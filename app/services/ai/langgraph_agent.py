"""LangGraph interview agent scaffolding.

This file now serves as the contract-first base for the no-embedding interview
engine. It holds the runtime state contract plus a deterministic interview plan
generator that prepares the one-time planning payload consumed by the later
LangGraph execution loop.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, TypedDict, cast

import logging
import hashlib
import json
import re

from app.services.company_data_cache import company_data_cache
from app.services.session_state_contract import build_persisted_session_record

logger = logging.getLogger(__name__)


class InterviewState(TypedDict):
    """Runtime interview state shared across the LangGraph execution loop."""

    history: List[dict]
    performance_summary: str
    context_summary: str
    has_jd: bool
    skills_remaining: List[str]
    skills_covered: List[str]
    questions_asked: List[str]
    last_user_input: Optional[str]
    current_question: str
    current_evaluation: Optional[dict]
    follow_up_hint: str
    interview_type: str
    time_context: str
    role: str
    company: str
    is_developer_role: bool
    coding_questions_asked: int
    last_question_was_coding: bool
    turn_number: int
    last_answer_type: str
    consecutive_non_answers: int
    ended: bool
    end_reason: str
    max_questions: int
    max_questions_per_topic: int
    topic_question_counts: Dict[str, int]
    max_confusion_retries: int
    consecutive_disengaged: int
    input_tokens: int
    output_tokens: int
    opening_line: str
    closing_line: str
    source_question_candidates: List[Dict[str, Any]]
    question_source_mode: str


class AgentState(TypedDict):
    """Wrapper state for the agent graph around the core interview state."""

    interview_state: InterviewState
    messages: List[Dict[str, Any]]
    current_node: str
    next_action: Literal["continue", "evaluate", "end_interview", "tool_call"]
    last_tool_result: Optional[str]
    tool_calls_count: int
    session_id: str
    created_at: datetime
    last_updated_at: datetime


DEFAULT_MAX_QUESTIONS = 0
DEFAULT_MAX_QUESTIONS_PER_TOPIC = 0
DEFAULT_MAX_CONFUSION_RETRIES = 2

DEFAULT_END_REASON = ""
DEFAULT_LAST_ANSWER_TYPE = "not_applicable"


class InterviewPlan(TypedDict):
    """Structured one-time plan produced from resume, JD, and company context."""

    company: str
    role: str
    interview_type: str
    topics: List[str]
    question_strategy: Dict[str, Any]
    evaluation_criteria: Dict[str, Any]
    company_signals: Dict[str, Any]
    token_budget: Dict[str, int]
    metadata: Dict[str, Any]
    source_evidence: List[Dict[str, Any]]
    question_candidates: List[Dict[str, Any]]
    opening_line: str
    closing_line: str


_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "Python": ["python", "pandas", "numpy", "fastapi", "flask", "django"],
    "JavaScript": ["javascript", "typescript", "node", "react", "next.js", "vite"],
    "System Design": ["system design", "microservices", "scalability", "distributed", "architecture"],
    "Databases": ["sql", "postgres", "mysql", "mongodb", "redis", "database"],
    "APIs": ["api", "rest", "graphql", "websocket", "grpc"],
    "Testing": ["test", "pytest", "jest", "unit testing", "integration testing"],
    "DevOps": ["docker", "kubernetes", "ci/cd", "aws", "gcp", "azure"],
    "DSA": ["algorithm", "data structure", "leetcode", "dynamic programming", "graph"],
    "Behavioral": ["leadership", "teamwork", "conflict", "communication", "ownership"],
    "Product Thinking": ["product", "trade-off", "user impact", "metrics", "stakeholder"],
}

_COMPANY_SIGNAL_KEYWORDS: Dict[str, List[str]] = {
    "leetcode": ["leetcode", "companywise", "interview questions", "coding round"],
    "system design": ["scale", "architecture", "performance", "latency", "throughput"],
    "backend": ["api", "database", "redis", "queue", "service"],
    "frontend": ["ui", "react", "state", "performance", "accessibility"],
}


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _find_topic_matches(text: str) -> List[str]:
    lowered = text.lower()
    matches: List[str] = []
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            matches.append(topic)
    return matches


def _infer_company_signals(company_name: str, resume_text: str, jd_text: str) -> Dict[str, Any]:
    combined = f"{company_name}\n{resume_text}\n{jd_text}".lower()
    detected = [
        signal
        for signal, keywords in _COMPANY_SIGNAL_KEYWORDS.items()
        if any(keyword in combined for keyword in keywords)
    ]
    return {
        "detected_signals": detected,
        "source_pool": [
            "leetcode company tags",
            "geeksforgeeks interview experiences",
            "glassdoor interview insights",
            "reddit r/leetcode",
            "company-wise github question lists",
        ],
    }


def _estimate_question_budget(duration_minutes: int) -> int:
    if duration_minutes <= 0:
        return 0
    if duration_minutes <= 15:
        return 8
    if duration_minutes <= 30:
        return 13
    if duration_minutes <= 45:
        return 16
    return 18


def _estimate_token_budget(question_budget: int) -> Dict[str, int]:
    if question_budget <= 0:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = question_budget * 1400
    output_tokens = question_budget * 350
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def generate_interview_plan(
    *,
    resume_text: str,
    jd_text: str,
    company_name: str,
    role: str = "",
    interview_type: str = "technical",
    duration_minutes: int = 30,
) -> InterviewPlan:
    """Generate a deterministic interview plan from resume, JD, and company context."""

    resume_topics = _find_topic_matches(resume_text)
    jd_topics = _find_topic_matches(jd_text)
    company_topics = _find_topic_matches(company_name)

    topics = _dedupe_preserve_order(company_topics + resume_topics + jd_topics)
    if interview_type == "hr":
        topics = _dedupe_preserve_order(["Behavioral", "Product Thinking"] + topics)
    elif interview_type == "behavioral":
        topics = _dedupe_preserve_order(["Behavioral"] + topics)
    elif interview_type == "problem":
        topics = _dedupe_preserve_order(["DSA", "Product Thinking"] + topics)
    elif interview_type == "technical" and not topics:
        topics = ["Python", "APIs", "Databases"]

    question_budget = _estimate_question_budget(duration_minutes)
    max_questions_per_topic = 0 if question_budget == 0 else (1 if duration_minutes <= 15 else 2 if duration_minutes <= 45 else 3)
    confusion_retries = 1 if duration_minutes <= 15 else 2

    jd_present = bool(jd_text.strip())
    company_signals = _infer_company_signals(company_name, resume_text, jd_text)
    token_budget = _estimate_token_budget(question_budget)

    question_strategy: Dict[str, Any] = {
        "interview_type": interview_type,
        "duration_minutes": duration_minutes,
        "question_budget": question_budget,
        "max_questions_per_topic": max_questions_per_topic,
        "max_confusion_retries": confusion_retries,
        "priority_order": ["company", "resume", "jd"] if interview_type == "technical" else (["company", "resume", "jd"] if company_name.strip() else (["resume", "jd"] if jd_present else ["resume", "company"])),
        "difficulty_progression": "adaptive",
        "coding_required_for_developer_roles": True,
        "one_question_per_turn": True,
        "anti_repetition": True,
    }

    evaluation_criteria: Dict[str, Any] = {
        "answer_types": [
            "genuine_answer",
            "confused",
            "refused",
            "off_topic",
            "incomplete",
            "wait_requested",
            "coding_requested",
            "end_requested",
            "not_applicable",
        ],
        "answer_quality": ["strong", "adequate", "weak", "incorrect", "not_applicable"],
        "follow_up_policy": "Only follow up on genuine weak/incorrect answers.",
        "termination_reasons": ["USER_EXIT", "AI_ENDED", "TIME_EXCEEDED", "DISENGAGED", "BUDGET_EXHAUSTED"],
        "token_tracking": "Track cumulative input_tokens and output_tokens every turn.",
    }

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "resume_hash": hashlib.sha256(resume_text.strip().encode("utf-8")).hexdigest(),
        "jd_hash": hashlib.sha256(jd_text.strip().encode("utf-8")).hexdigest(),
        "company_hash": hashlib.sha256(company_name.strip().lower().encode("utf-8")).hexdigest(),
        "question_count": len(topics),
        "uses_jd": jd_present,
    }

    return {
        "company": company_name,
        "role": role,
        "interview_type": interview_type,
        "topics": topics,
        "question_strategy": question_strategy,
        "evaluation_criteria": evaluation_criteria,
        "company_signals": company_signals,
        "token_budget": token_budget,
        "metadata": metadata,
    }


def serialize_interview_plan(plan: InterviewPlan) -> str:
    """Return a stable JSON payload for storage or API responses."""

    return json.dumps(plan, ensure_ascii=True, sort_keys=True)


async def get_or_create_interview_plan(
    *,
    resume_text: str,
    jd_text: str,
    company_name: str,
    role: str = "",
    interview_type: str = "technical",
    duration_minutes: int = 30,
    excluded_question_hashes: Iterable[str] | None = None,
) -> InterviewPlan:
    """Fetch interview plan from cache, or generate and persist on miss."""

    cached_plan = await company_data_cache.get_plan(
        resume_text=resume_text,
        jd_text=jd_text,
        company_name=company_name,
        role=role,
        interview_type=interview_type,
        duration_minutes=duration_minutes,
    )

    if cached_plan:
        plan = cast(InterviewPlan, cached_plan)
        metadata = dict(plan.get("metadata", {}))
        metadata["cache_hit"] = True
        plan["metadata"] = metadata
        if excluded_question_hashes:
            excluded = set(excluded_question_hashes)
            plan["question_candidates"] = [
                candidate for candidate in plan.get("question_candidates", []) if str(candidate.get("fingerprint", "")) not in excluded
            ]
        return plan

    try:
        from app.services.ai.source_interview_planner import source_interview_planner

        excluded = set()
        plan = await source_interview_planner.build_plan(
            resume_text=resume_text,
            jd_text=jd_text,
            company_name=company_name,
            role=role,
            interview_type=interview_type,
            duration_minutes=duration_minutes,
            excluded_question_hashes=excluded.union(set(excluded_question_hashes or [])),
        )
    except Exception as exc:
        logger.warning("[InterviewPlan] Falling back to keyword planner: %s", exc)
        plan = generate_interview_plan(
            resume_text=resume_text,
            jd_text=jd_text,
            company_name=company_name,
            role=role,
            interview_type=interview_type,
            duration_minutes=duration_minutes,
        )
        plan.setdefault("source_evidence", [])
        plan.setdefault("question_candidates", [])
        plan.setdefault("opening_line", f"Hi, thanks for joining us today. Let's begin with a question about {role or interview_type}.")
        plan.setdefault("closing_line", "Thank you so much for interviewing with us today. We really appreciate your time, and we’ll share the next steps soon.")

    metadata = dict(plan.get("metadata", {}))
    metadata["cache_hit"] = False
    plan["metadata"] = metadata

    if excluded_question_hashes:
        excluded = set(excluded_question_hashes)
        plan["question_candidates"] = [
            candidate for candidate in plan.get("question_candidates", []) if str(candidate.get("fingerprint", "")) not in excluded
        ]

    await company_data_cache.set_plan(
        resume_text=resume_text,
        jd_text=jd_text,
        company_name=company_name,
        role=role,
        interview_type=interview_type,
        duration_minutes=duration_minutes,
        plan=cast(Dict[str, Any], plan),
    )
    return plan


def create_default_interview_state(
    *,
    context_summary: str = "",
    interview_type: str = "technical",
    time_context: str = "",
    role: str = "",
    company: str = "",
    has_jd: bool = False,
    is_developer_role: bool = False,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
    max_questions_per_topic: int = DEFAULT_MAX_QUESTIONS_PER_TOPIC,
    max_confusion_retries: int = DEFAULT_MAX_CONFUSION_RETRIES,
) -> InterviewState:
    """Create a fully initialized InterviewState with deterministic defaults."""

    return {
        "history": [],
        "performance_summary": "",
        "context_summary": context_summary,
        "has_jd": has_jd,
        "skills_remaining": [],
        "skills_covered": [],
        "questions_asked": [],
        "last_user_input": None,
        "current_question": "",
        "current_evaluation": None,
        "follow_up_hint": "",
        "interview_type": interview_type,
        "time_context": time_context,
        "role": role,
        "company": company,
        "is_developer_role": is_developer_role,
        "coding_questions_asked": 0,
        "last_question_was_coding": False,
        "turn_number": 0,
        "last_answer_type": DEFAULT_LAST_ANSWER_TYPE,
        "consecutive_non_answers": 0,
        "ended": False,
        "end_reason": DEFAULT_END_REASON,
        "max_questions": max_questions,
        "max_questions_per_topic": max_questions_per_topic,
        "topic_question_counts": {},
        "max_confusion_retries": max_confusion_retries,
        "consecutive_disengaged": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def normalize_interview_state(
    state: Dict[str, Any] | InterviewState,
    *,
    context_summary: str = "",
) -> InterviewState:
    """Fill missing keys in a partial interview state without overwriting values."""

    normalized = create_default_interview_state(context_summary=context_summary)
    normalized.update(cast(Dict[str, Any], state))

    normalized["history"] = list(normalized.get("history", []))
    normalized["skills_remaining"] = list(normalized.get("skills_remaining", []))
    normalized["skills_covered"] = list(normalized.get("skills_covered", []))
    normalized["questions_asked"] = list(normalized.get("questions_asked", []))
    normalized["topic_question_counts"] = dict(normalized.get("topic_question_counts", {}))

    return normalized


def create_agent_state(
    interview_state: InterviewState,
    session_id: str,
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> AgentState:
    """Create the wrapper state used by the LangGraph runner."""

    now = datetime.now(timezone.utc)
    return {
        "interview_state": interview_state,
        "messages": messages or [],
        "current_node": "START",
        "next_action": "continue",
        "last_tool_result": None,
        "tool_calls_count": 0,
        "session_id": session_id,
        "created_at": now,
        "last_updated_at": now,
    }


def update_token_counters(
    interview_state: InterviewState,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> InterviewState:
    """Increment cumulative token counters in-place and return the state."""

    interview_state["input_tokens"] = interview_state.get("input_tokens", 0) + input_tokens
    interview_state["output_tokens"] = interview_state.get("output_tokens", 0) + output_tokens
    return interview_state


def get_state_snapshot(state: AgentState) -> Dict[str, Any]:
    """Return a JSON-friendly snapshot for logging or persistence."""

    interview_state = state["interview_state"]
    return {
        "session_id": state["session_id"],
        "current_node": state["current_node"],
        "next_action": state["next_action"],
        "tool_calls_count": state["tool_calls_count"],
        "created_at": state["created_at"].isoformat(),
        "last_updated_at": state["last_updated_at"].isoformat(),
        "interview_state": {
            "turn_number": interview_state["turn_number"],
            "last_answer_type": interview_state["last_answer_type"],
            "ended": interview_state["ended"],
            "end_reason": interview_state["end_reason"],
            "questions_asked": len(interview_state["questions_asked"]),
            "skills_covered": interview_state["skills_covered"],
            "skills_remaining": interview_state["skills_remaining"],
            "consecutive_non_answers": interview_state["consecutive_non_answers"],
            "consecutive_disengaged": interview_state["consecutive_disengaged"],
            "input_tokens": interview_state["input_tokens"],
            "output_tokens": interview_state["output_tokens"],
        },
    }


def build_session_persistence_payload(
    *,
    user_id: str,
    client_id: str,
    state: AgentState,
    ttl_seconds: int,
    previous_token_summary: Dict[str, Any] | None = None,
    accounting_source: str = "graph_state_cumulative",
    telemetry: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the canonical session payload for Redis persistence."""

    snapshot = get_state_snapshot(state)
    return build_persisted_session_record(
        session_id=state["session_id"],
        user_id=user_id,
        client_id=client_id,
        interview_state=state["interview_state"],
        agent_meta={
            "current_node": snapshot["current_node"],
            "next_action": snapshot["next_action"],
            "tool_calls_count": snapshot["tool_calls_count"],
            "created_at": snapshot["created_at"],
            "last_updated_at": snapshot["last_updated_at"],
        },
        ttl_seconds=ttl_seconds,
        status="active",
        previous_token_summary=previous_token_summary,
        accounting_source=accounting_source,
        telemetry=telemetry,
    )


def build_interview_agent() -> Dict[str, Any]:
    """Return the base graph contract as a serializable spec."""

    return {
        "state": "AgentState",
        "nodes": ["assess_state", "evaluate_turn", "generate_question", "end_interview"],
        "edges": [
            {"from": "START", "to": "assess_state"},
            {"from": "assess_state", "to": "evaluate_turn", "when": "next_action == evaluate"},
            {"from": "assess_state", "to": "end_interview", "when": "next_action == end_interview"},
            {"from": "evaluate_turn", "to": "generate_question"},
            {"from": "generate_question", "to": "assess_state"},
            {"from": "end_interview", "to": "END"},
        ],
        "notes": "Contract-first placeholder only; actual LangGraph implementation comes in the next step.",
    }


def compile_interview_agent() -> Any:
    """Return the base agent spec for now."""

    return build_interview_agent()


def initialize_agent_state(
    interview_state: Dict[str, Any] | InterviewState,
    session_id: str,
) -> AgentState:
    """Initialize a new agent state from a partial or complete interview state."""

    normalized_state = normalize_interview_state(interview_state)
    agent_state = create_agent_state(normalized_state, session_id)
    logger.info("[%s] Initialized agent state", session_id)
    return agent_state


if __name__ == "__main__":
    sample = create_default_interview_state(
        context_summary="Sample candidate context",
        interview_type="technical",
        role="Software Engineer",
        company="ExampleCo",
        has_jd=True,
    )
    state = create_agent_state(sample, "demo-session")
    print(get_state_snapshot(state))
