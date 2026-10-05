from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user
from app.core.key_manager import key_manager
from app.services.ai.langgraph_agent import (
    create_agent_state,
    create_default_interview_state,
    get_or_create_interview_plan,
    normalize_interview_state,
)
from app.services.ai.topic_interview_planner import generate_topic_plan
from app.services.ai.source_interview_planner import question_fingerprint
from app.services.ai.response_builder import build_turn_response
from app.services.ai.token_optimized_agent import InterviewGraph
from app.services.session_state_store import session_state_store

router = APIRouter()
logger = logging.getLogger(__name__)


_DEVELOPER_ROLE_KEYWORDS = (
    "developer",
    "engineer",
    "programmer",
    "software",
    "backend",
    "frontend",
    "full stack",
    "fullstack",
    "sde",
    "swe",
    "architect",
    "devops",
    "data scientist",
    "ml engineer",
)


class InitInterviewRequest(BaseModel):
    resume_text: str = Field(default="", min_length=0)
    jd_text: str = ""
    company: str = ""
    role: str = ""
    interview_type: str = "technical"
    duration_minutes: int = 30
    client_id: str = ""
    session_id: str | None = None
    # Topic-specific interview fields (alternative to resume_text)
    topic_id: str | None = None
    topic_name: str | None = None
    difficulty: str | None = None  # junior|mid|senior|expert
    subtopics: list[str] | None = None


class RunTurnRequest(BaseModel):
    session_id: str
    user_input: str = Field(..., min_length=1)
    client_id: str = ""


class EndInterviewRequest(BaseModel):
    session_id: str


def _validate_user(user: dict | None) -> dict:
    if not user or not user.get("sub"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def _ensure_model_ready() -> InterviewGraph:
    return InterviewGraph()


@router.post("/interview-engine/init")
async def init_interview_engine(
    payload: InitInterviewRequest,
    user: dict = Depends(get_current_user),
):
    user = _validate_user(user)
    user_id = user["sub"]
    start_ts = time.perf_counter()
    load_history = getattr(session_state_store, "load_user_question_history", None)
    prior_question_hashes = await load_history(user_id) if callable(load_history) else []

    t_plan_start = time.perf_counter()
    
    # Determine if this is a topic-specific interview
    is_topic_specific = (
        payload.interview_type == "topic-specific" 
        or payload.topic_id is not None
    )
    
    if is_topic_specific:
        # Generate plan from topic configuration
        if not payload.topic_id:
            raise HTTPException(
                status_code=400, 
                detail="topic_id is required for topic-specific interviews"
            )
        plan = await generate_topic_plan(
            topic_id=payload.topic_id,
            difficulty=payload.difficulty or "mid",
            duration_minutes=payload.duration_minutes,
            subtopics=payload.subtopics,
        )
        interview_type = "topic-specific"
    else:
        # Generate plan from resume and JD (existing flow)
        if not payload.resume_text.strip():
            raise HTTPException(
                status_code=400,
                detail="resume_text is required for standard interviews"
            )
        plan = await get_or_create_interview_plan(
            resume_text=payload.resume_text,
            jd_text=payload.jd_text,
            company_name=payload.company,
            role=payload.role,
            interview_type=payload.interview_type,
            duration_minutes=payload.duration_minutes,
            excluded_question_hashes=prior_question_hashes,
        )
        interview_type = payload.interview_type
    
    plan_ms = int((time.perf_counter() - t_plan_start) * 1000)

    session_id = payload.session_id or str(uuid4())
    role_lower = (payload.role or "").lower()
    is_developer_role = any(keyword in role_lower for keyword in _DEVELOPER_ROLE_KEYWORDS)

    strategy = plan.get("question_strategy", {})
    state = create_default_interview_state(
        context_summary=(
            f"Role: {payload.role or payload.topic_name or 'N/A'}\n"
            f"Company: {payload.company or 'N/A'}\n"
            f"Interview type: {interview_type}\n"
            f"Planning topics: {', '.join(plan.get('topics', []))}"
        ),
        interview_type=interview_type,
        role=payload.role,
        company=payload.company,
        has_jd=bool(payload.jd_text.strip()) if payload.jd_text else False,
        is_developer_role=is_developer_role,
        max_questions=int(strategy.get("question_budget", 0) or 0),
        max_questions_per_topic=int(strategy.get("max_questions_per_topic", 0) or 0),
        max_confusion_retries=int(strategy.get("max_confusion_retries", 2) or 2),
        time_context=f"Target duration: {payload.duration_minutes} minutes",
    )
    state["skills_remaining"] = list(plan.get("topics", []))
    state["performance_summary"] = "Interview initialized."
    state["opening_line"] = str(plan.get("opening_line", ""))
    state["closing_line"] = str(plan.get("closing_line", ""))
    state["source_question_candidates"] = list(plan.get("question_candidates", []))
    state["source_evidence"] = list(plan.get("source_evidence", []))
    state["must_ask_coding_question"] = bool(plan.get("needs_coding_question", False))
    state["question_source_mode"] = str(plan.get("question_source_mode", "company_first_jd_fallback"))

    t_graph_start = time.perf_counter()
    graph = _ensure_model_ready()
    updated_state = await graph.run_turn(state)
    normalized_state = normalize_interview_state(updated_state)
    graph_ms = int((time.perf_counter() - t_graph_start) * 1000)

    agent_state = create_agent_state(normalized_state, session_id)
    client_id = payload.client_id or session_id
    t_persist_start = time.perf_counter()
    latency = {
        "stage": "init",
        "plan_ms": plan_ms,
        "graph_ms": graph_ms,
    }
    saved_record = await session_state_store.save_agent_state(
        user_id=user_id,
        client_id=client_id,
        state=agent_state,
        accounting_source="graph_init_run_turn",
        telemetry=latency,
    )
    append_history = getattr(session_state_store, "append_user_question_history", None)
    if callable(append_history):
        await append_history(
            user_id,
            [question_fingerprint(normalized_state.get("current_question", ""))],
        )
    persist_ms = int((time.perf_counter() - t_persist_start) * 1000)
    total_ms = int((time.perf_counter() - start_ts) * 1000)
    latency["persist_ms"] = persist_ms
    latency["total_ms"] = total_ms
    logger.info("[InterviewEngine] init session=%s user=%s total_ms=%s plan_ms=%s graph_ms=%s persist_ms=%s", session_id, user_id, total_ms, plan_ms, graph_ms, persist_ms)

    return build_turn_response(
        session_id,
        normalized_state,
        plan,
        token_summary=saved_record.get("token_summary", {}),
        telemetry=latency,
    )


@router.post("/interview-engine/turn")
async def run_interview_turn(
    payload: RunTurnRequest,
    user: dict = Depends(get_current_user),
):
    user = _validate_user(user)
    user_id = user["sub"]
    start_ts = time.perf_counter()

    t_load_start = time.perf_counter()
    record = await session_state_store.load_record(payload.session_id)
    load_ms = int((time.perf_counter() - t_load_start) * 1000)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    if record.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Session does not belong to this user")

    lock_owner = f"{user_id}:{payload.client_id or 'api'}"
    t_lock_start = time.perf_counter()
    acquired = await session_state_store.acquire_turn_lock(
        session_id=payload.session_id,
        owner=lock_owner,
    )
    lock_wait_ms = int((time.perf_counter() - t_lock_start) * 1000)
    if not acquired:
        raise HTTPException(status_code=409, detail="Another turn is already in progress for this session")

    try:
        state = normalize_interview_state(record.get("interview_state", {}))
        state["last_user_input"] = payload.user_input

        t_graph_start = time.perf_counter()
        graph = _ensure_model_ready()
        updated_state = await graph.run_turn(state)
        normalized_state = normalize_interview_state(updated_state)
        graph_ms = int((time.perf_counter() - t_graph_start) * 1000)

        agent_state = create_agent_state(normalized_state, payload.session_id)
        agent_state["current_node"] = "END" if normalized_state.get("ended") else "RUN_TURN"

        t_persist_start = time.perf_counter()
        latency = {
            "stage": "turn",
            "load_ms": load_ms,
            "lock_wait_ms": lock_wait_ms,
            "graph_ms": graph_ms,
        }
        saved_record = await session_state_store.save_agent_state(
            user_id=user_id,
            client_id=payload.client_id or str(record.get("client_id", "")) or payload.session_id,
            state=agent_state,
            accounting_source="graph_turn_run_turn",
            telemetry=latency,
        )
        append_history = getattr(session_state_store, "append_user_question_history", None)
        if callable(append_history):
            await append_history(
                user_id,
                [question_fingerprint(normalized_state.get("current_question", ""))],
            )
        persist_ms = int((time.perf_counter() - t_persist_start) * 1000)
        total_ms = int((time.perf_counter() - start_ts) * 1000)
        latency["persist_ms"] = persist_ms
        latency["total_ms"] = total_ms
        logger.info("[InterviewEngine] turn session=%s user=%s turn=%s total_ms=%s graph_ms=%s", payload.session_id, user_id, normalized_state.get("turn_number", 0), total_ms, graph_ms)

        return build_turn_response(
            payload.session_id,
            normalized_state,
            token_summary=saved_record.get("token_summary", {}),
            telemetry=latency,
        )
    finally:
        await session_state_store.release_turn_lock(session_id=payload.session_id, owner=lock_owner)


@router.post("/interview-engine/end")
async def end_interview_engine(
    payload: EndInterviewRequest,
    user: dict = Depends(get_current_user),
):
    user = _validate_user(user)
    user_id = user["sub"]
    start_ts = time.perf_counter()

    record = await session_state_store.load_record(payload.session_id)
    if record and record.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Session does not belong to this user")

    ended = await session_state_store.mark_ended(payload.session_id)
    await session_state_store.clear_session(session_id=payload.session_id, user_id=user_id)

    total_ms = int((time.perf_counter() - start_ts) * 1000)
    logger.info("[InterviewEngine] end session=%s user=%s total_ms=%s", payload.session_id, user_id, total_ms)

    return {
        "session_id": payload.session_id,
        "status": "ended",
        "found": bool(ended),
        "telemetry": {
            "stage": "end",
            "total_ms": total_ms,
        },
    }
