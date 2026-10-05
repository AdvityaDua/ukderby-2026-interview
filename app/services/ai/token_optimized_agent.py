"""Token-optimized interview agent.

This is the replacement for the previous LangGraph-based flow. It performs one
compact LLM call per turn, using a compressed prompt and a single structured
output schema that combines answer evaluation and next-question generation.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.key_manager import key_manager
from .gemini_client import GeminiClient
from .fine_tuned_gateway import gateway_client
from .source_interview_planner import question_fingerprint
from .schemas import (
    Action,
    AnswerEvaluation,
    ClarityDepth,
    ConfidenceLevel,
    Correctness,
    Difficulty,
    Decision,
    NextStep,
    NextStepType,
    SignalStrength,
)

logger = logging.getLogger(__name__)


class InterviewState(TypedDict):
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


class OptimizedTurnOutput(BaseModel):
    performance_summary: str = Field(..., description="Concise updated summary of the candidate's trajectory.")
    answer_type: str = Field(
        "genuine_answer",
        description="One of: genuine_answer, confused, refused, off_topic, incomplete, wait_requested, end_requested, not_applicable",
    )
    answer_quality: str = Field(..., description="One of: strong, adequate, weak, incorrect, not_applicable")
    should_follow_up: bool = False
    follow_up_hint: str = ""
    newly_covered_skills: List[str] = Field(default_factory=list)
    last_answer_evaluation: Optional[AnswerEvaluation] = None
    decision: Decision
    next_step: NextStep
    confidence_in_candidate: ConfidenceLevel = ConfidenceLevel.MEDIUM


def _max_questions_for_duration(duration_minutes: int) -> int:
    if duration_minutes <= 0:
        return 0
    if duration_minutes <= 15:
        return 8
    if duration_minutes <= 30:
        return 13
    if duration_minutes <= 45:
        return 16
    return 18


def _max_questions_per_topic(duration_minutes: int) -> int:
    if duration_minutes <= 0:
        return 0
    if duration_minutes <= 15:
        return 1
    if duration_minutes <= 30:
        return 2
    if duration_minutes <= 45:
        return 2
    return 3


def _max_confusion_retries_for_duration(duration_minutes: int) -> int:
    if duration_minutes <= 0:
        return 2
    if duration_minutes <= 15:
        return 1
    return 2


def _take_recent(items: List[str], limit: int) -> List[str]:
    if limit <= 0:
        return []
    return items[-limit:]


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_text_for_comparison(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _looks_like_greeting(text: str) -> bool:
    normalized = _normalize_text_for_comparison(text)
    return normalized.startswith(("hi", "hello", "good morning", "good afternoon", "good evening"))


def _looks_like_closing(text: str) -> bool:
    normalized = _normalize_text_for_comparison(text)
    return normalized.startswith(("thank you", "thanks", "that concludes", "we appreciate", "it was great"))


def _pick_coding_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    coding_candidates = [item for item in candidates if isinstance(item, dict) and item.get("is_coding_question")]
    if coding_candidates:
        return coding_candidates[0]
    return None


class InterviewGraph:
    """Token-optimized interview engine."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        client: GeminiClient | None = None,
    ):
        if client is not None:
            self.client = client
        else:
            resolved_api_key = api_key or key_manager.get_gemini_key()
            self.client = GeminiClient(api_key=resolved_api_key, model_name=model_name)
        self.model_name = getattr(self.client, "model_name", model_name)

    def _build_system_instruction(self, state: InterviewState) -> str:
        interview_type = state.get("interview_type", "technical")
        role = state.get("role", "")
        company = state.get("company", "")
        has_jd = state.get("has_jd", False)
        return f"""
You are a compact interview engine for a {interview_type} interview.
Role: {role or 'Not specified'}
Company: {company or 'Not specified'}
JD present: {has_jd}

Your job:
- Evaluate the user's answer briefly.
- Update the performance summary compactly.
- Decide whether to follow up, move on, or end.
- Generate exactly one next question.
- Keep prompts and outputs concise to reduce tokens.
- Ask a coding question only when it is actually required.
- Do not restate the entire history.
- Use the candidate's resume/JD context already supplied in the state.
- Use the provided source-backed question candidates when available.
- For technical interviews, balance company, resume, and JD evidence instead of leaning on only one source.
- Start with a brief greeting on the first turn.
- End with a polite closing statement when the interview is finished.
""".strip()

    def _build_prompt(self, state: InterviewState) -> str:
        skills_remaining = _take_recent(state.get("skills_remaining", []), 12)
        skills_covered = _take_recent(state.get("skills_covered", []), 12)
        questions_asked = _take_recent(state.get("questions_asked", []), 8)
        recent_history = state.get("history", [])[-6:]
        history_text = "\n".join(
            f"{item.get('role', 'unknown').upper()}: {item.get('content', '')}" for item in recent_history
        ) or "(No prior exchange)"
        last_user_input = state.get("last_user_input") or ""
        current_question = state.get("current_question") or ""
        time_context = state.get("time_context") or ""
        opening_line = state.get("opening_line") or ""
        closing_line = state.get("closing_line") or ""
        candidates = state.get("source_question_candidates", []) or []
        max_questions = int(state.get("max_questions", 0) or 0)
        questions_used = len(questions_asked)
        questions_left = max(0, max_questions - questions_used) if max_questions > 0 else 0

        return f"""
STATE SNAPSHOT
- turn_number: {state.get('turn_number', 0)}
- interview_type: {state.get('interview_type', '')}
- role: {state.get('role', '')}
- company: {state.get('company', '')}
- time_context: {time_context}
- ended: {state.get('ended', False)}
- end_reason: {state.get('end_reason', '')}
- last_answer_type: {state.get('last_answer_type', '')}
- consecutive_non_answers: {state.get('consecutive_non_answers', 0)}
- consecutive_disengaged: {state.get('consecutive_disengaged', 0)}
- questions_used: {questions_used}
- questions_left: {questions_left}
- max_questions_per_topic: {state.get('max_questions_per_topic', 0)}
- max_confusion_retries: {state.get('max_confusion_retries', 0)}
- is_developer_role: {state.get('is_developer_role', False)}
- question_source_mode: {state.get('question_source_mode', '')}

COMPRESSED CONTEXT
{state.get('context_summary', '')}

SKILLS TO COVER
{', '.join(skills_remaining) if skills_remaining else '(none)'}

SKILLS ALREADY COVERED
{', '.join(skills_covered) if skills_covered else '(none)'}

PREVIOUS QUESTIONS
{chr(10).join(f'- {q}' for q in questions_asked) if questions_asked else '(none)'}

RECENT HISTORY
{history_text}

CURRENT TURN
- current_question: {current_question}
- last_user_input: {last_user_input or '(none)'}
- opening_line: {opening_line or '(none)'}
- closing_line: {closing_line or '(none)'}

SOURCE-BACKED QUESTION CANDIDATES
{chr(10).join(f'- {item.get("question", "")} [{item.get("topic", "")}]' for item in candidates[:5]) if candidates else '(none)'}

RULES
- Produce one concise next question only.
- If the candidate is confused, rephrase simply and stay on the same topic.
- If the candidate asks for coding, return a coding question only when appropriate.
- Avoid repeating previously asked questions.
- For technical interviews, prefer a balanced mix of company, resume, and JD-backed candidates.
- Prefer the smallest prompt possible that still moves the interview forward.
- If time is running out or budget is exhausted, end cleanly.
- If source-backed candidates are available, prefer them over inventing a new question.

Return JSON matching the schema exactly.
""".strip()

    # ── Fine-tuned gateway integration ────────────────────────────────────────

    async def _try_gateway_turn(
        self, state: InterviewState, system_instruction: str, prompt: str
    ) -> Optional[OptimizedTurnOutput]:
        """Attempt to run the turn through the fine-tuned gateway.

        Splits the single combined call into two sequential requests:
          1. /evaluator  — evaluates the candidate's last answer
          2. /interviewer — generates the next question

        Merges results into an OptimizedTurnOutput.
        Returns None on any failure (caller should fall back).
        """
        use_evaluator = settings.FINE_TUNED_EVALUATOR_ENABLED
        use_interviewer = settings.FINE_TUNED_INTERVIEWER_ENABLED

        if not use_evaluator and not use_interviewer:
            return None

        start = time.monotonic()
        eval_data: Optional[dict] = None
        interviewer_data: Optional[dict] = None

        # ── Step 1: Evaluate the candidate's last answer ─────────────────────
        if use_evaluator and state.get("last_user_input"):
            eval_messages = [
                {"role": "system", "content": system_instruction},
                {
                    "role": "user",
                    "content": (
                        f"Evaluate the candidate's answer.\n\n{prompt}\n\n"
                        f"You MUST return ONLY a valid JSON object matching this exact structure, with no markdown formatting or extra text:\n"
                        f"{{\n"
                        f'  "performance_summary": "...",\n'
                        f'  "answer_type": "genuine_answer",\n'
                        f'  "answer_quality": "strong",\n'
                        f'  "should_follow_up": false,\n'
                        f'  "follow_up_hint": "...",\n'
                        f'  "newly_covered_skills": [],\n'
                        f'  "confidence_in_candidate": "high"\n'
                        f"}}"
                    ),
                },
            ]
            eval_data = await gateway_client.generate_evaluator_response(
                messages=eval_messages, temperature=0.25, max_tokens=1024
            )
            if eval_data is None:
                logger.warning(
                    "[FineTunedGateway] Evaluator call failed — falling back for entire turn"
                )
                return None

        # ── Step 2: Generate the next question ───────────────────────────────
        if use_interviewer:
            # Enrich the interviewer prompt with evaluation context if available
            eval_context = ""
            if eval_data:
                eval_context = (
                    f"\nEVALUATION CONTEXT (from evaluator):\n"
                    f"- answer_type: {eval_data.get('answer_type', 'genuine_answer')}\n"
                    f"- answer_quality: {eval_data.get('answer_quality', 'not_applicable')}\n"
                    f"- should_follow_up: {eval_data.get('should_follow_up', False)}\n"
                    f"- follow_up_hint: {eval_data.get('follow_up_hint', '')}\n"
                    f"- performance_summary: {eval_data.get('performance_summary', '')}\n"
                )

            interviewer_messages = [
                {"role": "system", "content": system_instruction},
                {
                    "role": "user",
                    "content": (
                        f"Generate the next interview question.\n\n{prompt}"
                        f"{eval_context}\n\n"
                        f"You MUST return ONLY a valid JSON object matching this exact structure, with no markdown formatting or extra text:\n"
                        f"{{\n"
                        f'  "decision": {{"action": "continue", "reason": "...", "termination_flag": false}},\n'
                        f'  "next_step": {{"type": "follow_up", "difficulty": "medium", "question": "...", "target_skill": "...", "is_coding_question": false}},\n'
                        f'  "performance_summary": "...",\n'
                        f'  "answer_type": "genuine_answer",\n'
                        f'  "answer_quality": "strong",\n'
                        f'  "should_follow_up": false,\n'
                        f'  "follow_up_hint": "...",\n'
                        f'  "newly_covered_skills": [],\n'
                        f'  "confidence_in_candidate": "high"\n'
                        f"}}"
                    ),
                },
            ]
            interviewer_data = await gateway_client.generate_interviewer_response(
                messages=interviewer_messages, temperature=0.7, max_tokens=1024
            )
            if interviewer_data is None:
                logger.warning(
                    "[FineTunedGateway] Interviewer call failed — falling back for entire turn"
                )
                return None

        # ── Step 3: Merge results into OptimizedTurnOutput ───────────────────
        try:
            merged: Dict[str, Any] = {}

            # Start with evaluator data if available
            if eval_data:
                merged.update(eval_data)

            # Overlay interviewer data (question generation takes precedence for
            # decision/next_step fields)
            if interviewer_data:
                merged.update(interviewer_data)
                # But keep evaluator's evaluation fields if interviewer didn't provide them
                if eval_data:
                    for key in ("answer_type", "answer_quality", "should_follow_up",
                                "follow_up_hint", "newly_covered_skills"):
                        if key not in interviewer_data and key in eval_data:
                            merged[key] = eval_data[key]

            # If only one endpoint was called, use its data directly
            if not eval_data and interviewer_data:
                merged = interviewer_data
            elif eval_data and not interviewer_data:
                merged = eval_data

            parsed = OptimizedTurnOutput.model_validate(merged)
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "[FineTunedGateway] Turn completed via gateway (%dms, "
                "evaluator=%s, interviewer=%s)",
                duration_ms,
                "used" if eval_data else "skipped",
                "used" if interviewer_data else "skipped",
            )
            return parsed

        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "[FineTunedGateway] Failed to parse merged gateway response (%dms): "
                "%s: %s — falling back",
                duration_ms, type(exc).__name__, exc,
            )
            return None

    # ── Main turn execution ──────────────────────────────────────────────────

    async def run_turn(self, state: InterviewState) -> InterviewState:
        if state.get("ended"):
            return state

        system_instruction = self._build_system_instruction(state)
        prompt = self._build_prompt(state)

        # ── Try fine-tuned gateway first (if enabled) ────────────────────────
        parsed: Optional[OptimizedTurnOutput] = None
        input_tokens = 0
        output_tokens = 0

        if settings.FINE_TUNED_INTERVIEWER_ENABLED or settings.FINE_TUNED_EVALUATOR_ENABLED:
            parsed = await self._try_gateway_turn(state, system_instruction, prompt)
            # Gateway does not report token usage — counts stay at 0 for gateway turns

        # ── Fallback to Gemini ────────────────────────────────────────────────
        if parsed is None:
            response = await asyncio.to_thread(
                self.client.client.models.generate_content,
                model=self.model_name,
                contents=f"{system_instruction}\n\n{prompt}",
                config={
                    "response_mime_type": "application/json",
                    "response_schema": OptimizedTurnOutput,
                },
            )

            parsed = OptimizedTurnOutput.model_validate_json(response.text)
            usage_meta = getattr(response, "usage_metadata", None)
            input_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0) if usage_meta else 0
            output_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0) if usage_meta else 0

        skills_remaining = list(state.get("skills_remaining", []))
        skills_covered = list(state.get("skills_covered", []))
        new_skills = [skill for skill in parsed.newly_covered_skills if skill and skill not in skills_covered]
        if new_skills:
            skills_covered.extend(new_skills)
            skills_remaining = [skill for skill in skills_remaining if skill not in new_skills]

        if parsed.should_follow_up and parsed.follow_up_hint:
            follow_up_hint = parsed.follow_up_hint
        else:
            follow_up_hint = ""

        questions_asked = list(state.get("questions_asked", []))
        questions_asked.append(parsed.next_step.question)

        opening_line = (state.get("opening_line") or "").strip()
        closing_line = (state.get("closing_line") or "").strip()
        candidate_pool = list(state.get("source_question_candidates", []) or [])
        must_include_coding = bool(state.get("must_ask_coding_question", False) or state.get("needs_coding_question", False))
        ended = False

        current_question = parsed.next_step.question.strip()
        if state.get("turn_number", 0) == 0 and opening_line:
            if current_question and not _looks_like_greeting(current_question):
                current_question = f"{opening_line} {current_question}".strip()
            elif not current_question:
                current_question = opening_line

        if (
            must_include_coding
            and state.get("coding_questions_asked", 0) == 0
            and state.get("turn_number", 0) > 0
            and not ended
        ):
            coding_candidate = _pick_coding_candidate(candidate_pool)
            if coding_candidate is not None:
                current_question = str(coding_candidate.get("question", current_question)).strip() or current_question
                parsed.next_step.is_coding_question = True
                parsed.next_step.question = current_question
                parsed.next_step.target_skill = str(coding_candidate.get("topic", parsed.next_step.target_skill))
                parsed.next_step.type = parsed.next_step.type or NextStepType.FOLLOW_UP

        last_answer_type = parsed.answer_type
        if last_answer_type in {"confused", "refused", "off_topic", "wait_requested"}:
            consecutive_non_answers = state.get("consecutive_non_answers", 0) + 1
        else:
            consecutive_non_answers = 0

        if last_answer_type in {"refused", "off_topic"}:
            consecutive_disengaged = state.get("consecutive_disengaged", 0) + 1
        else:
            consecutive_disengaged = 0

        ended = parsed.decision.action == Action.END or parsed.decision.termination_flag
        end_reason = parsed.decision.reason if ended else state.get("end_reason", "")

        if ended and closing_line:
            current_question = closing_line

        current_evaluation = parsed.model_dump()
        current_evaluation["next_step"] = parsed.next_step.model_dump()
        current_evaluation["decision"] = parsed.decision.model_dump()
        if parsed.last_answer_evaluation is not None:
            current_evaluation["last_answer_evaluation"] = parsed.last_answer_evaluation.model_dump()

        if ended and closing_line:
            current_evaluation["next_step"]["question"] = closing_line

        if candidate_pool:
            candidate_fingerprints = {str(item.get("fingerprint", "")) for item in candidate_pool if isinstance(item, dict)}
            if candidate_fingerprints and question_fingerprint(current_question) in candidate_fingerprints:
                current_evaluation["next_step"]["question"] = current_question

        history = list(state.get("history", []))
        if state.get("last_user_input"):
            history.append({"role": "user", "content": state["last_user_input"]})
        history.append({"role": "model", "content": current_question})

        return {
            **state,
            "history": history,
            "performance_summary": parsed.performance_summary,
            "skills_remaining": skills_remaining,
            "skills_covered": skills_covered,
            "questions_asked": questions_asked,
            "current_question": current_question,
            "current_evaluation": current_evaluation,
            "follow_up_hint": follow_up_hint,
            "coding_questions_asked": state.get("coding_questions_asked", 0) + (1 if parsed.next_step.is_coding_question else 0),
            "last_question_was_coding": parsed.next_step.is_coding_question and not ended,
            "turn_number": state.get("turn_number", 0) + 1,
            "last_answer_type": last_answer_type,
            "consecutive_non_answers": consecutive_non_answers,
            "ended": ended,
            "end_reason": end_reason,
            "topic_question_counts": self._bump_topic_counts(state, parsed.next_step.target_skill),
            "consecutive_disengaged": consecutive_disengaged,
            "input_tokens": state.get("input_tokens", 0) + input_tokens,
            "output_tokens": state.get("output_tokens", 0) + output_tokens,
        }

    @staticmethod
    def _bump_topic_counts(state: InterviewState, target_skill: str) -> Dict[str, int]:
        topic_counts = dict(state.get("topic_question_counts", {}) or {})
        if target_skill:
            topic_counts[target_skill] = topic_counts.get(target_skill, 0) + 1
        return topic_counts
