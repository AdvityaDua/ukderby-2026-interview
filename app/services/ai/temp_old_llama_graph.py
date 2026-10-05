"""Interview orchestrator for fine-tuned LLaMA models.

Matches the EXACT request/response format the model was trained on
(see API demo doc).  Every mismatch in the request schema degrades
question quality — this version is schema-correct.

Interviewer request shape:
  {jd, round_type, difficulty, duration, style,
   conversation_history: [{question, answer}],
   candidate_answer: str}

Evaluator request shape:
  {jd, question, answer, round_type, difficulty,
   question_type: "main"|"followup",
   previous_scores: [float, ...]}        ← plain floats, NOT objects

Per-turn flow:
  1. Build rich JD from state
  2. Evaluate last answer  →  POST /evaluator/score
  3. Plan strategy         →  adaptive difficulty / topic / follow-up
  4. Generate question     →  POST /interviewer/question
                              (or re-use evaluator's follow_up verbatim)
  5. Apply framing         →  opening / closing lines
  6. Update state
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, List, Optional

import httpx

LLAMA_BASE_URL: str = os.getenv(
    "LLAMA_API_URL", "https://nutlike-rants-earache.ngrok-free.dev"
).rstrip("/")

_HEADERS = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
}

_SCORE_ADVANCE   = 5.0   # final_score >= this → move to next topic
_SCORE_HARD      = 7.5   # rolling avg >= this → difficulty = hard
_SCORE_EASY      = 4.0   # rolling avg <= this → difficulty = easy
_MAX_RETRIES     = 2
_RETRY_DELAY     = 1.5


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

async def _post(endpoint: str, payload: dict) -> dict:
    url = f"{LLAMA_BASE_URL}{endpoint}"
    last: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=40.0, headers=_HEADERS) as http:
                resp = await http.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            last = exc
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_DELAY)
    raise RuntimeError(f"[LlamaGraph] {endpoint} failed after {_MAX_RETRIES} tries: {last}")


# ---------------------------------------------------------------------------
# JD builder  — sends the FULL schema the model was trained on
# ---------------------------------------------------------------------------

def _extract_jd_section(context_summary: str) -> str:
    """Pull the raw JD block out of the context summary string."""
    marker = "JOB DESCRIPTION:"
    if marker in context_summary:
        return context_summary.split(marker, 1)[-1].strip()[:800]
    return ""


def _infer_level(context: str) -> tuple[str, str]:
    """Return (level_label, experience_required) from context text."""
    c = context.lower()
    if any(k in c for k in ("senior", "lead", "principal", "staff", "architect", "10+ years", "8+ years")):
        return "senior", "6+ years"
    if any(k in c for k in ("junior", "fresher", "intern", "entry", "graduate", "0-2", "1 year")):
        return "fresher", "0-2 years"
    return "mid-level", "3-5 years"


def _build_jd(state: dict) -> dict:
    """Build the rich JD dict the model expects."""
    role        = (state.get("role") or "Software Engineer").strip()
    context     = state.get("context_summary", "")
    company     = (state.get("company") or "").strip()
    int_type    = state.get("interview_type", "technical")

    # Skills: required = skills we still need to cover + already covered
    skills_rem  = list(state.get("skills_remaining", []))
    skills_cov  = list(state.get("skills_covered", []))
    required    = skills_rem[:6] + [s for s in skills_cov if s not in skills_rem][:2]
    good_to_have = skills_rem[6:9]
    if not required:
        required = ["Problem Solving", "Communication"]

    level, exp = _infer_level(context)

    jd_raw = _extract_jd_section(context)
    about_role = jd_raw[:250] if jd_raw else (
        f"Build and maintain production {int_type} systems as a {role}"
        + (f" at {company}" if company else "") + "."
    )

    # Key responsibilities from JD text or sensible defaults
    responsibilities: List[str] = []
    if jd_raw:
        # Look for bullet-like lines in the JD block
        for line in jd_raw.splitlines():
            line = line.strip().lstrip("-•*").strip()
            if 10 < len(line) < 120:
                responsibilities.append(line)
        responsibilities = responsibilities[:5]
    if not responsibilities:
        responsibilities = [
            f"Design and build {int_type} systems",
            "Write clean, tested, production-ready code",
            "Participate in code reviews",
            "Collaborate with cross-functional teams",
        ]

    # what_exceptional_looks_like — drives harder questions
    exceptional = (
        f"Candidate explains trade-offs, not just mechanics. "
        f"Has shipped {', '.join(required[:3])} at production scale. "
        f"Discusses failure modes, observability, and operational concerns proactively."
    )

    return {
        "job_title":                role,
        "level":                    level,
        "experience_required":      exp,
        "about_role":               about_role,
        "key_responsibilities":     responsibilities,
        "required_skills":          required,
        "good_to_have":             good_to_have,
        "what_exceptional_looks_like": exceptional,
    }


# ---------------------------------------------------------------------------
# Conversation history converter  →  [{question, answer}] format
# ---------------------------------------------------------------------------

def _to_qa_history(history: List[dict]) -> List[dict]:
    """Convert internal [{role, content}] to [{question, answer}] pairs."""
    pairs: List[dict] = []
    i = 0
    while i < len(history):
        msg = history[i]
        role = msg.get("role", "")
        if role in ("model", "interviewer"):
            q = msg.get("content", "").strip()
            a = ""
            if i + 1 < len(history) and history[i + 1].get("role") in ("user", "candidate"):
                a = history[i + 1].get("content", "").strip()
                i += 2
            else:
                i += 1
            if q:
                pairs.append({"question": q, "answer": a})
        else:
            i += 1
    return pairs


def _previous_score_floats(llama_scores: List[dict]) -> List[float]:
    """Extract plain final_score floats — what the evaluator API actually expects."""
    return [float(s.get("final_score", 5) or 5) for s in llama_scores]


# ---------------------------------------------------------------------------
# Rolling stats helpers
# ---------------------------------------------------------------------------

def _rolling_avg(scores: List[dict], window: int = 3) -> float:
    recent = [float(s.get("final_score", 5) or 5) for s in scores[-window:]]
    return round(sum(recent) / len(recent), 2) if recent else 5.0


def _classify_answer(text: str | None, score: float) -> str:
    if not text or not text.strip():
        return "confused"
    if len(text.strip().split()) <= 4:
        return "incomplete"
    return "genuine_answer"


def _build_perf_summary(prev: str, turn: int, topic: str, score: float,
                        strength: str, gap: str, rolling: float) -> str:
    entry = (f"T{turn} [{topic}] {score:.1f}/10 | "
             f"strength: {strength or 'N/A'} | gap: {gap or 'N/A'}")
    past = [l for l in prev.splitlines() if re.match(r"T\d+", l)][-6:]
    trend = ("↑ improving" if rolling >= 6.5
             else ("↓ declining" if rolling < 4.5 else "→ steady"))
    return f"Rolling avg: {rolling}/10 ({trend})\n" + "\n".join(past + [entry])


# ---------------------------------------------------------------------------
# Strategy planner
# ---------------------------------------------------------------------------

class _Strategy:
    __slots__ = (
        "difficulty", "active_topic", "topic_hint",
        "use_direct_followup", "direct_question",
        "question_type", "advance_topic",
        "ended", "end_reason",
        "answer_type", "answer_quality",
        "rolling", "last_score",
        "new_consec_poor", "new_consec_non",
    )

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def _plan(state: dict, evaluation: dict | None, last_input: str | None) -> _Strategy:
    turn         = int(state.get("turn_number", 0))
    all_scores   = list(state.get("llama_scores", []))
    if evaluation:
        all_scores = all_scores + [evaluation]

    rolling   = _rolling_avg(all_scores)
    last_score = float(evaluation.get("final_score", 5) or 5) if evaluation else 5.0

    # Adaptive difficulty
    if rolling >= _SCORE_HARD:
        difficulty = "hard"
    elif rolling <= _SCORE_EASY:
        difficulty = "easy"
    else:
        difficulty = "medium"

    # Answer classification
    answer_type    = _classify_answer(last_input, last_score) if last_input else "not_applicable"
    answer_quality = "not_applicable"
    if last_input and evaluation:
        answer_quality = ("strong" if last_score >= 7
                          else ("adequate" if last_score >= 5 else "weak"))

    # Consecutive poor tracking
    prev_poor    = int(state.get("_consec_poor", 0))
    max_retries  = int(state.get("max_confusion_retries", 2))
    new_consec_poor = (prev_poor + 1 if evaluation and last_score < _SCORE_ADVANCE else 0)
    new_consec_non  = (
        int(state.get("consecutive_non_answers", 0)) + 1
        if answer_type in {"confused", "refused", "off_topic", "wait_requested"}
        else 0
    )

    # Follow-up decision
    should_followup = (
        evaluation is not None
        and last_score < _SCORE_ADVANCE
        and new_consec_poor <= max_retries
        and bool(last_input)
        and answer_type not in {"refused", "end_requested"}
    )

    follow_up_q  = (evaluation.get("follow_up", "") or "").strip() if evaluation else ""
    use_direct   = should_followup and len(follow_up_q) > 15

    # Topic selection
    skills_rem   = list(state.get("skills_remaining", []))
    topic_counts = dict(state.get("topic_question_counts", {}))
    max_per_topic = int(state.get("max_questions_per_topic", 3) or 3)

    if should_followup:
        active_topic  = state.get("_active_topic") or (skills_rem[0] if skills_rem else "general")
        gap           = (evaluation.get("key_gap", "") or "") if evaluation else ""
        topic_hint    = f"{active_topic} — probe: {gap}" if gap else active_topic
        advance_topic = False
        question_type = "followup"
    else:
        # Advance to next un-exhausted skill
        active_topic = skills_rem[0] if skills_rem else state.get("role", "general")
        for skill in skills_rem:
            if topic_counts.get(skill, 0) < max_per_topic:
                active_topic = skill
                break
        topic_hint    = active_topic
        advance_topic = True
        question_type = "main"

    # End detection
    max_q    = int(state.get("max_questions", 0) or 0)
    ended    = max_q > 0 and (turn + 1) >= max_q
    end_reason = "max_questions_reached" if ended else ""

    # Persistent poor: 3 consecutive "no" signals after turn 5
    recent_signals = [s.get("hiring_signal", "yes") for s in all_scores[-3:]]
    if turn >= 5 and len(recent_signals) >= 3 and all(sig == "no" for sig in recent_signals):
        ended, end_reason = True, "persistent_poor_performance"

    if answer_type == "end_requested":
        ended, end_reason = True, "candidate_ended"

    return _Strategy(
        difficulty        = difficulty,
        active_topic      = active_topic,
        topic_hint        = topic_hint,
        use_direct_followup = use_direct,
        direct_question   = follow_up_q,
        question_type     = question_type,
        advance_topic     = advance_topic,
        ended             = ended,
        end_reason        = end_reason,
        answer_type       = answer_type,
        answer_quality    = answer_quality,
        rolling           = rolling,
        last_score        = last_score,
        new_consec_poor   = new_consec_poor,
        new_consec_non    = new_consec_non,
    )


# ---------------------------------------------------------------------------
# LlamaInterviewGraph
# ---------------------------------------------------------------------------

class LlamaInterviewGraph:
    """Schema-correct interview orchestrator for fine-tuned LLaMA models."""

    def __init__(self, api_key: str | None = None,
                 model_name: str | None = None,
                 client: Any | None = None) -> None:
        self._client   = client
        self.model_name = "llama-finetuned"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run_turn(self, state: dict) -> dict:
        if state.get("ended"):
            return state

        turn       = int(state.get("turn_number", 0))
        last_input: str | None = state.get("last_user_input") or None
        history    = list(state.get("history", []))

        # Build once per turn — reused in both evaluate & generate calls
        jd = state.get("llama_jd") or _build_jd(state)

        # 1. Evaluate
        evaluation: dict | None = None
        if last_input and turn > 0:
            evaluation = await self._evaluate(state, jd, last_input, history)

        # 2. Plan
        strategy = _plan(state, evaluation, last_input)

        # 3. Generate / select question
        if strategy.ended:
            question = ""
        elif strategy.use_direct_followup:
            question = strategy.direct_question
            print(f"[LlamaGraph] Direct follow-up: {question[:80]}…")
        else:
            question = await self._generate(state, jd, history, last_input, strategy, turn)

        # 4. Frame
        question = self._frame(state, question, turn, strategy.ended)

        # 5. Build updated state
        return self._build_state(state, jd, history, question, evaluation, strategy, turn, last_input)

    # ------------------------------------------------------------------
    # Evaluate  — schema-correct evaluator call
    # ------------------------------------------------------------------

    async def _evaluate(self, state: dict, jd: dict,
                        answer: str, history: List[dict]) -> dict:
        # Find the last question we asked
        last_q = next(
            (m["content"] for m in reversed(history)
             if m.get("role") in ("model", "interviewer")),
            "",
        )
        if not last_q:
            return {}

        prev_scores = _previous_score_floats(state.get("llama_scores", []))[-5:]
        q_type      = "followup" if not state.get("_advance_topic", True) else "main"

        try:
            result = await _post("/evaluator/score", {
                "jd":              jd,
                "question":        last_q,
                "answer":          answer,
                "round_type":      state.get("interview_type", "technical"),
                "difficulty":      state.get("_last_difficulty", "medium"),
                "question_type":   q_type,
                "previous_scores": prev_scores,
            })
            print(f"[LlamaGraph] Eval: score={result.get('final_score')} "
                  f"hire={result.get('hiring_signal')} "
                  f"gap={str(result.get('key_gap', ''))[:60]}")
            return result
        except Exception as exc:
            print(f"[LlamaGraph] Evaluator fallback ({exc})")
            return {"content_score": 5, "final_score": 5.0,
                    "key_strength": "", "key_gap": "", "feedback": "",
                    "follow_up": "", "hiring_signal": "yes"}

    # ------------------------------------------------------------------
    # Generate  — schema-correct interviewer call
    # ------------------------------------------------------------------

    async def _generate(self, state: dict, jd: dict,
                        history: List[dict], last_input: str | None,
                        strategy: _Strategy, turn: int) -> str:
        # Build Q&A history including the current answer
        qa_history = _to_qa_history(history)
        candidate_answer = last_input or ""

        # Duration from state or sensible default
        max_q = int(state.get("max_questions", 0) or 0)
        duration = f"{max(15, max_q * 3)} minutes" if max_q else "30 minutes"

        try:
            result = await _post("/interviewer/question", {
                "jd":                  jd,
                "round_type":          state.get("interview_type", "technical"),
                "difficulty":          strategy.difficulty,
                "duration":            duration,
                "style":               "direct",
                "conversation_history": qa_history,
                "candidate_answer":    candidate_answer,
            })
            q = (result.get("question") or "").strip()
            if q:
                return q
            raise ValueError("Empty question")
        except Exception as exc:
            print(f"[LlamaGraph] Interviewer fallback ({exc})")
            topic = strategy.active_topic or state.get("role", "the role")
            return f"Could you walk me through your experience with {topic} in more detail?"

    # ------------------------------------------------------------------
    # Framing
    # ------------------------------------------------------------------

    @staticmethod
    def _frame(state: dict, question: str, turn: int, ended: bool) -> str:
        opening = (state.get("opening_line") or "").strip()
        closing = (state.get("closing_line") or "").strip()
        if ended:
            return closing or (
                "Thank you so much for your time today — that wraps up our session. "
                "We'll be in touch with detailed feedback soon!"
            )
        if turn == 0 and opening:
            return f"{opening} {question}".strip() if question else opening
        return question

    # ------------------------------------------------------------------
    # State builder
    # ------------------------------------------------------------------

    def _build_state(self, state: dict, jd: dict, history: List[dict],
                     question: str, evaluation: dict | None,
                     strategy: _Strategy, turn: int,
                     last_input: str | None) -> dict:

        # Skills
        skills_rem = list(state.get("skills_remaining", []))
        skills_cov = list(state.get("skills_covered", []))
        if strategy.advance_topic and strategy.active_topic in skills_rem:
            skills_rem = [s for s in skills_rem if s != strategy.active_topic]
            if strategy.active_topic not in skills_cov:
                skills_cov.append(strategy.active_topic)

        # Scores accumulation
        all_scores = list(state.get("llama_scores", []))
        if evaluation:
            all_scores.append(evaluation)

        # Sync to LlamaClient so generate_feedback can use them
        if self._client is not None and hasattr(self._client, "_llama_scores"):
            self._client._llama_scores = all_scores

        # Performance summary (cumulative log)
        perf = state.get("performance_summary", "")
        if evaluation:
            perf = _build_perf_summary(
                perf, turn, strategy.active_topic,
                strategy.last_score,
                evaluation.get("key_strength", ""),
                evaluation.get("key_gap", ""),
                strategy.rolling,
            )

        # History
        new_history = list(history)
        if last_input:
            new_history.append({"role": "user",  "content": last_input})
        new_history.append({"role": "model", "content": question})

        # Topic counts
        topic_counts = dict(state.get("topic_question_counts", {}))
        topic_counts[strategy.active_topic] = topic_counts.get(strategy.active_topic, 0) + 1

        questions_asked = list(state.get("questions_asked", []))
        questions_asked.append(question)

        current_evaluation = {
            "next_step": {
                "question":          question,
                "is_coding_question": False,
                "target_skill":      strategy.active_topic,
                "type":              strategy.question_type,
                "difficulty":        strategy.difficulty,
            },
            "decision": {
                "action":           "end" if strategy.ended else "continue",
                "reason":           strategy.end_reason or "continuing",
                "termination_flag": strategy.ended,
            },
            "last_answer_evaluation": evaluation,
            "answer_type":          strategy.answer_type,
            "answer_quality":       strategy.answer_quality,
            "performance_summary":  perf,
            "newly_covered_skills": (
                [strategy.active_topic]
                if strategy.advance_topic and strategy.active_topic in skills_cov
                else []
            ),
            "should_follow_up":     not strategy.advance_topic,
            "follow_up_hint":       (evaluation.get("follow_up", "") if evaluation else ""),
            "confidence_in_candidate": (
                "high" if strategy.rolling >= 7
                else ("low" if strategy.rolling < 4 else "medium")
            ),
        }

        return {
            **state,
            "history":               new_history,
            "performance_summary":   perf,
            "skills_remaining":      skills_rem,
            "skills_covered":        skills_cov,
            "questions_asked":       questions_asked,
            "current_question":      question,
            "current_evaluation":    current_evaluation,
            "follow_up_hint":        (evaluation.get("follow_up", "") if evaluation else ""),
            "turn_number":           turn + 1,
            "last_answer_type":      strategy.answer_type,
            "consecutive_non_answers": strategy.new_consec_non,
            "consecutive_disengaged": (
                int(state.get("consecutive_disengaged", 0)) + 1
                if strategy.answer_type in {"refused", "off_topic"} else 0
            ),
            "ended":      strategy.ended,
            "end_reason": strategy.end_reason,
            "llama_jd":   jd,
            "llama_scores": all_scores,
            "input_tokens":  int(state.get("input_tokens", 0)),
            "output_tokens": int(state.get("output_tokens", 0)),
            "topic_question_counts": topic_counts,
            # Internal tracking keys (prefixed _ so they don't clash with typed state)
            "_active_topic":    strategy.active_topic,
            "_consec_poor":     strategy.new_consec_poor,
            "_last_difficulty": strategy.difficulty,
            "_advance_topic":   strategy.advance_topic,
        }

    @staticmethod
    def _bump_topic_counts(state: dict, target_skill: str) -> Dict[str, int]:
        counts = dict(state.get("topic_question_counts", {}))
        if target_skill:
            counts[target_skill] = counts.get(target_skill, 0) + 1
        return counts
