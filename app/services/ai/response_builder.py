"""Utilities for building interview turn responses.

This module is intentionally framework-agnostic so routing and termination
behavior can be unit tested without FastAPI dependencies.
"""

from __future__ import annotations

from typing import Any


def _extract_is_coding_question(current_eval: Any) -> bool:
    if isinstance(current_eval, dict):
        next_step = current_eval.get("next_step") or {}
        if isinstance(next_step, dict):
            return bool(next_step.get("is_coding_question", False))
        return bool(getattr(next_step, "is_coding_question", False))

    next_step = getattr(current_eval, "next_step", None)
    if isinstance(next_step, dict):
        return bool(next_step.get("is_coding_question", False))
    return bool(getattr(next_step, "is_coding_question", False))


def build_turn_response(
    session_id: str,
    state: dict[str, Any],
    plan: dict[str, Any] | None = None,
    token_summary: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_tokens = int(state.get("input_tokens", 0) or 0)
    output_tokens = int(state.get("output_tokens", 0) or 0)
    persisted = token_summary or {}

    is_coding = False
    current_eval = state.get("current_evaluation")
    if current_eval is not None:
        is_coding = _extract_is_coding_question(current_eval)

    coding_ux = {
        "open_editor": is_coding,
        "close_editor": bool(state.get("ended", False)) and not is_coding,
        "question_mode": "coding" if is_coding else "verbal",
        "show_coding_badge": is_coding,
        "allow_code_submission": is_coding,
        "suppress_auto_close": is_coding,
        "hint": (
            "Open the editor for this question." if is_coding else "No editor needed for this question."
        ),
    }

    return {
        "session_id": session_id,
        "question": state.get("current_question", ""),
        "ended": bool(state.get("ended", False)),
        "end_reason": state.get("end_reason", ""),
        "turn_number": int(state.get("turn_number", 0) or 0),
        "is_coding_question": is_coding,
        "tokens": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "last_turn_input_tokens": int(persisted.get("last_turn_input_tokens", 0) or 0),
            "last_turn_output_tokens": int(persisted.get("last_turn_output_tokens", 0) or 0),
            "last_turn_total_tokens": int(persisted.get("last_turn_total_tokens", 0) or 0),
            "accounting_source": str(persisted.get("accounting_source", "graph_state_cumulative")),
        },
        "telemetry": telemetry or {},
        "coding_ux": coding_ux,
        "plan": {
            "topics": (plan or {}).get("topics", []),
            "strategy": (plan or {}).get("question_strategy", {}),
            "cache_hit": ((plan or {}).get("metadata") or {}).get("cache_hit", False),
            "source_evidence": (plan or {}).get("source_evidence", []),
            "opening_line": (plan or {}).get("opening_line", ""),
            "closing_line": (plan or {}).get("closing_line", ""),
            "question_candidates": (plan or {}).get("question_candidates", []),
        }
        if plan
        else None,
    }
