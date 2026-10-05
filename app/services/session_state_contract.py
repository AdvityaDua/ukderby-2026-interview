"""Session state persistence contract for Redis.

This module defines the canonical schema and key strategy for persisting
LangGraph interview sessions so reconnects can resume safely.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Literal, TypedDict


SESSION_STATE_VERSION = 1
SESSION_KEY_PREFIX = "interview:state:"
USER_ACTIVE_SESSION_PREFIX = "interview:user_active:"
SESSION_TURN_LOCK_PREFIX = "interview:turn_lock:"
USER_QUESTION_HISTORY_PREFIX = "interview:user_questions:"


class PersistedAgentMeta(TypedDict):
    current_node: str
    next_action: str
    tool_calls_count: int
    created_at: str
    last_updated_at: str


class PersistedTokenSummary(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    last_turn_input_tokens: int
    last_turn_output_tokens: int
    last_turn_total_tokens: int
    accounting_source: str


class PersistedInterviewSession(TypedDict):
    schema_version: int
    session_id: str
    user_id: str
    client_id: str
    status: Literal["active", "ended", "expired"]
    created_at: str
    updated_at: str
    heartbeat_at: str
    ttl_seconds: int
    interview_state: Dict[str, Any]
    agent_meta: PersistedAgentMeta
    token_summary: PersistedTokenSummary
    telemetry: Dict[str, Any]


def session_state_key(session_id: str) -> str:
    return f"{SESSION_KEY_PREFIX}{session_id}"


def active_session_key_for_user(user_id: str) -> str:
    return f"{USER_ACTIVE_SESSION_PREFIX}{user_id}"


def session_turn_lock_key(session_id: str) -> str:
    return f"{SESSION_TURN_LOCK_PREFIX}{session_id}"


def user_question_history_key(user_id: str) -> str:
    return f"{USER_QUESTION_HISTORY_PREFIX}{user_id}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_persisted_session_record(
    *,
    session_id: str,
    user_id: str,
    client_id: str,
    interview_state: Dict[str, Any],
    agent_meta: Dict[str, Any],
    ttl_seconds: int,
    status: Literal["active", "ended", "expired"] = "active",
    previous_token_summary: Dict[str, Any] | None = None,
    accounting_source: str = "graph_state_cumulative",
    telemetry: Dict[str, Any] | None = None,
) -> PersistedInterviewSession:
    now = utc_now_iso()

    input_tokens = int(interview_state.get("input_tokens", 0) or 0)
    output_tokens = int(interview_state.get("output_tokens", 0) or 0)
    prev_input = int((previous_token_summary or {}).get("input_tokens", 0) or 0)
    prev_output = int((previous_token_summary or {}).get("output_tokens", 0) or 0)
    delta_input = max(0, input_tokens - prev_input)
    delta_output = max(0, output_tokens - prev_output)

    return {
        "schema_version": SESSION_STATE_VERSION,
        "session_id": session_id,
        "user_id": user_id,
        "client_id": client_id,
        "status": status,
        "created_at": str(agent_meta.get("created_at", now)),
        "updated_at": str(agent_meta.get("last_updated_at", now)),
        "heartbeat_at": now,
        "ttl_seconds": int(ttl_seconds),
        "interview_state": interview_state,
        "agent_meta": {
            "current_node": str(agent_meta.get("current_node", "START")),
            "next_action": str(agent_meta.get("next_action", "continue")),
            "tool_calls_count": int(agent_meta.get("tool_calls_count", 0) or 0),
            "created_at": str(agent_meta.get("created_at", now)),
            "last_updated_at": str(agent_meta.get("last_updated_at", now)),
        },
        "token_summary": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "last_turn_input_tokens": delta_input,
            "last_turn_output_tokens": delta_output,
            "last_turn_total_tokens": delta_input + delta_output,
            "accounting_source": accounting_source,
        },
        "telemetry": telemetry or {},
    }


def mark_record_heartbeat(record: PersistedInterviewSession) -> PersistedInterviewSession:
    now = utc_now_iso()
    record["heartbeat_at"] = now
    record["updated_at"] = now
    record["agent_meta"]["last_updated_at"] = now
    return record


def mark_record_status(
    record: PersistedInterviewSession,
    status: Literal["active", "ended", "expired"],
) -> PersistedInterviewSession:
    now = utc_now_iso()
    record["status"] = status
    record["updated_at"] = now
    record["agent_meta"]["last_updated_at"] = now
    return record
