"""Redis-backed session state store for LangGraph interview sessions.

This service implements the contract in session_state_contract.py and provides
load/save helpers for reconnect-safe interview continuity.
"""

from __future__ import annotations

import importlib
import json
import asyncio
from typing import Any, Dict

from app.core.config import settings
from app.services.ai.langgraph_agent import AgentState, build_session_persistence_payload
from app.services.session_state_contract import (
    PersistedInterviewSession,
    active_session_key_for_user,
    mark_record_heartbeat,
    mark_record_status,
    session_state_key,
    session_turn_lock_key,
    user_question_history_key,
)


DEFAULT_SESSION_STATE_TTL = 7_200  # 2 hours


try:
    redis = importlib.import_module("redis.asyncio")
except Exception:  # pragma: no cover - dependency may be unavailable in some environments
    redis = None


class SessionStateStore:
    def __init__(self, ttl_seconds: int = DEFAULT_SESSION_STATE_TTL) -> None:
        self._redis: Any | None = None
        self.ttl_seconds = ttl_seconds
        self._local_turn_locks: dict[str, asyncio.Lock] = {}

    async def _get_client(self) -> Any | None:
        if redis is None:
            return None
        if self._redis is None:
            self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
        return self._redis

    async def save_record(self, record: PersistedInterviewSession) -> None:
        """Persist a canonical session record and user active-session pointer."""
        client = await self._get_client()
        if client is None:
            return

        sid_key = session_state_key(record["session_id"])
        user_key = active_session_key_for_user(record["user_id"])
        payload = json.dumps(record)
        ttl = int(record.get("ttl_seconds", self.ttl_seconds) or self.ttl_seconds)

        await client.set(sid_key, payload, ex=ttl)
        await client.set(user_key, record["session_id"], ex=ttl)

    async def save_agent_state(
        self,
        *,
        user_id: str,
        client_id: str,
        state: AgentState,
        ttl_seconds: int | None = None,
        accounting_source: str = "graph_state_cumulative",
        telemetry: Dict[str, Any] | None = None,
    ) -> PersistedInterviewSession:
        """Build and persist canonical record from runtime AgentState."""
        effective_ttl = int(ttl_seconds or self.ttl_seconds)
        previous_record = await self.load_record(state["session_id"])
        record = build_session_persistence_payload(
            user_id=user_id,
            client_id=client_id,
            state=state,
            ttl_seconds=effective_ttl,
            previous_token_summary=(previous_record or {}).get("token_summary", {}),
            accounting_source=accounting_source,
            telemetry=telemetry,
        )
        await self.save_record(record)
        return record

    async def load_record(self, session_id: str) -> PersistedInterviewSession | None:
        """Load persisted record by session_id."""
        client = await self._get_client()
        if client is None:
            return None

        sid_key = session_state_key(session_id)
        raw = await client.get(sid_key)
        if not raw:
            return None
        return json.loads(raw)

    async def load_active_record_for_user(self, user_id: str) -> PersistedInterviewSession | None:
        """Load active session record for a user via user->session mapping."""
        client = await self._get_client()
        if client is None:
            return None

        user_key = active_session_key_for_user(user_id)
        session_id = await client.get(user_key)
        if not session_id:
            return None
        return await self.load_record(session_id)

    async def touch_session(self, session_id: str) -> PersistedInterviewSession | None:
        """Refresh heartbeat and TTL for an existing record."""
        client = await self._get_client()
        if client is None:
            return None

        record = await self.load_record(session_id)
        if not record:
            return None

        record = mark_record_heartbeat(record)
        ttl = int(record.get("ttl_seconds", self.ttl_seconds) or self.ttl_seconds)

        sid_key = session_state_key(session_id)
        user_key = active_session_key_for_user(record["user_id"])
        await client.set(sid_key, json.dumps(record), ex=ttl)
        await client.set(user_key, session_id, ex=ttl)
        return record

    async def mark_ended(self, session_id: str) -> PersistedInterviewSession | None:
        """Mark a session as ended and persist the updated status."""
        record = await self.load_record(session_id)
        if not record:
            return None

        record = mark_record_status(record, "ended")
        await self.save_record(record)
        return record

    async def clear_session(
        self,
        *,
        session_id: str,
        user_id: str | None = None,
    ) -> None:
        """Delete session record and user pointer."""
        client = await self._get_client()
        if client is None:
            return

        resolved_user_id = user_id
        if resolved_user_id is None:
            existing = await self.load_record(session_id)
            if existing:
                resolved_user_id = existing.get("user_id")

        sid_key = session_state_key(session_id)
        await client.delete(sid_key)

        if resolved_user_id:
            user_key = active_session_key_for_user(resolved_user_id)
            await client.delete(user_key)

    async def load_user_question_history(self, user_id: str) -> list[str]:
        client = await self._get_client()
        if client is None:
            return []

        raw = await client.get(user_question_history_key(user_id))
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except Exception:
            return []
        return [str(item) for item in data if str(item)]

    async def append_user_question_history(
        self,
        user_id: str,
        question_hashes: list[str],
        *,
        ttl_seconds: int = 30 * 24 * 60 * 60,
    ) -> None:
        if not question_hashes:
            return

        client = await self._get_client()
        if client is None:
            return

        key = user_question_history_key(user_id)
        existing = await self.load_user_question_history(user_id)
        merged = list(dict.fromkeys([*existing, *[item for item in question_hashes if item]]))
        await client.set(key, json.dumps(merged), ex=ttl_seconds)

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def acquire_turn_lock(
        self,
        *,
        session_id: str,
        owner: str,
        lock_ttl_seconds: int = 30,
    ) -> bool:
        """Acquire per-session turn lock. Returns False if already locked."""
        client = await self._get_client()
        if client is not None:
            key = session_turn_lock_key(session_id)
            result = await client.set(key, owner, ex=lock_ttl_seconds, nx=True)
            return bool(result)

        # Local fallback for environments without Redis client
        lock = self._local_turn_locks.setdefault(session_id, asyncio.Lock())
        if lock.locked():
            return False
        await lock.acquire()
        return True

    async def release_turn_lock(self, *, session_id: str, owner: str) -> None:
        """Release per-session turn lock if held by owner."""
        client = await self._get_client()
        if client is not None:
            key = session_turn_lock_key(session_id)
            current_owner = await client.get(key)
            if current_owner == owner:
                await client.delete(key)
            return

        lock = self._local_turn_locks.get(session_id)
        if lock and lock.locked():
            lock.release()


session_state_store = SessionStateStore()
