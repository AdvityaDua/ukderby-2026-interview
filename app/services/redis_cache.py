"""
Redis cache service for persisting interview session state.
Sessions survive WebSocket disconnects/page refreshes until
the interview formally ends or the TTL expires.
"""

import json
import redis.asyncio as redis
from app.core.config import settings

SESSION_TTL = 7200  # 2 hours in seconds
SESSION_PREFIX = "interview_session:"


class RedisCache:
    def __init__(self):
        self._redis: redis.Redis | None = None

    async def _get_client(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
            )
        return self._redis

    async def save_session(self, user_id: str, session_data: dict) -> None:
        """Persist session state to Redis with TTL."""
        r = await self._get_client()
        key = f"{SESSION_PREFIX}{user_id}"
        await r.set(key, json.dumps(session_data), ex=SESSION_TTL)

    async def load_session(self, user_id: str) -> dict | None:
        """Load session state from Redis. Returns None if not found."""
        r = await self._get_client()
        key = f"{SESSION_PREFIX}{user_id}"
        data = await r.get(key)
        if data:
            return json.loads(data)
        return None

    async def delete_session(self, user_id: str) -> None:
        """Delete session from Redis (interview ended)."""
        r = await self._get_client()
        key = f"{SESSION_PREFIX}{user_id}"
        await r.delete(key)

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


redis_cache = RedisCache()
