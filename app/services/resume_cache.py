"""
Resume Context Cache
====================
Caches the expensive resume+JD analysis (summarize_context + skills extraction)
in Redis, keyed by a hash of (resume_text + jd_text + interview_type + role + company).

This means: if the user takes multiple rounds (technical → behavioral → HR) with the
same resume and JD, the analysis is only performed ONCE. Subsequent rounds hit Redis
and skip the 2 LLM calls entirely, saving ~2,000–5,000 input tokens per round.

Cache TTL: 24 hours (86,400 seconds) — long enough to cover a full interview day
           but short enough to respect updated resumes.
"""

import hashlib
import json
import redis.asyncio as redis
from app.core.config import settings

RESUME_CACHE_PREFIX = "resume_ctx:"
RESUME_CACHE_TTL    = 86_400        # 24 hours


class ResumeContextCache:
    def __init__(self):
        self._redis: redis.Redis | None = None

    async def _get_client(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
        return self._redis

    # ── Key derivation ──────────────────────────────────────────────────────────
    @staticmethod
    def _make_key(
        resume_text: str,
        jd_text: str,
        interview_type: str,
        role: str,
        company: str,
    ) -> str:
        """SHA-256 fingerprint of the inputs → unique cache key."""
        raw = f"{resume_text.strip()}|{jd_text.strip()}|{interview_type}|{role.lower()}|{company.lower()}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{RESUME_CACHE_PREFIX}{digest}"

    # ── Public API ──────────────────────────────────────────────────────────────
    async def get(
        self,
        resume_text: str,
        jd_text: str,
        interview_type: str,
        role: str,
        company: str,
    ) -> dict | None:
        """
        Return cached {"context_summary": str, "skills": List[str]} or None.
        """
        try:
            r   = await self._get_client()
            key = self._make_key(resume_text, jd_text, interview_type, role, company)
            raw = await r.get(key)
            if raw:
                print(f"[ResumeCache] ✅ Cache HIT  → {key[:32]}…")
                return json.loads(raw)
            print(f"[ResumeCache] ⚠️  Cache MISS → {key[:32]}…")
        except Exception as e:
            print(f"[ResumeCache] Redis error on GET: {e}")
        return None

    async def set(
        self,
        resume_text: str,
        jd_text: str,
        interview_type: str,
        role: str,
        company: str,
        context_summary: str,
        skills: list,
    ) -> None:
        """
        Persist the context summary + skill list to Redis.
        """
        try:
            r    = await self._get_client()
            key  = self._make_key(resume_text, jd_text, interview_type, role, company)
            data = json.dumps({"context_summary": context_summary, "skills": skills})
            await r.set(key, data, ex=RESUME_CACHE_TTL)
            print(f"[ResumeCache] 💾 Stored → {key[:32]}… (TTL={RESUME_CACHE_TTL}s)")
        except Exception as e:
            print(f"[ResumeCache] Redis error on SET: {e}")

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None


# Singleton — imported everywhere that needs resume caching
resume_cache = ResumeContextCache()
