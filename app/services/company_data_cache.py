"""Company data cache for interview planning artifacts.

This cache stores one-time planning outputs so repeated starts with the same
resume/JD/company/role/interview_type reuse the generated plan and avoid
recomputation.
"""

import hashlib
import importlib
import json
from typing import Any

try:
    redis = importlib.import_module("redis.asyncio")
except Exception:  # pragma: no cover - dependency may be unavailable in some dev environments
    redis = None

from app.core.config import settings

COMPANY_PLAN_CACHE_PREFIX = "company_plan:"
COMPANY_PLAN_CACHE_TTL = 86_400  # 24 hours


class CompanyDataCache:
    def __init__(self) -> None:
        self._redis: Any | None = None

    async def _get_client(self) -> Any | None:
        if redis is None:
            return None
        if self._redis is None:
            self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
        return self._redis

    @staticmethod
    def _make_key(
        *,
        resume_text: str,
        jd_text: str,
        company_name: str,
        role: str,
        interview_type: str,
        duration_minutes: int,
    ) -> str:
        raw = (
            f"{resume_text.strip()}|{jd_text.strip()}|{company_name.strip().lower()}|"
            f"{role.strip().lower()}|{interview_type.strip().lower()}|{duration_minutes}"
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{COMPANY_PLAN_CACHE_PREFIX}{digest}"

    async def get_plan(
        self,
        *,
        resume_text: str,
        jd_text: str,
        company_name: str,
        role: str,
        interview_type: str,
        duration_minutes: int,
    ) -> dict[str, Any] | None:
        try:
            client = await self._get_client()
            if client is None:
                return None
            key = self._make_key(
                resume_text=resume_text,
                jd_text=jd_text,
                company_name=company_name,
                role=role,
                interview_type=interview_type,
                duration_minutes=duration_minutes,
            )
            raw = await client.get(key)
            if raw:
                print(f"[CompanyDataCache] cache hit: {key[:36]}...")
                return json.loads(raw)
            print(f"[CompanyDataCache] cache miss: {key[:36]}...")
        except Exception as exc:
            print(f"[CompanyDataCache] get failed: {exc}")
        return None

    async def set_plan(
        self,
        *,
        resume_text: str,
        jd_text: str,
        company_name: str,
        role: str,
        interview_type: str,
        duration_minutes: int,
        plan: dict[str, Any],
    ) -> None:
        try:
            client = await self._get_client()
            if client is None:
                return
            key = self._make_key(
                resume_text=resume_text,
                jd_text=jd_text,
                company_name=company_name,
                role=role,
                interview_type=interview_type,
                duration_minutes=duration_minutes,
            )
            await client.set(key, json.dumps(plan), ex=COMPANY_PLAN_CACHE_TTL)
            print(f"[CompanyDataCache] stored: {key[:36]}... ttl={COMPANY_PLAN_CACHE_TTL}")
        except Exception as exc:
            print(f"[CompanyDataCache] set failed: {exc}")

    async def invalidate_plan(
        self,
        *,
        resume_text: str,
        jd_text: str,
        company_name: str,
        role: str,
        interview_type: str,
        duration_minutes: int,
    ) -> None:
        try:
            client = await self._get_client()
            if client is None:
                return
            key = self._make_key(
                resume_text=resume_text,
                jd_text=jd_text,
                company_name=company_name,
                role=role,
                interview_type=interview_type,
                duration_minutes=duration_minutes,
            )
            await client.delete(key)
        except Exception as exc:
            print(f"[CompanyDataCache] invalidate failed: {exc}")

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None


company_data_cache = CompanyDataCache()
