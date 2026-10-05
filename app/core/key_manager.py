"""
key_manager.py — Fetches active Gemini / Groq keys from the NestJS admin API.
Falls back silently to local env vars if the backend is unreachable.
Auto-refreshes every REFRESH_INTERVAL seconds so admin key changes take effect
without restarting the Python service.
"""
import os
import asyncio
import logging
import httpx

logger = logging.getLogger(__name__)

REFRESH_INTERVAL = 60  # seconds between background re-fetches


class KeyManager:
    """Singleton that holds the resolved API keys and active model names for Gemini and Groq."""

    def __init__(self):
        self._gemini: str | None = None
        self._groq: str | None = None
        self._gemini_model: str | None = None
        self._groq_model: str | None = None
        self._backend_url: str | None = None
        self._task: asyncio.Task | None = None

    async def refresh(self, backend_url: str) -> None:
        """Fetch active keys from NestJS /ai-config/keys/internal (no auth required)."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(f"{backend_url}/ai-config/keys/internal")
                if r.status_code == 200:
                    data = r.json()
                    new_gemini = data.get("gemini")
                    new_groq   = data.get("groq")
                    if new_gemini:
                        self._gemini = new_gemini
                    if new_groq:
                        self._groq = new_groq
                    # Model names (optional fields — may not exist on older backends)
                    new_gm = data.get("geminiModel")
                    new_rm = data.get("groqModel")
                    if new_gm:
                        self._gemini_model = new_gm
                    if new_rm:
                        self._groq_model = new_rm
                    logger.info("[KeyManager] Keys refreshed from backend")
                else:
                    logger.warning(f"[KeyManager] Backend returned {r.status_code}, using cached/env fallback")
        except Exception as e:
            logger.warning(f"[KeyManager] Could not reach backend ({type(e).__name__}: {e}), using cached/env fallback")

    async def _auto_refresh_loop(self) -> None:
        """Background task: re-fetch keys every REFRESH_INTERVAL seconds."""
        while True:
            await asyncio.sleep(REFRESH_INTERVAL)
            if self._backend_url:
                await self.refresh(self._backend_url)

    def start_auto_refresh(self, backend_url: str) -> None:
        """Start the background refresh loop (call once from lifespan startup)."""
        self._backend_url = backend_url
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._auto_refresh_loop())
            logger.info(f"[KeyManager] Auto-refresh started (every {REFRESH_INTERVAL}s)")

    def stop_auto_refresh(self) -> None:
        """Cancel the background task (call from lifespan shutdown)."""
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("[KeyManager] Auto-refresh stopped")

    def get_gemini_key(self) -> str:
        return self._gemini or os.getenv("GOOGLE_API_KEY", "")

    def get_groq_key(self) -> str:
        return self._groq or os.getenv("GROQ_API_KEY", "")

    def get_gemini_model(self) -> str:
        """Returns the admin-configured Gemini model, falling back to the cost-effective default."""
        return self._gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def get_groq_model(self) -> str:
        """Returns the admin-configured Groq model, falling back to the cost-effective default."""
        return self._groq_model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# Global singleton
key_manager = KeyManager()
