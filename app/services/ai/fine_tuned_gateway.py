"""Fine-tuned LLM gateway client.

Async client for the self-hosted vLLM interviewer/evaluator endpoints.
Uses httpx with configurable timeout, simple retry, robust JSON extraction,
and automatic fallback signaling on any failure.

Gateway endpoints:
    POST /interviewer  — question generation
    POST /evaluator    — answer evaluation
    GET  /health       — gateway liveness
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[dict]:
    """Extract a JSON object from raw text.

    Handles:
      - Pure JSON strings
      - Markdown-wrapped ```json ... ``` blocks
      - JSON embedded in surrounding prose
    Returns None if no valid JSON object is found.
    """
    if not text or not text.strip():
        return None

    stripped = text.strip()

    # 1. Try direct parse
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Try markdown code-fence extraction
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if md_match:
        try:
            parsed = json.loads(md_match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. Try to find the first { ... } block
    brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Gateway client
# ─────────────────────────────────────────────────────────────────────────────

class FineTunedGateway:
    """Async client for the fine-tuned LLM gateway."""

    # Retry configuration
    MAX_RETRIES = 2
    RETRY_BACKOFF_SECONDS = 1.5

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = (base_url or settings.FINE_TUNED_GATEWAY_URL).rstrip("/")
        self.timeout = timeout

    # ── Core request method ──────────────────────────────────────────────────

    async def _request(
        self,
        endpoint: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Optional[dict]:
        """Send a chat-completion-style request to the gateway.

        Returns the parsed JSON dict on success, or None on failure
        (caller should fallback to Gemini).
        """
        url = f"{self.base_url}{endpoint}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)

                duration_ms = int((time.monotonic() - start) * 1000)

                if response.status_code != 200:
                    logger.warning(
                        "[FineTunedGateway] %s returned HTTP %d (attempt %d/%d, %dms): %s",
                        endpoint, response.status_code, attempt, self.MAX_RETRIES,
                        duration_ms, response.text[:300],
                    )
                    last_error = Exception(f"HTTP {response.status_code}")
                    if attempt < self.MAX_RETRIES:
                        import asyncio
                        await asyncio.sleep(self.RETRY_BACKOFF_SECONDS * attempt)
                    continue

                # Parse the response body
                raw_text = response.text
                # The gateway may return OpenAI-compatible format:
                #   { "choices": [{ "message": { "content": "..." } }] }
                # Or it may return the content directly as JSON.
                body = None
                try:
                    body = response.json()
                except json.JSONDecodeError:
                    pass

                content_text = raw_text
                if isinstance(body, dict):
                    # OpenAI-compatible: extract from choices[0].message.content
                    choices = body.get("choices", [])
                    if choices and isinstance(choices, list):
                        message = choices[0].get("message", {})
                        content_text = message.get("content", raw_text)

                result = _extract_json(content_text)
                if result is None:
                    logger.warning(
                        "[FineTunedGateway] %s JSON extraction failed (attempt %d/%d, %dms). "
                        "Raw response (first 500 chars): %s",
                        endpoint, attempt, self.MAX_RETRIES, duration_ms,
                        content_text[:500],
                    )
                    last_error = Exception("JSON extraction failed")
                    if attempt < self.MAX_RETRIES:
                        import asyncio
                        await asyncio.sleep(self.RETRY_BACKOFF_SECONDS * attempt)
                    continue

                logger.info(
                    "[FineTunedGateway] %s OK (model=%s, %dms, attempt %d)",
                    endpoint, model, duration_ms, attempt,
                )
                return result

            except httpx.TimeoutException as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                logger.warning(
                    "[FineTunedGateway] %s timeout after %dms (attempt %d/%d): %s",
                    endpoint, duration_ms, attempt, self.MAX_RETRIES, exc,
                )
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    import asyncio
                    await asyncio.sleep(self.RETRY_BACKOFF_SECONDS * attempt)

            except Exception as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                logger.warning(
                    "[FineTunedGateway] %s error after %dms (attempt %d/%d): %s: %s",
                    endpoint, duration_ms, attempt, self.MAX_RETRIES,
                    type(exc).__name__, exc,
                )
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    import asyncio
                    await asyncio.sleep(self.RETRY_BACKOFF_SECONDS * attempt)

        logger.error(
            "[FineTunedGateway] %s FAILED after %d attempts. Last error: %s. "
            "Falling back to Gemini.",
            endpoint, self.MAX_RETRIES, last_error,
        )
        return None

    # ── Public interface ─────────────────────────────────────────────────────

    async def generate_interviewer_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Optional[dict]:
        """Call the /interviewer endpoint for question generation.

        Returns parsed JSON dict on success, None on failure (triggers Gemini fallback).
        """
        return await self._request(
            endpoint="/interviewer",
            model="interviewer",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def generate_evaluator_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.25,
        max_tokens: int = 1024,
    ) -> Optional[dict]:
        """Call the /evaluator endpoint for answer evaluation.

        Returns parsed JSON dict on success, None on failure (triggers Gemini fallback).
        """
        return await self._request(
            endpoint="/evaluator",
            model="evaluator",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def health_check(self) -> bool:
        """Non-blocking health check. Returns True if gateway is reachable."""
        url = f"{self.base_url}/health"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                is_healthy = response.status_code == 200
                if is_healthy:
                    logger.info("[FineTunedGateway] Health check OK: %s", url)
                else:
                    logger.warning(
                        "[FineTunedGateway] Health check returned HTTP %d: %s",
                        response.status_code, url,
                    )
                return is_healthy
        except Exception as exc:
            logger.warning(
                "[FineTunedGateway] Health check failed for %s: %s: %s",
                url, type(exc).__name__, exc,
            )
            return False


# ── Module-level singleton ───────────────────────────────────────────────────
# Import this from other modules:
#   from app.services.ai.fine_tuned_gateway import gateway_client

gateway_client = FineTunedGateway()
