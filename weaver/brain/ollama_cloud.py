# ollama_cloud.py — Ollama Cloud LLM bridge for Weaver's cortex
#
# This module provides an async interface to the Ollama Cloud REST API.
# It's the default LLM backend because it frees the Pi 5's RAM and CPU
# for sensor processing and gait control — the heavy LLM inference happens
# on a remote server.
#
# Key design decisions:
# - Uses httpx for async HTTP (already a project dependency)
# - Retry logic with exponential backoff (networks fail, especially over Wi-Fi)
# - Timeout handling per-call (cloud can be slow)
# - API key read from config (which reads from env var WEAVER_CORTEX_CLOUD_API_KEY)
# - Same interface as OllamaLocalBridge so they can be swapped at runtime
#
# LEARNING NOTE: Ollama Cloud exposes the same /api/chat or /api/generate
# endpoints as local Ollama, but over HTTPS with an API key in the
# Authorization header. This means the request/response format is identical
# and we don't need a separate SDK.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from loguru import logger

from weaver.config import CortexConfig, get_config


class OllamaCloudBridge:
    """Async bridge to the Ollama Cloud REST API.

    This class wraps the Ollama Cloud endpoints with:
    - Connection pooling (via httpx.AsyncClient)
    - Retry logic with exponential backoff
    - Configurable timeouts
    - Structured error handling

    The Ollama API follows the pattern:
        POST /api/chat
        {
            "model": "llama3.2:3b",
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."}
            ],
            "stream": false,
            "options": {"temperature": 0.7}
        }

    Response (non-streaming):
        {
            "model": "llama3.2:3b",
            "message": {"role": "assistant", "content": "..."},
            "done": true,
            "eval_count": 42,
            "eval_duration": 1500000000
        }

    Attributes:
        config: CortexConfig with cloud settings
        _client: httpx.AsyncClient for connection pooling
        _call_count: Number of successful calls (for telemetry)
        _error_count: Number of failed calls (for telemetry)

    Example:
        bridge = OllamaCloudBridge()
        await bridge.start()
        response = await bridge.generate("Say hello", "You are a robot.")
        print(response)  # "Hello! I am Weaver..."
        await bridge.stop()
    """

    def __init__(self, config: CortexConfig | None = None) -> None:
        """Initialize the cloud bridge.

        Args:
            config: CortexConfig instance. If None, loads from global config.
        """
        self.config: CortexConfig = config or get_config().cortex
        self._client: httpx.AsyncClient | None = None
        self._call_count: int = 0
        self._error_count: int = 0
        self._last_response_time: float = 0.0
        # Base URL — strip trailing slash for consistent path joining
        self._base_url: str = self.config.cloud_base_url.rstrip("/")

    async def start(self) -> None:
        """Start the bridge — create the HTTP client pool.

        Creates an httpx.AsyncClient with:
        - Connection pooling (keep-alive)
        - Configured timeout
        - Default headers (Authorization if API key present)

        Raises:
            httpx.HTTPError: If client creation fails (rare)
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # API key from config (env var WEAVER_CORTEX_CLOUD_API_KEY flows here)
        if self.config.cloud_api_key:
            headers["Authorization"] = f"Bearer {self.config.cloud_api_key}"
        else:
            logger.warning(
                "Ollama Cloud API key is not set — "
                "set WEAVER_CORTEX_CLOUD_API_KEY env var. "
                "Calls will likely fail with 401."
            )

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(self.config.cloud_timeout_seconds),
            # Connection pool settings — conservative for Pi's limited sockets
            limits=httpx.Limits(
                max_connections=5,
                max_keepalive_connections=2,
            ),
        )
        logger.info(
            "☁️  Ollama Cloud bridge started — "
            "model: {}, base_url: {}",
            self.config.cloud_model,
            self._base_url,
        )

    async def stop(self) -> None:
        """Stop the bridge — close the HTTP client and release connections."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info(
            "Ollama Cloud bridge stopped "
            f"(calls={self._call_count}, errors={self._error_count})"
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_retries: int = 2,
    ) -> str:
        """Generate a response from the LLM.

        This is the main entry point. It sends the prompt + system prompt
        to the Ollama Cloud API and returns the generated text.

        Args:
            prompt: The user prompt / question
            system_prompt: System prompt describing the robot's role
            temperature: Sampling temperature (0.0=deterministic, 1.0=creative).
                        Lower is better for robot control — we want consistency.
            max_retries: Number of retry attempts on transient failures.
                        Each retry doubles the wait time (exponential backoff).

        Returns:
            The generated text response from the LLM.

        Raises:
            RuntimeError: If all retries fail or the bridge hasn't started.
            httpx.HTTPError: On non-retryable HTTP errors.

        LEARNING NOTE: We use the /api/chat endpoint (not /api/generate) because
        the chat endpoint properly handles system prompts as a separate message
        role. The generate endpoint mashes everything into one string.
        """
        if self._client is None:
            raise RuntimeError("OllamaCloudBridge not started — call start() first")

        # Build the request payload (Ollama chat format)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.config.cloud_model,
            "messages": messages,
            "stream": False,  # Non-streaming — we want the full response
            "options": {"temperature": temperature},
        }

        # Retry loop with exponential backoff
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                start_time = time.monotonic()
                response = await self._client.post("/api/chat", json=payload)
                elapsed = time.monotonic() - start_time
                self._last_response_time = elapsed

                # Check HTTP status
                response.raise_for_status()

                data = response.json()
                # Ollama chat response: {"message": {"content": "..."}}
                content = data.get("message", {}).get("content", "")
                if not content:
                    logger.warning("Ollama Cloud returned empty response")
                    # This is a soft error — return empty rather than raise

                self._call_count += 1
                logger.debug(
                    f"☁️  LLM response in {elapsed:.2f}s "
                    f"(tokens={data.get('eval_count', '?')}): "
                    f"{content[:80]}..."
                )
                return content.strip()

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"Ollama Cloud timeout (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
            except httpx.HTTPStatusError as e:
                # 4xx errors (except 429) are not retryable
                status = e.response.status_code
                if 400 <= status < 500 and status != 429:
                    logger.error(f"Ollama Cloud client error {status}: {e}")
                    raise RuntimeError(f"Ollama Cloud error {status}: {e}") from e
                last_error = e
                logger.warning(
                    f"Ollama Cloud HTTP {status} (attempt {attempt + 1}/{max_retries + 1})"
                )
            except httpx.HTTPError as e:
                last_error = e
                logger.warning(
                    f"Ollama Cloud connection error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
            except (KeyError, ValueError) as e:
                last_error = e
                logger.error(f"Ollama Cloud response parse error: {e}")

            # Exponential backoff before retry (skip on last attempt)
            if attempt < max_retries:
                wait = 0.5 * (2 ** attempt)  # 0.5s, 1.0s, 2.0s, ...
                logger.debug(f"Retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)

        # All retries exhausted
        self._error_count += 1
        raise RuntimeError(
            f"Ollama Cloud failed after {max_retries + 1} attempts: {last_error}"
        )

    async def is_available(self) -> bool:
        """Check if the cloud LLM is reachable.

        Does a lightweight request to see if the server is up.
        Used by the cortex to decide whether to fall back to rules.

        Returns:
            True if the server responds (even with an error),
            False if the connection fails entirely.
        """
        if self._client is None:
            return False
        try:
            # Hit the /api/tags endpoint — it lists available models
            # and is very cheap to call (no inference)
            response = await self._client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except (httpx.HTTPError, OSError, Exception):
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get bridge statistics for telemetry/debugging.

        Returns:
            Dict with call_count, error_count, last_response_time, etc.
        """
        return {
            "backend": "ollama_cloud",
            "model": self.config.cloud_model,
            "base_url": self._base_url,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "last_response_time_s": round(self._last_response_time, 3),
            "api_key_set": bool(self.config.cloud_api_key),
        }