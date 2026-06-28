# ollama_local.py — Ollama Local LLM bridge for Weaver's cortex
#
# This module provides an async interface to a locally-running Ollama instance
# on the Raspberry Pi 5. It has the same interface as OllamaCloudBridge so
# the cortex can swap between them at runtime without code changes.
#
# When to use local vs. cloud:
# - CLOUD (default): Frees Pi RAM/CPU. Best when Wi-Fi is reliable.
# - LOCAL: No network dependency. Best for offline operation, low-latency
#   needs, or when the cloud API is down. Uses ~2-4GB of Pi RAM for the model.
#
# The ollama Python package handles the HTTP communication with the local
# Ollama server (which runs at localhost:11434 by default).
#
# LEARNING NOTE: The ollama Python package is synchronous. We wrap its calls
# in asyncio.to_thread() to avoid blocking the event loop. This is the
# standard pattern for wrapping sync APIs in async code.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from weaver.config import CortexConfig, get_config


class OllamaLocalBridge:
    """Async bridge to a local Ollama instance running on the Pi.

    Uses the `ollama` Python package which communicates with the Ollama
    server (daemon) running on localhost:11434.

    Same interface as OllamaCloudBridge:
        bridge = OllamaLocalBridge()
        await bridge.start()
        response = await bridge.generate("Say hello", "You are a robot.")
        await bridge.stop()

    Key differences from cloud:
    - No API key needed (local server)
    - Longer timeout (Pi CPU is slower than cloud GPU)
    - Model must be pre-pulled: `ollama pull llama3.2:3b`
    - Synchronous ollama package wrapped in asyncio.to_thread()

    Attributes:
        config: CortexConfig with local settings
        _client: The ollama Client instance (sync)
        _call_count: Number of successful calls
        _error_count: Number of failed calls
        _last_response_time: Wall-clock time of last successful call

    Example:
        bridge = OllamaLocalBridge()
        await bridge.start()
        response = await bridge.generate(
            "What should I do about the obstacle?",
            system_prompt="You are Weaver, a hexapod robot."
        )
        await bridge.stop()
    """

    def __init__(self, config: CortexConfig | None = None) -> None:
        """Initialize the local bridge.

        Args:
            config: CortexConfig instance. If None, loads from global config.
        """
        self.config: CortexConfig = config or get_config().cortex
        self._client: Any = None  # ollama.Client (typed as Any to avoid hard dep)
        self._call_count: int = 0
        self._error_count: int = 0
        self._last_response_time: float = 0.0

    async def start(self) -> None:
        """Start the bridge — create the ollama client.

        This imports the ollama package lazily so that the module can be
        imported even if the ollama package isn't installed (e.g., on a
        dev machine that's only using the cloud bridge).

        Raises:
            ImportError: If the `ollama` package is not installed.
            ConnectionError: If the local Ollama server is not running.

        LEARNING NOTE: Lazy imports are important on the Pi because importing
        heavy packages at module load time slows startup. By importing inside
        start(), we only load ollama when local mode is actually selected.
        """
        try:
            import ollama  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "The 'ollama' package is required for local LLM mode. "
                "Install it with: pip install ollama"
            ) from e

        # Create the client pointing at the local Ollama server
        self._client = ollama.Client(
            host=self.config.local_base_url,
            timeout=self.config.local_timeout_seconds,
        )

        # Check if the model is available — warn (don't crash) if not
        try:
            models_response = await asyncio.to_thread(self._client.list)
            model_names = self._extract_model_names(models_response)
            if self.config.local_model not in model_names:
                logger.warning(
                    f"Model '{self.config.local_model}' not found locally. "
                    f"Available: {model_names}. "
                    f"Pull it with: ollama pull {self.config.local_model}"
                )
        except Exception as e:
            logger.warning(
                f"Could not list local models (is Ollama running at "
                f"{self.config.local_base_url}?): {e}"
            )

        logger.info(
            "🏠 Ollama Local bridge started — "
            "model: {}, host: {}",
            self.config.local_model,
            self.config.local_base_url,
        )

    async def stop(self) -> None:
        """Stop the bridge — release the client.

        The ollama package doesn't have an explicit close method (it uses
        requests under the hood), so we just clear the reference.
        """
        self._client = None
        logger.info(
            "Ollama Local bridge stopped "
            f"(calls={self._call_count}, errors={self._error_count})"
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_retries: int = 1,
    ) -> str:
        """Generate a response from the local LLM.

        Uses the ollama Python package's chat() method. Since the ollama
        package is synchronous, we run it in a thread to avoid blocking
        the asyncio event loop.

        Args:
            prompt: The user prompt / question
            system_prompt: System prompt describing the robot's role
            temperature: Sampling temperature. Lower = more deterministic.
            max_retries: Number of retry attempts. Local is usually more
                        reliable than cloud (no network), so default is 1.

        Returns:
            The generated text response from the LLM.

        Raises:
            RuntimeError: If all retries fail or bridge not started.
        """
        if self._client is None:
            raise RuntimeError("OllamaLocalBridge not started — call start() first")

        # Build messages list
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                # Run the sync ollama call in a thread pool
                # This prevents blocking the asyncio event loop
                import time as _time
                start = _time.monotonic()

                response = await asyncio.to_thread(
                    self._client.chat,
                    model=self.config.local_model,
                    messages=messages,
                    stream=False,
                    options={"temperature": temperature},
                )

                elapsed = _time.monotonic() - start
                self._last_response_time = elapsed

                # ollama package returns a ChatResponse object/dict
                # The content is in response["message"]["content"]
                content = self._extract_content(response)
                if not content:
                    logger.warning("Ollama Local returned empty response")

                self._call_count += 1
                logger.debug(
                    f"🏠 LLM response in {elapsed:.2f}s: {content[:80]}..."
                )
                return content.strip()

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Ollama Local error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                if attempt < max_retries:
                    wait = 0.5 * (2 ** attempt)
                    logger.debug(f"Retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)

        self._error_count += 1
        raise RuntimeError(
            f"Ollama Local failed after {max_retries + 1} attempts: {last_error}"
        )

    async def is_available(self) -> bool:
        """Check if the local Ollama server is running and responsive.

        Returns:
            True if the server responds to a list models request.
        """
        if self._client is None:
            return False
        try:
            await asyncio.to_thread(self._client.list)
            return True
        except Exception:
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get bridge statistics for telemetry/debugging.

        Returns:
            Dict with call_count, error_count, last_response_time, etc.
        """
        return {
            "backend": "ollama_local",
            "model": self.config.local_model,
            "base_url": self.config.local_base_url,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "last_response_time_s": round(self._last_response_time, 3),
        }

    # ─── Private helpers ──────────────────────────────────────────────

    def _extract_content(self, response: Any) -> str:
        """Extract text content from an ollama response.

        The ollama Python package can return either a dict or a ChatResponse
        object depending on the version. This helper handles both.

        Args:
            response: The return value of client.chat()

        Returns:
            The text content from the assistant's message.
        """
        # Handle dict-like response (older ollama package)
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "")
        # Handle object-like response (newer ollama package)
        # ChatResponse has .message.content
        try:
            return response.message.content
        except AttributeError:
            pass
        # Fallback: try to convert to dict and access
        try:
            return dict(response).get("message", {}).get("content", "")
        except Exception:
            logger.warning(f"Could not extract content from ollama response: {type(response)}")
            return ""

    def _extract_model_names(self, models_response: Any) -> list[str]:
        """Extract model names from an ollama list() response.

        Handles different response formats across ollama package versions.

        Args:
            models_response: The return value of client.list()

        Returns:
            List of model name strings.
        """
        names: list[str] = []
        try:
            # Newer format: response.models = [Model(name=..., ...)]
            if hasattr(models_response, "models"):
                for m in models_response.models:
                    name = getattr(m, "name", None) or getattr(m, "model", None)
                    if name:
                        names.append(name)
            # Older format: dict with "models" key containing list of dicts
            elif isinstance(models_response, dict):
                for m in models_response.get("models", []):
                    name = m.get("name") or m.get("model")
                    if name:
                        names.append(name)
        except Exception as e:
            logger.debug(f"Error parsing model list: {e}")
        return names