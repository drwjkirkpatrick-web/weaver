"""Tests for the Ollama Cloud and Local LLM bridges.

Tests cover:
- Bridge instantiation and configuration
- generate() with mocked HTTP responses (cloud)
- generate() with mocked ollama package (local)
- Retry logic
- Error handling
- is_available() health check
- Stats reporting
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weaver.config import CortexConfig, LLMMode


# ─── OllamaCloudBridge Tests ─────────────────────────────────────────────


class TestOllamaCloudBridge:
    """Tests for the Ollama Cloud LLM bridge."""

    @pytest.fixture
    def bridge(self):
        from weaver.brain.ollama_cloud import OllamaCloudBridge
        return OllamaCloudBridge()

    def test_instantiation(self, bridge):
        """Bridge should instantiate with default config."""
        assert bridge.config is not None
        assert bridge.config.cloud_model == "llama3.2:3b"
        assert bridge._client is None
        assert bridge._call_count == 0

    @pytest.mark.asyncio
    async def test_start_creates_client(self, bridge):
        """start() should create an httpx client."""
        await bridge.start()
        assert bridge._client is not None
        await bridge.stop()
        assert bridge._client is None

    @pytest.mark.asyncio
    async def test_generate_without_start_raises(self, bridge):
        """generate() before start() should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not started"):
            await bridge.generate("hello")

    @pytest.mark.asyncio
    async def test_generate_success(self, bridge):
        """generate() should return the content from the API response."""
        await bridge.start()

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Hello from the cloud!"},
            "eval_count": 10,
        }
        bridge._client.post = AsyncMock(return_value=mock_response)

        result = await bridge.generate("Say hello", "You are a robot.")
        assert result == "Hello from the cloud!"
        assert bridge._call_count == 1
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_generate_empty_response(self, bridge):
        """generate() with empty content should return empty string."""
        await bridge.start()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": {"content": ""}}
        bridge._client.post = AsyncMock(return_value=mock_response)

        result = await bridge.generate("test")
        assert result == ""
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_generate_retries_on_timeout(self, bridge):
        """generate() should retry on timeout."""
        import httpx

        await bridge.start()

        # First call times out, second succeeds
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Success"}}
        bridge._client.post = AsyncMock(
            side_effect=[
                httpx.TimeoutException("timeout"),
                mock_response,
            ]
        )

        result = await bridge.generate("test", max_retries=2)
        assert result == "Success"
        assert bridge._call_count == 1
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_generate_fails_after_max_retries(self, bridge):
        """generate() should raise after exhausting retries."""
        import httpx

        await bridge.start()
        bridge._client.post = AsyncMock(
            side_effect=httpx.TimeoutException("timeout")
        )

        with pytest.raises(RuntimeError, match="failed after"):
            await bridge.generate("test", max_retries=1)

        assert bridge._error_count == 1
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_generate_4xx_no_retry(self, bridge):
        """4xx errors (except 429) should not be retried."""
        import httpx

        await bridge.start()

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Unauthorized", request=MagicMock(), response=mock_response
            )
        )
        bridge._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="401"):
            await bridge.generate("test", max_retries=3)

        # Should have only been called once (no retry)
        assert bridge._client.post.call_count == 1
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_is_available_true(self, bridge):
        """is_available() should return True when server responds 200."""
        await bridge.start()
        mock_response = MagicMock()
        mock_response.status_code = 200
        bridge._client.get = AsyncMock(return_value=mock_response)
        assert await bridge.is_available() is True
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_is_available_false_on_error(self, bridge):
        """is_available() should return False on connection error."""
        import httpx

        await bridge.start()
        bridge._client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        assert await bridge.is_available() is False
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_is_available_false_when_not_started(self, bridge):
        """is_available() should return False when bridge not started."""
        assert await bridge.is_available() is False

    def test_get_stats(self, bridge):
        """get_stats() should return bridge statistics."""
        stats = bridge.get_stats()
        assert stats["backend"] == "ollama_cloud"
        assert stats["model"] == bridge.config.cloud_model
        assert "call_count" in stats
        assert "error_count" in stats


# ─── OllamaLocalBridge Tests ──────────────────────────────────────────────


class TestOllamaLocalBridge:
    """Tests for the Ollama Local LLM bridge."""

    @pytest.fixture
    def bridge(self):
        from weaver.brain.ollama_local import OllamaLocalBridge
        return OllamaLocalBridge()

    def test_instantiation(self, bridge):
        """Bridge should instantiate with default config."""
        assert bridge.config is not None
        assert bridge.config.local_model == "llama3.2:3b"
        assert bridge._client is None

    @pytest.mark.asyncio
    async def test_generate_without_start_raises(self, bridge):
        """generate() before start() should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not started"):
            await bridge.generate("hello")

    def test_extract_content_dict(self, bridge):
        """_extract_content should handle dict responses."""
        response = {"message": {"content": "Hello!"}}
        assert bridge._extract_content(response) == "Hello!"

    def test_extract_content_object(self, bridge):
        """_extract_content should handle object responses."""
        response = MagicMock()
        response.message.content = "Hello from object!"
        assert bridge._extract_content(response) == "Hello from object!"

    def test_extract_content_fallback(self, bridge):
        """_extract_content should return empty on unknown format."""
        response = "just a string"
        result = bridge._extract_content(response)
        assert result == ""

    def test_extract_model_names_from_list(self, bridge):
        """_extract_model_names should parse model list."""
        response = {
            "models": [
                {"name": "llama3.2:3b"},
                {"name": "mistral:7b"},
            ]
        }
        names = bridge._extract_model_names(response)
        assert "llama3.2:3b" in names
        assert "mistral:7b" in names

    def test_get_stats(self, bridge):
        """get_stats() should return local bridge statistics."""
        stats = bridge.get_stats()
        assert stats["backend"] == "ollama_local"
        assert stats["model"] == bridge.config.local_model
        assert "call_count" in stats