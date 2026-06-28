"""Tests for the voice pipeline (STT and TTS).

Tests cover:
- STT/TTS instantiation and configuration
- Mock mode operation
- Event publishing (VOICE_LISTENING, VOICE_TRANSCRIPT, VOICE_SPEAKING)
- TTS speak() in mock mode
- Stats reporting
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weaver.config import HardwareMode, VoiceConfig
from weaver.event_bus import Event, EventBus, EventType


# ─── STT Tests ───────────────────────────────────────────────────────────


class TestSpeechToText:
    """Tests for the SpeechToText module."""

    @pytest.fixture
    def stt(self):
        from weaver.voice.stt import SpeechToText
        return SpeechToText()

    def test_instantiation(self, stt):
        """STT should instantiate with default config."""
        assert stt.config is not None
        assert stt._mock_mode is True  # Default is MOCK hardware

    @pytest.mark.asyncio
    async def test_start_stop_mock_mode(self, stt):
        """start()/stop() should work in mock mode."""
        await stt.start()
        assert stt._running is True
        await stt.stop()
        assert stt._running is False

    @pytest.mark.asyncio
    async def test_disabled_stt_does_nothing(self):
        """Disabled STT should not start the listening loop."""
        from weaver.voice.stt import SpeechToText
        stt = SpeechToText(config=VoiceConfig(stt_enabled=False))
        await stt.start()
        assert stt._listen_task is None
        await stt.stop()

    @pytest.mark.asyncio
    async def test_publish_listening(self, stt):
        """_publish_listening should publish VOICE_LISTENING event."""
        stt.bus = EventBus()
        received = []
        async def capture(event: Event):
            received.append(event)
        stt.bus.subscribe(EventType.VOICE_LISTENING, capture)

        await stt._publish_listening()
        assert len(received) == 1
        assert "timestamp" in received[0].data

    @pytest.mark.asyncio
    async def test_publish_transcript(self, stt):
        """_publish_transcript should publish VOICE_TRANSCRIPT event."""
        stt.bus = EventBus()
        received = []
        async def capture(event: Event):
            received.append(event)
        stt.bus.subscribe(EventType.VOICE_TRANSCRIPT, capture)

        await stt._publish_transcript("walk forward")
        assert len(received) == 1
        assert received[0].data["text"] == "walk forward"
        assert stt._transcript_count == 1

    def test_wake_word_check_above_threshold(self, stt):
        """Wake word should trigger when energy exceeds threshold."""
        stt.config.wake_word_enabled = True
        stt.config.wake_word_energy_threshold = 500.0
        stt._last_wake_time = 0.0  # No cooldown
        assert stt._check_wake_word(600.0) is True

    def test_wake_word_check_below_threshold(self, stt):
        """Wake word should not trigger below threshold."""
        stt.config.wake_word_enabled = True
        stt.config.wake_word_energy_threshold = 500.0
        stt._last_wake_time = 0.0
        assert stt._check_wake_word(100.0) is False

    def test_wake_word_disabled_always_triggers(self, stt):
        """When wake word is disabled, should always trigger."""
        stt.config.wake_word_enabled = False
        assert stt._check_wake_word(10.0) is True

    def test_wake_word_cooldown(self, stt):
        """Wake word should not trigger during cooldown."""
        import time
        stt.config.wake_word_enabled = True
        stt.config.wake_word_energy_threshold = 500.0
        stt.config.wake_word_cooldown_seconds = 5.0
        stt._last_wake_time = time.time()  # Just triggered
        assert stt._check_wake_word(600.0) is False

    def test_get_stats(self, stt):
        """get_stats should return STT statistics."""
        stats = stt.get_stats()
        assert stats["mode"] == "mock"
        assert stats["provider"] == stt.config.stt_provider
        assert "transcript_count" in stats


# ─── TTS Tests ───────────────────────────────────────────────────────────


class TestTextToSpeech:
    """Tests for the TextToSpeech module."""

    @pytest.fixture
    def tts(self):
        from weaver.voice.tts import TextToSpeech
        return TextToSpeech()

    def test_instantiation(self, tts):
        """TTS should instantiate with default config."""
        assert tts.config is not None
        assert tts._mock_mode is True  # Default is MOCK hardware

    @pytest.mark.asyncio
    async def test_start_stop_mock_mode(self, tts):
        """start()/stop() should work in mock mode."""
        await tts.start()
        assert tts._running is True
        await tts.stop()
        assert tts._running is False

    @pytest.mark.asyncio
    async def test_disabled_tts_does_nothing(self):
        """Disabled TTS should not start."""
        from weaver.voice.tts import TextToSpeech
        tts = TextToSpeech(config=VoiceConfig(tts_enabled=False))
        await tts.start()
        assert tts._running is False
        await tts.stop()

    @pytest.mark.asyncio
    async def test_speak_mock_mode(self, tts):
        """speak() in mock mode should simulate speech."""
        await tts.start()
        await tts.speak("Hello, I am Weaver.")
        assert tts._speak_count == 1
        await tts.stop()

    @pytest.mark.asyncio
    async def test_speak_publishes_events(self, tts):
        """speak() should publish VOICE_SPEAKING events."""
        tts.bus = EventBus()
        received = []
        async def capture(event: Event):
            received.append(event)
        tts.bus.subscribe(EventType.VOICE_SPEAKING, capture)

        await tts.start()
        await tts.speak("Test message")
        # Should publish start (active=True) and stop (active=False)
        assert len(received) == 2
        assert received[0].data["active"] is True
        assert received[1].data["active"] is False
        await tts.stop()

    @pytest.mark.asyncio
    async def test_speak_empty_string_ignored(self, tts):
        """Empty strings should be ignored."""
        await tts.start()
        await tts.speak("")
        assert tts._speak_count == 0
        await tts.stop()

    @pytest.mark.asyncio
    async def test_speak_whitespace_ignored(self, tts):
        """Whitespace-only strings should be ignored."""
        await tts.start()
        await tts.speak("   ")
        assert tts._speak_count == 0
        await tts.stop()

    @pytest.mark.asyncio
    async def test_speak_queue(self, tts):
        """speak_queue should speak multiple texts."""
        tts.bus = EventBus()
        await tts.start()
        await tts.speak_queue(["Hello", "World"])
        assert tts._speak_count == 2
        await tts.stop()

    @pytest.mark.asyncio
    async def test_speak_not_running_ignored(self, tts):
        """speak() before start() should be ignored."""
        await tts.speak("test")
        assert tts._speak_count == 0

    def test_get_stats(self, tts):
        """get_stats should return TTS statistics."""
        stats = tts.get_stats()
        assert stats["mode"] == "mock"
        assert stats["provider"] == tts.config.tts_provider
        assert "speak_count" in stats
        assert "is_speaking" in stats