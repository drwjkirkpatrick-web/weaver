# tts.py — Text-to-Speech for Weaver's voice pipeline
#
# This module handles:
# 1. Converting text responses to speech
# 2. Playing audio through the Pi's audio output (speaker/headphones)
# 3. Publishing VOICE_SPEAKING events
#
# Providers:
# - edge-tts (default, free): Microsoft Edge's online TTS. High quality,
#   many voices, no API key needed. Requires internet.
# - openai: OpenAI's TTS API. High quality, requires API key.
# - elevenlabs: ElevenLabs TTS. Best quality, requires API key.
#
# In MOCK mode, text is just logged (no audio output). This allows testing
# the voice pipeline without a speaker.
#
# LEARNING NOTE: edge-tts is an async library that streams audio from
# Microsoft's free TTS service. It's perfect for the Pi because:
# 1. No model to load (saves RAM)
# 2. High quality neural voices
# 3. Low latency (streaming)
# 4. Free, no API key
# The downside is it requires internet connectivity.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import io
import time
from typing import Any

from loguru import logger

from weaver.config import HardwareMode, VoiceConfig, get_config
from weaver.event_bus import Event, EventBus, EventType, get_event_bus


class TextToSpeech:
    """Text-to-speech output for the Weaver robot.

    Converts text responses to speech and plays them through the audio
    output. Also publishes VOICE_SPEAKING events so the rest of the system
    knows when the robot is talking.

    Lifecycle:
        tts = TextToSpeech()
        await tts.start()
        await tts.speak("Hello, I am Weaver.")
        await tts.stop()

    In MOCK mode, speak() just logs the text — no audio is played.

    Attributes:
        config: VoiceConfig with TTS settings
        bus: The event bus
        _is_speaking: Whether TTS is currently producing speech
        _speak_count: Number of speak() calls
        _mock_mode: True if running without audio hardware
        _audio_player: Audio output handle (for real mode)

    Events published:
        - VOICE_SPEAKING: When TTS starts speaking (with text and active flag)
    """

    def __init__(
        self,
        config: VoiceConfig | None = None,
        hardware_mode: HardwareMode | None = None,
    ) -> None:
        """Initialize the TTS module.

        Args:
            config: VoiceConfig instance. If None, loads from global config.
            hardware_mode: Hardware mode (REAL or MOCK). If None, loads from config.
        """
        self.config: VoiceConfig = config or get_config().voice
        self.main_config = get_config()
        self.hardware_mode: HardwareMode = hardware_mode or self.main_config.hardware_mode
        self.bus: EventBus = get_event_bus()

        # State
        self._is_speaking: bool = False
        self._speak_count: int = 0
        self._mock_mode: bool = (
            self.hardware_mode == HardwareMode.MOCK
            or not self.config.tts_enabled
        )
        self._running: bool = False

        # Edge TTS communicator (lazy-loaded)
        self._edge_communicator: Any = None

        # Audio player for playback (pygame.mixer or similar)
        self._audio_player: Any = None

    async def start(self) -> None:
        """Start the TTS module.

        In REAL mode:
        1. Initializes the TTS provider (edge-tts, openai, etc.)
        2. Sets up audio playback

        In MOCK mode:
        1. Just logs that mock mode is active
        """
        if not self.config.tts_enabled:
            logger.info("🔇 TTS disabled in config")
            return

        mode = "MOCK" if self._mock_mode else "REAL"
        logger.info(
            f"🔊 TTS starting ({mode} mode, provider={self.config.tts_provider}, "
            f"voice={self.config.tts_voice})"
        )

        if not self._mock_mode:
            await self._init_provider()

        # Initialize audio playback
        if not self._mock_mode:
            await self._init_audio_player()

        self._running = True
        logger.info("✅ TTS active — ready to speak")

    async def stop(self) -> None:
        """Stop the TTS module.

        Cleans up the TTS provider and audio player.
        """
        logger.info("🔊 TTS shutting down...")
        self._running = False

        # Wait for current speech to finish (with timeout)
        if self._is_speaking:
            logger.debug("Waiting for current speech to finish...")
            for _ in range(50):  # 5 second timeout
                if not self._is_speaking:
                    break
                await asyncio.sleep(0.1)

        # Cleanup edge-tts
        if self._edge_communicator:
            try:
                # edge-tts Communicate objects don't need explicit close
                pass
            except Exception:
                pass
            self._edge_communicator = None

        # Cleanup audio player
        if self._audio_player:
            try:
                # Stop any playing audio
                pass
            except Exception:
                pass
            self._audio_player = None

        logger.info(f"TTS stopped (spoken={self._speak_count})")

    # ─── Public API ───────────────────────────────────────────────────

    async def speak(self, text: str) -> None:
        """Speak the given text through the audio output.

        This is the main entry point. It converts text to speech and
        plays it. The method is async and returns when speech is complete.

        Args:
            text: The text to speak. Empty strings are ignored.

        Example:
            await tts.speak("Hello! I am Weaver, your hexapod robot.")
            await tts.speak("Obstacle detected. Stopping.")
        """
        if not self._running or not self.config.tts_enabled:
            return

        text = text.strip()
        if not text:
            return

        self._speak_count += 1
        self._is_speaking = True

        # Publish speaking event
        await self.bus.publish(Event(
            type=EventType.VOICE_SPEAKING,
            data={
                "text": text,
                "active": True,
                "timestamp": time.time(),
                "speak_id": self._speak_count,
            },
            source="tts",
        ))

        logger.info(f"🔊 Speaking: '{text[:80]}...'")

        try:
            if self._mock_mode:
                # Mock mode — just log the text
                logger.info(f"🔊 [MOCK TTS] \"{text}\"")
                # Simulate speech duration (100ms per word, min 0.5s)
                word_count = len(text.split())
                duration = max(0.5, word_count * 0.1)
                await asyncio.sleep(duration)
            else:
                # Real mode — generate and play audio
                await self._speak_real(text)

        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            self._is_speaking = False

            # Publish speaking-end event
            await self.bus.publish(Event(
                type=EventType.VOICE_SPEAKING,
                data={
                    "active": False,
                    "timestamp": time.time(),
                    "speak_id": self._speak_count,
                },
                source="tts",
            ))

    async def speak_queue(self, texts: list[str]) -> None:
        """Speak multiple texts sequentially.

        Args:
            texts: List of text strings to speak in order.
        """
        for text in texts:
            await self.speak(text)

    # ─── Provider Initialization ──────────────────────────────────────

    async def _init_provider(self) -> None:
        """Initialize the TTS provider.

        For edge-tts, we don't need to pre-initialize — each speak()
        call creates a new Communicate object. We just validate that
        the package is installed.
        """
        if self.config.tts_provider == "edge":
            try:
                import edge_tts  # type: ignore[import-untyped]  # noqa: F401
                logger.debug("edge-tts package available")
            except ImportError:
                logger.error(
                    "edge-tts is not installed. "
                    "Install with: pip install edge-tts"
                )
                logger.warning("Falling back to MOCK mode for TTS")
                self._mock_mode = True

        elif self.config.tts_provider == "openai":
            import os
            if not os.environ.get("OPENAI_API_KEY"):
                logger.error("OPENAI_API_KEY not set for OpenAI TTS")
                logger.warning("Falling back to MOCK mode for TTS")
                self._mock_mode = True

        elif self.config.tts_provider == "elevenlabs":
            import os
            if not os.environ.get("ELEVENLABS_API_KEY"):
                logger.error("ELEVENLABS_API_KEY not set for ElevenLabs TTS")
                logger.warning("Falling back to MOCK mode for TTS")
                self._mock_mode = True

    async def _init_audio_player(self) -> None:
        """Initialize the audio player for playback.

        Uses pygame.mixer for audio playback. It's the most reliable
        audio library on the Pi for simple WAV/MP3 playback.

        LEARNING NOTE: We try pygame first, then fall back to aplay
        (command-line ALSA player). If neither works, we fall back to
        mock mode.
        """
        try:
            import pygame  # type: ignore[import-untyped]
            pygame.mixer.init()
            self._audio_player = pygame.mixer
            logger.debug("Audio player: pygame.mixer initialized")
        except ImportError:
            logger.debug("pygame not installed — will try aplay for audio playback")
            self._audio_player = "aplay"
        except Exception as e:
            logger.warning(f"pygame.mixer init failed: {e} — will try aplay")
            self._audio_player = "aplay"

    # ─── Real Speech Generation ──────────────────────────────────────

    async def _speak_real(self, text: str) -> None:
        """Generate and play speech in real mode.

        Routes to the appropriate provider (edge, openai, elevenlabs).

        Args:
            text: The text to speak.
        """
        if self.config.tts_provider == "edge":
            await self._speak_edge(text)
        elif self.config.tts_provider == "openai":
            await self._speak_openai(text)
        elif self.config.tts_provider == "elevenlabs":
            await self._speak_elevenlabs(text)
        else:
            logger.error(f"Unknown TTS provider: {self.config.tts_provider}")
            logger.info(f"[FALLBACK TTS] \"{text}\"")

    async def _speak_edge(self, text: str) -> None:
        """Generate speech using edge-tts (Microsoft Edge TTS).

        edge-tts streams audio data as MP3. We collect it into a buffer
        and then play it.

        Args:
            text: The text to speak.
        """
        import edge_tts  # type: ignore[import-untyped]

        # Create a Communicate object for this text
        communicate = edge_tts.Communicate(text, self.config.tts_voice)

        # Collect audio data into a buffer
        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        # Play the audio
        audio_buffer.seek(0)
        await self._play_audio(audio_buffer.read())

    async def _speak_openai(self, text: str) -> None:
        """Generate speech using OpenAI's TTS API.

        Args:
            text: The text to speak.
        """
        import os

        import httpx

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.error("OPENAI_API_KEY not set")
            return

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "tts-1",
                    "input": text,
                    "voice": "alloy",
                    "response_format": "mp3",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            await self._play_audio(response.content)

    async def _speak_elevenlabs(self, text: str) -> None:
        """Generate speech using ElevenLabs TTS API.

        Args:
            text: The text to speak.
        """
        import os

        import httpx

        api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "")
        if not api_key or not voice_id:
            logger.error("ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID not set")
            return

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            await self._play_audio(response.content)

    # ─── Audio Playback ───────────────────────────────────────────────

    async def _play_audio(self, audio_data: bytes) -> None:
        """Play audio data through the audio output.

        Supports pygame.mixer and aplay (command-line). Audio data
        should be in MP3 format (from edge-tts and OpenAI).

        Args:
            audio_data: Audio bytes (MP3 format)
        """
        if self._audio_player is None:
            logger.warning("No audio player available — skipping playback")
            return

        if self._audio_player == "aplay":
            # Use aplay (ALSA command-line player)
            # aplay doesn't play MP3 directly — use mpg123 instead
            await self._play_with_mpg123(audio_data)
        elif hasattr(self._audio_player, "music"):
            # pygame.mixer
            await self._play_with_pygame(audio_data)
        else:
            logger.warning(f"Unknown audio player type: {type(self._audio_player)}")

    async def _play_with_pygame(self, audio_data: bytes) -> None:
        """Play audio using pygame.mixer.

        Args:
            audio_data: MP3 audio bytes
        """
        import tempfile
        from pathlib import Path

        # pygame needs a file, so write to a temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            self._audio_player.music.load(temp_path)
            self._audio_player.music.play()

            # Wait for playback to complete
            while self._audio_player.music.get_busy():
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"pygame playback error: {e}")
        finally:
            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except Exception:
                pass

    async def _play_with_mpg123(self, audio_data: bytes) -> None:
        """Play MP3 audio using mpg123 command-line player.

        mpg123 is a lightweight MP3 player available on most Linux systems.
        It's more reliable than pygame on the Pi for simple playback.

        Args:
            audio_data: MP3 audio bytes
        """
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "mpg123", "-q", temp_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except FileNotFoundError:
            logger.error("mpg123 not installed — install with: sudo apt install mpg123")
        except Exception as e:
            logger.error(f"mpg123 playback error: {e}")
        finally:
            try:
                Path(temp_path).unlink()
            except Exception:
                pass

    # ─── Utilities ────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get TTS statistics for telemetry/debugging.

        Returns:
            Dict with speak count, mode, provider info, etc.
        """
        return {
            "mode": "mock" if self._mock_mode else "real",
            "provider": self.config.tts_provider,
            "voice": self.config.tts_voice,
            "speak_count": self._speak_count,
            "is_speaking": self._is_speaking,
        }