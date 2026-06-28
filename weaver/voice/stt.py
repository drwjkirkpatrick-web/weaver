# stt.py — Speech-to-Text for Weaver's voice pipeline
#
# This module handles:
# 1. Listening to a USB microphone for voice commands
# 2. Wake word detection (simple energy-based — no Porcupine/Snowboy needed)
# 3. Transcribing speech to text using faster-whisper (local) or an API
# 4. Publishing VOICE_LISTENING and VOICE_TRANSCRIPT events
#
# Two modes:
# - MOCK: No microphone needed — for development on non-Pi machines.
#         Simulates voice commands from stdin or generates fake ones.
# - REAL: Uses faster-whisper with a USB microphone (or default audio device).
#         Transcribes speech in real-time.
#
# LEARNING NOTE: faster-whisper is a CTranslate2-backed reimplementation of
# OpenAI's Whisper. It's 4x faster than the original and uses less memory.
# The "base" model is a good tradeoff for the Pi 5 — it's accurate enough
# for short commands and runs in real-time.
#
# Wake word detection uses a simple energy threshold (RMS). When the audio
# energy exceeds the threshold, we start recording. When it drops below the
# threshold for stt_timeout_seconds, we stop and transcribe. This is not
# as accurate as a dedicated wake word engine, but it's dependency-free
# and works well for a controlled environment.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

from loguru import logger

from weaver.config import HardwareMode, VoiceConfig, get_config
from weaver.event_bus import Event, EventBus, EventType, get_event_bus


class SpeechToText:
    """Speech-to-text with wake word detection for the Weaver robot.

    Listens to a microphone, detects speech via energy threshold, and
    transcribes using faster-whisper (local) or a remote API.

    Lifecycle:
        stt = SpeechToText()
        await stt.start()    # Starts listening
        # ... publishes VOICE_LISTENING and VOICE_TRANSCRIPT events ...
        await stt.stop()

    In MOCK mode, it reads from stdin (or generates fake commands) so you
    can test the voice pipeline without a microphone.

    Attributes:
        config: VoiceConfig with STT settings
        bus: The event bus
        _model: The whisper model (lazy-loaded)
        _listen_task: Background task for the listening loop
        _mock_mode: True if running without hardware
        _is_listening: Whether currently recording audio
        _last_wake_time: Timestamp of last wake word detection

    Events published:
        - VOICE_LISTENING: When speech detection starts (wake word triggered)
        - VOICE_TRANSCRIPT: When speech has been transcribed
    """

    def __init__(
        self,
        config: VoiceConfig | None = None,
        hardware_mode: HardwareMode | None = None,
    ) -> None:
        """Initialize the STT module.

        Args:
            config: VoiceConfig instance. If None, loads from global config.
            hardware_mode: Hardware mode (REAL or MOCK). If None, loads from config.
        """
        self.config: VoiceConfig = config or get_config().voice
        self.main_config = get_config()
        self.hardware_mode: HardwareMode = hardware_mode or self.main_config.hardware_mode
        self.bus: EventBus = get_event_bus()

        # State
        self._model: Any = None  # whisper model (lazy-loaded)
        self._listen_task: asyncio.Task | None = None
        self._running: bool = False
        self._is_listening: bool = False
        self._last_wake_time: float = 0.0
        self._transcript_count: int = 0

        # Audio device handle (for real mode)
        self._audio_stream: Any = None

        # Mock mode — determined by hardware mode and config
        self._mock_mode: bool = (
            self.hardware_mode == HardwareMode.MOCK
            or not self.config.stt_enabled
        )

    async def start(self) -> None:
        """Start the STT module.

        In REAL mode:
        1. Loads the whisper model (this takes a few seconds)
        2. Opens the microphone
        3. Starts the listening loop

        In MOCK mode:
        1. Starts a stdin reader for simulated voice commands

        Raises:
            ImportError: If faster-whisper is not installed (real mode)
            OSError: If microphone cannot be opened
        """
        if not self.config.stt_enabled:
            logger.info("🔇 STT disabled in config")
            return

        mode = "MOCK" if self._mock_mode else "REAL"
        logger.info(f"🎙️  STT starting ({mode} mode, provider={self.config.stt_provider})")

        if not self._mock_mode:
            # Load the whisper model
            await self._load_model()

        self._running = True
        self._listen_task = asyncio.create_task(self._listening_loop())

        logger.info("✅ STT active — listening for voice commands")

    async def stop(self) -> None:
        """Stop the STT module.

        Cancels the listening loop and releases the model/audio resources.
        """
        logger.info("🎙️  STT shutting down...")
        self._running = False

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        # Release audio resources
        if self._audio_stream:
            try:
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None

        # Unload model to free RAM
        self._model = None

        logger.info(f"STT stopped (transcripts={self._transcript_count})")

    # ─── Model Loading ───────────────────────────────────────────────

    async def _load_model(self) -> None:
        """Load the whisper model.

        Uses faster-whisper for local transcription. The model is loaded
        in a thread to avoid blocking the event loop (model loading is CPU/IO
        intensive).

        LEARNING NOTE: faster-whisper loads the model from disk on first use.
        The "base" model is ~150MB. On the Pi 5, this takes 3-5 seconds.
        We load it in a background thread so start() doesn't block.
        """
        if self.config.stt_provider != "local":
            logger.info(
                f"Using {self.config.stt_provider} API for STT — no local model needed"
            )
            return

        def _load() -> Any:
            """Load whisper model in a thread."""
            try:
                from faster_whisper import WhisperModel  # type: ignore[import-untyped]
                # Load model — "base" is good for Pi 5
                # device="cpu" because Pi 5 doesn't have CUDA
                # compute_type="int8" for speed (quantized)
                model = WhisperModel(
                    self.config.stt_model,
                    device="cpu",
                    compute_type="int8",
                )
                logger.info(f"📦 Whisper model '{self.config.stt_model}' loaded")
                return model
            except ImportError:
                raise ImportError(
                    "faster-whisper is not installed. "
                    "Install with: pip install faster-whisper"
                )

        try:
            self._model = await asyncio.to_thread(_load)
        except Exception as e:
            logger.error(f"Failed to load whisper model: {e}")
            logger.warning("Falling back to MOCK mode for STT")
            self._mock_mode = True

    # ─── Listening Loop ───────────────────────────────────────────────

    async def _listening_loop(self) -> None:
        """Main listening loop.

        In REAL mode:
        1. Capture audio from microphone in chunks
        2. Check if energy exceeds wake word threshold
        3. If yes, start recording
        4. When silence detected, transcribe and publish
        5. Repeat

        In MOCK mode:
        1. Read commands from stdin
        2. Publish them as transcripts
        """
        if self._mock_mode:
            await self._mock_listening_loop()
        else:
            await self._real_listening_loop()

    async def _mock_listening_loop(self) -> None:
        """Mock listening loop — reads from stdin.

        This allows testing the voice pipeline without a microphone.
        Commands are read line-by-line from stdin and published as transcripts.
        """
        logger.info("📝 STT MOCK mode — type commands and press Enter (Ctrl+D to quit)")

        loop = asyncio.get_event_loop()

        while self._running:
            try:
                # Read from stdin in a thread (stdin is blocking)
                line = await loop.run_in_executor(
                    None, sys.stdin.readline
                )
                if not line:  # EOF
                    logger.info("STT mock: stdin EOF")
                    break

                text = line.strip()
                if not text:
                    continue

                # Simulate the listening → transcript flow
                await self._publish_listening()
                await asyncio.sleep(0.1)  # Simulate processing time
                await self._publish_transcript(text)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"STT mock loop error: {e}")
                await asyncio.sleep(1.0)

    async def _real_listening_loop(self) -> None:
        """Real listening loop — captures audio and transcribes.

        Uses pyaudio (or sounddevice) to capture audio chunks, checks
        energy for wake word detection, and transcribes when speech ends.
        """
        try:
            import numpy as np
            import pyaudio  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "pyaudio and numpy are required for real STT mode. "
                "Install with: pip install pyaudio numpy"
            )
            logger.warning("Falling back to MOCK mode")
            self._mock_mode = True
            await self._mock_listening_loop()
            return

        # Audio configuration
        # LEARNING NOTE: Whisper expects 16kHz mono audio
        sample_rate = 16000
        chunk_duration = 0.03  # 30ms chunks for responsive wake word detection
        chunk_size = int(sample_rate * chunk_duration)

        pa = pyaudio.PyAudio()

        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
            )
            self._audio_stream = stream
            logger.debug(f"Microphone opened: {sample_rate}Hz, {chunk_size} samples/chunk")
        except OSError as e:
            logger.error(f"Failed to open microphone: {e}")
            logger.warning("Falling back to MOCK mode")
            self._mock_mode = True
            await self._mock_listening_loop()
            return

        # Audio buffer for recording
        recording_buffer: list[np.ndarray] = []
        is_recording = False
        silence_start_time: float | None = None

        logger.debug("Microphone listening for wake word...")

        while self._running:
            try:
                # Read audio chunk (in a thread — pyaudio.read is blocking)
                audio_data = await asyncio.to_thread(
                    stream.read, chunk_size, exception_on_overflow=False
                )

                # Convert to numpy array for energy calculation
                samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

                # Calculate RMS energy
                rms_energy = float(np.sqrt(np.mean(samples ** 2)))

                if not is_recording:
                    # ─── Wake word detection (energy-based) ───
                    if self._check_wake_word(rms_energy):
                        is_recording = True
                        recording_buffer = [samples]
                        silence_start_time = None
                        self._is_listening = True
                        await self._publish_listening()
                        logger.debug(
                            f"Wake word triggered (energy={rms_energy:.0f})"
                        )
                else:
                    # ─── Recording ───
                    recording_buffer.append(samples)

                    # Check for silence (end of speech)
                    if rms_energy < self.config.wake_word_energy_threshold * 0.3:
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        elif time.time() - silence_start_time > self.config.stt_timeout_seconds:
                            # Silence detected for timeout — transcribe
                            is_recording = False
                            self._is_listening = False
                            silence_start_time = None

                            # Concatenate buffer and transcribe
                            audio_array = np.concatenate(recording_buffer)
                            recording_buffer = []

                            # Transcribe in a thread (CPU-intensive)
                            text = await asyncio.to_thread(
                                self._transcribe_audio, audio_array, sample_rate
                            )

                            if text:
                                await self._publish_transcript(text)
                            else:
                                logger.debug("STT: no speech detected in recording")
                    else:
                        # Reset silence timer
                        silence_start_time = None

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"STT listening loop error: {e}")
                await asyncio.sleep(0.1)

        # Cleanup
        try:
            stream.close()
        except Exception:
            pass
        pa.terminate()

    def _check_wake_word(self, energy: float) -> bool:
        """Check if energy indicates speech (wake word detection).

        Simple energy-based detection: if energy exceeds threshold AND
        we're not in cooldown, trigger.

        Args:
            energy: RMS energy of the current audio chunk

        Returns:
            True if wake word is detected (speech started).
        """
        if not self.config.wake_word_enabled:
            # If wake word is disabled, always trigger (continuous listening)
            return True

        # Check cooldown
        now = time.time()
        if now - self._last_wake_time < self.config.wake_word_cooldown_seconds:
            return False

        # Check energy threshold
        if energy >= self.config.wake_word_energy_threshold:
            self._last_wake_time = now
            return True

        return False

    # ─── Transcription ───────────────────────────────────────────────

    def _transcribe_audio(self, audio: Any, sample_rate: int) -> str:
        """Transcribe audio data using the loaded whisper model.

        This is a SYNCHRONOUS method — it should be called in a thread
        (via asyncio.to_thread) to avoid blocking the event loop.

        Args:
            audio: numpy array of float32 audio samples
            sample_rate: Sample rate of the audio (should be 16000)

        Returns:
            Transcribed text, or empty string if no speech detected.
        """
        if self._model is None:
            logger.error("Whisper model not loaded")
            return ""

        try:
            # faster-whisper API:
            # segments, info = model.transcribe(audio, language="en")
            segments, _info = self._model.transcribe(
                audio,
                language=self.config.stt_language,
                beam_size=1,  # Fast setting — 1 beam for real-time
                vad_filter=True,  # Filter out non-speech
            )
            # segments is a generator — collect text
            text = " ".join(segment.text for segment in segments).strip()
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    async def transcribe_api(self, audio_data: bytes) -> str:
        """Transcribe audio using a remote API (Groq or OpenAI).

        Used when stt_provider is "groq" or "openai" instead of "local".

        Args:
            audio_data: Raw audio bytes (WAV format)

        Returns:
            Transcribed text.
        """
        if self.config.stt_provider == "groq":
            return await self._transcribe_groq(audio_data)
        elif self.config.stt_provider == "openai":
            return await self._transcribe_openai(audio_data)
        else:
            logger.error(f"Unknown STT provider: {self.config.stt_provider}")
            return ""

    async def _transcribe_groq(self, audio_data: bytes) -> str:
        """Transcribe using Groq's Whisper API.

        Args:
            audio_data: WAV audio bytes

        Returns:
            Transcribed text.
        """
        import os

        import httpx

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            logger.error("GROQ_API_KEY not set")
            return ""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("audio.wav", audio_data, "audio/wav")},
                data={"model": "whisper-large-v3"},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json().get("text", "")

    async def _transcribe_openai(self, audio_data: bytes) -> str:
        """Transcribe using OpenAI's Whisper API.

        Args:
            audio_data: WAV audio bytes

        Returns:
            Transcribed text.
        """
        import os

        import httpx

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.error("OPENAI_API_KEY not set")
            return ""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("audio.wav", audio_data, "audio/wav")},
                data={"model": "whisper-1"},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json().get("text", "")

    # ─── Event Publishing ─────────────────────────────────────────────

    async def _publish_listening(self) -> None:
        """Publish a VOICE_LISTENING event.

        This tells the rest of the system that the robot is actively
        listening to a voice command. The LED module can show a
        "listening" color, and the cortex can pause motion.
        """
        await self.bus.publish(Event(
            type=EventType.VOICE_LISTENING,
            data={
                "timestamp": time.time(),
                "wake_word": self.config.wake_word_enabled,
            },
            source="stt",
        ))

    async def _publish_transcript(self, text: str) -> None:
        """Publish a VOICE_TRANSCRIPT event.

        This delivers the transcribed text to the cortex, which will
        parse it into a motion command via the intent parser.

        Args:
            text: The transcribed speech
        """
        self._transcript_count += 1
        logger.info(f"🎙️  Transcript: '{text}'")

        await self.bus.publish(Event(
            type=EventType.VOICE_TRANSCRIPT,
            data={
                "text": text,
                "timestamp": time.time(),
                "transcript_id": self._transcript_count,
            },
            source="stt",
        ))

    # ─── Utilities ────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get STT statistics for telemetry/debugging.

        Returns:
            Dict with transcript count, mode, model info, etc.
        """
        return {
            "mode": "mock" if self._mock_mode else "real",
            "provider": self.config.stt_provider,
            "model": self.config.stt_model,
            "transcript_count": self._transcript_count,
            "is_listening": self._is_listening,
            "wake_word_enabled": self.config.wake_word_enabled,
        }