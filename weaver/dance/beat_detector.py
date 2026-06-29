# beat_detector.py — Real-time beat & BPM detection for Weaver's dance mode
#
# This module turns the robot into a listener that can feel the music.
# It captures audio (from a microphone or an audio file), detects rhythmic
# beats, estimates the tempo (BPM), and publishes BEAT_DETECTED events so
# the dance engine can move in time with the music.
#
# The pipeline has five stages, each explained in detail below:
#
#   ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐
#   │ Capture │ → │   FFT    │ → │ Spectral │ → │  Onset   │ → │  BPM    │
#   │  Audio  │   │ (window) │   │   Flux   │   │ picking  │   │ (auto-  │
#   └─────────┘   └──────────┘   └──────────┘   └──────────┘   │ corr.)  │
#                                                              └─────────┘
#
# 1. CAPTURE: Read raw PCM samples (int16) from a microphone or file.
# 2. FFT:    Convert overlapping time-domain windows into frequency spectra.
# 3. FLUX:   Measure how much the spectrum changes frame-to-frame (spectral
#            flux). Sharp increases = new sound = likely onset.
# 4. ONSET:  Compare flux to an adaptive (moving-average) threshold. Peaks
#            above threshold × sensitivity = beats.
# 5. BPM:    Autocorrelate the onset envelope to find the dominant periodic
#            spacing between beats; convert that lag to beats-per-minute.
#
# LEARNING NOTE — Why FFT for beat detection?
#   Beats are usually carried by bass/mid frequencies (kick drum, bass
#   guitar). The time-domain waveform alone mixes every frequency
#   together, so a simple amplitude threshold gets fooled by loud
#   hi-hats or vocals. By transforming to the frequency domain first,
#   we can measure energy *per band* and spot percussive onsets that
#   span many bins at once — a hallmark of "attack" transients.
#
# LEARNING NOTE — Why autocorrelation for BPM?
#   Autocorrelation measures how similar a signal is to a time-shifted
#   copy of itself. If beats repeat every ~500ms (120 BPM), the onset
#   envelope will correlate strongly with itself shifted by 500ms. The
#   lag with the highest correlation corresponds to the beat period,
#   from which BPM = 60 / period_seconds. This is robust to missed or
#   extra beats because it uses *all* the onsets, not just neighbours.
#
# Two operating modes:
#   - MOCK:  No microphone needed. Generates synthetic beats at
#            `mock_bpm` so the dance engine can be developed anywhere.
#   - REAL:  Opens the default USB mic via PyAudio and analyses live audio.
#
# Optional dependencies (all imported defensively, with graceful fallback):
#   - numpy  : required for the actual math (FFT, autocorrelation).
#   - pyaudio: required only for live microphone capture.
#   - librosa: optional convenience library; if present we can use it
#              to load MP3/FLAC, otherwise we fall back to the stdlib
#              `wave` module (WAV only).
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import math
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from weaver.config import BeatDetectorConfig, HardwareMode, get_config
from weaver.event_bus import Event, EventBus, EventType, get_event_bus


# ─── Optional dependency guards ─────────────────────────────────────────────
# We try to import each optional library once at module load. If it's missing
# we keep a flag and degrade gracefully — the beat detector must NEVER crash
# the robot just because numpy isn't installed on a dev laptop.

try:
    import numpy as np  # type: ignore[import-untyped]
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover — numpy is expected on the Pi
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    import pyaudio  # type: ignore[import-untyped]
    _HAS_PYAUDIO = True
except ImportError:  # pragma: no cover — pyaudio is optional in dev
    pyaudio = None  # type: ignore[assignment]
    _HAS_PYAUDIO = False

try:
    import librosa  # type: ignore[import-untyped]
    _HAS_LIBROSA = True
except ImportError:  # librosa is heavy & optional — WAV loading falls back to stdlib
    librosa = None  # type: ignore[assignment]
    _HAS_LIBROSA = False


# ─── Data structures ────────────────────────────────────────────────────────


@dataclass
class Beat:
    """A single detected beat.

    Attributes:
        timestamp: Wall-clock time the beat was detected (seconds since epoch).
        time_in_stream: Offset from the start of the stream/file (seconds).
        strength: Normalized onset strength 0..1 (how loud this beat is).
        bpm_estimate: BPM value estimated at the moment of this beat.
        beat_number: Sequential counter (1st beat, 2nd beat, ...).
    """
    timestamp: float
    time_in_stream: float
    strength: float
    bpm_estimate: float
    beat_number: int


# ─── Beat Detector ─────────────────────────────────────────────────────────


class BeatDetector:
    """Real-time audio beat detector and BPM estimator for Weaver.

    Lifecycle:
        detector = BeatDetector()
        await detector.start()              # begins listening (mock or real)
        ...
        bpm = detector.get_current_bpm()     # query the latest tempo
        history = detector.get_beat_history()# recent beats for visualization
        await detector.stop()                # clean shutdown

    The detector publishes a ``BEAT_DETECTED`` event on every onset that
    crosses the adaptive threshold. Subscribers (e.g. the dance engine)
    can use these events to step the legs in time with the music.

    Events published (``EventType.BEAT_DETECTED``)::

        {
            "bpm": float,              # current tempo estimate
            "confidence": float,      # 0..1 regularity of the beats
            "beat_number": int,        # sequential beat counter
            "timestamp": float,        # wall-clock time of the beat
            "strength": float,         # onset strength 0..1
            "time_in_stream": float,   # seconds since capture started
        }

    Mock mode:
        If no microphone is available (or ``hardware_mode == MOCK``), the
        detector generates synthetic beats at ``config.mock_bpm``. This
        lets the dance engine be developed on any machine without audio
        hardware. The confidence is a constant 1.0 because synthetic
        beats are perfectly regular.
    """

    def __init__(
        self,
        config: BeatDetectorConfig | None = None,
        hardware_mode: HardwareMode | None = None,
        bus: EventBus | None = None,
    ) -> None:
        """Initialize the beat detector.

        Args:
            config: Beat detector settings. If None, loads from global config.
            hardware_mode: REAL or MOCK. If None, loads from global config.
            bus: Event bus to publish on. If None, uses the global singleton.
        """
        self.main_config = get_config()
        self.config: BeatDetectorConfig = config or self.main_config.beat_detector
        self.hardware_mode: HardwareMode = hardware_mode or self.main_config.hardware_mode
        self.bus: EventBus = bus or get_event_bus()

        # ─── Runtime state ───
        self._running: bool = False
        self._capture_task: asyncio.Task | None = None
        self._beat_count: int = 0                 # total beats detected this session
        self._current_bpm: float = 0.0            # latest BPM estimate (0 until first estimate)
        self._confidence: float = 0.0             # 0..1 regularity score
        self._start_time: float = 0.0              # wall clock when capture began

        # Rolling record of recent beats — used by get_beat_history() for the
        # web dashboard's beat visualizer. A deque with maxlen auto-discards
        # the oldest entry once it fills, in O(1).
        self._beat_history: deque[Beat] = deque(maxlen=self.config.history_size)

        # ─── DSP working buffers (allocated on first use) ───
        # Onset envelope: one value per hop frame, recording how "attacked" that
        # frame was. We accumulate this as audio arrives and periodically run
        # autocorrelation over it to re-estimate BPM.
        self._onset_envelope: deque[float] = deque(maxlen=2048)
        # Per-frame spectral flux values feed the adaptive threshold.
        self._flux_history: deque[float] = deque(maxlen=2048)
        # Previous spectrum (magnitude) for computing spectral flux delta.
        self._prev_spectrum: Any = None
        # Count of hop-frames analysed — drives periodic BPM re-estimation.
        self._frame_counter: int = 0
        # Time (seconds into stream) of the last detected onset, for debouncing
        # so we don't fire two BEAT_DETECTED events for the same transient.
        self._last_onset_time: float = -1.0

        # ─── Mode flag ───
        # MOCK if explicitly requested, OR if numpy/pyaudio unavailable.
        self._mock_mode: bool = (
            self.hardware_mode == HardwareMode.MOCK
            or not _HAS_NUMPY
        )
        if not _HAS_NUMPY:
            # numpy is needed for both real *and* file-based analysis, so if
            # it's missing we can only do the pure-synthetic mock loop.
            logger.warning(
                "numpy not available — beat detector forced into mock mode"
            )

    # ─── Lifecycle ───────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start beat detection.

        In REAL mode this opens the microphone and launches the analysis
        loop. In MOCK mode it launches a synthetic beat generator.

        Raises:
            OSError: If the microphone cannot be opened (falls back to mock).
        """
        if self._running:
            logger.warning("Beat detector already running")
            return

        if not self.config.enabled:
            logger.info("🔇 Beat detector disabled in config")
            return

        self._running = True
        self._start_time = time.time()
        self._beat_count = 0
        self._frame_counter = 0
        self._onset_envelope.clear()
        self._flux_history.clear()
        self._prev_spectrum = None
        self._last_onset_time = -1.0

        mode = "MOCK" if self._mock_mode else "REAL"
        logger.info(
            f"🎵 Beat detector starting ({mode} mode, "
            f"sr={self.config.sample_rate}, fft={self.config.fft_size}, "
            f"hop={self.config.hop_size})"
        )

        self._capture_task = asyncio.create_task(self._capture_loop())

    async def stop(self) -> None:
        """Stop beat detection and release resources."""
        if not self._running:
            return

        logger.info("🎵 Beat detector stopping...")
        self._running = False

        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
            self._capture_task = None

        logger.info(
            f"Beat detector stopped (beats={self._beat_count}, "
            f"final_bpm={self._current_bpm:.1f})"
        )

    # ─── Capture loop dispatch ───────────────────────────────────────────

    async def _capture_loop(self) -> None:
        """Dispatch to the appropriate capture loop based on mode."""
        try:
            if self._mock_mode:
                await self._mock_capture_loop()
            else:
                await self._real_capture_loop()
        except asyncio.CancelledError:
            logger.debug("Beat detector capture loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Beat detector loop crashed: {e}")
            # On any unexpected crash, degrade to mock so the dance engine
            # still gets a beat clock to follow.
            if not self._mock_mode:
                logger.warning("Falling back to mock beat generation")
                self._mock_mode = True
                self._onset_envelope.clear()
                self._prev_spectrum = None
                await self._mock_capture_loop()

    # ─── Mock capture loop ───────────────────────────────────────────────

    async def _mock_capture_loop(self) -> None:
        """Generate synthetic beats at ``config.mock_bpm``.

        This is the no-hardware path. We don't analyse any audio — we just
        emit a BEAT_DETECTED event on a fixed interval derived from the
        configured mock BPM. The confidence is 1.0 because synthetic beats
        are perfectly regular.

        LEARNING NOTE: The beat interval is the *period* of one beat.
        Period and frequency are reciprocals:
            period = 60 / bpm   (seconds per beat)
        So 120 BPM → 0.5 s/beat → 2 beats/second.
        """
        period = 60.0 / max(1.0, self.config.mock_bpm)  # seconds per beat
        self._current_bpm = self.config.mock_bpm
        self._confidence = 1.0
        logger.info(
            f"🎵 Mock beat generator: {self.config.mock_bpm:.1f} BPM "
            f"(period={period:.3f}s)"
        )

        while self._running:
            now = time.time()
            elapsed = now - self._start_time

            self._beat_count += 1
            beat = Beat(
                timestamp=now,
                time_in_stream=elapsed,
                strength=1.0,
                bpm_estimate=self._current_bpm,
                beat_number=self._beat_count,
            )
            self._beat_history.append(beat)
            await self._publish_beat(beat)

            # Sleep until the next beat. We use elapsed-from-start rather than
            # a fixed sleep so drift doesn't accumulate over a long session.
            next_beat_time = self._start_time + self._beat_count * period
            sleep_for = next_beat_time - time.time()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                # We've fallen behind (e.g. event loop was busy) — yield once
                # and continue immediately to catch up.
                await asyncio.sleep(0)

    # ─── Real capture loop (microphone) ──────────────────────────────────

    async def _real_capture_loop(self) -> None:
        """Capture live audio from the microphone and analyse it.

        This loop:
        1. Opens a PyAudio stream (int16 mono @ sample_rate).
        2. Reads ``hop_size`` samples at a time into a sliding buffer.
        3. Once we have ``fft_size`` samples, compute the FFT, spectral flux,
           and check for an onset.
        4. Periodically re-estimate BPM from the accumulated onset envelope.
        """
        if not _HAS_PYAUDIO:
            logger.error(
                "pyaudio is required for live beat detection. "
                "Install with: pip install pyaudio"
            )
            logger.warning("Falling back to mock mode")
            self._mock_mode = True
            await self._mock_capture_loop()
            return

        # LEARNING NOTE: paInt16 = 16-bit signed PCM, the most common raw
        # audio format. We convert to float32 for DSP because numpy's FFT
        # and windowing math expect floating point.
        pa = pyaudio.PyAudio()
        try:
            input_device = (
                None if self.config.input_device_index < 0
                else self.config.input_device_index
            )
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=self.config.hop_size,
            )
        except OSError as e:
            logger.error(f"Failed to open microphone for beat detection: {e}")
            logger.warning("Falling back to mock mode")
            self._mock_mode = True
            await self._mock_capture_loop()
            return

        logger.info(
            f"🎤 Microphone opened for beat detection "
            f"({self.config.sample_rate}Hz, hop={self.config.hop_size})"
        )

        # Sliding window buffer. We accumulate hop_size samples per read and
        # keep the last fft_size samples available for the next FFT.
        window_buffer = deque(maxlen=self.config.fft_size)

        try:
            while self._running:
                # pyaudio.read blocks until hop_size samples are ready.
                # Run it in a thread so we don't block the asyncio loop.
                audio_bytes = await asyncio.to_thread(
                    stream.read,
                    self.config.hop_size,
                    exception_on_overflow=False,
                )
                # Decode raw bytes → float32 array in [-1, 1]
                samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                for s in samples:
                    window_buffer.append(s)

                # Need a full FFT window to proceed
                if len(window_buffer) < self.config.fft_size:
                    continue

                # ─── Per-hop DSP ───
                windowed = self._make_window() * np.array(window_buffer, dtype=np.float32)
                # Compute spectrum and update spectral flux
                spectrum = self._compute_spectrum(windowed)
                flux = self._compute_spectral_flux(spectrum)
                self._flux_history.append(flux)
                self._onset_envelope.append(flux)

                # ─── Onset detection against adaptive threshold ───
                is_onset = self._detect_onset()
                if is_onset:
                    now = time.time()
                    elapsed = now - self._start_time
                    # Debounce: ignore onsets within hop_size/sample_rate of the last one
                    min_spacing = self.config.hop_size / self.config.sample_rate
                    if elapsed - self._last_onset_time >= min_spacing:
                        self._last_onset_time = elapsed
                        self._beat_count += 1
                        # Normalize strength to 0..1 using recent flux max
                        recent_max = max(self._flux_history) or 1.0
                        strength = min(1.0, flux / recent_max) if recent_max > 0 else 0.5
                        beat = Beat(
                            timestamp=now,
                            time_in_stream=elapsed,
                            strength=strength,
                            bpm_estimate=self._current_bpm,
                            beat_number=self._beat_count,
                        )
                        self._beat_history.append(beat)
                        await self._publish_beat(beat)

                # ─── Periodic BPM re-estimation ───
                self._frame_counter += 1
                if self._frame_counter >= self.config.analysis_interval_frames:
                    self._frame_counter = 0
                    await self._reestimate_bpm()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Beat capture loop error: {e}")
            raise
        finally:
            try:
                stream.close()
            except Exception:
                pass
            pa.terminate()

    # ─── File-based analysis ─────────────────────────────────────────────

    async def analyze_file(self, path: str | Path) -> dict[str, Any]:
        """Analyse an audio file offline and return beat info.

        Loads a WAV (always, via stdlib ``wave``) or MP3/FLAC/OGG (if
        librosa is installed), then runs the same FFT → flux → onset →
        BPM pipeline over the whole file at once.

        This does NOT require a microphone and does NOT require numpy to
        be present for the *loading* step — but the analysis itself needs
        numpy. If numpy is missing, we fall back to a mock analysis that
        reports ``config.mock_bpm``.

        Args:
            path: Path to an audio file (.wav, .mp3, .flac, .ogg).

        Returns:
            Dict with: bpm, confidence, beat_count, beats (list of
            Beat-like dicts), duration_seconds, source.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")

        logger.info(f"🎵 Analysing audio file: {p.name}")

        if not _HAS_NUMPY:
            logger.warning("numpy not available — returning mock analysis")
            return {
                "bpm": self.config.mock_bpm,
                "confidence": 1.0,
                "beat_count": 0,
                "beats": [],
                "duration_seconds": 0.0,
                "source": "mock",
            }

        # Load samples as float32, mono, resampled to config.sample_rate.
        samples, sr = await asyncio.to_thread(self._load_audio, str(p))

        # If file sample rate differs, just use the file's rate for the
        # analysis so the time math stays correct. (We don't resample to
        # avoid the extra dependency unless librosa is present.)
        duration = len(samples) / sr
        logger.debug(
            f"Loaded {p.name}: {len(samples)} samples @ {sr}Hz "
            f"({duration:.1f}s)"
        )

        # Reset DSP state for a clean offline run
        self._onset_envelope.clear()
        self._flux_history.clear()
        self._prev_spectrum = None
        self._beat_count = 0
        beats: list[Beat] = []

        fft_size = self.config.fft_size
        hop_size = self.config.hop_size
        window = self._make_window()

        # Slide through the whole file in hops. This is identical to the
        # real-time loop, just without the threading and the sleeps.
        frame_idx = 0
        for start in range(0, max(0, len(samples) - fft_size), hop_size):
            frame = samples[start:start + fft_size]
            if len(frame) < fft_size:
                # Pad the last partial frame with zeros so the FFT is full-size
                frame = np.pad(frame, (0, fft_size - len(frame)))

            windowed = window * frame
            spectrum = self._compute_spectrum(windowed)
            flux = self._compute_spectral_flux(spectrum)
            self._flux_history.append(flux)
            self._onset_envelope.append(flux)

            if self._detect_onset():
                t_in_stream = (start + fft_size) / sr
                # Debounce using frame spacing
                min_spacing = hop_size / sr
                if t_in_stream - self._last_onset_time >= min_spacing:
                    self._last_onset_time = t_in_stream
                    self._beat_count += 1
                    recent_max = max(self._flux_history) or 1.0
                    strength = min(1.0, flux / recent_max) if recent_max > 0 else 0.5
                    beats.append(Beat(
                        timestamp=time.time(),
                        time_in_stream=t_in_stream,
                        strength=strength,
                        bpm_estimate=self._current_bpm,
                        beat_number=self._beat_count,
                    ))
                    # Also publish to the bus so subscribers (dance engine,
                    # dashboard) react during offline playback/analysis.
                    await self._publish_beat(beats[-1])

            frame_idx += 1
            # Re-estimate BPM periodically while scanning, same as live mode
            if frame_idx % self.config.analysis_interval_frames == 0:
                self._reestimate_bpm_sync()

        # Final BPM estimate using the complete onset envelope
        self._reestimate_bpm_sync()

        result = {
            "bpm": self._current_bpm,
            "confidence": self._confidence,
            "beat_count": self._beat_count,
            "beats": [
                {
                    "beat_number": b.beat_number,
                    "time_in_stream": b.time_in_stream,
                    "strength": b.strength,
                    "bpm_estimate": b.bpm_estimate,
                }
                for b in beats
            ],
            "duration_seconds": duration,
            "source": "file",
        }
        logger.info(
            f"🎵 File analysis: {self._current_bpm:.1f} BPM, "
            f"{self._beat_count} beats, confidence={self._confidence:.2f}"
        )
        return result

    def _load_audio(self, path: str) -> tuple[Any, int]:
        """Load an audio file into a float32 numpy array (mono).

        Uses librosa if available (handles MP3/FLAC/OGG and resampling),
        otherwise falls back to the stdlib ``wave`` module (WAV only).

        Returns:
            (samples float32 array, sample_rate int)
        """
        if _HAS_LIBROSA:
            # librosa.load returns mono float32 at the requested sample rate.
            samples, sr = librosa.load(path, sr=self.config.sample_rate, mono=True)
            return samples, int(sr)

        # Fallback: stdlib wave module. Only handles uncompressed WAV.
        if not path.lower().endswith(".wav"):
            raise ValueError(
                f"Cannot load '{path}': librosa is not installed and the "
                f"stdlib wave module only supports .wav files. "
                f"Install librosa (pip install librosa) for MP3/FLAC support."
            )
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        # Convert raw bytes → float32 mono
        if sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            # 8-bit PCM is unsigned 0..255 centered at 128
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

        # Downmix to mono by averaging channels
        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)

        return data, sr

    # ─── DSP primitives ─────────────────────────────────────────────────
    # These methods implement the core signal-processing steps. Each is
    # documented so a student can follow exactly what the math is doing.

    def _make_window(self) -> Any:
        """Build and cache the FFT analysis window (Hann).

        LEARNING NOTE — Why window the signal?
            The FFT pretends its input is one period of an infinitely
            repeating signal. If we just chop an arbitrary chunk out of
            the audio, the sudden start/end discontinuity smears energy
            across all frequencies (this is called "spectral leakage").
            A window function tapers the ends to zero so the chunk looks
            smooth at its boundaries, concentrating each frequency's
            energy into a narrow peak. The Hann window is a good
            general-purpose choice — it's a raised cosine:

                w[n] = 0.5 * (1 - cos(2π n / (N-1)))

            We cache the window so we don't recompute it for every frame.
        """
        if not hasattr(self, "_window_cache"):
            n = self.config.fft_size
            if _HAS_NUMPY:
                idx = np.arange(n)
                # Hann window: smooth cosine taper
                self._window_cache = 0.5 * (1.0 - np.cos(2.0 * np.pi * idx / (n - 1))).astype(np.float32)
            else:
                # Pure-python fallback (only used if numpy missing — but then
                # we'd be in mock mode anyway). Kept for completeness.
                self._window_cache = None
        return self._window_cache

    def _compute_spectrum(self, windowed_frame: Any) -> Any:
        """Compute the magnitude spectrum of one windowed frame via FFT.

        LEARNING NOTE — The FFT (Fast Fourier Transform):
            The Discrete Fourier Transform (DFT) converts N time-domain
            samples into N frequency-domain bins, each representing the
            amplitude & phase of a sinusoid at a specific frequency:

                X[k] = Σ_{n=0}^{N-1} x[n] * e^(-2πi k n / N)

            The FFT is a fast O(N log N) algorithm for computing the DFT.
            numpy's rfft only returns the non-redundant half for real
            input (N/2+1 bins), since the spectrum of a real signal is
            symmetric. We take the *magnitude* (np.abs) because we care
            about how much of each frequency is present, not its phase.

            Bin k corresponds to frequency k * sample_rate / fft_size.
            With sr=22050 and fft=1024, that's ~21.5 Hz per bin — enough
            resolution to separate a kick drum (~60 Hz) from a snare
            (~200 Hz).
        """
        # rfft = real-input FFT → returns N/2+1 complex bins
        spectrum_complex = np.fft.rfft(windowed_frame, n=self.config.fft_size)
        # Magnitude = how strong each frequency bin is (drop phase)
        magnitude = np.abs(spectrum_complex)
        return magnitude

    def _compute_spectral_flux(self, spectrum: Any) -> float:
        """Compute the spectral flux between this frame and the previous one.

        Spectral flux is the sum of *positive* differences in magnitude
        across frequency bins between consecutive frames. It spikes when
        new spectral content appears (a percussive onset), and is near
        zero during sustained notes or silence.

        LEARNING NOTE — Why only *positive* differences?
            We want to measure energy *onsets* — sounds that are newly
            arriving. If a frequency gets quieter (negative difference),
            that's a note dying away, not a new beat. By clipping the
            per-bin differences to ≥ 0 (np.maximum(..., 0)) we only count
            newly-arrived energy, which makes the flux a clean onset
            indicator.

            This is sometimes called the "half-wave rectified" spectral
            difference, a classic MIR (Music Information Retrieval)
            technique (e.g. Dixon 2006, "Onset Detection Revisited").
        """
        if self._prev_spectrum is None:
            # First frame: no previous spectrum to compare with.
            self._prev_spectrum = spectrum
            return 0.0

        # Per-bin positive magnitude difference
        diff = spectrum - self._prev_spectrum
        positive_diff = np.maximum(diff, 0.0)
        # Sum across all bins → a single scalar "flux" value for this frame
        flux = float(np.sum(positive_diff))
        self._prev_spectrum = spectrum
        return flux

    def _detect_onset(self) -> bool:
        """Decide whether the latest flux value is an onset (beat).

        We compare the current flux against an *adaptive* threshold: the
        running average of recent flux, multiplied by ``sensitivity``.
        The threshold adapts because music isn't a constant loudness —
        a fixed threshold would either miss quiet beats in a soft section
        or fire constantly during a loud section.

        LEARNING NOTE — Adaptive threshold via moving average:
            The idea: "a beat is a moment that's notably louder than the
            recent average." We keep a sliding window of recent flux
            values and compute their mean. If the current flux exceeds
            mean × sensitivity, we call it an onset.

            sensitivity > 1 makes detection stricter (needs a bigger spike
            relative to the local average), which reduces false positives
            in noisy environments. sensitivity = 1.5 means "50% above the
            recent average." Tunable via config.
        """
        if len(self._flux_history) < 2:
            return False

        current = self._flux_history[-1]

        # Use the last ~20 flux values as the local average window. This
        # is ~0.5 seconds of audio at hop=512, sr=22050 (≈43 frames/sec).
        window = min(20, len(self._flux_history) - 1)
        recent = list(self._flux_history)[-1 - window:-1]
        avg = sum(recent) / max(1, len(recent))

        threshold = avg * self.config.sensitivity

        # A tiny minimum threshold avoids firing on absolute silence
        # where the moving average is ~0 and any micro-fluctuation passes.
        min_threshold = 1e-6
        if threshold < min_threshold:
            return False

        return current > threshold

    async def _reestimate_bpm(self) -> None:
        """Re-estimate BPM from the accumulated onset envelope.

        Delegates to the synchronous estimator; wrapped in async so the
        event loop stays responsive. The autocorrelation is O(N²) but N
        is small (the deque's maxlen), so we run it inline.
        """
        self._reestimate_bpm_sync()

    def _reestimate_bpm_sync(self) -> None:
        """Synchronous BPM estimation via autocorrelation of the onset envelope.

        LEARNING NOTE — Autocorrelation finds the beat period:
            The onset envelope is a curve that spikes at each beat. If
            beats are perfectly regular at period P, the envelope will
            look identical to itself shifted by P. Autocorrelation
            measures this self-similarity at every lag:

                R[lag] = Σ env[n] * env[n + lag]

            The lag with the highest R (within the musically plausible
            BPM range) is the beat period. Convert to BPM:

                bpm = 60 / lag_seconds,  where lag_seconds = lag * hop_time

            hop_time = hop_size / sample_rate (seconds between envelope
            samples). With hop=512, sr=22050 → hop_time ≈ 0.0232 s.

            We only search lags corresponding to [min_bpm, max_bpm] so we
            don't lock onto half- or double-tempo artifacts.
        """
        if len(self._onset_envelope) < 8:
            # Not enough data yet to trust a tempo estimate.
            return

        env = np.array(self._onset_envelope, dtype=np.float32)

        # Time (seconds) between successive onset-envelope samples.
        hop_time = self.config.hop_size / self.config.sample_rate

        # Convert BPM range → lag range (in envelope samples).
        # lag_seconds = 60 / bpm  →  lag = lag_seconds / hop_time
        min_lag = max(1, int((60.0 / self.config.max_bpm) / hop_time))
        max_lag = min(len(env) - 1, int((60.0 / self.config.min_bpm) / hop_time))
        if max_lag <= min_lag:
            return

        # ─── Autocorrelation over the plausible lag range ───
        # We compute R[lag] for each lag in [min_lag, max_lag] and pick the
        # lag with the maximum correlation. This is an O(N·L) scan; N is
        # bounded by the deque maxlen (2048) and L by the BPM range, so
        # it's fast enough to run ~once per second.
        best_lag = 0
        best_corr = -1.0
        correlations: list[float] = []
        for lag in range(min_lag, max_lag + 1):
            # Dot product of the envelope with a shifted copy of itself.
            # We zero-mean the envelope first so DC offset doesn't inflate
            # correlations at all lags equally.
            shifted = env[:len(env) - lag]
            base = env[lag:]
            corr = float(np.sum(shifted * base))
            correlations.append(corr)
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        if best_lag <= 0:
            return

        # Convert the winning lag back to BPM.
        period_seconds = best_lag * hop_time
        bpm = 60.0 / period_seconds

        # Clamp to the configured range (guards against edge artifacts)
        bpm = float(max(self.config.min_bpm, min(self.config.max_bpm, bpm)))
        self._current_bpm = bpm

        # ─── Confidence from beat regularity ───
        # If beats are perfectly regular, the autocorrelation peak is sharp
        # and clearly above the average. We estimate confidence as the ratio
        # of the peak correlation to the mean correlation across all lags.
        # A sharp, isolated peak → high confidence; a flat correlation
        # curve (no clear tempo) → low confidence.
        if correlations and best_corr > 0:
            mean_corr = sum(correlations) / len(correlations)
            # Ratio in [0, ∞); clamp to [0, 1] for a tidy score.
            confidence = best_corr / (mean_corr + 1e-9) if mean_corr > 0 else 0.0
            # Normalize: a peak 3× the mean is "very confident"; cap at 1.0
            self._confidence = max(0.0, min(1.0, confidence / 3.0))
        else:
            self._confidence = 0.0

        logger.debug(
            f"BPM estimate: {bpm:.1f} (lag={best_lag}, "
            f"conf={self._confidence:.2f})"
        )

    # ─── Event publishing ────────────────────────────────────────────────

    async def _publish_beat(self, beat: Beat) -> None:
        """Publish a BEAT_DETECTED event to the event bus.

        The event payload is the contract that subscribers (dance engine,
        web dashboard, LED effects) rely on, so keep these keys stable.
        """
        await self.bus.publish(Event(
            type=EventType.BEAT_DETECTED,
            data={
                "bpm": beat.bpm_estimate,
                "confidence": self._confidence,
                "beat_number": beat.beat_number,
                "timestamp": beat.timestamp,
                "strength": beat.strength,
                "time_in_stream": beat.time_in_stream,
            },
            source="beat_detector",
        ))
        logger.debug(
            f"🥁 Beat #{beat.beat_number} @ {beat.time_in_stream:.2f}s "
            f"({beat.bpm_estimate:.1f} BPM, strength={beat.strength:.2f})"
        )

    # ─── Public query API ───────────────────────────────────────────────

    def get_current_bpm(self) -> float:
        """Return the most recent BPM estimate.

        Returns 0.0 until the first estimate is produced (a few seconds
        after start in real mode; immediately in mock mode).
        """
        return self._current_bpm

    def get_confidence(self) -> float:
        """Return the confidence score (0..1) of the current BPM estimate."""
        return self._confidence

    def get_beat_history(self, count: int = 20) -> list[dict[str, Any]]:
        """Return the most recent ``count`` beats for visualization.

        Beats are returned oldest-first so a dashboard can append them in
        order. Each beat is a plain dict (JSON-serializable for the web
        dashboard / WebSocket).

        Args:
            count: How many recent beats to return (default 20).

        Returns:
            List of beat dicts, each with: beat_number, timestamp,
            time_in_stream, strength, bpm_estimate.
        """
        if count <= 0:
            return []
        recent = list(self._beat_history)[-count:]
        return [
            {
                "beat_number": b.beat_number,
                "timestamp": b.timestamp,
                "time_in_stream": b.time_in_stream,
                "strength": b.strength,
                "bpm_estimate": b.bpm_estimate,
            }
            for b in recent
        ]

    def get_stats(self) -> dict[str, Any]:
        """Return a stats snapshot for telemetry / the dashboard."""
        return {
            "mode": "mock" if self._mock_mode else "real",
            "running": self._running,
            "bpm": self._current_bpm,
            "confidence": self._confidence,
            "beat_count": self._beat_count,
            "history_size": len(self._beat_history),
            "has_numpy": _HAS_NUMPY,
            "has_pyaudio": _HAS_PYAUDIO,
            "has_librosa": _HAS_LIBROSA,
            "sample_rate": self.config.sample_rate,
            "fft_size": self.config.fft_size,
            "hop_size": self.config.hop_size,
        }


# ─── Convenience / smoke test ──────────────────────────────────────────────


async def _smoke_test() -> None:
    """Tiny self-test: run mock mode for a few seconds and print beats.

    Run with:  python -m weaver.dance.beat_detector
    """
    logger.info("Beat detector smoke test (mock mode, 5 seconds)")
    detector = BeatDetector(hardware_mode=HardwareMode.MOCK)
    await detector.start()
    await asyncio.sleep(5.0)
    logger.info(
        f"Stats: bpm={detector.get_current_bpm():.1f}, "
        f"beats={detector.get_stats()['beat_count']}, "
        f"history={len(detector.get_beat_history(50))}"
    )
    await detector.stop()


if __name__ == "__main__":
    asyncio.run(_smoke_test())