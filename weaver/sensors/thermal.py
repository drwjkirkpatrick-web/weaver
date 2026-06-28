# thermal.py — Pi 5 CPU temperature monitor for Weaver
#
# The Raspberry Pi 5's CPU runs hot under load — especially when running
# an LLM, camera CV processing, and 18 servos simultaneously. This module
# monitors the CPU temperature via the Linux sysfs interface and alerts
# the system when it gets too hot.
#
# No external libraries needed! The Pi's thermal zone is exposed as a
# plain text file at /sys/class/thermal/thermal_zone0/temp. We just read
# it and divide by 1000 (it's in millicelsius).
#
# Why this matters for a robot:
#   - At ~80°C, the Pi 5 starts thermal throttling (reduces CPU clock).
#   - At ~85°C, the firmware forces shutdown to prevent damage.
#   - When throttled, the LLM cortex slows down → delayed decisions →
#     the robot might walk into things.
#   - Our thermal monitor lets the cortex proactively reduce its workload
#     (skip non-essential LLM calls, slow down gait) before throttling kicks in.
#
# Learning notes:
#   - The Linux thermal framework exposes sensors under /sys/class/thermal/.
#     Each "thermal zone" has a temp file (in millicelsius) and trip points
#     (threshold temperatures). The kernel handles hardware throttling; we
#     just read and report.
#   - "thermal_zone0" is the CPU/GPU SoC temperature on the Pi. On Pi 5,
#     this is the BCM2712 SoC temperature.
#   - We poll rather than use inotify because: (1) the file always exists,
#     (2) polling at 1Hz is cheap, and (3) async inotify adds complexity
#     for no real benefit at this sample rate.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from loguru import logger

from weaver.config import HardwareMode, ThermalConfig, get_config
from weaver.event_bus import Event, EventType, get_event_bus

# ─── Constants ────────────────────────────────────────────────────────────

# The Pi's CPU thermal zone sysfs path (in millicelsius).
# This is the same on all Pi models (Pi 3, 4, 5).
_THERMAL_ZONE_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


class ThermalSensor:
    """Raspberry Pi 5 CPU temperature monitor.

    This sensor reads the CPU/SoC temperature from the Linux sysfs
    thermal zone interface. No external libraries are required — it's
    just reading a file.

    Temperature thresholds:
        - Normal:    < 70°C  (green)
        - Caution:  70-75°C  (yellow — start throttling LLM usage)
        - Warning:  75-80°C  (orange — reduce motion speed)
        - Critical: ≥ 80°C   (red — stop all motion, alert)

    Events published:
        - THERMAL_STATUS:   periodic temperature update
        - THERMAL_WARNING:  when temp crosses the warning threshold
        - THERMAL_CRITICAL: when temp crosses the critical threshold

    Unlike other sensors, this module always runs in "real" mode on a Pi
    (the sysfs file always exists). In mock mode (e.g., on a dev laptop),
    it generates synthetic temperatures since the sysfs path won't exist.
    """

    def __init__(self, config: ThermalConfig | None = None) -> None:
        """Initialize the thermal sensor.

        Args:
            config: Thermal configuration. If None, loads from global config.
        """
        cfg = get_config()
        self.config: ThermalConfig = config or cfg.thermal
        self.hardware_mode: HardwareMode = cfg.hardware_mode

        self.bus = get_event_bus()

        # Running task
        self._task: asyncio.Task | None = None

        # Whether we can read the real sysfs file
        self._sysfs_available: bool = _THERMAL_ZONE_PATH.exists()

        # State tracking for alert edges (avoid republishing every cycle)
        self._warning_active: bool = False
        self._critical_active: bool = False

        # Current temperature (for on-demand queries)
        self._current_temp: float = 45.0

        # Mock mode state — used when sysfs isn't available OR hardware mode
        # is explicitly MOCK.
        self._mock: bool = self._should_use_mock()
        self._mock_time: float = 0.0

    def _should_use_mock(self) -> bool:
        """Determine whether to use mock mode.

        Thermal sensor is special: it uses sysfs (no external library), so
        "real mode" works on any Linux machine with a thermal zone. But in
        MOCK hardware mode, or when the sysfs file doesn't exist (e.g.,
        macOS dev machine), we simulate.
        """
        if self.hardware_mode == HardwareMode.MOCK:
            return True
        if not self._sysfs_available:
            logger.warning(
                f"🌡️  Thermal zone not found at {_THERMAL_ZONE_PATH} — "
                "running in MOCK mode (synthetic temperatures)"
            )
            return True
        return False

    async def start(self) -> None:
        """Start the thermal monitoring loop."""
        logger.info("🌡️  Thermal sensor starting (mode: {})", "MOCK" if self._mock else "REAL")

        if not self.config.enabled:
            logger.info("🌡️  Thermal monitoring disabled in config — skipping start")
            return

        self._task = asyncio.create_task(self._poll_loop())

        logger.info(
            "✅ Thermal sensor started (sample rate: {}Hz, "
            "warning: {}°C, critical: {}°C)",
            self.config.sample_rate_hz,
            self.config.warning_temp,
            self.config.critical_temp,
        )

    async def stop(self) -> None:
        """Stop the thermal monitoring loop."""
        logger.info("🌡️  Thermal sensor stopping...")

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("✅ Thermal sensor stopped")

    # ─── Polling Loop ─────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Main polling loop: read temperature, check thresholds, publish events."""
        interval = 1.0 / self.config.sample_rate_hz

        while True:
            try:
                temp = await self._read_temperature()

                if temp is not None:
                    self._current_temp = temp

                    # Publish thermal status (always)
                    await self.bus.publish(Event(
                        type=EventType.THERMAL_STATUS,
                        data={
                            "temp_c": round(temp, 1),
                            "warning": temp >= self.config.warning_temp,
                            "critical": temp >= self.config.critical_temp,
                            "throttle_recommended": temp >= self.config.llm_throttle_temp,
                            "timestamp": time.time(),
                        },
                        source="thermal",
                    ))

                    # ── Check warning threshold (edge-triggered) ──────────
                    if temp >= self.config.critical_temp:
                        if not self._critical_active:
                            await self.bus.publish(Event(
                                type=EventType.THERMAL_CRITICAL,
                                data={
                                    "temp_c": round(temp, 1),
                                    "threshold": self.config.critical_temp,
                                },
                                source="thermal",
                            ))
                            self._critical_active = True
                            self._warning_active = True  # Critical implies warning
                            logger.error(
                                f"🛑 THERMAL CRITICAL: {temp:.1f}°C — "
                                f"stop all motion!"
                            )

                    elif temp >= self.config.warning_temp:
                        if not self._warning_active:
                            await self.bus.publish(Event(
                                type=EventType.THERMAL_WARNING,
                                data={
                                    "temp_c": round(temp, 1),
                                    "threshold": self.config.warning_temp,
                                },
                                source="thermal",
                            ))
                            self._warning_active = True
                            self._critical_active = False  # No longer critical
                            logger.warning(
                                f"⚠️  THERMAL WARNING: {temp:.1f}°C — "
                                f"reduce activity"
                            )

                    else:
                        # Temperature is back to normal — clear alert flags
                        if self._warning_active or self._critical_active:
                            self._warning_active = False
                            self._critical_active = False
                            logger.info(
                                f"🌡️  Temperature normal: {temp:.1f}°C — alerts cleared"
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Thermal poll loop error: {e}")

            await asyncio.sleep(interval)

    async def _read_temperature(self) -> float | None:
        """Read the CPU temperature from the sysfs thermal zone.

        Returns:
            Temperature in degrees Celsius, or None on error.

        The sysfs file contains the temperature in **millicelsius** — e.g.,
        "45670" means 45.670°C. We divide by 1000 to get Celsius.

        Reading a small text file is fast enough (~0.1ms) that we don't
        need a separate thread, but we yield to the event loop anyway.
        """
        if self._mock:
            return self._mock_temperature()

        try:
            # Read the sysfs file (contains millicelsius as ASCII text)
            raw = _THERMAL_ZONE_PATH.read_text().strip()
            temp_mc = int(raw)
            temp_c = temp_mc / 1000.0

            # Yield to event loop (file read is technically blocking, but ~0.1ms)
            await asyncio.sleep(0)

            return temp_c

        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Thermal zone read error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected thermal read error: {e}")
            return None

    def _mock_temperature(self) -> float:
        """Generate a simulated CPU temperature for mock mode.

        Simulates the Pi 5 CPU temperature under varying load:
        - Base temperature ~50°C (idle)
        - Periodic spikes to ~72°C (LLM inference bursts)
        - Occasional sustained high temp (simulates sustained load)

        This exercises both the warning and critical thresholds to test
        the alerting logic.
        """
        self._mock_time += 1.0 / self.config.sample_rate_hz
        t = self._mock_time

        # Base idle temperature
        base = 50.0

        # Simulate periodic load spikes every 30 seconds (LLM inference)
        # Each spike lasts ~5 seconds and raises temp by ~20°C
        spike_phase = t % 30.0
        if 10.0 < spike_phase < 15.0:
            # Rising and falling temperature during a load spike
            spike_progress = (spike_phase - 10.0) / 5.0  # 0.0 → 1.0
            # Bell curve: peaks at spike_progress=0.5
            spike = 20.0 * (1.0 - (2.0 * spike_progress - 1.0) ** 2)
            base += spike

        # Add small random noise (±0.5°C) for realism
        noise = (hash(str(t)) % 100 - 50) / 100.0  # Deterministic noise
        temp = base + noise

        return round(temp, 1)

    # ─── Public API ───────────────────────────────────────────────────

    def get_temperature(self) -> float:
        """Get the current CPU temperature (for dashboard/queries).

        Returns:
            Current temperature in degrees Celsius.
        """
        return self._current_temp

    def get_status(self) -> dict[str, Any]:
        """Get current thermal status (for dashboard/queries).

        Returns:
            Dict with temperature and alert states.
        """
        return {
            "temp_c": round(self._current_temp, 1),
            "warning": self._warning_active,
            "critical": self._critical_active,
            "throttle_recommended": self._current_temp >= self.config.llm_throttle_temp,
            "warning_threshold": self.config.warning_temp,
            "critical_threshold": self.config.critical_temp,
        }