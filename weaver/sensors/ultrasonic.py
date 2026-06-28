# ultrasonic.py — HC-SR04 ultrasonic distance sensor for Weaver
#
# The HC-SR04 is the classic cheap ultrasonic rangefinder. It sends a 40kHz
# ultrasonic pulse and measures the time for the echo to return. Using the
# speed of sound, we convert that time to distance.
#
# Hardware: HC-SR04 module
#   - TRIG pin → GPIO 23 (output: we send the pulse)
#   - ECHO pin → GPIO 24 (input: we measure the response)
#   - IMPORTANT: ECHO is 5V, Pi GPIO is 3.3V! A voltage divider (1kΩ + 2kΩ)
#     is required to protect the Pi pin. The RaspClaws kit includes this.
#
# Library: lgpio — the native GPIO library for Pi 5 (replaces RPi.GPIO,
# which doesn't support the Pi 5's new RP1 chip).
#
# Learning notes:
#   - Speed of sound at 20°C ≈ 343 m/s = 0.0343 cm/µs
#   - Distance = (echo_time × speed_of_sound) / 2  (divide by 2: round trip)
#   - The sensor has a minimum range of ~2cm (too close = echo returns
#     before the pulse finishes transmitting) and max ~400cm.
#   - Ultrasonic sensors have a cone-shaped beam (~15° half-angle). They
#     detect the nearest object within that cone, but can miss objects
#     outside it (e.g., thin table legs at an angle).
#   - We take multiple samples and use the **median** (not mean!) because
#     the mean is skewed by outliers (e.g., a misread 400cm max when the
#     pulse is lost), while the median is robust to a few bad readings.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import random
import statistics
import time
from typing import Any

from loguru import logger

from weaver.config import HardwareMode, UltrasonicConfig, get_config
from weaver.event_bus import Event, EventType, get_event_bus

# ─── Optional hardware import ─────────────────────────────────────────────

try:
    import lgpio  # type: ignore[import-untyped]
    _LGPIO_AVAILABLE = True
except ImportError:
    _LGPIO_AVAILABLE = False


class UltrasonicSensor:
    """HC-SR04 ultrasonic distance sensor with noise reduction.

    This sensor measures distance by timing ultrasonic echoes. It takes
    multiple samples per reading and uses the **median** to filter out
    noise and outliers — much more robust than a single reading.

    The median filter is key here: ultrasonic sensors frequently produce
    "ghost" readings (e.g., 400cm when the echo is lost, or 2cm from
    crosstalk). The mean would be pulled by these outliers, but the median
    ignores them as long as they're fewer than half the samples.

    Events published:
        - ULTRASONIC_RANGE:    every reading with distance_cm
        - ULTRASONIC_OBSTACLE: when distance < obstacle threshold

    Mock mode:
        Generates random distances centered around a "wall" at 50cm,
        simulating the kind of readings you'd get in a room.
    """

    # Speed of sound at 20°C in cm/µs (for distance calculation)
    SPEED_OF_SOUND_CM_PER_US: float = 0.0343

    # Obstacle threshold (cm) — read from safety config at init.
    _OBSTACLE_THRESHOLD_DEFAULT: float = 30.0

    def __init__(self, config: UltrasonicConfig | None = None) -> None:
        """Initialize the ultrasonic sensor.

        Args:
            config: Ultrasonic configuration. If None, loads from global config.
        """
        cfg = get_config()
        self.config: UltrasonicConfig = config or cfg.ultrasonic
        self.hardware_mode: HardwareMode = cfg.hardware_mode

        # Pull obstacle threshold from safety config
        self._obstacle_threshold: float = cfg.safety.obstacle_stop_distance_cm

        self.bus = get_event_bus()

        # lgpio chip handle (real mode)
        self._gpio_chip: int = 0
        self._gpio_handle: int = 0

        # Running task
        self._task: asyncio.Task | None = None

        # Mock mode
        self._mock: bool = self._should_use_mock()

    def _should_use_mock(self) -> bool:
        """Determine whether to use mock mode."""
        if self.hardware_mode == HardwareMode.MOCK:
            return True
        if not _LGPIO_AVAILABLE:
            logger.warning(
                "📏 lgpio not installed — ultrasonic running in MOCK mode. "
                "Install with: pip install lgpio"
            )
            return True
        return False

    async def start(self) -> None:
        """Start the ultrasonic sensor polling loop."""
        logger.info("📏 Ultrasonic sensor starting (mode: {})", "MOCK" if self._mock else "REAL")

        if not self._mock:
            await self._init_real_hardware()

        self._task = asyncio.create_task(self._poll_loop())

        logger.info(
            "✅ Ultrasonic sensor started (sample rate: {}Hz, {} samples/reading)",
            self.config.sample_rate_hz,
            self.config.samples_per_reading,
        )

    async def stop(self) -> None:
        """Stop the sensor and release GPIO resources."""
        logger.info("📏 Ultrasonic sensor stopping...")

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if not self._mock and _LGPIO_AVAILABLE:
            try:
                lgpio.gpiochip_close(self._gpio_handle)
            except Exception as e:
                logger.error(f"Error closing GPIO chip: {e}")

        logger.info("✅ Ultrasonic sensor stopped")

    # ─── Initialization ───────────────────────────────────────────────

    async def _init_real_hardware(self) -> None:
        """Initialize GPIO pins for the HC-SR04."""
        try:
            # lgpio uses a "chip" handle. On Pi 5, gpiochip4 is the RP1.
            # lgpio.gpiochip_open(0) opens the default chip.
            self._gpio_handle = lgpio.gpiochip_open(0)

            # Claim TRIG pin as output (we send the pulse)
            lgpio.gpio_claim_output(
                self._gpio_handle,
                self.config.trigger_pin,
                0,  # Initial level: LOW
            )

            # Claim ECHO pin as input (we read the response)
            lgpio.gpio_claim_input(
                self._gpio_handle,
                self.config.echo_pin,
            )

            logger.info(
                f"📏 HC-SR04 initialized: TRIG=GPIO{self.config.trigger_pin}, "
                f"ECHO=GPIO{self.config.echo_pin}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize GPIO: {e} — falling back to MOCK mode")
            self._mock = True

    # ─── Polling Loop ─────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Main polling loop: take readings and publish events."""
        interval = 1.0 / self.config.sample_rate_hz

        while True:
            try:
                # Take multiple samples and use median for noise reduction
                distance = await self._take_reading()

                if distance is not None:
                    # Publish range reading
                    await self.bus.publish(Event(
                        type=EventType.ULTRASONIC_RANGE,
                        data={
                            "distance_cm": round(distance, 1),
                            "timestamp": time.time(),
                        },
                        source="ultrasonic",
                    ))

                    # Check for obstacle
                    if distance < self._obstacle_threshold:
                        await self.bus.publish(Event(
                            type=EventType.ULTRASONIC_OBSTACLE,
                            data={
                                "distance_cm": round(distance, 1),
                                "threshold": self._obstacle_threshold,
                            },
                            source="ultrasonic",
                        ))
                        logger.warning(
                            f"🚧 Obstacle detected at {distance:.1f}cm "
                            f"(threshold: {self._obstacle_threshold}cm)"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ultrasonic poll loop error: {e}")

            await asyncio.sleep(interval)

    async def _take_reading(self) -> float | None:
        """Take a single distance reading with multi-sample median filtering.

        Takes ``samples_per_reading`` individual measurements and returns
        the median. This filters out outliers (ghost readings).

        Returns:
            Distance in cm, or None if all samples failed.
        """
        samples: list[float] = []

        for _ in range(self.config.samples_per_reading):
            dist = await self._single_measurement()
            if dist is not None and 2.0 <= dist <= self.config.max_distance_cm:
                samples.append(dist)
            # Small delay between samples to let the echo settle
            await asyncio.sleep(0.01)

        if not samples:
            return None

        # Use median — robust to outliers (e.g., one 400cm ghost reading)
        return statistics.median(samples)

    async def _single_measurement(self) -> float | None:
        """Take a single ultrasonic distance measurement.

        The HC-SR04 protocol:
        1. Send a 10µs HIGH pulse on TRIG
        2. Wait for ECHO to go HIGH
        3. Measure how long ECHO stays HIGH (that's the echo time)
        4. Distance = (echo_time × speed_of_sound) / 2

        Returns:
            Distance in cm, or None if timeout/error.
        """
        if self._mock:
            return self._mock_measurement()

        try:
            # ── Step 1: Send trigger pulse ────────────────────────────
            # A 10µs HIGH pulse on TRIG starts the measurement.
            lgpio.gpio_write(self._gpio_handle, self.config.trigger_pin, 0)
            time.sleep(0.000002)  # 2µs settle
            lgpio.gpio_write(self._gpio_handle, self.config.trigger_pin, 1)
            time.sleep(0.000010)  # 10µs trigger pulse
            lgpio.gpio_write(self._gpio_handle, self.config.trigger_pin, 0)

            # ── Step 2: Wait for ECHO to go HIGH ──────────────────────
            # The sensor processes the trigger for ~4.5ms before sending
            # the ultrasonic burst and raising ECHO.
            pulse_start = time.time()
            timeout_start = pulse_start

            while lgpio.gpio_read(self._gpio_handle, self.config.echo_pin) == 0:
                pulse_start = time.time()
                if pulse_start - timeout_start > self.config.timeout_seconds:
                    logger.debug("Ultrasonic: timeout waiting for ECHO HIGH")
                    return None

            # ── Step 3: Measure ECHO pulse duration ──────────────────
            # ECHO stays HIGH for the duration of the ultrasonic round trip.
            pulse_end = time.time()
            timeout_start = pulse_end

            while lgpio.gpio_read(self._gpio_handle, self.config.echo_pin) == 1:
                pulse_end = time.time()
                if pulse_end - timeout_start > self.config.timeout_seconds:
                    logger.debug("Ultrasonic: timeout waiting for ECHO LOW")
                    return None

            # ── Step 4: Calculate distance ────────────────────────────
            # Duration of ECHO HIGH pulse (in microseconds)
            pulse_duration_us = (pulse_end - pulse_start) * 1_000_000

            # Distance = (time × speed) / 2 (round trip)
            # Speed of sound ≈ 0.0343 cm/µs at 20°C
            distance_cm = (pulse_duration_us * self.SPEED_OF_SOUND_CM_PER_US) / 2.0

            # Yield to event loop (the GPIO timing is blocking)
            await asyncio.sleep(0)

            return distance_cm

        except Exception as e:
            logger.error(f"Ultrasonic measurement error: {e}")
            return None

    def _mock_measurement(self) -> float | None:
        """Generate a simulated distance reading for mock mode.

        Simulates a robot in a room with a wall at ~50cm. Occasionally
        introduces outliers (ghost readings at max range) to exercise the
        median filter.
        """
        # Base distance: 50cm with some noise
        base = 50.0 + random.gauss(0, 2.0)

        # 10% chance of a ghost reading (simulates echo loss → max range)
        if random.random() < 0.10:
            return self.config.max_distance_cm

        # 5% chance of a very close reading (simulates obstacle)
        if random.random() < 0.05:
            return random.uniform(5.0, 25.0)

        return max(2.0, min(self.config.max_distance_cm, base))

    # ─── Public API ───────────────────────────────────────────────────

    async def get_distance(self) -> float | None:
        """Take a single distance reading (for on-demand queries).

        Returns:
            Distance in cm (median of samples), or None on failure.
        """
        return await self._take_reading()