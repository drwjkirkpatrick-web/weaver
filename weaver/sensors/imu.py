# imu.py — MPU6050 gyroscope/accelerometer for Weaver
#
# The MPU6050 is a 6-axis IMU (3-axis accelerometer + 3-axis gyroscope)
# connected via I2C. We use it to:
#   1. Measure body orientation (roll, pitch) — is the hexapod level?
#   2. Detect excessive tilt — trigger stabilization if it's falling over.
#   3. Measure acceleration — detect sudden movements or impacts.
#
# Hardware: MPU6050 breakout board, I2C address 0x68, Pi 5 I2C bus 1.
# Library: adafruit-circuitpython-mpu6050 (uses Blinka for I2C).
#
# Learning notes:
#   - An accelerometer measures *proper acceleration* (includes gravity).
#     When the sensor is flat, it reads ~9.81 m/s² on the Z axis. We use
#     the direction of gravity to compute tilt angles — this is accurate
#     when stationary but drifts during motion.
#   - A gyroscope measures *angular velocity* (rotation rate). Integrating
#     angular velocity over time gives angle, but this drifts due to noise
#     and bias. Over long periods, the integrated angle diverges.
#   - **Complementary filter**: combines both — use the accelerometer for
#     long-term accuracy (gravity direction) and the gyroscope for short-term
#     responsiveness. The formula:
#         angle = α * (angle + gyro * dt) + (1 - α) * accel_angle
#     where α (alpha) is typically 0.98 (trust gyro 98%, accel 2%).
#   - The accelerometer is noisy in the short term (vibration from servos!),
#     so we filter it heavily. The gyro is smooth but drifts over minutes.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

from loguru import logger

from weaver.config import HardwareMode, IMUConfig, get_config
from weaver.event_bus import Event, EventType, get_event_bus

# ─── Optional hardware import ─────────────────────────────────────────────

try:
    import board  # type: ignore[import-untyped]  # Blinka: maps Pi pins
    import busio  # type: ignore[import-untyped]  # Blinka: I2C/SPI/UART
    import adafruit_mpu6050  # type: ignore[import-untyped]
    _MPU6050_AVAILABLE = True
except ImportError:
    _MPU6050_AVAILABLE = False


class IMUSensor:
    """MPU6050 inertial measurement unit with complementary filtering.

    This sensor reads raw acceleration and gyroscope data from the MPU6050,
    applies a complementary filter to compute stable roll and pitch angles,
    and publishes the results to the event bus.

    The complementary filter is the key algorithm here. It fuses two
    noisy sensors into one good estimate:

    - **Accelerometer** gives accurate absolute orientation (relative to
      gravity) when stationary, but is very noisy during motion (servo
      vibration, impacts). Good for long-term reference.
    - **Gyroscope** gives smooth, responsive angular velocity but
      accumulates error over time (drift). Good for short-term tracking.

    The filter: ``angle = α × (angle + gyro×dt) + (1-α) × accel_angle``
    where α ≈ 0.98 means "trust the gyro for fast changes, correct slowly
    with the accelerometer."

    Events published:
        - IMU_DATA:          every sample with acceleration, gyro, orientation
        - IMU_TILT_WARNING:  when tilt exceeds the safety threshold

    Mock mode:
        Generates sinusoidal acceleration/gyro data simulating gentle sway,
        with occasional tilt spikes to test warning behavior.
    """

    # Tilt threshold (degrees) for warning — read from safety config.
    # Default 30° matches SafetyConfig.max_tilt_degrees.
    _TILT_WARNING_THRESHOLD: float = 30.0

    def __init__(self, config: IMUConfig | None = None) -> None:
        """Initialize the IMU sensor.

        Args:
            config: IMU configuration. If None, loads from global config.
        """
        cfg = get_config()
        self.config: IMUConfig = config or cfg.imu
        self.hardware_mode: HardwareMode = cfg.hardware_mode

        # Pull tilt threshold from safety config (cross-module dependency)
        self._tilt_threshold: float = cfg.safety.max_tilt_degrees

        self.bus = get_event_bus()

        # Hardware handle (real mode)
        self._mpu: Any = None
        self._i2c: Any = None

        # Complementary filter state
        self._roll: float = 0.0   # Rotation around X axis (left-right tilt)
        self._pitch: float = 0.0  # Rotation around Y axis (forward-back tilt)
        self._last_update: float = 0.0

        # Running task
        self._task: asyncio.Task | None = None

        # Mock mode state
        self._mock: bool = self._should_use_mock()
        self._mock_time: float = 0.0

    def _should_use_mock(self) -> bool:
        """Determine whether to use mock mode."""
        if self.hardware_mode == HardwareMode.MOCK:
            return True
        if not _MPU6050_AVAILABLE:
            logger.warning(
                "📐 adafruit-circuitpython-mpu6050 not installed — "
                "IMU running in MOCK mode. "
                "Install with: pip install adafruit-circuitpython-mpu6050"
            )
            return True
        return False

    async def start(self) -> None:
        """Start the IMU polling loop."""
        logger.info("📐 IMU sensor starting (mode: {})", "MOCK" if self._mock else "REAL")

        if not self._mock:
            await self._init_real_hardware()

        self._last_update = time.time()
        self._task = asyncio.create_task(self._poll_loop())

        logger.info("✅ IMU sensor started (sample rate: {}Hz)", self.config.sample_rate_hz)

    async def stop(self) -> None:
        """Stop the IMU and release hardware."""
        logger.info("📐 IMU sensor stopping...")

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # adafruit_mpu6050 doesn't need explicit cleanup, but we null it
        self._mpu = None
        self._i2c = None

        logger.info("✅ IMU sensor stopped")

    # ─── Initialization ───────────────────────────────────────────────

    async def _init_real_hardware(self) -> None:
        """Initialize the real MPU6050 over I2C."""
        try:
            # Blinka's board/busio modules provide a cross-platform I2C
            # interface. On the Pi, this maps to /dev/i2c-1.
            self._i2c = busio.I2C(board.SCL, board.SDA)
            self._mpu = adafruit_mpu6050.MPU6050(
                self._i2c,
                address=self.config.i2c_address,
            )

            logger.info(
                f"📐 MPU6050 initialized at I2C 0x{self.config.i2c_address:02X} "
                f"(bus {self.config.i2c_bus})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize MPU6050: {e} — falling back to MOCK mode")
            self._mock = True
            self._mpu = None

    # ─── Polling Loop ─────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Main polling loop that reads sensor data and publishes events."""
        interval = 1.0 / self.config.sample_rate_hz

        while True:
            try:
                now = time.time()
                dt = now - self._last_update
                self._last_update = now

                # Read raw sensor data
                accel, gyro = await self._read_sensor()

                # Apply calibration offsets (subtract bias)
                accel = (
                    accel[0] - self.config.accel_offset[0],
                    accel[1] - self.config.accel_offset[1],
                    accel[2] - self.config.accel_offset[2],
                )
                gyro = (
                    gyro[0] - self.config.gyro_offset[0],
                    gyro[1] - self.config.gyro_offset[1],
                    gyro[2] - self.config.gyro_offset[2],
                )

                # Compute orientation using complementary filter (returns degrees)
                roll, pitch = self._complementary_filter(accel, gyro, dt)

                # Compute total tilt magnitude (how far from level)
                # roll and pitch are already in degrees
                tilt_degrees = math.sqrt(roll ** 2 + pitch ** 2)

                # Publish IMU_DATA event
                await self.bus.publish(Event(
                    type=EventType.IMU_DATA,
                    data={
                        "acceleration": {
                            "x": round(accel[0], 3),
                            "y": round(accel[1], 3),
                            "z": round(accel[2], 3),
                        },
                        "gyroscope": {
                            "x": round(gyro[0], 3),
                            "y": round(gyro[1], 3),
                            "z": round(gyro[2], 3),
                        },
                        "orientation": {
                            "roll": round(roll, 2),
                            "pitch": round(pitch, 2),
                        },
                        "tilt_degrees": round(tilt_degrees, 2),
                        "timestamp": now,
                    },
                    source="imu",
                ))

                # Check tilt threshold — publish warning if exceeded
                if tilt_degrees > self._tilt_threshold:
                    await self.bus.publish(Event(
                        type=EventType.IMU_TILT_WARNING,
                        data={
                            "tilt_degrees": round(tilt_degrees, 2),
                            "threshold": self._tilt_threshold,
                            "roll": round(roll, 2),
                            "pitch": round(pitch, 2),
                        },
                        source="imu",
                    ))
                    logger.warning(
                        f"⚠️  Tilt warning: {tilt_degrees:.1f}° "
                        f"(threshold: {self._tilt_threshold}°)"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"IMU poll loop error: {e}")

            await asyncio.sleep(interval)

    async def _read_sensor(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Read acceleration and gyroscope data from the MPU6050.

        Returns:
            Tuple of (acceleration_xyz, gyroscope_xyz).
            Acceleration is in m/s². Gyroscope is in rad/s.

        In mock mode, generates sinusoidal data simulating gentle body sway
        with occasional tilt events.
        """
        if self._mock:
            return self._generate_mock_data()

        try:
            accel = self._mpu.acceleration   # (x, y, z) in m/s²
            gyro = self._mpu.gyro            # (x, y, z) in rad/s
            await asyncio.sleep(0)  # Yield to event loop (I2C read is blocking)
            return accel, gyro
        except Exception as e:
            logger.error(f"IMU read error: {e}")
            return (0.0, 0.0, 9.81), (0.0, 0.0, 0.0)

    def _generate_mock_data(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Generate simulated IMU data for mock mode.

        Simulates a hexapod gently swaying while walking, with occasional
        tilt spikes (as if stepping over an obstacle). Gravity (~9.81 m/s²)
        is on the Z axis when level.
        """
        t = self._mock_time
        self._mock_time += 1.0 / self.config.sample_rate_hz

        # Gentle sinusoidal sway (simulates walking gait oscillation)
        sway = 0.5  # m/s² amplitude
        accel_x = sway * math.sin(2 * math.pi * 1.5 * t)  # 1.5 Hz sway
        accel_y = sway * math.sin(2 * math.pi * 1.0 * t)  # 1.0 Hz sway
        accel_z = 9.81 + 0.3 * math.sin(2 * math.pi * 2.0 * t)  # Gravity + tiny bob

        # Occasional tilt spike every 10 seconds (simulate obstacle)
        tilt_spike = 0.0
        if int(t) % 10 == 0 and t % 1.0 < (1.0 / self.config.sample_rate_hz):
            tilt_spike = 3.0  # Sudden acceleration spike

        accel_x += tilt_spike

        # Gyroscope: small angular velocities matching the sway
        gyro_x = 0.05 * math.sin(2 * math.pi * 1.5 * t)  # rad/s
        gyro_y = 0.05 * math.cos(2 * math.pi * 1.0 * t)
        gyro_z = 0.02 * math.sin(2 * math.pi * 0.5 * t)

        return (accel_x, accel_y, accel_z), (gyro_x, gyro_y, gyro_z)

    # ─── Complementary Filter ─────────────────────────────────────────

    def _complementary_filter(
        self,
        accel: tuple[float, float, float],
        gyro: tuple[float, float, float],
        dt: float,
    ) -> tuple[float, float]:
        """Compute roll and pitch using a complementary filter.

        The complementary filter fuses accelerometer and gyroscope data:

            angle = α × (angle + gyro × dt) + (1 - α) × accel_angle

        where α (alpha) is the filter coefficient. A high α (0.98) trusts
        the gyroscope (smooth, responsive) and slowly corrects with the
        accelerometer (noisy but absolute reference).

        Args:
            accel: Acceleration (x, y, z) in m/s².
            gyro: Angular velocity (x, y, z) in rad/s.
            dt: Time delta since last update (seconds).

        Returns:
            Tuple of (roll, pitch) in **degrees**.
        """
        # ── Step 1: Compute angle from accelerometer ──────────────────
        ax, ay, az = accel

        # Guard against division by zero (freefall: all zeros)
        accel_roll = math.atan2(ay, az) if az != 0 else 0.0
        accel_pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))

        # Convert to degrees for consistent units
        accel_roll_deg = math.degrees(accel_roll)
        accel_pitch_deg = math.degrees(accel_pitch)

        # ── Step 2: Integrate gyroscope (in degrees) ──────────────────
        gx, gy, gz = gyro
        gyro_roll = self._roll + math.degrees(gx) * dt
        gyro_pitch = self._pitch + math.degrees(gy) * dt

        # ── Step 3: Fuse with complementary filter ────────────────────
        alpha = self.config.complementary_alpha
        self._roll = alpha * gyro_roll + (1 - alpha) * accel_roll_deg
        self._pitch = alpha * gyro_pitch + (1 - alpha) * accel_pitch_deg

        return self._roll, self._pitch

    # ─── Public API ───────────────────────────────────────────────────

    def get_orientation(self) -> dict[str, float]:
        """Get the current orientation (for dashboard/queries).

        Returns:
            Dict with roll, pitch (in degrees), and tilt_degrees.
        """
        # _roll and _pitch are already in degrees after _complementary_filter
        roll_deg = self._roll
        pitch_deg = self._pitch
        tilt = math.sqrt(roll_deg ** 2 + pitch_deg ** 2)
        return {
            "roll": round(roll_deg, 2),
            "pitch": round(pitch_deg, 2),
            "tilt_degrees": round(tilt, 2),
        }