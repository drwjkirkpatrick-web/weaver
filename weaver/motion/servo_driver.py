# servo_driver.py — PCA9685 servo controller driver
#
# The PCA9685 is a 16-channel, 12-bit PWM controller connected via I2C.
# The Adeept Robot HAT V3.3 uses it to drive all 17 servos on the RaspClaws.
# (18 channels available, 17 used — 1 spare).
#
# Key concepts:
#   - PWM (Pulse Width Modulation): we control servos by sending them pulses
#     of a specific width. A standard hobby servo expects a pulse every 20ms
#     (50Hz), and the pulse width determines the angle:
#       500μs  = 0°    (12-bit value ≈ 150)
#       1500μs = 90°   (12-bit value ≈ 375, center)
#       2500μs = 180°  (12-bit value ≈ 600)
#   - The PCA9685 has a 12-bit resolution (0-4095) for the pulse width.
#     At 50Hz, one full PWM cycle = 20ms = 4096 ticks.
#     So 1 tick ≈ 4.88μs.
#   - We configure the "on" and "off" points within each cycle.
#     A pulse starts at "on" and ends at "off".
#
# Mock mode: When no hardware is present, all writes are logged but not sent.
# This allows full development and testing on any machine.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
from loguru import logger

from weaver.config import HardwareMode, ServoConfig, get_config
from weaver.event_bus import Event, EventBus, EventType, get_event_bus

# Try to import the real hardware library
try:
    import board  # type: ignore
    import busio  # type: ignore
    import adafruit_pca9685  # type: ignore
    _PCA9685_AVAILABLE = True
except ImportError:
    _PCA9685_AVAILABLE = False


class ServoDriver:
    """PCA9685 servo controller for the RaspClaws hexapod.
    
    This driver manages 18 PWM channels (6 legs × 3 joints each):
    - Coxa:  horizontal rotation (left-right swing)
    - Femur: vertical rotation (lift leg up/down)
    - Tibia: vertical rotation (extend/retract knee)
    
    All angles are in degrees (0-180, with 90 = center).
    Calibration offsets (per-servo) are applied before writing to hardware.
    
    Mock mode:
        When hardware is unavailable, all set_angle() calls are logged
        to debug output instead of being sent to the PCA9685. This allows
        full development on any machine without real servos.
    """
    
    def __init__(self, config: ServoConfig | None = None):
        """Initialize the servo driver.
        
        Args:
            config: Servo configuration. If None, loads from global config.
        """
        cfg = get_config()
        self.config: ServoConfig = config or cfg.servo
        self.hardware_mode: HardwareMode = cfg.hardware_mode
        self.bus = get_event_bus()
        
        # Hardware handle
        self._pca: Any = None
        self._i2c: Any = None
        
        # Determine if we should use mock mode
        self._mock: bool = self._should_use_mock()
        
        # Track current angles (for status queries and smooth transitions)
        self._current_angles: list[float] = [90.0] * 18  # All centered
        
        # Servo enable state
        self._enabled: bool = False
        
        logger.info(
            "🔧 Servo driver initialized (mode: {}, channels: 18)",
            "MOCK" if self._mock else "REAL",
        )
    
    def _should_use_mock(self) -> bool:
        """Check whether to use mock mode."""
        if self.hardware_mode == HardwareMode.MOCK:
            return True
        if not _PCA9685_AVAILABLE:
            logger.warning(
                "🔧 adafruit-circuitpython-pca9685 not installed — "
                "servo driver running in MOCK mode. "
                "Install with: pip install adafruit-circuitpython-pca9685"
            )
            return True
        return False
    
    async def start(self) -> None:
        """Initialize the PCA9685 and center all servos."""
        logger.info("🔧 Servo driver starting (mode: {})", "MOCK" if self._mock else "REAL")
        
        if not self._mock:
            await self._init_hardware()
        
        # Center all servos on startup (safe position)
        await self.center_all()
        self._enabled = True
        
        logger.info("✅ Servo driver started — all 18 servos centered")
    
    async def stop(self) -> None:
        """Stop the servo driver and disable all servos."""
        logger.info("🔧 Servo driver stopping...")
        
        await self.disable_all()
        
        if self._pca:
            try:
                self._pca.deinit()
            except Exception:
                pass
        self._pca = None
        self._i2c = None
        
        logger.info("✅ Servo driver stopped")
    
    async def _init_hardware(self) -> None:
        """Initialize the PCA9685 chip over I2C."""
        try:
            self._i2c = busio.I2C(board.SCL, board.SDA)
            self._pca = adafruit_pca9685.PCA9685(self._i2c, address=self.config.i2c_address)
            self._pca.frequency = self.config.pwm_frequency
            
            logger.info(
                "🔧 PCA9685 initialized at I2C 0x{:02X}, freq={:.0f}Hz",
                self.config.i2c_address, self.config.pwm_frequency,
            )
        except Exception as e:
            logger.error(f"Failed to initialize PCA9685: {e} — falling back to MOCK mode")
            self._mock = True
            self._pca = None
    
    # ─── Core Methods ─────────────────────────────────────────────────
    
    def _angle_to_pulse(self, angle: float) -> int:
        """Convert a servo angle (0-180°) to a 12-bit PWM value (150-600).
        
        The mapping is linear:
            0°   → servo_min_pulse (150)
            90°  → servo_center_pulse (375)
            180° → servo_max_pulse (600)
        
        Args:
            angle: Servo angle in degrees (0-180).
        
        Returns:
            12-bit PWM pulse value (150-600).
        """
        # Clamp angle to valid range
        angle = max(0.0, min(180.0, angle))
        
        # Linear interpolation: angle → pulse
        pulse_range = self.config.servo_max_pulse - self.config.servo_min_pulse
        pulse = self.config.servo_min_pulse + (angle / 180.0) * pulse_range
        return int(round(pulse))
    
    async def set_angle(self, channel: int, angle: float, apply_calibration: bool = True) -> None:
        """Set the angle of a single servo.
        
        Args:
            channel: PCA9685 channel (0-17).
            angle: Target angle in degrees (0-180, 90=center).
            apply_calibration: If True, apply per-servo calibration offset.
        """
        if not 0 <= channel < 18:
            logger.error(f"Invalid servo channel: {channel}")
            return
        
        # Apply calibration offset if enabled
        if apply_calibration and channel < len(self.config.calibration_offsets):
            angle += self.config.calibration_offsets[channel]
        
        # Clamp to safe range
        angle = max(0.0, min(180.0, angle))
        
        # Track current angle
        self._current_angles[channel] = angle
        
        if self._mock:
            logger.debug(f"Servo[{channel:2d}] → {angle:6.1f}° (mock)")
            return
        
        # Write to PCA9685
        pulse = self._angle_to_pulse(angle)
        try:
            self._pca.channels[channel].duty_cycle = pulse
            await asyncio.sleep(0)  # Yield to event loop
        except Exception as e:
            logger.error(f"Servo write error (ch {channel}): {e}")
    
    async def set_angles(self, angles: list[float], apply_calibration: bool = True) -> None:
        """Set angles for all 18 servos at once.
        
        Args:
            angles: List of 18 angles (0-180 degrees, 90=center).
            apply_calibration: If True, apply per-servo offsets.
        """
        if len(angles) != 18:
            logger.error(f"Expected 18 angles, got {len(angles)}")
            return
        
        # Write all angles (sequentially — I2C is a shared bus)
        for ch, angle in enumerate(angles):
            await self.set_angle(ch, angle, apply_calibration)
    
    async def center_all(self) -> None:
        """Center all 18 servos to 90° (neutral position).
        
        This is the safe startup and shutdown position. All legs should
        be in a natural standing pose when centered.
        """
        await self.set_angles([90.0] * 18, apply_calibration=False)
        logger.info("🔧 All servos centered to 90°")
    
    async def disable_all(self) -> None:
        """Disable all servo outputs (turn off PWM signals).
        
        This makes the servos go limp — they won't hold position.
        Used for emergency stop and shutdown.
        """
        self._enabled = False
        
        if self._mock:
            logger.info("🔧 All servos disabled (mock)")
            return
        
        try:
            for ch in range(18):
                self._pca.channels[ch].duty_cycle = 0
            logger.info("🔧 All servos disabled (PWM off)")
        except Exception as e:
            logger.error(f"Error disabling servos: {e}")
    
    # ─── Smooth Motion ────────────────────────────────────────────────
    
    async def smooth_move(
        self,
        target_angles: list[float],
        duration: float = 1.0,
        steps: int = 50,
    ) -> None:
        """Smoothly transition from current angles to target angles.
        
        Interpolates linearly between current and target positions over
        the specified duration. This prevents jerky motion that could
        damage servos or destabilize the robot.
        
        Args:
            target_angles: List of 18 target angles (0-180 degrees).
            duration: Transition duration in seconds.
            steps: Number of interpolation steps (more = smoother).
        """
        if len(target_angles) != 18:
            logger.error(f"Expected 18 angles, got {len(target_angles)}")
            return
        
        start_angles = list(self._current_angles)
        step_delay = duration / steps
        
        for i in range(1, steps + 1):
            # Linear interpolation: start + (target - start) * (i / steps)
            progress = i / steps
            # Ease in-out (smoother than linear — uses cosine)
            eased = 0.5 * (1 - np.cos(np.pi * progress))
            
            interpolated = [
                start + (target - start) * eased
                for start, target in zip(start_angles, target_angles)
            ]
            await self.set_angles(interpolated, apply_calibration=True)
            await asyncio.sleep(step_delay)
    
    # ─── Status ────────────────────────────────────────────────────────
    
    def get_current_angles(self) -> list[float]:
        """Get the current angle of all 18 servos."""
        return list(self._current_angles)
    
    def get_status(self) -> dict[str, Any]:
        """Get driver status for dashboard."""
        return {
            "enabled": self._enabled,
            "mock_mode": self._mock,
            "channels": 18,
            "current_angles": [round(a, 1) for a in self._current_angles],
            "i2c_address": f"0x{self.config.i2c_address:02X}",
        }