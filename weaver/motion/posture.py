# posture.py — Body posture and balance controller
#
# The posture controller maintains the hexapod's body height and level.
# It uses IMU data to auto-level the body (compensate for terrain) and
# supports smooth transitions between different body poses.
#
# Key concepts:
#   - BODY HEIGHT: distance from body center to ground (mm).
#     Higher = robot stands taller, lower = crouches.
#   - AUTO-LEVEL: reads IMU roll/pitch, applies inverse rotation to
#     foot positions so the body stays level on uneven terrain.
#   - LEAN: deliberately tilting the body for looking or reaching.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import math
from typing import Any

from loguru import logger

from weaver.config import GaitConfig, get_config
from weaver.event_bus import Event, EventBus, EventType, get_event_bus
from weaver.motion.kinematics import HexapodKinematics, LegPosition
from weaver.motion.servo_driver import ServoDriver


class PostureController:
    """Manages body posture, height, and auto-leveling.
    
    The posture controller sits between the gait engine and the servo driver.
    During walking, the gait engine sets foot positions. Between steps
    (or when standing), the posture controller fine-tunes the body pose
    using IMU feedback.
    
    Auto-leveling loop:
    1. Read IMU roll/pitch (subscribed to IMU_DATA events)
    2. If tilt exceeds threshold, compute corrective body rotation
    3. Apply inverse rotation to foot positions via IK
    4. Write corrected angles to servos
    
    This runs at ~20Hz (every 50ms) for smooth correction without
    fighting the gait engine.
    """
    
    def __init__(
        self,
        config: GaitConfig | None = None,
        servo: ServoDriver | None = None,
        kinematics: HexapodKinematics | None = None,
    ):
        self.config = config or get_config().gait
        self.bus = get_event_bus()
        self.servo = servo or ServoDriver()
        self.kinematics = kinematics or HexapodKinematics()
        
        # Current posture state
        self._body_height: float = self.config.body_height
        self._body_roll: float = 0.0   # degrees
        self._body_pitch: float = 0.0  # degrees
        self._body_yaw: float = 0.0    # degrees
        
        # Auto-level state
        self._auto_level_enabled: bool = True
        self._current_roll: float = 0.0   # From IMU
        self._current_pitch: float = 0.0  # From IMU
        self._level_task: asyncio.Task | None = None
        
        # Smoothing: target vs current (for transitions)
        self._target_height: float = self._body_height
        self._target_roll: float = 0.0
        self._target_pitch: float = 0.0
        
        logger.info("🦿 Posture controller initialized (height: {:.0f}mm)", self._body_height)
    
    async def start(self) -> None:
        """Start the posture controller."""
        if not self.servo._enabled:
            await self.servo.start()
        
        # Subscribe to IMU data for auto-leveling
        self.bus.subscribe(EventType.IMU_DATA, self._on_imu_data)
        
        # Start auto-level loop
        self._level_task = asyncio.create_task(self._auto_level_loop())
        
        logger.info("✅ Posture controller started (auto-level: {})", self._auto_level_enabled)
    
    async def stop(self) -> None:
        """Stop the posture controller."""
        if self._level_task:
            self._level_task.cancel()
            try:
                await self._level_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Posture controller stopped")
    
    # ─── IMU Handler ──────────────────────────────────────────────────
    
    async def _on_imu_data(self, event: Event) -> None:
        """Update current body orientation from IMU."""
        orientation = event.data.get("orientation", {})
        self._current_roll = orientation.get("roll", 0.0)
        self._current_pitch = orientation.get("pitch", 0.0)
    
    # ─── Auto-Level Loop ──────────────────────────────────────────────
    
    async def _auto_level_loop(self) -> None:
        """Continuously correct body posture based on IMU feedback.
        
        Runs at ~20Hz. If auto-level is enabled and the body is tilted
        beyond a small deadzone, it applies a corrective rotation to
        bring the body back to level.
        """
        deadzone = 2.0  # degrees — ignore tiny tilts (reduces jitter)
        max_correction = 15.0  # degrees — cap how much we correct per step
        correction_rate = 0.3  # How fast to correct (0-1, lower = smoother)
        
        while True:
            try:
                if self._auto_level_enabled:
                    # Compute correction needed
                    roll_error = self._current_roll
                    pitch_error = self._current_pitch
                    
                    # Apply deadzone
                    if abs(roll_error) < deadzone:
                        roll_error = 0.0
                    if abs(pitch_error) < deadzone:
                        pitch_error = 0.0
                    
                    # Clamp correction
                    roll_correction = max(-max_correction, min(max_correction, -roll_error * correction_rate))
                    pitch_correction = max(-max_correction, min(max_correction, -pitch_error * correction_rate))
                    
                    # Smoothly update target
                    self._target_roll = self._target_roll * 0.9 + roll_correction * 0.1
                    self._target_pitch = self._target_pitch * 0.9 + pitch_correction * 0.1
                    
                    # Apply posture (only if not being overridden by gait engine)
                    # Note: In a real system, we'd check if the gait engine is active
                    # and skip posture updates during walking. For simplicity, we
                    # apply small corrections that won't interfere much.
                    if abs(self._target_roll) > 0.5 or abs(self._target_pitch) > 0.5:
                        await self._apply_posture()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-level error: {e}")
            
            await asyncio.sleep(0.05)  # 20Hz
    
    async def _apply_posture(self) -> None:
        """Compute and apply current posture to servos."""
        # Smoothly interpolate height
        self._body_height = self._body_height * 0.95 + self._target_height * 0.05
        
        # Compute IK with body rotation
        angles = self.kinematics.solve_body_ik(
            body_roll=self._target_roll,
            body_pitch=self._target_pitch,
            body_yaw=self._body_yaw,
            body_z=self._body_height - self.config.body_height,  # Height offset
        )
        
        await self.servo.set_angles(angles)
    
    # ─── Posture Commands ────────────────────────────────────────────
    
    async def set_height(self, height_mm: float) -> None:
        """Set body height smoothly.
        
        Args:
            height_mm: Target height (40-120mm).
        """
        height_mm = max(self.config.body_height_min, min(self.config.body_height_max, height_mm))
        self._target_height = height_mm
        
        # Smooth transition
        angles = self.kinematics.standing_stance(height=-height_mm)
        await self.servo.smooth_move(angles, duration=1.0, steps=50)
        
        self._body_height = height_mm
        logger.info(f"🦿 Body height set to {height_mm:.0f}mm")
    
    async def crouch(self) -> None:
        """Crouch low to the ground."""
        await self.set_height(self.config.body_height_min)
    
    async def stand_tall(self) -> None:
        """Stand at maximum height."""
        await self.set_height(self.config.body_height_max)
    
    async def lean(self, direction: str, amount: float = 10.0) -> None:
        """Lean the body in a direction.
        
        Args:
            direction: "forward", "backward", "left", "right"
            amount: Lean angle in degrees (5-20 recommended)
        """
        amount = max(0.0, min(20.0, amount))
        
        self._target_roll = 0.0
        self._target_pitch = 0.0
        
        if direction == "forward":
            self._target_pitch = amount
        elif direction == "backward":
            self._target_pitch = -amount
        elif direction == "left":
            self._target_roll = amount
        elif direction == "right":
            self._target_roll = -amount
        
        # Disable auto-level during deliberate lean
        self._auto_level_enabled = False
        
        angles = self.kinematics.lean_stance(direction, amount)
        await self.servo.smooth_move(angles, duration=0.8, steps=40)
        
        logger.info(f"🦿 Leaning {direction} by {amount:.0f}°")
        
        # Re-enable auto-level after a delay
        await asyncio.sleep(2.0)
        self._auto_level_enabled = True
        self._target_roll = 0.0
        self._target_pitch = 0.0
    
    async def level(self) -> None:
        """Return to level standing posture."""
        self._target_roll = 0.0
        self._target_pitch = 0.0
        self._auto_level_enabled = True
        
        angles = self.kinematics.standing_stance(height=-self._body_height)
        await self.servo.smooth_move(angles, duration=0.8, steps=40)
        
        logger.info("🦿 Body leveled")
    
    # ─── Auto-Level Toggle ───────────────────────────────────────────
    
    def enable_auto_level(self) -> None:
        """Enable automatic body leveling."""
        self._auto_level_enabled = True
        logger.info("🦿 Auto-level enabled")
    
    def disable_auto_level(self) -> None:
        """Disable automatic body leveling."""
        self._auto_level_enabled = False
        logger.info("🦿 Auto-level disabled")
    
    # ─── Status ────────────────────────────────────────────────────────
    
    def get_status(self) -> dict[str, Any]:
        """Get posture status."""
        return {
            "body_height": round(self._body_height, 1),
            "target_height": round(self._target_height, 1),
            "roll": round(self._current_roll, 2),
            "pitch": round(self._current_pitch, 2),
            "target_roll": round(self._target_roll, 2),
            "target_pitch": round(self._target_pitch, 2),
            "auto_level": self._auto_level_enabled,
        }