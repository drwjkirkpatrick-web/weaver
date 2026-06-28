# gait_engine.py — Hexapod gait controller
#
# The gait engine generates walking patterns for the 6-legged hexapod.
# It coordinates which legs move, when, and in what sequence.
#
# ─── Gait Types ────────────────────────────────────────────────────────
#
# 1. TRIPOD GAIT (default, fastest, most stable per step):
#    Legs are grouped into two tripods:
#      Group A: Front-Left, Mid-Right, Rear-Left  (legs 0, 4, 2)
#      Group B: Front-Right, Mid-Left, Rear-Right  (legs 5, 1, 3)
#    Group A lifts and swings forward while Group B stays on ground
#    and pushes the body forward. Then they swap.
#    → Always 3 feet on ground → very stable.
#
# 2. WAVE GAIT (slowest, most careful):
#    Only one leg moves at a time: L0 → L1 → L2 → L3 → L4 → L5 → repeat.
#    → 5 feet always on ground → extremely stable, but very slow.
#
# 3. RIPPLE GAIT (medium speed):
#    Two legs move at a time, offset so they're not adjacent.
#    → Balance between speed and stability.
#
# ─── Gait Cycle ────────────────────────────────────────────────────────
#
# Each gait has two phases per leg:
#   SWING phase:  leg lifts up, swings forward, places down
#   STANCE phase: foot is on ground, pushes body forward (or backward)
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import math
import time
from enum import Enum
from typing import Any

import numpy as np
from loguru import logger

from weaver.config import GaitConfig, get_config
from weaver.event_bus import Event, EventBus, EventType, get_event_bus
from weaver.motion.kinematics import HexapodKinematics, LegPosition
from weaver.motion.servo_driver import ServoDriver
from weaver.safety import SafetyGovernor


class GaitType(Enum):
    """Available gait patterns."""
    TRIPOD = "tripod"
    WAVE = "wave"
    RIPPLE = "ripple"


class GaitEngine:
    """Controls hexapod walking patterns.
    
    The gait engine uses inverse kinematics (from kinematics.py) to compute
    foot positions during walking, and the servo driver (servo_driver.py) to
    execute the motion. Every motion is checked by the safety governor
    before execution.
    
    Usage:
        gait = GaitEngine()
        await gait.start()
        await gait.walk("forward", speed=0.3)
        await gait.stop()
    """
    
    def __init__(
        self,
        config: GaitConfig | None = None,
        servo: ServoDriver | None = None,
        kinematics: HexapodKinematics | None = None,
    ):
        self.config = config or get_config().gait
        self.bus = get_event_bus()
        
        # Dependencies
        self.servo = servo or ServoDriver()
        self.kinematics = kinematics or HexapodKinematics()
        
        # Safety governor reference (set by main.py or injected)
        self._safety: SafetyGovernor | None = None
        
        # Current gait type
        self._current_gait: GaitType = GaitType.TRIPOD
        
        # Walking state
        self._walking = False
        self._direction = "stop"
        self._speed = 0.0
        self._walk_task: asyncio.Task | None = None
        
        # Gait phase tracking
        self._phase = 0.0  # 0.0 to 1.0 (one full gait cycle)
        
        logger.info("🦿 Gait engine initialized (gait: {})", self._current_gait.value)
    
    async def start(self) -> None:
        """Start the gait engine."""
        # Ensure servo driver is started
        if not self.servo._enabled:
            await self.servo.start()
        logger.info("✅ Gait engine started")
    
    async def stop(self) -> None:
        """Stop the gait engine."""
        await self.stop_walking()
        logger.info("Gait engine stopped")
    
    def set_safety_governor(self, safety: SafetyGovernor) -> None:
        """Inject the safety governor (called by main.py)."""
        self._safety = safety
    
    # ─── Gait Selection ───────────────────────────────────────────────
    
    def set_gait(self, gait: GaitType) -> None:
        """Change the active gait type."""
        self._current_gait = gait
        logger.info("🦿 Gait changed to: {}", gait.value)
        # Publish gait change event
        asyncio.create_task(self.bus.publish(Event(
            type=EventType.GAIT_CHANGE,
            data={"gait": gait.value},
            source="gait_engine",
        )))
    
    # ─── Walking ──────────────────────────────────────────────────────
    
    async def walk(self, direction: str = "forward", speed: float = 0.5) -> None:
        """Start walking in a direction at a given speed.
        
        Args:
            direction: "forward", "backward", "left", "right", "turn_left", "turn_right"
            speed: 0.0 to 1.0 (will be clamped by safety governor)
        """
        # Safety check
        if self._safety:
            allowed, adjusted_speed, reason = self._safety.check_motion(direction, speed)
            if not allowed:
                logger.warning(f"🛡️ Walk blocked by safety: {reason}")
                await self.bus.publish(Event(
                    type=EventType.MOTION_ABORTED,
                    data={"reason": reason, "direction": direction},
                    source="gait_engine",
                ))
                return
            speed = adjusted_speed
        
        self._direction = direction
        self._speed = speed
        self._walking = True
        
        # Publish motion started event
        await self.bus.publish(Event(
            type=EventType.MOTION_STARTED,
            data={"direction": direction, "speed": speed, "gait": self._current_gait.value},
            source="gait_engine",
        ))
        
        # Start the walk loop if not already running
        if self._walk_task is None or self._walk_task.done():
            self._walk_task = asyncio.create_task(self._walk_loop())
        
        logger.info(f"🦿 Walking {direction} at speed {speed:.2f}")
    
    async def stop_walking(self) -> None:
        """Stop walking and return to standing pose."""
        self._walking = False
        self._direction = "stop"
        self._speed = 0.0
        
        if self._walk_task:
            self._walk_task.cancel()
            try:
                await self._walk_task
            except asyncio.CancelledError:
                pass
            self._walk_task = None
        
        # Return to standing stance
        await self.stand_up()
        
        await self.bus.publish(Event(
            type=EventType.MOTION_COMPLETED,
            data={"direction": "stop"},
            source="gait_engine",
        ))
        logger.info("🦿 Stopped walking")
    
    async def _walk_loop(self) -> None:
        """Main walking animation loop."""
        step_duration = self.config.step_duration / max(0.1, self._speed + 0.3)
        
        while self._walking:
            try:
                # Compute foot positions for this gait cycle
                feet = self._compute_gait_feet(self._phase, self._direction, self._speed)
                
                # Solve IK for all legs
                angles = self.kinematics.solve_body_ik(feet)
                
                # Safety check on servo angles
                if self._safety:
                    allowed, _, reason = self._safety.check_motion(
                        self._direction, self._speed, servo_angles=angles
                    )
                    if not allowed:
                        logger.warning(f"🛡️ Motion blocked during walk: {reason}")
                        await self.stop_walking()
                        return
                
                # Write to servos
                await self.servo.set_angles(angles)
                
                # Advance phase
                self._phase += 1.0 / (step_duration * 60.0)  # Approximate steps per second
                if self._phase >= 1.0:
                    self._phase -= 1.0
                
                # Small delay for smooth motion (~60fps target)
                await asyncio.sleep(1.0 / 60.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Walk loop error: {e}")
                await asyncio.sleep(0.1)
    
    def _compute_gait_feet(
        self, phase: float, direction: str, speed: float
    ) -> list[LegPosition]:
        """Compute foot positions for all 6 legs at a given gait phase.
        
        Args:
            phase: Gait phase (0.0 to 1.0).
            direction: Walking direction.
            speed: Walking speed (0.0 to 1.0).
        
        Returns:
            List of 6 LegPosition objects.
        """
        feet = []
        step_len = self.config.step_length * speed
        step_h = self.config.step_height
        body_h = self.config.body_height
        
        for i in range(6):
            # Get phase offset for this leg based on gait type
            leg_phase = self._get_leg_phase(i, phase)
            
            if leg_phase < 0.5:
                # SWING phase (0.0 to 0.5): leg lifts and moves forward
                swing_t = leg_phase * 2.0  # Normalize to 0-1
                
                # X position: move from back to front
                x = step_len * (2 * swing_t - 1) * 0.5
                
                # Z position: lift up during swing (parabolic arc)
                z = -body_h + step_h * 4 * swing_t * (1 - swing_t)
            else:
                # STANCE phase (0.5 to 1.0): foot on ground, body moves forward
                stance_t = (leg_phase - 0.5) * 2.0  # Normalize to 0-1
                
                # X position: foot stays planted, body moves (relative: foot goes backward)
                x = step_len * (1 - 2 * stance_t) * 0.5
                
                # Z position: on the ground
                z = -body_h
            
            # Default Y position (per leg, from kinematics defaults)
            y = self.kinematics.DEFAULT_STANCE[i].y
            
            # Modify for direction
            if direction == "turn_left":
                # Rotate foot positions for turning
                angle = step_len * 0.01 * (1 if i < 3 else -1)
                x_rot = x * math.cos(angle) - y * math.sin(angle)
                y_rot = x * math.sin(angle) + y * math.cos(angle)
                x, y = x_rot, y_rot
            elif direction == "turn_right":
                angle = -step_len * 0.01 * (1 if i >= 3 else -1)
                x_rot = x * math.cos(angle) - y * math.sin(angle)
                y_rot = x * math.sin(angle) + y * math.cos(angle)
                x, y = x_rot, y_rot
            elif direction == "left":
                y += step_len * 0.3 * (1 if leg_phase < 0.5 else -1)
            elif direction == "right":
                y -= step_len * 0.3 * (1 if leg_phase < 0.5 else -1)
            elif direction == "backward":
                x = -x  # Reverse forward motion
            
            feet.append(LegPosition(x=x, y=y, z=z))
        
        return feet
    
    def _get_leg_phase(self, leg_index: int, phase: float) -> float:
        """Get the phase offset for a given leg based on gait type.
        
        Args:
            leg_index: 0-5 (FL, ML, RL, RR, MR, FR).
            phase: Global gait phase (0.0 to 1.0).
        
        Returns:
            This leg's phase (0.0 to 1.0).
        """
        if self._current_gait == GaitType.TRIPOD:
            # Two groups of 3 legs, 180° out of phase
            # Group A: legs 0, 4, 2 → phase offset 0.0
            # Group B: legs 5, 1, 3 → phase offset 0.5
            if leg_index in (0, 2, 4):
                return phase
            else:
                return (phase + 0.5) % 1.0
        
        elif self._current_gait == GaitType.WAVE:
            # Each leg offset by 1/6 of a cycle
            return (phase + leg_index / 6.0) % 1.0
        
        elif self._current_gait == GaitType.RIPPLE:
            # Pairs of legs, offset by 1/3
            offsets = [0.0, 0.33, 0.66, 0.0, 0.33, 0.66]
            return (phase + offsets[leg_index]) % 1.0
        
        return phase
    
    # ─── Static Poses ─────────────────────────────────────────────────
    
    async def stand_up(self) -> None:
        """Move to a standing pose."""
        angles = self.kinematics.standing_stance(height=-self.config.body_height)
        
        if self._safety:
            allowed, _, reason = self._safety.check_motion("stop", 0.0, servo_angles=angles)
            if not allowed:
                logger.warning(f"Stand blocked by safety: {reason}")
                return
        
        await self.servo.smooth_move(angles, duration=1.0, steps=50)
        await self.bus.publish(Event(
            type=EventType.MOTION_COMPLETED,
            data={"action": "stand"},
            source="gait_engine",
        ))
        logger.info("🦿 Standing pose")
    
    async def sit_down(self) -> None:
        """Lower body to resting/crouching position."""
        angles = self.kinematics.crouch_stance()
        
        if self._safety:
            allowed, _, reason = self._safety.check_motion("stop", 0.0, servo_angles=angles)
            if not allowed:
                logger.warning(f"Sit blocked by safety: {reason}")
                return
        
        await self.servo.smooth_move(angles, duration=1.5, steps=50)
        await self.bus.publish(Event(
            type=EventType.MOTION_COMPLETED,
            data={"action": "sit"},
            source="gait_engine",
        ))
        logger.info("🦿 Sitting down")
    
    async def crouch(self) -> None:
        """Crouch low to the ground."""
        await self.sit_down()
    
    async def stand_tall(self) -> None:
        """Raise body to maximum height."""
        angles = self.kinematics.tall_stance()
        
        if self._safety:
            allowed, _, reason = self._safety.check_motion("stop", 0.0, servo_angles=angles)
            if not allowed:
                logger.warning(f"Tall blocked by safety: {reason}")
                return
        
        await self.servo.smooth_move(angles, duration=1.0, steps=50)
        await self.bus.publish(Event(
            type=EventType.MOTION_COMPLETED,
            data={"action": "stand_tall"},
            source="gait_engine",
        ))
        logger.info("🦿 Standing tall")
    
    async def wave_hello(self) -> None:
        """Wave one leg in a greeting gesture.
        
        Lifts the front-right leg and waves it back and forth.
        Fun for demos and kid interaction!
        """
        logger.info("🦿 👋 Waving hello!")
        
        await self.bus.publish(Event(
            type=EventType.MOTION_STARTED,
            data={"action": "wave"},
            source="gait_engine",
        ))
        
        # Get current angles
        current = self.servo.get_current_angles()
        
        # Lift front-right leg (leg 5: channels 15, 16, 17)
        # Raise femur and tibia
        wave_angles = list(current)
        wave_angles[15] = 120  # Coxa: rotate outward
        wave_angles[16] = 60   # Femur: lift up
        wave_angles[17] = 40   # Tibia: bend knee
        
        await self.servo.smooth_move(wave_angles, duration=0.8, steps=30)
        
        # Wave back and forth 3 times
        for i in range(3):
            wave_angles[15] = 140 if i % 2 == 0 else 100
            await self.servo.smooth_move(wave_angles, duration=0.4, steps=20)
        
        # Return to standing
        await self.stand_up()
        
        await self.bus.publish(Event(
            type=EventType.MOTION_COMPLETED,
            data={"action": "wave"},
            source="gait_engine",
        ))
    
    # ─── Status ────────────────────────────────────────────────────────
    
    def get_status(self) -> dict[str, Any]:
        """Get gait engine status."""
        return {
            "walking": self._walking,
            "direction": self._direction,
            "speed": round(self._speed, 2),
            "gait": self._current_gait.value,
            "phase": round(self._phase, 3),
        }