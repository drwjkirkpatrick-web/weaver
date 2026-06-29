# dance_engine.py — Hexapod dance engine
#
# The dance engine coordinates beat-synced movements for the hexapod.
# It takes a dance routine (list of moves) and executes them in time
# with the beat detector, all under the watch of the dance safety governor.
#
# Architecture:
#
#   BeatDetector ──beat events──→ DanceEngine ──moves──→ DanceSafetyGov
#                                                      │
#                                                      ▼ safe angles
#                                                  ServoDriver
#
# A "dance" is a sequence of "dance moves". Each move is:
#   - A set of 18 servo angles (a pose)
#   - A beat count (how many beats to hold this pose)
#   - A transition style (smooth, snap, bounce)
#
# The engine plays moves in sequence, syncing to the beat detector.
# Students can create their own dances by defining a list of moves.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
from loguru import logger

from weaver.config import GaitConfig, get_config
from weaver.dance.dance_safety import DanceSafetyGovernor, DanceSafetyLevel
from weaver.event_bus import Event, EventBus, EventType, get_event_bus
from weaver.motion.servo_driver import ServoDriver


# ─── Move Types ────────────────────────────────────────────────────────


class TransitionType(str, Enum):
    """How to transition between poses."""
    SMOOTH = "smooth"    # Interpolate smoothly (cosine ease)
    SNAP = "snap"        # Jump immediately to new pose
    BOUNCE = "bounce"    # Go past target then settle (springy)
    WAVE = "wave"        # Sinusoidal oscillation


@dataclass
class DanceMove:
    """A single dance pose held for a number of beats.
    
    Attributes:
        name: Human-readable name (e.g., "arms up", "lean left")
        angles: 18 servo angles (0-180, 90=center)
        beats: How many beats to hold this pose
        transition: How to transition into this pose
        feet_on_ground: How many feet stay on ground (for safety check)
        height_change: Body height change from previous (mm, for safety)
    """
    name: str
    angles: list[float]
    beats: int = 1
    transition: TransitionType = TransitionType.SMOOTH
    feet_on_ground: int = 6
    height_change: float = 0.0


@dataclass
class DanceRoutine:
    """A complete dance routine (sequence of moves).
    
    Attributes:
        name: Dance name (e.g., "Macarena")
        moves: Ordered list of DanceMove objects
        bpm: Recommended BPM for this dance
        description: Human-readable description
        creator: Who made this dance (student name, etc.)
    """
    name: str
    moves: list[DanceMove]
    bpm: float = 120.0
    description: str = ""
    creator: str = "Weaver"
    
    @property
    def total_beats(self) -> int:
        """Total number of beats in the routine."""
        return sum(m.beats for m in self.moves)
    
    @property
    def estimated_duration(self) -> float:
        """Estimated duration in seconds based on BPM."""
        return self.total_beats * (60.0 / self.bpm)


# ─── Dance Engine ──────────────────────────────────────────────────────


class DanceEngine:
    """Executes dance routines synced to beats.
    
    The engine:
    1. Takes a DanceRoutine (list of moves)
    2. Waits for beat events from the beat detector
    3. On each beat, transitions to the next pose
    4. All moves pass through the dance safety governor
    5. Publishes DANCE_STARTED, DANCE_MOVE, DANCE_COMPLETED events
    
    Usage:
        engine = DanceEngine(servo_driver, safety_governor)
        await engine.start()
        
        routine = DanceRoutine(name="My Dance", moves=[...], bpm=120)
        await engine.dance(routine)
        
        await engine.stop()
    
    Student API:
        Students create DanceMove objects with servo angles and beat counts.
        They don't need to worry about safety — the governor handles that.
        See weaver/dance/student_api.py for a simpler interface.
    """
    
    def __init__(
        self,
        servo: ServoDriver | None = None,
        safety_governor: DanceSafetyGovernor | None = None,
        config: GaitConfig | None = None,
    ):
        self.servo = servo or ServoDriver()
        self.safety = safety_governor or DanceSafetyGovernor(DanceSafetyLevel.STRICT)
        self.config = config or get_config().gait
        self.bus = get_event_bus()
        
        self._running: bool = False
        self._dancing: bool = False
        self._current_routine: DanceRoutine | None = None
        self._move_index: int = 0
        self._beat_count: int = 0
        self._last_angles: list[float] = [90.0] * 18  # Track for transitions
        
        logger.info("🕺 Dance engine initialized")
    
    async def start(self) -> None:
        """Start the dance engine."""
        if not self.servo._enabled:
            await self.servo.start()
        self._running = True
        logger.info("✅ Dance engine started")
    
    async def stop(self) -> None:
        """Stop the dance engine."""
        self._running = False
        if self._dancing:
            await self.stop_dance()
        logger.info("Dance engine stopped")
    
    # ─── Dancing ────────────────────────────────────────────────────────
    
    async def dance(self, routine: DanceRoutine) -> None:
        """Execute a dance routine.
        
        Args:
            routine: The DanceRoutine to perform.
        """
        if not self._running:
            logger.warning("Dance engine not started — call start() first")
            return
        
        # Start safety session
        if not self.safety.start_session():
            logger.warning("🛡️ Dance blocked by safety governor — session not allowed")
            return
        
        self._dancing = True
        self._current_routine = routine
        self._move_index = 0
        self._beat_count = 0
        
        # Compute beat interval from BPM
        beat_interval = 60.0 / routine.bpm
        
        logger.info(
            f"🕺 Starting dance: '{routine.name}' ({len(routine.moves)} moves, "
            f"{routine.total_beats} beats, {routine.bpm:.0f} BPM, "
            f"~{routine.estimated_duration:.0f}s)"
        )
        
        # Publish dance started event
        await self.bus.publish(Event(
            type=EventType.MOTION_STARTED,
            data={
                "action": "dance",
                "routine": routine.name,
                "bpm": routine.bpm,
                "total_beats": routine.total_beats,
                "estimated_duration": routine.estimated_duration,
                "creator": routine.creator,
            },
            source="dance_engine",
        ))
        
        # Execute each move
        try:
            for move in routine.moves:
                if not self._dancing:
                    break
                
                # Execute the move (holds for move.beats beats)
                await self._execute_move(move, beat_interval)
                
                self._move_index += 1
            
            # Dance complete
            await self.bus.publish(Event(
                type=EventType.MOTION_COMPLETED,
                data={
                    "action": "dance",
                    "routine": routine.name,
                    "moves_executed": self._move_index,
                    "beats_played": self._beat_count,
                },
                source="dance_engine",
            ))
            
            logger.info(f"🕺 Dance complete: '{routine.name}' ({self._move_index} moves)")
        
        except asyncio.CancelledError:
            logger.info(f"🕺 Dance cancelled: '{routine.name}'")
            await self.bus.publish(Event(
                type=EventType.MOTION_ABORTED,
                data={"reason": "cancelled", "routine": routine.name},
                source="dance_engine",
            ))
        
        finally:
            self._dancing = False
            self.safety.end_session()
    
    async def stop_dance(self) -> None:
        """Stop the current dance immediately."""
        self._dancing = False
        # Return to standing position
        standing = [90.0] * 18
        await self.servo.smooth_move(standing, duration=0.5, steps=20)
        logger.info("🕺 Dance stopped — returning to standing")
    
    async def _execute_move(self, move: DanceMove, beat_interval: float) -> None:
        """Execute a single dance move, holding for the specified beats.
        
        Args:
            move: The dance move to execute
            beat_interval: Seconds per beat (60 / BPM)
        """
        # Check safety
        safe_angles = self.safety.check_move(
            angles=move.angles,
            feet_on_ground=move.feet_on_ground,
            height_change=move.height_change,
        )
        
        if safe_angles is None:
            logger.warning(f"🕺 Move '{move.name}' blocked by safety — skipping")
            return
        
        # Transition to the new pose
        hold_time = beat_interval * move.beats
        transition_time = min(beat_interval * 0.7, 0.5)  # 70% of beat or max 0.5s
        
        if move.transition == TransitionType.SNAP:
            # Immediate jump
            await self.servo.set_angles(safe_angles)
            await asyncio.sleep(hold_time)
        
        elif move.transition == TransitionType.BOUNCE:
            # Go past target, then settle back (springy)
            overshoot = [a + (a - prev) * 0.15 for a, prev in zip(safe_angles, self._last_angles)]
            # Clamp overshoot to safe range
            overshoot = self.safety._clamp_angles(overshoot) if self.safety else overshoot
            await self.servo.smooth_move(overshoot, duration=transition_time * 0.4, steps=15)
            await self.servo.smooth_move(safe_angles, duration=transition_time * 0.6, steps=15)
            await asyncio.sleep(hold_time - transition_time)
        
        elif move.transition == TransitionType.WAVE:
            # Sinusoidal oscillation (sway back and forth)
            steps = max(10, int(transition_time * 30))
            for i in range(steps):
                t = i / steps
                wave = 0.5 * (1 - np.cos(2 * np.pi * t * 2))  # 2 oscillations
                interp = [prev + (target - prev) * t + np.sin(2 * np.pi * t * 2) * 5 * (1 - t)
                          for prev, target in zip(self._last_angles, safe_angles)]
                interp = self.safety._clamp_angles(interp) if self.safety else interp
                await self.servo.set_angles(interp)
                await asyncio.sleep(transition_time / steps)
            await asyncio.sleep(hold_time - transition_time)
        
        else:  # SMOOTH (default)
            # Cosine-eased smooth transition
            await self.servo.smooth_move(safe_angles, duration=transition_time, steps=30)
            await asyncio.sleep(hold_time - transition_time)
        
        # Update tracking
        self._last_angles = list(safe_angles)
        self._beat_count += move.beats
        
        # Publish move event
        await self.bus.publish(Event(
            type=EventType.MOTION_COMPLETED,
            data={
                "action": "dance_move",
                "move_name": move.name,
                "move_index": self._move_index,
                "beats": move.beats,
                "transition": move.transition.value,
            },
            source="dance_engine",
        ))
    
    # ─── Status ────────────────────────────────────────────────────────
    
    def get_status(self) -> dict[str, Any]:
        """Get dance engine status."""
        return {
            "running": self._running,
            "dancing": self._dancing,
            "routine": self._current_routine.name if self._current_routine else None,
            "move_index": self._move_index,
            "total_moves": len(self._current_routine.moves) if self._current_routine else 0,
            "beat_count": self._beat_count,
            "safety": self.safety.get_status(),
        }