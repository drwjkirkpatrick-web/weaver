# student_api.py — Simple dance creation API for students
#
# This module provides a simplified interface for students to create
# their own robot dances WITHOUT needing to understand servo angles,
# I2C channels, or safety thresholds.
#
# Students create dances by chaining named moves from the move library:
#
#   from weaver.dance.student_api import StudentDance
#
#   my_dance = StudentDance("My Dance", bpm=120)
#   my_dance.add("arms_up", beats=2)
#   my_dance.add("sway_right", beats=1)
#   my_dance.add("sway_left", beats=1)
#   my_dance.add("spin_right", beats=4)
#   my_dance.add("standing", beats=2)
#
#   await my_dance.perform(engine)
#
# The student API handles:
#   - Converting move names to servo angles (via move_library)
#   - Safety checking (via dance_safety governor)
#   - Beat synchronization (via dance_engine)
#   - Smooth transitions between moves
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any

from loguru import logger

from weaver.dance.dance_engine import DanceEngine, DanceRoutine, DanceMove, TransitionType
from weaver.dance.move_library import ALL_MOVES, get_move, list_moves


class StudentDance:
    """A student-friendly dance builder.
    
    Students create dances by adding named moves with beat counts.
    No need to touch servo angles directly.
    
    Example:
        dance = StudentDance("My First Dance", bpm=120)
        dance.add("arms_up", beats=2)
        dance.add("sway_right", beats=1)
        dance.add("sway_left", beats=1)
        dance.add("standing", beats=2)
        
        await dance.perform(engine)
    
    Available moves (use list_available_moves() to see all):
        Body: sway_right, sway_left, bounce_up, bounce_down, lean_forward, lean_back, wiggle
        Legs: lift_FL, lift_FR, lift_ML, lift_MR, lift_RL, lift_RR, wave_FL, wave_FR, tap_FL, tap_FR
        Arms: arms_up, arm_out_left, arm_out_right, arms_crossed
        Poses: standing, tall, crouch, twist_left, twist_right
        Combos: spin_right, spin_left, bob_weave
    """
    
    def __init__(
        self,
        name: str = "Student Dance",
        bpm: float = 120.0,
        creator: str = "Student",
        description: str = "",
    ):
        """Create a new student dance.
        
        Args:
            name: Name of your dance (e.g., "My Cool Dance")
            bpm: Tempo in beats per minute (60-200). 120 is a good default.
            creator: Your name (for credit!)
            description: A short description of your dance
        """
        self.name = name
        self.bpm = max(60.0, min(200.0, bpm))
        self.creator = creator
        self.description = description
        self._moves: list[DanceMove] = []
        
        logger.info(f"🕺 Student dance created: '{name}' by {creator} ({bpm:.0f} BPM)")
    
    def add(self, move_name: str, beats: int = 1) -> "StudentDance":
        """Add a move to the dance.
        
        Args:
            move_name: Name of the move (see list_available_moves())
            beats: How many beats to hold this move (1-8)
        
        Returns:
            self (for chaining: dance.add("arms_up").add("spin_right"))
        
        Raises:
            KeyError: If the move name is not recognized.
        """
        if move_name not in ALL_MOVES:
            raise KeyError(
                f"Unknown move: '{move_name}'. "
                f"Available: {list(ALL_MOVES.keys())}"
            )
        
        beats = max(1, min(8, beats))
        move = get_move(move_name, beats=beats)
        self._moves.append(move)
        
        logger.debug(f"  + {move_name} ({beats} beat{'s' if beats > 1 else ''})")
        return self
    
    def add_pose(self, move_name: str, beats: int = 2) -> "StudentDance":
        """Add a pose (alias for add, with default 2 beats)."""
        return self.add(move_name, beats)
    
    def add_pause(self, beats: int = 1) -> "StudentDance":
        """Add a pause (standing still for the given beats)."""
        return self.add("standing", beats)
    
    def repeat(self, move_name: str, times: int = 2, beats: int = 1) -> "StudentDance":
        """Repeat a move multiple times.
        
        Args:
            move_name: Name of the move to repeat
            times: How many times to repeat
            beats: Beats per repetition
        """
        for _ in range(times):
            self.add(move_name, beats)
        return self
    
    def sequence(self, *move_names: str, beats: int = 1) -> "StudentDance":
        """Add a sequence of moves, each with the same beat count.
        
        Args:
            move_names: Multiple move names to add in sequence
            beats: Beats for each move
        """
        for name in move_names:
            self.add(name, beats)
        return self
    
    def build(self) -> DanceRoutine:
        """Build the dance into a DanceRoutine for the engine.
        
        Returns:
            A DanceRoutine object ready to be performed.
        """
        if not self._moves:
            logger.warning(f"🕺 Dance '{self.name}' has no moves!")
            self.add("standing", beats=2)
        
        routine = DanceRoutine(
            name=self.name,
            moves=list(self._moves),
            bpm=self.bpm,
            description=self.description,
            creator=self.creator,
        )
        
        logger.info(
            f"🕺 Dance '{self.name}' built: {len(self._moves)} moves, "
            f"{routine.total_beats} beats, ~{routine.estimated_duration:.0f}s"
        )
        return routine
    
    async def perform(self, engine: DanceEngine) -> None:
        """Perform this dance using the given dance engine.
        
        Args:
            engine: A started DanceEngine instance
        """
        routine = self.build()
        await engine.dance(routine)
    
    # ─── Info ──────────────────────────────────────────────────────────
    
    def info(self) -> str:
        """Get a human-readable description of this dance."""
        lines = [
            f"🕺 Dance: '{self.name}' by {self.creator}",
            f"   BPM: {self.bpm:.0f}",
            f"   Moves: {len(self._moves)}",
            f"   Total beats: {sum(m.beats for m in self._moves)}",
            f"   Duration: ~{sum(m.beats for m in self._moves) * 60 / self.bpm:.0f}s",
            f"   Moves list:",
        ]
        for i, move in enumerate(self._moves):
            lines.append(f"     {i+1}. {move.name} ({move.beats} beat{'s' if move.beats > 1 else ''})")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"StudentDance(name='{self.name}', bpm={self.bpm:.0f}, "
            f"moves={len(self._moves)}, creator='{self.creator}')"
        )


# ─── Static helpers ────────────────────────────────────────────────────


def list_available_moves() -> list[str]:
    """List all available dance move names that students can use."""
    return list_moves()


def print_move_guide() -> None:
    """Print a guide of all available moves for students."""
    print("\n🕷️ Weaver Dance Move Guide")
    print("=" * 50)
    print("\nBODY MOVES (safest — all feet stay down):")
    for name in sorted(ALL_MOVES.keys()):
        if "sway" in name or "bounce" in name or "lean" in name or "wiggle" in name:
            print(f"  {name}")
    
    print("\nLEG MOVES (one leg lifts — 5 feet on ground):")
    for name in sorted(ALL_MOVES.keys()):
        if "lift" in name or "tap" in name:
            print(f"  {name}")
    
    print("\nARM MOVES (front legs gesture — like arms):")
    for name in sorted(ALL_MOVES.keys()):
        if "arm" in name:
            print(f"  {name}")
    
    print("\nWAVE MOVES (wave legs hello):")
    for name in sorted(ALL_MOVES.keys()):
        if "wave" in name:
            print(f"  {name}")
    
    print("\nPOSES (static positions):")
    for name in sorted(ALL_MOVES.keys()):
        if name in ("standing", "tall", "crouch", "twist_left", "twist_right"):
            print(f"  {name}")
    
    print("\nCOMBOS (multi-beat sequences):")
    for name in sorted(ALL_MOVES.keys()):
        if "spin" in name or "bob" in name:
            print(f"  {name}")
    
    print("\n" + "=" * 50)
    print("Usage: dance.add('move_name', beats=2)")
    print("Example: dance.add('arms_up', beats=1).add('sway_right', beats=2)")
    print("=" * 50 + "\n")