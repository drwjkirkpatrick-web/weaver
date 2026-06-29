# move_library.py — Pre-built dance moves for the hexapod
#
# This library contains safe, pre-tested dance moves that students can
# use to build their own dances. Each move is a DanceMove object with
# 18 servo angles that have been verified safe by the dance safety governor.
#
# Moves are organized by category:
#   - BODY: Body sways, leans, bounces (all 6 feet stay down)
#   - LEGS: Individual leg lifts, waves, taps (5 feet stay down)
#   - ARMS: Front leg gestures (like waving arms — 4 feet down)
#   - POSES: Static poses (standing tall, crouching, etc.)
#
# Students mix and match these moves to create a DanceRoutine.
#
# All angles are in degrees (0-180, 90 = center).
# Leg channel mapping: [leg0_coxa, leg0_femur, leg0_tibia,
#                       leg1_coxa, leg1_femur, leg1_tibia, ...]
# Legs: 0=FL, 1=ML, 2=RL, 3=RR, 4=MR, 5=FR
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from weaver.dance.dance_engine import DanceMove, TransitionType


# ─── Helper: standing pose (all centered) ──────────────────────────────

def _standing() -> list[float]:
    """All 18 servos centered — standing pose."""
    return [90.0] * 18


def _with_leg(angles: list[float], leg: int, coxa: float, femur: float, tibia: float) -> list[float]:
    """Set one leg's 3 joint angles in a copy of the angles list.
    
    Args:
        angles: Base angles (will be copied, not modified)
        leg: Leg index (0-5)
        coxa: Coxa angle (horizontal rotation)
        femur: Femur angle (vertical lift)
        tibia: Tibia angle (knee)
    
    Returns:
        New list of 18 angles with the specified leg's joints set.
    """
    result = list(angles)
    result[leg * 3] = coxa
    result[leg * 3 + 1] = femur
    result[leg * 3 + 2] = tibia
    return result


# ═════════════════════════════════════════════════════════════════════════
# BODY MOVES (all 6 feet stay on ground — very safe)
# ═════════════════════════════════════════════════════════════════════════

def body_sway_right(beats: int = 1) -> DanceMove:
    """Sway body to the right (all feet stay on ground)."""
    angles = _standing()
    # Tilt body right: lower right legs, raise left legs slightly
    for leg in [0, 1, 2]:  # Left legs raise
        angles[leg * 3 + 1] = 95  # Femur slightly up
    for leg in [3, 4, 5]:  # Right legs lower
        angles[leg * 3 + 1] = 85  # Femur slightly down
    return DanceMove(
        name="sway_right",
        angles=angles,
        beats=beats,
        transition=TransitionType.SMOOTH,
        feet_on_ground=6,
        height_change=0,
    )


def body_sway_left(beats: int = 1) -> DanceMove:
    """Sway body to the left (all feet stay on ground)."""
    angles = _standing()
    for leg in [0, 1, 2]:  # Left legs lower
        angles[leg * 3 + 1] = 85
    for leg in [3, 4, 5]:  # Right legs raise
        angles[leg * 3 + 1] = 95
    return DanceMove(
        name="sway_left",
        angles=angles,
        beats=beats,
        transition=TransitionType.SMOOTH,
        feet_on_ground=6,
        height_change=0,
    )


def body_bounce_up(beats: int = 1) -> DanceMove:
    """Bounce body up slightly (all feet on ground, body raises)."""
    angles = _standing()
    for leg in range(6):
        angles[leg * 3 + 1] = 80  # Femur up (lift body)
    return DanceMove(
        name="bounce_up",
        angles=angles,
        beats=beats,
        transition=TransitionType.BOUNCE,
        feet_on_ground=6,
        height_change=15,
    )


def body_bounce_down(beats: int = 1) -> DanceMove:
    """Bounce body down slightly (all feet on ground, body lowers)."""
    angles = _standing()
    for leg in range(6):
        angles[leg * 3 + 1] = 100  # Femur down (lower body)
    return DanceMove(
        name="bounce_down",
        angles=angles,
        beats=beats,
        transition=TransitionType.BOUNCE,
        feet_on_ground=6,
        height_change=-15,
    )


def body_lean_forward(beats: int = 1) -> DanceMove:
    """Lean body forward (all feet on ground)."""
    angles = _standing()
    for leg in range(6):
        angles[leg * 3] = 85  # Coxa forward
    return DanceMove(
        name="lean_forward",
        angles=angles,
        beats=beats,
        transition=TransitionType.SMOOTH,
        feet_on_ground=6,
    )


def body_lean_back(beats: int = 1) -> DanceMove:
    """Lean body backward (all feet on ground)."""
    angles = _standing()
    for leg in range(6):
        angles[leg * 3] = 95  # Coxa back
    return DanceMove(
        name="lean_back",
        angles=angles,
        beats=beats,
        transition=TransitionType.SMOOTH,
        feet_on_ground=6,
    )


def body_wiggle(beats: int = 2) -> DanceMove:
    """Wiggle body side to side (small coxa oscillation)."""
    angles = _standing()
    for leg in range(6):
        if leg % 2 == 0:
            angles[leg * 3] = 82  # Left legs forward
        else:
            angles[leg * 3] = 98  # Right legs back
    return DanceMove(
        name="wiggle",
        angles=angles,
        beats=beats,
        transition=TransitionType.WAVE,
        feet_on_ground=6,
    )


# ═════════════════════════════════════════════════════════════════════════
# LEG MOVES (one leg lifts — 5 feet on ground — still very safe)
# ═════════════════════════════════════════════════════════════════════════

def leg_lift_front_left(beats: int = 1) -> DanceMove:
    """Lift front-left leg up."""
    angles = _with_leg(_standing(), 0, coxa=90, femur=70, tibia=70)
    return DanceMove(name="leg_lift_FL", angles=angles, beats=beats,
                     feet_on_ground=5, transition=TransitionType.SMOOTH)


def leg_lift_front_right(beats: int = 1) -> DanceMove:
    """Lift front-right leg up."""
    angles = _with_leg(_standing(), 5, coxa=90, femur=70, tibia=70)
    return DanceMove(name="leg_lift_FR", angles=angles, beats=beats,
                     feet_on_ground=5, transition=TransitionType.SMOOTH)


def leg_lift_mid_left(beats: int = 1) -> DanceMove:
    """Lift mid-left leg up."""
    angles = _with_leg(_standing(), 1, coxa=90, femur=70, tibia=70)
    return DanceMove(name="leg_lift_ML", angles=angles, beats=beats,
                     feet_on_ground=5, transition=TransitionType.SMOOTH)


def leg_lift_mid_right(beats: int = 1) -> DanceMove:
    """Lift mid-right leg up."""
    angles = _with_leg(_standing(), 4, coxa=90, femur=70, tibia=70)
    return DanceMove(name="leg_lift_MR", angles=angles, beats=beats,
                     feet_on_ground=5, transition=TransitionType.SMOOTH)


def leg_lift_rear_left(beats: int = 1) -> DanceMove:
    """Lift rear-left leg up."""
    angles = _with_leg(_standing(), 2, coxa=90, femur=70, tibia=70)
    return DanceMove(name="leg_lift_RL", angles=angles, beats=beats,
                     feet_on_ground=5, transition=TransitionType.SMOOTH)


def leg_lift_rear_right(beats: int = 1) -> DanceMove:
    """Lift rear-right leg up."""
    angles = _with_leg(_standing(), 3, coxa=90, femur=70, tibia=70)
    return DanceMove(name="leg_lift_RR", angles=angles, beats=beats,
                     feet_on_ground=5, transition=TransitionType.SMOOTH)


def leg_wave_front_right(beats: int = 2) -> DanceMove:
    """Wave front-right leg like waving hello."""
    angles = _with_leg(_standing(), 5, coxa=110, femur=65, tibia=55)
    return DanceMove(name="wave_FR", angles=angles, beats=beats,
                     feet_on_ground=5, transition=TransitionType.WAVE)


def leg_wave_front_left(beats: int = 2) -> DanceMove:
    """Wave front-left leg."""
    angles = _with_leg(_standing(), 0, coxa=70, femur=65, tibia=55)
    return DanceMove(name="wave_FL", angles=angles, beats=beats,
                     feet_on_ground=5, transition=TransitionType.WAVE)


def leg_tap_front_left(beats: int = 1) -> DanceMove:
    """Tap front-left leg (lift slightly and set down quickly)."""
    angles = _with_leg(_standing(), 0, coxa=90, femur=80, tibia=80)
    return DanceMove(name="tap_FL", angles=angles, beats=beats,
                     feet_on_ground=5, transition=TransitionType.BOUNCE)


def leg_tap_front_right(beats: int = 1) -> DanceMove:
    """Tap front-right leg."""
    angles = _with_leg(_standing(), 5, coxa=90, femur=80, tibia=80)
    return DanceMove(name="tap_FR", angles=angles, beats=beats,
                     feet_on_ground=5, transition=TransitionType.BOUNCE)


# ═════════════════════════════════════════════════════════════════════════
# ARM-LIKE MOVES (both front legs gesture — 4 feet on ground)
# ═════════════════════════════════════════════════════════════════════════

def arms_up(beats: int = 1) -> DanceMove:
    """Lift both front legs up like raising arms."""
    angles = _standing()
    # Front-left leg up
    angles = _with_leg(angles, 0, coxa=90, femur=60, tibia=50)
    # Front-right leg up
    angles = _with_leg(angles, 5, coxa=90, femur=60, tibia=50)
    return DanceMove(name="arms_up", angles=angles, beats=beats,
                     feet_on_ground=4, transition=TransitionType.SMOOTH)


def arms_out_left(beats: int = 1) -> DanceMove:
    """Front-left leg out to the side like arm extended."""
    angles = _with_leg(_standing(), 0, coxa=70, femur=70, tibia=60)
    return DanceMove(name="arm_out_left", angles=angles, beats=beats,
                     feet_on_ground=5, transition=TransitionType.SMOOTH)


def arms_out_right(beats: int = 1) -> DanceMove:
    """Front-right leg out to the side."""
    angles = _with_leg(_standing(), 5, coxa=110, femur=70, tibia=60)
    return DanceMove(name="arm_out_right", angles=angles, beats=beats,
                     feet_on_ground=5, transition=TransitionType.SMOOTH)


def arms_crossed(beats: int = 1) -> DanceMove:
    """Cross front legs inward (like crossing arms)."""
    angles = _standing()
    angles = _with_leg(angles, 0, coxa=105, femur=85, tibia=85)
    angles = _with_leg(angles, 5, coxa=75, femur=85, tibia=85)
    return DanceMove(name="arms_crossed", angles=angles, beats=beats,
                     feet_on_ground=4, transition=TransitionType.SMOOTH)


# ═════════════════════════════════════════════════════════════════════════
# POSES (static positions)
# ═════════════════════════════════════════════════════════════════════════

def pose_standing(beats: int = 1) -> DanceMove:
    """Normal standing pose."""
    return DanceMove(name="standing", angles=_standing(), beats=beats,
                     transition=TransitionType.SMOOTH, feet_on_ground=6)


def pose_tall(beats: int = 1) -> DanceMove:
    """Standing tall on tip-toes."""
    angles = _standing()
    for leg in range(6):
        angles[leg * 3 + 1] = 75  # Femur up
        angles[leg * 3 + 2] = 75  # Tibia up
    return DanceMove(name="tall", angles=angles, beats=beats,
                     feet_on_ground=6, height_change=20)


def pose_crouch(beats: int = 1) -> DanceMove:
    """Crouching low."""
    angles = _standing()
    for leg in range(6):
        angles[leg * 3 + 1] = 105  # Femur down
        angles[leg * 3 + 2] = 105  # Tibia down
    return DanceMove(name="crouch", angles=angles, beats=beats,
                     feet_on_ground=6, height_change=-20)


def pose_twist_left(beats: int = 1) -> DanceMove:
    """Twist body to face left."""
    angles = _standing()
    for leg in range(6):
        if leg < 3:
            angles[leg * 3] = 80  # Left legs forward
        else:
            angles[leg * 3] = 80  # Right legs back
    return DanceMove(name="twist_left", angles=angles, beats=beats,
                     feet_on_ground=6)


def pose_twist_right(beats: int = 1) -> DanceMove:
    """Twist body to face right."""
    angles = _standing()
    for leg in range(6):
        if leg < 3:
            angles[leg * 3] = 100
        else:
            angles[leg * 3] = 100
    return DanceMove(name="twist_right", angles=angles, beats=beats,
                     feet_on_ground=6)


# ═════════════════════════════════════════════════════════════════════════
# COMBO MOVES (multi-beat sequences in a single move)
# ═════════════════════════════════════════════════════════════════════════

def spin_right(beats: int = 4) -> DanceMove:
    """Rotate body right (all feet shuffle)."""
    angles = _standing()
    for leg in range(6):
        angles[leg * 3] = 95  # All coxae rotate right
    return DanceMove(name="spin_right", angles=angles, beats=beats,
                     transition=TransitionType.WAVE, feet_on_ground=6)


def spin_left(beats: int = 4) -> DanceMove:
    """Rotate body left (all feet shuffle)."""
    angles = _standing()
    for leg in range(6):
        angles[leg * 3] = 85  # All coxae rotate left
    return DanceMove(name="spin_left", angles=angles, beats=beats,
                     transition=TransitionType.WAVE, feet_on_ground=6)


def bob_and_weave(beats: int = 4) -> DanceMove:
    """Bob up and down while swaying (boxing-style)."""
    angles = _standing()
    for leg in [0, 1, 2]:
        angles[leg * 3 + 1] = 85  # Left side up
    for leg in [3, 4, 5]:
        angles[leg * 3 + 1] = 85  # Right side also up (body rises)
    return DanceMove(name="bob_weave", angles=angles, beats=beats,
                     transition=TransitionType.BOUNCE, feet_on_ground=6,
                     height_change=10)


# ─── Move Library Catalog ─────────────────────────────────────────────

ALL_MOVES = {
    # Body moves
    "sway_right": body_sway_right,
    "sway_left": body_sway_left,
    "bounce_up": body_bounce_up,
    "bounce_down": body_bounce_down,
    "lean_forward": body_lean_forward,
    "lean_back": body_lean_back,
    "wiggle": body_wiggle,
    
    # Leg moves
    "lift_FL": leg_lift_front_left,
    "lift_FR": leg_lift_front_right,
    "lift_ML": leg_lift_mid_left,
    "lift_MR": leg_lift_mid_right,
    "lift_RL": leg_lift_rear_left,
    "lift_RR": leg_lift_rear_right,
    "wave_FL": leg_wave_front_left,
    "wave_FR": leg_wave_front_right,
    "tap_FL": leg_tap_front_left,
    "tap_FR": leg_tap_front_right,
    
    # Arm moves
    "arms_up": arms_up,
    "arm_out_left": arms_out_left,
    "arm_out_right": arms_out_right,
    "arms_crossed": arms_crossed,
    
    # Poses
    "standing": pose_standing,
    "tall": pose_tall,
    "crouch": pose_crouch,
    "twist_left": pose_twist_left,
    "twist_right": pose_twist_right,
    
    # Combos
    "spin_right": spin_right,
    "spin_left": spin_left,
    "bob_weave": bob_and_weave,
}


def get_move(name: str, beats: int = 1) -> DanceMove:
    """Get a pre-built dance move by name.
    
    Args:
        name: Move name from ALL_MOVES
        beats: Number of beats to hold
    
    Returns:
        A DanceMove object.
    
    Raises:
        KeyError: If the move name is not found.
    """
    if name not in ALL_MOVES:
        raise KeyError(f"Unknown move: '{name}'. Available: {list(ALL_MOVES.keys())}")
    return ALL_MOVES[name](beats=beats)


def list_moves() -> list[str]:
    """List all available move names."""
    return sorted(ALL_MOVES.keys())