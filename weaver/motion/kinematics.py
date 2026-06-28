# kinematics.py — Hexapod inverse kinematics
#
# Inverse kinematics (IK) is the math of "given a desired foot position,
# what angles should the joints be at?"
#
# For a hexapod with 6 legs, each having 3 joints (coxa, femur, tibia),
# we need to solve 3 angles per leg given an (x, y, z) foot target.
#
# This is a simplified 2D IK per leg — the coxa handles the horizontal
# rotation (in the XY plane), and the femur+tibia handle the vertical
# (in the XZ plane after coxa rotation).
#
# Coordinate system (per leg, body-relative):
#   X = forward (away from body center)
#   Y = sideways (left/right from body center)
#   Z = vertical (up/down, negative = below body)
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from weaver.config import GaitConfig, get_config


@dataclass
class LegPosition:
    """Desired foot position relative to the leg's coxa joint."""
    x: float  # Forward (mm)
    y: float  # Sideways (mm)
    z: float  # Vertical (mm, negative = down)


@dataclass
class JointAngles:
    """Computed joint angles for a single leg."""
    coxa: float   # degrees (0-180, 90=center)
    femur: float  # degrees
    tibia: float  # degrees


class HexapodKinematics:
    """Inverse kinematics for a 6-leg, 3-DOF hexapod.
    
    Each leg has:
    - Coxa:  horizontal rotation (swings leg forward/backward)
    - Femur: vertical rotation (lifts leg up/down at the hip)
    - Tibia: vertical rotation (bends the knee)
    
    The IK is solved per-leg in two steps:
    1. Coxa angle from the XY position (atan2)
    2. Femur + tibia from the XZ position (law of cosines)
    
    Body-level transforms (roll, pitch, yaw, translation) are applied
    to each leg's foot position before solving IK, so the body can
    tilt/shift while feet stay planted.
    """
    
    # Default leg dimensions (mm) — typical RaspClaws measurements
    # These can be overridden per-leg or via config
    COXA_LENGTH: float = 30.0    # Distance from body center to coxa joint
    FEMUR_LENGTH: float = 55.0   # Hip to knee
    TIBIA_LENGTH: float = 75.0    # Knee to foot
    
    # Leg mounting positions on the body (x, y) relative to body center
    # Order: Front-Left, Mid-Left, Rear-Left, Rear-Right, Mid-Right, Front-Right
    LEG_MOUNT_POSITIONS: list[tuple[float, float]] = [
        (35.0,  50.0),   # Leg 0: Front-Left
        (0.0,   60.0),   # Leg 1: Mid-Left
        (-35.0, 50.0),   # Leg 2: Rear-Left
        (-35.0, -50.0),  # Leg 3: Rear-Right
        (0.0,  -60.0),   # Leg 4: Mid-Right
        (35.0, -50.0),   # Leg 5: Front-Right
    ]
    
    # Default standing foot positions (relative to coxa joint)
    # x=forward, y=sideways, z=down (negative)
    DEFAULT_STANCE: list[LegPosition] = [
        LegPosition(80.0,  30.0, -80.0),   # FL
        LegPosition(90.0,  20.0, -80.0),   # ML
        LegPosition(80.0,  10.0, -80.0),   # RL
        LegPosition(80.0, -10.0, -80.0),   # RR
        LegPosition(90.0, -20.0, -80.0),   # MR
        LegPosition(80.0, -30.0, -80.0),   # FR
    ]
    
    def __init__(self, config: GaitConfig | None = None):
        self.config = config or get_config().gait
        logger.info("📐 Hexapod kinematics initialized")
    
    # ─── Per-Leg IK ───────────────────────────────────────────────────
    
    def solve_leg_ik(self, foot: LegPosition, leg_index: int = 0) -> JointAngles:
        """Solve inverse kinematics for a single leg.
        
        Given a desired foot position (relative to the coxa joint),
        compute the three joint angles.
        
        Args:
            foot: Target foot position (x, y, z) in mm.
            leg_index: Which leg (0-5) — affects coxa sign convention.
        
        Returns:
            JointAngles with coxa, femur, tibia in degrees (0-180, 90=center).
        """
        x, y, z = foot.x, foot.y, foot.z
        
        # ── Step 1: Coxa angle ──
        # The coxa rotates the leg in the XY plane.
        # atan2(y, x) gives the angle from forward direction.
        coxa_rad = math.atan2(y, x)
        coxa_deg = math.degrees(coxa_rad) + 90.0  # Offset: 90° = center
        
        # ── Step 2: Project to leg plane ──
        # After coxa rotation, the leg operates in a 2D plane defined by
        # the radial distance (in XY) and the Z (vertical).
        r = math.sqrt(x * x + y * y)  # Radial distance from coxa
        # Subtract coxa length (foot is beyond the coxa joint)
        r_eff = r - self.COXA_LENGTH
        if r_eff < 0:
            r_eff = 0
        
        # ── Step 3: Femur + Tibia (2D IK via law of cosines) ──
        # We have a triangle: hip → knee → foot
        #   Side a = femur length
        #   Side b = tibia length
        #   Side c = distance from hip to foot = sqrt(r_eff² + z²)
        a = self.FEMUR_LENGTH
        b = self.TIBIA_LENGTH
        c = math.sqrt(r_eff * r_eff + z * z)
        
        # Clamp c to reachable range
        c = max(abs(a - b) + 1, min(a + b - 1, c))
        
        # Femur angle: angle at the hip joint
        # Using law of cosines: cos(A) = (a² + c² - b²) / (2ac)
        cos_femur = (a * a + c * c - b * b) / (2 * a * c)
        cos_femur = max(-1.0, min(1.0, cos_femur))  # Clamp for float safety
        femur_angle = math.acos(cos_femur)
        
        # The femur angle is measured from the horizontal, so we need to
        # add the angle from the hip-to-foot direction
        hip_angle = math.atan2(-z, r_eff)  # Negative z because z is down
        femur_rad = hip_angle + (femur_angle - math.pi / 2)
        femur_deg = math.degrees(femur_rad) + 90.0
        
        # Tibia angle: angle at the knee joint
        # cos(B) = (a² + b² - c²) / (2ab)
        cos_tibia = (a * a + b * b - c * c) / (2 * a * b)
        cos_tibia = max(-1.0, min(1.0, cos_tibia))
        tibia_angle = math.acos(cos_tibia)
        tibia_deg = math.degrees(tibia_angle)
        
        # Clamp all angles to servo range (0-180)
        coxa_deg = max(0.0, min(180.0, coxa_deg))
        femur_deg = max(0.0, min(180.0, femur_deg))
        tibia_deg = max(0.0, min(180.0, tibia_deg))
        
        return JointAngles(coxa=coxa_deg, femur=femur_deg, tibia=tibia_deg)
    
    # ─── Full-Body IK ─────────────────────────────────────────────────
    
    def solve_body_ik(
        self,
        foot_positions: list[LegPosition] | None = None,
        body_roll: float = 0.0,
        body_pitch: float = 0.0,
        body_yaw: float = 0.0,
        body_x: float = 0.0,
        body_y: float = 0.0,
        body_z: float = 0.0,
    ) -> list[float]:
        """Solve IK for all 6 legs with body transform applied.
        
        The body transform (roll/pitch/yaw/translation) is applied to each
        foot position BEFORE solving per-leg IK. This lets the body tilt
        or shift while feet stay planted on the ground.
        
        Args:
            foot_positions: Target foot positions per leg. If None, uses default stance.
            body_roll: Body roll in degrees (left-right tilt).
            body_pitch: Body pitch in degrees (forward-backward tilt).
            body_yaw: Body yaw in degrees (rotation).
            body_x: Body X translation in mm.
            body_y: Body Y translation in mm.
            body_z: Body Z translation in mm (positive = body up).
        
        Returns:
            List of 18 joint angles (6 legs × 3 joints) in degrees.
        """
        if foot_positions is None:
            foot_positions = list(self.DEFAULT_STANCE)
        
        # Convert body angles to radians
        roll_rad = math.radians(body_roll)
        pitch_rad = math.radians(body_pitch)
        yaw_rad = math.radians(body_yaw)
        
        # Rotation matrix elements (ZYX Euler convention)
        cr, sr = math.cos(roll_rad), math.sin(roll_rad)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        
        all_angles: list[float] = []
        
        for i, foot in enumerate(foot_positions):
            # Get leg mount position
            mount_x, mount_y = self.LEG_MOUNT_POSITIONS[i]
            
            # Foot position in body frame (relative to body center)
            fx = foot.x + mount_x
            fy = foot.y + mount_y
            fz = foot.z
            
            # Apply body translation (inverse — body moves, feet stay)
            fx -= body_x
            fy -= body_y
            fz -= body_z
            
            # Apply body rotation (inverse rotation to foot positions)
            # Roll (rotation around X axis)
            new_y = fy * cr - fz * sr
            new_z = fy * sr + fz * cr
            fy, fz = new_y, new_z
            
            # Pitch (rotation around Y axis)
            new_x = fx * cp + fz * sp
            new_z = -fx * sp + fz * cp
            fx, fz = new_x, new_z
            
            # Yaw (rotation around Z axis)
            new_x = fx * cy - fy * sy
            new_y = fx * sy + fy * cy
            fx, fy = new_x, new_y
            
            # Convert back to leg-relative coordinates
            leg_foot = LegPosition(
                x=fx - mount_x,
                y=fy - mount_y,
                z=fz,
            )
            
            # Solve per-leg IK
            angles = self.solve_leg_ik(leg_foot, i)
            all_angles.extend([angles.coxa, angles.femur, angles.tibia])
        
        return all_angles
    
    # ─── Stance Presets ────────────────────────────────────────────────
    
    def standing_stance(self, height: float = -80.0) -> list[float]:
        """Compute angles for a standing pose at given height.
        
        Args:
            height: Body height in mm (negative = below coxa level).
        
        Returns:
            List of 18 joint angles.
        """
        feet = [
            LegPosition(80.0,  30.0, height),
            LegPosition(90.0,  20.0, height),
            LegPosition(80.0,  10.0, height),
            LegPosition(80.0, -10.0, height),
            LegPosition(90.0, -20.0, height),
            LegPosition(80.0, -30.0, height),
        ]
        return self.solve_body_ik(feet)
    
    def crouch_stance(self) -> list[float]:
        """Low crouch — body close to ground."""
        return self.standing_stance(height=-40.0)
    
    def tall_stance(self) -> list[float]:
        """Standing tall — maximum height."""
        return self.standing_stance(height=-120.0)
    
    def lean_stance(self, direction: str = "forward", amount: float = 10.0) -> list[float]:
        """Lean the body in a direction.
        
        Args:
            direction: "forward", "backward", "left", "right"
            amount: Lean angle in degrees
        """
        pitch = 0.0
        roll = 0.0
        if direction == "forward":
            pitch = amount
        elif direction == "backward":
            pitch = -amount
        elif direction == "left":
            roll = amount
        elif direction == "right":
            roll = -amount
        
        return self.solve_body_ik(
            body_roll=roll,
            body_pitch=pitch,
        )