# dance_safety.py — Servo safety thresholds for dance moves
#
# Dancing pushes servos harder than walking. Rapid repeated movements,
# wide angle sweeps, and bouncy rhythms all increase heat and stress.
# This module enforces dance-specific safety limits ON TOP of the
# general safety governor.
#
# Key safety concerns when dancing:
#   1. SERVO HEAT: Fast repetitive motion heats servos quickly.
#      → Limit dance session duration, enforce cooldown periods.
#   2. ANGLE RANGE: Dance moves may want extreme poses.
#      → Clamp all angles to dance-safe range (narrower than walking range).
#   3. SPEED: Dance tempos can be fast (120-140 BPM).
#      → Limit how fast servos can move between positions.
#   4. STABILITY: Some dance moves lift multiple legs simultaneously.
#      → Require minimum 3 feet on ground at all times.
#   5. DUTY CYCLE: Servos have a rated duty cycle for continuous operation.
#      → Limit moves per second to prevent overheating.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from weaver.config import ServoConfig, get_config


# ─── Dance Safety Levels ──────────────────────────────────────────────


class DanceSafetyLevel(str, Enum):
    """Safety levels for dancing — more restrictive than walking safety."""
    STRICT = "strict"      # Kids/demo: slow, small movements, short sessions
    NORMAL = "normal"      # Regular dancing: moderate speed, normal range
    PERFORMANCE = "performance"  # Stage performance: full range, longer sessions
    DISABLED = "disabled"  # No limits (maintenance/calibration only — DANGEROUS)


@dataclass
class DanceThresholds:
    """Servo safety thresholds for dance moves.
    
    These are NARROWER than the walking thresholds because dancing
    involves faster, more repetitive motion that stresses servos more.
    """
    # ─── Angle limits (degrees, 90 = center) ───
    # Dancing uses a narrower safe range than walking to reduce stress
    coxa_min: float = 60.0    # Horizontal rotation min (was 45 for walking)
    coxa_max: float = 120.0   # Horizontal rotation max
    femur_min: float = 50.0   # Vertical lift min
    femur_max: float = 130.0  # Vertical lift max
    tibia_min: float = 50.0   # Knee bend min
    tibia_max: float = 130.0  # Knee bend max
    
    # ─── Speed limits ───
    # Max degrees per second a servo can move during dance
    max_servo_speed_deg_s: float = 60.0   # Conservative for repetitive motion
    # Max moves per second (each move = one set of 18 servo writes)
    max_moves_per_second: float = 4.0     # 240 BPM max (very fast!)
    # Minimum time between moves (ms) — prevents servo buzzing
    min_move_interval_ms: float = 50.0
    
    # ─── Session limits ───
    # Max continuous dance time before mandatory cooldown (seconds)
    max_session_seconds: float = 120.0   # 2 minutes
    # Required cooldown after max session (seconds)
    cooldown_seconds: float = 30.0       # 30 sec rest
    # Max total dance time per hour (prevents cumulative heat buildup)
    max_total_per_hour: float = 300.0     # 5 minutes per hour
    
    # ─── Stability requirements ───
    # Minimum feet on ground at all times during dance
    min_feet_on_ground: int = 3          # Tripod stability minimum
    # Max legs that can lift simultaneously
    max_simultaneous_lift: int = 2       # Only 2 legs can lift at once
    # Max body height change per beat (mm) — prevents jarring drops
    max_height_change_per_beat: float = 20.0
    
    # ─── Temperature monitoring ───
    # Max servo temperature before pausing dance (°C)
    max_servo_temp_c: float = 65.0       # Lower than walking (70°C)
    # Pause duration if temp exceeded (seconds)
    temp_pause_seconds: float = 10.0


@dataclass
class DanceSessionState:
    """Tracks the current dance session safety state."""
    active: bool = False
    started_at: float = 0.0
    duration: float = 0.0
    move_count: int = 0
    last_move_time: float = 0.0
    moves_this_second: int = 0
    moves_this_second_window: float = 0.0  # Timestamp of current 1s window
    total_dance_time_today: float = 0.0
    cooldown_until: float = 0.0  # Timestamp when cooldown ends
    temp_paused: bool = False
    temp_pause_until: float = 0.0
    safety_level: DanceSafetyLevel = DanceSafetyLevel.STRICT
    violations: int = 0
    last_violation: str = ""


class DanceSafetyGovernor:
    """Safety governor specifically for dance moves.
    
    This sits between the dance engine and the servo driver. Every dance
    move must pass through this governor before reaching the servos.
    
    Checks performed:
    1. Session duration limit — enforce max session time
    2. Cooldown enforcement — block dancing during cooldown
    3. Move rate limiting — prevent too many moves per second
    4. Angle clamping — keep all servos within dance-safe range
    5. Stability check — ensure enough feet stay on ground
    6. Temperature monitoring — pause if servos too hot
    7. Hourly limit — prevent cumulative overheating
    
    Usage:
        gov = DanceSafetyGovernor()
        gov.start_session()
        
        angles = [90.0] * 18  # Some dance move
        safe_angles = gov.check_move(angles, feet_on_ground=4)
        if safe_angles is not None:
            await servo_driver.set_angles(safe_angles)
        else:
            # Move was blocked by safety
            pass
    """
    
    def __init__(self, level: DanceSafetyLevel = DanceSafetyLevel.STRICT):
        self.thresholds = self._get_thresholds(level)
        self.state = DanceSessionState(safety_level=level)
        self._servo_config = get_config().servo
        
        logger.info(f"🕺 Dance safety governor initialized (level: {level.value})")
    
    def _get_thresholds(self, level: DanceSafetyLevel) -> DanceThresholds:
        """Get thresholds appropriate for the safety level."""
        base = DanceThresholds()
        
        if level == DanceSafetyLevel.STRICT:
            # Kids/demo: very conservative
            base.coxa_min = 70.0
            base.coxa_max = 110.0
            base.femur_min = 65.0
            base.femur_max = 115.0
            base.tibia_min = 65.0
            base.tibia_max = 115.0
            base.max_servo_speed_deg_s = 30.0
            base.max_moves_per_second = 2.0
            base.max_session_seconds = 60.0  # 1 minute max
            base.cooldown_seconds = 60.0
            base.max_total_per_hour = 120.0  # 2 min/hour
            base.max_simultaneous_lift = 1
            base.max_height_change_per_beat = 10.0
            
        elif level == DanceSafetyLevel.NORMAL:
            # Regular dancing: moderate
            base.max_moves_per_second = 3.0
            base.max_session_seconds = 120.0
            base.cooldown_seconds = 30.0
            base.max_total_per_hour = 300.0
            
        elif level == DanceSafetyLevel.PERFORMANCE:
            # Stage: full range, longer sessions
            base.coxa_min = 50.0
            base.coxa_max = 130.0
            base.femur_min = 40.0
            base.femur_max = 140.0
            base.tibia_min = 40.0
            base.tibia_max = 140.0
            base.max_servo_speed_deg_s = 80.0
            base.max_moves_per_second = 5.0
            base.max_session_seconds = 300.0  # 5 minutes
            base.cooldown_seconds = 15.0
            base.max_total_per_hour = 600.0
            base.max_simultaneous_lift = 3
            
        elif level == DanceSafetyLevel.DISABLED:
            # No limits — DANGEROUS
            base.coxa_min = 0.0
            base.coxa_max = 180.0
            base.femur_min = 0.0
            base.femur_max = 180.0
            base.tibia_min = 0.0
            base.tibia_max = 180.0
            base.max_servo_speed_deg_s = 180.0
            base.max_moves_per_second = 100.0
            base.max_session_seconds = 999999.0
            base.cooldown_seconds = 0.0
            base.max_total_per_hour = 999999.0
            base.max_simultaneous_lift = 6
            base.max_height_change_per_beat = 999.0
        
        return base
    
    # ─── Session Management ──────────────────────────────────────────
    
    def start_session(self) -> bool:
        """Start a dance session. Returns True if allowed, False if blocked."""
        now = time.time()
        
        # Check cooldown
        if now < self.state.cooldown_until:
            remaining = self.state.cooldown_until - now
            self._violation(f"Cooldown active ({remaining:.0f}s remaining)")
            return False
        
        # Check hourly limit
        if self.state.total_dance_time_today >= self.thresholds.max_total_per_hour:
            self._violation("Hourly dance limit reached")
            return False
        
        self.state.active = True
        self.state.started_at = now
        self.state.move_count = 0
        self.state.last_move_time = 0.0  # Allow first move immediately
        logger.info(f"🕺 Dance session started (max: {self.thresholds.max_session_seconds:.0f}s)")
        return True
    
    def end_session(self) -> None:
        """End the current dance session and start cooldown if needed."""
        if not self.state.active:
            return
        
        now = time.time()
        self.state.duration = now - self.state.started_at
        self.state.total_dance_time_today += self.state.duration
        self.state.active = False
        
        # Start cooldown if session was long
        if self.state.duration >= self.thresholds.max_session_seconds:
            self.state.cooldown_until = now + self.thresholds.cooldown_seconds
            logger.info(
                f"🕺 Dance session ended ({self.state.duration:.0f}s). "
                f"Cooldown for {self.thresholds.cooldown_seconds:.0f}s."
            )
        else:
            logger.info(f"🕺 Dance session ended ({self.state.duration:.0f}s)")
    
    # ─── Move Checking ─────────────────────────────────────────────────
    
    def check_move(
        self,
        angles: list[float],
        feet_on_ground: int = 6,
        height_change: float = 0.0,
    ) -> list[float] | None:
        """Check if a dance move is safe to execute.
        
        Args:
            angles: List of 18 servo angles (0-180, 90=center)
            feet_on_ground: How many feet will remain on ground
            height_change: Body height change in mm (for bouncy moves)
        
        Returns:
            Clamped safe angles if move is allowed, None if blocked.
        """
        now = time.time()
        
        # 1. Session duration check
        if self.state.active:
            elapsed = now - self.state.started_at
            if elapsed >= self.thresholds.max_session_seconds:
                self._violation(f"Session too long ({elapsed:.0f}s)")
                self.end_session()
                return None
        
        # 2. Move rate limiting
        self._check_rate(now)
        if self.state.moves_this_second >= self.thresholds.max_moves_per_second:
            self._violation("Too many moves per second")
            return None
        
        # 3. Minimum move interval
        if (now - self.state.last_move_time) * 1000 < self.thresholds.min_move_interval_ms:
            # Too soon after last move — skip but don't count as violation
            return None
        
        # 4. Temperature check
        if self.state.temp_paused and now < self.state.temp_pause_until:
            self._violation("Temp pause active")
            return None
        
        # 5. Stability check
        if feet_on_ground < self.thresholds.min_feet_on_ground:
            self._violation(f"Only {feet_on_ground} feet on ground (min: {self.thresholds.min_feet_on_ground})")
            return None
        
        # 6. Height change check
        if abs(height_change) > self.thresholds.max_height_change_per_beat:
            self._violation(f"Height change too large ({height_change:.0f}mm)")
            return None
        
        # 7. Angle clamping — ensure all servos stay within dance-safe range
        safe_angles = self._clamp_angles(angles)
        
        # Update state
        self.state.move_count += 1
        self.state.last_move_time = now
        
        return safe_angles
    
    def _clamp_angles(self, angles: list[float]) -> list[float]:
        """Clamp all servo angles to the dance-safe range.
        
        For each leg (6 legs × 3 joints), the safe range depends on the
        joint type (coxa/femur/tibia).
        """
        safe = list(angles)
        
        for leg in range(6):
            # Channels for this leg: coxa, femur, tibia
            coxa_ch = leg * 3
            femur_ch = leg * 3 + 1
            tibia_ch = leg * 3 + 2
            
            # Clamp coxa (horizontal rotation)
            if coxa_ch < len(safe):
                safe[coxa_ch] = max(
                    self.thresholds.coxa_min,
                    min(self.thresholds.coxa_max, safe[coxa_ch])
                )
            
            # Clamp femur (vertical lift)
            if femur_ch < len(safe):
                safe[femur_ch] = max(
                    self.thresholds.femur_min,
                    min(self.thresholds.femur_max, safe[femur_ch])
                )
            
            # Clamp tibia (knee)
            if tibia_ch < len(safe):
                safe[tibia_ch] = max(
                    self.thresholds.tibia_min,
                    min(self.thresholds.tibia_max, safe[tibia_ch])
                )
        
        return safe
    
    def _check_rate(self, now: float) -> None:
        """Track moves per second for rate limiting."""
        if now - self.state.moves_this_second_window >= 1.0:
            # Reset the 1-second window
            self.state.moves_this_second = 0
            self.state.moves_this_second_window = now
    
    # ─── Temperature ───────────────────────────────────────────────────
    
    def check_temperature(self, temp_c: float) -> bool:
        """Check if temperature is safe for dancing.
        
        Args:
            temp_c: Current servo/CPU temperature
        
        Returns:
            True if safe to continue, False if must pause.
        """
        if temp_c >= self.thresholds.max_servo_temp_c:
            self.state.temp_paused = True
            self.state.temp_pause_until = time.time() + self.thresholds.temp_pause_seconds
            self._violation(f"Temperature too high: {temp_c:.0f}°C")
            logger.warning(
                f"🕺🔥 Dance temp pause: {temp_c:.0f}°C (max: {self.thresholds.max_servo_temp_c:.0f}°C). "
                f"Pausing for {self.thresholds.temp_pause_seconds:.0f}s."
            )
            return False
        
        # Clear temp pause if time elapsed
        if self.state.temp_paused and time.time() >= self.state.temp_pause_until:
            self.state.temp_paused = False
            logger.info("🕺 Temperature OK — resuming dance")
        
        return not self.state.temp_paused
    
    # ─── Status ────────────────────────────────────────────────────────
    
    def get_status(self) -> dict[str, Any]:
        """Get dance safety status for dashboard."""
        now = time.time()
        remaining_session = max(0, self.thresholds.max_session_seconds - (now - self.state.started_at)) if self.state.active else 0
        remaining_cooldown = max(0, self.state.cooldown_until - now) if not self.state.active else 0
        
        return {
            "active": self.state.active,
            "safety_level": self.state.safety_level.value,
            "session_duration": round(now - self.state.started_at, 1) if self.state.active else 0,
            "session_remaining": round(remaining_session, 1),
            "cooldown_remaining": round(remaining_cooldown, 1),
            "move_count": self.state.move_count,
            "moves_per_second": self.state.moves_this_second,
            "max_moves_per_second": self.thresholds.max_moves_per_second,
            "total_dance_today": round(self.state.total_dance_time_today, 1),
            "max_per_hour": self.thresholds.max_total_per_hour,
            "temp_paused": self.state.temp_paused,
            "violations": self.state.violations,
            "last_violation": self.state.last_violation,
            "angle_range": {
                "coxa": [self.thresholds.coxa_min, self.thresholds.coxa_max],
                "femur": [self.thresholds.femur_min, self.thresholds.femur_max],
                "tibia": [self.thresholds.tibia_min, self.thresholds.tibia_max],
            },
        }
    
    def _violation(self, reason: str) -> None:
        """Log a dance safety violation."""
        self.state.violations += 1
        self.state.last_violation = reason
        logger.warning(f"🕺⚠️ Dance safety: {reason}")
    
    # ─── Angle Helpers ─────────────────────────────────────────────────
    
    def get_safe_range(self, joint: str) -> tuple[float, float]:
        """Get the safe angle range for a joint type.
        
        Args:
            joint: "coxa", "femur", or "tibia"
        
        Returns:
            (min_angle, max_angle) in degrees.
        """
        if joint == "coxa":
            return self.thresholds.coxa_min, self.thresholds.coxa_max
        elif joint == "femur":
            return self.thresholds.femur_min, self.thresholds.femur_max
        elif joint == "tibia":
            return self.thresholds.tibia_min, self.thresholds.tibia_max
        return 0.0, 180.0
    
    def is_angle_safe(self, channel: int, angle: float) -> bool:
        """Check if a single angle is within the safe dance range.
        
        Args:
            channel: Servo channel (0-17)
            angle: Angle in degrees
        
        Returns:
            True if the angle is within safe range.
        """
        leg = channel // 3
        joint_idx = channel % 3
        
        if joint_idx == 0:  # coxa
            lo, hi = self.thresholds.coxa_min, self.thresholds.coxa_max
        elif joint_idx == 1:  # femur
            lo, hi = self.thresholds.femur_min, self.thresholds.femur_max
        else:  # tibia
            lo, hi = self.thresholds.tibia_min, self.thresholds.tibia_max
        
        return lo <= angle <= hi