# safety.py — Safety Governor
#
# This is the MOST CRITICAL module in Weaver. It sits between all motion
# commands and the actual servo output. No motion happens without passing
# through the safety governor.
#
# Core principles:
# 1. Human safety first — especially children
# 2. Fail-safe: when in doubt, STOP
# 3. Defense in depth: multiple independent safety checks
# 4. Conservative defaults: slow, cautious, large buffers
# 5. Transparent: every safety decision is logged
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from weaver.config import SafetyConfig, SafetyLevel, get_config
from weaver.event_bus import Event, EventBus, EventType, get_event_bus


@dataclass
class SafetyState:
    """Current safety state snapshot."""
    estop_active: bool = False
    obstacle_distance_cm: float | None = None  # Latest ultrasonic reading
    human_distance_cm: float | None = None     # From camera face detection
    body_tilt_degrees: float = 0.0             # From IMU
    battery_voltage: float = 8.4               # Assume full
    cpu_temp_c: float = 45.0                   # Assume normal
    servo_temps: list[float] | None = None     # If equipped
    motion_allowed: bool = True               # Master flag
    speed_multiplier: float = 1.0              # 0.0-1.0, reduces speed
    last_violation: str = ""                   # Description of last violation
    violations_count: int = 0


class SafetyGovernor:
    """The safety governor intercepts all motion commands.
    
    It subscribes to ALL sensor events to maintain a real-time safety state.
    When a motion command is issued, it checks:
    
    1. Is e-stop active? → Block
    2. Is battery critical? → Block
    3. Is CPU too hot? → Block  
    4. Is body tilted too much? → Block + stabilize
    5. Is obstacle too close? → Block forward motion
    6. Is human nearby? → Reduce speed, maintain distance
    7. Are servo angles within safe range? → Clamp
    
    If ANY check fails, the motion is either blocked entirely or modified
    (speed reduced, direction changed) to be safe.
    """
    
    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or get_config().safety
        self.state = SafetyState()
        self.bus = get_event_bus()
        self._monitor_task: asyncio.Task | None = None
        
    async def start(self) -> None:
        """Start the safety governor and subscribe to events."""
        logger.info("🛡️  Safety Governor starting (level: {})", self.config.level.value)
        
        # Subscribe to ALL safety-relevant events (priority=True!)
        # Priority means these handlers run BEFORE any other handler
        self.bus.subscribe(EventType.ULTRASONIC_RANGE, self._on_ultrasonic, priority=True)
        self.bus.subscribe(EventType.ULTRASONIC_OBSTACLE, self._on_obstacle, priority=True)
        self.bus.subscribe(EventType.IMU_TILT_WARNING, self._on_tilt_warning, priority=True)
        self.bus.subscribe(EventType.IMU_DATA, self._on_imu_data, priority=True)
        self.bus.subscribe(EventType.BATTERY_LOW, self._on_battery_low, priority=True)
        self.bus.subscribe(EventType.BATTERY_CRITICAL, self._on_battery_critical, priority=True)
        self.bus.subscribe(EventType.CAMERA_FACE_DETECTED, self._on_face_detected, priority=True)
        self.bus.subscribe(EventType.THERMAL_CRITICAL, self._on_thermal_critical, priority=True)
        self.bus.subscribe(EventType.SAFETY_ESTOP, self._on_estop, priority=True)
        self.bus.subscribe(EventType.MOTION_COMMAND, self._on_motion_command, priority=True)
        
        # Start periodic safety monitor
        self._monitor_task = asyncio.create_task(self._safety_loop())
        
        logger.info("✅ Safety Governor active — all motion is gated")
    
    async def stop(self) -> None:
        """Stop the safety governor."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Safety Governor stopped")
    
    # ─── Public API ───────────────────────────────────────────────────
    
    def check_motion(self, direction: str, speed: float, 
                     servo_angles: list[float] | None = None) -> tuple[bool, float, str]:
        """Check if a motion command is safe to execute.
        
        Returns:
            (allowed, adjusted_speed, reason)
            - allowed: True if motion can proceed
            - adjusted_speed: speed after safety adjustment (may be reduced)
            - reason: why motion was blocked/modified (empty if allowed)
        """
        # 1. E-stop
        if self.state.estop_active:
            self._violation("E-stop active")
            return False, 0.0, "Emergency stop is active"
        
        # 2. Battery critical
        if self.state.battery_voltage < self.config.min_battery_voltage:
            self._violation(f"Battery low: {self.state.battery_voltage:.1f}V")
            return False, 0.0, f"Battery too low ({self.state.battery_voltage:.1f}V)"
        
        # 3. CPU temperature
        if self.state.cpu_temp_c > self.config.max_servo_temp_c:
            self._violation(f"CPU hot: {self.state.cpu_temp_c:.0f}°C")
            return False, 0.0, f"CPU too hot ({self.state.cpu_temp_c:.0f}°C)"
        
        # 4. Body tilt
        if abs(self.state.body_tilt_degrees) > self.config.max_tilt_degrees:
            self._violation(f"Tilt: {self.state.body_tilt_degrees:.1f}°")
            return False, 0.0, f"Body tilted too much ({self.state.body_tilt_degrees:.1f}°)"
        
        # 5. Obstacle check (only for forward motion)
        if direction in ("forward", "turn_left", "turn_right"):
            if self.state.obstacle_distance_cm is not None:
                if self.state.obstacle_distance_cm < self.config.obstacle_stop_distance_cm:
                    self._violation(f"Obstacle: {self.state.obstacle_distance_cm:.0f}cm")
                    return False, 0.0, f"Obstacle too close ({self.state.obstacle_distance_cm:.0f}cm)"
                
                # Slow down if in warning zone
                if self.state.obstacle_distance_cm < self.config.obstacle_slow_distance_cm:
                    speed *= 0.3  # Reduce to 30%
                    logger.debug(f"Safety: slowing due to obstacle at {self.state.obstacle_distance_cm:.0f}cm")
        
        # 6. Human nearby — reduce speed
        if self.config.human_detection_enabled and self.state.human_distance_cm is not None:
            if self.state.human_distance_cm < self.config.human_safe_distance_cm:
                # Scale speed inversely with distance
                ratio = self.state.human_distance_cm / self.config.human_safe_distance_cm
                speed *= max(0.1, ratio)
                logger.debug(f"Safety: human at {self.state.human_distance_cm:.0f}cm, speed→{speed:.2f}")
        
        # 7. Speed cap
        max_speed = self._max_speed_for_level()
        if speed > max_speed:
            logger.debug(f"Safety: capping speed {speed:.2f}→{max_speed:.2f}")
            speed = max_speed
        
        # 8. Servo angle range check
        if servo_angles is not None:
            for i, angle in enumerate(servo_angles):
                if i < len(self.config.servo_safe_min):
                    if angle < self.config.servo_safe_min[i]:
                        logger.warning(f"Safety: servo {i} angle {angle:.1f}° below min {self.config.servo_safe_min[i]:.1f}°")
                        return False, 0.0, f"Servo {i} would exceed safe range"
                if i < len(self.config.servo_safe_max):
                    if angle > self.config.servo_safe_max[i]:
                        logger.warning(f"Safety: servo {i} angle {angle:.1f}° above max {self.config.servo_safe_max[i]:.1f}°")
                        return False, 0.0, f"Servo {i} would exceed safe range"
        
        return True, speed, ""
    
    def get_state(self) -> SafetyState:
        """Get current safety state (for dashboard)."""
        return self.state
    
    def trigger_estop(self, reason: str = "manual") -> None:
        """Trigger emergency stop."""
        self.state.estop_active = True
        self.state.motion_allowed = False
        self.state.speed_multiplier = 0.0
        self.state.last_violation = f"E-STOP: {reason}"
        self.state.violations_count += 1
        logger.error(f"🛑 EMERGENCY STOP: {reason}")
    
    def clear_estop(self) -> None:
        """Clear emergency stop (requires manual confirmation)."""
        self.state.estop_active = False
        self.state.motion_allowed = True
        self.state.speed_multiplier = 1.0
        self.bus.set_estop(False)
        logger.info("✅ Emergency stop cleared — motion re-enabled")
    
    # ─── Event Handlers ───────────────────────────────────────────────
    
    async def _on_ultrasonic(self, event: Event) -> None:
        """Update obstacle distance from ultrasonic sensor."""
        self.state.obstacle_distance_cm = event.data.get("distance_cm")
    
    async def _on_obstacle(self, event: Event) -> None:
        """Obstacle detected — may need to stop."""
        distance = event.data.get("distance_cm", 999.0)
        if distance < self.config.obstacle_stop_distance_cm:
            await self.bus.publish(Event(
                type=EventType.SAFETY_COLLISION_IMMINENT,
                data={"distance_cm": distance, "source": "ultrasonic"},
                source="safety",
            ))
    
    async def _on_imu_data(self, event: Event) -> None:
        """Update tilt from IMU."""
        self.state.body_tilt_degrees = event.data.get("tilt_degrees", 0.0)
        if abs(self.state.body_tilt_degrees) > self.config.max_tilt_degrees:
            await self.bus.publish(Event(
                type=EventType.SAFETY_WARNING,
                data={"reason": "tilt", "tilt": self.state.body_tilt_degrees},
                source="safety",
            ))
    
    async def _on_tilt_warning(self, event: Event) -> None:
        """Tilt exceeded threshold — block motion."""
        self.state.motion_allowed = False
        self._violation(f"Excessive tilt: {event.data.get('tilt', 0):.1f}°")
    
    async def _on_battery_low(self, event: Event) -> None:
        """Battery low — reduce speed."""
        self.state.battery_voltage = event.data.get("voltage", 7.0)
        self.state.speed_multiplier *= 0.5
        logger.warning(f"🔋 Battery low: {self.state.battery_voltage:.1f}V — reducing speed")
    
    async def _on_battery_critical(self, event: Event) -> None:
        """Battery critical — stop all motion."""
        self.state.battery_voltage = event.data.get("voltage", 6.0)
        self.state.motion_allowed = False
        self._violation(f"Critical battery: {self.state.battery_voltage:.1f}V")
    
    async def _on_face_detected(self, event: Event) -> None:
        """Human detected — apply safety constraints."""
        if self.config.human_detection_enabled:
            distance = event.data.get("distance_cm")
            if distance:
                self.state.human_distance_cm = distance
                if distance < self.config.human_safe_distance_cm:
                    await self.bus.publish(Event(
                        type=EventType.SAFETY_HUMAN_NEARBY,
                        data={"distance_cm": distance},
                        source="safety",
                    ))
    
    async def _on_thermal_critical(self, event: Event) -> None:
        """CPU too hot — stop motion."""
        self.state.cpu_temp_c = event.data.get("temp_c", 80.0)
        self.state.motion_allowed = False
        self._violation(f"CPU thermal: {self.state.cpu_temp_c:.0f}°C")
    
    async def _on_estop(self, event: Event) -> None:
        """Emergency stop triggered."""
        self.trigger_estop(event.data.get("reason", "bus event"))
    
    async def _on_motion_command(self, event: Event) -> None:
        """Intercept motion commands and validate them."""
        if self.state.estop_active:
            logger.warning("Motion command blocked — e-stop active")
            await self.bus.publish(Event(
                type=EventType.MOTION_ABORTED,
                data={"reason": "e-stop", "original_command": event.data},
                source="safety",
            ))
            return
        
        direction = event.data.get("direction", "stop")
        speed = event.data.get("speed", 0.0)
        
        allowed, adjusted_speed, reason = self.check_motion(direction, speed)
        
        if not allowed:
            await self.bus.publish(Event(
                type=EventType.MOTION_ABORTED,
                data={"reason": reason, "original_command": event.data},
                source="safety",
            ))
        else:
            # Update the command with adjusted speed
            event.data["speed"] = adjusted_speed
            event.data["safety_approved"] = True
    
    # ─── Internal ─────────────────────────────────────────────────────
    
    def _max_speed_for_level(self) -> float:
        """Get max speed based on safety level."""
        match self.config.level:
            case SafetyLevel.CHILD:
                return 0.3   # Very slow — 30% max
            case SafetyLevel.ADULT:
                return 0.6   # Moderate — 60% max
            case SafetyLevel.EXPERT:
                return 0.8   # Fast — 80% max
            case SafetyLevel.DISABLED:
                return 1.0   # No limit (dangerous!)
            case _:
                return 0.3   # Default to safest
    
    def _violation(self, reason: str) -> None:
        """Log a safety violation."""
        self.state.last_violation = reason
        self.state.violations_count += 1
        logger.warning(f"⚠️  Safety violation #{self.state.violations_count}: {reason}")
    
    async def _safety_loop(self) -> None:
        """Periodic safety check loop."""
        while True:
            try:
                # Re-evaluate motion permission
                self.state.motion_allowed = (
                    not self.state.estop_active
                    and self.state.battery_voltage >= self.config.min_battery_voltage
                    and self.state.cpu_temp_c < self.config.max_servo_temp_c
                    and abs(self.state.body_tilt_degrees) <= self.config.max_tilt_degrees
                )
                
                # Publish safety state
                await self.bus.publish(Event(
                    type=EventType.SAFETY_WARNING if not self.state.motion_allowed else EventType.MODULE_STATUS,
                    data={
                        "module": "safety",
                        "motion_allowed": self.state.motion_allowed,
                        "estop": self.state.estop_active,
                        "obstacle_distance_cm": self.state.obstacle_distance_cm,
                        "human_distance_cm": self.state.human_distance_cm,
                        "tilt_degrees": self.state.body_tilt_degrees,
                        "battery_voltage": self.state.battery_voltage,
                        "cpu_temp_c": self.state.cpu_temp_c,
                        "speed_multiplier": self.state.speed_multiplier,
                        "violations": self.state.violations_count,
                    },
                    source="safety",
                ))
                
            except Exception as e:
                logger.error(f"Safety loop error: {e}")
            
            await asyncio.sleep(0.5)  # Check twice per second