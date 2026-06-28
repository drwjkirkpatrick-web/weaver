"""Tests for the safety governor — the most critical module."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from weaver.config import SafetyConfig, SafetyLevel
from weaver.event_bus import Event, EventBus, EventType
from weaver.safety import SafetyGovernor, SafetyState


@pytest.fixture
def safety_governor():
    """Create a safety governor with CHILD level (most restrictive)."""
    config = SafetyConfig(level=SafetyLevel.CHILD)
    gov = SafetyGovernor(config)
    gov.bus = EventBus()
    return gov


class TestSafetyGovernor:
    """Test the safety governor's motion checking logic."""
    
    def test_estop_blocks_all_motion(self, safety_governor):
        """E-stop should block all motion."""
        safety_governor.trigger_estop("test")
        
        allowed, speed, reason = safety_governor.check_motion("forward", 0.5)
        
        assert not allowed
        assert speed == 0.0
        assert "Emergency stop" in reason
    
    def test_low_battery_blocks_motion(self, safety_governor):
        """Low battery should block motion."""
        safety_governor.state.battery_voltage = 5.5  # Below 6.0V threshold
        
        allowed, speed, reason = safety_governor.check_motion("forward", 0.5)
        
        assert not allowed
        assert "Battery" in reason
    
    def test_high_temp_blocks_motion(self, safety_governor):
        """High CPU temperature should block motion."""
        safety_governor.state.cpu_temp_c = 85.0  # Above 70°C threshold
        
        allowed, speed, reason = safety_governor.check_motion("forward", 0.5)
        
        assert not allowed
        assert "CPU" in reason or "hot" in reason.lower()
    
    def test_excessive_tilt_blocks_motion(self, safety_governor):
        """Excessive body tilt should block motion."""
        safety_governor.state.body_tilt_degrees = 45.0  # Above 30° threshold
        
        allowed, speed, reason = safety_governor.check_motion("forward", 0.5)
        
        assert not allowed
        assert "tilt" in reason.lower()
    
    def test_obstacle_blocks_forward_motion(self, safety_governor):
        """Obstacle closer than stop distance should block forward motion."""
        safety_governor.state.obstacle_distance_cm = 20.0  # Below 30cm threshold
        
        allowed, speed, reason = safety_governor.check_motion("forward", 0.5)
        
        assert not allowed
        assert "Obstacle" in reason
    
    def test_obstacle_doesnt_block_backward_motion(self, safety_governor):
        """Obstacle should NOT block backward motion (retreating)."""
        safety_governor.state.obstacle_distance_cm = 20.0
        
        allowed, speed, reason = safety_governor.check_motion("backward", 0.5)
        
        assert allowed  # Backward is fine even with obstacle ahead
    
    def test_obstacle_slows_motion(self, safety_governor):
        """Obstacle in warning zone should slow down but not stop."""
        safety_governor.state.obstacle_distance_cm = 45.0  # Between 30-60cm
        
        allowed, speed, reason = safety_governor.check_motion("forward", 0.5)
        
        assert allowed
        assert speed < 0.5  # Speed should be reduced
        assert reason == ""
    
    def test_speed_cap_child_level(self, safety_governor):
        """CHILD safety level should cap speed at 0.3."""
        allowed, speed, reason = safety_governor.check_motion("forward", 1.0)
        
        assert allowed
        assert speed <= 0.3
    
    def test_speed_cap_adult_level(self):
        """ADULT safety level should cap speed at 0.6."""
        config = SafetyConfig(level=SafetyLevel.ADULT)
        gov = SafetyGovernor(config)
        gov.bus = EventBus()
        
        allowed, speed, reason = gov.check_motion("forward", 1.0)
        
        assert allowed
        assert speed <= 0.6
    
    def test_speed_cap_expert_level(self):
        """EXPERT safety level should cap speed at 0.8."""
        config = SafetyConfig(level=SafetyLevel.EXPERT)
        gov = SafetyGovernor(config)
        gov.bus = EventBus()
        
        allowed, speed, reason = gov.check_motion("forward", 1.0)
        
        assert allowed
        assert speed <= 0.8
    
    def test_servo_angle_range_check_blocks(self, safety_governor):
        """Servo angles outside safe range should be blocked."""
        # Angle of 90° exceeds the default max of 45°
        angles = [0.0] * 18
        angles[5] = 90.0  # Servo 5 way too far
        
        allowed, speed, reason = safety_governor.check_motion(
            "forward", 0.3, servo_angles=angles
        )
        
        assert not allowed
        assert "Servo" in reason
    
    def test_human_nearby_reduces_speed(self, safety_governor):
        """Human detection should reduce speed."""
        safety_governor.config.human_detection_enabled = True
        safety_governor.state.human_distance_cm = 50.0  # Within 100cm safe distance
        
        allowed, speed, reason = safety_governor.check_motion("forward", 0.3)
        
        assert allowed
        assert speed < 0.3  # Should be reduced
    
    def test_clear_estop(self, safety_governor):
        """Clearing e-stop should re-enable motion."""
        safety_governor.trigger_estop("test")
        assert not safety_governor.state.motion_allowed
        
        safety_governor.clear_estop()
        
        allowed, _, _ = safety_governor.check_motion("forward", 0.3)
        assert allowed
    
    @pytest.mark.asyncio
    async def test_ultrasonic_event_updates_state(self, safety_governor):
        """Ultrasonic range event should update obstacle distance."""
        await safety_governor._on_ultrasonic(Event(
            type=EventType.ULTRASONIC_RANGE,
            data={"distance_cm": 75.0},
        ))
        
        assert safety_governor.state.obstacle_distance_cm == 75.0
    
    @pytest.mark.asyncio
    async def test_face_detected_updates_human_distance(self, safety_governor):
        """Face detection event should update human distance."""
        safety_governor.config.human_detection_enabled = True
        
        await safety_governor._on_face_detected(Event(
            type=EventType.CAMERA_FACE_DETECTED,
            data={"distance_cm": 120.0},
        ))
        
        assert safety_governor.state.human_distance_cm == 120.0
    
    @pytest.mark.asyncio
    async def test_battery_critical_stops_motion(self, safety_governor):
        """Battery critical event should stop motion."""
        await safety_governor._on_battery_critical(Event(
            type=EventType.BATTERY_CRITICAL,
            data={"voltage": 5.8},
        ))
        
        assert not safety_governor.state.motion_allowed
        assert safety_governor.state.battery_voltage == 5.8