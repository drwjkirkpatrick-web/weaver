"""Tests for the cortex — LLM decision-making brain.

Tests cover:
- World model updates from sensor events
- Rule-based safety fallback (obstacle, battery, tilt, e-stop)
- Voice command processing
- LLM decision parsing
- Context window management
- Event publishing (CORTEX_DECISION, CORTEX_THINKING, etc.)
- Stats reporting
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weaver.brain.cortex import Cortex, CortexState, WorldModel
from weaver.config import CortexConfig, HardwareMode, LLMMode, get_config
from weaver.event_bus import Event, EventBus, EventType


# ─── WorldModel Tests ────────────────────────────────────────────────────


class TestWorldModel:
    """Tests for the WorldModel dataclass."""

    def test_default_world_model(self):
        """World model should have safe defaults."""
        wm = WorldModel()
        assert wm.obstacle_distance_cm is None
        assert wm.battery_voltage == 8.4
        assert wm.last_motion == "stop"
        assert wm.current_speed == 0.0

    def test_to_llm_context_contains_obstacle(self):
        """LLM context should include obstacle distance."""
        wm = WorldModel(obstacle_distance_cm=50.0)
        ctx = wm.to_llm_context()
        assert "50" in ctx
        assert "Obstacle" in ctx

    def test_to_llm_context_contains_tilt(self):
        """LLM context should include tilt angle."""
        wm = WorldModel(body_tilt_degrees=15.0)
        ctx = wm.to_llm_context()
        assert "15.0" in ctx
        assert "Tilt" in ctx

    def test_to_llm_context_contains_battery(self):
        """LLM context should include battery voltage."""
        wm = WorldModel(battery_voltage=7.2, battery_low=True)
        ctx = wm.to_llm_context()
        assert "7.2" in ctx
        assert "LOW" in ctx

    def test_to_llm_context_contains_human(self):
        """LLM context should mention detected humans."""
        wm = WorldModel(human_detected=True, human_distance_cm=80.0)
        ctx = wm.to_llm_context()
        assert "Human" in ctx
        assert "80" in ctx

    def test_to_llm_context_no_human(self):
        """LLM context should say no humans when none detected."""
        wm = WorldModel(human_detected=False)
        ctx = wm.to_llm_context()
        assert "No humans" in ctx

    def test_to_llm_context_contains_motion(self):
        """LLM context should include current motion state."""
        wm = WorldModel(last_motion="forward", current_speed=0.5)
        ctx = wm.to_llm_context()
        assert "forward" in ctx


# ─── Cortex Instantiation ─────────────────────────────────────────────────


def test_cortex_instantiation():
    """Cortex should instantiate with default config."""
    cortex = Cortex()
    assert cortex.config is not None
    assert cortex.state == CortexState.WAITING
    assert cortex.world_model is not None
    assert cortex.llm_bridge is None  # Not initialized until start()


def test_cortex_state_enum():
    """CortexState should have all expected states."""
    assert CortexState.THINKING
    assert CortexState.WAITING
    assert CortexState.ACTING
    assert CortexState.FALLBACK
    assert CortexState.ERROR


# ─── Sensor Event Handler Tests ──────────────────────────────────────────


class TestCortexSensorHandlers:
    """Tests for sensor event handlers updating the world model."""

    @pytest.fixture
    def cortex(self):
        """Cortex with a fresh event bus (no LLM, no decision loop)."""
        # Patch LLM bridge init to avoid network calls
        with patch.object(Cortex, '_init_llm_bridge', new_callable=AsyncMock):
            cortex = Cortex()
            cortex.bus = EventBus()
            cortex.intent_parser = MagicMock()
            cortex.intent_parser.start = AsyncMock()
            cortex.intent_parser.stop = AsyncMock()
            cortex.intent_parser.parse = AsyncMock(
                return_value={"direction": "stop", "confidence": 0.9, "raw_text": "test"}
            )
            return cortex

    @pytest.mark.asyncio
    async def test_ultrasonic_range_updates_obstacle(self, cortex):
        """ULTRASONIC_RANGE event should update obstacle_distance_cm."""
        await cortex._on_ultrasonic_range(Event(
            type=EventType.ULTRASONIC_RANGE,
            data={"distance_cm": 45.0},
        ))
        assert cortex.world_model.obstacle_distance_cm == 45.0

    @pytest.mark.asyncio
    async def test_ultrasonic_range_sets_warning(self, cortex):
        """Obstacle within slow distance should set warning flag."""
        slow_dist = cortex.main_config.safety.obstacle_slow_distance_cm
        await cortex._on_ultrasonic_range(Event(
            type=EventType.ULTRASONIC_RANGE,
            data={"distance_cm": slow_dist - 5},
        ))
        assert cortex.world_model.obstacle_warning is True

    @pytest.mark.asyncio
    async def test_imu_data_updates_tilt(self, cortex):
        """IMU_DATA event should update body_tilt_degrees."""
        await cortex._on_imu_data(Event(
            type=EventType.IMU_DATA,
            data={"tilt_degrees": 12.0},
        ))
        assert cortex.world_model.body_tilt_degrees == 12.0

    @pytest.mark.asyncio
    async def test_battery_status_updates_voltage(self, cortex):
        """BATTERY_STATUS event should update battery_voltage."""
        await cortex._on_battery_status(Event(
            type=EventType.BATTERY_STATUS,
            data={"voltage": 7.1},
        ))
        assert cortex.world_model.battery_voltage == 7.1

    @pytest.mark.asyncio
    async def test_battery_low_sets_flag(self, cortex):
        """BATTERY_LOW event should set battery_low flag."""
        await cortex._on_battery_low(Event(
            type=EventType.BATTERY_LOW,
            data={"voltage": 6.5},
        ))
        assert cortex.world_model.battery_low is True

    @pytest.mark.asyncio
    async def test_thermal_status_updates_temp(self, cortex):
        """THERMAL_STATUS event should update cpu_temp_c."""
        await cortex._on_thermal_status(Event(
            type=EventType.THERMAL_STATUS,
            data={"temp_c": 65.0},
        ))
        assert cortex.world_model.cpu_temp_c == 65.0

    @pytest.mark.asyncio
    async def test_face_detected_sets_human(self, cortex):
        """CAMERA_FACE_DETECTED event should set human_detected."""
        await cortex._on_face_detected(Event(
            type=EventType.CAMERA_FACE_DETECTED,
            data={"distance_cm": 90.0},
        ))
        assert cortex.world_model.human_detected is True
        assert cortex.world_model.human_distance_cm == 90.0

    @pytest.mark.asyncio
    async def test_object_detected_adds_to_list(self, cortex):
        """CAMERA_OBJECT_DETECTED event should add object to list."""
        await cortex._on_object_detected(Event(
            type=EventType.CAMERA_OBJECT_DETECTED,
            data={"object": "chair"},
        ))
        assert "chair" in cortex.world_model.detected_objects

    @pytest.mark.asyncio
    async def test_voice_transcript_queues_command(self, cortex):
        """VOICE_TRANSCRIPT event should queue the command."""
        await cortex._on_voice_transcript(Event(
            type=EventType.VOICE_TRANSCRIPT,
            data={"text": "walk forward"},
        ))
        assert cortex._pending_voice_command == "walk forward"

    @pytest.mark.asyncio
    async def test_estop_stops_motion(self, cortex):
        """SAFETY_ESTOP event should stop motion in world model."""
        cortex.world_model.last_motion = "forward"
        cortex.world_model.current_speed = 0.5
        await cortex._on_safety_estop(Event(
            type=EventType.SAFETY_ESTOP,
            data={"reason": "test"},
        ))
        assert cortex.world_model.last_motion == "stop"
        assert cortex.world_model.current_speed == 0.0

    @pytest.mark.asyncio
    async def test_motion_completed_resets_state(self, cortex):
        """MOTION_COMPLETED event should reset motion state."""
        cortex.world_model.last_motion = "forward"
        cortex.world_model.current_speed = 0.5
        await cortex._on_motion_completed(Event(
            type=EventType.MOTION_COMPLETED,
        ))
        assert cortex.world_model.last_motion == "stop"
        assert cortex.world_model.current_speed == 0.0


# ─── Safety Rule Tests ───────────────────────────────────────────────────


class TestCortexSafetyRules:
    """Tests for the rule-based safety checks in the cortex."""

    @pytest.fixture
    def cortex(self):
        """Cortex with a fresh event bus, no LLM."""
        with patch.object(Cortex, '_init_llm_bridge', new_callable=AsyncMock):
            cortex = Cortex()
            cortex.bus = EventBus()
            cortex.bus.set_estop(False)
            cortex.intent_parser = MagicMock()
            cortex.intent_parser.start = AsyncMock()
            cortex.intent_parser.stop = AsyncMock()
            cortex.intent_parser.parse = AsyncMock(
                return_value={"direction": "stop", "confidence": 0.9}
            )
            return cortex

    @pytest.mark.asyncio
    async def test_obstacle_too_close_returns_stop(self, cortex):
        """Obstacle within stop distance should trigger safety stop."""
        cortex.world_model.obstacle_distance_cm = 15.0  # Very close
        action = cortex._check_safety_rules()
        assert action is not None
        assert action["direction"] == "stop"

    @pytest.mark.asyncio
    async def test_obstacle_far_returns_none(self, cortex):
        """No obstacle (or far obstacle) should not trigger safety stop."""
        cortex.world_model.obstacle_distance_cm = 200.0
        action = cortex._check_safety_rules()
        assert action is None

    @pytest.mark.asyncio
    async def test_estop_returns_stop(self, cortex):
        """E-stop active should trigger safety stop."""
        cortex.bus.set_estop(True)
        action = cortex._check_safety_rules()
        assert action is not None
        assert action["direction"] == "stop"
        assert "estop" in action.get("reason", "")

    @pytest.mark.asyncio
    async def test_tilt_warning_returns_stop(self, cortex):
        """Tilt warning should trigger safety stop."""
        cortex.world_model.tilt_warning = True
        action = cortex._check_safety_rules()
        assert action is not None
        assert action["direction"] == "stop"
        assert "tilt" in action.get("reason", "")

    @pytest.mark.asyncio
    async def test_battery_critical_returns_stop(self, cortex):
        """Critical battery should trigger safety stop."""
        cortex.world_model.battery_voltage = 5.5  # Below min
        action = cortex._check_safety_rules()
        assert action is not None
        assert action["direction"] == "stop"
        assert "battery" in action.get("reason", "")

    @pytest.mark.asyncio
    async def test_human_very_close_returns_stop(self, cortex):
        """Human within 30cm should trigger safety stop."""
        cortex.world_model.human_detected = True
        cortex.world_model.human_distance_cm = 20.0
        action = cortex._check_safety_rules()
        assert action is not None
        assert action["direction"] == "stop"
        assert "human" in action.get("reason", "")


# ─── LLM Decision Parsing Tests ──────────────────────────────────────────


class TestCortexLLMParsing:
    """Tests for parsing LLM responses into motion commands."""

    @pytest.fixture
    def cortex(self):
        with patch.object(Cortex, '_init_llm_bridge', new_callable=AsyncMock):
            cortex = Cortex()
            cortex.bus = EventBus()
            cortex.intent_parser = MagicMock()
            cortex.intent_parser.start = AsyncMock()
            cortex.intent_parser.stop = AsyncMock()
            return cortex

    def test_parse_valid_direction(self, cortex):
        """Valid direction JSON should parse correctly."""
        response = '{"direction": "forward", "speed": 0.5}'
        result = cortex._parse_llm_decision(response)
        assert result is not None
        assert result["direction"] == "forward"
        assert result["speed"] == 0.5

    def test_parse_valid_action(self, cortex):
        """Valid action JSON should parse correctly."""
        response = '{"action": "wave"}'
        result = cortex._parse_llm_decision(response)
        assert result is not None
        assert result["action"] == "wave"

    def test_parse_json_with_extra_text(self, cortex):
        """JSON embedded in text should be extracted."""
        response = 'I think you should do this: {"direction": "stop", "speed": 0.0}'
        result = cortex._parse_llm_decision(response)
        assert result is not None
        assert result["direction"] == "stop"

    def test_parse_invalid_json_returns_none(self, cortex):
        """Non-JSON response should return None."""
        response = "I cannot decide right now."
        result = cortex._parse_llm_decision(response)
        assert result is None

    def test_parse_invalid_direction_returns_none(self, cortex):
        """Invalid direction value should return None."""
        response = '{"direction": "fly", "speed": 1.0}'
        result = cortex._parse_llm_decision(response)
        assert result is None

    def test_validate_valid_direction(self, cortex):
        """Valid directions should pass validation."""
        for d in ["forward", "backward", "turn_left", "turn_right",
                   "strafe_left", "strafe_right", "stop"]:
            assert cortex._validate_action({"direction": d}) is True

    def test_validate_valid_action(self, cortex):
        """Valid actions should pass validation."""
        for a in ["sit", "stand", "wave", "crouch", "dance",
                   "gait_change", "body_height"]:
            assert cortex._validate_action({"action": a}) is True

    def test_validate_invalid_returns_false(self, cortex):
        """Invalid actions should fail validation."""
        assert cortex._validate_action({"direction": "teleport"}) is False
        assert cortex._validate_action({"action": "fly"}) is False
        assert cortex._validate_action({"foo": "bar"}) is False


# ─── Context Window Tests ────────────────────────────────────────────────


class TestContextWindow:
    """Tests for the rolling context window."""

    @pytest.fixture
    def cortex(self):
        with patch.object(Cortex, '_init_llm_bridge', new_callable=AsyncMock):
            cortex = Cortex()
            cortex.bus = EventBus()
            cortex.intent_parser = MagicMock()
            cortex.intent_parser.start = AsyncMock()
            cortex.intent_parser.stop = AsyncMock()
            return cortex

    def test_context_window_default_size(self, cortex):
        """Context window should have the configured max size."""
        assert cortex.context_window.maxlen == cortex.config.max_context_messages

    def test_add_to_context(self, cortex):
        """Adding to context should increase its size."""
        cortex._add_to_context({"test": "data"})
        assert len(cortex.context_window) == 1

    def test_context_window_rolls_over(self, cortex):
        """Context window should roll over when full."""
        max_size = cortex.config.max_context_messages
        for i in range(max_size + 5):
            cortex._add_to_context({"index": i})
        assert len(cortex.context_window) == max_size
        # Should keep the most recent entries
        last = cortex.context_window[-1]
        assert last["index"] == max_size + 4


# ─── Action Execution Tests ──────────────────────────────────────────────


class TestActionExecution:
    """Tests for action execution and event publishing."""

    @pytest.fixture
    def cortex(self):
        with patch.object(Cortex, '_init_llm_bridge', new_callable=AsyncMock):
            cortex = Cortex()
            cortex.bus = EventBus()
            cortex.intent_parser = MagicMock()
            cortex.intent_parser.start = AsyncMock()
            cortex.intent_parser.stop = AsyncMock()
            cortex.intent_parser.parse = AsyncMock(
                return_value={"direction": "forward", "speed": 0.5, "confidence": 0.9}
            )
            return cortex

    @pytest.mark.asyncio
    async def test_execute_action_publishes_motion_command(self, cortex):
        """Executing an action should publish a MOTION_COMMAND event."""
        received = []
        async def capture(event: Event):
            received.append(event)
        cortex.bus.subscribe(EventType.MOTION_COMMAND, capture)

        await cortex._execute_action({"direction": "forward", "speed": 0.5})
        assert len(received) == 1
        assert received[0].data["direction"] == "forward"
        assert received[0].data["speed"] == 0.5

    @pytest.mark.asyncio
    async def test_execute_action_publishes_decision(self, cortex):
        """Executing an action should publish a CORTEX_DECISION event."""
        received = []
        async def capture(event: Event):
            received.append(event)
        cortex.bus.subscribe(EventType.CORTEX_DECISION, capture)

        await cortex._execute_action({"direction": "stop", "speed": 0.0})
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_execute_action_updates_world_model(self, cortex):
        """Executing a direction action should update the world model."""
        await cortex._execute_action({"direction": "forward", "speed": 0.5})
        assert cortex.world_model.last_motion == "forward"
        assert cortex.world_model.current_speed == 0.5

    @pytest.mark.asyncio
    async def test_execute_action_updates_state(self, cortex):
        """Executing an action should set state to ACTING."""
        await cortex._execute_action({"direction": "stop", "speed": 0.0})
        assert cortex.state == CortexState.ACTING


# ─── Voice Command Tests ─────────────────────────────────────────────────


class TestVoiceCommandProcessing:
    """Tests for voice command processing through the cortex."""

    @pytest.fixture
    def cortex(self):
        with patch.object(Cortex, '_init_llm_bridge', new_callable=AsyncMock):
            cortex = Cortex()
            cortex.bus = EventBus()
            cortex.intent_parser = MagicMock()
            cortex.intent_parser.start = AsyncMock()
            cortex.intent_parser.stop = AsyncMock()
            cortex.intent_parser.parse = AsyncMock(
                return_value={"direction": "forward", "speed": 0.5, "confidence": 0.9}
            )
            return cortex

    @pytest.mark.asyncio
    async def test_voice_command_triggers_action(self, cortex):
        """A voice command should be parsed and executed."""
        received = []
        async def capture(event: Event):
            received.append(event)
        cortex.bus.subscribe(EventType.MOTION_COMMAND, capture)

        await cortex._process_voice_command("walk forward")
        assert len(received) == 1
        assert received[0].data["direction"] == "forward"
        assert received[0].data["source"] == "voice"

    @pytest.mark.asyncio
    async def test_voice_command_low_confidence_publishes_error(self, cortex):
        """A low-confidence parse should publish a CORTEX_ERROR."""
        cortex.intent_parser.parse = AsyncMock(
            return_value={"confidence": 0.0, "error": "unparseable"}
        )
        errors = []
        async def capture(event: Event):
            errors.append(event)
        cortex.bus.subscribe(EventType.CORTEX_ERROR, capture)

        await cortex._process_voice_command("xyzzy gibberish")
        assert len(errors) == 1


# ─── Stats Tests ──────────────────────────────────────────────────────────


class TestCortexStats:
    """Tests for cortex statistics."""

    @pytest.fixture
    def cortex(self):
        with patch.object(Cortex, '_init_llm_bridge', new_callable=AsyncMock):
            cortex = Cortex()
            cortex.bus = EventBus()
            cortex.intent_parser = MagicMock()
            cortex.intent_parser.start = AsyncMock()
            cortex.intent_parser.stop = AsyncMock()
            return cortex

    def test_stats_initial(self, cortex):
        """Initial stats should show zero decisions."""
        stats = cortex.get_stats()
        assert stats["decision_count"] == 0
        assert stats["error_count"] == 0
        assert stats["fallback_count"] == 0
        assert stats["state"] == CortexState.WAITING.value

    def test_stats_after_error(self, cortex):
        """Error count should increment after publishing an error."""
        import asyncio
        async def run():
            await cortex._publish_error("test error")
        asyncio.run(run())
        stats = cortex.get_stats()
        assert stats["error_count"] == 1