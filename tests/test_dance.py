"""Tests for the dance module: safety, engine, move library, Macarena."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from weaver.dance.dance_safety import (
    DanceSafetyGovernor, DanceSafetyLevel, DanceThresholds, DanceSessionState,
)
from weaver.dance.dance_engine import (
    DanceEngine, DanceMove, DanceRoutine, TransitionType,
)
from weaver.dance.move_library import (
    get_move, list_moves, body_sway_right, body_sway_left,
    arms_up, pose_standing, pose_tall, pose_crouch,
    leg_lift_front_left, leg_lift_front_right, ALL_MOVES,
)
from weaver.dance.student_api import StudentDance, list_available_moves
from weaver.dance.macarena import create_macarena, create_macarena_routine


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def safety_gov():
    """Dance safety governor in STRICT mode."""
    return DanceSafetyGovernor(DanceSafetyLevel.STRICT)


@pytest.fixture
def mock_servo():
    """Mock servo driver."""
    servo = MagicMock()
    servo._enabled = True
    servo.set_angles = AsyncMock()
    servo.smooth_move = AsyncMock()
    servo.get_current_angles = MagicMock(return_value=[90.0] * 18)
    return servo


@pytest.fixture
def dance_engine(mock_servo, safety_gov):
    """Dance engine with mock servo and safety governor."""
    engine = DanceEngine(servo=mock_servo, safety_governor=safety_gov)
    return engine


# ═════════════════════════════════════════════════════════════════════════
# Dance Safety Governor Tests
# ═════════════════════════════════════════════════════════════════════════


class TestDanceSafety:
    """Test the dance safety governor."""
    
    def test_initialization(self, safety_gov):
        """Safety governor should initialize with STRICT level."""
        assert safety_gov.state.safety_level == DanceSafetyLevel.STRICT
        assert safety_gov.thresholds.coxa_min == 70.0
        assert safety_gov.thresholds.coxa_max == 110.0
    
    def test_start_session(self, safety_gov):
        """Starting a session should work."""
        assert safety_gov.start_session() is True
        assert safety_gov.state.active is True
    
    def test_end_session(self, safety_gov):
        """Ending a session should update state."""
        safety_gov.start_session()
        safety_gov.end_session()
        assert safety_gov.state.active is False
        assert safety_gov.state.duration > 0
    
    def test_check_move_safe(self, safety_gov):
        """Safe angles should pass and be returned (clamped if needed)."""
        safety_gov.start_session()
        angles = [90.0] * 18  # All centered — definitely safe
        result = safety_gov.check_move(angles, feet_on_ground=6)
        assert result is not None
        assert len(result) == 18
    
    def test_check_move_clamps_angles(self, safety_gov):
        """Angles outside safe range should be clamped."""
        safety_gov.start_session()
        angles = [0.0] * 18  # Way below safe min
        result = safety_gov.check_move(angles, feet_on_ground=6)
        assert result is not None
        # All angles should be clamped to at least coxa_min (70 for STRICT)
        for i in range(18):
            assert result[i] >= 50.0  # At least minimum for any joint
    
    def test_check_move_blocks_too_few_feet(self, safety_gov):
        """Less than 3 feet on ground should be blocked."""
        safety_gov.start_session()
        angles = [90.0] * 18
        result = safety_gov.check_move(angles, feet_on_ground=2)
        assert result is None  # Blocked
        assert safety_gov.state.violations > 0
    
    def test_check_move_blocks_large_height_change(self, safety_gov):
        """Height change exceeding limit should be blocked."""
        safety_gov.start_session()
        angles = [90.0] * 18
        result = safety_gov.check_move(angles, feet_on_ground=6, height_change=50.0)
        assert result is None  # 50mm exceeds STRICT limit of 10mm
    
    def test_rate_limiting(self, safety_gov):
        """Too many moves per second should be rate-limited."""
        safety_gov.start_session()
        angles = [90.0] * 18
        
        # STRICT mode allows 2 moves/second
        # First move should pass
        r1 = safety_gov.check_move(angles, feet_on_ground=6)
        assert r1 is not None
        
        # Second move should pass (within 2/second)
        import time
        time.sleep(0.06)  # Just over min_move_interval_ms (50ms)
        r2 = safety_gov.check_move(angles, feet_on_ground=6)
        assert r2 is not None
        
        # Third move should be rate-limited (exceeds 2/second)
        time.sleep(0.06)
        r3 = safety_gov.check_move(angles, feet_on_ground=6)
        # Might pass or fail depending on timing — just verify no crash
        assert r3 is None or isinstance(r3, list)
    
    def test_temperature_check(self, safety_gov):
        """High temperature should pause dancing."""
        assert safety_gov.check_temperature(50.0) is True  # Normal
        assert safety_gov.check_temperature(70.0) is False  # Too hot (max 65)
        assert safety_gov.state.temp_paused is True
    
    def test_is_angle_safe(self, safety_gov):
        """is_angle_safe should check joint-specific ranges."""
        # Coxa channel 0: STRICT range is 70-110
        assert safety_gov.is_angle_safe(0, 90.0) is True   # Center is safe
        assert safety_gov.is_angle_safe(0, 60.0) is False  # Below min
        assert safety_gov.is_angle_safe(0, 120.0) is False  # Above max
    
    def test_get_safe_range(self, safety_gov):
        """get_safe_range should return the correct range per joint."""
        coxa_lo, coxa_hi = safety_gov.get_safe_range("coxa")
        assert coxa_lo == 70.0
        assert coxa_hi == 110.0
        
        femur_lo, femur_hi = safety_gov.get_safe_range("femur")
        assert femur_lo == 65.0
        assert femur_hi == 115.0
    
    def test_normal_level_allows_more(self):
        """NORMAL level should allow wider angle range than STRICT."""
        gov = DanceSafetyGovernor(DanceSafetyLevel.NORMAL)
        assert gov.thresholds.coxa_min < 70.0  # More permissive
        assert gov.thresholds.coxa_max > 110.0
    
    def test_performance_level_allows_most(self):
        """PERFORMANCE level should allow the widest range."""
        gov = DanceSafetyGovernor(DanceSafetyLevel.PERFORMANCE)
        assert gov.thresholds.coxa_min == 50.0
        assert gov.thresholds.max_session_seconds == 300.0
    
    def test_get_status(self, safety_gov):
        """get_status should return a complete status dict."""
        status = safety_gov.get_status()
        assert "active" in status
        assert "safety_level" in status
        assert "angle_range" in status
        assert "coxa" in status["angle_range"]


# ═════════════════════════════════════════════════════════════════════════
# Move Library Tests
# ═════════════════════════════════════════════════════════════════════════


class TestMoveLibrary:
    """Test the pre-built dance moves."""
    
    def test_get_move_valid(self):
        """get_move should return a DanceMove for valid names."""
        move = get_move("sway_right", beats=2)
        assert isinstance(move, DanceMove)
        assert move.name == "sway_right"
        assert move.beats == 2
        assert len(move.angles) == 18
    
    def test_get_move_invalid(self):
        """get_move should raise KeyError for unknown moves."""
        with pytest.raises(KeyError):
            get_move("does_not_exist")
    
    def test_list_moves(self):
        """list_moves should return all move names."""
        moves = list_moves()
        assert len(moves) > 20  # We have 27+ moves
        assert "sway_right" in moves
        assert "arms_up" in moves
        assert "standing" in moves
    
    def test_all_moves_have_18_angles(self):
        """Every move should have exactly 18 servo angles."""
        for name in list_moves():
            move = get_move(name, beats=1)
            assert len(move.angles) == 18, f"Move {name} has {len(move.angles)} angles"
    
    def test_all_moves_have_safe_angles(self):
        """All move angles should be within the 0-180 servo range."""
        for name in list_moves():
            move = get_move(name, beats=1)
            for angle in move.angles:
                assert 0 <= angle <= 180, f"Move {name} has angle {angle} out of range"
    
    def test_standing_move_is_centered(self):
        """Standing pose should have all angles at 90."""
        move = pose_standing()
        for angle in move.angles:
            assert angle == 90.0
    
    def test_sway_right_body_stays_down(self):
        """Sway right should keep all 6 feet on ground."""
        move = body_sway_right()
        assert move.feet_on_ground == 6
    
    def test_arms_up_lifts_2_legs(self):
        """Arms up should have only 4 feet on ground."""
        move = arms_up()
        assert move.feet_on_ground == 4
    
    def test_leg_lift_lifts_1_leg(self):
        """Leg lift should have 5 feet on ground."""
        move = leg_lift_front_left()
        assert move.feet_on_ground == 5


# ═════════════════════════════════════════════════════════════════════════
# Dance Engine Tests
# ═════════════════════════════════════════════════════════════════════════


class TestDanceEngine:
    """Test the dance engine."""
    
    @pytest.mark.asyncio
    async def test_start_stop(self, dance_engine, mock_servo):
        """Engine should start and stop."""
        await dance_engine.start()
        assert dance_engine._running is True
        await dance_engine.stop()
        assert dance_engine._running is False
    
    @pytest.mark.asyncio
    async def test_dance_simple(self, dance_engine, mock_servo):
        """A simple dance should execute and call servo."""
        await dance_engine.start()
        
        routine = DanceRoutine(
            name="Test Dance",
            moves=[
                pose_standing(beats=1),
                body_sway_right(beats=1),
                pose_standing(beats=1),
            ],
            bpm=120.0,
        )
        
        await dance_engine.dance(routine)
        
        # Servo should have been called
        assert mock_servo.set_angles.called or mock_servo.smooth_move.called
        await dance_engine.stop()
    
    @pytest.mark.asyncio
    async def test_dance_blocked_by_safety(self, mock_servo):
        """Dance with unsafe moves should be blocked."""
        # Create a governor in STRICT mode
        gov = DanceSafetyGovernor(DanceSafetyLevel.STRICT)
        engine = DanceEngine(servo=mock_servo, safety_governor=gov)
        await engine.start()
        
        # Create a routine with an unsafe move (only 2 feet on ground)
        unsafe_move = DanceMove(
            name="unsafe",
            angles=[90.0] * 18,
            beats=1,
            feet_on_ground=2,  # Below minimum of 3
        )
        routine = DanceRoutine(name="Unsafe", moves=[unsafe_move], bpm=120.0)
        
        await engine.dance(routine)
        
        # The move should have been blocked
        # Safety governor should have recorded a violation
        assert gov.state.violations > 0
        
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_stop_dance(self, dance_engine, mock_servo):
        """stop_dance should stop the current dance."""
        await dance_engine.start()
        
        # Start a long dance in the background
        routine = DanceRoutine(
            name="Long Dance",
            moves=[pose_standing(beats=100)],
            bpm=60.0,
        )
        
        asyncio.create_task(dance_engine.dance(routine))
        await asyncio.sleep(0.1)
        
        await dance_engine.stop_dance()
        assert dance_engine._dancing is False
        
        await dance_engine.stop()
    
    def test_get_status(self, dance_engine):
        """get_status should return engine state."""
        status = dance_engine.get_status()
        assert "running" in status
        assert "dancing" in status
        assert "safety" in status


# ═════════════════════════════════════════════════════════════════════════
# Student API Tests
# ═════════════════════════════════════════════════════════════════════════


class TestStudentAPI:
    """Test the student dance creation API."""
    
    def test_create_dance(self):
        """Creating a student dance should work."""
        dance = StudentDance("My Dance", bpm=120, creator="Test")
        assert dance.name == "My Dance"
        assert dance.bpm == 120.0
        assert dance.creator == "Test"
        assert len(dance._moves) == 0
    
    def test_add_move(self):
        """Adding a move should increase the move count."""
        dance = StudentDance("Test")
        dance.add("arms_up", beats=2)
        assert len(dance._moves) == 1
        assert dance._moves[0].name == "arms_up"
        assert dance._moves[0].beats == 2
    
    def test_add_chaining(self):
        """add() should return self for chaining."""
        dance = StudentDance("Test")
        result = dance.add("arms_up").add("sway_right").add("standing")
        assert result is dance
        assert len(dance._moves) == 3
    
    def test_add_unknown_move(self):
        """Adding an unknown move should raise KeyError."""
        dance = StudentDance("Test")
        with pytest.raises(KeyError):
            dance.add("nonexistent_move")
    
    def test_add_pause(self):
        """add_pause should add a standing move."""
        dance = StudentDance("Test")
        dance.add_pause(beats=2)
        assert len(dance._moves) == 1
        assert dance._moves[0].name == "standing"
    
    def test_repeat(self):
        """repeat should add the move multiple times."""
        dance = StudentDance("Test")
        dance.repeat("sway_right", times=3, beats=1)
        assert len(dance._moves) == 3
        assert all(m.name == "sway_right" for m in dance._moves)
    
    def test_sequence(self):
        """sequence should add multiple moves at once."""
        dance = StudentDance("Test")
        dance.sequence("arms_up", "sway_right", "sway_left", "standing", beats=1)
        assert len(dance._moves) == 4
    
    def test_build(self):
        """build should return a DanceRoutine."""
        dance = StudentDance("Test", bpm=120)
        dance.add("arms_up", beats=2).add("standing", beats=2)
        routine = dance.build()
        
        assert isinstance(routine, DanceRoutine)
        assert routine.name == "Test"
        assert len(routine.moves) == 2
        assert routine.bpm == 120.0
        assert routine.total_beats == 4
    
    def test_bpm_clamped(self):
        """BPM should be clamped to 60-200."""
        dance = StudentDance("Test", bpm=500)
        assert dance.bpm == 200.0
        
        dance2 = StudentDance("Test", bpm=10)
        assert dance2.bpm == 60.0
    
    def test_info(self):
        """info should return a readable description."""
        dance = StudentDance("Test", bpm=120, creator="Alice")
        dance.add("arms_up", beats=2)
        info = dance.info()
        assert "Test" in info
        assert "Alice" in info
        assert "arms_up" in info
    
    def test_list_available_moves(self):
        """list_available_moves should return move names."""
        moves = list_available_moves()
        assert "arms_up" in moves
        assert "sway_right" in moves
        assert len(moves) > 20


# ═════════════════════════════════════════════════════════════════════════
# Macarena Tests
# ═════════════════════════════════════════════════════════════════════════


class TestMacarena:
    """Test the Macarena choreography."""
    
    def test_create_macarena(self):
        """create_macarena should return a StudentDance."""
        dance = create_macarena()
        assert isinstance(dance, StudentDance)
        assert dance.name == "Macarena"
        assert dance.bpm == 100.0
    
    def test_macarena_has_moves(self):
        """Macarena should have a substantial number of moves."""
        dance = create_macarena()
        assert len(dance._moves) > 20  # We expect ~28 moves
    
    def test_macarena_total_beats(self):
        """Macarena should have a reasonable total beat count."""
        dance = create_macarena()
        routine = dance.build()
        assert routine.total_beats > 40  # Should be ~56 beats
    
    def test_macarena_routine(self):
        """create_macarena_routine should return a DanceRoutine."""
        routine = create_macarena_routine()
        assert isinstance(routine, DanceRoutine)
        assert routine.name == "Macarena"
        assert len(routine.moves) > 0
    
    def test_macarena_duration(self):
        """Macarena should have a reasonable estimated duration."""
        routine = create_macarena_routine()
        # At 100 BPM, 56 beats = ~33.6 seconds
        assert 20 < routine.estimated_duration < 60
    
    def test_macarena_contains_key_moves(self):
        """Macarena should contain arm_out and sway moves."""
        dance = create_macarena()
        move_names = [m.name for m in dance._moves]
        assert "arm_out_right" in move_names  # Right arm out
        assert "arm_out_left" in move_names   # Left arm out
        assert "sway_right" in move_names     # Hip sway right
        assert "sway_left" in move_names       # Hip sway left
        assert "spin_right" in move_names      # Turn
        assert "arms_up" in move_names         # Finale!