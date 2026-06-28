"""Tests for the intent parser — natural language to robot action parsing.

Tests cover:
- Rule-based parsing for all supported commands
- Speed modifier extraction
- Duration extraction
- LLM fallback (mocked)
- Edge cases (empty, unknown commands)
"""

from __future__ import annotations

import pytest

from weaver.brain.intent_parser import IntentParser


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def parser():
    """IntentParser with no LLM bridge (rule-only mode)."""
    return IntentParser(llm_bridge=None)


# ─── Direction Commands ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_walk_forward(parser):
    """'walk forward' → {direction: forward, speed: 0.5}"""
    result = await parser.parse("walk forward")
    assert result["direction"] == "forward"
    assert result["speed"] == 0.5
    assert result["confidence"] >= 0.7


@pytest.mark.asyncio
async def test_parse_turn_left(parser):
    """'turn left' → {direction: turn_left, speed: 0.3}"""
    result = await parser.parse("turn left")
    assert result["direction"] == "turn_left"
    assert result["speed"] == 0.3


@pytest.mark.asyncio
async def test_parse_turn_right(parser):
    """'turn right' → {direction: turn_right, speed: 0.3}"""
    result = await parser.parse("turn right")
    assert result["direction"] == "turn_right"


@pytest.mark.asyncio
async def test_parse_stop(parser):
    """'stop' → {direction: stop, speed: 0.0}"""
    result = await parser.parse("stop")
    assert result["direction"] == "stop"
    assert result["speed"] == 0.0


@pytest.mark.asyncio
async def test_parse_halt(parser):
    """'halt' should also stop."""
    result = await parser.parse("halt")
    assert result["direction"] == "stop"


@pytest.mark.asyncio
async def test_parse_walk_backward(parser):
    """'walk backward' → {direction: backward}"""
    result = await parser.parse("walk backward")
    assert result["direction"] == "backward"


@pytest.mark.asyncio
async def test_parse_strafe_left(parser):
    """'strafe left' → {direction: strafe_left}"""
    result = await parser.parse("strafe left")
    assert result["direction"] == "strafe_left"


@pytest.mark.asyncio
async def test_parse_strafe_right(parser):
    """'strafe right' → {direction: strafe_right}"""
    result = await parser.parse("strafe right")
    assert result["direction"] == "strafe_right"


@pytest.mark.asyncio
async def test_parse_reverse(parser):
    """'reverse' → {direction: backward}"""
    result = await parser.parse("reverse")
    assert result["direction"] == "backward"


# ─── Action Commands ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_sit_down(parser):
    """'sit down' → {action: sit}"""
    result = await parser.parse("sit down")
    assert result["action"] == "sit"


@pytest.mark.asyncio
async def test_parse_stand_up(parser):
    """'stand up' → {action: stand}"""
    result = await parser.parse("stand up")
    assert result["action"] == "stand"


@pytest.mark.asyncio
async def test_parse_wave(parser):
    """'wave' → {action: wave}"""
    result = await parser.parse("wave")
    assert result["action"] == "wave"


@pytest.mark.asyncio
async def test_parse_crouch(parser):
    """'crouch' → {action: sit}"""
    result = await parser.parse("crouch")
    assert result["action"] == "sit"


# ─── Speed Modifiers ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_speed_slowly(parser):
    """'walk forward slowly' → speed should be 0.3"""
    result = await parser.parse("walk forward slowly")
    assert result["direction"] == "forward"
    assert result["speed"] == 0.3


@pytest.mark.asyncio
async def test_parse_speed_fast(parser):
    """'walk forward fast' → speed should be 0.7"""
    result = await parser.parse("walk forward fast")
    assert result["direction"] == "forward"
    assert result["speed"] == 0.7


@pytest.mark.asyncio
async def test_parse_speed_percentage(parser):
    """'walk forward at 80%' → speed should be 0.8"""
    result = await parser.parse("walk forward at 80%")
    assert result["direction"] == "forward"
    assert result["speed"] == 0.8


# ─── Duration Extraction ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_duration_explicit(parser):
    """'walk forward for 3 seconds' → duration should be 3.0"""
    result = await parser.parse("walk forward for 3 seconds")
    assert result["direction"] == "forward"
    assert result["duration"] == 3.0


@pytest.mark.asyncio
async def test_parse_duration_keyword(parser):
    """'walk forward for a bit' → duration should be set"""
    result = await parser.parse("walk forward for a bit")
    assert result["direction"] == "forward"
    assert "duration" in result


# ─── Edge Cases ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_empty_string(parser):
    """Empty string should return an error result."""
    result = await parser.parse("")
    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_parse_unknown_command(parser):
    """Unknown command should have low confidence."""
    result = await parser.parse("xyzzy frobnicate")
    assert result["confidence"] < 0.7


@pytest.mark.asyncio
async def test_parse_raw_text_preserved(parser):
    """The raw_text field should preserve the original input."""
    result = await parser.parse("walk forward")
    assert "raw_text" in result


@pytest.mark.asyncio
async def test_parse_source_field(parser):
    """Result should include a source field ('rule' or 'llm')."""
    result = await parser.parse("stop")
    assert result["source"] == "rule"


# ─── LLM Fallback ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_llm_fallback():
    """When LLM bridge is available, low-confidence rule results should
    trigger LLM parsing."""
    from unittest.mock import AsyncMock, MagicMock

    mock_bridge = MagicMock()
    mock_bridge.generate = AsyncMock(return_value='{"direction": "forward", "speed": 0.4}')

    parser = IntentParser(llm_bridge=mock_bridge)
    await parser.start()

    # A complex command that rules can't parse well
    result = await parser.parse("carefully navigate to the kitchen")

    # LLM should have been called
    assert mock_bridge.generate.called
    assert result["source"] == "llm"
    await parser.stop()


@pytest.mark.asyncio
async def test_parse_llm_json_extraction():
    """LLM response with extra text should still have JSON extracted."""
    from unittest.mock import AsyncMock, MagicMock

    mock_bridge = MagicMock()
    mock_bridge.generate = AsyncMock(
        return_value='Here is the command: {"direction": "stop", "speed": 0.0}'
    )

    parser = IntentParser(llm_bridge=mock_bridge)
    await parser.start()

    result = await parser.parse("please stop moving immediately")
    assert result["direction"] == "stop"
    await parser.stop()


# ─── Stats ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stats(parser):
    """Stats should reflect parsing activity."""
    await parser.start()
    await parser.parse("walk forward")
    await parser.parse("stop")
    stats = parser.get_stats()
    assert stats["total_parsed"] == 2
    assert stats["rule_parsed"] == 2
    assert stats["llm_parsed"] == 0
    await parser.stop()