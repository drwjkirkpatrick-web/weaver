"""Tests for the event bus — the central nervous system of Weaver."""

from __future__ import annotations

import asyncio
import pytest

from weaver.event_bus import Event, EventBus, EventType


@pytest.mark.asyncio
async def test_publish_subscribe():
    """Test basic pub/sub: publish an event, handler receives it."""
    bus = EventBus()
    received = []
    
    async def handler(event: Event):
        received.append(event)
    
    bus.subscribe(EventType.ULTRASONIC_RANGE, handler)
    await bus.publish(Event(
        type=EventType.ULTRASONIC_RANGE,
        data={"distance_cm": 50.0},
        source="test",
    ))
    
    assert len(received) == 1
    assert received[0].data["distance_cm"] == 50.0
    assert received[0].source == "test"


@pytest.mark.asyncio
async def test_multiple_subscribers():
    """Test that multiple subscribers all receive the event."""
    bus = EventBus()
    received_a = []
    received_b = []
    
    async def handler_a(event: Event):
        received_a.append(event)
    
    async def handler_b(event: Event):
        received_b.append(event)
    
    bus.subscribe(EventType.IMU_DATA, handler_a)
    bus.subscribe(EventType.IMU_DATA, handler_b)
    
    await bus.publish(Event(type=EventType.IMU_DATA, data={"tilt": 5.0}))
    
    assert len(received_a) == 1
    assert len(received_b) == 1


@pytest.mark.asyncio
async def test_priority_handlers_run_first():
    """Priority handlers should execute before normal handlers."""
    bus = EventBus()
    execution_order = []
    
    async def normal_handler(event: Event):
        execution_order.append("normal")
    
    async def priority_handler(event: Event):
        execution_order.append("priority")
    
    bus.subscribe(EventType.MOTION_COMMAND, normal_handler, priority=False)
    bus.subscribe(EventType.MOTION_COMMAND, priority_handler, priority=True)
    
    await bus.publish(Event(type=EventType.MOTION_COMMAND, data={}))
    
    assert execution_order == ["priority", "normal"]


@pytest.mark.asyncio
async def test_estop_activation():
    """E-stop event should activate the e-stop flag."""
    bus = EventBus()
    assert not bus.estop_active
    
    await bus.publish(Event(
        type=EventType.SAFETY_ESTOP,
        data={"reason": "test"},
        source="test",
    ))
    
    assert bus.estop_active


@pytest.mark.asyncio
async def test_handler_error_doesnt_crash_bus():
    """A failing handler shouldn't prevent other handlers from running."""
    bus = EventBus()
    received = []
    
    async def bad_handler(event: Event):
        raise ValueError("Boom!")
    
    async def good_handler(event: Event):
        received.append(event)
    
    bus.subscribe(EventType.BATTERY_STATUS, bad_handler)
    bus.subscribe(EventType.BATTERY_STATUS, good_handler)
    
    await bus.publish(Event(type=EventType.BATTERY_STATUS, data={"voltage": 7.4}))
    
    assert len(received) == 1  # Good handler still ran


@pytest.mark.asyncio
async def test_event_history():
    """Events should be stored in history for debugging."""
    bus = EventBus(history_size=10)
    
    for i in range(15):
        await bus.publish(Event(
            type=EventType.ULTRASONIC_RANGE,
            data={"distance_cm": float(i)},
        ))
    
    history = bus.get_recent_events(5)
    assert len(history) == 5
    # Should be the last 5 events (10-14)
    assert history[0].data["distance_cm"] == 10.0
    assert history[-1].data["distance_cm"] == 14.0


@pytest.mark.asyncio
async def test_unsubscribe():
    """Unsubscribed handlers should not receive events."""
    bus = EventBus()
    received = []
    
    async def handler(event: Event):
        received.append(event)
    
    bus.subscribe(EventType.CAMERA_FRAME, handler)
    bus.unsubscribe(EventType.CAMERA_FRAME, handler)
    
    await bus.publish(Event(type=EventType.CAMERA_FRAME, data={}))
    
    assert len(received) == 0


def test_event_type_enum():
    """All expected event types should exist."""
    assert EventType.CAMERA_FRAME
    assert EventType.MOTION_COMMAND
    assert EventType.SAFETY_ESTOP
    assert EventType.CORTEX_DECISION
    assert EventType.VOICE_TRANSCRIPT
    assert EventType.SYSTEM_READY


def test_event_repr():
    """Event should have a useful repr."""
    event = Event(
        type=EventType.BATTERY_LOW,
        data={"voltage": 6.5},
        source="battery",
    )
    repr_str = repr(event)
    assert "BATTERY_LOW" in repr_str
    assert "battery" in repr_str