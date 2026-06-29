# event_bus.py — Async event bus for Weaver
#
# This is the central nervous system. All subsystems communicate through it.
# Sensors publish readings, the cortex subscribes to make decisions,
# and motion commands flow back through the same bus.
#
# Why an event bus?
#   - Decoupled: sensors don't need to know about the brain or motion
#   - Async: non-blocking, perfect for Pi's limited cores
#   - Extensible: add new modules by subscribing to existing events
#   - Debuggable: every event is logged, can be replayed
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine

from loguru import logger


# ─── Event Types ───────────────────────────────────────────────────────


class EventType(Enum):
    """All event types that flow through the bus."""
    
    # Sensor events
    CAMERA_FRAME = auto()           # New camera frame available
    CAMERA_OBJECT_DETECTED = auto() # CV detected an object
    CAMERA_FACE_DETECTED = auto()   # Face/person detected (safety!)
    CAMERA_COLOR_TRACKED = auto()   # Color tracking result
    
    IMU_DATA = auto()               # Gyro/accel reading
    IMU_TILT_WARNING = auto()       # Body tilt exceeded threshold
    
    ULTRASONIC_RANGE = auto()       # Distance reading
    ULTRASONIC_OBSTACLE = auto()    # Obstacle detected in path
    
    BATTERY_STATUS = auto()         # Battery voltage/level
    BATTERY_LOW = auto()            # Battery below threshold
    BATTERY_CRITICAL = auto()       # Battery critical — stop everything
    
    THERMAL_STATUS = auto()         # Pi CPU temperature
    THERMAL_WARNING = auto()        # Temperature high
    THERMAL_CRITICAL = auto()       # Temperature critical
    
    # Motion events
    MOTION_COMMAND = auto()         # Command to move (from cortex or web)
    MOTION_STARTED = auto()         # A motion has begun
    MOTION_COMPLETED = auto()       # A motion finished
    MOTION_ABORTED = auto()         # Motion was aborted (safety)
    GAIT_CHANGE = auto()            # Gait type changed
    
    # Safety events
    SAFETY_ESTOP = auto()           # EMERGENCY STOP
    SAFETY_WARNING = auto()         # Safety warning (soft)
    SAFETY_VIOLATION = auto()       # Safety constraint violated
    SAFETY_HUMAN_NEARBY = auto()    # Human detected within safe distance
    SAFETY_COLLISION_IMMINENT = auto()  # About to hit something
    
    # Brain events
    CORTEX_THINKING = auto()        # LLM is processing
    CORTEX_DECISION = auto()        # LLM made a decision
    CORTEX_ERROR = auto()           # LLM call failed
    CORTEX_FALLBACK = auto()        # Fell back to rule-based
    
    # Voice events
    VOICE_LISTENING = auto()        # Started listening
    VOICE_TRANSCRIPT = auto()       # STT result available
    VOICE_SPEAKING = auto()         # TTS is speaking
    VOICE_COMMAND = auto()          # Parsed voice command
    
    # System events
    SYSTEM_STARTUP = auto()         # Weaver is starting up
    SYSTEM_READY = auto()           # All subsystems initialized
    SYSTEM_SHUTDOWN = auto()        # Weaver is shutting down
    SYSTEM_ERROR = auto()           # Unhandled error
    MODULE_STATUS = auto()          # Module status update
    TELEMETRY_LOG = auto()          # Periodic telemetry record
    
    # Dance events
    BEAT_DETECTED = auto()          # Beat detector found a beat
    DANCE_STARTED = auto()          # Dance routine started
    DANCE_MOVE = auto()             # A dance move is being executed
    DANCE_COMPLETED = auto()        # Dance routine finished
    DANCE_ABORTED = auto()          # Dance was aborted (safety)


# ─── Event ─────────────────────────────────────────────────────────────


@dataclass
class Event:
    """A single event on the bus."""
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""                # Which module published
    
    def __repr__(self) -> str:
        return f"Event({self.type.name}, src={self.source}, data_keys={list(self.data.keys())})"


# ─── Event Bus ──────────────────────────────────────────────────────────


# Type alias for async event handlers
Handler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Async publish/subscribe event bus.
    
    Features:
    - Async handlers (non-blocking)
    - Multiple subscribers per event type
    - Event history (ring buffer) for debugging
    - Priority handlers (safety events jump the queue)
    - Graceful error handling (one bad handler doesn't crash the bus)
    
    Usage:
        bus = EventBus()
        
        # Subscribe
        async def handle_obstacle(event: Event):
            logger.info(f"Obstacle at {event.data['distance_cm']}cm")
        bus.subscribe(EventType.ULTRASONIC_OBSTACLE, handle_obstacle)
        
        # Publish
        await bus.publish(Event(
            type=EventType.ULTRASONIC_OBSTACLE,
            data={"distance_cm": 15.0},
            source="ultrasonic",
        ))
    """
    
    def __init__(self, history_size: int = 500):
        self._subscribers: dict[EventType, list[Handler]] = defaultdict(list)
        self._priority_subscribers: dict[EventType, list[Handler]] = defaultdict(list)
        self._history: deque[Event] = deque(maxlen=history_size)
        self._running = False
        self._event_count = 0
        # Emergency stop flag — checked by motion controller
        self._estop_active = False
    
    @property
    def estop_active(self) -> bool:
        """Is emergency stop active?"""
        return self._estop_active
    
    def set_estop(self, active: bool) -> None:
        """Set/clear emergency stop."""
        self._estop_active = active
        if active:
            logger.error("🛑 EMERGENCY STOP ACTIVATED")
        else:
            logger.info("✅ Emergency stop cleared")
    
    def subscribe(self, event_type: EventType, handler: Handler, priority: bool = False) -> None:
        """Subscribe a handler to an event type.
        
        Args:
            event_type: Which event to listen for
            handler: Async function called with the Event
            priority: If True, this handler runs before non-priority ones.
                      Use for safety-critical handlers.
        """
        if priority:
            self._priority_subscribers[event_type].append(handler)
        else:
            self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__name__} to {event_type.name} (priority={priority})")
    
    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        """Remove a handler subscription."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
        if handler in self._priority_subscribers[event_type]:
            self._priority_subscribers[event_type].remove(handler)
    
    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.
        
        Priority handlers run first (sequential, to ensure safety logic
        executes before anything else). Non-priority handlers run concurrently.
        """
        self._event_count += 1
        self._history.append(event)
        
        # Handle e-stop specially
        if event.type == EventType.SAFETY_ESTOP:
            self.set_estop(True)
        
        # Get all handlers for this event type
        priority_handlers = self._priority_subscribers.get(event.type, [])
        normal_handlers = self._subscribers.get(event.type, [])
        
        # Run priority handlers first (sequential — safety must complete)
        for handler in priority_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Priority handler {handler.__name__} error: {e}")
        
        # Run normal handlers concurrently
        if normal_handlers:
            tasks = [self._safe_call(handler, event) for handler in normal_handlers]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _safe_call(self, handler: Handler, event: Event) -> None:
        """Call a handler with error handling."""
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Handler {handler.__name__} error on {event.type.name}: {e}")
    
    def get_recent_events(self, count: int = 50) -> list[Event]:
        """Get recent events from history (for debugging/dashboard)."""
        return list(self._history)[-count:]
    
    def get_stats(self) -> dict[str, Any]:
        """Get bus statistics."""
        return {
            "total_events": self._event_count,
            "history_size": len(self._history),
            "estop_active": self._estop_active,
            "subscriber_count": sum(
                len(h) for h in self._subscribers.values()
            ) + sum(
                len(h) for h in self._priority_subscribers.values()
            ),
        }


# Singleton
_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus singleton."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus