# cortex.py — Main LLM decision-making brain for Weaver
#
# The cortex is the "thinking" part of the robot. It:
# 1. Subscribes to ALL sensor events to build a world model
# 2. Periodically evaluates the situation (every 0.5s by default)
# 3. Uses an LLM (Ollama Cloud or Local) to decide what to do
# 4. Issues motion commands (which pass through the safety governor)
# 5. Falls back to rule-based behavior if the LLM is unavailable
#
# Architecture:
#
#   Sensor Events �───────► Cortex ──► World Model
#   Voice Commands �─┐        │
#                   │        ▼
#                   └──► Intent Parser ─► LLM Decision ─► MOTION_COMMAND
#                                                        (→ Safety Governor)
#
# The cortex NEVER directly commands servos. It publishes MOTION_COMMAND events
# which are intercepted by the safety governor (safety.py) before reaching
# the motion controller. This is a critical safety invariant.
#
# LEARNING NOTE: The LLM is called asynchronously with a timeout. If the LLM
# is slow or unavailable, the cortex falls back to rule-based behavior. This
# means the robot is ALWAYS safe — even if the cloud goes down, it will still
# avoid obstacles and respond to basic commands.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from weaver.config import CortexConfig, HardwareMode, LLMMode, get_config
from weaver.event_bus import Event, EventBus, EventType, get_event_bus


# ─── World Model ─────────────────────────────────────────────────────────


@dataclass
class WorldModel:
    """Rolling snapshot of the robot's understanding of its environment.

    Updated by sensor event handlers. Read by the cortex when it's time
    to make a decision. This is the "state" that gets passed to the LLM.

    Attributes:
        obstacle_distance_cm: Distance to nearest obstacle (from ultrasonic)
        obstacle_warning: True if obstacle is within slow/stop range
        body_tilt_degrees: Current body tilt (from IMU)
        tilt_warning: True if tilt exceeds safe threshold
        battery_voltage: Current battery voltage
        battery_low: True if battery is low
        cpu_temp_c: Pi CPU temperature
        human_detected: True if camera detects a human
        human_distance_cm: Distance to nearest human (if detected)
        last_motion: Direction of last issued motion command
        current_speed: Current movement speed (0.0-1.0)
        last_updated: Timestamp of most recent sensor update
    """

    obstacle_distance_cm: float | None = None
    obstacle_warning: bool = False
    body_tilt_degrees: float = 0.0
    tilt_warning: bool = False
    battery_voltage: float = 8.4
    battery_low: bool = False
    cpu_temp_c: float = 45.0
    human_detected: bool = False
    human_distance_cm: float | None = None
    last_motion: str = "stop"
    current_speed: float = 0.0
    last_updated: float = field(default_factory=time.time)

    # Object detection (from camera)
    detected_objects: list[str] = field(default_factory=list)
    tracked_color: str = ""

    def to_llm_context(self) -> str:
        """Convert the world model to a concise text summary for the LLM.

        The LLM doesn't need every field — it needs a concise summary of
        the current situation. This method formats the relevant data into
        a short, readable string.

        Returns:
            A text summary like:
            "Obstacle: 45cm (warning). Battery: 7.2V (low). Tilt: 5°. No humans detected."
        """
        parts: list[str] = []

        # Obstacle info
        if self.obstacle_distance_cm is not None:
            status = "WARNING" if self.obstacle_warning else "clear"
            parts.append(f"Obstacle: {self.obstacle_distance_cm:.0f}cm ({status})")
        else:
            parts.append("Obstacle: unknown")

        # Tilt
        parts.append(f"Tilt: {self.body_tilt_degrees:.1f}°")
        if self.tilt_warning:
            parts.append("TILT WARNING")

        # Battery
        bat_status = "LOW" if self.battery_low else "OK"
        parts.append(f"Battery: {self.battery_voltage:.1f}V ({bat_status})")

        # Temperature
        parts.append(f"CPU: {self.cpu_temp_c:.0f}°C")

        # Humans
        if self.human_detected:
            dist = f" at {self.human_distance_cm:.0f}cm" if self.human_distance_cm else ""
            parts.append(f"Human detected{dist}")
        else:
            parts.append("No humans detected")

        # Current motion
        parts.append(f"Moving: {self.last_motion} @ {self.current_speed:.1%}")

        # Objects
        if self.detected_objects:
            parts.append(f"Objects: {', '.join(self.detected_objects)}")

        return ". ".join(parts) + "."


# ─── Cortex State ────────────────────────────────────────────────────────


class CortexState(str, Enum):
    """States the cortex can be in.

    THINKING: LLM is processing a decision
    WAITING: Between decisions (idle, no action needed)
    ACTING: A motion command is being executed
    FALLBACK: LLM unavailable, using rule-based behavior
    ERROR: Something went wrong
    """
    THINKING = "thinking"
    WAITING = "waiting"
    ACTING = "acting"
    FALLBACK = "fallback"
    ERROR = "error"


# ─── Cortex ──────────────────────────────────────────────────────────────


class Cortex:
    """The LLM-powered decision-making brain of the Weaver robot.

    The cortex subscribes to all sensor events and maintains a world model.
    Every `decision_interval_seconds` (default 0.5s), it evaluates the
    situation and decides what the robot should do next.

    Decision flow:
    1. Gather sensor data from the world model
    2. Check if an immediate safety action is needed (rule-based, instant)
    3. If no immediate action needed, consult the LLM for a decision
    4. Parse the LLM's response into a motion command
    5. Publish the motion command (→ safety governor → motion controller)

    Safety invariant: The cortex NEVER bypasses the safety governor.
    All motion commands go through the event bus where safety.py intercepts
    them. The cortex can REQUEST motion, but safety can VETO it.

    Attributes:
        config: CortexConfig instance
        bus: The event bus
        world_model: Current world model snapshot
        state: Current cortex state (THINKING/WAITING/ACTING/etc.)
        llm_bridge: The active LLM bridge (cloud or local)
        intent_parser: Parser for voice commands
        context_window: Rolling list of recent sensor data / decisions
        _decision_task: The async decision loop task

    Example:
        cortex = Cortex()
        await cortex.start()
        # ... robot runs, cortex makes decisions every 0.5s ...
        await cortex.stop()
    """

    def __init__(
        self,
        config: CortexConfig | None = None,
        hardware_mode: HardwareMode | None = None,
    ) -> None:
        """Initialize the cortex.

        Args:
            config: CortexConfig instance. If None, loads from global config.
            hardware_mode: Hardware mode (REAL or MOCK). If None, loads from config.
        """
        self.config: CortexConfig = config or get_config().cortex
        self.main_config = get_config()
        self.hardware_mode: HardwareMode = hardware_mode or self.main_config.hardware_mode
        self.bus: EventBus = get_event_bus()

        # World model — updated by sensor handlers
        self.world_model: WorldModel = WorldModel()

        # Cortex state
        self.state: CortexState = CortexState.WAITING
        self._decision_count: int = 0
        self._error_count: int = 0
        self._fallback_count: int = 0
        self._last_decision_time: float = 0.0

        # LLM bridge — lazily initialized in start()
        self.llm_bridge: Any = None
        self.intent_parser: Any = None

        # Rolling context window — recent sensor summaries and decisions
        self.context_window: deque[dict[str, Any]] = deque(
            maxlen=self.config.max_context_messages
        )

        # Async task for the decision loop
        self._decision_task: asyncio.Task | None = None
        self._running: bool = False

        # Track if we have an active voice command to process
        self._pending_voice_command: str | None = None

    async def start(self) -> None:
        """Start the cortex.

        This:
        1. Initializes the LLM bridge (cloud or local based on config)
        2. Creates the intent parser
        3. Subscribes to all sensor events
        4. Starts the periodic decision loop

        Raises:
            ImportError: If local mode is selected but ollama package isn't installed.
        """
        logger.info(
            f"🧠 Cortex starting — mode: {self.config.mode.value}, "
            f"hardware: {self.hardware_mode.value}"
        )

        # ─── Initialize LLM bridge ───
        await self._init_llm_bridge()

        # ─── Initialize intent parser ───
        # Imported here to avoid circular import at module load time
        from weaver.brain.intent_parser import IntentParser

        self.intent_parser = IntentParser(
            llm_bridge=self.llm_bridge,
            config=self.config,
        )
        await self.intent_parser.start()

        # ─── Subscribe to sensor events ───
        self._subscribe_to_events()

        # ─── Start the decision loop ───
        self._running = True
        self._decision_task = asyncio.create_task(self._decision_loop())

        logger.info("✅ Cortex online — making decisions every {:.1f}s", self.config.decision_interval_seconds)

    async def stop(self) -> None:
        """Stop the cortex.

        Cancels the decision loop and shuts down the LLM bridge.
        """
        logger.info("🧠 Cortex shutting down...")
        self._running = False

        if self._decision_task:
            self._decision_task.cancel()
            try:
                await self._decision_task
            except asyncio.CancelledError:
                pass
            self._decision_task = None

        if self.llm_bridge:
            await self.llm_bridge.stop()

        if self.intent_parser:
            await self.intent_parser.stop()

        logger.info(
            f"Cortex stopped (decisions={self._decision_count}, "
            f"errors={self._error_count}, fallbacks={self._fallback_count})"
        )

    # ─── Initialization ───────────────────────────────────────────────

    async def _init_llm_bridge(self) -> None:
        """Initialize the LLM bridge based on config mode.

        Creates either OllamaCloudBridge or OllamaLocalBridge.
        The bridges have the same interface, so the cortex doesn't care
        which one is active.
        """
        if self.config.mode == LLMMode.OLLAMA_CLOUD:
            from weaver.brain.ollama_cloud import OllamaCloudBridge
            self.llm_bridge = OllamaCloudBridge(config=self.config)
        else:
            from weaver.brain.ollama_local import OllamaLocalBridge
            self.llm_bridge = OllamaLocalBridge(config=self.config)

        await self.llm_bridge.start()

        # Check availability — warn but don't crash if LLM is down
        available = await self.llm_bridge.is_available()
        if not available:
            logger.warning(
                f"⚠️  LLM ({self.config.mode.value}) is not available. "
                f"Cortex will use rule-based fallback."
            )
            if not self.config.fallback_to_rules:
                logger.error("fallback_to_rules is False and LLM is down — cortex cannot make decisions!")
        else:
            logger.info(f"✅ LLM bridge available: {self.llm_bridge.get_stats()}")

    def _subscribe_to_events(self) -> None:
        """Subscribe to all sensor and system events.

        The cortex needs to know about EVERYTHING happening on the robot.
        Each handler updates the world model.
        """
        # Ultrasonic / obstacle detection
        self.bus.subscribe(EventType.ULTRASONIC_RANGE, self._on_ultrasonic_range)
        self.bus.subscribe(EventType.ULTRASONIC_OBSTACLE, self._on_ultrasonic_obstacle)

        # IMU / tilt
        self.bus.subscribe(EventType.IMU_DATA, self._on_imu_data)
        self.bus.subscribe(EventType.IMU_TILT_WARNING, self._on_tilt_warning)

        # Battery
        self.bus.subscribe(EventType.BATTERY_STATUS, self._on_battery_status)
        self.bus.subscribe(EventType.BATTERY_LOW, self._on_battery_low)
        self.bus.subscribe(EventType.BATTERY_CRITICAL, self._on_battery_critical)

        # Thermal
        self.bus.subscribe(EventType.THERMAL_STATUS, self._on_thermal_status)
        self.bus.subscribe(EventType.THERMAL_WARNING, self._on_thermal_warning)

        # Camera
        self.bus.subscribe(EventType.CAMERA_OBJECT_DETECTED, self._on_object_detected)
        self.bus.subscribe(EventType.CAMERA_FACE_DETECTED, self._on_face_detected)
        self.bus.subscribe(EventType.CAMERA_COLOR_TRACKED, self._on_color_tracked)

        # Safety
        self.bus.subscribe(EventType.SAFETY_ESTOP, self._on_safety_estop)
        self.bus.subscribe(EventType.SAFETY_COLLISION_IMMINENT, self._on_collision_imminent)

        # Voice commands — these trigger immediate action
        self.bus.subscribe(EventType.VOICE_TRANSCRIPT, self._on_voice_transcript)

        # Motion feedback
        self.bus.subscribe(EventType.MOTION_COMPLETED, self._on_motion_completed)
        self.bus.subscribe(EventType.MOTION_ABORTED, self._on_motion_aborted)

        logger.debug("Cortex subscribed to all sensor events")

    # ─── Decision Loop ────────────────────────────────────────────────

    async def _decision_loop(self) -> None:
        """The main decision-making loop.

        Runs every `decision_interval_seconds` (default 0.5s). Each iteration:
        1. Check for immediate safety actions (obstacle, tilt, battery)
        2. Check for pending voice commands
        3. If no immediate action needed, consult the LLM
        4. Execute the decision

        LEARNING NOTE: The loop uses asyncio.sleep() which means it's
        cooperative — other handlers can run between iterations. This is
        important because sensor events arrive asynchronously and the
        event bus needs to process them.
        """
        logger.debug("Decision loop started")
        while self._running:
            try:
                await self._evaluate_situation()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Decision loop error: {e}", exc_info=True)
                self._error_count += 1
                self.state = CortexState.ERROR
                await self._publish_error(f"Decision loop error: {e}")

            await asyncio.sleep(self.config.decision_interval_seconds)

        logger.debug("Decision loop stopped")

    async def _evaluate_situation(self) -> None:
        """Evaluate the current situation and decide what to do.

        This is called on every loop iteration. It:
        1. Checks for immediate safety actions (rule-based, instant)
        2. Processes any pending voice command
        3. If safe and no voice command, consults the LLM

        The safety checks happen BEFORE the LLM call because they're
        time-critical. If there's an obstacle 15cm away, we don't have
        time to wait for an LLM response.
        """
        self._decision_count += 1
        self._last_decision_time = time.time()

        # ─── Step 1: Check for immediate safety actions ───
        safety_action = self._check_safety_rules()
        if safety_action:
            # Immediate safety action overrides everything
            await self._execute_action(safety_action, source="safety_rule")
            return

        # ─── Step 2: Process pending voice command ───
        if self._pending_voice_command:
            voice_cmd = self._pending_voice_command
            self._pending_voice_command = None
            await self._process_voice_command(voice_cmd)
            return

        # ─── Step 3: LLM evaluation (if available) ───
        # Only call LLM if the situation warrants thinking.
        # We don't call the LLM every 0.5s — that's wasteful.
        # Only call when:
        # - There's something new in the environment, OR
        # - The robot is idle and might need to explore, OR
        # - It's been a while since the last LLM call
        if self._should_consult_llm():
            await self._consult_llm()

    def _check_safety_rules(self) -> dict[str, Any] | None:
        """Check rule-based safety conditions.

        These are INSTANT checks that don't need the LLM. They're the
        fail-safe layer that keeps the robot safe even if the LLM is slow.

        Returns:
            A motion command dict if a safety action is needed, None otherwise.
        """
        # E-stop
        if self.bus.estop_active:
            return {"direction": "stop", "speed": 0.0, "reason": "estop"}

        # Obstacle too close — STOP immediately
        if self.world_model.obstacle_distance_cm is not None:
            obstacle_stop = self.main_config.safety.obstacle_stop_distance_cm
            if self.world_model.obstacle_distance_cm < obstacle_stop:
                return {
                    "direction": "stop",
                    "speed": 0.0,
                    "reason": f"obstacle_close_{self.world_model.obstacle_distance_cm:.0f}cm",
                }

        # Tilt warning — stabilize
        if self.world_model.tilt_warning:
            return {"direction": "stop", "speed": 0.0, "reason": "tilt_warning"}

        # Battery critical — stop
        if self.world_model.battery_voltage < self.main_config.safety.min_battery_voltage:
            return {"direction": "stop", "speed": 0.0, "reason": "battery_critical"}

        # Human very close — stop
        if (self.world_model.human_detected and
                self.world_model.human_distance_cm is not None and
                self.world_model.human_distance_cm < 30.0):
            return {"direction": "stop", "speed": 0.0, "reason": "human_close"}

        return None

    def _should_consult_llm(self) -> bool:
        """Decide if the LLM should be consulted this cycle.

        We don't want to call the LLM every 0.5s — that's expensive and
        slow. Only consult the LLM when:
        - The robot is not currently moving (idle)
        - OR there's a significant change in the environment
        - AND the LLM is available

        Returns:
            True if the LLM should be consulted, False otherwise.
        """
        # Don't consult LLM if it's not available
        if self.llm_bridge is None:
            return False

        # If currently moving, don't interrupt with LLM thinking
        # (unless there's a new obstacle or safety concern)
        if self.world_model.last_motion != "stop" and self.world_model.current_speed > 0:
            # Only re-consult if something changed
            if not self.world_model.obstacle_warning:
                return False

        # Consult LLM every few cycles when idle
        # (every ~2 seconds, not every 0.5s)
        time_since_last = time.time() - self._last_decision_time
        if time_since_last < 2.0 and self.state != CortexState.THINKING:
            return False

        return True

    async def _consult_llm(self) -> None:
        """Consult the LLM for a decision.

        Builds a prompt from the world model and context window, sends it
        to the LLM, and parses the response into a motion command.

        If the LLM fails, falls back to rule-based behavior.
        """
        self.state = CortexState.THINKING

        # Publish thinking event (for LED/UI feedback)
        await self.bus.publish(Event(
            type=EventType.CORTEX_THINKING,
            data={"world_model": self._world_model_dict()},
            source="cortex",
        ))

        # Build the prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt()

        try:
            # Call the LLM with a timeout
            response = await asyncio.wait_for(
                self.llm_bridge.generate(user_prompt, system_prompt, temperature=0.3),
                timeout=self.config.cloud_timeout_seconds if self.config.mode == LLMMode.OLLAMA_CLOUD
                else self.config.local_timeout_seconds,
            )

            # Parse the LLM response
            action = self._parse_llm_decision(response)

            if action:
                # Add to context window
                self._add_to_context({
                    "timestamp": time.time(),
                    "world_model": self._world_model_dict(),
                    "llm_response": response[:200],
                    "action": action,
                })

                await self._execute_action(action, source="llm")
            else:
                # LLM response was unparseable — stay idle
                logger.debug(f"LLM response unparseable, staying idle: {response[:100]}")

        except asyncio.TimeoutError:
            logger.warning("⏱️  LLM timeout — falling back to rules")
            await self._fallback_to_rules("LLM timeout")
        except Exception as e:
            logger.error(f"LLM error: {e}")
            await self._publish_error(f"LLM error: {e}")
            await self._fallback_to_rules(str(e))

        self.state = CortexState.WAITING

    async def _fallback_to_rules(self, reason: str) -> None:
        """Fall back to rule-based behavior.

        When the LLM is unavailable, the cortex uses simple rules:
        - If obstacle ahead: stop or turn
        - If idle: stay stopped
        - If battery low: stop

        Args:
            reason: Why the fallback was triggered (for logging/events)
        """
        self.state = CortexState.FALLBACK
        self._fallback_count += 1

        await self.bus.publish(Event(
            type=EventType.CORTEX_FALLBACK,
            data={"reason": reason},
            source="cortex",
        ))

        logger.warning(f"🔄 Cortex fallback: {reason}")

        # Simple rule-based behavior
        action = self._check_safety_rules()
        if action:
            await self._execute_action(action, source="fallback_rule")
        else:
            # Default: stay stopped when in fallback
            if self.world_model.last_motion != "stop":
                await self._execute_action(
                    {"direction": "stop", "speed": 0.0, "reason": "fallback_idle"},
                    source="fallback_rule",
                )

    # ─── Action Execution ─────────────────────────────────────────────

    async def _execute_action(self, action: dict[str, Any], source: str = "cortex") -> None:
        """Execute a motion action by publishing a MOTION_COMMAND event.

        CRITICAL: This does NOT directly command servos. It publishes an
        event that the safety governor will intercept and validate. The
        safety governor may modify or block the command.

        Args:
            action: The action dict with direction/action/speed
            source: Who initiated this action (for logging)
        """
        # Update world model
        if "direction" in action:
            self.world_model.last_motion = action["direction"]
            self.world_model.current_speed = action.get("speed", 0.0)
        elif "action" in action:
            # One-shot action (sit, stand, wave, etc.)
            self.world_model.last_motion = action["action"]
            self.world_model.current_speed = 0.0

        self.state = CortexState.ACTING

        # Build the event data
        event_data = {
            **action,
            "source": source,
            "cortex_decision_id": self._decision_count,
        }

        # Publish the decision event (for logging/UI)
        await self.bus.publish(Event(
            type=EventType.CORTEX_DECISION,
            data={
                "action": action,
                "source": source,
                "world_model": self._world_model_dict(),
            },
            source="cortex",
        ))

        # Publish the motion command — safety governor will intercept this
        await self.bus.publish(Event(
            type=EventType.MOTION_COMMAND,
            data=event_data,
            source="cortex",
        ))

        logger.info(
            f"🎯 Decision #{self._decision_count}: {action.get('direction', action.get('action', '?'))} "
            f"(source={source})"
        )

    async def _process_voice_command(self, text: str) -> None:
        """Process a voice command by parsing intent and executing.

        Args:
            text: The transcribed voice command
        """
        logger.info(f"🎙️  Processing voice command: '{text}'")

        try:
            result = await self.intent_parser.parse(text)

            if result.get("confidence", 0.0) > 0:
                # Execute the parsed command
                await self._execute_action(result, source="voice")
            else:
                logger.warning(f"Could not parse voice command: '{text}'")
                await self._publish_error(f"Voice command unparseable: {text}")

        except Exception as e:
            logger.error(f"Voice command processing error: {e}")
            await self._publish_error(f"Voice command error: {e}")

    # ─── Prompt Building ──────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM.

        The system prompt describes:
        - The robot's identity and capabilities
        - Safety rules (never violate these)
        - The expected response format

        Returns:
            The system prompt string.
        """
        return (
            "You are Weaver, an autonomous hexapod spider robot running on a Raspberry Pi 5. "
            "You have 6 legs with 18 servos, ultrasonic distance sensing, IMU (tilt), "
            "battery monitoring, and a camera. Your goal is to navigate safely and respond "
            "to voice commands.\n\n"
            "SAFETY RULES (never violate):\n"
            "- Never move forward if an obstacle is within 30cm\n"
            "- Always stop if a human is within 30cm\n"
            "- Never exceed speed 0.8\n"
            "- If battery is low, minimize movement\n"
            "- If body tilt is high, stop and stabilize\n\n"
            "CAPABILITIES:\n"
            "- Directions: forward, backward, turn_left, turn_right, strafe_left, strafe_right, stop\n"
            "- Actions: sit, stand, wave, crouch\n"
            "- Gait changes: tripod (fast), wave (stable), ripple (medium)\n\n"
            "Respond with ONLY a JSON object on one line:\n"
            '{"direction": "forward", "speed": 0.5}\n'
            "or for actions:\n"
            '{"action": "wave"}\n'
            "or if no action needed:\n"
            '{"direction": "stop", "speed": 0.0}\n\n'
            "Keep responses minimal. Do not add explanations."
        )

    def _build_user_prompt(self) -> str:
        """Build the user prompt with current world model and context.

        Returns:
            The user prompt string with sensor data and recent context.
        """
        world_summary = self.world_model.to_llm_context()

        # Include recent context for continuity
        recent_context = ""
        if self.context_window:
            last = self.context_window[-1]
            recent_context = f"\nPrevious action: {last.get('action', {})}"

        return (
            f"Current situation: {world_summary}\n"
            f"{recent_context}\n"
            f"What should I do next? Respond with a JSON motion command."
        )

    # ─── LLM Response Parsing ─────────────────────────────────────────

    def _parse_llm_decision(self, response: str) -> dict[str, Any] | None:
        """Parse the LLM's response into a motion command.

        The LLM should return a JSON object like:
            {"direction": "forward", "speed": 0.5}
        or
            {"action": "wave"}

        Args:
            response: Raw LLM response text

        Returns:
            Parsed action dict, or None if unparseable.
        """
        response = response.strip()

        # Try direct JSON parse
        try:
            result = json.loads(response)
            if self._validate_action(result):
                return result
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                if self._validate_action(result):
                    return result
            except json.JSONDecodeError:
                pass

        logger.debug(f"Could not parse LLM decision as JSON: {response[:100]}")
        return None

    def _validate_action(self, action: dict[str, Any]) -> bool:
        """Validate that a parsed action dict has the required fields.

        Args:
            action: The action dict to validate

        Returns:
            True if valid, False otherwise.
        """
        valid_directions = {
            "forward", "backward", "turn_left", "turn_right",
            "strafe_left", "strafe_right", "stop",
        }
        valid_actions = {
            "sit", "stand", "wave", "crouch", "dance",
            "gait_change", "body_height",
        }

        if "direction" in action:
            return action["direction"] in valid_directions
        if "action" in action:
            return action["action"] in valid_actions
        return False

    # ─── Context Window ────────────────────────────────────────────────

    def _add_to_context(self, entry: dict[str, Any]) -> None:
        """Add an entry to the rolling context window.

        The context window maintains a history of recent decisions and
        sensor states. This gives the LLM continuity — it can see what
        it decided recently and what happened.

        Args:
            entry: A dict with timestamp, world_model, action, etc.
        """
        self.context_window.append(entry)

    # ─── Sensor Event Handlers ────────────────────────────────────────

    async def _on_ultrasonic_range(self, event: Event) -> None:
        """Update obstacle distance from ultrasonic sensor."""
        dist = event.data.get("distance_cm")
        if dist is not None:
            self.world_model.obstacle_distance_cm = dist
            # Update warning flag
            slow_dist = self.main_config.safety.obstacle_slow_distance_cm
            self.world_model.obstacle_warning = dist < slow_dist
            self.world_model.last_updated = time.time()

    async def _on_ultrasonic_obstacle(self, event: Event) -> None:
        """Obstacle detected event — update world model."""
        dist = event.data.get("distance_cm")
        if dist is not None:
            self.world_model.obstacle_distance_cm = dist
            self.world_model.obstacle_warning = True

    async def _on_imu_data(self, event: Event) -> None:
        """Update tilt from IMU."""
        tilt = event.data.get("tilt_degrees", 0.0)
        self.world_model.body_tilt_degrees = tilt
        self.world_model.tilt_warning = abs(tilt) > self.main_config.safety.max_tilt_degrees
        self.world_model.last_updated = time.time()

    async def _on_tilt_warning(self, event: Event) -> None:
        """Tilt warning event."""
        self.world_model.tilt_warning = True
        self.world_model.body_tilt_degrees = event.data.get("tilt", self.world_model.body_tilt_degrees)

    async def _on_battery_status(self, event: Event) -> None:
        """Update battery voltage."""
        self.world_model.battery_voltage = event.data.get("voltage", self.world_model.battery_voltage)
        self.world_model.battery_low = self.world_model.battery_voltage < self.main_config.battery.low_battery_threshold

    async def _on_battery_low(self, event: Event) -> None:
        """Battery low event."""
        self.world_model.battery_low = True
        self.world_model.battery_voltage = event.data.get("voltage", self.world_model.battery_voltage)

    async def _on_battery_critical(self, event: Event) -> None:
        """Battery critical event — stop everything."""
        self.world_model.battery_low = True
        self.world_model.battery_voltage = event.data.get("voltage", 6.0)

    async def _on_thermal_status(self, event: Event) -> None:
        """Update CPU temperature."""
        self.world_model.cpu_temp_c = event.data.get("temp_c", self.world_model.cpu_temp_c)

    async def _on_thermal_warning(self, event: Event) -> None:
        """Thermal warning event."""
        self.world_model.cpu_temp_c = event.data.get("temp_c", self.world_model.cpu_temp_c)

    async def _on_object_detected(self, event: Event) -> None:
        """Object detected by camera."""
        obj = event.data.get("object", "")
        if obj:
            # Keep last 5 detected objects
            self.world_model.detected_objects = (
                [obj] + self.world_model.detected_objects[:4]
            )

    async def _on_face_detected(self, event: Event) -> None:
        """Human/face detected by camera — update safety model."""
        self.world_model.human_detected = True
        self.world_model.human_distance_cm = event.data.get("distance_cm")

    async def _on_color_tracked(self, event: Event) -> None:
        """Color tracking result."""
        self.world_model.tracked_color = event.data.get("color", "")

    async def _on_safety_estop(self, event: Event) -> None:
        """Emergency stop — immediately stop the robot."""
        self.world_model.last_motion = "stop"
        self.world_model.current_speed = 0.0
        logger.error("🛑 Cortex received e-stop — halting all motion")

    async def _on_collision_imminent(self, event: Event) -> None:
        """Collision imminent — stop immediately."""
        logger.warning("⚠️  Collision imminent — cortex stopping")
        await self._execute_action(
            {"direction": "stop", "speed": 0.0, "reason": "collision_imminent"},
            source="safety_rule",
        )

    async def _on_voice_transcript(self, event: Event) -> None:
        """Voice transcript received — queue for processing.

        We don't process immediately because we're in the decision loop.
        Instead, we queue it and the next loop iteration will process it.
        """
        text = event.data.get("text", "").strip()
        if text:
            self._pending_voice_command = text
            logger.debug(f"Voice command queued: '{text}'")

    async def _on_motion_completed(self, event: Event) -> None:
        """Motion completed — update state."""
        self.world_model.last_motion = "stop"
        self.world_model.current_speed = 0.0
        self.state = CortexState.WAITING

    async def _on_motion_aborted(self, event: Event) -> None:
        """Motion was aborted (likely by safety governor)."""
        reason = event.data.get("reason", "unknown")
        self.world_model.last_motion = "stop"
        self.world_model.current_speed = 0.0
        self.state = CortexState.WAITING
        logger.warning(f"⚠️  Motion aborted: {reason}")
        await self._publish_error(f"Motion aborted: {reason}")

    # ─── Utilities ────────────────────────────────────────────────────

    def _world_model_dict(self) -> dict[str, Any]:
        """Convert world model to a dict for events/telemetry."""
        return {
            "obstacle_distance_cm": self.world_model.obstacle_distance_cm,
            "obstacle_warning": self.world_model.obstacle_warning,
            "body_tilt_degrees": self.world_model.body_tilt_degrees,
            "tilt_warning": self.world_model.tilt_warning,
            "battery_voltage": self.world_model.battery_voltage,
            "battery_low": self.world_model.battery_low,
            "cpu_temp_c": self.world_model.cpu_temp_c,
            "human_detected": self.world_model.human_detected,
            "human_distance_cm": self.world_model.human_distance_cm,
            "last_motion": self.world_model.last_motion,
            "current_speed": self.world_model.current_speed,
        }

    async def _publish_error(self, message: str) -> None:
        """Publish a cortex error event.

        Args:
            message: Error description
        """
        self._error_count += 1
        await self.bus.publish(Event(
            type=EventType.CORTEX_ERROR,
            data={"message": message, "error_count": self._error_count},
            source="cortex",
        ))

    def get_stats(self) -> dict[str, Any]:
        """Get cortex statistics for telemetry/debugging.

        Returns:
            Dict with decision counts, error counts, state, etc.
        """
        return {
            "state": self.state.value,
            "mode": self.config.mode.value,
            "decision_count": self._decision_count,
            "error_count": self._error_count,
            "fallback_count": self._fallback_count,
            "context_window_size": len(self.context_window),
            "llm_stats": self.llm_bridge.get_stats() if self.llm_bridge else None,
            "world_model": self._world_model_dict(),
        }