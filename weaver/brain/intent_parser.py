# intent_parser.py — Natural language → robot action parser
#
# This module parses voice/text commands into structured motion commands that
# the motion controller and safety governor can understand.
#
# Two modes:
# 1. RULE-BASED (fast, no LLM needed): Handles common commands like
#    "walk forward", "turn left", "stop", "sit down" using regex/keyword
#    matching. Instant, reliable, no network needed.
# 2. LLM-BASED (powerful, uses LLM): For complex/natural commands like
#    "go to the kitchen" or "find the red ball" — uses the LLM to interpret
#    intent and generate appropriate motion commands.
#
# The parser tries rule-based first (fast path). If the rule-based parser
# can't match with sufficient confidence, it falls back to the LLM.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
import time
from typing import Any

from loguru import logger

from weaver.config import CortexConfig, get_config


# ─── Action schemas ─────────────────────────────────────────────────────
# These define the structure of parsed motion commands.
# The motion controller expects these keys in the MOTION_COMMAND event data.

# Direction commands (continuous motion):
#   {"direction": "forward", "speed": 0.5, "duration": 2.0}
#   {"direction": "backward", "speed": 0.3, "duration": 1.0}
#   {"direction": "turn_left", "speed": 0.3}
#   {"direction": "turn_right", "speed": 0.3}
#   {"direction": "strafe_left", "speed": 0.3}
#   {"direction": "strafe_right", "speed": 0.3}
#   {"direction": "stop"}

# Action commands (one-shot poses):
#   {"action": "sit"}
#   {"action": "stand"}
#   {"action": "wave"}
#   {"action": "crouch"}
#   {"action": "body_height", "height_mm": 60.0}
#   {"action": "gait_change", "gait": "tripod"}


class IntentParser:
    """Parse natural language commands into structured robot actions.

    Uses a two-tier strategy:
    1. Rule-based matching for common commands (fast, deterministic)
    2. LLM-based parsing for complex/natural commands (flexible, slower)

    The rule-based parser uses keyword + regex matching. It handles the
    most common commands that a user would give a hexapod robot.

    The LLM-based parser sends the command to the LLM with a structured
    output prompt. This handles commands like "carefully walk to the door"
    or "do a little dance".

    Attributes:
        config: CortexConfig for confidence threshold settings
        llm_bridge: The LLM bridge (OllamaCloud or OllamaLocal) for complex parsing
        _rule_patterns: Compiled regex patterns for rule-based matching

    Example:
        parser = IntentParser(llm_bridge=bridge)
        await parser.start()

        result = await parser.parse("walk forward slowly")
        # result = {"direction": "forward", "speed": 0.3, "confidence": 0.95}

        result = await parser.parse("go check out that sound")
        # Falls back to LLM, returns complex action
    """

    def __init__(self, llm_bridge: Any = None, config: CortexConfig | None = None) -> None:
        """Initialize the intent parser.

        Args:
            llm_bridge: An LLM bridge (OllamaCloudBridge or OllamaLocalBridge)
                       for complex command parsing. Can be None for rule-only mode.
            config: CortexConfig instance. If None, loads from global config.
        """
        self.config: CortexConfig = config or get_config().cortex
        self.llm_bridge = llm_bridge
        self._parse_count: int = 0
        self._llm_parse_count: int = 0
        self._rule_parse_count: int = 0
        self._failed_count: int = 0

        # Build rule patterns — each tuple is (compiled_regex, handler_function)
        self._rule_patterns: list[tuple[re.Pattern, Any]] = self._build_rule_patterns()

    async def start(self) -> None:
        """Start the intent parser.

        No async initialization needed currently, but included for interface
        consistency with other Weaver modules.
        """
        mode = "rule+LLM" if self.llm_bridge else "rule-only"
        logger.info(f"🎯 Intent parser started ({mode} mode)")

    async def stop(self) -> None:
        """Stop the intent parser."""
        logger.info(
            f"Intent parser stopped "
            f"(parsed={self._parse_count}, rules={self._rule_parse_count}, "
            f"llm={self._llm_parse_count}, failed={self._failed_count})"
        )

    async def parse(self, text: str) -> dict[str, Any]:
        """Parse a natural language command into a structured action.

        This is the main entry point. It first tries the fast rule-based
        parser. If confidence is below the threshold, it falls back to LLM.

        Args:
            text: The voice or text command (e.g., "walk forward slowly")

        Returns:
            A dict with the parsed command. Always includes:
            - "raw_text": The original input
            - "confidence": 0.0-1.0 confidence score
            - "source": "rule" or "llm"

            For motion commands:
            - "direction": forward/backward/turn_left/turn_right/strafe_left/strafe_right/stop
            - "speed": float 0.0-1.0
            - "duration": float (seconds, optional)

            For action commands:
            - "action": sit/stand/wave/crouch/etc.

            If parsing fails:
            - "error": str, "confidence": 0.0
        """
        self._parse_count += 1
        text_lower = text.lower().strip()

        if not text_lower:
            self._failed_count += 1
            return {"raw_text": text, "confidence": 0.0, "source": "rule", "error": "empty"}

        # ─── Step 1: Try rule-based parsing (fast path) ───
        rule_result = self._rule_based_parse(text_lower)
        if rule_result["confidence"] >= self.config.intent_confidence_threshold:
            self._rule_parse_count += 1
            logger.debug(f"🎯 Rule parsed: '{text}' → {rule_result}")
            return rule_result

        # ─── Step 2: Try LLM-based parsing (slow path) ───
        if self.llm_bridge is not None:
            try:
                llm_result = await self._llm_based_parse(text)
                self._llm_parse_count += 1
                logger.debug(f"🤖 LLM parsed: '{text}' → {llm_result}")
                return llm_result
            except Exception as e:
                logger.warning(f"LLM intent parsing failed: {e}")

        # ─── Step 3: Return best rule-based guess or error ───
        if rule_result["confidence"] > 0:
            # Return the low-confidence rule result as a best guess
            self._rule_parse_count += 1
            return rule_result

        self._failed_count += 1
        logger.warning(f"❌ Could not parse intent: '{text}'")
        return {
            "raw_text": text,
            "confidence": 0.0,
            "source": "none",
            "error": "unparseable",
        }

    # ─── Rule-Based Parser ────────────────────────────────────────────

    def _build_rule_patterns(self) -> list[tuple[re.Pattern, Any]]:
        """Build the regex patterns for rule-based intent matching.

        Returns a list of (compiled_pattern, handler_function) tuples.
        Each handler takes the regex match object and returns a result dict.

        LEARNING NOTE: We use word boundaries (\b) in patterns to avoid
        matching substrings. "stop" shouldn't match inside "stopping".
        Pattern order matters — more specific patterns should come first.
        """
        patterns: list[tuple[re.Pattern, Any]] = []

        # ── Stop ──
        patterns.append((
            re.compile(r'\b(stop|halt|freeze|brake|cease|hold)\b'),
            self._handle_stop,
        ))

        # ── Sit / crouch ──
        patterns.append((
            re.compile(r'\b(sit|sit down|crouch|get down|lower body|lie down)\b'),
            self._handle_sit,
        ))

        # ── Stand up ──
        patterns.append((
            re.compile(r'\b(stand|stand up|get up|rise|upright)\b'),
            self._handle_stand,
        ))

        # ── Wave ──
        patterns.append((
            re.compile(r'\b(wave|hello|greet|say hi)\b'),
            self._handle_wave,
        ))

        # ── Walk backward ──  (MUST be before forward — "walk" alone = forward)
        patterns.append((
            re.compile(r'\b(backward|back up|reverse|backwards|retreat)\b'),
            self._handle_backward,
        ))

        # ── Walk forward ──
        patterns.append((
            re.compile(
                r'\b(walk|go|move|step|crawl|forward|ahead|straight)\b'
                r'.*\b(forward|ahead|straight|forward|onward|front)\b'
                r'|\b(forward|ahead|onward)\b'
                r'|\b(walk|go|move|step)\s*(it)?\b'  # "walk" alone = forward
            ),
            self._handle_forward,
        ))

        # ── Turn left ──
        patterns.append((
            re.compile(r'\b(turn|rotate|spin|pivot)\s*(left|port)\b|^\s*left\b|\bgo left\b'),
            self._handle_turn_left,
        ))

        # ── Turn right ──
        patterns.append((
            re.compile(r'\b(turn|rotate|spin|pivot)\s*(right|starboard)\b|^\s*right\b|\bgo right\b'),
            self._handle_turn_right,
        ))

        # ── Strafe left ──
        patterns.append((
            re.compile(r'\b(strafe|slide|sidestep)\s*(left|port)\b'),
            self._handle_strafe_left,
        ))

        # ── Strafe right ──
        patterns.append((
            re.compile(r'\b(strafe|slide|sidestep)\s*(right|starboard)\b'),
            self._handle_strafe_right,
        ))

        # ── Speed modifiers ──
        # These don't produce commands by themselves, they modify speed
        # Handled in _extract_speed()

        # ── Gait change ──
        patterns.append((
            re.compile(r'\b(tripod|wave gait|ripple gait|change gait)\b'),
            self._handle_gait_change,
        ))

        # ── Body height ──
        patterns.append((
            re.compile(r'\b(body height|stand tall|stand high|raise body|lower body height)\b'),
            self._handle_body_height,
        ))

        return patterns

    def _rule_based_parse(self, text: str) -> dict[str, Any]:
        """Run the rule-based parser on the text.

        Tries each pattern in order. First match wins.
        Also extracts speed and duration modifiers.

        Args:
            text: Lowercase command text

        Returns:
            Parsed result dict with confidence score.
        """
        for pattern, handler in self._rule_patterns:
            match = pattern.search(text)
            if match:
                result = handler(match)
                result["raw_text"] = text
                result["source"] = "rule"

                # Extract speed modifier (overrides default)
                speed_mod = self._extract_speed(text)
                if speed_mod is not None and "speed" in result:
                    result["speed"] = speed_mod

                # Extract duration
                duration = self._extract_duration(text)
                if duration is not None:
                    result["duration"] = duration

                return result

        # No pattern matched
        return {"raw_text": text, "confidence": 0.0, "source": "rule"}

    def _extract_speed(self, text: str) -> float | None:
        """Extract speed modifier from text.

        Keywords like "slowly", "fast", "quickly" map to speed values.

        Args:
            text: Lowercase command text

        Returns:
            Speed float (0.0-1.0) or None if no modifier found.
        """
        # Explicit speed percentages: "at 50%", "at 80 percent"
        pct_match = re.search(r'(\d+)\s*(?:%|percent)', text)
        if pct_match:
            pct = int(pct_match.group(1))
            return max(0.0, min(1.0, pct / 100.0))

        # Keyword-based speed
        speed_keywords: list[tuple[str, float]] = [
            (r'\b(very fast|fastest|full speed|maximum)\b', 0.8),
            (r'\b(fast|quick|quickly|rapid|hurry)\b', 0.7),
            (r'\b(normal|regular|default|standard)\b', 0.5),
            (r'\b(slowly|slow|gentle|careful|cautious)\b', 0.3),
            (r'\b(very slow|slowest|creep|crawl|inch)\b', 0.15),
        ]
        for pattern, speed in speed_keywords:
            if re.search(pattern, text):
                return speed

        return None

    def _extract_duration(self, text: str) -> float | None:
        """Extract duration from text.

        Keywords like "for 3 seconds", "for a bit", "briefly"

        Args:
            text: Lowercase command text

        Returns:
            Duration in seconds, or None if not specified.
        """
        # Explicit duration: "for 3 seconds", "for 2.5 secs"
        dur_match = re.search(r'for\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)', text)
        if dur_match:
            return float(dur_match.group(1))

        # Implicit duration keywords
        duration_keywords: list[tuple[str, float]] = [
            (r'\b(briefly|quick tap|momentarily)\b', 0.5),
            (r'\b(for a (bit|moment|sec|second))\b', 1.0),
            (r'\b(for a while|for some time|keep going)\b', 5.0),
        ]
        for pattern, dur in duration_keywords:
            if re.search(pattern, text):
                return dur

        return None

    # ─── Rule Handlers ───────────────────────────────────────────────

    def _handle_stop(self, match: re.Match) -> dict[str, Any]:
        """Handle stop/halt commands."""
        return {"direction": "stop", "speed": 0.0, "confidence": 0.95}

    def _handle_sit(self, match: re.Match) -> dict[str, Any]:
        """Handle sit/crouch commands."""
        return {"action": "sit", "confidence": 0.95}

    def _handle_stand(self, match: re.Match) -> dict[str, Any]:
        """Handle stand up commands."""
        return {"action": "stand", "confidence": 0.95}

    def _handle_wave(self, match: re.Match) -> dict[str, Any]:
        """Handle wave/greet commands."""
        return {"action": "wave", "confidence": 0.90}

    def _handle_forward(self, match: re.Match) -> dict[str, Any]:
        """Handle walk forward commands."""
        return {"direction": "forward", "speed": 0.5, "confidence": 0.90}

    def _handle_backward(self, match: re.Match) -> dict[str, Any]:
        """Handle walk backward commands."""
        return {"direction": "backward", "speed": 0.3, "confidence": 0.90}

    def _handle_turn_left(self, match: re.Match) -> dict[str, Any]:
        """Handle turn left commands."""
        return {"direction": "turn_left", "speed": 0.3, "confidence": 0.90}

    def _handle_turn_right(self, match: re.Match) -> dict[str, Any]:
        """Handle turn right commands."""
        return {"direction": "turn_right", "speed": 0.3, "confidence": 0.90}

    def _handle_strafe_left(self, match: re.Match) -> dict[str, Any]:
        """Handle strafe left commands."""
        return {"direction": "strafe_left", "speed": 0.3, "confidence": 0.85}

    def _handle_strafe_right(self, match: re.Match) -> dict[str, Any]:
        """Handle strafe right commands."""
        return {"direction": "strafe_right", "speed": 0.3, "confidence": 0.85}

    def _handle_gait_change(self, match: re.Match) -> dict[str, Any]:
        """Handle gait change commands."""
        text = match.group(0)
        if "tripod" in text:
            return {"action": "gait_change", "gait": "tripod", "confidence": 0.85}
        elif "wave" in text:
            return {"action": "gait_change", "gait": "wave", "confidence": 0.85}
        elif "ripple" in text:
            return {"action": "gait_change", "gait": "ripple", "confidence": 0.85}
        return {"action": "gait_change", "gait": "tripod", "confidence": 0.60}

    def _handle_body_height(self, match: re.Match) -> dict[str, Any]:
        """Handle body height commands."""
        text = match.group(0)
        if "tall" in text or "high" in text or "raise" in text:
            return {"action": "body_height", "height_mm": 110.0, "confidence": 0.80}
        return {"action": "body_height", "height_mm": 60.0, "confidence": 0.75}

    # ─── LLM-Based Parser ─────────────────────────────────────────────

    async def _llm_based_parse(self, text: str) -> dict[str, Any]:
        """Use the LLM to parse a complex command.

        Sends the command to the LLM with a structured output prompt.
        The LLM returns a JSON-like structure that we parse.

        Args:
            text: The original command text

        Returns:
            Parsed result dict with confidence and source="llm".

        LEARNING NOTE: We instruct the LLM to return ONLY a JSON object.
        This makes parsing reliable. If the LLM adds extra text, we try
        to extract the JSON portion.
        """
        system_prompt = self._build_llm_system_prompt()
        user_prompt = self._build_llm_user_prompt(text)

        response = await self.llm_bridge.generate(user_prompt, system_prompt, temperature=0.1)

        # Parse the LLM response as JSON
        result = self._parse_llm_json(response, text)
        result["source"] = "llm"
        return result

    def _build_llm_system_prompt(self) -> str:
        """Build the system prompt for LLM-based intent parsing.

        Returns:
            System prompt string describing the expected output format.
        """
        return (
            "You are the intent parser for Weaver, a hexapod spider robot. "
            "Your job is to translate natural language commands into a JSON motion command.\n\n"
            "Respond with ONLY a JSON object, no markdown, no explanation.\n\n"
            "Valid JSON schemas:\n"
            '  Direction: {"direction": "forward|backward|turn_left|turn_right|strafe_left|strafe_right|stop", "speed": 0.0-1.0}\n'
            '  Action: {"action": "sit|stand|wave|crouch|dance"}\n'
            '  Gait: {"action": "gait_change", "gait": "tripod|wave|ripple"}\n'
            '  Height: {"action": "body_height", "height_mm": 40-120}\n\n'
            "If the command is unclear, respond with: "
            '{"error": "unclear", "confidence": 0.3}\n\n'
            "Always include a confidence field (0.0-1.0). "
            "Default speed is 0.5. Use lower speed (0.3) for cautious/slow commands."
        )

    def _build_llm_user_prompt(self, text: str) -> str:
        """Build the user prompt for LLM-based intent parsing."""
        return f'Parse this command into a JSON motion command:\n"{text}"'

    def _parse_llm_json(self, response: str, original_text: str) -> dict[str, Any]:
        """Parse the LLM's JSON response.

        Handles cases where the LLM wraps JSON in markdown or adds extra text.

        Args:
            response: Raw LLM response text
            original_text: Original command (for fallback)

        Returns:
            Parsed dict with confidence score.
        """
        import json

        # Try direct JSON parse first
        try:
            result = json.loads(response.strip())
            if isinstance(result, dict):
                result.setdefault("confidence", 0.7)
                result["raw_text"] = original_text
                return result
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from the response (LLM may add wrapper text)
        # Look for first { ... } block
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                if isinstance(result, dict):
                    result.setdefault("confidence", 0.6)
                    result["raw_text"] = original_text
                    return result
            except json.JSONDecodeError:
                pass

        # Could not parse — return error
        logger.warning(f"Could not parse LLM intent response as JSON: {response[:100]}")
        return {
            "raw_text": original_text,
            "confidence": 0.0,
            "source": "llm",
            "error": "json_parse_failed",
            "llm_response": response[:200],
        }

    # ─── Utility ──────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get parser statistics for telemetry.

        Returns:
            Dict with parse counts and success rates.
        """
        return {
            "total_parsed": self._parse_count,
            "rule_parsed": self._rule_parse_count,
            "llm_parsed": self._llm_parse_count,
            "failed": self._failed_count,
            "rule_rate": (
                self._rule_parse_count / self._parse_count
                if self._parse_count > 0
                else 0.0
            ),
            "llm_available": self.llm_bridge is not None,
        }