# leds.py — WS2812 RGB LED status indicators
#
# Controls the 2x WS2812 RGB LEDs on the RaspClaws body.
# Colors indicate robot state:
#   Blue   = idle / waiting
#   Green  = moving
#   Orange = warning (low battery, obstacle, etc.)
#   Red    = danger / e-stop / error
#   Purple = LLM thinking
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from weaver.config import LEDConfig, get_config
from weaver.event_bus import Event, EventBus, EventType, get_event_bus


class LEDController:
    """WS2812 RGB LED controller for status indication.
    
    In real mode: uses rpi_ws2812 library.
    In mock mode: logs color changes to console.
    """
    
    def __init__(self, config: LEDConfig | None = None):
        self.config = config or get_config().led
        self.bus = get_event_bus()
        self._strip = None
        self._running = False
        self._current_color = (0, 0, 255)  # Start blue (idle)
        self._blinking = False
        self._blink_task: asyncio.Task | None = None
        
    async def start(self) -> None:
        """Initialize the LED strip."""
        try:
            import board
            import neopixel
            
            self._strip = neopixel.NeoPixel(
                board.D12,              # GPIO 12 (PWM0)
                self.config.led_count,
                brightness=self.config.brightness,
                auto_write=False,
            )
            self._set_color(*self.config.color_idle)
            logger.info("💡 WS2812 LEDs initialized ({} LEDs)", self.config.led_count)
        except ImportError:
            logger.warning("💡 LEDs: rpi_ws2812 not installed — using mock LEDs")
            self._strip = None
        except Exception as e:
            logger.warning(f"💡 LED init failed: {e} — using mock LEDs")
            self._strip = None
        
        # Subscribe to events for status changes
        self.bus.subscribe(EventType.MOTION_STARTED, self._on_motion_start)
        self.bus.subscribe(EventType.MOTION_COMPLETED, self._on_motion_complete)
        self.bus.subscribe(EventType.SAFETY_ESTOP, self._on_estop)
        self.bus.subscribe(EventType.SAFETY_WARNING, self._on_warning)
        self.bus.subscribe(EventType.CORTEX_THINKING, self._on_thinking)
        self.bus.subscribe(EventType.CORTEX_DECISION, self._on_decision)
        self.bus.subscribe(EventType.BATTERY_LOW, self._on_battery_low)
        self.bus.subscribe(EventType.BATTERY_CRITICAL, self._on_battery_critical)
        self.bus.subscribe(EventType.SYSTEM_ERROR, self._on_error)
        
        self._running = True
        logger.info("💡 LED controller started")
    
    async def stop(self) -> None:
        """Stop the LED controller."""
        self._running = False
        if self._blink_task:
            self._blink_task.cancel()
            try:
                await self._blink_task
            except asyncio.CancelledError:
                pass
        # Turn off LEDs
        self._set_color(0, 0, 0)
        logger.info("LED controller stopped")
    
    def _set_color(self, r: int, g: int, b: int) -> None:
        """Set all LEDs to a color."""
        r = int(r * self.config.brightness)
        g = int(g * self.config.brightness)
        b = int(b * self.config.brightness)
        self._current_color = (r, g, b)
        
        if self._strip:
            for i in range(self.config.led_count):
                self._strip[i] = (r, g, b)
            self._strip.show()
        else:
            logger.debug(f"LED color: rgb({r},{g},{b})")
    
    async def _blink(self, r: int, g: int, b: int, interval: float = 0.5) -> None:
        """Blink LEDs between a color and off."""
        on = True
        while self._blinking and self._running:
            if on:
                self._set_color(r, g, b)
            else:
                self._set_color(0, 0, 0)
            on = not on
            await asyncio.sleep(interval)
    
    def _start_blink(self, r: int, g: int, b: int, interval: float = 0.5) -> None:
        """Start blinking."""
        self._blinking = True
        if self._blink_task:
            self._blink_task.cancel()
        self._blink_task = asyncio.create_task(self._blink(r, g, b, interval))
    
    def _stop_blink(self) -> None:
        """Stop blinking."""
        self._blinking = False
        if self._blink_task:
            self._blink_task.cancel()
            self._blink_task = None
    
    # ─── Event Handlers ───────────────────────────────────────────────
    
    async def _on_motion_start(self, event: Event) -> None:
        self._stop_blink()
        self._set_color(*self.config.color_moving)
    
    async def _on_motion_complete(self, event: Event) -> None:
        self._stop_blink()
        self._set_color(*self.config.color_idle)
    
    async def _on_estop(self, event: Event) -> None:
        self._start_blink(*self.config.color_danger, interval=0.3)
    
    async def _on_warning(self, event: Event) -> None:
        self._stop_blink()
        self._set_color(*self.config.color_warning)
    
    async def _on_thinking(self, event: Event) -> None:
        self._stop_blink()
        self._set_color(*self.config.color_thinking)
    
    async def _on_decision(self, event: Event) -> None:
        # Return to idle after decision made
        self._stop_blink()
        self._set_color(*self.config.color_idle)
    
    async def _on_battery_low(self, event: Event) -> None:
        self._stop_blink()
        self._set_color(*self.config.color_warning)
    
    async def _on_battery_critical(self, event: Event) -> None:
        self._start_blink(*self.config.color_danger, interval=0.5)
    
    async def _on_error(self, event: Event) -> None:
        self._start_blink(*self.config.color_error, interval=0.2)