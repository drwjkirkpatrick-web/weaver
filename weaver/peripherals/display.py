# display.py — OLED display controller (SSD1306)
#
# Shows robot status on the 0.96" OLED display mounted on the RaspClaws body.
# Cycles through: battery, sensors, safety, IP address.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from weaver.config import OLEDConfig, get_config
from weaver.event_bus import Event, EventBus, EventType, get_event_bus


class OLEDDisplay:
    """OLED display controller for status readout.
    
    In real mode: uses adafruit-circuitpython-ssd1306 over I2C.
    In mock mode: logs display content to console.
    """
    
    def __init__(self, config: OLEDConfig | None = None):
        self.config = config or get_config().oled
        self.bus = get_event_bus()
        self._display = None
        self._running = False
        self._current_page = 0
        self._pages = ["status", "sensors", "network"]
        self._page_interval = 5.0  # seconds per page
        self._last_page_switch = time.time()
        
        # Latest data (updated via event subscriptions)
        self._battery_voltage = 8.4
        self._battery_percent = 100.0
        self._obstacle_distance: float | None = None
        self._body_tilt = 0.0
        self._motion_status = "idle"
        self._safety_level = "CHILD"
        self._estop = False
        self._cpu_temp = 45.0
        self._llm_mode = "ollama_cloud"
        
    async def start(self) -> None:
        """Initialize the OLED display."""
        try:
            import board
            import busio
            import adafruit_ssd1306
            
            i2c = busio.I2C(board.SCL, board.SDA)
            self._display = adafruit_ssd1306.SSD1306_I2C(
                self.config.width, self.config.height, i2c,
                addr=self.config.i2c_address
            )
            self._display.fill(0)
            self._display.show()
            logger.info("📱 OLED display initialized (SSD1306)")
        except ImportError:
            logger.warning("📱 OLED: adafruit-circuitpython-ssd1306 not installed — using mock display")
            self._display = None
        except Exception as e:
            logger.warning(f"📱 OLED init failed: {e} — using mock display")
            self._display = None
        
        # Subscribe to events for data updates
        self.bus.subscribe(EventType.BATTERY_STATUS, self._on_battery)
        self.bus.subscribe(EventType.ULTRASONIC_RANGE, self._on_ultrasonic)
        self.bus.subscribe(EventType.IMU_DATA, self._on_imu)
        self.bus.subscribe(EventType.MOTION_STARTED, self._on_motion)
        self.bus.subscribe(EventType.THERMAL_STATUS, self._on_thermal)
        self.bus.subscribe(EventType.SAFETY_ESTOP, self._on_estop)
        
        self._running = True
        asyncio.create_task(self._display_loop())
        logger.info("📱 OLED display controller started")
    
    async def stop(self) -> None:
        """Stop the display controller."""
        self._running = False
        if self._display:
            self._display.fill(0)
            self._display.text("Weaver", 0, 0, 1)
            self._display.text("OFFLINE", 0, 16, 1)
            self._display.show()
        logger.info("OLED display stopped")
    
    async def _display_loop(self) -> None:
        """Main display update loop."""
        while self._running:
            try:
                # Cycle pages
                if time.time() - self._last_page_switch > self._page_interval:
                    self._current_page = (self._current_page + 1) % len(self._pages)
                    self._last_page_switch = time.time()
                
                page = self._pages[self._current_page]
                
                if page == "status":
                    self._render_status()
                elif page == "sensors":
                    self._render_sensors()
                elif page == "network":
                    self._render_network()
                
                await asyncio.sleep(1.0 / self.config.update_rate_hz)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"OLED display error: {e}")
                await asyncio.sleep(1.0)
    
    def _render_status(self) -> None:
        """Render the status page."""
        if self._display:
            self._display.fill(0)
            self._display.text("WEAVER", 0, 0, 1)
            self._display.text(f"Mode: {self._motion_status}", 0, 12, 1)
            self._display.text(f"Safe: {self._safety_level}", 0, 24, 1)
            self._display.text(f"LLM: {self._llm_mode[:8]}", 0, 36, 1)
            if self._estop:
                self._display.text("** E-STOP **", 0, 52, 1)
            self._display.show()
        else:
            logger.debug(f"OLED [status]: motion={self._motion_status} safety={self._safety_level} estop={self._estop}")
    
    def _render_sensors(self) -> None:
        """Render the sensors page."""
        if self._display:
            self._display.fill(0)
            self._display.text("SENSORS", 0, 0, 1)
            self._display.text(f"Bat: {self._battery_voltage:.1f}V", 0, 12, 1)
            obs = f"{self._obstacle_distance:.0f}cm" if self._obstacle_distance else "---cm"
            self._display.text(f"Obs: {obs}", 0, 24, 1)
            self._display.text(f"Tilt: {self._body_tilt:.1f}d", 0, 36, 1)
            self._display.text(f"CPU: {self._cpu_temp:.0f}C", 0, 48, 1)
            self._display.show()
        else:
            logger.debug(f"OLED [sensors]: bat={self._battery_voltage:.1f}V obs={self._obstacle_distance} tilt={self._body_tilt:.1f} cpu={self._cpu_temp:.0f}")
    
    def _render_network(self) -> None:
        """Render the network page."""
        import socket
        try:
            hostname = socket.gethostname()
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            ip = "unknown"
        
        if self._display:
            self._display.fill(0)
            self._display.text("NETWORK", 0, 0, 1)
            self._display.text(f"Host: {hostname}", 0, 12, 1)
            self._display.text(f"IP: {ip}", 0, 24, 1)
            self._display.text(f"Port: {get_config().web.port}", 0, 36, 1)
            self._display.text(f"http://{ip}:8080", 0, 52, 1)
            self._display.show()
        else:
            logger.debug(f"OLED [network]: host={hostname} ip={ip}")
    
    # ─── Event Handlers ───────────────────────────────────────────────
    
    async def _on_battery(self, event: Event) -> None:
        self._battery_voltage = event.data.get("voltage", 8.4)
        self._battery_percent = event.data.get("percent", 100.0)
    
    async def _on_ultrasonic(self, event: Event) -> None:
        self._obstacle_distance = event.data.get("distance_cm")
    
    async def _on_imu(self, event: Event) -> None:
        self._body_tilt = event.data.get("tilt_degrees", 0.0)
    
    async def _on_motion(self, event: Event) -> None:
        self._motion_status = event.data.get("status", "idle")
    
    async def _on_thermal(self, event: Event) -> None:
        self._cpu_temp = event.data.get("temp_c", 45.0)
    
    async def _on_estop(self, event: Event) -> None:
        self._estop = True