#!/usr/bin/env python3
# main.py — Weaver Robot Brain main entry point
#
# This is the top-level orchestrator. It:
# 1. Loads configuration
# 2. Initializes all subsystems (sensors, brain, motion, peripherals, web)
# 3. Manages startup/shutdown order
# 4. Runs the main asyncio event loop
#
# Startup order (dependencies flow downward):
#   Config → EventBus → Safety → Sensors → Motion → Brain → Voice → Peripherals → Web
#
# Shutdown order (reverse):
#   Web → Peripherals → Voice → Brain → Motion → Sensors → Safety → EventBus
#
# Usage:
#   python -m weaver.main                    # Run with mock hardware
#   WEAVER_HARDWARE_MODE=real python -m weaver.main  # Run on Pi
#   python -m weaver.main --web-only         # Just the web dashboard (for testing)
#   python -m weaver.main --safety child     # Override safety level
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import signal
import sys
import time
from typing import Any

from loguru import logger

from weaver import __version__
from weaver.config import (
    HardwareMode,
    LLMMode,
    SafetyLevel,
    get_config,
    reload_config,
)
from weaver.event_bus import Event, EventBus, EventType, get_event_bus


# ─── Logger Setup ───────────────────────────────────────────────────────


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    # Also log to file
    logger.add(
        "weaver/data/weaver.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
    )


# ─── Robot Brain Orchestrator ───────────────────────────────────────────


class WeaverBrain:
    """Top-level robot brain orchestrator.
    
    Manages the lifecycle of all subsystems:
    - Safety Governor (always first to start, last to stop)
    - Sensor modules (camera, IMU, ultrasonic, battery, thermal)
    - Motion system (servo driver, gait engine, kinematics, posture)
    - Brain (LLM cortex, intent parser)
    - Voice pipeline (STT, TTS)
    - Peripherals (OLED, LEDs, telemetry)
    - Web dashboard (FastAPI + WebSocket)
    """
    
    def __init__(self):
        self.config = get_config()
        self.bus = get_event_bus()
        self.modules: dict[str, Any] = {}
        self._running = False
        self._start_time = time.time()
    
    async def start(self) -> None:
        """Start all subsystems in dependency order."""
        mode = self.config.hardware_mode
        logger.info(
            "🕷️  Weaver v{} starting up (mode: {}, safety: {})",
            __version__, mode.value, self.config.safety.level.value,
        )
        
        # 1. Safety Governor (CRITICAL — must be first)
        from weaver.safety import SafetyGovernor
        self.modules["safety"] = SafetyGovernor(self.config.safety)
        await self.modules["safety"].start()
        
        # Publish startup event
        await self.bus.publish(Event(
            type=EventType.SYSTEM_STARTUP,
            data={"version": __version__, "mode": mode.value},
            source="main",
        ))
        
        if mode == HardwareMode.MOCK:
            logger.info("🧪 Running in MOCK mode — all hardware simulated")
        else:
            logger.info("🔧 Running in REAL hardware mode")
        
        # 2. Sensor modules
        await self._start_sensors()
        
        # 3. Motion system
        await self._start_motion()
        
        # 4. Brain (cortex)
        await self._start_brain()
        
        # 5. Voice pipeline
        await self._start_voice()
        
        # 6. Peripherals
        await self._start_peripherals()
        
        # 6.5. Keepalive (keeps STT/TTS/LLM warm)
        await self._start_keepalive()
        
        # 7. Web dashboard (last — depends on all above)
        await self._start_web()
        
        self._running = True
        await self.bus.publish(Event(
            type=EventType.SYSTEM_READY,
            data={"uptime": time.time() - self._start_time},
            source="main",
        ))
        
        elapsed = time.time() - self._start_time
        logger.info("✅ Weaver is ready! (startup took {:.1f}s)", elapsed)
        logger.info(
            "🌐 Dashboard: http://{}:{}/",
            self._get_ip(),
            self.config.web.port,
        )
    
    async def stop(self) -> None:
        """Stop all subsystems in reverse order."""
        logger.info("🕷️  Weaver shutting down...")
        self._running = False
        
        await self.bus.publish(Event(
            type=EventType.SYSTEM_SHUTDOWN,
            data={},
            source="main",
        ))
        
        # Stop in reverse order
        for name in reversed(list(self.modules.keys())):
            module = self.modules[name]
            try:
                if hasattr(module, "stop"):
                    await module.stop()
                logger.debug(f"Stopped {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        self.modules.clear()
        logger.info("👋 Weaver stopped. Goodbye!")
    
    async def _start_sensors(self) -> None:
        """Initialize all sensor modules."""
        enabled = self.config.enabled_modules
        
        if "camera" in enabled:
            from weaver.sensors.camera import CameraModule
            self.modules["camera"] = CameraModule()
            await self.modules["camera"].start()
        
        if "imu" in enabled:
            from weaver.sensors.imu import IMUModule
            self.modules["imu"] = IMUModule()
            await self.modules["imu"].start()
        
        if "ultrasonic" in enabled:
            from weaver.sensors.ultrasonic import UltrasonicModule
            self.modules["ultrasonic"] = UltrasonicModule()
            await self.modules["ultrasonic"].start()
        
        if "battery" in enabled:
            from weaver.sensors.battery import BatteryMonitor
            self.modules["battery"] = BatteryMonitor()
            await self.modules["battery"].start()
        
        if "thermal" in enabled:
            from weaver.sensors.thermal import ThermalMonitor
            self.modules["thermal"] = ThermalMonitor()
            await self.modules["thermal"].start()
    
    async def _start_motion(self) -> None:
        """Initialize the motion system."""
        enabled = self.config.enabled_modules
        
        if "servo" in enabled:
            from weaver.motion.servo_driver import ServoDriver
            self.modules["servo"] = ServoDriver()
            await self.modules["servo"].start()
        
        if "gait" in enabled:
            from weaver.motion.gait_engine import GaitEngine
            self.modules["gait"] = GaitEngine()
            await self.modules["gait"].start()
        
        if "posture" in enabled:
            from weaver.motion.posture import PostureController
            self.modules["posture"] = PostureController()
            await self.modules["posture"].start()
    
    async def _start_brain(self) -> None:
        """Initialize the LLM brain."""
        if "cortex" not in self.config.enabled_modules:
            return
        
        from weaver.brain.cortex import Cortex
        self.modules["cortex"] = Cortex()
        await self.modules["cortex"].start()
    
    async def _start_voice(self) -> None:
        """Initialize voice pipeline."""
        if "voice" not in self.config.enabled_modules:
            return
        
        from weaver.voice.stt import STTModule
        from weaver.voice.tts import TTSModule
        
        self.modules["stt"] = STTModule()
        await self.modules["stt"].start()
        
        self.modules["tts"] = TTSModule()
        await self.modules["tts"].start()
    
    async def _start_peripherals(self) -> None:
        """Initialize peripheral devices."""
        enabled = self.config.enabled_modules
        
        if "oled" in enabled:
            from weaver.peripherals.display import OLEDDisplay
            self.modules["oled"] = OLEDDisplay()
            await self.modules["oled"].start()
        
        if "led" in enabled:
            from weaver.peripherals.leds import LEDController
            self.modules["led"] = LEDController()
            await self.modules["led"].start()
        
        if "telemetry" in enabled:
            from weaver.peripherals.telemetry import TelemetryLogger
            self.modules["telemetry"] = TelemetryLogger()
            await self.modules["telemetry"].start()
    
    async def _start_keepalive(self) -> None:
        """Initialize the keepalive manager for STT, TTS, and LLM."""
        if "keepalive" not in self.config.enabled_modules:
            return
        if not self.config.keepalive.enabled:
            return
        
        from weaver.keepalive import create_default_keepalive_manager
        km = create_default_keepalive_manager()
        await km.start()
        self.modules["keepalive"] = km
        
        # Inject keepalive reference into cortex if available
        if "cortex" in self.modules:
            self.modules["cortex"].keepalive = km
    
    async def _start_web(self) -> None:
        """Start the web dashboard in a background task."""
        if "web" not in self.config.enabled_modules:
            return
        
        # Start uvicorn in a background thread
        import threading
        import uvicorn
        
        config = uvicorn.Config(
            "weaver.web.server:app",
            host=self.config.web.host,
            port=self.config.web.port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        
        # Run uvicorn in a thread so it doesn't block the asyncio loop
        def run_uvicorn():
            asyncio.set_event_loop(asyncio.new_event_loop())
            server.run()
        
        thread = threading.Thread(target=run_uvicorn, daemon=True)
        thread.start()
        self.modules["web"] = thread
        logger.info(f"🌐 Web dashboard on port {self.config.web.port}")
    
    def _get_ip(self) -> str:
        """Get the Pi's local IP address."""
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"
    
    async def run_forever(self) -> None:
        """Run the robot brain until interrupted."""
        await self.start()
        
        # Keep running until interrupted
        try:
            while self._running:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()


# ─── CLI Entry Point ─────────────────────────────────────────────────────


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Weaver Robot Brain")
    parser.add_argument("--mode", choices=["real", "mock"], default=None,
                        help="Hardware mode (default: from config)")
    parser.add_argument("--safety", choices=["child", "adult", "expert", "disabled"],
                        default=None, help="Safety level override")
    parser.add_argument("--llm", choices=["cloud", "local"], default=None,
                        help="LLM mode override")
    parser.add_argument("--web-only", action="store_true",
                        help="Start only the web dashboard (for testing)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Apply CLI overrides
    config = reload_config()
    if args.mode:
        config.hardware_mode = HardwareMode(args.mode)
    if args.safety:
        config.safety.level = SafetyLevel(args.safety)
    if args.llm:
        if args.llm == "cloud":
            config.cortex.mode = LLMMode.OLLAMA_CLOUD
        else:
            config.cortex.mode = LLMMode.OLLAMA_LOCAL
    
    # Create and run the brain
    brain = WeaverBrain()
    
    # Handle Ctrl+C gracefully
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, lambda: asyncio.create_task(brain.stop()))
    
    try:
        loop.run_until_complete(brain.run_forever())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()