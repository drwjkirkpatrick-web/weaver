# server.py — FastAPI web dashboard and WebSocket control
#
# This is the web interface for Weaver. It provides:
# 1. A real-time dashboard showing all sensor data, robot state, camera feed
# 2. WebSocket for bidirectional communication (control robot from browser)
# 3. Voice command endpoint (send text/voice commands)
# 4. REST API for programmatic control
# 5. Static file serving for the dashboard HTML/JS
#
# The dashboard is designed to be FAST — it runs on the Pi's local network
# and responds in <100ms. No external dependencies, no CDN, no build step.
# Pure vanilla JS + WebSocket for real-time updates.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from weaver.config import WebConfig, get_config
from weaver.event_bus import Event, EventBus, EventType, get_event_bus
from weaver.safety import SafetyGovernor


class ConnectionManager:
    """Manages active WebSocket connections.
    
    Each connected client (browser tab, phone) gets a WebSocket.
    We broadcast sensor updates and state changes to all connected clients.
    """
    
    def __init__(self):
        self.active: list[WebSocket] = []
    
    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WebSocket client connected (total: {len(self.active)})")
    
    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)
        logger.info(f"WebSocket client disconnected (total: {len(self.active)})")
    
    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a message to all connected clients."""
        if not self.active:
            return
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)
    
    async def broadcast_text(self, text: str) -> None:
        """Send a text message to all clients."""
        if not self.active:
            return
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# ─── Robot State ───────────────────────────────────────────────────────

class RobotState:
    """Aggregated robot state for the dashboard.
    
    This is a read-only snapshot that the dashboard displays.
    Updated by subscribing to the event bus.
    """
    
    def __init__(self):
        self.battery_voltage: float = 8.4
        self.battery_percent: float = 100.0
        self.cpu_temp: float = 45.0
        self.obstacle_distance: float | None = None
        self.body_tilt: float = 0.0
        self.motion_status: str = "idle"  # idle, walking, turning, stopped
        self.safety_level: str = "child"
        self.safety_violations: int = 0
        self.estop_active: bool = False
        self.llm_mode: str = "ollama_cloud"
        self.llm_thinking: bool = False
        self.current_gait: str = "tripod"
        self.last_command: str = ""
        self.last_voice_transcript: str = ""
        self.uptime: float = time.time()
        self.camera_streaming: bool = False
        self.legs_positions: list[dict] = []
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "battery_voltage": round(self.battery_voltage, 2),
            "battery_percent": round(self.battery_percent, 1),
            "cpu_temp": round(self.cpu_temp, 1),
            "obstacle_distance": self.obstacle_distance,
            "body_tilt": round(self.body_tilt, 1),
            "motion_status": self.motion_status,
            "safety_level": self.safety_level,
            "safety_violations": self.safety_violations,
            "estop_active": self.estop_active,
            "llm_mode": self.llm_mode,
            "llm_thinking": self.llm_thinking,
            "current_gait": self.current_gait,
            "last_command": self.last_command,
            "last_voice_transcript": self.last_voice_transcript,
            "uptime_seconds": round(time.time() - self.uptime, 0),
            "camera_streaming": self.camera_streaming,
        }


# ─── FastAPI App ────────────────────────────────────────────────────────

# Global state (set during lifespan)
_manager: ConnectionManager | None = None
_state: RobotState | None = None
_bus: EventBus | None = None
_safety: SafetyGovernor | None = None


def get_manager() -> ConnectionManager:
    assert _manager is not None, "ConnectionManager not initialized"
    return _manager


def get_state() -> RobotState:
    assert _state is not None, "RobotState not initialized"
    return _state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    global _manager, _state, _bus, _safety
    
    config = get_config()
    _manager = ConnectionManager()
    _state = RobotState()
    _bus = get_event_bus()
    _safety = SafetyGovernor(config.safety)
    
    # Subscribe to events to update robot state
    async def on_battery(event: Event):
        _state.battery_voltage = event.data.get("voltage", 8.4)
        _state.battery_percent = event.data.get("percent", 100.0)
    
    async def on_thermal(event: Event):
        _state.cpu_temp = event.data.get("temp_c", 45.0)
    
    async def on_ultrasonic(event: Event):
        _state.obstacle_distance = event.data.get("distance_cm")
    
    async def on_imu(event: Event):
        _state.body_tilt = event.data.get("tilt_degrees", 0.0)
    
    async def on_motion(event: Event):
        _state.motion_status = event.data.get("status", "idle")
        _state.last_command = event.data.get("direction", "")
    
    async def on_cortex(event: Event):
        _state.llm_thinking = event.type == EventType.CORTEX_THINKING
        if event.type == EventType.CORTEX_DECISION:
            _state.last_command = event.data.get("action", "")
    
    async def on_voice(event: Event):
        if event.type == EventType.VOICE_TRANSCRIPT:
            _state.last_voice_transcript = event.data.get("text", "")
    
    async def on_safety(event: Event):
        _state.estop_active = event.data.get("estop", False)
        _state.safety_violations = event.data.get("violations", 0)
    
    _bus.subscribe(EventType.BATTERY_STATUS, on_battery)
    _bus.subscribe(EventType.THERMAL_STATUS, on_thermal)
    _bus.subscribe(EventType.ULTRASONIC_RANGE, on_ultrasonic)
    _bus.subscribe(EventType.IMU_DATA, on_imu)
    _bus.subscribe(EventType.MOTION_STARTED, on_motion)
    _bus.subscribe(EventType.MOTION_COMPLETED, on_motion)
    _bus.subscribe(EventType.CORTEX_THINKING, on_cortex)
    _bus.subscribe(EventType.CORTEX_DECISION, on_cortex)
    _bus.subscribe(EventType.VOICE_TRANSCRIPT, on_voice)
    _bus.subscribe(EventType.MODULE_STATUS, on_safety)
    _bus.subscribe(EventType.SAFETY_ESTOP, on_safety)
    
    # Start state broadcast loop
    broadcast_task = asyncio.create_task(_broadcast_loop())
    
    logger.info(f"🌐 Web dashboard starting on {_config.web.host}:{_config.web.port}")
    yield
    
    broadcast_task.cancel()
    try:
        await broadcast_task
    except asyncio.CancelledError:
        pass
    logger.info("Web dashboard stopped")


async def _broadcast_loop():
    """Periodically broadcast robot state to all connected clients."""
    while True:
        try:
            if _manager and _state:
                await _manager.broadcast({
                    "type": "state",
                    "data": _state.to_dict(),
                    "timestamp": time.time(),
                })
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
        await asyncio.sleep(0.5)  # 2 updates per second


# Create app
_config = get_config()
app = FastAPI(
    title="Weaver Robot Brain",
    description="AI-powered hexapod spider robot control dashboard",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS (allow all for development — restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_config.web.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (dashboard HTML/JS/CSS)
static_path = Path(__file__).parent.parent / _config.web.static_dir
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# ─── Routes ─────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard HTML page."""
    html_path = static_path / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Weaver Robot Brain</h1><p>Dashboard not found. Run setup.</p>")


@app.get("/api/state")
async def api_state():
    """Get current robot state as JSON."""
    return JSONResponse(get_state().to_dict())


@app.get("/api/safety")
async def api_safety():
    """Get safety governor state."""
    if _safety:
        s = _safety.get_state()
        return {
            "estop_active": s.estop_active,
            "motion_allowed": s.motion_allowed,
            "obstacle_distance_cm": s.obstacle_distance_cm,
            "human_distance_cm": s.human_distance_cm,
            "body_tilt_degrees": s.body_tilt_degrees,
            "battery_voltage": s.battery_voltage,
            "cpu_temp_c": s.cpu_temp_c,
            "speed_multiplier": s.speed_multiplier,
            "violations_count": s.violations_count,
            "last_violation": s.last_violation,
        }
    return {"error": "safety governor not initialized"}


@app.post("/api/command")
async def api_command(direction: str, speed: float = 0.5, action: str | None = None):
    """Send a motion command to the robot."""
    if action:
        await _bus.publish(Event(
            type=EventType.MOTION_COMMAND,
            data={"action": action, "direction": direction, "speed": speed},
            source="web_api",
        ))
    else:
        await _bus.publish(Event(
            type=EventType.MOTION_COMMAND,
            data={"direction": direction, "speed": speed},
            source="web_api",
        ))
    return {"status": "sent", "direction": direction, "speed": speed}


@app.post("/api/estop")
async def api_estop():
    """Trigger emergency stop."""
    _bus.set_estop(True)
    await _bus.publish(Event(
        type=EventType.SAFETY_ESTOP,
        data={"reason": "web_button"},
        source="web_api",
    ))
    return {"status": "estop_activated"}


@app.post("/api/estop/clear")
async def api_estop_clear():
    """Clear emergency stop."""
    if _safety:
        _safety.clear_estop()
        return {"status": "estop_cleared"}
    return {"error": "safety governor not initialized"}


@app.post("/api/voice")
async def api_voice(text: str):
    """Send a voice/text command to the robot brain."""
    await _bus.publish(Event(
        type=EventType.VOICE_COMMAND,
        data={"text": text, "source": "web_text"},
        source="web_api",
    ))
    return {"status": "sent", "text": text}


@app.get("/api/llm/swap")
async def api_llm_swap():
    """Swap between Ollama Cloud and Ollama Local."""
    from weaver.config import LLMMode, reload_config
    config = reload_config()
    if config.cortex.mode == LLMMode.OLLAMA_CLOUD:
        config.cortex.mode = LLMMode.OLLAMA_LOCAL
        mode = "ollama_local"
    else:
        config.cortex.mode = LLMMode.OLLAMA_CLOUD
        mode = "ollama_cloud"
    get_state().llm_mode = mode
    logger.info(f"LLM swapped to {mode}")
    return {"status": "swapped", "mode": mode}


@app.get("/api/events/recent")
async def api_recent_events(count: int = 50):
    """Get recent events from the event bus (for debugging)."""
    events = _bus.get_recent_events(count)
    return [
        {
            "type": e.type.name,
            "source": e.source,
            "timestamp": e.timestamp,
            "data": e.data,
        }
        for e in events
    ]


@app.get("/api/bus/stats")
async def api_bus_stats():
    """Get event bus statistics."""
    return _bus.get_stats()


@app.get("/api/health")
async def api_health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "robot_name": _config.robot_name,
        "hardware_mode": _config.hardware_mode.value,
        "uptime": time.time() - get_state().uptime,
    }


# ─── WebSocket ──────────────────────────────────────────────────────────


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for real-time bidirectional communication.
    
    Client → Server messages:
    - {"type": "command", "direction": "forward", "speed": 0.5}
    - {"type": "voice", "text": "walk forward"}
    - {"type": "estop"}
    - {"type": "estop_clear"}
    - {"type": "llm_swap"}
    
    Server → Client messages:
    - {"type": "state", "data": {...}}  (2x per second)
    - {"type": "event", "event_type": "...", "data": {...}}
    """
    manager = get_manager()
    await manager.connect(ws)
    
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue
            
            msg_type = msg.get("type", "")
            
            if msg_type == "command":
                direction = msg.get("direction", "stop")
                speed = msg.get("speed", 0.5)
                await _bus.publish(Event(
                    type=EventType.MOTION_COMMAND,
                    data={"direction": direction, "speed": speed},
                    source="websocket",
                ))
                await ws.send_json({"type": "ack", "command": direction})
            
            elif msg_type == "voice":
                text = msg.get("text", "")
                await _bus.publish(Event(
                    type=EventType.VOICE_COMMAND,
                    data={"text": text, "source": "websocket"},
                    source="websocket",
                ))
                await ws.send_json({"type": "ack", "voice": text})
            
            elif msg_type == "estop":
                _bus.set_estop(True)
                await _bus.publish(Event(
                    type=EventType.SAFETY_ESTOP,
                    data={"reason": "websocket"},
                    source="websocket",
                ))
                await ws.send_json({"type": "ack", "estop": True})
            
            elif msg_type == "estop_clear":
                if _safety:
                    _safety.clear_estop()
                await ws.send_json({"type": "ack", "estop": False})
            
            elif msg_type == "llm_swap":
                # Swap LLM mode
                config = get_config()
                if config.cortex.mode.value == "ollama_cloud":
                    config.cortex.mode = LLMMode.OLLAMA_LOCAL
                else:
                    config.cortex.mode = LLMMode.OLLAMA_CLOUD
                get_state().llm_mode = config.cortex.mode.value
                await ws.send_json({"type": "ack", "llm_mode": config.cortex.mode.value})
            
            elif msg_type == "gait":
                gait_type = msg.get("gait", "tripod")
                await _bus.publish(Event(
                    type=EventType.GAIT_CHANGE,
                    data={"gait": gait_type},
                    source="websocket",
                ))
                await ws.send_json({"type": "ack", "gait": gait_type})
            
            else:
                await ws.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})
    
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(ws)


def run_server():
    """Run the web server (blocking — call from main.py)."""
    config = get_config()
    uvicorn.run(
        "weaver.web.server:app",
        host=config.web.host,
        port=config.web.port,
        log_level="info",
        access_log=False,  # Reduce log spam
    )