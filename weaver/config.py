# config.py — Central configuration for Weaver
#
# All subsystems read from this config. Environment variables override defaults.
# On the Pi, set WEAVER_HARDWARE=true to enable real hardware.
# In dev/mock mode, everything runs with simulated hardware.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Enums ─────────────────────────────────────────────────────────────


class LLMMode(str, Enum):
    """Which LLM backend to use for the cortex."""
    OLLAMA_CLOUD = "ollama_cloud"  # Remote Ollama API (default, frees Pi RAM)
    OLLAMA_LOCAL = "ollama_local"  # Local Ollama on Pi 5 (swap when RAM is tight)


class HardwareMode(str, Enum):
    """Whether we're running on real Pi hardware or in simulation."""
    REAL = "real"      # Real Raspberry Pi 5 + RaspClaws hardware
    MOCK = "mock"      # Development/simulation mode (no hardware needed)


class SafetyLevel(str, Enum):
    """Safety constraint levels for motion."""
    CHILD = "child"     # Most restrictive — very slow, tiny steps, big buffer
    ADULT = "adult"     # Moderate — normal walking speed, safe distance
    EXPERT = "expert"   # Least restrictive — faster, smaller buffer
    DISABLED = "disabled"  # Safety off (maintenance only, NOT for interactive use)


# ─── Subsystem Configs ─────────────────────────────────────────────────


class ServoConfig(BaseModel):
    """PCA9685 servo driver configuration."""
    i2c_bus: int = 1                # Pi 5 default I2C bus
    i2c_address: int = 0x40         # PCA9685 default address
    pwm_frequency: float = 50.0     # Hz — standard for hobby servos
    servo_min_pulse: int = 150      # 12-bit min (out of 4096)
    servo_max_pulse: int = 600      # 12-bit max
    servo_center_pulse: int = 375   # Center position
    # RaspClaws has 6 legs, 3 joints each = 18 channels
    # Channel mapping: leg[joint] = channel number on PCA9685
    # Leg order: Front-Left, Mid-Left, Rear-Left, Rear-Right, Mid-Right, Front-Right
    # Joint order: coxa (horizontal), femur (vertical), tibia (vertical)
    leg_channels: list[list[int]] = Field(default=[
        [0, 1, 2],    # Leg 0: Front-Left  (coxa, femur, tibia)
        [3, 4, 5],    # Leg 1: Mid-Left
        [6, 7, 8],    # Leg 2: Rear-Left
        [9, 10, 11],  # Leg 3: Rear-Right
        [12, 13, 14], # Leg 4: Mid-Right
        [15, 16, 17], # Leg 5: Front-Right
    ])
    # Servo calibration offsets (degrees to add/subtract per channel)
    # Populated during calibration — start at 0 for all
    calibration_offsets: list[float] = Field(default=[0.0] * 18)


class GaitConfig(BaseModel):
    """Hexapod gait parameters."""
    # Speed multipliers
    default_speed: float = 0.5      # 0.0=stopped, 1.0=max
    max_speed: float = 0.8          # Hard cap (safety)
    # Step parameters
    step_height: float = 30.0       # mm — how high legs lift
    step_length: float = 60.0       # mm — forward travel per step
    step_duration: float = 1.2      # seconds per step cycle
    # Gait types
    tripod_gait: bool = True        # 3-leg alternating (fast, stable)
    wave_gait: bool = True          # 1-leg at a time (slow, very stable)
    ripple_gait: bool = True        # 2-leg alternating (medium)
    # Body height (mm above ground)
    body_height: float = 80.0       # Default standing height
    body_height_min: float = 40.0   # Crouching
    body_height_max: float = 120.0  # Standing tall
    # Turn parameters
    turn_radius: float = 200.0     # mm — min turning radius
    turn_speed: float = 0.3        # Turn speed multiplier


class SafetyConfig(BaseModel):
    """Human-interaction safety governor.
    
    This is the MOST important config in the system. It constrains all motion
    to be safe for humans nearby — especially children.
    """
    level: SafetyLevel = SafetyLevel.CHILD
    # Obstacle avoidance
    obstacle_stop_distance_cm: float = 30.0   # Stop if object closer than this
    obstacle_slow_distance_cm: float = 60.0   # Slow down if object in this range
    # Servo torque limiting (software-side)
    max_servo_angle_per_sec: float = 45.0    # Max rotation speed (deg/s)
    max_body_speed_ms: float = 0.3            # Max body translation (m/s)
    # Safe zones (degrees) — servos won't exceed these
    servo_safe_min: list[float] = Field(default=[-45.0] * 18)
    servo_safe_max: list[float] = Field(default=[45.0] * 18)
    # Emergency stop
    estop_enabled: bool = True
    estop_gpio_pin: int = 21              # BCM 21 — physical e-stop button
    # Tilt protection
    max_tilt_degrees: float = 30.0       # Auto-stabilize if body tilts past this
    # Temperature
    max_servo_temp_c: float = 70.0       # Pause if servos too hot
    # Battery
    min_battery_voltage: float = 6.0     # Stop all motion if battery low
    # Human detection
    human_detection_enabled: bool = True  # Use camera to detect humans and slow down
    human_safe_distance_cm: float = 100.0 # Maintain at least 1m from detected humans
    # Force feedback (if equipped)
    force_feedback_enabled: bool = False
    collision_force_threshold: float = 2.0  # Newtons


class CameraConfig(BaseModel):
    """Pi Camera configuration."""
    resolution: tuple[int, int] = (640, 480)
    framerate: int = 30
    # Computer vision
    color_tracking_enabled: bool = True
    object_detection_enabled: bool = True
    face_detection_enabled: bool = True     # For human safety
    # Night vision
    night_mode: bool = False
    # Video stream for web dashboard
    stream_quality: int = 70                # JPEG quality (0-100)
    stream_framerate: int = 15              # FPS for web stream


class IMUConfig(BaseModel):
    """MPU6050 gyroscope/accelerometer."""
    i2c_bus: int = 1
    i2c_address: int = 0x68          # MPU6050 default
    sample_rate_hz: float = 100.0   # Sampling rate
    # Calibration offsets (populated during calibration)
    accel_offset: list[float] = Field(default=[0.0, 0.0, 0.0])
    gyro_offset: list[float] = Field(default=[0.0, 0.0, 0.0])
    # Complementary filter coefficient (0-1)
    # Higher = trust accelerometer more, lower = trust gyroscope more
    complementary_alpha: float = 0.98


class UltrasonicConfig(BaseModel):
    """HC-SR04 ultrasonic range sensor."""
    trigger_pin: int = 23           # BCM 23
    echo_pin: int = 24               # BCM 24
    max_distance_cm: float = 400.0   # Sensor max range
    sample_rate_hz: float = 10.0     # How often to ping
    # Multi-sample averaging
    samples_per_reading: int = 3     # Take 3 readings, use median
    timeout_seconds: float = 0.1     # Per-reading timeout


class BatteryConfig(BaseModel):
    """18650 battery pack monitoring."""
    # ADC pin if using voltage divider, or I2C if using INA219
    monitor_type: str = "gpio"      # "gpio" (voltage divider) or "ina219"
    # GPIO voltage divider
    adc_pin: int = 0                 # MCP3008 channel 0 (if used)
    # INA219 config
    ina219_address: int = 0x41
    ina219_bus: int = 1
    # Battery specs (2x 18650 in series = ~7.4V nominal)
    cells: int = 2
    nominal_voltage: float = 7.4
    full_voltage: float = 8.4        # 2 × 4.2V
    empty_voltage: float = 6.0       # 2 × 3.0V
    # Alerts
    low_battery_threshold: float = 6.8   # Start warning
    critical_battery_threshold: float = 6.2  # Stop motion


class OLEDConfig(BaseModel):
    """0.96" OLED display (SSD1306)."""
    i2c_bus: int = 1
    i2c_address: int = 0x3C          # SSD1306 default
    width: int = 128
    height: int = 64
    # What to show on the display
    show_status: bool = True         # Mode, battery, safety level
    show_sensor_data: bool = True    # Distance, tilt
    update_rate_hz: float = 2.0      # Refresh rate


class LEDConfig(BaseModel):
    """WS2812 RGB LED status indicators."""
    # 2 LEDs on RaspClaws
    led_count: int = 2
    # GPIO pin (WS2812 uses PWM — Pi 5 uses GPIO 12 or 13 or 18)
    gpio_pin: int = 12               # BCM 12 (PWM0)
    brightness: float = 0.3          # 0.0-1.0 (keep dim for safety)
    # Color meanings (RGB tuples)
    color_idle: list[int] = Field(default=[0, 0, 255])       # Blue
    color_moving: list[int] = Field(default=[0, 255, 0])    # Green
    color_warning: list[int] = Field(default=[255, 100, 0]) # Orange
    color_danger: list[int] = Field(default=[255, 0, 0])    # Red
    color_thinking: list[int] = Field(default=[128, 0, 255]) # Purple (LLM thinking)
    color_error: list[int] = Field(default=[255, 0, 0])     # Red blink


class VoiceConfig(BaseModel):
    """Voice pipeline: STT (Whisper) + TTS."""
    stt_enabled: bool = True
    stt_provider: str = "local"      # "local" (faster-whisper) or "groq" or "openai"
    stt_model: str = "base"          # Whisper model size: tiny/base/small/medium
    stt_language: str = "en"         # Language code
    stt_timeout_seconds: float = 5.0 # Max silence before stopping
    tts_enabled: bool = True
    tts_provider: str = "edge"       # "edge" (free) or "openai" or "elevenlabs"
    tts_voice: str = "en-US-AriaNeural"  # Edge TTS voice
    # Wake word detection (simple energy-based, no external dependency)
    wake_word_enabled: bool = True
    wake_word_energy_threshold: float = 500.0  # RMS energy to trigger
    wake_word_cooldown_seconds: float = 2.0    # Min time between triggers


class CortexConfig(BaseModel):
    """LLM brain (cortex) configuration."""
    mode: LLMMode = LLMMode.OLLAMA_CLOUD
    # Ollama Cloud (remote API)
    cloud_base_url: str = "https://api.ollama.cloud"  # Or ngrok/tunnel URL
    cloud_model: str = "llama3.2:3b"       # Fast, capable, small context
    cloud_api_key: str = ""                # Set via WEAVER_CORTEX_CLOUD_API_KEY
    cloud_timeout_seconds: float = 30.0
    # Ollama Local (on Pi 5)
    local_base_url: str = "http://localhost:11434"
    local_model: str = "llama3.2:3b"      # 3B fits in 8GB with room to spare
    local_timeout_seconds: float = 60.0   # Slower on Pi CPU
    # Decision making
    decision_interval_seconds: float = 0.5  # How often cortex evaluates situation
    max_context_messages: int = 20          # Rolling context window
    # Intent classification
    intent_confidence_threshold: float = 0.7
    # Fallback: if LLM is unavailable, use rule-based behavior
    fallback_to_rules: bool = True


class WebConfig(BaseModel):
    """FastAPI web dashboard."""
    host: str = "0.0.0.0"            # Listen on all interfaces (Pi on LAN)
    port: int = 8080
    # WebSocket settings
    ws_ping_interval: float = 10.0
    ws_ping_timeout: float = 20.0
    # Authentication (simple token for now)
    auth_enabled: bool = False       # Set True for production
    auth_token: str = ""             # Set via WEAVER_WEB_AUTH_TOKEN
    # Static files
    static_dir: str = "static/dashboard"
    # CORS
    cors_origins: list[str] = Field(default=["*"])  # Restrict in production


class ThermalConfig(BaseModel):
    """Pi 5 thermal monitoring."""
    enabled: bool = True
    sample_rate_hz: float = 1.0      # Check temp every second
    # Thermal throttling thresholds (°C)
    warning_temp: float = 75.0       # Log warning, reduce motion speed
    critical_temp: float = 80.0      # Stop all motion, alert
    # Auto-reduce LLM usage if too hot
    llm_throttle_temp: float = 70.0  # Skip non-essential LLM calls


class PhraseCacheConfig(BaseModel):
    """Local phrase cache for instant robot responses."""
    enabled: bool = True
    db_path: str = "weaver/data/phrases.db"
    # When True, learned phrases from LLM are saved to cache
    learn_new_phrases: bool = True
    # Minimum similarity for fuzzy matching (0.0-1.0)
    fuzzy_min_similarity: float = 0.4
    # How many recent phrases to exclude from selection (variety)
    variety_window: int = 20


class KeepaliveConfig(BaseModel):
    """Keepalive pings for STT, TTS, and local LLM."""
    enabled: bool = True
    # Ollama Local LLM keepalive (seconds between pings)
    ollama_local_interval: float = 120.0   # Every 2 minutes (active mode)
    ollama_local_idle_interval: float = 300.0  # Every 5 minutes (fallback mode)
    # Whisper STT keepalive
    whisper_interval: float = 300.0       # Every 5 minutes
    # Edge TTS keepalive
    edge_tts_interval: float = 600.0      # Every 10 minutes


class TelemetryConfig(BaseModel):
    """Wi-Fi telemetry and remote logging."""
    enabled: bool = True
    # Log to SQLite database
    db_path: str = "weaver/data/telemetry.db"
    # How often to persist sensor readings
    persist_interval_seconds: float = 5.0
    # Max records to keep (auto-prune old data)
    max_records: int = 100_000


# ─── Main Config ─────────────────────────────────────────────────────────


class MainConfig(BaseSettings):
    """Top-level Weaver configuration.
    
    Environment variables override these defaults.
    Prefix: WEAVER_ (e.g., WEAVER_HARDWARE_MODE=real)
    """
    model_config = SettingsConfigDict(
        env_prefix="WEAVER_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Core
    hardware_mode: HardwareMode = HardwareMode.MOCK
    robot_name: str = "Weaver"
    log_level: str = "INFO"
    
    # Subsystems
    servo: ServoConfig = Field(default_factory=ServoConfig)
    gait: GaitConfig = Field(default_factory=GaitConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    imu: IMUConfig = Field(default_factory=IMUConfig)
    ultrasonic: UltrasonicConfig = Field(default_factory=UltrasonicConfig)
    battery: BatteryConfig = Field(default_factory=BatteryConfig)
    oled: OLEDConfig = Field(default_factory=OLEDConfig)
    led: LEDConfig = Field(default_factory=LEDConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    cortex: CortexConfig = Field(default_factory=CortexConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    phrase_cache: PhraseCacheConfig = Field(default_factory=PhraseCacheConfig)
    keepalive: KeepaliveConfig = Field(default_factory=KeepaliveConfig)
    
    # Enabled subsystems (toggle individual modules)
    enabled_modules: list[str] = Field(default=[
        "servo", "gait", "safety", "camera", "imu", "ultrasonic",
        "battery", "oled", "led", "voice", "cortex", "web", "thermal", "telemetry",
        "phrase_cache", "keepalive",
    ])


# Singleton accessor
_config: MainConfig | None = None


def get_config() -> MainConfig:
    """Get the global config singleton (lazy-initialized)."""
    global _config
    if _config is None:
        _config = MainConfig()
    return _config


def reload_config() -> MainConfig:
    """Force reload config from environment."""
    global _config
    _config = MainConfig()
    return _config