"""Tests for the configuration system."""

from __future__ import annotations

import os
import pytest

from weaver.config import (
    BatteryConfig, CameraConfig, CortexConfig, GaitConfig,
    HardwareMode, IMUConfig, LEDConfig, LLMMode, MainConfig,
    OLEDConfig, SafetyConfig, SafetyLevel, ServoConfig,
    ThermalConfig, UltrasonicConfig, VoiceConfig, WebConfig,
    get_config, reload_config,
)


class TestConfig:
    """Test configuration loading and defaults."""
    
    def test_default_config(self):
        """Default config should be MOCK mode, CHILD safety."""
        config = MainConfig()
        assert config.hardware_mode == HardwareMode.MOCK
        assert config.safety.level == SafetyLevel.CHILD
        assert config.cortex.mode == LLMMode.OLLAMA_CLOUD
        assert config.robot_name == "Weaver"
    
    def test_servo_config_defaults(self):
        """Servo config should have 18 channels (6 legs × 3 joints)."""
        config = ServoConfig()
        assert len(config.leg_channels) == 6
        for leg in config.leg_channels:
            assert len(leg) == 3
        assert config.i2c_address == 0x40
        assert config.pwm_frequency == 50.0
    
    def test_gait_config_defaults(self):
        """Gait config should have safe defaults."""
        config = GaitConfig()
        assert config.default_speed <= 0.5
        assert config.max_speed <= 1.0
        assert config.step_height > 0
        assert config.step_length > 0
    
    def test_safety_config_defaults(self):
        """Safety config should default to CHILD level."""
        config = SafetyConfig()
        assert config.level == SafetyLevel.CHILD
        assert config.obstacle_stop_distance_cm == 30.0
        assert config.human_detection_enabled is True
    
    def test_cortex_config_defaults(self):
        """Cortex should default to Ollama Cloud."""
        config = CortexConfig()
        assert config.mode == LLMMode.OLLAMA_CLOUD
        assert config.cloud_model == "llama3.2:3b"
        assert config.fallback_to_rules is True
    
    def test_voice_config_defaults(self):
        """Voice should default to local STT and edge TTS."""
        config = VoiceConfig()
        assert config.stt_enabled is True
        assert config.stt_provider == "local"
        assert config.tts_provider == "edge"
        assert config.wake_word_enabled is True
    
    def test_web_config_defaults(self):
        """Web should default to port 8080, all interfaces."""
        config = WebConfig()
        assert config.port == 8080
        assert config.host == "0.0.0.0"
    
    def test_battery_config_defaults(self):
        """Battery config should be for 2S 18650 pack."""
        config = BatteryConfig()
        assert config.cells == 2
        assert config.nominal_voltage == 7.4
        assert config.full_voltage == 8.4
        assert config.empty_voltage == 6.0
    
    def test_led_config_defaults(self):
        """LED config should have 2 LEDs (RaspClaws has 2 WS2812s)."""
        config = LEDConfig()
        assert config.led_count == 2
        assert config.brightness <= 0.5  # Dim for safety
    
    def test_enabled_modules_default(self):
        """All core modules should be enabled by default."""
        config = MainConfig()
        for module in ["servo", "gait", "safety", "camera", "imu",
                       "ultrasonic", "battery", "web", "cortex"]:
            assert module in config.enabled_modules
    
    def test_env_override_hardware_mode(self, monkeypatch):
        """Environment variable should override hardware mode."""
        monkeypatch.setenv("WEAVER_HARDWARE_MODE", "real")
        config = reload_config()
        assert config.hardware_mode == HardwareMode.REAL
        # Reset for other tests
        monkeypatch.setenv("WEAVER_HARDWARE_MODE", "mock")
        reload_config()
    
    def test_safety_level_enum_values(self):
        """Safety levels should have correct values."""
        assert SafetyLevel.CHILD.value == "child"
        assert SafetyLevel.ADULT.value == "adult"
        assert SafetyLevel.EXPERT.value == "expert"
        assert SafetyLevel.DISABLED.value == "disabled"
    
    def test_llm_mode_enum_values(self):
        """LLM modes should have correct values."""
        assert LLMMode.OLLAMA_CLOUD.value == "ollama_cloud"
        assert LLMMode.OLLAMA_LOCAL.value == "ollama_local"