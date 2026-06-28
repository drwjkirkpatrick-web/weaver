"""Test configuration and shared fixtures for Weaver tests.

Provides mock event bus, mock config, and mock hardware fixtures so all
tests run without real Raspberry Pi hardware.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Patch hardware modules BEFORE importing weaver modules that use them.
# This ensures clean mock mode for all tests.
import sys
for mod_name in [
    "lgpio", "smbus2", "board", "busio", "picamera2",
    "adafruit_pca9685", "adafruit_mpu6050", "adafruit_ssd1306",
    "neopixel", "rpi_ws2812", "spidev", "cv2",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from weaver.config import (
    BatteryConfig, CameraConfig, CortexConfig, GaitConfig, IMUConfig,
    LEDConfig, MainConfig, OLEDConfig, SafetyConfig, SafetyLevel,
    ServoConfig, ThermalConfig, UltrasonicConfig, VoiceConfig,
    WebConfig, TelemetryConfig, HardwareMode, LLMMode,
)
from weaver.event_bus import EventBus, EventType, Event, get_event_bus


@pytest.fixture
def event_bus():
    """Fresh event bus for each test."""
    return EventBus()


@pytest.fixture
def mock_config():
    """Mock config for testing (MOCK mode, CHILD safety)."""
    return MainConfig(
        hardware_mode=HardwareMode.MOCK,
        safety=SafetyConfig(level=SafetyLevel.CHILD),
    )


@pytest.fixture
def safety_config():
    return SafetyConfig(level=SafetyLevel.CHILD)


@pytest.fixture
def servo_config():
    return ServoConfig()


@pytest.fixture
def gait_config():
    return GaitConfig()


@pytest.fixture
def mock_event_bus():
    """Mock event bus that doesn't actually run async handlers."""
    bus = MagicMock(spec=EventBus)
    bus.publish = AsyncMock()
    bus.subscribe = MagicMock()
    bus.unsubscribe = MagicMock()
    bus.set_estop = MagicMock()
    bus.estop_active = False
    bus.get_stats = MagicMock(return_value={"total_events": 0})
    bus.get_recent_events = MagicMock(return_value=[])
    return bus