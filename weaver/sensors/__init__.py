"""Weaver sensors: camera, IMU, ultrasonic, battery, thermal.

All sensor modules follow the same pattern:
  - Try to import hardware libraries at module level (graceful fallback)
  - ``_should_use_mock()`` decides real vs mock based on HardwareMode + lib availability
  - ``async start()`` initializes hardware and starts the polling loop
  - ``async stop()`` cancels tasks and releases hardware
  - Async polling loop reads data and publishes events to the bus

Import the sensor classes directly:
    from weaver.sensors import CameraSensor, IMUSensor, UltrasonicSensor, BatterySensor, ThermalSensor
"""

from __future__ import annotations

from weaver.sensors.camera import CameraSensor
from weaver.sensors.imu import IMUSensor
from weaver.sensors.ultrasonic import UltrasonicSensor
from weaver.sensors.battery import BatterySensor
from weaver.sensors.thermal import ThermalSensor

__all__ = [
    "CameraSensor",
    "IMUSensor",
    "UltrasonicSensor",
    "BatterySensor",
    "ThermalSensor",
]