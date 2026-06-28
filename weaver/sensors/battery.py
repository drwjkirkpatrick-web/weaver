# battery.py — 18650 battery pack monitor for Weaver
#
# The RaspClaws hexapod is powered by 2× 18650 lithium-ion cells in series
# (a "2S" pack). This module monitors the pack voltage and estimates the
# remaining charge, alerting the system when battery is low or critical.
#
# Battery specs (2S 18650 pack):
#   - Full:    8.4V  (4.2V per cell — 100% charge)
#   - Nominal: 7.4V  (3.7V per cell — ~50% charge)
#   - Empty:   6.0V  (3.0V per cell — 0% charge, cutoff!)
#   - NEVER discharge below 3.0V/cell — permanently damages the cells
#
# Hardware options (configurable):
#   1. "gpio" — Voltage divider + MCP3008 ADC (SPI). The 8.4V pack is too
#      high for the Pi's 3.3V ADC, so a divider scales it down.
#   2. "ina219" — INA219 I2C current/voltage sensor. More accurate, gives
#      current draw and power too, but requires extra I2C wiring.
#
# Learning notes:
#   - Lithium-ion discharge curve is NOT linear. It stays near 4.0V for
#     most of the discharge, then drops steeply near the end. Our linear
#     percentage estimate is a simplification — for a real product you'd
#     use a lookup table or coulomb counting. But for a robot that gets
#     checked regularly, linear is "good enough."
#   - Voltage sag: under load (servos!), the voltage drops temporarily.
#     We take the reading when servos are idle if possible, and smooth
#     with a moving average to avoid false low-battery alerts.
#   - The 2S pack voltage = cell1 + cell2. We monitor the *total* pack
#     voltage, not individual cells. For proper cell balancing, a BMS
#     (Battery Management System) is needed — the RaspClaws kit includes
#     a basic one.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from typing import Any

from loguru import logger

from weaver.config import BatteryConfig, HardwareMode, get_config
from weaver.event_bus import Event, EventType, get_event_bus

# ─── Optional hardware imports ────────────────────────────────────────────

try:
    import busio  # type: ignore[import-untyped]
    import board  # type: ignore[import-untyped]
    _BLINKA_AVAILABLE = True
except ImportError:
    _BLINKA_AVAILABLE = False

try:
    import adafruit_ina219  # type: ignore[import-untyped]
    _INA219_AVAILABLE = True
except ImportError:
    _INA219_AVAILABLE = False

try:
    import spidev  # type: ignore[import-untyped]
    _SPIDEV_AVAILABLE = True
except ImportError:
    _SPIDEV_AVAILABLE = False


class BatterySensor:
    """18650 2S battery pack monitor with voltage-based charge estimation.

    This sensor reads the battery pack voltage (via INA219 I2C or MCP3008
# SPI ADC) and estimates remaining charge. It publishes periodic status
    events and alerts when the battery is low or critical.

    Voltage-to-percentage conversion uses a simple linear mapping between
    empty_voltage (0%) and full_voltage (100%). This is an approximation
    — real Li-ion discharge is non-linear — but it's sufficient for
    safety monitoring.

    A moving average filter (window = 5 readings) smooths out voltage
    sag from servo current spikes, preventing false low-battery alerts
    when the robot takes a step.

    Events published:
        - BATTERY_STATUS:   periodic voltage + percentage update
        - BATTERY_LOW:      when voltage drops below low threshold
        - BATTERY_CRITICAL: when voltage drops below critical threshold

    Mock mode:
        Simulates a slowly discharging battery starting at ~8.0V.
    """

    # Default sample rate (Hz) — battery doesn't need fast polling.
    # 0.2 Hz (every 5 seconds) is plenty for a battery that drains over hours.
    _DEFAULT_SAMPLE_RATE_HZ: float = 0.2

    # Moving average window for voltage smoothing
    _SMOOTHING_WINDOW: int = 5

    def __init__(self, config: BatteryConfig | None = None) -> None:
        """Initialize the battery monitor.

        Args:
            config: Battery configuration. If None, loads from global config.
        """
        cfg = get_config()
        self.config: BatteryConfig = config or cfg.battery
        self.hardware_mode: HardwareMode = cfg.hardware_mode

        self.bus = get_event_bus()

        # Hardware handles
        self._ina219: Any = None
        self._spi: Any = None
        self._i2c: Any = None

        # Moving average buffer for voltage smoothing
        self._voltage_history: deque[float] = deque(maxlen=self._SMOOTHING_WINDOW)

        # State tracking for alert edges (avoid republishing every cycle)
        self._low_alerted: bool = False
        self._critical_alerted: bool = False

        # Running task
        self._task: asyncio.Task | None = None

        # Mock mode state
        self._mock: bool = self._should_use_mock()
        self._mock_voltage: float = 8.0  # Start at ~80% charge
        self._mock_time: float = 0.0

    def _should_use_mock(self) -> bool:
        """Determine whether to use mock mode based on config and available libs."""
        if self.hardware_mode == HardwareMode.MOCK:
            return True

        # Check if required libraries are available for the configured monitor type
        if self.config.monitor_type == "ina219":
            if not (_BLINKA_AVAILABLE and _INA219_AVAILABLE):
                logger.warning(
                    "🔋 INA219 libraries not installed — battery running in MOCK mode. "
                    "Install with: pip install adafruit-circuitpython-ina219"
                )
                return True
        elif self.config.monitor_type == "gpio":
            if not _SPIDEV_AVAILABLE:
                logger.warning(
                    "🔋 spidev not installed — battery running in MOCK mode. "
                    "Install with: pip install spidev"
                )
                return True
        else:
            logger.warning(
                f"🔋 Unknown monitor_type '{self.config.monitor_type}' — using MOCK mode"
            )
            return True

        return False

    async def start(self) -> None:
        """Start the battery monitoring loop."""
        logger.info("🔋 Battery monitor starting (mode: {}, type: {})",
                     "MOCK" if self._mock else "REAL",
                     self.config.monitor_type)

        if not self._mock:
            await self._init_real_hardware()

        self._task = asyncio.create_task(self._poll_loop())

        logger.info("✅ Battery monitor started ({}S {} pack, {:.1f}-{:.1f}V)",
                     self.config.cells, "18650",
                     self.config.empty_voltage, self.config.full_voltage)

    async def stop(self) -> None:
        """Stop the battery monitor and release hardware."""
        logger.info("🔋 Battery monitor stopping...")

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._spi is not None:
            try:
                self._spi.close()
            except Exception:
                pass
        self._ina219 = None
        self._i2c = None
        self._spi = None

        logger.info("✅ Battery monitor stopped")

    # ─── Initialization ───────────────────────────────────────────────

    async def _init_real_hardware(self) -> None:
        """Initialize the configured battery monitoring hardware."""
        try:
            if self.config.monitor_type == "ina219":
                await self._init_ina219()
            elif self.config.monitor_type == "gpio":
                await self._init_mcp3008()
        except Exception as e:
            logger.error(f"Failed to initialize battery monitor: {e} — falling back to MOCK mode")
            self._mock = True

    async def _init_ina219(self) -> None:
        """Initialize the INA219 I2C current/voltage sensor.

        The INA219 can measure up to 32V and 3.2A with 12-bit resolution.
        It communicates over I2C and gives us voltage, current, and power.
        """
        self._i2c = busio.I2C(board.SCL, board.SDA)
        self._ina219 = adafruit_ina219.INA219(
            self._i2c,
            address=self.config.ina219_address,
        )
        logger.info(
            f"🔋 INA219 initialized at I2C 0x{self.config.ina219_address:02X} "
            f"(bus {self.config.ina219_bus})"
        )

    async def _init_mcp3008(self) -> None:
        """Initialize the MCP3008 SPI ADC for voltage divider reading.

        The MCP3008 is an 8-channel 10-bit SPI ADC. We read one channel
        that's connected to a voltage divider scaling the 8.4V pack down
        to <3.3V for the Pi.

        Voltage divider ratio: V_pack = V_adc × (R1 + R2) / R2
        We assume the divider ratio is baked into the reading — a common
        approach is R1=30kΩ, R2=10kΩ (ratio 4:1), so 8.4V → 2.1V.
        """
        self._spi = spidev.SpiDev()
        # bus 0, device 0 (CE0) — standard Pi SPI config
        self._spi.open(0, 0)
        self._spi.max_speed_hz = 1350000  # MCP3008 max: 1.35 MHz
        self._spi.mode = 0b00  # SPI mode 0 (CPOL=0, CPHA=0)

        logger.info(
            f"🔋 MCP3008 initialized on SPI (channel {self.config.adc_pin})"
        )

    # ─── Polling Loop ─────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Main polling loop: read voltage, compute percentage, publish events."""
        interval = 1.0 / self._DEFAULT_SAMPLE_RATE_HZ

        while True:
            try:
                voltage = await self._read_voltage()

                if voltage is not None:
                    # Smooth with moving average (reduces servo sag spikes)
                    self._voltage_history.append(voltage)
                    smoothed = sum(self._voltage_history) / len(self._voltage_history)

                    # Calculate charge percentage
                    percentage = self._voltage_to_percentage(smoothed)

                    # Publish battery status
                    await self.bus.publish(Event(
                        type=EventType.BATTERY_STATUS,
                        data={
                            "voltage": round(smoothed, 2),
                            "raw_voltage": round(voltage, 2),
                            "percentage": round(percentage, 1),
                            "cells": self.config.cells,
                            "timestamp": time.time(),
                        },
                        source="battery",
                    ))

                    # Check alert thresholds (edge-triggered: only publish once)
                    if smoothed <= self.config.critical_battery_threshold:
                        if not self._critical_alerted:
                            await self.bus.publish(Event(
                                type=EventType.BATTERY_CRITICAL,
                                data={
                                    "voltage": round(smoothed, 2),
                                    "threshold": self.config.critical_battery_threshold,
                                    "percentage": round(percentage, 1),
                                },
                                source="battery",
                            ))
                            self._critical_alerted = True
                            logger.error(
                                f"🛑 BATTERY CRITICAL: {smoothed:.2f}V "
                                f"({percentage:.0f}%) — STOP ALL MOTION"
                            )

                    elif smoothed <= self.config.low_battery_threshold:
                        if not self._low_alerted:
                            await self.bus.publish(Event(
                                type=EventType.BATTERY_LOW,
                                data={
                                    "voltage": round(smoothed, 2),
                                    "threshold": self.config.low_battery_threshold,
                                    "percentage": round(percentage, 1),
                                },
                                source="battery",
                            ))
                            self._low_alerted = True
                            logger.warning(
                                f"⚠️  BATTERY LOW: {smoothed:.2f}V "
                                f"({percentage:.0f}%) — reduce activity"
                            )

                    else:
                        # Reset alert flags when battery recovers
                        # (e.g., after replacing batteries or voltage sag clears)
                        if self._low_alerted or self._critical_alerted:
                            self._low_alerted = False
                            self._critical_alerted = False
                            logger.info(
                                f"🔋 Battery recovered: {smoothed:.2f}V — alerts cleared"
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Battery poll loop error: {e}")

            await asyncio.sleep(interval)

    async def _read_voltage(self) -> float | None:
        """Read the battery pack voltage.

        Returns:
            Pack voltage in volts, or None on error.
        """
        if self._mock:
            return self._mock_voltage_reading()

        try:
            if self.config.monitor_type == "ina219":
                voltage = self._ina219.bus_voltage  # INA219 gives bus voltage directly
                await asyncio.sleep(0)
                return float(voltage)

            elif self.config.monitor_type == "gpio":
                # Read MCP3008 ADC channel and convert to voltage.
                # The MCP3008 is a 10-bit ADC: 0-1023 → 0-3.3V (Pi Vref).
                # The voltage divider scales the pack voltage down, so we
                # multiply by the divider ratio to recover the actual voltage.
                raw = self._read_mcp3008(self.config.adc_pin)
                if raw is None:
                    return None

                # ADC voltage: raw / 1023 × 3.3V
                adc_voltage = (raw / 1023.0) * 3.3

                # Voltage divider ratio: assuming R1=30kΩ, R2=10kΩ → 4:1
                # (This should ideally be in config, but we hardcode a common
                #  ratio. Adjust if your divider is different.)
                divider_ratio = 4.0
                pack_voltage = adc_voltage * divider_ratio

                await asyncio.sleep(0)
                return pack_voltage

        except Exception as e:
            logger.error(f"Battery voltage read error: {e}")
            return None

        return None

    def _read_mcp3008(self, channel: int) -> int | None:
        """Read a single channel from the MCP3008 SPI ADC.

        The MCP3008 uses a 3-byte SPI transaction:
        - Byte 1: start bit + single-ended mode + channel (3 bits)
        - Byte 2: don't care (reads 0)
        - Byte 3: don't care (reads low 2 bits of result)

        The result is 10 bits: (byte2 & 0x03) << 8 | byte3.

        Args:
            channel: ADC channel (0-7).

        Returns:
            10-bit value (0-1023), or None on error.
        """
        try:
            # Construct SPI command for single-ended read on the given channel.
            # Format: 0b00001_000 | channel (single-ended mode)
            cmd = 0b00001000 | (channel & 0x07) << 4
            # SPI transfer: 3 bytes
            r = self._spi.xfer2([1, cmd, 0])
            # Result: combine the low 2 bits of byte 2 and all of byte 3
            return ((r[1] & 0x03) << 8) | r[2]
        except Exception as e:
            logger.error(f"MCP3008 read error: {e}")
            return None

    def _voltage_to_percentage(self, voltage: float) -> float:
        """Convert pack voltage to charge percentage.

        Uses a simple linear mapping between empty and full voltage.
        Clamps to [0, 100].

        Note: This is an approximation. Real Li-ion cells have a non-linear
        discharge curve — they hold ~4.0V for most of the discharge, then
        drop steeply at the end. For a more accurate estimate, use a
        lookup table based on the cell's discharge profile.

        Args:
            voltage: Pack voltage in volts.

        Returns:
            Estimated charge percentage (0.0 - 100.0).
        """
        v_range = self.config.full_voltage - self.config.empty_voltage
        if v_range <= 0:
            return 0.0

        percentage = ((voltage - self.config.empty_voltage) / v_range) * 100.0
        return max(0.0, min(100.0, percentage))

    def _mock_voltage_reading(self) -> float:
        """Generate a simulated battery voltage for mock mode.

        Simulates a battery slowly discharging over time, with small
        fluctuations from servo current draw.
        """
        # Simulate slow discharge: ~0.01V per reading (very slow)
        # In real life, a 2S pack powers the robot for 30-60 minutes.
        self._mock_time += 1.0 / self._DEFAULT_SAMPLE_RATE_HZ

        # Discharge: lose 0.1V per minute of mock time
        discharge = self._mock_time / 60.0 * 0.1
        base_voltage = max(self.config.empty_voltage, self._mock_voltage - discharge)

        # Add small noise (simulates servo current spikes causing voltage sag)
        noise = random.gauss(0, 0.05)

        return round(base_voltage + noise, 3)

    # ─── Public API ───────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Get current battery status (for dashboard/queries).

        Returns:
            Dict with voltage, percentage, and alert state.
        """
        if self._voltage_history:
            voltage = sum(self._voltage_history) / len(self._voltage_history)
        else:
            voltage = 0.0

        return {
            "voltage": round(voltage, 2),
            "percentage": round(self._voltage_to_percentage(voltage), 1),
            "low_alert": self._low_alerted,
            "critical_alert": self._critical_alerted,
            "cells": self.config.cells,
        }