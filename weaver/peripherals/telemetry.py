# telemetry.py — SQLite telemetry logger
#
# Records all sensor readings and robot events to a local SQLite database.
# This provides historical data for debugging, ML training, and analytics.
# Auto-prunes old data to keep the database from growing unbounded.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from loguru import logger

from weaver.config import TelemetryConfig, get_config
from weaver.event_bus import Event, EventBus, EventType, get_event_bus


class TelemetryLogger:
    """Logs sensor data and events to SQLite.
    
    Tables:
    - sensor_readings: timestamp, sensor_name, value_json
    - events: timestamp, event_type, source, data_json
    - motion_log: timestamp, direction, speed, result
    """
    
    def __init__(self, config: TelemetryConfig | None = None):
        self.config = config or get_config().telemetry
        self.bus = get_event_bus()
        self._db: sqlite3.Connection | None = None
        self._running = False
        self._buffer: list[tuple[str, str, str]] = []  # (timestamp, name, json)
        self._flush_task: asyncio.Task | None = None
    
    async def start(self) -> None:
        """Initialize the telemetry database."""
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                sensor_name TEXT NOT NULL,
                value_json TEXT NOT NULL
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                source TEXT,
                data_json TEXT
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS motion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                direction TEXT,
                speed REAL,
                result TEXT,
                reason TEXT
            )
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_sensor_time ON sensor_readings(timestamp)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp)")
        self._db.commit()
        
        # Subscribe to all sensor events
        self.bus.subscribe(EventType.BATTERY_STATUS, self._on_sensor)
        self.bus.subscribe(EventType.ULTRASONIC_RANGE, self._on_sensor)
        self.bus.subscribe(EventType.IMU_DATA, self._on_sensor)
        self.bus.subscribe(EventType.THERMAL_STATUS, self._on_sensor)
        self.bus.subscribe(EventType.CAMERA_OBJECT_DETECTED, self._on_sensor)
        
        # Subscribe to motion events
        self.bus.subscribe(EventType.MOTION_STARTED, self._on_motion)
        self.bus.subscribe(EventType.MOTION_COMPLETED, self._on_motion)
        self.bus.subscribe(EventType.MOTION_ABORTED, self._on_motion)
        
        # Subscribe to safety events
        self.bus.subscribe(EventType.SAFETY_ESTOP, self._on_safety)
        self.bus.subscribe(EventType.SAFETY_VIOLATION, self._on_safety)
        
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        self._prune_old_data()
        
        logger.info(f"📊 Telemetry logger started (db: {db_path})")
    
    async def stop(self) -> None:
        """Stop telemetry logging."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        if self._buffer:
            self._flush()
        if self._db:
            self._db.close()
        logger.info("Telemetry logger stopped")
    
    def _buffer_reading(self, sensor_name: str, data: dict) -> None:
        """Buffer a sensor reading for batch insert."""
        self._buffer.append((
            time.time(),
            sensor_name,
            json.dumps(data),
        ))
    
    def _flush(self) -> None:
        """Flush buffer to database."""
        if not self._buffer or not self._db:
            return
        
        try:
            self._db.executemany(
                "INSERT INTO sensor_readings (timestamp, sensor_name, value_json) VALUES (?, ?, ?)",
                self._buffer,
            )
            self._db.commit()
            self._buffer.clear()
        except Exception as e:
            logger.error(f"Telemetry flush error: {e}")
    
    async def _flush_loop(self) -> None:
        """Periodically flush buffer to database."""
        while self._running:
            try:
                self._flush()
                await asyncio.sleep(self.config.persist_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Telemetry flush loop error: {e}")
                await asyncio.sleep(5.0)
    
    def _prune_old_data(self) -> None:
        """Remove old records beyond max_records."""
        if not self._db:
            return
        try:
            for table in ["sensor_readings", "events", "motion_log"]:
                count = self._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if count > self.config.max_records:
                    excess = count - self.config.max_records
                    self._db.execute(f"DELETE FROM {table} WHERE id IN (SELECT id FROM {table} ORDER BY id LIMIT {excess})")
                    logger.info(f"Pruned {excess} old records from {table}")
            self._db.commit()
        except Exception as e:
            logger.error(f"Telemetry prune error: {e}")
    
    # ─── Event Handlers ───────────────────────────────────────────────
    
    async def _on_sensor(self, event: Event) -> None:
        """Buffer sensor readings."""
        sensor_name = event.type.name.lower()
        self._buffer_reading(sensor_name, event.data)
    
    async def _on_motion(self, event: Event) -> None:
        """Log motion events."""
        if not self._db:
            return
        try:
            self._db.execute(
                "INSERT INTO motion_log (timestamp, direction, speed, result, reason) VALUES (?, ?, ?, ?, ?)",
                (
                    time.time(),
                    event.data.get("direction", ""),
                    event.data.get("speed", 0.0),
                    event.type.name,
                    event.data.get("reason", ""),
                )
            )
            self._db.commit()
        except Exception as e:
            logger.error(f"Motion log error: {e}")
    
    async def _on_safety(self, event: Event) -> None:
        """Log safety events."""
        if not self._db:
            return
        try:
            self._db.execute(
                "INSERT INTO events (timestamp, event_type, source, data_json) VALUES (?, ?, ?, ?)",
                (
                    time.time(),
                    event.type.name,
                    event.source,
                    json.dumps(event.data),
                )
            )
            self._db.commit()
        except Exception as e:
            logger.error(f"Safety log error: {e}")