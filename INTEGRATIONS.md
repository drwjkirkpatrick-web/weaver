# 🔧 Weaver: 20 Raspberry Pi 5 Integrations

This document details all 20 ways Weaver integrates the Pi 5 into the RaspClaws hexapod platform.

## Integration Matrix

| # | Integration | Pi 5 Feature | Interface | Module | Hardware Needed |
|---|---|---|---|---|---|
| 1 | PCA9685 Servo Control | I2C Bus 1 | I2C 0x40 | `motion/servo_driver.py` | Adeept Robot HAT |
| 2 | Pi Camera Vision | CSI-2 Port | CSI | `sensors/camera.py` | Pi Camera Module |
| 3 | MPU6050 IMU | I2C Bus 1 | I2C 0x68 | `sensors/imu.py` | MPU6050 (in kit) |
| 4 | Ultrasonic Ranging | GPIO 23/24 | GPIO | `sensors/ultrasonic.py` | HC-SR04 (in kit) |
| 5 | Battery Monitoring | ADC/I2C | Analog/I2C | `sensors/battery.py` | Voltage divider or INA219 |
| 6 | CPU Thermal Monitor | /sys/class/thermal | Sysfs | `sensors/thermal.py` | None (built-in) |
| 7 | OLED Status Display | I2C Bus 1 | I2C 0x3C | `peripherals/display.py` | SSD1306 (in kit) |
| 8 | RGB LED Status | PWM GPIO 12 | PWM | `peripherals/leds.py` | WS2812 (in kit) |
| 9 | Telemetry Database | NVMe/SD Card | File I/O | `peripherals/telemetry.py` | None |
| 10 | FastAPI Web Server | Ethernet/WiFi | TCP 8080 | `web/server.py` | None (built-in) |
| 11 | WebSocket Real-time | Ethernet/WiFi | TCP WS | `web/server.py` | None (built-in) |
| 12 | Voice STT (Whisper) | USB Audio | USB ALSA | `voice/stt.py` | USB Microphone |
| 13 | Voice TTS | Audio Jack/USB | ALSA | `voice/tts.py` | Speaker/Headphones |
| 14 | LLM Cloud Bridge | Ethernet/WiFi | HTTPS | `brain/ollama_cloud.py` | Internet + API key |
| 15 | LLM Local Bridge | CPU/RAM | localhost:11434 | `brain/ollama_local.py` | Ollama installed |
| 16 | Intent Parser | CPU | NLP | `brain/intent_parser.py` | None |
| 17 | Safety Governor | GPIO 21 | GPIO | `safety.py` | E-Stop button (optional) |
| 18 | Event Bus | Async | asyncio | `event_bus.py` | None |
| 19 | Hexapod IK Engine | CPU FPU | Math | `motion/kinematics.py` | None |
| 20 | Gait Engine | CPU + I2C | Async | `motion/gait_engine.py` | PCA9685 + servos |

---

## Detailed Integration Descriptions

### 1. PCA9685 Servo Control (I2C)
- **Chip**: PCA9685 on Adeept Robot HAT V3.3
- **Address**: 0x40 (default)
- **Channels**: 18 used (6 legs × 3 joints), 1 spare
- **PWM**: 50Hz, 12-bit resolution (4096 steps)
- **Pulse range**: 150-600 (150=0°, 375=90°, 600=180°)
- **Library**: adafruit-circuitpython-pca9685
- **Calibration**: Per-servo angle offsets in config

### 2. Pi Camera Vision (CSI)
- **Camera**: Pi Camera Module (included in kit)
- **Resolution**: 640×480 @ 30fps (configurable)
- **Features**: Color tracking, object detection, face detection
- **Web stream**: JPEG frames @ 15fps via WebSocket
- **Library**: picamera2 + OpenCV
- **Safety use**: Face detection for human-safe motion

### 3. MPU6050 IMU (I2C)
- **Chip**: MPU6050 6-axis gyro/accel
- **Address**: 0x68
- **Data**: 3-axis acceleration + 3-axis gyroscope
- **Filter**: Complementary filter (α=0.98) for roll/pitch
- **Rate**: 100Hz sampling
- **Safety use**: Tilt protection, auto-leveling

### 4. Ultrasonic Range Sensor (GPIO)
- **Sensor**: HC-SR04
- **Pins**: Trigger=GPIO 23, Echo=GPIO 24
- **Range**: 2-400cm
- **Rate**: 10Hz, 3-sample median filter
- **Safety use**: Obstacle avoidance (stop at 30cm, slow at 60cm)

### 5. Battery Monitoring
- **Pack**: 2× 18650 in series (2S)
- **Voltage range**: 6.0V (empty) to 8.4V (full)
- **Methods**: GPIO voltage divider (MCP3008 ADC) or INA219 (I2C 0x41)
- **Alerts**: Low at 6.8V, critical at 6.2V
- **Safety use**: Stops motion when battery depleted

### 6. CPU Thermal Monitor
- **Source**: `/sys/class/thermal/thermal_zone0/temp`
- **Rate**: 1Hz
- **Thresholds**: Warning 75°C, Critical 80°C, LLM throttle 70°C
- **Safety use**: Stops motion if Pi overheats

### 7. OLED Status Display (I2C)
- **Display**: 0.96" SSD1306, 128×64 monochrome
- **Address**: 0x3C
- **Pages**: Auto-cycling status, sensors, network info
- **Rate**: 2Hz refresh
- **Library**: adafruit-circuitpython-ssd1306

### 8. RGB LED Status (PWM)
- **LEDs**: 2× WS2812 addressable RGB
- **Pin**: GPIO 12 (PWM0)
- **Colors**: Blue=idle, Green=moving, Orange=warning, Red=danger, Purple=thinking
- **Brightness**: 30% (safety — not blinding)
- **Library**: rpi_ws2812 / neopixel

### 9. Telemetry Database
- **Storage**: SQLite database at `weaver/data/telemetry.db`
- **Tables**: sensor_readings, events, motion_log
- **Buffer**: Batch inserts every 5 seconds
- **Auto-prune**: Max 100,000 records per table

### 10. FastAPI Web Server
- **Framework**: FastAPI + Uvicorn
- **Port**: 8080 (configurable)
- **Features**: REST API, static file serving, auto-docs at /docs
- **Endpoints**: /api/state, /api/safety, /api/command, /api/estop, /api/voice

### 11. WebSocket Real-time
- **Protocol**: WebSocket on /ws
- **Features**: Bidirectional, auto-reconnect, 2Hz state broadcast
- **Client → Server**: Commands, voice text, e-stop, LLM swap, gait change
- **Server → Client**: Robot state, event notifications

### 12. Voice STT (Speech-to-Text)
- **Engine**: faster-whisper (local, free) or Groq/OpenAI API
- **Model**: base (39MB, good accuracy/speed balance)
- **Wake word**: Energy-based detection (RMS threshold)
- **Library**: faster-whisper, PyAudio

### 13. Voice TTS (Text-to-Speech)
- **Engine**: Edge TTS (free, Microsoft) or OpenAI TTS
- **Voice**: en-US-AriaNeural (configurable)
- **Output**: Audio jack or USB DAC
- **Library**: edge-tts (async)

### 14. LLM Cloud Bridge
- **Protocol**: HTTP/HTTPS to Ollama Cloud API
- **Model**: llama3.2:3b (fast, capable)
- **Timeout**: 30 seconds
- **Retry**: Exponential backoff
- **RAM usage**: ~0MB on Pi (cloud does the work)

### 15. LLM Local Bridge
- **Protocol**: HTTP to localhost:11434
- **Model**: llama3.2:3b (~2GB RAM on Pi 5)
- **Timeout**: 60 seconds (CPU inference is slower)
- **Swap**: Can switch to/from cloud at runtime via dashboard button

### 16. Intent Parser
- **Function**: Natural language → structured motion commands
- **Simple commands**: Rule-based ("forward" → walk forward)
- **Complex commands**: LLM-based ("carefully approach the red object")
- **Output**: JSON action dict consumed by gait engine

### 17. Safety Governor
- **Core**: Intercepts all MOTION_COMMAND events (priority=True)
- **Checks**: E-stop, battery, thermal, tilt, obstacle, human, speed, servo range
- **GPIO**: BCM 21 for physical e-stop button
- **Override**: Can block or modify any motion command

### 18. Event Bus
- **Architecture**: Async pub/sub on asyncio
- **Subscribers**: Multiple per event type, priority queue for safety
- **History**: 500-event ring buffer for debugging
- **Events**: 30+ event types covering sensors, motion, safety, brain, voice, system

### 19. Hexapod IK Engine
- **Math**: Forward + inverse kinematics for 6-leg, 3-DOF hexapod
- **Transforms**: Body-to-leg coordinate frames
- **Features**: Body translation, rotation (roll/pitch/yaw)
- **Output**: 18 joint angles (degrees) for PCA9685

### 20. Gait Engine
- **Patterns**: Tripod (3-leg, fast), Wave (1-leg, stable), Ripple (2-leg, medium)
- **Phases**: Swing (lift+move) → Stance (push body)
- **Actions**: Walk, turn, strafe, stand, sit, crouch, wave hello
- **Safety**: Every motion checked by safety governor before execution