# 🕷️ Weaver — AI Robot Brain for Adeept RaspClaws Hexapod

<div align="center">

**A Hermes-powered AI brain for the Adeept RaspClaws Hexapod Spider Robot on Raspberry Pi 5**

*The robot that weaves through the world — sensor data into decisions, decisions into safe motion.*

</div>

---

## 🌟 Overview

Weaver is a complete AI robot brain that transforms the Adeept RaspClaws Hexapod kit into an intelligent, autonomous, voice-controlled spider robot. It runs on a Raspberry Pi 5 (8GB) and uses LLMs (Ollama Cloud or Local) for reasoning, with a full sensor suite for world awareness and a safety-first motion system designed for human interaction.

### Key Features

| Feature | Description |
|---|---|
| 🧠 **LLM Cortex** | AI decision-making via Ollama Cloud (default) or Ollama Local — swap at runtime |
| 🛡️ **Safety Governor** | Human-safe motion — child-safe defaults, obstacle avoidance, e-stop, tilt protection |
| 🎮 **Fast Web Dashboard** | Real-time control via any browser — WebSocket, voice commands, camera feed |
| 🎤 **Voice Control** | Whisper STT + Edge TTS — talk to your spider, it talks back |
| 🎵 **Beat Detection & Dance** | Real-time FFT beat detection → autonomous dance engine with Macarena choreography |
| 📡 **20 Pi Integrations** | Full utilization of Pi 5 hardware (see [INTEGRATIONS.md](INTEGRATIONS.md)) |
| 📊 **Telemetry** | SQLite logging of all sensor data and events for analysis |
| 🧪 **Mock Mode** | Full development without hardware — simulate everything |

---

## 📋 Hardware Requirements

| Component | Status |
|---|---|
| Raspberry Pi 5 (8GB) | ✅ Required |
| Adeept RaspClaws Hexapod Kit | ✅ Required |
| Adeept Robot HAT V3.3 (PCA9685) | ✅ Included in kit |
| 17× AD002 Metal Gear Servos | ✅ Included in kit |
| Pi Camera Module | ✅ Included in kit |
| MPU6050 Gyroscope | ✅ Included in kit |
| HC-SR04 Ultrasonic Sensor | ✅ Included in kit |
| 0.96" OLED (SSD1306) | ✅ Included in kit |
| 2× WS2812 RGB LEDs | ✅ Included in kit |
| 2× 18650 Batteries | ✅ Required (not in kit) |
| USB Microphone | ➕ Add for voice control |
| USB Speaker/Audio Out | ➕ Add for TTS |

See [HARDWARE.md](HARDWARE.md) for detailed specs and wiring.

---

## 🚀 Quick Start

### Development (Mock Mode — No Hardware Needed)

```bash
git clone https://github.com/drwjkirkpatrick-web/weaver.git
cd weaver
pip install -e ".[dev]"

# Run in mock mode
python -m weaver.main

# Open dashboard
open http://localhost:8080
```

### On Raspberry Pi 5 (Real Hardware)

```bash
# Install system dependencies
sudo apt install -y python3-pip i2c-tools
sudo raspi-config  # Enable I2C, Camera, SSH

# Clone and install
git clone https://github.com/drwjkirkpatrick-web/weaver.git
cd weaver
pip install -e ".[pi]"

# Configure
cp .env.example .env
nano .env  # Set WEAVER_HARDWARE_MODE=real, API keys, etc.

# Run
WEAVER_HARDWARE_MODE=real python -m weaver.main

# Access dashboard from any device on your network
# http://<pi-ip-address>:8080
```

### Install Ollama (for Local LLM Mode)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b  # 3B model fits in 8GB Pi 5
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WEAVER ROBOT BRAIN                            │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Camera  │  │   IMU    │  │Ultrasonic│  │ Battery  │  SENSORS  │
│  │  Module  │  │  Module  │  │  Module  │  │ Monitor  │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       └──────────────┴──────────────┴─────────────┘                  │
│                          ▼                                          │
│                   ┌──────────────┐                                  │
│                   │   EVENT BUS   │  ← async pub/sub               │
│                   └──────┬───────┘                                  │
│              ┌──────────┼──────────────┐                            │
│              ▼          ▼              ▼                            │
│     ┌────────────┐ ┌──────────┐ ┌────────────┐                     │
│     │   CORTEX   │ │ SAFETY   │ │  PERIPHERALS│                     │
│     │ (LLM brain)│ │ GOVERNOR │ │ (OLED/LED) │                     │
│     └─────┬──────┘ └────┬─────┘ └────────────┘                     │
│           │              │                                         │
│           ▼              ▼                                         │
│     ┌──────────────────────────┐                                   │
│     │    MOTION CONTROLLER     │                                   │
│     │  (gait engine + IK)      │  ACTUATORS                       │
│     └────────────┬─────────────┘                                   │
│                  ▼                                                  │
│     ┌──────────────────────────┐                                   │
│     │   SERVO DRIVER (PCA9685)│  → 17 metal-gear servos           │
│     └──────────────────────────┘                                   │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │  WEB DASHBOARD (FastAPI + WebSocket)  +  VOICE PIPELINE   │     │
│   └──────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design.

---

## 🛡️ Safety System

Weaver is designed to be **human-interaction safe**, especially around children.

| Safety Level | Max Speed | Obstacle Stop | Human Distance | Use Case |
|---|---|---|---|---|
| **CHILD** (default) | 30% | 30cm | 100cm | Kids, demos, indoor |
| **ADULT** | 60% | 30cm | 100cm | General use |
| **EXPERT** | 80% | 30cm | 100cm | Experienced users |
| **DISABLED** | 100% | — | — | Maintenance only |

### Safety Features

- **Emergency Stop**: Physical button (GPIO 21) + web button + voice command
- **Obstacle Avoidance**: Ultrasonic sensor stops forward motion within 30cm
- **Human Detection**: Camera-based face detection slows/stops near humans
- **Tilt Protection**: Auto-stabilizes if body tilt exceeds 30°
- **Battery Protection**: Stops all motion below 6.0V
- **Thermal Protection**: Stops motion if Pi CPU exceeds 70°C
- **Servo Range Limits**: Software clamps prevent servo over-travel
- **Speed Governing**: All motion is speed-capped by safety level

See [SAFETY.md](SAFETY.md) for full safety documentation.

---

## 🎮 Web Dashboard

The dashboard provides real-time control from any browser on your network:

- **Live sensor data**: Battery, temperature, distance, tilt
- **Camera feed**: Real-time video stream
- **Motion controls**: D-pad, action buttons, speed slider
- **Voice commands**: Type or speak commands
- **E-Stop**: Big red emergency stop button
- **LLM swap**: Switch between Cloud and Local with one click
- **Event log**: Real-time event stream for debugging

**Keyboard controls**: Arrow keys (move), Space (e-stop)

---

## 🎤 Voice Commands

Weaver understands natural language:

| Say... | Robot does... |
|---|---|
| "walk forward" | Starts walking forward |
| "stop" | Stops all motion |
| "turn left" / "turn right" | Turns in place |
| "sit down" | Lower body to resting position |
| "stand up" | Rise to standing height |
| "wave hello" | Waves a leg |
| "what do you see?" | Cortex describes camera input |
| "are you okay?" | Cortex reports status |

---

## 🔧 Configuration

All configuration is in `weaver/config.py` with environment variable overrides:

```bash
# .env file
WEAVER_HARDWARE_MODE=real          # or mock
WEAVER_SAFETY_LEVEL=child          # child, adult, expert, disabled
WEAVER_CORTEX_MODE=ollama_cloud    # or ollama_local
WEAVER_CORTEX_CLOUD_API_KEY=...    # Ollama Cloud API key
WEAVER_WEB_PORT=8080               # Dashboard port
```

---

## 🧪 Testing

```bash
# Run all tests (mock mode — no hardware needed)
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=weaver --cov-report=term-missing

# Run specific module
python -m pytest tests/test_safety.py -v
```

---

## 📦 The 20 Pi Integrations

Weaver deeply integrates the Raspberry Pi 5 into the RaspClaws platform:

| # | Integration | Pi Feature | Module |
|---|---|---|---|
| 1 | PCA9685 Servo Control | I2C Bus | `motion/servo_driver.py` |
| 2 | Pi Camera Vision | CSI Camera | `sensors/camera.py` |
| 3 | MPU6050 IMU | I2C Bus | `sensors/imu.py` |
| 4 | Ultrasonic Ranging | GPIO | `sensors/ultrasonic.py` |
| 5 | Battery Monitoring | ADC/I2C | `sensors/battery.py` |
| 6 | CPU Thermal Monitor | /sys/class/thermal | `sensors/thermal.py` |
| 7 | OLED Status Display | I2C | `peripherals/display.py` |
| 8 | RGB LED Status | PWM GPIO | `peripherals/leds.py` |
| 9 | Telemetry Database | SQLite on NVMe | `peripherals/telemetry.py` |
| 10 | FastAPI Web Server | Ethernet/WiFi | `web/server.py` |
| 11 | WebSocket Real-time | TCP | `web/server.py` |
| 12 | Voice STT (Whisper) | USB Audio | `voice/stt.py` |
| 13 | Voice TTS | Audio Jack | `voice/tts.py` |
| 14 | LLM Cloud Bridge | HTTP Client | `brain/ollama_cloud.py` |
| 15 | LLM Local Bridge | Ollama | `brain/ollama_local.py` |
| 16 | Intent Parser | NLP | `brain/intent_parser.py` |
| 17 | Safety Governor | GPIO E-Stop | `safety.py` |
| 18 | Event Bus | Async | `event_bus.py` |
| 19 | Hexapod IK Engine | CPU Math | `motion/kinematics.py` |
| 20 | Gait Engine | CPU + I2C | `motion/gait_engine.py` |

See [INTEGRATIONS.md](INTEGRATIONS.md) for detailed integration documentation.

---

## 🔄 LLM Mode: Cloud vs Local

Weaver supports two LLM backends, swappable at runtime:

### Ollama Cloud (Default)
- **Pros**: Zero RAM usage on Pi, faster inference, supports large models
- **Cons**: Requires internet, API key, slight latency
- **Setup**: Set `WEAVER_CORTEX_MODE=ollama_cloud` and `WEAVER_CORTEX_CLOUD_API_KEY`

### Ollama Local
- **Pros**: Works offline, no API costs, low latency
- **Cons**: Uses ~2GB Pi RAM, CPU-intensive, limited to small models
- **Setup**: Install Ollama, run `ollama pull llama3.2:3b`, set `WEAVER_CORTEX_MODE=ollama_local`

**Swap at runtime**: Click the "Swap Cloud/Local" button in the dashboard, or call `GET /api/llm/swap`.

---

## 📁 Project Structure

```
weaver/
├── weaver/                     # Main package
│   ├── __init__.py
│   ├── main.py               # Entry point + orchestrator
│   ├── config.py             # All configuration (Pydantic)
│   ├── event_bus.py          # Async pub/sub event bus
│   ├── safety.py             # Safety governor (CRITICAL)
│   ├── brain/                # LLM decision-making
│   │   ├── cortex.py         # Main brain loop
│   │   ├── ollama_cloud.py   # Cloud LLM bridge
│   │   ├── ollama_local.py   # Local LLM bridge
│   │   └── intent_parser.py  # NLP command parser
│   ├── motion/               # Movement system
│   │   ├── servo_driver.py   # PCA9685 controller
│   │   ├── kinematics.py     # Hexapod inverse kinematics
│   │   ├── gait_engine.py    # Walking patterns
│   │   └── posture.py        # Body balance
│   ├── sensors/              # Sensor modules
│   │   ├── camera.py         # Pi Camera + OpenCV
│   │   ├── imu.py            # MPU6050 gyro/accel
│   │   ├── ultrasonic.py     # HC-SR04 distance
│   │   ├── battery.py        # 18650 voltage monitor
│   │   └── thermal.py        # CPU temperature
│   ├── voice/                # Voice pipeline
│   │   ├── stt.py            # Whisper speech-to-text
│   │   └── tts.py            # Edge TTS text-to-speech
│   ├── peripherals/          # Status devices
│   │   ├── display.py        # OLED display
│   │   ├── leds.py           # WS2812 RGB LEDs
│   │   └── telemetry.py     # SQLite data logger
│   ├── web/                  # Web dashboard
│   │   └── server.py         # FastAPI + WebSocket
│   └── data/                 # Runtime data (gitignored)
├── static/dashboard/          # Web UI (vanilla JS)
│   └── index.html
├── tests/                    # Test suite
├── config/                   # Config templates
├── scripts/                  # Utility scripts
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 🛠️ Development

### Running in Mock Mode

Mock mode simulates all hardware — perfect for development on any machine:

```bash
python -m weaver.main --mode mock
```

All sensors return simulated data, servos log to console instead of moving, and the web dashboard works normally.

### Adding a New Sensor Module

1. Create `weaver/sensors/your_sensor.py`
2. Follow the `SensorModule` pattern (async `start()`/`stop()`)
3. Add config class to `weaver/config.py`
4. Add event types to `weaver/event_bus.py` if needed
5. Register in `weaver/main.py` `_start_sensors()`
6. Add tests in `tests/test_your_sensor.py`

### Running Tests

```bash
python -m pytest tests/ -v --cov=weaver
```

---

## 📜 License

MIT — see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- [Adeept](https://www.adeept.com/) — RaspClaws Hexapod kit
- [Nous Research](https://nousresearch.com/) — Hermes Agent framework
- [Ollama](https://ollama.com/) — Local LLM runtime
- [Adafruit](https://adafruit.com/) — CircuitPython libraries

---

<div align="center">

*Weaver — the spider that thinks. 🕷️*

</div>