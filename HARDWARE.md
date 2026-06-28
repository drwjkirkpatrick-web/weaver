# 🔩 Weaver Hardware Reference

## RaspClaws Hexapod Kit

### Kit Contents (Adeept RaspClaws / RaspClaws-Metal)

| Component | Qty | Interface | Purpose |
|---|---|---|---|
| Aluminum alloy body parts | 1 set | Mechanical | Robot frame |
| Adeept Robot HAT V3.3 | 1 | I2C + GPIO | Servo controller + motor driver |
| Pi Camera Module | 1 | CSI-2 | Vision, color tracking, navigation |
| MPU6050 Gyro/Accel | 1 | I2C 0x68 | Orientation, tilt, stabilization |
| HC-SR04 Ultrasonic | 1 | GPIO 23/24 | Distance measurement |
| 0.96" OLED (SSD1306) | 1 | I2C 0x3C | Status display |
| WS2812 RGB LED | 2 | GPIO 12 (PWM) | Status indicators |
| AD002 Metal Gear Servo | 17 | PCA9685 ch 0-16 | Leg joints |
| Battery holder | 1 | USB-C | 2× 18650 power |
| USB-C cable | 1 | USB-C | Charging |

### Servo Layout

The RaspClaws has 6 legs, each with 3 joints (18 total). The kit uses 17 servos (one spare channel).

```
         Front
    ┌─────────────────┐
    │  L0   │   L5    │     L = Left, R = Right
    │ FL    │   FR    │     F = Front, M = Mid, R = Rear
    │       │         │
    │  L1   │   L4    │     Each leg has 3 servos:
    │ ML    │   MR    │       1. Coxa (horizontal rotation)
    │       │         │       2. Femur (vertical lift)
    │  L2   │   L3    │       3. Tibia (vertical extend)
    │ RL    │   RR    │
    └─────────────────┘
         Rear
```

### PCA9685 Channel Mapping

| Channel | Leg | Joint | Description |
|---|---|---|---|
| 0 | 0 (Front-Left) | Coxa | Horizontal rotation |
| 1 | 0 (Front-Left) | Femur | Vertical lift |
| 2 | 0 (Front-Left) | Tibia | Vertical extend |
| 3 | 1 (Mid-Left) | Coxa | |
| 4 | 1 (Mid-Left) | Femur | |
| 5 | 1 (Mid-Left) | Tibia | |
| 6 | 2 (Rear-Left) | Coxa | |
| 7 | 2 (Rear-Left) | Femur | |
| 8 | 2 (Rear-Left) | Tibia | |
| 9 | 3 (Rear-Right) | Coxa | |
| 10 | 3 (Rear-Right) | Femur | |
| 11 | 3 (Rear-Right) | Tibia | |
| 12 | 4 (Mid-Right) | Coxa | |
| 13 | 4 (Mid-Right) | Femur | |
| 14 | 4 (Mid-Right) | Tibia | |
| 15 | 5 (Front-Right) | Coxa | |
| 16 | 5 (Front-Right) | Femur | |
| 17 | 5 (Front-Right) | Tibia | |

### Servo Specifications (AD002 Metal Gear)

- **Type**: Analog servo with metal gears
- **Rotation**: 0-180°
- **Pulse range**: 500-2500μs
- **PWM frequency**: 50Hz (20ms period)
- **Torque**: ~10 kg·cm at 6V
- **Speed**: ~0.12s/60° at 6V
- **Voltage**: 4.8-7.4V

### PCA9685 PWM Mapping

| Angle | Pulse (μs) | 12-bit Value |
|---|---|---|
| 0° | 500 | 150 |
| 45° | 1000 | 262 |
| 90° (center) | 1500 | 375 |
| 135° | 2000 | 487 |
| 180° | 2500 | 600 |

## Raspberry Pi 5 Pin Assignments

| Function | BCM Pin | Physical Pin | Interface |
|---|---|---|---|
| I2C SDA (PCA9685, MPU6050, SSD1306) | GPIO 2 | Pin 3 | I2C Bus 1 |
| I2C SCL | GPIO 3 | Pin 5 | I2C Bus 1 |
| Ultrasonic Trigger | GPIO 23 | Pin 16 | GPIO Output |
| Ultrasonic Echo | GPIO 24 | Pin 18 | GPIO Input |
| WS2812 LED Data | GPIO 12 | Pin 32 | PWM0 |
| E-Stop Button | GPIO 21 | Pin 40 | GPIO Input (pull-up) |
| Camera | CSI-2 | FFC | MIPI CSI |
| Audio Out | 3.5mm Jack | — | Analog |

## Power System

### Battery Pack
- **Configuration**: 2× 18650 Li-ion in series (2S)
- **Nominal voltage**: 7.4V
- **Full charge**: 8.4V (4.2V per cell)
- **Empty**: 6.0V (3.0V per cell)
- **Charging**: USB-C (built-in charger on Robot HAT)

### Power Distribution
```
18650 Pack (7.4V)
    │
    ├── Robot HAT (servo power via 6V BEC)
    │       ├── 17× servos
    │       ├── WS2812 LEDs
    │       └── OLED display
    │
    ├── Pi 5 (via USB-C, 5V/3A)
    │       ├── Camera
    │       ├── USB microphone
    │       └── USB DAC (if used)
    │
    └── Battery monitor (voltage divider or INA219)
```

## Adding Hardware (Expansion)

### USB Microphone (for Voice Control)
- Any USB microphone works (Blue Snowball, cheap USB mic)
- Connects to Pi 5 USB 3.0 port
- ALSA device: `plughw:1,0` (or auto-detect)

### USB Speaker (for TTS)
- USB audio adapter or 3.5mm jack to powered speaker
- ALSA device: `plughw:0,0` (or auto-detect)

### INA219 Current/Voltage Sensor (Better Battery Monitoring)
- I2C address: 0x41
- Measures voltage + current + power
- More accurate than voltage divider

### Additional Sensors (Spare I2C Addresses)
| Address | Available? | Suggested Use |
|---|---|---|
| 0x1E | ✅ | Magnetometer (HMC5883L) |
| 0x76 | ✅ | Pressure/Temp (BME280) |
| 0x39 | ✅ | Light sensor (TSL2561) |

## Assembly Tips

1. **Center servos BEFORE assembly** — connect to Robot HAT, run `center_all()`, then attach to body
2. **Route cables carefully** — servo cables can tangle in joints
3. **Tighten all screws** — vibration loosens them quickly
4. **Battery access** — ensure you can swap batteries without disassembling
5. **Camera angle** — mount pointing slightly down (15°) for best obstacle view

## Calibration

After assembly, calibrate each servo:

```bash
# On the Pi
WEAVER_HARDWARE_MODE=real python -m weaver.scripts.calibrate
```

This will:
1. Center all servos
2. Ask you to adjust offsets if legs aren't straight
3. Save calibration to config
4. Test each leg individually