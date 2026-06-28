# 🛡️ Weaver Safety System

## Overview

Weaver is designed to be **human-interaction safe** from the ground up. The safety governor sits between all motion commands and servo output. No motion happens without passing through multiple safety checks.

## Safety Architecture

```
                    Motion Command
                         │
                         ▼
              ┌─────────────────────┐
              │  SAFETY GOVERNOR    │
              │                     │
              │  1. E-Stop check    │──→ BLOCK
              │  2. Battery check   │──→ BLOCK
              │  3. Thermal check   │──→ BLOCK
              │  4. Tilt check      │──→ BLOCK
              │  5. Obstacle check   │──→ BLOCK or SLOW
              │  6. Human check     │──→ SLOW
              │  7. Speed cap       │──→ CLAMP
              │  8. Servo range     │──→ CLAMP or BLOCK
              │                     │
              └─────────┬───────────┘
                        │
                   ▼  or  ▼
              APPROVED    BLOCKED
                        │
                        ▼
              ┌─────────────────────┐
              │   SERVO DRIVER      │
              │   (PCA9685)         │
              └─────────────────────┘
```

## Safety Levels

| Level | Speed Cap | Description |
|---|---|---|
| **CHILD** | 30% | Default. Very slow, cautious. Safe for kids and indoor demos. |
| **ADULT** | 60% | Moderate speed. General use with supervision. |
| **EXPERT** | 80% | Faster operation. Experienced users only. |
| **DISABLED** | 100% | No limits. **MAINTENANCE ONLY.** Not for interactive use. |

## Emergency Stop

The E-Stop immediately halts all motion and disables servos.

**Activation methods:**
1. Physical button (GPIO 21 — connect NC button to GND)
2. Web dashboard "EMERGENCY STOP" button
3. WebSocket command `{"type": "estop"}`
4. Voice command "stop" or "emergency stop"
5. Programmatic: `safety_governor.trigger_estop(reason)`

**Clearing:**
- Must be done manually via web dashboard "Clear E-Stop" button
- Or programmatic: `safety_governor.clear_estop()`

## Safety Checks

### 1. Obstacle Avoidance
- Ultrasonic sensor pings 10× per second
- **Stop**: Object closer than 30cm → all forward motion blocked
- **Slow**: Object 30-60cm → speed reduced to 30%
- Backward motion is never blocked by obstacles

### 2. Human Detection
- Camera runs face detection (Haar cascades)
- Human within 100cm → speed scales inversely with distance
- Human within 30cm → all motion stops

### 3. Tilt Protection
- IMU reads body orientation 100× per second
- Tilt > 30° → motion blocked, auto-stabilization engaged
- Tilt > 15° → warning logged, speed reduced

### 4. Battery Protection
- Voltage monitored continuously
- Below 6.8V → warning, speed reduced to 50%
- Below 6.2V → critical, all motion stops
- Below 6.0V → system shutdown initiated

### 5. Thermal Protection
- Pi 5 CPU temperature monitored 1× per second
- Above 70°C → non-essential LLM calls skipped
- Above 75°C → speed reduced, warning logged
- Above 80°C → all motion stops, alert published

### 6. Servo Range Limits
- Each of the 18 servos has configurable min/max angles
- Default range: ±45° from center
- Commands exceeding range are blocked, not clamped (safety-first)

### 7. Speed Governing
- All motion commands are speed-capped by safety level
- Speed is checked AFTER obstacle/human adjustments
- Final speed = min(requested_speed, obstacle_adjusted, human_adjusted, level_cap)

## Testing

The safety system has comprehensive tests in `tests/test_safety.py`:

```bash
python -m pytest tests/test_safety.py -v
```

All 17 safety tests must pass before any deployment.