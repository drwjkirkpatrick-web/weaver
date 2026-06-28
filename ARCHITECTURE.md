# 🏗️ Weaver Architecture

## Design Principles

1. **Safety First** — The safety governor gates ALL motion. No exceptions.
2. **Event-Driven** — All subsystems communicate via async pub/sub event bus.
3. **Mock-Safe** — Every module works in mock mode (no hardware) and real mode.
4. **LLM-Optional** — Rule-based fallback when LLM is unavailable.
5. **Modular** — Each subsystem is independent and swappable.

## Startup Order

```
1. Config          → Load settings from env/file
2. EventBus       → Initialize async pub/sub
3. Safety         → Safety governor subscribes to ALL events (priority)
4. Sensors        → Camera, IMU, Ultrasonic, Battery, Thermal
5. Motion         → Servo driver, gait engine, posture controller
6. Brain          → Cortex (subscribes to sensors, publishes motion commands)
7. Voice          → STT + TTS pipelines
8. Peripherals    → OLED display, LEDs, telemetry logger
9. Web            → FastAPI server (depends on all above for state)
```

## Shutdown Order (Reverse)

```
9. Web → 8. Peripherals → 7. Voice → 6. Brain → 5. Motion → 4. Sensors → 3. Safety → 2. EventBus
```

## Event Flow

```
Sensors ──publish──→ EventBus ──dispatch──→ Safety (priority)
                         │                      │
                         │                      ├── block/modify
                         │                      ▼
                         └──────────────→ Cortex ──→ MotionCommand
                                                      │
                                                      ▼
                                              Safety Governor
                                                      │
                                              ┌───────┴───────┐
                                              │               │
                                          APPROVED         BLOCKED
                                              │               │
                                              ▼               ▼
                                        GaitEngine     MotionAborted
                                              │
                                              ▼
                                        ServoDriver
                                              │
                                              ▼
                                         PCA9685
                                              │
                                              ▼
                                         17 Servos
```

## Module Dependencies

```
config.py ←── (all modules)
event_bus.py ←── (all modules)
safety.py ←── gait_engine, cortex
sensors/* ←── cortex (via event bus)
motion/servo_driver ←── motion/gait_engine, motion/posture
motion/kinematics ←── motion/gait_engine, motion/posture
brain/cortex ←── brain/ollama_cloud OR brain/ollama_local, brain/intent_parser
voice/stt ←── brain/cortex (via VOICE_COMMAND event)
voice/tts ←── brain/cortex (via VOICE_SPEAKING event)
web/server ←── (subscribes to all for dashboard state)
```

## LLM Integration

### Cloud Mode (Default)
```
Cortex ──HTTP POST──→ Ollama Cloud API ──→ Response
                            (no Pi resources used)
```

### Local Mode
```
Cortex ──localhost:11434──→ Ollama on Pi ──→ Response
                            (~2GB RAM, CPU intensive)
```

### Swap Mechanism
The cortex holds a reference to the current LLM bridge. Swapping is:
1. Stop current bridge
2. Create new bridge (cloud or local)
3. Start new bridge
4. Resume cortex loop

This happens without restarting the robot.