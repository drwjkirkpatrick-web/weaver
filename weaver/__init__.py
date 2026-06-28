# __init__.py — Weaver robot brain
#
# Weaver: A Hermes-powered AI brain for the Adeept RaspClaws Hexapod Spider Robot
# running on a Raspberry Pi 5 (8GB).
#
# The brain coordinates sensor data, makes navigation decisions via LLM,
# and controls a 6-leg, 18-servo hexapod with human-safe motion.
#
# Architecture:
#
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │                        WEAVER ROBOT BRAIN                            │
#   │                                                                     │
#   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
#   │  │  Camera  │  │   IMU    │  │Ultrasonic│  │ Battery  │  SENSORS  │
#   │  │  Module  │  │  Module  │  │  Module  │  │ Monitor  │            │
#   │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
#   │       └──────────────┴──────────────┴─────────────┘                  │
#   │                          ▼                                          │
#   │                   ┌──────────────┐                                  │
#   │                   │   EVENT BUS   │  ← async pub/sub               │
#   │                   │  (asyncio)    │                                  │
#   │                   └──────┬───────┘                                  │
#   │              ┌──────────┼──────────────┐                            │
#   │              ▼          ▼              ▼                            │
#   │     ┌────────────┐ ┌──────────┐ ┌────────────┐                     │
#   │     │   CORTEX   │ │ SAFETY   │ │  PERIPHERALS│                     │
#   │     │ (LLM brain)│ │ GOVERNOR │ │ (OLED/LED) │                     │
#   │     └─────┬──────┘ └────┬─────┘ └────────────┘                     │
#   │           │              │                                         │
#   │           ▼              ▼                                         │
#   │     ┌──────────────────────────┐                                   │
#   │     │    MOTION CONTROLLER     │                                   │
#   │     │  (gait engine + IK)      │  ACTUATORS                       │
#   │     └────────────┬─────────────┘                                   │
#   │                  ▼                                                  │
#   │     ┌──────────────────────────┐                                   │
#   │     │   SERVO DRIVER (PCA9685)│  → 17 metal-gear servos           │
#   │     └──────────────────────────┘                                   │
#   │                                                                     │
#   │   ┌──────────────────────────────────────────────────────────┐     │
#   │   │  WEB DASHBOARD (FastAPI + WebSocket)  +  VOICE PIPELINE   │     │
#   │   └──────────────────────────────────────────────────────────┘     │
#   └─────────────────────────────────────────────────────────────────────┘
#
# 17 servos = 6 legs × 3 joints (coxa/femur/tibia) - 1 (center body servo)
# Actually 18 channels on PCA9685, 17 used by RaspClaws (1 spare)
#
# LLM Mode: Ollama Cloud (default) → Ollama Local (fallback/swap)
#
# ──────────────────────────────────────────────────────────────────────

__version__ = "0.1.0"
__all__ = ["__version__"]