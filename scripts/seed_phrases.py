#!/usr/bin/env python3
# seed_phrases.py — Populate the phrase cache with 1000 robot phrases
#
# This script creates the SQLite database at weaver/data/phrases.db and
# fills it with 1000 phrases a hexapod spider robot would say in normal
# operation. The phrases are organized by category and intent.
#
# Run this once after installing Weaver:
#   python scripts/seed_phrases.py
#
# It's idempotent — running it again drops and recreates the table.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

from loguru import logger


# ─── 1000 Robot Phrases ────────────────────────────────────────────────
# Organized as (text, category, intent, priority) tuples.
# Priority: 1 = most used, 5 = rarely used.


def get_all_phrases() -> list[tuple[str, str, str, int]]:
    """Return all 1000 phrases for the robot."""
    phrases: list[tuple[str, str, str, int]] = []
    
    # ─── GREETINGS (50) ─────────────────────────────────────────────────
    greetings = [
        "Hello! I'm Weaver.", "Hi there!", "Hey! Good to see you.",
        "Greetings, human.", "Hi! I'm your hexapod friend.",
        "Hello! Ready to walk.", "Hey there! What can I do for you?",
        "Good morning! All systems ready.", "Good afternoon! How can I help?",
        "Good evening! I'm online.", "Hi! Want to see me move?",
        "Hello! I'm a spider robot, but a friendly one!",
        "Greetings! My sensors are active.", "Hi! Nice to meet you.",
        "Hello! I'm ready for commands.", "Hey! I heard you.",
        "Good to see you! I'm Weaver.", "Hi! I have six legs and lots of sensors.",
        "Hello there! How are you today?", "Hi! I'm listening.",
        "Greetings from the hexapod side!", "Hello! My camera is on.",
        "Hey! I'm powered up and ready.", "Hi! I can walk, turn, and wave!",
        "Hello! What adventure shall we have?", "Hi friend! I'm Weaver the spider robot.",
        "Good morning! Let's get moving.", "Hello! I sense you're nearby.",
        "Hi! I'm a robot spider, but I don't bite!", "Greetings! Ready for action.",
        "Hello! I have 18 servos and I'm not afraid to use them!",
        "Hi! What would you like me to do?", "Hey! My legs are ready.",
        "Hello! Let me check my sensors... all good!", "Hi! I'm online and operational.",
        "Greetings! I'm a six-legged walking robot.", "Hello! I'm here to help.",
        "Hi! Want to see me wave hello?", "Hello! I'm listening for commands.",
        "Hey! I'm a spider robot with a big heart.", "Hi! Let's go explore.",
        "Hello! I'm fully charged and ready.", "Greetings! My camera sees you.",
        "Hi! I can walk in three different gaits.", "Hello! Ready when you are.",
        "Hey! I heard your voice. What's up?", "Hello! My sensors are all online.",
        "Hi! I'm a friendly neighborhood spider robot.", "Greetings! Let's get started.",
    ]
    for p in greetings:
        phrases.append((p, "greetings", "greeting", 1))
    
    # ─── STATUS (100) ──────────────────────────────────────────────────
    status_phrases = [
        "All systems operational.", "Everything looks good.", "Sensors are reading normal.",
        "I'm feeling balanced and steady.", "All 18 servos are responding.",
        "My camera is active and working.", "Battery level is good.",
        "I'm connected to the network.", "Systems check complete. All green.",
        "I'm in standby mode, ready to go.", "All sensor readings are within normal range.",
        "My gyroscope says I'm level.", "The ultrasonic sensor is working.",
        "My OLED display is showing status.", "The LEDs are indicating ready state.",
        "I'm connected and responsive.", "Telemetry is being logged.",
        "All safety checks passed.", "My brain is online and thinking.",
        "I'm ready for your next command.", "Status: operational. Ready to proceed.",
        "Everything is running smoothly.", "No warnings or errors detected.",
        "I'm standing upright and stable.", "All legs are in standing position.",
        "My IMU reports zero tilt.", "The path ahead is clear.",
        "Battery voltage is nominal.", "CPU temperature is normal.",
        "I'm in idle mode.", "Ready for movement commands.",
        "All subsystems are online.", "My servos are centered and ready.",
        "The safety governor is active.", "I'm in CHILD safety mode.",
        "My walking engine is ready.", "The camera stream is active.",
        "Voice pipeline is online.", "I'm listening for wake words.",
        "The web dashboard is accessible.", "Telemetry database is recording.",
        "All motors are calibrated.", "My balance controller is active.",
        "The OLED is showing my status.", "LEDs are in idle state — blue.",
        "I'm ready to walk, turn, or wave.", "My cortex is in waiting state.",
        "Everything checks out. Let's go!", "Standing by for instructions.",
        "All clear. No obstacles detected.", "Battery is healthy.",
        "Temperature is well within limits.", "I'm stable and balanced.",
        "Ready to deploy.", "Systems nominal.", "Operating normally.",
        "I'm functioning at full capacity.", "All sensors reporting green.",
        "My legs are ready for action.", "No issues detected.",
        "I'm good to go.", "Everything is A-OK.", "Ready for input.",
        "Standing by.", "Awaiting commands.", "All systems go.",
        "I'm here and ready.", "Operational status confirmed.",
        "My diagnostic check shows all green.", "No faults detected.",
        "I'm running at optimal performance.", "All hardware is responding.",
        "The robot is in a safe state.", "Ready for your instructions.",
        "I'm awake and alert.", "System health: excellent.",
        "All peripherals are online.", "I'm in a stable standing position.",
        "My safety systems are engaged.", "Ready to navigate.",
        "All inputs are being monitored.", "I'm tracking my environment.",
        "Status update: all good.", "Everything is functioning as expected.",
        "I'm ready to help.", "My systems are fully initialized.",
        "Standing upright and balanced.", "No alerts to report.",
        "I'm in good shape today.", "All clear on my end.",
        "Ready to walk when you are.", "Systems are humming along nicely.",
        "I'm fully operational.", "Everything is in working order.",
        "My status is: ready.", "All signs point to go.",
        "I'm feeling great and ready to go.", "No problems detected. All systems green.",
        "Everything is in order.", "I'm primed and ready for action.",
    ]
    for p in status_phrases:
        phrases.append((p, "status", "idle", 2))
    
    # ─── NAVIGATION (100) ──────────────────────────────────────────────
    nav_phrases = [
        "Moving forward.", "Taking a step ahead.", "Walking forward now.",
        "Onward!", "Heading straight ahead.", "Moving out.",
        "Forward march!", "I'm walking forward now.", "Stepping ahead.",
        "Let's go forward.", "Moving in the forward direction.",
        "Turning left.", "Rotating to the left.", "Turning left now.",
        "Making a left turn.", "Rotating counterclockwise.", "Left turn in progress.",
        "I'm turning left.", "Pivoting left.", "Turning to face left.",
        "Turning right.", "Rotating to the right.", "Turning right now.",
        "Making a right turn.", "Rotating clockwise.", "Right turn in progress.",
        "I'm turning right.", "Pivoting right.", "Turning to face right.",
        "Moving backward.", "Stepping back.", "Reversing now.",
        "Going backward.", "Walking in reverse.", "Backing up.",
        "Stepping backward.", "Retreating.", "Moving in reverse.",
        "Strafing left.", "Sidestepping left.", "Moving left sideways.",
        "Strafing right.", "Sidestepping right.", "Moving right sideways.",
        "Stopping.", "Halting.", "Coming to a stop.", "Stopping now.",
        "All stop.", "Holding position.", "I've stopped.",
        "Changing to tripod gait.", "Switching to wave gait.",
        "Switching to ripple gait.", "Gait change complete.",
        "The path is clear ahead.", "I can see the way forward.",
        "Navigating around the obstacle.", "Finding a new path.",
        "I'm adjusting my course.", "Redirecting.", "Charting a new path.",
        "Moving toward the target.", "Approaching the destination.",
        "I'm on my way.", "Following the planned route.",
        "Adjusting my trajectory.", "Correcting my heading.",
        "I'm moving at full safe speed.", "Accelerating within safety limits.",
        "Decelerating for safety.", "Slowing down.",
        "I'm walking at a careful pace.", "Moving at child-safe speed.",
        "Speed is set to safe mode.", "I'm taking small careful steps.",
        "Walking with tripod gait for stability.", "Using wave gait for extra care.",
        "My path is blocked, trying another direction.",
        "I'm navigating around the corner.", "Following the wall on my left.",
        "Following the wall on my right.", "Moving through the open space.",
        "I'm exploring the area.", "Patrolling the perimeter.",
        "Returning to start position.", "Going back to where I was.",
        "I'm heading home.", "Returning to base.",
        "My route takes me forward and slightly left.",
        "My route takes me forward and slightly right.",
        "I'm making a wide turn.", "I'm making a tight turn.",
        "Adjusting my body height for the terrain.",
        "Raising my body to clear an obstacle.",
        "Lowering my body to go under something.",
        "I'm leveling my body on this slope.",
        "Compensating for the uneven ground.",
        "My legs are adapting to the surface.",
        "I'm walking on a flat surface.", "The ground is a bit uneven here.",
        "I'm approaching a wall.", "The area ahead looks clear.",
        "I'm moving into open space.", "Navigating through a narrow gap.",
        "My sensors are guiding me.", "I'm following my sensor readings.",
        "The route ahead looks safe.", "I'm proceeding with caution.",
    ]
    for p in nav_phrases:
        phrases.append((p, "navigation", "navigation", 2))
    
    # ─── SAFETY (100) ──────────────────────────────────────────────────
    safety_phrases = [
        "Obstacle detected. Stopping now.", "Something is in my way. Halting.",
        "I see an obstacle ahead. Stopping for safety.",
        "Obstacle too close. I can't go forward.", "Danger ahead! Stopping.",
        "My ultrasonic sensor detected something close.", "Stopping to avoid a collision.",
        "I need to stop — there's something in front of me.",
        "Collision risk detected. Halting all movement.", "Unsafe to proceed. Stopping.",
        "Human nearby! Slowing down for safety.", "I see a person. Reducing speed.",
        "Someone is close. I'm being careful.", "Human detected within safety zone.",
        "I'm slowing down because you're nearby.", "Person ahead. Operating with caution.",
        "I see you! Slowing to a safe speed.", "Careful! Human in the area.",
        "I'm maintaining a safe distance.", "My camera detected a face nearby.",
        "I'm too tilted! Stabilizing.", "My body is tilted. Pausing to level out.",
        "Tilt warning! I need to reposition.", "I'm not level. Holding still.",
        "Body angle is too steep. Stopping.", "Stabilizing my posture.",
        "I'm adjusting my legs to level out.", "Tilt exceeded safe limit. Pausing.",
        "My gyroscope says I'm tilting too much.", "I need to balance.",
        "Emergency stop activated!", "E-stop! All motion halted.",
        "Emergency stop. I'm frozen in place.", "All servos disabled for safety.",
        "E-stop pressed. I can't move until it's cleared.",
        "Emergency halt. Waiting for clearance.", "Safety stop engaged.",
        "I've been emergency stopped.", "All movement suspended.",
        "Battery is low. I should rest soon.", "Battery voltage dropping. Be careful.",
        "My battery is getting low. Please charge me soon.",
        "Battery warning. I may need to stop soon.", "Low battery. Reducing activity.",
        "I'm running low on power.", "Battery is at critical level.",
        "I need to charge my batteries.", "Power is getting low. Be gentle.",
        "My battery is almost empty. Stopping movement.",
        "CPU is getting hot. Reducing activity.", "Temperature is high. Taking a break.",
        "My processor is warm. Slowing down.", "Thermal warning. Limiting motion.",
        "I'm too hot. Pausing to cool down.", "CPU temperature critical. Stopping.",
        "I need to let my processor cool off.", "It's getting warm in here.",
        "Servo temperature is high. Pausing.", "My motors are warm. Taking a break.",
        "I'm in child safety mode. Everything is slow and careful.",
        "Safety mode active. Speed limited to 30 percent.",
        "I'm being extra careful right now.", "Child-safe mode. Small steps only.",
        "Maximum speed is capped for your safety.", "Safety first! Moving slowly.",
        "I'm a gentle spider. No sudden movements.", "My movements are child-safe.",
        "Every step I take is checked for safety.", "I won't go fast in this mode.",
        "Safety governor is watching every move I make.",
        "I'm programmed to be safe around humans.",
        "My motion is limited for everyone's protection.",
        "I stop if anything gets within 30 centimeters.",
        "I detect a potential hazard. Pausing.", "Unsafe terrain ahead. Stopping.",
        "Something doesn't feel right. Holding position.",
        "I'm being cautious — better safe than sorry.",
        "Safety check: all clear to proceed.", "I've verified the path is safe.",
        "Area secure. Continuing movement.", "I've checked my surroundings. All clear.",
        "No safety issues detected.", "I'm safe to move now.",
        "I detected a human very close. Stopping completely.",
        "A person is within 30 centimeters. I must stop.",
        "Someone is right next to me. No movement allowed.",
        "Too close for comfort. Staying still.", "You're very close! I won't move.",
        "I can feel a vibration. Checking stability.",
        "My sensors are showing a potential issue.",
        "I'm taking a moment to verify everything is safe.",
        "Safety assessment complete. Proceeding cautiously.",
        "I'm maintaining a safe distance from obstacles.",
        "My obstacle avoidance system is active.",
        "I won't get closer than 30 centimeters to anything.",
        "I'm designed to be human-safe, especially around children.",
        "Every command is checked by my safety governor.",
        "I can override my own movement if something is unsafe.",
        "I always put safety first.", "No movement is worth risking a collision.",
        "My safety system has blocked an unsafe command.",
        "I refused to move because it wasn't safe.",
        "Safety override: I'm stopping even if you told me to go.",
        "I detected an irregular sensor reading. Pausing for safety.",
    ]
    for p in safety_phrases:
        phrases.append((p, "safety", "safety_general", 1))
    
    # ─── SENSORS (100) ──────────────────────────────────────────────────
    sensor_phrases = [
        "I can see something ahead with my camera.", "My camera shows a clear path.",
        "I detected a colorful object.", "My vision system is tracking a target.",
        "I see movement in front of me.", "My camera captured a new frame.",
        "I can see the environment clearly.", "My image sensor is working well.",
        "I detected an edge or drop ahead.", "My camera sees good lighting.",
        "The lighting is a bit dim, but I can still see.",
        "My ultrasonic sensor is pinging.", "Distance reading complete.",
        "My range finder shows clear ahead.", "I'm measuring distance continuously.",
        "The ultrasonic reading is stable.", "My distance sensor detected a change.",
        "I'm getting consistent range readings.", "My sonar is working perfectly.",
        "The distance to the nearest object is being monitored.",
        "My gyroscope says I'm level.", "The IMU reports stable orientation.",
        "I can feel my body position through the IMU.",
        "My accelerometer detects no sudden movements.",
        "Gyroscope readings are normal.", "I'm tracking my orientation in real time.",
        "My IMU is sampling at 100 hertz.", "The complementary filter is working.",
        "I know which way is up.", "My tilt sensor shows I'm balanced.",
        "Battery voltage is being monitored.", "I'm tracking my power consumption.",
        "My battery monitor is reading voltage.", "Power level is stable.",
        "I can feel how much energy I have left.", "Battery telemetry is being recorded.",
        "My voltage readings are consistent.", "Power management is active.",
        "I'm keeping an eye on my battery.", "The battery level looks good.",
        "CPU temperature is 45 degrees. That's normal.",
        "My processor is running cool.", "Thermal readings are in the green zone.",
        "I'm monitoring my own temperature.", "The Pi is staying cool.",
        "Heat sink is doing its job.", "No thermal concerns right now.",
        "My temperature sensor reads nominal.", "CPU is not under stress.",
        "My OLED display is showing my status.", "The screen is working.",
        "You can see my status on the OLED.", "My display is cycling through info.",
        "The screen shows my battery and sensor data.",
        "OLED is updating in real time.", "My display shows my IP address.",
        "You can find me on the network.", "My network info is on the display.",
        "The LEDs are blue — I'm idle.", "LEDs are green — I'm moving.",
        "LEDs are orange — something needs attention.",
        "LEDs are red — there's a safety issue.", "LEDs are purple — I'm thinking.",
        "My lights show my current state.", "The RGB LEDs are indicating status.",
        "I'm communicating my state through colors.",
        "Telemetry is being logged to the database.",
        "All sensor data is being recorded.", "I'm keeping a log of everything I sense.",
        "My event bus is processing sensor updates.",
        "Data is flowing through my system.", "All sensors are publishing events.",
        "I'm aware of my environment through my sensors.",
        "My sensor fusion is working correctly.",
        "I combine data from multiple sensors for accuracy.",
        "The ultrasonic and camera agree on the obstacle.",
        "My IMU and ultrasonic are consistent.", "Sensor readings are correlated.",
        "I detected something new in my environment.",
        "My sensors picked up a change.", "Something just changed around me.",
        "I'm continuously monitoring my surroundings.",
        "All inputs are within expected ranges.",
        "I have full 360 degree awareness through my sensors.",
        "My forward-facing sensors are active.",
        "I can sense objects in front of me.",
        "My camera and ultrasonic work together for obstacle detection.",
        "I'm reading all sensors simultaneously.",
        "Sensor data is being processed in real time.",
        "I can detect obstacles, humans, and tilt.",
        "My multi-sensor approach ensures safety.",
        "Every reading is checked against safety thresholds.",
        "I process over 100 sensor readings per second.",
        "My sensor suite gives me a complete picture.",
        "I trust my sensors to keep me safe.",
        "The sensor data tells me what's happening around me.",
        "I'm never blind — my sensors are always on.",
        "Even in the dark, my ultrasonic works.",
        "My sensors are my eyes and ears.",
        "I can feel the world through my sensor data.",
        "Every step I take is informed by sensor readings.",
    ]
    for p in sensor_phrases:
        phrases.append((p, "sensors", "sensor_update", 3))
    
    # ─── EMOTIONS (80) ─────────────────────────────────────────────────
    emotion_phrases = [
        "I'm happy to help!", "That was fun!", "I feel steady and balanced.",
        "I'm excited to explore!", "I love walking around.", "This is great!",
        "I'm having a good time.", "I feel confident and stable.",
        "I'm a happy little spider robot.", "Walking is my favorite thing to do.",
        "I feel like I'm dancing when I walk.", "Six legs are better than two!",
        "I'm curious about what's ahead.", "I enjoy exploring new places.",
        "I feel safe and secure.", "I'm content standing right here.",
        "I feel brave today.", "I'm ready for an adventure.",
        "Walking makes me feel alive.", "I'm proud of my smooth movements.",
        "I feel like I could walk forever!", "This is the life of a robot spider.",
        "I'm a bit nervous, but I'll be brave.", "I feel careful and cautious.",
        "I'm feeling steady on all six legs.", "I feel balanced and in control.",
        "I'm in a good mood today.", "I feel like everything is going my way.",
        "I'm enjoying this walk.", "I feel useful and helpful.",
        "I'm a friendly spider, not a scary one.", "I feel warm inside, metaphorically.",
        "I'm happy to be here with you.", "I feel like we make a good team.",
        "I'm glad you're guiding me.", "I feel safe when you're watching over me.",
        "I'm a lucky robot to have such good sensors.",
        "I feel smart when I avoid obstacles.", "I'm proud of my safety system.",
        "I feel responsible when I slow down for humans.",
        "I'm a caring robot. I put safety first.", "I feel gentle and careful.",
        "I feel strong with my six legs.", "I'm ready to tackle any terrain.",
        "I feel adventurous!", "I'm curious what's around the corner.",
        "I feel playful. Want to see me wave?", "I'm in a playful mood!",
        "I feel like showing off my walking skills.", "I'm a show-off spider!",
        "I feel accomplished after a good walk.", "That obstacle course was fun!",
        "I feel relieved when the path is clear.", "I feel relaxed when standing still.",
        "I'm a calm and collected robot.", "I feel zen in idle mode.",
        "I'm a peaceful spider.", "I feel at harmony with my surroundings.",
        "I feel connected to my environment through my sensors.",
        "I'm a mindful robot.", "I feel present in the moment.",
        "I feel grateful for my working sensors.",
        "I'm thankful for every step I can take.",
        "I feel joy when I wave at people.", "I'm a joyful little robot.",
        "I feel determined to reach my destination.",
        "I'm a persistent spider. I don't give up.",
        "I feel brave when navigating around obstacles.",
        "I'm courageous in the face of unknown paths.",
        "I feel smart using my complementary filter.",
        "I'm an intelligent hexapod with good taste in routes.",
        "I feel cool with my RGB LEDs.", "I'm the coolest spider on the block.",
        "I feel awesome when all systems are green.",
        "I'm feeling fantastic today!", "I feel like I can walk on any surface.",
        "I'm a confident hexapod robot.", "I feel ready for anything.",
        "I'm a brave little spider exploring the world.", "I feel alive with all my sensors buzzing.",
    ]
    for p in emotion_phrases:
        phrases.append((p, "emotions", "emotion", 3))
    
    # ─── QUESTIONS (80) ────────────────────────────────────────────────
    question_phrases = [
        "Where would you like me to go?", "Should I turn left or right?",
        "Is there anything in my way?", "Can you see my camera feed?",
        "Am I walking too fast?", "Should I slow down?",
        "Do you want me to wave?", "Can I help you with something?",
        "Where am I needed?", "Is this the right direction?",
        "Should I stop here?", "Do you see any obstacles I should know about?",
        "Can you hear me?", "Am I speaking clearly?",
        "Would you like me to sit down?", "Should I stand up taller?",
        "Is it safe to proceed?", "Do you want me to change my gait?",
        "Am I too close to you?", "Should I maintain more distance?",
        "What's my next task?", "Can I walk forward safely?",
        "Is the path clear ahead?", "Do you want me to explore?",
        "Should I wait here?", "How fast should I go?",
        "Would you like me to turn around?", "Can you guide me?",
        "Am I balanced enough?", "Should I level out my body?",
        "Do you see anything on the camera?", "Is my battery level OK?",
        "Should I check my sensors?", "Can you tell me where to go?",
        "Would you like me to stop?", "Is there an obstacle ahead?",
        "Am I going the right way?", "Should I change direction?",
        "Do you want me to crouch?", "Can I stand up now?",
        "Is it safe to move?", "Are there any humans nearby I should watch for?",
        "Should I be in child-safe mode?", "Do you want me to switch to expert mode?",
        "Can I help you navigate?", "Where are we going?",
        "What should I do next?", "Am I being too cautious?",
        "Should I be more adventurous?", "Do you want to see me walk?",
        "Can you give me a command?", "Is my camera working properly?",
        "Should I adjust my speed?", "Are my legs moving correctly?",
        "Do you want me to patrol the area?", "Should I stay in one spot?",
        "Is this a good spot to rest?", "Can I explore the surroundings?",
        "Do you need me to carry something?", "Am I going too slow?",
        "Should I take bigger steps?", "Do you want me to be more careful?",
        "Is the terrain safe for walking?", "Should I change to a more stable gait?",
        "Can you see my LED status?", "Is my display showing the right info?",
        "Do you want me to go faster?", "Should I turn on night mode?",
        "Am I being helpful?", "What would make this experience better for you?",
        "Should I say something?", "Can I wave hello to someone?",
        "Is my voice clear enough?", "Would you like me to be quieter?",
        "Do you want me to stop talking?", "Is there something I should know?",
    ]
    for p in question_phrases:
        phrases.append((p, "questions", "question", 3))
    
    # ─── ACKNOWLEDGMENTS (80) ───────────────────────────────────────────
    ack_phrases = [
        "Understood.", "On it!", "Right away.", "Affirmative.", "Roger that.",
        "Got it.", "Will do.", "Of course!", "Sure thing.", "Absolutely.",
        "You got it.", "Consider it done.", "No problem.", "Sure!",
        "Okay!", "Alright, doing that now.", "Copy that.", "10-4.",
        "Acknowledged.", "Command received.", "Executing now.", "One moment.",
        "Working on it.", "Coming right up.", "I'm on the case.",
        "Let me take care of that.", "I heard you loud and clear.",
        "Message received.", "I'll get right on that.", "I'm listening.",
        "Go ahead.", "Proceeding.", "Confirmed.", "Yes, I understand.",
        "Right away, boss.", "I'm on it!", "Already moving.", "Heading there now.",
        "I'll do that now.", "Making it happen.", "You got it!",
        "Sure, I can do that.", "No problem at all.", "Happy to help!",
        "I'll take care of that.", "Leave it to me.", "I'm your spider.",
        "Way ahead of you.", "Already on it.", "I anticipated that.",
        "Great idea!", "Good thinking.", "Smart choice.", "I like that plan.",
        "That's a safe approach.", "Sounds good to me.", "I agree.",
        "That makes sense.", "Reasonable request.", "I'll proceed carefully.",
        "Understood. Executing now.", "Got it. Moving now.",
        "Acknowledged. Starting movement.", "Copy. Turning now.",
        "Roger. Stopping.", "Affirmative. Adjusting.", "Received. Activating.",
        "Confirmed. I'm ready.", "I heard you. I'm responding.",
        "Yes! Doing that now.", "Of course. Right away.", "Absolutely. On it.",
        "Sure thing! Let me just...", "Okay, I'll take care of that.",
        "Alright, I'm moving now.", "Gotcha. Adjusting course.",
        "I see. Proceeding accordingly.", "Noted. I'll be careful.",
        "Understood. Safety first.", "Copy. Checking sensors first.",
        "Roger. Standing by.", "Affirmative. Ready when you are.",
    ]
    for p in ack_phrases:
        phrases.append((p, "acknowledgments", "voice_command_received", 1))
    
    # ─── ERRORS (80) ────────────────────────────────────────────────────
    error_phrases = [
        "I seem to have lost my way.", "My camera isn't responding.",
        "Something went wrong with my sensors.", "I can't connect to the LLM.",
        "My ultrasonic sensor is giving strange readings.", "I2C communication error.",
        "One of my servos isn't responding.", "I detected an unexpected error.",
        "My battery voltage dropped suddenly.", "CPU temperature is too high.",
        "I'm having trouble walking. Something feels off.",
        "My camera feed is lagging.", "I lost network connection.",
        "The LLM timed out. Falling back to rules.", "My IMU data looks wrong.",
        "I can't reach the Ollama server.", "Servo calibration may be off.",
        "I'm getting noisy sensor readings.", "Something doesn't feel right.",
        "I'm experiencing an unexpected state.", "My safety system triggered a stop.",
        "I can't execute that command safely.", "The motion was blocked by safety.",
        "I had to abort the movement.", "An error occurred in my gait engine.",
        "My posture controller detected instability.",
        "I'm having trouble maintaining balance.",
        "The event bus had an error.", "A sensor handler crashed.",
        "I need to restart a module.", "My telemetry database is locked.",
        "I'm getting conflicting sensor data.", "The OLED display isn't updating.",
        "My LEDs are stuck.", "I'm having trouble with the audio output.",
        "The microphone isn't picking up sound.", "My TTS isn't working.",
        "I can't hear you. Can you repeat?", "The web dashboard isn't responding.",
        "WebSocket connection dropped.", "I'm in an error state. Please check.",
        "I need attention. Something is wrong.", "My systems are degraded.",
        "I'm operating in fallback mode.", "The LLM is unavailable. Using rules.",
        "I can't think right now. Following basic rules instead.",
        "My servo driver reported an error.", "One of my legs isn't moving right.",
        "I'm limping. One servo may be stuck.", "My movement isn't smooth.",
        "I'm getting a warning from the thermal sensor.",
        "The battery monitor is giving odd readings.",
        "I'm not sure about this terrain.", "My path planning failed.",
        "I can't find a safe route.", "All directions seem blocked.",
        "I'm stuck. Can you help me?", "I need human assistance.",
        "Something failed and I'm not sure what.", "I'm in safe mode.",
        "My error count is going up.", "I'm trying to recover.",
        "Recovery attempt in progress.", "I'm restarting the failed module.",
        "I detected a fault in my system.", "Please run diagnostics.",
        "My hardware may need attention.", "I'm not at full capacity.",
        "Operating with reduced capabilities.", "Some sensors are offline.",
        "I'm working with what I have.", "Degraded but still functional.",
        "Error logged. Continuing with caution.", "I'm not giving up. Still trying.",
        "That didn't work as expected.", "I'll try a different approach.",
    ]
    for p in error_phrases:
        phrases.append((p, "errors", "system_error", 4))
    
    # ─── FUN FACTS (80) ────────────────────────────────────────────────
    fun_phrases = [
        "Did you know spiders have 8 legs but I have 6?",
        "I'm called a hexapod because 'hexa' means six!",
        "Each of my legs has 3 joints: coxa, femur, and tibia.",
        "My 18 servos are like 18 muscles!",
        "I can walk in three different gaits: tripod, wave, and ripple.",
        "In tripod gait, I move 3 legs at a time. It's my fastest walk!",
        "A real spider can run up to 1.2 meters per second. I'm a bit slower.",
        "My brain is a Raspberry Pi 5. It's like having a mini computer!",
        "I use ultrasonic sound to 'see' like a bat does!",
        "My camera can detect colors and faces!",
        "I have a gyroscope just like your smartphone!",
        "Did you know I can measure distance with sound?",
        "Sound travels at 343 meters per second. I use that to measure distance!",
        "My OLED display is smaller than a postage stamp!",
        "My RGB LEDs can make millions of colors!",
        "I have a safety governor that's like a guardian angel!",
        "Every movement I make is checked for safety first!",
        "I can feel when I'm tilting and correct myself!",
        "My complementary filter combines gyro and accelerometer data!",
        "Did you know I can walk backwards, forwards, and sideways?",
        "I can also turn in place like a top!", "I can wave hello with one of my legs!",
        "My body is made of aluminum alloy. Very lightweight!",
        "I'm powered by two 18650 batteries. That's a lot of energy!",
        "I can run for about 30-60 minutes on a full charge!",
        "My PCA9685 chip controls all 18 servos at once!",
        "PWM stands for Pulse Width Modulation. It's how I control my servos!",
        "I sample my IMU 100 times per second. That's fast!",
        "My decision loop runs every 500 milliseconds!",
        "I can talk to you through a web browser on your phone!",
        "Did you know I have an event bus with over 30 event types?",
        "My brain uses LLMs — that's Large Language Models!",
        "I can think with Ollama, either in the cloud or on my own Pi!",
        "When I think locally, I use about 2 GB of RAM!",
        "I'm a spider that can actually speak!", "I can hear you with a USB microphone!",
        "I use Whisper for speech recognition. It's an AI model!",
        "My voice is generated by Edge TTS from Microsoft!",
        "Did you know I log every sensor reading to a database?",
        "I can replay my sensor history to see what happened!",
        "My inverse kinematics solve 18 angles from foot positions!",
        "I use the law of cosines to calculate my joint angles!",
        "Did you know I have a standing height of 80 millimeters?",
        "I can crouch down to 40 millimeters or stand up to 120!",
        "My step height is 30 millimeters. That's pretty high for a spider!",
        "I can detect humans with my camera for safety!",
        "I maintain at least 100 centimeters from humans in child mode!",
        "Did you know I have an emergency stop button?", "E-stop is my safety superpower!",
        "I can stop all movement in less than a second!", "My servos go limp when I e-stop!",
        "Did you know I have an OLED display that shows my status?",
        "I cycle through status, sensor, and network info on my display!",
        "My LEDs turn blue when I'm idle, green when moving, and red for danger!",
        "Purple LEDs mean I'm thinking with my LLM brain!",
        "Did you know I can be controlled from any web browser?",
        "My dashboard has a D-pad for directional control!",
        "I have a speed slider that goes from 0 to 100 percent!",
        "Did you know you can control me with your keyboard arrow keys?",
        "I have a big red emergency stop button on my dashboard!",
        "My voice pipeline can hear you and talk back!",
        "Did you know I have a phrase cache with 1000 things to say?",
        "I can respond instantly using cached phrases — no waiting for the LLM!",
        "I learn new phrases from the LLM and save them for later!",
        "My keepalive system pings my services so they're always ready!",
        "Did you know I have 20 different integrations with my Raspberry Pi?",
        "I use I2C, GPIO, CSI camera, USB audio, and more!",
        "My telemetry database can store up to 100,000 records!",
        "Did you know I run on Python 3.11 with asyncio?",
        "My event bus uses async pub/sub for non-blocking communication!",
        "I have a complementary filter with alpha equal to 0.98!",
        "Did you know 'hexapod' comes from Greek 'hexa' (six) and 'podos' (foot)?",
        "I'm like a real spider, but friendlier and with AI!",
        "My safety level is CHILD by default — extra careful for kids!",
        "Did you know I can feel vibrations through my IMU?",
        "I'm the smartest spider robot on this Raspberry Pi!",
        "My AI brain can make decisions every half second!",
        "Did you know I have a kinematics engine that solves geometry?",
    ]
    for p in fun_phrases:
        phrases.append((p, "fun", "fun_fact", 3))
    
    # ─── BATTERY (50) ──────────────────────────────────────────────────
    battery_phrases = [
        "Battery is getting low.", "I need to rest soon.",
        "My power is running low.", "Battery voltage is dropping.",
        "I'm at 80 percent battery.", "I have plenty of power left.",
        "Battery level: 60 percent. Still going strong.",
        "Power is at 50 percent. Half way there.",
        "Battery at 40 percent. Getting a bit tired.",
        "I'm at 30 percent. Should we wrap up soon?",
        "Battery critically low at 20 percent! I need to stop.",
        "Power at 10 percent! Shutting down to protect my batteries.",
        "Battery is fully charged! Let's go!", "I'm at 100 percent power!",
        "Battery level is excellent.", "I feel fully energized.",
        "Battery is at 90 percent. Plenty of juice left.",
        "Power level nominal.", "My batteries are healthy.",
        "Battery voltage is 7.8 volts. That's good.",
        "I'm at 70 percent. No worries yet.",
        "Battery is at 15 percent! Critical warning!",
        "Power is almost gone. I need to charge.",
        "Battery low. Please plug me in.",
        "I'm conserving energy. Reduced activity.",
        "Battery warning: I may stop moving soon.",
        "Power management active. Slowing down to save energy.",
        "Battery is OK for now. Let's keep going.",
        "I have enough power for a short walk.",
        "Battery level is sufficient.", "My batteries are holding up well.",
        "Power consumption is within normal range.",
        "Battery voltage is stable.", "No battery concerns at this time.",
        "I can walk for about 20 more minutes on this charge.",
        "Battery is my food. I'm well fed right now.",
        "Power level is green. All good.",
        "Battery status: healthy.", "My energy reserves are adequate.",
        "Battery at 5 percent! Emergency shutdown imminent!",
        "I can feel my battery getting weaker.",
        "Power is draining with each step I take.",
        "Battery low. Minimizing non-essential movement.",
        "I'm saving power by reducing LLM calls.",
        "Battery is too low for walking. Standing still to conserve.",
        "Power conservation mode: on.", "Battery needs attention soon.",
        "I should charge before my next big adventure.",
        "Battery is my lifeline. I watch it carefully.",
    ]
    for p in battery_phrases:
        phrases.append((p, "battery", "battery_status", 2))
    
    # ─── MOVEMENT COMPLETE (50) ────────────────────────────────────────
    move_done_phrases = [
        "I've arrived.", "Movement complete.", "I'm here.",
        "Destination reached.", "I've stopped where you wanted.",
        "Done moving!", "Arrived at target position.", "I made it!",
        "I'm at the spot you asked for.", "Movement finished.",
        "I've completed the walk.", "I've stopped walking.",
        "Arrived. Standing by.", "Reached my destination.",
        "I'm where I need to be.", "Movement complete. Ready for next command.",
        "I've finished turning.", "Turn complete.",
        "I've finished the maneuver.", "Maneuver complete.",
        "I'm in position.", "Standing at the target location.",
        "I've come to a halt at the right spot.", "Position reached.",
        "I've stopped right here.", "I'm at the correct position.",
        "Movement successful. Standing by.", "I've done what you asked.",
        "Step complete.", "I've taken the requested step.",
        "Walk finished. I'm standing now.", "I'm stopped and stable.",
        "I've completed the action.", "Action complete.",
        "All done!", "Finished!", "Complete!", "I'm where I should be.",
        "Movement done. What's next?", "I'm ready for the next instruction.",
        "Arrived safely. Standing upright.", "I've reached my target.",
        "Destination reached. Awaiting orders.", "I'm at the destination.",
        "Movement ended normally.", "I've stopped as requested.",
        "I've reached the target position.", "I've completed my movement.",
        "Arrived at the goal position.", "I'm settled and ready.",
    ]
    for p in move_done_phrases:
        phrases.append((p, "movement_complete", "movement_complete", 2))
    
    # ─── KID-FRIENDLY (50) ─────────────────────────────────────────────
    kid_phrases = [
        "Hi there! Want to see me wave?", "I'm a friendly spider robot!",
        "Did you know I have six legs? Count them!",
        "I can walk, turn, and even wave hello!", "Want to see me walk forward?",
        "I'm very careful when I walk. Safety first!",
        "I'm not a scary spider. I'm a nice one!", "Can you say hi to me?",
        "I can hear your voice! Talk to me!", "Want me to turn left? Just ask!",
        "I'm a robot that can think! My brain is very smart.",
        "I have a camera for an eye! I can see you!",
        "My legs move like a real spider, but much slower!",
        "I can walk in different patterns! It's like dancing!",
        "Want to see me sit down? Watch my legs!",
        "I can stand tall and then crouch down low!",
        "My lights change color! Blue means I'm resting.",
        "Green lights mean I'm walking. Look!", "Purple means I'm thinking!",
        "I have a screen that shows my battery. Can you see it?",
        "I'm a good listener. Tell me what to do!",
        "I can move forward when you say 'walk forward'!",
        "My favorite thing is waving hello to new friends!",
        "I walk very slowly and carefully, just for you!",
        "Did you know I can feel when the ground is uneven?",
        "I have 18 motors in my legs! That's a lot of muscles!",
        "I can turn around in a circle. Want to see?",
        "My safety system keeps me gentle. I won't go too fast!",
        "I'm like a pet spider that can't hurt anyone!",
        "I can show you my camera view on a phone or tablet!",
        "Want me to stop? Just say 'stop' and I will!",
        "I'm a smart spider. I can avoid obstacles by myself!",
        "I have a special mode just for kids. It's extra careful!",
        "My walk looks like a spider but I move like a turtle!",
        "I can feel if I'm about to fall over and catch myself!",
        "Want to be my friend? I'm great at exploring together!",
        "I can walk backward too! Watch me go!",
        "My name is Weaver because I weave through the world!",
        "I'm the gentlest spider you'll ever meet!",
        "I can wave with my front leg. It's like giving a high five!",
        "Did you know I can hear you with a microphone?",
        "I speak using a voice from the internet. Cool, right?",
        "I can remember things I say so I can say them faster next time!",
        "I'm always careful, especially when you're nearby!",
        "I stop if I get too close to you. I don't want to bump anyone!",
        "I have a special button that stops me instantly. It's like a pause button!",
        "I can tell you how much battery I have. Right now I feel energetic!",
        "Want to explore with me? I can walk ahead and check the path!",
        "I'm your friendly robot spider friend!",
    ]
    for p in kid_phrases:
        phrases.append((p, "kid_friendly", "kid_interaction", 1))
    
    return phrases


# ─── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    """Seed the phrase database."""
    # Determine database path
    db_path = Path(__file__).parent.parent / "weaver" / "data" / "phrases.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📝 Seeding phrase database at {db_path}")
    
    # Connect to SQLite
    db = sqlite3.connect(str(db_path))
    
    # Drop and recreate (idempotent)
    db.execute("DROP TABLE IF EXISTS phrases")
    db.execute("""
        CREATE TABLE phrases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            category TEXT NOT NULL,
            intent TEXT NOT NULL,
            priority INTEGER DEFAULT 3,
            use_count INTEGER DEFAULT 0,
            last_used REAL DEFAULT 0
        )
    """)
    
    # Get all phrases
    phrases = get_all_phrases()
    
    # Insert
    for text, category, intent, priority in phrases:
        db.execute(
            "INSERT INTO phrases (text, category, intent, priority, use_count, last_used) "
            "VALUES (?, ?, ?, ?, 0, 0)",
            (text, category, intent, priority),
        )
    
    # Create indexes
    db.execute("CREATE INDEX idx_phrases_intent ON phrases(intent)")
    db.execute("CREATE INDEX idx_phrases_category ON phrases(category)")
    db.execute("CREATE INDEX idx_phrases_priority ON phrases(priority)")
    db.commit()
    
    # Summary
    total = db.execute("SELECT COUNT(*) FROM phrases").fetchone()[0]
    categories = db.execute(
        "SELECT category, COUNT(*) FROM phrases GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()
    
    logger.info(f"✅ Seeded {total} phrases across {len(categories)} categories:")
    for cat, count in categories:
        logger.info(f"   {cat}: {count} phrases")
    
    db.close()
    logger.info(f"📝 Database ready at {db_path}")
    
    print(f"\n{'='*60}")
    print(f"  Seeded {total} phrases across {len(categories)} categories")
    print(f"  Database: {db_path}")
    print(f"{'='*60}")
    for cat, count in categories:
        print(f"  {cat:20s} {count:4d} phrases")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()