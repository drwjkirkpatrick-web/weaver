# macarena.py — The Macarena dance choreography for Weaver
#
# The Macarena is a classic dance from 1996 (Los del Río).
# BPM: ~103 (original) — we use 100 for a comfortable robot pace.
#
# The Macarena has a repeating sequence of arm/hand movements:
#   1. Right arm out (palm down)
#   2. Left arm out (palm down)
#   3. Right arm flip (palm up)
#   4. Left arm flip (palm up)
#   5. Right arm to left shoulder
#   6. Left arm to right shoulder
#   7. Right arm behind head
#   8. Left arm behind head
#   9. Hip sway right
#   10. Hip sway left
#
# For the hexapod, we translate arm movements to front leg movements
# and hip sways to body sways. The hexapod "arms" are the front legs.
#
# Each phrase is 8 beats. The full Macarena repeats the 10-move sequence.
# We program 2 full repetitions (20 moves × ~8 beats each = ~160 beats).
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from weaver.dance.dance_engine import DanceRoutine
from weaver.dance.student_api import StudentDance


def create_macarena() -> StudentDance:
    """Create the Macarena dance as a StudentDance.
    
    Returns:
        A StudentDance object with the Macarena choreography.
    
    The Macarena translates to hexapod moves like this:
    
    Beat 1-2: "Right arm out" → front-right leg extends out
    Beat 3-4: "Left arm out" → front-left leg extends out
    Beat 5-6: "Right arm flip" → front-right leg waves
    Beat 7-8: "Left arm flip" → front-left leg waves
    Beat 9-10: "Right to shoulder" → front-right leg crosses in
    Beat 11-12: "Left to shoulder" → front-left leg crosses in
    Beat 13-14: "Right behind head" → front-right leg up high
    Beat 15-16: "Left behind head" → front-left leg up high
    Beat 17-18: "Hip right" → body sway right
    Beat 19-20: "Hip left" → body sway left
    Beat 21-24: "Turn" → spin right (4 beats)
    
    Then repeat!
    """
    dance = StudentDance(
        name="Macarena",
        bpm=100.0,  # Macarena original is ~103, we slow slightly for robot
        creator="Weaver Team",
        description=(
            "The classic Macarena! Right arm out, left arm out, "
            "flip, cross, behind head, hip sway, and turn. "
            "Translated to hexapod front-leg gestures and body sways."
        ),
    )
    
    # ─── Phrase 1: Arms out (beats 1-8) ───────────────────────────────
    
    # "Right arm out — palm down" (beat 1-2)
    dance.add("arm_out_right", beats=2)
    
    # "Left arm out — palm down" (beat 3-4)
    dance.add("arm_out_left", beats=2)
    
    # "Right arm — flip palm up" (beat 5-6)
    dance.add("wave_FR", beats=2)
    
    # "Left arm — flip palm up" (beat 7-8)
    dance.add("wave_FL", beats=2)
    
    # ─── Phrase 2: Cross arms (beats 9-16) ────────────────────────────
    
    # "Right arm to left shoulder" (beat 9-10)
    dance.add("arms_crossed", beats=2)
    
    # "Left arm to right shoulder" (beat 11-12)
    dance.add("arms_crossed", beats=2)
    
    # "Right arm behind head" (beat 13-14)
    dance.add("lift_FR", beats=2)
    
    # "Left arm behind head" (beat 15-16)
    dance.add("lift_FL", beats=2)
    
    # ─── Phrase 3: Hip sway + turn (beats 17-24) ─────────────────────
    
    # "Hip sway right" (beat 17-18)
    dance.add("sway_right", beats=2)
    
    # "Hip sway left" (beat 19-20)
    dance.add("sway_left", beats=2)
    
    # "Hip sway right again" (beat 21-22)
    dance.add("sway_right", beats=2)
    
    # "Turn right!" (beat 23-24)
    dance.add("spin_right", beats=2)
    
    # ─── Phrase 4: Second repetition (beats 25-48) ────────────────────
    
    # Repeat the full sequence (now facing a new direction after the turn)
    dance.add("arm_out_right", beats=2)   # 25-26
    dance.add("arm_out_left", beats=2)    # 27-28
    dance.add("wave_FR", beats=2)         # 29-30
    dance.add("wave_FL", beats=2)         # 31-32
    dance.add("arms_crossed", beats=2)    # 33-34
    dance.add("arms_crossed", beats=2)    # 35-36
    dance.add("lift_FR", beats=2)         # 37-38
    dance.add("lift_FL", beats=2)         # 39-40
    dance.add("sway_right", beats=2)      # 41-42
    dance.add("sway_left", beats=2)       # 43-44
    dance.add("sway_right", beats=2)      # 45-46
    dance.add("spin_right", beats=2)      # 47-48
    
    # ─── Phrase 5: Finale! (beats 49-56) ──────────────────────────────
    
    # Big finish: arms up + bounce
    dance.add("arms_up", beats=2)         # 49-50: "Hey Macarena!"
    dance.add("bounce_up", beats=2)       # 51-52: bounce
    dance.add("arms_up", beats=2)         # 53-54: arms up again
    dance.add("standing", beats=2)        # 55-56: finish standing
    
    return dance


def create_macarena_routine() -> DanceRoutine:
    """Create the Macarena as a DanceRoutine (for direct engine use).
    
    Returns:
        A DanceRoutine ready to pass to DanceEngine.dance().
    """
    return create_macarena().build()


# ─── Quick demo ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Print the choreography
    dance = create_macarena()
    print(dance.info())
    print("\n🕷️ This is the Macarena choreography for Weaver the hexapod robot!")
    print("   Each move translates arm gestures to front-leg movements.")
    print("   The dance repeats twice with a big finale!\n")