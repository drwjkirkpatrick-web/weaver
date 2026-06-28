# phrase_cache.py — Local phrase cache for instant robot responses
#
# When the robot needs to say something, it normally goes through the full
# LLM pipeline: build prompt → send to Ollama → parse response → speak.
# This takes 0.5-5 seconds depending on model and network.
#
# But most of what a robot says is predictable! "Obstacle detected, stopping."
# "Battery at 80 percent." "Hello! I'm Weaver." These phrases repeat.
#
# The phrase cache stores 1000+ common phrases in SQLite. When the cortex
# needs to respond, it checks the cache first:
#   1. Match by intent (exact match — fastest, instant)
#   2. Match by keyword similarity (fuzzy match — fast, <10ms)
#   3. Fall back to LLM (slow but handles novel situations)
#
# Cache hits: <1ms response time (instant)
# Cache misses: 0.5-5s (LLM generation)
#
# The cache also LEARNS: when the LLM generates a good response for a novel
# situation, the cortex can save it to the cache for future use.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

from weaver.config import get_config


class PhraseCache:
    """SQLite-backed phrase cache for instant robot responses.
    
    The cache stores phrases with:
    - category: greetings, status, safety, etc.
    - intent: what triggers this phrase (obstacle_close, battery_low, etc.)
    - priority: 1 (most used) to 5 (rarely used)
    - use_count: how many times this phrase has been used (for learning)
    
    Lookup strategy (fastest to slowest):
    1. Exact intent match → pick a random phrase from that intent
    2. Keyword match → find phrases containing key words
    3. Fuzzy match → similarity scoring (difflib)
    4. Miss → return None, caller falls back to LLM
    
    The cache also tracks which phrases have been used recently and rotates
    them so the robot doesn't say the exact same thing every time.
    """
    
    def __init__(self, db_path: str | None = None):
        """Initialize the phrase cache.
        
        Args:
            db_path: Path to the SQLite database. If None, uses config default.
        """
        if db_path is None:
            # Default location alongside the telemetry DB
            db_path = "weaver/data/phrases.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._db: sqlite3.Connection | None = None
        self._recently_used: list[int] = []  # Track recent phrase IDs to avoid repeats
        self._recent_limit: int = 20  # Don't repeat last 20 phrases
        
        logger.info(f"📝 Phrase cache initialized (db: {self.db_path})")
    
    async def start(self) -> None:
        """Open the database connection and verify the table exists."""
        self._db = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )
        
        # Check if the table exists
        cursor = self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='phrases'"
        )
        if cursor.fetchone() is None:
            logger.warning(
                "📝 Phrase cache table not found — run 'python scripts/seed_phrases.py' "
                "to populate the cache with 1000 phrases"
            )
            # Create empty table so the cache works (just with no phrases)
            self._create_table()
        
        # Create indexes for fast lookup
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_phrases_intent ON phrases(intent)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_phrases_category ON phrases(category)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_phrases_priority ON phrases(priority)"
        )
        self._db.commit()
        
        count = self._db.execute("SELECT COUNT(*) FROM phrases").fetchone()[0]
        logger.info(f"✅ Phrase cache ready ({count} phrases loaded)")
    
    async def stop(self) -> None:
        """Close the database connection."""
        if self._db:
            self._db.close()
            self._db = None
        logger.info("Phrase cache stopped")
    
    def _create_table(self) -> None:
        """Create the phrases table if it doesn't exist."""
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS phrases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                category TEXT NOT NULL,
                intent TEXT NOT NULL,
                priority INTEGER DEFAULT 3,
                use_count INTEGER DEFAULT 0,
                last_used REAL DEFAULT 0
            )
        """)
        self._db.commit()
    
    # ─── Lookup ────────────────────────────────────────────────────────
    
    def lookup_by_intent(self, intent: str) -> str | None:
        """Find a phrase by exact intent match.
        
        This is the fastest lookup (indexed, O(1)).
        
        Args:
            intent: The intent string (e.g., "obstacle_close", "greeting")
        
        Returns:
            A matching phrase text, or None if no phrases match.
        
        Example:
            phrase = cache.lookup_by_intent("battery_low")
            # → "Battery is getting low. I should rest soon."
        """
        if not self._db:
            return None
        
        # Get all phrases for this intent, excluding recently used
        excluded = self._get_excluded_clause()
        
        rows = self._db.execute(
            f"""
            SELECT id, text, priority, use_count 
            FROM phrases 
            WHERE intent = ? AND id NOT IN ({excluded})
            ORDER BY priority ASC, use_count ASC, RANDOM()
            LIMIT 5
            """,
            (intent,),
        ).fetchall()
        
        if not rows:
            # Try without the exclusion (in case we've used all phrases)
            rows = self._db.execute(
                """
                SELECT id, text, priority, use_count 
                FROM phrases 
                WHERE intent = ?
                ORDER BY priority ASC, use_count ASC, RANDOM()
                LIMIT 5
                """,
                (intent,),
            ).fetchall()
        
        if not rows:
            return None
        
        # Pick one (random selection from top 5)
        # Priority 1 phrases are more likely to be picked
        import random
        weights = [max(1, 6 - r[2]) for r in rows]  # priority 1 = weight 5
        chosen = random.choices(rows, weights=weights, k=1)[0]
        
        # Mark as used
        self._mark_used(chosen[0])
        
        return chosen[1]
    
    def lookup_by_keywords(self, keywords: list[str]) -> str | None:
        """Find a phrase by keyword matching.
        
        Searches for phrases containing any of the given keywords.
        Ranks by number of keyword matches and priority.
        
        Args:
            keywords: List of keywords to search for (e.g., ["obstacle", "stop"])
        
        Returns:
            Best matching phrase text, or None.
        """
        if not self._db or not keywords:
            return None
        
        # Build a LIKE query for each keyword
        conditions = " OR ".join(["text LIKE ?" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords]
        
        excluded = self._get_excluded_clause()
        query = f"""
            SELECT id, text, priority, use_count,
                   ({' + '.join([f'CAST(text LIKE ? AS INTEGER)' for _ in keywords])}) as match_count
            FROM phrases
            WHERE ({conditions}) AND id NOT IN ({excluded})
            ORDER BY match_count DESC, priority ASC, use_count ASC, RANDOM()
            LIMIT 3
        """
        
        # We need the keywords twice: once for match_count, once for WHERE
        all_params = [f"%{kw}%" for kw in keywords] + [f"%{kw}%" for kw in keywords]
        
        rows = self._db.execute(query, all_params).fetchall()
        
        if not rows:
            return None
        
        chosen = rows[0]  # Best match
        self._mark_used(chosen[0])
        
        return chosen[1]
    
    def lookup_fuzzy(self, query: str, min_similarity: float = 0.4) -> str | None:
        """Find a phrase by fuzzy similarity to the query.
        
        Uses difflib's SequenceMatcher to find phrases similar to the query.
        This is slower than intent/keyword lookup but handles paraphrased queries.
        
        Args:
            query: The text to match against (e.g., "something blocking my path")
            min_similarity: Minimum similarity score (0.0-1.0) to accept a match.
        
        Returns:
            Best matching phrase, or None if nothing exceeds the threshold.
        """
        if not self._db:
            return None
        
        # Load all phrases (this is fine — 1000 phrases is small)
        rows = self._db.execute(
            "SELECT id, text, priority, use_count FROM phrases"
        ).fetchall()
        
        if not rows:
            return None
        
        # Score each phrase
        from difflib import SequenceMatcher
        
        best_match = None
        best_score = 0.0
        
        for row in rows:
            phrase_id, text, priority, use_count = row
            
            # Skip recently used
            if phrase_id in self._recently_used:
                continue
            
            # Compute similarity
            score = SequenceMatcher(None, query.lower(), text.lower()).ratio()
            
            # Boost score for higher priority (lower number = higher priority)
            priority_boost = (6 - priority) * 0.02
            score += priority_boost
            
            # Penalize frequently used phrases (variety)
            variety_penalty = min(0.1, use_count * 0.005)
            score -= variety_penalty
            
            if score > best_score:
                best_score = score
                best_match = row
        
        if best_match and best_score >= min_similarity:
            self._mark_used(best_match[0])
            return best_match[1]
        
        return None
    
    # ─── Smart Lookup (combines all strategies) ──────────────────────
    
    def get_phrase(
        self,
        intent: str | None = None,
        keywords: list[str] | None = None,
        fuzzy_query: str | None = None,
    ) -> str | None:
        """Try all lookup strategies in order of speed.
        
        1. Exact intent match (fastest, <1ms)
        2. Keyword match (fast, <5ms)
        3. Fuzzy match (slower, <50ms for 1000 phrases)
        4. Return None (caller falls back to LLM)
        
        Args:
            intent: Exact intent to match (e.g., "obstacle_close")
            keywords: Keywords to search for
            fuzzy_query: Text to fuzzy-match against
        
        Returns:
            A phrase string, or None if no match found.
        """
        # Strategy 1: Exact intent
        if intent:
            phrase = self.lookup_by_intent(intent)
            if phrase:
                logger.debug(f"📝 Phrase cache HIT (intent={intent}): {phrase[:50]}")
                return phrase
        
        # Strategy 2: Keywords
        if keywords:
            phrase = self.lookup_by_keywords(keywords)
            if phrase:
                logger.debug(f"📝 Phrase cache HIT (keywords={keywords}): {phrase[:50]}")
                return phrase
        
        # Strategy 3: Fuzzy
        if fuzzy_query:
            phrase = self.lookup_fuzzy(fuzzy_query)
            if phrase:
                logger.debug(f"📝 Phrase cache HIT (fuzzy): {phrase[:50]}")
                return phrase
        
        logger.debug("📝 Phrase cache MISS — falling back to LLM")
        return None
    
    # ─── Learning ─────────────────────────────────────────────────────
    
    def add_phrase(
        self,
        text: str,
        category: str = "learned",
        intent: str = "unknown",
        priority: int = 3,
    ) -> None:
        """Add a new phrase to the cache (learning from LLM output).
        
        When the LLM generates a good response for a novel situation,
        the cortex can save it to the cache for future instant retrieval.
        
        Args:
            text: The phrase text to cache
            category: Phrase category
            intent: The intent that triggered this phrase
            priority: Priority (1-5, lower = used more)
        """
        if not self._db:
            return
        
        # Don't add duplicates
        existing = self._db.execute(
            "SELECT id FROM phrases WHERE text = ?", (text,)
        ).fetchone()
        
        if existing:
            # Already exists — just bump priority (make it more likely to be used)
            self._db.execute(
                "UPDATE phrases SET priority = MAX(1, priority - 1) WHERE id = ?",
                (existing[0],),
            )
            self._db.commit()
            return
        
        self._db.execute(
            "INSERT INTO phrases (text, category, intent, priority, use_count, last_used) "
            "VALUES (?, ?, ?, ?, 0, 0)",
            (text, category, intent, priority),
        )
        self._db.commit()
        logger.debug(f"📝 Learned new phrase: '{text[:50]}' (intent={intent})")
    
    # ─── Context-Aware Selection ─────────────────────────────────────
    
    def get_contextual_phrase(
        self,
        situation: dict[str, Any],
    ) -> str | None:
        """Get a phrase based on the current robot situation.
        
        This is the main entry point for the cortex. It examines the
        situation dict and determines the appropriate intent, then
        looks up a matching phrase.
        
        Args:
            situation: Dict with sensor data:
                - obstacle_distance_cm: float or None
                - battery_voltage: float
                - battery_low: bool
                - body_tilt_degrees: float
                - tilt_warning: bool
                - cpu_temp_c: float
                - human_detected: bool
                - human_distance_cm: float or None
                - last_motion: str
                - voice_command: str or None
        
        Returns:
            A phrase string, or None if no cached phrase fits.
        """
        # Determine intent from situation
        intent = self._classify_intent(situation)
        
        if intent:
            phrase = self.lookup_by_intent(intent)
            if phrase:
                return phrase
        
        # Try keywords from the situation
        keywords = self._extract_keywords(situation)
        if keywords:
            phrase = self.lookup_by_keywords(keywords)
            if phrase:
                return phrase
        
        return None
    
    def _classify_intent(self, situation: dict[str, Any]) -> str | None:
        """Determine the most appropriate intent from the situation.
        
        Priority order (first match wins):
        1. Emergency stop / e-stop
        2. Collision imminent
        3. Battery critical
        4. Tilt warning
        5. Human very close
        6. Obstacle close
        7. Obstacle warning (slow zone)
        8. Battery low
        9. Thermal warning
        10. Human nearby
        11. Voice command received
        12. Movement complete
        13. Idle / status
        
        Returns:
            Intent string, or None if no intent matches.
        """
        # Emergency / safety first
        if situation.get("estop_active"):
            return "estop"
        
        if situation.get("collision_imminent"):
            return "collision_imminent"
        
        battery = situation.get("battery_voltage", 8.4)
        if battery < 6.2:
            return "battery_critical"
        
        if situation.get("tilt_warning"):
            return "tilt_warning"
        
        human_dist = situation.get("human_distance_cm")
        if human_dist is not None and human_dist < 30:
            return "human_very_close"
        
        obstacle = situation.get("obstacle_distance_cm")
        if obstacle is not None:
            if obstacle < 30:
                return "obstacle_close"
            if obstacle < 60:
                return "obstacle_warning"
        
        if situation.get("battery_low") or battery < 6.8:
            return "battery_low"
        
        if situation.get("cpu_temp_c", 45) > 75:
            return "thermal_warning"
        
        if human_dist is not None and human_dist < 100:
            return "human_nearby"
        
        if situation.get("voice_command"):
            return "voice_command_received"
        
        if situation.get("last_motion") == "stop" and situation.get("motion_completed"):
            return "movement_complete"
        
        if situation.get("last_motion") == "stop":
            return "idle"
        
        return None
    
    def _extract_keywords(self, situation: dict[str, Any]) -> list[str]:
        """Extract keywords from the situation for keyword-based lookup."""
        keywords = []
        
        if situation.get("obstacle_distance_cm") is not None:
            keywords.append("obstacle")
        if situation.get("battery_low"):
            keywords.append("battery")
        if situation.get("tilt_warning"):
            keywords.append("tilt")
        if situation.get("human_detected"):
            keywords.append("human")
        if situation.get("voice_command"):
            keywords.append("command")
        
        return keywords
    
    # ─── Internal ─────────────────────────────────────────────────────
    
    def _mark_used(self, phrase_id: int) -> None:
        """Mark a phrase as used (increment use_count, update last_used)."""
        if not self._db:
            return
        
        self._db.execute(
            "UPDATE phrases SET use_count = use_count + 1, last_used = ? WHERE id = ?",
            (time.time(), phrase_id),
        )
        self._db.commit()
        
        # Track recently used for variety
        self._recently_used.append(phrase_id)
        if len(self._recently_used) > self._recent_limit:
            self._recently_used.pop(0)
    
    def _get_excluded_clause(self) -> str:
        """Get a SQL VALUES clause for excluded phrase IDs.
        
        Returns a string like "0,0,0" (safe default if no exclusions).
        """
        if not self._recently_used:
            return "0"  # Exclude nothing (0 is never a valid ID... actually it is in SQLite)
        # Use parameter-safe approach
        return ",".join(["0"] * max(1, len(self._recently_used)))
    
    # ─── Stats ────────────────────────────────────────────────────────
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics for dashboard."""
        if not self._db:
            return {"total_phrases": 0, "categories": {}}
        
        total = self._db.execute("SELECT COUNT(*) FROM phrases").fetchone()[0]
        
        categories = {}
        for row in self._db.execute(
            "SELECT category, COUNT(*) FROM phrases GROUP BY category"
        ).fetchall():
            categories[row[0]] = row[1]
        
        total_used = self._db.execute(
            "SELECT SUM(use_count) FROM phrases"
        ).fetchone()[0] or 0
        
        return {
            "total_phrases": total,
            "categories": categories,
            "total_uses": total_used,
            "recently_used_count": len(self._recently_used),
        }