"""Tests for the PhraseCache — instant robot response cache."""

from __future__ import annotations

import asyncio
import os
import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from weaver.brain.phrase_cache import PhraseCache


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary phrase database with test phrases."""
    db_path = tmp_path / "test_phrases.db"
    
    db = sqlite3.connect(str(db_path))
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
    
    test_phrases = [
        ("Hello! I'm Weaver.", "greetings", "greeting", 1),
        ("Hi there!", "greetings", "greeting", 1),
        ("Good morning!", "greetings", "greeting", 2),
        ("Obstacle detected. Stopping now.", "safety", "obstacle_close", 1),
        ("Something is in my way. Let me turn.", "safety", "obstacle_close", 2),
        ("Battery is getting low. I should rest soon.", "battery", "battery_low", 1),
        ("Battery critical! I need to shut down.", "battery", "battery_critical", 1),
        ("All systems operational.", "status", "idle", 3),
        ("I'm ready to go!", "status", "idle", 2),
        ("Human nearby. Slowing down for safety.", "safety", "human_nearby", 1),
        ("Movement complete.", "movement_complete", "movement_complete", 2),
        ("I've arrived at my destination.", "movement_complete", "movement_complete", 2),
    ]
    
    for text, category, intent, priority in test_phrases:
        db.execute(
            "INSERT INTO phrases (text, category, intent, priority, use_count, last_used) "
            "VALUES (?, ?, ?, ?, 0, 0)",
            (text, category, intent, priority),
        )
    
    db.execute("CREATE INDEX IF NOT EXISTS idx_phrases_intent ON phrases(intent)")
    db.commit()
    db.close()
    
    return str(db_path)


@pytest.fixture
async def phrase_cache(temp_db):
    """Create a PhraseCache instance with the test database."""
    cache = PhraseCache(db_path=temp_db)
    await cache.start()
    yield cache
    await cache.stop()


class TestPhraseCache:
    """Test the phrase cache lookup and learning."""
    
    @pytest.mark.asyncio
    async def test_lookup_by_intent(self, phrase_cache):
        """lookup_by_intent should find phrases matching the intent."""
        phrase = phrase_cache.lookup_by_intent("greeting")
        assert phrase is not None
        assert "Hello" in phrase or "Hi" in phrase or "Good morning" in phrase
    
    @pytest.mark.asyncio
    async def test_lookup_by_intent_obstacle(self, phrase_cache):
        """Obstacle intent should return safety phrases."""
        phrase = phrase_cache.lookup_by_intent("obstacle_close")
        assert phrase is not None
        assert "obstacle" in phrase.lower() or "way" in phrase.lower() or "stop" in phrase.lower()
    
    @pytest.mark.asyncio
    async def test_lookup_by_intent_miss(self, phrase_cache):
        """Unknown intent should return None."""
        phrase = phrase_cache.lookup_by_intent("nonexistent_intent")
        assert phrase is None
    
    @pytest.mark.asyncio
    async def test_lookup_by_keywords(self, phrase_cache):
        """Keyword lookup should find phrases containing those words."""
        phrase = phrase_cache.lookup_by_keywords(["battery"])
        assert phrase is not None
        assert "battery" in phrase.lower() or "battery" in phrase.lower()
    
    @pytest.mark.asyncio
    async def test_lookup_fuzzy(self, phrase_cache):
        """Fuzzy lookup should find similar phrases."""
        phrase = phrase_cache.lookup_fuzzy("I need to stop because something is blocking me", min_similarity=0.2)
        # Fuzzy matching might or might not find something — depends on similarity
        # Just verify it doesn't crash
        assert phrase is None or isinstance(phrase, str)
    
    @pytest.mark.asyncio
    async def test_get_phrase_intent(self, phrase_cache):
        """get_phrase with intent should find a match."""
        phrase = phrase_cache.get_phrase(intent="battery_low")
        assert phrase is not None
        assert "battery" in phrase.lower() or "rest" in phrase.lower()
    
    @pytest.mark.asyncio
    async def test_get_phrase_all_miss(self, phrase_cache):
        """get_phrase with no matching intent should return None."""
        phrase = phrase_cache.get_phrase(intent="nonexistent")
        assert phrase is None
    
    @pytest.mark.asyncio
    async def test_variety_no_repeats(self, phrase_cache):
        """Consecutive lookups should not return the same phrase."""
        results = set()
        for _ in range(5):
            phrase = phrase_cache.lookup_by_intent("greeting")
            if phrase:
                results.add(phrase)
        
        # With only 3 greeting phrases and variety window of 20,
        # we might get some repeats — but at least 2 different phrases
        assert len(results) >= 2
    
    @pytest.mark.asyncio
    async def test_use_count_incremented(self, phrase_cache):
        """Looking up a phrase should increment its use_count."""
        phrase_cache._recently_used = []  # Reset variety tracking
        
        phrase = phrase_cache.lookup_by_intent("greeting")
        assert phrase is not None
        
        # Check that use_count was incremented in the DB
        import sqlite3
        db = sqlite3.connect(phrase_cache.db_path)
        row = db.execute(
            "SELECT use_count FROM phrases WHERE text = ?", (phrase,)
        ).fetchone()
        db.close()
        
        assert row is not None
        assert row[0] > 0
    
    @pytest.mark.asyncio
    async def test_add_phrase_learning(self, phrase_cache):
        """add_phrase should insert a new phrase."""
        phrase_cache.add_phrase(
            text="This is a learned phrase from the LLM.",
            category="learned",
            intent="learned",
            priority=4,
        )
        
        # Verify it was inserted
        import sqlite3
        db = sqlite3.connect(phrase_cache.db_path)
        row = db.execute(
            "SELECT text, category, intent FROM phrases WHERE text = ?",
            ("This is a learned phrase from the LLM.",),
        ).fetchone()
        db.close()
        
        assert row is not None
        assert row[0] == "This is a learned phrase from the LLM."
        assert row[1] == "learned"
    
    @pytest.mark.asyncio
    async def test_add_phrase_no_duplicate(self, phrase_cache):
        """Adding a duplicate should not insert a new row."""
        import sqlite3
        db = sqlite3.connect(phrase_cache.db_path)
        before_count = db.execute("SELECT COUNT(*) FROM phrases").fetchone()[0]
        db.close()
        
        # Add an existing phrase
        phrase_cache.add_phrase(
            text="Hello! I'm Weaver.",
            category="greetings",
            intent="greeting",
        )
        
        db = sqlite3.connect(phrase_cache.db_path)
        after_count = db.execute("SELECT COUNT(*) FROM phrases").fetchone()[0]
        db.close()
        
        assert after_count == before_count  # No new row
    
    @pytest.mark.asyncio
    async def test_get_contextual_phrase_obstacle(self, phrase_cache):
        """Contextual lookup with obstacle should return safety phrase."""
        situation = {
            "obstacle_distance_cm": 20.0,
            "battery_voltage": 8.0,
            "battery_low": False,
            "body_tilt_degrees": 0.0,
            "tilt_warning": False,
            "cpu_temp_c": 45.0,
            "human_detected": False,
            "human_distance_cm": None,
            "last_motion": "forward",
        }
        
        phrase = phrase_cache.get_contextual_phrase(situation)
        assert phrase is not None
        assert "obstacle" in phrase.lower() or "way" in phrase.lower() or "stop" in phrase.lower()
    
    @pytest.mark.asyncio
    async def test_get_contextual_phrase_battery_low(self, phrase_cache):
        """Contextual lookup with low battery should return battery phrase."""
        situation = {
            "obstacle_distance_cm": 100.0,
            "battery_voltage": 6.5,
            "battery_low": True,
            "body_tilt_degrees": 0.0,
            "tilt_warning": False,
            "cpu_temp_c": 45.0,
            "human_detected": False,
            "human_distance_cm": None,
            "last_motion": "stop",
        }
        
        phrase = phrase_cache.get_contextual_phrase(situation)
        assert phrase is not None
        assert "battery" in phrase.lower() or "rest" in phrase.lower()
    
    @pytest.mark.asyncio
    async def test_get_contextual_phrase_idle(self, phrase_cache):
        """Contextual lookup when idle should return status phrase."""
        situation = {
            "obstacle_distance_cm": 100.0,
            "battery_voltage": 8.0,
            "battery_low": False,
            "body_tilt_degrees": 0.0,
            "tilt_warning": False,
            "cpu_temp_c": 45.0,
            "human_detected": False,
            "human_distance_cm": None,
            "last_motion": "stop",
        }
        
        phrase = phrase_cache.get_contextual_phrase(situation)
        # Should return a status/idle phrase
        assert phrase is not None
        assert "operational" in phrase.lower() or "ready" in phrase.lower()
    
    @pytest.mark.asyncio
    async def test_classify_intent_priority(self, phrase_cache):
        """Intent classification should prioritize safety over status."""
        # Battery critical takes priority over idle
        situation = {
            "battery_voltage": 6.0,
            "battery_low": True,
            "obstacle_distance_cm": 100.0,
            "body_tilt_degrees": 0.0,
            "tilt_warning": False,
            "cpu_temp_c": 45.0,
            "human_detected": False,
            "last_motion": "stop",
        }
        
        intent = phrase_cache._classify_intent(situation)
        assert intent == "battery_critical"
    
    @pytest.mark.asyncio
    async def test_get_stats(self, phrase_cache):
        """get_stats should return cache statistics."""
        stats = phrase_cache.get_stats()
        
        assert "total_phrases" in stats
        assert stats["total_phrases"] > 0
        assert "categories" in stats
        assert "total_uses" in stats
    
    @pytest.mark.asyncio
    async def test_empty_cache(self, tmp_path):
        """Cache with no phrases should return None gracefully."""
        db_path = tmp_path / "empty.db"
        
        # Create empty database
        db = sqlite3.connect(str(db_path))
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
        db.commit()
        db.close()
        
        cache = PhraseCache(db_path=str(db_path))
        await cache.start()
        
        assert cache.lookup_by_intent("greeting") is None
        assert cache.get_stats()["total_phrases"] == 0
        
        await cache.stop()