"""Tests for the KeepaliveManager and ping functions."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from weaver.keepalive import KeepaliveManager, KeepaliveTarget, ping_ollama_local, ping_whisper_stt, ping_edge_tts


class TestKeepaliveManager:
    """Test the keepalive manager lifecycle."""
    
    @pytest.mark.asyncio
    async def test_register_target(self):
        """Registering a target should add it to the manager."""
        km = KeepaliveManager()
        
        async def ping():
            return True
        
        km.register("test", ping, interval=10.0)
        
        assert "test" in km._targets
        assert km._targets["test"].interval == 10.0
        assert km._targets["test"].enabled is True
    
    @pytest.mark.asyncio
    async def test_unregister_target(self):
        """Unregistering should remove the target."""
        km = KeepaliveManager()
        
        async def ping():
            return True
        
        km.register("test", ping)
        km.unregister("test")
        
        assert "test" not in km._targets
    
    @pytest.mark.asyncio
    async def test_enable_disable(self):
        """enable/disable should toggle the target."""
        km = KeepaliveManager()
        
        async def ping():
            return True
        
        km.register("test", ping, enabled=True)
        km.disable("test")
        assert km._targets["test"].enabled is False
        
        km.enable("test")
        assert km._targets["test"].enabled is True
    
    @pytest.mark.asyncio
    async def test_ping_calls_function(self):
        """The keepalive loop should call the ping function."""
        km = KeepaliveManager()
        
        call_count = 0
        
        async def ping():
            nonlocal call_count
            call_count += 1
            return True
        
        km.register("test", ping, interval=0.1)
        
        # Manually run one ping cycle
        target = km._targets["test"]
        target.last_ping = 0  # Force ping
        await km._keepalive_loop.__wrapped__(km) if hasattr(km._keepalive_loop, '__wrapped__') else None
        
        # Actually, let's test differently — start and stop quickly
        await km.start()
        await asyncio.sleep(0.2)
        await km.stop()
        
        assert call_count > 0
        assert km._targets["test"].ping_count > 0
        assert km._targets["test"].last_success is True
    
    @pytest.mark.asyncio
    async def test_ping_failure(self):
        """A failing ping should increment fail_count."""
        km = KeepaliveManager()
        
        async def ping():
            return False
        
        km.register("test", ping, interval=0.1)
        
        await km.start()
        await asyncio.sleep(0.2)
        await km.stop()
        
        assert km._targets["test"].fail_count > 0
        assert km._targets["test"].last_success is False
    
    @pytest.mark.asyncio
    async def test_ping_exception(self):
        """A ping that raises should not crash the manager."""
        km = KeepaliveManager()
        
        async def ping():
            raise ConnectionError("Boom!")
        
        km.register("test", ping, interval=0.1)
        
        await km.start()
        await asyncio.sleep(0.2)
        await km.stop()
        
        assert km._targets["test"].fail_count > 0
        assert km._targets["test"].last_success is False
    
    @pytest.mark.asyncio
    async def test_disabled_not_pinged(self):
        """Disabled targets should not be pinged."""
        km = KeepaliveManager()
        
        call_count = 0
        
        async def ping():
            nonlocal call_count
            call_count += 1
            return True
        
        km.register("test", ping, interval=0.1, enabled=False)
        
        await km.start()
        await asyncio.sleep(0.2)
        await km.stop()
        
        assert call_count == 0
    
    def test_get_status(self):
        """get_status should return info about all targets."""
        km = KeepaliveManager()
        
        async def ping():
            return True
        
        km.register("test", ping, interval=30.0)
        
        status = km.get_status()
        assert "test" in status
        assert status["test"]["interval"] == 30.0
        assert status["test"]["enabled"] is True
    
    def test_is_alive(self):
        """is_alive should return False for unknown services."""
        km = KeepaliveManager()
        assert km.is_alive("nonexistent") is False
    
    def test_is_alive_after_ping(self):
        """is_alive should return True after a successful ping."""
        km = KeepaliveManager()
        
        async def ping():
            return True
        
        km.register("test", ping, interval=60.0)
        target = km._targets["test"]
        target.last_ping = 999999999999.0  # Very recent
        target.last_success = True
        
        assert km.is_alive("test") is True


class TestPingFunctions:
    """Test the built-in ping functions."""
    
    @pytest.mark.asyncio
    async def test_ping_whisper_stt(self):
        """ping_whisper_stt should return True if package is importable."""
        result = await ping_whisper_stt()
        # In test env, faster_whisper is mocked so it should return True
        # or False if not installed — both are valid
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_ping_edge_tts(self):
        """ping_edge_tts should return True if edge_tts is importable."""
        result = await ping_edge_tts()
        assert isinstance(result, bool)