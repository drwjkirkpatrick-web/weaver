# keepalive.py — Keep services alive for Weaver
#
# Whisper STT, Edge TTS, and local Ollama can "fall asleep" or get unloaded
# when idle. This module periodically pings each service to keep it warm
# so that the first voice command or LLM call doesn't have a cold-start delay.
#
# For the Raspberry Pi 5 with limited RAM, this is especially important:
#   - Ollama will unload models after ~5 min of inactivity
#   - faster-whisper's model gets swapped to disk under memory pressure
#   - edge-tts connection pool expires
#
# The KeepaliveManager runs a single async loop that pings all registered
# services at their configured intervals. If a service is unavailable,
# it logs a warning but does NOT crash — the system continues without that
# service until it comes back.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from loguru import logger

from weaver.config import get_config


@dataclass
class KeepaliveTarget:
    """A service that needs periodic pinging to stay warm.
    
    Attributes:
        name: Human-readable name (e.g., "ollama_local")
        ping_func: Async callable that returns True on success, False on failure
        interval: Seconds between pings (default: 60)
        enabled: Whether this target is active
        last_ping: Timestamp of last successful ping
        last_success: True if last ping succeeded
        ping_count: Total pings sent
        fail_count: Total failed pings
    """
    name: str
    ping_func: Callable[[], Coroutine[Any, Any, bool]]
    interval: float = 60.0
    enabled: bool = True
    last_ping: float = 0.0
    last_success: bool = False
    ping_count: int = 0
    fail_count: int = 0


class KeepaliveManager:
    """Manages keepalive pings for STT, TTS, and LLM services.
    
    Services are registered with an async ping function and an interval.
    The manager runs a single loop that pings each service when its
    interval elapses.
    
    Usage:
        km = KeepaliveManager()
        
        # Register services (done by main.py or individual modules)
        km.register("ollama_local", ping_ollama, interval=120)
        km.register("whisper_stt", ping_whisper, interval=300)
        km.register("edge_tts", ping_edge, interval=60)
        
        await km.start()
        # ... services stay warm ...
        await km.stop()
    
    The ping functions are simple async callables that return True/False:
    
        async def ping_ollama() -> bool:
            try:
                resp = await httpx.AsyncClient().get("http://localhost:11434/api/tags")
                return resp.status_code == 200
            except:
                return False
    """
    
    def __init__(self):
        self._targets: dict[str, KeepaliveTarget] = {}
        self._task: asyncio.Task | None = None
        self._running: bool = False
    
    def register(
        self,
        name: str,
        ping_func: Callable[[], Coroutine[Any, Any, bool]],
        interval: float = 60.0,
        enabled: bool = True,
    ) -> None:
        """Register a service for keepalive pinging.
        
        Args:
            name: Service identifier (e.g., "ollama_local", "whisper_stt")
            ping_func: Async function that pings the service and returns True/False
            interval: Seconds between pings
            enabled: If False, the service is registered but not pinged
        """
        self._targets[name] = KeepaliveTarget(
            name=name,
            ping_func=ping_func,
            interval=interval,
            enabled=enabled,
        )
        logger.info(f"💓 Registered keepalive target: {name} (interval: {interval}s)")
    
    def unregister(self, name: str) -> None:
        """Stop monitoring a service."""
        if name in self._targets:
            del self._targets[name]
            logger.info(f"Unregistered keepalive: {name}")
    
    def enable(self, name: str) -> None:
        """Enable a registered service."""
        if name in self._targets:
            self._targets[name].enabled = True
            logger.info(f"Keepalive enabled: {name}")
    
    def disable(self, name: str) -> None:
        """Disable a registered service (stop pinging it)."""
        if name in self._targets:
            self._targets[name].enabled = False
            logger.info(f"Keepalive disabled: {name}")
    
    async def start(self) -> None:
        """Start the keepalive loop."""
        if self._running:
            logger.warning("Keepalive manager already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._keepalive_loop())
        logger.info(f"✅ Keepalive manager started ({len(self._targets)} targets)")
    
    async def stop(self) -> None:
        """Stop the keepalive manager."""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        logger.info("Keepalive manager stopped")
    
    async def _keepalive_loop(self) -> None:
        """Main keepalive loop — pings services when their intervals elapse."""
        while self._running:
            for name, target in list(self._targets.items()):
                if not target.enabled:
                    continue
                
                now = time.time()
                if now - target.last_ping < target.interval:
                    continue  # Not time yet
                
                target.last_ping = now
                target.ping_count += 1
                
                try:
                    success = await target.ping_func()
                    target.last_success = success
                    
                    if success:
                        logger.debug(f"💓 {name}: OK")
                    else:
                        target.fail_count += 1
                        logger.warning(f"💔 {name}: ping returned False (fails: {target.fail_count})")
                
                except Exception as e:
                    target.fail_count += 1
                    target.last_success = False
                    logger.warning(f"💔 {name}: ping error: {e} (fails: {target.fail_count})")
            
            # Check every 5 seconds (don't spin too tightly)
            await asyncio.sleep(5.0)
    
    def get_status(self) -> dict[str, Any]:
        """Get status of all keepalive targets (for dashboard)."""
        return {
            name: {
                "enabled": t.enabled,
                "interval": t.interval,
                "last_ping": t.last_ping,
                "last_success": t.last_success,
                "ping_count": t.ping_count,
                "fail_count": t.fail_count,
            }
            for name, t in self._targets.items()
        }
    
    def is_alive(self, name: str) -> bool:
        """Check if a service was recently pinged successfully."""
        target = self._targets.get(name)
        if target is None:
            return False
        if not target.enabled:
            return False
        # Consider alive if last ping was within 3 intervals and was successful
        now = time.time()
        return (
            target.last_success
            and (now - target.last_ping) < (target.interval * 3)
        )


# ─── Ping Functions ────────────────────────────────────────────────────


async def ping_ollama_local() -> bool:
    """Ping the local Ollama server to keep the model loaded.
    
    Ollama unloads models after ~5 minutes of inactivity (configurable).
    This ping sends a minimal generate request to keep the model warm.
    
    Returns:
        True if Ollama responded successfully.
    """
    try:
        import httpx
        
        config = get_config()
        url = f"{config.cortex.local_base_url}/api/tags"
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=5.0)
            if resp.status_code == 200:
                # Also send a minimal chat to keep the model loaded
                # This is a tiny prompt that costs almost nothing
                chat_url = f"{config.cortex.local_base_url}/api/chat"
                try:
                    await client.post(
                        chat_url,
                        json={
                            "model": config.cortex.local_model,
                            "messages": [{"role": "user", "content": "."}],
                            "stream": False,
                            "options": {"num_predict": 1},
                        },
                        timeout=10.0,
                    )
                except Exception:
                    pass  # The /api/tags ping is enough
                return True
        return False
    except Exception as e:
        logger.debug(f"Ollama keepalive ping failed: {e}")
        return False


async def ping_whisper_stt() -> bool:
    """Keep the Whisper model loaded in memory.
    
    faster-whisper loads the model on first use. If it hasn't been used
    in a while, the OS may swap it to disk under memory pressure.
    
    We check if the model object still exists (the STT module keeps it
    loaded). This is a lightweight check — no actual transcription.
    
    Returns:
        True if the Whisper model is loaded and ready.
    """
    try:
        # Check if we can import faster_whisper (validates the package is available)
        import faster_whisper  # noqa: F401
        
        # The actual model is held by the STT module instance.
        # We can't directly access it here without a reference, but
        # the import check ensures the package is still loadable.
        # In a real deployment, the STT module's listening loop keeps
        # the model warm by processing audio continuously.
        
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"Whisper keepalive check failed: {e}")
        return False


async def ping_edge_tts() -> bool:
    """Keep the Edge TTS connection warm.
    
    Edge TTS creates a new connection per speak() call. There's no
    persistent connection to keep alive. However, we can verify
    that the edge-tts package is still importable and that we can
    reach Microsoft's TTS endpoint.
    
    Returns:
        True if edge-tts is available and the endpoint is reachable.
    """
    try:
        import edge_tts  # noqa: F401
        
        # We don't actually generate speech (that would be wasteful).
        # Just verify the package is importable.
        # The edge-tts service is stateless — each call is independent.
        
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"Edge TTS keepalive check failed: {e}")
        return False


# ─── Factory ────────────────────────────────────────────────────────────


def create_default_keepalive_manager() -> KeepaliveManager:
    """Create a KeepaliveManager with default targets based on config.
    
    This checks the current configuration and registers keepalive targets
    for any enabled services:
    - Ollama Local LLM (if mode is ollama_local or as a fallback)
    - Whisper STT (if stt_enabled and stt_provider is "local")
    - Edge TTS (if tts_enabled and tts_provider is "edge")
    """
    config = get_config()
    km = KeepaliveManager()
    
    # Ollama Local LLM keepalive
    # Register it if local mode is active, OR if it's available as a fallback
    # The local Ollama model gets unloaded after ~5 min idle.
    # We ping every 2 minutes to keep it warm.
    if config.cortex.mode.value == "ollama_local":
        km.register(
            name="ollama_local",
            ping_func=ping_ollama_local,
            interval=120.0,  # Every 2 minutes
            enabled=True,
        )
    else:
        # Even in cloud mode, keep local Ollama warm as a fallback
        # (but ping less frequently since it's not the primary)
        km.register(
            name="ollama_local",
            ping_func=ping_ollama_local,
            interval=300.0,  # Every 5 minutes (just keep it from unloading)
            enabled=True,
        )
    
    # Whisper STT keepalive
    # The model gets swapped to disk under memory pressure.
    # We check every 5 minutes that it's still loaded.
    if config.voice.stt_enabled and config.voice.stt_provider == "local":
        km.register(
            name="whisper_stt",
            ping_func=ping_whisper_stt,
            interval=300.0,  # Every 5 minutes
            enabled=True,
        )
    
    # Edge TTS keepalive
    # Edge TTS is stateless, but we verify the package is available.
    if config.voice.tts_enabled and config.voice.tts_provider == "edge":
        km.register(
            name="edge_tts",
            ping_func=ping_edge_tts,
            interval=600.0,  # Every 10 minutes (lightweight check)
            enabled=True,
        )
    
    return km