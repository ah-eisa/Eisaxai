"""
EisaX Redis Store — F-5
Distributed rate limiting + session caching for multi-worker setups.
Falls back to in-memory if Redis is unavailable (graceful degradation).
"""
import os
import time
import logging
import functools
from typing import Optional

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── Redis client (lazy singleton) ─────────────────────────────────────────────
_redis_client = None
_redis_ok = None  # None = untested, True/False = last test result


def get_redis():
    global _redis_client, _redis_ok
    if _redis_client is not None and _redis_ok:
        return _redis_client
    try:
        import redis
        client = redis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
        client.ping()
        _redis_client = client
        _redis_ok = True
        logger.info("[redis] Connected to %s", REDIS_URL)
        return _redis_client
    except Exception as exc:
        _redis_ok = False
        logger.warning("[redis] Unavailable (%s) — using in-memory fallback", exc)
        return None


def redis_available() -> bool:
    """Quick health check."""
    r = get_redis()
    return r is not None


# ── In-memory fallback stores ──────────────────────────────────────────────────
_mem_rate: dict = {}     # key -> (count, window_start)
_mem_cache: dict = {}    # key -> (value, expires_at)


# ── Rate Limiting ─────────────────────────────────────────────────────────────

def check_rate_limit(key: str, limit: int, window_seconds: int = 60) -> tuple[bool, int]:
    """
    Sliding-window rate limiter.
    Returns (allowed: bool, remaining: int).
    Uses Redis if available, otherwise in-memory.
    """
    r = get_redis()
    now = int(time.time())

    if r:
        try:
            pipe_key = f"rl:{key}"
            pipe = r.pipeline()
            pipe.incr(pipe_key)
            pipe.expire(pipe_key, window_seconds)
            count, _ = pipe.execute()
            remaining = max(0, limit - count)
            return count <= limit, remaining
        except Exception as exc:
            logger.debug("[redis] rate_limit error: %s", exc)
            # fall through to in-memory

    # In-memory fallback
    window_start, count = _mem_rate.get(key, (now, 0))
    if now - window_start >= window_seconds:
        window_start, count = now, 0
    count += 1
    _mem_rate[key] = (window_start, count)
    remaining = max(0, limit - count)
    return count <= limit, remaining


# ── Distributed Cache ─────────────────────────────────────────────────────────

def cache_set(key: str, value: str, ttl_seconds: int = 3600):
    """Store a string value in Redis/memory with TTL."""
    r = get_redis()
    if r:
        try:
            r.setex(f"cache:{key}", ttl_seconds, value)
            return
        except Exception as exc:
            logger.debug("[redis] cache_set error: %s", exc)
    # fallback
    _mem_cache[key] = (value, time.time() + ttl_seconds)


def cache_get(key: str) -> Optional[str]:
    """Retrieve a cached value or None if expired/missing."""
    r = get_redis()
    if r:
        try:
            val = r.get(f"cache:{key}")
            return val.decode() if val else None
        except Exception as exc:
            logger.debug("[redis] cache_get error: %s", exc)
    # fallback
    entry = _mem_cache.get(key)
    if entry:
        value, expires_at = entry
        if time.time() < expires_at:
            return value
        del _mem_cache[key]
    return None


def cache_delete(key: str):
    r = get_redis()
    if r:
        try:
            r.delete(f"cache:{key}")
            return
        except Exception:
            pass
    _mem_cache.pop(key, None)


# ── Session Presence ──────────────────────────────────────────────────────────

def mark_session_active(session_id: str, user_id: str, ttl: int = 300):
    """Mark a session as active (for presence tracking)."""
    r = get_redis()
    if r:
        try:
            r.setex(f"sess:{session_id}", ttl, user_id)
            r.sadd("active_sessions", session_id)
            r.expire("active_sessions", ttl + 60)
        except Exception as exc:
            logger.debug("[redis] mark_session_active: %s", exc)


def count_active_sessions() -> int:
    """Count currently active sessions across all workers."""
    r = get_redis()
    if r:
        try:
            return r.scard("active_sessions")
        except Exception:
            pass
    return 0


# ── slowapi Redis storage (optional integration) ──────────────────────────────

def get_slowapi_storage():
    """
    Return a slowapi-compatible storage backend.
    Uses Redis if available, otherwise returns None (slowapi uses in-memory by default).
    """
    r = get_redis()
    if r:
        try:
            from slowapi.util import get_remote_address
            # slowapi supports redis via limits library
            return f"redis://{REDIS_URL.split('://')[-1]}"
        except Exception:
            pass
    return None


# ── Health info ───────────────────────────────────────────────────────────────

def redis_info() -> dict:
    r = get_redis()
    if not r:
        return {"available": False, "url": REDIS_URL}
    try:
        info = r.info("server")
        return {
            "available": True,
            "version": info.get("redis_version", "unknown"),
            "uptime_seconds": info.get("uptime_in_seconds", 0),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "url": REDIS_URL,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}
