"""
Phase H — latency-discipline cache.

Thread-safe TTL memoization for engine outputs. Engines that compute on
the same `weights + benchmark + window` combination should consult the
cache to avoid recomputing Monte Carlo paths, factor regressions, and
benchmark statistics on every report render.

Key design rules:
- Bounded LRU (default 256 entries) so RAM doesn't grow unbounded.
- Per-entry TTL (default 900s — controllable via FeatureRegistry).
- Cache key is the SHA256 of a stable JSON dump of all inputs that
  meaningfully change the output. Engine code is responsible for
  passing those inputs to `cache_get` / `cache_set`.
- Cache is process-local; multi-worker gunicorn deployments will hold
  separate caches per worker (acceptable — entries are cheap to rebuild).
- `disable_cache()` decorator-style context for tests.

Public surface:
    from phase_h.cache import cache_key, cache_get, cache_set, memoize, stats
"""

from __future__ import annotations

import functools
import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .registry import FeatureRegistry


_LOCK = threading.RLock()
_STORE: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
_MAX_ENTRIES = 256

_metrics = {
    "hits":      0,
    "misses":    0,
    "evictions": 0,
    "writes":    0,
}


# ──────────────────────────────────────────────────────────────────────
# Stable cache-key helper
# ──────────────────────────────────────────────────────────────────────

def _coerce(o: Any) -> Any:
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, Mapping):
        return {str(k): _coerce(v) for k, v in sorted(o.items())}
    if isinstance(o, (list, tuple)):
        return [_coerce(v) for v in o]
    if isinstance(o, set):
        return sorted(_coerce(v) for v in o)
    if hasattr(o, "tolist"):  # numpy arrays / pandas
        try:
            return _coerce(o.tolist())
        except Exception:  # pragma: no cover
            return repr(o)
    return repr(o)


def cache_key(engine: str, **inputs: Any) -> str:
    """Build a stable, content-addressed cache key for an engine call."""
    payload = {"engine": engine, "inputs": _coerce(inputs)}
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# Storage primitives
# ──────────────────────────────────────────────────────────────────────

def _ttl() -> int:
    try:
        return max(0, int(FeatureRegistry.get("cache_ttl_seconds")))
    except Exception:
        return 900


def _enabled() -> bool:
    try:
        return bool(FeatureRegistry.is_enabled("cache_enabled"))
    except Exception:
        return True


def cache_get(key: str) -> Optional[Any]:
    """Return the cached value or None if absent / expired / disabled."""
    if not _enabled():
        return None
    now = time.monotonic()
    with _LOCK:
        entry = _STORE.get(key)
        if entry is None:
            _metrics["misses"] += 1
            return None
        expires_at, value = entry
        if expires_at < now:
            del _STORE[key]
            _metrics["misses"] += 1
            _metrics["evictions"] += 1
            return None
        # LRU touch
        _STORE.move_to_end(key)
        _metrics["hits"] += 1
        return value


def cache_set(key: str, value: Any, *, ttl_seconds: Optional[int] = None) -> None:
    if not _enabled():
        return
    ttl = ttl_seconds if ttl_seconds is not None else _ttl()
    expires_at = time.monotonic() + ttl
    with _LOCK:
        _STORE[key] = (expires_at, value)
        _STORE.move_to_end(key)
        while len(_STORE) > _MAX_ENTRIES:
            _STORE.popitem(last=False)
            _metrics["evictions"] += 1
        _metrics["writes"] += 1


def clear() -> None:
    with _LOCK:
        _STORE.clear()


def stats() -> Dict[str, int]:
    with _LOCK:
        return {**_metrics, "size": len(_STORE), "max_entries": _MAX_ENTRIES}


# ──────────────────────────────────────────────────────────────────────
# Decorator
# ──────────────────────────────────────────────────────────────────────

def memoize(engine: str, *, ttl_seconds: Optional[int] = None) -> Callable:
    """
    Decorator: caches a function's return value keyed on its kwargs.
    Positional args are NOT part of the key — engines should pass all
    cache-relevant inputs as keyword arguments.

        @memoize("benchmark_relative")
        def compute_benchmark_relative(*, weights, ...): ...
    """
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = cache_key(engine, **kwargs)
            hit = cache_get(key)
            if hit is not None:
                return hit
            out = fn(*args, **kwargs)
            cache_set(key, out, ttl_seconds=ttl_seconds)
            return out
        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        return wrapper
    return deco


# ──────────────────────────────────────────────────────────────────────
# Test context manager
# ──────────────────────────────────────────────────────────────────────

class disable_cache:
    """Context manager: force-disable cache for a block (e.g., tests)."""

    def __enter__(self) -> None:
        FeatureRegistry.override("cache_enabled", False)
        clear()

    def __exit__(self, *_: Any) -> None:
        FeatureRegistry.reset_override("cache_enabled")


__all__ = [
    "cache_key",
    "cache_get",
    "cache_set",
    "memoize",
    "clear",
    "stats",
    "disable_cache",
]
