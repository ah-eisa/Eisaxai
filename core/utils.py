"""
core/utils.py
─────────────
Shared utilities for the EisaX intelligence platform.

Contents
────────
• TTLCache   — thread-safe in-memory cache with per-instance TTL (not global,
               avoids race conditions under concurrent requests)
• retry      — exponential-backoff retry wrapper for any callable
• yf_retry   — yfinance-specific wrapper (returns Ticker + info dict)

Usage
─────
    from core.utils import TTLCache, retry, yf_retry

    _cache = TTLCache(ttl_seconds=600)      # per-instance, not module-level global
    result = retry(lambda: fetch_data(), max_attempts=3)
    ticker_obj, info = yf_retry("AAPL")
"""

import logging
import threading
import time
from typing import Any, Callable, Tuple, Type

logger = logging.getLogger(__name__)


# ── TTLCache ──────────────────────────────────────────────────────────────────

class TTLCache:
    """
    Thread-safe in-memory cache with TTL expiry.

    Design notes
    ────────────
    • Per-instance (not module-level global) so different callers can have
      independent TTLs and there are no cross-request pollution risks.
    • Uses a single threading.Lock — safe for multi-threaded ASGI servers.
    • Expired entries are evicted lazily on next .get(); no background thread.
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._store: dict[str, tuple[Any, float]] = {}   # key → (value, expire_ts)
        self._ttl   = ttl_seconds
        self._lock  = threading.Lock()

    # ── public API ───────────────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """Return cached value if present and not expired, else None."""
        with self._lock:
            entry = self._store.get(key)
            if entry is not None and time.time() < entry[1]:
                return entry[0]
            # Evict stale entry
            self._store.pop(key, None)
            return None

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key* with the cache's TTL."""
        with self._lock:
            self._store[key] = (value, time.time() + self._ttl)

    def invalidate(self, key: str) -> None:
        """Explicitly remove a single key."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Flush all entries."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ── retry ─────────────────────────────────────────────────────────────────────

def retry(
    fn: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Any:
    """
    Call *fn()* with exponential back-off on transient failures.

    Parameters
    ──────────
    fn            : zero-argument callable to invoke
    max_attempts  : total number of tries (default 3)
    base_delay    : initial wait in seconds; doubles each retry (default 1.0)
    exceptions    : tuple of exception types that trigger a retry

    Raises
    ──────
    The last exception if every attempt fails.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except exceptions as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = base_delay * (2 ** attempt)
                logger.warning(
                    "[retry] attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1, max_attempts, exc, wait,
                )
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ── yf_retry ─────────────────────────────────────────────────────────────────

def yf_retry(
    ticker: str,
    max_attempts: int = 3,
    base_delay: float = 1.5,
) -> Tuple[Any, dict]:
    """
    Create a yfinance Ticker and fetch `.info` with exponential back-off.

    Returns
    ───────
    (ticker_obj, info_dict)   — same as ``yf.Ticker(ticker), ticker_obj.info``

    Raises
    ──────
    The last exception if all attempts fail.

    Notes
    ─────
    • Uses fast_info where sufficient (no network round-trip), but some callers
      need the full .info dict (analyst data, earnings, etc.), so we keep .info
      as default and let callers switch to .fast_info themselves for hot paths.
    """
    import yfinance as yf

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            t    = yf.Ticker(ticker)
            info = t.info          # triggers network call
            return t, info
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = base_delay * (2 ** attempt)
                logger.warning(
                    "[yf_retry] %s attempt %d/%d failed: %s — retrying in %.1fs",
                    ticker, attempt + 1, max_attempts, exc, wait,
                )
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]
