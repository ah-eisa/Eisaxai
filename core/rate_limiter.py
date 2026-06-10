"""
SQLite sliding-window rate limiter shared across multiple workers.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Dict

from core.config import BASE_DIR

logger = logging.getLogger(__name__)

DB_PATH = str(BASE_DIR / "rate_limits.db")
DEFAULT_LIMIT = 20
EXEMPT_USERS = {"admin", "admin_test"}


def _init_db() -> None:
    try:
        with sqlite3.connect(DB_PATH, timeout=5.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_events (
                    user_id TEXT NOT NULL,
                    ts REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rate_events_user_ts ON rate_events(user_id, ts)"
            )
    except Exception as exc:
        logger.warning("[RateLimiter] init failed: %s", exc)


_init_db()


def _normalize_user_id(user_id: str) -> str:
    uid = (user_id or "").strip()
    return uid or "anonymous"


def is_rate_limited(user_id: str, limit: int = DEFAULT_LIMIT, window_sec: int = 60) -> bool:
    uid = _normalize_user_id(user_id)
    if uid in EXEMPT_USERS:
        return False

    now = time.time()
    cutoff = now - float(window_sec)

    try:
        conn = sqlite3.connect(DB_PATH, timeout=5.0, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("BEGIN IMMEDIATE")

            conn.execute(
                "DELETE FROM rate_events WHERE user_id = ? AND ts < ?",
                (uid, cutoff),
            )
            row = conn.execute(
                "SELECT COUNT(*) FROM rate_events WHERE user_id = ? AND ts >= ?",
                (uid, cutoff),
            ).fetchone()
            count = int(row[0] if row else 0)

            if count >= int(limit):
                conn.execute("COMMIT")
                return True

            conn.execute(
                "INSERT INTO rate_events (user_id, ts) VALUES (?, ?)",
                (uid, now),
            )
            conn.execute("COMMIT")
            return False
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("[RateLimiter] is_rate_limited failed for %s: %s", uid, exc)
        # Fail open to avoid false blocking in production.
        return False


def get_usage(user_id: str, window_sec: int = 60) -> Dict[str, float]:
    uid = _normalize_user_id(user_id)
    if uid in EXEMPT_USERS:
        return {"count": 0, "limit": DEFAULT_LIMIT, "remaining": DEFAULT_LIMIT, "reset_in": 0.0}

    now = time.time()
    cutoff = now - float(window_sec)

    try:
        with sqlite3.connect(DB_PATH, timeout=5.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                "DELETE FROM rate_events WHERE user_id = ? AND ts < ?",
                (uid, cutoff),
            )
            row = conn.execute(
                "SELECT COUNT(*), MIN(ts) FROM rate_events WHERE user_id = ? AND ts >= ?",
                (uid, cutoff),
            ).fetchone()

            count = int(row[0] if row and row[0] is not None else 0)
            min_ts = float(row[1]) if row and row[1] is not None else None
            remaining = max(0, DEFAULT_LIMIT - count)
            reset_in = max(0.0, float(window_sec) - (now - min_ts)) if min_ts else 0.0
            return {
                "count": count,
                "limit": DEFAULT_LIMIT,
                "remaining": remaining,
                "reset_in": reset_in,
            }
    except Exception as exc:
        logger.warning("[RateLimiter] get_usage failed for %s: %s", uid, exc)
        return {"count": 0, "limit": DEFAULT_LIMIT, "remaining": DEFAULT_LIMIT, "reset_in": 0.0}
