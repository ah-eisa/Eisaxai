"""
core/polish_cache.py
────────────────────
SQLite-backed cache for LLM-polished editorial outputs.

Keys
────
  rule:{report_id}  → rule-based cleaned text (stored at analyze time)
  llm:{report_id}   → full LLM-polished text  (stored after /v1/polish-report)

TTL: 24 hours.  report_id = MD5[:16] of rule-based report text.
"""
import logging
import sqlite3
import time

from core.config import BASE_DIR

log = logging.getLogger("polish_cache")

_DB   = str(BASE_DIR / "polish_cache.db")
_TTL  = 24 * 3600   # 24 h
_TABLE = "polish_cache"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("PRAGMA busy_timeout=5000")
    return c


def _init() -> None:
    with _conn() as c:
        c.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                cache_key  TEXT PRIMARY KEY,
                text       TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        c.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{_TABLE}_created ON {_TABLE}(created_at)"
        )


# ── Public API ─────────────────────────────────────────────────────────────────

def get(cache_key: str) -> str | None:
    """Return cached text for *cache_key*, or None if missing / expired."""
    try:
        with _conn() as c:
            row = c.execute(
                f"SELECT text, created_at FROM {_TABLE} WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        text, created_at = row[0], float(row[1])
        if time.time() - created_at > _TTL:
            _delete(cache_key)
            return None
        return text
    except Exception as exc:
        log.warning("[polish_cache] get(%s) failed: %s", cache_key, exc)
        return None


def set(cache_key: str, text: str) -> None:
    """Store *text* under *cache_key* with current timestamp."""
    try:
        with _conn() as c:
            c.execute(
                f"INSERT OR REPLACE INTO {_TABLE} (cache_key, text, created_at) VALUES (?, ?, ?)",
                (cache_key, text, time.time()),
            )
    except Exception as exc:
        log.warning("[polish_cache] set(%s) failed: %s", cache_key, exc)


def _delete(cache_key: str) -> None:
    try:
        with _conn() as c:
            c.execute(f"DELETE FROM {_TABLE} WHERE cache_key = ?", (cache_key,))
    except Exception:
        pass


def cleanup() -> None:
    """Purge entries older than TTL — call from a maintenance cron."""
    try:
        with _conn() as c:
            c.execute(
                f"DELETE FROM {_TABLE} WHERE created_at < ?",
                (time.time() - _TTL,),
            )
    except Exception as exc:
        log.warning("[polish_cache] cleanup failed: %s", exc)


_init()
