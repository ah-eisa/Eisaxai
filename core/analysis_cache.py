"""
Multi-level analysis cache.

Levels
------
L1: in-memory, session-scoped, 5 minutes
L2: in-memory, user-scoped, 30 minutes
L3: SQLite-backed, global, 1 hour
"""

from __future__ import annotations

import copy
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.config import BASE_DIR

logger = logging.getLogger(__name__)

DB = str(BASE_DIR / "analysis_cache.db")

L1_TTL_SECONDS = 5 * 60
L2_TTL_SECONDS = 30 * 60
L3_TTL_SECONDS = 60 * 60
TIMESTAMP_BUCKET_SECONDS = 5 * 60

_L3_TABLE = "analysis_cache_entries"
_lock = threading.RLock()


@dataclass(frozen=True)
class CacheKey:
    ticker: str
    analysis_type: str
    timestamp_bucket: str
    user_id: str | None = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", (self.ticker or "").upper())
        object.__setattr__(self, "analysis_type", (self.analysis_type or "full").lower())
        object.__setattr__(self, "timestamp_bucket", self.timestamp_bucket or _bucket_for())

    @property
    def global_key(self) -> str:
        return f"{self.ticker}:{self.analysis_type}:{self.timestamp_bucket}"

    @property
    def user_key(self) -> str | None:
        if not self.user_id:
            return None
        return f"{self.user_id}:{self.global_key}"

    @property
    def session_key(self) -> str | None:
        if not self.user_id or not self.session_id:
            return None
        return f"{self.user_id}:{self.session_id}:{self.global_key}"


@dataclass
class _CacheEntry:
    value: Any
    created_at: float
    expires_at: float
    ticker: str


def _empty_level_stats() -> dict[str, float | int]:
    return {"requests": 0, "hits": 0, "misses": 0, "sets": 0}


_stats: dict[str, dict[str, float | int]] = {
    "l1": _empty_level_stats(),
    "l2": _empty_level_stats(),
    "l3": _empty_level_stats(),
}

_l1_store: dict[str, _CacheEntry] = {}
_l2_store: dict[str, _CacheEntry] = {}


def _now() -> float:
    return time.time()


def _bucket_for(ts: float | None = None) -> str:
    ts = _now() if ts is None else ts
    bucket_ts = int(ts // TIMESTAMP_BUCKET_SECONDS) * TIMESTAMP_BUCKET_SECONDS
    return datetime.fromtimestamp(bucket_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def configure(db_path: str | None = None) -> None:
    global DB
    if db_path:
        DB = db_path
    _init()


def _sqlite_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _init() -> None:
    with _sqlite_connect() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_L3_TABLE} (
                global_key       TEXT PRIMARY KEY,
                ticker           TEXT NOT NULL,
                analysis_type    TEXT NOT NULL,
                timestamp_bucket TEXT NOT NULL,
                value_json       TEXT NOT NULL,
                created_at       REAL NOT NULL
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{_L3_TABLE}_ticker ON {_L3_TABLE}(ticker)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{_L3_TABLE}_created_at ON {_L3_TABLE}(created_at)"
        )


def make_key(
    ticker: str,
    analysis_type: str = "full",
    *,
    mode: str | None = None,
    timestamp: float | None = None,
    timestamp_bucket: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> CacheKey:
    return CacheKey(
        ticker=ticker,
        analysis_type=mode or analysis_type,
        timestamp_bucket=timestamp_bucket or _bucket_for(timestamp),
        user_id=user_id,
        session_id=session_id,
    )


def _coerce_key(
    key: CacheKey | dict[str, Any] | str,
    *,
    analysis_type: str = "full",
    mode: str | None = None,
    timestamp: float | None = None,
    timestamp_bucket: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> CacheKey:
    if isinstance(key, CacheKey):
        return key
    if isinstance(key, dict):
        return make_key(
            key.get("ticker", ""),
            analysis_type=key.get("analysis_type", analysis_type),
            mode=key.get("mode", mode),
            timestamp=key.get("timestamp", timestamp),
            timestamp_bucket=key.get("timestamp_bucket", timestamp_bucket),
            user_id=key.get("user_id", user_id),
            session_id=key.get("session_id", session_id),
        )
    return make_key(
        key,
        analysis_type=analysis_type,
        mode=mode,
        timestamp=timestamp,
        timestamp_bucket=timestamp_bucket,
        user_id=user_id,
        session_id=session_id,
    )


def _record_hit(level: str, key: CacheKey, age_seconds: int) -> None:
    with _lock:
        _stats[level]["requests"] += 1
        _stats[level]["hits"] += 1
    logger.info(
        "[AnalysisCache] HIT level=%s ticker=%s type=%s bucket=%s age=%ss",
        level,
        key.ticker,
        key.analysis_type,
        key.timestamp_bucket,
        age_seconds,
    )


def _record_miss(level: str, key: CacheKey, reason: str) -> None:
    with _lock:
        _stats[level]["requests"] += 1
        _stats[level]["misses"] += 1
    logger.info(
        "[AnalysisCache] MISS level=%s ticker=%s type=%s bucket=%s reason=%s",
        level,
        key.ticker,
        key.analysis_type,
        key.timestamp_bucket,
        reason,
    )


def _record_set(level: str, key: CacheKey) -> None:
    with _lock:
        _stats[level]["sets"] += 1
    logger.info(
        "[AnalysisCache] SET level=%s ticker=%s type=%s bucket=%s",
        level,
        key.ticker,
        key.analysis_type,
        key.timestamp_bucket,
    )


def _copy_value(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _decorate_value(level: str, entry: _CacheEntry) -> dict[str, Any]:
    age_seconds = max(0, int(_now() - entry.created_at))
    payload = _copy_value(entry.value)
    if isinstance(payload, dict):
        payload["from_cache"] = True
        payload["cache_age"] = age_seconds
        payload["cache_level"] = level
        return payload
    return {
        "value": payload,
        "from_cache": True,
        "cache_age": age_seconds,
        "cache_level": level,
    }


def _memory_store(
    store: dict[str, _CacheEntry],
    scoped_key: str | None,
    ttl_seconds: int,
    key: CacheKey,
    value: Any,
    *,
    created_at: float | None = None,
    record_set: bool = False,
    level: str,
) -> None:
    if not scoped_key:
        return
    created_at = _now() if created_at is None else created_at
    expires_at = created_at + ttl_seconds
    if expires_at <= _now():
        return
    with _lock:
        store[scoped_key] = _CacheEntry(
            value=_copy_value(value),
            created_at=created_at,
            expires_at=expires_at,
            ticker=key.ticker,
        )
    if record_set:
        _record_set(level, key)


def _memory_get(
    store: dict[str, _CacheEntry],
    scoped_key: str | None,
    key: CacheKey,
    *,
    level: str,
) -> _CacheEntry | None:
    if not scoped_key:
        _record_miss(level, key, "scope_unavailable")
        return None
    with _lock:
        entry = store.get(scoped_key)
        if entry is None:
            _record_miss(level, key, "not_found")
            return None
        if entry.expires_at <= _now():
            store.pop(scoped_key, None)
            _record_miss(level, key, "expired")
            return None
        return _CacheEntry(
            value=_copy_value(entry.value),
            created_at=entry.created_at,
            expires_at=entry.expires_at,
            ticker=entry.ticker,
        )


def _sqlite_get(key: CacheKey) -> _CacheEntry | None:
    try:
        with _sqlite_connect() as conn:
            row = conn.execute(
                f"""
                SELECT value_json, created_at
                FROM {_L3_TABLE}
                WHERE global_key = ?
                """,
                (key.global_key,),
            ).fetchone()
            if row is None:
                _record_miss("l3", key, "not_found")
                return None
            created_at = float(row[1])
            if created_at + L3_TTL_SECONDS <= _now():
                conn.execute(
                    f"DELETE FROM {_L3_TABLE} WHERE global_key = ?",
                    (key.global_key,),
                )
                _record_miss("l3", key, "expired")
                return None
            _record_hit("l3", key, max(0, int(_now() - created_at)))
            return _CacheEntry(
                value=json.loads(row[0]),
                created_at=created_at,
                expires_at=created_at + L3_TTL_SECONDS,
                ticker=key.ticker,
            )
    except Exception as exc:
        logger.warning("[AnalysisCache] l3 get failed: %s", exc)
        _record_miss("l3", key, "error")
        return None


def _sqlite_set(key: CacheKey, value: Any, *, created_at: float | None = None, record_set: bool = False) -> None:
    created_at = _now() if created_at is None else created_at
    try:
        payload = json.dumps(_copy_value(value), ensure_ascii=False)
        with _sqlite_connect() as conn:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {_L3_TABLE}
                    (global_key, ticker, analysis_type, timestamp_bucket, value_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key.global_key,
                    key.ticker,
                    key.analysis_type,
                    key.timestamp_bucket,
                    payload,
                    created_at,
                ),
            )
        if record_set:
            _record_set("l3", key)
    except Exception as exc:
        logger.warning("[AnalysisCache] l3 set failed: %s", exc)


def _promote_from(level: str, key: CacheKey, entry: _CacheEntry) -> None:
    if level == "l2":
        _memory_store(
            _l1_store,
            key.session_key,
            L1_TTL_SECONDS,
            key,
            entry.value,
            created_at=entry.created_at,
            level="l1",
        )
        return
    if level == "l3":
        _memory_store(
            _l2_store,
            key.user_key,
            L2_TTL_SECONDS,
            key,
            entry.value,
            created_at=entry.created_at,
            level="l2",
        )
        _memory_store(
            _l1_store,
            key.session_key,
            L1_TTL_SECONDS,
            key,
            entry.value,
            created_at=entry.created_at,
            level="l1",
        )


def get(
    key: CacheKey | dict[str, Any] | str,
    level: str = "all",
    *,
    analysis_type: str = "full",
    mode: str | None = None,
    timestamp: float | None = None,
    timestamp_bucket: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    cache_key = _coerce_key(
        key,
        analysis_type=analysis_type,
        mode=mode,
        timestamp=timestamp,
        timestamp_bucket=timestamp_bucket,
        user_id=user_id,
        session_id=session_id,
    )
    level = (level or "all").lower()

    if level == "l1":
        entry = _memory_get(_l1_store, cache_key.session_key, cache_key, level="l1")
        if entry is None:
            return None
        _record_hit("l1", cache_key, max(0, int(_now() - entry.created_at)))
        return _decorate_value("l1", entry)

    if level == "l2":
        entry = _memory_get(_l2_store, cache_key.user_key, cache_key, level="l2")
        if entry is None:
            return None
        _record_hit("l2", cache_key, max(0, int(_now() - entry.created_at)))
        return _decorate_value("l2", entry)

    if level == "l3":
        entry = _sqlite_get(cache_key)
        if entry is None:
            return None
        return _decorate_value("l3", entry)

    for source_level, fetcher in (
        ("l1", lambda: _memory_get(_l1_store, cache_key.session_key, cache_key, level="l1")),
        ("l2", lambda: _memory_get(_l2_store, cache_key.user_key, cache_key, level="l2")),
        ("l3", lambda: _sqlite_get(cache_key)),
    ):
        entry = fetcher()
        if entry is None:
            continue
        if source_level in {"l1", "l2"}:
            _record_hit(source_level, cache_key, max(0, int(_now() - entry.created_at)))
        _promote_from(source_level, cache_key, entry)
        return _decorate_value(source_level, entry)
    return None


def set(
    key: CacheKey | dict[str, Any] | str,
    value: Any,
    level: str = "l3",
    *,
    analysis_type: str = "full",
    mode: str | None = None,
    timestamp: float | None = None,
    timestamp_bucket: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    cache_key = _coerce_key(
        key,
        analysis_type=analysis_type,
        mode=mode,
        timestamp=timestamp,
        timestamp_bucket=timestamp_bucket,
        user_id=user_id,
        session_id=session_id,
    )
    level = (level or "l3").lower()
    created_at = _now()

    if level in {"all", "l1"}:
        _memory_store(
            _l1_store,
            cache_key.session_key,
            L1_TTL_SECONDS,
            cache_key,
            value,
            created_at=created_at,
            record_set=True,
            level="l1",
        )
    if level in {"all", "l2"}:
        _memory_store(
            _l2_store,
            cache_key.user_key,
            L2_TTL_SECONDS,
            cache_key,
            value,
            created_at=created_at,
            record_set=True,
            level="l2",
        )
    if level in {"all", "l3"}:
        _sqlite_set(cache_key, value, created_at=created_at, record_set=True)


def invalidate(ticker: str | None = None) -> None:
    ticker = ticker.upper() if ticker else None
    with _lock:
        if ticker:
            _l1_keys = [key for key, entry in _l1_store.items() if entry.ticker == ticker]
            _l2_keys = [key for key, entry in _l2_store.items() if entry.ticker == ticker]
            for key in _l1_keys:
                _l1_store.pop(key, None)
            for key in _l2_keys:
                _l2_store.pop(key, None)
        else:
            _l1_store.clear()
            _l2_store.clear()

    try:
        with _sqlite_connect() as conn:
            if ticker:
                conn.execute(f"DELETE FROM {_L3_TABLE} WHERE ticker = ?", (ticker,))
                logger.info("[AnalysisCache] INVALIDATE ticker=%s", ticker)
            else:
                conn.execute(f"DELETE FROM {_L3_TABLE}")
                logger.info("[AnalysisCache] INVALIDATE all")
    except Exception as exc:
        logger.warning("[AnalysisCache] invalidate failed: %s", exc)


def cleanup() -> None:
    now = _now()
    with _lock:
        for store in (_l1_store, _l2_store):
            stale_keys = [key for key, entry in store.items() if entry.expires_at <= now]
            for key in stale_keys:
                store.pop(key, None)
    try:
        with _sqlite_connect() as conn:
            conn.execute(
                f"DELETE FROM {_L3_TABLE} WHERE created_at < ?",
                (now - L3_TTL_SECONDS,),
            )
    except Exception as exc:
        logger.warning("[AnalysisCache] cleanup failed: %s", exc)


def stats() -> dict[str, dict[str, float | int]]:
    snapshot: dict[str, dict[str, float | int]] = {}
    with _lock:
        for level, level_stats in _stats.items():
            requests = int(level_stats["requests"])
            hits = int(level_stats["hits"])
            misses = int(level_stats["misses"])
            snapshot[level] = {
                "requests": requests,
                "hits": hits,
                "misses": misses,
                "sets": int(level_stats["sets"]),
                "hit_rate": round(hits / requests, 4) if requests else 0.0,
                "miss_rate": round(misses / requests, 4) if requests else 0.0,
            }
    return snapshot


def reset(*, clear_persistent: bool = True, clear_stats: bool = True) -> None:
    with _lock:
        _l1_store.clear()
        _l2_store.clear()
        if clear_stats:
            for level in _stats:
                _stats[level] = _empty_level_stats()
    if clear_persistent:
        try:
            with _sqlite_connect() as conn:
                conn.execute(f"DELETE FROM {_L3_TABLE}")
        except Exception:
            _init()


_init()
