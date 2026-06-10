"""
core/services/market_updates.py
────────────────────────────────
EisaX Market Updates — institutional-grade daily pulse + weekly strategy brief.

Public API (unchanged signatures):
    generate_daily_update()  -> dict
    generate_weekly_update() -> dict
    get_latest_updates()     -> dict
    format_for_linkedin(update_json: dict) -> str

New helpers (internal):
    build_eisax_stance(moves, regime, fg)   -> dict
    build_invalidation_logic(moves, regime) -> list
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path("/home/ubuntu/investwise/data/market_updates.db")
_MARKET_SNAPSHOT_PATH = Path("/home/ubuntu/investwise/data/market_updates_snapshot.json")
_MARKET_CACHE_TTL = timedelta(minutes=15)
_MARKET_CACHE: dict[str, Any] = {
    "lookback_days": None,
    "fetched_at": None,
    "data_timestamp": None,
    "data": None,
}
_LAST_MARKET_DATA_TIMESTAMP: Optional[str] = None

# ── Benchmarks ────────────────────────────────────────────────────────────────

_BENCHMARKS = {
    "SPY":      "S&P 500",
    "QQQ":      "Nasdaq 100",
    "^VIX":     "VIX",
    "GLD":      "Gold",
    "SLV":      "Silver",
    "USO":      "Oil (WTI)",
    "BTC-USD":  "Bitcoin",
    "^TNX":     "10Y Treasury Yield",
    "UUP":      "US Dollar (DXY)",
    "^TASI":    "Saudi Market Composite",
    "^DFMGI":   "UAE Market Composite",
    "EGX30.CA": "Egypt Market Composite",
}

_PIPELINE_REGIONAL_BENCHMARKS = {
    "^TASI": {"market": "ksa", "label": "Saudi Market Composite"},
    "^DFMGI": {"market": "uae", "label": "UAE Market Composite"},
    "EGX30.CA": {"market": "egypt", "label": "Egypt Market Composite"},
}

_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")


# ── Storage ───────────────────────────────────────────────────────────────────

# DB and cache layer

def _init_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS market_updates (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                type       TEXT    NOT NULL,
                json_data  TEXT    NOT NULL,
                created_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_mu_type_time ON market_updates(type, created_at)")

def _save_update(update_type: str, data: dict) -> int:
    _init_db()
    with sqlite3.connect(_DB_PATH) as con:
        cur = con.execute(
            "INSERT INTO market_updates (type, json_data, created_at) VALUES (?, ?, ?)",
            (update_type, json.dumps(data, ensure_ascii=False),
             datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid

def _get_latest(update_type: str) -> Optional[dict]:
    _init_db()
    with sqlite3.connect(_DB_PATH) as con:
        row = con.execute(
            "SELECT json_data, created_at FROM market_updates WHERE type=? ORDER BY id DESC LIMIT 1",
            (update_type,),
        ).fetchone()
    if not row:
        return None
    data = json.loads(row[0])
    data["_generated_at"] = row[1]
    return data

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _set_market_cache(lookback_days: int, data: dict, data_timestamp: str) -> None:
    global _LAST_MARKET_DATA_TIMESTAMP
    _MARKET_CACHE["lookback_days"] = lookback_days
    _MARKET_CACHE["fetched_at"] = datetime.now(timezone.utc)
    _MARKET_CACHE["data_timestamp"] = data_timestamp
    _MARKET_CACHE["data"] = data
    _LAST_MARKET_DATA_TIMESTAMP = data_timestamp

def _get_cached_market_data(lookback_days: int) -> Optional[dict]:
    fetched_at = _MARKET_CACHE.get("fetched_at")
    if (
        _MARKET_CACHE.get("lookback_days") == lookback_days
        and isinstance(fetched_at, datetime)
        and datetime.now(timezone.utc) - fetched_at <= _MARKET_CACHE_TTL
        and isinstance(_MARKET_CACHE.get("data"), dict)
        and _MARKET_CACHE.get("data")
    ):
        data_timestamp = _MARKET_CACHE.get("data_timestamp")
        if isinstance(data_timestamp, str):
            global _LAST_MARKET_DATA_TIMESTAMP
            _LAST_MARKET_DATA_TIMESTAMP = data_timestamp
        return dict(_MARKET_CACHE["data"])
    return None

def _persist_last_good_snapshot(lookback_days: int, data: dict, data_timestamp: str) -> None:
    _MARKET_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lookback_days": lookback_days,
        "data_timestamp": data_timestamp,
        "saved_at": _utc_now_iso(),
        "data": data,
    }
    _MARKET_SNAPSHOT_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

def _load_last_good_snapshot(lookback_days: int) -> Optional[dict]:
    global _LAST_MARKET_DATA_TIMESTAMP
    if not _MARKET_SNAPSHOT_PATH.exists():
        return None
    try:
        payload = json.loads(_MARKET_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        data = payload.get("data")
        if not isinstance(data, dict) or not data:
            return None
        data_timestamp = payload.get("data_timestamp") or _utc_now_iso()
        _set_market_cache(int(payload.get("lookback_days", lookback_days)), data, data_timestamp)
        _LAST_MARKET_DATA_TIMESTAMP = data_timestamp
        return dict(data)
    except Exception as exc:
        logger.warning("[market_updates] Failed to load last good snapshot: %s", exc)
        return None

def _get_market_data_timestamp() -> str:
    return _LAST_MARKET_DATA_TIMESTAMP or _utc_now_iso()

