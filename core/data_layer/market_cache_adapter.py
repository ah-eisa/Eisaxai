"""
core.data_layer.market_cache_adapter — read-only wrapper over the existing
15-minute parquet cache at /home/ubuntu/investwise/market_cache/.

Never rebuilds. Never writes. Discovers the latest snapshot per market via
index.json (preferred) and falls back to glob ordering when index.json is
absent or malformed. All heavy reads are memoised via phase_h.cache.

Public surface:
    list_markets()                              -> tuple[str, ...]
    snapshot_timestamp(market)                  -> str | None
    get_latest_snapshot(market)                 -> pd.DataFrame | None
    get_ticker_row(ticker, market=None)         -> dict | None
    get_universe_panel(tickers, columns=None)   -> pd.DataFrame
    MarketCacheAdapter (class-level façade)
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from phase_h.cache import memoize
from phase_h.registry import FeatureRegistry

from . import _flags  # noqa: F401 — side-effect: register data_layer flags
from .base import (
    DEFAULT_SNAPSHOT_TTL_SECONDS,
    STALE_SNAPSHOT_HARD_LIMIT_SECONDS,
    ALL_MARKETS,
    RecordNotFound,
    StaleSnapshotError,
)

# The literal "market_cache" filesystem path is intentionally constrained
# to this single module — every other consumer reaches the cache through
# the exported helpers below.
CACHE_ROOT = "/home/ubuntu/investwise/market_cache"
FACTOR_PANEL_DIR = os.path.join(CACHE_ROOT, "factor_panels")

logger = logging.getLogger("data_layer.market_cache_adapter")

_INDEX_PATH = os.path.join(CACHE_ROOT, "index.json")
_FILENAME_TS_RE = re.compile(r"^(?P<market>[a-z]+)_(?P<date>\d{8})_(?P<time>\d{4})\.parquet$")

_LOCK = threading.RLock()
_INDEX_CACHE: Dict[str, Any] = {"loaded_at": 0.0, "data": None}


# ──────────────────────────────────────────────────────────────────────
# index.json + filename discovery
# ──────────────────────────────────────────────────────────────────────

def _load_index_json(max_age_seconds: int = 60) -> Dict[str, Any]:
    """Cached index.json read (60s TTL — file changes every 15 min)."""
    with _LOCK:
        now = time.time()
        if _INDEX_CACHE["data"] is not None and (now - _INDEX_CACHE["loaded_at"]) < max_age_seconds:
            return _INDEX_CACHE["data"]  # type: ignore[return-value]
        data: Dict[str, Any] = {}
        try:
            with open(_INDEX_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
        except (OSError, ValueError) as exc:
            logger.debug("index.json unavailable (%s) — falling back to glob", exc)
            data = {}
        _INDEX_CACHE["data"] = data
        _INDEX_CACHE["loaded_at"] = now
        return data


def _glob_latest_parquet(market: str) -> Optional[str]:
    """Filename-sorted fallback when index.json is missing."""
    pattern = os.path.join(CACHE_ROOT, f"{market}_*.parquet")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _latest_filename(market: str) -> Optional[str]:
    """Resolve the latest parquet filename for `market`."""
    idx = _load_index_json()
    entries = idx.get(market) or []
    if entries:
        # index.json entries are ordered ascending; last is newest
        latest = entries[-1].get("filename")
        if latest:
            return os.path.join(CACHE_ROOT, latest)
    return _glob_latest_parquet(market)


def _filename_timestamp(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = _FILENAME_TS_RE.match(base)
    if not m:
        return None
    return f"{m.group('date')}T{m.group('time')}"


def _age_seconds_from_path(path: str) -> Optional[float]:
    try:
        return time.time() - os.path.getmtime(path)
    except OSError:
        return None


# ──────────────────────────────────────────────────────────────────────
# Public lookups
# ──────────────────────────────────────────────────────────────────────

def list_markets() -> Tuple[str, ...]:
    """All markets currently visible in the cache (subset of ALL_MARKETS)."""
    idx = _load_index_json()
    if idx:
        seen = tuple(m for m in ALL_MARKETS if m in idx and idx[m])
        if seen:
            return seen
    # glob fallback
    seen_g: List[str] = []
    for m in ALL_MARKETS:
        if _glob_latest_parquet(m):
            seen_g.append(m)
    return tuple(seen_g)


def snapshot_timestamp(market: str) -> Optional[str]:
    """ISO-ish snapshot timestamp string for the latest snapshot of `market`."""
    idx = _load_index_json()
    entries = idx.get(market) or []
    if entries and entries[-1].get("timestamp"):
        return str(entries[-1]["timestamp"])
    path = _latest_filename(market)
    return _filename_timestamp(path) if path else None


@memoize("data_layer.market_snapshot", ttl_seconds=DEFAULT_SNAPSHOT_TTL_SECONDS)
def _read_snapshot(*, market: str) -> Optional["pd.DataFrame"]:
    """Memoised parquet read for one market. Returns None when missing."""
    if pd is None:
        return None
    path = _latest_filename(market)
    if path is None or not os.path.exists(path):
        return None
    age = _age_seconds_from_path(path)
    if age is not None and age > STALE_SNAPSHOT_HARD_LIMIT_SECONDS:
        if FeatureRegistry.is_enabled("data_layer_strict_stale"):
            raise StaleSnapshotError(
                f"{market} snapshot age {age:.0f}s exceeds stale limit "
                f"{STALE_SNAPSHOT_HARD_LIMIT_SECONDS}s"
            )
        logger.warning("data_layer: %s snapshot is stale (%.0fs)", market, age)
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("data_layer: failed to read %s (%s)", path, exc)
        return None
    return df


def get_latest_snapshot(market: str) -> Optional["pd.DataFrame"]:
    """Public accessor: latest snapshot DataFrame for one market."""
    if not FeatureRegistry.is_enabled("data_layer_enabled"):
        return None
    market = (market or "").strip().lower()
    if not market:
        return None
    return _read_snapshot(market=market)


def get_ticker_row(ticker: str, market: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Return the latest cached row for one ticker as a plain dict.
    Scans every market when `market` is unspecified — useful for cross-market
    universes (e.g. KSA + UAE + Egypt in one portfolio).
    """
    if not ticker or pd is None:
        return None
    ticker = ticker.strip()
    markets = [market.lower()] if market else list(list_markets())
    for m in markets:
        df = get_latest_snapshot(m)
        if df is None or "ticker" not in df.columns:
            continue
        hits = df[df["ticker"] == ticker]
        if not hits.empty:
            row = hits.iloc[0].to_dict()
            row.setdefault("_market", m)
            return row
    return None


def get_universe_panel(
    tickers: Sequence[str],
    columns: Optional[Sequence[str]] = None,
) -> Optional["pd.DataFrame"]:
    """
    Build a DataFrame of one row per ticker across all markets.
    `columns` restricts the returned columns when provided.
    """
    if pd is None or not tickers:
        return None
    wanted = {t.strip() for t in tickers if t}
    frames: List["pd.DataFrame"] = []
    for m in list_markets():
        df = get_latest_snapshot(m)
        if df is None or "ticker" not in df.columns:
            continue
        sub = df[df["ticker"].isin(wanted)]
        if not sub.empty:
            frames.append(sub.assign(_market=m))
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    if columns is not None:
        keep = [c for c in columns if c in out.columns]
        out = out[keep]
    return out


# ──────────────────────────────────────────────────────────────────────
# Class-level façade — convenient when an engine wants one handle.
# ──────────────────────────────────────────────────────────────────────

class MarketCacheAdapter:
    """Stateless façade — methods delegate to module-level functions."""

    @staticmethod
    def is_enabled() -> bool:
        return FeatureRegistry.is_enabled("data_layer_enabled")

    @staticmethod
    def markets() -> Tuple[str, ...]:
        return list_markets()

    @staticmethod
    def snapshot(market: str) -> Optional["pd.DataFrame"]:
        return get_latest_snapshot(market)

    @staticmethod
    def ticker(symbol: str, market: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return get_ticker_row(symbol, market=market)

    @staticmethod
    def panel(tickers: Sequence[str], columns: Optional[Sequence[str]] = None):
        return get_universe_panel(tickers, columns=columns)

    @staticmethod
    def stats() -> Dict[str, Any]:
        markets = list_markets()
        return {
            "markets_available": list(markets),
            "snapshots": {m: snapshot_timestamp(m) for m in markets},
        }


def find_records(field: str, value: Any) -> List[Dict[str, Any]]:
    """Cross-market lookup: every row where `field == value`. Defensive — empty list on miss."""
    if pd is None:
        return []
    out: List[Dict[str, Any]] = []
    for m in list_markets():
        df = get_latest_snapshot(m)
        if df is None or field not in df.columns:
            continue
        for _, row in df[df[field] == value].iterrows():
            d = row.to_dict()
            d.setdefault("_market", m)
            out.append(d)
    return out


def raise_if_missing(ticker: str) -> Dict[str, Any]:
    """Convenience: get_ticker_row that raises RecordNotFound on miss."""
    row = get_ticker_row(ticker)
    if row is None:
        raise RecordNotFound(f"ticker {ticker!r} not in any cached market")
    return row


# ──────────────────────────────────────────────────────────────────────
# Returns-panel builder — replaces the per-engine cache walker that
# previously lived inside phase_h.benchmarks._load_cached_returns and
# phase_h.factor_model._load_cached_returns.
# ──────────────────────────────────────────────────────────────────────

def _canonical_ticker(symbol: str) -> str:
    """Lower-noise canonical form used to match across exchange prefixes."""
    if not symbol:
        return ""
    s = str(symbol).split(":")[-1]
    return s.strip().upper()


@memoize("data_layer.returns_panel", ttl_seconds=DEFAULT_SNAPSHOT_TTL_SECONDS)
def _returns_panel(*, canon_tickers: tuple) -> Optional["pd.DataFrame"]:
    """Memoised returns-panel build for a set of canonical tickers."""
    if pd is None or not canon_tickers:
        return None
    wanted = set(canon_tickers)
    price_rows: List["pd.Series"] = []
    idx = _load_index_json()
    if not idx:
        return None
    for market_code, snaps in idx.items():
        for snap in snaps:
            filename = snap.get("filename")
            if not filename:
                continue
            path = os.path.join(CACHE_ROOT, filename)
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path, columns=["ticker", "close", "_snapshot_ts"])
            except Exception:  # pragma: no cover — defensive
                continue
            if df.empty:
                continue
            df = df.copy()
            df["_canon"] = df["ticker"].map(lambda x: _canonical_ticker(str(x)))
            df = df[df["_canon"].isin(wanted)]
            if df.empty:
                continue
            ts = pd.to_datetime(df["_snapshot_ts"].iloc[0], errors="coerce")
            if pd.isna(ts):
                ts = pd.Timestamp(snap.get("timestamp") or snap.get("created_at") or None)
            row = df.drop_duplicates("_canon").set_index("_canon")["close"]
            row.name = ts
            price_rows.append(pd.to_numeric(row, errors="coerce"))
    if not price_rows:
        return None
    prices = pd.concat(price_rows, axis=1).T.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices.pct_change().dropna(how="all")


def get_returns_panel(tickers: Iterable[str]) -> Optional["pd.DataFrame"]:
    """
    Build a wide DataFrame of pct-change returns across snapshots.

    Index   : snapshot timestamp
    Columns : canonical ticker (exchange prefix stripped, upper-cased)
    Values  : period returns

    Returns None when no rows match — engines should treat this as
    "fall through to Indicative tier" rather than raising.
    """
    if not FeatureRegistry.is_enabled("data_layer_enabled"):
        return None
    canon = tuple(sorted({_canonical_ticker(t) for t in tickers if t}))
    if not canon:
        return None
    return _returns_panel(canon_tickers=canon)


__all__ = [
    "CACHE_ROOT",
    "FACTOR_PANEL_DIR",
    "list_markets",
    "snapshot_timestamp",
    "get_latest_snapshot",
    "get_ticker_row",
    "get_universe_panel",
    "find_records",
    "raise_if_missing",
    "get_returns_panel",
    "MarketCacheAdapter",
]
