"""
core.data_layer.benchmarks — canonical benchmark catalog + lookup.

Mirrors phase_h.benchmarks but is region-aware and exposes a stable
interface so future engines can request a benchmark by ticker, region,
or asset_class without rediscovering the catalog.

This module is intentionally small — heavy-lifting computations (rolling
beta, Brinson attribution, capture ratios, …) stay inside phase_h.benchmarks.
The data layer only owns the canonical CATALOG and the cached series.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from phase_h.cache import memoize
from phase_h.registry import FeatureRegistry

from . import _flags  # noqa: F401
from .base import DEFAULT_SNAPSHOT_TTL_SECONDS
from .market_cache_adapter import get_ticker_row

logger = logging.getLogger("data_layer.benchmarks")


# Single source of truth — kept in sync with phase_h.benchmarks (additive).
BENCHMARK_CATALOG: Dict[str, Dict[str, Any]] = {
    # US
    "SPY":      {"label": "S&P 500",                "region": "US",     "asset_class": "Equity", "source_market": "america"},
    "QQQ":      {"label": "NASDAQ-100",             "region": "US",     "asset_class": "Equity", "source_market": "america"},
    "DIA":      {"label": "Dow Jones Industrial",   "region": "US",     "asset_class": "Equity", "source_market": "america"},
    "IWM":      {"label": "Russell 2000",           "region": "US",     "asset_class": "Equity", "source_market": "america"},
    # Global / DM / EM
    "ACWI":     {"label": "MSCI All-Country World", "region": "Global", "asset_class": "Equity", "source_market": "america"},
    "EFA":      {"label": "MSCI EAFE",              "region": "DM",     "asset_class": "Equity", "source_market": "america"},
    "EEM":      {"label": "MSCI Emerging Markets",  "region": "EM",     "asset_class": "Equity", "source_market": "america"},
    "GCC":      {"label": "MSCI GCC Composite",     "region": "GCC",    "asset_class": "Equity", "source_market": "ksa"},
    # GCC native indices (proxied by largest single-country ETFs / aggregates)
    "KSA":      {"label": "iShares MSCI Saudi",     "region": "KSA",    "asset_class": "Equity", "source_market": "ksa"},
    "QAT":      {"label": "iShares MSCI Qatar",     "region": "Qatar",  "asset_class": "Equity", "source_market": "qatar"},
    "UAE":      {"label": "iShares MSCI UAE",       "region": "UAE",    "asset_class": "Equity", "source_market": "uae"},
    "EGPT":     {"label": "VanEck Egypt Index",     "region": "Egypt",  "asset_class": "Equity", "source_market": "egypt"},
    # Bonds
    "AGG":      {"label": "US Aggregate Bond",      "region": "US",     "asset_class": "Bond",   "source_market": "america"},
    "TLT":      {"label": "20+ Year Treasury",      "region": "US",     "asset_class": "Bond",   "source_market": "america"},
    "EMB":      {"label": "EM USD Sovereign Bond",  "region": "EM",     "asset_class": "Bond",   "source_market": "america"},
    # Commodities / Alt
    "GLD":      {"label": "SPDR Gold Shares",       "region": "Global", "asset_class": "Commodity", "source_market": "america"},
    "SLV":      {"label": "iShares Silver Trust",   "region": "Global", "asset_class": "Commodity", "source_market": "america"},
    "USO":      {"label": "Crude Oil",              "region": "Global", "asset_class": "Commodity", "source_market": "commodities"},
    "BIL":      {"label": "1-3 Month T-Bill",       "region": "US",     "asset_class": "Cash",   "source_market": "america"},
}


def list_benchmarks(region: Optional[str] = None, asset_class: Optional[str] = None) -> List[str]:
    """List benchmark tickers; optional region / asset_class filter."""
    out = []
    for tk, meta in BENCHMARK_CATALOG.items():
        if region and meta["region"].lower() != region.lower():
            continue
        if asset_class and meta["asset_class"].lower() != asset_class.lower():
            continue
        out.append(tk)
    return out


def get_benchmark(ticker: str) -> Optional[Dict[str, Any]]:
    """Return catalog entry for `ticker` (case-insensitive) or None."""
    if not ticker:
        return None
    key = ticker.strip().upper()
    return BENCHMARK_CATALOG.get(key)


@memoize("data_layer.benchmark_series", ttl_seconds=DEFAULT_SNAPSHOT_TTL_SECONDS)
def _benchmark_close(*, ticker: str) -> Optional[float]:
    """Latest close price for a benchmark from the cache. Memoised."""
    row = get_ticker_row(ticker)
    if row is None:
        return None
    val = row.get("close")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def get_benchmark_series(ticker: str) -> Dict[str, Any]:
    """
    Return the latest-known benchmark snapshot as a flat dict:
        {ticker, label, region, asset_class, close, snapshot_market}.
    Sparse-friendly: missing fields are `None`.
    """
    meta = get_benchmark(ticker) or {}
    close = _benchmark_close(ticker=ticker.upper()) if ticker else None
    return {
        "ticker": (ticker or "").upper(),
        "label": meta.get("label"),
        "region": meta.get("region"),
        "asset_class": meta.get("asset_class"),
        "close": close,
        "snapshot_market": meta.get("source_market"),
    }


__all__ = [
    "BENCHMARK_CATALOG",
    "list_benchmarks",
    "get_benchmark",
    "get_benchmark_series",
]
