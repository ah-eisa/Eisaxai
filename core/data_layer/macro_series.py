"""
core.data_layer.macro_series — read-only macro series accessors.

Today this layer exposes a static set of macro reference series sourced
from the existing commodities snapshot (Brent, Gold, USD index, US 10Y).
A full FRED / IMF integration is out of scope for this phase — the data
contract is the priority so engines can depend on stable keys.

Whenever the underlying snapshot is unavailable, the series is returned
with `value=None` and `notes=["unavailable"]` — never raises.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from phase_h.cache import memoize
from phase_h.registry import FeatureRegistry

from . import _flags  # noqa: F401
from .base import DEFAULT_SNAPSHOT_TTL_SECONDS
from .market_cache_adapter import get_ticker_row
from .utils.validation import coerce_float

logger = logging.getLogger("data_layer.macro_series")


# Key → ticker mapping. Tickers must be present in the commodities or
# america cache. Engines should call `get_macro_series("brent")`.
_KEYS: Dict[str, Dict[str, Any]] = {
    "brent":       {"ticker": "TVC:UKOIL",    "label": "Brent Crude",          "unit": "USD/bbl"},
    "wti":         {"ticker": "TVC:USOIL",    "label": "WTI Crude",            "unit": "USD/bbl"},
    "gold":        {"ticker": "TVC:GOLD",     "label": "Gold Spot",            "unit": "USD/oz"},
    "silver":      {"ticker": "TVC:SILVER",   "label": "Silver Spot",          "unit": "USD/oz"},
    "copper":      {"ticker": "TVC:COPPER",   "label": "Copper",               "unit": "USD/lb"},
    "dxy":         {"ticker": "TVC:DXY",      "label": "US Dollar Index",      "unit": "index"},
    "us10y":       {"ticker": "TVC:US10Y",    "label": "US 10Y Treasury",      "unit": "%"},
    "us02y":       {"ticker": "TVC:US02Y",    "label": "US 2Y Treasury",       "unit": "%"},
    "vix":         {"ticker": "TVC:VIX",      "label": "CBOE VIX",             "unit": "index"},
}


def list_macro_keys() -> List[str]:
    return list(_KEYS.keys())


@memoize("data_layer.macro_series", ttl_seconds=DEFAULT_SNAPSHOT_TTL_SECONDS)
def get_macro_series(*, key: str) -> Dict[str, Any]:
    """
    Latest value + metadata for a named macro series.
    Always returns a dict (never None) so engines never branch on absence
    at the type level.
    """
    if not FeatureRegistry.is_enabled("data_layer_macro_series"):
        return {"key": key, "value": None, "label": None, "unit": None,
                "snapshot": None, "notes": ["macro_disabled"]}
    meta = _KEYS.get(key.lower())
    if meta is None:
        return {"key": key, "value": None, "label": None, "unit": None,
                "snapshot": None, "notes": ["unknown_key"]}
    row = get_ticker_row(meta["ticker"])
    if row is None:
        return {
            "key": key, "value": None,
            "label": meta["label"], "unit": meta["unit"],
            "snapshot": None, "notes": ["unavailable"],
        }
    return {
        "key": key,
        "value": coerce_float(row.get("close")),
        "label": meta["label"],
        "unit": meta["unit"],
        "snapshot": row.get("_snapshot_ts"),
        "notes": [],
    }


__all__ = ["list_macro_keys", "get_macro_series"]
