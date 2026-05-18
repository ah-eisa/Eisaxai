"""
core.data_layer.sectors — GICS-aligned sector classification helpers.

Sector strings on parquet snapshots are vendor-tagged (TradingView /
Refinitiv) and can drift in casing. This module normalises them into a
fixed GICS-style vocabulary so engines and tests have a stable surface.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from phase_h.registry import FeatureRegistry

from . import _flags  # noqa: F401
from .market_cache_adapter import get_ticker_row

# Canonical 11 GICS sectors (Real Estate split out per 2016 reclassification).
GICS_SECTORS = (
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
)


_ALIASES: Dict[str, str] = {
    "energy minerals":        "Energy",
    "oil & gas":              "Energy",
    "oil and gas":            "Energy",
    "energy":                 "Energy",
    "process industries":     "Materials",
    "non-energy minerals":    "Materials",
    "materials":              "Materials",
    "producer manufacturing": "Industrials",
    "industrial services":    "Industrials",
    "industrials":            "Industrials",
    "retail trade":           "Consumer Discretionary",
    "consumer durables":      "Consumer Discretionary",
    "consumer services":      "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "consumer non-durables":  "Consumer Staples",
    "consumer staples":       "Consumer Staples",
    "health technology":      "Health Care",
    "health services":        "Health Care",
    "healthcare":             "Health Care",
    "health care":            "Health Care",
    "finance":                "Financials",
    "financials":             "Financials",
    "electronic technology":  "Information Technology",
    "technology services":    "Information Technology",
    "technology":             "Information Technology",
    "information technology": "Information Technology",
    "communications":         "Communication Services",
    "communication services": "Communication Services",
    "utilities":              "Utilities",
    "real estate":            "Real Estate",
}


def list_sectors() -> List[str]:
    return list(GICS_SECTORS)


def sector_classification(raw: Optional[str]) -> str:
    """Normalise a raw sector string into a GICS bucket. Unknowns return 'Unknown'."""
    if not raw:
        return "Unknown"
    norm = raw.strip().lower()
    return _ALIASES.get(norm, "Unknown")


def get_sector(ticker: str) -> str:
    """Look up a ticker's sector via the market cache and normalise it."""
    if not FeatureRegistry.is_enabled("data_layer_enabled"):
        return "Unknown"
    row = get_ticker_row(ticker)
    if row is None:
        return "Unknown"
    return sector_classification(row.get("sector"))


__all__ = ["GICS_SECTORS", "list_sectors", "sector_classification", "get_sector"]
