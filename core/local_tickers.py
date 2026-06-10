"""
EisaX Local Market Tickers Database
=====================================
Supports: Saudi (Tadawul), UAE (ADX/DFM), Egypt (EGX), Kuwait, Qatar
Data loaded from data/tickers/*.json — edit JSON files to add/update tickers.

Usage:
    from core.local_tickers import MARKET_DB, get_all_tickers_flat
"""
import json
import functools
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data" / "tickers"


def _load(market: str) -> dict:
    path = _DATA_DIR / f"{market}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Market dicts — same names as before for backward compatibility
SAUDI_TICKERS  = _load("saudi")
UAE_TICKERS    = _load("uae")
EGYPT_TICKERS  = _load("egypt")
KUWAIT_TICKERS = _load("kuwait")
QATAR_TICKERS  = _load("qatar")

# Market indices (small static dict — kept inline)
MARKET_INDICES = {
    "^TASI": {
        "name_en": "Tadawul All Share Index (TASI)",
        "name_ar": "تاسي",
        "aliases_ar": ["المؤشر العام", "تداول", "مؤشر تاسي"],
        "aliases_en": ["tasi", "tadawul index"],
        "market": "Saudi",
    },
    "^EGX30": {
        "name_en": "EGX 30 Index",
        "name_ar": "إي جي إكس 30",
        "aliases_ar": ["المؤشر المصري", "مؤشر البورصة المصرية", "egx30", "egx"],
        "aliases_en": ["egx30", "egx 30", "egypt index"],
        "market": "Egypt",
    },
    "^ADSMI": {
        "name_en": "Abu Dhabi Securities Market Index",
        "name_ar": "مؤشر أبوظبي",
        "aliases_ar": ["سوق أبوظبي"],
        "aliases_en": ["adx index", "adsmi"],
        "market": "UAE",
    },
    "^DFMGI": {
        "name_en": "Dubai Financial Market Index",
        "name_ar": "مؤشر دبي",
        "aliases_ar": ["سوق دبي"],
        "aliases_en": ["dfm index", "dfmgi"],
        "market": "UAE",
    },
}

MARKET_SUFFIXES = {
    "saudi":   ".SR",
    "egypt":   ".CA",
    "uae_adx": ".AE",
    "uae_dfm": ".DU",
    "kuwait":  ".KW",
    "qatar":   ".QA",
}

SUPPORTED_CURRENCIES = {
    "SAR": {"name": "Saudi Riyal",    "name_ar": "ريال سعودي",   "symbol": "﷼"},
    "AED": {"name": "UAE Dirham",     "name_ar": "درهم إماراتي", "symbol": "د.إ"},
    "EGP": {"name": "Egyptian Pound", "name_ar": "جنيه مصري",    "symbol": "ج.م"},
    "KWF": {"name": "Kuwait Fils",    "name_ar": "فلس كويتي",    "symbol": "ف"},
    "QAR": {"name": "Qatari Riyal",   "name_ar": "ريال قطري",    "symbol": "ر.ق"},
}

MARKET_DB = {
    "saudi":   SAUDI_TICKERS,
    "egypt":   EGYPT_TICKERS,
    "uae":     UAE_TICKERS,
    "kuwait":  KUWAIT_TICKERS,
    "qatar":   QATAR_TICKERS,
    "indices": MARKET_INDICES,
}


@functools.cache
def get_all_tickers_flat() -> dict:
    """Returns a flat dict of all tickers across all markets."""
    flat = {}
    for key, tickers in MARKET_DB.items():
        if key != "indices":
            flat.update(tickers)
    return flat


def get_market_sectors(market: str) -> list:
    """Returns sorted list of unique sectors for a market."""
    tickers = MARKET_DB.get(market, {})
    sectors = {info.get("sector", "Other") for info in tickers.values()}
    return sorted(sectors)


def get_tickers_by_sector(market: str, sector: str) -> dict:
    """Returns all tickers in a given sector for a market."""
    tickers = MARKET_DB.get(market, {})
    return {
        t: info for t, info in tickers.items()
        if info.get("sector", "").lower() == sector.lower()
    }


def get_ticker_currency(ticker: str) -> str:
    """Returns the currency code for a ticker."""
    info = get_all_tickers_flat().get(ticker, {})
    return info.get("currency", "USD")
