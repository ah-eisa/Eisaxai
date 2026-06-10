"""
excel_stock_lookup.py — Stock sector/name/industry lookup from Excel data files.
Sources:
  - Final_Stocks_Report_All_Countries.xlsx  → UAE (143), Saudi (395), Egypt (238)
  - Regional_Stocks_Cleaned.xlsx            → Kuwait (133), Qatar (54)

Usage:
    from core.excel_stock_lookup import get_stock_info
    info = get_stock_info("ADNOCGAS.AE")
    # → {"name": "ADNOC", "sector": "Energy", "industry": "Oil & Gas",
    #    "exchange": "Abu Dhabi / Dubai", "market_cap": "AED 249.33B"}
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Suffix stripping map ─────────────────────────────────────────────────────
_STRIP_SUFFIXES = (".AE", ".DU", ".SR", ".CA", ".KW", ".QA", ".BH", ".OM")

# ── Ticker alias normalization (system ticker → Excel ticker base) ────────────
# Cases where the system uses a different ticker than what's in the Excel files
_ALIASES: dict = {
    "EMAAR":    "EMAR",       # Emaar Properties (system: EMAAR.DU, Excel: EMAR)
    "DIB":      "DISB",       # Dubai Islamic Bank (system: DIB.DU, Excel: DISB)
    "DEWA":     "DEWAA",      # Dubai Electricity (system: DEWA.DU, Excel: DEWAA)
    "AIRARABI": "AIRA",       # Air Arabia (system: AIRARABI.DU, Excel: AIRA)
    "SHUAA":    "SHUA",       # SHUAA Capital (system: SHUAA.DU, Excel: SHUA)
    "DEYAAR":   "DEYR",       # Deyaar Development
    "TABRD":    "TABR",       # National Central Cooling
    "DUBAITAXI":"DTC",        # Dubai Taxi
    "ENBDREIT": "ENBD",       # ENBD REIT → map to parent
    # Kuwait: Excel Company column vs system ticker base
    "NBK":      "NBKK",       # National Bank of Kuwait
    "BOUBYAN":  "BOUK",       # Boubyan Bank
    "MABANEE":  "MABK",       # Mabanee
    "GBK":      "GBKK",       # Gulf Bank of Kuwait
    "WARBABANK":"WARB",       # Warba Bank
    "CBK":      "CBKK",       # Commercial Bank of Kuwait
    "OOREDOO":  "OORE",       # Ooredoo Kuwait
    "ABK":      "ABKK",       # Al-Ahli Bank of Kuwait
    # Qatar
    "QNBK":     "QNBK",       # Already matches
    "QGAS":     "QGTS",       # Qatar Gas Transport
}

# ── Exchange → suffix (for reverse lookup) ──────────────────────────────────
_EXCH_SUFFIX = {
    "abu dhabi / dubai": (".AE", ".DU"),
    "saudi arabia":      (".SR",),
    "egypt":             (".CA",),
    "kuwait city":       (".KW",),
    "doha":              (".QA",),
}

# ── In-memory lookup dict: base_ticker → info dict ──────────────────────────
_LOOKUP: dict = {}
_LOADED = False

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "..")  # /home/ubuntu/investwise/


def _load():
    global _LOOKUP, _LOADED
    if _LOADED:
        return

    try:
        import pandas as pd

        final_path    = os.path.join(_DATA_DIR, "Final_Stocks_Report_All_Countries.xlsx")
        regional_path = os.path.join(_DATA_DIR, "Regional_Stocks_Cleaned.xlsx")

        rows = []

        # ── File 1: Final_Stocks_Report_All_Countries.xlsx ────────────────────
        if os.path.exists(final_path):
            df = pd.read_excel(final_path)
            for _, r in df.iterrows():
                ticker = str(r.get("Ticker", "")).strip().upper()
                if not ticker:
                    continue
                rows.append({
                    "ticker_base": ticker,
                    "name":        str(r.get("Name", "")).strip(),
                    "exchange":    str(r.get("Exchange", "")).strip(),
                    "sector":      str(r.get("Sector", "")).strip(),
                    "industry":    str(r.get("Industry", "")).strip(),
                    "market_cap":  str(r.get("Market Cap", "")).strip(),
                    "pe_ratio":    str(r.get("P/E Ratio", "")).strip(),
                    "last_price":  str(r.get("Last Trade Price", "")).strip(),
                })
            logger.info("[ExcelLookup] Loaded %d rows from Final_Stocks", len(rows))

        # ── File 2: Regional_Stocks_Cleaned.xlsx (Kuwait + Qatar) ────────────
        if os.path.exists(regional_path):
            df2 = pd.read_excel(regional_path)
            regional_start = len(rows)
            for _, r in df2.iterrows():
                ticker = str(r.get("Company", "")).strip().upper()
                if not ticker:
                    continue
                rows.append({
                    "ticker_base": ticker,
                    "name":        str(r.get("Name", "")).strip(),
                    "exchange":    str(r.get("Exchange", "")).strip(),
                    "sector":      str(r.get("Sector", "")).strip(),
                    "industry":    str(r.get("Industry", "")).strip(),
                    "market_cap":  str(r.get("Market Cap", "")).strip(),
                    "pe_ratio":    str(r.get("P/E Ratio", "")).strip(),
                    "last_price":  str(r.get("Last Trade Price", "")).strip(),
                })
            logger.info("[ExcelLookup] Loaded %d rows from Regional_Stocks",
                        len(rows) - regional_start)

        # ── Build lookup ──────────────────────────────────────────────────────
        for row in rows:
            key = row["ticker_base"]
            if key and key not in ("NAN", "TICKER"):
                _LOOKUP[key] = row

        logger.info("[ExcelLookup] Total lookup entries: %d", len(_LOOKUP))
        _LOADED = True

    except Exception as e:
        logger.error("[ExcelLookup] Failed to load Excel data: %s", e)
        _LOADED = True  # Prevent retry loops


def _strip_suffix(ticker: str) -> str:
    """Return base ticker without exchange suffix."""
    t = ticker.upper().strip()
    for sfx in _STRIP_SUFFIXES:
        if t.endswith(sfx):
            return t[: -len(sfx)]
    return t


def get_stock_info(ticker: str) -> Optional[dict]:
    """
    Return sector/name/industry for a ticker.
    Tries: exact match → suffix-stripped → alias → partial fuzzy.
    Returns None if not found.
    """
    _load()

    if not ticker:
        return None

    t = ticker.upper().strip()

    # 1. Exact match
    if t in _LOOKUP:
        return _LOOKUP[t]

    # 2. Strip suffix → base ticker
    base = _strip_suffix(t)
    if base in _LOOKUP:
        return _LOOKUP[base]

    # 3. Alias normalization (system ticker → Excel ticker)
    aliased = _ALIASES.get(base)
    if aliased and aliased in _LOOKUP:
        return _LOOKUP[aliased]

    # 4. Also try the reverse: maybe Excel ticker matches a longer system name
    #    e.g. "EMAAR" → look for any key starting with base
    for key in _LOOKUP:
        if key.startswith(base) or base.startswith(key):
            if abs(len(key) - len(base)) <= 2:  # allow 2-char difference
                return _LOOKUP[key]

    return None


def get_sector(ticker: str, default: str = "Unknown") -> str:
    """Return sector for ticker, default if not found."""
    info = get_stock_info(ticker)
    if info:
        sector = info.get("sector", "")
        if sector and sector not in ("nan", "NaN", "Sector"):
            return sector
    return default


def get_industry(ticker: str, default: str = "") -> str:
    """Return industry for ticker."""
    info = get_stock_info(ticker)
    if info:
        ind = info.get("industry", "")
        if ind and ind not in ("nan", "NaN"):
            return ind
    return default


def get_company_name(ticker: str, default: str = "") -> str:
    """Return English company name for ticker."""
    info = get_stock_info(ticker)
    if info:
        name = info.get("name", "")
        if name and name not in ("nan", "NaN"):
            return name
    return default


def enrich_fund_dict(ticker: str, fund: dict) -> dict:
    """
    Enrich a fundamentals dict with Excel data.
    Only fills fields that are empty/Unknown/N/A — never overwrites real data.
    """
    info = get_stock_info(ticker)
    if not info:
        return fund

    def _fill(key: str, val: str):
        if val and val not in ("nan", "NaN", "", "N/A", "Unknown"):
            if not fund.get(key) or fund.get(key) in ("Unknown", "N/A", "", "nan"):
                fund[key] = val

    _fill("sector",       info.get("sector", ""))
    _fill("industry",     info.get("industry", ""))
    # For company name: also replace if it looks like a raw ticker symbol
    _base = _strip_suffix(ticker)
    if (not fund.get("company_name")
            or fund.get("company_name") in ("Unknown", "N/A", "", "nan")
            or fund.get("company_name", "").upper() in (ticker.upper(), _base.upper())):
        xl_name = info.get("name", "")
        if xl_name and xl_name not in ("nan", "NaN", ""):
            fund["company_name"] = xl_name

    return fund


# ── Pre-load on import ───────────────────────────────────────────────────────
try:
    _load()
except Exception:
    pass


if __name__ == "__main__":
    # Quick test
    tests = [
        "ADNOCGAS.AE", "FAB.AE", "EMAAR.DU", "ADNOCDRILL.AE",
        "2222.SR", "COMI.CA", "EMAR.AE", "KFH.KW", "QNBK.QA",
        "TAQA.AE", "IHC.AE", "ENBD.DU",
    ]
    for t in tests:
        info = get_stock_info(t)
        if info:
            print(f"{t:20s} → {info['name'][:30]:30s} | {info['sector']:25s} | {info['industry']}")
        else:
            print(f"{t:20s} → NOT FOUND")
