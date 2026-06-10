"""
core/services/regional_handler.py
──────────────────────────────────
Regional market enrichment — fills yfinance data gaps using local SQLite DB
for Saudi (.SR), UAE (.AE / .DU), Egyptian (.CA) and other GCC tickers.

Public API
──────────
    merge_regional_data(target, fund) -> dict
        Merges the appropriate regional DB rows into the *fund* dict in-place
        (and also returns the enriched dict for chaining convenience).

    detect_currency(target) -> tuple[str, str]
        Returns (symbol, label) e.g. ("﷼", "SAR") for Saudi tickers.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_DB_PATH = Path(__file__).parent.parent / "investwise.db"

# Field mappings: (fund_key, db_column)
_COMMON_FIELDS = [
    ("net_margin",       "net_margin"),
    ("gross_margin",     "gross_margin"),
    ("roe",              "roe"),
    ("revenue_growth",   "revenue_growth"),
    ("eps",              "eps"),
    ("pe_ratio",         "pe_ratio"),
    ("forward_pe",       "forward_pe"),
    ("beta",             "beta"),
    ("debt_equity",      "debt_equity"),
    ("sector",           "sector"),
    ("company_name",     "company_name"),
]

_SA_EXTRA_FIELDS = _COMMON_FIELDS + [
    ("earnings_growth",  "earnings_growth"),
    ("div_yield",        "div_yield"),
    ("revenue",          "revenue"),
]

_EGX_FIELDS = [
    ("roe",             "roe"),
    ("eps",             "eps"),
    ("pe_ratio",        "pe_ratio"),
    ("forward_pe",      "forward_pe"),
    ("beta",            "beta"),
    ("net_margin",      "net_margin"),
    ("gross_margin",    "gross_margin"),
    ("revenue_growth",  "revenue_growth"),
]

# Approximate EGP/USD rate (used when market cap looks like USD for EGX stocks)
_EGP_USD_RATE = 49.0


# ── Internal DB helper ────────────────────────────────────────────────────────

def _query_db(table: str, ticker: str) -> dict | None:
    """Return one row from *table* matching *ticker*, or None."""
    try:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"SELECT * FROM {table} WHERE ticker=?", (ticker.upper(),)  # noqa: S608
            ).fetchone()
        return dict(row) if row else None
    except Exception as exc:
        logger.debug("[RegionalDB] %s/%s query failed: %s", table, ticker, exc)
        return None


def _merge_fields(
    fund: dict,
    db_row: dict,
    field_map: list[tuple[str, str]],
) -> None:
    """Overwrite empty fund fields with DB values (in-place)."""
    for fund_key, db_col in field_map:
        if not fund.get(fund_key) and db_row.get(db_col) is not None:
            fund[fund_key] = db_row[db_col]


# ── Public API ────────────────────────────────────────────────────────────────

def merge_regional_data(target: str, fund: dict) -> dict:
    """
    Enrich *fund* with regional SQLite data and apply market-specific fixes.

    Steps executed (only those relevant to the ticker suffix):
      1. Saudi   (.SR)  — uae_fundamentals table (extended field set)
      2. UAE     (.AE / .DU) — uae_fundamentals table
      3. Egypt   (.CA)  — egx_fundamentals table + market-cap EGP conversion
      4. All     — global sector classification fallback
      5. All     — crash / extreme-move flag injected into fund

    Returns the enriched *fund* dict (modified in-place + returned for chaining).
    """
    t_upper = target.upper()

    # ── 1. Saudi ──────────────────────────────────────────────────────────────
    if t_upper.endswith(".SR"):
        row = _query_db("uae_fundamentals", t_upper)
        if row:
            _merge_fields(fund, row, _SA_EXTRA_FIELDS)
            if not fund.get("market_cap") and row.get("market_cap"):
                fund["market_cap"] = row["market_cap"]
            logger.info(
                "[SA-DB] %s: nm=%s roe=%s pe=%s",
                target, fund.get("net_margin"), fund.get("roe"), fund.get("pe_ratio"),
            )

    # ── 2. UAE ────────────────────────────────────────────────────────────────
    elif t_upper.endswith((".AE", ".DU")):
        row = _query_db("uae_fundamentals", t_upper)
        if row:
            _merge_fields(fund, row, _COMMON_FIELDS)
            if not fund.get("market_cap") and row.get("market_cap"):
                fund["market_cap"] = row["market_cap"]
            logger.info(
                "[UAE-DB] %s: nm=%s roe=%s gm=%s",
                target, fund.get("net_margin"), fund.get("roe"), fund.get("gross_margin"),
            )

    # ── 3. Egypt ──────────────────────────────────────────────────────────────
    elif t_upper.endswith(".CA"):
        row = _query_db("egx_fundamentals", t_upper)
        if row:
            _merge_fields(fund, row, _EGX_FIELDS)
            if not fund.get("market_cap") and row.get("market_cap"):
                fund["market_cap"] = row["market_cap"]
            logger.info(
                "[EGX-DB] %s: pe=%s roe=%s eps=%s",
                target, fund.get("pe_ratio"), fund.get("roe"), fund.get("eps"),
            )
        # Market cap for EGX stocks is stored in USD in some sources — convert
        mc = fund.get("market_cap")
        if mc and float(mc) < 1e10:  # looks like USD (< $10B) → convert to EGP
            fund["market_cap"] = float(mc) * _EGP_USD_RATE
            logger.info("[EGP] %s: converted market_cap USD → EGP", target)

    # ── 4. Global sector fallback (all tickers) ───────────────────────────────
    if not fund.get("sector") or fund.get("sector") in ("Unknown", "N/A", ""):
        try:
            from core.fundamental_engine import _classify_sector as _clf
            classified = _clf(target)
            if classified:
                fund["sector"] = classified
                logger.info("[Sector] %s classified as '%s'", target, classified)
        except Exception as exc:
            logger.debug("[Sector] classification fallback failed for %s: %s", target, exc)

    return fund


def detect_currency(target: str) -> tuple[str, str]:
    """
    Return (symbol, label) for the ticker's trading currency.

    Examples
    ────────
    "2222.SR"      → ("﷼", "SAR")
    "EMAAR.DU"     → ("د.إ", "AED")
    "COMI.CA"      → ("ج.م", "EGP")
    "KFH.KW"       → ("ف", "KWF")
    "QNBK.QA"      → ("ر.ق", "QAR")
    "AAPL"         → ("$", "USD")
    """
    t = target.upper()
    if t.endswith(".SR"):
        return "﷼", "SAR"
    if t.endswith((".AE", ".DU")):
        return "د.إ", "AED"
    if t.endswith(".CA"):
        return "ج.م", "EGP"
    if t.endswith(".KW"):
        return "ف", "KWF"
    if t.endswith(".QA"):
        return "ر.ق", "QAR"
    return "$", "USD"
