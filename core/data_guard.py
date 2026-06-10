"""
Data Guard — EisaX.
Pre-flight data completeness check + waterfall enrichment + last-resort scraping.

Workflow for every ticker before deep analysis:
    1. check_completeness(ticker, level) → identify missing fields
    2. If missing → enrich(ticker, missing_fields):
         a. Search local DB sources (uae_fundamentals, stock_knowledge)
         b. Try cached APIs (RapidAPI, pipeline parquet)
         c. Try live APIs (yfinance for non-Arab, TradingView for Arab)
         d. Last resort: scrape StockAnalysis.com / Investing.com
    3. Re-check completeness, return final report.

Usage:
    from core.data_guard import ensure_complete, DataLevel
    report = ensure_complete("ADNOCGAS.AE", DataLevel.INSTITUTIONAL)
    if not report.is_complete:
        # report.missing_fields tells you what couldn't be found
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("eisax.data_guard")


# ── Data level definitions ────────────────────────────────────────────────────
class DataLevel(str, Enum):
    BASIC         = "basic"
    TECHNICAL     = "technical"
    FUNDAMENTAL   = "fundamental"
    INSTITUTIONAL = "institutional"


REQUIRED_FIELDS: dict[str, list[str]] = {
    "basic": [
        "name", "sector", "price",
    ],
    "technical": [
        "name", "price", "rsi", "sma50", "sma200",
    ],
    "fundamental": [
        "name", "sector", "price",
        "pe_ratio", "eps", "market_cap", "div_yield", "beta",
    ],
    "institutional": [
        "name", "sector", "industry", "price",
        "pe_ratio", "eps", "market_cap", "div_yield", "beta",
        "revenue", "net_income", "net_margin",
        "gross_margin", "operating_margin",
        "roe", "roa", "debt_equity", "current_ratio",
        "ebitda", "free_cash_flow", "price_book",
        "revenue_growth", "earnings_growth",
        "week_52_high", "week_52_low",
    ],
}


# ── Report dataclass ──────────────────────────────────────────────────────────
@dataclass
class CompletenessReport:
    ticker:           str
    level:            str
    is_complete:      bool
    completeness_pct: float
    present_fields:   dict        = field(default_factory=dict)
    missing_fields:   list[str]   = field(default_factory=list)
    sources_used:     list[str]   = field(default_factory=list)
    scrape_performed: bool        = False
    duration_ms:      int         = 0
    error:            Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ticker":           self.ticker,
            "level":            self.level,
            "is_complete":      self.is_complete,
            "completeness_pct": round(self.completeness_pct, 1),
            "present_fields":   {k: v for k, v in self.present_fields.items() if v is not None},
            "missing_fields":   self.missing_fields,
            "sources_used":     self.sources_used,
            "scrape_performed": self.scrape_performed,
            "duration_ms":      self.duration_ms,
            "error":            self.error,
        }


# ── Helper: classify ticker market ────────────────────────────────────────────
def _market_of(ticker: str) -> str:
    """Return one of: ksa, uae, egypt, kuwait, qatar, us, other."""
    t = ticker.upper()
    if t.endswith(".SR"):                          return "ksa"
    if t.endswith((".AE", ".DU", ".AD")):          return "uae"
    if t.endswith(".CA"):                          return "egypt"
    if t.endswith(".KW"):                          return "kuwait"
    if t.endswith(".QA"):                          return "qatar"
    if "." not in t or t.endswith(("-USD", "=F", "=X")):  return "us"
    return "other"


def _is_arab(ticker: str) -> bool:
    return _market_of(ticker) in ("ksa", "uae", "egypt", "kuwait", "qatar")


# ── Source: DB lookup (uae_fundamentals table) ────────────────────────────────
_DB_COLUMN_MAP: dict[str, str] = {
    "name":              "company_name",
    "sector":            "sector",
    "industry":          "industry",
    "price":             "price",
    "pe_ratio":          "pe_ratio",
    "forward_pe":        "forward_pe",
    "eps":               "eps",
    "market_cap":        "market_cap",
    "div_yield":         "div_yield",
    "beta":              "beta",
    "revenue":           "revenue",
    "net_income":        "net_income",
    "net_margin":        "net_margin",
    "gross_margin":      "gross_margin",
    "operating_margin":  "operating_margin",
    "roe":               "roe",
    "roa":               "roa",
    "debt_equity":       "debt_equity",
    "current_ratio":     "current_ratio",
    "ebitda":            "ebitda",
    "free_cash_flow":    "free_cash_flow",
    "price_book":        "price_book",
    "revenue_growth":    "revenue_growth",
    "earnings_growth":   "earnings_growth",
    "shares_out":        "shares_out",
    "week_52_high":      "week_52_high",
    "week_52_low":       "week_52_low",
    "rsi":               "rsi",
    "sma50":             "sma50",
    "sma200":            "sma200",
    "macd":              "macd",
    "volume":            "volume",
}


def _fetch_from_db(ticker: str, fields: list[str]) -> dict:
    """Read fields from uae_fundamentals table."""
    try:
        from core.config import CORE_DB
        conn = sqlite3.connect(str(CORE_DB))
        db_cols = [_DB_COLUMN_MAP[f] for f in fields if f in _DB_COLUMN_MAP]
        if not db_cols:
            return {}
        sql = f"SELECT {', '.join(db_cols)} FROM uae_fundamentals WHERE ticker=? LIMIT 1"
        row = conn.execute(sql, (ticker.upper(),)).fetchone()
        conn.close()
        if not row:
            return {}
        out: dict = {}
        for field_name, val in zip([f for f in fields if f in _DB_COLUMN_MAP], row):
            if val is not None and val != "":
                out[field_name] = val
        return out
    except Exception as e:
        logger.warning(f"[data_guard] DB lookup failed for {ticker}: {e}")
        return {}


# ── Source: yfinance ──────────────────────────────────────────────────────────
def _fetch_from_yfinance(ticker: str, fields: list[str]) -> dict:
    """Pull fields from yfinance. Skips Arab markets that Yahoo doesn't cover."""
    market = _market_of(ticker)
    if market in ("uae",):
        return {}   # ADX/.AE blocked at yfinance
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        if not info:
            return {}

        mapping = {
            "name":            info.get("longName") or info.get("shortName"),
            "sector":          info.get("sector"),
            "industry":        info.get("industry"),
            "price":           info.get("currentPrice") or info.get("regularMarketPrice"),
            "pe_ratio":        info.get("trailingPE"),
            "forward_pe":      info.get("forwardPE"),
            "eps":             info.get("trailingEps"),
            "market_cap":      info.get("marketCap"),
            "div_yield":       info.get("dividendYield"),
            "beta":            info.get("beta"),
            "revenue":         info.get("totalRevenue"),
            "net_income":      info.get("netIncomeToCommon"),
            "net_margin":      (info.get("profitMargins", 0) * 100) if info.get("profitMargins") else None,
            "gross_margin":    (info.get("grossMargins", 0) * 100) if info.get("grossMargins") else None,
            "roe":             (info.get("returnOnEquity", 0) * 100) if info.get("returnOnEquity") else None,
            "debt_equity":     info.get("debtToEquity"),
            "revenue_growth":  (info.get("revenueGrowth", 0) * 100) if info.get("revenueGrowth") else None,
            "earnings_growth": (info.get("earningsGrowth", 0) * 100) if info.get("earningsGrowth") else None,
            "shares_out":      info.get("sharesOutstanding"),
            "week_52_high":    info.get("fiftyTwoWeekHigh"),
            "week_52_low":     info.get("fiftyTwoWeekLow"),
        }
        return {k: v for k, v in mapping.items() if k in fields and v is not None}
    except Exception as e:
        logger.warning(f"[data_guard] yfinance failed for {ticker}: {e}")
        return {}


# ── Source: TradingView Query (Arab markets, broad coverage) ──────────────────
# Expanded field set — TV exposes institutional-grade metrics for ADX/Tadawul/EGX
# that StockAnalysis.com doesn't. ROE, gross margin, EBITDA, FCF, etc.
_TV_FIELDS = [
    "name", "close", "market_cap_basic",
    "price_earnings_ttm", "dividend_yield_recent",
    "earnings_per_share_diluted_ttm", "sector", "industry",
    "beta_1_year",
    "total_revenue_ttm", "net_income_ttm",
    # Profitability
    "return_on_equity", "return_on_assets",
    "gross_margin_ttm", "operating_margin_ttm", "net_margin_ttm",
    "ebitda_ttm",
    # Balance sheet
    "debt_to_equity", "current_ratio",
    # Cash flow + Valuation
    "free_cash_flow_ttm", "price_book_fq",
    # Technicals
    "RSI",
]


def _fetch_from_tradingview(ticker: str, fields: list[str]) -> dict:
    """Use TradingView Query for Arab tickers — broad fundamental coverage."""
    if not _is_arab(ticker):
        return {}
    try:
        from tradingview_screener import Query, Column
        symbol = ticker.upper().replace(".AE", "").replace(".DU", "").replace(".SR", "")
        market_map = {"ksa": "ksa", "uae": "uae", "egypt": "egypt",
                      "kuwait": "kuwait", "qatar": "qatar"}
        market = market_map.get(_market_of(ticker))
        if not market:
            return {}

        _, df = (Query()
                 .set_markets(market)
                 .select(*_TV_FIELDS)
                 .where(Column("name") == symbol)
                 .limit(1)
                 .get_scanner_data())

        if df.empty:
            return {}
        row = df.iloc[0]
        mapping = {
            "name":              row.get("name"),
            "price":             row.get("close"),
            "market_cap":        row.get("market_cap_basic"),
            "pe_ratio":          row.get("price_earnings_ttm"),
            "div_yield":         row.get("dividend_yield_recent"),
            "eps":               row.get("earnings_per_share_diluted_ttm"),
            "sector":            row.get("sector"),
            "industry":          row.get("industry"),
            "beta":              row.get("beta_1_year"),
            "revenue":           row.get("total_revenue_ttm"),
            "net_income":        row.get("net_income_ttm"),
            "rsi":               row.get("RSI"),
            # ── Newly-fetched institutional metrics ───────────────────────────
            "roe":               row.get("return_on_equity"),
            "roa":               row.get("return_on_assets"),
            "gross_margin":      row.get("gross_margin_ttm"),
            "operating_margin":  row.get("operating_margin_ttm"),
            "net_margin":        row.get("net_margin_ttm"),
            "ebitda":            row.get("ebitda_ttm"),
            "debt_equity":       row.get("debt_to_equity"),
            "current_ratio":     row.get("current_ratio"),
            "free_cash_flow":    row.get("free_cash_flow_ttm"),
            "price_book":        row.get("price_book_fq"),
        }
        return {k: v for k, v in mapping.items()
                if k in fields and v is not None
                and not (isinstance(v, float) and str(v).lower() == "nan")}
    except Exception as e:
        logger.warning(f"[data_guard] TradingView failed for {ticker}: {e}")
        return {}


# ── Source: StockAnalysis scrape (last resort for Arab tickers) ───────────────
def _scrape_stockanalysis(ticker: str, fields: list[str]) -> dict:
    """Scrape StockAnalysis.com via existing realtime_data helper."""
    if not _is_arab(ticker):
        return {}
    try:
        from core.realtime_data import _stockanalysis_uae
        raw = _stockanalysis_uae(ticker)
        if not raw:
            return {}

        def _pct(v):
            if v is None: return None
            try: return float(str(v).replace("%", "").replace("+", "").strip())
            except: return None

        def _size(v):
            if not v: return None
            s = str(v).strip()
            try:
                if "T" in s: return float(s.split("T")[0].replace(",", "")) * 1e12
                if "B" in s: return float(s.split("B")[0].replace(",", "")) * 1e9
                if "M" in s: return float(s.split("M")[0].replace(",", "")) * 1e6
                return float(s.replace(",", ""))
            except: return None

        def _yield(v):
            if not v: return None
            try:
                f = float(str(v).replace("%", "").strip())
                return f / 100 if f > 1 else f
            except: return None

        mapping = {
            "name":            raw.get("company_name"),
            "sector":          raw.get("sector"),
            "industry":        raw.get("industry"),
            "price":           raw.get("price"),
            "market_cap":      _size(raw.get("market_cap")),
            "eps":             raw.get("eps"),
            "pe_ratio":        raw.get("pe_ratio"),
            "forward_pe":      raw.get("forward_pe"),
            "beta":            raw.get("beta"),
            "div_yield":       _yield(raw.get("dividend_yield")),
            "revenue":         _size(raw.get("revenue")),
            "net_income":      _size(raw.get("net_income")),
            "revenue_growth":  _pct(raw.get("rev_growth")),
            "earnings_growth": _pct(raw.get("earnings_growth")),
            "week_52_high":    raw.get("week_52_high"),
            "week_52_low":     raw.get("week_52_low"),
            "shares_out":      raw.get("shares_out_raw"),
        }
        result = {k: v for k, v in mapping.items() if k in fields and v is not None}
        # Compute net_margin if not present but revenue + net_income are
        if "net_margin" in fields and "net_margin" not in result:
            rev = mapping.get("revenue")
            ni  = mapping.get("net_income")
            if rev and ni and rev > 0:
                result["net_margin"] = round((ni / rev) * 100, 2)
        return result
    except Exception as e:
        logger.warning(f"[data_guard] StockAnalysis scrape failed for {ticker}: {e}")
        return {}


# ── Source: pipeline cache (parquet) for technicals/prices ────────────────────
def _fetch_from_pipeline_cache(ticker: str, fields: list[str]) -> dict:
    """Load price/technical fields from pipeline parquet cache."""
    technical_fields = {"price", "rsi", "sma50", "sma200", "macd", "volume"}
    needed = [f for f in fields if f in technical_fields]
    if not needed:
        return {}
    try:
        from core.market_data import get_full_stock_profile
        profile = get_full_stock_profile(ticker) or {}
        mapping = {
            "price":   profile.get("price"),
            "rsi":     profile.get("rsi"),
            "sma50":   profile.get("sma50"),
            "sma200":  profile.get("sma200"),
            "macd":    profile.get("macd"),
            "volume":  profile.get("volume"),
            "name":    profile.get("name"),
            "sector":  profile.get("sector"),
        }
        return {k: v for k, v in mapping.items() if k in fields and v is not None}
    except Exception as e:
        logger.warning(f"[data_guard] pipeline cache failed for {ticker}: {e}")
        return {}


# ── DB save (after enrichment) ─────────────────────────────────────────────────
def _save_to_db(ticker: str, fields_dict: dict, source: str) -> None:
    """Persist enriched fields back to uae_fundamentals (works for any ticker)."""
    if not fields_dict:
        return
    try:
        from core.config import CORE_DB
        conn = sqlite3.connect(str(CORE_DB))

        db_data = {}
        for field_name, val in fields_dict.items():
            db_col = _DB_COLUMN_MAP.get(field_name)
            if db_col and val is not None:
                # Coerce numpy / pandas types to Python native — otherwise SQLite
                # stores them as BLOB (binary pickle of the numpy scalar).
                try:
                    if hasattr(val, "item"):  # numpy scalar
                        val = val.item()
                except Exception:
                    pass
                if isinstance(val, (bytes, bytearray)):
                    # Last-resort fallback for pre-stored BLOB values
                    try:
                        import struct
                        val = struct.unpack("<q", val[:8])[0] if len(val) == 8 else None
                    except Exception:
                        val = None
                db_data[db_col] = val

        if not db_data:
            conn.close()
            return

        from datetime import datetime
        db_data["source"] = source
        db_data["updated_at"] = datetime.utcnow().isoformat() + "Z"

        cols      = list(db_data.keys())
        placeholders = ", ".join(["?"] * (len(cols) + 1))   # +1 for ticker
        col_list  = ", ".join(["ticker"] + cols)
        updates   = ", ".join([f"{c}=COALESCE(excluded.{c}, uae_fundamentals.{c})" for c in cols])

        sql = (f"INSERT INTO uae_fundamentals ({col_list}) VALUES ({placeholders}) "
               f"ON CONFLICT(ticker) DO UPDATE SET {updates}")
        conn.execute(sql, [ticker.upper()] + [db_data[c] for c in cols])
        conn.commit()
        conn.close()
        logger.info(f"[data_guard] saved {len(cols)} fields for {ticker} (source={source})")
    except Exception as e:
        logger.warning(f"[data_guard] DB save failed for {ticker}: {e}")


# ── Public API ────────────────────────────────────────────────────────────────
def check_completeness(ticker: str, level: str = "fundamental") -> CompletenessReport:
    """Cheap local-only check: DB + pipeline cache. No live calls."""
    t0 = time.time()
    if level not in REQUIRED_FIELDS:
        return CompletenessReport(
            ticker=ticker, level=level, is_complete=False,
            completeness_pct=0.0, error=f"unknown level: {level}"
        )

    required = REQUIRED_FIELDS[level]
    collected: dict = {}
    sources: list[str] = []

    # Layer 1: DB
    db_data = _fetch_from_db(ticker, required)
    if db_data:
        collected.update(db_data)
        sources.append("db")

    # Layer 2: pipeline cache (only for technical/price fields, fast)
    still_missing = [f for f in required if f not in collected]
    if still_missing:
        pc_data = _fetch_from_pipeline_cache(ticker, still_missing)
        if pc_data:
            collected.update(pc_data)
            sources.append("pipeline_cache")

    missing = [f for f in required if f not in collected or collected[f] is None]
    pct = ((len(required) - len(missing)) / len(required) * 100) if required else 100.0

    return CompletenessReport(
        ticker=ticker, level=level,
        is_complete=(len(missing) == 0),
        completeness_pct=pct,
        present_fields=collected,
        missing_fields=missing,
        sources_used=sources,
        duration_ms=int((time.time() - t0) * 1000),
    )


def enrich(ticker: str, missing_fields: list[str], allow_scrape: bool = True) -> tuple[dict, list[str]]:
    """
    Try all sources to fill missing fields. Returns (enriched_dict, sources_used).
    Saves to DB at the end.
    """
    if not missing_fields:
        return {}, []

    collected: dict = {}
    sources: list[str] = []
    needed = list(missing_fields)

    # Source priority by market
    market = _market_of(ticker)
    if market == "us" or market == "other":
        chain = [
            ("yfinance",      _fetch_from_yfinance),
            ("tradingview",   _fetch_from_tradingview),
        ]
    else:   # Arab markets
        chain = [
            ("tradingview",   _fetch_from_tradingview),
            ("yfinance",      _fetch_from_yfinance),   # tries for .SR/.CA/.KW/.QA
            ("pipeline",      _fetch_from_pipeline_cache),
        ]

    for src_name, src_fn in chain:
        if not needed:
            break
        new = src_fn(ticker, needed)
        if new:
            collected.update(new)
            sources.append(src_name)
            needed = [f for f in needed if f not in collected]

    # Last resort: scrape (only for Arab markets where SA covers fundamentals)
    if needed and allow_scrape and _is_arab(ticker):
        new = _scrape_stockanalysis(ticker, needed)
        if new:
            collected.update(new)
            sources.append("stockanalysis_scrape")
            needed = [f for f in needed if f not in collected]

    # Persist
    if collected:
        _save_to_db(ticker, collected, source=", ".join(sources))

    return collected, sources


def ensure_complete(
    ticker: str,
    level: str = "fundamental",
    allow_scrape: bool = True,
) -> CompletenessReport:
    """
    Full guardrail: check → enrich → re-check. Use before every analysis.
    Saves anything new to DB so subsequent calls are instant.
    """
    t0 = time.time()
    # Step 1: cheap local check
    report = check_completeness(ticker, level)
    if report.is_complete:
        return report

    # Step 2: enrich missing fields
    logger.info(f"[data_guard] {ticker}: {len(report.missing_fields)} fields missing — enriching...")
    enriched, sources = enrich(ticker, report.missing_fields, allow_scrape=allow_scrape)
    scrape_performed = "stockanalysis_scrape" in sources

    # Step 3: re-check after enrichment
    final = check_completeness(ticker, level)
    final.sources_used = list(set(final.sources_used + sources))
    final.scrape_performed = scrape_performed
    final.duration_ms = int((time.time() - t0) * 1000)

    return final


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    ticker = sys.argv[1] if len(sys.argv) > 1 else "ADNOCGAS.AE"
    level  = sys.argv[2] if len(sys.argv) > 2 else "institutional"
    print(json.dumps(ensure_complete(ticker, level).to_dict(), indent=2, default=str))
