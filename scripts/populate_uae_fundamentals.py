#!/usr/bin/env python3
"""
populate_uae_fundamentals.py — جمع بيانات الأسهم الإماراتية من StockAnalysis.com
وحفظها في SQLite (uae_fundamentals) لتغطية شاملة.

يشغل مرة واحدة أو عبر cron يومياً:
    python3 scripts/populate_uae_fundamentals.py

المصادر:
1. StockAnalysis.com  → PE, EPS, Beta, Dividend, Revenue, Growth, MarketCap
2. Investing.com cache → Price (via market_data_engine)
3. Excel file         → Sector, Industry, Name
"""

import sys, os, time, re, json, sqlite3, logging
sys.path.insert(0, "/home/ubuntu/investwise")
os.chdir("/home/ubuntu/investwise")

from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("populate_uae")

DB_PATH = Path("/home/ubuntu/investwise/core/investwise.db")

# ── Step 1: Upgrade schema ────────────────────────────────────────────────
def upgrade_schema(conn):
    """Add missing columns to uae_fundamentals if they don't exist."""
    new_cols = {
        "forward_pe":       "REAL",
        "net_margin":       "REAL",
        "gross_margin":     "REAL",
        "roe":              "REAL",
        "debt_equity":      "REAL",
        "revenue_growth":   "REAL",
        "earnings_growth":  "REAL",
        "sector":           "TEXT",
        "industry":         "TEXT",
        "net_income":       "REAL",
        "shares_out":       "REAL",
        "week_52_high":     "REAL",
        "week_52_low":      "REAL",
        "price":            "REAL",
        "company_name":     "TEXT",
        "source":           "TEXT",
    }
    existing = {row[1] for row in conn.execute("PRAGMA table_info(uae_fundamentals)")}
    for col, dtype in new_cols.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE uae_fundamentals ADD COLUMN {col} {dtype}")
            logger.info(f"  Added column: {col} ({dtype})")
    conn.commit()


# ── Step 2: Fetch via realtime_data._stockanalysis_uae (battle-tested scraper) ─
def fetch_stockanalysis(ticker: str) -> dict:
    """Use the production-grade _stockanalysis_uae scraper from realtime_data.py.
    This function is already used by the live agent and handles all field extraction
    correctly — no need to duplicate the scraping logic here."""
    try:
        from core.realtime_data import _stockanalysis_uae
        raw = _stockanalysis_uae(ticker)
        if not raw:
            return {}

        def _pct_to_float(val):
            """Convert '+21.0%' or '-2.9%' → 21.0 or -2.9"""
            if val is None: return None
            try:
                s = str(val).replace('%','').replace('+','').strip()
                return float(s)
            except (ValueError, TypeError):
                return None

        def _size_to_float(val):
            """Convert '253.17B AED' → 253170000000.0"""
            if not val: return None
            s = str(val).strip()
            try:
                if 'T' in s: return float(re.sub(r'[^\d.]', '', s.split('T')[0])) * 1e12
                if 'B' in s: return float(re.sub(r'[^\d.]', '', s.split('B')[0])) * 1e9
                if 'M' in s: return float(re.sub(r'[^\d.]', '', s.split('M')[0])) * 1e6
                return float(re.sub(r'[^\d.]', '', s))
            except Exception:
                return None

        def _pct_yield(val):
            """Convert '5.20%' → 0.0520"""
            if not val: return None
            try:
                v = float(str(val).replace('%','').strip())
                return v / 100 if v > 1 else v
            except Exception:
                return None

        result = {
            "ticker":          raw.get("ticker", ticker.upper()),
            "source":          raw.get("source", "StockAnalysis"),
            "price":           raw.get("price"),
            "company_name":    raw.get("company_name"),
            "sector":          raw.get("sector"),
            "industry":        raw.get("industry"),
            "market_cap":      _size_to_float(raw.get("market_cap")),
            "eps":             raw.get("eps"),
            "pe_ratio":        raw.get("pe_ratio"),
            "forward_pe":      raw.get("forward_pe"),
            "beta":            raw.get("beta"),
            "div_yield":       _pct_yield(raw.get("dividend_yield")),
            "revenue":         _size_to_float(raw.get("revenue")),
            "net_income":      _size_to_float(raw.get("net_income")),
            "revenue_growth":  _pct_to_float(raw.get("rev_growth")),
            "earnings_growth": _pct_to_float(raw.get("earnings_growth")),
            "week_52_high":    raw.get("week_52_high"),
            "week_52_low":     raw.get("week_52_low"),
            "shares_out":      raw.get("shares_out_raw"),
        }
        # Compute net margin from revenue + net_income if available
        if result.get("net_income") and result.get("revenue") and result["revenue"] > 0:
            result["net_margin"] = round((result["net_income"] / result["revenue"]) * 100, 2)

        return {k: v for k, v in result.items() if v is not None}

    except Exception as e:
        logger.warning(f"  [{ticker}] fetch_stockanalysis error: {e}")
        return {}

    # (all fields now handled inside fetch_stockanalysis via _stockanalysis_uae)
    return result


# ── Step 3: Get price from Investing.com cache ───────────────────────────
def get_cached_price(ticker: str) -> float:
    """Get latest price from market_data_engine cache."""
    try:
        from core.market_data_engine import get_stock_data
        df = get_stock_data(ticker, "AE", period="1m")
        if df is not None and not df.empty:
            return round(float(df["Close"].iloc[-1]), 3)
    except Exception:
        pass
    return 0.0


# ── Step 4: Get name/sector from Excel ───────────────────────────────────
def get_excel_info(ticker: str) -> dict:
    """Get company info from Excel file."""
    try:
        from core.excel_stock_lookup import get_stock_info
        base = ticker.upper().replace(".AE", "").replace(".DU", "")
        info = get_stock_info(base)
        if info:
            return {
                "company_name": info.get("name"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }
    except Exception:
        pass
    return {}


# ── Step 5: Get 52-week range from historical data ──────────────────────
def get_52w_range(ticker: str) -> dict:
    """Get 52-week high/low from cached historical data."""
    try:
        from core.market_data_engine import get_stock_data
        import pandas as pd
        df = get_stock_data(ticker, "AE", period="1y")
        if df is not None and len(df) > 10:
            return {
                "week_52_high": round(float(df["High"].max()), 3),
                "week_52_low": round(float(df["Low"].min()), 3),
            }
    except Exception:
        pass
    return {}


# ── Step 6: Main Population Loop ────────────────────────────────────────
def populate():
    from core.market_data_engine import UAE_INVESTING

    tickers = sorted(UAE_INVESTING.keys())
    total = len(tickers)
    logger.info(f"🚀 Starting UAE fundamentals population for {total} tickers")

    conn = sqlite3.connect(str(DB_PATH))
    upgrade_schema(conn)

    success = 0
    failed = 0
    skipped = 0

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i}/{total}] {ticker}...")

        # Fetch from StockAnalysis.com
        sa_data = fetch_stockanalysis(ticker)

        # Get price
        price = get_cached_price(ticker)

        # Compute PE if we have price + EPS (cast to float — may arrive as str)
        pe_ratio = None
        try:
            _eps_v = float(sa_data["eps"]) if sa_data.get("eps") else None
            _mc_v  = float(sa_data["market_cap"]) if sa_data.get("market_cap") else None
            _ni_v  = float(sa_data["net_income"]) if sa_data.get("net_income") else None
            if _eps_v and price and _eps_v > 0:
                pe_ratio = round(price / _eps_v, 2)
            elif _mc_v and _ni_v and _ni_v > 0:
                pe_ratio = round(_mc_v / _ni_v, 2)
        except Exception:
            pass
        # Also cast numeric fields that may come as str
        for _fld in ("eps", "pe_ratio", "forward_pe", "beta"):
            if sa_data.get(_fld) is not None:
                try: sa_data[_fld] = float(sa_data[_fld])
                except Exception: sa_data.pop(_fld, None)

        # Get Excel enrichment
        excel = get_excel_info(ticker)

        # Get 52W range
        w52 = get_52w_range(ticker)

        # Merge: SA data > Excel data > existing
        sector = sa_data.get("sector") or excel.get("sector")
        industry = sa_data.get("industry") or excel.get("industry")
        company_name = excel.get("company_name") or ticker.replace(".AE", "").replace(".DU", "")

        # Count useful fields
        useful_fields = sum(1 for v in [
            sa_data.get("market_cap"), pe_ratio, sa_data.get("beta"),
            sa_data.get("eps"), sa_data.get("div_yield"), sa_data.get("revenue"),
            sa_data.get("net_margin"), sa_data.get("revenue_growth")
        ] if v)

        if useful_fields < 1 and not price:
            logger.warning(f"  ⚠️ {ticker}: No useful data found, skipping")
            skipped += 1
        else:
            try:
                conn.execute("""
                    INSERT INTO uae_fundamentals
                    (ticker, name, company_name, market_cap, pe_ratio, beta,
                     eps, div_yield, revenue, forward_pe, net_margin, gross_margin,
                     roe, debt_equity, revenue_growth, earnings_growth,
                     net_income, shares_out, sector, industry,
                     week_52_high, week_52_low, price, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        name = COALESCE(excluded.name, uae_fundamentals.name),
                        company_name = COALESCE(excluded.company_name, uae_fundamentals.company_name),
                        market_cap = COALESCE(excluded.market_cap, uae_fundamentals.market_cap),
                        pe_ratio = COALESCE(excluded.pe_ratio, uae_fundamentals.pe_ratio),
                        beta = COALESCE(excluded.beta, uae_fundamentals.beta),
                        eps = COALESCE(excluded.eps, uae_fundamentals.eps),
                        div_yield = COALESCE(excluded.div_yield, uae_fundamentals.div_yield),
                        revenue = COALESCE(excluded.revenue, uae_fundamentals.revenue),
                        forward_pe = COALESCE(excluded.forward_pe, uae_fundamentals.forward_pe),
                        net_margin = COALESCE(excluded.net_margin, uae_fundamentals.net_margin),
                        gross_margin = COALESCE(excluded.gross_margin, uae_fundamentals.gross_margin),
                        roe = COALESCE(excluded.roe, uae_fundamentals.roe),
                        debt_equity = COALESCE(excluded.debt_equity, uae_fundamentals.debt_equity),
                        revenue_growth = COALESCE(excluded.revenue_growth, uae_fundamentals.revenue_growth),
                        earnings_growth = COALESCE(excluded.earnings_growth, uae_fundamentals.earnings_growth),
                        net_income = COALESCE(excluded.net_income, uae_fundamentals.net_income),
                        shares_out = COALESCE(excluded.shares_out, uae_fundamentals.shares_out),
                        sector = COALESCE(excluded.sector, uae_fundamentals.sector),
                        industry = COALESCE(excluded.industry, uae_fundamentals.industry),
                        week_52_high = COALESCE(excluded.week_52_high, uae_fundamentals.week_52_high),
                        week_52_low = COALESCE(excluded.week_52_low, uae_fundamentals.week_52_low),
                        price = COALESCE(excluded.price, uae_fundamentals.price),
                        source = COALESCE(excluded.source, uae_fundamentals.source),
                        updated_at = excluded.updated_at
                """, (
                    ticker, company_name, company_name,
                    sa_data.get("market_cap"), pe_ratio, sa_data.get("beta"),
                    sa_data.get("eps"), sa_data.get("div_yield"), sa_data.get("revenue"),
                    sa_data.get("forward_pe"), sa_data.get("net_margin"), sa_data.get("gross_margin"),
                    sa_data.get("roe"), sa_data.get("debt_equity"),
                    sa_data.get("revenue_growth"), sa_data.get("earnings_growth"),
                    sa_data.get("net_income"), sa_data.get("shares_out"),
                    sector, industry,
                    w52.get("week_52_high"), w52.get("week_52_low"),
                    price, sa_data.get("source"),
                    datetime.now().isoformat()
                ))
                conn.commit()
                success += 1
                logger.info(f"  ✅ {ticker}: {useful_fields} fields | PE={pe_ratio} | Beta={sa_data.get('beta')} | Div={sa_data.get('div_yield')}")
            except Exception as e:
                logger.error(f"  ❌ {ticker}: DB insert failed: {e}")
                failed += 1

        # Rate limit: 1.5s between requests to avoid SA blocking
        if i < total:
            time.sleep(1.5)

    conn.close()

    logger.info(f"""
{'='*60}
✅ UAE Fundamentals Population Complete
{'='*60}
Total tickers:  {total}
Success:        {success}
Failed:         {failed}
Skipped:        {skipped}
{'='*60}
""")

    # Print summary stats
    conn2 = sqlite3.connect(str(DB_PATH))
    stats = conn2.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN pe_ratio IS NOT NULL THEN 1 ELSE 0 END) as has_pe,
            SUM(CASE WHEN beta IS NOT NULL THEN 1 ELSE 0 END) as has_beta,
            SUM(CASE WHEN div_yield IS NOT NULL THEN 1 ELSE 0 END) as has_div,
            SUM(CASE WHEN revenue IS NOT NULL THEN 1 ELSE 0 END) as has_rev,
            SUM(CASE WHEN net_margin IS NOT NULL THEN 1 ELSE 0 END) as has_margin,
            SUM(CASE WHEN sector IS NOT NULL THEN 1 ELSE 0 END) as has_sector
        FROM uae_fundamentals
    """).fetchone()
    conn2.close()

    logger.info(f"""
📊 Coverage Summary:
  Total rows:    {stats[0]}
  Has PE:        {stats[1]}
  Has Beta:      {stats[2]}
  Has Dividend:  {stats[3]}
  Has Revenue:   {stats[4]}
  Has Margin:    {stats[5]}
  Has Sector:    {stats[6]}
""")


if __name__ == "__main__":
    populate()
