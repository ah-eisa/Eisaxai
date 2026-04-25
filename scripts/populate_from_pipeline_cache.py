#!/usr/bin/env python3
"""
populate_from_pipeline_cache.py
────────────────────────────────
Populate uae_fundamentals (and sister tables) from the TradingView
parquet cache — no scraping, no network calls, runs in <5 seconds.

Covers: UAE (ADX/DFM), KSA, Egypt, Kuwait, Qatar, Bahrain, Morocco, Tunisia.

Run:
    python3 scripts/populate_from_pipeline_cache.py
"""

import sys, os, glob, math, sqlite3, logging
from datetime import datetime, timezone

sys.path.insert(0, "/home/ubuntu/investwise")
os.chdir("/home/ubuntu/investwise")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline_populate")

from core.config import CORE_DB

CACHE_DIR = "/home/ubuntu/investwise/market_cache"

# market → (dot-suffix, table-name)
MARKETS = {
    "uae":     (".AE", "uae_fundamentals"),
    "ksa":     (".SR", "uae_fundamentals"),   # same table — ticker suffix differentiates
    "egypt":   (".CA", "uae_fundamentals"),
    "kuwait":  (".KW", "uae_fundamentals"),
    "qatar":   (".QA", "uae_fundamentals"),
    "bahrain": (".BH", "uae_fundamentals"),
    "morocco": (".MA", "uae_fundamentals"),
    "tunisia": (".TN", "uae_fundamentals"),
}


def _safe_float(v):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f) or f == 0) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _safe_str(v):
    s = str(v).strip() if v is not None else ""
    return s if s.lower() not in ("", "nan", "none", "n/a", "unknown") else None


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uae_fundamentals (
            ticker          TEXT PRIMARY KEY,
            name            TEXT,
            company_name    TEXT,
            market_cap      REAL,
            pe_ratio        REAL,
            eps             REAL,
            beta            REAL,
            div_yield       REAL,
            revenue         REAL,
            forward_pe      REAL,
            net_margin      REAL,
            gross_margin    REAL,
            roe             REAL,
            debt_equity     REAL,
            revenue_growth  REAL,
            earnings_growth REAL,
            net_income      REAL,
            shares_out      REAL,
            sector          TEXT,
            industry        TEXT,
            week_52_high    REAL,
            week_52_low     REAL,
            price           REAL,
            source          TEXT,
            updated_at      TEXT
        )
    """)
    # Add any missing columns from older schema versions
    existing = {row[1] for row in conn.execute("PRAGMA table_info(uae_fundamentals)")}
    extras = {
        "rsi": "REAL", "sma50": "REAL", "sma200": "REAL",
        "macd": "REAL", "volume": "INTEGER",
    }
    for col, dtype in extras.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE uae_fundamentals ADD COLUMN {col} {dtype}")
    conn.commit()
    log.info("Table uae_fundamentals ready")


def load_latest(market: str):
    files = sorted(glob.glob(f"{CACHE_DIR}/{market}_*.parquet"))
    if not files:
        return None
    import pandas as pd
    return pd.read_parquet(files[-1])


def populate():
    import pandas as pd

    conn = sqlite3.connect(str(CORE_DB))
    create_table(conn)

    now_iso = datetime.now(timezone.utc).isoformat()
    total_inserted = 0

    for market, (suffix, table) in MARKETS.items():
        df = load_latest(market)
        if df is None or df.empty:
            log.warning("No parquet found for %s — skipping", market)
            continue

        # bare ticker = part after ":"
        df = df.copy()
        df["_bare"] = df["ticker"].astype(str).str.split(":").str[-1].str.upper()
        df["_full"] = df["_bare"] + suffix

        inserted = 0
        for _, row in df.iterrows():
            ticker = row["_full"]
            name   = _safe_str(row.get("name")) or row["_bare"]

            pe     = _safe_float(row.get("price_earnings_ttm"))
            eps    = _safe_float(row.get("earnings_per_share_diluted_ttm"))
            mc     = _safe_float(row.get("market_cap_basic"))
            sect   = _safe_str(row.get("sector"))
            divy   = _safe_float(row.get("dividend_yield_recent"))
            if divy and divy > 1:
                divy = round(divy / 100, 6)
            price  = _safe_float(row.get("close"))
            rsi    = _safe_float(row.get("RSI"))
            sma50  = _safe_float(row.get("SMA50"))
            sma200 = _safe_float(row.get("SMA200"))
            macd   = _safe_float(row.get("MACD.macd"))
            vol    = int(row.get("volume") or 0) or None

            conn.execute("""
                INSERT INTO uae_fundamentals
                    (ticker, name, company_name, market_cap, pe_ratio, eps, div_yield,
                     sector, price, rsi, sma50, sma200, macd, volume, source, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ticker) DO UPDATE SET
                    name        = COALESCE(excluded.name,        uae_fundamentals.name),
                    company_name= COALESCE(excluded.company_name,uae_fundamentals.company_name),
                    market_cap  = COALESCE(excluded.market_cap,  uae_fundamentals.market_cap),
                    pe_ratio    = COALESCE(excluded.pe_ratio,    uae_fundamentals.pe_ratio),
                    eps         = COALESCE(excluded.eps,         uae_fundamentals.eps),
                    div_yield   = COALESCE(excluded.div_yield,   uae_fundamentals.div_yield),
                    sector      = COALESCE(excluded.sector,      uae_fundamentals.sector),
                    price       = COALESCE(excluded.price,       uae_fundamentals.price),
                    rsi         = COALESCE(excluded.rsi,         uae_fundamentals.rsi),
                    sma50       = COALESCE(excluded.sma50,       uae_fundamentals.sma50),
                    sma200      = COALESCE(excluded.sma200,      uae_fundamentals.sma200),
                    macd        = COALESCE(excluded.macd,        uae_fundamentals.macd),
                    volume      = COALESCE(excluded.volume,      uae_fundamentals.volume),
                    source      = excluded.source,
                    updated_at  = excluded.updated_at
            """, (
                ticker, name, name, mc, pe, eps, divy,
                sect, price, rsi, sma50, sma200, macd, vol,
                "TradingView Pipeline Cache", now_iso,
            ))
            inserted += 1

        conn.commit()
        log.info("  %s: %d rows upserted → uae_fundamentals", market.upper(), inserted)
        total_inserted += inserted

    conn.close()
    log.info("Done — %d total rows in uae_fundamentals", total_inserted)


if __name__ == "__main__":
    populate()
