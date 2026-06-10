"""
market_data_engine.py — EisaX Hybrid Data Engine
Cache-first: يجيب من المخزن لو موجود، أو يسكرب ويخزن
"""

import yfinance as yf
import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path("/home/ubuntu/investwise/data/historical")
DB_PATH  = Path("/home/ubuntu/investwise/core/investwise.db")

# ─── Ticker Maps ─────────────────────────────────────────────────────────────

# السعودي والمصري عبر yfinance
YFINANCE_MARKETS = {"SA": ".SR", "EG": ".CA"}

# الإماراتي عبر Investing.com
UAE_INVESTING = {
    "EMAAR.DU":     {"id": "1055159", "ref": "emaar-properties-historical-data"},
    "ENBD.DU":      {"id": "12548",   "ref": "emirates-nbd-historical-data"},
    "DIB.DU":       {"id": "941311",  "ref": "dubai-islamic-bank-historical-data"},
    "DEWA.DU":      {"id": "941326",  "ref": "dewa-historical-data"},
    "SALIK.DU":     {"id": "1194944", "ref": "salik-historical-data"},
    "FAB.AE":       {"id": "999060",  "ref": "first-abu-dhabi-bank-historical-data"},
    "ALDAR.AE":     {"id": "941317",  "ref": "aldar-properties-historical-data"},
    "ADIB.AE":      {"id": "941307",  "ref": "abu-dhabi-islamic-bank-historical-data"},
    "TAQA.AE":      {"id": "941330",  "ref": "taqa-historical-data"},
    "IHC.AE":       {"id": "941296",  "ref": "international-holding-historical-data"},
    "ADNOCDIST.AE": {"id": "1194945", "ref": "adnoc-distribution-historical-data"},
}

# ─── DB Setup ─────────────────────────────────────────────────────────────────

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_reports (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker    TEXT NOT NULL,
            market    TEXT NOT NULL,
            report    TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_cache_meta (
            ticker     TEXT PRIMARY KEY,
            market     TEXT,
            last_updated TEXT,
            rows       INTEGER
        )
    """)
    conn.commit()
    conn.close()

_init_db()

# ─── Cache Helpers ────────────────────────────────────────────────────────────

def _parquet_path(ticker: str, market: str) -> Path:
    safe = ticker.replace(".", "_").replace("^", "IDX_")
    return DATA_DIR / market / f"{safe}.parquet"

def _load_cache(ticker: str, market: str) -> Optional[pd.DataFrame]:
    path = _parquet_path(ticker, market)
    if path.exists():
        return pd.read_parquet(path)
    return None

def _save_cache(df: pd.DataFrame, ticker: str, market: str):
    path = _parquet_path(ticker, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    # حدّث الـ meta
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO data_cache_meta (ticker, market, last_updated, rows)
        VALUES (?, ?, ?, ?)
    """, (ticker, market, datetime.today().strftime("%Y-%m-%d"), len(df)))
    conn.commit()
    conn.close()
    logger.info(f"✅ Cached {ticker} ({len(df)} rows)")

def _is_fresh(ticker: str, market: str) -> bool:
    """الداتا fresh لو آخر تحديث كان النهارده"""
    path = _parquet_path(ticker, market)
    if not path.exists():
        return False
    df = pd.read_parquet(path)
    if df.empty:
        return False
    last = pd.to_datetime(df.index[-1])
    # السوق مغلق الجمعة والسبت — نحسب آخر يوم تداول
    today = datetime.today()
    days_back = 1 if today.weekday() not in [4, 5] else (today.weekday() - 3)
    last_trading = today - timedelta(days=days_back)
    return last.date() >= last_trading.date()

# ─── yfinance Scraper (SA + EG) ───────────────────────────────────────────────

def _fetch_yfinance(ticker: str, market: str, start: str = "2018-01-01") -> Optional[pd.DataFrame]:
    old = _load_cache(ticker, market)
    if old is not None:
        last = pd.to_datetime(old.index[-1])
        start = (last + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df.empty:
            return old
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)

        if old is not None:
            df = pd.concat([old, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()

        _save_cache(df, ticker, market)
        return df
    except Exception as e:
        logger.error(f"yfinance error {ticker}: {e}")
        return old

# ─── Investing.com Scraper (AE) ───────────────────────────────────────────────

def _parse_volume(v: str) -> float:
    v = v.replace(",", "").strip()
    if "M" in v: return float(v.replace("M", "")) * 1_000_000
    if "K" in v: return float(v.replace("K", "")) * 1_000
    try: return float(v)
    except: return 0

def _fetch_investing(ticker: str, info: dict, start: str = "2018-01-01") -> Optional[pd.DataFrame]:
    old = _load_cache(ticker, "AE")
    if old is not None:
        last = pd.to_datetime(old.index[-1])
        start = (last + timedelta(days=1)).strftime("%Y-%m-%d")
        if start >= datetime.today().strftime("%Y-%m-%d"):
            return old

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.today()

    # chunks بحجم سنة
    chunks = []
    cur = start_dt
    while cur < end_dt:
        chunk_end = min(cur + timedelta(days=365), end_dt)
        chunks.append((cur.strftime("%m/%d/%Y"), chunk_end.strftime("%m/%d/%Y")))
        cur = chunk_end + timedelta(days=1)

    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer": f"https://www.investing.com/equities/{info['ref']}",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    all_records = []
    for st, en in chunks:
        try:
            r = scraper.post(
                "https://www.investing.com/instruments/HistoricalDataAjax",
                headers=headers,
                data={"curr_id": info["id"], "st_date": st, "end_date": en,
                      "interval_sec": "Daily", "sort_col": "date",
                      "sort_ord": "DESC", "action": "historical_data"},
                timeout=20,
            )
            soup = BeautifulSoup(r.text, "html.parser")
            for row in soup.select("#curr_table tbody tr"):
                cols = row.find_all("td")
                if len(cols) >= 5 and "No results" not in cols[0].text:
                    try:
                        all_records.append({
                            "Date":   pd.to_datetime(cols[0].text.strip()),
                            "Close":  float(cols[1].text.strip().replace(",", "")),
                            "Open":   float(cols[2].text.strip().replace(",", "")),
                            "High":   float(cols[3].text.strip().replace(",", "")),
                            "Low":    float(cols[4].text.strip().replace(",", "")),
                            "Volume": _parse_volume(cols[5].text.strip()) if len(cols) > 5 else 0,
                        })
                    except:
                        continue
            time.sleep(1.2)
        except Exception as e:
            logger.error(f"Investing.com chunk error {ticker} {st}-{en}: {e}")

    if not all_records:
        return old

    df = pd.DataFrame(all_records).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if old is not None:
        df = pd.concat([old, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()

    _save_cache(df, ticker, "AE")
    return df

# ─── Main API ─────────────────────────────────────────────────────────────────

def get_stock_data(
    ticker: str,
    market: str,
    period: str = "1y",
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """
    الواجهة الرئيسية — cache-first
    
    Args:
        ticker: "2222.SR", "COMI.CA", "EMAAR.DU"
        market: "SA", "EG", "AE"
        period: "1y", "6m", "3m", "2y", "5y", "all"
        force_refresh: يجيب فريش حتى لو عندنا cache
    
    Returns:
        DataFrame: OHLCV مفلتر بالـ period
    """
    # إذا الداتا fresh ومش محتاج refresh
    if not force_refresh and _is_fresh(ticker, market):
        df = _load_cache(ticker, market)
    else:
        # جيب وخزّن
        if market == "AE" and ticker in UAE_INVESTING:
            df = _fetch_investing(ticker, UAE_INVESTING[ticker])
        elif market in ("SA", "EG"):
            df = _fetch_yfinance(ticker, market)
        else:
            logger.warning(f"Unknown market/ticker: {ticker}/{market}")
            df = _load_cache(ticker, market)

    if df is None or df.empty:
        return None

    # فلتر الـ period
    df.index = pd.to_datetime(df.index)
    today = datetime.today()
    periods = {
        "1m": 30, "3m": 90, "6m": 180,
        "1y": 365, "2y": 730, "5y": 1825, "all": 99999,
    }
    days = periods.get(period, 365)
    return df[df.index >= today - timedelta(days=days)]


def get_latest_price(ticker: str, market: str) -> Optional[dict]:
    """آخر سعر للسهم"""
    df = get_stock_data(ticker, market, period="1m")
    if df is None or len(df) < 2:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    change = ((last["Close"] - prev["Close"]) / prev["Close"]) * 100
    currency = "SAR" if market == "SA" else "EGP" if market == "EG" else "AED"
    return {
        "ticker": ticker, "market": market,
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "close": round(last["Close"], 2),
        "open": round(last["Open"], 2),
        "high": round(last["High"], 2),
        "low": round(last["Low"], 2),
        "volume": int(last["Volume"]),
        "change_pct": round(change, 2),
        "currency": currency,
    }


def get_returns(ticker: str, market: str, period: str = "1y") -> Optional[pd.Series]:
    """Daily returns للـ Portfolio Optimizer"""
    df = get_stock_data(ticker, market, period=period)
    if df is None:
        return None
    return df["Close"].pct_change().dropna()


# ─── Report Cache ─────────────────────────────────────────────────────────────

def save_report(ticker: str, market: str, report: str):
    """يحفظ تقرير التحليل في SQLite"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO analysis_reports (ticker, market, report) VALUES (?, ?, ?)",
        (ticker, market, report)
    )
    conn.commit()
    conn.close()


def get_latest_report(ticker: str, market: str, max_age_hours: int = 24) -> Optional[str]:
    """يجيب آخر تقرير محفوظ لو مش قديم"""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT report, created_at FROM analysis_reports
        WHERE ticker=? AND market=?
        ORDER BY created_at DESC LIMIT 1
    """, (ticker, market)).fetchone()
    conn.close()

    if not row:
        return None

    report, created_at = row
    age = datetime.now() - datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
    if age.total_seconds() / 3600 > max_age_hours:
        return None
    return report


def get_data_summary() -> dict:
    """إحصائيات الداتا المخزنة"""
    summary = {}
    for market in ["SA", "EG", "AE"]:
        path = DATA_DIR / market
        if path.exists():
            files = list(path.glob("*.parquet"))
            total = sum(len(pd.read_parquet(f)) for f in files)
            summary[market] = {"tickers": len(files), "rows": total}
        else:
            summary[market] = {"tickers": 0, "rows": 0}
    return summary


# ─── للـ Finance Agent ────────────────────────────────────────────────────────

def prepare_analysis_context(ticker: str, market: str, period: str = "1y") -> dict:
    """
    يحضّر كل الداتا اللي DeepSeek محتاجها للتحليل
    Returns dict جاهز تبعته في الـ prompt
    """
    df = get_stock_data(ticker, market, period=period)
    if df is None:
        return {"error": f"No data for {ticker}"}

    latest = get_latest_price(ticker, market)
    returns = df["Close"].pct_change().dropna()

    # إحصائيات أساسية
    stats = {
        "ticker": ticker,
        "market": market,
        "period": period,
        "latest_price": latest,
        "data_points": len(df),
        "date_range": {
            "from": df.index[0].strftime("%Y-%m-%d"),
            "to": df.index[-1].strftime("%Y-%m-%d"),
        },
        "price_stats": {
            "high_52w": round(df["Close"].max(), 2),
            "low_52w": round(df["Close"].min(), 2),
            "avg_volume": int(df["Volume"].mean()),
        },
        "returns_stats": {
            "total_return_pct": round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2),
            "volatility_annual": round(returns.std() * (252 ** 0.5) * 100, 2),
            "sharpe_approx": round((returns.mean() / returns.std()) * (252 ** 0.5), 2) if returns.std() > 0 else 0,
        },
        "recent_prices": df["Close"].tail(30).round(2).to_dict(),
    }

    return stats


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ticker = sys.argv[1] if len(sys.argv) > 1 else "EMAAR.DU"
    market = sys.argv[2] if len(sys.argv) > 2 else "AE"

    print(f"\n📊 Testing: {ticker} / {market}")
    ctx = prepare_analysis_context(ticker, market, period="1y")
    import json
    print(json.dumps(ctx, indent=2, default=str))