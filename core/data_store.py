"""
data_store.py — EisaX Data Store
واجهة قراءة الداتا المحفوظة للـ Finance Agent
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from core.config import DATA_DIR as _cfg_data_dir
DATA_DIR = _cfg_data_dir


def get_price_history(
    ticker: str,
    market: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    يجيب Historical OHLCV لسهم معين

    Args:
        ticker: مثال "2222.SR", "COMI.CA", "EMAAR.DU"
        market: "SA", "EG", "AE"
        start: "2022-01-01" (اختياري)
        end: "2024-12-31" (اختياري)
        period: "1y", "6m", "3m", "1m", "ytd" (بديل عن start/end)

    Returns:
        DataFrame: Open, High, Low, Close, Volume
    """
    safe = ticker.replace(".", "_").replace("^", "IDX_")
    path = DATA_DIR / "historical" / market / f"{safe}.parquet"

    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # تطبيق الـ period لو موجود
    if period:
        end_dt = datetime.today()
        if period == "1y":
            start_dt = end_dt - timedelta(days=365)
        elif period == "6m":
            start_dt = end_dt - timedelta(days=180)
        elif period == "3m":
            start_dt = end_dt - timedelta(days=90)
        elif period == "1m":
            start_dt = end_dt - timedelta(days=30)
        elif period == "ytd":
            start_dt = datetime(end_dt.year, 1, 1)
        else:
            start_dt = end_dt - timedelta(days=365)
        df = df[df.index >= start_dt]

    else:
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]

    return df if not df.empty else None


def get_latest_price(ticker: str, market: str) -> Optional[dict]:
    """
    يجيب آخر سعر مسجّل للسهم

    Returns:
        dict: {"date": ..., "close": ..., "change_pct": ...}
    """
    df = get_price_history(ticker, market)
    if df is None or len(df) < 2:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    change_pct = ((last["Close"] - prev["Close"]) / prev["Close"]) * 100

    return {
        "ticker": ticker,
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "open": round(last["Open"], 2),
        "high": round(last["High"], 2),
        "low": round(last["Low"], 2),
        "close": round(last["Close"], 2),
        "volume": int(last["Volume"]),
        "change_pct": round(change_pct, 2),
    }


def get_market_summary(market: str) -> dict:
    """
    ملخص عن كل سوق — عدد الأسهم، آخر تحديث، إلخ
    """
    path = DATA_DIR / "historical" / market
    if not path.exists():
        return {"market": market, "files": 0, "rows": 0}

    files = list(path.glob("*.parquet"))
    total_rows = 0
    latest_date = None

    for f in files:
        try:
            df = pd.read_parquet(f)
            total_rows += len(df)
            if not df.empty:
                d = pd.to_datetime(df.index[-1])
                if latest_date is None or d > latest_date:
                    latest_date = d
        except Exception:
            continue

    return {
        "market": market,
        "tickers": len(files),
        "total_rows": total_rows,
        "last_updated": latest_date.strftime("%Y-%m-%d") if latest_date else None,
    }


def get_returns(ticker: str, market: str, period: str = "1y") -> Optional[pd.Series]:
    """يحسب الـ daily returns للاستخدام في Portfolio Optimizer"""
    df = get_price_history(ticker, market, period=period)
    if df is None:
        return None
    return df["Close"].pct_change().dropna()


def search_available_tickers(market: str) -> list[str]:
    """يرجع قائمة الـ tickers المتاحة محلياً"""
    path = DATA_DIR / "historical" / market
    if not path.exists():
        return []
    files = list(path.glob("*.parquet"))
    return [f.stem.replace("_", ".").replace("IDX.", "^") for f in files]


# ─── للـ Finance Agent ────────────────────────────────────────────────────────

def is_data_available(ticker: str, market: str) -> bool:
    """يتحقق لو عندنا داتا محلية للسهم ده"""
    safe = ticker.replace(".", "_").replace("^", "IDX_")
    path = DATA_DIR / "historical" / market / f"{safe}.parquet"
    return path.exists()


def get_multi_ticker_prices(tickers: list, market: str, period: str = "1y") -> pd.DataFrame:
    """
    يجيب Close prices لمجموعة أسهم — للـ Portfolio Optimizer
    Returns DataFrame: columns = tickers, index = dates
    """
    dfs = {}
    for t in tickers:
        df = get_price_history(t, market, period=period)
        if df is not None:
            dfs[t] = df["Close"]

    if not dfs:
        return pd.DataFrame()

    return pd.DataFrame(dfs).dropna()