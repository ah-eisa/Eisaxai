"""
rapidapi_client.py — EisaX RapidAPI Data Client
=================================================
Uses: investing-com-ultimate-api.p.rapidapi.com (الأقوى لـ UAE + Egypt)
Covers: ADX / DFM / EGX — أي ticker بدون ID investing.com مسبق

Why this API?
  ✅ investing.com coverage شامل لـ UAE (DFM/ADX) + Egypt (EGX)
  ✅ لا يحتاج numeric ID — بيقبل ticker + country
  ✅ Historical + Quote + Fundamentals في نفس الـ endpoint
  ✅ أكثر استقراراً من cloudscraper
"""

import os
import http.client
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

RAPIDAPI_KEY  = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "investing-com-ultimate-api.p.rapidapi.com"

# ─── Country mapping ───────────────────────────────────────────────────────────
MARKET_TO_COUNTRY = {
    "AE": "dubai",
    "EG": "egypt",
    "SA": "saudi arabia",
    "KW": "kuwait",
    "QA": "qatar",
}

# ─── Ticker normalizer (يشيل الـ suffix) ──────────────────────────────────────
def _clean_ticker(ticker: str) -> str:
    """DAMAC.DU → DAMAC | COMI.CA → COMI | 2222.SR → 2222"""
    return ticker.split(".")[0]


def _make_request(path: str) -> Optional[dict]:
    """HTTP GET wrapper مع error handling"""
    try:
        conn = http.client.HTTPSConnection(RAPIDAPI_HOST, timeout=15)
        headers = {
            "x-rapidapi-key":  RAPIDAPI_KEY,
            "x-rapidapi-host": RAPIDAPI_HOST,
            "Content-Type":    "application/json",
        }
        conn.request("GET", path, headers=headers)
        res  = conn.getresponse()
        body = res.read().decode("utf-8")
        conn.close()

        if res.status != 200:
            logger.warning(f"RapidAPI HTTP {res.status} for {path}: {body[:200]}")
            return None

        return json.loads(body)

    except Exception as e:
        logger.error(f"RapidAPI request error: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
#  📊 Quote — آخر سعر + بيانات فورية
# ════════════════════════════════════════════════════════════════════════════════

def get_quote(ticker: str, market: str) -> Optional[dict]:
    """
    جيب Quote للسهم من RapidAPI investing.com
    
    Returns dict مع: price, change_pct, volume, market_cap, pe, eps, 52w_high/low
    """
    clean   = _clean_ticker(ticker)
    country = MARKET_TO_COUNTRY.get(market, "")
    if not country:
        logger.warning(f"Unknown market: {market}")
        return None

    path = f"/stocks/quote?ticker={clean}&country={country.replace(' ', '%20')}"
    data = _make_request(path)

    if not data:
        return None

    try:
        currency_map = {"AE": "AED", "EG": "EGP", "SA": "SAR", "KW": "KWF", "QA": "QAR"}
        return {
            "ticker":     ticker,
            "market":     market,
            "name":       data.get("name", ticker),
            "price":      float(data.get("price", 0) or 0),
            "change_pct": float(data.get("changePercent", 0) or 0),
            "change":     float(data.get("change", 0) or 0),
            "volume":     int(data.get("volume", 0) or 0),
            "market_cap": data.get("marketCap"),
            "pe":         data.get("pe"),
            "eps":        data.get("eps"),
            "high_52w":   data.get("high52Week"),
            "low_52w":    data.get("low52Week"),
            "open":       float(data.get("open", 0) or 0),
            "high":       float(data.get("high", 0) or 0),
            "low":        float(data.get("low", 0) or 0),
            "prev_close": float(data.get("previousClose", 0) or 0),
            "currency":   currency_map.get(market, "AED"),
            "source":     "rapidapi_investing",
            "timestamp":  datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Quote parse error for {ticker}: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
#  📈 Historical OHLCV — تاريخ الأسعار
# ════════════════════════════════════════════════════════════════════════════════

def get_historical(
    ticker: str,
    market: str,
    start: str = "2018-01-01",
    end:   str  = None,
) -> Optional[pd.DataFrame]:
    """
    جيب تاريخ الأسعار OHLCV من RapidAPI investing.com
    
    Returns: DataFrame مفهرس بالتاريخ [Open, High, Low, Close, Volume]
    """
    clean   = _clean_ticker(ticker)
    country = MARKET_TO_COUNTRY.get(market, "")
    if not country:
        return None

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    path = (
        f"/stocks/history?ticker={clean}"
        f"&country={country.replace(' ', '%20')}"
        f"&start_date={start}"
        f"&end_date={end}"
        f"&interval=Daily"
    )

    data = _make_request(path)
    if not data:
        return None

    # الـ response ممكن يكون list مباشرة أو dict فيه key "data"
    rows = data if isinstance(data, list) else data.get("data", data.get("history", []))

    if not rows:
        logger.warning(f"No historical rows for {ticker}")
        return None

    records = []
    for row in rows:
        try:
            records.append({
                "Date":   pd.to_datetime(row.get("date", row.get("time", ""))),
                "Open":   float(row.get("open",  0) or 0),
                "High":   float(row.get("high",  0) or 0),
                "Low":    float(row.get("low",   0) or 0),
                "Close":  float(row.get("close", 0) or 0),
                "Volume": int(row.get("volume", 0)  or 0),
            })
        except Exception:
            continue

    if not records:
        return None

    df = (
        pd.DataFrame(records)
        .set_index("Date")
        .sort_index()
    )
    df = df[~df.index.duplicated(keep="last")]
    df = df[df["Close"] > 0]           # شيل الصفوف الفارغة
    return df if not df.empty else None


# ════════════════════════════════════════════════════════════════════════════════
#  📋 Fundamentals — بيانات أساسية
# ════════════════════════════════════════════════════════════════════════════════

def get_fundamentals(ticker: str, market: str) -> Optional[dict]:
    """
    جيب البيانات الأساسية: Revenue, EPS, Margins, ROE, Debt, etc.
    """
    clean   = _clean_ticker(ticker)
    country = MARKET_TO_COUNTRY.get(market, "")
    if not country:
        return None

    path = f"/stocks/profile?ticker={clean}&country={country.replace(' ', '%20')}"
    data = _make_request(path)

    if not data:
        return None

    try:
        return {
            "ticker":        ticker,
            "market":        market,
            "revenue":       data.get("revenue"),
            "net_income":    data.get("netIncome"),
            "eps":           data.get("eps"),
            "pe":            data.get("pe"),
            "forward_pe":    data.get("forwardPe"),
            "pb":            data.get("pb"),
            "ps":            data.get("ps"),
            "roe":           data.get("roe"),
            "debt_equity":   data.get("debtEquity"),
            "gross_margin":  data.get("grossMargin"),
            "net_margin":    data.get("netMargin"),
            "dividend_yield":data.get("dividendYield"),
            "beta":          data.get("beta"),
            "sector":        data.get("sector"),
            "industry":      data.get("industry"),
            "description":   data.get("description"),
            "source":        "rapidapi_investing",
        }
    except Exception as e:
        logger.error(f"Fundamentals parse error for {ticker}: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
#  🔍 Search — ابحث عن ticker بالاسم
# ════════════════════════════════════════════════════════════════════════════════

def search_ticker(name: str, market: str = "AE") -> Optional[list]:
    """
    ابحث عن ticker بالاسم — مفيد لتعريف الأسهم الجديدة
    
    Example: search_ticker("DAMAC", "AE") → [{"ticker": "DAMAC", "name": ...}]
    """
    country = MARKET_TO_COUNTRY.get(market, "")
    path    = f"/search?query={name.replace(' ', '%20')}&country={country.replace(' ', '%20')}"
    data    = _make_request(path)

    if not data:
        return None

    results = data if isinstance(data, list) else data.get("results", [])
    return results[:10]  # أول 10 نتائج


# ════════════════════════════════════════════════════════════════════════════════
#  🛡️ Smart Fetch — wrapper موحد مع fallback
# ════════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv_with_fallback(
    ticker: str,
    market: str,
    start:  str = "2018-01-01",
) -> Optional[pd.DataFrame]:
    """
    الـ entry point الرئيسي — جيب OHLCV مع retry وlogging
    """
    logger.info(f"RapidAPI fetch: {ticker} ({market})")
    df = get_historical(ticker, market, start=start)

    if df is not None and not df.empty:
        logger.info(f"✅ RapidAPI got {len(df)} rows for {ticker}")
        return df

    logger.warning(f"❌ RapidAPI: no historical data for {ticker}")
    return None
