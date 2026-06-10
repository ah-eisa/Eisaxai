"""
egx_lookup.py — EGX Stock Fundamentals
مثل dfm_lookup بس للبورصة المصرية
"""
import sqlite3, logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def calc_technicals(df) -> dict:
    """حساب RSI + MACD + SMA50 + SMA200 من OHLCV DataFrame"""
    if df is None or len(df) < 20:
        return {}
    try:
        import pandas as pd, numpy as np
        close = df["Close"].dropna()

        # SMA
        sma50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else None
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        price  = float(close.iloc[-1])

        # RSI (14)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, 1e-10)
        rsi   = float(100 - (100 / (1 + rs.iloc[-1])))

        # MACD (12,26,9)
        ema12  = close.ewm(span=12).mean()
        ema26  = close.ewm(span=26).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        macd_v = float(macd.iloc[-1])
        sig_v  = float(signal.iloc[-1])

        # Signals
        rsi_signal  = "Oversold 🟢"  if rsi < 30 else "Overbought 🔴" if rsi > 70 else "Neutral ⚪"
        macd_signal = "Bullish 🟢"   if macd_v > sig_v else "Bearish 🔴"
        trend       = "Above SMA50 📈" if sma50 and price > sma50 else "Below SMA50 📉"

        return {
            "rsi":         round(rsi, 1),
            "rsi_signal":  rsi_signal,
            "macd":        round(macd_v, 3),
            "macd_signal": macd_signal,
            "sma50":       round(sma50, 3)  if sma50  else None,
            "sma200":      round(sma200, 3) if sma200 else None,
            "trend":       trend,
            "price":       round(price, 3),
        }
    except Exception as e:
        return {}

from core.config import CORE_DB as _cfg_core_db
DB_PATH = _cfg_core_db

NAME_ALIASES = {
    "comi": "Commercial International Bank",
    "cib": "Commercial International Bank",
    "كوميرشيال": "Commercial International Bank",
    "بنك كوميرشيال": "Commercial International Bank",
    "hrho": "Hermes Holding",
    "هيرمس": "Hermes Holding",
    "tmgh": "Talaat Moustafa Group",
    "طلعت مصطفى": "Talaat Moustafa Group",
    "etel": "Telecom Egypt",
    "telecom egypt": "Telecom Egypt",
    "المصرية للاتصالات": "Telecom Egypt",
    "swdy": "El Sewedy Electric",
    "السويدي": "El Sewedy Electric",
    "sewedy": "El Sewedy Electric",
    "mfpc": "Misr Fertilizers",
    "موبكو": "Misr Fertilizers",
    "ocdi": "Orascom Construction",
    "اوراسكوم": "Orascom Construction",
    "phdc": "Palm Hills",
    "بالم هيلز": "Palm Hills",
    "jufo": "Juhayna Food",
    "جهينة": "Juhayna Food",
    "ekho": "Eastern Company",
    "الشرقية للدخان": "Eastern Company",
    "masr": "Madinet Nasr",
    "مدينة نصر": "Madinet Nasr",
}

def lookup_egx(name: str) -> Optional[dict]:
    key = name.strip().lower().replace(".ca", "")
    canonical = NAME_ALIASES.get(key)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = None
    if canonical:
        row = conn.execute(
            "SELECT * FROM egx_fundamentals WHERE LOWER(name)=LOWER(?)", (canonical,)
        ).fetchone()
    if not row:
        row = conn.execute(
            "SELECT * FROM egx_fundamentals WHERE LOWER(name) LIKE LOWER(?) OR LOWER(ticker) LIKE LOWER(?)",
            (f"%{key}%", f"%{key}%")
        ).fetchone()
    conn.close()
    return dict(row) if row else None

def get_egx_context(name: str) -> str:
    d = lookup_egx(name)
    if not d:
        words = name.lower().split()
        for w in words:
            if len(w) > 3:
                d = lookup_egx(w)
                if d: break
    if not d:
        return ""

    price_line = ""
    try:
        import sys; sys.path.insert(0, '/home/ubuntu/investwise')
        from core.market_data_engine import _load_cache
        df = _load_cache(d["ticker"], "EG")
        if df is not None and not df.empty:
            last  = float(df["Close"].iloc[-1])
            prev  = float(df["Close"].iloc[-2]) if len(df) > 1 else last
            chg   = ((last - prev) / prev * 100) if prev else 0
            date  = str(df.index[-1])[:10]
            arrow = "📈" if chg >= 0 else "📉"
            price_line = f"- **Live Price:** EGP {last:.2f} ({'+' if chg>=0 else ''}{chg:.2f}%) {arrow} | *as of {date}*\n"
    except Exception as e:
        logger.debug(f"EGX price error: {e}")

    tech_block = ""
    try:
        import sys; sys.path.insert(0, '/home/ubuntu/investwise')
        from core.market_data_engine import _load_cache, get_stock_data
        _df = _load_cache(d["ticker"], "EG")
        if _df is None or len(_df) < 200:
            _df = get_stock_data(d["ticker"], "EG", period="2y")
        tech = calc_technicals(_df)
        if tech:
            tech_block = (
                f"\n**التحليل الفني:**\n"
                f"- **RSI (14):** {tech['rsi']} — {tech['rsi_signal']}\n"
                f"- **MACD:** {tech['macd']} — {tech['macd_signal']}\n"
                f"- **SMA50:** {tech['sma50']} | **SMA200:** {tech['sma200']}\n"
                f"- **الاتجاه:** {tech['trend']}\n"
            )
    except: pass

    return "\n".join([
        f"## EGX Fundamentals: {d['name']}",
        price_line,
        f"- **Ticker:** {d['ticker']}",
        f"- **Market Cap:** EGP {d['market_cap']}",
        f"- **Revenue:** EGP {d['revenue']}",
        f"- **P/E Ratio:** {d['pe_ratio'] if d['pe_ratio'] else 'N/A'}",
        f"- **Beta:** {d['beta'] if d['beta'] else 'N/A'}",
        f"- **Avg Volume (3M):** {d['avg_vol_3m']}",
        f"- **Exchange:** Egyptian Exchange (EGX)",
        tech_block,
    ])

def is_egx_query(message: str) -> bool:
    msg = message.lower()
    return any(k in msg for k in [
        ".ca", "egx", "بورصة مصر", "comi", "cib", "hrho", "tmgh",
        "هيرمس", "طلعت مصطفى", "كوميرشيال", "etel", "swdy", "السويدي",
        "موبكو", "اوراسكوم", "بالم هيلز", "جهينة"
    ])
