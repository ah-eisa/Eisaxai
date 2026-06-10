"""
dfm_lookup.py — DFM Stock Name Recognition & Fundamentals
Converts Arabic/English company names to DFM tickers
and provides fundamentals context for the agent.
"""
import sqlite3
import logging
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

# Arabic + short name aliases → canonical DB name
NAME_ALIASES = {
    # Emaar
    "emaar": "Emaar Properties",
    "اعمار": "Emaar Properties",
    "إعمار": "Emaar Properties",
    "عمار": "Emaar Properties",
    "emaar properties": "Emaar Properties",
    "emaar dev": "Emaar Develop",
    "emaar develop": "Emaar Develop",
    # Emirates NBD
    "enbd": "Emirates NBD PJSC",
    "emirates nbd": "Emirates NBD PJSC",
    "النبد": "Emirates NBD PJSC",
    "بنك الامارات": "Emirates NBD PJSC",
    # DIB
    "dib": "Dubai Islamic Bank",
    "dubai islamic": "Dubai Islamic Bank",
    "بنك دبي الاسلامي": "Dubai Islamic Bank",
    "دبي الاسلامي": "Dubai Islamic Bank",
    # DEWA
    "dewa": "Dubai Electricity and Water",
    "ديوا": "Dubai Electricity and Water",
    "كهرباء دبي": "Dubai Electricity and Water",
    # Air Arabia
    "air arabia": "Air Arabia PJSC",
    "العربية للطيران": "Air Arabia PJSC",
    "العربية": "Air Arabia PJSC",
    # Aramex
    "aramex": "ARAMEX PJSC",
    "ارامكس": "ARAMEX PJSC",
    # Salik
    "salik": "Salik Company PJSC",
    "سالك": "Salik Company PJSC",
    # Talabat
    "talabat": "Talabat Holding",
    "طلبات": "Talabat Holding",
    # Tecom
    "tecom": "Tecom PJSC",
    "تيكوم": "Tecom PJSC",
    # DU telecom
    "du": "Emirate Integrated Telecom",
    "دو": "Emirate Integrated Telecom",
    # Tabreed
    "tabreed": "National Central Cooling",
    "تبريد": "National Central Cooling",
    # Amlak
    "amlak": "Amlak Finance",
    "املاك": "Amlak Finance",
    # Mashreq
    "mashreq": "Mashreqbank PSC",
    "مشرق": "Mashreqbank PSC",
    "mashreqbank": "Mashreqbank PSC",
    # SHUAA
    "shuaa": "SHUAA Capital PSC",
    "شعاع": "SHUAA Capital PSC",
    # Parkin
    "parkin": "Parkin Company PJSC",
    "باركن": "Parkin Company PJSC",
    # Spinneys
    "spinneys": "Spinneys 1961 Holding",
    "سبينيس": "Spinneys 1961 Holding",
    # ALEC
    "alec": "ALEC Holdings PJSC",
    # Dubai taxi
    "dubai taxi": "Dubai Taxi Company PJSC",
    "تاكسي دبي": "Dubai Taxi Company PJSC",
    # CBD
    "cbd": "Commercial Bank of Dubai",
    "بنك دبي التجاري": "Commercial Bank of Dubai",
    # Deyaar
    "deyaar": "Deyaar Development",
    "ديار": "Deyaar Development",
    # DFM exchange
    "dfm": "Dubai Financial Market",
    "سوق دبي المالي": "Dubai Financial Market",
    # Empower
    "empower": "Emirates Central Cooling Systems",
    "امباور": "Emirates Central Cooling Systems",
    # Taaleem
    "taaleem": "Taaleem Holdings",
    "تعليم": "Taaleem Holdings",
    # Al Ansari
    "al ansari": "Al Ansari Financial Services PJSC",
    "الانصاري": "Al Ansari Financial Services PJSC",
    # Amanat
    "amanat": "Amanat Holdings PJSC",
    "امانات": "Amanat Holdings PJSC",
    # Gulf Navigation
    "gulf navigation": "Gulf Navigation Hld",
    "الملاحة الخليجية": "Gulf Navigation Hld",
    # DAMAC
    "damac": "DAMAC Properties Dubai PJSC",
    "داماك": "DAMAC Properties Dubai PJSC",
    "damac properties": "DAMAC Properties Dubai PJSC",
    # Union Properties
    "union properties": "Union Properties",
    "الاتحاد للعقارات": "Union Properties",
}


def lookup_dfm(name: str) -> Optional[dict]:
    """
    Given any name/alias, return fundamentals dict or None.
    Tries: alias map → partial DB name match → ticker match
    """
    key = name.strip().lower()
    
    # Step 1: alias map
    canonical = NAME_ALIASES.get(key)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    row = None
    if canonical:
        row = conn.execute(
            "SELECT * FROM uae_fundamentals WHERE LOWER(name)=LOWER(?)", (canonical,)
        ).fetchone()
    
    # Step 2: partial name match
    if not row:
        row = conn.execute(
            "SELECT * FROM uae_fundamentals WHERE LOWER(name) LIKE LOWER(?)",
            (f"%{key}%",)
        ).fetchone()
    
    # Step 3: ticker match
    if not row:
        row = conn.execute(
            "SELECT * FROM uae_fundamentals WHERE LOWER(ticker) LIKE LOWER(?)",
            (f"%{key}%",)
        ).fetchone()
    
    conn.close()
    
    if not row:
        return None
    
    return dict(row)


def get_dfm_context(name: str) -> str:
    """
    Returns a formatted string for injection into agent prompt.
    Tries the full string first, then each word/phrase.
    Includes live price from cache (no blocking HTTP).
    """
    # Try full string
    d = lookup_dfm(name)

    # Try each word and 2-word combos from the message
    if not d:
        words = name.lower().split()
        for i in range(len(words)-1):
            d = lookup_dfm(f"{words[i]} {words[i+1]}")
            if d: break
    if not d:
        words = name.lower().split()
        for word in words:
            if len(word) > 3:
                d = lookup_dfm(word)
                if d: break
    if not d:
        return ""

    # ─── Try live price from cache (no HTTP blocking) ───
    price_line = ""
    try:
        if d.get("ticker"):
            from core.market_data_engine import get_latest_price, _load_cache
            # Only use cache — never trigger fresh HTTP fetch
            market = "AE" if d["ticker"].endswith(".DU") or d["ticker"].endswith(".AE") else "AE"
            df = _load_cache(d["ticker"], market)
            if df is not None and not df.empty:
                last_price = float(df["Close"].iloc[-1])
                prev_price = float(df["Close"].iloc[-2]) if len(df) > 1 else last_price
                chg_pct = ((last_price - prev_price) / prev_price * 100) if prev_price else 0
                last_date = str(df.index[-1])[:10]
                arrow = "📈" if chg_pct >= 0 else "📉"
                price_line = f"- **Live Price:** AED {last_price:.3f} ({'+' if chg_pct>=0 else ''}{chg_pct:.2f}%) {arrow} | *as of {last_date}*\n"
    except Exception as _pe:
        logger.debug(f"[DFM] Price from cache failed: {_pe}")
    
    # ── Technical Indicators ──────────────────────────────────────────
    tech_block = ""
    try:
        from core.market_data_engine import _load_cache, get_stock_data
        _df = _load_cache(d["ticker"], "AE")
        # لو الداتا قليلة جيب أكثر عشان SMA200
        if _df is None or len(_df) < 200:
            _df = get_stock_data(d["ticker"], "AE", period="2y")
        tech = calc_technicals(_df)
        if tech:
            tech_block = (
                f"\n**التحليل الفني:**\n"
                f"- **RSI (14):** {tech['rsi']} — {tech['rsi_signal']}\n"
                f"- **MACD:** {tech['macd']} — {tech['macd_signal']}\n"
                f"- **SMA50:** {tech['sma50']} | **SMA200:** {tech['sma200']}\n"
                f"- **الاتجاه:** {tech['trend']}\n"
            )
    except Exception as _te:
        pass

    lines = [
        f"## DFM Fundamentals: {d['name']}",
        price_line,
        f"- **Ticker:** {d['ticker'] or 'N/A'}",
        f"- **Market Cap:** AED {d['market_cap']}",
        f"- **Revenue:** AED {d['revenue']}",
        f"- **P/E Ratio:** {d['pe_ratio'] if d['pe_ratio'] else 'N/A'}",
        f"- **Beta:** {d['beta'] if d['beta'] else 'N/A'}",
        f"- **Avg Volume (3M):** {d['avg_vol_3m']}",
        f"- **Exchange:** Dubai Financial Market (DFM)",
        tech_block,
    ]
    return "\n".join(lines)


def screen_dfm(criterion: str = "low_pe", top_n: int = 10) -> list:
    """
    Screen DFM stocks by criterion.
    criterion: 'low_pe' | 'high_volume' | 'low_beta' | 'large_cap'
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    if criterion == "low_pe":
        rows = conn.execute("""
            SELECT * FROM uae_fundamentals 
            WHERE pe_ratio > 0 
            ORDER BY pe_ratio ASC LIMIT ?
        """, (top_n,)).fetchall()
    elif criterion == "high_pe":
        rows = conn.execute("""
            SELECT * FROM uae_fundamentals 
            WHERE pe_ratio > 0 AND pe_ratio < 500
            ORDER BY pe_ratio DESC LIMIT ?
        """, (top_n,)).fetchall()
    elif criterion == "low_beta":
        rows = conn.execute("""
            SELECT * FROM uae_fundamentals 
            WHERE beta IS NOT NULL
            ORDER BY ABS(beta) ASC LIMIT ?
        """, (top_n,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM uae_fundamentals LIMIT ?", (top_n,)
        ).fetchall()
    
    conn.close()
    return [dict(r) for r in rows]


def is_dfm_query(message: str) -> bool:
    """Quick check if message is about a DFM stock."""
    msg = message.lower()
    dfm_keywords = [
        "dfm", "dubai financial market", "سوق دبي",
        ".du", "emaar", "enbd", "dewa", "dib", "salik",
        "talabat", "aramex", "air arabia", "du telecom",
        "اعمار", "إعمار", "ديوا", "طلبات", "damac", "داماك"
    ]
    return any(kw in msg for kw in dfm_keywords)


if __name__ == "__main__":
    tests = ["emaar", "DEWA", "طلبات", "enbd", "salik", "aramex"]
    for t in tests:
        d = lookup_dfm(t)
        if d:
            print(f"✅ '{t}' → {d['name']} ({d['ticker']}) | PE={d['pe_ratio']} | Beta={d['beta']}")
        else:
            print(f"❌ '{t}' → not found")
    
    print("\n=== Top 5 Lowest P/E ===")
    for s in screen_dfm("low_pe", 5):
        print(f"  {s['name']:<40} PE={s['pe_ratio']}")


def get_dfm_price(ticker: str) -> dict:
    """
    Get live price for DFM/ADX stock using investing.com scraper.
    Returns dict with price, change, change_pct or empty dict.
    """
    try:
        from core.market_data_engine import _fetch_investing, UAE_INVESTING
        if ticker in UAE_INVESTING:
            df = _fetch_investing(ticker, UAE_INVESTING[ticker])
            if df is not None and not df.empty:
                last = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
                chg = last - prev
                chg_pct = (chg / prev * 100) if prev else 0
                return {
                    "price": round(last, 3),
                    "change": round(chg, 3),
                    "change_pct": round(chg_pct, 2),
                    "source": "investing.com"
                }
    except Exception as e:
        logger.warning(f"[DFM Price] {ticker} failed: {e}")
    return {}


def get_full_dfm_context(message: str) -> str:
    """
    Full context including live price if available.
    """
    base = get_dfm_context(message)
    if not base:
        return ""

    # Extract ticker from context
    import re
    m = re.search(r'\*\*Ticker:\*\* (\S+)', base)
    if m:
        ticker = m.group(1)
        price_data = get_dfm_price(ticker)
        if price_data:
            price_line = (
                f"- **Live Price:** AED {price_data['price']} "
                f"({'+' if price_data['change'] >= 0 else ''}{price_data['change_pct']}%)"
            )
            base = base.replace("## DFM Fundamentals:", f"## DFM Live Data:\n{price_line}\n##")

    return base
