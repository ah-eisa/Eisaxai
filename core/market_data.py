"""
market_data.py — EisaX Resilient Market Data Layer
=====================================================
get_realtime_quote() implements a TRUE waterfall — never returns 0 price:

  1. yfinance fast_info      (US/Global — free, fast)
  2. market_data_engine      (UAE/Gulf — Investing.com scraper)
  3. StockAnalysis scraper   (UAE/Saudi/Egypt — web scraper)
  4. RapidAPI / DB cache     (paid fallback + last known price)
  5. Serper web search       (last resort — extract price from Google)
"""

import os, logging
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ─── Ticker aliases — smart self-healing before waterfall ───────────────────
_TICKER_ALIASES = {
    # Forex/spot → ETF equivalent
    "XAUUSD":    "GC=F",    # Gold spot → Gold Futures (yfinance works)
    "XAU/USD":   "GC=F",
    "GOLD":      "GC=F",
    "XAGUSD":    "SI=F",    # Silver spot → Silver Futures
    "XAG/USD":   "SI=F",
    "SILVER":    "SI=F",
    "XTIUSD":    "CL=F",    # WTI Oil spot → WTI Futures
    "OIL":       "CL=F",
    "WTIUSD":    "CL=F",
    "BRENTUSD":  "BZ=F",    # Brent Oil spot → Brent Futures
    # UAE aliases
    "ETISALAT":    "EAND.AE",
    "ETISALAT.AE": "EAND.AE",
    "ETISALAT.DU": "EAND.DU",
    # Saudi aliases
    "ARAMCO":    "2222.SR",
    "ADNOC":     "ADNOCGAS.AE",
    # BTC/crypto — map to ETF equivalent
    "BTCUSD":    "IBIT",
    "BTC/USD":   "IBIT",
    "BITCOIN":   "IBIT",
    "ETHUSD":    "ETHA",
    "ETH/USD":   "ETHA",
}

# ─── 1. Core waterfall price fetch ──────────────────────────────────────────
def get_realtime_quote(ticker: str) -> dict:
    """
    Resilient price fetch — tries every source before giving up.
    Returns: {"ticker", "price", "change_pct", "source"}
    Never returns price=0 if ANY source has data.
    """
    _orig = ticker.upper()
    # ── Self-healing: resolve aliases before anything else ─────────────────
    _t = _TICKER_ALIASES.get(_orig, _orig)
    if _t != _orig:
        logger.info(f"[Quote/alias] {_orig} → {_t}")

    _is_local = _t.endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA", ".BH", ".MA", ".TN"))
    try:
        from core.price_cache import get as _pc_get, set as _pc_set
        _cached_price = _pc_get(_orig)
        if _cached_price:
            return {"ticker": _orig, "price": _cached_price, "change_pct": 0.0, "source": "cache"}
    except Exception:
        _pc_set = None
    _mkt_map  = {".AE":"AE",".DU":"AE",".SR":"SA",".CA":"EG",".KW":"KW",".QA":"QA",".BH":"BH",".MA":"MA",".TN":"TN"}
    _mkt = next((v for k,v in _mkt_map.items() if _t.endswith(k)), "US")

    price = 0.0
    change_pct = 0.0
    source = "unknown"

    # ── Source 0: TradingView pipeline cache — authoritative for ALL tickers ──
    # Policy: TV is the source of truth across markets. Yahoo, Investing.com,
    # StockAnalysis are labeled fallbacks only for fields TV doesn't expose
    # or tickers TV cache doesn't carry.
    try:
        from core.data_layer import market_cache_adapter as _mca
        _tv_market_map = {"AE": "uae", "SA": "ksa", "EG": "egypt",
                          "KW": "kuwait", "QA": "qatar", "BH": "bahrain",
                          "MA": "morocco", "TN": "tunisia"}
        if _mkt in _tv_market_map:
            _tv_markets = [_tv_market_map[_mkt]]
        elif _t.endswith("=F"):
            _tv_markets = ["commodities"]
        else:
            # US equities (and unknown markets) — try america first, then crypto
            _tv_markets = ["america", "crypto"]
        _bare = _t.split(".")[0]
        for _tv_mkt in _tv_markets:
            _df = _mca.get_latest_snapshot(_tv_mkt)
            if _df is None or _df.empty or "ticker" not in _df.columns:
                continue
            _col = _df["ticker"].astype(str).str.upper()
            _match = _df[
                _col.str.endswith(":" + _bare)
                | (_col == _t)
                | (_col == _bare)
            ]
            if _match.empty:
                continue
            _row = _match.iloc[0]
            _tv_close = float(_row.get("close") or 0)
            _tv_chg = float(_row.get("change") or 0)
            if _tv_close > 0:
                price = _tv_close
                change_pct = _tv_chg
                source = "tradingview_cache"
                try:
                    if _pc_set:
                        _pc_set(_orig, price)
                except Exception:
                    pass
                logger.info(f"[Quote/TV] {_t}: {price} (market={_tv_mkt}, {source})")
                break
    except Exception as e:
        logger.debug(f"[Quote/TV] {_t} failed: {e}")

    # ── Short-circuit when TV provided a price ─────────────────────────────
    # No need to chain through YF/Investing/SA — TV is authoritative.
    if price > 0 and source == "tradingview_cache":
        return {
            "ticker":     _orig,
            "price":      round(price, 4),
            "change_pct": round(float(change_pct), 2),
            "source":     source,
        }

    # ── Source 1: yfinance (fallback when TV cache misses) ─────────────────
    try:
        import warnings; warnings.filterwarnings("ignore")
        tk = yf.Ticker(_t)
        fi = tk.fast_info
        info = tk.info or {}
        last_price = (
            getattr(fi, 'last_price', None) or
            info.get("regularMarketPrice") or
            info.get("currentPrice") or 0
        )
        prev_close = (
            getattr(fi, 'previous_close', None) or
            info.get("regularMarketPreviousClose") or
            info.get("previousClose") or 0
        )
        last_price = float(last_price or 0)
        prev_close = float(prev_close or 0)
        if last_price > 0:
            price = last_price
            change_pct = ((last_price - prev_close) / prev_close * 100) if prev_close else 0
            source = "yfinance"
            try:
                if _pc_set: _pc_set(_orig, price)
            except Exception: pass
            logger.debug(f"[Quote/yf] {_t}: {price}")
    except Exception as e:
        logger.debug(f"[Quote/yf] {_t} failed: {e}")

    # ── Source 2: market_data_engine (Investing.com scraper — UAE/Gulf) ───
    if price <= 0 and _is_local:
        try:
            from core.market_data_engine import get_latest_price as _glp
            q = _glp(_t, _mkt)
            if q and float(q.get("close") or 0) > 0:
                price      = float(q["close"])
                change_pct = float(q.get("change_pct", 0))
                source     = "investing.com"
                logger.info(f"[Quote/MDE] {_t}: {price} ({source})")
        except Exception as e:
            logger.debug(f"[Quote/MDE] {_t} failed: {e}")

    # ── Source 3: StockAnalysis scraper ────────────────────────────────────
    if price <= 0:
        try:
            from core.realtime_data import _stockanalysis_uae as _sa_fetch
            sa = _sa_fetch(_t)
            if sa and float(sa.get("price") or 0) > 0:
                price  = float(sa["price"])
                source = "stockanalysis"
                logger.info(f"[Quote/SA] {_t}: {price}")
        except Exception as e:
            logger.debug(f"[Quote/SA] {_t} failed: {e}")

    # ── Source 4: RapidAPI (Investing.com API) ─────────────────────────────
    if price <= 0:
        try:
            from core.rapidapi_client import get_quote as _rapi_q
            rq = _rapi_q(_t, _mkt)
            if rq and float(rq.get("price") or 0) > 0:
                price      = float(rq["price"])
                change_pct = float(rq.get("change_pct", 0))
                source     = "rapidapi"
                logger.info(f"[Quote/RapidAPI] {_t}: {price}")
        except Exception as e:
            logger.debug(f"[Quote/RapidAPI] {_t} failed: {e}")

    # ── Source 5: DB last-known price ──────────────────────────────────────
    if price <= 0:
        try:
            import sqlite3
            from core.config import CORE_DB as _cfg_core_db
            conn = sqlite3.connect(str(_cfg_core_db))
            # Search both original and resolved ticker
            row = conn.execute(
                "SELECT price FROM uae_fundamentals WHERE ticker IN (?,?) AND price>0 LIMIT 1",
                (_t, _orig)
            ).fetchone()
            conn.close()
            if row and float(row[0] or 0) > 0:
                price  = float(row[0])
                source = "db_cache"
                logger.info(f"[Quote/DB] {_t}: {price} (last known)")
        except Exception as e:
            logger.debug(f"[Quote/DB] {_t} failed: {e}")

    # ── Source 6: Serper web search (last resort) ──────────────────────────
    if price <= 0:
        try:
            import requests, re
            serper_key = os.getenv("SERPER_API_KEY", "")
            if serper_key:
                # Use original requested name for better search results
                search_q = _orig.split(".")[0].replace("=F","")
                q_suffix = {"AE":"UAE ADX stock","SA":"Saudi Tadawul stock","EG":"Egypt EGX stock"}.get(_mkt,"stock")
                r = requests.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                    json={"q": f"{search_q} {q_suffix} price today", "num": 3},
                    timeout=8
                )
                if r.status_code == 200:
                    text = str(r.json())
                    # Try to extract price pattern like "$19.52" or "19.52 AED"
                    m = re.search(r'[\$£€]?\s*(\d{1,6}(?:[.,]\d{1,4})?)\s*(?:USD|AED|SAR|EGP|$)', text)
                    if m:
                        p = float(m.group(1).replace(",", ""))
                        if 0.01 < p < 1_000_000:
                            price  = p
                            source = "serper_search"
                            logger.info(f"[Quote/Serper] {_orig}: extracted price {price}")
        except Exception as e:
            logger.debug(f"[Quote/Serper] {_orig} failed: {e}")

    if price <= 0:
        logger.warning(f"[Quote] ALL sources failed for {_t} — returning price=0")

    # Mark fallback explicitly for GCC tickers when TV missed
    if _is_local and source not in ("tradingview_cache", "cache"):
        source = f"{source} (fallback)"
        logger.warning(
            f"[Quote/GCCFallback] {_orig}: TV cache missed -> using {source}"
        )

    return {
        "ticker":     _orig,   # always return original ticker requested
        "price":      round(price, 4),
        "change_pct": round(float(change_pct), 2),
        "source":     source,
    }


# ─── 2. Full Stock Profile ────────────────────────────────────────────────────
def get_full_stock_profile(ticker: str) -> dict:
    """جمع كل البيانات الحقيقية لسهم واحد — resilient waterfall"""
    quote      = get_realtime_quote(ticker)
    sentiment  = get_news_sentiment_av(ticker)
    macro      = get_macro_context()
    rf_rate    = macro.get("treasury_10y", {}).get("value", 4.5)
    if rf_rate: rf_rate = rf_rate / 100

    # Fundamentals via yfinance (+ store raw info for ETF detection)
    fundamentals = {}
    _yf_raw = {}
    try:
        import warnings; warnings.filterwarnings("ignore")
        t    = yf.Ticker(ticker)
        fi   = t.fast_info
        info = t.info or {}
        _yf_raw = info   # store for ETF detection
        fundamentals = {
            "year_high":          round(float(getattr(fi,'year_high',0) or 0), 2),
            "year_low":           round(float(getattr(fi,'year_low',0) or 0), 2),
            "market_cap":         getattr(fi, 'market_cap', None),
            "sma50":              round(float(getattr(fi,'fifty_day_average',0) or 0), 2),
            "sma200":             round(float(getattr(fi,'two_hundred_day_average',0) or 0), 2),
            "beta":               info.get("beta"),
            "pe_ttm":             info.get("trailingPE"),
            "pe_forward":         info.get("forwardPE"),
            "ps_ratio":           info.get("priceToSalesTrailing12Months"),
            "pb_ratio":           info.get("priceToBook"),
            "ev_ebitda":          info.get("enterpriseToEbitda"),
            "gross_margin":       info.get("grossMargins"),
            "operating_margin":   info.get("operatingMargins"),
            "roe":                info.get("returnOnEquity"),
            "roa":                info.get("returnOnAssets"),
            "revenue_growth":     info.get("revenueGrowth"),
            "earnings_growth":    info.get("earningsGrowth"),
            "dividend_yield":     info.get("dividendYield"),
            "short_ratio":        info.get("shortRatio"),
            "analyst_target":     info.get("targetMeanPrice"),
            "analyst_low":        info.get("targetLowPrice"),
            "analyst_high":       info.get("targetHighPrice"),
            "recommendation":     info.get("recommendationKey"),
            "num_analysts":       info.get("numberOfAnalystOpinions"),
            "sector":             info.get("sector"),
            "industry":           info.get("industry"),
            "employees":          info.get("fullTimeEmployees"),
            "cash":               info.get("totalCash"),
            "debt":               info.get("totalDebt"),
            "current_ratio":      info.get("currentRatio"),
            "next_earnings":      info.get("earningsTimestampStart"),
            # ETF fields
            "quoteType":          info.get("quoteType"),
            "fundFamily":         info.get("fundFamily"),
            "category":           info.get("category"),
            "annualReportExpenseRatio": info.get("annualReportExpenseRatio"),
            "totalAssets":        info.get("totalAssets"),
            "yield":              info.get("yield"),
        }
    except Exception as e:
        logger.debug(f"[Profile/yf] {ticker}: {e}")

    return {
        "quote":        quote,
        "sentiment":    sentiment,
        "macro":        macro,
        "rf_rate":      rf_rate,
        "fundamentals": fundamentals,
        "_yf_raw":      _yf_raw,   # raw yfinance info for ETF detection
        "timestamp":    __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
    }


# ─── 3. News Sentiment ────────────────────────────────────────────────────────
def get_news_sentiment_av(ticker: str) -> dict:
    try:
        import requests, numpy as np
        key = os.getenv("ALPHA_VANTAGE_KEY", "")
        if not key:
            return {"sentiment": "Neutral ➡️", "score": 0, "news": []}
        url  = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={key}&limit=5"
        feed = requests.get(url, timeout=10).json().get("feed", [])
        scores, news = [], []
        for item in feed[:5]:
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker") == ticker:
                    scores.append(float(ts.get("ticker_sentiment_score", 0)))
            news.append({"title": item.get("title",""), "source": item.get("source","")})
        avg  = float(np.mean(scores)) if scores else 0
        sent = "Bullish 📈" if avg > 0.15 else "Bearish 📉" if avg < -0.15 else "Neutral ➡️"
        return {"sentiment": sent, "score": round(avg, 3), "news": news}
    except Exception:
        return {"sentiment": "Neutral ➡️", "score": 0, "news": []}


# ─── 4. Macro Context (FRED) ─────────────────────────────────────────────────
def get_macro_context() -> dict:
    try:
        import requests
        key = os.getenv("FRED_API_KEY", "")
        result = {}
        for sid in ["DGS10", "FEDFUNDS", "UNRATE"]:
            url = (f"https://api.stlouisfed.org/fred/series/observations"
                   f"?series_id={sid}&api_key={key}&file_type=json&limit=1&sort_order=desc")
            obs = requests.get(url, timeout=8).json().get("observations", [])
            if obs:
                names = {"DGS10":"treasury_10y","FEDFUNDS":"fed_funds","UNRATE":"unemployment"}
                result[names[sid]] = {"series": sid, "value": float(obs[0].get("value",0)), "date": obs[0].get("date")}
        try:
            url = (f"https://api.stlouisfed.org/fred/series/observations"
                   f"?series_id=CPIAUCSL&api_key={key}&file_type=json&limit=13&sort_order=desc")
            obs = requests.get(url, timeout=8).json().get("observations", [])
            if len(obs) >= 13:
                latest  = float(obs[0].get("value", 0))
                year_ago = float(obs[12].get("value", 1))
                result["inflation"] = {"series":"CPIAUCSL","value":round((latest-year_ago)/year_ago*100,2),"date":obs[0].get("date")}
        except Exception: pass
        try:
            url = (f"https://api.stlouisfed.org/fred/series/observations"
                   f"?series_id=A191RL1Q225SBEA&api_key={key}&file_type=json&limit=1&sort_order=desc")
            obs = requests.get(url, timeout=8).json().get("observations", [])
            if obs:
                result["gdp_growth"] = {"series":"GDP","value":float(obs[0].get("value",0)),"date":obs[0].get("date")}
        except Exception: pass
        return result
    except Exception:
        return {}


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tests = ["EAND.AE", "ETISALAT.AE", "MSFT", "2222.SR", "COMI.CA", "GLD", "XAUUSD", "GOLD", "OIL"]
    for t in tests:
        q = get_realtime_quote(t)
        print(f"{t}: ${q['price']} ({q['change_pct']:+.2f}%) [{q['source']}]")
