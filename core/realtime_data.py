"""
EisaX Real-Time Data Module
Sources: FMP, Finnhub, NewsAPI, CoinGecko, FRED, StockData
+ Local market support (Saudi, Egypt, UAE via yfinance)
"""
import os, requests, logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv("/home/ubuntu/investwise/.env")
logger = logging.getLogger(__name__)

FINNHUB_KEY  = os.getenv("FINNHUB_API_KEY", "")
NEWS_KEY     = os.getenv("NEWS_API_KEY", "")
COINGECKO_KEY= os.getenv("COINGECKO_API_KEY", "")
FMP_KEY      = os.getenv("FMP_API_KEY", "")
FRED_KEY     = os.getenv("FRED_API_KEY", "")
STOCKDATA_KEY= os.getenv("STOCKDATA_TOKEN", "")

# ═══════════════════════════════════════════════════════════════
# LOCAL MARKET DETECTION
# ═══════════════════════════════════════════════════════════════
LOCAL_SUFFIXES = (".SR", ".CA", ".AE", ".DU")

def _is_local_ticker(ticker: str) -> bool:
    """Check if ticker is from a local Arab market (not supported by US APIs)."""
    t = (ticker or "").upper()
    return any(t.endswith(s) for s in LOCAL_SUFFIXES)

def _get_local_news_query(ticker: str) -> str:
    """Build a better news search query for local tickers."""
    t = (ticker or "").upper()

    # ── Sector-enriched queries for energy tickers ──────────────────────────
    _energy_keywords = {
        "ADNOCGAS": "ADNOC Gas LNG oil energy UAE",
        "ADNOCDIST": "ADNOC Distribution fuel UAE oil",
        "TAQA": "TAQA Abu Dhabi energy utilities",
        "2222":  "Saudi Aramco oil crude production OPEC",
        "ARAMCO": "Saudi Aramco oil crude OPEC production",
        "DEWA":  "DEWA Dubai electricity water utilities",
        "DANA":  "Dana Gas Egypt Kurdistan gas",
    }
    for key, query in _energy_keywords.items():
        if key in t:
            return query

    # ── General: resolve company name from ticker resolver ──────────────────
    try:
        from core.ticker_resolver import TickerResolver
        resolver = TickerResolver()
        info = resolver.get_ticker_info(ticker.upper())
        if info:
            name = info.get("name_en", ticker)
            market_map = {
                "saudi": "Saudi Arabia Tadawul",
                "egypt": "Egypt EGX",
                "uae": "UAE ADX DFM",
                "kuwait": "Kuwait Stock Exchange",
                "qatar": "Qatar Exchange QSE",
            }
            market = info.get("market", "")
            context = market_map.get(market, "")
            sector = info.get("sector", "")
            # Add sector for richer search signal
            if "energy" in sector.lower() or "oil" in sector.lower():
                return f"{name} oil energy {context}"
            elif "bank" in sector.lower() or "financ" in sector.lower():
                return f"{name} bank earnings {context}"
            return f"{name} {context} stock"
    except Exception:
        pass
    return f"{ticker} stock"


def get_live_news(ticker: str, company_name: str = "", limit: int = 5) -> list:
    """Latest news - FMP first, Finnhub fallback, NewsAPI last resort"""
    
    # ── LOCAL MARKET: Skip FMP/Finnhub, go straight to NewsAPI ──
    if _is_local_ticker(ticker):
        return _get_local_news(ticker, company_name, limit)
    
    # 1. FMP - best for stock news (real-time)
    try:
        url = f"https://financialmodelingprep.com/api/v3/stock_news"
        params = {"tickers": ticker, "limit": limit, "apikey": FMP_KEY}
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        if isinstance(data, list) and data:
            return [{
                "headline": d.get("title", ""),
                "source": d.get("site", ""),
                "url": d.get("url", ""),
                "datetime": d.get("publishedDate", "")[:10]
            } for d in data[:limit] if d.get("title")]
    except Exception as e:
        logger.warning(f"FMP news failed: {e}")

    # 2. Finnhub fallback
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        url = "https://finnhub.io/api/v1/company-news"
        params = {"symbol": ticker, "from": week_ago, "to": today, "token": FINNHUB_KEY}
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        if isinstance(data, list) and data:
            return [{
                "headline": d.get("headline", ""),
                "source": d.get("source", ""),
                "url": d.get("url", ""),
                "datetime": datetime.fromtimestamp(d.get("datetime", 0)).strftime("%b %d, %Y")
            } for d in data[:limit]]
    except Exception as e:
        logger.warning(f"Finnhub news failed: {e}")

    # 3. NewsAPI last resort
    try:
        query = company_name or ticker
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{query} stock",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "language": "en",
            "apiKey": NEWS_KEY,
            "from": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        }
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        return [{
            "headline": a.get("title", ""),
            "source": a.get("source", {}).get("name", ""),
            "url": a.get("url", ""),
            "datetime": a.get("publishedAt", "")[:10]
        } for a in data.get("articles", [])[:limit] if a.get("title")]
    except Exception as e:
        logger.warning(f"NewsAPI failed: {e}")
    
    return []


def _get_local_news(ticker: str, company_name: str = "", limit: int = 5) -> list:
    """News for local market tickers — GNews primary, NewsAPI fallback."""
    query = company_name or _get_local_news_query(ticker)
    _q = query.replace(" stock market 2026","").replace(" stock 2026","").strip()

    # ── 1. GNews API (free 100/day, separate quota) ──
    gnews_key = os.getenv("GNEWS_API_KEY", "")
    if gnews_key:
        try:
            r = requests.get("https://gnews.io/api/v4/search", params={
                "q": _q,
                "lang": "en",
                "max": limit,
                "apikey": gnews_key,
                "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }, timeout=8)
            if r.status_code == 200:
                articles = r.json().get("articles", [])
                if articles:
                    logger.info(f"[GNews] {ticker}: {len(articles)} articles")
                    return [{
                        "headline": a.get("title", ""),
                        "source": a.get("source", {}).get("name", ""),
                        "url": a.get("url", ""),
                        "datetime": a.get("publishedAt", "")[:10]
                    } for a in articles[:limit] if a.get("title")]
        except Exception as e:
            logger.warning(f"GNews local failed: {e}")

    # ── 2. NewsAPI fallback ──
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": _q,
            "sortBy": "publishedAt",
            "pageSize": limit,
            "language": "en",
            "apiKey": NEWS_KEY,
            "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        }
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        articles = data.get("articles", [])
        if articles:
            return [{
                "headline": a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "url": a.get("url", ""),
                "datetime": a.get("publishedAt", "")[:10]
            } for a in articles[:limit] if a.get("title")]
    except Exception as e:
        logger.warning(f"Local NewsAPI failed: {e}")
    
    # Try Arabic news search as fallback
    try:
        from core.ticker_resolver import TickerResolver
        resolver = TickerResolver()
        info = resolver.get_ticker_info(ticker.upper())
        if info:
            ar_name = info.get("name_ar", "")
            if ar_name:
                params["q"] = f"{ar_name} سهم"
                params["language"] = "ar"
                r = requests.get(url, params=params, timeout=8)
                data = r.json()
                articles = data.get("articles", [])
                if articles:
                    return [{
                        "headline": a.get("title", ""),
                        "source": a.get("source", {}).get("name", ""),
                        "url": a.get("url", ""),
                        "datetime": a.get("publishedAt", "")[:10]
                    } for a in articles[:limit] if a.get("title")]
    except Exception as e:
        logger.warning(f"Arabic news fallback failed: {e}")

    return []


def get_upcoming_dividend(ticker: str) -> dict:
    """Upcoming dividend via yfinance"""
    try:
        import yfinance as yf
        from datetime import datetime
        tk = yf.Ticker(ticker)
        info = tk.info
        
        div_rate = info.get("dividendRate", 0) or 0
        div_yield = info.get("dividendYield", 0) or 0
        last_div = info.get("lastDividendValue", 0) or 0
        ex_ts = info.get("exDividendDate") or info.get("lastDividendDate")
        
        if not div_rate and not last_div:
            return {"note": "No dividend data - company may not pay dividends"}
        
        # Convert timestamp to date
        ex_date = "N/A"
        if ex_ts:
            try:
                ex_date = datetime.fromtimestamp(int(ex_ts)).strftime("%b %d, %Y")
            except Exception as _e:
                ex_date = str(ex_ts)
        
        return {
            "amount_per_share": round(last_div, 4),
            "annual_rate": round(div_rate, 4),
            "annual_yield_pct": round(div_yield * 100, 2),
            "ex_dividend_date": ex_date,
            "quarterly_est": round(div_rate / 4, 4) if div_rate else round(last_div, 4)
        }
    except Exception as e:
        logger.warning(f"yfinance dividend failed: {e}")
    return {}

def get_earnings_calendar(ticker: str) -> dict:
    """Next earnings via FMP earnings calendar"""
    
    # ── LOCAL MARKET: Use yfinance instead of FMP ──
    if _is_local_ticker(ticker):
        return _get_local_earnings(ticker)
    
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        future = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")
        
        url = "https://financialmodelingprep.com/api/v3/earning_calendar"
        params = {"from": today, "to": future, "apikey": FMP_KEY}
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        
        if isinstance(data, list):
            matches = [d for d in data if d.get("symbol", "").upper() == ticker.upper()]
            if matches:
                next_e = matches[0]
                return {
                    "date": next_e.get("date", "N/A"),
                    "eps_estimate": next_e.get("epsEstimated", "N/A"),
                    "revenue_estimate": next_e.get("revenueEstimated", "N/A"),
                    "time": next_e.get("time", "N/A")
                }
    except Exception as e:
        logger.warning(f"FMP earnings calendar failed: {e}")
    
    # Alpha Vantage fallback for earnings
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "EARNINGS_CALENDAR",
            "symbol": ticker,
            "horizon": "3month",
            "apikey": os.getenv("ALPHA_VANTAGE_KEY", "")
        }
        r = requests.get(url, params=params, timeout=10)
        # Returns CSV
        lines = r.text.strip().split("\n")
        if len(lines) > 1:
            headers = lines[0].split(",")
            for line in lines[1:]:
                vals = line.split(",")
                if len(vals) >= 3 and ticker.upper() in vals[0].upper():
                    return {
                        "date": vals[2] if len(vals) > 2 else "N/A",
                        "eps_estimate": vals[4] if len(vals) > 4 else "N/A",
                        "time": "N/A"
                    }
    except Exception as e:
        logger.warning(f"Alpha Vantage earnings failed: {e}")
    
    return {"date": "N/A"}


def _get_local_earnings(ticker: str) -> dict:
    """Get earnings data for local tickers via yfinance."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is not None and not cal.empty:
            return {
                "date": str(cal.iloc[0, 0]) if cal.shape[1] > 0 else "N/A",
                "eps_estimate": str(cal.iloc[1, 0]) if cal.shape[0] > 1 else "N/A",
                "revenue_estimate": str(cal.iloc[2, 0]) if cal.shape[0] > 2 else "N/A",
                "time": "N/A"
            }
    except Exception as e:
        logger.warning(f"yfinance earnings for {ticker}: {e}")
    return {"date": "N/A"}


def get_crypto_price(coin_id: str = "bitcoin") -> dict:
    """Real-time crypto via CoinGecko"""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "x_cg_demo_api_key": COINGECKO_KEY
        }
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        if coin_id in data:
            d = data[coin_id]
            return {
                "price": d.get("usd", 0),
                "change_24h": round(d.get("usd_24h_change", 0), 2),
                "market_cap": d.get("usd_market_cap", 0),
                "volume_24h": d.get("usd_24h_vol", 0)
            }
    except Exception as e:
        logger.warning(f"CoinGecko failed: {e}")
    return {}

def get_macro_data(series_id: str = "FEDFUNDS") -> dict:
    """Macro data from FRED (Fed Funds Rate, CPI, GDP, etc.)"""
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 3
        }
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        obs = data.get("observations", [])
        if obs:
            latest = obs[0]
            return {
                "series": series_id,
                "value": latest.get("value", "N/A"),
                "date": latest.get("date", "N/A"),
                "previous": obs[1].get("value", "N/A") if len(obs) > 1 else "N/A"
            }
    except Exception as e:
        logger.warning(f"FRED failed for {series_id}: {e}")
    return {}

def get_key_macro_snapshot() -> dict:
    """Get key macro indicators in one call"""
    return {
        "fed_funds_rate": get_macro_data("FEDFUNDS"),
        "cpi_yoy": get_macro_data("CPIAUCSL"),
        "us_10yr_yield": get_macro_data("DGS10"),
        "unemployment": get_macro_data("UNRATE"),
        "gdp_growth": get_macro_data("A191RL1Q225SBEA")
    }

def get_macro_news(topic: str = "Federal Reserve interest rates", limit: int = 5) -> list:
    """Macro news via NewsAPI"""
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": topic,
            "sortBy": "publishedAt",
            "pageSize": limit,
            "language": "en",
            "apiKey": NEWS_KEY,
            "from": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        }
        r = requests.get(url, params=params, timeout=8)
        return [{
            "headline": a.get("title", ""),
            "source": a.get("source", {}).get("name", ""),
            "url": a.get("url", ""),
            "datetime": a.get("publishedAt", "")[:10]
        } for a in r.json().get("articles", [])[:limit] if a.get("title")]
    except Exception as e:
        logger.warning(f"Macro news failed: {e}")
    return []


# ═══════════════════════════════════════════════════════════════
# STOCKANALYSIS — UAE (ADX / DFM) FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════

# Exchange routing: .AE → adx, .DU → dfm
_UAE_EXCHANGE = {".AE": "adx", ".DU": "dfm"}

_SA_UAE_CACHE: dict = {}
_SA_UAE_TTL   = 3600  # 1 hour cache

def _stockanalysis_uae(ticker: str) -> dict:
    """
    Fetch fundamentals for UAE stocks (ADX / DFM) from StockAnalysis.com.
    Returns same keys as _deepcrawl_local_fallback for full compatibility.
    Fields: price, market_cap, eps, pe_ratio, forward_pe, beta,
            dividend_yield, revenue, net_income, shares_out,
            week_52_range, rev_growth, earnings_growth, source.
    """
    import re, time

    # ── cache check ─────────────────────────────────────────────
    cached = _SA_UAE_CACHE.get(ticker)
    if cached and time.time() - cached["_ts"] < _SA_UAE_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    # ── build URL ────────────────────────────────────────────────
    suffix = ".AE" if ticker.upper().endswith(".AE") else ".DU"
    exch   = _UAE_EXCHANGE.get(suffix, "adx")
    slug   = ticker.upper().replace(suffix, "").lower()
    url    = f"https://stockanalysis.com/quote/{exch}/{slug}/"

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            logger.warning("[SA-UAE] %s returned %d", ticker, r.status_code)
            return {}
        text = r.text
    except Exception as e:
        logger.warning("[SA-UAE] fetch failed for %s: %s", ticker, e)
        return {}

    def _rx(pattern, grp=1):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(grp).strip() if m else None

    result = {"ticker": ticker.upper(), "source": f"StockAnalysis ({exch.upper()})"}

    # ── Price from market_data_engine (more reliable than SA for UAE) ────────
    try:
        from core.market_data_engine import get_stock_data
        import pandas as pd
        df = get_stock_data(ticker, "AE", period="1m")
        if df is not None and not df.empty:
            result["price"] = round(float(df["Close"].iloc[-1]), 3)
    except Exception as e:
        logger.warning("[realtime_data] AE price fetch failed for %s: %s", ticker, e)

    # ── Market Cap & Shares ──────────────────────────────────────
    mc_raw = _rx(r'marketCap[\"\':\s]+([0-9.e+\-]+)')
    if mc_raw:
        mc = float(mc_raw)
        if mc >= 1e9:    result["market_cap"] = f"{mc/1e9:.2f}B AED"
        elif mc >= 1e6:  result["market_cap"] = f"{mc/1e6:.0f}M AED"
        else:            result["market_cap"] = f"{mc:.2f}B AED"   # already in B
        result["market_cap_raw"] = mc

    shares_raw = _rx(r'sharesOut[\"\':\s]+([0-9.e+\-]+)')
    if shares_raw:
        sh = float(shares_raw)
        result["shares_out_raw"] = sh
        # StockAnalysis stores sharesOut in billions (e.g. 76.72 = 76.72B shares)
        if sh >= 1e9:
            result["shares_out"] = f"{sh/1e9:.2f}B"
        elif sh >= 1e6:
            result["shares_out"] = f"{sh/1e6:.0f}M"
        else:
            result["shares_out"] = f"{sh:.2f}B"  # assume billions

    # ── EPS & P/E ────────────────────────────────────────────────
    eps_raw = _rx(r'eps[\"\':\s]+([0-9.]+)')
    if eps_raw:
        eps = float(eps_raw)
        result["eps"] = str(round(eps, 3))
        # Calculate P/E TTM from live price / EPS
        price = result.get("price")
        if price and eps > 0:
            result["pe_ratio"] = str(round(price / eps, 1))

    fpe = _rx(r'forwardPe[\"\':\s]+([0-9.]+)')
    if fpe:
        result["forward_pe"] = fpe

    # ── Beta ─────────────────────────────────────────────────────
    beta = _rx(r'beta[\"\':\s]+([0-9.]+)')
    if beta:
        result["beta"] = beta

    # ── Dividend ─────────────────────────────────────────────────
    div = _rx(r'dividendYield[\"\':\s]+([0-9.]+)')
    if div:
        result["dividend_yield"] = f"{div}%"

    # ── Revenue & Earnings (most recent row) ─────────────────────
    rev_all = re.findall(r'revenue:\s*([0-9.]+e?\+?[0-9]*)', text)
    earn_all = re.findall(r'earnings:\s*([0-9.]+e?\+?[0-9]*)', text)
    if rev_all:
        rev = float(rev_all[-1])
        result["revenue"] = f"{rev/1e9:.2f}B AED" if rev >= 1e9 else f"{rev/1e6:.0f}M AED"
    if earn_all:
        ni = float(earn_all[0])
        result["net_income"] = f"{ni/1e9:.2f}B AED" if ni >= 1e9 else f"{ni/1e6:.0f}M AED"

    # ── Growth rates ─────────────────────────────────────────────
    rev_g = _rx(r'revenueGrowth:\s*(-?[0-9.]+)')
    if rev_g:
        result["rev_growth"] = f"{float(rev_g):+.1f}%"

    earn_g = _rx(r'earningsGrowth:\s*(-?[0-9.]+)')
    if earn_g:
        result["earnings_growth"] = f"{float(earn_g):+.1f}%"

    # ── 52-Week Range from historical data (split-adjusted) ─────
    try:
        from core.market_data_engine import get_stock_data
        import pandas as pd, numpy as np
        df_1y = get_stock_data(ticker, "AE", period="1y")
        if df_1y is not None and not df_1y.empty:
            # Detect stock split: large single-day drop (>40%) → use only post-split data
            pct_chg = df_1y["Close"].pct_change()
            split_idx = pct_chg[pct_chg < -0.40].index
            if len(split_idx) > 0:
                # Use data only from last split date onwards
                df_1y = df_1y[df_1y.index >= split_idx[-1]]
            h52 = round(df_1y["High"].max(), 3)
            l52 = round(df_1y["Low"].min(), 3)
            result["week_52_range"] = f"{l52} - {h52}"
            result["week_52_high"]  = h52
            result["week_52_low"]   = l52
    except Exception:
        pass

    # ── Sector & Industry ────────────────────────────────────────
    sector_raw = _rx(r'"sector"\s*:\s*"([^"]+)"')
    if not sector_raw:
        sector_raw = _rx(r'Sector[^|]*\|\s*([A-Za-z &/]+)\s*\|')
    if sector_raw:
        result["sector"] = sector_raw.strip()
        result["industry"] = sector_raw.strip()

    # ── Excel fallback for sector/industry/name ───────────────────
    try:
        from core.excel_stock_lookup import get_stock_info as _xl_info
        _xl = _xl_info(ticker)
        if _xl:
            if not result.get("sector") or result.get("sector") in ("Unknown", "N/A"):
                if _xl.get("sector") and _xl["sector"] not in ("nan", "NaN"):
                    result["sector"] = _xl["sector"]
            if not result.get("industry") or result.get("industry") in ("Unknown", "N/A"):
                if _xl.get("industry") and _xl["industry"] not in ("nan", "NaN"):
                    result["industry"] = _xl["industry"]
            if not result.get("company_name"):
                if _xl.get("name") and _xl["name"] not in ("nan", "NaN"):
                    result["company_name"] = _xl["name"]
    except Exception:
        pass

    # ── Analyst Price Target (from forecast page) ─────────────────
    try:
        forecast_url = f"https://stockanalysis.com/quote/{exch}/{slug}/forecast/"
        rf = requests.get(forecast_url, headers=headers, timeout=10)
        if rf.status_code == 200:
            ft = rf.text
            def _rxf(pattern, grp=1):
                m = re.search(pattern, ft, re.IGNORECASE)
                return m.group(grp).strip() if m else None
            # Average target price
            pt_avg = _rxf(r'"priceTarget"\s*:\s*([0-9.]+)')
            if not pt_avg:
                pt_avg = _rxf(r'Average[^|]*\|\s*([0-9.]+)')
            pt_high = _rxf(r'High[^|]*\|\s*([0-9.]+)')
            pt_low  = _rxf(r'Low[^|]*\|\s*([0-9.]+)')
            if pt_avg:
                price_now = result.get("price", 0)
                try:
                    upside = ((float(pt_avg) - price_now) / price_now * 100) if price_now else 0
                    result["price_target"] = f"{float(pt_avg):.2f} ({upside:+.1f}%)"
                    result["analyst_target"] = float(pt_avg)
                except Exception:
                    result["price_target"] = pt_avg
            if pt_high: result["price_target_high"] = pt_high
            if pt_low:  result["price_target_low"]  = pt_low
            # Analyst consensus
            consensus = _rxf(r'"consensus"\s*:\s*"([^"]+)"')
            if not consensus:
                consensus = _rxf(r'(Strong Buy|Buy|Hold|Sell|Strong Sell)')
            if consensus:
                result["analyst_rating"] = consensus
    except Exception:
        pass

    # ── Currency & market label ──────────────────────────────────
    result["currency"] = "AED"

    if not result.get("price") and not result.get("market_cap"):
        logger.warning("[SA-UAE] No useful data extracted for %s", ticker)
        return {}

    # ── store cache ──────────────────────────────────────────────
    _SA_UAE_CACHE[ticker] = {**result, "_ts": time.time()}
    logger.info("[SA-UAE] %s — %d fields fetched", ticker, len(result))
    return result


# ═══════════════════════════════════════════════════════════════
# DEEPCRAWL — MULTI-SOURCE DATA ENGINE (3 sources in parallel)
# ═══════════════════════════════════════════════════════════════
DEEPCRAWL_URL = "https://deepcrawl-worker-v0.eisax.workers.dev/read"
_DC_TIMEOUT   = 7   # seconds per source

def _dc_fetch(url: str) -> str:
    """Fetch any URL via the DeepCrawl worker and return clean Markdown."""
    try:
        r = requests.get(DEEPCRAWL_URL, params={"url": url}, timeout=_DC_TIMEOUT)
        text = r.text
        return text if len(text) > 100 and "INTERNAL_SERVER_ERROR" not in text else ""
    except Exception as _e:
        logger.debug("[DeepCrawl] fetch failed for %s: %s", url, _e)
        return ""


def _dc_stockanalysis(ticker: str) -> dict:
    """Source 1 — StockAnalysis overview: price, PE, market cap, analyst rating."""
    import re
    content = _dc_fetch(f"https://stockanalysis.com/stocks/{ticker.lower()}/")
    if not content:
        return {}
    result = {}
    for exch in ("NASDAQ", "NYSE", "NYSE ARCA"):
        m = re.search(rf'{exch}.*?USD\s*\n([\d.]+)', content)
        if m:
            result["price"] = float(m.group(1))
            break
    patterns = {
        "market_cap":     r'Market Cap.*?\|\s*([\d.]+[TBMK]?)',
        "revenue":        r'Revenue.*?\|\s*([\d.]+[TBMK]?)',
        "net_income":     r'Net Income.*?\|\s*([\d.]+[TBMK]?)',
        "eps":            r'\bEPS\b.*?\|\s*([\d.]+)',
        "pe_ratio":       r'PE Ratio.*?\|\s*([\d.]+)',
        "forward_pe":     r'Forward PE.*?\|\s*([\d.]+)',
        "dividend":       r'Dividend.*?\|\s*\$([\d.]+)',
        "ex_div_date":    r'Ex-Dividend Date.*?\|\s*([A-Za-z]+ \d+, \d{4})',
        "volume":         r'Volume.*?\|\s*([\d,]+)',
        "week_52_range":  r'52-Week Range.*?\|\s*([\d.]+ - [\d.]+)',
        "beta":           r'Beta.*?\|\s*([\d.]+)',
        "analyst_rating": r'Analysts.*?\|\s*([^\n|]+)',
        "price_target":   r'Price Target.*?\|\s*([\d.]+ \([^)]+\))',
        "earnings_date":  r'Earnings Date.*?\|\s*([A-Za-z]+ \d+, \d{4})',
    }
    for key, pat in patterns.items():
        m = re.search(pat, content)
        if m:
            result[key] = m.group(1).strip()
    return result


def _dc_financials(ticker: str) -> dict:
    """Source 2 — StockAnalysis financials: revenue trend, EPS, FCF, margins."""
    import re
    content = _dc_fetch(f"https://stockanalysis.com/stocks/{ticker.lower()}/financials/")
    if not content:
        return {}
    result = {}

    # Extract fiscal years from header row (e.g. "FY 2026 | FY 2025 | ...")
    years = re.findall(r'FY (\d{4})', content)[:5]

    def _extract_row(label_pattern: str) -> list:
        """Find a table row by label and return the numeric values."""
        # Wrap label in non-capturing group so OR doesn't split the capture group
        m = re.search(r'(?:' + label_pattern + r')[^\|]*\|(.*?)(?:\n|$)', content, re.IGNORECASE)
        if not m or m.group(1) is None:
            return []
        raw = m.group(1)
        # Values may be like "215,938" or "4.93" or "-"
        vals = re.findall(r'([\d,]+\.?\d*)', raw)
        return [v.replace(',', '') for v in vals[:5]]

    def _zip(years, vals):
        return {y: v for y, v in zip(years, vals) if v and v != '-'}

    # Revenue history
    rev_vals = _extract_row(r'\[Revenue\]|Revenue(?!.*Margin|.*Growth)')
    if rev_vals and years:
        result["revenue_history"] = _zip(years, rev_vals)

    # EPS history — prefer Diluted, fallback to Basic, skip Growth/Adjusted
    eps_vals = _extract_row(r'EPS \(Diluted\)')
    if not eps_vals:
        eps_vals = _extract_row(r'EPS \(Basic\)')
    if eps_vals and years:
        result["eps_history"] = _zip(years, eps_vals)

    # Free Cash Flow
    fcf_vals = _extract_row(r'Free Cash Flow')
    if fcf_vals:
        result["free_cash_flow"] = fcf_vals[0]   # most recent year

    # Net Margin — try pattern with %
    nm = re.search(r'Net (?:Profit )?Margin[^\|]*\|\s*([+-]?[\d.]+)%', content, re.IGNORECASE)
    if nm:
        result["net_margin_annual"] = f"{nm.group(1)}%"

    # Gross Margin
    gm = re.search(r'Gross (?:Profit )?Margin[^\|]*\|\s*([+-]?[\d.]+)%', content, re.IGNORECASE)
    if gm:
        result["gross_margin"] = f"{gm.group(1)}%"

    return result


def _dc_finviz(ticker: str) -> dict:
    """Source 3 — Finviz: RSI, short float, institutional %, technicals, performance.
    Falls back to StockAnalysis ratios page if Finviz is blocked."""
    import re
    result = {}

    # Try Finviz first
    content = _dc_fetch(f"https://finviz.com/quote.ashx?t={ticker.upper()}")
    if content and len(content) > 200:
        patterns = {
            "rsi":           r'RSI.*?\|\s*([\d.]+)',
            "short_float":   r'Short Float.*?\|\s*([\d.]+)%',
            "inst_own":      r'Inst Own.*?\|\s*([\d.]+)%',
            "insider_own":   r'Insider Own.*?\|\s*([\d.]+)%',
            "sma50":         r'SMA50.*?\|\s*([\d.]+)',
            "sma200":        r'SMA200.*?\|\s*([\d.]+)',
            "avg_volume":    r'Avg Volume.*?\|\s*([\d.]+[MK]?)',
            "debt_equity":   r'Debt/Eq.*?\|\s*([\d.]+)',
            "roe":           r'ROE.*?\|\s*([\d.]+)%',
            "roa":           r'ROA.*?\|\s*([\d.]+)%',
            "profit_margin": r'Profit Margin.*?\|\s*([\d.]+)%',
            "perf_week":     r'Perf Week.*?\|\s*([+-]?[\d.]+)%',
            "perf_month":    r'Perf Month.*?\|\s*([+-]?[\d.]+)%',
            "perf_ytd":      r'Perf YTD.*?\|\s*([+-]?[\d.]+)%',
        }
        for key, pat in patterns.items():
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                result[key] = m.group(1).strip()
        if result:
            return result

    # Fallback — StockAnalysis ratios page (public, no blocking)
    ratios = _dc_fetch(f"https://stockanalysis.com/stocks/{ticker.lower()}/financials/ratios/")
    if ratios:
        ratio_patterns = {
            "roe":          r'Return on Equity[^\|]*\|\s*([+-]?[\d.]+)%',
            "roa":          r'Return on Assets[^\|]*\|\s*([+-]?[\d.]+)%',
            "debt_equity":  r'Debt\s*/\s*Equity[^\|]*\|\s*([+-]?[\d.]+)',
            "profit_margin":r'(?:Net |Profit )Margin[^\|]*\|\s*([+-]?[\d.]+)%',
            "gross_margin": r'Gross Margin[^\|]*\|\s*([+-]?[\d.]+)%',
        }
        for key, pat in ratio_patterns.items():
            if key not in result:
                m = re.search(pat, ratios, re.IGNORECASE)
                if m:
                    result[key] = m.group(1).strip()

    # Performance from StockAnalysis overview (52w change, etc.)
    overview = _dc_fetch(f"https://stockanalysis.com/stocks/{ticker.lower()}/")
    if overview:
        perf_patterns = {
            "week_52_range": r'52.Week Range[^\|]*\|\s*([^\|\n]+)',
            "avg_volume":    r'Average Volume[^\|]*\|\s*([\d,]+)',
        }
        for key, pat in perf_patterns.items():
            if key not in result:
                m = re.search(pat, overview, re.IGNORECASE)
                if m:
                    result[key] = m.group(1).strip()

    return result


def _dc_forecast(ticker: str) -> dict:
    """Source 4 — StockAnalysis forecast: analyst buy/hold/sell counts, price targets."""
    import re
    content = _dc_fetch(f"https://stockanalysis.com/stocks/{ticker.lower()}/forecast/")
    if not content:
        return {}
    result = {}
    # Analyst counts from rating table: Strong Buy | Buy | Hold | Sell | Strong Sell
    sb = re.search(r'Strong Buy[^\|]*\|\s*(\d+)', content, re.IGNORECASE)
    b  = re.search(r'\|\s*Buy\s*\|\s*(\d+)', content)
    h  = re.search(r'Hold[^\|]*\|\s*(\d+)', content, re.IGNORECASE)
    s  = re.search(r'(?:Sell|Strong Sell)[^\|]*\|\s*(\d+)', content, re.IGNORECASE)
    buy_cnt  = (int(sb.group(1)) if sb else 0) + (int(b.group(1))  if b else 0)
    hold_cnt = int(h.group(1)) if h else 0
    sell_cnt = int(s.group(1)) if s else 0
    if buy_cnt + hold_cnt + sell_cnt > 0:
        result["analyst_buy"]  = buy_cnt
        result["analyst_hold"] = hold_cnt
        result["analyst_sell"] = sell_cnt
    # Price targets
    pt_avg  = re.search(r'\$\s*([\d.]+)[^\n]*\+[\d.]+%', content)
    pt_low  = re.search(r'Low[^\|]*\|\s*\$([\d.]+)', content, re.IGNORECASE)
    pt_high = re.search(r'High[^\|]*\|\s*\$([\d.]+)', content, re.IGNORECASE)
    pt_med  = re.search(r'Median[^\|]*\|\s*\$([\d.]+)', content, re.IGNORECASE)
    if pt_avg:  result["price_target_mean"]   = pt_avg.group(1)
    if pt_low:  result["price_target_low"]    = pt_low.group(1)
    if pt_high: result["price_target_high"]   = pt_high.group(1)
    if pt_med:  result["price_target_median"] = pt_med.group(1)
    return result


def deepcrawl_stock(ticker: str) -> dict:
    """
    Fetch rich stock data from 4 sources in parallel:
      1. StockAnalysis overview  — price, PE, rating, earnings date
      2. StockAnalysis financials — revenue trend, margins, FCF, EPS history
      3. Finviz / SA ratios      — ROE, ROA, debt/equity, margins (Finviz fallback to SA)
      4. StockAnalysis forecast  — analyst buy/hold/sell counts, price targets (low/avg/high)
    Falls back to yfinance for local market tickers (UAE, Saudi, Egypt).
    """
    if _is_local_ticker(ticker):
        # ── UAE (ADX / DFM): StockAnalysis primary + dfm_lookup context ──────
        if ticker.upper().endswith((".DU", ".AE")):
            sa_data = _stockanalysis_uae(ticker)
            if sa_data:
                # Enrich with dfm_lookup sector/name context if available
                try:
                    import sys; sys.path.insert(0, '/home/ubuntu/investwise')
                    from core.dfm_lookup import get_dfm_context
                    ctx = get_dfm_context(ticker)
                    if ctx:
                        # Only add fields not already present
                        for k, v in ctx.items():
                            if k not in sa_data or not sa_data[k]:
                                sa_data[k] = v
                except Exception as e:
                    logger.debug("[realtime_data] DFM context enrichment failed for %s: %s", ticker, e)
                return sa_data
            # Fallback: dfm_lookup only
            try:
                import sys; sys.path.insert(0, '/home/ubuntu/investwise')
                from core.dfm_lookup import get_dfm_context
                ctx = get_dfm_context(ticker)
                if ctx:
                    return {"local_context": ctx, "source": "dfm_lookup", "ticker": ticker}
            except Exception as _dle:
                logger.warning(f"dfm_lookup failed for {ticker}: {_dle}")
        # ── Egypt: استخدم egx_lookup ──
        elif ticker.upper().endswith(".CA"):
            try:
                import sys; sys.path.insert(0, '/home/ubuntu/investwise')
                from core.egx_lookup import get_egx_context
                ctx = get_egx_context(ticker)
                if ctx:
                    return {"local_context": ctx, "source": "egx_lookup", "ticker": ticker}
            except Exception as _ele:
                logger.warning(f"egx_lookup failed for {ticker}: {_ele}")
        return _deepcrawl_local_fallback(ticker)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs = {
        "overview":   (_dc_stockanalysis, ticker),
        "financials": (_dc_financials,    ticker),
        "technical":  (_dc_finviz,        ticker),
        "forecast":   (_dc_forecast,      ticker),
    }
    parts = {}
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fn, arg): name for name, (fn, arg) in jobs.items()}
            for future in as_completed(futures, timeout=18):
                name = futures[future]
                try:
                    parts[name] = future.result()
                except Exception as _fe:
                    logger.warning("[DeepCrawl] %s failed for %s: %s", name, ticker, _fe)
                    parts[name] = {}
    except Exception as e:
        logger.error("[DeepCrawl] Parallel fetch failed for %s: %s", ticker, e)
        return {}

    merged = {"ticker": ticker.upper(), "source": "DeepCrawl (StockAnalysis + Finviz + Forecast)"}
    merged.update(parts.get("overview",   {}))
    merged.update(parts.get("financials", {}))
    merged.update(parts.get("technical",  {}))
    # Forecast enriches price targets and buy/hold/sell counts (don't overwrite overview rating)
    for k, v in (parts.get("forecast") or {}).items():
        if k not in merged or not merged[k]:
            merged[k] = v

    if not merged.get("price"):
        logger.warning("[DeepCrawl] No price returned for %s — all sources empty", ticker)
        return {}

    logger.info("[DeepCrawl] %s enriched — %d fields from %d sources",
                ticker, len(merged), sum(1 for v in parts.values() if v))
    return merged


def _deepcrawl_local_fallback(ticker: str) -> dict:
    """
    Fallback data fetcher for local market tickers using yfinance.
    Returns same format as deepcrawl_stock() for compatibility.
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)

        # ── fast_info first: fail-fast check + grab price/cap/volume cheaply ──
        # fast_info uses a lightweight endpoint (~300ms vs ~2s for .info).
        # If there's no price here, the ticker is invalid — skip the slow .info call.
        _fi = tk.fast_info
        _fi_price = float(getattr(_fi, "last_price",  None) or 0)
        _fi_mc    = getattr(_fi, "market_cap", None)
        _fi_vol   = getattr(_fi, "volume",     None)

        if not _fi_price:
            logger.warning(f"yfinance fast_info returned no price for {ticker}")
            return {"source": "yfinance (limited)", "ticker": ticker}

        # ── .info for detailed fundamentals (PE, beta, dividends, etc.) ──
        info = tk.info or {}

        # Get currency info
        try:
            from core.local_tickers import get_ticker_currency, SUPPORTED_CURRENCIES
            currency = get_ticker_currency(ticker)
            currency_info = SUPPORTED_CURRENCIES.get(currency, {})
            currency_symbol = currency_info.get("symbol", "$")
        except Exception as _e:
            currency_symbol = "$"
            currency = "USD"

        result = {
            "source": f"yfinance ({currency})",
            "ticker": ticker.upper(),
            "currency": currency,
        }

        # Price — prefer fast_info value (already fetched, no extra cost)
        price = _fi_price or info.get("regularMarketPrice") or info.get("currentPrice") or 0
        if price:
            # Never write price_cache from the yfinance fallback path.
            # TV cache is the authoritative source across markets; a YF value
            # cached here would override TV on subsequent reads within the TTL.
            result["price"] = float(price)

        # Market cap — prefer fast_info (avoids one .info dict lookup)
        mc = _fi_mc or info.get("marketCap", 0)
        if mc:
            if mc >= 1e12:
                result["market_cap"] = f"{mc/1e12:.2f}T"
            elif mc >= 1e9:
                result["market_cap"] = f"{mc/1e9:.2f}B"
            elif mc >= 1e6:
                result["market_cap"] = f"{mc/1e6:.0f}M"
            else:
                result["market_cap"] = str(mc)
        
        # Key metrics
        if info.get("trailingPE"):
            result["pe_ratio"] = str(round(float(info["trailingPE"]), 2))
        if info.get("forwardPE"):
            result["forward_pe"] = str(round(float(info["forwardPE"]), 2))
        if info.get("trailingEps"):
            result["eps"] = str(round(float(info["trailingEps"]), 2))
        if info.get("beta"):
            result["beta"] = str(round(float(info["beta"]), 2))
        if info.get("dividendYield"):
            # yfinance API change (2024+): dividendYield is now returned as a
            # percent (e.g. 5.03 for a 5.03% yield) instead of the older
            # decimal form (0.0503). The previous *100 multiplication produced
            # values like 503% that were silently capped at 30% by
            # _safe_div_yield, breaking dividend-aware verdict logic.
            # Detect both formats: values >1 are already percent, ≤1 are
            # legacy decimal.
            _dy_raw = float(info['dividendYield'])
            _dy_pct = _dy_raw if _dy_raw > 1.0 else _dy_raw * 100
            result["dividend_yield"] = f"{_dy_pct:.2f}%"
        if info.get("fiftyTwoWeekLow") and info.get("fiftyTwoWeekHigh"):
            result["week_52_range"] = f"{info['fiftyTwoWeekLow']:.2f} - {info['fiftyTwoWeekHigh']:.2f}"
        _vol = _fi_vol or info.get("volume")
        if _vol:
            result["volume"] = f"{int(_vol):,}"
        if info.get("averageVolume"):
            result["avg_volume"] = f"{info['averageVolume']:,}"
        
        # Revenue & Income
        if info.get("totalRevenue"):
            rev = info["totalRevenue"]
            if rev >= 1e9:
                result["revenue"] = f"{rev/1e9:.2f}B"
            elif rev >= 1e6:
                result["revenue"] = f"{rev/1e6:.0f}M"
        if info.get("netIncomeToCommon"):
            ni = info["netIncomeToCommon"]
            if ni >= 1e9:
                result["net_income"] = f"{ni/1e9:.2f}B"
            elif ni >= 1e6:
                result["net_income"] = f"{ni/1e6:.0f}M"
        
        # Profitability
        if info.get("profitMargins"):
            result["net_margin"] = f"{float(info['profitMargins'])*100:.1f}%"
        if info.get("returnOnEquity"):
            result["roe"] = f"{float(info['returnOnEquity'])*100:.1f}%"
        if info.get("revenueGrowth"):
            result["rev_growth"] = f"{float(info['revenueGrowth'])*100:.1f}%"
        
        # Sector
        if info.get("sector"):
            result["sector"] = info["sector"]
        if info.get("industry"):
            result["industry"] = info["industry"]
        
        # Target & recommendation
        if info.get("targetMeanPrice"):
            target = float(info["targetMeanPrice"])
            upside = ((target - price) / price * 100) if price else 0
            result["price_target"] = f"{target:.2f} ({upside:+.1f}%)"
            result["analyst_target"] = target
        if info.get("recommendationKey"):
            result["analyst_rating"] = info["recommendationKey"].title()
        
        return result
        
    except Exception as e:
        logger.warning(f"yfinance fallback failed for {ticker}: {e}")
        return {"source": "yfinance (error)", "ticker": ticker}


def deepcrawl_news(ticker: str, limit: int = 5) -> list:
    """Fetch latest news for ticker via DeepCrawl + StockAnalysis news page"""
    
    # ── LOCAL MARKET: Use NewsAPI instead ──
    if _is_local_ticker(ticker):
        return _get_local_news(ticker, limit=limit)
    
    try:
        url = f"https://stockanalysis.com/stocks/{ticker.lower()}/news/"
        r = requests.get(DEEPCRAWL_URL, params={"url": url}, timeout=5)
        content = r.text
        
        if "INTERNAL_SERVER_ERROR" in content:
            return []
        
        import re
        # Extract news headlines and dates
        news = []
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if re.match(r'^#{1,3}\s+\[', line):
                title_m = re.search(r'\[([^\]]+)\]\(([^)]+)\)', line)
                if title_m:
                    headline = title_m.group(1)
                    url_part = title_m.group(2)
                    # Look for date in nearby lines
                    date = ""
                    for j in range(i+1, min(i+4, len(lines))):
                        date_m = re.search(r'([A-Za-z]+ \d+, \d{4}|\d+ \w+ ago)', lines[j])
                        if date_m:
                            date = date_m.group(1)
                            break
                    news.append({
                        "headline": headline,
                        "url": url_part if url_part.startswith("http") else f"https://stockanalysis.com{url_part}",
                        "datetime": date,
                        "source": "StockAnalysis"
                    })
                    if len(news) >= limit:
                        break
        return news
    except Exception as e:
        logger.warning(f"DeepCrawl news failed: {e}")
        return []


if __name__ == "__main__":
    logger.debug("=" * 50)
    logger.debug("EisaX Real-Time Data Module Test")
    logger.debug("=" * 50)
    logger.debug("\n[NVDA NEWS — US]")
    news = get_live_news("NVDA", "NVIDIA", 3)
    for n in news:
        logger.debug(f"  [{n['datetime']}] {n['headline'][:80]}")
    logger.debug("\n[2222.SR — Saudi Aramco (Local)]")
    dc = deepcrawl_stock("2222.SR")
    logger.debug(f"  Source: {dc.get('source', 'N/A')}")
    logger.debug(f"  Price: {dc.get('price', 'N/A')}")
    logger.debug(f"  P/E: {dc.get('pe_ratio', 'N/A')}")
    logger.debug(f"  Market Cap: {dc.get('market_cap', 'N/A')}")
    logger.debug("\n[COMI.CA — CIB Egypt (Local)]")
    dc2 = deepcrawl_stock("COMI.CA")
    logger.debug(f"  Source: {dc2.get('source', 'N/A')}")
    logger.debug(f"  Price: {dc2.get('price', 'N/A')}")
    logger.debug("\n[AAPL DIVIDEND]")
    div = get_upcoming_dividend("AAPL")
    logger.debug(f"  {div}")
    logger.debug("\n[MSFT EARNINGS]")
    earn = get_earnings_calendar("MSFT")
    logger.debug(f"  Next: {earn}")
    logger.debug("\n[BTC PRICE]")
    btc = get_crypto_price("bitcoin")
    logger.debug(f"  BTC: ${btc.get('price', 0):,.0f} ({btc.get('change_24h', 0):+.2f}%)")
    logger.debug("\n[MACRO SNAPSHOT]")
    macro = get_key_macro_snapshot()
    for k, v in macro.items():
        logger.debug(f"  {k}: {v.get('value', 'N/A')} ({v.get('date', '')})")