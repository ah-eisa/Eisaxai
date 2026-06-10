"""
EisaX RapidAPI + Free Data Layer
=================================
6 data sources integrated with RapidAPI primary + free public fallbacks.
All functions are safe to call — they never raise, always return {} or [].

Sources:
  1. Fear & Greed   — alternative.me (free) / RapidAPI CNN (when subscribed)
  2. Forex Calendar — ForexFactory JSON (free) / RapidAPI Forex (when subscribed)
  3. CNBC News      — DeepCrawl scrape (free) / RapidAPI CNBC (when subscribed)
  4. Cash Flow      — yfinance quarterly (free) / RapidAPI Real-Time Finance (when subscribed)
  5. Events Calendar— yfinance calendar (free) / RapidAPI TradingView (when subscribed)
  6. Tadawul        — yfinance .SR (free) / RapidAPI Tadawul (when subscribed)
"""

import requests, logging, time
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

RAPIDAPI_KEY  = "f11dbd45f8msh92f122b3d8936f3p137a6fjsn83b242116387"
RAPID_HEADERS = {"x-rapidapi-key": RAPIDAPI_KEY}
_TIMEOUT      = 8   # seconds per request

# ── Simple TTL cache ──────────────────────────────────────────────────────────
_cache: dict = {}

def _cached(key: str, ttl: int, fn):
    """Return cached value if fresh, else call fn() and cache result."""
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < ttl:
        return entry["val"]
    val = fn()
    _cache[key] = {"val": val, "ts": time.time()}
    return val


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEAR & GREED INDEX
# ══════════════════════════════════════════════════════════════════════════════

def get_fear_greed() -> dict:
    """
    Returns Fear & Greed index.
    Primary  : RapidAPI CNN Fear & Greed (when subscribed)
    Fallback : alternative.me free API (Crypto F&G proxy — widely used)
    Returns  : {score, rating, label_ar, timestamp, source}
    """
    def _fetch():
        # ── Try RapidAPI CNN first ──
        try:
            r = requests.get(
                "https://cnn-fear-and-greed-index.p.rapidapi.com/cnn/v1/fear-and-greed/latest",
                headers={**RAPID_HEADERS, "x-rapidapi-host": "cnn-fear-and-greed-index.p.rapidapi.com"},
                timeout=_TIMEOUT
            )
            if r.status_code == 200:
                d = r.json()
                score = d.get("fearGreedIndex", {}).get("score") or d.get("score")
                rating = d.get("fearGreedIndex", {}).get("rating") or d.get("rating", "")
                if score:
                    return _build_fg(float(score), rating, "CNN Fear & Greed (RapidAPI)")
        except Exception:
            pass

        # ── Fallback: alternative.me ──
        try:
            r = requests.get(
                "https://api.alternative.me/fng/?limit=1&format=json",
                timeout=_TIMEOUT
            )
            if r.status_code == 200:
                item = r.json().get("data", [{}])[0]
                score  = float(item.get("value", 0))
                rating = item.get("value_classification", "")
                return _build_fg(score, rating, "alternative.me")
        except Exception as e:
            logger.warning("[RapidData] Fear&Greed fallback failed: %s", e)

        return {}

    return _cached("fear_greed", ttl=3600, fn=_fetch)   # cache 1 hour


def _build_fg(score: float, rating: str, source: str) -> dict:
    """Build standardized Fear & Greed dict with Arabic label."""
    r = rating.lower()
    if "extreme fear" in r:
        label_ar = "خوف شديد"
        color = "#f43f5e"
    elif "fear" in r:
        label_ar = "خوف"
        color = "#fb923c"
    elif "neutral" in r or "greed" not in r:
        label_ar = "محايد"
        color = "#fbbf24"
    elif "extreme greed" in r:
        label_ar = "طمع شديد"
        color = "#00d97e"
    else:
        label_ar = "طمع"
        color = "#34d399"
    return {
        "score":    round(score, 1),
        "rating":   rating,
        "label_ar": label_ar,
        "color":    color,
        "source":   source,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. FOREX ECONOMIC CALENDAR
# ══════════════════════════════════════════════════════════════════════════════

def get_forex_calendar(days_ahead: int = 7,
                       countries: Optional[list] = None,
                       min_impact: str = "Medium") -> list:
    """
    Returns upcoming economic events.
    Primary  : RapidAPI Forex Calendar (when subscribed)
    Fallback : ForexFactory free JSON API
    Returns  : list of {date, country, title, impact, forecast, previous}
    """
    countries = countries or ["USD", "EUR", "GBP", "JPY", "AED", "SAR"]

    def _fetch():
        today = datetime.now(timezone.utc).date()
        end   = today + timedelta(days=days_ahead)

        # ── Try RapidAPI ──
        try:
            iso_countries = "%3B".join([c.lower() for c in countries[:6]])
            url = (f"https://forex-api2.p.rapidapi.com/v2/calendar/get"
                   f"?startDate={today}&endDate={end}"
                   f"&includeVolatilities=medium%3Bhigh"
                   f"&includeCountries={iso_countries}")
            r = requests.get(url,
                headers={**RAPID_HEADERS, "x-rapidapi-host": "forex-api2.p.rapidapi.com"},
                timeout=_TIMEOUT)
            if r.status_code == 200:
                raw = r.json()
                events = raw if isinstance(raw, list) else raw.get("data", raw.get("events", []))
                return _normalize_calendar(events, "RapidAPI Forex")
        except Exception:
            pass

        # ── Fallback: ForexFactory ──
        try:
            r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                             timeout=_TIMEOUT)
            if r.status_code == 200:
                raw = r.json()
                # Filter by impact and country
                impact_map = {"Holiday": 0, "Low": 1, "Medium": 2, "High": 3}
                min_lvl = impact_map.get(min_impact, 2)
                filtered = [
                    e for e in raw
                    if impact_map.get(e.get("impact", "Low"), 0) >= min_lvl
                    and e.get("country", "").upper() in [c.upper() for c in countries]
                ]
                return _normalize_calendar(filtered, "ForexFactory")
        except Exception as e:
            logger.warning("[RapidData] Forex calendar failed: %s", e)

        return []

    return _cached(f"forex_cal_{days_ahead}", ttl=3600, fn=_fetch)


def _normalize_calendar(events: list, source: str) -> list:
    """Normalize calendar events to standard format."""
    out = []
    for e in events[:20]:
        title   = e.get("title") or e.get("name") or e.get("event") or ""
        date_raw = e.get("date") or e.get("time") or e.get("datetime") or ""
        country  = (e.get("country") or e.get("currency") or "").upper()
        impact   = e.get("impact") or e.get("volatility") or ""
        forecast = e.get("forecast") or e.get("forecastValue") or ""
        previous = e.get("previous") or e.get("previousValue") or ""
        actual   = e.get("actual") or e.get("actualValue") or ""
        if title:
            out.append({
                "date":     date_raw[:16] if date_raw else "",
                "country":  country,
                "title":    title,
                "impact":   impact,
                "forecast": str(forecast) if forecast else "",
                "previous": str(previous) if previous else "",
                "actual":   str(actual)   if actual   else "",
                "source":   source,
            })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. CNBC LATEST NEWS
# ══════════════════════════════════════════════════════════════════════════════

def get_cnbc_news(limit: int = 6) -> list:
    """
    Returns latest CNBC financial news headlines.
    Primary  : RapidAPI CNBC (when subscribed)
    Fallback : DeepCrawl scrape of CNBC RSS
    Returns  : list of {headline, url, datetime, source}
    """
    def _fetch():
        # ── Try RapidAPI CNBC ──
        try:
            r = requests.get(
                "https://cnbc-markets-and-news-data.p.rapidapi.com/news/latest",
                headers={**RAPID_HEADERS, "x-rapidapi-host": "cnbc-markets-and-news-data.p.rapidapi.com"},
                timeout=_TIMEOUT
            )
            if r.status_code == 200:
                raw = r.json()
                items = raw if isinstance(raw, list) else raw.get("data", raw.get("articles", []))
                news = []
                for item in items[:limit]:
                    headline = item.get("title") or item.get("headline") or item.get("name") or ""
                    url      = item.get("url") or item.get("link") or ""
                    dt       = item.get("datePublished") or item.get("date") or item.get("pubDate") or ""
                    if headline:
                        news.append({"headline": headline, "url": url, "datetime": dt[:16], "source": "CNBC"})
                if news:
                    return news
        except Exception:
            pass

        # ── Fallback: DeepCrawl CNBC RSS ──
        try:
            from core.realtime_data import _dc_fetch
            import re
            content = _dc_fetch("https://www.cnbc.com/id/10001147/device/rss/rss.html")
            if content and len(content) > 200:
                # CNBC RSS headlines are run-together — split by sentence boundaries
                # Remove markdown header
                clean = re.sub(r'^#+ [^\n]+\n', '', content).strip()
                # Try to extract linked titles: [Title](url)
                linked = re.findall(r'\[([^\]]{15,120})\]\((https?://[^\)]+)\)', clean)
                if linked:
                    return [{"headline": h, "url": u, "datetime": "", "source": "CNBC"} for h, u in linked[:limit]]
                # Plain text fallback — split sentences ending with dot/question/exclamation
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean)
                news = []
                for s in sentences:
                    s = s.strip()
                    if 20 < len(s) < 200:
                        news.append({"headline": s, "url": "https://www.cnbc.com", "datetime": "", "source": "CNBC"})
                    if len(news) >= limit:
                        break
                return news
        except Exception as e:
            logger.warning("[RapidData] CNBC news fallback failed: %s", e)

        return []

    return _cached("cnbc_news", ttl=1800, fn=_fetch)   # cache 30 minutes


# ══════════════════════════════════════════════════════════════════════════════
# 4. CASH FLOW (QUARTERLY)
# ══════════════════════════════════════════════════════════════════════════════

def get_cashflow(ticker: str) -> dict:
    """
    Returns quarterly cash flow data.
    Primary  : RapidAPI Real-Time Finance Data (when subscribed)
    Fallback : yfinance quarterly_cashflow
    Returns  : {operating_cf, investing_cf, financing_cf, free_cf, quarters:[...]}
    """
    def _fetch():
        # ── Try RapidAPI ──
        try:
            symbol = ticker.replace(".","_") + ":NASDAQ"
            r = requests.get(
                f"https://real-time-finance-data.p.rapidapi.com/company-cash-flow"
                f"?symbol={ticker}%3ANASDAQ&period=QUARTERLY&language=en",
                headers={**RAPID_HEADERS, "x-rapidapi-host": "real-time-finance-data.p.rapidapi.com"},
                timeout=_TIMEOUT
            )
            if r.status_code == 200:
                d = r.json().get("data", {})
                cf = d.get("cash_flow_statement", d)
                if cf:
                    return _normalize_cashflow(cf, "RapidAPI")
        except Exception:
            pass

        # ── Fallback: yfinance ──
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            cf = t.quarterly_cashflow
            if cf is not None and not cf.empty:
                quarters = [str(c)[:10] for c in cf.columns[:4]]
                def _row(name):
                    row = cf[cf.index == name]
                    if row.empty:
                        # Try partial match
                        matches = [i for i in cf.index if name.lower() in str(i).lower()]
                        if matches:
                            row = cf[cf.index == matches[0]]
                    return row.iloc[0].tolist()[:4] if not row.empty else [None]*4

                op  = _row("Operating Cash Flow")
                inv = _row("Investing Cash Flow")
                fin = _row("Financing Cash Flow")
                fcf = _row("Free Cash Flow")
                cap = _row("Capital Expenditure")

                def _fmt(vals):
                    return [round(v/1e9, 2) if v is not None else None for v in vals]

                return {
                    "quarters":      quarters,
                    "operating_cf":  _fmt(op),
                    "investing_cf":  _fmt(inv),
                    "financing_cf":  _fmt(fin),
                    "free_cf":       _fmt(fcf),
                    "capex":         _fmt(cap),
                    "source":        "yfinance",
                    "unit":          "B USD"
                }
        except Exception as e:
            logger.warning("[RapidData] Cash flow failed for %s: %s", ticker, e)

        return {}

    return _cached(f"cashflow_{ticker}", ttl=86400, fn=_fetch)   # cache 24 hours


def _normalize_cashflow(data: dict, source: str) -> dict:
    """Normalize RapidAPI cash flow to standard format."""
    return {
        "operating_cf": data.get("operating_activities") or data.get("operating_cf"),
        "investing_cf": data.get("investing_activities") or data.get("investing_cf"),
        "financing_cf": data.get("financing_activities") or data.get("financing_cf"),
        "free_cf":      data.get("free_cash_flow") or data.get("free_cf"),
        "source":       source,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVENTS CALENDAR (EARNINGS / DIVIDENDS / SPLITS)
# ══════════════════════════════════════════════════════════════════════════════

def get_events_calendar(ticker: str) -> dict:
    """
    Returns upcoming earnings, dividend dates and analyst estimates.
    Primary  : RapidAPI TradingView Events (when subscribed)
    Fallback : yfinance calendar
    Returns  : {earnings_date, ex_div_date, div_date, eps_est_avg, eps_est_high,
                eps_est_low, rev_est_avg, source}
    """
    def _fetch():
        # ── Try RapidAPI TradingView ──
        try:
            r = requests.get(
                f"https://tradingview18.p.rapidapi.com/symbols/get-events-calendar?symbol={ticker}",
                headers={**RAPID_HEADERS, "x-rapidapi-host": "tradingview18.p.rapidapi.com"},
                timeout=_TIMEOUT
            )
            if r.status_code == 200:
                d = r.json()
                events = d.get("data", d.get("events", d))
                if events:
                    return _normalize_events(events, "TradingView (RapidAPI)")
        except Exception:
            pass

        # ── Fallback: yfinance calendar ──
        try:
            import yfinance as yf
            cal = yf.Ticker(ticker).calendar
            if cal:
                def _d(v):
                    if hasattr(v, 'isoformat'):
                        return v.isoformat()
                    if isinstance(v, list) and v:
                        return str(v[0])
                    return str(v) if v else None

                ed_list = cal.get("Earnings Date", [])
                # Filter for upcoming dates only (skip past earnings)
                from datetime import datetime as _dt_cls, timezone as _tz
                _today = _dt_cls.now(_tz.utc).date()
                _future = []
                for _ed in (ed_list or []):
                    try:
                        _d_val = _ed.date() if hasattr(_ed, 'date') else _dt_cls.fromisoformat(str(_ed).split("T")[0]).date()
                        if _d_val >= _today:
                            _future.append(_ed)
                    except Exception:
                        pass
                _selected = _future[0] if _future else (ed_list[0] if ed_list else None)
                return {
                    "earnings_date":  _d(_selected) if _selected else None,
                    "ex_div_date":    _d(cal.get("Ex-Dividend Date")),
                    "div_date":       _d(cal.get("Dividend Date")),
                    "eps_est_avg":    round(cal.get("Earnings Average", 0) or 0, 3) or None,
                    "eps_est_high":   round(cal.get("Earnings High", 0) or 0, 3)    or None,
                    "eps_est_low":    round(cal.get("Earnings Low", 0) or 0, 3)     or None,
                    "rev_est_avg":    cal.get("Revenue Average"),
                    "rev_est_high":   cal.get("Revenue High"),
                    "rev_est_low":    cal.get("Revenue Low"),
                    "source":         "yfinance",
                }
        except Exception as e:
            logger.warning("[RapidData] Events calendar failed for %s: %s", ticker, e)

        return {}

    return _cached(f"events_{ticker}", ttl=21600, fn=_fetch)   # cache 6 hours


def _normalize_events(data, source: str) -> dict:
    """Normalize TradingView events to standard format."""
    if isinstance(data, dict):
        return {
            "earnings_date": data.get("earnings_date") or data.get("nextEarningsDate"),
            "ex_div_date":   data.get("ex_dividend_date") or data.get("exDividendDate"),
            "source":        source,
        }
    return {"source": source}


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAUDI TADAWUL PRICES
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_tadawul_candles(tadawul_id: str) -> list:
    """
    Internal: fetch 1min candles from Tadawul RapidAPI.
    Returns raw candles list (newest-first). Empty list on failure.
    Cached 60s with key tadawul_raw_{tadawul_id}.
    """
    def _do():
        try:
            r = requests.get(
                f"https://saudi-exchange-stocks-tadawul.p.rapidapi.com/v1/stock/get-stock-prices-tadawul/"
                f"?tadawul_id={tadawul_id}&timeframe=1min&from=2024-01-01&to=2024-01-01",
                headers={**RAPID_HEADERS, "x-rapidapi-host": "saudi-exchange-stocks-tadawul.p.rapidapi.com"},
                timeout=12   # increased from 8
            )
            if r.status_code == 200:
                return r.json().get("data", [])
        except Exception as e:
            logger.warning("[RapidData] Tadawul candles fetch failed for %s: %s", tadawul_id, e)
        return []
    return _cached(f"tadawul_raw_{tadawul_id}", ttl=60, fn=_do)


def get_tadawul_quote(tadawul_id: str) -> dict:
    """
    Returns Saudi Tadawul stock live quote.
    Primary  : RapidAPI Saudi Exchange Tadawul — 1min timeframe (confirmed working)
    Fallback : yfinance with .SR suffix
    Returns  : {price, open, high, low, volume, change, change_pct, source}
    Cache    : 60 seconds (near-realtime)
    """
    def _fetch():
        # ── Primary: use shared candle cache (avoids duplicate HTTP calls) ──
        candles = _fetch_tadawul_candles(tadawul_id)
        if candles:
            # data is sorted newest-first → [0] = latest, [-1] = earliest (day open)
            latest   = candles[0]
            earliest = candles[-1]
            _price   = float(latest.get("close") or 0)
            _open    = float(earliest.get("open") or latest.get("open") or _price)
            _high    = max(float(c.get("high", 0)) for c in candles)
            _low     = min(float(c.get("low", 9999)) for c in candles)
            _vol     = sum(int(c.get("volume", 0)) for c in candles)
            _change  = round(_price - _open, 3)
            _chg_pct = round((_change / _open) * 100, 3) if _open else 0
            return {
                "price":      _price,
                "open":       _open,
                "high":       _high,
                "low":        _low,
                "volume":     _vol,
                "change":     _change,
                "change_pct": _chg_pct,
                "source":     "Tadawul RapidAPI",
            }

        # ── Fallback: yfinance fast_info (price-only, ~3x faster than .info) ──
        try:
            import yfinance as yf
            _tk = yf.Ticker(f"{tadawul_id}.SR")
            _fi = _tk.fast_info
            _price = float(getattr(_fi, "last_price",     None) or 0)
            if _price:
                _prev_c  = float(getattr(_fi, "previous_close", None) or 0)
                _chg     = round(_price - _prev_c, 4) if _prev_c else 0
                _chg_pct = round((_chg / _prev_c) * 100, 4) if _prev_c else 0
                return {
                    "price":      _price,
                    "open":       float(getattr(_fi, "open",     None) or 0) or None,
                    "high":       float(getattr(_fi, "day_high", None) or 0) or None,
                    "low":        float(getattr(_fi, "day_low",  None) or 0) or None,
                    "volume":     int(getattr(_fi, "volume",     None) or 0) or None,
                    "change":     _chg,
                    "change_pct": _chg_pct,
                    "name":       None,   # fast_info omits longName — acceptable for price fallback
                    "source":     "yfinance (.SR fast_info)",
                }
        except Exception as e:
            logger.warning("[RapidData] Tadawul yfinance fallback failed for %s: %s", tadawul_id, e)

        return {}

    return _cached(f"tadawul_{tadawul_id}", ttl=60, fn=_fetch)   # 60-second cache (near-realtime)


def get_tadawul_history(tadawul_id: str, timeframe: str = "1min", days: int = 1) -> list:
    """
    Returns OHLCV history candles for a Saudi Tadawul stock.
    Primary  : RapidAPI Tadawul (1min intraday data — current day)
    Fallback : yfinance .SR daily bars

    Args:
        tadawul_id : Tadawul stock ID (e.g. "2222" for Aramco)
        timeframe  : "1min" | "5min" | "15min" | "1hour"
        days       : number of past trading days (currently only today is returned by API)

    Returns: list of {date, open, high, low, close, volume} dicts
    """
    def _fetch():
        # ── Primary: use shared candle cache (1min) — avoids duplicate HTTP calls ──
        candles = _fetch_tadawul_candles(tadawul_id)
        if candles:
            # candles sorted newest-first; reverse for chart display (oldest-first)
            return list(reversed(candles))

        # ── Fallback: yfinance daily bars ──
        try:
            import yfinance as yf
            end   = datetime.now(timezone.utc)
            start = end - timedelta(days=days + 5)
            df = yf.Ticker(f"{tadawul_id}.SR").history(
                start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
            )
            if not df.empty:
                result = []
                for ts, row in df.iterrows():
                    result.append({
                        "date":   str(ts)[:16],
                        "open":   round(float(row["Open"]), 3),
                        "high":   round(float(row["High"]), 3),
                        "low":    round(float(row["Low"]), 3),
                        "close":  round(float(row["Close"]), 3),
                        "volume": int(row["Volume"]),
                    })
                return result[-days:]
        except Exception as e:
            logger.warning("[RapidData] Tadawul yf history failed for %s: %s", tadawul_id, e)

        return []

    return _cached(f"tadawul_hist_{tadawul_id}_{timeframe}", ttl=120, fn=_fetch)


# Saudi stock metadata — Tadawul ID → {name_ar, name_en, sector}
SAUDI_STOCKS = {
    "2222": {"name_en": "Saudi Aramco",         "name_ar": "أرامكو السعودية",        "sector": "Energy"},
    "7010": {"name_en": "STC",                  "name_ar": "الاتصالات السعودية",     "sector": "Telecom"},
    "1120": {"name_en": "Al Rajhi Bank",         "name_ar": "مصرف الراجحي",          "sector": "Banking"},
    "2010": {"name_en": "SABIC",                 "name_ar": "سابك",                  "sector": "Chemicals"},
    "1180": {"name_en": "Al Bilad Bank",         "name_ar": "البلاد",                "sector": "Banking"},
    "2350": {"name_en": "Saudi Kayan",           "name_ar": "سعودي كيان",            "sector": "Chemicals"},
    "4001": {"name_en": "Tawuniya Insurance",    "name_ar": "التعاونية للتأمين",     "sector": "Insurance"},
    "9200": {"name_en": "Elm",                   "name_ar": "علم",                   "sector": "Technology"},
}


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: load all global market data at once
# ══════════════════════════════════════════════════════════════════════════════

def get_market_pulse() -> dict:
    """
    Returns global market sentiment and upcoming events in one call.
    Used by the dashboard endpoint for all tickers.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    jobs = {
        "fear_greed": get_fear_greed,
        "calendar":   lambda: get_forex_calendar(days_ahead=5, min_impact="High"),
        "cnbc_news":  lambda: get_cnbc_news(limit=5),
    }
    results = {}
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(fn): name for name, fn in jobs.items()}
            for future in as_completed(futures, timeout=12):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.warning("[RapidData] market_pulse %s failed: %s", name, e)
                    results[name] = {} if name != "calendar" and name != "cnbc_news" else []
    except Exception as e:
        logger.error("[RapidData] get_market_pulse failed: %s", e)
    return results
