"""
EisaX News Aggregator
Sources: RSS (free) + NewsAPI + GNews
Cache: 5 minutes per ticker
"""
import os, time, requests, logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# ── 5-minute cache ────────────────────────────────────────────────
_cache: Dict[str, dict] = {}
CACHE_TTL = 300  # 5 minutes

def _cached(key: str) -> list | None:
    if key in _cache:
        if time.time() - _cache[key]["ts"] < CACHE_TTL:
            return _cache[key]["data"]
    return None

def _store(key: str, data: list):
    _cache[key] = {"ts": time.time(), "data": data}

# ── RSS Feeds (مجاني) ─────────────────────────────────────────────
RSS_FEEDS = {
    "global": [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.ft.com/rss/home/us",
    ],
    "crypto": [
        "https://cointelegraph.com/rss",
        "https://coindesk.com/arc/outboundfeeds/rss/",
    ],
    "mena": [
        "https://www.arabnews.com/taxonomy/term/318/feed",
        "https://gulfnews.com/rss/business",
    ],
}

def _fetch_rss(urls: list, limit: int = 5) -> list:
    try:
        import feedparser
    except ImportError:
        return []
    
    items = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:limit]:
                items.append({
                    "title": entry.get("title", ""),
                    "url": entry.get("link", ""),
                    "source": feed.feed.get("title", "RSS"),
                    "publishedAt": entry.get("published", ""),
                })
        except Exception as e:
            logger.warning(f"RSS failed {url}: {e}")
    return items[:limit]

# ── NewsAPI ───────────────────────────────────────────────────────
def _fetch_newsapi(query: str, limit: int = 5) -> list:
    key = os.getenv("NEWS_API_KEY", "")
    if not key:
        return []
    try:
        r = requests.get("https://newsapi.org/v2/everything", params={
            "q": query, "pageSize": limit,
            "sortBy": "publishedAt", "language": "en",
            "apiKey": key,
        }, timeout=8)
        articles = r.json().get("articles", [])
        return [{
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "source": a.get("source", {}).get("name", "NewsAPI"),
            "publishedAt": a.get("publishedAt", ""),
        } for a in articles]
    except Exception as e:
        logger.warning(f"NewsAPI failed: {e}")
        return []

# ── GNews ─────────────────────────────────────────────────────────
def _fetch_gnews(query: str, limit: int = 5) -> list:
    key = os.getenv("GNEWS_API_KEY", "")
    if not key:
        return []
    try:
        r = requests.get("https://gnews.io/api/v4/search", params={
            "q": query, "max": limit,
            "lang": "en", "apikey": key,
        }, timeout=8)
        articles = r.json().get("articles", [])
        return [{
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "source": a.get("source", {}).get("name", "GNews"),
            "publishedAt": a.get("publishedAt", ""),
        } for a in articles]
    except Exception as e:
        logger.warning(f"GNews failed: {e}")
        return []

# ── Main Function ─────────────────────────────────────────────────
def get_news(ticker: str = "", query: str = "", limit: int = 7) -> list:
    cache_key = f"{ticker}:{query}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    # ← الحل: استخدم اسم الشركة مش الـ ticker
    _commodity_map = {
        # Gold futures + ETFs + aliases
        "GC=F": "gold price", "XAUUSD": "gold price", "GOLD": "gold price",
        "GLD": "gold price ETF", "IAU": "gold price ETF", "SGOL": "gold price ETF",
        "GLDM": "gold price ETF", "AAAU": "gold price ETF", "PHYS": "gold price ETF",
        "BAR": "gold price ETF",
        # Silver
        "SI=F": "silver price", "XAGUSD": "silver price", "SILVER": "silver price",
        "SLV": "silver price ETF", "SIVR": "silver price ETF",
        # Oil
        "CL=F": "crude oil price WTI", "BZ=F": "brent oil price",
        "OIL": "crude oil price", "XTIUSD": "oil price",
        "USO": "crude oil ETF price", "BNO": "brent oil ETF price",
    }
    ticker_base = ticker.split('.')[0]
    search_query = query or _commodity_map.get(ticker.upper(), ticker_base)
    # Detect type
    is_crypto = ticker in ["BTC", "ETH", "BNB", "XRP", "SOL", "DOGE"]
    is_mena = any(ticker.endswith(x) for x in [".AD", ".DU", ".SR", ".KW", ".QA"])

    # RSS
    if is_crypto:
        rss = _fetch_rss(RSS_FEEDS["crypto"], limit)
    elif is_mena:
        rss = _fetch_rss(RSS_FEEDS["mena"], limit)
    else:
        rss = _fetch_rss(RSS_FEEDS["global"], limit)

    # APIs
    newsapi = _fetch_newsapi(search_query, limit)
    gnews   = _fetch_gnews(search_query, limit)

    # Combine + deduplicate by title
    seen, results = set(), []
    for item in rss + newsapi + gnews:
        title = item.get("title", "")
        if title and title not in seen:
            seen.add(title)
            results.append(item)

    final = results[:limit]
    _store(cache_key, final)
    return final

# ── Global Market News ────────────────────────────────────────────
def get_global_news(limit: int = 5) -> list:
    return get_news(query="stock market finance economy", limit=limit)
