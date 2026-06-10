"""
EisaX News Engine Client
Queries the local news engine (running on localhost:8000/v1/news)
and formats results for injection into stock analysis prompts.

Usage:
    from core.news_engine_client import get_ticker_news, build_news_prompt_block

Features:
- TTL cache (5 min) to avoid hammering the API on repeated queries
- Fast timeout (3s) so it never blocks the analysis pipeline
- Smart query construction: tries ticker, company name, and aliases
- Returns direct + sector + country articles separately
"""

import logging
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ── TTL Cache ──────────────────────────────────────────────────────────────────
_cache_lock  = threading.Lock()
_cache_store: dict = {}   # key → (result_dict, expire_ts)
_CACHE_TTL   = 300        # 5 minutes

def _cache_get(key: str):
    with _cache_lock:
        entry = _cache_store.get(key)
        if entry and time.time() < entry[1]:
            return entry[0]
    return None

def _cache_set(key: str, value):
    with _cache_lock:
        _cache_store[key] = (value, time.time() + _CACHE_TTL)

# ── Ticker → human query mapping ───────────────────────────────────────────────
_TICKER_TO_NAME: dict = {
    # ── UAE ───────────────────────────────────────────────────────────────
    "ADNOCGAS.AE":   "ADNOC",      "ADNOCGAS.DU":   "ADNOC",
    "ADNOCDIST.AE":  "ADNOC",      "ADNOCDIST.DU":  "ADNOC",    # Distribution → parent brand
    "ADNOCDRILLING.AE": "ADNOC",   "ADNOCDRILLING.DU": "ADNOC",
    "EMAAR.DU":      "Emaar",      "EMAAR.AE":      "Emaar",
    "ALDAR.AE":      "Aldar",      "ALDAR.DU":      "Aldar",
    "FAB.AE":        "FAB",        "FAB.DU":        "FAB",
    "ENBD.DU":       "Emirates NBD",
    "TAQA.AE":       "TAQA",       "TAQA.DU":       "TAQA",
    "DEWA.DU":       "DEWA",       "DEWA.AE":       "DEWA",
    "EAND.AE":       "Etisalat",   "EAND.DU":       "Etisalat e&",
    "ADCB.AE":       "ADCB",       "ADCB.DU":       "ADCB",
    "DIB.DU":        "Dubai Islamic Bank",
    "DAMAC.DU":      "DAMAC",
    "AIRARABIA.AE":  "Air Arabia",
    # ── Saudi Arabia ──────────────────────────────────────────────────────
    "2222.SR":       "Aramco Saudi",
    "1120.SR":       "Al Rajhi Bank",
    "2010.SR":       "SABIC",
    "7010.SR":       "STC Saudi Telecom",
    "2030.SR":       "ACWA Power",
    "1111.SR":       "Riyad Bank",
    "1180.SR":       "Saudi Kayan",
    "2350.SR":       "Maaden Saudi",
    "1050.SR":       "Banque Saudi Fransi",
    "4260.SR":       "Mobily Saudi",
    "7020.SR":       "Saudi Arabia Telecom",
    # ── Egypt ─────────────────────────────────────────────────────────────
    "COMI.CA":       "CIB Egypt",
    "TMGH.CA":       "Talaat Moustafa Egypt",
    "EFID.CA":       "EFG Hermes Egypt",
    "DCRC.CA":       "Orascom Egypt",
    "EAST.CA":       "Eastern Company Egypt",
    # ── Kuwait ────────────────────────────────────────────────────────────
    "KFH.KW":        "Kuwait Finance House",
    "NBK.KW":        "National Bank Kuwait",
    "ZAIN.KW":       "Zain Kuwait",
    # ── Qatar ─────────────────────────────────────────────────────────────
    "QNBK.QA":       "QNB Qatar",
    "ORDS.QA":       "Ooredoo Qatar",
    "QEWS.QA":       "Qatar Electricity Water",
    # ── Global equities ───────────────────────────────────────────────────
    "AAPL":          "Apple",
    "MSFT":          "Microsoft",
    "NVDA":          "Nvidia semiconductor",
    "GOOGL":         "Google Alphabet",
    "AMZN":          "Amazon",
    "META":          "Meta Facebook",
    "TSLA":          "Tesla",
    "JPM":           "JPMorgan Chase",
    "GS":            "Goldman Sachs",
    "XOM":           "ExxonMobil oil",
    "CVX":           "Chevron oil",
    # ── Commodities & Crypto ──────────────────────────────────────────────
    "GC=F":          "gold price bullion",
    "SI=F":          "silver price",
    "CL=F":          "crude oil WTI OPEC",
    "NG=F":          "natural gas price",
    "BTC-USD":       "Bitcoin crypto",
    "ETH-USD":       "Ethereum crypto",
    "BTC":           "Bitcoin crypto",
    "ETH":           "Ethereum crypto",
    # ── Market indices ────────────────────────────────────────────────────
    "^TASI":         "Saudi Tadawul stock market",
    "^EGX30":        "Egypt EGX stock market",
    "^GSPC":         "S&P 500",
    "^DJI":          "Dow Jones",
    "^IXIC":         "Nasdaq",
}

_EMPTY = {"direct": [], "sector": [], "country": [], "related": [], "meta": {}}

NEWS_ENGINE_URL = "http://localhost:8000/v1/news"


def get_ticker_news(
    ticker: str,
    company_name: str = "",
    sector: str = "",
    country: str = "",
    limit: int = 8,
) -> dict:
    """
    Query the local news engine for a given ticker/company.
    Returns dict with keys: direct, sector, country, related, meta.
    Each item: {title, url, summary, sentiment, source, published_at}

    Falls back to empty result on timeout or any error — never blocks.
    """
    if not ticker:
        return _EMPTY.copy()

    # Build the search query: prefer human name over raw ticker
    base_ticker = ticker.split(".")[0].split("-")[0].split("=")[0].upper()
    human_name  = (
        _TICKER_TO_NAME.get(ticker.upper())
        or _TICKER_TO_NAME.get(base_ticker)
        or company_name
        or base_ticker
    )

    # Cache key uses both ticker and human name
    cache_key = f"{ticker.upper()}::{human_name}"
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug("[NewsEngine] Cache hit for %s", cache_key)
        return cached

    result = _EMPTY.copy()
    try:
        import httpx as _hx
        # Primary query: by human name (e.g. "ADNOC", "Aramco Saudi Arabia")
        resp = _hx.get(
            NEWS_ENGINE_URL,
            params={"query": human_name, "limit": limit},
            timeout=3.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            result = {
                "direct":  data.get("direct",  []),
                "sector":  data.get("sector",  []),
                "country": data.get("country", []),
                "related": data.get("related", []),
                "meta":    data.get("meta",    {}),
            }

        # If direct results are sparse, try base ticker as a second query
        if len(result["direct"]) < 2 and base_ticker != human_name:
            resp2 = _hx.get(
                NEWS_ENGINE_URL,
                params={"query": base_ticker, "limit": limit},
                timeout=2.0,
            )
            if resp2.status_code == 200:
                data2   = resp2.json()
                direct2 = data2.get("direct", [])
                seen    = {a.get("url") for a in result["direct"]}
                for art in direct2:
                    if art.get("url") not in seen:
                        result["direct"].append(art)
                        seen.add(art.get("url"))

        total = len(result["direct"]) + len(result["sector"]) + len(result["country"])
        logger.info(
            "[NewsEngine] %s → %d direct | %d sector | %d country",
            ticker, len(result["direct"]), len(result["sector"]), len(result["country"])
        )

    except Exception as e:
        logger.debug("[NewsEngine] query failed for %s: %s", ticker, e)
        result = _EMPTY.copy()

    _cache_set(cache_key, result)
    return result


def build_news_prompt_block(news_data: dict, ticker: str = "") -> str:
    """
    Build a structured news context block for injection into AI prompts.
    Returns an empty string if no news available.

    Format fed to the AI:
        FRESH NEWS CONTEXT (from EisaX live news engine):
        COMPANY NEWS:
          - [headline] (source, sentiment)
        SECTOR NEWS:
          - [headline] (source)
        COUNTRY/MACRO NEWS:
          - [headline] (source)
    """
    if not news_data:
        return ""

    direct  = news_data.get("direct",  [])[:5]
    sector  = news_data.get("sector",  [])[:3]
    country = news_data.get("country", [])[:3]

    if not (direct or sector or country):
        return ""

    def _fmt(art: dict) -> str:
        title     = art.get("title", "").strip()
        source    = art.get("source", "").strip()
        sentiment = art.get("sentiment", "").strip()
        if not title:
            return ""
        parts = []
        if source:
            parts.append(source)
        if sentiment and sentiment != "neutral":
            parts.append(f"{'🟢' if sentiment=='bullish' else '🔴'} {sentiment}")
        suffix = f" ({', '.join(parts)})" if parts else ""
        return f"  - {title}{suffix}"

    lines = [
        "FRESH NEWS CONTEXT (EisaX live news engine — published within last 6 hours):",
        "Use these headlines when writing Section 4 (Key Risks) and Section 7 (Why Now).",
        "Reference specific headlines by name — do NOT just list them generically at the end.",
    ]

    if direct:
        lines.append(f"\nCOMPANY NEWS ({ticker.split('.')[0] if ticker else 'Direct'}):")
        for art in direct:
            line = _fmt(art)
            if line:
                lines.append(line)

    if sector:
        meta   = news_data.get("meta", {})
        s_name = meta.get("inferred_sector", "Sector")
        lines.append(f"\nSECTOR NEWS ({s_name}):")
        for art in sector:
            line = _fmt(art)
            if line:
                lines.append(line)

    if country:
        meta   = news_data.get("meta", {})
        c_name = meta.get("inferred_country", "Region")
        lines.append(f"\nCOUNTRY / MACRO NEWS ({c_name}):")
        for art in country:
            line = _fmt(art)
            if line:
                lines.append(line)

    return "\n".join(lines)


def format_news_links(news_data: dict) -> list[dict]:
    """
    Convert news engine results into the finance agent's news_links format:
    [{title: str, url: str, source: str, sentiment: str}]

    Priority: direct > sector > country (deduplicated by URL)
    """
    seen  = set()
    links = []

    for bucket in ("direct", "sector", "country", "related"):
        for art in news_data.get(bucket, []):
            url   = art.get("url", "")
            title = art.get("title", "")
            if not url or not title or url in seen:
                continue
            seen.add(url)
            links.append({
                "title":     title[:120],
                "url":       url,
                "source":    art.get("source", ""),
                "sentiment": art.get("sentiment", "neutral"),
                "summary":   art.get("summary", ""),
            })
            if len(links) >= 8:
                break
        if len(links) >= 8:
            break

    return links
