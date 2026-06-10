"""
core/services/news_filter.py
─────────────────────────────
News deduplication, hard-noise filtering, and GLM/OpenAI relevance scoring.

Public API
──────────
    dedup_buckets(direct, sector, country, related) -> dict
        Cross-bucket deduplication using first 60 chars of title.

    apply_hard_noise(articles, *, field="title") -> list
        Remove articles whose title matches any _HARD_NOISE pattern.

    filter_all_buckets(direct, sector, country, related,
                       asset_name, ticker, sector_name, asset_type,
                       etf_meta=None) -> dict
        Full pipeline: dedup → hard-noise → GLM/OpenAI relevance scoring.
        Returns {"direct": [...], "sector": [...], "country": [...], "related": [...]}.

    build_news_block(buckets, ticker, fund,
                     news_links=None, x_data=None,
                     is_local_ticker=False, is_regional_energy=False) -> str
        Build the markdown news section ready to append to a report.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── Hard-noise keyword list ───────────────────────────────────────────────────
_HARD_NOISE: list[str] = [
    # Military / conflict (non-market-moving)
    "airstrike", "air strike", "military strike", "troops killed", "soldiers killed",
    "soldiers wounded", "bombing", "mortar attack", "drone strike",
    "security capabilities assessment", "launches security", "security assessment",
    "military exercise", "naval exercise", "military operation",
    "peacekeeping", "coup", "civil war", "insurgent",
    # Weather / natural disasters
    "rain alert", "rainfall alert", "flood warning", "flash flood",
    "earthquake", "tsunami", "hurricane", "tornado warning", "typhoon",
    "uae weather", "weather alert", "weather:", "weather forecast",
    "work remotely on friday", "employees to work remotely",
    # Social / ceremonial
    "warm moment", "casual restaurant", "restaurant visit", "family visit",
    "president visits", "royal visit", "official visit to",
    # Sports / entertainment
    "cricket score", "football match", "soccer match", "olympics", "world cup",
    "celebrity", "recipe", "fashion week", "movie review", "tv show",
    "horoscope", "dating", "workout tips", "music album",
    # Non-financial Arabic content
    "يتعافى الإنسان", "ولحافه", "بسريره",
]

# Sector keyword whitelist (used for secondary keyword filter)
_SECTOR_KEYWORDS: dict[str, list[str]] = {
    "technology":    ["tech", "software", "semiconductor", "ai ", "cloud", "chip", "data center"],
    "real estate":   ["real estate", "property", "reit", "construction", "developer", "mortgage", "rent"],
    "energy":        ["oil", "gas", "energy", "petroleum", "opec", "refinery", "crude", "barrel"],
    "financials":    ["bank", "financial", "lending", "credit", "rate", "fed", "interest", "loan"],
    "healthcare":    ["pharma", "biotech", "drug", "fda", "clinical trial", "hospital", "medical"],
    "crypto":        ["bitcoin", "crypto", "blockchain", "defi", "nft", "ethereum", "solana", "token"],
    "saudi":         ["saudi", "aramco", "tadawul", "riyadh", "vision 2030", "pif", "neom"],
    "gcc":           ["gcc", "gulf", "opec", "oil price", "crude", "mena", "middle east market"],
    "global":        ["global market", "world economy", "imf", "world bank", "trade war", "tariff", "g7", "g20"],
}

_COUNTRY_KEYWORDS: dict[str, list[str]] = {
    "saudi":     ["saudi", "aramco", "riyadh", "tadawul", "vision 2030", "pif"],
    "uae":       ["uae", "dubai", "abu dhabi", "adnoc", "emaar", "dfm", "adx"],
    "egypt":     ["egypt", "egx", "cairo", "cib", "nbe", "central bank of egypt"],
    "us":        ["fed", "federal reserve", "us economy", "nasdaq", "s&p", "dow jones", "treasury"],
    "gcc":       ["gcc", "gulf", "opec", "crude", "mena"],
    "global":    ["global", "world", "imf", "world bank", "g7", "g20"],
}


# ── Deduplication ─────────────────────────────────────────────────────────────

def dedup_buckets(
    direct: list,
    sector: list,
    country: list,
    related: list,
) -> dict[str, list]:
    """
    Cross-bucket deduplication using first 60 chars of lowercased title.
    Processes buckets in priority order (direct first) so higher-priority
    buckets keep the article.
    """
    seen: set[str] = set()

    def _dedup(articles: list) -> list:
        out = []
        for a in articles:
            key = (a.get("title") or "")[:60].lower().strip()
            if key and key not in seen:
                seen.add(key)
                out.append(a)
        return out

    return {
        "direct":  _dedup(direct),
        "sector":  _dedup(sector),
        "country": _dedup(country),
        "related": _dedup(related),
    }


# ── Hard-noise filter ─────────────────────────────────────────────────────────

def apply_hard_noise(articles: list, *, field: str = "title") -> list:
    """Remove articles whose *field* matches any hard-noise keyword."""
    result = []
    for a in articles:
        text = (a.get(field) or "").lower()
        if not any(n in text for n in _HARD_NOISE):
            result.append(a)
    return result


# ── GLM/OpenAI relevance filter ───────────────────────────────────────────────

def _glm_filter_bucket(
    articles: list,
    glm_client: Any,
    asset_name: str,
    ticker: str,
    sector: str,
    asset_type: str,
    bucket: str,
    min_score: int = 60,
) -> list:
    """Apply GLM/OpenAI relevance scoring to one bucket."""
    if not articles:
        return articles
    try:
        return glm_client.filter_news_relevance(
            articles, asset_name, ticker, sector, asset_type,
            bucket=bucket, min_score=min_score,
        )
    except Exception as exc:
        logger.warning("[NewsFilter] GLM filter failed for %s/%s: %s", ticker, bucket, exc)
        return articles


# ── Full pipeline ─────────────────────────────────────────────────────────────

def filter_all_buckets(
    direct: list,
    sector: list,
    country: list,
    related: list,
    asset_name: str,
    ticker: str,
    sector_name: str,
    asset_type: str,
    etf_meta: dict | None = None,
) -> dict[str, list]:
    """
    Full pipeline:
      1. Cross-bucket dedup
      2. Hard-noise removal
      3. GLM/OpenAI relevance scoring (each bucket scored independently)

    Returns dict with keys: direct, sector, country, related.
    If GLM fails, buckets are returned as-is (no data loss).
    """
    # Step 1 — dedup
    buckets = dedup_buckets(direct, sector, country, related)

    # Step 2 — hard noise (applied before GLM to save API tokens)
    for k in buckets:
        buckets[k] = apply_hard_noise(buckets[k])

    # Step 3 — GLM/OpenAI relevance (all 4 buckets in parallel)
    try:
        from concurrent.futures import ThreadPoolExecutor as _TPEx, as_completed as _as_completed
        from core.glm_client import GLMClient as _GLMClient
        glm = _GLMClient()
        _etype = (
            etf_meta.get("etf_type", "etf") if etf_meta
            else ("crypto" if ticker.endswith("-USD") else "stock")
        )
        _bucket_configs = [
            ("direct",  buckets["direct"],  60),
            ("sector",  buckets["sector"],  60),
            ("country", buckets["country"], 70),
            ("related", buckets["related"], 60),
        ]
        with _TPEx(max_workers=4) as _exe:
            _futs = {
                _exe.submit(
                    _glm_filter_bucket, arts, glm, asset_name, ticker, sector_name, _etype, bname, mscore
                ): bname
                for bname, arts, mscore in _bucket_configs
            }
            for _fut in _as_completed(_futs):
                _bname = _futs[_fut]
                try:
                    buckets[_bname] = _fut.result()
                except Exception as _be:
                    logger.warning("[NewsFilter] bucket %s failed for %s: %s", _bname, ticker, _be)
    except Exception as exc:
        logger.warning("[NewsFilter] GLM init failed for %s: %s — using unfiltered news", ticker, exc)

    return buckets


# ── Markdown builder ──────────────────────────────────────────────────────────

def _sentiment_icon(sentiment: str) -> str:
    return {"bullish": "🟢", "bearish": "🔴"}.get(sentiment, "⚪")


def build_news_block(
    buckets: dict[str, list],
    ticker: str,
    fund: dict,
    news_links: list | None = None,
    x_data: dict | None = None,
    is_local_ticker: bool = False,
    is_regional_energy: bool = False,
) -> str:
    """
    Build the full markdown news + X sentiment block.

    Parameters
    ──────────
    buckets           : output of filter_all_buckets()
    ticker            : resolved ticker symbol
    fund              : fundamentals dict (for company_name etc.)
    news_links        : flat fallback list (yfinance/FMP)
    x_data            : Grok sentiment dict
    is_local_ticker   : True for MENA regional tickers
    is_regional_energy: True for regional energy names
    """
    direct  = buckets.get("direct",  [])
    sector  = buckets.get("sector",  [])
    country = buckets.get("country", [])
    related = buckets.get("related", [])

    has_engine_news = bool(direct or sector or country or related)

    news_block = ""

    if has_engine_news:
        news_block = "\n\n---\n📰 **Latest News** *(EisaX live news engine)*\n"

        if direct:
            co_label = ticker.split(".")[0]
            news_block += f"\n**📌 {co_label} — Company News**\n"
            for a in direct[:5]:
                ico = _sentiment_icon(a.get("sentiment", "neutral"))
                src = f" *({a['source']})*" if a.get("source") else ""
                url = a.get("url", "")
                ttl = a.get("title", "")
                news_block += f"- {ico} [{ttl}]({url}){src}\n" if url else f"- {ico} {ttl}{src}\n"

        sector_name = fund.get("sector", "Sector") or "Sector"
        if sector:
            news_block += f"\n**🏭 {sector_name} — Sector News**\n"
            for a in sector[:3]:
                url = a.get("url", "")
                ttl = a.get("title", "")
                src = f" *({a['source']})*" if a.get("source") else ""
                news_block += f"- [{ttl}]({url}){src}\n" if url else f"- {ttl}{src}\n"

        if country:
            _country_code = (
                "Saudi Arabia" if ticker.upper().endswith(".SR") else
                "UAE" if ticker.upper().endswith((".AE", ".DU")) else
                "Egypt" if ticker.upper().endswith(".CA") else
                "Market"
            )
            news_block += f"\n**🌍 {_country_code} — Market News**\n"
            for a in country[:3]:
                url = a.get("url", "")
                ttl = a.get("title", "")
                src = f" *({a['source']})*" if a.get("source") else ""
                news_block += f"- [{ttl}]({url}){src}\n" if url else f"- {ttl}{src}\n"

    elif news_links:
        # Fallback: flat list with GLM filter applied
        try:
            from core.glm_client import GLMClient as _GLMC
            g = _GLMC()
            news_links = g.filter_news_relevance(
                news_links,
                fund.get("company_name") or ticker,
                ticker,
                fund.get("sector", "General") or "General",
                "crypto" if ticker.endswith("-USD") else "stock",
            )
        except Exception as exc:
            logger.warning(
                "[NewsFilter] flat fallback GLM filter failed for %s: %s — using raw news_links",
                ticker,
                exc,
            )
        news_block = "\n\n---\n📰 **Latest News** *(live at time of query)*\n"
        for n in news_links:
            url = n.get("url", "")
            ttl = n.get("title", "")
            news_block += f"- [{ttl}]({url})\n" if url else f"- {ttl}\n"

    else:
        # No news at all — mandatory fallback message
        if is_regional_energy:
            msg = (
                "No live news fetched. Monitor: "
                "**Argaam**, **Mubasher**, **Reuters Energy**, "
                "and OPEC+ statements for real-time catalysts."
            )
        elif is_local_ticker:
            msg = (
                "No live news fetched. Check **Argaam**, **The National**, "
                "or the issuer's investor relations page for latest updates."
            )
        else:
            msg = (
                "No live news fetched. Check **Bloomberg**, **Reuters**, "
                "or **Seeking Alpha** for the latest updates."
            )
        news_block = f"\n\n---\n📰 **Latest News**\n> ⚠️ {msg}\n"

    # ── X / Twitter Posts Block ───────────────────────────────────────────────
    x_posts_block = ""
    xp_list = (x_data or {}).get("top_posts", [])
    if xp_list and (x_data or {}).get("source") == "grok-live":
        xs_label = (x_data or {}).get("sentiment", "")
        xs_score = (x_data or {}).get("score", 0.0)
        xs_themes = ", ".join((x_data or {}).get("themes", [])[:3])
        xs_breaking = (x_data or {}).get("breaking", "")

        _sent_emoji = {"Bullish": "🟢", "Bearish": "🔴", "Mixed": "🟡"}.get(xs_label, "🟡")
        x_posts_block = (
            f"\n---\n📱 **X / Twitter Sentiment** *(Grok live · last 48h)*\n"
            f"> {_sent_emoji} **{xs_label}** (score: {xs_score:+.2f}) · {xs_themes}\n"
        )
        if xs_breaking:
            x_posts_block += f"> ⚡ **BREAKING:** {xs_breaking}\n"
        for post in xp_list[:4]:
            _p_sent = {"bullish": "🟢", "bearish": "🔴"}.get(
                (post.get("sentiment") or "").lower(), "🟡"
            )
            _p_handle = post.get("handle", "")
            _p_likes  = post.get("likes", 0)
            _p_date   = post.get("date", "")
            _p_text   = post.get("text", "")
            x_posts_block += (
                f"- {_p_sent} **@{_p_handle}** *({_p_likes} likes)* · {_p_date}: \"{_p_text}\"\n"
            )

    return news_block + x_posts_block
