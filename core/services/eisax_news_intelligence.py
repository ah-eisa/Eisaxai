"""
eisax_news_intelligence.py
──────────────────────────
Drop-in news layer for EisaX market_updates.py.

Adds:
  fetch_news_context()           -> NewsContext (dataclass)
  inject_news_into_daily_prompt  -> str (upgraded prompt)
  inject_news_into_weekly_prompt -> str (upgraded prompt)
  generate_gcc_intelligence()    -> dict (standalone GCC section)

No new dependencies beyond the stdlib + requests (already used).
Feedparser is optional — falls back to raw RSS parsing if not installed.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
import json

logger = logging.getLogger(__name__)

# ── RSS Feed Registry ─────────────────────────────────────────────────────────
# Curated feeds — all publicly accessible, no auth required.
# Grouped by theme so we can pick intelligently per report type.

_FEEDS: dict[str, list[dict]] = {

    # ── Global Macro / Markets ────────────────────────────────────────────────
    "global_macro": [
        {"url": "https://feeds.reuters.com/reuters/businessNews",        "source": "Reuters Business"},
        {"url": "https://feeds.reuters.com/reuters/companyNews",         "source": "Reuters Companies"},
        {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "source": "NYT Business"},
        {"url": "https://www.ft.com/rss/home",                           "source": "Financial Times"},
        {"url": "https://feeds.marketwatch.com/marketwatch/topstories/", "source": "MarketWatch"},
        {"url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "source": "CNBC Markets"},
        {"url": "https://www.cnbc.com/id/20910258/device/rss/rss.html",  "source": "CNBC Economy"},
        {"url": "https://feeds.bloomberg.com/markets/news.rss",          "source": "Bloomberg Markets"},
    ],

    # ── Central Banks / Rates ─────────────────────────────────────────────────
    "rates_macro": [
        {"url": "https://feeds.reuters.com/reuters/businessNews",        "source": "Reuters Business"},
        {"url": "https://www.cnbc.com/id/20910258/device/rss/rss.html",  "source": "CNBC Economy"},
        {"url": "https://www.federalreserve.gov/feeds/press_all.xml",    "source": "Federal Reserve"},
    ],

    # ── GCC / MENA ────────────────────────────────────────────────────────────
    "gcc_mena": [
        {"url": "https://www.arabnews.com/rss.xml",                      "source": "Arab News"},
        {"url": "https://gulfnews.com/rss/business",                     "source": "Gulf News Business"},
        {"url": "https://www.thenationalnews.com/rss/business.xml",      "source": "The National"},
        {"url": "https://www.zawya.com/rss/economy.xml",                 "source": "Zawya Economy"},
        {"url": "https://www.zawya.com/rss/markets.xml",                 "source": "Zawya Markets"},
        {"url": "https://www.reuters.com/world/middle-east/rss.xml",     "source": "Reuters MENA"},
        {"url": "https://english.mubasher.info/rss",                     "source": "Mubasher"},
    ],

    # ── Energy / Oil ──────────────────────────────────────────────────────────
    "energy": [
        {"url": "https://feeds.reuters.com/reuters/oilReport",           "source": "Reuters Energy"},
        {"url": "https://oilprice.com/rss/main",                         "source": "OilPrice.com"},
        {"url": "https://www.cnbc.com/id/10000045/device/rss/rss.html",  "source": "CNBC Energy"},
    ],

    # ── Crypto ────────────────────────────────────────────────────────────────
    "crypto": [
        {"url": "https://cointelegraph.com/rss",                         "source": "CoinTelegraph"},
        {"url": "https://decrypt.co/feed",                               "source": "Decrypt"},
    ],

    # ── Geopolitics ───────────────────────────────────────────────────────────
    "geopolitics": [
        {"url": "https://feeds.reuters.com/Reuters/worldNews",           "source": "Reuters World"},
        {"url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml","source": "NYT World"},
        {"url": "https://www.bbc.co.uk/news/world/rss.xml",              "source": "BBC World"},
    ],
}

# ── Market-relevant keywords for filtering ────────────────────────────────────
_MARKET_KEYWORDS = {
    "must_include": [
        "fed", "rate", "inflation", "cpi", "gdp", "recession", "yield",
        "s&p", "nasdaq", "dow", "earnings", "opec", "oil", "gold", "bitcoin",
        "dollar", "treasury", "bond", "equity", "stock", "market", "bank",
        "tasi", "dfm", "tadawul", "saudi", "uae", "dubai", "abu dhabi",
        "iran", "hormuz", "geopolit", "tariff", "trade", "china", "powell",
        "ecb", "boe", "crypto", "etf", "ipo", "merger", "acquisition",
        "quarter", "profit", "revenue", "guidance", "outlook"
    ],
    "gcc_specific": [
        "saudi", "uae", "dubai", "abu dhabi", "qatar", "kuwait", "bahrain",
        "oman", "aramco", "adnoc", "emaar", "aldar", "mada", "nbd", "fab",
        "tasi", "dfm", "adx", "tadawul", "vision 2030", "neom", "gcc",
        "mena", "gulf", "opec+", "riyadh", "doha"
    ]
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    published: str
    category: str
    is_gcc: bool = False


@dataclass
class NewsContext:
    global_headlines: list[NewsItem] = field(default_factory=list)
    gcc_headlines: list[NewsItem] = field(default_factory=list)
    energy_headlines: list[NewsItem] = field(default_factory=list)
    rates_headlines: list[NewsItem] = field(default_factory=list)
    geopolitical_headlines: list[NewsItem] = field(default_factory=list)
    crypto_headlines: list[NewsItem] = field(default_factory=list)
    fetched_at: str = ""
    total_items: int = 0

    def is_empty(self) -> bool:
        return self.total_items == 0

    def top_global(self, n: int = 5) -> list[NewsItem]:
        return self.global_headlines[:n]

    def top_gcc(self, n: int = 4) -> list[NewsItem]:
        return self.gcc_headlines[:n]

    def top_energy(self, n: int = 3) -> list[NewsItem]:
        return self.energy_headlines[:n]

    def format_for_prompt(self, max_global: int = 6, max_gcc: int = 4, max_energy: int = 3) -> str:
        """Format news context as clean text block for AI prompt injection."""
        lines = []

        if self.global_headlines:
            lines.append("=== GLOBAL MARKET HEADLINES (TODAY) ===")
            for item in self.global_headlines[:max_global]:
                lines.append(f"[{item.source}] {item.title}")
                if item.summary and len(item.summary) > 20:
                    # Truncate summary to keep prompt lean
                    summ = item.summary[:200].rstrip() + ("…" if len(item.summary) > 200 else "")
                    lines.append(f"  → {summ}")
            lines.append("")

        if self.rates_headlines:
            lines.append("=== RATES & CENTRAL BANK ===")
            for item in self.rates_headlines[:2]:
                lines.append(f"[{item.source}] {item.title}")
            lines.append("")

        if self.energy_headlines:
            lines.append("=== ENERGY / OIL ===")
            for item in self.energy_headlines[:max_energy]:
                lines.append(f"[{item.source}] {item.title}")
            lines.append("")

        if self.geopolitical_headlines:
            lines.append("=== GEOPOLITICAL ===")
            for item in self.geopolitical_headlines[:2]:
                lines.append(f"[{item.source}] {item.title}")
            lines.append("")

        if self.gcc_headlines:
            lines.append("=== GCC / MENA (CRITICAL FOR REGIONAL VIEW) ===")
            for item in self.gcc_headlines[:max_gcc]:
                lines.append(f"[{item.source}] {item.title}")
                if item.summary and len(item.summary) > 20:
                    summ = item.summary[:180].rstrip() + ("…" if len(item.summary) > 180 else "")
                    lines.append(f"  → {summ}")
            lines.append("")

        if self.crypto_headlines:
            lines.append("=== CRYPTO ===")
            for item in self.crypto_headlines[:2]:
                lines.append(f"[{item.source}] {item.title}")
            lines.append("")

        if not lines:
            return "NEWS CONTEXT: No headlines available — rely on price data and regime signals."

        lines.insert(0, f"NEWS CONTEXT (fetched {self.fetched_at}) — USE THESE TO GROUND YOUR ANALYSIS:\n")
        return "\n".join(lines)


# ── RSS Fetching ──────────────────────────────────────────────────────────────

def _fetch_rss(url: str, source: str, timeout: int = 8) -> list[dict]:
    """
    Fetch and parse RSS feed. Returns list of {title, summary, published} dicts.
    Tries feedparser first, falls back to regex-based parsing.
    """
    try:
        import requests
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "EisaX-Intelligence/2.0 (market research bot)",
            "Accept": "application/rss+xml, application/xml, text/xml",
        })
        resp.raise_for_status()
        content = resp.text
    except Exception as exc:
        logger.debug("[news] Failed to fetch %s: %s", url, exc)
        return []

    # Try feedparser first (cleaner parsing)
    try:
        import feedparser
        feed = feedparser.parse(content)
        items = []
        for entry in feed.entries[:15]:
            title = getattr(entry, "title", "").strip()
            summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
            # Strip HTML tags from summary
            summary = re.sub(r"<[^>]+>", " ", summary).strip()
            summary = re.sub(r"\s+", " ", summary)
            published = getattr(entry, "published", "") or getattr(entry, "updated", "")
            if title:
                items.append({"title": title, "summary": summary[:300], "published": published, "source": source})
        return items
    except ImportError:
        pass

    # Fallback: regex-based XML parsing
    items = []
    pattern = re.compile(
        r"<item[^>]*>.*?<title[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>.*?(?:<description[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>)?",
        re.DOTALL | re.IGNORECASE
    )
    for m in pattern.finditer(content):
        title = re.sub(r"<[^>]+>", "", m.group(1) or "").strip()
        desc = re.sub(r"<[^>]+>", " ", m.group(2) or "").strip()
        desc = re.sub(r"\s+", " ", desc)
        if title and len(title) > 5:
            items.append({"title": title, "summary": desc[:300], "published": "", "source": source})
        if len(items) >= 12:
            break
    return items


def _is_market_relevant(title: str, category: str = "global") -> bool:
    """Filter headlines to only market-moving content."""
    lower = title.lower()

    # Always relevant categories
    if category in ("rates_macro", "energy", "gcc_mena", "crypto", "geopolitics"):
        return True

    # Must contain at least one market keyword
    return any(kw in lower for kw in _MARKET_KEYWORDS["must_include"])


def _is_gcc_relevant(title: str, summary: str = "") -> bool:
    """Check if headline is GCC/MENA specific."""
    text = (title + " " + summary).lower()
    return any(kw in text for kw in _MARKET_KEYWORDS["gcc_specific"])


def _deduplicate(items: list[NewsItem], seen_titles: set) -> list[NewsItem]:
    """Remove near-duplicate headlines."""
    out = []
    for item in items:
        # Simple dedup: first 6 words as fingerprint
        fp = " ".join(item.title.lower().split()[:6])
        if fp not in seen_titles:
            seen_titles.add(fp)
            out.append(item)
    return out


# ── Main fetcher ──────────────────────────────────────────────────────────────

def fetch_news_context(
    include_gcc: bool = True,
    include_energy: bool = True,
    include_crypto: bool = True,
    timeout_per_feed: int = 7,
    max_feeds_per_category: int = 3,
) -> NewsContext:
    """
    Fetch and categorize news from multiple RSS sources.
    Returns a NewsContext object ready to inject into AI prompts.

    Designed to be fast: parallel-friendly, fails gracefully.
    Total expected time: 3-8 seconds with good network.
    """
    import concurrent.futures

    ctx = NewsContext(fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    seen: set = set()

    # Define fetch jobs
    jobs: list[tuple[str, str, str]] = []  # (category, url, source)

    for cat, feeds in _FEEDS.items():
        limit = max_feeds_per_category
        for feed in feeds[:limit]:
            jobs.append((cat, feed["url"], feed["source"]))

    # Fetch concurrently
    raw_by_cat: dict[str, list[dict]] = {cat: [] for cat in _FEEDS}

    def _fetch_job(job: tuple) -> tuple[str, list[dict]]:
        cat, url, source = job
        items = _fetch_rss(url, source, timeout=timeout_per_feed)
        return cat, items

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            futures = {executor.submit(_fetch_job, job): job for job in jobs}
            for future in concurrent.futures.as_completed(futures, timeout=15):
                try:
                    cat, items = future.result(timeout=5)
                    raw_by_cat[cat].extend(items)
                except Exception:
                    pass
    except Exception as exc:
        logger.warning("[news] Concurrent fetch error: %s", exc)

    # Process and categorize
    all_gcc_items: list[NewsItem] = []
    all_global_items: list[NewsItem] = []
    all_rates_items: list[NewsItem] = []
    all_energy_items: list[NewsItem] = []
    all_geo_items: list[NewsItem] = []
    all_crypto_items: list[NewsItem] = []

    for cat, raw_items in raw_by_cat.items():
        for raw in raw_items:
            title = raw.get("title", "").strip()
            if not title or len(title) < 10:
                continue

            summary = raw.get("summary", "").strip()
            source = raw.get("source", "")
            published = raw.get("published", "")

            if not _is_market_relevant(title, cat):
                continue

            is_gcc = _is_gcc_relevant(title, summary) or cat == "gcc_mena"

            item = NewsItem(
                title=title,
                summary=summary,
                source=source,
                published=published,
                category=cat,
                is_gcc=is_gcc,
            )

            # Route to appropriate bucket
            if is_gcc or cat == "gcc_mena":
                all_gcc_items.append(item)
            if cat == "rates_macro":
                all_rates_items.append(item)
            elif cat == "energy":
                all_energy_items.append(item)
            elif cat == "crypto":
                all_crypto_items.append(item)
            elif cat == "geopolitics":
                all_geo_items.append(item)
            else:
                all_global_items.append(item)

    # Deduplicate and store
    ctx.global_headlines      = _deduplicate(all_global_items[:20], seen)[:10]
    ctx.gcc_headlines         = _deduplicate(all_gcc_items[:15], seen)[:8]
    ctx.energy_headlines      = _deduplicate(all_energy_items[:10], seen)[:5]
    ctx.rates_headlines       = _deduplicate(all_rates_items[:8], seen)[:4]
    ctx.geopolitical_headlines= _deduplicate(all_geo_items[:8], seen)[:4]
    ctx.crypto_headlines      = _deduplicate(all_crypto_items[:6], seen)[:3]

    ctx.total_items = (
        len(ctx.global_headlines) + len(ctx.gcc_headlines) +
        len(ctx.energy_headlines) + len(ctx.rates_headlines) +
        len(ctx.geopolitical_headlines) + len(ctx.crypto_headlines)
    )

    logger.info(
        "[news] Context: %d global | %d GCC | %d energy | %d rates | %d geo | %d crypto",
        len(ctx.global_headlines), len(ctx.gcc_headlines),
        len(ctx.energy_headlines), len(ctx.rates_headlines),
        len(ctx.geopolitical_headlines), len(ctx.crypto_headlines),
    )
    return ctx


# ── Prompt Injectors ──────────────────────────────────────────────────────────

def inject_news_into_daily_prompt(
    base_prompt: str,
    news: NewsContext,
    moves_summary: dict,
    regime: str,
    conf: str,
    fg: dict,
    today: str,
) -> str:
    """
    Upgrade the daily AI prompt with real news context + cross-asset correlation instructions.
    This is the core upgrade over the existing system.
    """
    news_block = news.format_for_prompt(max_global=6, max_gcc=4, max_energy=3)

    # Build a cross-asset correlation hint based on actual data
    correlation_hints = _build_correlation_hints(moves_summary)

    upgraded_prompt = f"""You are EisaX — institutional AI investment intelligence used by portfolio managers.
Your edge: you combine REAL price data + REAL news context + regime signals into insights no human analyst can produce in under 3 hours.

TODAY: {today}
REGIME: {regime} (confidence: {conf})
FEAR & GREED: {fg.get('score', 50)}/100 ({fg.get('rating', 'Neutral')})

{news_block}

REAL PRICE DATA (yfinance — verified):
{json.dumps(moves_summary, indent=2)}

CROSS-ASSET CORRELATIONS DETECTED:
{correlation_hints}

Generate a Daily Market Pulse as valid JSON only.

CRITICAL RULES FOR THIS UPGRADE vs GROK/GENERIC AI:
1. what_matters_now: Reference SPECIFIC headlines from the news context + connect to price data. Never generic.
2. key_moves: Explain WHY each asset moved using SPECIFIC news context above — not "risk sentiment."
3. why_now: Must reference at least ONE specific news catalyst from above + ONE price signal.
4. gcc_note: Add a 2-sentence GCC-specific observation using the GCC headlines above.
5. Never write: "amid uncertainty" / "markets showed resilience" / "investor confidence"
6. Every insight must be falsifiable: a reader must be able to act on it or disagree with it.

Return ONLY this JSON (no markdown fences):
{{
  "date": "{today}",
  "market_regime": "{regime}",
  "regime_confidence": "{conf}",
  "what_matters_now": [
    "<News-grounded insight: specific headline + price reaction + portfolio implication>",
    "<Cross-asset signal that retail misses — connect 2+ assets from the price data>",
    "<One forward-looking observation based on news + regime: what happens NEXT?>"
  ],
  "key_moves": [
    {{"asset": "<name>", "move": "<±X.X% 1d>", "reason": "<SPECIFIC news cause — not 'risk sentiment'>"}},
    {{"asset": "<name>", "move": "<±X.X% 1d>", "reason": "<specific cause>"}},
    {{"asset": "<name>", "move": "<±X.X% 1d>", "reason": "<specific cause>"}},
    {{"asset": "<name>", "move": "<±X.X% 1d>", "reason": "<specific cause>"}},
    {{"asset": "<name>", "move": "<±X.X% 1d>", "reason": "<specific cause>"}}
  ],
  "eisax_view": {{
    "stance": "<Tactical BUY|HOLD|REDUCE RISK>",
    "overweight_assets": ["<specific>"],
    "underweight_assets": ["<specific>"],
    "neutral_assets": ["<specific>"],
    "focus": "<4-6 words>",
    "horizon": "<short-term|tactical|swing|defensive>"
  }},
  "why_now": "<2 sentences MAX: 1 specific news catalyst + 1 price signal. Zero filler.>",
  "gcc_note": "<2 sentences: GCC-specific insight from the MENA headlines above. Include TASI/DFM levels if available.>",
  "what_invalidates": [
    "<price level from data + specific threshold>",
    "<volatility trigger with VIX level>",
    "<macro event or data release that changes regime>"
  ],
  "tactical_positioning": "<1-2 lines: concrete portfolio action — name actual instruments>",
  "next_triggers": ["<specific event/level/date>", "<specific event>", "<level to watch>"],
  "fear_greed_index": {fg.get('score', 50)},
  "news_sources_used": ["<source1>", "<source2>", "<source3>"]
}}
"""
    return upgraded_prompt


def inject_news_into_weekly_prompt(
    news: NewsContext,
    moves_summary: dict,
    regime: str,
    conf: str,
    fg: dict,
    week_range: str,
    stance: dict,
    invali: list,
) -> str:
    """
    Upgraded weekly prompt — news-grounded with deep GCC section.
    """
    news_block = news.format_for_prompt(max_global=7, max_gcc=5, max_energy=4)
    correlation_hints = _build_correlation_hints(moves_summary)

    return f"""You are EisaX — institutional AI investment intelligence. Style: Goldman Sachs strategy note. Decisive.
Your weekly brief must be impossible to replicate without your combination of: real price data + real news + regime engine.

WEEK: {week_range}
REGIME: {regime} (confidence: {conf})
FEAR & GREED: {fg.get('score', 50)}/100 ({fg.get('rating', 'Neutral')})

{news_block}

REAL PRICE DATA (full week):
{json.dumps(moves_summary, indent=2)}

CROSS-ASSET SIGNALS:
{correlation_hints}

PRE-COMPUTED STANCE: {json.dumps(stance)}
PRE-COMPUTED INVALIDATION: {json.dumps(invali)}

WHAT SEPARATES THIS FROM GROK/CHATGPT REPORTS:
- Every claim anchored to a SPECIFIC headline OR a SPECIFIC price level — not both generic
- GCC section must be substantive: TASI, DFM, ADNOC/Aramco, oil linkage, Vision 2030 context
- Highest conviction idea must include: instrument + catalyst + entry context + invalidation level
- Regional views must disagree with consensus where data supports it

Return ONLY this JSON:
{{
  "week_range": "{week_range}",
  "market_summary": "<3 sharp sentences: what DROVE markets this week — each sentence must reference a specific news event OR price move. No vague causation.>",
  "positioning": "<How to be positioned NOW — name overweight/underweight asset classes explicitly with WHY>",
  "asset_allocation_view": {{
    "equities": "<Overweight|Neutral|Underweight>",
    "crypto": "<Overweight|Neutral|Underweight>",
    "metals": "<Overweight|Neutral|Underweight>",
    "commodities": "<Overweight|Neutral|Underweight>",
    "cash": "<Overweight|Neutral|Underweight>"
  }},
  "regional_view": {{
    "US": "<Specific view on US equities with price context — reference actual index levels>",
    "GCC": "<2-sentence GCC view: TASI/DFM direction + oil linkage + any specific corporate/sovereign news from above>",
    "Egypt": "<1-sentence EM view with EGX30 context + dollar impact>"
  }},
  "winners_losers": {{
    "winners": ["<asset> <±X.X%>", "<asset> <±X.X%>", "<asset> <±X.X%>"],
    "losers": ["<asset> <±X.X%>", "<asset> <±X.X%>", "<asset> <±X.X%>"]
  }},
  "highest_conviction_opportunity": "<ONE specific trade: instrument + specific catalyst from news + entry context (price level or condition) + time horizon + what invalidates it>",
  "key_risks": [
    "<Risk 1: specific event/level from news context + why it matters NOW for portfolios>",
    "<Risk 2: second specific risk with data>",
    "<Risk 3: GCC-specific or regional risk>"
  ],
  "what_changes_this_view": ["<specific price level or macro release>", "<second specific trigger>"],
  "portfolio_angle": "<2-3 sentences: concrete portfolio action across asset classes this week>",
  "eisax_verdict": "<1 sentence starting with action verb: Reduce/Add/Hold/Rotate + specific instruction>",
  "gcc_intelligence": {{
    "tasi_view": "<TASI-specific observation with direction and oil context>",
    "dfm_view": "<DFM and UAE equity context>",
    "oil_gcc_link": "<How this week's oil move affects GCC earnings visibility>",
    "sovereign_wealth": "<Any SWF or Vision 2030 news this week>"
  }}
}}
"""


# ── Cross-Asset Correlation Engine ────────────────────────────────────────────

def _build_correlation_hints(moves_summary: dict) -> str:
    """
    Detect and describe cross-asset correlations from live price data.
    This is EisaX's deterministic edge — no AI required, no hallucination.
    """
    hints = []

    spy = moves_summary.get("SPY", {})
    qqq = moves_summary.get("QQQ", {})
    vix = moves_summary.get("^VIX", {})
    gld = moves_summary.get("GLD", {})
    uso = moves_summary.get("USO", {})
    tnx = moves_summary.get("^TNX", {})
    btc = moves_summary.get("BTC-USD", {})
    uup = moves_summary.get("UUP", {})
    tasi = moves_summary.get("^TASI", {})

    spy_d1 = spy.get("d1_pct", 0) or 0
    vix_px = vix.get("price", 20) or 20
    gld_d1 = gld.get("d1_pct", 0) or 0
    uso_d1 = uso.get("d1_pct", 0) or 0
    tnx_px = tnx.get("price", 4.25) or 4.25
    btc_d1 = btc.get("d1_pct", 0) or 0
    uup_d1 = uup.get("d1_pct", 0) or 0
    qqq_d1 = qqq.get("d1_pct", 0) or 0
    tasi_d1 = tasi.get("d1_pct", 0) or 0

    # SPY vs VIX divergence
    if spy_d1 > 0.5 and vix_px > 20:
        hints.append(f"⚡ DIVERGENCE: SPY +{spy_d1:.1f}% but VIX still elevated at {vix_px:.1f} — equity rally without fear premium collapsing = institutional hedging persists")
    elif spy_d1 < -0.5 and vix_px < 18:
        hints.append(f"⚡ DIVERGENCE: SPY {spy_d1:.1f}% but VIX only {vix_px:.1f} — market selling into complacency = distribution, not panic")

    # Gold vs Equities
    if gld_d1 > 0.3 and spy_d1 > 0.3:
        hints.append(f"⚡ RISK-ON + GOLD BOTH UP: SPY +{spy_d1:.1f}% with Gold +{gld_d1:.1f}% = dual hedging demand, not pure risk-on")
    elif gld_d1 > 0.5 and spy_d1 < -0.3:
        hints.append(f"⚡ CLASSIC FLIGHT: SPY {spy_d1:.1f}% while Gold +{gld_d1:.1f}% = safe haven rotation confirmed")

    # Oil vs Equities (energy cost signal)
    if uso_d1 < -2 and spy_d1 > 0.3:
        hints.append(f"⚡ OIL DOWN / EQUITIES UP: USO {uso_d1:.1f}% — markets treating oil drop as disinflationary tailwind, not demand slowdown signal")
    elif uso_d1 > 2 and spy_d1 < -0.3:
        hints.append(f"⚡ STAGFLATION SIGNAL: Oil +{uso_d1:.1f}% + SPY {spy_d1:.1f}% — cost-push pressure on margins is the read, not growth")

    # Dollar vs Risk
    if uup_d1 < -0.3 and spy_d1 > 0.3:
        hints.append(f"⚡ DOLLAR WEAKNESS SUPPORTING RISK: DXY {uup_d1:.1f}% feeding into equity strength + EM tailwind")
    elif uup_d1 > 0.3 and btc_d1 < -1:
        hints.append(f"⚡ DOLLAR STRENGTH HITTING CRYPTO: BTC {btc_d1:.1f}% as DXY +{uup_d1:.1f}% — dollar liquidity tightening")

    # Rates vs Equities
    if tnx_px > 4.5 and spy_d1 < 0:
        hints.append(f"⚡ HIGH RATES WEIGHING: 10Y at {tnx_px:.2f}% — multiple compression pressure on long-duration equities")
    elif tnx_px < 4.0 and spy_d1 > 0.5:
        hints.append(f"⚡ RATE RELIEF: 10Y at {tnx_px:.2f}% creating multiple expansion space for equities")

    # BTC vs SPY correlation
    if abs(btc_d1) > 2:
        if btc_d1 * spy_d1 > 0:
            hints.append(f"⚡ BTC-SPY CORRELATED: Both moving same direction — risk appetite is broad ({btc_d1:+.1f}% BTC, {spy_d1:+.1f}% SPY)")
        else:
            hints.append(f"⚡ BTC-SPY DECOUPLING: BTC {btc_d1:+.1f}% vs SPY {spy_d1:+.1f}% = crypto-specific driver, not macro")

    # Nasdaq vs SPY (tech premium)
    if abs(qqq_d1 - spy_d1) > 0.5:
        if qqq_d1 > spy_d1:
            hints.append(f"⚡ TECH LEADING: QQQ +{qqq_d1:.1f}% vs SPY +{spy_d1:.1f}% — growth factor outperforming, rate sensitivity is low")
        else:
            hints.append(f"⚡ TECH LAGGING: QQQ {qqq_d1:.1f}% vs SPY {spy_d1:.1f}% — value/cyclicals absorbing the rotation")

    # GCC oil link
    if tasi_d1 != 0 and uso_d1 != 0:
        if tasi_d1 * uso_d1 > 0:
            hints.append(f"⚡ TASI-OIL ALIGNED: TASI {tasi_d1:+.1f}% tracking oil {uso_d1:+.1f}% — energy-driven GCC beta intact")
        else:
            hints.append(f"⚡ TASI-OIL DIVERGING: TASI {tasi_d1:+.1f}% vs oil {uso_d1:+.1f}% — domestic liquidity decoupling from commodity cycle")

    if not hints:
        hints.append("No significant cross-asset divergences detected — regime signals are internally consistent.")

    return "\n".join(hints)


# ── GCC Intelligence Standalone Generator ─────────────────────────────────────

def generate_gcc_intelligence(
    moves_summary: dict,
    news: NewsContext,
    regime: str,
) -> dict:
    """
    Generate a rich standalone GCC intelligence block.
    Used to power the regional section in the frontend.
    Deterministic + news-enriched — no AI required.
    """
    tasi = moves_summary.get("^TASI", {})
    dfm_proxy = moves_summary.get("^DFMGI", {})
    uso = moves_summary.get("USO", {})
    uup = moves_summary.get("UUP", {})
    gld = moves_summary.get("GLD", {})

    tasi_d1 = tasi.get("d1_pct", 0) or 0
    tasi_d5 = tasi.get("d5_pct", 0) or 0
    tasi_px = tasi.get("price", 0) or 0
    uso_d1  = uso.get("d1_pct", 0) or 0
    uso_d5  = uso.get("d5_pct", 0) or 0
    uso_px  = uso.get("price", 0) or 0
    uup_d5  = uup.get("d5_pct", 0) or 0

    # Oil-GCC directional signal
    oil_gcc_sentiment = "supportive" if uso_d5 > 0 else "pressuring"
    oil_dir = "higher" if uso_d1 > 0 else "lower"

    # Forex impact on GCC (USD pegged)
    dollar_context = (
        "Dollar strength supports GCC pegs but dampens import competitiveness and EM flows."
        if uup_d5 > 0.5
        else "Dollar softness eases peg maintenance costs and opens EM capital flow potential."
        if uup_d5 < -0.5
        else "Dollar neutral — GCC peg stability stable, no FX-driven distortions."
    )

    # Top GCC headlines for context
    gcc_news_lines = [f"• [{h.source}] {h.title}" for h in news.top_gcc(4)]
    gcc_news_text = "\n".join(gcc_news_lines) if gcc_news_lines else "• No GCC-specific headlines available this session."

    # Build the output
    return {
        "tasi": {
            "d1_pct": tasi_d1,
            "d5_pct": tasi_d5,
            "price": tasi_px,
            "direction": "↑" if tasi_d1 > 0.25 else ("↓" if tasi_d1 < -0.25 else "→"),
            "note": (
                f"TASI {'gaining' if tasi_d5 > 0 else 'losing'} {abs(tasi_d5):.1f}% on the week. "
                f"Oil at {uso_px:.1f} is {oil_gcc_sentiment} GCC earnings visibility."
            ),
        },
        "oil_link": {
            "uso_d1": uso_d1,
            "uso_d5": uso_d5,
            "price": uso_px,
            "gcc_impact": (
                f"Oil {oil_dir} {abs(uso_d1):.1f}% today. "
                f"{'Aramco and ADNOC margins under pressure — watch energy sector weights in TASI/ADX.' if uso_d1 < -1 else 'Energy windfall supporting sovereign revenues — GCC fiscal buffer intact.' if uso_d1 > 1 else 'Oil range-bound — GCC budget neutrality ~$70-80/bbl maintains stability.'}"
            ),
        },
        "dollar_context": dollar_context,
        "gcc_headlines": gcc_news_lines[:4],
        "gcc_news_text": gcc_news_text,
        "regional_regime": (
            "Cautious" if regime == "Bearish" else
            "Constructive" if regime == "Bullish" else
            "Selective"
        ),
        "summary": (
            f"GCC markets remain {'resilient' if tasi_d5 > 0 else 'under pressure'} with TASI "
            f"{tasi_d5:+.1f}% on the week. "
            f"{dollar_context} "
            f"Key watch: oil at {uso_px:.1f} — {'above' if uso_px > 80 else 'below'} the GCC fiscal breakeven zone."
        ),
    }


# ── Integration helper ────────────────────────────────────────────────────────

def enrich_daily_update(update: dict, news: NewsContext, moves_summary: dict) -> dict:
    """
    Post-generation enrichment: add news sources, GCC section, and
    cross-asset signals to an already-generated daily update dict.
    Call this AFTER the AI generates the base update.
    """
    regime = update.get("market_regime", "Cautious")

    # Add GCC intelligence block
    gcc_intel = generate_gcc_intelligence(moves_summary, news, regime)
    update["gcc_intelligence"] = gcc_intel

    # Add correlation signals
    update["cross_asset_signals"] = _build_correlation_hints(moves_summary)

    # Add news attribution
    all_sources = set()
    for bucket in [news.global_headlines, news.gcc_headlines, news.energy_headlines]:
        for item in bucket[:3]:
            all_sources.add(item.source)
    update["news_sources"] = sorted(all_sources)[:8]
    update["news_fetched_at"] = news.fetched_at

    # If gcc_note missing (AI didn't fill it), build deterministic version
    if not update.get("gcc_note") and gcc_intel:
        update["gcc_note"] = gcc_intel["summary"]

    return update


# ── Cache layer ───────────────────────────────────────────────────────────────

_NEWS_CACHE: dict = {"data": None, "fetched_at": None}
_NEWS_CACHE_TTL = timedelta(minutes=20)


def get_news_context(force_refresh: bool = False) -> NewsContext:
    """
    Cached news fetch — refreshes every 20 minutes.
    Use this in generate_daily_update() and generate_weekly_update().
    """
    now = datetime.now(timezone.utc)
    cached_at = _NEWS_CACHE.get("fetched_at")

    if (
        not force_refresh
        and isinstance(cached_at, datetime)
        and (now - cached_at) < _NEWS_CACHE_TTL
        and _NEWS_CACHE.get("data") is not None
    ):
        logger.debug("[news] Using cached news context")
        return _NEWS_CACHE["data"]

    logger.info("[news] Fetching fresh news context")
    ctx = fetch_news_context()
    _NEWS_CACHE["data"] = ctx
    _NEWS_CACHE["fetched_at"] = now
    return ctx
