"""
macro_intelligence.py — EisaX Global Macro Awareness Engine
=============================================================

Provides:
  1. Live commodity / index / currency snapshot  (yfinance — free, no key)
  2. Macro news headlines                         (NewsAPI / Serper fallback)
  3. Cross-domain linkage map                     (geopolitical → asset → stock)
  4. Prompt-ready formatted string                (injected into every DeepSeek call)

Usage:
    from core.macro_intelligence import get_macro_context
    ctx = get_macro_context(ticker="EMAAR.DU", sector="Real Estate")
    # → inject ctx["prompt_block"] into DeepSeek prompt
"""

import os, time, logging, re
from functools import lru_cache
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ─── TTL in-memory cache ────────────────────────────────────────────────────
_MACRO_CACHE: dict = {}
_MACRO_TTL   = 900   # 15 min — commodities move slowly intra-day

# ─── Live macro tickers (all free via yfinance) ──────────────────────────────
_MACRO_YF = {
    # Commodities
    "oil_brent":  ("BZ=F",       "Brent Crude",     "$/bbl"),
    "oil_wti":    ("CL=F",       "WTI Crude",        "$/bbl"),
    "gold":       ("GC=F",       "Gold",             "$/oz"),
    "silver":     ("SI=F",       "Silver",           "$/oz"),
    "copper":     ("HG=F",       "Copper",           "$/lb"),
    "nat_gas":    ("NG=F",       "Natural Gas",      "$/MMBtu"),
    "wheat":      ("ZW=F",       "Wheat",            "¢/bu"),
    "corn":       ("ZC=F",       "Corn",             "¢/bu"),
    "sugar":      ("SB=F",       "Sugar #11",        "¢/lb"),
    "coffee":     ("KC=F",       "Coffee",           "¢/lb"),
    "cotton":     ("CT=F",       "Cotton",           "¢/lb"),
    # "palmoil": ("POO.KL", "Palm Oil", "MYR/t"),  # delisted
    # Indices / Risk
    "vix":        ("^VIX",       "VIX Fear Index",   "pts"),
    "sp500":      ("^GSPC",      "S&P 500",          "pts"),
    "dxy":        ("DX-Y.NYB",   "US Dollar (DXY)",  "pts"),
    "us10y":      ("^TNX",       "US 10-Y Yield",    "%"),
    "em_etf":     ("EEM",        "EM ETF",           "$"),
    # Currencies
    "eurusd":     ("EURUSD=X",   "EUR/USD",          ""),
    "usdjpy":     ("USDJPY=X",   "USD/JPY",          ""),
    "usdinr":     ("USDINR=X",   "USD/INR",          ""),
    "usdbrl":     ("USDBRL=X",   "USD/BRL",          ""),
    "usdegp":     ("USDEGP=X",   "USD/EGP",          ""),
    "usdsar":     ("USDSAR=X",   "USD/SAR",          ""),
}

# ─── Cross-domain linkage knowledge base ────────────────────────────────────
# Structure: trigger_keywords → {affected_asset: (direction, reasoning_chain)}
MACRO_LINKAGES = {

    # ── Geopolitical / Middle East ──────────────────────────────────────────
    "iran_conflict": {
        "keywords": ["iran", "hormuz", "tehran", "irgc", "persian gulf",
                     "strait of hormuz", "iran sanctions", "iran nuclear"],
        "impacts": {
            "oil":             ("+15–25%", "Strait of Hormuz handles 20% of global oil → closure = supply shock"),
            "gold":            ("+8–12%",  "Safe-haven demand spikes in Middle East conflict"),
            "shipping":        ("+30–50%", "Tanker rerouting around Cape of Good Hope adds 2–3 weeks"),
            "sugar_brazil":    ("+3–7%",   "Higher shipping costs raise Brazilian sugar export price"),
            "uae_realestate":  ("-10–20%", "Risk premium on Gulf assets expands; foreign capital pulls back"),
            "gulf_banks":      ("-5–12%",  "Loan-book risk re-pricing; Saudi/UAE sovereign spreads widen"),
            "defense_stocks":  ("+10–20%", "US/Israel defense sector benefits from elevated tensions"),
        }
    },

    "russia_ukraine": {
        "keywords": ["ukraine", "russia", "moscow", "kyiv", "nato", "russian invasion",
                     "crimea", "donbas", "zelensky", "putin"],
        "impacts": {
            "wheat":           ("+20–40%", "Ukraine = 10% of global wheat exports; Black Sea corridor closure"),
            "corn":            ("+15–25%", "Ukraine = 15% of global corn; war disrupts planting/harvest"),
            "nat_gas":         ("+25–60%", "Russia = 40% of EU gas supply; sanctions → LNG premium"),
            "gold":            ("+5–10%",  "War = safe-haven demand; Russia sells USD reserves → gold"),
            "fertilizer":      ("+30–50%", "Russia/Belarus = 40% of global potash; sanctions cut supply"),
            "sugar_indirect":  ("+5–10%",  "Fertilizer price rise → Brazil sugarcane cost inflation"),
            "defense_stocks":  ("+15–30%", "NATO re-armament spending: Lockheed, BAE, Rheinmetall"),
        }
    },

    "china_taiwan": {
        "keywords": ["taiwan", "tsmc", "taiwan strait", "china invasion", "pla", "chip war",
                     "semiconductor blockade", "china taiwan"],
        "impacts": {
            "semiconductors":  ("-30–50%", "Taiwan = 90% of advanced chips; blockade = global tech shock"),
            "tech_stocks":     ("-20–35%", "Supply chain freeze; Apple, NVIDIA, AMD, ASML all exposed"),
            "gold":            ("+10–20%", "Largest geopolitical risk in decades → peak safe-haven"),
            "oil":             ("+5–10%",  "South China Sea = major shipping lane for Gulf oil to Asia"),
        }
    },

    "saudi_opec_cut": {
        "keywords": ["opec", "opec+", "saudi production cut", "aramco output",
                     "oil cut", "production quota", "barrel cut"],
        "impacts": {
            "oil":             ("+5–15%",  "Supply reduction directly lifts spot price"),
            "saudi_stocks":    ("+3–8%",   "Aramco earnings rise; Saudi fiscal surplus improves"),
            "airlines":        ("-5–10%",  "Jet fuel is 25–30% of airline costs; margins compress"),
            "petrochemicals":  ("-3–7%",   "Higher feedstock cost squeezes SABIC/Kayan margins"),
            "inflation_em":    ("+1–3%",   "Oil-importing EMs (Egypt, India, Turkey) see FX pressure"),
        }
    },

    # ── Monetary Policy ─────────────────────────────────────────────────────
    "fed_rate_hike": {
        "keywords": ["federal reserve", "fed hike", "rate increase", "fomc hawkish",
                     "powell hike", "interest rate rise", "tightening cycle"],
        "impacts": {
            "growth_stocks":   ("-15–25%", "Higher discount rate crushes DCF valuations; NASDAQ first"),
            "gold":            ("-5–10%",  "Opportunity cost of holding non-yielding gold rises"),
            "usd":             ("+3–5%",   "Rate differential draws capital to USD assets"),
            "em_currencies":   ("-5–15%",  "Capital flight from EM; EGP/INR/BRL weaken"),
            "bonds":           ("-8–15%",  "Existing bond prices fall inversely to yield rise"),
            "real_estate":     ("-10–20%", "Mortgage rate spike suppresses demand; cap rates re-price"),
            "egp":             ("-5–10%",  "Egypt: higher USD rates → pressure on EGP peg/float"),
        }
    },

    "fed_rate_cut": {
        "keywords": ["federal reserve cut", "fed pivot", "rate cut", "fomc dovish",
                     "powell cut", "easing cycle", "quantitative easing"],
        "impacts": {
            "gold":            ("+5–15%",  "Lower opportunity cost → gold re-rates higher"),
            "growth_stocks":   ("+10–20%", "Lower discount rate expands tech/growth multiples"),
            "em_currencies":   ("+3–8%",   "Capital flows back to EM carry trades"),
            "real_estate":     ("+8–15%",  "Lower mortgages reignite housing demand"),
            "em_bonds":        ("+5–12%",  "Spread compression; EM sovereign bonds rally"),
        }
    },

    "china_stimulus": {
        "keywords": ["china stimulus", "pboc cut", "beijing stimulus", "chinese bazooka",
                     "china property rescue", "china rrr cut", "li keqiang stimulus"],
        "impacts": {
            "copper":          ("+8–15%",  "China = 55% of global copper demand; construction restart"),
            "iron_ore":        ("+10–20%", "Steel production for infrastructure projects"),
            "oil":             ("+5–8%",   "Industrial activity recovery boosts energy demand"),
            "em_etf":          ("+5–12%",  "Risk-on; EM equities re-rate on China growth hopes"),
            "australian_dollar": ("+3–5%", "Australia = largest iron ore/coal exporter to China"),
        }
    },

    # ── Commodity-Specific ──────────────────────────────────────────────────
    "drought_brazil": {
        "keywords": ["brazil drought", "brazil rainfall", "mato grosso", "sao paulo drought",
                     "brazil crop", "brazil agriculture", "el nino brazil"],
        "impacts": {
            "sugar":           ("+10–20%", "Brazil = 25% of global sugar; drought cuts cane yield"),
            "coffee":          ("+15–25%", "Brazil = 40% of Arabica; drought damages coffee cherries"),
            "soybeans":        ("+8–15%",  "Brazil = largest soy exporter; crop failure → global shortage"),
            "brl":             ("-3–8%",   "Commodity export revenue falls → BRL weakens"),
            "india_sugar":     ("+5–10%",  "India imports more → domestic prices rise → restaurant inflation"),
        }
    },

    "india_inflation": {
        "keywords": ["india cpi", "india inflation", "rbi rate", "india food price",
                     "india restaurant", "india gst", "india fuel price"],
        "impacts": {
            "usdinr":          ("+2–5%",   "Inflation erodes INR purchasing power vs USD"),
            "india_banks":     ("-5–10%",  "RBI tightening squeezes NIMs and loan growth"),
            "india_consumer":  ("-8–15%",  "Discretionary spending falls as food/fuel eat income"),
            "gold_india":      ("+5–10%",  "India = world's largest gold consumer; inflation hedge buying"),
            "indian_food_cos": ("-5–12%",  "Input cost inflation squeezes margins of ITC, Nestle India"),
        }
    },

    "us_tariffs_china": {
        "keywords": ["tariff", "trade war", "us china tariff", "import duty",
                     "section 301", "trade deficit", "protectionism"],
        "impacts": {
            "tech_supply_chain": ("-10–20%", "Apple/NVIDIA Asia manufacturing costs rise"),
            "vietnam":         ("+5–10%",  "Manufacturing relocates to Vietnam, Mexico, India"),
            "copper":          ("-3–8%",   "Slower global trade → industrial metals weaker"),
            "usd":             ("+2–4%",   "US trade deficit narrows → structural USD demand"),
        }
    },

    # ── Sector-Specific ─────────────────────────────────────────────────────
    "ai_boom": {
        "keywords": ["ai chips", "nvidia earnings", "chatgpt", "llm demand",
                     "data center", "ai infrastructure", "openai", "artificial intelligence boom"],
        "impacts": {
            "nvidia":          ("+20–40%", "CUDA moat; every AI model trains on NVDA GPUs"),
            "electricity":     ("+5–10%",  "Data centers consume 2–3% of global electricity; rising"),
            "copper":          ("+3–7%",   "Data centers need 4× more copper than traditional buildings"),
            "real_estate_dc":  ("+10–20%", "Data center REITs in premium locations re-rate"),
        }
    },

    "oil_glut": {
        "keywords": ["oil oversupply", "oil glut", "opec discord", "oil inventory surge",
                     "shale production surge", "oil bear market"],
        "impacts": {
            "oil":             ("-15–30%", "Supply glut → spot price collapse"),
            "aramco":          ("-8–15%",  "Revenue and dividend sustainability questioned"),
            "gulf_sovereign":  ("-5–10%",  "Fiscal break-even price pressure on GCC budgets"),
            "airlines":        ("+5–12%",  "Lower jet fuel → margin expansion"),
            "fertilizers":     ("-5–10%",  "Gas-based ammonia cost falls → nitrogen fertilizer cheaper"),
        }
    },
}

# ─── Sector → relevant macro keys ───────────────────────────────────────────
SECTOR_MACRO_MAP = {
    "energy":          ["oil_brent", "oil_wti", "nat_gas", "dxy"],
    "oil":             ["oil_brent", "oil_wti", "nat_gas", "dxy"],
    "petrochemicals":  ["oil_brent", "nat_gas", "dxy", "us10y"],
    "real estate":     ["us10y", "dxy", "gold", "vix"],
    "banks":           ["us10y", "dxy", "vix", "sp500"],
    "financial":       ["us10y", "dxy", "vix", "sp500"],
    "technology":      ["vix", "sp500", "us10y", "dxy"],
    "consumer":        ["wheat", "corn", "sugar", "coffee", "oil_brent"],
    "food":            ["wheat", "corn", "sugar", "coffee", "cotton"],
    "agriculture":     ["wheat", "corn", "sugar", "coffee", "palmoil"],
    "materials":       ["copper", "gold", "silver", "nat_gas"],
    "mining":          ["copper", "gold", "silver", "iron_ore"],
    "utilities":       ["nat_gas", "us10y", "dxy"],
    "pharma":          ["dxy", "vix", "us10y"],
    "telecom":         ["dxy", "us10y"],
    "industrials":     ["copper", "oil_brent", "dxy"],
    "airlines":        ["oil_brent", "oil_wti", "dxy"],
    "shipping":        ["oil_brent", "nat_gas", "dxy"],
    "default":         ["oil_brent", "gold", "vix", "dxy", "us10y"],
}


# ─── 1. Live Macro Snapshot ──────────────────────────────────────────────────
def get_live_macro(force_refresh: bool = False) -> dict:
    """
    Returns live prices + % changes for all macro tickers.
    Cached for 15 minutes.  Uses yfinance (free, no key needed).

    Returns:
        {
          "oil_brent": {"label": "Brent Crude", "price": 82.5, "chg_pct": +1.2,
                        "unit": "$/bbl", "trend": "↑"},
          "vix": {...},
          "timestamp": "2026-03-20 19:00 UTC",
          "error_count": 0,
        }
    """
    cache_key = "live_macro"
    cached = _MACRO_CACHE.get(cache_key)
    if cached and not force_refresh and time.time() - cached["_ts"] < _MACRO_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    import yfinance as yf
    import warnings
    warnings.filterwarnings("ignore")

    result: dict = {}
    error_count = 0

    # Batch download for speed
    symbols = [v[0] for v in _MACRO_YF.values()]
    try:
        data = yf.download(
            symbols, period="2d", interval="1d",
            progress=False, auto_adjust=True, threads=True
        )
        closes = data.get("Close", {}) if hasattr(data, 'get') else data["Close"]

        for key, (sym, label, unit) in _MACRO_YF.items():
            try:
                col = closes[sym] if sym in closes.columns else None
                if col is None or len(col.dropna()) < 2:
                    continue
                clean = col.dropna()
                price = float(clean.iloc[-1])
                prev  = float(clean.iloc[-2])
                chg   = ((price - prev) / prev * 100) if prev else 0.0
                result[key] = {
                    "label":   label,
                    "price":   round(price, 4),
                    "chg_pct": round(chg, 2),
                    "unit":    unit,
                    "trend":   "↑" if chg > 0.1 else ("↓" if chg < -0.1 else "→"),
                }
            except Exception as e:
                logger.debug(f"[Macro] {key} ({sym}): {e}")
                error_count += 1

    except Exception as e:
        logger.warning(f"[Macro] batch yf.download failed: {e}")
        error_count = len(_MACRO_YF)

    result["timestamp"]   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    result["error_count"] = error_count

    _MACRO_CACHE[cache_key] = {**result, "_ts": time.time()}
    logger.info(f"[Macro] Fetched {len(result)-2} live prices ({error_count} errors)")
    return result


# ─── 2. Macro News ───────────────────────────────────────────────────────────
_NEWS_CACHE: dict = {}
_NEWS_TTL = 1800  # 30 min

# Trusted financial news sources — macro headlines should only come from these.
# Regional/local press (Times of India, etc.) produces noise for global macro analysis.
_TRUSTED_MACRO_SOURCES = {
    'bloomberg', 'reuters', 'financial times', 'ft', 'wall street journal', 'wsj',
    'cnbc', 'marketwatch', "barron's", 'barrons', 'the economist', 'economist',
    'associated press', 'ap news', 'bbc news', 'bbc', 'forbes', 'investing.com',
    'yahoo finance', 'business insider', 'axios', 'the guardian', 'ft.com',
    'fx street', 'fxstreet', 'seeking alpha', 'motley fool', 'nasdaq', 'nytimes',
    'new york times', 'washington post', 'politico', 'fortune', 'thestreet',
}

def _is_trusted_macro_source(source: str) -> bool:
    """Return True if source is a known trusted financial/macro outlet."""
    s = (source or '').lower().strip()
    return any(t in s for t in _TRUSTED_MACRO_SOURCES)

def get_macro_news(limit: int = 12) -> list[dict]:
    """
    Fetch top macro/market-moving headlines.
    Sources: NewsAPI → Serper (Google) fallback.
    Returns list of {title, source, url, published}.
    """
    cached = _NEWS_CACHE.get("macro_news")
    if cached and time.time() - cached["_ts"] < _NEWS_TTL:
        return cached["data"]

    news = []

    # ── Source 1: NewsAPI ────────────────────────────────────────────────────
    NEWS_KEY = os.getenv("NEWS_API_KEY", "")
    if NEWS_KEY:
        try:
            import requests
            queries = [
                "oil price geopolitics markets",
                "Federal Reserve interest rates inflation",
                "China economy commodities",
                "Middle East conflict markets",
            ]
            seen = set()
            for q in queries:
                if len(news) >= limit:
                    break
                r = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={"q": q, "pageSize": 8, "sortBy": "publishedAt",
                            "language": "en", "apiKey": NEWS_KEY},
                    timeout=8
                )
                if r.status_code == 200:
                    for art in r.json().get("articles", []):
                        title  = art.get("title", "")
                        src    = art.get("source", {}).get("name", "NewsAPI")
                        if title and title not in seen and "[Removed]" not in title:
                            if _is_trusted_macro_source(src):
                                seen.add(title)
                                news.append({
                                    "title":     title,
                                    "source":    src,
                                    "url":       art.get("url", ""),
                                    "published": art.get("publishedAt", "")[:10],
                                })
        except Exception as e:
            logger.warning(f"[Macro News] NewsAPI error: {e}")

    # ── Source 2: Serper (Google Search) ────────────────────────────────────
    if len(news) < 4:
        SERPER_KEY = os.getenv("SERPER_API_KEY", "")
        if SERPER_KEY:
            try:
                import requests
                r = requests.post(
                    "https://google.serper.dev/news",
                    headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
                    json={"q": "global markets geopolitics commodities 2026", "num": limit},
                    timeout=8
                )
                if r.status_code == 200:
                    for item in r.json().get("news", []):
                        src = item.get("source", "Google")
                        if _is_trusted_macro_source(src):
                            news.append({
                                "title":     item.get("title", ""),
                                "source":    src,
                                "url":       item.get("link", ""),
                                "published": item.get("date", ""),
                            })
            except Exception as e:
                logger.warning(f"[Macro News] Serper error: {e}")

    news = news[:limit]
    _NEWS_CACHE["macro_news"] = {"data": news, "_ts": time.time()}
    logger.info(f"[Macro News] Fetched {len(news)} headlines")
    return news


# ─── 3. Cross-Domain Linkage Detector ───────────────────────────────────────
def get_relevant_linkages(
    ticker: str,
    sector: str = "",
    news_headlines: list[str] | None = None,
    macro_prices: dict | None = None,
) -> list[dict]:
    """
    Detects which macro events from the knowledge base are relevant to:
    - The current news headlines
    - The current commodity/index moves
    - The ticker's sector

    Returns list of {
        "event": str,
        "confidence": "HIGH"|"MEDIUM"|"LOW",
        "relevant_impacts": {asset: (direction, reasoning)},
        "triggered_by": "news"|"price_move"|"sector",
    }
    """
    found = []
    headlines_text = " ".join(news_headlines or []).lower()

    # Check each linkage scenario
    for event_key, event_data in MACRO_LINKAGES.items():
        keywords   = event_data["keywords"]
        impacts    = event_data["impacts"]
        confidence = None
        trigger    = None

        # Check news headlines
        kw_hits = sum(1 for kw in keywords if kw in headlines_text)
        if kw_hits >= 2:
            confidence = "HIGH"
            trigger    = "news"
        elif kw_hits == 1:
            confidence = "MEDIUM"
            trigger    = "news"

        # Check price anomalies (>2% single-day move in related asset)
        if macro_prices and confidence is None:
            event_assets = _event_price_assets(event_key)
            for asset_key in event_assets:
                md = macro_prices.get(asset_key, {})
                chg = abs(md.get("chg_pct", 0))
                if chg >= 3.0:
                    confidence = "MEDIUM"
                    trigger    = "price_move"
                    break

        if confidence is None:
            continue

        # Filter impacts relevant to this ticker/sector
        sector_lo = sector.lower()
        ticker_lo = ticker.lower()
        relevant  = {}
        for asset, (direction, reasoning) in impacts.items():
            if _is_relevant_impact(asset, sector_lo, ticker_lo):
                relevant[asset] = (direction, reasoning)

        if relevant or confidence == "HIGH":
            found.append({
                "event":            event_key,
                "label":            event_key.replace("_", " ").title(),
                "confidence":       confidence,
                "relevant_impacts": relevant if relevant else impacts,
                "triggered_by":     trigger,
            })

    return found


def _event_price_assets(event_key: str) -> list[str]:
    """Map event → which macro prices to check for anomalies."""
    mapping = {
        "iran_conflict":    ["oil_brent", "gold"],
        "russia_ukraine":   ["wheat", "nat_gas", "gold"],
        "china_taiwan":     ["gold", "em_etf"],
        "saudi_opec_cut":   ["oil_brent", "oil_wti"],
        "fed_rate_hike":    ["us10y", "dxy"],
        "fed_rate_cut":     ["gold", "us10y"],
        "china_stimulus":   ["copper", "em_etf"],
        "drought_brazil":   ["sugar", "coffee"],
        "india_inflation":  ["usdinr"],
        "us_tariffs_china": ["em_etf", "copper"],
        "ai_boom":          ["sp500"],
        "oil_glut":         ["oil_brent", "oil_wti"],
    }
    return mapping.get(event_key, [])


def _is_relevant_impact(asset: str, sector_lo: str, ticker_lo: str) -> bool:
    """Check if a macro impact is relevant to this stock's sector/name."""
    sector_relevance = {
        "oil":          ["energy", "oil", "petro", "aramco", "adnoc", "refineri"],
        "gold":         ["mining", "gold", "metal", "precious"],
        "wheat":        ["food", "consumer", "bakery", "cereal", "flour"],
        "sugar":        ["food", "consumer", "sugar", "beverage"],
        "coffee":       ["food", "consumer", "beverage", "coffee"],
        "copper":       ["mining", "industrial", "material", "utilities"],
        "nat_gas":      ["energy", "utilities", "petro", "dewa", "taqa"],
        "em_currencies": ["bank", "financial", "insurance"],
        "shipping":     ["transport", "logistics", "shipping"],
        "gulf_banks":   ["bank", "financial"],
        "uae_realestate": ["real estate", "property", "emaar", "aldar"],
        "growth_stocks": ["tech", "technology", "software"],
        "airlines":     ["aviation", "airline", "transport"],
        "tech_supply_chain": ["tech", "semiconductor", "hardware"],
    }
    clues = sector_relevance.get(asset, [])
    return any(c in sector_lo or c in ticker_lo for c in clues) if clues else True


# ─── 4. Sector-Specific Macro Prices ────────────────────────────────────────
def get_sector_macro(sector: str, macro: dict) -> dict:
    """Return only the macro metrics most relevant to this sector."""
    sector_lo = sector.lower()
    keys = SECTOR_MACRO_MAP.get("default")
    for skey, mkeys in SECTOR_MACRO_MAP.items():
        if skey in sector_lo:
            keys = mkeys
            break
    return {k: macro[k] for k in keys if k in macro}


# ─── 5. Prompt Block Builder ─────────────────────────────────────────────────
def get_macro_context(
    ticker: str = "",
    sector: str = "",
    news_headlines: list[str] | None = None,
) -> dict:
    """
    Main entry point. Returns:
        {
          "macro":        dict of all live prices,
          "sector_macro": dict of sector-relevant prices,
          "linkages":     list of detected cross-domain events,
          "macro_news":   list of headline dicts,
          "prompt_block": str ready to inject into DeepSeek prompt,
        }
    """
    macro      = get_live_macro()
    sec_macro  = get_sector_macro(sector, macro)
    macro_news = get_macro_news(limit=10)
    headlines  = [n["title"] for n in macro_news] + (news_headlines or [])
    linkages   = get_relevant_linkages(ticker, sector, headlines, macro)

    prompt_block = _build_prompt_block(ticker, sector, macro, sec_macro, linkages, macro_news)

    return {
        "macro":        macro,
        "sector_macro": sec_macro,
        "linkages":     linkages,
        "macro_news":   macro_news,
        "prompt_block": prompt_block,
    }


def _build_prompt_block(
    ticker, sector, macro, sec_macro, linkages, macro_news
) -> str:
    """Format a rich, analyst-quality macro context block for the DeepSeek prompt."""

    ts = macro.get("timestamp", "")
    lines = [f"\n---\n🌍 **GLOBAL MACRO CONTEXT** *(live as of {ts})*\n"]

    # ── Key commodities & risk indicators ───────────────────────────────────
    priority_keys = ["oil_brent", "gold", "vix", "dxy", "us10y", "copper",
                     "nat_gas", "wheat", "sugar", "sp500"]
    # Sector-relevant first
    shown = list(sec_macro.keys())
    for k in priority_keys:
        if k not in shown and k in macro:
            shown.append(k)

    commodity_lines = []
    for k in shown[:8]:
        md = macro.get(k)
        if not md or not isinstance(md, dict):
            continue
        p   = md["price"]
        chg = md["chg_pct"]
        lbl = md["label"]
        unit = md["unit"]
        trend = md["trend"]
        chg_str = f"+{chg:.1f}%" if chg > 0 else f"{chg:.1f}%"
        commodity_lines.append(f"  • **{lbl}**: {p:,.2f} {unit} ({chg_str} {trend})")

    if commodity_lines:
        lines.append("**Market Snapshot:**")
        lines.extend(commodity_lines)
        lines.append("")

    # ── VIX risk level ───────────────────────────────────────────────────────
    vix_data = macro.get("vix", {})
    vix_val  = vix_data.get("price", 0) if isinstance(vix_data, dict) else 0
    if vix_val:
        if vix_val > 30:
            risk_label = "🔴 EXTREME FEAR — Markets in panic mode"
        elif vix_val > 20:
            risk_label = "🟡 ELEVATED — Risk-off environment"
        elif vix_val > 15:
            risk_label = "🟢 NORMAL — Orderly markets"
        else:
            risk_label = "🟢 COMPLACENT — Low volatility / risk-on"
        lines.append(f"**Market Sentiment:** VIX {vix_val:.1f} — {risk_label}\n")

    # ── Cross-domain linkages ────────────────────────────────────────────────
    high_linkages = [l for l in linkages if l["confidence"] in ("HIGH", "MEDIUM")]
    if high_linkages:
        lines.append("⚡ **CROSS-ASSET LINKAGES DETECTED** *(relevant to this stock)*:")
        for lnk in high_linkages[:3]:
            conf_icon = "🔴" if lnk["confidence"] == "HIGH" else "🟡"
            lines.append(f"\n**{conf_icon} {lnk['label']}** (triggered by: {lnk['triggered_by']})")
            for asset, (direction, reasoning) in list(lnk["relevant_impacts"].items())[:4]:
                lines.append(f"  → **{asset.replace('_',' ').title()}**: `{direction}` — {reasoning}")
        lines.append("")

    # ── Macro headlines ──────────────────────────────────────────────────────
    if macro_news:
        lines.append("📰 **Macro Headlines** *(market-moving events)*:")
        for n in macro_news[:6]:
            src = n.get("source", "")
            lines.append(f"  • {n['title']} *({src})*")
        lines.append("")

    # ── Instruction to DeepSeek ──────────────────────────────────────────────
    lines.append(
        "🎯 **ANALYST INSTRUCTION**: Use the above macro context to explain "
        "**indirect causality chains** that affect this stock. "
        "Example format: *'Iran tensions → Hormuz risk → oil supply shock → "
        "shipping costs +35% → Brazilian sugar export cost rises → Sugar #11 +7% → "
        "food inflation in Egypt accelerates → EGP pressure'*. "
        "Connect global dots to the specific stock/sector. "
        "Do NOT ignore macro tailwinds/headwinds even if not directly sector-related.\n---"
    )

    return "\n".join(lines)


# ─── 6. Resilient News Fetcher ───────────────────────────────────────────────
def get_news_resilient(
    ticker: str,
    company: str = "",
    sector: str  = "",
    limit:  int  = 6,
) -> list[dict]:
    """
    Waterfall news fetcher:
    1. realtime_data.get_live_news (FMP → Finnhub → NewsAPI)
    2. realtime_data._get_local_news (GNews → NewsAPI Arabic)
    3. Serper web search as last resort
    4. Returns macro_news if still empty (never empty-handed)
    """
    from core.realtime_data import get_live_news, _get_local_news, _is_local_ticker

    news = []

    # Step 1: Primary source
    try:
        raw = get_live_news(ticker, limit=limit)
        if raw:
            news = raw
            logger.info(f"[News/{ticker}] got {len(news)} from get_live_news")
    except Exception as e:
        logger.warning(f"[News/{ticker}] get_live_news failed: {e}")

    # Step 2: Local market source
    if not news and _is_local_ticker(ticker):
        try:
            raw = _get_local_news(ticker, company_name=company, limit=limit)
            if raw:
                news = raw
                logger.info(f"[News/{ticker}] got {len(news)} from _get_local_news")
        except Exception as e:
            logger.warning(f"[News/{ticker}] _get_local_news failed: {e}")

    # Step 3: Serper web search
    if not news:
        SERPER_KEY = os.getenv("SERPER_API_KEY", "")
        if SERPER_KEY:
            try:
                import requests
                query = f"{company or ticker} stock news {sector} {datetime.now().year}"
                r = requests.post(
                    "https://google.serper.dev/news",
                    headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
                    json={"q": query, "num": limit},
                    timeout=8
                )
                if r.status_code == 200:
                    for item in r.json().get("news", []):
                        news.append({
                            "title":  item.get("title", ""),
                            "url":    item.get("link", ""),
                            "source": item.get("source", "Google"),
                        })
                    logger.info(f"[News/{ticker}] got {len(news)} from Serper")
            except Exception as e:
                logger.warning(f"[News/{ticker}] Serper failed: {e}")

    # Step 4: Macro news as fallback (never empty)
    if not news:
        macro_n = get_macro_news(limit=4)
        news = [{
            "title":  n["title"],
            "url":    n.get("url", ""),
            "source": f"{n.get('source','')} [macro context]",
        } for n in macro_n]
        logger.info(f"[News/{ticker}] using macro news as fallback ({len(news)} items)")

    return news[:limit]


# ─── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fetching live macro...")
    m = get_live_macro()
    for k, v in m.items():
        if isinstance(v, dict):
            print(f"  {k}: {v['price']} {v['unit']} ({v['chg_pct']:+.1f}% {v['trend']})")

    print("\nBuilding macro context for EMAAR.DU / Real Estate...")
    ctx = get_macro_context("EMAAR.DU", "Real Estate")
    print(ctx["prompt_block"])
