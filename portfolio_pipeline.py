"""
portfolio_pipeline.py — EisaX Clean Portfolio Pipeline
=======================================================
4-step clean architecture:

  Step 1 — Intent Parser (LLM)
    "balanced portfolio US + GCC + Egypt + commodities"
    → {risk: "balanced", markets: ["us","gcc","egypt"], include: ["commodities"]}

  Step 2 — Allocation Engine (Rules-based)
    → {us_equity: 0.20, gcc_equity: 0.25, egypt_equity: 0.05,
       bonds: 0.25, commodities: 0.10, crypto: 0.05, cash: 0.05}

  Step 3 — Asset Selector (Live Market Cache)
    → Real tickers with verified names, scores, expected returns

  Step 4 — Report Generator (DeepSeek)
    → Full institutional report with real numbers
"""

import os, json, logging, sys
from typing import Any
import pandas as pd

logger = logging.getLogger("eisax.portfolio_pipeline")

# Ensure project root is on sys.path so imports work when called from any location
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.data_layer import market_cache_adapter as _mca

# ── TradingView prefix → yfinance suffix ──────────────────────────────────────
_TV_SUFFIX = {
    "TADAWUL": ".SR", "DFM": ".DU", "ADX": ".AE",
    "EGX": ".CA", "KSE": ".KW", "QSE": ".QA",
}

def _tv_to_yf(tv: str) -> str:
    if ":" in tv:
        exch, sym = tv.split(":", 1)
        return sym + _TV_SUFFIX.get(exch.upper(), "")
    return tv

# ── Verified company names ────────────────────────────────────────────────────
_NAMES = {
    "2222.SR": "Saudi Aramco",          "1120.SR": "Al Rajhi Bank",
    "1180.SR": "Alinma Bank",           "1150.SR": "Riyad Bank",
    "7010.SR": "STC (Saudi Telecom)",   "7030.SR": "Zain KSA",
    "2010.SR": "SABIC",                 "2060.SR": "TASNEE",
    "4001.SR": "Abdullah Al Othaim",    "1211.SR": "Ma'aden",
    "1050.SR": "Banque Saudi Fransi",   "1140.SR": "Al-Awwal Bank",
    "EMAAR.DU": "Emaar Properties",     "FAB.AE": "First Abu Dhabi Bank",
    "ADCB.AE": "Abu Dhabi Commercial Bank",
    "COMI.CA": "CIB Egypt",             "TMGH.CA": "Talaat Moustafa Group",
    "ETEL.CA": "Telecom Egypt",         "TAQA.CA": "TAQA Arabia",
    "KSA":  "iShares MSCI Saudi Arabia ETF",
    "GULF": "WisdomTree Middle East Dividend ETF",
    "EGPT": "VanEck Egypt Index ETF",
    "QAT":  "iShares MSCI Qatar ETF",
    "SCHR": "Schwab Intermediate-Term Treasury ETF",
    "GSG":  "iShares S&P GSCI Commodity ETF",
    "PDBC": "Invesco Optimum Yield Diversified Commodity ETF",
    "IBIT": "iShares Bitcoin Trust ETF",
    "FETH": "Fidelity Ethereum Fund ETF",
    "4290.SR": "Jarir Marketing Co.",
    "EMAAR.DU": "Emaar Properties",
    "DEWA.AE":  "Dubai Electricity & Water Authority (DEWA)",
    "ALDAR.AE": "Aldar Properties (Abu Dhabi)",
    "FAB.AE":   "First Abu Dhabi Bank",
    "GLD":  "Gold ETF (SPDR)",          "IAU": "iShares Gold Trust",
    "SLV":  "Silver ETF (iShares)",     "USO": "US Oil Fund",
    "BND":  "Vanguard Total Bond ETF",  "TLT": "iShares 20Y Treasury ETF",
    "AGG":  "iShares Core Bond ETF",    "HYG": "iShares High Yield Bond ETF",
    "SPY":  "S&P 500 ETF",              "QQQ": "Nasdaq-100 ETF",
    "AAPL": "Apple",                    "MSFT": "Microsoft",
    "NVDA": "Nvidia",                   "GOOGL": "Alphabet (Google)",
    "AMZN": "Amazon",                   "META": "Meta Platforms",
    "TSLA": "Tesla",
    "BTC-USD": "Bitcoin",               "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    # Common US large-caps
    "JPM": "JPMorgan Chase",            "BAC": "Bank of America",
    "WFC": "Wells Fargo",               "GS": "Goldman Sachs",
    "JNJ": "Johnson & Johnson",         "PFE": "Pfizer",
    "UNH": "UnitedHealth Group",        "MRK": "Merck",
    "ABBV": "AbbVie",                   "LLY": "Eli Lilly",
    "XOM": "ExxonMobil",               "CVX": "Chevron",
    "V": "Visa",                        "MA": "Mastercard",
    "WMT": "Walmart",                   "HD": "Home Depot",
    "KO": "Coca-Cola",                  "PEP": "PepsiCo",
    "MCD": "McDonald's",               "SBUX": "Starbucks",
    "DIS": "Walt Disney",              "NFLX": "Netflix",
    "INTC": "Intel",                   "AMD": "AMD",
    "CSCO": "Cisco",                   "IBM": "IBM",
    "ORCL": "Oracle",                  "CRM": "Salesforce",
    "ADBE": "Adobe Inc.",
    "DELL": "Dell Technologies",        "HPQ": "HP Inc.",
    "CAT": "Caterpillar",              "BA": "Boeing",
    "GE": "GE Aerospace",              "MMM": "3M",
    "VZ": "Verizon",                   "T": "AT&T",
    "PG": "Procter & Gamble",          "CL": "Colgate-Palmolive",
    "MU": "Micron Technology",         "AMAT": "Applied Materials",
    "LRCX": "Lam Research",            "TXN": "Texas Instruments",
    # UAE stocks
    "BOROUGE.AE": "Borouge",           "AGILITY.AE": "Agility Logistics",
    "DIB.AE": "Dubai Islamic Bank",    "ALDAR.AE": "Aldar Properties",
    "SALIK.AE": "Salik (Dubai Toll)",  "DEWA.AE": "DEWA",
    "TABREED.AE": "National Central Cooling (Tabreed)",
    # KSA additional
    "4290.SR": "Jarir Marketing",      "8010.SR": "STC Pay",
    "2380.SR": "Petro Rabigh",         "1180.SR": "Alinma Bank",
    "4030.SR": "Tihama",               "6010.SR": "NADEC",
    # Egypt additional
    "EGAL.CA": "El Gawhara Real Estate","HRHO.CA": "Heliopolis Housing",
    "SWDY.CA": "El Sewedy Electric",   "OCDI.CA": "Oriental Weavers",
}

def name_of(ticker: str) -> str:
    return _NAMES.get(ticker, ticker)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — INTENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_intent(user_message: str) -> dict:
    """
    Use LLM to extract structured portfolio intent from user message.
    Returns a dict with: risk, markets, include, horizon, max_drawdown, currency
    """
    try:
        import sys; sys.path.insert(0, os.path.dirname(__file__))
        from core.llm import get_client
        import config

        client = get_client()
        resp = client.create_completion(
            model=config.DEFAULT_MODEL,
            temperature=0,
            max_tokens=500,
            messages=[{
                "role": "system",
                "content": (
                    "Extract portfolio parameters from the user message. "
                    "Return ONLY valid JSON with these exact keys:\n"
                    '{\n'
                    '  "risk": "conservative|balanced|growth|aggressive",\n'
                    '  "markets": ["us","gcc","ksa","uae","egypt","kuwait","qatar","global"],\n'
                    '  "include": ["bonds","gold","commodities","crypto","reits","cash"],\n'
                    '  "horizon": "short|medium|long",\n'
                    '  "max_drawdown": 0.25,\n'
                    '  "currency": "USD|SAR|EGP"\n'
                    '}\n\n'
                    'RULES:\n'
                    '- "ksa" or "saudi" → markets: ["gcc"] (use gcc which covers both KSA+UAE)\n'
                    '- "aggressive" or "هجومي" → risk: "aggressive"\n'
                    '- "balanced" or "متوازن" → risk: "balanced"\n'
                    '- "conservative" or "محافظ" → risk: "conservative"\n'
                    '- If no risk mentioned → "balanced"\n'
                    '- If no horizon → "long"\n'
                    '- max_drawdown: extract number if mentioned (e.g. "max 25%" → 0.25), else 0.25\n'
                    '- Return ONLY the JSON object, nothing else'
                )
            }, {
                "role": "user",
                "content": user_message
            }]
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        # clean markdown code blocks if present
        raw = raw.strip("```json").strip("```").strip()
        intent = json.loads(raw)
        logger.info("[Pipeline] Parsed intent: %s", intent)
        return intent
    except Exception as e:
        logger.warning("[Pipeline] Intent parse failed: %s — using defaults", e)
        return {
            "risk": "balanced",
            "markets": ["us", "gcc"],
            "include": ["bonds", "gold"],
            "horizon": "long",
            "max_drawdown": 0.25,
            "currency": "USD"
        }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ALLOCATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

# Base allocation templates per risk level
# Buckets: us_equity, gcc_equity, egypt_equity, bonds, gold, commodities, crypto, cash
_BASE_ALLOC = {
    "conservative": {
        "bonds": 0.45, "gold": 0.10, "us_equity": 0.20,
        "gcc_equity": 0.15, "egypt_equity": 0.00,
        "commodities": 0.05, "crypto": 0.00, "cash": 0.05,
    },
    "balanced": {
        "bonds": 0.25, "gold": 0.08, "us_equity": 0.22,
        "gcc_equity": 0.22, "egypt_equity": 0.05,
        "commodities": 0.08, "crypto": 0.05, "cash": 0.05,
    },
    "growth": {
        "bonds": 0.10, "gold": 0.05, "us_equity": 0.30,
        "gcc_equity": 0.28, "egypt_equity": 0.08,
        "commodities": 0.10, "crypto": 0.07, "cash": 0.02,
    },
    "aggressive": {
        "bonds": 0.00, "gold": 0.05, "us_equity": 0.38,
        "gcc_equity": 0.33, "egypt_equity": 0.00,
        "commodities": 0.08, "crypto": 0.07, "cash": 0.00,
    },
}

def build_allocation(intent: dict) -> dict:
    """
    Given parsed intent, return allocation % per bucket.
    Adjusts base template based on requested markets and asset classes.
    """
    risk = intent.get("risk", "balanced")
    markets = [m.lower() for m in intent.get("markets", [])]
    include = [i.lower() for i in intent.get("include", [])]

    alloc = dict(_BASE_ALLOC.get(risk, _BASE_ALLOC["balanced"]))

    # Zero out markets not requested
    has_us     = any(m in markets for m in ["us", "america", "global"])
    has_gcc    = any(m in markets for m in ["gcc", "ksa", "saudi", "uae", "gulf", "خليج"])
    has_egypt  = any(m in markets for m in ["egypt", "egx", "مصر"])
    has_bonds  = any(i in include for i in ["bonds", "fixed income", "سندات"]) or risk in ["conservative", "balanced"]
    has_gold   = any(i in include for i in ["gold", "ذهب"])
    has_comm   = any(i in include for i in ["commodities", "oil", "silver", "copper", "سلع"])
    has_crypto = any(i in include for i in ["crypto", "bitcoin", "btc", "كريبتو"])

    if not has_us:      alloc["us_equity"] = 0.0
    if not has_gcc:     alloc["gcc_equity"] = 0.0
    if not has_egypt:   alloc["egypt_equity"] = 0.0
    if not has_bonds:   alloc["bonds"] = 0.0
    if not has_gold:    alloc["gold"] = 0.0
    if not has_comm:    alloc["commodities"] = 0.0
    if not has_crypto:  alloc["crypto"] = 0.0

    # Normalize to 1.0
    total = sum(alloc.values())
    if total > 0:
        alloc = {k: round(v / total, 4) for k, v in alloc.items()}

    # Drop zero buckets for cleanliness
    alloc = {k: v for k, v in alloc.items() if v > 0}

    logger.info("[Pipeline] Allocation buckets: %s", alloc)
    return alloc


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — ASSET SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache(market: str) -> pd.DataFrame | None:
    """Latest snapshot for a market — routed through core.data_layer."""
    try:
        return _mca.get_latest_snapshot(market)
    except Exception as e:
        logger.warning("[Pipeline] Cache load failed for %s: %s", market, e)
        return None


def _score_stock(row) -> float:
    """Composite quality score: RSI + MACD + uptrend + P/E + market cap."""
    s = 0.0
    try:
        rsi = float(row.get("RSI") or 0)
        s += 1.0 if 40 <= rsi <= 65 else 0.4 if 30 <= rsi <= 75 else 0.0

        chg = float(row.get("change") or 0)
        s += 0.5 if chg > 0 else 0.0

        macd = row.get("MACD.macd"); sig = row.get("MACD.signal")
        if macd is not None and sig is not None:
            s += 0.5 if float(macd or 0) > float(sig or 0) else 0.0

        cl   = float(row.get("close")  or 0)
        s50  = float(row.get("SMA50")  or 0)
        s200 = float(row.get("SMA200") or 0)
        if cl > 0 and s50 > 0 and s200 > 0:
            s += 1.0 if cl > s50 > s200 else 0.0

        pe = float(row.get("price_earnings_ttm") or 0)
        s += 1.0 if 5 < pe < 25 else 0.5 if pe == 0 else 0.0

        mc = float(row.get("market_cap_basic") or 0)
        s += 1.0 if mc > 5e9 else 0.5 if mc > 1e9 else 0.0
    except Exception:
        pass
    return s


def _top_from_cache(market: str, n: int = 3) -> list[dict]:
    """Return top-N stocks from market cache as dicts with ticker + meta."""
    df = _load_cache(market)
    if df is None or df.empty:
        return []
    df = df.copy()
    df = df[(df["close"] > 0) & (df["market_cap_basic"].fillna(0) > 0)]
    df["_score"] = df.apply(_score_stock, axis=1)
    top = df.nlargest(n, "_score")
    result = []
    for _, row in top.iterrows():
        tv  = str(row.get("ticker") or "")
        yf  = _tv_to_yf(tv)
        result.append({
            "ticker": yf,
            "name":   name_of(yf) if name_of(yf) != yf else str(row.get("name") or yf),
            "sector": str(row.get("sector") or ""),
            "close":  float(row.get("close") or 0),
            "rsi":    float(row.get("RSI") or 0),
            "market": market,
            "score":  round(float(row.get("_score") or 0), 2),
        })
    return result


# Fixed ETF/asset lists per bucket
_BOND_ETFS   = [
    {"ticker": "BND",  "name": "Vanguard Total Bond ETF",      "type": "bonds"},
    {"ticker": "TLT",  "name": "iShares 20Y Treasury ETF",     "type": "bonds"},
    {"ticker": "AGG",  "name": "iShares Core Bond ETF",        "type": "bonds"},
    {"ticker": "SCHR", "name": "Schwab Intermediate-Term Treasury ETF", "type": "bonds"},
]
_GOLD_ETFS   = [
    {"ticker": "GLD",  "name": "SPDR Gold Shares",             "type": "gold"},
    {"ticker": "IAU",  "name": "iShares Gold Trust",            "type": "gold"},
]
_COMM_ETFS   = [
    {"ticker": "GSG",  "name": "iShares S&P GSCI Commodity ETF","type": "commodities"},
    {"ticker": "PDBC", "name": "Invesco Diversified Commodity ETF","type": "commodities"},
    {"ticker": "DBA",  "name": "Invesco DB Agriculture ETF",   "type": "commodities"},
    {"ticker": "SLV",  "name": "iShares Silver Trust",         "type": "commodities"},
]
_CRYPTO_ETFS = [
    {"ticker": "IBIT",    "name": "iShares Bitcoin Trust ETF",        "type": "crypto"},
    {"ticker": "FETH",    "name": "Fidelity Ethereum Fund ETF",        "type": "crypto"},
]
_GCC_ETFS    = [
    {"ticker": "KSA",  "name": "iShares MSCI Saudi Arabia ETF",         "type": "gcc_etf"},
    {"ticker": "GULF", "name": "WisdomTree Middle East Dividend ETF",   "type": "gcc_etf"},
    {"ticker": "QAT",  "name": "iShares MSCI Qatar ETF",                "type": "gcc_etf"},
]
_EGYPT_ETFS  = [
    {"ticker": "EGPT", "name": "VanEck Egypt Index ETF",       "type": "egypt_etf"},
]

# ── Risk-profiled US equity lists ─────────────────────────────────────────────
# Conservative: dividend / defensive blue chips
_US_CONSERVATIVE = [
    {"ticker": "JNJ",  "name": "Johnson & Johnson",    "sector": "Healthcare",   "type": "us_equity"},
    {"ticker": "PG",   "name": "Procter & Gamble",     "sector": "Consumer",     "type": "us_equity"},
    {"ticker": "KO",   "name": "Coca-Cola",             "sector": "Consumer",     "type": "us_equity"},
    {"ticker": "VZ",   "name": "Verizon",               "sector": "Telecom",      "type": "us_equity"},
    {"ticker": "BND",  "name": "Vanguard Total Bond ETF","sector": "Bonds",       "type": "us_equity"},
]
# Balanced: mix of growth + stability
_US_BALANCED = [
    {"ticker": "MSFT", "name": "Microsoft",            "sector": "Technology",   "type": "us_equity"},
    {"ticker": "AAPL", "name": "Apple",                 "sector": "Technology",   "type": "us_equity"},
    {"ticker": "JPM",  "name": "JPMorgan Chase",        "sector": "Financials",   "type": "us_equity"},
    {"ticker": "V",    "name": "Visa",                  "sector": "Financials",   "type": "us_equity"},
    {"ticker": "SPY",  "name": "S&P 500 ETF",           "sector": "Broad Market", "type": "us_equity"},
]
# Growth: high-growth tech + innovation
_US_GROWTH = [
    {"ticker": "NVDA", "name": "Nvidia",                "sector": "Semiconductors","type": "us_equity"},
    {"ticker": "MSFT", "name": "Microsoft",             "sector": "Technology",    "type": "us_equity"},
    {"ticker": "AMZN", "name": "Amazon",                "sector": "Consumer Tech", "type": "us_equity"},
    {"ticker": "GOOGL","name": "Alphabet (Google)",     "sector": "Technology",    "type": "us_equity"},
    {"ticker": "META", "name": "Meta Platforms",        "sector": "Technology",    "type": "us_equity"},
    {"ticker": "QQQ",  "name": "Nasdaq-100 ETF",        "sector": "Broad Tech",    "type": "us_equity"},
]
# Aggressive: high-beta growth + AI + semis + disruptors
_US_AGGRESSIVE = [
    {"ticker": "NVDA", "name": "Nvidia",                "sector": "Semiconductors",           "type": "us_equity"},
    {"ticker": "AMD",  "name": "AMD",                   "sector": "Semiconductors",           "type": "us_equity"},
    {"ticker": "META", "name": "Meta Platforms",        "sector": "Digital Advertising/AI",   "type": "us_equity"},
    {"ticker": "AMZN", "name": "Amazon",                "sector": "Cloud/E-Commerce",         "type": "us_equity"},
    # ADBE replaces TSLA: different driver — creative software subscriptions + AI licensing
    # (not hardware capex, not consumer EV sentiment, not energy storage)
    {"ticker": "ADBE", "name": "Adobe",                 "sector": "Creative Software/SaaS",   "type": "us_equity"},
    {"ticker": "CRM",  "name": "Salesforce",            "sector": "Enterprise Software/SaaS", "type": "us_equity"},
    {"ticker": "QQQ",  "name": "Nasdaq-100 ETF",        "sector": "Broad Tech",               "type": "us_equity"},
]
_US_BY_RISK = {
    "conservative": _US_CONSERVATIVE,
    "balanced":     _US_BALANCED,
    "growth":       _US_GROWTH,
    "aggressive":   _US_AGGRESSIVE,
}

# ── Risk-profiled GCC lists ────────────────────────────────────────────────────
# Balanced/Conservative: dividend-heavy blue chips
_GCC_CONSERVATIVE = [
    {"ticker": "2222.SR", "name": "Saudi Aramco",             "sector": "Energy",      "type": "gcc_equity"},
    {"ticker": "1120.SR", "name": "Al Rajhi Bank",            "sector": "Financials",  "type": "gcc_equity"},
    {"ticker": "KSA",     "name": "iShares MSCI Saudi Arabia ETF","sector": "Broad",   "type": "gcc_etf"},
    {"ticker": "GULF",    "name": "WisdomTree Middle East Dividend ETF","sector": "Broad","type": "gcc_etf"},
]
# Growth: mix sectors — banking + real estate + telecom + diversified
_GCC_GROWTH_SECTORS = [
    {"ticker": "2222.SR",    "name": "Saudi Aramco",           "sector": "Energy",      "type": "gcc_equity"},
    {"ticker": "1120.SR",    "name": "Al Rajhi Bank",          "sector": "Financials",  "type": "gcc_equity"},
    {"ticker": "7010.SR",    "name": "STC (Saudi Telecom)",    "sector": "Telecom",     "type": "gcc_equity"},
    {"ticker": "EMAAR.DU",   "name": "Emaar Properties",      "sector": "Real Estate", "type": "gcc_equity"},
    {"ticker": "FAB.AE",     "name": "First Abu Dhabi Bank",  "sector": "Financials",  "type": "gcc_equity"},
    {"ticker": "GULF",       "name": "WisdomTree Middle East Dividend ETF","sector": "Broad","type": "gcc_etf"},
]
# Aggressive: growth sectors — infra + digital + non-oil diversification
_GCC_AGGRESSIVE_SECTORS = [
    {"ticker": "2222.SR",    "name": "Saudi Aramco",           "sector": "Energy",          "type": "gcc_equity"},
    {"ticker": "1120.SR",    "name": "Al Rajhi Bank",          "sector": "Financials",      "type": "gcc_equity"},
    {"ticker": "7010.SR",    "name": "STC (Saudi Telecom)",    "sector": "Telecom/Digital", "type": "gcc_equity"},
    {"ticker": "4290.SR",    "name": "Jarir Marketing",        "sector": "Consumer/Retail", "type": "gcc_equity"},
    {"ticker": "EMAAR.DU",   "name": "Emaar Properties",      "sector": "Real Estate",     "type": "gcc_equity"},
    {"ticker": "DEWA.AE",    "name": "DEWA",                  "sector": "Utilities/Infra",      "type": "gcc_equity"},
    {"ticker": "ALDAR.AE",  "name": "Aldar Properties",      "sector": "Real Estate/Abu Dhabi","type": "gcc_equity"},
    {"ticker": "KSA",        "name": "iShares MSCI Saudi Arabia ETF","sector": "Broad",     "type": "gcc_etf"},
    {"ticker": "QAT",        "name": "iShares MSCI Qatar ETF","sector": "Broad/Diversifier","type": "gcc_etf"},
]
_GCC_BY_RISK = {
    "conservative": _GCC_CONSERVATIVE,
    "balanced":     _GCC_GROWTH_SECTORS,
    "growth":       _GCC_GROWTH_SECTORS,
    "aggressive":   _GCC_AGGRESSIVE_SECTORS,
}


def _top_from_cache_filtered(market: str, n: int, exclude_sectors: list = None) -> list[dict]:
    """
    Like _top_from_cache but filters out specified sectors to avoid concentration.
    """
    df = _load_cache(market)
    if df is None or df.empty:
        return []
    df = df.copy()
    df = df[(df["close"] > 0) & (df["market_cap_basic"].fillna(0) > 0)]
    if exclude_sectors:
        df = df[~df["sector"].isin(exclude_sectors)]
    df["_score"] = df.apply(_score_stock, axis=1)
    top = df.nlargest(n, "_score")
    result = []
    for _, row in top.iterrows():
        tv  = str(row.get("ticker") or "")
        yf  = _tv_to_yf(tv)
        result.append({
            "ticker": yf,
            "name":   name_of(yf) if name_of(yf) != yf else str(row.get("name") or yf),
            "sector": str(row.get("sector") or ""),
            "close":  float(row.get("close") or 0),
            "rsi":    float(row.get("RSI") or 0),
            "market": market,
            "score":  round(float(row.get("_score") or 0), 2),
        })
    return result


def select_assets(allocation: dict, intent: dict) -> dict:
    """
    For each allocation bucket, pick the best real assets.
    - Risk-profiled US lists (growth/tech for aggressive, defensive for conservative)
    - GCC sector-diversified lists (not just energy/banks)
    - Correct commodities for diversification, not as "hedge" language
    Returns: {bucket: [asset_dict, ...]}
    """
    risk    = intent.get("risk", "balanced")
    assets  = {}

    n_us  = {"conservative": 3, "balanced": 4, "growth": 5, "aggressive": 5}.get(risk, 4)
    n_gcc = {"conservative": 3, "balanced": 5, "growth": 6, "aggressive": 6}.get(risk, 5)

    if "us_equity" in allocation:
        # Use risk-profiled list — not raw cache (cache gives value/defensive by score)
        profiled = list(_US_BY_RISK.get(risk, _US_BALANCED)[:n_us])
        # For growth/aggressive: augment ONLY with tech/growth sector picks from cache
        if risk in ("growth", "aggressive"):
            _GROWTH_SECTORS = {"Technology", "Consumer Cyclical", "Communication Services",
                               "Semiconductors", "Software", "Consumer Tech"}
            cache_picks = _top_from_cache("america", n=6)
            existing_tickers = {a["ticker"] for a in profiled}
            for cp in cache_picks:
                if (cp["ticker"] not in existing_tickers and
                        cp.get("sector","") in _GROWTH_SECTORS):
                    profiled = profiled + [cp]
                    existing_tickers.add(cp["ticker"])
                    if len(profiled) >= n_us + 2:
                        break
        assets["us_equity"] = profiled

    if "gcc_equity" in allocation:
        # Use risk-profiled GCC list for sector diversity
        profiled_gcc = _GCC_BY_RISK.get(risk, _GCC_GROWTH_SECTORS)[:n_gcc]
        # Augment with cache — but exclude Energy & Financials if already well represented
        # to avoid economic concentration
        existing_sectors = [a.get("sector","") for a in profiled_gcc]
        energy_count    = sum(1 for s in existing_sectors if "Energy" in s)
        finance_count   = sum(1 for s in existing_sectors if "Financial" in s)
        exclude = []
        if energy_count >= 2:    exclude.append("Energy")
        if finance_count >= 2:   exclude.append("Finance")
        cache_gcc = (
            _top_from_cache_filtered("ksa", n=2, exclude_sectors=exclude or None) +
            _top_from_cache_filtered("uae", n=1, exclude_sectors=exclude or None)
        )
        existing_tickers = {a["ticker"] for a in profiled_gcc}
        for cg in cache_gcc:
            if cg["ticker"] not in existing_tickers:
                profiled_gcc = profiled_gcc + [cg]
                existing_tickers.add(cg["ticker"])
                if len(profiled_gcc) >= n_gcc + 2:
                    break
        assets["gcc_equity"] = profiled_gcc

    if "egypt_equity" in allocation:
        eg_stocks = _top_from_cache("egypt", n=2)
        assets["egypt_equity"] = eg_stocks + _EGYPT_ETFS if eg_stocks else _EGYPT_ETFS

    if "bonds" in allocation:
        if risk == "conservative":
            # Short + intermediate duration — lower rate sensitivity
            assets["bonds"] = [_BOND_ETFS[0], _BOND_ETFS[3]]   # BND + SCHR
        elif risk == "balanced":
            assets["bonds"] = _BOND_ETFS[:2]                    # BND + TLT
        else:
            assets["bonds"] = [_BOND_ETFS[0]]                   # BND only for growth/aggressive

    if "gold" in allocation:
        assets["gold"] = _GOLD_ETFS[:1]   # GLD

    if "commodities" in allocation:
        if risk in ("growth", "aggressive"):
            # Broad diversified commodity ETF — NOT just oil+silver (avoids single-commodity risk)
            assets["commodities"] = [_COMM_ETFS[0], _COMM_ETFS[1]]   # GSG + PDBC
        else:
            assets["commodities"] = [_COMM_ETFS[0], _COMM_ETFS[3]]   # GSG + SLV

    if "crypto" in allocation:
        # Split crypto across BTC + ETH to avoid single-asset concentration
        assets["crypto"] = _CRYPTO_ETFS[:2]

    logger.info("[Pipeline] Selected assets: %s",
                {k: [a["ticker"] for a in v] for k, v in assets.items()})
    return assets


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — BUILD PORTFOLIO (weights within each bucket)
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio(allocation: dict, assets: dict) -> list[dict]:
    """
    Assign final weights to each asset.
    Within each bucket, distribute the bucket's allocation equally.
    Returns flat list of {ticker, name, weight, bucket, ...}
    """
    portfolio = []
    for bucket, bucket_weight in allocation.items():
        bucket_assets = assets.get(bucket, [])
        if not bucket_assets:
            continue
        per_asset = bucket_weight / len(bucket_assets)
        for a in bucket_assets:
            portfolio.append({
                "ticker":  a["ticker"],
                "name":    a.get("name", a["ticker"]),
                "weight":  round(per_asset, 4),
                "bucket":  bucket,
                "sector":  a.get("sector", ""),
                "score":   a.get("score", 0),
                "rsi":     a.get("rsi", 0),
                "close":   a.get("close", 0),
            })

    # Normalize weights to exactly 1.0
    total = sum(p["weight"] for p in portfolio)
    if total > 0:
        for p in portfolio:
            p["weight"] = round(p["weight"] / total, 4)

    # Sort by weight descending
    portfolio.sort(key=lambda x: -x["weight"])
    return portfolio


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — REPORT GENERATOR (DeepSeek)
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(portfolio: list[dict], allocation: dict,
                    intent: dict, metrics: dict) -> str:
    """
    Send the complete, structured portfolio data to DeepSeek.
    Returns a full institutional markdown report.
    """
    try:
        from core.llm import get_client
        import config
        from datetime import datetime as _dt

        # Build the allocation table for the prompt
        alloc_table = "\n".join(
            f"| {p['name']} | {p['ticker']} | {p['weight']*100:.1f}% | {p['bucket'].replace('_',' ').title()} |"
            for p in portfolio
        )

        # Build bucket summary
        bucket_summary = "\n".join(
            f"- {k.replace('_',' ').title()}: {v*100:.0f}%"
            for k, v in sorted(allocation.items(), key=lambda x: -x[1])
        )

        risk     = intent.get("risk", "balanced").upper()
        horizon  = intent.get("horizon", "long")
        mdd      = intent.get("max_drawdown", 0.25)
        today    = _dt.now().strftime("%B %d, %Y")

        exp_ret  = metrics.get("expected_return", 0)
        vol      = metrics.get("volatility", 0)
        sharpe   = metrics.get("sharpe", 0)

        # ── Institutional diagnostics (Effective N, buckets, drawdown) ────
        from core.services.portfolio_analytics import (
            compute_effective_n,
            diversification_label,
            diversification_emoji,
            compute_economic_buckets,
            bucket_concentration_warning,
            estimate_worst_case_from_vol,
            readiness_with_drawdown,
            sharpe_context_note,
            diversification_soft_suggestion,
        )
        _weights_list = [float(p.get("weight") or 0) for p in portfolio]
        eff_n = compute_effective_n(_weights_list)
        eff_n_label = diversification_label(eff_n)
        eff_n_icon = diversification_emoji(eff_n)
        econ_buckets = compute_economic_buckets(portfolio)
        bucket_warn = bucket_concentration_warning(econ_buckets, threshold=50.0)
        worst_case_dd = estimate_worst_case_from_vol(vol, z=2.0)   # decimal, negative
        sharpe_note = sharpe_context_note(sharpe, vol)
        soft_diversif_note = diversification_soft_suggestion(eff_n, econ_buckets)

        # Determine base readiness — then apply the drawdown gate
        placeholders = [p["ticker"] for p in portfolio
                        if p["ticker"].upper() in {"NEEDED","UAE","SAUDI","ARABIA","GCC","AS","AN"}]
        _base_readiness = (
            "✅ APPROVED"
            if not placeholders and exp_ret > 0.045 and sharpe > 0
            else "⚠️ CONDITIONAL"
        )
        _readiness_verdict = readiness_with_drawdown(
            base_status=_base_readiness,
            worst_case=worst_case_dd,
            target_drawdown=mdd,
        )
        readiness = _readiness_verdict.status
        drawdown_gate_note = _readiness_verdict.note

        # Render bucket table for prompt
        bucket_table_lines = [
            f"- {name}: {pct:.0f}%"
            for name, pct in sorted(econ_buckets.items(), key=lambda kv: -kv[1])
        ]
        econ_bucket_block = "\n".join(bucket_table_lines) or "- (no classified buckets)"

        # Compute sector/economic concentration for the prompt
        sector_counts: dict = {}
        for p in portfolio:
            s = p.get("sector", "") or p.get("bucket", "")
            sector_counts[s] = sector_counts.get(s, 0) + 1
        concentration_note = "; ".join(
            f"{s}: {c} positions" for s, c in sorted(sector_counts.items(), key=lambda x: -x[1]) if c >= 2
        ) or "well distributed"

        # Top-2 weight concentration
        top2_weight = sum(p["weight"] for p in portfolio[:2])
        position_sizing_alert = top2_weight > 0.35  # flag if top 2 > 35%

        # Accurate strategy label based on actual composition
        equity_pct = sum(p["weight"] for p in portfolio
                         if "equity" in p.get("bucket","") or p.get("bucket","") == "us_equity")
        strategy_label = {
            "conservative": "Conservative Income & Capital Preservation",
            "balanced":     "Balanced Growth & Income",
            "growth":       "Growth-Oriented Diversified",
            "aggressive":   "Aggressive Growth — High Conviction Equity",
        }.get(risk.lower(), f"{risk} Strategy")

        prompt = f"""You are EisaX AI, an elite institutional portfolio strategist.
Today: {today}
Generate a complete, professional Institutional Portfolio Strategy Report in English.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLIENT MANDATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy Name: {strategy_label}
Risk Profile: {risk}
Investment Horizon: {horizon}
Target Drawdown: {mdd*100:.0f}% (mandate floor — not a guarantee; in extreme market conditions losses may exceed this)
Estimated Worst-Case Drawdown: {worst_case_dd*100:.0f}% (derived from 2σ of annual volatility; may breach target in stress regimes)
Strategy Readiness: {readiness}{(chr(10) + 'Readiness Note: ' + drawdown_gate_note) if drawdown_gate_note else ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGIC ALLOCATION (BUCKET LEVEL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{bucket_summary}
Total Equity Exposure: {equity_pct*100:.0f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PORTFOLIO HOLDINGS (USE THESE EXACT NAMES AND WEIGHTS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
| Asset Name | Ticker | Weight | Bucket |
|---|---|---|---|
{alloc_table}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERFORMANCE METRICS (USE EXACT NUMBERS — DO NOT CHANGE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Expected Annual Return: {exp_ret*100:.2f}%  (fundamental estimate; methodology: weighted bucket base rates, risk-free rate 4.5%)
- Annual Volatility: {vol*100:.2f}%  (weighted average; true portfolio vol lower due to cross-asset diversification)
- Sharpe Ratio: {sharpe:.2f}  (excess return per unit of risk)
- Target Drawdown: {mdd*100:.0f}% (mandate floor)
- Estimated Worst-Case Drawdown: {worst_case_dd*100:.0f}% (2σ of annual volatility)
- Effective Diversification (N): {eff_n:.1f} — {eff_n_icon} {eff_n_label}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ECONOMIC EXPOSURE BREAKDOWN (correlation-aware — USE THESE LABELS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{econ_bucket_block}
{("⚠️ " + bucket_warn) if bucket_warn else "✅ No single economic bucket exceeds 50%."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCENTRATION FLAGS (MUST ADDRESS IN REPORT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sector/Theme Concentration: {concentration_note}
Top-2 Positions Combined Weight: {top2_weight*100:.1f}%
Position Sizing Alert Required: {"YES — top-2 positions exceed 35% combined" if position_sizing_alert else "No — weights well distributed"}
Effective N Verdict: {eff_n_label} (N = {eff_n:.1f}) — if N < 5 the report MUST describe the portfolio as a "concentrated high-conviction structure", not "well diversified".
{("Sharpe Context: " + sharpe_note) if sharpe_note else ""}
{("Advisory: " + soft_diversif_note) if soft_diversif_note else ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED REPORT STRUCTURE (10 SECTIONS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 0. Strategy Readiness: {readiness}
Show a verification table: assets verified | return positive | Sharpe positive | drawdown guideline met

## 1. Portfolio Overview (2-3 paragraphs)
- State the CORRECT strategy label: "{strategy_label}"
- Describe what the portfolio actually IS (not just its name)
- Note the equity exposure percentage and asset class mix

## 2. Risk Profile Analysis
- Stress test table: S&P -10% / -30% / -50% / Fed +2% / Bull +20%
- IMPORTANT: For the -50% scenario, honestly state if the portfolio may breach the {mdd*100:.0f}% drawdown target
- Add a paragraph on ECONOMIC CONCENTRATION risk (sectors/themes that move together)

## 3. Investment Thesis — Why This Portfolio?
- 3 specific, distinct pillars — make them real (not marketing language)
- For aggressive/growth: reference AI, digital transformation, regional growth themes
- For conservative: reference income generation, capital preservation, yield

## 4. Asset Selection Rationale
One bullet per asset: what it is | why selected for THIS risk profile | role in portfolio | what it is NOT (e.g. "PFE is a defensive healthcare name — suitable for conservative/balanced, NOT a high-growth aggressive pick")

## 5. Allocation Table (exact weights — never change them)

## 6. Performance Metrics + Methodology
Table with all 4 metrics + plain English interpretation.
Add a "Methodology Note" box with EXACTLY this content:

> **Methodology Note**
> Expected returns are fundamental estimates using weighted bucket base rates (US Equity: 12%, GCC Equity: 13%, Bonds: 5%, Gold: 7%, Commodities: 8%, Crypto: 18%). The volatility figure is a weighted average of bucket-level standard deviations; the true portfolio volatility **may vary materially depending on correlation regimes** — particularly during stress events when asset-class correlations converge toward 1.0. The Sharpe Ratio uses a risk-free rate of 4.5%.
>
> **Estimated Inter-Asset Correlations (approximate, long-run):**
> | Asset Pair | Estimated Correlation | Regime Note |
> |---|---|---|
> | US Tech vs GCC Equity | 0.45 – 0.60 | Converges higher (~0.75+) in global risk-off |
> | US Tech vs Commodities | 0.20 – 0.35 | Can turn negative during supply shocks |
> | GCC Equity vs Commodities | 0.55 – 0.65 | Strong structural link via oil revenues |
> | GCC Equity vs US Tech | 0.40 – 0.55 | Partly decoupled via regional reform narrative |
>
> *These correlations are qualitative estimates based on observed historical relationships. A formal covariance matrix requires live market data and is updated quarterly.*

## 7. Implementation Plan
- Phased execution (3 tranches)
- Explicit rebalancing triggers: asset class drift ±5%, single-name cap {100/max(len(portfolio),1)*1.5:.0f}%, country/sector cap
- Monitoring: daily drawdown, weekly GCC macro, monthly attribution

## 8. Benchmark Comparison (table format with numbers)
Use this table structure:
| Benchmark | Est. Annual Return | Est. Volatility | This Portfolio vs Benchmark |
vs 60/40 Portfolio (~6-8% return, ~10% vol)
vs MSCI Emerging Markets (~8-10% return, ~18% vol)
vs S&P GCC Composite (~10-12% return, ~19% vol)
- Show THIS portfolio's metrics in the same table for direct comparison
- Include honest statement: "This is a [regional thematic / balanced / aggressive] mandate — performance will diverge significantly from global benchmarks during regional cycles"

## 9. Risk Warning
MUST include ALL of these:
- Market risk (equity drawdown)
- Concentration risk: economic concentration (not just position count) — sectors/themes that correlate
- Drawdown disclaimer: "The {mdd*100:.0f}% max drawdown is a risk management TARGET, not a guaranteed floor. In severe, correlated market dislocations, losses may exceed this threshold."
- Commodity-specific risk: "Commodity ETFs can decline alongside equities in global risk-off events and may not provide the hedging benefit anticipated in all market scenarios."
- Liquidity risk for GCC single names

STRICT RULES:
- Strategy name in Section 1 MUST be: "{strategy_label}" — not just "Aggressive Growth"
- Use ONLY the asset names/tickers above — never invent alternatives
- Use EXACT weights and metrics — do not round or change them
- Never write "Assumed" or "?" next to any asset
- If position_sizing_alert is YES: add a ⚠️ Position Sizing Alert box in Section 5
- Return clean Markdown only — no code blocks wrapping the whole document
"""

        client = get_client()
        resp = client.create_completion(
            model=config.DEFAULT_MODEL,
            temperature=0.2,
            max_tokens=7000,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an elite institutional portfolio strategist. "
                        "Produce complete, professional investment reports with exact numbers provided. "
                        "Never invent data. "
                        "Output ONLY clean Markdown — do NOT wrap the entire report in a ```markdown code block. "
                        "Use ## for section headers, | for tables, **bold** for emphasis. "
                        "The output will be rendered as HTML so must be valid Markdown."
                    )
                },
                {"role": "user", "content": prompt},
            ]
        )
        report = (resp.choices[0].message.content or "").strip()

        # Strip outer markdown code-block wrapper if LLM added one anyway
        import re as _re
        report = _re.sub(r"^```(?:markdown)?\s*\n", "", report)
        report = _re.sub(r"\n```\s*$", "", report)
        report = report.strip()

        logger.info("[Pipeline] Report generated (%d chars)", len(report))
        return report

    except Exception as e:
        logger.error("[Pipeline] Report generation failed: %s", e)
        return f"# Portfolio Report\n\nError generating report: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_metrics(portfolio: list[dict], risk: str) -> dict:
    """
    Estimate portfolio metrics from bucket composition.
    Uses conservative fundamental estimates, not historical data.
    """
    # Expected return estimates per bucket
    _BUCKET_MU = {
        "us_equity":    0.12,
        "gcc_equity":   0.13,
        "egypt_equity": 0.10,
        "bonds":        0.05,
        "gold":         0.07,
        "commodities":  0.08,
        "crypto":       0.18,
        "cash":         0.045,
    }
    # Volatility estimates per bucket
    _BUCKET_VOL = {
        "us_equity":    0.18,
        "gcc_equity":   0.20,
        "egypt_equity": 0.25,
        "bonds":        0.06,
        "gold":         0.15,
        "commodities":  0.22,
        "crypto":       0.55,
        "cash":         0.00,
    }

    exp_ret = sum(p["weight"] * _BUCKET_MU.get(p["bucket"], 0.08) for p in portfolio)
    vol     = sum(p["weight"] * _BUCKET_VOL.get(p["bucket"], 0.15) for p in portfolio)
    rf      = 0.045
    sharpe  = (exp_ret - rf) / vol if vol > 0 else 0

    return {
        "expected_return": round(exp_ret, 4),
        "volatility":      round(vol, 4),
        "sharpe":          round(sharpe, 4),
    }


def run(user_message: str) -> str:
    """
    Main pipeline entry point.
    Takes user message, runs all 4 steps, returns full report markdown.
    """
    logger.info("[Pipeline] Starting pipeline for: %s", user_message[:80])

    # Step 1 — Parse intent
    intent = parse_intent(user_message)

    # Step 2 — Allocation buckets
    allocation = build_allocation(intent)

    # Step 3 — Select real assets
    assets = select_assets(allocation, intent)

    # Step 4 — Assign weights
    portfolio = build_portfolio(allocation, assets)

    if not portfolio:
        return "# ⚠️ Could not build portfolio — no assets found in cache. Try again later."

    # Estimate metrics
    metrics = _estimate_metrics(portfolio, intent.get("risk", "balanced"))

    # Step 5 — Generate report
    report = generate_report(portfolio, allocation, intent, metrics)

    logger.info("[Pipeline] Pipeline complete — %d assets, return=%.1f%%, sharpe=%.2f",
                len(portfolio), metrics["expected_return"]*100, metrics["sharpe"])
    return report


# ─────────────────────────────────────────────────────────────────────────────
# INTENT DETECTION — is this message a pipeline portfolio request?
# ─────────────────────────────────────────────────────────────────────────────

_PIPELINE_SIGNALS_AR = [
    "ابني", "ابنى", "إبني", "اعمل محفظة", "اعمل محفظه",
    "عايز محفظة", "عاوز محفظة", "عايز محفظه", "عاوز محفظه",
    "بناء محفظة", "صمم محفظة", "محفظة متوازنة", "محفظة هجومية",
    "محفظة محافظة", "محفظة نمو", "انشئ محفظة", "أنشئ محفظة",
    "محفظه متوازنه", "محفظه هجوميه", "محفظه محافظه",
]
_PIPELINE_SIGNALS_EN = [
    "build me a portfolio", "build a portfolio", "create a portfolio",
    "construct a portfolio", "design a portfolio", "make me a portfolio",
    "set up a portfolio", "build portfolio", "i want a portfolio",
    "i need a portfolio",
]

# Action verbs + portfolio noun — catch "build me a balanced portfolio", etc.
_BUILD_VERBS = ["build", "create", "construct", "design", "make", "generate", "set up", "setup"]
_PORTFOLIO_NOUNS = ["portfolio", "fund", "allocation"]

def is_pipeline_request(message: str) -> bool:
    ml = message.lower()
    # Arabic signals
    if any(s in ml for s in _PIPELINE_SIGNALS_AR):
        return True
    # Exact English phrases
    if any(s in ml for s in _PIPELINE_SIGNALS_EN):
        return True
    # Flexible: verb + portfolio anywhere in message
    has_verb = any(v in ml for v in _BUILD_VERBS)
    has_noun = any(n in ml for n in _PORTFOLIO_NOUNS)
    return has_verb and has_noun
