from __future__ import annotations
import logging
import json
import re
import time
from typing import Any, List, Optional, Dict
import pandas as pd
import config
from core.llm import get_client
from core.data import get_prices, to_returns
from core.policy import apply_policy
from core.metrics import perf_metrics
from state import SYSTEM_PROMPTS
logger = logging.getLogger(__name__)

from core.pm_helpers import _kv, get_param, parse_float, parse_int, _fmt_pct, _fmt_float, render_weights, detect_risk_pref, recommend_etfs, method_from_risk

def _normalize_tickers(tickers: list[str]) -> list[str]:
    """Apply _TICKER_MAP and deduplicate."""
    result = []
    seen = set()
    for t in tickers:
        mapped = _TICKER_MAP.get(t.upper(), t)
        if mapped and mapped not in seen:
            seen.add(mapped)
            result.append(mapped)
    return result


# ── Verified ticker → company name lookup ─────────────────────────────────────
# Used by the report generation to replace "Assumed" with verified company names.
# Prevents the LLM from hallucinating "STC BANK?" or "TASNEE?".
_TICKER_NAMES: dict[str, str] = {
    # Saudi Arabia (Tadawul)
    "2222.SR": "Saudi Aramco",
    "1120.SR": "Al Rajhi Bank",
    "1180.SR": "Alinma Bank",
    "1150.SR": "Riyad Bank",
    "1140.SR": "Al-Awwal Bank (STC Bank)",
    "1060.SR": "National Commercial Bank (SNB)",
    "1010.SR": "Riyad Capital",
    "2010.SR": "SABIC (Saudi Basic Industries)",
    "2380.SR": "Petro Rabigh",
    "4001.SR": "Abdullah Al Othaim Markets",
    "3010.SR": "Saudi Kayan Petrochemical",
    "1050.SR": "Banque Saudi Fransi",
    "2060.SR": "National Industrialization Co. (TASNEE)",
    "1211.SR": "Ma'aden (Saudi Arabian Mining)",
    "7010.SR": "STC (Saudi Telecom Company)",
    "7030.SR": "Zain KSA (Mobile Telecom)",
    "4240.SR": "Nahdi Medical Company",
    "8010.SR": "Tawuniya (Cooperative Insurance)",
    "2220.SR": "SIPCHEM",
    "2280.SR": "Riyad REIT ETF",
    "4002.SR": "Al-Shiddi International",
    "1302.SR": "Amlak International Finance",
    "1324.SR": "Ataa Educational Company",
    # UAE
    "EMAAR.DU":      "Emaar Properties",
    "FAB.AE":        "First Abu Dhabi Bank",
    "ADCB.AE":       "Abu Dhabi Commercial Bank",
    "DIB.DU":        "Dubai Islamic Bank",
    "ADNOCDIST.AE":  "ADNOC Distribution",
    "ETISALAT.AE":   "e& (Etisalat)",
    "DU.DU":         "du (Emirates Integrated Telecom)",
    "EMAARDEV.DU":   "Emaar Development",
    "BOROUGE.AE":    "Borouge (Abu Dhabi Polymers)",
    # Egypt
    "COMI.CA":  "Commercial International Bank (CIB)",
    "TMGH.CA":  "Talaat Moustafa Group Holding",
    "ETEL.CA":  "Telecom Egypt",
    "HRHO.CA":  "Heliopolis Housing",
    "SWDY.CA":  "Edita Food Industries (Swady)",
    "AMIA.CA":  "Arab Moltaka Investments",
    "TAQA.CA":  "TAQA Arabia",
    # ETFs & proxies
    "KSA":    "iShares MSCI Saudi Arabia ETF",
    "GULF":   "WisdomTree Middle East Dividend Fund",
    "EGPT":   "VanEck Egypt Index ETF",
    "QAT":    "iShares MSCI Qatar ETF",
    "EWU":    "iShares MSCI United Kingdom ETF",
    "EEM":    "iShares MSCI Emerging Markets ETF",
    "GLD":    "SPDR Gold Shares",
    "IAU":    "iShares Gold Trust",
    "SPY":    "SPDR S&P 500 ETF",
    "QQQ":    "Invesco Nasdaq-100 ETF",
    "BND":    "Vanguard Total Bond Market ETF",
    "AGG":    "iShares Core U.S. Aggregate Bond ETF",
    "USO":    "United States Oil Fund",
    "SLV":    "iShares Silver Trust",
    "CPER":   "United States Copper Index Fund",
}

# All known placeholder / fake tickers that must never appear in any report
_ALL_FAKE_TICKERS: set[str] = {
    "NEEDED", "ASSET", "STOCK", "INDEX", "TICKER", "SYMBOL", "PLACEHOLDER",
    "ARAB", "MARKET", "EQUITY", "SHARE", "ITEM", "OTHER", "CASH", "FUND",
    "AS", "AN", "IN", "TO", "OF", "OR", "AT", "BY",
    "ADD", "NEW", "SET", "USE", "GET", "PUT", "TBD", "NA", "XX",
    "UAE", "SAUDI", "ARABIA", "GCC", "EGYPT", "DUBAI", "KUWAIT",
    "BAHRAIN", "OMAN", "MENA", "QATAR",
}


def has_placeholder_tickers(weights: dict) -> list[str]:
    """Return list of any placeholder/fake tickers found in a weights dict."""
    return [t for t in weights if t.upper() in _ALL_FAKE_TICKERS]


def get_ticker_name(ticker: str) -> str:
    """Return verified company name for a ticker, or the ticker itself if unknown."""
    return _TICKER_NAMES.get(ticker, ticker)

# Leveraged / inverse ETFs — never include unless user explicitly asks for "leverage"
_LEVERAGED_ETFS = {
    "UPRO", "SPXU", "TQQQ", "SQQQ", "SOXL", "SOXS",
    "FNGU", "FNGD", "TNA", "TZA", "NUGT", "DUST",
    "JNUG", "JDST", "UVXY", "VXX", "LABU", "LABD",
    "SSO", "SDS", "QLD", "QID", "USD", "ERX", "ERY",
    "SPXL", "SPXS", "TECL", "TECS", "NAIL", "DRN", "DRV",
}


# ── TradingView exchange prefix → yfinance suffix ────────────────────────────
_TV_SUFFIX_MAP = {
    "TADAWUL": ".SR",
    "DFM":     ".DU",
    "ADX":     ".AE",
    "EGX":     ".CA",
    "KSE":     ".KW",
    "QSE":     ".QA",
    "BSE":     ".BH",
    "MSM":     ".OM",
    "CSE":     ".TN",
    "CBSE":    ".MA",
}

def _tv_to_yfinance(tv_ticker: str) -> str:
    """Convert TradingView format (TADAWUL:2222) to yfinance format (2222.SR)."""
    if ":" in tv_ticker:
        exchange, symbol = tv_ticker.split(":", 1)
        suffix = _TV_SUFFIX_MAP.get(exchange.upper(), "")
        return f"{symbol}{suffix}"
    return tv_ticker


def get_top_regional_tickers(user_message: str, n_each: int = 3) -> list[str]:
    """
    Pull top-scored stocks from the live market cache for any GCC/Arab
    market mentioned in the user message.
    Returns yfinance-format tickers (e.g. 2222.SR, COMI.CA, EMAAR.DU).
    Falls back to ETF proxies if cache is unavailable.
    """
    ml = user_message.lower()

    # Detect which markets to load
    market_map = []
    if any(w in ml for w in ["سعودي", "سعودية", "سعوديه", "saudi", "ksa", "تداول", "ارامكو", "أرامكو", "aramco"]):
        market_map.append("ksa")
    if any(w in ml for w in ["امارات", "إمارات", "اماراتي", "إماراتي", "uae", "dubai", "دبي", "ابوظبي", "أبوظبي"]):
        market_map.append("uae")
    if any(w in ml for w in ["مصر", "مصري", "مصريه", "egypt", "egx", "البورصه", "البورصة"]):
        market_map.append("egypt")
    if any(w in ml for w in ["كويت", "kuwait"]):
        market_map.append("kuwait")
    if any(w in ml for w in ["قطر", "قطري", "qatar"]):
        market_map.append("qatar")
    if any(w in ml for w in ["خليج", "gcc", "gulf", "خليجي", "خليجيه"]) and not market_map:
        market_map.extend(["ksa", "uae"])

    if not market_map:
        return []

    try:
        import sys, os
        _base = os.path.join(os.path.dirname(os.path.dirname(__file__)))
        if _base not in sys.path:
            sys.path.insert(0, _base)
        from global_allocator import _select_top_stocks, _load_latest_snapshot
        import pandas as pd

        results = []
        for market_code in market_map:
            df = _load_latest_snapshot(market_code)
            if df is None or df.empty:
                logger.warning("[RegionalTickers] No cache for %s", market_code)
                continue

            # Score stocks (same logic as global_allocator)
            def _score(row) -> float:
                s = 0.0
                rsi = row.get("RSI")
                if rsi is not None:
                    try:
                        r = float(rsi)
                        s += 1.0 if 40 <= r <= 65 else 0.4 if 30 <= r <= 75 else 0.0
                    except Exception:
                        pass
                chg = float(row.get("change") or 0)
                s += 0.5 if chg > 0 else 0.0
                macd = row.get("MACD.macd"); sig = row.get("MACD.signal")
                if macd is not None and sig is not None:
                    try:
                        s += 0.5 if float(macd) > float(sig) else 0.0
                    except Exception:
                        pass
                cl = float(row.get("close") or 0)
                s50 = float(row.get("SMA50") or 0)
                s200 = float(row.get("SMA200") or 0)
                if cl > 0 and s50 > 0 and s200 > 0:
                    s += 1.0 if cl > s50 > s200 else 0.0
                pe = float(row.get("price_earnings_ttm") or 0)
                s += 1.0 if 5 < pe < 25 else 0.5 if pe == 0 else 0.0
                mc = float(row.get("market_cap_basic") or 0)
                s += 1.0 if mc > 5e9 else 0.5 if mc > 1e9 else 0.0
                return s

            df = df.copy()
            df = df[(df["close"] > 0) & (df["market_cap_basic"].fillna(0) > 0)]
            df["_score"] = df.apply(_score, axis=1)
            top = df.nlargest(n_each, "_score")

            for _, row in top.iterrows():
                tv_ticker = str(row.get("ticker") or "")
                yf_ticker = _tv_to_yfinance(tv_ticker)
                if yf_ticker and len(yf_ticker) >= 3:
                    results.append(yf_ticker)
                    logger.info("[RegionalTickers] %s → %s (score=%.1f, sector=%s)",
                                tv_ticker, yf_ticker, row["_score"], row.get("sector","?"))

        if results:
            logger.info("[RegionalTickers] Selected from cache: %s", results)
        return results

    except Exception as e:
        logger.warning("[RegionalTickers] Cache lookup failed: %s", e)
        return []


def smart_expand_tickers(user_message: str, extracted_tickers: list[str]) -> list[str]:
    """
    Use GPT to intelligently expand user's asset requests into valid tickers.
    Leveraged/inverse ETFs are blocked unless the user explicitly mentions 'leverage'.
    For GCC/Arab market requests, uses the live market cache instead of LLM guessing.
    """
    low = user_message.lower()
    asset_keywords = [
        "crypto", "bitcoin", "ethereum", "btc", "eth", "solana", "xrp",
        "bonds", "gold", "silver", "commodities", "oil", "treasury",
        "real estate", "reit", "property",
        "emerging", "international", "uae", "gcc", "abu dhabi", "dubai", "saudi",
        "tech", "healthcare", "energy", "financial", "utilities", "industrial",
        "aggressive", "conservative", "balanced", "growth", "value", "dividend",
        "all-weather", "defensive", "high yield",
        "etf", "stocks", "diversified", "mix"
    ]

    needs_expansion = any(kw in low for kw in asset_keywords)
    if not needs_expansion and len(extracted_tickers) >= 2:
        return extracted_tickers

    # ── Step 1: Try live market cache for GCC/Arab markets FIRST ─────────────
    # This gives real, scored, fundamentally-sound tickers instead of LLM guesses.
    _regional_kws = [
        "سعودي", "سعودية", "saudi", "ksa", "تداول", "ارامكو",
        "امارات", "إمارات", "uae", "dubai", "دبي",
        "مصر", "egypt", "egx", "البورصة",
        "كويت", "kuwait", "قطر", "qatar",
        "خليج", "gcc", "gulf", "خليجي",
    ]
    if any(kw in low for kw in _regional_kws):
        cache_tickers = get_top_regional_tickers(user_message, n_each=3)
        if cache_tickers:
            # Validate with yfinance — GCC suffixed tickers (.SR/.CA/.AE/.DU) are
            # trusted even if yfinance is slow, because they follow the real format.
            # Only skip a ticker if it explicitly has NO price AND no recognised suffix.
            _GCC_SUFFIXES = (".SR", ".AE", ".DU", ".CA", ".KW", ".QA", ".BH", ".OM")
            try:
                import yfinance as _yf
                validated = []
                for _t in cache_tickers:
                    # Trust GCC-format tickers — don't let slow yfinance drop them
                    if any(_t.endswith(sfx) for sfx in _GCC_SUFFIXES):
                        validated.append(_t)
                        continue
                    try:
                        _p = getattr(_yf.Ticker(_t).fast_info, "last_price", None)
                        if _p and float(_p) > 0:
                            validated.append(_t)
                    except Exception:
                        pass
                if validated:
                    logger.info("[SmartExpand] Using cache tickers: %s", validated)
                    return validated
            except Exception:
                # yfinance unavailable — trust GCC format tickers directly
                gcc_trusted = [t for t in cache_tickers if any(t.endswith(sfx) for sfx in _GCC_SUFFIXES)]
                if gcc_trusted:
                    logger.info("[SmartExpand] yfinance unavailable, trusting GCC tickers: %s", gcc_trusted)
                    return gcc_trusted

        # Cache found regional request but returned no tickers — use safe ETF fallbacks
        # instead of falling through to LLM (which may hallucinate AS, NEEDED, etc.)
        _etf_fallback = []
        if any(kw in low for kw in ["سعودي", "سعودية", "saudi", "ksa", "تداول"]):
            _etf_fallback.append("KSA")
        if any(kw in low for kw in ["امارات", "إمارات", "uae", "dubai", "دبي"]):
            _etf_fallback.append("GULF")
        if any(kw in low for kw in ["مصر", "egypt", "egx"]):
            _etf_fallback.append("EGPT")
        if any(kw in low for kw in ["خليج", "gcc", "خليجي"]) and not _etf_fallback:
            _etf_fallback.extend(["KSA", "GULF"])
        if _etf_fallback:
            logger.info("[SmartExpand] Cache empty, using ETF fallbacks: %s", _etf_fallback)
            return _etf_fallback
    # ── End cache lookup ──────────────────────────────────────────────────────

    # Only allow leveraged ETFs when explicitly requested
    user_wants_leverage = any(kw in low for kw in ["leveraged", "leverage", "3x", "2x", "ultra pro"])

    try:
        client = get_client()
        response = client.create_completion(
            model=config.DEFAULT_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial ticker expert. The user wants to build a portfolio.\n"
                        "Convert their request into a list of VALID Yahoo Finance ticker symbols.\n\n"
                        "RULES:\n"
                        "- For US stocks: use standard symbols (AAPL, MSFT, GOOGL, NVDA, AMZN)\n"
                        "- For crypto: use Yahoo format (BTC-USD, ETH-USD, SOL-USD, XRP-USD)\n"
                        "- For bonds: use ETFs (TLT, BND, AGG, IEF, SHY)\n"
                        "- For gold: use GLD or IAU\n"
                        "- For silver: use SLV\n"
                        "- For oil/commodities: use USO, XLE, XOP\n"
                        "- For real estate/REIT: use VNQ, IYR, XLRE\n"
                        "- For international/emerging: use VEA, VWO, EFA, IEMG\n"
                        "- For Saudi Arabia: use KSA (iShares MSCI Saudi Arabia ETF)\n"
                        "- For UAE: use GULF (WisdomTree Middle East) — NEVER use 'UAE' as a ticker\n"
                        "- For GCC region: use KSA + GULF\n"
                        "- For individual Saudi stocks: use real tickers like 2222.SR, 2010.SR, 7010.SR, 7030.SR\n"
                        "- For individual UAE stocks: use EMAAR.DU, FAB.AE, ADCB.AE\n"
                        "- For Egypt: use EGPT (VanEck Egypt Index ETF)\n"
                        "- For sectors: XLK=tech, XLF=financial, XLE=energy, XLV=healthcare, XLI=industrial\n"
                        "- For aggressive/growth: lean toward tech, crypto, growth (QQQ, NVDA, BTC-USD, SPY)\n"
                        "- For conservative/safe: bonds, utilities, dividend stocks\n"
                        "- For balanced: mix of stocks, bonds, alternatives\n"
                        "- For all-weather: stocks, bonds, gold, REITs\n"
                        "- Keep portfolio to 4-8 tickers for good diversification\n"
                        "- NEVER use: 'UAE', 'SAUDI', 'ARABIA', 'GCC', 'EGYPT', 'DUBAI', 'KUWAIT' as tickers — these are NOT valid\n"
                        "- NEVER include leveraged or inverse ETFs unless explicitly asked\n\n"
                        "Respond with ONLY a comma-separated list of ticker symbols, nothing else.\n"
                        "Example: KSA, GULF, EGPT, GLD, QQQ"
                    ),
                },
                {"role": "user", "content": user_message},
            ],
        )
        content = response.choices[0].message.content.strip()
        clean_tickers = [t.strip().upper() for t in content.split(",") if t.strip()]
        valid = [t for t in clean_tickers if 2 <= len(t) <= 12]

        # ── Remove fake/placeholder tickers ──────────────────────────────────
        _FAKE_TICKERS = {
            "TEST", "YEAR", "STRESS", "MONTE", "CARLO", "OIL", "TECH", "BOND",
            "GOLD", "CASH", "BULL", "BEAR", "RISK", "SAFE", "FUND", "PLAN",
            "CIO", "RSI", "MACD", "EMA", "SMA", "ATR", "ADX", "HOLD", "SELL",
            "BUY", "STOP", "LOSS", "GAIN", "COST", "BASE", "RATE", "TIME",
            # Regional names that are NOT valid yfinance tickers
            "UAE", "SAUDI", "ARABIA", "GCC", "EGYPT", "DUBAI", "KUWAIT",
            "BAHRAIN", "OMAN", "MENA",
            # Common LLM placeholder words that slip through
            "NEEDED", "ASSET", "STOCK", "INDEX", "TICKER", "SYMBOL", "NAME",
            "ARAB", "MARKET", "EQUITY", "SHARE", "ITEM", "TYPE", "OTHER",
            "ADD", "NEW", "SET", "USE", "GET", "PUT", "ONE", "TWO",
            "AS", "AN", "THE", "IN", "TO", "OF",  # common 2-letter english words
        }

        def _is_plausible_ticker(t: str) -> bool:
            """
            Reject tickers that look like English placeholder words.
            Real tickers: AAPL, BTC-USD, 2222.SR, EMAAR.DU, QQQ, KSA, GLD
            Fake ones: NEEDED, AS, ARAB, ASSET
            """
            # Allow GCC-style tickers: digits + dot + 2-letter market code
            if "." in t:
                return True
            # Allow crypto format: XXX-USD
            if "-" in t:
                return True
            # Allow known short real tickers (1-3 chars): SPY, QQQ, GLD, KSA…
            if len(t) <= 3:
                return t not in _FAKE_TICKERS
            # For 4+ char tickers: reject if they look like plain English words
            # (all alpha AND appear to be a common word — heuristic: mixed or all-caps fine)
            # Flag if the lowercase version is a common English word > 4 letters
            _ENGLISH_WORDS = {
                "needed", "asset", "stock", "index", "ticker", "market",
                "equity", "share", "other", "added", "total", "value",
                "small", "large", "fixed", "short", "long", "type",
            }
            if t.lower() in _ENGLISH_WORDS:
                return False
            return True

        # Apply normalization map first, then fake/plausibility filters
        valid = [_TICKER_MAP.get(t, t) for t in valid if t not in _FAKE_TICKERS]
        valid = [t for t in valid if t and _is_plausible_ticker(t)]

        # ── Validate tickers actually exist on yfinance (batch check) ────────
        try:
            import yfinance as _yf
            _validated = []
            for _t in valid:
                try:
                    _info = _yf.Ticker(_t).fast_info
                    _price = getattr(_info, "last_price", None)
                    if _price and float(_price) > 0:
                        _validated.append(_t)
                    else:
                        logger.info("[SmartExpand] Skipping invalid ticker: %s (no price)", _t)
                except Exception:
                    logger.info("[SmartExpand] Skipping invalid ticker: %s", _t)
            if _validated:
                valid = _validated
                logger.info("[SmartExpand] Validated tickers: %s", valid)
        except Exception as _ve:
            logger.warning("[SmartExpand] Ticker validation failed: %s", _ve)

        # Strip leveraged ETFs unless user explicitly wants them
        if not user_wants_leverage:
            filtered = [t for t in valid if t not in _LEVERAGED_ETFS]
            if filtered:
                removed = set(valid) - set(filtered)
                if removed:
                    logger.info("[SmartExpand] Removed leveraged ETFs: %s", removed)
                valid = filtered

        return valid
    except Exception as e:
        logger.error(f"[SmartExpand] Error: {e}")
        return extracted_tickers

# ============================================================
# RISK SCORING
# ============================================================
_RISK_LABELS = {
    1: "Very Conservative", 2: "Very Conservative",
    3: "Conservative",      4: "Conservative",
    5: "Moderate",          6: "Balanced",
    7: "Growth",            8: "Growth",
    9: "Aggressive",       10: "Very Aggressive",
}
_RISK_EMOJIS = {
    1: "🔵", 2: "🔵", 3: "🟢", 4: "🟢",
    5: "🟡", 6: "🟡", 7: "🟠", 8: "🟠",
    9: "🔴", 10: "🔴",
}

