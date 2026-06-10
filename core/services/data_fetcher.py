"""
core/services/data_fetcher.py
──────────────────────────────
Parallel data fetching for the EisaX analytics pipeline.

All 9 network sources run concurrently inside a ThreadPoolExecutor,
reducing total fetch time from ~15 s to ~4 s.

Public API
──────────
    fetch_all(target, *, timeout=25) -> FetchResult
        Fire all 9 sources in parallel and return a unified FetchResult
        dataclass.  Falls back gracefully if individual sources fail.

    resolve_technicals(target, prices, is_local_market) -> dict
        Compute technical summary, VaR, and max-drawdown from price series.
        Includes UAE/regional Parquet-cache fallback.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    """All data produced by the parallel fetch stage."""
    # Price / technical
    real_price:       float | None = None
    change_pct:       float        = 0.0
    summary:          dict         = field(default_factory=dict)
    series:           Any          = None        # pandas Series
    var_95:           float        = 0.02
    max_dd:           float        = 0.20

    # Fundamentals
    fund:             dict         = field(default_factory=dict)
    dc_data:          dict         = field(default_factory=dict)
    profile:          dict         = field(default_factory=dict)
    yf_info:          dict         = field(default_factory=dict)

    # Macro / market
    fg_data:          dict         = field(default_factory=lambda: {"score": 50, "rating": "Neutral", "label_ar": ""})
    t10y:             str          = "N/A"
    fed:              str          = "N/A"
    unemp:            str          = "N/A"
    inflation:        str          = "N/A"
    gdp:              str          = "N/A"
    next_earnings:    str | None   = None
    ev_out:           dict         = field(default_factory=dict)

    # News / sentiment
    engine_news:      dict         = field(default_factory=dict)  # {direct,sector,country,related,meta}
    news_links:       list         = field(default_factory=list)
    news_sent:        str          = "N/A"
    news_score:       float        = 0.0

    # X / Grok
    x_data:           dict | None  = None

    # Local enrichment (UAE/GCC fallback)
    local_enriched:   dict         = field(default_factory=dict)

    # Analyst
    analyst_target:   float | None = None
    analyst_consensus: str | None  = None
    analyst_count:    int | None   = None
    forward_pe:       float | None = None


# ── Individual fetchers (each callable by the executor) ───────────────────────

def _fetch_profile(target: str) -> dict:
    try:
        from core.market_data import get_full_stock_profile
        return get_full_stock_profile(target) or {}
    except Exception as exc:
        logger.debug("[DataFetcher] profile failed %s: %s", target, exc)
        return {}


def _fetch_fund(target: str) -> dict:
    try:
        from core.fundamental_engine import get_fundamentals
        return get_fundamentals(target) or {}
    except Exception as exc:
        logger.debug("[DataFetcher] fund failed %s: %s", target, exc)
        return {}


def _fetch_deepcrawl(target: str) -> dict:
    try:
        from core.realtime_data import deepcrawl_stock
        return deepcrawl_stock(target) or {}
    except Exception as exc:
        logger.debug("[DataFetcher] deepcrawl failed %s: %s", target, exc)
        return {}


def _fetch_yf(target: str) -> tuple[Any, dict]:
    try:
        from core.utils import yf_retry
        return yf_retry(target)
    except Exception as exc:
        logger.debug("[DataFetcher] yfinance failed %s: %s", target, exc)
        import yfinance as yf
        return yf.Ticker(target), {}


def _fetch_prices(target: str) -> Any:
    try:
        from core.data import get_prices
        return get_prices([target], "2023-01-01", None)
    except Exception as exc:
        logger.debug("[DataFetcher] prices failed %s: %s", target, exc)
        import pandas as pd
        return pd.DataFrame()


def _fetch_fear_greed() -> dict:
    try:
        from core.rapid_data import get_fear_greed
        return get_fear_greed() or {}
    except Exception as exc:
        logger.debug("[DataFetcher] fear&greed failed: %s", exc)
        return {"score": 50, "rating": "Neutral", "label_ar": ""}


def _fetch_events(target: str) -> dict:
    try:
        from core.rapid_data import get_events_calendar
        return get_events_calendar(target) or {}
    except Exception as exc:
        logger.debug("[DataFetcher] events failed %s: %s", target, exc)
        return {}


def _fetch_engine_news(target: str) -> dict:
    try:
        from core.news_engine_client import get_ticker_news
        return get_ticker_news(target) or {}
    except Exception as exc:
        logger.debug("[DataFetcher] engine_news failed %s: %s", target, exc)
        return {}


def _fetch_grok(target: str, asset_name: str = "", sector: str = "") -> dict | None:
    try:
        from core.grok_client import get_x_sentiment
        return get_x_sentiment(target, asset_name, sector)
    except Exception as exc:
        logger.debug("[DataFetcher] grok failed %s: %s", target, exc)
        return None


# ── UAE / regional Parquet-cache fallback ─────────────────────────────────────

def _uae_parquet_fallback(target: str) -> dict:
    """
    When yfinance fails for a regional ticker (.AE/.DU/.SR/.CA etc.),
    load historical price data from the local Parquet cache and recompute
    technical indicators from it.

    Returns a dict with keys matching FetchResult fields that need updating.
    """
    import pandas as _pd
    import core.analytics as ca

    result: dict = {
        "series":   _pd.Series(dtype=float),
        "summary":  {
            "price": 0, "trend": "N/A", "momentum": "N/A", "condition": "N/A",
            "rsi": 50.0, "sma_50": 0.0, "sma_200": 0.0,
            "adx": 0.0, "atr": 0.0, "macd": 0.0, "macd_signal": 0.0,
        },
        "var_95": 0.02,
        "max_dd": 0.20,
        "local_enriched": {},
    }

    t_upper = target.upper()
    _mkt = (
        "AE" if t_upper.endswith((".AE", ".DU")) else
        "SA" if t_upper.endswith(".SR") else
        "EG" if t_upper.endswith(".CA") else
        "KW" if t_upper.endswith(".KW") else
        "QA" if t_upper.endswith(".QA") else None
    )
    if not _mkt:
        return result

    # 1. Load from Parquet cache
    _df_cache = None
    try:
        from core.market_data_engine import get_stock_data as _get_mde
        _df_cache = _get_mde(target, _mkt, period="5y", force_refresh=False)
        if _df_cache is not None and not _df_cache.empty and "Close" in _df_cache.columns:
            result["series"] = _df_cache["Close"].copy()
            logger.info("[UAE Fallback] Loaded %d rows from Parquet cache", len(result["series"]))
    except Exception as exc:
        logger.warning("[UAE Fallback] Parquet load failed: %s", exc)

    # 2. Compute technicals from historical data
    series = result["series"]
    if not series.empty and len(series) > 30:
        try:
            _tech_input = (
                _df_cache if (
                    _df_cache is not None
                    and not _df_cache.empty
                    and all(c in _df_cache.columns for c in ("High", "Low", "Close"))
                ) else series
            )
            result["summary"] = ca.generate_technical_summary(target, _tech_input)
            _returns = series.pct_change().dropna()
            result["var_95"] = ca.calculate_var(_returns)
            result["max_dd"] = ca.calculate_max_drawdown(series)
            logger.info(
                "[UAE Fallback] ✅ Calculated from %d data points: RSI=%s SMA50/200=%s",
                len(series), result["summary"].get("rsi"), result["summary"].get("sma_50"),
            )
        except Exception as exc:
            logger.warning("[UAE Fallback] Technical calc failed: %s", exc)

    # 3. Enrich with local fundamentals (DFM/sector data)
    try:
        from core.local_market_enricher import enrich_local_analysis
        result["local_enriched"] = enrich_local_analysis(target) or {}
    except Exception as exc:
        logger.debug("[UAE Fallback] local enrichment failed: %s", exc)

    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def fetch_all(target: str, *, timeout: int = 25) -> FetchResult:
    """
    Cache-first fetch strategy:

    LOCAL MARKET TICKERS (.SR .AE .CA .KW .QA .BH):
      1. Check pipeline cache (TradingView, 15-min snapshots)
      2. If cache is fresh  → build FetchResult from cache instantly
                              then enrich with live extras (news, F&G, earnings)
      3. If cache is stale/missing → fall through to full 9-source parallel fetch

    GLOBAL TICKERS (NVDA, BTC-USD, GC=F ...):
      → Always run full 9-source parallel fetch (yfinance is reliable here)
    """
    _LOCAL_SUFFIXES = (".SR", ".AE", ".DU", ".CA", ".KW", ".QA", ".BH")
    _is_local = target.upper().endswith(_LOCAL_SUFFIXES)

    if _is_local:
        _cache_res = _try_build_from_cache(target)
        if _cache_res is not None:
            logger.info("[DataFetcher] %s: cache-first hit — skipping slow fetch", target)
            return _enrich_with_live_extras(_cache_res, target, timeout)
        logger.info("[DataFetcher] %s: cache miss/stale — falling back to full fetch", target)

    import core.analytics as ca

    res = FetchResult()

    with ThreadPoolExecutor(max_workers=9) as exe:
        futures = {
            "profile":    exe.submit(_fetch_profile,     target),
            "fund":       exe.submit(_fetch_fund,        target),
            "dc":         exe.submit(_fetch_deepcrawl,   target),
            "yf":         exe.submit(_fetch_yf,          target),
            "prices":     exe.submit(_fetch_prices,      target),
            "fg":         exe.submit(_fetch_fear_greed),
            "events":     exe.submit(_fetch_events,      target),
            "news":       exe.submit(_fetch_engine_news, target),
            "grok":       exe.submit(_fetch_grok,        target),
        }

        # ── profile ───────────────────────────────────────────────────────────
        try:
            res.profile = futures["profile"].result(timeout=timeout) or {}
        except Exception as exc:
            logger.debug("[DataFetcher] profile timeout/error %s: %s", target, exc)

        # ── fund (fundamentals) ───────────────────────────────────────────────
        try:
            res.fund = futures["fund"].result(timeout=timeout) or {}
        except Exception as exc:
            logger.debug("[DataFetcher] fund timeout/error %s: %s", target, exc)

        # ── deepcrawl ─────────────────────────────────────────────────────────
        try:
            res.dc_data = futures["dc"].result(timeout=timeout) or {}
        except Exception as exc:
            logger.debug("[DataFetcher] deepcrawl timeout/error %s: %s", target, exc)

        # ── yfinance: price + macro + news_links ──────────────────────────────
        try:
            _yf_ticker, _yf_info = futures["yf"].result(timeout=timeout)
            res.yf_info = _yf_info or {}
            _yf_info = res.yf_info  # alias

            # Live price
            _price_raw = (
                _yf_info.get("currentPrice") or _yf_info.get("regularMarketPrice") or
                _yf_info.get("ask") or _yf_info.get("bid") or 0
            )
            if _price_raw:
                res.real_price = float(_price_raw)
                res.change_pct = float(
                    _yf_info.get("regularMarketChangePercent") or
                    _yf_info.get("52WeekChange") or 0
                ) * (100 if abs(float(_yf_info.get("regularMarketChangePercent", 1) or 1)) < 1 else 1)

            # Macro
            _macro_keys = {
                "t10y":      ("10-year", "us10y", "t10y"),
                "fed":       ("fed", "federal_funds_rate", "fedfunds"),
                "unemp":     ("unemployment", "unemp"),
                "inflation": ("cpi", "inflation"),
                "gdp":       ("gdp",),
            }
            _macro = res.profile.get("macro") or {}
            for attr, keys in _macro_keys.items():
                for k in keys:
                    v = _macro.get(k) or _yf_info.get(k)
                    if v is not None:
                        setattr(res, attr, str(round(float(v), 2)))
                        break

            # Flat news links (fallback)
            res.news_links = _yf_info.get("news") or []

            # Forward PE
            fp = _yf_info.get("forwardPE")
            if fp:
                try:
                    res.forward_pe = float(fp)
                except Exception as exc:
                    logger.warning(
                        "[DataFetcher] invalid forwardPE for %s: %r (%s)",
                        target,
                        fp,
                        exc,
                    )
        except Exception as exc:
            logger.debug("[DataFetcher] yfinance timeout/error %s: %s", target, exc)

        # ── prices → technicals ───────────────────────────────────────────────
        _is_local_market = target.upper().endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
        try:
            prices_df = futures["prices"].result(timeout=timeout)
            if prices_df is None or prices_df.empty:
                raise ValueError("empty prices")
            series = prices_df[target]
            res.series  = series
            res.summary = ca.generate_technical_summary(target, series)
            _returns    = series.pct_change().dropna()
            res.var_95  = ca.calculate_var(_returns)
            res.max_dd  = ca.calculate_max_drawdown(series)
            # Sync price from series if not already set
            if not res.real_price and not series.empty:
                res.real_price = float(series.iloc[-1])
        except Exception as exc:
            logger.debug("[DataFetcher] prices timeout/error %s: %s — trying UAE fallback", target, exc)
            if _is_local_market:
                _fb = _uae_parquet_fallback(target)
                res.series        = _fb["series"]
                res.summary       = _fb["summary"]
                res.var_95        = _fb["var_95"]
                res.max_dd        = _fb["max_dd"]
                res.local_enriched= _fb["local_enriched"]

        # ── fear & greed ──────────────────────────────────────────────────────
        try:
            fg = futures["fg"].result(timeout=10) or {}
            if fg:
                res.fg_data = fg
        except Exception as exc:
            logger.debug("[DataFetcher] fear&greed timeout: %s", exc)

        # ── events / earnings ─────────────────────────────────────────────────
        try:
            ev = futures["events"].result(timeout=10) or {}
            res.next_earnings = ev.get("next_earnings_date")
            res.ev_out        = ev
        except Exception as exc:
            logger.debug("[DataFetcher] events timeout: %s", exc)

        # ── engine news ───────────────────────────────────────────────────────
        try:
            res.engine_news = futures["news"].result(timeout=timeout) or {}
        except Exception as exc:
            logger.debug("[DataFetcher] engine_news timeout %s: %s", target, exc)

        # ── Grok (X/Twitter) ─────────────────────────────────────────────────
        try:
            res.x_data = futures["grok"].result(timeout=timeout)
        except Exception as exc:
            logger.debug("[DataFetcher] grok timeout %s: %s", target, exc)

    # ── Price re-validation fallback ──────────────────────────────────────────
    if not res.real_price:
        res.real_price = (
            float(res.fund.get("price") or 0) or
            float(res.summary.get("price") or 0) or
            float(res.dc_data.get("price") or 0) or None
        )
        if res.real_price:
            logger.info("[DataFetcher] real_price recovered from fallback: %s", res.real_price)

    logger.info(
        "[DataFetcher] %s → price=%.2f fg=%s earnings=%s",
        target, res.real_price or 0,
        res.fg_data.get("score", "?"), res.next_earnings,
    )
    return res


_SUFFIX_TO_MARKET = {
    ".SR": "ksa", ".AE": "uae", ".DU": "uae",
    ".CA": "egypt", ".KW": "kuwait", ".QA": "qatar", ".BH": "bahrain",
}


def _try_build_from_cache(target: str) -> "FetchResult | None":
    """
    Attempt to build a FetchResult entirely from the pipeline cache.
    Returns a populated FetchResult if cache is fresh, else None.
    """
    try:
        import sys as _sys
        from core.config import BASE_DIR as _BASE_DIR
        _root = str(_BASE_DIR)
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from pipeline     import cache as _pc
        from query_engine import QueryEngine

        t_upper = target.upper()
        market  = next((m for sfx, m in _SUFFIX_TO_MARKET.items() if t_upper.endswith(sfx)), None)
        if market is None:
            return None

        # reject stale cache (> 30 min)
        age = _pc.cache_age_minutes(market)
        if age is None or age > 30:
            logger.debug("[DataFetcher] %s: cache age=%s — stale", target, age)
            return None

        stock = QueryEngine(_pc).get_stock(target, market)
        if stock is None:
            return None

        res = FetchResult()

        # ── Price ──────────────────────────────────────────────────────────────
        res.real_price = float(stock["close"])   if stock.get("close")  else None
        res.change_pct = float(stock["change"])  if stock.get("change") is not None else 0.0

        # ── Technical summary ──────────────────────────────────────────────────
        rsi    = stock.get("RSI")
        sma50  = stock.get("SMA50")
        sma200 = stock.get("SMA200")
        macd   = stock.get("MACD.macd")
        price  = res.real_price or 0

        res.summary = {
            "price":    price,
            "rsi":      float(rsi)    if rsi    is not None else 50.0,
            "sma_50":   float(sma50)  if sma50  is not None else None,
            "sma_200":  float(sma200) if sma200 is not None else None,
            "macd":     float(macd)   if macd   is not None else None,
            "trend":    ("Bullish" if (price and sma200 and price > float(sma200)) else
                         "Bearish" if (price and sma200) else "Neutral"),
            "momentum": ("Bullish" if (price and sma50  and price > float(sma50))  else
                         "Bearish" if (price and sma50)  else "Neutral"),
        }

        # ── Fundamentals ───────────────────────────────────────────────────────
        res.fund = {
            "name":          stock.get("name", target),
            "sector":        stock.get("sector"),
            "pe_ratio":      stock.get("price_earnings_ttm"),
            "dividend_yield":stock.get("dividend_yield_recent"),
            "market_cap":    stock.get("market_cap_basic"),
            "eps":           stock.get("earnings_per_share_diluted_ttm"),
            "price":         price,
        }

        logger.info(
            "[DataFetcher] %s: built from cache (age=%.1f min) — price=%.2f RSI=%.1f",
            target, age, price, res.summary["rsi"]
        )
        return res

    except Exception as exc:
        logger.debug("[DataFetcher] _try_build_from_cache failed for %s: %s", target, exc)
        return None


def _enrich_with_live_extras(res: FetchResult, target: str, timeout: int) -> FetchResult:
    """
    After a cache-first build, still fetch the live-only data in parallel:
      - Fear & Greed index
      - Earnings calendar
      - News (EisaX engine)
      - Grok / X sentiment
    These are NOT in the cache — they require live API calls.
    """
    try:
        with ThreadPoolExecutor(max_workers=4) as exe:
            f_fg     = exe.submit(_fetch_fear_greed)
            f_events = exe.submit(_fetch_events,      target)
            f_news   = exe.submit(_fetch_engine_news, target)
            f_grok   = exe.submit(_fetch_grok,        target)

            try:
                fg = f_fg.result(timeout=10) or {}
                if fg: res.fg_data = fg
            except Exception as exc:
                logger.warning("[DataFetcher] live extras fear&greed failed for %s: %s", target, exc)

            try:
                ev = f_events.result(timeout=10) or {}
                res.next_earnings = ev.get("next_earnings_date")
                res.ev_out        = ev
            except Exception as exc:
                logger.warning("[DataFetcher] live extras events failed for %s: %s", target, exc)

            try:
                res.engine_news = f_news.result(timeout=timeout) or {}
            except Exception as exc:
                logger.warning("[DataFetcher] live extras engine_news failed for %s: %s", target, exc)

            try:
                res.x_data = f_grok.result(timeout=timeout)
            except Exception as exc:
                logger.warning("[DataFetcher] live extras grok failed for %s: %s", target, exc)

    except Exception as exc:
        logger.debug("[DataFetcher] _enrich_with_live_extras failed for %s: %s", target, exc)

    return res


def _enrich_from_pipeline_cache(res: FetchResult, target: str) -> FetchResult:
    """
    Fills any None/empty fields in FetchResult from the pipeline cache.
    Cache is a supplement — never overrides a value already fetched from a primary source.
    Most useful for local-market tickers (.SR .AE .CA .KW .QA) where yfinance is unreliable.
    """
    # Only run for local-market tickers and crypto/commodities known to the cache
    _LOCAL_SUFFIXES = (".SR", ".AE", ".DU", ".CA", ".KW", ".QA", ".BH")
    _is_local = target.upper().endswith(_LOCAL_SUFFIXES)
    if not _is_local:
        return res   # US/global stocks: yfinance is reliable — skip

    try:
        from pipeline     import cache as _pipeline_cache
        from query_engine import QueryEngine

        # suffix → market code
        _SUFFIX_MAP = {
            ".SR": "ksa", ".AE": "uae", ".DU": "uae",
            ".CA": "egypt", ".KW": "kuwait", ".QA": "qatar", ".BH": "bahrain",
        }
        t_upper = target.upper()
        market  = next((m for sfx, m in _SUFFIX_MAP.items() if t_upper.endswith(sfx)), None)
        if market is None:
            return res

        qe    = QueryEngine(_pipeline_cache)   # fetcher=None → no auto-refresh inside fetch
        stock = qe.get_stock(target, market)
        if stock is None:
            return res

        logger.debug("[DataFetcher] cache hit for %s [%s]", target, market)

        # ── Price ──────────────────────────────────────────────────────────────
        _cache_price  = stock.get("close")
        _cache_change = stock.get("change")
        if not res.real_price and _cache_price:
            res.real_price = float(_cache_price)
            logger.info("[DataFetcher] %s: price from pipeline cache → %.2f", target, res.real_price)
        if not res.change_pct and _cache_change is not None:
            res.change_pct = float(_cache_change)

        # ── Technical summary ──────────────────────────────────────────────────
        if not isinstance(res.summary, dict):
            res.summary = {}

        _TECH_MAP = {
            "RSI":        "rsi",
            "SMA50":      "sma_50",
            "SMA200":     "sma_200",
            "MACD.macd":  "macd",
            "Stoch.K":    "stoch_k",
        }
        for cache_key, summary_key in _TECH_MAP.items():
            val = stock.get(cache_key)
            if val is not None and not res.summary.get(summary_key):
                try:
                    res.summary[summary_key] = float(val)
                except (TypeError, ValueError):
                    logger.warning(
                        "[DataFetcher] invalid cache technical for %s: %s=%r",
                        target,
                        cache_key,
                        val,
                    )

        # Derive trend / momentum from cache if summary was empty
        if not res.summary.get("trend"):
            price  = res.real_price or 0
            sma50  = res.summary.get("sma_50",  0) or 0
            sma200 = res.summary.get("sma_200", 0) or 0
            if price and sma50 and sma200:
                res.summary["trend"]    = "Bullish" if price > sma200 else "Bearish"
                res.summary["momentum"] = "Bullish" if price > sma50  else "Bearish"

        if not res.summary.get("price") and res.real_price:
            res.summary["price"] = res.real_price

        # ── Fundamentals ───────────────────────────────────────────────────────
        if not isinstance(res.fund, dict):
            res.fund = {}

        _FUND_MAP = {
            "sector":                          "sector",
            "price_earnings_ttm":              "pe_ratio",
            "dividend_yield_recent":           "dividend_yield",
            "market_cap_basic":                "market_cap",
            "earnings_per_share_diluted_ttm":  "eps",
        }
        for cache_key, fund_key in _FUND_MAP.items():
            val = stock.get(cache_key)
            if val is not None and not res.fund.get(fund_key):
                res.fund[fund_key] = val

        if not res.fund.get("name"):
            res.fund["name"] = stock.get("name", target)

    except Exception as _exc:
        logger.debug("[DataFetcher] cache enrich skipped for %s: %s", target, _exc)

    return res
