"""
core/services/analytics_builder.py
────────────────────────────────────
Complex logic extracted from FinanceAgent._handle_analytics.

Public API
──────────
    enrich_after_fetch(target, fr) -> dict
        Derives analyst targets, beta, energy/crypto flags, fair value, etc.

    collect_news_waterfall(target, fr, dc_data, fund) -> tuple[list, str, float]
        Full news collection with 8+ fallback levels.

    build_data_block(target, fr, ctx, original_target=None) -> str
        Builds the structured text block for the LLM.

    build_analytics_prompt(target, data_block, ctx, scorecard_verdict_hint,
                           is_arabic, brain_ctx, local_injection,
                           research_summary, original_target=None,
                           macro_block="") -> str
        Builds the full DeepSeek prompt.

    assemble_report(target, fr, ctx, deepseek_reply, news_block, pos,
                    pre_scorecard_md, original_target=None) -> str
        Assembles the final markdown report.
"""

from __future__ import annotations

import logging
import math
import os
import re as _re

from core.services.data_fetcher import FetchResult

logger = logging.getLogger(__name__)


# ── A. enrich_after_fetch ─────────────────────────────────────────────────────

def enrich_after_fetch(target: str, fr: FetchResult) -> dict:
    """
    Compute all derived values from FetchResult after the parallel fetch.
    Returns a ctx dict with keys documented in the module docstring.
    """
    import re as _re_loc

    dc_data = fr.dc_data or {}
    yf_info = fr.yf_info or {}
    fund    = fr.fund    or {}

    real_price = fr.real_price

    # ── Analyst target / consensus / count ────────────────────────────────────
    analyst_target    = fund.get("analyst_target") or None
    analyst_consensus = fund.get("analyst_consensus") or None
    analyst_count     = fund.get("analyst_count") or None
    forward_pe        = fr.forward_pe
    dividend_yield    = None

    # DeepCrawl primary source
    if dc_data.get("price_target"):
        pt_m = _re_loc.search(r"([\d.]+)", dc_data["price_target"])
        if pt_m:
            analyst_target = float(pt_m.group(1))
    if dc_data.get("analyst_rating"):
        analyst_consensus = dc_data["analyst_rating"]
    if dc_data.get("forward_pe"):
        try:
            forward_pe = float(dc_data["forward_pe"]) or None
        except Exception as exc:
            logger.warning(
                "[enrich] invalid DeepCrawl forward_pe for %s: %r (%s)",
                target,
                dc_data.get("forward_pe"),
                exc,
            )

    # DeepCrawl dividend: dollar amount → decimal yield
    _dc_div_dollar = float(dc_data.get("dividend", 0) or 0)
    _dc_price = float(dc_data.get("price", 0) or 0) or (real_price or 0)
    if _dc_div_dollar > 0 and _dc_price > 0:
        _dy = _dc_div_dollar / _dc_price
        if _dy <= 0.20:
            dividend_yield = _dy

    # yfinance fill
    if not analyst_target:
        analyst_target = yf_info.get("targetMeanPrice") or yf_info.get("targetMedianPrice")
        if analyst_target:
            analyst_target = float(analyst_target)
    if not analyst_consensus:
        analyst_consensus = yf_info.get("recommendationKey", "").replace("_", " ").title()
    if not analyst_count:
        analyst_count = yf_info.get("numberOfAnalystOpinions")
    if not forward_pe:
        _fpe_raw = yf_info.get("forwardPE")
        if _fpe_raw:
            try:
                _fpe = float(_fpe_raw)
                if _fpe > 0:
                    forward_pe = _fpe
            except Exception as exc:
                logger.warning(
                    "[enrich] invalid yfinance forwardPE for %s: %r (%s)",
                    target,
                    _fpe_raw,
                    exc,
                )
    if not dividend_yield:
        _trail_dy = float(yf_info.get("trailingAnnualDividendYield") or 0)
        if _trail_dy > 0.50:
            _trail_dy = _trail_dy / 100
        if _trail_dy > 0.50:
            _trail_dy = 0
        dividend_yield = _trail_dy if _trail_dy > 0 else None

    # Volume + 52W range → store into fr.fund
    _vol_today = yf_info.get("volume") or yf_info.get("regularMarketVolume") or 0
    _vol_avg   = yf_info.get("averageVolume") or 0
    _vol_10d   = yf_info.get("averageVolume10days") or 0
    _52w_high  = yf_info.get("fiftyTwoWeekHigh") or 0
    _52w_low   = yf_info.get("fiftyTwoWeekLow") or 0
    if _vol_today: fund["volume_today"]  = int(_vol_today)
    if _vol_avg:   fund["volume_avg90d"] = int(_vol_avg)
    if _vol_10d:   fund["volume_avg10d"] = int(_vol_10d)
    if _52w_high:  fund["week52_high"]   = float(_52w_high)
    if _52w_low:   fund["week52_low"]    = float(_52w_low)
    fr.fund = fund  # write back

    # ── DC data merge for local suffixes ──────────────────────────────────────
    _LOCAL_SUFFIXES = (".AE", ".DU", ".SR", ".CA", ".KW", ".QA")
    if dc_data and target.upper().endswith(_LOCAL_SUFFIXES):
        def _dc_f(key):
            v = dc_data.get(key)
            try:
                return float(str(v).strip()) if v not in (None, "", "N/A") else None
            except Exception:
                logger.debug("[enrich] _dc_f: cannot coerce %r for key %r", v, key, exc_info=True)
                return None

        def _dc_size(key):
            v = str(dc_data.get(key, "") or "")
            try:
                if "T" in v: return float(v.split("T")[0]) * 1e12
                if "B" in v: return float(v.split("B")[0]) * 1e9
                if "M" in v: return float(v.split("M")[0]) * 1e6
            except Exception as exc:
                logger.debug(
                    "[enrich] unable to parse DeepCrawl size for %s/%s: %r (%s)",
                    target,
                    key,
                    v,
                    exc,
                )
            return None

        def _dc_pct(key):
            v = str(dc_data.get(key, "") or "")
            try:
                return float(v.strip().rstrip("%"))
            except Exception:
                logger.debug("[enrich] _dc_pct: cannot parse %r for key %r", v, key, exc_info=True)
                return None

        _db = _dc_f("beta")
        if _db is not None and (not fund.get("beta") or abs(float(fund.get("beta", 1.0)) - 1.0) < 0.01):
            fund["beta"] = _db
        _dp = _dc_f("pe_ratio")
        if _dp and not fund.get("pe_ratio"):
            fund["pe_ratio"] = _dp
        _dfpe = _dc_f("forward_pe")
        if _dfpe and not forward_pe:
            forward_pe = _dfpe
        _de = _dc_f("eps")
        if _de and not fund.get("eps"):
            fund["eps"] = _de
        _dr = _dc_size("revenue")
        if _dr and not fund.get("revenue"):
            fund["revenue"] = _dr
        _dni = _dc_size("net_income")
        if _dni and not fund.get("net_income"):
            fund["net_income"] = _dni
        _mc_raw = dc_data.get("market_cap_raw")
        if _mc_raw and not fund.get("market_cap"):
            fund["market_cap"] = (_mc_raw * 1e9 if _mc_raw < 1e6 else _mc_raw)
        _drg = _dc_pct("rev_growth")
        if _drg is not None and not fund.get("revenue_growth"):
            fund["revenue_growth"] = _drg
        _deg = _dc_pct("earnings_growth")
        if _deg is not None and not fund.get("eps_growth"):
            fund["eps_growth"] = _deg
        if dc_data.get("dividend_yield") and not dividend_yield:
            try:
                _dy_str = str(dc_data["dividend_yield"]).strip().rstrip("%")
                _dy2 = float(_dy_str) / 100
                if _dy2 > 0:
                    dividend_yield = _dy2
            except Exception as exc:
                logger.warning(
                    "[enrich] invalid DeepCrawl dividend_yield for %s: %r (%s)",
                    target,
                    dc_data.get("dividend_yield"),
                    exc,
                )
        fr.fund = fund  # write back

    # ── Sequential analyst fallback ───────────────────────────────────────────
    if not analyst_target and real_price:
        try:
            import yfinance as _yf_seq
            _seq_info = _yf_seq.Ticker(target).info or {}
            _at_seq = _seq_info.get("targetMeanPrice") or _seq_info.get("targetMedianPrice")
            if _at_seq:
                analyst_target = float(_at_seq)
                if not analyst_consensus:
                    analyst_consensus = _seq_info.get("recommendationKey", "").replace("_", " ").title()
                if not analyst_count:
                    analyst_count = _seq_info.get("numberOfAnalystOpinions")
                logger.info("[enrich] analyst_target (sequential): %s", analyst_target)
        except Exception as _seq_e:
            logger.debug("[enrich] sequential analyst fetch failed: %s", _seq_e)

    # ── Sequential fundamentals re-fetch if sparse ────────────────────────────
    _missing_count = sum(1 for k in ["net_margin", "roe", "revenue_growth"] if not fund.get(k))
    if _missing_count >= 2:
        try:
            import yfinance as _yf_fund_seq
            import time as _t_seq
            _t_seq.sleep(1.5)
            _fi_seq = _yf_fund_seq.Ticker(target).info or {}
            if _fi_seq.get("profitMargins"):
                fund["net_margin"] = round(_fi_seq["profitMargins"] * 100, 1)
            if _fi_seq.get("returnOnEquity"):
                fund["roe"] = round(_fi_seq["returnOnEquity"] * 100, 2)
            if _fi_seq.get("revenueGrowth"):
                fund["revenue_growth"] = round(_fi_seq["revenueGrowth"] * 100, 1)
            if _fi_seq.get("earningsGrowth"):
                fund["eps_growth"] = round(_fi_seq["earningsGrowth"] * 100, 1)
            if _fi_seq.get("grossMargins"):
                fund["gross_margin"] = round(_fi_seq["grossMargins"] * 100, 1)
            if _fi_seq.get("operatingMargins"):
                fund["operating_margin"] = round(_fi_seq["operatingMargins"] * 100, 1)
            if not fund.get("pe_ratio") and _fi_seq.get("trailingPE"):
                fund["pe_ratio"] = round(_fi_seq["trailingPE"], 1)
            if not fund.get("current_ratio") and _fi_seq.get("currentRatio"):
                fund["current_ratio"] = round(_fi_seq["currentRatio"], 2)
            if not fund.get("beta") and _fi_seq.get("beta"):
                fund["beta"] = round(_fi_seq["beta"], 2)
            if not fund.get("eps") and _fi_seq.get("trailingEps"):
                fund["eps"] = round(_fi_seq["trailingEps"], 2)
            if not fund.get("market_cap") and _fi_seq.get("marketCap"):
                fund["market_cap"] = _fi_seq["marketCap"]
            fr.fund = fund
            logger.info("[enrich] sparse re-fetch recovered nm=%s roe=%s rg=%s",
                        fund.get("net_margin"), fund.get("roe"), fund.get("revenue_growth"))
        except Exception as _ff_e:
            logger.debug("[enrich] sequential re-fetch failed: %s", _ff_e)

    # ── Beta calculation ──────────────────────────────────────────────────────
    _is_crypto_asset = (
        target.endswith("-USD") and
        any(c in target for c in ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX"])
    )
    effective_beta = 1.0
    # NOTE: _compute_rolling_beta is a method on FinanceAgent; caller must handle crypto beta.
    # For the builder, we do the non-crypto path only.
    if not _is_crypto_asset:
        _dc_beta_v = float(dc_data.get("beta") or 0)
        _yf_beta_v = float(fund.get("beta") or 0)
        _is_local_stock = any(target.upper().endswith(sfx) for sfx in (".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
        if _is_local_stock and abs(_yf_beta_v - 1.0) < 0.005:
            _yf_beta_v = 0
        effective_beta = _dc_beta_v or _yf_beta_v or 0
        if not effective_beta:
            _s_eb = (fund.get("sector", "") or "").lower()
            effective_beta = (
                0.3 if any(x in _s_eb for x in ("energy", "oil", "gas", "utilities"))
                else 0.7 if any(x in _s_eb for x in ("real estate", "financials", "banks"))
                else 1.1
            )
    else:
        # Crypto: placeholder; caller should override with self._compute_rolling_beta(target)
        effective_beta = 1.5

    # ── Summary NaN/inf sanitization ─────────────────────────────────────────
    summary = fr.summary or {}
    _summary_defaults = {
        "rsi": 50.0, "sma_50": 0.0, "sma_200": 0.0,
        "adx": 0.0, "atr": 0.0, "macd": 0.0, "macd_signal": 0.0, "price": 0.0,
    }
    for _sk, _sd in _summary_defaults.items():
        _sv = summary.get(_sk, _sd)
        try:
            _svf = float(_sv or 0)
            summary[_sk] = _sd if (math.isnan(_svf) or math.isinf(_svf)) else _svf
        except Exception:
            logger.debug("[build_fr] summary coercion failed for key %r value %r — using default", _sk, _sv, exc_info=True)
            summary[_sk] = _sd
    fr.summary = summary

    # ── On-chain data (crypto) ─────────────────────────────────────────────
    # Caller is responsible for invoking self._fetch_onchain; we just default here.
    onchain_data: dict = {}

    # ── Energy detection + oil price ──────────────────────────────────────────
    _ENERGY_SECTORS = {"energy", "oil & gas", "oil", "petroleum", "integrated oil", "gas"}
    _ENERGY_PREFIXES = ("ADNOC", "2222", "2030", "2010", "TAQA", "DANA", "ARAMCO")
    _t_base = target.split(".")[0].upper()
    is_energy = (
        fund.get("sector", "").lower() in _ENERGY_SECTORS
        or fund.get("industry", "").lower() in {
            "oil & gas integrated", "oil & gas e&p",
            "oil & gas refining & marketing", "oil & gas equipment & services",
        }
        or any(_t_base.startswith(pfx) for pfx in _ENERGY_PREFIXES)
        or "GAS" in _t_base or "OIL" in _t_base or "PETRO" in _t_base or "ENERG" in _t_base
    )
    oil_data: dict = {}
    if is_energy:
        try:
            import yfinance as _yf_oil
            _brent   = _yf_oil.Ticker("BZ=F")
            _oil_fi  = _brent.fast_info
            _oil_price = float(getattr(_oil_fi, "last_price", None) or 0) or None
            _prev      = float(getattr(_oil_fi, "previous_close", None) or 0) or None
            _oil_change = 0.0
            if _oil_price and _prev:
                _oil_change = ((_oil_price - _prev) / _prev) * 100
            oil_data = {"price": _oil_price, "change_pct": round(_oil_change, 2), "name": "Brent Crude"}
            logger.info("[enrich] Brent=$%.2f (%+.1f%%)", _oil_price or 0, _oil_change)
        except Exception as _oil_e:
            logger.warning("[enrich] Brent fetch failed: %s", _oil_e)

    # ── Crash detection ───────────────────────────────────────────────────────
    change_pct = fr.change_pct or 0.0
    is_crash = abs(change_pct) >= 20
    crash_direction = (
        "CRASH 📉" if change_pct <= -20
        else "CIRCUIT BREAKER RALLY 📈" if change_pct >= 20
        else ""
    )

    # ── Fair value estimate ───────────────────────────────────────────────────
    fv_estimate = None
    fv_label = "Analyst consensus"
    valuation_pe = 15
    if not analyst_target and real_price:
        try:
            _eps_ttm = float(fund.get("eps") or dc_data.get("eps") or 0)
            _eg_raw = fund.get("eps_growth") or str(dc_data.get("earnings_growth", "0")).strip("%+")
            _eg = float(_eg_raw) if _eg_raw else 0
            _sector_pe_map = {
                "energy": 14, "financials": 12, "real estate": 15,
                "technology": 22, "utilities": 16, "healthcare": 18,
                "industrials": 15, "consumer cyclicals": 14,
                "consumer non-cyclicals": 17, "basic materials": 12,
            }
            _s = fund.get("sector", "").lower()
            _peer_pe = _sector_pe_map.get(_s, 15)
            _fpe_val = float(forward_pe or 0)
            valuation_pe = int(_fpe_val if _fpe_val > 0 else _peer_pe)
            if _eps_ttm > 0:
                _fwd_eps = _eps_ttm * (1 + _eg / 100)
                fv_estimate = round(_fwd_eps * valuation_pe, 3)
                fv_label = f"EisaX Fair Value (EPS×{valuation_pe}x)"
                logger.info("[enrich] FairValue=%s (FwdEPS=%.3f × PE=%s)", fv_estimate, _fwd_eps, valuation_pe)
        except Exception as _fve:
            logger.debug("[enrich] FairValue calc failed: %s", _fve)

    display_target = analyst_target or fv_estimate
    target_is_estimate = (analyst_target is None)

    # ── SMA tech target ───────────────────────────────────────────────────────
    _sma50_sc  = float(summary.get("sma_50", 0) or 0)
    _sma200_sc = float(summary.get("sma_200", 0) or 0)
    if _sma50_sc and math.isnan(_sma50_sc): _sma50_sc = 0.0
    if _sma200_sc and math.isnan(_sma200_sc): _sma200_sc = 0.0
    sma_tech_target = None
    if not display_target and real_price:
        if _sma200_sc and real_price < _sma200_sc:
            sma_tech_target = round(_sma200_sc, 3)
        elif _sma50_sc and real_price < _sma50_sc:
            sma_tech_target = round(_sma50_sc, 3)
    scorecard_target = display_target or sma_tech_target

    # ── ETF detection ──────────────────────────────────────────────────────────
    etf_meta = None
    try:
        from core.etf_intelligence import detect_etf as _detect_etf
        _profile = getattr(fr, "profile", {}) or {}
        _yf_info_for_etf = _profile.get("_yf_raw", {}) or fund.get("_yf_raw", {}) or {}
        etf_meta = _detect_etf(target, _yf_info_for_etf)
        if etf_meta:
            logger.info("[enrich] ETF detected: %s — %s", etf_meta.get("etf_type"), etf_meta.get("etf_label"))
    except Exception as _etf_e:
        logger.debug("[enrich] ETF detection skipped: %s", _etf_e)

    return {
        "analyst_target":    analyst_target,
        "analyst_consensus": analyst_consensus,
        "analyst_count":     analyst_count,
        "forward_pe":        forward_pe,
        "dividend_yield":    dividend_yield,
        "effective_beta":    effective_beta,
        "is_energy":         is_energy,
        "oil_data":          oil_data,
        "is_crypto":         _is_crypto_asset,
        "onchain_data":      onchain_data,
        "is_crash":          is_crash,
        "crash_direction":   crash_direction,
        "fv_estimate":       fv_estimate,
        "fv_label":          fv_label,
        "valuation_pe":      valuation_pe,
        "display_target":    display_target,
        "target_is_estimate": target_is_estimate,
        "scorecard_target":  scorecard_target,
        "etf_meta":          etf_meta,
    }


# ── B. collect_news_waterfall ─────────────────────────────────────────────────

def collect_news_waterfall(
    target: str,
    fr: FetchResult,
    dc_data: dict,
    fund: dict,
) -> tuple:
    """
    Full news collection pipeline with 8+ fallback levels.
    Returns (news_links, news_sent, news_score).
    """
    news_links: list = list(fr.news_links or [])
    _engine_news_data = fr.engine_news or {}

    news_sent  = fund.get("news_sentiment", "N/A")
    news_score = float(fund.get("news_score", 0.0) or 0.0)

    # ── Seed from engine news (inject at FRONT) ───────────────────────────────
    if _engine_news_data:
        try:
            from core.news_engine_client import format_news_links as _fmt_eng_links
            _eng_links = _fmt_eng_links(_engine_news_data)
            _seen_eng  = {n["url"] for n in news_links}
            _injected  = []
            for _el in _eng_links:
                if _el["url"] not in _seen_eng:
                    _injected.append(_el)
                    _seen_eng.add(_el["url"])
            news_links = _injected + news_links  # engine links at FRONT
            logger.info("[newsfall] %s: injected %d engine links", target, len(_injected))
        except Exception as _ene:
            logger.debug("[newsfall] engine news format failed: %s", _ene)

    # ── FMP fallback ─────────────────────────────────────────────────────────
    if not news_links:
        try:
            from core.realtime_data import get_live_news
            fmp_news = get_live_news(target, limit=4)
            for n in fmp_news:
                if n.get("headline") and n.get("url"):
                    news_links.append({"title": n["headline"][:120], "url": n["url"]})
        except Exception as _fmpe:
            logger.error("[newsfall] FMP news failed: %s", _fmpe)

    # ── Regional energy supplement ────────────────────────────────────────────
    _t_upper_news = target.upper()
    _ENERGY_PREFIXES2 = ("ADNOC", "ARAMCO", "2222", "TAQA", "DANA", "GAS", "OIL", "ENERG")
    _is_regional_energy = (
        _t_upper_news.endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
        and any(k in _t_upper_news for k in _ENERGY_PREFIXES2)
    )
    if _is_regional_energy and len(news_links) < 3:
        try:
            from core.realtime_data import get_live_news as _gln
            _region_q = (
                "Gulf oil energy OPEC Iran war 2026"
                if _t_upper_news.endswith((".AE", ".DU"))
                else "Saudi Aramco oil energy OPEC 2026"
                if _t_upper_news.endswith(".SR")
                else "oil energy OPEC Middle East 2026"
            )
            _geo_news = _gln(target, company_name=_region_q, limit=5)
            for n in _geo_news:
                h = n.get("headline", "")
                u = n.get("url", "")
                if h and u and not any(x["title"] == h for x in news_links):
                    news_links.append({"title": h[:120], "url": u})
            logger.info("[newsfall] %s: supplemented with %d regional items", target, len(_geo_news))
        except Exception as _rne:
            logger.warning("[newsfall] regional supplement failed: %s", _rne)

    # ── Local non-energy: NewsAPI ─────────────────────────────────────────────
    _is_local_ticker = _t_upper_news.endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
    if _is_local_ticker and len(news_links) < 2:
        try:
            from core.realtime_data import get_live_news as _gln2
            _co_name = fund.get("company_name") or target.split(".")[0]
            _mkt_ctx = (
                "UAE" if _t_upper_news.endswith((".AE", ".DU"))
                else "Saudi Arabia" if _t_upper_news.endswith(".SR")
                else "Egypt" if _t_upper_news.endswith(".CA")
                else "Kuwait" if _t_upper_news.endswith(".KW")
                else "Qatar"
            )
            _ticker_base = target.split(".")[0]
            _local_news = _gln2(target, company_name=f"{_co_name}", limit=5)
            if len(_local_news) < 2:
                _local_news = _gln2(target, company_name=f"{_ticker_base} {_mkt_ctx}", limit=5)
            for n in _local_news:
                h = n.get("headline", "")
                u = n.get("url", "")
                if h and u and not any(x["title"] == h for x in news_links):
                    news_links.append({"title": h[:120], "url": u})
            if len(news_links) < 2:
                _sector = fund.get("sector", "") or "investment"
                _mkt_news = _gln2(target, company_name=f"{_sector} {_mkt_ctx} market 2026", limit=4)
                for n in _mkt_news:
                    h = n.get("headline", "")
                    u = n.get("url", "")
                    if h and u and not any(x["title"] == h for x in news_links):
                        news_links.append({"title": h[:120], "url": u})
            logger.info("[newsfall] %s: %d local news items", target, len(news_links))
        except Exception as _lne:
            logger.warning("[newsfall] local news failed: %s", _lne)

    # ── Serper last-resort ────────────────────────────────────────────────────
    if len(news_links) < 2:
        try:
            _serper_key = os.getenv("SERPER_API_KEY", "")
            if _serper_key:
                import requests as _req_serper
                _ticker_base_serper = target.split(".")[0]
                _co_name_serper = (
                    fund.get("company_name") or dc_data.get("company_name") or _ticker_base_serper
                )
                _is_gulf_ticker = _t_upper_news.endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
                _commodity_name_map = {
                    "GC": "gold price", "SI": "silver price", "CL": "crude oil WTI",
                    "NG": "natural gas price", "PL": "platinum price", "PA": "palladium price",
                    "HG": "copper price", "BZ": "brent oil price",
                    "GC=F": "gold price", "SI=F": "silver price", "CL=F": "crude oil WTI",
                    "PL=F": "platinum price", "PA=F": "palladium price",
                    "HG=F": "copper price", "BZ=F": "brent oil price",
                    "GLD": "gold ETF price", "IAU": "gold ETF price", "SGOL": "gold ETF",
                    "GLDM": "gold ETF", "SLV": "silver ETF price", "SIVR": "silver ETF",
                    "USO": "crude oil ETF", "BNO": "brent oil ETF",
                    "PPLT": "platinum ETF", "PALL": "palladium ETF", "CPER": "copper ETF",
                }
                _serper_commodity = _commodity_name_map.get(
                    _ticker_base_serper.upper(),
                    _commodity_name_map.get(target.upper(), ""),
                )
                if _is_gulf_ticker:
                    _sq = (
                        f'"{_co_name_serper}" OR "{_ticker_base_serper}" أخبار stock news '
                        f'site:zawya.com OR site:gulfnews.com OR site:arabianbusiness.com'
                    )
                elif _serper_commodity:
                    _sq = f"{_serper_commodity} market news 2026"
                else:
                    _sq = f'"{_co_name_serper}" stock news {(fund.get("sector", "") or "")}'
                _sr = _req_serper.post(
                    "https://google.serper.dev/news",
                    headers={"X-API-KEY": _serper_key, "Content-Type": "application/json"},
                    json={"q": _sq, "num": 6},
                    timeout=8,
                )
                if _sr.status_code == 200:
                    for _sn in _sr.json().get("news", []):
                        _sh = _sn.get("title", "")
                        _su = _sn.get("link", "")
                        if _sh and _su and not any(x["title"] == _sh for x in news_links):
                            news_links.append({"title": _sh[:120], "url": _su})
                    logger.info("[newsfall] %s: Serper got %d items", target, len(news_links))
        except Exception as _sne:
            logger.warning("[newsfall] Serper failed: %s", _sne)

    # ── EisaX Aggregator final fallback ───────────────────────────────────────
    if len(news_links) < 2:
        try:
            from core.news_aggregator import get_news as _agg_news
            _agg = _agg_news(ticker=target, limit=5)
            for _an in _agg:
                _at = _an.get("title", "")
                _au = _an.get("url", "")
                if _at and _au and not any(x["title"] == _at for x in news_links):
                    news_links.append({"title": _at[:120], "url": _au})
            logger.info("[newsfall] %s: aggregator got %d items", target, len(news_links))
        except Exception as _age:
            logger.warning("[newsfall] aggregator failed: %s", _age)

    # ── Relevance filter ──────────────────────────────────────────────────────
    def _is_relevant_news(title: str, ticker_str: str, company: str) -> bool:
        if not title:
            return False
        t_low  = title.lower()
        tk_low = ticker_str.lower().split(".")[0]
        co_low = (company or "").lower()

        _noise_sources = [
            "wallstreetbets", "reddit", "r/stocks", "memestocks",
            "mcdonald's", "mcdonalds", "coca-cola", "coca cola",
            "unrelated_company",
        ]
        if any(n in t_low for n in _noise_sources):
            return False

        _tk_clean = tk_low.split("=")[0]
        if tk_low and len(tk_low) > 2 and tk_low in t_low:
            return True
        _commodity_kw_map = {
            "gc": ["gold", "xau", "bullion", "precious metal"],
            "si": ["silver", "xag", "precious metal"],
            "cl": ["crude", "oil", "wti", "petroleum"],
            "ng": ["natural gas", "lng"],
            "pl": ["platinum", "pgm", "precious metal"],
            "pa": ["palladium", "pgm", "precious metal"],
            "hg": ["copper", "base metal", "industrial metal"],
            "pplt": ["platinum", "precious metal"],
            "pall": ["palladium", "precious metal"],
            "cper": ["copper", "base metal"],
            "gld": ["gold", "xau", "bullion", "precious metal"],
            "iau": ["gold", "xau", "bullion", "precious metal"],
            "sgol": ["gold", "xau", "bullion"],
            "gldm": ["gold", "xau", "bullion"],
            "slv": ["silver", "xag", "precious metal"],
            "sivr": ["silver", "xag"],
            "uso": ["crude", "oil", "wti", "petroleum"],
            "bno": ["brent", "oil", "crude"],
        }
        if _tk_clean in _commodity_kw_map:
            if any(k in t_low for k in _commodity_kw_map[_tk_clean]):
                return True
        if co_low and len(co_low) > 3:
            first_word = co_low.split()[0]
            if len(first_word) > 3 and first_word in t_low:
                return True

        _t_sector = (fund.get("sector") or "").lower()
        _sector_keys = {
            "energy":      ["oil", "opec", "brent", "crude", "gas", "lng", "iran", "hormuz"],
            "technology":  ["ai", "semiconductor", "tech", "chip", "cloud", "software"],
            "real estate": ["real estate", "property", "reit", "mortgage", "housing"],
            "financials":  ["bank", "lending", "fed", "rate", "credit", "loan"],
            "crypto":      ["bitcoin", "btc", "crypto", "ethereum", "blockchain"],
            "commodit":    ["gold", "xau", "bullion", "silver", "precious metal", "oil", "brent", "crude", "commodity"],
            "precious":    ["gold", "xau", "bullion", "silver", "platinum", "palladium", "precious metal"],
        }
        for sec, keys in _sector_keys.items():
            if sec in _t_sector:
                if any(k in t_low for k in keys):
                    return True

        _broad_ok = ["earnings", "revenue", "ipo", "dividend", "buyback",
                     "forecast", "outlook", "guidance", "acquisition", "merger"]
        if any(k in t_low for k in _broad_ok):
            if tk_low and tk_low in t_low:
                return True
            if co_low and len(co_low.split()[0]) > 3 and co_low.split()[0] in t_low:
                return True

        return False

    _co_name_for_filter = fund.get("company_name", target)
    _orig_count = len(news_links)
    news_links = [
        n for n in news_links
        if _is_relevant_news(n.get("title", ""), target, _co_name_for_filter)
    ]
    if len(news_links) < _orig_count:
        logger.info("[newsfall] %s: filtered %d irrelevant, kept %d",
                    target, _orig_count - len(news_links), len(news_links))

    # ── Post-filter Serper rescue ─────────────────────────────────────────────
    if len(news_links) == 0:
        try:
            _serper_key2 = os.getenv("SERPER_API_KEY", "")
            if _serper_key2:
                import requests as _req_s2
                _tb2  = target.split(".")[0]
                _cn2  = fund.get("company_name") or dc_data.get("company_name") or _tb2
                _gulf2 = _t_upper_news.endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
                _sq2 = (
                    f'"{_cn2}" OR "{_tb2}" stock news zawya arabianbusiness 2026'
                    if _gulf2
                    else f'"{_cn2}" stock news 2026'
                )
                _sr2 = _req_s2.post(
                    "https://google.serper.dev/news",
                    headers={"X-API-KEY": _serper_key2, "Content-Type": "application/json"},
                    json={"q": _sq2, "num": 6},
                    timeout=8,
                )
                if _sr2.status_code == 200:
                    for _sn2 in _sr2.json().get("news", []):
                        _sh2 = _sn2.get("title", "")
                        _su2 = _sn2.get("link", "")
                        if _sh2 and _su2:
                            news_links.append({"title": _sh2[:120], "url": _su2})
                    logger.info("[newsfall] %s: post-filter Serper rescue: %d items", target, len(news_links))
        except Exception as _sne2:
            logger.debug("[newsfall] post-filter Serper rescue: %s", _sne2)

    # ── EisaX Aggregator post-filter rescue ───────────────────────────────────
    if len(news_links) == 0:
        try:
            from core.news_aggregator import get_news as _agg_news2
            _agg2 = _agg_news2(ticker=target, limit=5)
            for _an2 in _agg2:
                _at2 = _an2.get("title", "")
                _au2 = _an2.get("url", "")
                if _at2 and _au2:
                    news_links.append({"title": _at2[:120], "url": _au2})
            logger.info("[newsfall] %s: aggregator rescue: %d items", target, len(news_links))
        except Exception as _age2:
            logger.warning("[newsfall] aggregator rescue failed: %s", _age2)

    return (news_links, news_sent, news_score)


# ── C. build_data_block ───────────────────────────────────────────────────────

def build_data_block(
    target: str,
    fr: FetchResult,
    ctx: dict,
    original_target: str | None = None,
) -> str:
    """
    Build the structured text block passed to the LLM.
    Extracted from _handle_analytics lines 3199–3411.
    """
    fund          = fr.fund or {}
    dc_data       = fr.dc_data or {}
    summary       = fr.summary or {}
    var_95        = fr.var_95 or 0.02
    max_dd        = fr.max_dd or 0.20
    ev_out        = fr.ev_out or {}
    fg_data       = fr.fg_data or {}
    real_price    = fr.real_price
    change_pct    = fr.change_pct or 0.0
    next_earnings = fr.next_earnings

    analyst_target    = ctx.get("analyst_target")
    analyst_consensus = ctx.get("analyst_consensus")
    analyst_count     = ctx.get("analyst_count")
    forward_pe        = ctx.get("forward_pe")
    dividend_yield    = ctx.get("dividend_yield")
    effective_beta    = ctx.get("effective_beta", 1.0)
    is_energy         = ctx.get("is_energy", False)
    oil_data          = ctx.get("oil_data", {})
    onchain_data      = ctx.get("onchain_data", {})
    fv_label          = ctx.get("fv_label", "Analyst consensus")
    valuation_pe      = ctx.get("valuation_pe", 15)
    display_target    = ctx.get("display_target")
    target_is_estimate = ctx.get("target_is_estimate", True)
    etf_meta          = ctx.get("etf_meta")

    currency_sym = ctx.get("currency_sym", "$")
    currency_lbl = ctx.get("currency_lbl", "USD")
    _is_local_currency = currency_lbl in ("SAR", "AED", "EGP", "KWF", "QAR")
    _is_local_mkt = currency_lbl != "USD"
    _t_upper = target.upper()

    news_sent  = ctx.get("news_sent", "N/A")
    news_score = ctx.get("news_score", 0.0)
    t10y       = getattr(fr, "t10y", "N/A")
    fed        = getattr(fr, "fed", "N/A")
    unemp      = getattr(fr, "unemp", "N/A")
    inflation  = getattr(fr, "inflation", "N/A")
    gdp        = getattr(fr, "gdp", "N/A")

    _fallback_price = real_price or summary.get("price", 0)
    price_str = (
        f"{_fallback_price:,.2f} {currency_sym} ({change_pct:+.2f}%)"
        if _fallback_price and _is_local_mkt and change_pct
        else f"{_fallback_price:,.2f} {currency_sym}"
        if _fallback_price and _is_local_mkt
        else f"${_fallback_price:,.2f} ({change_pct:+.2f}%)"
        if _fallback_price and change_pct
        else f"${_fallback_price:,.2f}"
        if _fallback_price else "N/A"
    )

    # ── Local format helpers ──────────────────────────────────────────────────
    def _B(n):
        try:
            if not n: return "N/A"
            v = float(n)
            if currency_lbl != "USD":
                if v >= 1e12: return f"{v/1e12:.2f}T {currency_sym}"
                if v >= 1e9:  return f"{v/1e9:.1f}B {currency_sym}"
                if v >= 1e6:  return f"{v/1e6:.0f}M {currency_sym}"
                return f"{v:,.0f} {currency_sym}"
            return f"${v/1e9:.1f}B" if v >= 1e9 else f"${v/1e6:.0f}M"
        except Exception:
            logger.debug("[_B] format error for value %r", n, exc_info=True)
            return "N/A"

    def _P(n):
        return f"{n:.1f}%" if n else "N/A"

    def _X(n):
        return f"{n:.1f}x" if n else "N/A"

    # Engine news data for inline block
    _engine_news_data = fr.engine_news or {}

    data_block = f"""
TICKER: {original_target if original_target else target} (resolved: {target})
COMPANY: {fund.get('company_name') or (original_target if original_target else target)}
SECTOR: {fund.get('sector', 'N/A')} | INDUSTRY: {fund.get('industry', 'N/A')}
CURRENCY: {currency_lbl} (use {currency_sym} symbol in ALL price references){chr(10) + "IMPORTANT: This is an Egyptian stock (EGX). Market Cap, prices and all monetary values are in EGP (Egyptian Pound ج.م). Do NOT convert to USD or display in USD." if _t_upper.endswith(".CA") else ""}
LIVE PRICE: {price_str}
MARKET CAP: {_B(fund.get('market_cap'))}
QUALITY SCORE: {fund.get('fundamental_score', 'N/A')}/100

NEWS SENTIMENT: {news_sent} (score: {news_score})

MACRO: 10Y Treasury: {t10y}% | Fed Funds: {fed}% | Unemployment: {unemp}% | CPI YoY: {inflation}% | GDP Growth: {gdp}%

GROWTH:
- Revenue Growth YoY: {_P(fund.get('revenue_growth'))}
- EPS Growth YoY: {_P(fund.get('eps_growth'))}
- Revenue (TTM): {_B(fund.get('revenue'))}
- EPS (TTM): ${fund.get('eps', 'N/A')}

PROFITABILITY:
- Gross Margin: {_P(fund.get('gross_margin'))}
- Operating Margin: {_P(fund.get('operating_margin'))}
- Net Margin: {_P(fund.get('net_margin'))}
- ROE: {_P(fund.get('roe'))}
- ROIC: {_P(fund.get('roic'))}

VALUATION:
- P/E (TTM): {_X(fund.get('pe_ratio'))}
- Forward P/E: {_X(float(dc_data.get("forward_pe") or 0) or forward_pe)}
- P/S (TTM): {_X(fund.get('ps_ratio'))}
- EV/EBITDA: {_X(fund.get('ev_ebitda'))}
- Beta: {effective_beta}
- Gross Margin: {_P(fund.get('gross_margin'))}{" (Non-GAAP; GAAP may vary ~2-3%)" if fund.get('gross_margin') else ""}
- Dividend Yield: {f"{dividend_yield*100:.2f}%" if dividend_yield and dividend_yield > 0.001 else "Minimal (<0.1%)"}

ANALYST CONSENSUS:
- Recommendation: {analyst_consensus or 'N/A'} ({analyst_count or 'N/A'} analysts)
- Price Target (Mean): {((currency_sym if _is_local_currency else "$") + str(round(display_target, 2))) if display_target else 'N/A'}{" [" + fv_label + "]" if target_is_estimate else ""}
- Upside Potential: {f"{((display_target/real_price)-1)*100:.1f}%" if display_target and real_price else 'N/A'}
{"- NOTE: No analyst coverage found. Target shown is EisaX Fair Value Estimate (Forward EPS × " + str(valuation_pe) + "x sector P/E). Present as 'EisaX Fair Value Estimate' in section 5, NOT as analyst consensus. Do NOT use SMA200 as a price target." if target_is_estimate else ""}

BALANCE SHEET:
- Cash: {_B(fund.get('cash'))}
- Total Debt: {_B(fund.get('total_debt'))}
- Debt/Equity: {fund.get('debt_equity', 'N/A')}
- Current Ratio: {fund.get('current_ratio') or 'N/A'}

EARNINGS:
- Last Earnings Date: {fund.get('last_earnings_date', 'N/A')}
- NEXT EARNINGS DATE: {next_earnings or 'N/A'}
- EPS Actual vs Est (last): ${fund.get('last_eps_actual', 'N/A')} vs ${fund.get('last_eps_estimate', 'N/A')}
- Earnings Surprise: {fund.get('earnings_surprise_pct', 'N/A')}%
- Next Quarter EPS Estimate: ${ev_out.get('eps_est_avg', 'N/A')} (range: ${ev_out.get('eps_est_low','?')} – ${ev_out.get('eps_est_high','?')})
- Next Quarter Revenue Estimate: {f"${ev_out['rev_est_avg']/1e9:.1f}B" if ev_out.get('rev_est_avg') else 'N/A'} (range: {f"${ev_out['rev_est_low']/1e9:.1f}B" if ev_out.get('rev_est_low') else '?'} – {f"${ev_out['rev_est_high']/1e9:.1f}B" if ev_out.get('rev_est_high') else '?'})

MARKET SENTIMENT (Fear & Greed Index):
- Score: {fg_data.get('score', 'N/A')} / 100
- Rating: {fg_data.get('rating', 'N/A')} ({fg_data.get('label_ar', '')})
- Implication: {"Extreme fear — historically a contrarian buy signal; staged entries become more favorable" if (fg_data.get('score') or 50) < 25 else "Fear zone — market is risk-off; tighter stop losses advised" if (fg_data.get('score') or 50) < 45 else "Neutral sentiment" if (fg_data.get('score') or 50) < 55 else "Greed — market momentum favors bulls, but watch for complacency" if (fg_data.get('score') or 50) < 75 else "Extreme greed — elevated risk of correction; use caution on new entries"}

TECHNICALS:
- Trend: {summary['trend']} (Price vs SMA200)
- Momentum: {summary['momentum']} (MACD)
- RSI: {summary['rsi']:.1f} → {summary['condition']}
- MACD: {summary.get('macd', 0):.2f} | Signal: {summary.get('macd_signal', 0):.2f} | {"Bullish crossover" if summary.get('macd', 0) > summary.get('macd_signal', 0) else "Bearish crossover"}
- SMA50: {currency_sym}{summary['sma_50']:,.2f} | SMA200: {currency_sym}{summary['sma_200']:,.2f}
- Price vs SMA50: {f"{((real_price - summary['sma_50']) / summary['sma_50'] * 100):+.1f}%" if real_price and summary.get('sma_50') and float(summary.get('sma_50',0)) != 0 else "N/A"} | vs SMA200: {f"{((real_price - summary['sma_200']) / summary['sma_200'] * 100):+.1f}%" if real_price and summary.get('sma_200') and float(summary.get('sma_200',0)) != 0 else "N/A"}
- ADX: {summary.get('adx', 0):.1f} ({"Strong trend" if summary.get('adx', 0) > 25 else "Weak/No trend"}) | ATR: {summary.get('atr', 0):.2f}
{"- ⚠️ Technical Note: Momentum indicators (MACD/RSI) reflect price-driven buying pressure, while ADX measures trend strength independently of direction. A bullish momentum reading alongside a weak ADX (< 25) indicates early-stage or range-bound price action — not a confirmed trend. Treat momentum signals with reduced confidence until ADX sustains above 25." if (summary.get('adx', 0) < 25 and (summary.get('macd', 0) > 0 or summary.get('rsi', 0) > 55)) else ""}
{(lambda v_t, v_a: f"""
VOLUME:
- Today: {v_t/1e6:.1f}M vs 90-day avg {v_a/1e6:.1f}M → {"🔴 LOW volume ({:.0f}% of avg) — weak conviction in move".format(v_t/v_a*100) if v_a and v_t/v_a < 0.75 else "🟢 HIGH volume ({:.0f}% of avg) — strong conviction".format(v_t/v_a*100) if v_a and v_t/v_a > 1.25 else "⚪ Normal volume ({:.0f}% of avg)".format(v_t/v_a*100) if v_a else "N/A"}
""" if v_a else "")(
    fund.get('volume_today', 0) or 0,
    fund.get('volume_avg90d', 0) or 0
)}
{(lambda h52, l52, p: f"""
TECHNICAL LEVEL LADDER (S/R — ordered by proximity):
RULE: ALL 7 rows MUST appear. Use R3→R2→R1→SPOT→S1→S2→S3. Sources: SMA/EMA > Fibonacci > Swing > 52W. If data insufficient for a level, show "N/A — insufficient data".

| Level | Price | Type | Basis |
|-------|-------|------|-------|
| R3    | {f"{currency_sym}{h52:,.2f}" if h52 else "N/A — insufficient data"} | Resistance | {f"52W High" if h52 else "—"} |
| R2    | {f"{currency_sym}{l52 + (h52-l52)*0.618:,.2f}" if (h52 and l52) else "N/A — insufficient data"} | Resistance | {f"Fib 61.8%" if (h52 and l52) else "—"} |
| R1    | {f"{currency_sym}{l52 + (h52-l52)*0.50:,.2f}" if (h52 and l52) else "N/A — insufficient data"} | Resistance | {f"Fib 50% / Mid-Range" if (h52 and l52) else "—"} |
| SPOT  | {f"{currency_sym}{p:,.2f}" if p else "N/A"} | Current Price | Live |
| S1    | {f"{currency_sym}{l52 + (h52-l52)*0.382:,.2f}" if (h52 and l52) else "N/A — insufficient data"} | Support | {f"Fib 38.2%" if (h52 and l52) else "—"} |
| S2    | {f"{currency_sym}{l52 + (h52-l52)*0.236:,.2f}" if (h52 and l52) else "N/A — insufficient data"} | Support | {f"Fib 23.6%" if (h52 and l52) else "—"} |
| S3    | {f"{currency_sym}{l52:,.2f}" if l52 else "N/A — insufficient data"} | Support | {f"52W Low" if l52 else "—"} |
INSTRUCTION: In Section 3, you MUST include this FULL 7-row S/R table. Never skip or omit any row. Reference S1/R1 for entry/stop placement. Mention volume to confirm level breaks.
""")(
    fund.get('week52_high', 0) or 0,
    fund.get('week52_low', 0) or 0,
    real_price or 0
)}
RISK:
- VaR (95%, daily): {var_95*100:.2f}%
- Max Historical Drawdown: {max_dd*100:.2f}%
{"" if not onchain_data else f"""
ON-CHAIN METRICS (LIVE):
- All-Time High: ${(onchain_data.get('ath') or 0):,.0f} (ATH change: {(onchain_data.get('ath_change_pct') or 0):.1f}%, date: {onchain_data.get('ath_date', 'N/A')})
- Supply: {(onchain_data.get('circulating_supply') or 0):,.0f} / {(onchain_data.get('max_supply') or 0):,.0f} ({onchain_data.get('supply_ratio', 0)}% mined)
- 24h Volume: ${(onchain_data.get('total_volume_24h') or 0)/1e9:.1f}B
- Market Cap Rank: #{onchain_data.get('mc_rank', 'N/A')}
{f'- Hash Rate: {onchain_data["hash_rate_eh"]:.0f} EH/s' if onchain_data.get('hash_rate_eh') else ''}
{f'- Active Addresses (24h): {onchain_data["active_addresses"]:,}' if onchain_data.get('active_addresses') else ''}
{f'- Transactions (24h): {onchain_data["n_tx_24h"]:,}' if onchain_data.get('n_tx_24h') else ''}
IMPORTANT: Use these on-chain metrics in your analysis. Discuss supply scarcity, network activity, and hash rate health.
"""}
{"" if not oil_data.get('price') else f"""
OIL PRICE DATA (LIVE):
- Brent Crude: ${oil_data['price']:.2f}/bbl ({oil_data['change_pct']:+.1f}%)
IMPORTANT: This is an ENERGY SECTOR stock. Oil prices are the #1 driver of revenue and valuation.
Include an Oil Price Sensitivity Analysis table in your report showing impact at $50, $60, $70, $80, $90/bbl.
Discuss OPEC+ dynamics and energy transition risks.

OIL PRICE SENSITIVITY (pre-computed):
| Oil Price (Brent) | Change from Current | Est. Revenue Impact | Est. Stock Price |
|-------------------|--------------------|--------------------|-----------------|
| ${oil_data['price']:.0f}/bbl (current) | — | Base | {currency_sym}{real_price or 0:,.2f} |
| $90/bbl | {((90 - oil_data['price']) / oil_data['price'] * 100):+.0f}% | {((90 - oil_data['price']) / oil_data['price'] * 70):+.0f}% | {currency_sym}{(real_price or 0) * (1 + (90 - oil_data['price']) / oil_data['price'] * 0.55):,.2f} |
| $80/bbl | {((80 - oil_data['price']) / oil_data['price'] * 100):+.0f}% | {((80 - oil_data['price']) / oil_data['price'] * 70):+.0f}% | {currency_sym}{(real_price or 0) * (1 + (80 - oil_data['price']) / oil_data['price'] * 0.55):,.2f} |
| $70/bbl | {((70 - oil_data['price']) / oil_data['price'] * 100):+.0f}% | {((70 - oil_data['price']) / oil_data['price'] * 70):+.0f}% | {currency_sym}{(real_price or 0) * (1 + (70 - oil_data['price']) / oil_data['price'] * 0.55):,.2f} |
| $60/bbl | {((60 - oil_data['price']) / oil_data['price'] * 100):+.0f}% | {((60 - oil_data['price']) / oil_data['price'] * 70):+.0f}% | {currency_sym}{(real_price or 0) * (1 + (60 - oil_data['price']) / oil_data['price'] * 0.55):,.2f} |
| $50/bbl | {((50 - oil_data['price']) / oil_data['price'] * 100):+.0f}% | {((50 - oil_data['price']) / oil_data['price'] * 70):+.0f}% | {currency_sym}{(real_price or 0) * (1 + (50 - oil_data['price']) / oil_data['price'] * 0.55):,.2f} |
"""}
{(f"""SCENARIO ANALYSIS (Energy-Sector — Oil-Price-Adjusted):
Note: Impact already pre-calculated using 0.55x oil sensitivity. Copy EXACTLY — do NOT add extra columns.
| Scenario | Impact | Implied Price | Suggested Hedge |
|----------|--------|---------------|-----------------|
| 🚀 Oil Spike $150+/bbl | +{((((150 - oil_data.get('price',80)) / oil_data.get('price',80)) * 55)):.1f}% | {currency_sym}{(real_price or 0) * (1 + (((150 - oil_data.get('price',80)) / oil_data.get('price',80)) * 0.55)):,.2f} | Hold / partial profit |
| 🛢️ Oil Crash to $50/bbl | {(-((oil_data.get('price',80)-50)/oil_data.get('price',80))*55):.1f}% | {currency_sym}{((real_price or 0) * (1 + (-((oil_data.get('price',80)-50)/oil_data.get('price',80))*55))/100):,.2f} | Gold + Tech |
| 📉 OPEC+ Production Surge | {(-18 * 0.55):.1f}% | {currency_sym}{(real_price or 0) * (1 + (-18 * 0.55)/100):,.2f} | Diversified equities |
| 🌱 Energy Transition (long-term) | {(-30 * 0.55 * 0.75):.1f}% | {currency_sym}{(real_price or 0) * (1 + (-30 * 0.55 * 0.75)/100):,.2f} | Clean energy + Tech |
| 🏦 Fed Rate Shock +2% | {((-8 * max(float(effective_beta), 0.4)) + (-5 * 0.55)):.1f}% | {currency_sym}{(real_price or 0) * (1 + ((-8 * max(float(effective_beta), 0.4)) + (-5 * 0.55))/100):,.2f} | Treasuries + Cash |
INSTRUCTION FOR SECTION 9 (Scenario Analysis):
- You MUST include 3 core scenarios: Bear, Base, Bull — each with probability + expected price + expected return
- Core scenario probabilities MUST sum to 100%
- Any Macro Shock / Black Swan / Tail Risk scenario must be labeled as "💥 Tail Risk Overlay" and shown SEPARATELY
- Tail Risk Overlay must NOT be included in Expected Value calculation
- Expected Value = Σ(core_probability × core_return) across Bear/Base/Bull ONLY
- Show the EV calculation explicitly: "Expected Value: X.X%"
- After the core table, show tail risk separately:
  💥 **Tail Risk Overlay** | ~-25% | [trigger] | [hedge]
  ⚠️ *Not included in Expected Value calculation*
""" if is_energy else (f"""SCENARIO ANALYSIS (UAE Real Estate — Geopolitical + Rate Sensitive):
Note: Dubai real estate reacts to regional geopolitics AND global rates, not just market beta ({effective_beta}).
Use -20% to -30% for geopolitical scenarios regardless of low beta — tourist/investor sentiment collapses in conflict.
| Scenario | Impact Driver | Est. Price Impact | Implied Price ({currency_sym}) | Suggested Hedge |
|----------|--------------|------------------|--------------------------|-----------------|
| 🚀 Dubai Tourism Boom | +35% tourism surge | +{(35 * 0.40):.1f}% | {currency_sym}{(real_price or 0) * (1 + (35 * 0.40)/100):,.2f} | Hold / add on dips |
| 🌍 Iran/Hormuz Conflict | Gulf security crisis | -{(28):.1f}% | {currency_sym}{(real_price or 0) * (1 - 28/100):,.2f} | Gold + global REITs |
| 📉 Dubai Bear Market | -30% DFM correction | -{(30 * 0.85):.1f}% | {currency_sym}{(real_price or 0) * (1 - 30 * 0.85/100):,.2f} | Cash + Bonds |
| 🏦 Fed Rate Shock +2% | Higher financing cost | -{(18 * max(float(effective_beta), 0.35)):.1f}% | {currency_sym}{(real_price or 0) * (1 - 18 * max(float(effective_beta), 0.35)/100):,.2f} | US Treasuries |
| 🌱 Expo/Infrastructure Catalyst | Mega-project boost | +{(20 * 0.50):.1f}% | {currency_sym}{(real_price or 0) * (1 + 20 * 0.50/100):,.2f} | Hold / add |
INSTRUCTION FOR SECTION 9 (Scenario Analysis):
- You MUST include 3 core scenarios: Bear, Base, Bull — each with probability + expected price + expected return
- Core scenario probabilities MUST sum to 100%
- Any Macro Shock / Black Swan / Tail Risk scenario must be labeled as "💥 Tail Risk Overlay" and shown SEPARATELY
- Tail Risk Overlay must NOT be included in Expected Value calculation
- Expected Value = Σ(core_probability × core_return) across Bear/Base/Bull ONLY
- Show the EV calculation explicitly: "Expected Value: X.X%"
- After the core table, show tail risk separately:
  💥 **Tail Risk Overlay** | ~-25% | [trigger] | [hedge]
  ⚠️ *Not included in Expected Value calculation*
""" if (
    any(x in (fund.get('sector','') or '').lower() for x in ('real estate', 'property', 'reits'))
    and target.upper().endswith(('.DU', '.AE'))
) else (f"""SCENARIO ANALYSIS (Crash-Recovery — Post -39%+ Event):
⚠️ This stock experienced a severe single-day crash. Beta-adjusted scenarios are NOT meaningful here.
Use event-driven scenarios instead (corporate action, mean-reversion, or further collapse).
| Scenario | Trigger | Price Impact | Implied Price ({currency_sym}) | Suggested Action |
|----------|---------|-------------|--------------------------|-----------------|
| ✅ Corporate Action Clarified | Rights issue priced in — stock normalises | +{(45):.0f}% | {currency_sym}{(real_price or 0) * 1.45:,.2f} | BUY on confirmed clarity |
| 🔄 Partial Mean Reversion | Stock recovers 50% of crash | +{(25):.0f}% | {currency_sym}{(real_price or 0) * 1.25:,.2f} | Hold / add gradually |
| ⚠️ Fundamental Impairment | Crash = real earnings deterioration | -{(30):.0f}% | {currency_sym}{(real_price or 0) * 0.70:,.2f} | STOP LOSS immediately |
| 📉 Continued Selling / Forced Liquidation | No buyers for 1-2 weeks | -{(20):.0f}% | {currency_sym}{(real_price or 0) * 0.80:,.2f} | Volume confirmation pending |
| 🏦 EM Currency Devaluation | Local currency weakens -15% | -{(15):.0f}% | {currency_sym}{(real_price or 0) * 0.85:,.2f} | Hedge with USD exposure |
CRITICAL INSTRUCTION: In section 8, present THESE crash-recovery scenarios instead of generic beta-adjusted ones.
The #1 question investors need answered is: WHY did the stock crash -39%? Address this directly.
INSTRUCTION FOR SECTION 9 (Scenario Analysis):
- You MUST include 3 core scenarios: Bear, Base, Bull — each with probability + expected price + expected return
- Core scenario probabilities MUST sum to 100%
- Any Macro Shock / Black Swan / Tail Risk scenario must be labeled as "💥 Tail Risk Overlay" and shown SEPARATELY
- Tail Risk Overlay must NOT be included in Expected Value calculation
- Expected Value = Σ(core_probability × core_return) across Bear/Base/Bull ONLY
- Show the EV calculation explicitly: "Expected Value: X.X%"
- After the core table, show tail risk separately:
  💥 **Tail Risk Overlay** | ~-25% | [trigger] | [hedge]
  ⚠️ *Not included in Expected Value calculation*
""" if abs(change_pct or 0) >= 20 else f"""SCENARIO ANALYSIS (Beta-Adjusted — use these in section 9 of your report):
Note: Beta = {effective_beta}. Impact already pre-calculated (Market_Move × Beta). Copy EXACTLY — do NOT add extra columns.
REQUIREMENT: Show at least 2 BULLISH rows (🚀💡📈) and at least 2 BEARISH rows (📉🏦🤖⚠️).
| Scenario | Impact | Implied Price | Suggested Hedge |
|----------|--------|---------------|-----------------|
| 🚀 Bull Market Rally (+20%) | {(20 * float(effective_beta)):.1f}% | ${(real_price or 0) * (1 + (20 * float(effective_beta))/100):.2f} | Hold / add on dips |
| 💡 Fed Pivot / Rate Cut (+15%) | {(15 * float(effective_beta)):.1f}% | ${(real_price or 0) * (1 + (15 * float(effective_beta))/100):.2f} | Growth + Tech |
| 📉 AI/Tech Slowdown (-20%) | {(-20 * float(effective_beta)):.1f}% | ${(real_price or 0) * (1 + (-20 * float(effective_beta))/100):.2f} | Healthcare + Staples |
| 🏦 Fed Rate Shock +2% (-18%) | {(-18 * float(effective_beta)):.1f}% | ${(real_price or 0) * (1 + (-18 * float(effective_beta))/100):.2f} | Value stocks + Cash |
INSTRUCTION FOR SECTION 9 (Scenario Analysis):
- You MUST include 3 core scenarios: Bear, Base, Bull — each with probability + expected price + expected return
- Core scenario probabilities MUST sum to 100%
- Any Macro Shock / Black Swan / Tail Risk scenario must be labeled as "💥 Tail Risk Overlay" and shown SEPARATELY
- Tail Risk Overlay must NOT be included in Expected Value calculation
- Expected Value = Σ(core_probability × core_return) across Bear/Base/Bull ONLY
- Show the EV calculation explicitly: "Expected Value: X.X%"
- After the core table, show tail risk separately:
  💥 **Tail Risk Overlay** | ~-25% | [trigger] | [hedge]
  ⚠️ *Not included in Expected Value calculation*
""")))}
{(lambda: (
    __import__('core.news_engine_client', fromlist=['build_news_prompt_block'])
    .build_news_prompt_block(_engine_news_data, target)
    if _engine_news_data and (_engine_news_data.get('direct') or _engine_news_data.get('sector') or _engine_news_data.get('country'))
    else (
        (chr(10) + "LATEST NEWS (LIVE — integrate into Section 4 Risks and Section 7 Why Now):" + chr(10)
         + chr(10).join(f"- {n['title']}" for n in (ctx.get('news_links') or [])[:5]) + chr(10)
         + "INSTRUCTION: Reference at least 1-2 of these headlines in Section 4 Key Risks and/or Section 7 Why Now.")
        if ctx.get('news_links') else ""
    )
)())}"""

    # ETF data_block override
    if etf_meta:
        try:
            from core.etf_intelligence import (
                build_etf_data_block as _build_etf_db,
                build_etf_scenarios as _build_etf_sc,
            )
            from core.macro_intelligence import get_live_macro as _etf_glm
            _etf_macro_live = {}
            try:
                _etf_macro_live = _etf_glm()
            except Exception as exc:
                logger.debug("[build_data_block] ETF macro fetch failed for %s: %s", target, exc)
            _etf_db = _build_etf_db(
                etf_meta, target, real_price or 0, change_pct or 0,
                summary, fg_data, macro=_etf_macro_live, var_95=var_95, max_dd=max_dd,
            )
            _etf_scenarios = _build_etf_sc(etf_meta["etf_type"], real_price or 100, _etf_macro_live)
            data_block = _etf_db + "\n\n" + _etf_scenarios
            logger.info("[build_data_block] ETF override: %s (%s)", target, etf_meta["etf_type"])
            if not fund.get("sector") or fund.get("sector") in ("Unknown", "N/A", ""):
                _is_futures_ticker = target.upper().endswith("=F") or target.upper() in (
                    "GC=F", "SI=F", "CL=F", "NG=F", "PL=F", "PA=F", "HG=F", "BZ=F"
                )
                _etf_sector_map = {
                    "commodity_gold": "Commodities - Precious Metals" if _is_futures_ticker else "ETF - Precious Metals",
                    "commodity_silver": "Commodities - Precious Metals" if _is_futures_ticker else "ETF - Precious Metals",
                    "commodity_platinum": "Commodities - Precious Metals" if _is_futures_ticker else "ETF - Precious Metals",
                    "commodity_palladium": "Commodities - Precious Metals" if _is_futures_ticker else "ETF - Precious Metals",
                    "commodity_copper": "Commodities - Industrial Metals" if _is_futures_ticker else "ETF - Industrial Metals",
                    "commodity_oil": "Commodities - Energy" if _is_futures_ticker else "ETF - Energy",
                    "commodity_other": "Commodities" if _is_futures_ticker else "ETF - Commodities",
                    "bond_treasury": "Fixed Income",
                    "bond_corporate": "Fixed Income",
                    "bond_tips": "Fixed Income",
                    "equity_index_us": "Equities - US Index",
                    "equity_index_intl": "Equities - International",
                    "equity_sector": "Equities - Sector",
                    "reit_etf": "Real Estate",
                    "leveraged": "Leveraged ETF",
                    "dividend": "Equities - Dividend",
                }
                fr.fund["sector"] = _etf_sector_map.get(etf_meta["etf_type"], "ETF")
        except Exception as _etf_db_e:
            logger.debug("[build_data_block] ETF override skipped: %s", _etf_db_e)

    # X sentiment block (appended to data_block)
    x_data = fr.x_data or {}
    if x_data and x_data.get("sentiment") and x_data.get("source") != "grok-unavailable":
        _xs   = x_data.get("sentiment", "")
        _xsc  = x_data.get("score", 0.0)
        _xsum = x_data.get("x_summary", "")
        _xbrk = x_data.get("breaking")
        _xthm = x_data.get("themes", [])
        _xpst = x_data.get("top_posts", [])

        _x_block = "\n\n--- X/Twitter Sentiment (Grok Live · last 48h) ---\n"
        _x_block += f"Overall: {_xs} (score: {_xsc:+.2f})\n"
        if _xsum:
            _x_block += f"Summary: {_xsum}\n"
        if _xbrk:
            _x_block += f"⚡ BREAKING: {_xbrk}\n"
        if _xthm:
            _x_block += f"Key Themes: {' · '.join(_xthm)}\n"
        if _xpst:
            _x_block += "Top Posts from X:\n"
            for _p in _xpst[:4]:
                _lk  = f" ({_p.get('likes',0):,} likes)" if _p.get("likes") else ""
                _src = _p.get("source", "")
                _txt = _p.get("text", "")[:160]
                _dt  = _p.get("date", "")
                _imp = _p.get("impact", "Neutral")
                _ico = "🟢" if _imp == "Positive" else "🔴" if _imp == "Negative" else "⚪"
                _x_block += f"  {_ico} {_src}{_lk} ({_dt}): \"{_txt}\"\n"
        _x_block += (
            "INSTRUCTION: Use this X sentiment data in Section 8 (Why Now?) under a "
            "'📱 X Sentiment' bullet. If there is BREAKING news, mention it in Section 4 "
            "(Key Risks). ONLY cite sources that appear in the Top Posts above."
        )
        data_block += _x_block
        logger.info("[build_data_block] X sentiment injected for %s: %s (%+.2f)", target, _xs, _xsc)

    return data_block


# ── D. build_analytics_prompt ─────────────────────────────────────────────────

def build_analytics_prompt(
    target: str,
    data_block: str,
    ctx: dict,
    scorecard_verdict_hint: str,
    is_arabic: bool,
    brain_ctx: str,
    local_injection: str,
    research_summary: str,
    original_target: str | None = None,
    macro_block: str = "",
    pre_entry: str = "N/A",
    pre_stop: str = "N/A",
    pre_target: str = "N/A",
    user_ctx_block: str = "",
    research_context: str = "",
) -> str:
    """
    Builds the full DeepSeek investment memo prompt.
    Extracted from _handle_analytics lines 3684–3809.
    """
    from datetime import datetime as _dt

    fg_data      = ctx.get("fg_data", {}) or {}
    is_energy    = ctx.get("is_energy", False)
    oil_data     = ctx.get("oil_data", {}) or {}
    is_crash     = ctx.get("is_crash", False)
    crash_direction = ctx.get("crash_direction", "")
    change_pct   = ctx.get("change_pct", 0.0) or 0.0
    etf_meta     = ctx.get("etf_meta")
    x_data       = ctx.get("x_data", {}) or {}
    currency_sym = ctx.get("currency_sym", "$")
    currency_lbl = ctx.get("currency_lbl", "USD")

    _display_ticker = original_target if original_target else target

    prompt = f"""You are EisaX, Chief Investment Officer - built by Eng. Ahmed Eisa.

🚨 CRITICAL: Today's date is {_dt.now().strftime("%B %d, %Y")}.{research_summary}
   - You MUST use this EXACT date in your memo header
   - Any historical data reference must be clearly labeled as "historical"
   - All analysis must reflect current 2026 market conditions
   - MEMO SUBJECT LINE: In the memo header, the "Re:" line MUST use the ticker exactly as the user typed it: **{_display_ticker}** — NOT the resolved symbol. E.g. if user typed "XAUUSD", write "Re: Analysis of XAUUSD" not "Re: Analysis of GC=F".

Your advantage over general AI assistants:
- You are a SPECIALIZED financial analyst with 20+ years CIO experience
- You have access to LIVE market data (not training data)
- You provide institutional-grade analysis with specific entry/exit levels

🎯 SCORECARD PRE-VERDICT (computed before this memo): **{scorecard_verdict_hint}**
⛔ TONE ALIGNMENT RULE — MANDATORY:
Your memo MUST be tonally consistent with the above verdict:
- If verdict = REDUCE or SELL → Executive Summary must reflect caution. No "compelling entry", "attractive opportunity", or buy language. Acknowledge the headwinds clearly.
- If verdict = HOLD → Balanced tone. Acknowledge both upside potential and risks equally.
- If verdict = BUY or ACCUMULATE → Constructive tone. State the opportunity while acknowledging risks.
- ⛔ NEVER write a bullish Executive Summary when the verdict is REDUCE/SELL. This creates a contradiction the client will immediately notice and destroys credibility.

🔴 LANGUAGE QUALITY RULES:
- ⛔ NEVER use boilerplate phrases like "according to recent analyst data", "market observers note", "analysts suggest", or "industry experts believe" — these are empty filler. Use the ACTUAL data provided or state explicitly that the data is unavailable.
- ⛔ NEVER cite a news source that is NOT in the LATEST NEWS section of the data below. Do NOT reference "The Times of India", "Hindustan Times", regional newspapers, blogs, or any outlet from your training knowledge. If you cite a source, it MUST appear verbatim in the LATEST NEWS section.
- ⛔ NEVER invent or paraphrase headlines not present in the LATEST NEWS data. If no relevant news exists, say "No relevant headlines at time of analysis."
- ⛔ BE CONSISTENT on valuation: if the Scorecard labels Forward P/E as "🟢 Reasonable", do NOT describe the same P/E as "elevated" in the memo body. Use the same label throughout.
- ⛔ EARNINGS DATE: Use ONLY the exact date from the data. NEVER combine a fiscal quarter label from one year with a date from another year (e.g. "Q1 2027 on April 29, 2026" is wrong). If unsure of the fiscal quarter label, just say "next earnings report on [date]".
- ✅ Peer comparisons in Section 6 MUST include actual numbers. E.g., "GOOGL trades at 22x forward P/E vs {target}'s Xx" — not just "GOOGL is a peer".
- ✅ If EPS growth estimate is available in the data, include the YoY % in Section 2.

Analyze the following data and write an institutional-grade investment memorandum.
{user_ctx_block}
{data_block}

{(f"""
⚠️ ETF ANALYSIS MODE — {etf_meta['etf_label'] if etf_meta else ''}
This is an ETF, NOT a stock. Follow ETF-specific rules:
- Section 2 = "{"Commodity Analysis" if etf_meta and etf_meta.get("etf_type","").startswith("commodity") else "Fund Analysis"}" (NOT Fundamental Analysis): Discuss what the fund/contract tracks, expense ratio cost drag, AUM liquidity, and how the underlying asset/index is valued. NO EPS, Revenue, ROE, ROIC, or corporate metrics.
- Section 5 = "Market Catalysts": No analyst consensus. Discuss macro catalysts that drive this fund (rate moves, commodity shifts, sector rotation, etc.).
- Section 6 = "⚔️ Peer Comparison": name 2 direct alternative funds (by ticker). Compare expense ratio, yield/return profile, and AUM in exactly 2 sentences. No corporate competitors — funds only.
- Section 7 = "EisaX Outlook": Compare to ALTERNATIVE investments (e.g., for GLD: compare to TLT, T-bills, TIPS; for TLT: compare to HYG, cash, SPY). Include one specific number and one risk/reward statement.
- Section 9 = Use the ETF-SPECIFIC scenario table provided in the data.
- Do NOT mention P/E ratio, EPS, Revenue, ROE, ROIC, analyst price targets, or earnings dates.
""") if etf_meta else ""}
Structure your response with these sections (ALL sections are MANDATORY — do NOT skip any):
1. **Executive Summary** (2-3 sentences with clear stance)
2. **{"Commodity Analysis" if etf_meta and etf_meta.get("etf_type","").startswith("commodity") else "Fund Analysis" if etf_meta else "Fundamental Analysis"}** ({"macro drivers, real yield sensitivity, USD relationship, central bank demand, and supply/demand dynamics for the underlying commodity. Do NOT use ETF/fund language — this is a commodity futures contract." if etf_meta and etf_meta.get("etf_type","").startswith("commodity") else "what the fund tracks, expense ratio drag, AUM size, macro drivers of the underlying asset" if etf_meta else "growth quality, profitability, valuation - mention Forward P/E and Gross Margin GAAP note"})
3. **Technical Outlook** (MANDATORY — you MUST include ALL of the following from the TECHNICALS data):
   - SMA50, SMA200, RSI, MACD, ADX values with trend direction and momentum condition
   - CRITICAL: use the exact RSI condition label from data — e.g. "RSI: 32.2 (Near Oversold)" not your own label
   - Volume vs average: state if volume is LOW/NORMAL/HIGH vs 90-day avg and what this means for conviction
   - Fibonacci levels: mention the nearest resistance ABOVE current price and the key support BELOW (from FIBONACCI LEVELS data)
   - ⚠️ Technical Note: Momentum indicators (MACD/RSI) reflect price-driven buying pressure, while ADX measures trend strength independently of direction. A bullish momentum reading alongside a weak ADX (< 25) indicates early-stage or range-bound price action — not a confirmed trend. Treat momentum signals with reduced confidence until ADX sustains above 25.
   - ⛔ Do NOT repeat these technical facts in Section 8 (Why Now) — Section 8 focuses on TIMING and CATALYSTS only
4. **Key Risks** (top 2-3 BUSINESS risks with severity rating):
   ⛔ DATA GAPS ARE NOT RISKS: If fundamental metrics (ROE, ROIC, Net Margin, etc.) are unavailable, note this ONCE in Section 2 as a data limitation. Do NOT list "Weak Fundamental Metrics" or "Data Unavailability" as a Key Risk in Section 4.
   ✅ Section 4 must contain only genuine business, macro, commodity, regulatory, or market risks (e.g., oil price volatility, competition, geopolitical risk, rate sensitivity, regulatory change).
   MANDATORY: If LATEST NEWS appears in the data, reference at least one relevant headline here as a named risk. E.g., "Geopolitical Risk (Severity: High): [headline about Hormuz/Iran/OPEC]..."
5. **Analyst Consensus & Catalysts** (mention price target, upside %, upcoming earnings)
6. **⚔️ Peer Comparison** (MANDATORY — do NOT skip — exactly 2 sentences, no more):
{"   ETF mode: name 2 direct alternative funds. Compare expense ratio, yield/return, and AUM. Format: \"vs [FUND]: [difference]. [why an investor would choose this one over it].\"" if etf_meta else
"   Stock mode — compare to the single closest DIRECT competitor in the same sub-industry:\n"
"   • Sentence 1 (Valuation): state both forward P/E (or EV/EBITDA, P/S for growth) values and the % premium or discount.\n"
"   • Sentence 2 (Edge): where does this company lead or lag vs the peer? (growth rate, margin, market share, moat, product pipeline)\n"
"   Format: \"vs [PEER_TICKER]: [valuation sentence]. [competitive position sentence].\"\n"
"   Example: \"vs NVDA: AMD trades at 24x fwd P/E vs NVDA's 35x — a 31% discount. AMD leads in CPU market share but lags NVDA's data center GPU dominance (NVDA holds ~80% market share vs AMD ~15%).\"\n"
"   ⛔ Do NOT write more than 2 sentences. ⛔ Do NOT include any rating or recommendation.\n"
"   ⛔ Do NOT say 'data unavailable', 'comparison unavailable', or 'data gaps'. Use your training knowledge for the peer's approximate forward P/E if not in the data.\n"
"   ⛔ If you truly cannot compare, name the peer and compare qualitatively (margins, growth, market share).\n"
"   ⚡ PEER SELECTION: Choose the MOST RELEVANT competitor — for cloud/software companies this may be AMZN (AWS) or META, not necessarily GOOGL. For UAE/Saudi companies compare to the closest regional peer."}"
"\n⛔ BANNED PHRASES — never write these regardless of verdict:\n"
"- \"bullish trends are expected in 2026\" or any variation\n"
"- \"diversification is recommended, aligning with our balanced view\"\n"
"- Any generic forward-looking phrase not supported by the data above.\n"

7. **EisaX Outlook** — Write 2-3 sentences with:
   - One specific number (e.g. implied return, EV/EBITDA vs peers, FCF yield, or PEG ratio)
   - One clear risk/reward statement
   - ⛔ DO NOT include any verdict, buy/sell/hold rating, or recommendation
   - ⛔ DO NOT write any score or scorecard
   - The official verdict is auto-generated in the EisaX Scorecard below

8. **⏰ Why Now?** (MANDATORY — focus on TIMING and CATALYSTS, not technical analysis which belongs in Section 3):
   • Market Sentiment: Fear & Greed at {fg_data.get('score','N/A')} ({fg_data.get('rating','N/A')}) — what extreme reading means for entry timing RIGHT NOW
   • Upcoming Catalyst: next earnings date, product launch, regulatory event, or sector-specific driver — cite LATEST NEWS if relevant; explain WHY this catalyst matters NOW
   • Risk/Timing: one specific risk to the entry timing (NOT a repeat of Section 4 risks — frame it as timing risk, e.g. "Could fall further before earnings", "Momentum may not reverse until X")
   {"• Oil Price: Brent at $" + str(round(oil_data.get('price',0),2)) + "/bbl — impact on revenue and margins" if is_energy else ""}
   {("• 📱 X Sentiment: Copy EXACTLY — sentiment is **" + str(x_data.get('sentiment','')) + "** (score: " + f"{x_data.get('score',0):+.2f}" + "). Key themes: " + ", ".join(x_data.get('themes',[])[:2]) + ". Do NOT change the sentiment label or score — use them verbatim. Cite specific accounts from the Top Posts if available.") if x_data and x_data.get("sentiment") else ""}
   Format: "• [Factor]: [Implication]"

9. **🌍 Advanced Scenario Analysis**
   {"Include the Oil Price Sensitivity table AND the Energy-Sector scenario table from the data. Show how different oil prices ($50-$90/bbl) affect this stock." if is_energy else "Include a markdown table of 4 beta-adjusted scenarios from the SCENARIO ANALYSIS section in the data. REQUIREMENT: At least 2 scenarios must be BULLISH (upside cases) and at least 1 must be BEARISH. Do NOT generate all-bearish or all-downside scenarios — this is for institutional investors who need balanced upside and downside analysis."}
   Format:
   Emoji rule: 🚀📈💡 for BULLISH rows · 📉🏦🤖⚠️ for BEARISH rows. NEVER use 📉 on a positive-impact row.
   ⛔ The SCENARIO ANALYSIS data already has exactly 4 columns: Scenario | Impact | Implied Price | Suggested Hedge. Copy this table EXACTLY — do NOT add a Market Move column or split any cell. Use "Expected Price" as the header for the price column.
   | Scenario | Impact | Expected Price | Suggested Hedge |
   |----------|--------|----------------|-----------------|

{"10. **🛢️ Oil Price Sensitivity** (MANDATORY for energy stocks): Include the full Oil Price Sensitivity table from the data showing revenue impact at $50, $60, $70, $80, $90/bbl. Discuss the breakeven oil price and OPEC+ production outlook." if is_energy else ""}

Use actual numbers. Be specific. Institutional tone.
{"CRITICAL: This is an ENERGY sector stock. Oil prices are the PRIMARY driver. You MUST discuss oil price impact throughout the report, include the sensitivity table, and reference Brent crude at $" + str(round(oil_data.get('price',0),2)) + "/bbl." if is_energy else ""}
{"CURRENCY: Use " + currency_sym + " (" + currency_lbl + ") for ALL price references — NOT USD." if currency_lbl != "USD" else ""}
{"LANGUAGE: The user's request was in Arabic. Write the FULL report in Arabic. IMPORTANT: Use the SAME number of sections, SAME level of detail, and ALL 9 sections — do NOT simplify or shorten because it is in Arabic. Arabic and English reports must be identical in depth and structure. Section 6 (Peer Comparison) must still be exactly 2 sentences with competitor ticker and valuation numbers in Arabic." if is_arabic else "LANGUAGE: Write in English."}
{"🚨 EXTREME PRICE MOVE ALERT — " + crash_direction + " (" + f"{change_pct:+.2f}%" + " single-day move detected): This MUST be the FIRST thing addressed in Section 1 (Executive Summary). In Section 4 (Key Risks), you MUST investigate and explain the likely cause: check if this is an ex-dividend drop, rights issue (capital increase), trading halt lifted, forced selling, major news event, or circuit-breaker trigger. State the most probable cause based on available data. Do NOT treat this as a normal trading day — this is an exceptional event requiring forensic analysis." if is_crash else ""}
IMPORTANT RULES:
- Do NOT mention dividend yield unless above 0.5%
- Entry zone must ALWAYS be BELOW the current live price
- Stop loss: one consistent value only
- Analyst count: use the EXACT number from the data. Do NOT round or cap it.
- ⛔ EARNINGS DATE RULE: The NEXT EARNINGS DATE in the data is the ONLY date to use. Do NOT derive or guess fiscal quarter labels (Q1/Q2/Q3/Q4) from calendar dates — the fiscal year varies by company. Use the date as-is (e.g. "April 29, 2026") and say "next earnings" not "Q1 FY2027".
- ⛔ NEVER write "Score: XX/100" in sections 1-8. That appears ONLY in the Scorecard.
- ⛔ DO NOT create any scorecard table, score breakdown, scoring methodology, or positioning section in your response. NO "Growth: X/30", "Valuation: X/20", "Score: XX/100", "Confidence Score", "Entry Zone", "Stop Loss", "Target" sections. The EisaX Proprietary Scorecard AND Positioning Guide are automatically appended below your memo — ANY duplication causes critical display errors and will be rejected.
- ⛔ Your response MUST end after section 9. Do NOT add any additional sections, tables, or blocks after section 9.
- ALL 9 sections above are MANDATORY. Do NOT skip Technical Outlook, Why Now, or Advanced Scenario Analysis.
- ⛔ NEWS INTEGRATION RULE: If FRESH NEWS CONTEXT is provided in the data, you MUST cite at least 1 specific headline by name in Section 1 (Executive Summary) AND at least 1 in Section 4 (Key Risks). Do NOT generically mention "recent news" — quote or paraphrase the actual headline title. Failing to integrate news is a critical quality failure.
- ⛔ CONSISTENCY RULE: Section 8 (Why Now) must be CONSISTENT with the Scorecard verdict. If the verdict is REDUCE or SELL, do NOT frame the analysis as a "contrarian opportunity" or suggest it is a good entry point. Instead, explain what would need to change for the thesis to improve. If the verdict is HOLD/BUY, you may describe constructive entry timing.
- ⛔ UPSIDE LANGUAGE RULE: Only use "strong upside" when upside potential is genuinely >20%. For <10% upside use "modest upside" or "limited upside". For 10-20% upside use "moderate upside". Never call +3% to +5% returns "strong upside" — that misleads investors.
Do NOT include a standalone Positioning section.{brain_ctx}
{macro_block}
"""

    prompt = prompt.replace("PLACEHOLDER_ENTRY", pre_entry)
    prompt = prompt.replace("PLACEHOLDER_TARGET", pre_target)
    prompt = prompt.replace("PLACEHOLDER_STOP", pre_stop)
    prompt += "\n\n🚨 MANDATORY: Entry=" + pre_entry + " | Stop=" + pre_stop + " | Target=" + pre_target + " — USE THESE EXACT LEVELS."
    if research_context:
        prompt += "\n\n" + research_context
    prompt += local_injection

    return prompt


# ── E. assemble_report ────────────────────────────────────────────────────────

def assemble_report(
    target: str,
    fr: FetchResult,
    ctx: dict,
    deepseek_reply: str,
    news_block: str,
    pos: dict,
    pre_scorecard_md: str,
    original_target: str | None = None,
) -> str:
    """
    Assembles the final markdown report from all pieces.
    Extracted from _handle_analytics lines 3862–4403.
    Does NOT set state.last_artifact or call _save_to_brain — caller is responsible.
    """
    import math as _math_pos
    import re as _re_rep

    fund       = fr.fund or {}
    summary    = fr.summary or {}
    var_95     = fr.var_95 or 0.02
    max_dd     = fr.max_dd or 0.20
    real_price = fr.real_price
    change_pct = fr.change_pct or 0.0

    currency_sym = ctx.get("currency_sym", "$")
    currency_lbl = ctx.get("currency_lbl", "USD")
    _is_local_mkt = currency_lbl != "USD"
    _is_local_currency = currency_lbl in ("SAR", "AED", "EGP", "KWF", "QAR")
    is_energy    = ctx.get("is_energy", False)
    oil_data     = ctx.get("oil_data", {}) or {}
    is_crash     = ctx.get("is_crash", False)
    effective_beta = ctx.get("effective_beta", 1.0)
    display_target = ctx.get("display_target")
    target_is_estimate = ctx.get("target_is_estimate", True)
    _is_regional_energy = ctx.get("is_regional_energy", False)
    _is_local_ticker    = ctx.get("is_local_ticker", False)
    x_data       = fr.x_data or {}
    _engine_news_data = fr.engine_news or {}

    _fallback_price = real_price or summary.get("price", 0)
    price_str = (
        f"{_fallback_price:,.2f} {currency_sym} ({change_pct:+.2f}%)"
        if _fallback_price and _is_local_mkt and change_pct
        else f"{_fallback_price:,.2f} {currency_sym}"
        if _fallback_price and _is_local_mkt
        else f"${_fallback_price:,.2f} ({change_pct:+.2f}%)"
        if _fallback_price and change_pct
        else f"${_fallback_price:,.2f}"
        if _fallback_price else "N/A"
    )
    _t_upper = target.upper()

    # ── EisaX score from scorecard markdown ──────────────────────────────────
    _eisax_score_match = _re_rep.search(r"EisaX Score:\s*\*\*(\d+)/100\*\*", pre_scorecard_md)
    _eisax_score = _eisax_score_match.group(1) if _eisax_score_match else "N/A"

    _exch_label = (
        "🇸🇦 Tadawul · SAR" if _t_upper.endswith(".SR") else
        "🇦🇪 ADX/DFM · AED" if _t_upper.endswith((".AE", ".DU")) else
        "🇪🇬 EGX · EGP" if _t_upper.endswith(".CA") else
        "🇰🇼 Boursa Kuwait · KWF" if _t_upper.endswith(".KW") else
        "🇶🇦 Qatar Exchange · QAR" if _t_upper.endswith(".QA") else ""
    )
    _oil_badge = f" | **🛢️ Brent: ${oil_data.get('price',0):.2f}**" if is_energy and oil_data.get("price") else ""
    _display_ticker = original_target if (original_target and original_target != target) else target

    header = (
        f"# EisaX Intelligence Report: {_display_ticker}\n\n"
        f"**🔴 Live Price:** {price_str} | "
        f"**Sector:** {fund.get('sector', 'N/A')} | "
        f"**EisaX Score:** {_eisax_score}/100"
        + (f" | **{_exch_label}**" if _exch_label else "")
        + _oil_badge
        + "\n\n---\n\n"
    )

    # ── Chart placeholder ─────────────────────────────────────────────────────
    chart_block = (
        f'\n\n---\n📈 **Price Chart (60 days)**\n'
        f'<div class="eisax-chart" data-ticker="{target}"></div>'
    )

    # ── Disclaimer ────────────────────────────────────────────────────────────
    _analysis_disclaimer = (
        "\n\n---\n"
        "> ⚠️ **Disclaimer:** This report is generated by EisaX AI and is for informational purposes only. "
        "It does not constitute financial advice, investment recommendation, or an offer to buy or sell any security. "
        "All prices and data are fetched live at the time of the query and may not reflect real-time market conditions. "
        "Past performance is not indicative of future results. Always verify data independently and consult a licensed financial advisor before making investment decisions."
    )

    # ── Positioning block ─────────────────────────────────────────────────────
    def _clean(v, d=0.0):
        try:
            f = float(v or 0)
            return d if (_math_pos.isnan(f) or _math_pos.isinf(f)) else f
        except Exception:
            logger.debug("[positioning] _clean: cannot coerce %r to float — using default %r", v, d, exc_info=True)
            return d

    sma50  = _clean(summary.get("sma_50", 0))
    sma200 = _clean(summary.get("sma_200", 0))
    ep     = pos.get("ep")
    sp     = pos.get("sp")

    _fp_ref = _clean(real_price or _fallback_price or 0)
    if _fp_ref and sma200:
        _pct_from_sma = (_fp_ref - sma200) / sma200
        if _pct_from_sma < -0.10:
            entry_price = _fp_ref * 0.97
            stop_price  = _fp_ref * 0.91
        elif _pct_from_sma < 0:
            entry_price = sma200 * 0.98
            stop_price  = sma200 * 0.92
        else:
            entry_price = ep if ep else sma200 * 1.01
            stop_price  = sp if sp else sma200 * 0.95
    else:
        entry_price = ep if ep else (_fp_ref * 0.96 if _fp_ref else None)
        stop_price  = sp if sp else (_fp_ref * 0.91 if _fp_ref else None)

    if _fp_ref and entry_price and entry_price >= _fp_ref:
        entry_price = _fp_ref * 0.97
        stop_price  = _fp_ref * 0.91

    def _fmt_price(p):
        if not p:
            return "N/A"
        return f"{p:,.2f} {currency_sym}" if _is_local_mkt else f"${p:,.2f}"

    entry_level = _fmt_price(entry_price)
    stop_level  = _fmt_price(stop_price)
    _pos_target = display_target
    _rp_pos = real_price or _fallback_price or 0

    _target_is_sma = False
    if _pos_target and _rp_pos:
        upside = ((_pos_target / _rp_pos) - 1) * 100
        target_level = (
            f"{_pos_target:,.2f} {currency_sym} ({upside:+.1f}%)"
            if _is_local_mkt
            else f"${_pos_target:,.2f} ({upside:+.1f}%)"
        )
    elif sma200 and _rp_pos:
        if _rp_pos < sma200:
            _tech_tgt = sma200 * 1.15
        elif sma50 and _rp_pos < sma50:
            _tech_tgt = sma50
        else:
            _tech_tgt = sma200 * 1.15
        _sma_used = "SMA50" if (sma50 and _rp_pos < sma50) else "SMA200"
        _tech_up  = ((_tech_tgt / _rp_pos) - 1) * 100
        target_level = (
            f"{_tech_tgt:,.2f} {currency_sym} ({_tech_up:+.1f}%)"
            if _is_local_mkt
            else f"${_tech_tgt:,.2f} ({_tech_up:+.1f}%)"
        )
        _target_is_sma = True
    elif sma50 and _rp_pos:
        _tech_tgt = sma50 if _rp_pos < sma50 else sma50 * 1.05
        _tech_up  = ((_tech_tgt / _rp_pos) - 1) * 100
        _sma_used = "SMA50"
        target_level = (
            f"{_tech_tgt:,.2f} {currency_sym} ({_tech_up:+.1f}%)"
            if _is_local_mkt
            else f"${_tech_tgt:,.2f} ({_tech_up:+.1f}%)"
        )
        _target_is_sma = True
    else:
        target_level = "N/A"

    if "_sma_used" not in dir():
        _sma_used = "SMA50" if (sma50 and not sma200) else "SMA200"

    _target_rationale = (
        f"⚠️ Technical target ({_sma_used} mean-reversion) — not analyst" if _target_is_sma
        else "⚠️ EisaX FV Estimate (no analyst coverage)" if target_is_estimate
        else "Analyst consensus target"
    )
    _rp_pos2 = real_price or _fallback_price or 0
    _stop_rationale = (
        "Below SMA200 (-5%)"
        if stop_price and sma200 and abs(stop_price - sma200 * 0.95) / (sma200 * 0.95) < 0.03
        else "Trailing stop (-9% from current)"
        if _rp_pos2 and stop_price and stop_price >= _rp_pos2 * 0.88
        else "Key support level (-9% from current)"
    )
    _rp_pos3 = real_price or _fallback_price or 0
    if entry_price and _rp_pos3 and _rp_pos3 > entry_price * 1.02:
        _pct_to_entry = ((_rp_pos3 - entry_price) / _rp_pos3) * 100
        _entry_note = (
            f"\n\n> ⏳ **Awaiting Pullback** — Current price "
            f"({_fmt_price(_rp_pos3)}) is **{_pct_to_entry:.1f}% above** the entry level. "
            f"Current price ({_fmt_price(_rp_pos3)}) is **{_pct_to_entry:.1f}% above** the identified entry zone of {_fmt_price(entry_price)}, which reduces the margin of safety relative to the defined risk parameters."
        )
    else:
        _entry_note = ""

    _entry_rationale = (
        "Near SMA200 support"
        if entry_price and sma200 and abs(entry_price - sma200) / sma200 < 0.05
        else "Pullback entry — below current price"
        if entry_price and _rp_pos3 and entry_price < _rp_pos3 * 0.98
        else "At current price — entry zone active"
    )

    positioning_block = (
        f"\n\n---\n"
        f"📊 **Positioning Guide**\n"
        f"| | Level | Rationale |\n"
        f"|---|---|---|\n"
        f"| 🟢 Entry | {entry_level} | {_entry_rationale} |\n"
        f"| 🎯 Target | {target_level} | {_target_rationale} |\n"
        f"| 🔴 Stop | {stop_level} | {_stop_rationale} |\n"
        f"{_entry_note}"
    )

    # ── Assemble with DeepSeek reply ──────────────────────────────────────────
    if deepseek_reply:
        try:
            from core.fact_checker import FactChecker
            fact_data = {**summary, "price": real_price or summary.get("price")}
            factcheck_block = FactChecker().verify_analysis(target, fact_data)
        except Exception as _fce:
            logger.error("[assemble] FactChecker failed: %s", _fce)
            factcheck_block = ""
        _report = (
            header
            + deepseek_reply
            + factcheck_block
            + news_block
            + positioning_block
            + pre_scorecard_md
            + chart_block
            + _analysis_disclaimer
        )
        try:
            import sys as _sys
            from core.config import BASE_DIR as _BASE_DIR
            _root = str(_BASE_DIR)
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from report_enhancer import ReportEnhancer
            from pipeline import cache as _cache, fetcher as _fetcher
            from query_engine import QueryEngine
            _qe = QueryEngine(_cache, _fetcher)
            _report = ReportEnhancer(_qe).enhance(_report, ticker=target)
            logger.info("[assemble] Enhancer applied to %s", target)
        except Exception as _enh_err:
            logger.warning("[assemble] Enhancer skipped for %s: %s", target, _enh_err)
        return _report

    # ── Fallback: structured reply without DeepSeek ───────────────────────────
    def _P(n):
        return f"{n:.1f}%" if n else "N/A"

    def _X(n):
        return f"{n:.1f}x" if n else "N/A"

    def _B(n):
        try:
            if not n: return "N/A"
            v = float(n)
            if currency_lbl != "USD":
                if v >= 1e9:  return f"{v/1e9:.1f}B {currency_sym}"
                if v >= 1e6:  return f"{v/1e6:.0f}M {currency_sym}"
                return f"{v:,.0f} {currency_sym}"
            return f"${v/1e9:.1f}B" if v >= 1e9 else f"${v/1e6:.0f}M"
        except Exception:
            logger.debug("[fallback_report] _B: format error for value %r", n, exc_info=True)
            return "N/A"

    verdict = (
        "ACCUMULATE" if summary.get("trend") == "Bullish" and summary.get("momentum") == "Bullish"
        else "REDUCE" if summary.get("trend") == "Bearish" and summary.get("momentum") == "Bearish"
        else "HOLD"
    )
    reply = (
        header
        + f"## Fundamentals\n"
        f"- Revenue Growth: {_P(fund.get('revenue_growth'))} | EPS Growth: {_P(fund.get('eps_growth'))}\n"
        f"- Net Margin: {_P(fund.get('net_margin'))} | ROE: {_P(fund.get('roe'))}\n"
        f"- P/E: {_X(fund.get('pe_ratio'))} | EV/EBITDA: {_X(fund.get('ev_ebitda'))}\n"
        f"- Market Cap: {_B(fund.get('market_cap'))} | Cash: {_B(fund.get('cash'))}\n\n"
        f"## Technicals\n"
        f"- Trend: {summary.get('trend','N/A')} | RSI: {summary.get('rsi',50):.1f} | MACD: {summary.get('momentum','N/A')}\n"
        f"- VaR(95%): {var_95*100:.2f}% | Max DD: {max_dd*100:.2f}%\n\n"
        f"**EisaX Verdict: {verdict}**"
    )
    try:
        from core.fact_checker import FactChecker
        fact_data = {**summary, "price": real_price or summary.get("price")}
        fact_block = FactChecker().verify_analysis(target, fact_data)
        reply += "\n\n" + fact_block
    except Exception as _fce2:
        logger.error("[assemble] FactChecker (fallback) failed: %s", _fce2)
    try:
        import sys as _sys
        from core.config import BASE_DIR as _BASE_DIR
        _root = str(_BASE_DIR)
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from report_enhancer import ReportEnhancer
        from pipeline import cache as _cache, fetcher as _fetcher
        from query_engine import QueryEngine
        _qe = QueryEngine(_cache, _fetcher)
        reply = ReportEnhancer(_qe).enhance(reply, ticker=target)
        logger.info("[assemble] Enhancer applied to %s", target)
    except Exception as _enh_err:
        logger.warning("[assemble] Enhancer skipped for %s: %s", target, _enh_err)
    return reply
