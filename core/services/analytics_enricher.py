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

    # Ensure canonical 52W key names — finance.py DB cache stores as year_high/year_low
    if not fund.get("week52_high") and fund.get("year_high"):
        fund["week52_high"] = float(fund["year_high"])
    if not fund.get("week52_low") and fund.get("year_low"):
        fund["week52_low"] = float(fund["year_low"])
    fr.fund = fund

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

