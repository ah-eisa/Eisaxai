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
from core.pm_tickers import _normalize_tickers, has_placeholder_tickers, get_ticker_name, _tv_to_yfinance, get_top_regional_tickers, smart_expand_tickers
from core.pm_reporting import compute_risk_score, render_optimize_reply, render_report, build_portfolio_report_body, generate_executive_report_llm, _compute_extras, generate_strategy_guide_llm

def optimize_and_get_data(
    *,
    tickers: list[str],
    start: str,
    end: str | None,
    method: str,
    min_w: float,
    max_w: float,
    min_assets: int,
    seed_w: float,
    rf: float,
    **kwargs,
) -> tuple[dict, dict]:
    """
    Performs the optimization and returns (weights_raw, performance).
    Does NOT handle memory side effects.
    """
    # ── Normalize tickers (fix fake/invalid regional names) ──────────────────
    tickers = _normalize_tickers(tickers)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Hard fake-ticker guard — last line of defence before the optimizer ────
    # Any ticker that looks like a placeholder word is removed here, regardless
    # of how it got in (LLM, user input, session memory, etc.)
    _HARD_FAKE = {
        "NEEDED", "ASSET", "STOCK", "INDEX", "TICKER", "SYMBOL", "PLACEHOLDER",
        "ARAB", "MARKET", "EQUITY", "SHARE", "ITEM", "OTHER", "CASH", "FUND",
        "AS", "AN", "IN", "TO", "OF", "OR", "AT", "BY",  # common 2-letter words
        "ADD", "NEW", "SET", "USE", "GET", "PUT", "TBD", "N/A", "NA", "XX",
    }
    _before = tickers[:]
    tickers = [t for t in tickers if t.upper() not in _HARD_FAKE]
    if len(tickers) < len(_before):
        _removed = set(_before) - set(tickers)
        logger.warning("[Optimizer] Removed fake tickers before optimization: %s", _removed)
    if not tickers:
        raise ValueError("No valid tickers remaining after fake-ticker cleanup.")
    # ─────────────────────────────────────────────────────────────────────────

    # ── Global hard cap: no single asset may exceed 20% ─────────────────────
    _GLOBAL_MAX_W = 0.20
    max_w = min(max_w, _GLOBAL_MAX_W)
    # ─────────────────────────────────────────────────────────────────────────

    # Auto-adjust constraints (ensures feasibility when ticker count is low)
    if tickers and max_w < 1.0:
        min_possible_max_w = 1.0 / len(tickers)
        if max_w < min_possible_max_w:
            max_w = min(1.0, min_possible_max_w + 0.05)

    # Lazy import to avoid hard dependency on pypfopt if unneeded
    from core.portfolio import optimize as optimize_core, parse_constraints

    # ── Mixed-market fix ─────────────────────────────────────────────────────
    # US + Local (SR/AE/DU/CA) in one optimizer → singular covariance matrix
    # because of different trading calendars and missing data.
    # Fix: optimize each market group separately, then blend weights by allocation.
    _LOCAL_SUFFIXES = (".SR", ".AE", ".DU", ".CA", ".KW", ".QA")
    _local = [t for t in tickers if t.upper().endswith(_LOCAL_SUFFIXES)]
    _us    = [t for t in tickers if not t.upper().endswith(_LOCAL_SUFFIXES)]

    if _local and _us:
        logger.info("[Portfolio] Mixed-market detected: US=%s, Local=%s — splitting optimization", _us, _local)
        total      = len(tickers)
        us_alloc   = len(_us)    / total
        local_alloc= len(_local) / total

        w_final   = {}
        perf_us   = {}
        perf_local= {}

        # ── US group ──
        try:
            prices_us = get_prices(_us, start=start, end=end, force_refresh=False)
            _us_max_w = min(0.40, 1.0 / len(_us) + 0.10)
            w_us, perf_us = optimize_core(
                prices_us, method=method,
                min_w=0.0, max_w=_us_max_w,
                return_performance=True,
                target_return=kwargs.get("target_return"),
                max_drawdown=kwargs.get("max_drawdown"),
            )
            for t, w in w_us.items():
                w_final[t] = w * us_alloc
        except Exception as _us_err:
            logger.warning("[Portfolio] US group optimization failed: %s — using equal weight", _us_err)
            for t in _us:
                w_final[t] = us_alloc / len(_us)

        # ── Local group ──
        try:
            from datetime import datetime as _dt, timedelta as _td
            # Use only 2 years of data for local markets (less history available)
            _local_start = (_dt.now() - _td(days=730)).strftime("%Y-%m-%d")
            prices_local = get_prices(_local, start=_local_start, end=end, force_refresh=False)
            # Drop columns with more than 30% missing data
            _min_rows = int(len(prices_local) * 0.70)
            prices_local = prices_local.dropna(thresh=_min_rows, axis=1)
            prices_local = prices_local.ffill().bfill()  # fill remaining gaps

            if prices_local.empty or len(prices_local.columns) == 0:
                raise ValueError("No valid local price data after cleaning")

            _surviving_local = list(prices_local.columns)
            _dropped = [t for t in _local if t not in _surviving_local]
            if _dropped:
                logger.warning("[Portfolio] Dropped local tickers (insufficient data): %s", _dropped)

            _local_max_w = min(0.60, 1.0 / len(_surviving_local) + 0.15)
            # Use min_vol for local markets — more stable with sparse data
            w_local, perf_local = optimize_core(
                prices_local, method="min_vol",
                min_w=0.0, max_w=_local_max_w,
                return_performance=True,
            )
            for t, w in w_local.items():
                w_final[t] = w * local_alloc

        except Exception as _local_err:
            logger.warning("[Portfolio] Local group optimization failed: %s — using equal weight", _local_err)
            for t in _local:
                w_final[t] = local_alloc / len(_local)

        # ── Normalize weights to sum = 1.0 ──
        _total_w = sum(w_final.values())
        if _total_w > 0:
            w_raw = {t: w / _total_w for t, w in w_final.items()}
        else:
            w_raw = {t: 1.0 / len(tickers) for t in tickers}

        # Use US perf as primary (more reliable), flag as mixed
        perf = perf_us if perf_us else perf_local
        perf["mixed_market"] = True
        perf["mixed_market_groups"] = {"us": _us, "local": _local}

        # ── Compute extras then return early ──
        try:
            _all_prices = get_prices(list(w_raw.keys()), start=start, end=end, force_refresh=False)
            _extras = _compute_extras(w_raw, _all_prices, perf)
            perf.update(_extras)
        except Exception as _ex_err:
            logger.warning("[Portfolio] Extras computation failed (mixed): %s", _ex_err)

        return w_raw, perf
    # ── End mixed-market fix ─────────────────────────────────────────────────

    prices = get_prices(tickers, start=start, end=end, force_refresh=False)
    w_raw, perf = optimize_core(
        prices, method=method, min_w=min_w, max_w=max_w, return_performance=True,
        target_return=kwargs.get("target_return"),
        max_drawdown=kwargs.get("max_drawdown"),
    )

    # ── Hedge-asset retry ────────────────────────────────────────────────────
    # If a drawdown limit was requested but is still violated after the first
    # optimization pass, blend in defensive assets (BND, GLD) and re-run
    # min_vol.  This changes the ASSET UNIVERSE rather than just the weights.
    max_drawdown = kwargs.get("max_drawdown")
    if max_drawdown and not perf.get("drawdown_satisfied", True):
        _DEFENSIVE = ["BND", "GLD"]
        hedge = [a for a in _DEFENSIVE if a not in tickers]
        if hedge:
            try:
                tickers_hedged = tickers + hedge
                prices2 = get_prices(tickers_hedged, start=start, end=end, force_refresh=False)
                w_raw2, perf2 = optimize_core(
                    prices2,
                    method="min_vol",
                    min_w=0,
                    max_w=0.20,
                    return_performance=True,
                    max_drawdown=max_drawdown,
                )
                # Accept the hedged portfolio only if its drawdown is better
                dd_orig   = perf.get("max_drawdown", -1.0)
                dd_hedged = perf2.get("max_drawdown", -1.0)
                if dd_hedged > dd_orig:          # less negative → improvement
                    w_raw, perf = w_raw2, perf2
                    prices = prices2
                    perf["hedge_assets_added"] = hedge
                    logger.info(
                        "[Portfolio] Hedge retry improved MDD %.1f%% → %.1f%% (added %s)",
                        dd_orig * 100, dd_hedged * 100, hedge,
                    )
                else:
                    logger.info(
                        "[Portfolio] Hedge retry did NOT improve MDD (%.1f%% vs %.1f%%); keeping original",
                        dd_orig * 100, dd_hedged * 100,
                    )
            except Exception as _hedge_err:
                logger.warning("[Portfolio] Hedge retry failed: %s", _hedge_err)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Compute extras (Correlation, Stress Test, Benchmark) ─────────────────
    try:
        _extras = _compute_extras(w_raw, prices, perf)
        perf.update(_extras)
    except Exception as _ex_err:
        logger.warning(f"[Portfolio] Extras computation failed: {_ex_err}")
    # ─────────────────────────────────────────────────────────────────────────

    return w_raw, perf
