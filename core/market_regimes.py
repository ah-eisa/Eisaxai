"""
Multi-Market Regime Comparison — EisaX F5.
Classifies historical periods into bull/bear/sideways/stagflation regimes
and projects portfolio performance under each.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from core.macro_elasticities import (
    REGIME_MACRO_PROFILES,
    REGIME_LABELS,
    MACRO_VAR_DEFAULTS,
)
from core.macro_simulator import MacroScenario, simulate_portfolio

log = logging.getLogger("eisax.market_regimes")

# Market proxy tickers for regime classification
_SPY  = "SPY"      # US equity (global proxy)
_VIX  = "^VIX"
_OIL  = "BZ=F"    # Brent crude
_BOND = "TLT"     # long-duration bonds (inflation proxy via yield)

_DEFAULT_REGIME_RETURNS: dict[str, float] = {
    "bull":        0.22,
    "sideways":    0.04,
    "bear":       -0.18,
    "stagflation": -0.08,
}


def _classify_regime_for_period(
    spy_ret_12m: float,
    vix_avg: float,
    inflation_est: float,
) -> str:
    """Classify a rolling 12-month window into one of 4 regimes."""
    if inflation_est > 5.0 and spy_ret_12m < 0:
        return "stagflation"
    if spy_ret_12m > 0.15 and vix_avg < 20:
        return "bull"
    if spy_ret_12m < -0.10 and vix_avg > 25:
        return "bear"
    return "sideways"


def get_regime_historical_returns(
    tickers: list[str],
    start_date: str = "2015-01-01",
) -> dict[str, dict[str, float]]:
    """
    Classify historical monthly windows into regimes and compute
    average annualized ticker return within each regime.

    Returns:
        {regime: {ticker: annualized_mean_return}}
        Falls back to _DEFAULT_REGIME_RETURNS on data failure.
    """
    try:
        import yfinance as yf
        from core.data import get_prices, to_returns

        # Fetch SPY + VIX for regime classification
        proxy_prices = yf.download(
            [_SPY, _VIX, _OIL],
            start=start_date,
            progress=False,
            auto_adjust=True,
        )
        if proxy_prices.empty:
            raise ValueError("proxy price fetch returned empty")

        # Handle multi-level columns
        if isinstance(proxy_prices.columns, pd.MultiIndex):
            spy_close  = proxy_prices["Close"][_SPY].dropna()
            vix_close  = proxy_prices["Close"][_VIX].dropna()
        else:
            spy_close  = proxy_prices["Close"].dropna()
            vix_close  = pd.Series(dtype=float)

        if spy_close.empty:
            raise ValueError("SPY data empty")

        # Resample to monthly
        spy_monthly = spy_close.resample("ME").last()
        vix_monthly = vix_close.resample("ME").mean() if not vix_close.empty else pd.Series(dtype=float)

        # Rolling 12-month SPY return
        spy_ret_12m = spy_monthly.pct_change(12)

        # Use TLT yield delta as rough inflation proxy (inverted — falling bond = rising inflation)
        try:
            tlt_prices   = yf.download(_BOND, start=start_date, progress=False, auto_adjust=True)
            tlt_monthly  = (tlt_prices["Close"] if not isinstance(tlt_prices.columns, pd.MultiIndex)
                           else tlt_prices["Close"][_BOND]).resample("ME").last()
            tlt_ret_12m  = tlt_monthly.pct_change(12)
            inflation_est = (-tlt_ret_12m * 8).fillna(3.0)  # rough mapping
        except Exception:
            inflation_est = pd.Series(3.0, index=spy_ret_12m.index)

        # Classify each month
        regime_labels: dict[str, str] = {}
        for dt in spy_ret_12m.index:
            if pd.isna(spy_ret_12m.get(dt)):
                continue
            vix_v = float(vix_monthly.get(dt, 20)) if not vix_monthly.empty else 20.0
            inf_v = float(inflation_est.get(dt, 3.0)) if hasattr(inflation_est, "get") else 3.0
            regime_labels[str(dt.date())] = _classify_regime_for_period(
                float(spy_ret_12m[dt]), vix_v, inf_v
            )

        # Fetch portfolio ticker returns
        ticker_prices = get_prices(tickers, start=start_date)
        if ticker_prices.empty:
            raise ValueError("ticker prices empty")

        daily_rets = ticker_prices.pct_change().dropna()
        monthly_rets = daily_rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)

        # Group ticker returns by regime
        regime_bucket: dict[str, list[float]] = {r: [] for r in ["bull", "bear", "sideways", "stagflation"]}
        ticker_regime_rets: dict[str, dict[str, list[float]]] = {
            t: {r: [] for r in regime_bucket} for t in tickers
        }

        for dt, regime in regime_labels.items():
            try:
                m_dt = pd.Timestamp(dt)
                # find matching row in monthly_rets
                matches = monthly_rets.index[monthly_rets.index.month == m_dt.month and
                                              monthly_rets.index.year == m_dt.year]
                if matches.empty:
                    continue
                row = monthly_rets.loc[matches[0]]
            except Exception:
                continue
            for ticker in tickers:
                col = ticker if ticker in row.index else None
                if col is None:
                    col_matches = [c for c in row.index if ticker.split(".")[0] in c]
                    col = col_matches[0] if col_matches else None
                if col and not pd.isna(row[col]):
                    ticker_regime_rets[ticker][regime].append(float(row[col]))

        result: dict[str, dict[str, float]] = {}
        for regime in ["bull", "bear", "sideways", "stagflation"]:
            result[regime] = {}
            for ticker in tickers:
                monthly_list = ticker_regime_rets[ticker][regime]
                if monthly_list:
                    ann_ret = float(np.mean(monthly_list)) * 12
                    result[regime][ticker] = round(max(min(ann_ret, 2.0), -0.95), 4)
                else:
                    result[regime][ticker] = _DEFAULT_REGIME_RETURNS[regime]

        return result

    except Exception as e:
        log.warning(f"[market_regimes] regime classification failed: {e} — using defaults")
        return {
            regime: {t: _DEFAULT_REGIME_RETURNS[regime] for t in tickers}
            for regime in ["bull", "bear", "sideways", "stagflation"]
        }


def compare_regimes(
    positions_df: pd.DataFrame,
    total_value: float,
    horizon_months: int = 12,
    tickers: Optional[list[str]] = None,
    start_date: str = "2015-01-01",
) -> dict:
    """
    Project portfolio value under each of 4 market regimes.

    For each regime:
      - Use REGIME_MACRO_PROFILES to get macro scenario
      - Compute sector-level macro adjustments via simulate_portfolio()
      - Blend with historical regime return data (50/50 weight)
      - Project portfolio value at horizon_months

    Args:
        positions_df: DataFrame from Portfolio.summary()["positions"].
        total_value: Current portfolio value.
        horizon_months: Projection horizon in months.
        tickers: Override ticker list; derived from positions_df if None.
        start_date: Historical start for regime classification.

    Returns:
        {
            "regimes": {
                "bull": {
                    "projected_value": float,
                    "expected_return_pct": float,
                    "macro_profile": dict,
                    "sector_impacts": dict,
                    "position_breakdown": pd.DataFrame,
                    "historical_base_return_pct": float,
                    "label_ar": str,
                    "label_en": str,
                },
                ...
            },
            "best_regime": str,
            "worst_regime": str,
            "regime_spread_pct": float,
            "horizon_months": int,
        }
    """
    if tickers is None:
        tickers = [str(r.get("ticker", "")) for _, r in positions_df.iterrows() if r.get("ticker")]
    tickers = [t for t in tickers if t]

    hist_returns = get_regime_historical_returns(tickers, start_date=start_date)

    years = horizon_months / 12
    regime_results: dict[str, dict] = {}

    for regime in ["bull", "bear", "sideways", "stagflation"]:
        macro_profile = REGIME_MACRO_PROFILES[regime]
        scenario = MacroScenario(
            gdp_growth=macro_profile["gdp_growth"],
            inflation=macro_profile["inflation"],
            fed_rate=macro_profile["fed_rate"],
            oil_brent=macro_profile["oil_brent"],
            usd_index=macro_profile["usd_index"],
        )

        sim = simulate_portfolio(positions_df, total_value, scenario)
        macro_return_pct = sim["total_impact_pct"]  # elasticity-based

        # Historical base return for this regime (portfolio-weighted average)
        ticker_hist = hist_returns.get(regime, {})
        weighted_hist = 0.0
        total_w = 0.0
        for _, pos in positions_df.iterrows():
            t = str(pos.get("ticker", ""))
            v = float(pos.get("value", 0) or 0)
            if t and total_value > 0:
                w = v / total_value
                hist_ret = ticker_hist.get(t, _DEFAULT_REGIME_RETURNS[regime])
                weighted_hist += w * hist_ret * 100   # annualized %
                total_w += w

        historical_base_pct = weighted_hist / total_w if total_w > 0 else _DEFAULT_REGIME_RETURNS[regime] * 100

        # Blend: 50% elasticity model + 50% historical
        blended_annual_pct = 0.5 * macro_return_pct + 0.5 * historical_base_pct
        projected_value    = total_value * ((1 + blended_annual_pct / 100) ** years)

        label_ar, label_en = REGIME_LABELS.get(regime, (regime, regime))

        regime_results[regime] = {
            "projected_value":           round(projected_value, 0),
            "expected_return_pct":       round(blended_annual_pct * years, 2),
            "macro_profile":             macro_profile,
            "sector_impacts":            sim["sector_impacts"],
            "position_breakdown":        sim["position_impacts"],
            "historical_base_return_pct": round(historical_base_pct, 2),
            "macro_elasticity_return_pct": round(macro_return_pct, 2),
            "label_ar":                  label_ar,
            "label_en":                  label_en,
        }

    sorted_by_return = sorted(
        regime_results.items(),
        key=lambda x: x[1]["expected_return_pct"],
        reverse=True,
    )
    best_regime  = sorted_by_return[0][0]
    worst_regime = sorted_by_return[-1][0]
    spread = (
        regime_results[best_regime]["expected_return_pct"]
        - regime_results[worst_regime]["expected_return_pct"]
    )

    return {
        "regimes":           regime_results,
        "best_regime":       best_regime,
        "worst_regime":      worst_regime,
        "regime_spread_pct": round(spread, 2),
        "horizon_months":    horizon_months,
        "total_value":       total_value,
    }
