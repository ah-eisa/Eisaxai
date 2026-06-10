"""
Forward-Looking Scenario Builder — EisaX F3.
Projects portfolio value at 3 / 6 / 12 months under a user-defined macro scenario.
Combines macro elasticity adjustments with historical base returns per ticker.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

from core.macro_simulator import MacroScenario, compute_sector_impacts
from core.macro_elasticities import MACRO_VAR_DEFAULTS

log = logging.getLogger("eisax.scenario_builder")

_DEFAULT_ANNUAL_RETURN = 0.07   # fallback when price history is unavailable
_DEFAULT_ANNUAL_VOL    = 0.25


def _fetch_base_returns(tickers: list[str], start: str = "2018-01-01") -> dict[str, float]:
    """
    Fetch annualized mean daily return per ticker from core.data.
    Falls back to _DEFAULT_ANNUAL_RETURN on any error.
    """
    base: dict[str, float] = {}
    try:
        from core.data import get_prices, to_returns
        prices = get_prices(tickers, start=start)
        if prices.empty:
            return {t: _DEFAULT_ANNUAL_RETURN for t in tickers}
        rets = to_returns(prices)
        for ticker in tickers:
            col = ticker if ticker in rets.columns else None
            if col is None:
                # try partial match
                matches = [c for c in rets.columns if ticker.split(".")[0] in c]
                col = matches[0] if matches else None
            if col is not None and not rets[col].dropna().empty:
                annual = float(rets[col].mean() * 252)
                base[ticker] = max(min(annual, 1.5), -0.9)   # clamp to [-90%, +150%]
            else:
                base[ticker] = _DEFAULT_ANNUAL_RETURN
    except Exception as e:
        log.warning(f"[scenario_builder] price fetch failed: {e}")
        for t in tickers:
            base.setdefault(t, _DEFAULT_ANNUAL_RETURN)
    return base


def build_forward_scenario(
    positions_df: pd.DataFrame,
    total_value: float,
    scenario: MacroScenario,
    baseline: Optional[MacroScenario] = None,
    horizons_months: list[int] = [3, 6, 12],
    tickers: Optional[list[str]] = None,
    start_date: str = "2018-01-01",
) -> dict:
    """
    Project portfolio value at each horizon under the given macro scenario.

    Method:
      1. Fetch historical annualized return per ticker.
      2. Compute sector-level macro adjustment from compute_sector_impacts().
      3. adjusted_return = base_return + sector_impact_pct / 100
      4. projected_value = current_value * (1 + adjusted_return) ^ (months / 12)

    Args:
        positions_df: DataFrame from Portfolio.summary()["positions"].
                      Required columns: ticker, name, sector, value.
        total_value: Current portfolio value.
        scenario: MacroScenario assumptions.
        baseline: Reference macro (defaults to MACRO_VAR_DEFAULTS).
        horizons_months: List of projection horizons in months.
        tickers: Override ticker list; derived from positions_df if None.
        start_date: Historical start for base return calculation.

    Returns:
        {
            "horizons": {
                3:  {"projected_value": float, "pct_change": float,
                     "position_projections": pd.DataFrame},
                6:  {...},
                12: {...},
            },
            "macro_adjustments": dict[str, float],
            "base_returns": dict[str, float],
            "scenario_label": str,
            "total_value": float,
        }
    """
    if baseline is None:
        baseline = MacroScenario.from_defaults()

    sector_impacts = compute_sector_impacts(scenario, baseline)

    if tickers is None:
        tickers = [str(r.get("ticker", "")) for _, r in positions_df.iterrows() if r.get("ticker")]

    base_returns = _fetch_base_returns(tickers, start=start_date)

    # Build per-position projections
    position_rows = []
    for _, pos in positions_df.iterrows():
        ticker   = str(pos.get("ticker", ""))
        name     = str(pos.get("name", ticker))
        sector   = str(pos.get("sector", "Unknown"))
        cur_val  = float(pos.get("value", 0) or 0)

        base_ret     = base_returns.get(ticker, _DEFAULT_ANNUAL_RETURN)
        macro_adj    = (sector_impacts.get(sector, sector_impacts.get("Unknown", 0.0))) / 100
        adj_return   = base_ret + macro_adj

        row = {
            "Ticker":               ticker,
            "Name":                 name,
            "Sector":               sector,
            "Current Value":        round(cur_val, 0),
            "Base Annual Return %": round(base_ret * 100, 2),
            "Macro Adjustment %":   round(macro_adj * 100, 2),
            "Adjusted Return %":    round(adj_return * 100, 2),
        }

        for h in horizons_months:
            years     = h / 12
            proj_val  = cur_val * ((1 + adj_return) ** years)
            row[f"Proj {h}m"] = round(proj_val, 0)

        position_rows.append(row)

    proj_df = pd.DataFrame(position_rows) if position_rows else pd.DataFrame()

    # Aggregate to portfolio level per horizon
    horizons_out: dict[int, dict] = {}
    for h in horizons_months:
        col = f"Proj {h}m"
        if not proj_df.empty and col in proj_df.columns:
            proj_val  = float(proj_df[col].sum())
        else:
            proj_val  = total_value
        pct_change = ((proj_val / total_value) - 1) * 100 if total_value else 0.0
        horizons_out[h] = {
            "projected_value":      round(proj_val, 0),
            "pct_change":           round(pct_change, 2),
            "position_projections": proj_df,
        }

    # Human-readable scenario label
    d = scenario.to_dict()
    b = baseline.to_dict()
    parts = []
    if abs(d["gdp_growth"] - b["gdp_growth"]) >= 0.5:
        parts.append(f"GDP {d['gdp_growth']:+.1f}%")
    if abs(d["inflation"] - b["inflation"]) >= 0.5:
        parts.append(f"Inf {d['inflation']:.1f}%")
    if abs(d["fed_rate"] - b["fed_rate"]) >= 0.25:
        parts.append(f"Rate {d['fed_rate']:.2f}%")
    if abs(d["oil_brent"] - b["oil_brent"]) >= 5:
        parts.append(f"Oil ${d['oil_brent']:.0f}")
    if abs(d["usd_index"] - b["usd_index"]) >= 2:
        parts.append(f"DXY {d['usd_index']:.0f}")
    scenario_label = ", ".join(parts) if parts else "Baseline scenario"

    return {
        "horizons":         horizons_out,
        "macro_adjustments": {k: round(v, 3) for k, v in sector_impacts.items()},
        "base_returns":      {k: round(v * 100, 2) for k, v in base_returns.items()},
        "scenario_label":    scenario_label,
        "total_value":       total_value,
    }
