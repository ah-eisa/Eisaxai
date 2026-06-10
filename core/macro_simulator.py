"""
Macroeconomic Simulation Engine — EisaX F1.
Maps macro variable changes to portfolio P&L via sector elasticities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from core.macro_elasticities import (
    SECTOR_ELASTICITIES,
    MACRO_VAR_DEFAULTS,
)

log = logging.getLogger("eisax.macro_simulator")


@dataclass
class MacroScenario:
    gdp_growth: float  # % annualized
    inflation:  float  # % YoY CPI
    fed_rate:   float  # % absolute fed funds level
    oil_brent:  float  # USD per barrel
    usd_index:  float  # DXY index points

    @classmethod
    def from_defaults(cls) -> "MacroScenario":
        d = MACRO_VAR_DEFAULTS
        return cls(
            gdp_growth=d["gdp_growth"],
            inflation=d["inflation"],
            fed_rate=d["fed_rate"],
            oil_brent=d["oil_brent"],
            usd_index=d["usd_index"],
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "gdp_growth": self.gdp_growth,
            "inflation":  self.inflation,
            "fed_rate":   self.fed_rate,
            "oil_brent":  self.oil_brent,
            "usd_index":  self.usd_index,
        }


def _get_elasticity(sector: str, variable: str) -> float:
    """Return elasticity for sector/variable, falling back to 'Unknown'."""
    row = SECTOR_ELASTICITIES.get(sector) or SECTOR_ELASTICITIES.get("Unknown", {})
    return row.get(variable, 0.0)


def compute_sector_impacts(
    scenario: MacroScenario,
    baseline: MacroScenario | None = None,
) -> dict[str, float]:
    """
    Compute expected return impact (%) per sector given macro scenario vs baseline.

    Oil delta is in $10 increments; USD delta is in 5-point DXY increments.
    All other variables: delta in percentage points.

    Returns:
        {sector_name: impact_pct}  — can be positive or negative
    """
    if baseline is None:
        baseline = MacroScenario.from_defaults()

    b = baseline.to_dict()
    s = scenario.to_dict()

    # Deltas — normalised to the elasticity unit
    deltas = {
        "gdp_growth": s["gdp_growth"] - b["gdp_growth"],
        "inflation":  s["inflation"]  - b["inflation"],
        "fed_rate":   s["fed_rate"]   - b["fed_rate"],
        "oil_brent":  (s["oil_brent"] - b["oil_brent"]) / 10.0,   # per $10
        "usd_index":  (s["usd_index"] - b["usd_index"]) / 5.0,    # per 5 pts
    }

    impacts: dict[str, float] = {}
    for sector in SECTOR_ELASTICITIES:
        total = sum(
            _get_elasticity(sector, var) * delta
            for var, delta in deltas.items()
        )
        impacts[sector] = round(total, 3)

    return impacts


def simulate_portfolio(
    positions_df: pd.DataFrame,
    total_value: float,
    scenario: MacroScenario,
    baseline: MacroScenario | None = None,
) -> dict:
    """
    Apply macro scenario to a portfolio and compute expected P&L.

    Args:
        positions_df: DataFrame from Portfolio.summary()["positions"].
                      Required columns: ticker, name, sector, value.
        total_value: Current total portfolio value.
        scenario: MacroScenario to evaluate.
        baseline: Reference macro state (defaults to MACRO_VAR_DEFAULTS).

    Returns:
        {
            "sector_impacts": dict[str, float],
            "position_impacts": pd.DataFrame,
            "total_impact_pct": float,
            "total_impact_value": float,
            "new_portfolio_value": float,
            "scenario": dict,
            "baseline": dict,
        }
    """
    if baseline is None:
        baseline = MacroScenario.from_defaults()

    sector_impacts = compute_sector_impacts(scenario, baseline)

    rows = []
    weighted_impact = 0.0

    for _, pos in positions_df.iterrows():
        ticker  = str(pos.get("ticker", ""))
        name    = str(pos.get("name", ticker))
        sector  = str(pos.get("sector", "Unknown"))
        value   = float(pos.get("value", 0) or 0)

        impact_pct = sector_impacts.get(sector, sector_impacts.get("Unknown", 0.0))
        impact_val = value * (impact_pct / 100)
        weight     = (value / total_value * 100) if total_value else 0
        weighted_impact += impact_val

        rows.append({
            "Ticker":       ticker,
            "Name":         name,
            "Sector":       sector,
            "Value":        round(value, 0),
            "Weight %":     round(weight, 1),
            "Impact %":     round(impact_pct, 2),
            "Impact Value": round(impact_val, 0),
            "New Value":    round(value + impact_val, 0),
        })

    position_impacts = pd.DataFrame(rows).sort_values("Impact Value") if rows else pd.DataFrame()
    total_impact_pct = (weighted_impact / total_value * 100) if total_value else 0.0

    return {
        "sector_impacts":      {k: round(v, 3) for k, v in sector_impacts.items()},
        "position_impacts":    position_impacts,
        "total_impact_pct":    round(total_impact_pct, 2),
        "total_impact_value":  round(weighted_impact, 0),
        "new_portfolio_value": round(total_value + weighted_impact, 0),
        "scenario":            scenario.to_dict(),
        "baseline":            baseline.to_dict(),
    }
