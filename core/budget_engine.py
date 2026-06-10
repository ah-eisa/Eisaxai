"""
Budget Assumptions Engine — EisaX F2.
Given a total investment budget and target sector weights, computes
exact share quantities to buy/sell per ticker.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import pandas as pd

log = logging.getLogger("eisax.budget_engine")


def compute_budget_allocation(
    total_budget: float,
    target_sector_weights: dict[str, float],
    positions_df: pd.DataFrame,
    total_value: float,
    current_prices: Optional[dict[str, float]] = None,
) -> dict:
    """
    Compute buy/sell actions to reach target sector allocation within a budget.

    Args:
        total_budget: Cash available to deploy (can be 0 for rebalance-only).
        target_sector_weights: {"Finance": 40, "Energy": 30, ...} — should sum ~100.
        positions_df: DataFrame from Portfolio.summary()["positions"].
                      Required columns: ticker, name, sector, value, price, qty.
        total_value: Current portfolio market value (positions only, no cash).
        current_prices: Optional {ticker: price} override. If None, uses positions_df["price"].

    Returns:
        {
            "allocations": pd.DataFrame,
            "total_buy_cost": float,
            "total_sell_proceeds": float,
            "net_cash_required": float,
            "remaining_cash": float,
            "allocation_drift_pct": float,
            "feasible": bool,
            "warnings": list[str],
            "summary_weight_check": float,  # sum of target weights
        }
    """
    warnings: list[str] = []

    # ── Validate inputs ────────────────────────────────────────────────────────
    weight_sum = sum(target_sector_weights.values())
    if abs(weight_sum - 100) > 5:
        warnings.append(f"Target weights sum to {weight_sum:.1f}% (expected ~100%)")

    if positions_df.empty or total_value <= 0:
        warnings.append("No positions found — budget plan is for fresh allocation only")

    # ── Total portfolio basis = existing + new budget ─────────────────────────
    total_basis = total_value + total_budget

    # ── Build per-position lookup ──────────────────────────────────────────────
    rows = []
    for _, pos in positions_df.iterrows():
        ticker  = str(pos.get("ticker", ""))
        name    = str(pos.get("name", ticker))
        sector  = str(pos.get("sector", "Unknown"))
        cur_val = float(pos.get("value", 0) or 0)
        price   = float((current_prices or {}).get(ticker) or pos.get("price") or 0)
        qty     = float(pos.get("qty", 0) or 0)

        rows.append({
            "ticker":          ticker,
            "name":            name,
            "sector":          sector,
            "current_value":   cur_val,
            "current_weight":  (cur_val / total_basis * 100) if total_basis else 0,
            "price":           price,
            "qty":             qty,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "ticker", "name", "sector", "current_value", "current_weight", "price", "qty"
    ])

    # ── Compute sector-level targets ───────────────────────────────────────────
    sector_target_value: dict[str, float] = {
        s: (w / 100) * total_basis
        for s, w in target_sector_weights.items()
    }

    # ── Distribute target within sector proportional to current holdings ───────
    # If no current holdings in a sector, warn and skip that sector
    sector_groups = df.groupby("sector") if not df.empty else {}

    allocation_rows = []

    # Sectors with existing holdings
    processed_sectors: set[str] = set()
    if not df.empty:
        for sector, grp in df.groupby("sector"):
            processed_sectors.add(sector)
            target_val = sector_target_value.get(sector, 0.0)
            sector_cur_val = grp["current_value"].sum()

            # Distribute target proportionally within sector
            for _, row in grp.iterrows():
                ticker    = row["ticker"]
                price     = row["price"]
                cur_val   = row["current_value"]
                cur_qty   = row["qty"]

                if sector_cur_val > 0:
                    pos_frac   = cur_val / sector_cur_val
                else:
                    pos_frac   = 1.0 / max(len(grp), 1)

                pos_target_val = target_val * pos_frac
                delta_val      = pos_target_val - cur_val

                if price > 0:
                    if delta_val >= price:
                        shares_to_buy  = math.floor(delta_val / price)
                        shares_to_sell = 0
                        action         = "BUY"
                        est_cost       = shares_to_buy * price
                    elif delta_val <= -price:
                        shares_to_buy  = 0
                        shares_to_sell = math.floor(abs(delta_val) / price)
                        action         = "SELL"
                        est_cost       = -shares_to_sell * price
                    else:
                        shares_to_buy  = 0
                        shares_to_sell = 0
                        action         = "HOLD"
                        est_cost       = 0.0
                else:
                    shares_to_buy  = 0
                    shares_to_sell = 0
                    action         = "HOLD (no price)"
                    est_cost       = 0.0
                    warnings.append(f"{ticker}: price unavailable — skipped")

                target_weight = (pos_target_val / total_basis * 100) if total_basis else 0

                allocation_rows.append({
                    "Ticker":            ticker,
                    "Name":              row["name"],
                    "Sector":            sector,
                    "Current Value":     round(cur_val, 0),
                    "Current Weight %":  round(row["current_weight"], 1),
                    "Target Weight %":   round(target_weight, 1),
                    "Target Value":      round(pos_target_val, 0),
                    "Delta Value":       round(delta_val, 0),
                    "Price":             round(price, 2),
                    "Shares to Buy":     shares_to_buy,
                    "Shares to Sell":    shares_to_sell,
                    "Est. Cost":         round(est_cost, 0),
                    "Action":            action,
                })

    # Sectors in target but NOT in portfolio — warn
    for sector, target_val in sector_target_value.items():
        if sector not in processed_sectors and target_val > 0:
            warnings.append(
                f"Sector '{sector}' has target {target_val:,.0f} "
                f"but no positions held — add tickers manually"
            )

    alloc_df = pd.DataFrame(allocation_rows) if allocation_rows else pd.DataFrame()

    # ── Aggregate costs ────────────────────────────────────────────────────────
    total_buy_cost      = alloc_df["Est. Cost"].clip(lower=0).sum() if not alloc_df.empty else 0.0
    total_sell_proceeds = alloc_df["Est. Cost"].clip(upper=0).abs().sum() if not alloc_df.empty else 0.0
    net_cash_required   = total_buy_cost - total_sell_proceeds
    remaining_cash      = total_budget - net_cash_required

    if remaining_cash < 0:
        warnings.append(
            f"Budget deficit: need {abs(remaining_cash):,.0f} more cash to fully execute plan"
        )

    # ── Allocation drift after trades ─────────────────────────────────────────
    drift = 0.0
    if not alloc_df.empty and total_basis > 0:
        alloc_df["Post-Trade Value"] = (
            alloc_df["Current Value"]
            + alloc_df["Shares to Buy"] * alloc_df["Price"]
            - alloc_df["Shares to Sell"] * alloc_df["Price"]
        )
        alloc_df["Post-Trade Weight %"] = alloc_df["Post-Trade Value"] / total_basis * 100
        drift = float((alloc_df["Post-Trade Weight %"] - alloc_df["Target Weight %"]).abs().max())

    return {
        "allocations":          alloc_df,
        "total_buy_cost":       round(total_buy_cost, 0),
        "total_sell_proceeds":  round(total_sell_proceeds, 0),
        "net_cash_required":    round(net_cash_required, 0),
        "remaining_cash":       round(remaining_cash, 0),
        "allocation_drift_pct": round(drift, 2),
        "feasible":             remaining_cash >= 0,
        "warnings":             warnings,
        "summary_weight_check": round(weight_sum, 1),
        "total_basis":          round(total_basis, 0),
    }
