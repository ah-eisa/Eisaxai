"""
Portfolio Optimization — EisaX.
Modern Portfolio Theory (Markowitz) — Sharpe maximization, minimum variance,
and efficient frontier construction.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

log = logging.getLogger("eisax.portfolio_optimizer")

_TRADING_DAYS = 252
_DEFAULT_RF   = 0.04   # 4% risk-free rate (close to US Treasuries / SAMA rate)


def _portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """Return (annualized_return, annualized_volatility) for given weights."""
    ann_ret = float(np.dot(weights, mean_returns) * _TRADING_DAYS)
    ann_vol = float(np.sqrt(weights.T @ cov @ weights) * np.sqrt(_TRADING_DAYS))
    return ann_ret, ann_vol


def _negative_sharpe(weights, mean_returns, cov, rf):
    ret, vol = _portfolio_stats(weights, mean_returns, cov)
    return -(ret - rf) / vol if vol > 0 else 1e9


def _portfolio_variance(weights, mean_returns, cov):
    return weights.T @ cov @ weights


def optimize_portfolio(
    positions_df: pd.DataFrame,
    objective: str = "max_sharpe",
    target_return: Optional[float] = None,
    risk_free_rate: float = _DEFAULT_RF,
    allow_short: bool = False,
    max_weight: float = 0.40,
    tickers: Optional[list[str]] = None,
    start_date: str = "2018-01-01",
) -> dict:
    """
    Find optimal portfolio weights.

    Args:
        positions_df: DataFrame from Portfolio.summary()["positions"].
                      Required columns: ticker, value.
        objective: "max_sharpe" | "min_variance" | "target_return"
        target_return: Required for objective="target_return" (annualized decimal).
        risk_free_rate: Annual risk-free rate (default 4%).
        allow_short: If True, allows negative weights.
        max_weight: Per-position weight cap (0-1).
        tickers: Override ticker list; derived from positions_df if None.
        start_date: Historical start for return statistics.

    Returns:
        {
            "tickers": list[str],
            "current_weights": dict[str, float],
            "optimal_weights": dict[str, float],
            "current_stats": {"return": float, "volatility": float, "sharpe": float},
            "optimal_stats":  {"return": float, "volatility": float, "sharpe": float},
            "rebalance_actions": pd.DataFrame,
            "objective": str,
            "error": str | None,
        }
    """
    if tickers is None:
        tickers = [str(r.get("ticker", "")) for _, r in positions_df.iterrows() if r.get("ticker")]
    tickers = [t for t in tickers if t]

    if len(tickers) < 2:
        return {
            "error":   "Need at least 2 holdings to optimize",
            "tickers": tickers,
            "current_weights": {}, "optimal_weights": {},
            "current_stats":   {}, "optimal_stats":   {},
            "rebalance_actions": pd.DataFrame(),
            "objective": objective,
        }

    # ── Fetch returns ──────────────────────────────────────────────────────────
    try:
        from core.data import get_prices, to_returns
        prices = get_prices(tickers, start=start_date)
        if prices.empty:
            raise ValueError("price fetch returned empty")
        rets = to_returns(prices, log=False).dropna()
        valid_tickers = [t for t in tickers if t in rets.columns]
        if len(valid_tickers) < 2:
            return {
                "error":   f"Insufficient price history for {tickers}",
                "tickers": tickers,
                "current_weights": {}, "optimal_weights": {},
                "current_stats":   {}, "optimal_stats":   {},
                "rebalance_actions": pd.DataFrame(),
                "objective": objective,
            }
        rets = rets[valid_tickers]
        tickers = valid_tickers
    except Exception as e:
        return {
            "error": f"Data fetch failed: {e}",
            "tickers": tickers,
            "current_weights": {}, "optimal_weights": {},
            "current_stats":   {}, "optimal_stats":   {},
            "rebalance_actions": pd.DataFrame(),
            "objective": objective,
        }

    mean_returns = rets.mean().values
    cov          = rets.cov().values
    n            = len(tickers)

    # ── Current weights ────────────────────────────────────────────────────────
    cur_w_map = {}
    total_val = float(positions_df["value"].sum()) if "value" in positions_df.columns else 0.0
    for _, pos in positions_df.iterrows():
        t = str(pos.get("ticker", ""))
        v = float(pos.get("value", 0) or 0)
        if t in tickers and total_val > 0:
            cur_w_map[t] = v / total_val
    cur_weights = np.array([cur_w_map.get(t, 0.0) for t in tickers])
    if cur_weights.sum() > 0:
        cur_weights = cur_weights / cur_weights.sum()
    else:
        cur_weights = np.ones(n) / n

    cur_ret, cur_vol = _portfolio_stats(cur_weights, mean_returns, cov)
    cur_sharpe = (cur_ret - risk_free_rate) / cur_vol if cur_vol > 0 else 0.0

    # ── Optimization setup ─────────────────────────────────────────────────────
    # Auto-relax max_weight when it would make the simplex constraint infeasible.
    # If n * max_weight < 1, the upper bounds can't sum to 1 ⇒ no feasible solution.
    # Bump max_weight to at least 1/n (equal-weight floor) so the optimizer
    # always has a feasible region.
    effective_max = max(max_weight, 1.0 / n + 0.01) if n > 0 else max_weight
    bound_low  = -effective_max if allow_short else 0.0
    bound_high = effective_max
    bounds     = [(bound_low, bound_high)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.ones(n) / n

    # Helper: try SLSQP first, fall back to trust-constr, then to multi-start.
    # SLSQP can choke on poorly-scaled problems ("Positive directional derivative
    # for linesearch") on small portfolios (n<3). Multi-start with biased x0s
    # picks the best feasible solution that converges.
    def _solve(fn, args, cons):
        r = minimize(fn, x0, args=args, method="SLSQP",
                     bounds=bounds, constraints=cons,
                     options={"maxiter": 200, "ftol": 1e-8})
        if r.success:
            return r
        r2 = minimize(fn, x0, args=args, method="trust-constr",
                      bounds=bounds, constraints=cons,
                      options={"maxiter": 500, "xtol": 1e-8})
        if r2.success:
            return r2
        # Multi-start fallback — try corner & biased weights
        candidates = []
        for seed in range(n):
            x_try = np.ones(n) * (0.1 / max(n - 1, 1))
            x_try[seed] = 1.0 - x_try.sum() + x_try[seed]
            x_try = np.clip(x_try, bound_low, bound_high)
            x_try = x_try / x_try.sum() if x_try.sum() > 0 else np.ones(n) / n
            rt = minimize(fn, x_try, args=args, method="SLSQP",
                          bounds=bounds, constraints=cons,
                          options={"maxiter": 200, "ftol": 1e-8})
            if rt.success:
                candidates.append(rt)
        if candidates:
            return min(candidates, key=lambda x: x.fun)
        return r

    if objective == "max_sharpe":
        result = _solve(_negative_sharpe,
                        (mean_returns, cov, risk_free_rate),
                        constraints)
    elif objective == "min_variance":
        result = _solve(_portfolio_variance, (mean_returns, cov), constraints)
    elif objective == "target_return":
        if target_return is None:
            return {"error": "target_return required for target_return objective",
                    "tickers": tickers,
                    "current_weights": {}, "optimal_weights": {},
                    "current_stats":   {}, "optimal_stats":   {},
                    "rebalance_actions": pd.DataFrame(),
                    "objective": objective}
        daily_target = target_return / _TRADING_DAYS
        target_constraints = constraints + [
            {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - daily_target}
        ]
        result = _solve(_portfolio_variance, (mean_returns, cov), target_constraints)
    else:
        return {"error": f"Unknown objective: {objective}",
                "tickers": tickers,
                "current_weights": {}, "optimal_weights": {},
                "current_stats":   {}, "optimal_stats":   {},
                "rebalance_actions": pd.DataFrame(),
                "objective": objective}

    if not result.success:
        return {"error": f"Optimization failed: {result.message}",
                "tickers": tickers,
                "current_weights": dict(zip(tickers, cur_weights.tolist())),
                "optimal_weights": {}, "current_stats": {}, "optimal_stats": {},
                "rebalance_actions": pd.DataFrame(),
                "objective": objective}

    opt_w = result.x
    opt_ret, opt_vol = _portfolio_stats(opt_w, mean_returns, cov)
    opt_sharpe = (opt_ret - risk_free_rate) / opt_vol if opt_vol > 0 else 0.0

    # ── Rebalance actions ──────────────────────────────────────────────────────
    rebal_rows = []
    for i, t in enumerate(tickers):
        cur_pct = cur_weights[i] * 100
        opt_pct = opt_w[i] * 100
        delta_pct = opt_pct - cur_pct
        delta_val = (opt_w[i] - cur_weights[i]) * total_val
        if abs(delta_pct) < 0.5:
            action = "✅ Hold"
        elif delta_pct > 0:
            action = f"⬆️ Add"
        else:
            action = f"⬇️ Reduce"
        rebal_rows.append({
            "Ticker":         t,
            "Current %":      round(cur_pct, 1),
            "Optimal %":      round(opt_pct, 1),
            "Delta %":        round(delta_pct, 1),
            "Delta Value":    round(delta_val, 0),
            "Action":         action,
        })

    return {
        "tickers":         tickers,
        "current_weights": dict(zip(tickers, [round(w, 4) for w in cur_weights.tolist()])),
        "optimal_weights": dict(zip(tickers, [round(w, 4) for w in opt_w.tolist()])),
        "current_stats": {
            "return":     round(cur_ret * 100, 2),
            "volatility": round(cur_vol * 100, 2),
            "sharpe":     round(cur_sharpe, 3),
        },
        "optimal_stats": {
            "return":     round(opt_ret * 100, 2),
            "volatility": round(opt_vol * 100, 2),
            "sharpe":     round(opt_sharpe, 3),
        },
        "rebalance_actions": pd.DataFrame(rebal_rows),
        "objective":         objective,
        "improvement": {
            "return_lift":     round((opt_ret - cur_ret) * 100, 2),
            "vol_change":      round((opt_vol - cur_vol) * 100, 2),
            "sharpe_lift":     round(opt_sharpe - cur_sharpe, 3),
        },
        "error": None,
    }


def efficient_frontier(
    positions_df: pd.DataFrame,
    n_points: int = 30,
    risk_free_rate: float = _DEFAULT_RF,
    max_weight: float = 0.40,
    tickers: Optional[list[str]] = None,
    start_date: str = "2018-01-01",
) -> dict:
    """
    Generate efficient frontier curve.

    Returns:
        {
            "frontier": pd.DataFrame with columns: return_pct, volatility_pct, sharpe
            "max_sharpe_point":   {"return": float, "volatility": float, "sharpe": float, "weights": dict}
            "min_variance_point": {...}
            "current_point":      {"return": float, "volatility": float, "sharpe": float}
            "tickers":            list[str]
            "error":              str | None
        }
    """
    # Get max sharpe and min variance optimums
    max_s = optimize_portfolio(positions_df, objective="max_sharpe",
                               risk_free_rate=risk_free_rate, max_weight=max_weight,
                               tickers=tickers, start_date=start_date)
    if max_s.get("error"):
        return {"error": max_s["error"], "frontier": pd.DataFrame(),
                "max_sharpe_point": {}, "min_variance_point": {}, "current_point": {},
                "tickers": tickers or []}

    min_v = optimize_portfolio(positions_df, objective="min_variance",
                               risk_free_rate=risk_free_rate, max_weight=max_weight,
                               tickers=tickers, start_date=start_date)

    if min_v.get("error"):
        return {"error": min_v["error"], "frontier": pd.DataFrame(),
                "max_sharpe_point": {}, "min_variance_point": {}, "current_point": {},
                "tickers": tickers or []}

    # Sweep target returns from min_var return to max_sharpe return
    min_ret = min_v["optimal_stats"]["return"] / 100
    max_ret = max_s["optimal_stats"]["return"] / 100

    if max_ret <= min_ret:
        target_returns = [min_ret]
    else:
        target_returns = np.linspace(min_ret, max_ret, n_points).tolist()

    frontier_rows = []
    for tr in target_returns:
        r = optimize_portfolio(
            positions_df, objective="target_return", target_return=tr,
            risk_free_rate=risk_free_rate, max_weight=max_weight,
            tickers=tickers, start_date=start_date,
        )
        if r.get("error"):
            continue
        s = r["optimal_stats"]
        frontier_rows.append({
            "return_pct":     s["return"],
            "volatility_pct": s["volatility"],
            "sharpe":         s["sharpe"],
        })

    return {
        "frontier":            pd.DataFrame(frontier_rows),
        "max_sharpe_point":    {
            **max_s["optimal_stats"],
            "weights": max_s["optimal_weights"],
        },
        "min_variance_point":  {
            **min_v["optimal_stats"],
            "weights": min_v["optimal_weights"],
        },
        "current_point":       max_s["current_stats"],
        "tickers":             max_s["tickers"],
        "error":               None,
    }
