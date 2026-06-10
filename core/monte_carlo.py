"""
Portfolio-Level Monte Carlo Simulation + VaR — EisaX F4.
Uses Cholesky decomposition to preserve ticker correlations.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("eisax.monte_carlo")

_DEFAULT_ANNUAL_RETURN = 0.07
_DEFAULT_ANNUAL_VOL    = 0.22


def run_portfolio_monte_carlo(
    positions_df: pd.DataFrame,
    total_value: float,
    n_simulations: int = 5000,
    horizon_days: int = 252,
    var_confidence_levels: list[float] = [0.95, 0.99],
    loss_threshold_pct: float = 0.10,
    tickers: Optional[list[str]] = None,
    start_date: str = "2018-01-01",
    sample_paths: int = 200,
) -> dict:
    """
    Run portfolio-level Monte Carlo using correlated return paths.

    Args:
        positions_df: DataFrame from Portfolio.summary()["positions"].
                      Required columns: ticker, value.
        total_value: Current portfolio market value.
        n_simulations: Number of Monte Carlo paths.
        horizon_days: Projection horizon in trading days (252 = 1yr).
        var_confidence_levels: Confidence levels for VaR/CVaR calculation.
        loss_threshold_pct: Threshold for P(loss > X%) metric.
        tickers: Override ticker list; derived from positions_df if None.
        start_date: Historical start date for return distribution.
        sample_paths: Number of paths to return for chart (subset of n_simulations).

    Returns:
        {
            "var": {0.95: float, 0.99: float},
            "cvar": {0.95: float, 0.99: float},
            "prob_loss_gt_threshold": float,
            "best_outcome": float,   # P90 terminal value
            "worst_outcome": float,  # P10 terminal value
            "median_outcome": float,
            "mean_outcome": float,
            "terminal_distribution": np.ndarray,  # (n_simulations,)
            "paths_sample": np.ndarray,            # (horizon_days+1, sample_paths)
            "inputs": dict,
            "error": str | None,
        }
    """
    if tickers is None:
        tickers = [str(r.get("ticker", "")) for _, r in positions_df.iterrows() if r.get("ticker")]

    tickers = [t for t in tickers if t]

    # ── Compute position weights ───────────────────────────────────────────────
    weight_map: dict[str, float] = {}
    for _, pos in positions_df.iterrows():
        t = str(pos.get("ticker", ""))
        v = float(pos.get("value", 0) or 0)
        if t and total_value > 0:
            weight_map[t] = v / total_value

    # ── Fetch price history ────────────────────────────────────────────────────
    try:
        from core.data import get_prices, to_returns
        prices = get_prices(tickers, start=start_date)
    except Exception as e:
        log.warning(f"[monte_carlo] price fetch failed: {e} — using default assumptions")
        prices = pd.DataFrame()

    # ── Build return matrix (daily log returns) ────────────────────────────────
    use_tickers = tickers
    if not prices.empty:
        try:
            from core.data import to_returns
            rets = to_returns(prices, log=True)
            valid_cols = [t for t in use_tickers if t in rets.columns]
            if not valid_cols:
                valid_cols = [c for c in rets.columns if any(t.split(".")[0] in c for t in use_tickers)]
            use_tickers = valid_cols if valid_cols else use_tickers
            rets = rets[use_tickers].dropna() if use_tickers and all(t in rets.columns for t in use_tickers) else pd.DataFrame()
        except Exception as e:
            log.warning(f"[monte_carlo] returns computation failed: {e}")
            rets = pd.DataFrame()
    else:
        rets = pd.DataFrame()

    # ── Derive stats — fallback to defaults if insufficient data ──────────────
    n_assets = len(use_tickers)
    if not rets.empty and len(rets) >= 30 and n_assets > 0:
        means = rets.mean().values                            # daily mean returns
        cov   = rets.cov().values                             # daily covariance matrix
    else:
        log.warning("[monte_carlo] insufficient history — using default return/vol assumptions")
        means = np.full(max(n_assets, 1), _DEFAULT_ANNUAL_RETURN / 252)
        vol   = _DEFAULT_ANNUAL_VOL / np.sqrt(252)
        cov   = np.eye(max(n_assets, 1)) * (vol ** 2)
        if n_assets == 0:
            n_assets  = 1
            use_tickers = ["Portfolio"]
            weight_map  = {"Portfolio": 1.0}

    # ── Cholesky decomposition (regularised for near-singular matrices) ────────
    reg = 1e-8 * np.eye(n_assets)
    try:
        L = np.linalg.cholesky(cov + reg)
    except np.linalg.LinAlgError:
        # Further regularise if needed
        reg = 1e-5 * np.eye(n_assets)
        try:
            L = np.linalg.cholesky(cov + reg)
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.diag(cov) + 1e-5))

    # ── Build portfolio weight vector ──────────────────────────────────────────
    weights = np.array([weight_map.get(t, 1.0 / n_assets) for t in use_tickers])
    weights = weights / weights.sum()   # re-normalise

    # ── Simulate paths ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    # Shape: (n_simulations, horizon_days, n_assets)
    Z = rng.standard_normal((n_simulations, horizon_days, n_assets))
    # Correlated returns: (n_sim, T, n_assets)
    corr_rets = Z @ L.T + means   # broadcast means over time axis

    # Cumulative log returns → portfolio path values
    cum_log_rets = np.cumsum(corr_rets, axis=1)                  # (n_sim, T, n_assets)
    asset_paths  = np.exp(cum_log_rets)                           # relative to 1.0
    ptf_paths    = asset_paths @ weights                           # (n_sim, T)

    # Prepend t=0 (value = 1.0)
    ones      = np.ones((n_simulations, 1))
    ptf_paths = np.concatenate([ones, ptf_paths], axis=1)         # (n_sim, T+1)

    # Terminal portfolio values
    terminal_relative = ptf_paths[:, -1]                          # (n_sim,)
    terminal_values   = terminal_relative * total_value

    # ── VaR / CVaR on terminal portfolio returns ───────────────────────────────
    terminal_returns = terminal_relative - 1.0                    # portfolio returns

    var_results:  dict[float, float] = {}
    cvar_results: dict[float, float] = {}
    for cl in var_confidence_levels:
        var_pct  = float(np.percentile(terminal_returns, (1 - cl) * 100))
        cvar_pct = float(terminal_returns[terminal_returns <= var_pct].mean())
        var_results[cl]  = round(var_pct * 100, 2)
        cvar_results[cl] = round(cvar_pct * 100, 2)

    prob_loss = float((terminal_returns < -loss_threshold_pct).mean()) * 100

    # ── Percentile outcomes ────────────────────────────────────────────────────
    best_outcome   = float(np.percentile(terminal_values, 90))
    worst_outcome  = float(np.percentile(terminal_values, 10))
    median_outcome = float(np.percentile(terminal_values, 50))
    mean_outcome   = float(terminal_values.mean())

    # ── Sample paths for chart ─────────────────────────────────────────────────
    idx      = rng.choice(n_simulations, size=min(sample_paths, n_simulations), replace=False)
    paths_s  = (ptf_paths[idx] * total_value).T    # (T+1, sample_paths)

    return {
        "var":                     var_results,
        "cvar":                    cvar_results,
        "prob_loss_gt_threshold":  round(prob_loss, 2),
        "best_outcome":            round(best_outcome, 0),
        "worst_outcome":           round(worst_outcome, 0),
        "median_outcome":          round(median_outcome, 0),
        "mean_outcome":            round(mean_outcome, 0),
        "terminal_distribution":   terminal_values,
        "paths_sample":            paths_s,
        "inputs": {
            "n_simulations":    n_simulations,
            "horizon_days":     horizon_days,
            "tickers":          use_tickers,
            "weights":          dict(zip(use_tickers, weights.tolist())),
            "loss_threshold":   loss_threshold_pct,
            "total_value":      total_value,
        },
        "error": None,
    }
