from __future__ import annotations

import re
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return, ema_historical_return
from pypfopt.risk_models import CovarianceShrinkage

# Minimum expected return floor — prevents optimizer from producing portfolios
# that are mathematically guaranteed to lose money due to short historical windows.
_MU_FLOOR = 0.03  # 3% — below risk-free but avoids catastrophic negative outputs


def _clean_weights(weights: dict) -> dict[str, float]:
    w = {str(k).upper().strip(): float(v) for k, v in (weights or {}).items()}
    w = {k: v for k, v in w.items() if v > 0}
    s = sum(w.values())
    if s > 0:
        w = {k: v / s for k, v in w.items()}
    return w


def _estimate_mu(prices: pd.DataFrame) -> pd.Series:
    """
    Robust expected return estimator:
    1. Compute both full-window mean and 6-month EMA
    2. Take the higher of the two (optimistic but grounded)
    3. Apply a floor of _MU_FLOOR to prevent catastrophic negatives
    """
    mu_mean = mean_historical_return(prices, frequency=252)
    try:
        mu_ema = ema_historical_return(prices, frequency=252, span=126)  # ~6 months
        mu = mu_mean.where(mu_mean >= mu_ema, mu_ema)  # take element-wise max
    except Exception:
        mu = mu_mean
    return mu.clip(lower=_MU_FLOOR)


def _build_ef(prices: pd.DataFrame, weight_bounds: tuple[float, float]) -> EfficientFrontier:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame")
    if prices.shape[1] < 2:
        raise ValueError("Need at least 2 assets for optimization")

    prices = prices.copy()
    prices.columns = [str(c).upper().strip() for c in prices.columns]

    mu = _estimate_mu(prices)
    cov = CovarianceShrinkage(prices).ledoit_wolf()
    return EfficientFrontier(mu, cov, weight_bounds=weight_bounds)


def _perf_dict(ef: EfficientFrontier, objective: str) -> dict:
    perf = ef.portfolio_performance(verbose=False)
    return {
        "expected_return": float(perf[0]),
        "volatility":      float(perf[1]),
        "sharpe":          float(perf[2]),
        "objective":       objective,
    }


# ── existing methods ───────────────────────────────────────────────────────────

def max_sharpe_weights(
    prices: pd.DataFrame,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    return_performance: bool = False,
):
    ef = _build_ef(prices, weight_bounds)
    ef.max_sharpe()
    weights = _clean_weights(ef.clean_weights())
    if not return_performance:
        return weights
    return weights, _perf_dict(ef, "max_sharpe")


def min_vol_weights(
    prices: pd.DataFrame,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    return_performance: bool = False,
):
    ef = _build_ef(prices, weight_bounds)
    ef.min_volatility()
    weights = _clean_weights(ef.clean_weights())
    if not return_performance:
        return weights
    return weights, _perf_dict(ef, "min_vol")


# ── NEW: target return ─────────────────────────────────────────────────────────

def efficient_return_weights(
    prices: pd.DataFrame,
    target_return: float,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    return_performance: bool = False,
):
    """
    Minimize volatility subject to achieving `target_return` (annualized, e.g. 0.12).
    Falls back to max_sharpe if target is unachievable.
    """
    ef = _build_ef(prices, weight_bounds)
    mu = _estimate_mu(prices)
    max_possible = float(mu.max())

    # Clamp target to what's achievable
    target = min(target_return, max_possible * 0.95)

    try:
        ef.efficient_return(target_return=target)
    except Exception:
        # fallback
        ef = _build_ef(prices, weight_bounds)
        ef.max_sharpe()

    weights = _clean_weights(ef.clean_weights())
    if not return_performance:
        return weights
    return weights, _perf_dict(ef, f"efficient_return_{target_return:.0%}")


# ── NEW: max drawdown constraint ───────────────────────────────────────────────

def _estimate_max_drawdown(weights: dict, prices: pd.DataFrame) -> float:
    """
    Estimate historical max drawdown for a given set of weights.
    """
    prices = prices.copy()
    prices.columns = [str(c).upper().strip() for c in prices.columns]
    w = pd.Series(weights)
    w = w.reindex(prices.columns).fillna(0)
    portfolio_returns = prices.pct_change().dropna().dot(w)
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())  # negative number


def constrained_weights(
    prices: pd.DataFrame,
    target_return: float | None = None,
    max_drawdown: float | None = None,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    return_performance: bool = False,
):
    """
    Smart optimizer that respects:
    - target_return: minimum annual return (e.g. 0.12 for 12%)
    - max_drawdown: maximum acceptable drawdown (e.g. 0.25 for 25%)

    Strategy:
    1. If target_return given → use efficient_return
    2. If result violates max_drawdown → tighten max_w and retry
    3. Fallback to max_sharpe
    """
    prices = prices.copy()
    prices.columns = [str(c).upper().strip() for c in prices.columns]

    min_w, max_w = weight_bounds

    # Step 1: optimize for target return
    if target_return is not None:
        weights, perf = efficient_return_weights(
            prices, target_return, (min_w, max_w), return_performance=True
        )
    else:
        weights, perf = max_sharpe_weights(
            prices, (min_w, max_w), return_performance=True
        )

    # Step 2: check drawdown constraint
    if max_drawdown is not None:
        actual_dd = _estimate_max_drawdown(weights, prices)
        dd_limit = -abs(max_drawdown)  # make negative for comparison

        if actual_dd < dd_limit:
            # Drawdown too large — tighten concentration and retry with min_vol
            tighter_max_w = min(max_w, 0.20)
            try:
                ef = _build_ef(prices, (min_w, tighter_max_w))
                ef.min_volatility()
                weights = _clean_weights(ef.clean_weights())
                perf = _perf_dict(ef, "min_vol_drawdown_constrained")
                actual_dd = _estimate_max_drawdown(weights, prices)
            except Exception:
                pass

        perf["max_drawdown"] = actual_dd
        perf["max_drawdown_constraint"] = max_drawdown
        perf["drawdown_satisfied"] = actual_dd >= dd_limit

    if not return_performance:
        return weights
    return weights, perf


# ── parse constraints from natural language ────────────────────────────────────

def parse_constraints(msg: str) -> dict:
    """
    Extract target_return and max_drawdown from user message.
    Examples:
      "annual profit 12%" → target_return=0.12
      "max drawdown 25%"  → max_drawdown=0.25
    """
    constraints = {}

    # Target return
    ret_match = re.search(
        r'(?:annual|yearly|return|profit|عائد|ربح)[^\d]*(\d+(?:\.\d+)?)\s*%',
        msg, re.I
    )
    if ret_match:
        constraints["target_return"] = float(ret_match.group(1)) / 100

    # Max drawdown — English + Arabic (خسارة/خساره/هبوط/اقصى خساره)
    dd_match = re.search(
        r'(?:max(?:imum)?[\s_]?draw(?:down)?|mdd|هبوط|خسار[هة]|اقصى\s*خسار[هة]|اقصي\s*خسار[هة])[^\d]*(\d+(?:\.\d+)?)\s*%',
        msg, re.I
    )
    if dd_match:
        constraints["max_drawdown"] = float(dd_match.group(1)) / 100

    return constraints


# ── main optimize function (updated) ──────────────────────────────────────────

def optimize(
    prices: pd.DataFrame,
    method: str = "max_sharpe",
    min_w: float = 0.0,
    max_w: float = 1.0,
    return_performance: bool = False,
    target_return: float | None = None,
    max_drawdown: float | None = None,
):
    method = (method or "max_sharpe").lower().strip()
    bounds = (float(min_w), float(max_w))

    # If constraints given → use smart constrained optimizer
    if target_return is not None or max_drawdown is not None:
        return constrained_weights(
            prices,
            target_return=target_return,
            max_drawdown=max_drawdown,
            weight_bounds=bounds,
            return_performance=return_performance,
        )

    if method == "max_sharpe":
        return max_sharpe_weights(prices, weight_bounds=bounds, return_performance=return_performance)

    if method in ("min_vol", "min_volatility", "minimum_volatility"):
        return min_vol_weights(prices, weight_bounds=bounds, return_performance=return_performance)

    if method in ("efficient_return", "target_return"):
        return efficient_return_weights(prices, target_return or 0.12, bounds, return_performance)

    raise ValueError(f"Unknown method: {method}")