"""
Portfolio Analytics API — EisaX.
8 endpoints exposing advanced portfolio analytics to the frontend.

All endpoints accept a `positions` list of dicts:
    [{"ticker": str, "name": str, "sector": str, "value": float, "qty": float,
      "price": float, "cost_basis": float (optional)}, ...]

Mounted at /v1/portfolio/*
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("eisax.api.portfolio_analytics")

router = APIRouter(prefix="/v1/portfolio", tags=["portfolio-analytics"])


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types and replace NaN/inf with None
    so the response can pass FastAPI's strict JSON encoder."""
    if obj is None:
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.ndarray,)):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


# ── Shared schemas ──────────────────────────────────────────────────────────────
class Position(BaseModel):
    model_config = {"populate_by_name": True, "extra": "allow"}
    ticker:     str
    name:       Optional[str] = ""
    sector:     Optional[str] = "Unknown"
    value:      float = 0.0
    qty:        float = Field(default=0.0, validation_alias="qty")
    price:      Optional[float] = 0.0
    cost_basis: Optional[float] = None


class _BasePayload(BaseModel):
    positions:   list[Position]
    total_value: Optional[float] = None  # derived from positions if None


def _positions_to_df(positions: list[Position]) -> tuple[pd.DataFrame, float]:
    """Convert Position list to DataFrame + total_value.
    If 'value' is 0 but qty/shares*price gives a positive number, fill it in
    so downstream analytics can show real per-position values."""
    rows = []
    for p in positions:
        d = p.model_dump(by_alias=False)
        # Forgiving aliases: accept "shares" as alternate name for qty
        extras = getattr(p, "__pydantic_extra__", None) or {}
        if not d.get("qty") and "shares" in extras:
            d["qty"] = float(extras["shares"] or 0)
        # Auto-fill value when caller sent qty+price but not value
        if not d.get("value") and d.get("qty") and d.get("price"):
            d["value"] = float(d["qty"]) * float(d["price"])
        rows.append(d)
    df = pd.DataFrame(rows)
    total = float(df["value"].sum()) if "value" in df.columns and not df.empty else 0.0
    return df, total


# ══════════════════════════════════════════════════════════════════════════════
# 1. MACRO SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
class MacroSimRequest(_BasePayload):
    gdp_growth: float = Field(default=2.3, description="Annualized GDP growth %")
    inflation:  float = Field(default=3.0, description="CPI YoY %")
    fed_rate:   float = Field(default=4.5, description="Fed funds rate %")
    oil_brent:  float = Field(default=75.0, description="Brent oil $/bbl")
    usd_index:  float = Field(default=102.0, description="DXY index")


@router.post("/macro-sim")
def macro_simulation(req: MacroSimRequest):
    """Run macroeconomic simulation on a portfolio."""
    try:
        from core.macro_simulator import MacroScenario, simulate_portfolio
        df, total = _positions_to_df(req.positions)
        if df.empty:
            raise HTTPException(status_code=400, detail="positions list is empty")
        scen = MacroScenario(
            gdp_growth=req.gdp_growth, inflation=req.inflation,
            fed_rate=req.fed_rate, oil_brent=req.oil_brent, usd_index=req.usd_index,
        )
        result = simulate_portfolio(df, req.total_value or total, scen)
        return {
            "ok": True,
            "scenario":            result["scenario"],
            "baseline":            result["baseline"],
            "sector_impacts":      result["sector_impacts"],
            "total_impact_pct":    result["total_impact_pct"],
            "total_impact_value":  result["total_impact_value"],
            "new_portfolio_value": result["new_portfolio_value"],
            "position_impacts":    result["position_impacts"].to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("macro_simulation failed")
        raise HTTPException(status_code=500, detail="Internal server error")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUDGET PLAN
# ══════════════════════════════════════════════════════════════════════════════
class BudgetRequest(_BasePayload):
    total_budget:          float = 0.0
    target_sector_weights: dict[str, float] = Field(default_factory=dict)


@router.post("/budget-plan")
def budget_plan(req: BudgetRequest):
    """Compute exact buy/sell quantities for a target sector allocation."""
    try:
        from core.budget_engine import compute_budget_allocation
        df, total = _positions_to_df(req.positions)
        result = compute_budget_allocation(
            total_budget=req.total_budget,
            target_sector_weights=req.target_sector_weights,
            positions_df=df,
            total_value=req.total_value or total,
        )
        result["allocations"] = result["allocations"].to_dict(orient="records") if not result["allocations"].empty else []
        return _json_safe({"ok": True, **result})
    except Exception as e:
        log.exception("budget_plan failed")
        raise HTTPException(status_code=500, detail="Internal server error")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FORWARD SCENARIO
# ══════════════════════════════════════════════════════════════════════════════
class ForwardScenarioRequest(MacroSimRequest):
    horizons_months: list[int] = [3, 6, 12]


@router.post("/forward-scenario")
def forward_scenario(req: ForwardScenarioRequest):
    """Project portfolio value at multiple horizons under a macro scenario."""
    try:
        from core.macro_simulator import MacroScenario
        from core.scenario_builder import build_forward_scenario
        df, total = _positions_to_df(req.positions)
        if df.empty:
            raise HTTPException(status_code=400, detail="positions list is empty")
        scen = MacroScenario(
            gdp_growth=req.gdp_growth, inflation=req.inflation,
            fed_rate=req.fed_rate, oil_brent=req.oil_brent, usd_index=req.usd_index,
        )
        result = build_forward_scenario(
            df, req.total_value or total, scen,
            horizons_months=req.horizons_months,
        )
        horizons_out = {}
        for h, hdata in result["horizons"].items():
            pp = hdata.get("position_projections", pd.DataFrame())
            horizons_out[str(h)] = {
                "projected_value":      hdata["projected_value"],
                "pct_change":           hdata["pct_change"],
                "position_projections": pp.to_dict(orient="records") if not pp.empty else [],
            }
        return {
            "ok":                 True,
            "horizons":           horizons_out,
            "macro_adjustments":  result["macro_adjustments"],
            "base_returns":       result["base_returns"],
            "scenario_label":     result["scenario_label"],
            "total_value":        result["total_value"],
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("forward_scenario failed")
        raise HTTPException(status_code=500, detail="Internal server error")


# ══════════════════════════════════════════════════════════════════════════════
# 4. MONTE CARLO / VAR
# ══════════════════════════════════════════════════════════════════════════════
class MonteCarloRequest(_BasePayload):
    n_simulations:      int = 5000
    horizon_days:       int = 252
    loss_threshold_pct: float = 0.10
    var_confidence:     list[float] = [0.95, 0.99]


@router.post("/monte-carlo")
def monte_carlo(req: MonteCarloRequest):
    """Run portfolio Monte Carlo + VaR/CVaR."""
    try:
        from core.monte_carlo import run_portfolio_monte_carlo
        df, total = _positions_to_df(req.positions)
        if df.empty:
            raise HTTPException(status_code=400, detail="positions list is empty")
        result = run_portfolio_monte_carlo(
            df, req.total_value or total,
            n_simulations=req.n_simulations,
            horizon_days=req.horizon_days,
            loss_threshold_pct=req.loss_threshold_pct,
            var_confidence_levels=req.var_confidence,
        )
        # Strip arrays — return summary stats only (frontend doesn't need full paths)
        return _json_safe({
            "ok":                     True,
            "var":                    {str(k): v for k, v in result["var"].items()},
            "cvar":                   {str(k): v for k, v in result["cvar"].items()},
            "prob_loss_gt_threshold": result["prob_loss_gt_threshold"],
            "best_outcome":           result["best_outcome"],
            "worst_outcome":          result["worst_outcome"],
            "median_outcome":         result["median_outcome"],
            "mean_outcome":           result["mean_outcome"],
            "inputs": {
                "n_simulations": result["inputs"]["n_simulations"],
                "horizon_days":  result["inputs"]["horizon_days"],
                "loss_threshold": result["inputs"]["loss_threshold"],
                "total_value":   result["inputs"]["total_value"],
            },
        })
    except HTTPException:
        raise
    except Exception as e:
        log.exception("monte_carlo failed")
        raise HTTPException(status_code=500, detail="Internal server error")


# ══════════════════════════════════════════════════════════════════════════════
# 5. MARKET REGIME COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
class RegimeRequest(_BasePayload):
    horizon_months: int = 12


@router.post("/regimes")
def market_regimes(req: RegimeRequest):
    """Compare portfolio under bull/bear/sideways/stagflation regimes."""
    try:
        from core.market_regimes import compare_regimes
        df, total = _positions_to_df(req.positions)
        if df.empty:
            raise HTTPException(status_code=400, detail="positions list is empty")
        result = compare_regimes(df, req.total_value or total,
                                 horizon_months=req.horizon_months)
        regimes_out = {}
        for rname, rdata in result["regimes"].items():
            pb = rdata.get("position_breakdown", pd.DataFrame())
            regimes_out[rname] = {
                "projected_value":            rdata["projected_value"],
                "expected_return_pct":        rdata["expected_return_pct"],
                "macro_profile":              rdata["macro_profile"],
                "sector_impacts":             rdata["sector_impacts"],
                "historical_base_return_pct": rdata["historical_base_return_pct"],
                "macro_elasticity_return_pct": rdata["macro_elasticity_return_pct"],
                "label_ar":                   rdata["label_ar"],
                "label_en":                   rdata["label_en"],
                "position_breakdown":         pb.to_dict(orient="records") if not pb.empty else [],
            }
        return {
            "ok":                True,
            "regimes":           regimes_out,
            "best_regime":       result["best_regime"],
            "worst_regime":      result["worst_regime"],
            "regime_spread_pct": result["regime_spread_pct"],
            "horizon_months":    result["horizon_months"],
            "total_value":       result["total_value"],
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("market_regimes failed")
        raise HTTPException(status_code=500, detail="Internal server error")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SHARIAH SCREENING
# ══════════════════════════════════════════════════════════════════════════════
@router.post("/shariah")
def shariah_screen(req: _BasePayload):
    """Screen portfolio for AAOIFI Shariah compliance."""
    try:
        from core.shariah_screener import screen_portfolio
        df, _ = _positions_to_df(req.positions)
        if df.empty:
            raise HTTPException(status_code=400, detail="positions list is empty")
        result = screen_portfolio(df)
        return {
            "ok":                    True,
            "compliance_rate_pct":   result["compliance_rate_pct"],
            "halal_count":           result["halal_count"],
            "haram_count":           result["haram_count"],
            "unknown_count":         result["unknown_count"],
            "total_halal_value":     result["total_halal_value"],
            "total_haram_value":     result["total_haram_value"],
            "total_unknown_value":   result["total_unknown_value"],
            "purification_estimate": result["purification_estimate"],
            "summary":               result["summary"],
            "results":               result["results"].to_dict(orient="records") if not result["results"].empty else [],
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("shariah_screen failed")
        raise HTTPException(status_code=500, detail="Internal server error")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PORTFOLIO OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
class OptimizerRequest(_BasePayload):
    objective:      str   = "max_sharpe"   # max_sharpe | min_variance | target_return
    target_return:  Optional[float] = None
    risk_free_rate: float = 0.04
    max_weight:     float = 0.40
    allow_short:    bool  = False
    include_frontier: bool = True


@router.post("/optimize")
def optimize(req: OptimizerRequest):
    """Optimize portfolio weights using Markowitz MPT."""
    try:
        from core.portfolio_optimizer import optimize_portfolio, efficient_frontier
        df, _ = _positions_to_df(req.positions)
        if df.empty:
            raise HTTPException(status_code=400, detail="positions list is empty")

        opt = optimize_portfolio(
            df, objective=req.objective,
            target_return=req.target_return,
            risk_free_rate=req.risk_free_rate,
            max_weight=req.max_weight,
            allow_short=req.allow_short,
        )
        if opt.get("error"):
            return {"ok": False, "error": opt["error"]}

        response = {
            "ok":               True,
            "tickers":          opt["tickers"],
            "current_weights":  opt["current_weights"],
            "optimal_weights":  opt["optimal_weights"],
            "current_stats":    opt["current_stats"],
            "optimal_stats":    opt["optimal_stats"],
            "improvement":      opt["improvement"],
            "rebalance_actions": opt["rebalance_actions"].to_dict(orient="records") if not opt["rebalance_actions"].empty else [],
            "objective":        opt["objective"],
        }

        if req.include_frontier:
            ef = efficient_frontier(
                df, n_points=20,
                risk_free_rate=req.risk_free_rate,
                max_weight=req.max_weight,
            )
            if not ef.get("error"):
                response["frontier"] = ef["frontier"].to_dict(orient="records") if not ef["frontier"].empty else []
                response["max_sharpe_point"] = ef["max_sharpe_point"]
                response["min_variance_point"] = ef["min_variance_point"]
            else:
                response["frontier_error"] = ef["error"]

        return response
    except HTTPException:
        raise
    except Exception as e:
        log.exception("optimize failed")
        raise HTTPException(status_code=500, detail="Internal server error")


# ══════════════════════════════════════════════════════════════════════════════
# 8. DIVIDEND INCOME PROJECTION
# ══════════════════════════════════════════════════════════════════════════════
class DividendRequest(_BasePayload):
    annual_contribution:   float = 0.0
    growth_assumption_pct: float = 0.0


@router.post("/dividend-income")
def dividend_income(req: DividendRequest):
    """Project portfolio dividend income."""
    try:
        from core.dividend_engine import project_portfolio_income
        df, _ = _positions_to_df(req.positions)
        if df.empty:
            raise HTTPException(status_code=400, detail="positions list is empty")
        result = project_portfolio_income(
            df,
            annual_contribution=req.annual_contribution,
            growth_assumption_pct=req.growth_assumption_pct,
        )
        return {
            "ok":                     True,
            "total_annual_income":    result["total_annual_income"],
            "monthly_average_income": result["monthly_average_income"],
            "portfolio_yield_pct":    result["portfolio_yield_pct"],
            "yield_on_cost_pct":      result["yield_on_cost_pct"],
            "weighted_payout_ratio":  result["weighted_payout_ratio"],
            "weighted_growth_rate":   result["weighted_growth_rate"],
            "sustainability_score":   result["sustainability_score"],
            "positions":              result["positions"].to_dict(orient="records") if not result["positions"].empty else [],
            "monthly_calendar":       result["monthly_calendar"].to_dict(orient="records") if not result["monthly_calendar"].empty else [],
            "projection_5y":          result["projection_5y"].to_dict(orient="records") if not result["projection_5y"].empty else [],
            "warnings":               result["warnings"],
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("dividend_income failed")
        raise HTTPException(status_code=500, detail="Internal server error")


# ══════════════════════════════════════════════════════════════════════════════
# DATA GUARD — completeness check + auto-enrichment
# ══════════════════════════════════════════════════════════════════════════════
class DataCheckRequest(BaseModel):
    ticker:       str
    level:        str  = "institutional"   # basic | technical | fundamental | institutional
    allow_scrape: bool = True


@router.post("/data-check")
def data_check(req: DataCheckRequest):
    """
    Pre-flight data completeness check.
    Returns what fields are present, what's missing, and (if allow_scrape) what was fetched.
    """
    try:
        from core.data_guard import ensure_complete, check_completeness
        if req.allow_scrape:
            report = ensure_complete(req.ticker, req.level, allow_scrape=True)
        else:
            report = check_completeness(req.ticker, req.level)
        return {"ok": True, **report.to_dict()}
    except Exception as e:
        log.exception("data_check failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/data-check/{ticker}")
def data_check_get(ticker: str, level: str = "institutional", allow_scrape: bool = True):
    """GET shortcut — useful for browser/curl testing."""
    return data_check(DataCheckRequest(ticker=ticker, level=level, allow_scrape=allow_scrape))


# ══════════════════════════════════════════════════════════════════════════════
# Index endpoint — list all analytics features
# ══════════════════════════════════════════════════════════════════════════════
@router.get("/analytics/list")
def list_analytics():
    """Return the catalog of available analytics features."""
    return {
        "ok": True,
        "features": [
            {"id": "macro-sim",          "name_ar": "محاكاة الاقتصاد الكلي",     "name_en": "Macro Simulation",     "endpoint": "/v1/portfolio/macro-sim",       "icon": "🌍"},
            {"id": "budget-plan",        "name_ar": "مخطط الميزانية",            "name_en": "Budget Planner",       "endpoint": "/v1/portfolio/budget-plan",     "icon": "💰"},
            {"id": "forward-scenario",   "name_ar": "سيناريو مستقبلي",          "name_en": "Forward Scenario",     "endpoint": "/v1/portfolio/forward-scenario","icon": "🔭"},
            {"id": "monte-carlo",        "name_ar": "مونت كارلو / VaR",         "name_en": "Monte Carlo / VaR",    "endpoint": "/v1/portfolio/monte-carlo",     "icon": "🎲"},
            {"id": "regimes",            "name_ar": "مقارنة أنظمة السوق",       "name_en": "Market Regimes",       "endpoint": "/v1/portfolio/regimes",         "icon": "🌐"},
            {"id": "shariah",            "name_ar": "الفحص الشرعي",             "name_en": "Shariah Screening",    "endpoint": "/v1/portfolio/shariah",         "icon": "🕌"},
            {"id": "optimize",           "name_ar": "تحسين المحفظة",            "name_en": "Portfolio Optimization", "endpoint": "/v1/portfolio/optimize",      "icon": "📊"},
            {"id": "dividend-income",    "name_ar": "دخل الأرباح الموزعة",      "name_en": "Dividend Income",      "endpoint": "/v1/portfolio/dividend-income", "icon": "💸"},
        ],
    }
