from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from core.data import get_prices, to_returns
from core.portfolio import optimize as optimize_core
from core.metrics import perf_metrics
from core.policy import apply_policy

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

class OptimizeRequest(BaseModel):
    tickers: list[str] = Field(..., example=["AAPL","MSFT","GOOGL","AMZN"])
    start: str = "2022-01-01"
    end: str | None = None
    method: str = "max_sharpe"
    min_w: float = 0.0
    max_w: float = 0.35
    min_assets: int = 4
    seed_w: float = 0.02
    rf: float = 0.0
    force_refresh: bool = False
    include_performance: bool = True

@router.post("/optimize")
def optimize(req: OptimizeRequest):
    try:
        prices = get_prices(req.tickers, start=req.start, end=req.end, force_refresh=req.force_refresh)

        if req.include_performance:
            w_raw, perf = optimize_core(
                prices,
                method=req.method,
                min_w=req.min_w,
                max_w=req.max_w,
                return_performance=True,
            )
        else:
            w_raw = optimize_core(
                prices,
                method=req.method,
                min_w=req.min_w,
                max_w=req.max_w,
                return_performance=False,
            )
            perf = None

        w = apply_policy(
            w_raw,
            universe=req.tickers,
            min_assets=req.min_assets,
            max_w=req.max_w,
            min_w=req.min_w,
            seed_w=req.seed_w,
        )

        out = {
            "weights_raw": w_raw,
            "weights": w,
            "policy": {"min_assets": req.min_assets, "seed_w": req.seed_w, "min_w": req.min_w, "max_w": req.max_w},
        }
        if perf is not None:
            out["performance"] = perf
        return out

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class MetricsRequest(BaseModel):
    tickers: list[str]
    weights: dict[str, float]
    start: str = "2022-01-01"
    end: str | None = None
    rf: float = 0.0
    force_refresh: bool = False

@router.post("/metrics")
def metrics(req: MetricsRequest):
    try:
        prices = get_prices(req.tickers, start=req.start, end=req.end, force_refresh=req.force_refresh)
        rets = to_returns(prices)

        w = {k.upper().strip(): float(v) for k, v in req.weights.items()}
        cols = [c for c in rets.columns if c.upper().strip() in w]
        if not cols:
            raise ValueError("No overlap between weights and downloaded tickers/returns.")

        w_series = pd.Series({c: w[c.upper().strip()] for c in cols})
        port = (rets[cols] * w_series).sum(axis=1)

        return {"metrics": perf_metrics(port, rf=req.rf)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class ReportRequest(BaseModel):
    tickers: list[str] = Field(..., example=["AAPL","MSFT","GOOGL","AMZN"])
    start: str = "2022-01-01"
    end: str | None = None
    method: str = "max_sharpe"
    min_w: float = 0.0
    max_w: float = 0.35
    min_assets: int = 4
    seed_w: float = 0.02
    rf: float = 0.0
    force_refresh: bool = False

@router.post("/report")
def report(req: ReportRequest):
    try:
        prices = get_prices(req.tickers, start=req.start, end=req.end, force_refresh=req.force_refresh)

        w_raw, perf = optimize_core(
            prices,
            method=req.method,
            min_w=req.min_w,
            max_w=req.max_w,
            return_performance=True,
        )

        w = apply_policy(
            w_raw,
            universe=req.tickers,
            min_assets=req.min_assets,
            max_w=req.max_w,
            min_w=req.min_w,
            seed_w=req.seed_w,
        )

        rets = to_returns(prices)
        cols = [c for c in rets.columns if c.upper().strip() in w]
        if not cols:
            raise ValueError("No overlap between policy weights and returns columns.")

        w_series = pd.Series({c: float(w[c.upper().strip()]) for c in cols})
        port = (rets[cols] * w_series).sum(axis=1)

        m = perf_metrics(port, rf=req.rf)

        return {
            "weights_raw": w_raw,
            "weights": w,
            "policy": {"min_assets": req.min_assets, "seed_w": req.seed_w, "min_w": req.min_w, "max_w": req.max_w},
            "performance": perf,
            "metrics": m,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
