"""
api/routers/portfolio_memory.py
────────────────────────────────
Portfolio snapshot history + global allocation routes.

Router exported:
  portfolio_memory_router  — no prefix, mounts /v1/portfolio/* and /v1/global-allocate/*
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.routes.auth import _require_jwt

logger = logging.getLogger("api_bridge")
limiter = Limiter(key_func=get_remote_address)

portfolio_memory_router = APIRouter()


@portfolio_memory_router.get("/v1/portfolio/history/{user_id}")
@limiter.limit("30/minute")
async def portfolio_history(
    request: Request,
    user_id: str,
    limit: int = 20,
    user: dict = Depends(_require_jwt),
):
    """
    Return portfolio snapshot history for a user.
    Shows how allocation + metrics evolved over time.
    """
    is_admin = user.get("role") == "admin" or user.get("user_id") == "admin"
    if not is_admin and user_id != str(user["sub"]):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        from portfolio_memory import get_user_snapshots, get_performance_history
        snapshots = get_user_snapshots(user_id, limit=limit)
        history   = get_performance_history(user_id, limit=limit)
        return {
            "user_id":    user_id,
            "count":      len(snapshots),
            "snapshots":  snapshots,
            "performance_history": history,
        }
    except Exception as e:
        return {"error": str(e)}


@portfolio_memory_router.get("/v1/portfolio/snapshot/{snapshot_id}")
@limiter.limit("30/minute")
async def get_portfolio_snapshot(
    request: Request,
    snapshot_id: str,
    user: dict = Depends(_require_jwt),
):
    """
    Retrieve a specific portfolio snapshot by ID — full report + audit data.
    Enables report reproducibility.
    """
    try:
        from portfolio_memory import get_snapshot
        snap = get_snapshot(snapshot_id)
        if not snap:
            raise HTTPException(status_code=404, detail=f"Snapshot {snapshot_id} not found")
        is_admin = user.get("role") == "admin" or user.get("user_id") == "admin"
        if snap.get("user_id") and snap["user_id"] != str(user["sub"]) and not is_admin:
            raise HTTPException(status_code=403, detail="Access denied")
        return snap
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


@portfolio_memory_router.get("/v1/portfolio/compare")
@limiter.limit("20/minute")
async def compare_portfolio_snapshots(
    request: Request,
    snap_a: str,
    snap_b: str,
    user: dict = Depends(_require_jwt),
):
    """
    Compare two portfolio snapshots: allocation drift + metric changes.
    Use to track how a portfolio evolved between rebalances.
    """
    try:
        from portfolio_memory import compare_snapshots, get_snapshot
        is_admin = user.get("role") == "admin" or user.get("user_id") == "admin"
        snapshot_a = get_snapshot(snap_a)
        snapshot_b = get_snapshot(snap_b)
        if not snapshot_a or not snapshot_b:
            raise HTTPException(status_code=404, detail="One or both snapshot IDs not found")
        for snap in (snapshot_a, snapshot_b):
            if snap.get("user_id") and snap["user_id"] != str(user["sub"]) and not is_admin:
                raise HTTPException(status_code=403, detail="Access denied")
        diff = compare_snapshots(snap_a, snap_b)
        if not diff:
            raise HTTPException(status_code=404, detail="One or both snapshot IDs not found")

        # Build human-readable markdown summary
        md_lines = [
            "## 📊 Portfolio Comparison",
            f"**{diff['date_a']}** → **{diff['date_b']}**",
            "",
            "### Allocation Changes",
            "| Ticker | Before | After | Δ | Direction |",
            "|--------|--------|-------|---|-----------|",
        ]
        for t, d in sorted(diff["allocation_diff"].items()):
            direction = (
                "🟢 Added" if d["before"] == 0
                else "🔴 Removed" if d["after"] == 0
                else ("🔼 Increased" if d["delta"] > 0 else "🔽 Decreased" if d["delta"] < 0 else "⚪ Unchanged")
            )
            md_lines.append(f"| **{t}** | {d['before']:.1f}% | {d['after']:.1f}% | {d['delta']:+.1f}pp | {direction} |")

        md_lines += [
            "", "### Metric Changes",
            "| Metric | Before | After | Δ | Better? |",
            "|--------|--------|-------|---|---------|",
        ]
        for key, label, better_if in [
            ("sharpe",       "Sharpe Ratio",    "higher"),
            ("beta",         "Portfolio Beta",  "lower"),
            ("cvar_95",      "CVaR 95%/day",    "higher"),
            ("ann_vol",      "Ann. Volatility", "lower"),
            ("total_return", "1Y Total Return", "higher"),
        ]:
            d = diff["metric_diff"].get(key)
            if d is None:
                continue
            improved = (d["delta"] > 0) == (better_if == "higher")
            icon = "✅" if improved else "🔴" if d["delta"] != 0 else "⚪"
            md_lines.append(f"| {label} | {d['before']:.2f} | {d['after']:.2f} | {d['delta']:+.2f} | {icon} |")

        diff["summary_md"] = "\n".join(md_lines)
        return diff
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


@portfolio_memory_router.post("/v1/global-allocate")
@limiter.limit("5/minute")
async def global_allocate(
    request: Request,
    user: dict = Depends(_require_jwt),
):
    """
    🌍 Global Allocation Engine — cross-market QP optimization.
    Allocates across US / GCC / Egypt / Crypto / Gold / Bonds.

    Body (JSON):
    {
        "profile":          "conservative" | "balanced" | "growth" | "aggressive",
        "region_include":   ["US","GCC","Gold"],      // optional: only these
        "region_exclude":   ["Crypto"],               // optional: exclude these
        "custom_caps":      {"Crypto": 0.10},         // optional: override caps
        "port_value_usd":   100000,                   // optional: $100k default
        "rf_rate":          0.045                     // optional: risk-free rate
    }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    profile        = body.get("profile", "balanced")
    region_include = body.get("region_include")
    region_exclude = body.get("region_exclude")
    custom_caps    = body.get("custom_caps")
    port_value_usd = float(body.get("port_value_usd", 100_000))
    rf_rate        = float(body.get("rf_rate", 0.045))

    try:
        from global_allocator import allocate
        result = allocate(
            profile=profile,
            region_include=region_include,
            region_exclude=region_exclude,
            custom_caps=custom_caps,
            rf_rate=rf_rate,
            port_value_usd=port_value_usd,
        )
        return result
    except Exception as e:
        logger.error("Global allocator error: %s", e, exc_info=True)
        return {"error": str(e)}


@portfolio_memory_router.get("/v1/global-allocate/profiles")
@limiter.limit("60/minute")
async def global_allocate_profiles(
    request: Request,
    user: dict = Depends(_require_jwt),
):
    """List available risk profiles and regions for the Global Allocation Engine."""
    try:
        from global_allocator import _PROFILES, _UNIVERSE
        regions = sorted(set(a.region for a in _UNIVERSE))
        return {
            "profiles": {
                k: {
                    "label": v["label"],
                    "description": v["description"],
                    "max_beta": v["max_beta"],
                    "max_vol": v["max_vol"],
                }
                for k, v in _PROFILES.items()
            },
            "regions": regions,
            "assets": [
                {"name": a.name, "region": a.region, "proxy": a.proxy,
                 "description": a.description}
                for a in _UNIVERSE
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@portfolio_memory_router.get("/v1/portfolio/performance/{user_id}")
@limiter.limit("30/minute")
async def portfolio_performance_chart(
    request: Request,
    user_id: str,
    user: dict = Depends(_require_jwt),
):
    """
    Return time-series of key metrics across all snapshots for chart rendering.
    Frontend can plot Sharpe / Beta / CVaR evolution over time.
    """
    is_admin = user.get("role") == "admin" or user.get("user_id") == "admin"
    if not is_admin and user_id != str(user["sub"]):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        from portfolio_memory import get_performance_history
        history = get_performance_history(user_id, limit=50)
        return {
            "user_id": user_id,
            "data_points": len(history),
            "series": {
                "dates":        [h["date"] for h in history],
                "sharpe":       [h["sharpe"] for h in history],
                "beta":         [h["beta"] for h in history],
                "cvar_95":      [h["cvar_95"] for h in history],
                "total_return": [h["total_return"] for h in history],
                "ann_vol":      [h["ann_vol"] for h in history],
            },
        }
    except Exception as e:
        return {"error": str(e)}
