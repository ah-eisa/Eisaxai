"""
core/dependencies/examples.py — How to use the new auth dependency in api_bridge_v2.py

These are EXAMPLE endpoints — do NOT add them to production.
They demonstrate the pattern that should replace the 40+ copy-pasted auth checks.
"""

from fastapi import APIRouter, Depends, Request
from core.dependencies.auth import require_auth

router = APIRouter()


# ── Example 1: Simple endpoint that replaces the old pattern ──────────────
#
# OLD (api_bridge_v2.py line 247-248):
#
#   async def chart_data(request: Request, ticker: str = "NVDA",
#                        access_token: str = Header(None, alias="X-API-Key"),
#                        access_token_alt: str = Header(None, alias="access-token")):
#       if (access_token or access_token_alt) != SECURE_TOKEN:
#           raise HTTPException(403, "Unauthorized")
#
# NEW:

@router.get("/v1/chart")
async def chart_data_new(
    request: Request,
    ticker: str = "NVDA",
    user: dict = Depends(require_auth),          # ← single line replaces 2 header params + 1 if-check
):
    # `user` is {"user_id": ..., "tier": ..., "method": ...}
    # or for JWT: {"user_id": ..., "email": ..., "role": ..., "tier": ..., "method": "jwt"}
    return {"ticker": ticker, "requested_by": user["user_id"]}


# ── Example 2: POST endpoint with body + auth ─────────────────────────────
#
# OLD (api_bridge_v2.py line 1783):
#
#   async def global_allocate(request: Request,
#                             access_token: str = Header(None, alias="X-API-Key"),
#                             access_token_alt: str = Header(None, alias="access-token")):
#       if (access_token or access_token_alt) != SECURE_TOKEN:
#           raise HTTPException(403, "Unauthorized")
#       body = await request.json()
#
# NEW:

@router.post("/v1/global-allocate")
async def global_allocate_new(
    request: Request,
    user: dict = Depends(require_auth),
):
    body = await request.json()
    return {
        "allocated": True,
        "user_id": user["user_id"],
        "tier": user.get("tier", "basic"),
        "portfolio": body.get("portfolio", []),
    }


# ── Optional: per-tier access control ─────────────────────────────────────
#
# You can build on top of require_auth for role-based gates:

def require_tier(min_tier: str):
    """Return a dependency that checks the user meets a minimum tier."""
    tier_order = {"basic": 0, "standard": 1, "premium": 2, "vip": 3}

    def _check(user: dict = Depends(require_auth)):
        if tier_order.get(user.get("tier", "basic"), 0) < tier_order.get(min_tier, 0):
            from fastapi import HTTPException
            raise HTTPException(403, f"Tier '{min_tier}' or higher required")
        return user

    return _check


@router.get("/v1/admin/stats")
async def admin_stats(
    request: Request,
    user: dict = Depends(require_tier("vip")),      # only vip / secure_token users
):
    return {"total_users": 42, "requested_by": user["user_id"]}
