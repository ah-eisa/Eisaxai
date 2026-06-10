"""
core/dependencies/auth.py — FastAPI authentication dependency for EisaX

Replaces the copy-pasted SECURE_TOKEN check across api_bridge_v2.py
with a single reusable `Depends()` function.

Supports three auth methods (checked in order):
  1. Personal API key  (X-API-Key or Authorization header starting with "eixa_")
  2. Legacy admin token (SECURE_TOKEN env var)
  3. JWT Bearer token  (Authorization: Bearer <jwt>)

Usage:
    @router.get("/some-endpoint")
    async def some_endpoint(request: Request, user = Depends(require_auth)):
        # user is a dict: {"user_id": ..., "tier": ..., "method": ...}
"""

import os
from functools import lru_cache

from fastapi import Header, HTTPException, Request
from typing import Optional


# ---------------------------------------------------------------------------
# Lazy imports — avoid circular dependencies at module load time
# ---------------------------------------------------------------------------

def _validate_api_key(key: str) -> dict | None:
    from core.api_keys import validate_key
    return validate_key(key)


def _decode_jwt(token: str) -> dict | None:
    from core.auth import decode_token
    try:
        return decode_token(token)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cached env read (reload only on process restart — change via env var)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _secure_token() -> str:
    return os.getenv("SECURE_TOKEN", "")


# ---------------------------------------------------------------------------
# Core resolver (shared logic for sync & async variants)
# ---------------------------------------------------------------------------

def _resolve(
    x_api_key: Optional[str],
    access_token_alt: Optional[str],
    authorization: Optional[str],
) -> dict:
    """Return user context dict or raise HTTPException."""

    x_api_key = (x_api_key or "").strip()
    access_token_alt = (access_token_alt or "").strip()
    auth_header = (authorization or "").strip()
    bearer = auth_header.removeprefix("Bearer ").removeprefix("bearer ").strip()

    token = x_api_key or access_token_alt

    # 1. Personal API key (starts with "eixa_")
    if token.startswith("eixa_"):
        info = _validate_api_key(token)
        if info:
            return {"user_id": info["user_id"], "tier": info["tier"], "method": "api_key"}
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    if bearer.startswith("eixa_"):
        info = _validate_api_key(bearer)
        if info:
            return {"user_id": info["user_id"], "tier": info["tier"], "method": "api_key"}
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    # 2. Legacy SECURE_TOKEN (admin bypass)
    secret = _secure_token()
    if secret and (token == secret or bearer == secret):
        return {"user_id": "admin", "tier": "vip", "method": "secure_token"}

    # 3. JWT Bearer token
    if bearer and not bearer.startswith("eixa_") and bearer != secret:
        payload = _decode_jwt(bearer)
        if payload:
            return {
                "user_id": payload.get("sub", "unknown"),
                "email": payload.get("email", ""),
                "role": payload.get("role", "user"),
                "tier": "premium",
                "method": "jwt",
            }
        raise HTTPException(status_code=401, detail="Invalid or expired JWT")

    # 4. No valid credentials
    raise HTTPException(status_code=401, detail="Missing authentication")


# ---------------------------------------------------------------------------
# Async variant — use this on normal FastAPI route handlers
# ---------------------------------------------------------------------------

async def require_auth(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    access_token_alt: Optional[str] = Header(None, alias="access-token"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> dict:
    """
    FastAPI dependency that resolves the authenticated user.

    Returns:
        {"user_id": str, "tier": str, "method": str}
        JWT auth also includes: {"email": str, "role": str}

    Raises:
        HTTPException 401 — invalid / missing credentials
    """
    return _resolve(x_api_key, access_token_alt, authorization)


# ---------------------------------------------------------------------------
# Sync variant — for background tasks or non-async endpoints
# ---------------------------------------------------------------------------

def require_auth_sync(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    access_token_alt: Optional[str] = Header(None, alias="access-token"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> dict:
    return _resolve(x_api_key, access_token_alt, authorization)
