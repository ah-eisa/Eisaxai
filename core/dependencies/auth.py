"""
core/dependencies/auth.py — FastAPI authentication dependency for EisaX

Replaces the copy-pasted SECURE_TOKEN check across api_bridge_v2.py
with a single reusable `Depends()` function.

Supports two auth methods (checked in order):
  1. Personal API key  (X-API-Key or Authorization header starting with "eixa_")
  2. JWT Bearer token  (Authorization: Bearer <jwt>)

The legacy shared SECURE_TOKEN is RETIRED (Phase 4, 2026-06-11): it is
detected and rejected with a loud log line so stragglers are identifiable.

Usage:
    @router.get("/some-endpoint")
    async def some_endpoint(request: Request, user = Depends(require_auth)):
        # user is a dict: {"user_id": ..., "tier": ..., "method": ...}
"""

import hmac
import logging
import os
import time
from functools import lru_cache

from fastapi import Header, HTTPException, Request
from typing import Optional

logger = logging.getLogger(__name__)


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


# SECURE_TOKEN is RETIRED (Phase 4, 2026-06-11). The env var is read only so
# stragglers are identified by name in the log — the token grants nothing.


# ---------------------------------------------------------------------------
# Core resolver (shared logic for sync & async variants)
# ---------------------------------------------------------------------------

def _resolve(
    x_api_key: Optional[str],
    access_token_alt: Optional[str],
    authorization: Optional[str],
    client: str = "",
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

    # 2. Legacy SECURE_TOKEN — RETIRED: reject loudly so any straggler client
    #    is visible in the log instead of failing as a generic 401.
    secret = _secure_token()
    if secret and (hmac.compare_digest(token, secret) or hmac.compare_digest(bearer, secret)):
        logger.error(
            "[auth] RETIRED SECURE_TOKEN rejected for %s — mint an eixa_ key for this client",
            client or "unknown-client",
        )
        raise HTTPException(status_code=401, detail="Legacy token retired — use a personal API key")

    # 3. JWT Bearer token (eixa_ bearers already returned above)
    if bearer:
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
# Public non-dependency form — for endpoints that accept a body-token fallback
# (e.g. /v1/export/html sends the key inside the JSON payload). Feed the body
# token as x_api_key so every token type (eixa_/JWT/legacy) resolves the same.
# ---------------------------------------------------------------------------

def resolve_auth(
    x_api_key: Optional[str],
    access_token_alt: Optional[str] = None,
    authorization: Optional[str] = None,
    client: str = "",
) -> dict:
    return _resolve(x_api_key, access_token_alt, authorization, client)


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
    client_host = getattr(request.client, "host", "") if request.client else ""
    return _resolve(
        x_api_key,
        access_token_alt,
        authorization,
        client=f"{client_host} {request.method} {request.url.path}",
    )


# ---------------------------------------------------------------------------
# Sync variant — for background tasks or non-async endpoints
# ---------------------------------------------------------------------------

def require_auth_sync(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    access_token_alt: Optional[str] = Header(None, alias="access-token"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> dict:
    return _resolve(x_api_key, access_token_alt, authorization)
