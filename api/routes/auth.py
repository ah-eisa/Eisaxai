"""
api/routes/auth.py — Extracted authentication & user-management routes

Extracted from api_bridge_v2.py (Phase 1 — Step 1 of 3).
All endpoints use Depends(require_auth) or Depends(_require_jwt) / Depends(_require_admin)
instead of raw token checks.

Endpoints extracted:
  POST   /auth/login                       (line 3046)
  POST   /auth/change-password              (line 3068)
  GET    /auth/me                           (line 3083)
  POST   /v1/keys                           (line 3096)
  GET    /v1/keys                           (line 3121)
  DELETE /v1/keys/{key_id}                  (line 3134)
  POST   /v1/keys/validate                  (line 3151)
  POST   /admin/users                       (line 3172)
  GET    /admin/users                       (line 3188)
  PATCH  /admin/users/{user_id}             (line 3194)
  DELETE /admin/users/{user_id}             (line 3203)
  POST   /admin/users/{user_id}/reset-password  (line 3211)
"""

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Rate limiter — instantiated at module load time so decorators work
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


def _limit(rate: str):
    """Apply rate limiting via the module-level limiter."""
    return limiter.limit(rate)


# ---------------------------------------------------------------------------
# Reuse auth dependency from core
# ---------------------------------------------------------------------------
from core.dependencies.auth import require_auth


# ---------------------------------------------------------------------------
# Internal auth helpers (mirrors api_bridge_v2.py lines 2958-3010)
# ---------------------------------------------------------------------------
import jwt as _jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from core.auth import hash_password, verify_password, create_token, decode_token, generate_temp_password
from core.user_db import (
    create_user, get_user_by_email, get_user_by_id,
    list_users, update_user, delete_user, record_login,
    increment_failed_attempts, reset_failed_attempts,
)

_bearer = HTTPBearer(auto_error=False)


def _require_jwt(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    """FastAPI dependency — validates Bearer JWT and returns payload."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_token(credentials.credentials)
    except _jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except _jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload


def _require_admin(payload: dict = Depends(_require_jwt)) -> dict:
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
    return payload


def _resolve_auth(
    x_api_key: str = Header(None, alias="X-API-Key"),
    access_token: str = Header(None, alias="access-token"),
    authorization: str = Header(None, alias="Authorization"),
) -> dict:
    token = x_api_key or access_token
    bearer = (authorization or "").removeprefix("Bearer ").strip()
    # 1. Personal API key (starts with eixa_)
    if token and token.startswith("eixa_"):
        from core.api_keys import validate_key
        info = validate_key(token)
        if info:
            return {"user_id": info["user_id"], "tier": info["tier"], "method": "api_key"}
        raise HTTPException(401, "Invalid API key")
    if bearer and bearer.startswith("eixa_"):
        from core.api_keys import validate_key
        info = validate_key(bearer)
        if info:
            return {"user_id": info["user_id"], "tier": info["tier"], "method": "api_key"}
        raise HTTPException(401, "Invalid API key")
    # 2. Legacy SECURE_TOKEN — retired (Phase 4); no token grant beyond eixa_ keys
    raise HTTPException(403, "Unauthorized")


def _resolve_user_context(
    access_token: Optional[str] = None,
    access_token_alt: Optional[str] = None,
    authorization: Optional[str] = None,
) -> dict:
    bearer = (authorization or "").removeprefix("Bearer ").strip()
    if bearer and not bearer.startswith("eixa_"):
        try:
            payload = decode_token(bearer)
        except _jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except _jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {
            "user_id": payload["sub"],
            "tier": "jwt",
            "method": "jwt",
            "role": payload.get("role", "user"),
        }
    auth = _resolve_auth(
        x_api_key=access_token,
        access_token=access_token_alt,
        authorization=authorization,
    )
    auth["role"] = "admin" if auth["user_id"] == "admin" else "user"
    return auth


import os


# ---------------------------------------------------------------------------
# Pydantic models (lines 3016-3044)
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class CreateUserRequest(BaseModel):
    email: str
    name: str
    role: str = "user"  # "user" | "admin"


class UpdateUserRequest(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[int] = None


class APIKeyCreateRequest(BaseModel):
    name: str = "Default"
    tier: str = "basic"
    daily_limit: int = 0


class APIKeyValidateRequest(BaseModel):
    key: Optional[str] = None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(prefix="", tags=["auth"])


# ── Auth endpoints (lines 3046-3093) ─────────────────────────────────────

@router.post("/auth/login")
@_limit("10/minute")
async def auth_login(request: Request, body: LoginRequest):
    from datetime import datetime, timezone
    _GENERIC_ERR = "Invalid credentials"
    user = get_user_by_email(body.email)
    if not user:
        raise HTTPException(status_code=401, detail=_GENERIC_ERR)
    # Lockout check
    locked_until = user.get("locked_until")
    if locked_until:
        try:
            lu = datetime.fromisoformat(locked_until)
            if datetime.now(timezone.utc) < lu:
                raise HTTPException(status_code=429, detail="Account locked — too many failed attempts. Try again later.")
        except ValueError:
            pass
    if not verify_password(body.password, user["password_hash"]):
        increment_failed_attempts(user["id"])
        raise HTTPException(status_code=401, detail=_GENERIC_ERR)
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account disabled")
    # Successful login — reset counter
    reset_failed_attempts(user["id"])
    record_login(user["id"])
    token = create_token(
        user["id"], user["email"], user["role"],
        must_change=bool(user["must_change_pw"])
    )
    return {
        "token": token,
        "must_change": bool(user["must_change_pw"]),
        "name": user["name"],
        "role": user["role"],
    }


@router.post("/auth/change-password")
@_limit("5/minute")
async def auth_change_password(request: Request, body: ChangePasswordRequest, payload: dict = Depends(_require_jwt)):
    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(body.old_password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Wrong current password")
    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")
    update_user(user["id"], password_hash=hash_password(body.new_password), must_change_pw=0)
    token = create_token(user["id"], user["email"], user["role"], must_change=False)
    return {"token": token, "message": "Password changed"}


@router.get("/auth/me")
@_limit("60/minute")
async def auth_me(request: Request, payload: dict = Depends(_require_jwt)):
    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
    }


# ── API Key endpoints (lines 3096-3169) ─────────────────────────────────

@router.post("/v1/keys")
@_limit("20/minute")
async def create_api_key(
    request: Request,
    body: APIKeyCreateRequest,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
    authorization: str = Header(None, alias="Authorization"),
):
    from core.api_keys import generate_key

    auth = _resolve_user_context(access_token, access_token_alt, authorization)
    raw_key = generate_key(str(auth["user_id"]), body.name, body.tier, body.daily_limit)
    return {
        "key": raw_key,
        "key_prefix": raw_key[:12],
        "user_id": str(auth["user_id"]),
        "name": body.name,
        "tier": body.tier,
        "daily_limit": body.daily_limit,
        "method": auth["method"],
    }


@router.get("/v1/keys")
@_limit("20/minute")
async def get_api_keys(
    request: Request,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
    authorization: str = Header(None, alias="Authorization"),
):
    from core.api_keys import list_user_keys

    auth = _resolve_user_context(access_token, access_token_alt, authorization)
    return {
        "user_id": str(auth["user_id"]),
        "keys": list_user_keys(str(auth["user_id"])),
    }


@router.delete("/v1/keys/{key_id}")
@_limit("20/minute")
async def delete_api_key(
    request: Request,
    key_id: int,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
    authorization: str = Header(None, alias="Authorization"),
):
    from core.api_keys import revoke_key

    auth = _resolve_user_context(access_token, access_token_alt, authorization)
    revoke_key(key_id, str(auth["user_id"]))
    return {"ok": True, "key_id": key_id}


@router.post("/v1/keys/validate")
@_limit("20/minute")
async def validate_api_key_endpoint(
    request: Request,
    body: Optional[APIKeyValidateRequest] = None,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
    authorization: str = Header(None, alias="Authorization"),
):
    resolved = _resolve_auth(
        x_api_key=(body.key if body else None) or access_token,
        access_token=access_token_alt,
        authorization=authorization,
    )
    return {"valid": True, **resolved}


# ── Admin user-management endpoints (lines 3172-3219) ───────────────────

@router.post("/admin/users")
@_limit("10/minute")
async def admin_create_user(request: Request, body: CreateUserRequest, _: dict = Depends(_require_admin)):
    if get_user_by_email(body.email):
        raise HTTPException(status_code=409, detail="Email already exists")
    temp_pw = generate_temp_password()
    uid = create_user(
        email=body.email,
        name=body.name,
        password_hash=hash_password(temp_pw),
        role=body.role,
        must_change_pw=True,
    )
    return {"id": uid, "email": body.email, "name": body.name, "temp_password": temp_pw}


@router.get("/admin/users")
@_limit("30/minute")
async def admin_list_users(request: Request, _: dict = Depends(_require_admin)):
    return list_users()


@router.patch("/admin/users/{user_id}")
@_limit("20/minute")
async def admin_update_user(request: Request, user_id: int, body: UpdateUserRequest, _: dict = Depends(_require_admin)):
    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    if not update_user(user_id, **changes):
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


@router.delete("/admin/users/{user_id}")
@_limit("10/minute")
async def admin_delete_user(request: Request, user_id: int, _: dict = Depends(_require_admin)):
    if not delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


@router.post("/admin/users/{user_id}/reset-password")
@_limit("10/minute")
async def admin_reset_password(request: Request, user_id: int, _: dict = Depends(_require_admin)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    temp_pw = generate_temp_password()
    update_user(user_id, password_hash=hash_password(temp_pw), must_change_pw=1)
    return {"temp_password": temp_pw}
