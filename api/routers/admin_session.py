"""
api/routers/admin_session.py
────────────────────────────
Admin session-management routes extracted from api_bridge_v2.py.

Routers exported:
  admin_session_router  — no prefix, mounts /admin/* endpoints
"""
from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt as _jwt
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.auth import JWT_ALGORITHM, JWT_SECRET, decode_token

logger = logging.getLogger("api_bridge")

# ── Rate limiter (decorator-only; enforcement via app.state.limiter) ──────────
limiter = Limiter(key_func=get_remote_address)

# ── Env-based globals ─────────────────────────────────────────────────────────
SECURE_TOKEN = os.getenv("SECURE_TOKEN", "")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
ADMIN_PASSPHRASE = os.getenv("ADMIN_PASSPHRASE", "") or os.getenv("ADMIN_TOKEN", "")

if not ADMIN_TOKEN:
    logger.warning("[STARTUP] ADMIN_TOKEN is not set — admin endpoints will be disabled")

# ── Pydantic models ───────────────────────────────────────────────────────────

class AdminAuthRequest(BaseModel):
    token: str = Field(..., min_length=1)

class AdminLoginRequest(BaseModel):
    password: str = Field(..., min_length=1)

# ── Admin auth helpers ────────────────────────────────────────────────────────

def _decode_admin_session_token(token: str) -> Optional[dict]:
    if not token:
        return None
    try:
        payload = decode_token(token)
    except (_jwt.ExpiredSignatureError, _jwt.InvalidTokenError):
        return None
    if payload.get("role") != "admin":
        return None
    return payload


def _check_secure_or_admin_session(token: str, request: Optional[Request] = None):
    if request:
        cookie_tok = request.cookies.get("eisax_admin_session", "")
        if cookie_tok and _decode_admin_session_token(cookie_tok):
            return
    if SECURE_TOKEN and token and token == SECURE_TOKEN:
        return
    if token and _decode_admin_session_token(token):
        return
    raise HTTPException(status_code=403, detail="Forbidden")


def _check_admin(token: str = "", request: Optional[Request] = None):
    # 1. Cookie session (primary — browser admin pages)
    if request:
        cookie_tok = request.cookies.get("eisax_admin_session", "")
        if cookie_tok and _decode_admin_session_token(cookie_tok):
            return
    # 2. SECURE_TOKEN header fallback (internal services / CLI)
    if SECURE_TOKEN and token and token == SECURE_TOKEN:
        return
    # 3. Previous signed JWT via header (transition)
    if token and _decode_admin_session_token(token):
        return
    # 4. ADMIN_TOKEN password fallback (legacy)
    from api_bridge_v2 import orchestrator as _orch
    if ADMIN_TOKEN and token and _orch.session_mgr.verify_admin_password(token, ADMIN_TOKEN):
        return
    if not SECURE_TOKEN and not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin access is not configured")
    raise HTTPException(status_code=403, detail="Forbidden")


def _require_admin_cookie(request: Request) -> dict:
    token = request.cookies.get("eisax_admin_session", "")
    if not token:
        raise HTTPException(status_code=401, detail="Admin session required")
    try:
        payload = decode_token(token)
    except _jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Admin session expired")
    except _jwt.InvalidTokenError:
        raise HTTPException(status_code=403, detail="Forbidden")
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    return payload

# ── Router ────────────────────────────────────────────────────────────────────
admin_session_router = APIRouter()


@admin_session_router.post("/admin/login")
@limiter.limit("5/minute")
async def admin_login(request: Request, body: AdminLoginRequest):
    if not ADMIN_PASSPHRASE or body.password != ADMIN_PASSPHRASE:
        raise HTTPException(status_code=403, detail="Invalid admin password")
    session_token = _jwt.encode(
        {"role": "admin", "exp": datetime.now(timezone.utc) + timedelta(hours=4)},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )
    response = JSONResponse({"status": "ok"})
    response.set_cookie(
        key="eisax_admin_session",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=14400,
        path="/",
    )
    return response


@admin_session_router.post("/admin/logout")
@limiter.limit("10/minute")
async def admin_logout(request: Request):
    response = JSONResponse({"status": "ok"})
    response.delete_cookie("eisax_admin_session", path="/")
    return response


@admin_session_router.get("/admin/sessions")
@limiter.limit("30/minute")
async def admin_sessions(request: Request, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    from collections import defaultdict
    from api_bridge_v2 import orchestrator
    sessions = orchestrator.session_mgr.get_all_sessions_admin()
    grouped = defaultdict(list)
    for s in sessions:
        grouped[s["user_id"]].append(s)
    result = []
    for uid, user_sessions in grouped.items():
        last = max((s["last_active"] or "") for s in user_sessions)
        is_blocked = any(s.get("blocked") for s in user_sessions)
        profile = orchestrator.session_mgr.get_user_profile(uid)
        daily_count = orchestrator.session_mgr.get_user_daily_count(uid)
        result.append({
            "user_id": uid,
            "session_count": len(user_sessions),
            "total_messages": sum(s["msg_count"] for s in user_sessions),
            "last_active": last,
            "ip": user_sessions[0].get("ip", "—"),
            "user_agent": user_sessions[0].get("user_agent", "—"),
            "blocked": is_blocked,
            "sessions": user_sessions,
            "daily_limit": profile.get("daily_limit", 0),
            "daily_count": daily_count,
            "note": profile.get("note", ""),
            "tier": profile.get("tier", "basic"),
        })
    result.sort(key=lambda x: x["last_active"] or "", reverse=True)
    return result


@admin_session_router.get("/admin/session/{session_id}")
@limiter.limit("60/minute")
async def admin_session_detail(request: Request, session_id: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    from api_bridge_v2 import orchestrator
    return orchestrator.session_mgr.get_chat_history(session_id)


@admin_session_router.get("/admin/stats")
@limiter.limit("30/minute")
async def admin_stats(request: Request, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    from api_bridge_v2 import orchestrator
    return orchestrator.session_mgr.get_admin_stats()


@admin_session_router.post("/admin/user/{user_id}/block")
@limiter.limit("30/minute")
async def block_user(request: Request, user_id: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    from api_bridge_v2 import orchestrator
    orchestrator.session_mgr.set_user_blocked(user_id, True)
    orchestrator.session_mgr.log_admin_action("block_user", user_id)
    return {"status": "blocked", "user_id": user_id}


@admin_session_router.post("/admin/user/{user_id}/unblock")
@limiter.limit("30/minute")
async def unblock_user(request: Request, user_id: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    from api_bridge_v2 import orchestrator
    orchestrator.session_mgr.set_user_blocked(user_id, False)
    orchestrator.session_mgr.log_admin_action("unblock_user", user_id)
    return {"status": "unblocked", "user_id": user_id}


@admin_session_router.post("/admin/user/{user_id}/message")
@limiter.limit("20/minute")
async def send_admin_message(request: Request, user_id: str, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    content = body.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    from api_bridge_v2 import orchestrator
    orchestrator.session_mgr.queue_admin_message(user_id, content)
    orchestrator.session_mgr.log_admin_action("message_user", user_id, content[:80])
    return {"status": "queued", "user_id": user_id}


@admin_session_router.get("/admin/messages")
@limiter.limit("30/minute")
async def get_admin_messages(request: Request, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    from api_bridge_v2 import orchestrator
    return orchestrator.session_mgr.get_admin_message_history()


@admin_session_router.post("/admin/settings/password")
@limiter.limit("5/minute")
async def change_admin_password(request: Request, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    new_password = body.get("new_password", "").strip()
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    from api_bridge_v2 import orchestrator
    orchestrator.session_mgr.change_admin_password(new_password)
    return {"status": "password updated"}


@admin_session_router.post("/admin/user/{user_id}/limit")
@limiter.limit("30/minute")
async def set_user_limit(request: Request, user_id: str, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    daily_limit = int(body.get("daily_limit", 0))
    if daily_limit < 0:
        raise HTTPException(status_code=400, detail="daily_limit must be >= 0")
    from api_bridge_v2 import orchestrator
    orchestrator.session_mgr.set_user_profile(user_id, daily_limit=daily_limit)
    orchestrator.session_mgr.log_admin_action("set_limit", user_id, str(daily_limit))
    return {"status": "ok", "user_id": user_id, "daily_limit": daily_limit}


@admin_session_router.post("/admin/user/{user_id}/note")
@limiter.limit("30/minute")
async def set_user_note(request: Request, user_id: str, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    note = body.get("note", "")
    from api_bridge_v2 import orchestrator
    orchestrator.session_mgr.set_user_profile(user_id, note=note)
    orchestrator.session_mgr.log_admin_action("set_note", user_id, note[:60] if note else "cleared")
    return {"status": "ok", "user_id": user_id}


@admin_session_router.post("/admin/user/{user_id}/tier")
@limiter.limit("30/minute")
async def set_user_tier(request: Request, user_id: str, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    tier = body.get("tier", "basic")
    if tier not in ("basic", "pro", "vip"):
        raise HTTPException(status_code=400, detail="tier must be basic, pro, or vip")
    from api_bridge_v2 import orchestrator
    orchestrator.session_mgr.set_user_profile(user_id, tier=tier)
    orchestrator.session_mgr.log_admin_action("set_tier", user_id, tier)
    return {"status": "ok", "user_id": user_id, "tier": tier}


@admin_session_router.post("/admin/broadcast")
@limiter.limit("5/minute")
async def broadcast_message(request: Request, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    content = body.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    from api_bridge_v2 import orchestrator
    count = orchestrator.session_mgr.broadcast_admin_message(content)
    orchestrator.session_mgr.log_admin_action("broadcast", f"{count} users", content[:80])
    return {"status": "broadcast", "recipients": count}


@admin_session_router.delete("/admin/user/{user_id}/sessions")
@limiter.limit("30/minute")
async def delete_user_sessions(request: Request, user_id: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    from api_bridge_v2 import orchestrator
    count = orchestrator.session_mgr.delete_user_sessions(user_id)
    orchestrator.session_mgr.log_admin_action("delete_sessions", user_id, f"{count} sessions deleted")
    return {"status": "deleted", "user_id": user_id, "sessions_deleted": count}


@admin_session_router.post("/admin/ip/{ip}/block")
@limiter.limit("30/minute")
async def block_ip_endpoint(request: Request, ip: str, body: dict = {}, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    reason = (body or {}).get("reason", "")
    from api_bridge_v2 import orchestrator
    orchestrator.session_mgr.block_ip(ip, reason)
    orchestrator.session_mgr.log_admin_action("block_ip", ip, reason or "no reason")
    return {"status": "blocked", "ip": ip}


@admin_session_router.post("/admin/ip/{ip}/unblock")
@limiter.limit("30/minute")
async def unblock_ip_endpoint(request: Request, ip: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    from api_bridge_v2 import orchestrator
    orchestrator.session_mgr.unblock_ip(ip)
    orchestrator.session_mgr.log_admin_action("unblock_ip", ip)
    return {"status": "unblocked", "ip": ip}


@admin_session_router.get("/admin/blocked-ips")
@limiter.limit("30/minute")
async def get_blocked_ips(request: Request, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    from api_bridge_v2 import orchestrator
    return orchestrator.session_mgr.get_blocked_ips()


@admin_session_router.get("/admin/audit-log")
@limiter.limit("30/minute")
async def get_audit_log(request: Request, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    from api_bridge_v2 import orchestrator
    return orchestrator.session_mgr.get_audit_log()


@admin_session_router.get("/admin/notifications")
@limiter.limit("60/minute")
async def get_notifications(request: Request, since: str = "", access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    if not since:
        since = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    from api_bridge_v2 import orchestrator
    return orchestrator.session_mgr.get_new_activity(since)


@admin_session_router.get("/admin/export/users")
@limiter.limit("10/minute")
async def export_users(request: Request, access_token: str = Header(None, alias="X-Admin-Key")):
    from collections import defaultdict
    from fastapi.responses import StreamingResponse as SR
    import csv
    _check_admin(access_token or "", request)
    from api_bridge_v2 import orchestrator
    sessions = orchestrator.session_mgr.get_all_sessions_admin()
    grouped = defaultdict(list)
    for s in sessions:
        grouped[s["user_id"]].append(s)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["User ID", "Sessions", "Total Messages", "Last Active", "IP", "Tier", "Daily Limit", "Blocked"])
    for uid, user_sessions in grouped.items():
        last = max((s["last_active"] or "") for s in user_sessions)
        is_blocked = any(s.get("blocked") for s in user_sessions)
        profile = orchestrator.session_mgr.get_user_profile(uid)
        writer.writerow([
            uid, len(user_sessions),
            sum(s["msg_count"] for s in user_sessions),
            last, user_sessions[0].get("ip", ""),
            profile.get("tier", "basic"),
            profile.get("daily_limit", 0),
            "Yes" if is_blocked else "No"
        ])
    output.seek(0)
    return SR(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=eisax_users_export.csv"}
    )
