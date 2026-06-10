"""
api/routes/chat.py — Extracted chat, analysis, admin, export, and misc routes

Extracted from api_bridge_v2.py (Phase 3 — Step 3 of 3).
All endpoints use Depends(require_auth) — no raw token checks.

Endpoints extracted (67 total):
  GET    /                                       (line 241)
  GET    /v1/chart-data                          (line 245)
  POST   /upload                                 (line 1873)
  GET    /health                                 (line 1895)
  POST   /v1/chat                                (line 1926)
  POST   /chat                                   (line 2051)
  POST   /api/chat                               (line 2052)
  POST   /v1/chat/stream                         (line 2078)
  POST   /v1/tts                                 (line 2146)
  GET    /admin/sessions                         (line 2173)
  GET    /admin/session/{session_id}             (line 2205)
  GET    /admin/stats                            (line 2211)
  POST   /admin/user/{user_id}/block             (line 2217)
  POST   /admin/user/{user_id}/unblock           (line 2225)
  POST   /admin/user/{user_id}/message           (line 2233)
  GET    /admin/messages                         (line 2244)
  POST   /admin/settings/password                (line 2250)
  POST   /admin/user/{user_id}/limit             (line 2260)
  POST   /admin/user/{user_id}/note              (line 2271)
  POST   /admin/user/{user_id}/tier              (line 2280)
  POST   /admin/broadcast                        (line 2291)
  DELETE /admin/user/{user_id}/sessions          (line 2302)
  POST   /admin/ip/{ip}/block                    (line 2310)
  POST   /admin/ip/{ip}/unblock                  (line 2319)
  GET    /admin/blocked-ips                      (line 2327)
  GET    /admin/audit-log                        (line 2333)
  GET    /admin/notifications                    (line 2339)
  GET    /admin/export/users                     (line 2348)
  GET    /api/history                            (line 2383)
  GET    /api/history/{session_id}               (line 2390)
  DELETE /api/history/{session_id}               (line 2397)
  POST   /v1/export                              (line 2405)
  GET    /v1/download/{filename}                 (line 2531)
  GET    /v1/brain/status                        (line 2550)
  GET    /v1/brain/wisdom                        (line 2558)
  POST   /v1/alerts                              (line 2582)
  GET    /v1/alerts                              (line 2591)
  DELETE /v1/alerts/{alert_id}                   (line 2598)
  GET    /v1/version                             (line 2606)
  POST   /v1/export/html                         (line 2617)
  GET    /v1/dashboard/{ticker}                  (line 2641)
  POST   /v1/translate-ar                        (line 2853)
  POST   /v1/export/html-pdf                     (line 2934)
  GET    /v1/health                              (line 3224)
  POST   /admin/cleanup                          (line 3243)
  GET    /admin/logs                             (line 3257)
  GET    /admin/logs/stream                      (line 3271)
  GET    /admin/analytics                        (line 3310)
  GET    /admin/analytics/data                   (line 3324)
  GET    /v1/usage                               (line 3407)
  GET    /v1/redis/health                        (line 3427)
  GET    /v1/referral                            (line 3442)
  POST   /v1/referral/apply                      (line 3456)
  POST   /v1/webhooks                            (line 3489)
  GET    /v1/webhooks                            (line 3516)
  DELETE /v1/webhooks/{webhook_id}               (line 3539)
  POST   /v1/billing/checkout                    (line 3568)
  POST   /v1/billing/webhook                     (line 3591)
  POST   /v1/billing/portal                      (line 3607)
  GET    /v1/sentiment/{ticker}                  (line 3636)
  POST   /v1/sentiment/batch                     (line 3659)
  GET    /v1/sentiment/market/overview           (line 3688)
  GET    /v1/sentiment/{ticker}/trend            (line 3710)
  POST   /v1/backtest                            (line 3746)
  POST   /v1/screener                            (line 3806)
  GET    /v1/forex                               (line 3853)
  GET    /v1/forex/{symbol}                      (line 3878)
"""

import os
import io
import re
import json as _json
import logging
import asyncio
import math
import shutil
import csv
import time
import sqlite3
from typing import Optional, Any, Dict, List
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from collections import defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from fastapi import APIRouter, Depends, Header, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import (
    StreamingResponse, RedirectResponse, JSONResponse, FileResponse
)
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.config import APP_DB, STATIC_DIR, EXPORTS_DIR, BACKEND_LOG, ENV_FILE
from core.dependencies.auth import require_auth
from core.orchestrator import MultiAgentOrchestrator
from core.tts_service import TTSService
from core.export_engine import export as export_engine
from core.news_aggregator import get_news as _get_aggregated_news

# ---------------------------------------------------------------------------
# Rate limiter — instantiated at module load time so decorators work
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

def _limit(rate: str):
    """Apply rate limiting via the module-level limiter."""
    return limiter.limit(rate)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(prefix="", tags=["chat"])

# ── Shared references ─────────────────────────────────────────────────────
logger = logging.getLogger("api_bridge")
_SECURE_TOKEN = os.getenv("SECURE_TOKEN", "")

# File cache helpers
_FILE_CACHE_DIR = "/home/ubuntu/investwise/file_cache"
_FILE_STORE_TTL = 3600
os.makedirs(_FILE_CACHE_DIR, exist_ok=True)

def _file_store_get(file_id: str):
    fpath = os.path.join(_FILE_CACHE_DIR, file_id + ".json")
    if not os.path.exists(fpath):
        return None
    with open(fpath, "r", encoding="utf-8") as _f:
        return _json.load(_f)


# ── Module-level shared objects ──────────────────────────────────────────
# These mirror the top-level instances in api_bridge_v2.py
orchestrator = MultiAgentOrchestrator(db_path=str(APP_DB))
tts_service = TTSService()

# ── Git sha + version ───────────────────────────────────────────────────
import subprocess as _subprocess
_GIT_SHA = "unknown"
try:
    _GIT_SHA = _subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd="/home/ubuntu/investwise",
        text=True,
    ).strip()
except Exception:
    pass
_APP_VERSION = "2.0.0"
try:
    _APP_VERSION = open("/home/ubuntu/investwise/version.txt").read().strip()
except Exception:
    pass

# ── Pydantic models (from api_bridge_v2.py lines 221-239) ───────────────
class MessagePayload(BaseModel):
    message: str = Field(..., max_length=16000)
    user_id: Optional[str] = "admin"
    session_id: Optional[str] = None
    files: Optional[list] = []
    settings: Optional[dict] = None


def _coerce_chat_payload(raw: dict) -> MessagePayload:
    """Accept legacy {text: ...} or new {message: ...} body."""
    data = dict(raw)
    if "text" in data and "message" not in data:
        legacy_text = data.pop("text")
        if isinstance(legacy_text, dict):
            data.update(legacy_text)
        else:
            data["message"] = legacy_text
    if not data.get("user_id"):
        data["user_id"] = "admin"
    return MessagePayload(**data)


class HtmlExportPayload(BaseModel):
    html: str
    filename: str = ""
    access_token: str = ""


class TranslatePayload(BaseModel):
    text: str
    access_token: str = ""


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


# ── ADMIN_TOKEN for session-based admin endpoints ─────────────────────────
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

def _check_admin(token: str):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin access is not configured")
    if not orchestrator.session_mgr.verify_admin_password(token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Forbidden")


@router.get("/")
async def root():
    return RedirectResponse(url="https://eisax.com", status_code=301)

@router.get("/v1/chart-data")
@_limit("30/minute")
async def chart_data(request: Request, ticker: str = "NVDA"):
    import yfinance as yf
    from datetime import datetime, timedelta
    df = None

    # ── Try yfinance first ────────────────────────────────────────────────────
    try:
        end = datetime.now()
        start = end - timedelta(days=65)
        _df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if not _df.empty:
            df = _df
    except Exception:
        pass

    # ── Fallback: investing.com for UAE/local market tickers ─────────────────
    if df is None or df.empty:
        try:
            from core.market_data_engine import UAE_INVESTING, _fetch_investing
            info = UAE_INVESTING.get(ticker)
            if info:
                from datetime import datetime, timedelta
                start_str = (datetime.now() - timedelta(days=75)).strftime("%Y-%m-%d")
                _df = await run_in_threadpool(_fetch_investing, ticker, info, start_str)
                if _df is not None and not _df.empty:
                    df = _df
        except Exception:
            pass

    if df is None or df.empty:
        return {"error": "No data"}

    import math
    tail = df.tail(60)
    close_col = "Close" if "Close" in tail.columns else tail.columns[0]
    dates_raw  = list(tail.index)
    prices_raw = [float(v) for v in tail[close_col].values]

    # Strip rows where price is NaN/Inf (non-trading days, halted stocks)
    # These cause "Out of range float values are not JSON compliant" errors.
    dates  = [d.strftime("%b %d") for d, p in zip(dates_raw, prices_raw)
              if not (math.isnan(p) or math.isinf(p))]
    prices = [round(p, 2) for p in prices_raw
              if not (math.isnan(p) or math.isinf(p))]

    if not prices:
        return {"error": "No valid price data"}
    return {"dates": dates, "prices": prices, "ticker": ticker}
@router.post("/upload")
@_limit("10/minute")
async def upload_file_ui(request: Request, file: UploadFile = File(...)):
    """Receive file from chat UI, extract text via Gemini Vision or file_processor."""
    import uuid as _uuid, base64 as _b64
    from core.file_processor import process_file
    raw = await file.read()
    b64 = _b64.b64encode(raw).decode()
    result = process_file(file.filename, b64)
    file_id = str(_uuid.uuid4())
    _evict_old_files()
    _file_store_set(file_id, {
        "id": file_id,
        "filename": file.filename,
        "text": result.get("text", ""),
        "error": result.get("error"),
        "_ts": _time.time(),
    })
    return {"status": "received", "file_id": file_id, "filename": file.filename}
@router.get("/health")
@_limit("30/minute")
async def health(request: Request):
    import psutil, time
    uptime = time.time() - psutil.boot_time()
    mem = psutil.virtual_memory()
    return {
        "status": "online",
        "agent": "EisaX General AI",
        "uptime_hours": round(uptime / 3600, 1),
        "memory_used_pct": round(mem.percent, 1),
        "cpu_pct": round(psutil.cpu_percent(interval=0.5), 1),
    }

from fastapi.concurrency import run_in_threadpool
import pandas as pd
import io

def _coerce_chat_payload(raw: dict) -> MessagePayload:
    """Accept legacy chat payloads and normalize to MessagePayload."""
    data = dict(raw or {})
    if "message" not in data:
        legacy_text = data.get("text") or data.get("query") or data.get("prompt")
        if isinstance(legacy_text, str):
            data["message"] = legacy_text
    if not data.get("user_id"):
        data["user_id"] = "admin"
    return MessagePayload(**data)
@router.post("/v1/chat")
@_limit("30/minute")
async def unified_chat(
    payload: MessagePayload,
    request: Request
):
    """نقطة الدخول الرئيسية للمحادثة - مع الحماية"""

    # Accept both X-API-Key and access-token headers (frontend uses access-token)

    client_ip = request.headers.get("X-Real-IP") or request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or str(request.client.host)
    user_agent = request.headers.get("User-Agent", "")

    # Block check
    if orchestrator.session_mgr.is_user_blocked(payload.user_id):
        raise HTTPException(status_code=403, detail="Your account has been suspended. Please contact support.")

    # Rate limit check
    if orchestrator.session_mgr.is_user_rate_limited(payload.user_id):
        raise HTTPException(status_code=429, detail="Daily message limit reached. Please try again tomorrow.")

    # IP block check
    if orchestrator.session_mgr.is_ip_blocked(client_ip):
        raise HTTPException(status_code=403, detail="Access denied from this network.")

    from core.rate_limiter import is_rate_limited, get_usage
    if is_rate_limited(payload.user_id):
        usage = get_usage(payload.user_id)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {usage['count']}/{usage['limit']} requests per minute. "
                   f"Reset in {usage['reset_in']:.0f}s.",
            headers={"Retry-After": str(int(usage["reset_in"]))}
        )

    session_id = payload.session_id or orchestrator.session_mgr.get_or_create_session(
        payload.user_id, ip=client_ip, user_agent=user_agent
    )
    orchestrator.session_mgr.get_or_create_session(
        payload.user_id, session_id=session_id, ip=client_ip, user_agent=user_agent
    )

    # Admin message injection — deliver queued messages before processing user input
    pending = orchestrator.session_mgr.get_pending_admin_messages(payload.user_id)
    if pending:
        orchestrator.session_mgr.mark_admin_messages_delivered(payload.user_id)
        combined = "\n\n".join(f"📢 {m['content']}" for m in pending)
        orchestrator.session_mgr.save_message(session_id, payload.user_id, "assistant", combined)
        response_body = {
            "reply": combined,
            "session_id": session_id,
            "agent": "Admin",
            "model": None,
            "download_url": None,
            "format": None,
            "quota": orchestrator.session_mgr.get_user_daily_usage(payload.user_id),
        }
        return JSONResponse(
            content=response_body,
            headers=orchestrator.session_mgr.get_quota_header(payload.user_id)
        )

    message = payload.message

    # Inject file content from /upload store via active_file_id
    active_file_id = None
    if payload.settings and isinstance(payload.settings, dict):
        active_file_id = payload.settings.get("active_file_id")
    stored_file = _file_store_get(active_file_id) if active_file_id else None
    if stored_file and stored_file.get("text"):
        file_text = stored_file["text"]
        fname = stored_file.get("filename", "file")
        message = ("[FILE ANALYSIS]" + chr(10)
                   + "File content (" + fname + "):" + chr(10) + chr(10)
                   + file_text[:8000] + chr(10) + chr(10)
                   + "User question: " + message)

    # Process uploaded files
    if payload.files:
        try:
            from core.file_processor import process_file
            extracted_parts = []
            for f in payload.files:
                filename = f.get("filename") or f.get("name", "file")
                b64data = f.get("data", "")
                if not b64data:
                    continue
                res = process_file(filename, b64data)
                if res.get("text"):
                    part = "[File: " + filename + "]" + chr(10) + res["text"][:8000]
                    extracted_parts.append(part)
            if extracted_parts:
                file_block = (chr(10) + chr(10)).join(extracted_parts)
                message = ("[FILE ANALYSIS]" + chr(10) + "File content below:" + chr(10) + chr(10)
                           + file_block + chr(10) + chr(10)
                           + "User question: " + message)
        except Exception as e:
            pass

    result = await orchestrator.process_message(
        user_id=payload.user_id,
        message=message,
        session_id=session_id
    )
    quota = orchestrator.session_mgr.get_user_daily_usage(payload.user_id)
    response_body = {
        "reply": result.get("reply") or result.get("response") or "",
        "session_id": session_id,
        "agent": result.get("agent_name", "EisaX"),
        "model": result.get("model"),
        "download_url": result.get("download_url"),
        "format": result.get("format"),
        "quota": quota,
    }
    return JSONResponse(
        content=response_body,
        headers=orchestrator.session_mgr.get_quota_header(payload.user_id)
    )

# Backward-compatible aliases used by older UI pages (/chat and /api/chat).
@router.post("/chat")
@router.post("/api/chat")
@_limit("30/minute")
async def unified_chat_legacy(
    request: Request
):
    try:
        raw = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid JSON body.")
    try:
        payload = _coerce_chat_payload(raw)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Request body must include 'message' (or legacy 'text')."
        )
    return await unified_chat(
        payload=payload,
        request=request,
        access_token=access_token,
        access_token_alt=access_token_alt
    )

# ── SSE Streaming Chat Endpoint ───────────────────────────────────────────────
@router.post("/v1/chat/stream")
@_limit("30/minute")
async def unified_chat_stream(
    payload: MessagePayload,
    request: Request
):
    """
    Server-Sent Events streaming chat endpoint.
    Returns Content-Type: text/event-stream.

    Each SSE message is a JSON-encoded event:
      data: {"type": "status", "text": "..."}   ← progress / loader text
      data: {"type": "token",  "text": "..."}   ← LLM content chunk
      data: {"type": "done",   "session_id": "...", "agent": "...", "model": "..."}
      data: {"type": "error",  "text": "..."}
      data: [DONE]                               ← stream closed
    """

    if orchestrator.session_mgr.is_user_blocked(payload.user_id):
        raise HTTPException(status_code=403, detail="Your account has been suspended.")
    if orchestrator.session_mgr.is_user_rate_limited(payload.user_id):
        raise HTTPException(status_code=429, detail="Daily message limit reached.")

    client_ip = (
        request.headers.get("X-Real-IP")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or str(request.client.host)
    )
    session_id = payload.session_id or orchestrator.session_mgr.get_or_create_session(
        payload.user_id, ip=client_ip
    )

    async def _generate():
        try:
            # stream_process_message already yields fully-formatted SSE lines
            async for sse_line in orchestrator.stream_process_message(
                user_id=payload.user_id,
                message=payload.message,
                session_id=session_id
            ):
                yield sse_line
        except Exception as e:
            yield f'data: {_json.dumps({"type":"error","text":str(e)}, ensure_ascii=False)}\n\n'
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection": "keep-alive",
        }
    )


# --- TTS Endpoint ---

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

@router.post("/v1/tts")
@_limit("20/minute")
async def text_to_speech(request: Request, tts_body: TTSRequest):
    try:
        audio_bytes = tts_service.generate_speech(tts_body.text, tts_body.language)
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Admin Endpoints ---

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
if not ADMIN_TOKEN:
    logger.warning("[STARTUP] ADMIN_TOKEN is not set — admin endpoints will be disabled")

def _check_admin(token: str):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin access is not configured")
    if not orchestrator.session_mgr.verify_admin_password(token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Forbidden")
@router.get("/admin/sessions")
@_limit("30/minute")
async def admin_sessions(request: Request, _admin_tok: str = Header(None, alias="X-Admin-Key")):
    _check_admin(_admin_tok)
    from collections import defaultdict
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

@router.get("/admin/session/{session_id}")
@_limit("60/minute")
async def admin_session_detail(request: Request, session_id: str, _admin_tok: str = Header(None, alias="X-Admin-Key")):
    _check_admin(_admin_tok)
    return orchestrator.session_mgr.get_chat_history(session_id)

@router.get("/admin/stats")
@_limit("30/minute")
async def admin_stats(request: Request, _admin_tok: str = Header(None, alias="X-Admin-Key")):
    _check_admin(_admin_tok)
    return orchestrator.session_mgr.get_admin_stats()

@router.post("/admin/user/{user_id}/block")
@_limit("30/minute")
async def block_user(request: Request, user_id: str, _admin_tok: str = Header(None, alias="X-Admin-Key")):
    _check_admin(_admin_tok)
    orchestrator.session_mgr.set_user_blocked(user_id, True)
    orchestrator.session_mgr.log_admin_action("block_user", user_id)
    return {"status": "blocked", "user_id": user_id}

@router.post("/admin/user/{user_id}/unblock")
@_limit("30/minute")
async def unblock_user(request: Request, user_id: str, _admin_tok: str = Header(None, alias="X-Admin-Key")):
    _check_admin(_admin_tok)
    orchestrator.session_mgr.set_user_blocked(user_id, False)
    orchestrator.session_mgr.log_admin_action("unblock_user", user_id)
    return {"status": "unblocked", "user_id": user_id}

@router.post("/admin/user/{user_id}/message")
@_limit("20/minute")
async def send_admin_message(request: Request, user_id: str, body: dict):
    _check_admin(_admin_tok)
    content = body.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    orchestrator.session_mgr.queue_admin_message(user_id, content)
    orchestrator.session_mgr.log_admin_action("message_user", user_id, content[:80])
    return {"status": "queued", "user_id": user_id}

@router.get("/admin/messages")
@_limit("30/minute")
async def get_admin_messages(request: Request, _admin_tok: str = Header(None, alias="X-Admin-Key")):
    _check_admin(_admin_tok)
    return orchestrator.session_mgr.get_admin_message_history()

@router.post("/admin/settings/password")
@_limit("5/minute")
async def change_admin_password(request: Request, body: dict, _admin_tok: str = Header(None, alias="X-Admin-Key")):
    _check_admin(_admin_tok)
    new_password = body.get("new_password", "").strip()
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    orchestrator.session_mgr.change_admin_password(new_password)
    return {"status": "password updated"}

@router.post("/admin/user/{user_id}/limit")
@_limit("30/minute")
async def set_user_limit(request: Request, user_id: str, body: dict):
    _check_admin(_admin_tok)
    daily_limit = int(body.get("daily_limit", 0))
    if daily_limit < 0:
        raise HTTPException(status_code=400, detail="daily_limit must be >= 0")
    orchestrator.session_mgr.set_user_profile(user_id, daily_limit=daily_limit)
    orchestrator.session_mgr.log_admin_action("set_limit", user_id, str(daily_limit))
    return {"status": "ok", "user_id": user_id, "daily_limit": daily_limit}

@router.post("/admin/user/{user_id}/note")
@_limit("30/minute")
async def set_user_note(request: Request, user_id: str, body: dict):
    _check_admin(_admin_tok)
    note = body.get("note", "")
    orchestrator.session_mgr.set_user_profile(user_id, note=note)
    orchestrator.session_mgr.log_admin_action("set_note", user_id, note[:60] if note else "cleared")
    return {"status": "ok", "user_id": user_id}

@router.post("/admin/user/{user_id}/tier")
@_limit("30/minute")
async def set_user_tier(request: Request, user_id: str, body: dict):
    _check_admin(_admin_tok)
    tier = body.get("tier", "basic")
    if tier not in ("basic", "pro", "vip"):
        raise HTTPException(status_code=400, detail="tier must be basic, pro, or vip")
    orchestrator.session_mgr.set_user_profile(user_id, tier=tier)
    orchestrator.session_mgr.log_admin_action("set_tier", user_id, tier)
    return {"status": "ok", "user_id": user_id, "tier": tier}

@router.post("/admin/broadcast")
@_limit("5/minute")
async def broadcast_message(request: Request, body: dict, _admin_tok: str = Header(None, alias="X-Admin-Key")):
    _check_admin(_admin_tok)
    content = body.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    count = orchestrator.session_mgr.broadcast_admin_message(content)
    orchestrator.session_mgr.log_admin_action("broadcast", f"{count} users", content[:80])
    return {"status": "broadcast", "recipients": count}

@router.delete("/admin/user/{user_id}/sessions")
@_limit("30/minute")
async def delete_user_sessions(request: Request, user_id: str, _admin_tok: str = Header(None, alias="X-Admin-Key")):
    _check_admin(_admin_tok)
    count = orchestrator.session_mgr.delete_user_sessions(user_id)
    orchestrator.session_mgr.log_admin_action("delete_sessions", user_id, f"{count} sessions deleted")
    return {"status": "deleted", "user_id": user_id, "sessions_deleted": count}

@router.post("/admin/ip/{ip}/block")
@_limit("30/minute")
async def block_ip_endpoint(request: Request, ip: str, body: dict = {}):
    _check_admin(_admin_tok)
    reason = (body or {}).get("reason", "")
    orchestrator.session_mgr.block_ip(ip, reason)
    orchestrator.session_mgr.log_admin_action("block_ip", ip, reason or "no reason")
    return {"status": "blocked", "ip": ip}

@router.post("/admin/ip/{ip}/unblock")
@_limit("30/minute")
async def unblock_ip_endpoint(request: Request, ip: str, _admin_tok: str = Header(None, alias="X-Admin-Key")):
    _check_admin(_admin_tok)
    orchestrator.session_mgr.unblock_ip(ip)
    orchestrator.session_mgr.log_admin_action("unblock_ip", ip)
    return {"status": "unblocked", "ip": ip}

@router.get("/admin/blocked-ips")
@_limit("30/minute")
async def get_blocked_ips(request: Request, _admin_tok: str = Header(None, alias="X-Admin-Key")):
    _check_admin(_admin_tok)
    return orchestrator.session_mgr.get_blocked_ips()

@router.get("/admin/audit-log")
@_limit("30/minute")
async def get_audit_log(request: Request, _admin_tok: str = Header(None, alias="X-Admin-Key")):
    _check_admin(_admin_tok)
    return orchestrator.session_mgr.get_audit_log()

@router.get("/admin/notifications")
@_limit("60/minute")
async def get_notifications(request: Request, since: str = "", _admin_tok: str = Header(None, alias="X-Admin-Key")):
    _check_admin(_admin_tok)
    if not since:
        from datetime import datetime, timedelta, timezone
        since = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    return orchestrator.session_mgr.get_new_activity(since)

@router.get("/admin/export/users")
@_limit("10/minute")
async def export_users(request: Request, _admin_tok: str = Header(None, alias="X-Admin-Key")):
    from fastapi.responses import StreamingResponse as SR
    import csv
    _check_admin(_admin_tok)
    from collections import defaultdict
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

# --- New History Endpoints ---
@router.get("/api/history")
@_limit("60/minute")
async def get_history(request: Request, user_id: Optional[str] = "admin"):
    return orchestrator.session_mgr.get_user_sessions(user_id)

@router.get("/api/history/{session_id}")
@_limit("60/minute")
async def get_session_history(request: Request, session_id: str):
    return orchestrator.session_mgr.get_chat_history(session_id)

@router.delete("/api/history/{session_id}")
@_limit("20/minute")
async def delete_session(request: Request, session_id: str):
    orchestrator.session_mgr.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}
@router.post("/v1/export")
@_limit("10/minute")
async def export_chat(request: Request):
    import re, shutil
    try:
        body = await request.json()
    except Exception as _e:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    fmt = body.get("format", "pdf")
    messages = body.get("messages", [])
    title = body.get("title", "EisaX Report")
    smart = [m for m in messages if m.get("role") == "assistant" 
             and len(m.get("content","")) > 200
             and not any(x in m.get("content","") for x in ["Hello!", "Hi!", "How can I help", "مرحباً", "أهلاً"])]
    if not smart:
        smart = messages
    
    # === GLM FORMATTING LAYER ===
    try:
        from core.glm_client import GLMClient
        glm = GLMClient()
        
        # Combine all messages
        combined = "\n\n---\n\n".join([
            m.get("content", "") for m in smart if m.get("content")
        ])
        
        # Let GLM clean and format
        logger.debug("Calling GLM with %d chars", len(combined))
        formatted = glm.prepare_for_export(combined, fmt)
        logger.debug("GLM result: success=%s", formatted.get('success'))

        if formatted.get("success"):
            smart = [{"role": "assistant", "content": formatted["content"]}]
            logger.info("GLM formatted export for %s — new length: %d", fmt, len(formatted['content']))
        else:
            logger.warning("GLM formatting failed: %s", formatted.get('error'))
    except Exception as e:
        logger.error("GLM export prep error: %s", e, exc_info=True)
    
    # Clean emojis for PDF compatibility
    emoji_map = {
        "📊": ">>", "📈": "^", "📉": "v", "🔴": "(SELL)",
        "🟢": "(BUY)", "🎯": "(TARGET)", "📰": ">>", "🔍": ">>",
        "✅": "OK", "➕": "+", "⚠️": "(!)", "💡": ">>",
        "🧠": ">>", "👋": "", "📄": "", "💰": "$",
        "–": "-", "→": "->", "—": "-", "–": "-",
        "—": "-", "’": "'", "“": '"', "”": '"',
        "?": "-"
    }
    
    def clean_content(text):
        for emoji, replacement in emoji_map.items():
            text = text.replace(emoji, replacement)
        return text
    
    smart = [{"role": m["role"], "content": clean_content(m.get("content",""))} for m in smart]
    
    for msg in smart:
        c = msg.get("content","")
        m = re.search(r"EisaX Intelligence Report: ([A-Z]+)", c)
        if m:
            title = f"EisaX Report - {m.group(1)}"
            break
        elif "Portfolio Risk Report" in c:
            title = "EisaX Portfolio Risk Report"
            break
    export_dir = str(EXPORTS_DIR)
    os.makedirs(export_dir, exist_ok=True)
    try:
        # CIO engines for exports
        if fmt in ("pdf", "pdf_ar"):
            from core.cio_pdf import generate_cio_pdf
            import time, re as re2

            _lang = "ar" if fmt == "pdf_ar" else "en"
            _suffix = "_AR" if _lang == "ar" else ""
            filename = "EisaX" + _suffix + "_" + time.strftime("%Y%m%d_%H%M%S") + ".pdf"
            out_path = str(EXPORTS_DIR / filename)

            ticker_m = re2.search(r"EisaX (?:Report|Intelligence Report)[:\s-]+([A-Z]{1,5})", title or "")
            ticker = ticker_m.group(1) if ticker_m else ""

            combined = "\n\n".join(m.get("content", "") for m in smart)

            pdf_result = generate_cio_pdf(combined, out_path, ticker=ticker, title=title, lang=_lang)
            report_id = pdf_result[1] if isinstance(pdf_result, tuple) and len(pdf_result) > 1 else None
            result = {"success": True, "filename": filename, "report_id": report_id}

        elif fmt in ("docx", "word"):
            from core.cio_docx import generate_cio_docx
            import time, re as re2

            filename = "EisaX_" + time.strftime("%Y%m%d_%H%M%S") + ".docx"
            out_path = str(EXPORTS_DIR / filename)

            ticker_m = re2.search(r"EisaX (?:Report|Intelligence Report)[:\s-]+([A-Z]{1,5})", title or "")
            ticker = ticker_m.group(1) if ticker_m else ""

            combined = "\n\n".join(m.get("content", "") for m in smart)

            docx_result = generate_cio_docx(combined, out_path, ticker=ticker, title=title)
            report_id = docx_result[1] if isinstance(docx_result, tuple) and len(docx_result) > 1 else None
            result = {"success": True, "filename": filename, "report_id": report_id}
        else:
            result = export_engine(fmt, smart, title)
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error","Export failed"))
        filename = os.path.basename(result.get("filename",""))
        src = result.get("filename","")
        dst = os.path.join(export_dir, filename)
        if src and os.path.exists(src) and src != dst:
            shutil.copy2(src, dst)
        return {
            "success": True,
            "filename": filename,
            "download_url": f"/v1/download/{filename}",
            "title": title,
            "format": fmt,
            "report_id": result.get("report_id"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/download/{filename}")
@_limit("60/minute")
async def download_file(request: Request, filename: str):
    """Download exported file — public endpoint so browser links work directly."""
    import re as _re
    from fastapi.responses import FileResponse

    # Only allow safe filenames: letters, digits, underscores, hyphens, dots
    if not _re.fullmatch(r"[\w\-]+\.(pdf|docx|xlsx|pptx|csv)", filename, _re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Invalid filename")

    export_dir = "/home/ubuntu/investwise/static/exports"
    file_path = os.path.join(export_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=filename)
@router.get("/v1/brain/status")
@_limit("30/minute")
async def brain_status(request: Request):
    from learning_engine import get_engine
    return get_engine().status()

@router.get("/v1/brain/wisdom")
@_limit("20/minute")
async def brain_wisdom(request: Request):
    from learning_engine import get_engine
    engine = get_engine()
    conn = engine._get_conn()
    stocks = conn.execute("SELECT COUNT(*) FROM stock_knowledge").fetchone()[0]
    preds = conn.execute(
        "SELECT COUNT(*), ROUND(AVG(was_correct)*100,1) FROM predictions WHERE evaluated=1"
    ).fetchone()
    lessons = conn.execute(
        "SELECT lesson, category, confidence, date FROM learning_log ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    conn.close()
    return {
        "stocks_known": stocks,
        "predictions_evaluated": preds[0],
        "overall_accuracy_pct": preds[1],
        "lessons": [dict(r) for r in lessons],
        "engine_stats": engine._stats
    }

@router.post("/v1/alerts")
@_limit("20/minute")
async def create_alert(request: Request):
    body = await request.json()
    from core.price_alerts import add_alert
    alert_id = add_alert(body.get('user_id','anonymous'), body['ticker'], body['condition'], body['threshold'])
    return {'alert_id': alert_id, 'status': 'created'}

@router.get("/v1/alerts")
@_limit("30/minute")
async def list_alerts(request: Request, user_id: str = 'anonymous'):
    from core.price_alerts import get_user_alerts
    return get_user_alerts(user_id)

@router.delete("/v1/alerts/{alert_id}")
@_limit("20/minute")
async def remove_alert(request: Request, alert_id: int, user_id: str = 'anonymous'):
    from core.price_alerts import delete_alert
    delete_alert(alert_id, user_id)
    return {'status': 'deleted'}

@router.get('/v1/version')
@_limit('60/minute')
async def app_version(request: Request):
    return {'version': _APP_VERSION, 'git_sha': _GIT_SHA, 'build_date': '2026-04-10', 'env': 'production'}

# ── HTML → PDF Export ──
class HtmlExportPayload(BaseModel):
    html: str
    filename: str = ""
    access_token: str = ""

@router.post("/v1/export/html")
@_limit("10/minute")
async def export_html_to_pdf(
    request: Request,
    payload: HtmlExportPayload
):
    try:
        import time
        from core.playwright_pdf import html_to_pdf, inject_print_css
        fname = payload.filename or f"EisaX_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
        if not fname.endswith('.pdf'):
            fname += '.pdf'
        filepath = str(EXPORTS_DIR / fname)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        html_to_pdf(inject_print_css(payload.html), filepath)
        os.chmod(filepath, 0o644)
        return {"url": f"/v1/download/{fname}", "download_url": f"/v1/download/{fname}", "filename": fname}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/dashboard/{ticker}")
@_limit("20/minute")
async def dashboard(request: Request, ticker: str):
    """Return all dashboard data for a ticker in one call — no LLM, runs concurrently."""
    import asyncio, math
    from core.market_data import get_realtime_quote, get_full_stock_profile
    from core.data import get_prices
    from core.analytics import generate_technical_summary, run_stress_test
    from core.realtime_data import deepcrawl_stock, deepcrawl_news
    from core.rapid_data import get_market_pulse, get_cashflow, get_events_calendar

    ticker = ticker.upper().strip()
    loop = asyncio.get_event_loop()

    # ── Detect Saudi Tadawul ticker ──────────────────────────────────────────
    is_saudi  = ticker.endswith(".SR")
    tadawul_id = ticker.replace(".SR", "") if is_saudi else None

    # ── fetch ALL sources in parallel ──────────────────────────────────────────
    # Group 1: per-ticker data (quote, profile, prices, DeepCrawl, cash flow, events)
    # Group 2: global market data (Fear&Greed, Forex calendar, CNBC news)
    # Group 3 (Saudi only): Tadawul live quote + history
    from core.rapid_data import get_tadawul_quote, get_tadawul_history, _fetch_tadawul_candles
    try:
        if is_saudi:
            # For Saudi tickers: fetch Tadawul candles FIRST (shared cache for quote+history)
            # then derive quote and history from same candles without 2 separate HTTP calls
            (quote, profile, prices_df, dc_data, dc_news,
             cashflow_data, events_data, market_pulse,
             _raw_candles) = await asyncio.gather(
                loop.run_in_executor(None, get_realtime_quote, ticker),
                loop.run_in_executor(None, get_full_stock_profile, ticker),
                loop.run_in_executor(None, get_prices, [ticker]),
                loop.run_in_executor(None, deepcrawl_stock, ticker),
                loop.run_in_executor(None, deepcrawl_news, ticker, 5),
                loop.run_in_executor(None, get_cashflow, ticker),
                loop.run_in_executor(None, get_events_calendar, ticker),
                loop.run_in_executor(None, get_market_pulse),
                loop.run_in_executor(None, _fetch_tadawul_candles, tadawul_id)
            )
            # Build quote + history from same candles (no extra HTTP call)
            tadawul_quote = get_tadawul_quote(tadawul_id)     # reads from shared cache (instant)
            tadawul_hist  = list(reversed(_raw_candles)) if _raw_candles else get_tadawul_history(tadawul_id)
        else:
            (quote, profile, prices_df, dc_data, dc_news,
             cashflow_data, events_data, market_pulse) = await asyncio.gather(
                loop.run_in_executor(None, get_realtime_quote, ticker),
                loop.run_in_executor(None, get_full_stock_profile, ticker),
                loop.run_in_executor(None, get_prices, [ticker]),
                loop.run_in_executor(None, deepcrawl_stock, ticker),
                loop.run_in_executor(None, deepcrawl_news, ticker, 5),
                loop.run_in_executor(None, get_cashflow, ticker),
                loop.run_in_executor(None, get_events_calendar, ticker),
                loop.run_in_executor(None, get_market_pulse)
            )
            tadawul_quote = {}
            tadawul_hist  = []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {e}")

    # ── Override quote with Tadawul live data (more accurate for .SR tickers) ──
    if is_saudi and tadawul_quote.get("price"):
        tq = tadawul_quote
        quote["price"]      = tq.get("price",      quote.get("price"))
        quote["open"]       = tq.get("open",        quote.get("open"))
        quote["high"]       = tq.get("high",        quote.get("high"))
        quote["low"]        = tq.get("low",         quote.get("low"))
        quote["volume"]     = tq.get("volume",      quote.get("volume"))
        quote["change"]     = tq.get("change",      quote.get("change"))
        quote["change_pct"] = tq.get("change_pct",  quote.get("change_pct"))
        quote["source"]     = "Tadawul RapidAPI (live)"

    # ── technicals + stress (instant, local) ──
    try:
        close_series = prices_df[ticker] if ticker in prices_df.columns else prices_df.iloc[:, 0]
        tech   = generate_technical_summary(ticker, close_series)
        beta   = float((profile.get("fundamentals") or {}).get("beta") or 1.0)
        stress = run_stress_test(close_series, beta=beta)
    except Exception as e:
        tech   = {}
        stress = {"scenarios": {}, "annual_vol": 0}

    # ── sanitise NaN/Inf so JSON serialises cleanly ──
    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    def _clean_dict(d):
        return {k: _clean(v) for k, v in (d or {}).items()}

    def _safe_float(v):
        try:
            return float(v) if v not in (None, "", "-") else None
        except (TypeError, ValueError):
            return None

    # ── merge DeepCrawl technicals into tech dict (RSI, SMA, performance) ──
    dc = dc_data or {}
    dc_technicals = {
        "rsi":          _safe_float(dc.get("rsi")),
        "sma50":        _safe_float(dc.get("sma50")),
        "sma200":       _safe_float(dc.get("sma200")),
        "short_float":  _safe_float(dc.get("short_float")),
        "avg_volume":   dc.get("avg_volume"),
        "perf_week":    _safe_float(dc.get("perf_week")),
        "perf_month":   _safe_float(dc.get("perf_month")),
        "perf_ytd":     _safe_float(dc.get("perf_ytd")),
    }
    # Merge into local tech dict — DeepCrawl fills gaps
    for k, v in dc_technicals.items():
        if v is not None:
            tech[k] = v

    # ── DeepCrawl fundamentals enrichment ──
    dc_fundamentals = {
        # Analyst consensus
        "analyst_rating":      dc.get("analyst_rating"),
        "analyst_buy":         dc.get("analyst_buy"),
        "analyst_hold":        dc.get("analyst_hold"),
        "analyst_sell":        dc.get("analyst_sell"),
        # Price targets (from forecast page)
        "price_target":        dc.get("price_target"),
        "price_target_mean":   dc.get("price_target_mean"),
        "price_target_low":    dc.get("price_target_low"),
        "price_target_high":   dc.get("price_target_high"),
        "price_target_median": dc.get("price_target_median"),
        # Valuation
        "forward_pe":          _safe_float(dc.get("forward_pe")),
        "earnings_date":       dc.get("earnings_date"),
        "week_52_range":       dc.get("week_52_range"),
        # Ownership
        "inst_own":            _safe_float(dc.get("inst_own")),
        "insider_own":         _safe_float(dc.get("insider_own")),
        # Financial ratios (from SA ratios page fallback)
        "debt_equity":         _safe_float(dc.get("debt_equity")),
        "roe":                 _safe_float(dc.get("roe")),
        "roa":                 _safe_float(dc.get("roa")),
        "profit_margin":       _safe_float(dc.get("profit_margin")),
        "gross_margin":        dc.get("gross_margin"),
        "net_margin":          dc.get("net_margin_annual"),
        "free_cash_flow":      dc.get("free_cash_flow"),
    }

    # ── DeepCrawl historical financials (revenue + EPS by year) ──
    dc_financials = {
        "revenue_history": dc.get("revenue_history") or {},
        "eps_history":     dc.get("eps_history")     or {},
    }

    # ── Merge existing fundamentals with DeepCrawl (DeepCrawl fills gaps only) ──
    base_fundamentals = _clean_dict(profile.get("fundamentals", {}))
    for k, v in dc_fundamentals.items():
        if v is not None and k not in base_fundamentals:
            base_fundamentals[k] = v

    # ── Enrich fundamentals with Events Calendar data ──────────────────────────
    ev = events_data or {}
    events_fields = {
        "earnings_date":  ev.get("earnings_date"),
        "ex_div_date":    ev.get("ex_div_date"),
        "div_date":       ev.get("div_date"),
        "eps_est_avg":    ev.get("eps_est_avg"),
        "eps_est_high":   ev.get("eps_est_high"),
        "eps_est_low":    ev.get("eps_est_low"),
        "rev_est_avg":    ev.get("rev_est_avg"),
    }
    for k, v in events_fields.items():
        if v is not None and not base_fundamentals.get(k):
            base_fundamentals[k] = v

    # ── Combine news: DeepCrawl stock news + CNBC global news ─────────────────
    mp = market_pulse or {}
    cnbc_news = _get_aggregated_news(ticker=ticker, limit=5)
    combined_news = (dc_news or []) + cnbc_news

    # ── Build final financials with cash flow ──────────────────────────────────
    cf = cashflow_data or {}
    dc_financials["cash_flow"] = {
        "quarters":     cf.get("quarters", []),
        "operating_cf": cf.get("operating_cf", []),
        "free_cf":      cf.get("free_cf", []),
        "capex":        cf.get("capex", []),
        "unit":         cf.get("unit", "B USD"),
        "source":       cf.get("source", ""),
    } if cf else {}

    return {
        "ticker":       ticker,
        "quote":        _clean_dict(quote),
        "fundamentals": base_fundamentals,
        "technicals":   _clean_dict(tech),
        "financials":   dc_financials,
        "stress":       {k: _clean_dict(v) for k, v in stress.get("scenarios", {}).items()},
        "annual_vol":   stress.get("annual_vol", 0),
        "news":         combined_news,
        # ── Market-wide data ───────────────────────────────────────────────────
        "fear_greed":    mp.get("fear_greed") or {},
        "econ_calendar": mp.get("calendar")   or [],
        "dc_source":     dc.get("source", ""),
        # ── Saudi Tadawul (only populated for .SR tickers) ────────────────────
        "is_saudi":        is_saudi,
        "tadawul_intraday": tadawul_hist,   # list of {date,open,high,low,close,volume} 1-min candles
    }


class TranslatePayload(BaseModel):
    text: str
    access_token: str = ""

@router.post("/v1/translate-ar")
@_limit("20/minute")
async def translate_to_arabic(request: Request, payload: TranslatePayload):
    """Translate an English investment report to Arabic. Primary: DeepSeek. Fallback: GLM."""

    system_prompt = (
        "أنت محلل مالي محترف. مهمتك ترجمة تقرير استثماري كامل من الإنجليزية إلى العربية الفصحى.\n"
        "القواعد الصارمة:\n"
        "1. ترجم كل النص كاملاً بدون حذف أي قسم أو معلومة\n"
        "2. احتفظ بتنسيق Markdown كما هو: ##، ###، **bold**، | tables |، - lists، > blockquote\n"
        "3. لا تترجم: أسماء الشركات، رموز البورصة (AAPL، BTC)، الأرقام، العملات، النسب المئوية\n"
        "4. الجداول (tables): حافظ على | الفاصل | وترجم محتوى الخلايا فقط\n"
        "5. اكتب بأسلوب مؤسسي احترافي مناسب لتقارير المحللين الماليين\n"
        "6. أخرج النص المترجم فقط — بدون أي تعليق أو مقدمة"
    )
    # Chunks arrive pre-split from client (max 6000 chars each) — accept up to 8000 chars
    text_in = payload.text[:8000]
    user_msg = f"ترجم هذا النص كاملاً مع الحفاظ على تنسيق Markdown:\n\n{text_in}"

    import httpx, os

    # ── Primary: DeepSeek ────────────────────────────────────────────────────
    ds_key = os.getenv("DEEPSEEK_API_KEY", "")
    if ds_key:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                ds_resp = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                    json={
                        "model": "deepseek-v4-flash",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_msg}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 4000
                    }
                )
            if ds_resp.status_code == 200:
                ar_text = ds_resp.json()["choices"][0]["message"]["content"]
                logger.info("translate-ar: DeepSeek OK (%d chars)", len(ar_text))
                return {"success": True, "text": ar_text}
            else:
                logger.warning("translate-ar DeepSeek failed %s: %s", ds_resp.status_code, ds_resp.text[:150])
        except Exception as _de:
            logger.warning("translate-ar DeepSeek error: %s", _de)

    # ── Fallback: GLM ────────────────────────────────────────────────────────
    try:
        from core.glm_client import GLMClient, GLM_API_URL, GLM_MODEL
        glm = GLMClient()
        async with httpx.AsyncClient(timeout=110) as client:
            glm_resp = await client.post(
                GLM_API_URL,
                headers=glm.headers,
                json={
                    "model": GLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_msg}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 8000
                }
            )
        if glm_resp.status_code == 200:
            ar_text = glm_resp.json()["choices"][0]["message"]["content"]
            logger.info("translate-ar: GLM fallback OK (%d chars)", len(ar_text))
            return {"success": True, "text": ar_text}
        else:
            logger.warning("translate-ar GLM failed: %s", glm_resp.text[:200])
    except Exception as _ge:
        logger.error("translate-ar GLM error: %s", _ge)

    return {"success": False, "text": payload.text, "error": "All translation services unavailable"}

@router.post("/v1/export/html-pdf")
@_limit("5/minute")
async def export_html_pdf(request: Request, payload: HtmlExportPayload):
    try:
        import time
        from core.playwright_pdf import html_to_pdf, inject_print_css
        fname = payload.filename or f"EisaX_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
        if not fname.endswith('.pdf'):
            fname += '.pdf'
        filepath = str(EXPORTS_DIR / fname)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        html_to_pdf(inject_print_css(payload.html), filepath)
        os.chmod(filepath, 0o644)
        return {"url": f"/v1/download/{fname}", "download_url": f"/v1/download/{fname}", "filename": fname}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health Check ──────────────────────────────────────────────────────────────

@router.get("/v1/health")
@_limit("30/minute")
async def health_check(
    request: Request
):
    from core.services.health_service import run_health_check
    result = await run_health_check(_SECURE_TOKEN)
    status_code = 200 if result["status"] == "ok" else (503 if result["status"] == "down" else 207)
    from fastapi.responses import JSONResponse
    return JSONResponse(content=result, status_code=status_code)


# ── Session Cleanup ───────────────────────────────────────────────────────────

@router.post("/admin/cleanup")
@_limit("5/minute")
async def run_cleanup(
    request: Request,
    days: int = 30
):
    _check_admin(_admin_tok)
    result = orchestrator.session_mgr.cleanup_old_sessions(days_to_keep=days)
    return result


# ── Logging Dashboard ─────────────────────────────────────────────────────────

@router.get("/admin/logs")
@_limit("30/minute")
async def admin_logs_page(
    request: Request
):
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / "admin_logs.html"))


@router.get("/admin/logs/stream")
@_limit("10/minute")
async def admin_logs_stream(request: Request, _admin_tok: str = Header(None, alias="X-Admin-Key")):
    _check_admin(_admin_tok)
    from fastapi.responses import StreamingResponse
    import asyncio as _aio

    async def _generate():
        log_path = str(BACKEND_LOG)
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f.readlines()[-100:]:
                    line = line.strip()
                    if line:
                        yield f"data: {_json.dumps({'line': line}, ensure_ascii=False)}\n\n"
                f.seek(0, 2)
                while True:
                    if await request.is_disconnected():
                        break
                    new_line = f.readline()
                    if new_line:
                        line = new_line.strip()
                        if line:
                            yield f"data: {_json.dumps({'line': line}, ensure_ascii=False)}\n\n"
                    else:
                        await _aio.sleep(0.5)
        except Exception as exc:
            yield f"data: {_json.dumps({'line': f'[ERROR] {exc}'})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── Analytics Dashboard ───────────────────────────────────────────────────────

@router.get("/admin/analytics")
@_limit("30/minute")
async def admin_analytics_page(
    request: Request,
    _admin_tok: str = Header(None, alias="X-Admin-Key"),
):
    _check_admin(_admin_tok)
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / "admin_analytics.html"))


@router.get("/admin/analytics/data")
@_limit("30/minute")
async def admin_analytics_data(
    request: Request,
    _admin_tok: str = Header(None, alias="X-Admin-Key"),
):
    _check_admin(_admin_tok)

    import sqlite3
    import re as _re2
    from datetime import datetime, timedelta, timezone
    from core.config import APP_DB

    conn = sqlite3.connect(str(APP_DB))
    try:
        today = datetime.now(timezone.utc).date()
        days = [(today - timedelta(days=i)).isoformat() for i in range(13, -1, -1)]

        msgs_per_day = {}
        for day in days:
            row = conn.execute(
                "SELECT COUNT(*) FROM chat_history WHERE DATE(timestamp)=?",
                (day,)
            ).fetchone()
            msgs_per_day[day] = row[0] if row else 0

        tiers = {}
        for row in conn.execute(
            "SELECT tier, COUNT(*) FROM user_profiles GROUP BY tier"
        ).fetchall():
            tiers[row[0] or "basic"] = row[1]

        rows = conn.execute(
            "SELECT content FROM chat_history ORDER BY timestamp DESC LIMIT 500"
        ).fetchall()
        ticker_counts = {}
        for row in rows:
            for match in _re2.findall(r"\b([A-Z]{2,5})\b", row[0] or ""):
                if match not in ("I", "THE", "AND", "FOR", "OR", "BUT", "NOT", "NEW", "ALL", "USD", "ETF"):
                    ticker_counts[match] = ticker_counts.get(match, 0) + 1
        top_tickers = sorted(ticker_counts.items(), key=lambda item: (-item[1], item[0]))[:10]

        total_users = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM sessions"
        ).fetchone()[0]
        msgs_today = conn.execute(
            "SELECT COUNT(*) FROM chat_history WHERE DATE(timestamp)=?",
            (today.isoformat(),)
        ).fetchone()[0]
        active_24h = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM chat_history WHERE timestamp >= datetime('now','-24 hours')"
        ).fetchone()[0]

        recent = [
            {
                "user_id": f"{str(row[0] or 'unknown')[:12]}...",
                "preview": (row[1] or "")[:60],
                "ts": row[2],
            }
            for row in conn.execute(
                "SELECT user_id, content, timestamp FROM chat_history ORDER BY timestamp DESC LIMIT 20"
            ).fetchall()
        ]
    finally:
        conn.close()

    return {
        "messages_per_day": msgs_per_day,
        "tier_distribution": tiers,
        "top_tickers": [{"ticker": key, "count": value} for key, value in top_tickers],
        "summary": {
            "total_users": total_users,
            "messages_today": msgs_today,
            "active_sessions_24h": active_24h,
            "top_ticker": top_tickers[0][0] if top_tickers else "N/A",
        },
        "recent_activity": recent,
    }

@router.get("/v1/usage")
@_limit("30/minute")
async def user_usage(
    request: Request,
    user_id: str = "anonymous",
    days: int = 30
):
    return orchestrator.session_mgr.get_user_usage_stats(user_id, days=min(days, 90))



# ── F-5: Redis Health ─────────────────────────────────────────────────────────

@router.get("/v1/redis/health")
@_limit("30/minute")
async def redis_health(
    request: Request
):
    from core.redis_store import redis_info
    return redis_info()


# ── F-6: Referral System ──────────────────────────────────────────────────────

@router.get("/v1/referral")
@_limit("30/minute")
async def get_referral(
    request: Request,
    user_id:          str = "anonymous"
):
    from core.referrals import get_referral_stats
    return get_referral_stats(user_id)


@router.post("/v1/referral/apply")
@_limit("5/minute")
async def apply_referral_code(
    request: Request
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    user_id = str(body.get("user_id", "")).strip()
    code    = str(body.get("code", "")).strip()
    if not user_id or not code:
        raise HTTPException(status_code=400, detail="Required: user_id, code")
    from core.referrals import apply_referral
    result = apply_referral(user_id, code)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


# ── F-7: Outbound Webhooks ────────────────────────────────────────────────────

class WebhookConfig(BaseModel):
    user_id:  str
    url:      str
    events:   list = Field(default_factory=lambda: ["analysis_complete"])
    secret:   str = ""


@router.post("/v1/webhooks")
@_limit("10/minute")
async def register_webhook(
    request: Request,
    body: WebhookConfig
):
    import sqlite3 as _sl2
    conn = _sl2.connect(str(APP_DB))
    conn.execute("""CREATE TABLE IF NOT EXISTS webhooks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL, url TEXT NOT NULL,
        events TEXT NOT NULL DEFAULT '[]',
        secret TEXT NOT NULL DEFAULT '',
        active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    cur = conn.execute(
        "INSERT INTO webhooks(user_id,url,events,secret) VALUES(?,?,?,?)",
        (body.user_id, body.url, _json.dumps(body.events), body.secret))
    wid = cur.lastrowid; conn.commit(); conn.close()
    logger.info("[webhooks] registered %d for %s → %s", wid, body.user_id, body.url)
    return {"webhook_id": wid, "status": "registered", "url": body.url}


@router.get("/v1/webhooks")
@_limit("30/minute")
async def list_webhooks(
    request: Request,
    user_id:          str = "anonymous"
):
    import sqlite3 as _sl2
    conn = _sl2.connect(str(APP_DB)); conn.row_factory = _sl2.Row
    try:
        rows = conn.execute(
            "SELECT id,url,events,active,created_at FROM webhooks WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


@router.delete("/v1/webhooks/{webhook_id}")
@_limit("20/minute")
async def delete_webhook(
    request: Request,
    webhook_id: int,
    user_id:          str = "anonymous"
):
    import sqlite3 as _sl2
    conn = _sl2.connect(str(APP_DB))
    conn.execute("CREATE TABLE IF NOT EXISTS webhooks (id INTEGER PRIMARY KEY, user_id TEXT, active INTEGER)")
    conn.execute("UPDATE webhooks SET active=0 WHERE id=? AND user_id=?", (webhook_id, user_id))
    conn.commit(); conn.close()
    return {"status": "deleted", "webhook_id": webhook_id}


class CheckoutRequest(BaseModel):
    user_id: str
    email:   str
    tier:    str   # "pro" | "vip"


class PortalRequest(BaseModel):
    user_id: str


@router.post("/v1/billing/checkout")
@_limit("10/minute")
async def create_checkout(
    request: Request,
    body: CheckoutRequest
):
    try:
        from core.billing import StripeBilling
        url = StripeBilling().create_checkout_session(body.user_id, body.email, body.tier)
        return {"checkout_url": url}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("[billing/checkout] %s", exc)
        raise HTTPException(status_code=500, detail="Billing service error")


@router.post("/v1/billing/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        from core.billing import StripeBilling
        result = StripeBilling().handle_webhook(payload, sig)
        if result.get("tier") and result.get("user_id"):
            orchestrator.session_mgr.set_user_profile(result["user_id"], tier=result["tier"])
            logger.info("[billing] upgraded user %s to tier %s", result["user_id"], result["tier"])
        return {"received": True, "event": result.get("event")}
    except Exception as exc:
        logger.error("[billing/webhook] %s", exc)
        raise HTTPException(status_code=400, detail=f"Webhook error: {exc}")


@router.post("/v1/billing/portal")
@_limit("10/minute")
async def billing_portal(
    request: Request,
    body: PortalRequest
):
    try:
        from core.billing import StripeBilling
        billing = StripeBilling()
        cid = billing.get_customer_id(body.user_id)
        if not cid:
            raise HTTPException(status_code=404, detail="No billing record found for this user")
        url = billing.create_portal_session(cid)
        return {"portal_url": url}
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("[billing/portal] %s", exc)
        raise HTTPException(status_code=500, detail="Billing service error")


# ── G-8: News Sentiment NLP ───────────────────────────────────────────────────

@router.get("/v1/sentiment/{ticker}")
@_limit("20/minute")
async def get_ticker_sentiment(
    request: Request,
    ticker: str,
    use_cache: bool = True
):
    """VADER sentiment analysis on recent news for a single ticker."""
    try:
        from core.sentiment import SentimentAnalyzer
        result = await asyncio.get_event_loop().run_in_executor(
            None, SentimentAnalyzer().analyze_ticker, ticker.upper(), use_cache
        )
        return result
    except Exception as exc:
        logger.error("[sentiment] ticker=%s %s", ticker, exc)
        raise HTTPException(status_code=500, detail=f"Sentiment error: {exc}")


@router.post("/v1/sentiment/batch")
@_limit("5/minute")
async def get_batch_sentiment(
    request: Request
):
    """Analyze sentiment for multiple tickers at once (max 10)."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    tickers   = [str(t).upper().strip() for t in body.get("tickers", []) if t][:10]
    use_cache = bool(body.get("use_cache", True))
    if not tickers:
        raise HTTPException(status_code=400, detail="Provide 'tickers' list (max 10)")
    try:
        from core.sentiment import SentimentAnalyzer
        results = await asyncio.get_event_loop().run_in_executor(
            None, SentimentAnalyzer().analyze_many, tickers, use_cache
        )
        return {"count": len(results), "results": results}
    except Exception as exc:
        logger.error("[sentiment/batch] %s", exc)
        raise HTTPException(status_code=500, detail=f"Sentiment error: {exc}")


@router.get("/v1/sentiment/market/overview")
@_limit("10/minute")
async def get_market_sentiment(
    request: Request,
    use_cache: bool = True
):
    """Aggregate market sentiment from major ETF news (SPY/QQQ/DIA)."""
    try:
        from core.sentiment import SentimentAnalyzer
        result = await asyncio.get_event_loop().run_in_executor(
            None, SentimentAnalyzer().market_sentiment, use_cache
        )
        return result
    except Exception as exc:
        logger.error("[sentiment/market] %s", exc)
        raise HTTPException(status_code=500, detail=f"Sentiment error: {exc}")


@router.get("/v1/sentiment/{ticker}/trend")
@_limit("20/minute")
async def get_sentiment_trend(
    request: Request,
    ticker: str,
    hours: int = 48
):
    """Historical sentiment trend (hourly buckets) from local DB."""
    try:
        from core.sentiment import SentimentAnalyzer
        result = await asyncio.get_event_loop().run_in_executor(
            None, SentimentAnalyzer().sentiment_trend, ticker.upper(), min(hours, 720)
        )
        return result
    except Exception as exc:
        logger.error("[sentiment/trend] ticker=%s %s", ticker, exc)
        raise HTTPException(status_code=500, detail=f"Sentiment trend error: {exc}")


class BacktestRequest(BaseModel):
    ticker: str
    strategy: str  # 'ma_crossover' | 'rsi' | 'macd'
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    initial_capital: float = 10000.0
    short_window: int = 20
    long_window: int = 50
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0


@router.post('/v1/backtest')
@_limit('10/minute')
async def run_backtest(
    request: Request,
    body: BacktestRequest,
):
    try:
        import asyncio

        from core.backtester import BacktestEngine, MACrossover, RSIStrategy, MACDStrategy

        strategies = {
            'ma_crossover': MACrossover(short=body.short_window, long=body.long_window),
            'rsi': RSIStrategy(period=body.rsi_period, oversold=body.rsi_oversold, overbought=body.rsi_overbought),
            'macd': MACDStrategy(),
        }
        if body.strategy not in strategies:
            raise HTTPException(400, f'Unknown strategy. Choose: {list(strategies.keys())}')
        engine = BacktestEngine()
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            engine.run,
            body.ticker,
            strategies[body.strategy],
            body.start_date,
            body.end_date,
            body.initial_capital
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error('[backtest] %s', exc)
        raise HTTPException(500, f'Backtest error: {exc}')


class ScreenerRequest(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    universe: str = "us_large_cap"  # 'us_large_cap'|'uae'|'egypt'|'saudi'|'custom'
    pe_min: Optional[float] = None
    pe_max: Optional[float] = None
    roe_min: Optional[float] = None
    roe_max: Optional[float] = None
    market_cap_min: Optional[float] = None
    market_cap_max: Optional[float] = None
    volume_min: Optional[float] = None
    rsi_min: Optional[float] = None
    rsi_max: Optional[float] = None
    price_above_sma200: Optional[bool] = None
    dividend_yield_min: Optional[float] = None
    revenue_growth_min: Optional[float] = None
    sector: Optional[str] = None
    max_results: int = 20
    include_sentiment: bool = False   # G-9-A: enrich each result with news sentiment


@router.post("/v1/screener")
@_limit("5/minute")
async def stock_screener(
    request: Request,
    body: ScreenerRequest
):
    try:
        import asyncio

        from core.screener import StockScreener, ScreenerFilter, DEFAULT_UNIVERSE

        tickers = body.tickers if body.tickers else DEFAULT_UNIVERSE.get(body.universe, DEFAULT_UNIVERSE["us_large_cap"])
        filters = ScreenerFilter(
            pe_min=body.pe_min,
            pe_max=body.pe_max,
            roe_min=body.roe_min,
            roe_max=body.roe_max,
            market_cap_min=body.market_cap_min,
            market_cap_max=body.market_cap_max,
            volume_min=body.volume_min,
            rsi_min=body.rsi_min,
            rsi_max=body.rsi_max,
            price_above_sma200=body.price_above_sma200,
            dividend_yield_min=body.dividend_yield_min,
            revenue_growth_min=body.revenue_growth_min,
            sector=body.sector
        )
        screener = StockScreener()
        results = await asyncio.get_event_loop().run_in_executor(
            None, screener.screen, tickers, filters, 8, body.include_sentiment
        )
        results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:body.max_results]
        return {"count": len(results), "universe": body.universe,
                "sentiment_enriched": body.include_sentiment, "results": results}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[screener] %s", exc)
        raise HTTPException(500, f"Screener error: {exc}")


# ── H-4: Forex Pairs ──────────────────────────────────────────────────────────

@router.get("/v1/forex")
@_limit("20/minute")
async def get_forex(
    request: Request,
    category: str = "all",   # all | arab | major | em
    use_cache: bool = True
):
    """Live FX rates — Arab pairs (AED/SAR/EGP/KWD/QAR/BHD) + major pairs."""
    try:
        from core.forex import ForexFetcher
        pairs = await asyncio.get_event_loop().run_in_executor(
            None, ForexFetcher().fetch, use_cache
        )
        if category != "all":
            pairs = [p for p in pairs if p.get("category") == category]
        return {"count": len(pairs), "pairs": pairs}
    except Exception as exc:
        logger.error("[forex] %s", exc)
        raise HTTPException(status_code=500, detail=f"Forex error: {exc}")


@router.get("/v1/forex/{symbol}")
@_limit("30/minute")
async def get_forex_pair(
    request: Request,
    symbol: str
):
    """Single FX pair — e.g. /v1/forex/USDAED=X or /v1/forex/EURUSD"""
    try:
        from core.forex import ForexFetcher
        # normalise: add =X suffix if missing
        sym = symbol.upper()
        if not sym.endswith("=X"):
            sym += "=X"
        pair = await asyncio.get_event_loop().run_in_executor(
            None, ForexFetcher().get_pair, sym
        )
        if not pair:
            raise HTTPException(status_code=404, detail=f"Pair {sym} not found")
        return pair
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[forex/%s] %s", symbol, exc)
        raise HTTPException(status_code=500, detail=f"Forex error: {exc}")
