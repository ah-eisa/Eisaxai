"""
api/routers/chat.py
────────────────────
Chat, report, and TTS routes extracted from api_bridge_v2.py.

Router exported:
  chat_router — no prefix, mounts /v1/pilot-report, /v1/report,
                /v1/chat, /chat, /api/chat, /v1/chat/stream, /v1/tts

Models re-exported (backward compat):
  MessagePayload, PilotReportPayload, TTSRequest
"""
from __future__ import annotations

import asyncio
import io
import json as _json
import logging
import os
import time as _time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.routers.staging import (
    _resolution_error_response,
    _should_resolve_direct_analysis_request,
)

logger = logging.getLogger("api_bridge")
limiter = Limiter(key_func=get_remote_address)

SECURE_TOKEN = os.getenv("SECURE_TOKEN", "")

chat_router = APIRouter()

# ── Models ────────────────────────────────────────────────────────────────────

class MessagePayload(BaseModel):
    message: str
    user_id: str = "admin"
    session_id: Optional[str] = None
    settings: Optional[dict] = None
    files: Optional[list] = None

class PilotReportPayload(BaseModel):
    symbol: str
    market: str = ""
    language: str = "en"
    report_type: str = "pilot_report"

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

# ── Helpers ────────────────────────────────────────────────────────────────────

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

# ── Report ─────────────────────────────────────────────────────────────────────

@chat_router.post("/v1/pilot-report")
@chat_router.post("/v1/report")
@limiter.limit("10/minute")
async def pilot_report(
    payload: PilotReportPayload,
    request: Request,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    from api_bridge_v2 import orchestrator, _APP_VERSION
    _token = access_token or access_token_alt
    if _token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    if str(payload.report_type or "").strip().lower() != "pilot_report":
        raise HTTPException(status_code=400, detail="report_type must be 'pilot_report'")
    if not orchestrator.financial_agent:
        raise HTTPException(status_code=503, detail="Financial agent is unavailable")

    client_ip = (
        request.headers.get("X-Real-IP")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or str(request.client.host)
    )
    user_agent = request.headers.get("User-Agent", "")
    session_user = f"pilot-report:{client_ip}"
    session_id = orchestrator.session_mgr.get_or_create_session(
        session_user, ip=client_ip, user_agent=user_agent,
    )
    orchestrator.session_mgr.get_or_create_session(
        session_user, session_id=session_id, ip=client_ip, user_agent=user_agent,
    )

    market = str(payload.market or "").strip().upper()
    symbol = str(payload.symbol or "").strip().upper()
    language = (str(payload.language or "").strip().lower() or "en")
    message = (
        f"تحليل كامل لـ {symbol}"
        if language.startswith("ar")
        else f"full analysis of {symbol}"
    )

    started_at = _time.perf_counter()
    generated_at = datetime.now().astimezone().isoformat()
    try:
        analysis_result = await asyncio.wait_for(
            run_in_threadpool(
                orchestrator.financial_agent._handle_analytics,
                session_id,
                {"user_id": session_user, "user_ctx": {"market": market, "language": language}},
                message,
                False,
                "full",
            ),
            timeout=300,
        )
    except asyncio.TimeoutError as exc:
        logger.warning("[pilot-report] timed out for %s (%s)", symbol, market)
        raise HTTPException(status_code=504, detail="Pilot report generation timed out") from exc
    except Exception as exc:
        logger.exception("[pilot-report] generation failed for %s (%s)", symbol, market)
        raise HTTPException(status_code=500, detail="Pilot report generation failed") from exc

    html_report = analysis_result.get("reply") or ""
    if not html_report:
        raise HTTPException(status_code=502, detail="Pilot report returned empty output")

    try:
        from core.services.pilot_report_json import build_pilot_report_json
        latency_seconds = max(0, int(round(_time.perf_counter() - started_at)))
        report_json = build_pilot_report_json(
            symbol=symbol,
            market=market,
            language=language,
            report_text=html_report,
            analysis_data=analysis_result.get("data") or {},
            system_version=_APP_VERSION,
            model_primary="DeepSeek V3",
            generated_at=generated_at,
            data_as_of=generated_at,
            latency_seconds=latency_seconds,
        )
    except Exception as exc:
        logger.exception("[pilot-report] json build failed for %s (%s)", symbol, market)
        raise HTTPException(status_code=500, detail="Pilot report JSON validation failed") from exc

    return JSONResponse(content={"html_report": html_report, "report_json": report_json})

# ── Chat ───────────────────────────────────────────────────────────────────────

@chat_router.post("/v1/chat")
@limiter.limit("30/minute")
async def unified_chat(
    payload: MessagePayload,
    request: Request,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    """نقطة الدخول الرئيسية للمحادثة - مع الحماية"""
    from api_bridge_v2 import orchestrator, _file_store_get_for_user

    _token = access_token or access_token_alt
    if _token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")

    client_ip = (
        request.headers.get("X-Real-IP")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or str(request.client.host)
    )
    user_agent = request.headers.get("User-Agent", "")

    if orchestrator.session_mgr.is_user_blocked(payload.user_id):
        raise HTTPException(status_code=403, detail="Your account has been suspended. Please contact support.")
    if orchestrator.session_mgr.is_user_rate_limited(payload.user_id):
        raise HTTPException(status_code=429, detail="Daily message limit reached. Please try again tomorrow.")
    if orchestrator.session_mgr.is_ip_blocked(client_ip):
        raise HTTPException(status_code=403, detail="Access denied from this network.")

    from core.rate_limiter import is_rate_limited, get_usage
    if is_rate_limited(payload.user_id):
        usage = get_usage(payload.user_id)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {usage['count']}/{usage['limit']} requests per minute. "
                   f"Reset in {usage['reset_in']:.0f}s.",
            headers={"Retry-After": str(int(usage["reset_in"]))},
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
        return JSONResponse(
            content={
                "reply": combined,
                "session_id": session_id,
                "agent": "Admin",
                "model": None,
                "download_url": None,
                "format": None,
                "quota": orchestrator.session_mgr.get_user_daily_usage(payload.user_id),
            },
            headers=orchestrator.session_mgr.get_quota_header(payload.user_id),
        )

    message = payload.message
    resolution_payload = None
    active_file_id = None
    if payload.settings and isinstance(payload.settings, dict):
        active_file_id = payload.settings.get("active_file_id")

    if not payload.files and not active_file_id and _should_resolve_direct_analysis_request(message):
        from core.services.entity_resolution import EntityResolution, resolve_asset_entity

        _chat_resolution: Optional[EntityResolution] = None
        try:
            from core.ticker_index import quick_scan as _qs_chat
            _qs_hit = _qs_chat(message)
            if _qs_hit and _qs_hit.match_type != "bare_ambiguous":
                _chat_resolution = EntityResolution(
                    query_raw=message,
                    normalized_query=_qs_hit.symbol,
                    resolution_status="resolved",
                    symbol=_qs_hit.symbol,
                    market=_qs_hit.market,
                    asset_type=_qs_hit.asset_type,
                    currency=_qs_hit.currency,
                    resolution_source="ticker_index",
                    confidence="high",
                    name=_qs_hit.name,
                    exchange=_qs_hit.exchange,
                )
                logger.info(
                    "[ticker_index] pre-routed chat '%s' → %s / %s (bypassing entity resolution)",
                    message, _qs_hit.symbol, _qs_hit.market,
                )
        except Exception as _qs_err:
            logger.debug("[ticker_index] chat scan error (non-fatal): %s", _qs_err)

        resolution = _chat_resolution or resolve_asset_entity(message)
        resolution_payload = resolution.to_dict()
        if not resolution.is_resolved:
            logger.info("[entity-resolution][chat] %s", resolution_payload)
            return _resolution_error_response(resolution_payload)
        logger.info(
            "[entity-resolution][chat] raw='%s' normalized='%s' -> %s / %s via %s (%s)",
            message,
            resolution.normalized_query,
            resolution.symbol,
            resolution.market,
            resolution.resolution_source,
            resolution.confidence,
        )
        message = resolution.analysis_instruction

    # Inject file content from /upload store via active_file_id
    stored_file = _file_store_get_for_user(active_file_id, payload.user_id) if active_file_id else None
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
        except Exception:
            pass

    result = await orchestrator.process_message(
        user_id=payload.user_id,
        message=message,
        session_id=session_id,
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
    if resolution_payload:
        response_body["resolution"] = resolution_payload
    return JSONResponse(
        content=response_body,
        headers=orchestrator.session_mgr.get_quota_header(payload.user_id),
    )


@chat_router.post("/chat")
@chat_router.post("/api/chat")
@limiter.limit("30/minute")
async def unified_chat_legacy(
    request: Request,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
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
            detail="Request body must include 'message' (or legacy 'text').",
        )
    return await unified_chat(
        payload=payload,
        request=request,
        access_token=access_token,
        access_token_alt=access_token_alt,
    )

# ── SSE Streaming Chat ─────────────────────────────────────────────────────────

@chat_router.post("/v1/chat/stream")
@limiter.limit("30/minute")
async def unified_chat_stream(
    payload: MessagePayload,
    request: Request,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
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
    from api_bridge_v2 import orchestrator
    _token = access_token or access_token_alt
    if _token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")

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
            async for sse_line in orchestrator.stream_process_message(
                user_id=payload.user_id,
                message=payload.message,
                session_id=session_id,
            ):
                yield sse_line
        except Exception as e:
            yield f'data: {_json.dumps({"type": "error", "text": str(e)}, ensure_ascii=False)}\n\n'
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

# ── TTS ────────────────────────────────────────────────────────────────────────

@chat_router.post("/v1/tts")
@limiter.limit("20/minute")
async def text_to_speech(
    request: Request,
    tts_body: TTSRequest,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    from api_bridge_v2 import tts_service
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        audio_bytes = tts_service.generate_speech(tts_body.text, tts_body.language)
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
