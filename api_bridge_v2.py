import numpy; import yfinance
import os
# Phase 5 hardening: app-created files (SQLite DBs, export PDFs, caches) get
# 640/750 instead of world-readable 644 — a second local user (e.g. opc) must
# not read api_keys.db / sessions.db / user data. Set before any DB/dir create.
os.umask(0o027)
import logging
import time as _time
import asyncio
import uuid
from core.config import (
    APP_DB, STATIC_DIR, EXPORTS_DIR, FILE_CACHE_DIR,
    BACKEND_LOG, ENV_FILE,
)
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Request, Depends
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta, timezone
import uvicorn
import io
import jwt as _jwt
import re as _re
import copy as _copy
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from core.tts_service import TTSService
from core.orchestrator import MultiAgentOrchestrator
from core.news_aggregator import get_news as _get_aggregated_news
from core.export_engine import export as export_engine
from core.dependencies.auth import require_auth
from contextlib import asynccontextmanager
# learning_engine runs as a separate service (eisax-learning.service)


# ── JWT auth dependency — defined early so routes above line 3816 can use it ──
_bearer = HTTPBearer(auto_error=False)


def _require_jwt(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    """FastAPI dependency — validates Bearer JWT, returns payload dict."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    from core.auth import decode_token as _decode_token
    try:
        return _decode_token(credentials.credentials)
    except _jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except _jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ── Logging with rotation (max 10MB per file, keep 3 backups) ──────────────
_log_handler = RotatingFileHandler(
    str(BACKEND_LOG),
    maxBytes=10 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_log_handler, logging.StreamHandler()])
logger = logging.getLogger("api_bridge")

limiter = Limiter(key_func=get_remote_address)
import subprocess as _subprocess
_GIT_SHA = 'unknown'
try:
    _GIT_SHA = _subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        cwd='/home/ubuntu/investwise',
        text=True,
    ).strip()
except Exception:
    pass
_APP_VERSION = '2.0.0'
try:
    _APP_VERSION = open('/home/ubuntu/investwise/version.txt').read().strip()
except Exception:
    pass

@asynccontextmanager
async def lifespan(app):
    yield

app = FastAPI(title="InvestWise & EisaX AI Gateway", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Request body size guard (prevent >4 MB payloads from crashing workers) ──
_MAX_BODY_BYTES = 4 * 1024 * 1024  # 4 MB

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_BODY_BYTES:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large (max 4 MB)"},
            )
    return await call_next(request)

static_dir = str(STATIC_DIR)
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ── Include EisaX News Engine router ─────────────────────────────────────
# IMPORTANT: use include_router (not app.mount) so /v1/news and /v1/chat coexist.
# app.mount("/v1", sub_app) hijacks ALL /v1/* routes including /v1/chat.
try:
    import sys as _sys
    _sys.path.insert(0, "/home/ubuntu/eisax-news")
    from db import init_db as _news_init_db
    from news_api import news_router as _news_router   # APIRouter, not FastAPI app
    from engine import start_scheduler as _start_news_scheduler
    _news_init_db()
    app.include_router(_news_router, prefix="/v1")     # → /v1/news, /v1/news/latest …
    # Multi-worker dedup: only the first-spawned worker owns the scheduler.
    # `EISAX_SCHEDULER_OWNER` is set by gunicorn's post_fork hook via an
    # O_EXCL lock file; absent (default → "1") means single-worker setup.
    import logging as _lg
    _scheduler_owner = os.getenv("EISAX_SCHEDULER_OWNER", "1") == "1"
    # News collection is owned by the dedicated eisax-news.service. Gunicorn only
    # serves the /v1/news API; it does NOT collect unless explicitly opted in via
    # EISAX_GUNICORN_NEWS_SCHEDULER=1. This avoids 3-4x redundant scrape+summarize
    # cycles (prod + staging + legacy app_new all wrote to the same news.db).
    _gunicorn_collects = os.getenv("EISAX_GUNICORN_NEWS_SCHEDULER", "0") == "1"
    if _scheduler_owner and _gunicorn_collects:
        _start_news_scheduler()
        _lg.getLogger(__name__).info("[NewsEngine] Router included at /v1/news — scheduler started (gunicorn opt-in)")
    else:
        _lg.getLogger(__name__).info(
            "[NewsEngine] Router included at /v1/news — collection delegated to eisax-news.service"
        )
except Exception as _ne:
    import logging as _lg
    _lg.getLogger(__name__).warning("[NewsEngine] Failed to include router: %s", _ne)

orchestrator = MultiAgentOrchestrator(db_path=str(APP_DB))
tts_service = TTSService()

_ENVIRONMENT = os.getenv("ENVIRONMENT", "production").strip().lower()
_STAGING_UPSTREAM_BASE = os.getenv("STAGING_UPSTREAM_BASE", "http://127.0.0.1:8000").rstrip("/")
_STAGING_LEADS_PATH = Path(
    os.getenv(
        "STAGING_LEADS_PATH",
        "/home/ubuntu/investwise/data/staging-agent-leads.jsonl",
    )
)
_STAGING_LEADS_PATH.parent.mkdir(parents=True, exist_ok=True)
# Disk-based file store (shared across all workers)
import json as _json
_FILE_CACHE_DIR = str(FILE_CACHE_DIR)
_FILE_STORE_TTL = 3600  # seconds
_DOWNLOAD_TOKENS = {}
os.makedirs(_FILE_CACHE_DIR, exist_ok=True)


def _create_download_token(filename: str, user_id: str) -> str:
    """Mint a one-hour download token (consumed by /v1/download/{token}).

    Restored in Phase 4: commit 3225b34 dropped this helper while
    api/routers/content.py still lazy-imports it — every /v1/export* call
    has 500'd at `from api_bridge_v2 import _create_download_token` since.
    """
    now = _time.time()
    for existing_token, entry in list(_DOWNLOAD_TOKENS.items()):
        if not isinstance(entry, dict) or entry.get("expires", 0) <= now:
            _DOWNLOAD_TOKENS.pop(existing_token, None)

    token = uuid.uuid4().hex
    _DOWNLOAD_TOKENS[token] = {
        "filename": filename,
        "user_id": user_id,
        "expires": now + 3600,
    }
    return token


# Guest/staging access helpers live in api/routers/staging.py (the only live
# copies). The duplicates that used to sit here had zero callers — removed in
# the Phase 4 cleanup; _GUEST_LIMIT_MESSAGE is re-imported below for compat.

# ── File-store helpers + portfolio upload (extracted to api/routers/portfolio_upload.py) ──
from api.routers.portfolio_upload import (
    portfolio_upload_router,
    _evict_old_files, _file_store_set, _file_store_get, _file_store_get_for_user,
)
app.include_router(portfolio_upload_router)

# ── Portfolio analytics endpoints (macro sim, MC/VaR, shariah, optimize, etc.) ──
try:
    from api.routers.portfolio_analytics import router as portfolio_analytics_router
    app.include_router(portfolio_analytics_router)
except Exception as _pa_err:
    import logging as _pa_log
    _pa_log.getLogger("eisax.startup").warning(f"portfolio_analytics router not loaded: {_pa_err}")

class MessagePayload(BaseModel):
    message: str = Field(..., max_length=16000)
    user_id: Optional[str] = "admin"
    session_id: Optional[str] = None
    files: Optional[list] = []
    settings: Optional[dict] = None


class PilotReportPayload(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    market: str = Field(..., min_length=1, max_length=16)
    language: str = Field(default="en", min_length=2, max_length=8)
    report_type: str = Field(..., min_length=4, max_length=32)


# ── Staging API + Guest Admin routers (extracted to api/routers/staging.py) ──
from api.routers.staging import (
    staging_router,
    guest_admin_router,
    _resolution_error_response,
    _should_resolve_direct_analysis_request,
    # Re-exports for backward compat (tests + callers)
    _guest_trial_check,
    _guest_trial_increment_success,
    _GUEST_LIMIT_MESSAGE,
)
app.include_router(staging_router)
app.include_router(guest_admin_router)

@app.get("/")
async def root():
    return RedirectResponse(url="https://eisax.com", status_code=301)

@app.get("/v1/chart-data")
@limiter.limit("60/minute")
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


@app.post("/upload")
@limiter.limit("10/minute")
async def upload_file_ui(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(require_auth),
):
    """Receive file from chat UI, extract text via Gemini Vision or file_processor."""
    import uuid as _uuid, base64 as _b64
    from core.file_processor import process_file
    raw = await file.read()
    b64 = _b64.b64encode(raw).decode()
    result = process_file(file.filename, b64)
    file_id = str(_uuid.uuid4())
    # require_auth normalizes all methods (JWT/eixa_/legacy) to user_id;
    # "sub" kept as fallback for any cached _require_jwt-shaped payloads
    uploader_user_id = str(user.get("user_id") or user.get("sub") or "") or None
    _evict_old_files()
    _file_store_set(file_id, {
        "id": file_id,
        "filename": file.filename,
        "text": result.get("text", ""),
        "user_id": uploader_user_id,
        "error": result.get("error"),
        "_ts": _time.time(),
    })
    return {"status": "received", "file_id": file_id, "filename": file.filename}

@app.get("/health")
@limiter.limit("30/minute")
async def health(request: Request, user: dict = Depends(require_auth)):
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


# ── Chat, report, TTS routes (extracted to api/routers/chat.py) ───────────────
from api.routers.chat import (
    chat_router, MessagePayload, PilotReportPayload, TTSRequest, _coerce_chat_payload,
)
app.include_router(chat_router)


# ── Admin session management (extracted to api/routers/admin_session.py) ──────
from api.routers.admin_session import (
    admin_session_router,
    AdminAuthRequest, AdminLoginRequest,
    _decode_admin_session_token, _check_secure_or_admin_session,
    _check_admin, _require_admin_cookie,
)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
ADMIN_PASSPHRASE = os.getenv("ADMIN_PASSPHRASE", "") or os.getenv("ADMIN_TOKEN", "")
app.include_router(admin_session_router)

# --- New History Endpoints ---


# ── Content / intelligence routes (extracted to api/routers/content.py) ───────
from api.routers.content import content_router, HtmlExportPayload, TranslatePayload
app.include_router(content_router)

# ── B2B Auth + Admin User Management (extracted to api/routes/auth.py) ────
# JWT constants needed by admin login (lines above)
from core.auth import JWT_SECRET, JWT_ALGORITHM, decode_token, decode_token as _decode_token_for_resolve
from core.user_db import init_users_table
init_users_table()  # idempotent — creates users table if not exists

from api.routes.auth import router as auth_router
app.include_router(auth_router)


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/v1/health")
@limiter.limit("30/minute")
async def health_check(
    request: Request,
    user: dict = Depends(require_auth),
):
    from core.services.health_service import run_health_check
    # arg is reserved/unused in health_service (noqa ARG001); SECURE_TOKEN retired
    result = await run_health_check("")
    status_code = 200 if result["status"] == "ok" else (503 if result["status"] == "down" else 207)
    from fastapi.responses import JSONResponse
    return JSONResponse(content=result, status_code=status_code)



# ── Admin misc + tail routes (extracted to api/routers/misc.py) ──────────────
from api.routers.misc import admin_misc_router, misc_router
app.include_router(admin_misc_router)
app.include_router(misc_router)
