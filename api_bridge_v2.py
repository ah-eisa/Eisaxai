import numpy; import yfinance
import os
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
from contextlib import asynccontextmanager
# learning_engine runs as a separate service (eisax-learning.service)


def _resolve_auth(
    x_api_key: str = Header(None, alias='X-API-Key'),
    access_token: str = Header(None, alias='access-token'),
    authorization: str = Header(None, alias='Authorization'),
) -> dict:
    token = x_api_key or access_token
    bearer = (authorization or '').removeprefix('Bearer ').strip()
    # 1. Personal API key (starts with eixa_)
    if token and token.startswith('eixa_'):
        from core.api_keys import validate_key
        info = validate_key(token)
        if info: return {'user_id': info['user_id'], 'tier': info['tier'], 'method': 'api_key'}
        raise HTTPException(401, 'Invalid API key')
    if bearer and bearer.startswith('eixa_'):
        from core.api_keys import validate_key
        info = validate_key(bearer)
        if info: return {'user_id': info['user_id'], 'tier': info['tier'], 'method': 'api_key'}
        raise HTTPException(401, 'Invalid API key')
    # 2. Legacy SECURE_TOKEN
    if SECURE_TOKEN and (token == SECURE_TOKEN or bearer == SECURE_TOKEN):
        return {'user_id': 'admin', 'tier': 'vip', 'method': 'secure_token'}
    raise HTTPException(403, 'Unauthorized')


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
    _start_news_scheduler()
    import logging as _lg
    _lg.getLogger(__name__).info("[NewsEngine] Router included at /v1/news — scheduler started")
except Exception as _ne:
    import logging as _lg
    _lg.getLogger(__name__).warning("[NewsEngine] Failed to include router: %s", _ne)

orchestrator = MultiAgentOrchestrator(db_path=str(APP_DB))
tts_service = TTSService()

SECURE_TOKEN = os.getenv("SECURE_TOKEN", "")
_ENVIRONMENT = os.getenv("ENVIRONMENT", "production").strip().lower()
_STAGING_UPSTREAM_BASE = os.getenv("STAGING_UPSTREAM_BASE", "http://127.0.0.1:8000").rstrip("/")
_STAGING_ADMIN_USERS = {
    item.strip().lower()
    for item in os.getenv("EISAX_ADMIN_USERS", "Admin,admin").split(",")
    if item.strip()
}
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


def _parse_guest_tokens() -> dict[str, str]:
    """Map configured guest tokens to revocable usernames without exposing tokens."""
    entries = []
    if os.getenv("EISAX_GUEST_TOKEN"):
        entries.append(f"guest:{os.getenv('EISAX_GUEST_TOKEN')}")
    entries.extend(part.strip() for part in os.getenv("EISAX_GUEST_TOKENS", "").split(",") if part.strip())
    tokens: dict[str, str] = {}
    for idx, entry in enumerate(entries, start=1):
        if ":" in entry:
            username, token = entry.split(":", 1)
        else:
            username, token = f"guest_{idx}", entry
        username = (username or f"guest_{idx}").strip()
        token = (token or "").strip()
        if token:
            tokens[token] = username
    return tokens


def _resolve_guest_token(request: Request) -> Optional[str]:
    supplied = (
        request.headers.get("X-Guest-Token")
        or request.query_params.get("guest_token")
        or request.query_params.get("demo_token")
        or ""
    ).strip()
    if not supplied:
        return None
    import hmac as _hmac
    for configured, username in _parse_guest_tokens().items():
        if _hmac.compare_digest(supplied, configured):
            return username
    return None


def _resolve_staging_access(request: Request) -> dict:
    auth_user = (request.headers.get("X-Auth-User") or "").strip()
    token_user = _resolve_guest_token(request)
    if token_user:
        return {"role": "guest", "username": token_user, "demo": True, "method": "guest_token"}
    if auth_user:
        if auth_user.lower() in _STAGING_ADMIN_USERS:
            return {"role": "admin", "username": auth_user, "demo": False, "method": "basic_auth"}
        return {"role": "guest", "username": auth_user, "demo": True, "method": "basic_auth"}
    # Preserve internal/backend callers that bypass the public nginx layer.
    host = (request.headers.get("host") or "").lower()
    client_host = getattr(request.client, "host", "") if request.client else ""
    if host.startswith(("127.0.0.1", "localhost")) or client_host in {"127.0.0.1", "::1", "localhost"}:
        return {"role": "admin", "username": "internal", "demo": False, "method": "internal"}
    return {"role": "guest", "username": "public-demo", "demo": True, "method": "public"}


def _is_guest_access(access_context: Optional[dict], restrict_for_trial: bool = False) -> bool:
    return bool(restrict_for_trial or ((access_context or {}).get("role") == "guest"))


def _safe_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except Exception:
        return default


_GUEST_LIMIT_MESSAGE = "This guest demo has reached its analysis limit. Please contact EisaX for extended access."

# ── File-store helpers + portfolio upload (extracted to api/routers/portfolio_upload.py) ──
from api.routers.portfolio_upload import (
    portfolio_upload_router,
    _evict_old_files, _file_store_set, _file_store_get, _file_store_get_for_user,
)
app.include_router(portfolio_upload_router)

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
    user: dict = Depends(_require_jwt),
):
    """Receive file from chat UI, extract text via Gemini Vision or file_processor."""
    import uuid as _uuid, base64 as _b64
    from core.file_processor import process_file
    raw = await file.read()
    b64 = _b64.b64encode(raw).decode()
    result = process_file(file.filename, b64)
    file_id = str(_uuid.uuid4())
    uploader_user_id = str(user["sub"])
    if not uploader_user_id:
        try:
            uploader_user_id = _resolve_user_context(
                access_token=None,
                access_token_alt=access_token_alt,
                authorization=authorization,
            ).get("user_id")
        except HTTPException:
            uploader_user_id = None
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
async def health(request: Request, access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
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

# _resolve_user_context used by /v1/upload-portfolio
def _resolve_user_context(
    access_token: Optional[str] = None,
    access_token_alt: Optional[str] = None,
    authorization: Optional[str] = None,
) -> dict:
    bearer = (authorization or "").removeprefix("Bearer ").strip()
    if bearer and not bearer.startswith("eixa_") and bearer != SECURE_TOKEN:
        try:
            payload = _decode_token_for_resolve(bearer)
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

from api.routes.auth import router as auth_router
app.include_router(auth_router)


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/v1/health")
@limiter.limit("30/minute")
async def health_check(
    request: Request,
    access_token:     str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    _token = access_token or access_token_alt
    if _token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    from core.services.health_service import run_health_check
    result = await run_health_check(SECURE_TOKEN)
    status_code = 200 if result["status"] == "ok" else (503 if result["status"] == "down" else 207)
    from fastapi.responses import JSONResponse
    return JSONResponse(content=result, status_code=status_code)



# ── Admin misc + tail routes (extracted to api/routers/misc.py) ──────────────
from api.routers.misc import admin_misc_router, misc_router
app.include_router(admin_misc_router)
app.include_router(misc_router)
