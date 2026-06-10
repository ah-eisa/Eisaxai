"""
app_new.py — New FastAPI entry point for EisaX / InvestWise

Replaces api_bridge_v2.py by composing 3 extracted route modules:
  - api.routes.auth        (Phase 1 — 12 endpoints)
  - api.routes.portfolio   (Phase 2 —  7 endpoints)
  - api.routes.chat        (Phase 3 — 67 endpoints)

Usage:
    python -m uvicorn app_new:app --reload --host 0.0.0.0 --port 8000
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager

import numpy  # noqa: F401 — side-effect import (yfinance needs it)
import yfinance  # noqa: F401

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from core.config import (
    APP_DB, STATIC_DIR, EXPORTS_DIR, FILE_CACHE_DIR, BACKEND_LOG,
)

# ── Logging ───────────────────────────────────────────────────────────────
_log_handler = RotatingFileHandler(
    str(BACKEND_LOG),
    maxBytes=10 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_log_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
)
logging.basicConfig(level=logging.INFO, handlers=[_log_handler, logging.StreamHandler()])
logger = logging.getLogger("app_new")

# ── Lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    # Initialise the orchestrator's session manager on startup
    from core.orchestrator import MultiAgentOrchestrator
    application.state.orchestrator = MultiAgentOrchestrator(db_path=str(APP_DB))
    yield

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="InvestWise & EisaX AI Gateway", lifespan=lifespan)

# ── Rate limiting ─────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Static files ──────────────────────────────────────────────────────────
static_dir = str(STATIC_DIR)
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ── Include EisaX News Engine router (if available) ──────────────────────
try:
    import sys as _sys
    _sys.path.insert(0, "/home/ubuntu/eisax-news")
    from db import init_db as _news_init_db
    from news_api import news_router as _news_router
    from engine import start_scheduler as _start_news_scheduler
    _news_init_db()
    app.include_router(_news_router, prefix="/v1")
    _start_news_scheduler()
    logger.info("[NewsEngine] Router included at /v1/news — scheduler started")
except Exception as _ne:
    logger.warning("[NewsEngine] Failed to include router: %s", _ne)

# ── Wire extracted routers ────────────────────────────────────────────────
from api.routes.auth import router as auth_router
from api.routes.portfolio import router as portfolio_router
from api.routes.chat import router as chat_router

app.include_router(auth_router)
app.include_router(portfolio_router)
app.include_router(chat_router)

# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_new:app", host="0.0.0.0", port=8000, workers=2)
