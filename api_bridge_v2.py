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

def _evict_old_files():
    """Remove file cache entries older than TTL."""
    now = _time.time()
    for fname in os.listdir(_FILE_CACHE_DIR):
        fpath = os.path.join(_FILE_CACHE_DIR, fname)
        try:
            if now - os.path.getmtime(fpath) > _FILE_STORE_TTL:
                os.remove(fpath)
        except Exception:
            pass

def _file_store_set(file_id: str, data: dict):
    fpath = os.path.join(_FILE_CACHE_DIR, file_id + ".json")
    with open(fpath, "w", encoding="utf-8") as _f:
        _json.dump(data, _f, ensure_ascii=False)

def _file_store_get(file_id: str):
    fpath = os.path.join(_FILE_CACHE_DIR, file_id + ".json")
    if not os.path.exists(fpath):
        return None
    with open(fpath, "r", encoding="utf-8") as _f:
        return _json.load(_f)


def _create_download_token(filename: str, user_id: str) -> str:
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


def _file_store_get_for_user(file_id: str, user_id: Optional[str]):
    entry = _file_store_get(file_id)
    if entry is None:
        return None
    if isinstance(entry, str):
        return {"text": entry}
    if not isinstance(entry, dict):
        return None

    stored_user_id = entry.get("user_id")
    if stored_user_id is not None and stored_user_id != user_id:
        raise HTTPException(status_code=403, detail="File access denied")
    return entry


def _soften_portfolio_advice(text: str) -> str:
    """Convert directive wording to advisory wording for safer decision support."""
    if not text:
        return text
    softened = str(text)
    replacements = [
        (r"(?i)\bexecute (?:the )?rebalancing plan immediately\b", "consider gradual rebalancing based on risk triggers"),
        (r"(?i)\bexecute immediately\b", "consider phased execution"),
        (r"(?i)\bexit 100%\b", "consider a full exit"),
        (r"(?i)\breduce to 50%\b", "consider reducing to 50%"),
        (r"(?i)\bmust\b", "should"),
    ]
    for pattern, repl in replacements:
        softened = _re.sub(pattern, repl, softened)
    return softened


def _compute_portfolio_confidence(sharpe: float,
                                  eff_n: float,
                                  avg_corr: float,
                                  beta_total: Optional[float],
                                  cvar_95: float,
                                  rolling_sharpe_now: Optional[float] = None) -> int:
    """Deterministic confidence score for portfolio verdict communication."""
    score = 66.0
    score += min(max((sharpe - 0.8) * 12.0, -10.0), 10.0)
    score += min(max((eff_n - 2.5) * 4.0, -8.0), 8.0)
    score -= min(max((avg_corr - 0.45) * 20.0, 0.0), 8.0)
    if isinstance(beta_total, (int, float)):
        score -= min(max((beta_total - 1.2) * 8.0, 0.0), 10.0)
    else:
        score -= 3.0  # uncertainty penalty when beta data coverage is missing
    score -= min(max((abs(cvar_95) - 3.0) * 3.0, 0.0), 8.0)
    if rolling_sharpe_now is not None:
        score += min(max((rolling_sharpe_now - 0.2) * 5.0, -6.0), 6.0)
    return int(max(45, min(88, round(score))))


def _build_portfolio_decision_layer(*,
                                    confidence: int,
                                    risk_label: str,
                                    rolling_sharpe_now: Optional[float],
                                    tech_weight: float,
                                    next_review_hint: str = "next review cycle") -> str:
    """Adds explicit decision discipline: uncertainty, no-action case, and boundary."""
    rolling_note = (
        f"rolling Sharpe currently {rolling_sharpe_now:+.2f}"
        if rolling_sharpe_now is not None else
        "rolling Sharpe trend requires ongoing confirmation"
    )
    if tech_weight >= 0.70:
        alt_case = (
            "If AI mega-cap momentum persists and earnings revisions remain positive, concentrated tech could continue to outperform in the near term."
        )
    else:
        alt_case = (
            "If macro volatility cools and correlations normalize, current allocation may remain acceptable with only minor rebalancing."
        )
    no_action_case = (
        f"If risk metrics stay stable and no catalyst break occurs before the {next_review_hint}, HOLD/no-trade remains a valid decision."
    )
    return (
        "\n\n---\n"
        "## 🎯 Decision Discipline Layer\n"
        f"- **Confidence:** {confidence}%\n"
        f"- **Risk Posture:** {risk_label}\n"
        f"- **Primary Uncertainty:** concentration regime sensitivity and {rolling_note}\n"
        f"- **No-Action Case:** {no_action_case}\n"
        f"- **Alternative Scenario:** {alt_case}\n"
        "> **Decision Boundary:** This analysis provides strategic guidance, not execution instructions."
    )


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


class StagingLeadPayload(BaseModel):
    email: str = Field(..., max_length=320)
    name: Optional[str] = Field(default=None, max_length=120)
    query: Optional[str] = Field(default=None, max_length=500)
    report_kind: Optional[str] = Field(default=None, max_length=40)


def _ensure_staging_public_enabled():
    if _ENVIRONMENT != "staging":
        raise HTTPException(status_code=404, detail="Not found")


def _staging_client_id(request: Request) -> str:
    raw_ip = (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or request.headers.get("X-Real-IP")
        or str(request.client.host)
    )
    safe_ip = _re.sub(r"[^0-9a-fA-F:.]", "", raw_ip) or "unknown"
    return f"staging:{safe_ip}"


def _staging_proxy_headers() -> dict:
    return {"X-API-Key": SECURE_TOKEN}


def _staging_clean_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = _re.sub(r"```.*?```", " ", cleaned, flags=_re.S)
    cleaned = _re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned)
    cleaned = _re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = cleaned.replace("|", " ")
    cleaned = _re.sub(r"[#>*_`]+", " ", cleaned)
    cleaned = _re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _staging_extract_section(markdown: str, markers: list[str]) -> str:
    lines = (markdown or "").splitlines()
    for idx, line in enumerate(lines):
        normalized = _staging_clean_text(line).lower()
        if not normalized:
            continue
        if any(marker.lower() in normalized for marker in markers):
            chunk = []
            for next_line in lines[idx + 1:]:
                if next_line.strip().startswith("#"):
                    break
                chunk.append(next_line)
            body = "\n".join(chunk).strip()
            if body:
                return body
    return ""


def _staging_extract_sentences(text: str) -> list[str]:
    sentences = []
    for raw in _re.split(r"(?<=[.!?])\s+", _staging_clean_text(text)):
        candidate = raw.strip(" -")
        if len(candidate) >= 40 and candidate not in sentences:
            sentences.append(candidate)
    return sentences


def _staging_extract_marked_line(markdown: str, patterns: list[str]) -> Optional[str]:
    for line in (markdown or "").splitlines():
        cleaned = _staging_clean_text(line)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(pattern.lower() in lowered for pattern in patterns):
            return cleaned
    return None


def _staging_extract_verdict(markdown: str, report_kind: str, result_type: str = "") -> str:
    # ── Construction results use a dedicated vocabulary ──────────────────
    if result_type == "construction":
        md = markdown or ""
        if _re.search(r"CONDITIONAL APPROVAL", md, _re.I):
            return "CONDITIONAL APPROVAL"
        if _re.search(r"EXCEEDS RISK|exceeds.*drawdown|drawdown.*exceeds", md, _re.I):
            return "EXCEEDS RISK LIMIT"
        if _re.search(r"APPROVED|within mandate|within.*constraint", md, _re.I):
            return "PORTFOLIO READY"
        return "WITHIN MANDATE"

    if report_kind == "portfolio":
        verdict_patterns = [
            r"\bRating:\s*(Conservative|Balanced|Aggressive|Speculative)\b",
            r"\bRisk Level:\s*([A-Za-z-]+)\b",
            r"\bRisk Profile:\s*([A-Za-z-]+)\b",
        ]
    else:
        verdict_patterns = [
            # Quick View headline: "Fundamental: HOLD 🟡" / "الأساسيات: REDUCE"
            r'\bFundamental:\s*\*?\*?\s*(BUY|HOLD|SELL|REDUCE|AVOID|ACCUMULATE|WATCHLIST)\b',
            r'\bالأساسيات:\s*\*?\*?\s*(BUY|HOLD|SELL|REDUCE|AVOID|ACCUMULATE)\b',
            # Scorecard / table cell fallback
            r"\|\s*(BUY|HOLD|SELL|WATCHLIST|ACCUMULATE|REDUCE)\b",
            r"\bVerdict Type:\s*([A-Za-z -]+)\b",
        ]
    for pattern in verdict_patterns:
        match = _re.search(pattern, markdown or "", flags=_re.I)
        if match:
            return match.group(1).strip().upper() if report_kind == "stock" else match.group(1).strip()
    return "Analysis Ready" if report_kind == "stock" else "Portfolio Review Ready"


def _staging_extract_confidence(markdown: str) -> Optional[str]:
    percent_match = _re.search(r"\bConfidence:\s*(\d+%)", markdown or "", flags=_re.I)
    if percent_match:
        return percent_match.group(1)
    conviction_match = _re.search(r"\bConviction:\s*([A-Za-z-]+)", markdown or "", flags=_re.I)
    if conviction_match:
        return conviction_match.group(1).strip().title()
    return None


def _staging_extract_risk_level(markdown: str) -> Optional[str]:
    top_risk = _re.search(r"Top Risk:\s*.*?Severity:\s*([A-Za-z-]+)", markdown or "", flags=_re.I)
    if top_risk:
        return top_risk.group(1).strip().title()
    total_risk = _re.search(r"Risk profile \(total beta\):\s*([A-Za-z-]+)", markdown or "", flags=_re.I)
    if total_risk:
        return total_risk.group(1).strip().title()
    risk_level = _re.search(r"\bRisk Level:\s*([A-Za-z-]+)", markdown or "", flags=_re.I)
    if risk_level:
        return risk_level.group(1).strip().title()
    risk_profile = _re.search(r"\bRisk Profile:\s*([A-Za-z-]+)", markdown or "", flags=_re.I)
    if risk_profile:
        return risk_profile.group(1).strip().title()
    return None


def _staging_extract_summary(markdown: str, report_kind: str, result_type: str = "") -> str:
    # ── Construction: pull from Portfolio Overview / Strategy Readiness ──────
    if result_type == "construction":
        overview = _staging_extract_section(markdown, ["Portfolio Overview", "Strategy Readiness"])
        for paragraph in _re.split(r"\n\s*\n", overview or ""):
            cleaned = _staging_clean_text(paragraph)
            # skip tables, bullets, headings, short lines
            if cleaned and len(cleaned) > 40 and not cleaned.startswith("|") and not cleaned.startswith("-") and not cleaned.startswith("#"):
                # Return max 2 sentences
                sentences = _re.findall(r"[^.!?]+[.!?]+", cleaned)
                return " ".join(sentences[:2]).strip() or cleaned[:280]
        # Fallback: first meaningful line from whole report
        for line in (markdown or "").splitlines():
            c = _staging_clean_text(line)
            if c and len(c) > 40 and not c.startswith("|") and not c.startswith("-") and not c.startswith("#"):
                return c[:280]
        return "Portfolio constructed and ready for review."

    markers = ["Executive Summary", "Executive Assessment"]
    section = _staging_extract_section(markdown, markers)
    for paragraph in _re.split(r"\n\s*\n", section or markdown or ""):
        cleaned = _staging_clean_text(paragraph)
        if cleaned and not cleaned.lower().startswith("metric "):
            return cleaned
    return _staging_clean_text(markdown)[:280]


def _staging_extract_insights(markdown: str, report_kind: str, summary: str, result_type: str = "") -> list[str]:
    # ── Construction insights: return / volatility / concentration ───────────
    if result_type == "construction":
        insights: list[str] = []
        # 1) Expected return
        ret_m = _re.search(r"Expected Annual Return[:\s]+([0-9.]+%[^.\n]{0,60})", markdown or "", _re.I)
        if ret_m:
            insights.append("Expected annual return: " + ret_m.group(1).strip().rstrip(","))
        # 2) Volatility / Sharpe / drawdown
        vol_m = _re.search(r"Annual Volatility[:\s]+([0-9.]+%[^.\n]{0,80})", markdown or "", _re.I)
        shr_m = _re.search(r"Sharpe Ratio[:\s]+([0-9.]+[^.\n]{0,60})", markdown or "", _re.I)
        if vol_m and shr_m:
            insights.append(f"Volatility {vol_m.group(1).strip().rstrip(',')} · Sharpe {shr_m.group(1).strip()}")
        elif vol_m:
            insights.append("Annual volatility: " + vol_m.group(1).strip().rstrip(","))
        # 3) Concentration / correlation risk from the report
        conc_m = _re.search(
            r"(Concentration[^.\n]{10,120}|correlation[^.\n]{10,120}|Effective N[^.\n]{10,120})",
            markdown or "", _re.I
        )
        if conc_m:
            insights.append(_staging_clean_text(conc_m.group(1)))
        # Fill with fallbacks if fewer than 3
        fallbacks = [
            "Allocation generated and ready for review.",
            "Open the full report for detailed asset rationale.",
            "Risk constraints and implementation plan included.",
        ]
        for fb in fallbacks:
            if len(insights) >= 3:
                break
            if fb not in insights:
                insights.append(fb)
        return insights[:3]

    candidates = []
    tracked_lines = [
        _staging_extract_marked_line(markdown, ["Top Risk:", "Risk Level:", "Risk Profile:"]),
        _staging_extract_marked_line(markdown, ["Technical Signal (Supporting):", "Decision Discipline Layer", "Alpha driver:"]),
        _staging_extract_marked_line(markdown, ["Next Earnings:", "NEAR-TERM CATALYST", "Decision Boundary:"]),
    ]
    for line in tracked_lines:
        if line and line not in candidates:
            candidates.append(line)
    for sentence in _staging_extract_sentences(summary):
        if sentence not in candidates:
            candidates.append(sentence)
    if report_kind == "portfolio":
        portfolio_note = _staging_extract_marked_line(markdown, ["EisaX Risk Assessment", "Portfolio Assessment"])
        if portfolio_note and portfolio_note not in candidates:
            candidates.append(portfolio_note)
    return candidates[:3] or [
        "Live analysis completed and ready for deeper review.",
        "Open the full report to inspect the detailed assumptions.",
        "Use the email form below to request the institutional report.",
    ]


def _staging_prepare_query(query: str) -> str:
    normalized = (query or "").strip()
    if not normalized:
        return normalized

    analyze_match = _re.match(r"(?i)^analyze\s+(.+)$", normalized)
    if analyze_match:
        subject = analyze_match.group(1).strip()
        return (
            f"Analyze {subject} for an investment decision. "
            "Respond in concise institutional tone and keep the answer under 350 words. "
            "Include a clear BUY, HOLD, or SELL verdict, a short executive summary, risk level, "
            "confidence if available, and exactly three key insights."
        )

    if _re.search(r"(?i)\bportfolio\b", normalized):
        return (
            f"{normalized}. Respond in institutional tone with a concise allocation rationale, risk framing, "
            "and portfolio construction logic."
        )

    return normalized


def _resolution_error_response(resolution: dict) -> JSONResponse:
    status = resolution.get("resolution_status") or "unresolved"
    normalized_query = resolution.get("normalized_query") or resolution.get("query_raw") or "this request"
    if status == "ambiguous":
        detail = f"'{normalized_query}' is ambiguous. Please specify the exact instrument."
    else:
        detail = f"Unable to resolve '{normalized_query}' safely. Please use the exact ticker."
    payload = {"ok": False, "detail": detail, **resolution}
    return JSONResponse(status_code=400, content=payload)


_INTENT_ASSET_ANALYSIS = "asset_analysis"
_INTENT_PORTFOLIO_ANALYSIS = "portfolio_analysis"
_INTENT_PORTFOLIO_CONSTRUCTION = "portfolio_construction"
_INTENT_FIXED_INCOME = "fixed_income"


def _classify_request_intent(message: str, *, has_file: bool = False) -> str:
    raw = (message or "").strip()
    if has_file:
        return _INTENT_PORTFOLIO_ANALYSIS
    if not raw:
        return _INTENT_ASSET_ANALYSIS

    lowered = raw.lower()
    portfolio_analysis_keywords = [
        "upload",
        "csv",
        "xlsx",
        "xls",
        "my portfolio",
        "my holdings",
        "my positions",
        "analyze portfolio",
        "analyze my portfolio",
        "portfolio analysis",
        "portfolio review",
        "portfolio risk",
        "holdings analysis",
        "upload portfolio",
    ]
    portfolio_analysis_keywords_ar = [
        "محفظتي",
        "محفظتى",
        "حلل محفظتي",
        "حلل محفظتى",
        "تحليل محفظتي",
        "تحليل محفظتى",
        "ارفع",
        "رفع ملف",
        "csv",
        "اكسل",
        "إكسل",
    ]
    if any(keyword in lowered for keyword in portfolio_analysis_keywords):
        return _INTENT_PORTFOLIO_ANALYSIS
    if any(keyword in raw for keyword in portfolio_analysis_keywords_ar):
        return _INTENT_PORTFOLIO_ANALYSIS

    construction_keywords = [
        "portfolio",
        "build",
        "create",
        "construct",
        "design",
        "allocate",
        "rebalance",
        "re-balance",
        "allocation",
        "max down",
        "max drawdown",
        "risk tolerance",
        "drawdown",
        "aggressive",
        "conservative",
        "balanced",
        "moderate",
        "equities",
        "stocks",
        "saudi",
        "us equities",
        "american equities",
    ]
    construction_keywords_ar = [
        "محفظة",
        "محفظه",
        "ابني",
        "ابنى",
        "ابنِ",
        "كوّن",
        "كون",
        "خصص",
        "خصّص",
        "وزع",
        "وزّع",
        "أعد موازنة",
        "اعد موازنة",
        "موازنة",
        "عدواني",
        "عدوانيه",
        "محافظ",
        "متوازن",
        "مخاطرة",
        "مخاطر",
    ]

    english_hits = sum(1 for keyword in construction_keywords if keyword in lowered)
    arabic_hits = sum(1 for keyword in construction_keywords_ar if keyword in raw)
    if english_hits >= 2 or arabic_hits >= 1:
        return _INTENT_PORTFOLIO_CONSTRUCTION

    try:
        from portfolio_pipeline import is_pipeline_request as _is_pipeline_request

        if _is_pipeline_request(raw):
            return _INTENT_PORTFOLIO_CONSTRUCTION
    except Exception:
        pass

    # Fixed income / sukuk / bond / ISIN detection
    try:
        from core.fixed_income import is_fixed_income_query as _is_fi_query

        if _is_fi_query(raw):
            return _INTENT_FIXED_INCOME
    except Exception:
        pass

    return _INTENT_ASSET_ANALYSIS


def _portfolio_analysis_prompt_response(query: str) -> str:
    subject = (query or "your portfolio").strip()
    return (
        "## Portfolio Analysis Intake\n\n"
        f"To analyze {subject}, upload a CSV or Excel portfolio file and I will run a holdings-based risk and exposure review.\n\n"
        "### Required next step\n"
        "- Upload a `.csv`, `.xlsx`, or `.xls` file with your holdings.\n\n"
        "### What the analysis will cover\n"
        "- Position concentration and allocation gaps\n"
        "- Risk exposure by asset and market\n"
        "- Portfolio-level observations and next-step actions\n"
    )


def _should_resolve_direct_analysis_request(message: str) -> bool:
    raw = (message or "").strip()
    if not raw:
        return False
    if _classify_request_intent(raw) != _INTENT_ASSET_ANALYSIS:
        return False
    if not _re.search(
        r"(?i)^\s*(?:please\s+)?(?:analyze|analysis of|full analysis of|quick analysis of|brief analysis of)\b",
        raw,
    ) and not _re.search(r"^\s*(?:حلل|حللي|حللى|تحليل)\b", raw):
        return False

    lowered = raw.lower()
    blockers = [" and ", " vs ", " versus ", ",", "\n", "portfolio", "compare", "محفظة", "قارن"]
    return not any(token in lowered for token in blockers)


def _staging_shape_result(
    *,
    query: str,
    report_text: str,
    report_kind: str,
    mode: str,
    download_url: Optional[str] = None,
    html_report: Optional[str] = None,
    report_json: Optional[dict] = None,
    resolution: Optional[dict] = None,
    result_type: str = "",
) -> dict:
    summary = _staging_extract_summary(report_text, report_kind=report_kind, result_type=result_type)
    confidence = _staging_extract_confidence(report_text)
    payload = {
        "ok": True,
        "mode": mode,
        "query": query,
        "report_kind": report_kind,
        "result_type": result_type or report_kind,
        "summary": summary,
        "verdict": _staging_extract_verdict(report_text, report_kind=report_kind, result_type=result_type),
        "risk_level": _staging_extract_risk_level(report_text),
        "confidence": confidence,
        "insights": _staging_extract_insights(report_text, report_kind=report_kind, summary=summary, result_type=result_type),
        "download_url": download_url,
        "teaser": (report_text[:500] + "...") if len(report_text) > 500 else report_text,
        "full_report": html_report or report_text,
        "report_json": report_json,
    }
    if resolution:
        payload["resolution"] = resolution
    return payload


def _staging_fallback_payload(query: str, file_name: Optional[str] = None) -> dict:
    subject = file_name or query or "your request"
    fallback_report = (
        "## Agent Preview\n\n"
        f"The live analysis service is temporarily unavailable, so this staging page is showing a fallback preview for {subject}.\n\n"
        "### What You Can Expect\n"
        "- Institutional summary with verdict and risk framing.\n"
        "- Three key insights surfaced from the live agent workflow.\n"
        "- A full report handoff path after lead capture.\n"
    )
    return _staging_shape_result(
        query=query or f"Analyze {file_name}" if file_name else "Analyze request",
        report_text=fallback_report,
        report_kind="portfolio" if file_name else "stock",
        mode="fallback",
    )


def _staging_proxy_chat(query: str, user_id: str) -> dict:
    import requests as _requests

    response = _requests.post(
        f"{_STAGING_UPSTREAM_BASE}/v1/chat",
        headers={**_staging_proxy_headers(), "Content-Type": "application/json"},
        json={"message": _staging_prepare_query(query), "user_id": user_id},
        timeout=120,
    )
    try:
        body = response.json()
    except Exception:
        body = {"detail": response.text[:500]}
    return {"status_code": response.status_code, "body": body}


def _staging_proxy_upload(file_name: str, file_bytes: bytes, user_id: str) -> dict:
    import requests as _requests

    response = _requests.post(
        f"{_STAGING_UPSTREAM_BASE}/v1/upload-portfolio",
        headers=_staging_proxy_headers(),
        files={"file": (file_name, file_bytes)},
        data={"user_id": user_id},
        timeout=120,
    )
    try:
        body = response.json()
    except Exception:
        body = {"detail": response.text[:500]}
    return {"status_code": response.status_code, "body": body}


def _staging_upstream_error(status_code: int, body: dict, default_message: str) -> HTTPException:
    if status_code == 429:
        return HTTPException(status_code=429, detail="The live agent is busy right now. Please retry in a moment.")
    if status_code == 413:
        return HTTPException(status_code=413, detail="The uploaded file is too large for the current staging limit.")
    if status_code == 400:
        return HTTPException(status_code=400, detail=body.get("error") or body.get("detail") or default_message)
    return HTTPException(status_code=502, detail=default_message)


@app.get("/staging-api/health")
@limiter.limit("30/minute")
async def staging_public_health(request: Request):
    _ensure_staging_public_enabled()
    return {"status": "ok"}


# ── Market Updates — public read, admin generate ──────────────────────────────

@app.get("/staging-api/updates")
@limiter.limit("60/minute")
async def api_updates_latest(request: Request):
    """Return latest daily + weekly updates with LinkedIn-formatted text."""
    import json as _json
    from core.services.market_updates import get_latest_updates
    result = get_latest_updates()
    if not result:
        return JSONResponse({"daily": None, "weekly": None}, status_code=200)
    print("DEBUG /api/updates response:", _json.dumps(result)[:1000])
    return result


@app.get("/staging-api/updates/daily")
@limiter.limit("60/minute")
async def api_updates_daily(request: Request):
    from core.services.market_updates import get_latest_updates
    result = get_latest_updates()
    daily = result.get("daily")
    if not daily:
        return JSONResponse({"detail": "No daily update available yet"}, status_code=404)
    return daily


@app.get("/staging-api/updates/weekly")
@limiter.limit("60/minute")
async def api_updates_weekly(request: Request):
    from core.services.market_updates import get_latest_updates
    result = get_latest_updates()
    weekly = result.get("weekly")
    if not weekly:
        return JSONResponse({"detail": "No weekly update available yet"}, status_code=404)
    return weekly


@app.post("/staging-api/updates/generate")
@limiter.limit("5/minute")
async def api_updates_generate(request: Request):
    """Admin-only: trigger generation of daily + weekly updates."""
    _require_admin_cookie(request)
    import asyncio as _aio
    from core.services.market_updates import generate_daily_update, generate_weekly_update
    loop = _aio.get_event_loop()
    daily  = await loop.run_in_executor(None, generate_daily_update)
    weekly = await loop.run_in_executor(None, generate_weekly_update)
    return {"status": "generated", "daily": daily, "weekly": weekly}


@app.post("/staging-api/analyze")
@limiter.limit("12/minute")
async def staging_public_analyze(
    request: Request,
    query: str = Form(""),
    symbol: str = Form(""),
    skip_resolution: bool = Form(False),
    report_language: str = Form("en"),
    language: str = Form(""),
    file: UploadFile = File(None),
):
    _ensure_staging_public_enabled()
    normalized_query = (query or "").strip()
    selected_symbol = (symbol or "").strip().upper()
    preferred_language = (report_language or language or "en").strip().lower() or "en"
    intent = _classify_request_intent(normalized_query, has_file=file is not None)
    if not normalized_query and not file and not selected_symbol:
        raise HTTPException(status_code=400, detail="Enter a request or upload a portfolio file.")

    user_id = _staging_client_id(request)
    try:
        if file is not None:
            file_bytes = await file.read()
            live_payload = await run_in_threadpool(
                _staging_proxy_upload,
                file.filename or "portfolio.csv",
                file_bytes,
                user_id,
            )
            if live_payload["status_code"] >= 500:
                raise RuntimeError(f"upstream upload error {live_payload['status_code']}")
            if live_payload["status_code"] >= 400:
                raise _staging_upstream_error(
                    live_payload["status_code"],
                    live_payload["body"],
                    "Portfolio analysis is temporarily unavailable.",
                )
            report_text = live_payload["body"].get("analysis") or ""
            if not report_text:
                raise HTTPException(status_code=502, detail="Portfolio analysis returned no content.")
            return _staging_shape_result(
                query=normalized_query or f"Analyze {file.filename}",
                report_text=report_text,
                report_kind="portfolio",
                mode="live",
                download_url=None,
            )

        from core.services.entity_resolution import EntityResolution, is_exact_ticker, resolve_asset_entity
        from core.services.market_route_handler import handle_stock_analysis as _handle_stock_analysis
        from portfolio_builder import detect_and_build as _detect_and_build

        if skip_resolution and selected_symbol:
            exact_symbol = is_exact_ticker(selected_symbol)
            if not exact_symbol:
                return _resolution_error_response(
                    {
                        "query_raw": normalized_query or selected_symbol,
                        "normalized_query": selected_symbol,
                        "resolution_status": "unresolved",
                    }
                )
            resolution = EntityResolution(
                query_raw=normalized_query or selected_symbol,
                normalized_query=selected_symbol,
                resolution_status="resolved",
                symbol=exact_symbol.symbol,
                market=exact_symbol.market,
                asset_type=exact_symbol.asset_type,
                currency=exact_symbol.currency,
                resolution_source="exact_symbol",
                confidence="high",
                name=exact_symbol.name,
                local_name=exact_symbol.local_name,
                exchange=exact_symbol.exchange,
                universe_source=exact_symbol.source_tag,
            )
        elif intent == _INTENT_PORTFOLIO_ANALYSIS:
            if not file:
                return _staging_shape_result(
                    query=normalized_query,
                    report_text=_portfolio_analysis_prompt_response(normalized_query),
                    report_kind="portfolio",
                    mode="guided",
                    download_url=None,
                )
        elif intent == _INTENT_PORTFOLIO_CONSTRUCTION:
            logger.info(
                "[intent-gate][staging] portfolio construction routed to builder: %s",
                normalized_query,
            )
            report_text = await run_in_threadpool(_detect_and_build, normalized_query)
            if not report_text:
                raise HTTPException(status_code=502, detail="Portfolio builder returned no content.")
            return _staging_shape_result(
                query=normalized_query,
                report_text=report_text,
                report_kind="portfolio",
                result_type="construction",
                mode="live",
                download_url=None,
            )
        elif intent == _INTENT_FIXED_INCOME:
            logger.info(
                "[intent-gate][staging] fixed-income query routed to finance agent: %s",
                normalized_query,
            )
            try:
                from core.agents.finance import FinancialAgent as _FinancialAgent

                _fi_agent = _FinancialAgent()
                _fi_result = await run_in_threadpool(
                    _fi_agent.think,
                    normalized_query,
                    {"session_id": f"staging-fi-{int(_time.time() * 1000)}"},
                    {"model": os.getenv("MODEL_NAME", "")},
                )
                report_text = _fi_result.get("reply") or ""
                if not report_text:
                    raise HTTPException(status_code=502, detail="Fixed income analysis returned no content.")
                return _staging_shape_result(
                    query=normalized_query,
                    report_text=report_text,
                    report_kind="fixed_income",
                    result_type="fixed_income",
                    mode="live",
                    download_url=None,
                )
            except HTTPException:
                raise
            except Exception as _fi_err:
                logger.error("[fixed-income][staging] error: %s", _fi_err, exc_info=True)
                raise HTTPException(status_code=502, detail=f"Fixed income analysis failed: {_fi_err}")
        else:
            # ── Fast pre-route: local ticker index (O(1)) before LLM resolution ──
            _idx_resolution: Optional[EntityResolution] = None
            try:
                from core.ticker_index import quick_scan as _quick_scan
                _idx_hit = _quick_scan(normalized_query or selected_symbol)
                if _idx_hit:
                    _idx_resolution = EntityResolution(
                        query_raw=normalized_query,
                        normalized_query=_idx_hit.symbol,
                        resolution_status="resolved",
                        symbol=_idx_hit.symbol,
                        market=_idx_hit.market,
                        asset_type=_idx_hit.asset_type,
                        currency=_idx_hit.currency,
                        resolution_source="ticker_index",
                        confidence="high",
                        name=_idx_hit.name,
                        exchange=_idx_hit.exchange,
                    )
                    logger.info(
                        "[ticker_index] pre-routed '%s' → %s / %s (bypassing entity resolution)",
                        normalized_query, _idx_hit.symbol, _idx_hit.market,
                    )
            except Exception as _idx_err:
                logger.debug("[ticker_index] scan error (non-fatal): %s", _idx_err)
                _idx_resolution = None

            resolution = _idx_resolution or resolve_asset_entity(normalized_query)
        resolution_payload = resolution.to_dict()
        if not resolution.is_resolved:
            logger.info("[entity-resolution] %s", resolution_payload)
            return _resolution_error_response(resolution_payload)

        stock_instruction = resolution.analysis_instruction
        logger.info(
            "[entity-resolution] raw='%s' normalized='%s' -> %s / %s via %s (%s)",
            normalized_query or selected_symbol,
            resolution.normalized_query,
            resolution.symbol,
            resolution.market,
            resolution.resolution_source,
            resolution.confidence,
        )
        started_at = _time.perf_counter()
        session_id = f"staging-{int(_time.time() * 1000)}"
        orchestrator.session_mgr.get_or_create_session(
            user_id,
            session_id=session_id,
            ip=request.headers.get("X-Real-IP") or str(request.client.host),
            user_agent=request.headers.get("User-Agent", ""),
        )
        live_payload = await _handle_stock_analysis(
            orchestrator=orchestrator,
            session_id=session_id,
            user_id=user_id,
            message=stock_instruction,
            instruction=stock_instruction,
            user_ctx={
                "market": resolution.market,
                "asset_type": resolution.asset_type,
                "currency": resolution.currency,
                "resolution_source": resolution.resolution_source,
                "resolution_confidence": resolution.confidence,
            },
        )
        report_text = live_payload.get("reply") or ""
        if not report_text:
            raise HTTPException(status_code=502, detail="Analysis returned no content.")
        report_json = None
        try:
            from core.services.pilot_report_json import build_pilot_report_json

            generated_at = datetime.now().astimezone().isoformat()
            report_json = build_pilot_report_json(
                symbol=resolution.symbol,
                market=resolution.market,
                language=preferred_language,
                report_text=report_text,
                analysis_data=live_payload.get("data") or {},
                system_version=_APP_VERSION,
                model_primary="DeepSeek V3",
                generated_at=generated_at,
                data_as_of=generated_at,
                latency_seconds=max(0, int(round(_time.perf_counter() - started_at))),
            )
        except Exception as exc:
            logger.warning("[staging-api] pilot json unavailable for '%s': %s", normalized_query, exc)
        return _staging_shape_result(
            query=normalized_query or selected_symbol,
            report_text=report_text,
            report_kind="stock",
            mode="live",
            download_url=live_payload.get("download_url"),
            html_report=report_text,
            report_json=report_json,
            resolution=resolution_payload,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("[staging-api] analyze degraded for '%s': %s", normalized_query or getattr(file, "filename", "portfolio"), exc)
        return JSONResponse(
            status_code=200,
            content=_staging_fallback_payload(
                normalized_query,
                file_name=getattr(file, "filename", None),
            ),
        )


@app.get("/staging-api/quick")
@limiter.limit("60/minute")
async def staging_quick_snapshot(request: Request, q: str = ""):
    """
    Fast pipeline-cache snapshot for a ticker — returns in <2s.
    Used by the frontend to show Quick View while full analysis loads.

    Returns: ticker, name, market, price, change, rsi, sma200,
             pe_ratio, div_yield, market_cap, sector, rsi_signal,
             sma_signal, quick_signal (BULLISH/NEUTRAL/BEARISH)
    """
    _ensure_staging_public_enabled()
    q = (q or "").strip().upper()
    if not q or len(q) > 20:
        raise HTTPException(status_code=400, detail="Invalid ticker")

    try:
        from pipeline import CacheManager as _QC
        import math as _qm

        _MARKETS_SUFFIX = {
            "uae": ".AE", "ksa": ".SR", "egypt": ".CA",
            "kuwait": ".KW", "qatar": ".QA", "bahrain": ".BH",
            "morocco": ".MA", "tunisia": ".TN",
            "america": "", "crypto": "-USD",
        }

        def _vf(v):
            try:
                f = float(v)
                return None if (_qm.isnan(f) or _qm.isinf(f) or f == 0) else f
            except (TypeError, ValueError):
                return None

        qcm = _QC()
        # Strip suffix to get bare ticker for cache lookup
        bare = q.split("-")[0]
        for sfx in (".AE", ".SR", ".CA", ".KW", ".QA", ".BH", ".MA", ".TN"):
            if bare.endswith(sfx):
                bare = bare[: -len(sfx)]
                break

        row = None
        mkt_found = None
        for mkt, sfx in _MARKETS_SUFFIX.items():
            df, _ = qcm.get_latest(mkt)
            if df is None or df.empty:
                continue
            df = df.copy()
            df["_bare"] = df["ticker"].astype(str).str.split(":").str[-1].str.upper()
            hits = df[df["_bare"] == bare]
            if not hits.empty:
                row = hits.iloc[0].to_dict()
                mkt_found = mkt
                break

        if row is None:
            raise HTTPException(status_code=404, detail=f"Ticker '{q}' not found in pipeline cache")

        sfx = _MARKETS_SUFFIX.get(mkt_found, "")
        full_ticker = bare + sfx

        price  = _vf(row.get("close"))
        change = round(float(row.get("change") or 0), 2)
        rsi    = _vf(row.get("RSI"))
        sma50  = _vf(row.get("SMA50"))
        sma200 = _vf(row.get("SMA200"))
        pe     = _vf(row.get("price_earnings_ttm"))
        divy   = _vf(row.get("dividend_yield_recent"))
        mc     = _vf(row.get("market_cap_basic"))
        sector = str(row.get("sector") or "").strip() or None
        macd   = _vf(row.get("MACD.macd"))
        macd_s = _vf(row.get("MACD.signal"))

        # RSI signal
        if rsi is not None:
            if rsi < 30:
                rsi_signal = "oversold"
            elif rsi > 70:
                rsi_signal = "overbought"
            else:
                rsi_signal = "neutral"
        else:
            rsi_signal = "unknown"

        # SMA200 signal
        if price and sma200:
            sma_signal = "above" if price > sma200 else "below"
        else:
            sma_signal = "unknown"

        # Simple composite quick signal (3 independent signals → majority vote)
        _bull, _bear = 0, 0
        if rsi_signal == "oversold":      _bull += 1
        elif rsi_signal == "overbought":  _bear += 1
        if sma_signal == "above":         _bull += 1
        elif sma_signal == "below":       _bear += 1
        if macd is not None and macd_s is not None:
            if macd > macd_s:  _bull += 1
            else:              _bear += 1
        if change > 1.5:       _bull += 0.5
        elif change < -1.5:    _bear += 0.5

        if _bull >= 2:         quick_signal = "BULLISH"
        elif _bear >= 2:       quick_signal = "BEARISH"
        else:                  quick_signal = "NEUTRAL"

        return {
            "ticker":      full_ticker,
            "name":        str(row.get("name") or bare),
            "market":      mkt_found.upper() if mkt_found else "UNKNOWN",
            "price":       round(price, 4) if price else None,
            "change":      change,
            "rsi":         round(rsi, 1) if rsi else None,
            "rsi_signal":  rsi_signal,
            "sma50":       round(sma50, 4) if sma50 else None,
            "sma200":      round(sma200, 4) if sma200 else None,
            "sma_signal":  sma_signal,
            "pe_ratio":    round(pe, 2) if pe else None,
            "div_yield":   round(divy, 2) if divy else None,
            "market_cap":  round(mc / 1e9, 2) if mc else None,
            "sector":      sector,
            "quick_signal": quick_signal,
        }
    except HTTPException:
        raise
    except Exception as _qe:
        logger.warning("[quick] error for %s: %s", q, _qe)
        raise HTTPException(status_code=500, detail="Quick snapshot unavailable")


@app.post("/staging-api/lead")
@limiter.limit("20/minute")
async def staging_public_lead(request: Request, payload: StagingLeadPayload):
    _ensure_staging_public_enabled()
    email = (payload.email or "").strip().lower()
    if not _re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")

    lead_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "email": email,
        "name": (payload.name or "").strip() or None,
        "query": (payload.query or "").strip() or None,
        "report_kind": (payload.report_kind or "").strip() or None,
        "ip": (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.headers.get("X-Real-IP")
            or str(request.client.host)
        ),
        "user_agent": request.headers.get("User-Agent", "")[:300],
    }
    with _STAGING_LEADS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(_json.dumps(lead_record, ensure_ascii=False) + "\n")
    return {"ok": True, "message": "Lead captured."}

@app.get("/")
async def root():
    return RedirectResponse(url="https://eisax.com", status_code=301)

@app.get("/v1/chart-data")
@limiter.limit("30/minute")
async def chart_data(request: Request, ticker: str = "NVDA", access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
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

@app.post("/v1/upload-portfolio")
@limiter.limit("10/minute")
async def upload_portfolio(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form("admin"),
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token")
):
    """Upload CSV/Excel portfolio file and analyze it"""
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        contents = await file.read()

        if len(contents) > 5 * 1024 * 1024:
            return {"error": "File too large. Maximum allowed size is 5MB."}

        # Parse file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Only CSV or Excel files supported"}
        
        # Normalize columns
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Find ticker and weight columns
        ticker_col = next((c for c in df.columns if c in ['ticker','symbol','stock','name']), df.columns[0])
        weight_col = next((c for c in df.columns if c in ['weight','allocation','%','percent','value']), None)
        
        tickers = df[ticker_col].str.upper().tolist()
        
        if weight_col:
            weights = df[weight_col].tolist()
            # Normalize to percentages
            total = sum(float(w) for w in weights)
            if total > 1.5:  # assume percentages not decimals
                weights = [float(w)/100 for w in weights]
            else:
                weights = [float(w) for w in weights]
        else:
            weights = [1/len(tickers)] * len(tickers)
        
        portfolio = dict(zip(tickers, weights))
        # Always work on normalized total-portfolio weights (including cash-like rows)
        # to avoid hidden basis-mismatch between "total portfolio" and "equity sleeve".
        _sum_input_w = sum(float(w) for w in portfolio.values())
        if _sum_input_w <= 0:
            return {"error": "Invalid portfolio weights: sum must be > 0"}
        portfolio = {t: float(w) / _sum_input_w for t, w in portfolio.items()}
        
        # Build Portfolio Risk Report
        import yfinance as yf
        import numpy as np
        from dotenv import load_dotenv
        load_dotenv(str(ENV_FILE))
        def _to_float(v):
            try:
                f = float(v)
                return f if np.isfinite(f) else None
            except Exception:
                return None
        def _safe_round(v, n=3):
            f = _to_float(v)
            return round(f, n) if f is not None else None

        def _clean_nan(obj):
            """Recursively replace NaN/Inf with None — prevents JSON serialization crash."""
            if isinstance(obj, dict):
                return {k: _clean_nan(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean_nan(v) for v in obj]
            if isinstance(obj, float):
                return None if (obj != obj or obj == float('inf') or obj == float('-inf')) else obj
            return obj

        def _fmt_safe(val, fmt=".1f", fallback="N/A"):
            """Format a float safely — return fallback if NaN/None."""
            if val is None:
                return fallback
            try:
                f = float(val)
                if f != f:  # NaN check
                    return fallback
                return f"{f:{fmt}}"
            except (ValueError, TypeError):
                return fallback

        valid_tickers = [t for t in tickers if t.upper() not in ["CASH","USD","AED"]]
        valid_weights_total = {t: portfolio.get(t, 0.0) for t in valid_tickers}
        
        # Normalize weights
        total_w = sum(valid_weights_total.values())
        equity_alloc_total = total_w
        if total_w > 0:
            # Equity-sleeve normalized weights (sum to 100% across non-cash assets)
            valid_weights = {t: w / total_w for t, w in valid_weights_total.items()}
        else:
            valid_weights = {}

        # ── Fetch 1yr price history + fundamentals ──────────────────────────────
        RF_RATE = 0.045   # US T-Bill risk-free rate (4.5%)
        LOOKBACK = "1y"

        price_data = {}
        stock_info = {}
        for t in valid_tickers:
            try:
                tk = yf.Ticker(t)
                hist = tk.history(period=LOOKBACK)
                if hist.empty or hist["Close"].isna().all():
                    hist = tk.history(period="6mo")  # fallback to 6M
                if not hist.empty and not hist["Close"].isna().all():
                    price_data[t] = hist["Close"].dropna()
                info = tk.info
                # trailingAnnualDividendYield is a reliable decimal fraction (0.004 = 0.4%).
                # dividendYield returns as a percentage value (0.4 for 0.4%) — avoid it.
                _raw_dy = float(info.get("trailingAnnualDividendYield") or 0)
                _safe_dy = min(max(_raw_dy, 0.0), 0.15)   # clamp decimal [0, 15%]
                stock_info[t] = {
                    "price":     info.get("regularMarketPrice") or info.get("previousClose", 0),
                    "beta":      _to_float(info.get("beta")),
                    "sector":    info.get("sector", "N/A"),
                    "pe":        _to_float(info.get("trailingPE")),
                    "mktcap":    _to_float(info.get("marketCap")),
                    "div_yield": _safe_dy,   # current market yield, NOT yield-on-cost
                }
            except Exception as _fe:
                logger.debug("Stock data fetch failed for %s: %s", t, _fe)

        # ── Benchmark: S&P 500 ───────────────────────────────────────────────
        spx_return = None
        try:
            spx = yf.Ticker("^GSPC").history(period=LOOKBACK)
            if spx.empty or spx["Close"].isna().all():
                spx = yf.Ticker("^GSPC").history(period="6mo")
            if not spx.empty:
                spx_return = float((spx["Close"].dropna().iloc[-1] / spx["Close"].dropna().iloc[0]) - 1)
        except Exception:
            pass

        # ── Calculate metrics ────────────────────────────────────────────────
        corr_matrix_str = ""
        high_corr = []
        sortino = 0.0
        max_dd = 0.0
        max_dd_duration = 0
        port_total_return = 0.0
        cvar_95 = 0.0
        rolling_sharpe_str = ""
        rolling_sharpe_now = None
        sector_concentration = {}
        factor_exposure = {}
        port_beta_equity = None

        if len(price_data) >= 2:
            prices_df = pd.DataFrame(price_data).dropna()
            returns_df = prices_df.pct_change().dropna()

            w_arr = np.array([valid_weights.get(t, 0) for t in returns_df.columns])
            w_arr = w_arr / w_arr.sum() if w_arr.sum() > 0 else w_arr

            port_returns = returns_df.values @ w_arr
            ann_return   = float(np.mean(port_returns) * 252)
            ann_vol      = float(np.std(port_returns) * np.sqrt(252))
            rf_daily     = (1 + RF_RATE) ** (1/252) - 1
            excess       = port_returns - rf_daily
            sharpe       = round(float(np.mean(excess) / np.std(excess) * np.sqrt(252)), 2) if np.std(excess) > 0 else 0

            # Sortino (downside deviation only)
            neg_excess   = excess[excess < 0]
            downside_std = float(np.std(neg_excess) * np.sqrt(252)) if len(neg_excess) > 0 else ann_vol
            sortino      = round(float(np.mean(excess) * 252 / downside_std), 2) if downside_std > 0 else 0

            # VaR 95% and CVaR 95% (Expected Shortfall — tail risk beyond VaR)
            var_95  = float(np.percentile(port_returns, 5) * 100)
            cvar_95 = float(port_returns[port_returns <= np.percentile(port_returns, 5)].mean() * 100)

            # Max drawdown + duration (calendar days in drawdown)
            cum      = (1 + pd.Series(port_returns)).cumprod()
            roll_max = cum.cummax()
            dd       = (cum - roll_max) / roll_max
            max_dd   = float(dd.min() * 100)
            # Duration: longest consecutive streak below previous peak
            in_dd = (dd < -0.001).astype(int)
            streak = max_dd_duration = 0
            for v in in_dd:
                streak = streak + 1 if v else 0
                max_dd_duration = max(max_dd_duration, streak)

            # Total return
            port_total_return = float((cum.iloc[-1] - 1) * 100)

            # Rolling 63-day Sharpe (quarterly window, annualised)
            if len(port_returns) >= 63:
                roll_win = 63
                roll_sh  = []
                for i in range(roll_win, len(port_returns) + 1):
                    w_ret = port_returns[i - roll_win:i]
                    ex    = w_ret - rf_daily
                    s     = float(np.mean(ex) / np.std(ex) * np.sqrt(252)) if np.std(ex) > 0 else 0
                    roll_sh.append(round(s, 2))
                if roll_sh:
                    rs_min = min(roll_sh); rs_max = max(roll_sh); rs_now = roll_sh[-1]
                    rolling_sharpe_now = float(rs_now)
                    rs_trend = "↗ Improving" if rs_now > np.mean(roll_sh) else "↘ Declining"
                    rolling_sharpe_str = (
                        f"Current (last 63d): **{rs_now:.2f}** | "
                        f"Range: {rs_min:.2f} → {rs_max:.2f} | {rs_trend}"
                    )

            # Correlation matrix
            corr = returns_df.corr()
            # ── Smart correlation insight: weight-adjusted cluster risk ──────
            # Don't just flag pairs >0.70; compute effective diversification ratio
            n_assets   = len(corr)
            avg_corr   = float(corr.values[np.triu_indices(n_assets, k=1)].mean()) if n_assets > 1 else 0
            # Effective N (Herfindahl on corr eigenvalues → measures true diversification)
            eigvals    = np.linalg.eigvalsh(corr.values)
            eigvals    = np.maximum(eigvals, 0)
            eff_n      = (eigvals.sum() ** 2) / (eigvals ** 2).sum() if (eigvals**2).sum() > 0 else 1
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    c = corr.iloc[i, j]
                    if abs(c) > 0.70:
                        high_corr.append(f"{corr.columns[i]}/{corr.columns[j]}: {c:.2f}")
            cols_c      = list(corr.columns)
            corr_header = "| " + " | ".join([""] + cols_c) + " |"
            corr_sep    = "|" + "|".join(["---"] * (len(cols_c) + 1)) + "|"
            corr_rows   = [corr_header, corr_sep]
            for row_t in cols_c:
                row_vals = [f"{corr.loc[row_t, col_t]:.2f}" for col_t in cols_c]
                corr_rows.append("| " + row_t + " | " + " | ".join(row_vals) + " |")
            corr_matrix_str = "\n".join(corr_rows)

            # Sector concentration (Herfindahl index)
            for t in valid_tickers:
                s = stock_info.get(t, {}).get("sector", "Unknown")
                sector_concentration[s] = sector_concentration.get(s, 0) + valid_weights.get(t, 0)
            hhi = sum(v**2 for v in sector_concentration.values())  # 0=diversified, 1=concentrated

        else:
            ann_return, ann_vol, sharpe, var_95, port_beta_equity = 0, 0, 0, 0, None
            port_total_return, max_dd, avg_corr, eff_n, hhi = 0.0, 0.0, 0.0, 1.0, 1.0

        # Keep both bases explicit:
        # - equity basis: normalized over non-cash holdings
        # - total basis: normalized over all uploaded rows (including cash-like rows)
        def _weighted_beta(weights_map):
            _num = 0.0
            _den = 0.0
            _coverage = 0.0
            for _t, _w in weights_map.items():
                _b = stock_info.get(_t, {}).get("beta")
                if _b is None:
                    continue
                _num += float(_b) * float(_w)
                _den += float(_w)
                _coverage += float(_w)
            if _den <= 0:
                return None, 0.0
            return float(_num / _den), float(_coverage)

        port_beta_equity, beta_cov_equity = _weighted_beta(valid_weights)
        port_beta_total, beta_cov_total = _weighted_beta(valid_weights_total)
        port_beta = port_beta_equity
        def _beta_is_valid(v):
            return isinstance(v, (int, float)) and np.isfinite(v)
        def _fmt_beta(v):
            return f"{v:.2f}" if _beta_is_valid(v) else "N/A"
        _has_beta_total = _beta_is_valid(port_beta_total)
        _has_beta_equity = _beta_is_valid(port_beta_equity)

        # ── Historical Stress Tests (actual crisis returns, NOT linear beta) ──
        # Each scenario uses historically observed market returns + portfolio beta
        # adjusted by sector composition during that crisis.
        _CRISIS_SCENARIOS = [
            # (label, spx_actual, tech_multiplier, note)
            ("🔴 2022 Rate Shock (actual)",  -0.195, 1.45, "Tech -33% vs SPX -19.5% in 2022"),
            ("🔴 COVID Crash Mar 2020",       -0.340, 1.20, "SPX -34% in 33 days"),
            ("🔴 2008 GFC (Sep–Nov)",         -0.420, 0.90, "Financials led; tech -40%"),
            ("🔴 Bear Market -30% Scenario",  -0.300, 1.30, "Projected; tech typically amplified"),
            ("🟢 2023 AI Bull Run (actual)",  +0.265, 1.80, "Tech +55% vs SPX +26.5%"),
            ("🟢 Rate Cut Cycle +15%",        +0.150, 1.40, "Growth stocks benefit most"),
        ]
        tech_weight = sum(
            valid_weights.get(t, 0)
            for t in valid_tickers
            if stock_info.get(t, {}).get("sector", "").lower() in
               ["technology", "communication services", "consumer cyclical"]
        )
        scenario_lines = []
        for label, spx_ret, tech_mult, note in _CRISIS_SCENARIOS:
            # Blend: non-tech portion tracks beta linearly; tech amplified by sector mult
            non_tech_w = max(0, 1 - tech_weight)
            if _has_beta_equity:
                blended = (tech_weight * spx_ret * tech_mult) + (non_tech_w * spx_ret * port_beta_equity)
                icon = "🔴" if blended < 0 else "🟢"
                scenario_lines.append(
                    f"| {label} | {spx_ret*100:+.1f}% | **{blended*100:+.1f}%** | *{note}* |"
                )
            else:
                scenario_lines.append(
                    f"| {label} | {spx_ret*100:+.1f}% | **N/A** | *{note} · beta unavailable* |"
                )

        # ── Portfolio Dividend Yield (weighted) ──────────────────────────────
        port_div_yield_equity = sum(
            stock_info.get(t, {}).get("div_yield", 0) * valid_weights.get(t, 0)
            for t in valid_tickers
        ) * 100
        port_div_yield_total = sum(
            stock_info.get(t, {}).get("div_yield", 0) * valid_weights_total.get(t, 0)
            for t in valid_tickers
        ) * 100
        port_div_yield = port_div_yield_equity

        cash_pct = max(0.0, (1 - equity_alloc_total) * 100)
        port_total_return_total_est = port_total_return * equity_alloc_total

        # ══════════════════════════════════════════════════════════════════════
        # Build report
        # ══════════════════════════════════════════════════════════════════════
        lines = []
        from datetime import datetime
        now_str = datetime.now().strftime("%B %d, %Y")

        # ── Executive Summary ──────────────────────────────────────────────
        alpha = (port_total_return - (spx_return or 0) * 100) if spx_return is not None else None
        alpha_total_est = (port_total_return_total_est - (spx_return or 0) * 100) if spx_return is not None else None
        risk_label = (
            "Aggressive 🔴" if (_has_beta_total and port_beta_total > 1.5) else
            "Moderate 🟡" if (_has_beta_total and port_beta_total > 1.0) else
            "Conservative 🟢" if _has_beta_total else
            "Unknown ⚪"
        )
        elevated_risk = ((_has_beta_total and port_beta_total > 1.5) or (tech_weight >= 0.70) or (eff_n < 2.8) or (cvar_95 <= -3.5))
        verdict_line = (
            f"Equity sleeve returned **{_fmt_safe(port_total_return, '+.1f')}%**; estimated total-portfolio return "
            f"(including {_fmt_safe(cash_pct)}% cash) is **{_fmt_safe(port_total_return_total_est, '+.1f')}%**. "
            f"Vs S&P 500 **{_fmt_safe(spx_return * 100 if spx_return is not None else None, '+.1f')}%**: equity alpha **{_fmt_safe(alpha, '+.1f')}%**, "
            f"estimated total alpha **{_fmt_safe(alpha_total_est, '+.1f')}%**. "
            f"Risk profile (total beta): **{risk_label}** ({_fmt_beta(port_beta_total)}). Sharpe (equity sleeve): **~{_fmt_safe(sharpe)}**. "
            f"Strategic guidance: {'Strong performance, but concentration-driven with elevated risk — consider gradual de-risking and defensive diversification.' if elevated_risk else 'Returns are solid, but correlation clustering remains — consider incremental diversification while monitoring rolling Sharpe.' if high_corr else 'Strong historical performance with balanced risk controls — continue disciplined monitoring before adding incremental risk.'}"
            if spx_return is not None else
            f"Equity sleeve 1Y Return: **{_fmt_safe(port_total_return, '+.1f')}%** (estimated total portfolio: **{_fmt_safe(port_total_return_total_est, '+.1f')}%**). "
            f"Alpha: N/A (benchmark unavailable). "
            f"Risk: **{risk_label}** ({_fmt_beta(port_beta_total)}). Sharpe (equity sleeve): **~{_fmt_safe(sharpe)}**."
        )
        lines.append("# 📊 EisaX Portfolio Risk Report")
        lines.append(f"**Date:** {now_str}  |  **Period:** 1 Year  |  **Risk-Free Rate:** 4.5% (US T-Bill)")
        lines.append("")
        lines.append("## 🎯 Executive Summary")
        lines.append(f"> {verdict_line}")
        lines.append("")

        # ── Holdings ───────────────────────────────────────────────────────
        lines.append("## 📋 Holdings")
        lines.append("| Ticker | Weight (Total) | Weight (Equity Sleeve) | Sector | Beta | P/E | Div Yield |")
        lines.append("|--------|----------------|------------------------|--------|------|-----|-----------|")
        for t in valid_tickers:
            info  = stock_info.get(t, {})
            w_total_pct = portfolio.get(t, 0) * 100
            w_equity_pct = valid_weights.get(t, 0) * 100
            dy    = info.get("div_yield", 0) * 100
            _beta_str = _fmt_beta(info.get("beta"))
            _pe_v = info.get("pe")
            _pe_str = f"{_pe_v:.1f}" if isinstance(_pe_v, (int, float)) and np.isfinite(_pe_v) else "N/A"
            lines.append(
                f"| {t} | {w_total_pct:.1f}% | {w_equity_pct:.1f}% | {info.get('sector','N/A')} "
                f"| {_beta_str} "
                f"| {_pe_str} "
                f"| {dy:.2f}% |"
            )
        if cash_pct > 0.5:
            lines.append(f"| CASH | {cash_pct:.1f}% | — | — | 0 | — | 0% |")
        lines.append(
            f"\n> **Weighted Dividend Yield (Total Portfolio):** {port_div_yield_total:.2f}%  "
            f"| **Equity Sleeve:** {port_div_yield_equity:.2f}%"
        )

        # ── Risk Metrics ───────────────────────────────────────────────────
        lines.append("")
        lines.append("## 📈 Risk Metrics")
        lines.append("*Method: Historical Simulation (252 trading days) · rf = 4.5% · equity-sleeve normalized*")
        lines.append("")
        lines.append("| Metric | Value | Assessment |")
        lines.append("|--------|-------|------------|")
        lines.append(f"| 1Y Return (Equity Sleeve) | {_fmt_safe(port_total_return, '+.1f')}% | {'🟢 Strong' if port_total_return > 15 else '🟡 Moderate' if port_total_return > 0 else '🔴 Negative'} |")
        lines.append(f"| Estimated 1Y Return (Total Portfolio) | {_fmt_safe(port_total_return_total_est, '+.1f')}% | Includes {_fmt_safe(cash_pct)}% cash drag |")
        if spx_return is not None and alpha is not None:
            alpha_icon = "🟢" if alpha > 0 else "🔴"
            lines.append(f"| vs S&P 500 (Alpha) | {_fmt_safe(alpha, '+.1f')}% | {alpha_icon} {'Outperforming' if alpha > 0 else 'Underperforming'} benchmark |")
        elif spx_return is None:
            lines.append(f"| vs S&P 500 (Alpha) | N/A | Alpha: N/A (benchmark unavailable) |")
        lines.append(f"| Annualized Volatility | {_fmt_safe(ann_vol * 100 if isinstance(ann_vol, (int, float)) else None)}% | {'🔴 High' if ann_vol > 0.30 else '🟡 Moderate' if ann_vol > 0.15 else '🟢 Low'} |")
        lines.append(f"| Sharpe Ratio (1Y, rf=4.5%) | {_fmt_safe(sharpe, '.2f')} | {'🟢 Excellent' if sharpe > 1.5 else '🟡 Acceptable' if sharpe > 0.5 else '🔴 Poor'} |")
        lines.append(f"| Sortino Ratio (1Y, rf=4.5%) | {_fmt_safe(sortino, '.2f')} | {'🟢 Good' if sortino > 1.0 else '🟡 Acceptable' if sortino > 0.5 else '🔴 Poor'} downside-adjusted |")
        lines.append(f"| Portfolio Beta (Total Weight) | {_fmt_beta(port_beta_total)} | {'🔴 High Risk' if (_has_beta_total and port_beta_total > 1.5) else '🟡 Moderate' if (_has_beta_total and port_beta_total > 1) else '🟢 Defensive' if _has_beta_total else '⚪ N/A (missing beta data)'} |")
        lines.append(f"| Portfolio Beta (Equity Sleeve) | {_fmt_beta(port_beta_equity)} | {'Normalized over non-cash assets' if _has_beta_equity else 'N/A (insufficient beta coverage)'} |")
        lines.append(f"| VaR 95% 1-Day (Historical) | {_fmt_safe(var_95, '.2f')}% | 95% of days, loss ≤ this |")
        lines.append(f"| CVaR 95% (Expected Shortfall) | {_fmt_safe(cvar_95, '.2f')}% | Avg loss **when** VaR is breached — tail risk |")
        lines.append(f"| Max Drawdown (1Y) | {_fmt_safe(max_dd)}% | Worst peak-to-trough |")
        lines.append(f"| Max Drawdown Duration | {max_dd_duration}d | Longest time underwater |")
        lines.append(f"| Portfolio Div Yield (Total) | {port_div_yield_total:.2f}% | Weighted annual income |")
        lines.append(f"| Portfolio Div Yield (Equity Sleeve) | {port_div_yield_equity:.2f}% | Normalized over non-cash assets |")
        if rolling_sharpe_str:
            lines.append(f"| Rolling Sharpe (63d) | — | {rolling_sharpe_str} |")

        # ── Sector Concentration ───────────────────────────────────────────
        lines.append("")
        lines.append("## 🏭 Sector Exposure")
        lines.append("| Sector | Weight | HHI Contribution |")
        lines.append("|--------|--------|-----------------|")
        for sec, w in sorted(sector_concentration.items(), key=lambda x: -x[1]):
            lines.append(f"| {sec} | {w*100:.1f}% | {w**2:.3f} |")
        hhi_label = "🔴 Highly Concentrated" if hhi > 0.35 else "🟡 Moderately Concentrated" if hhi > 0.18 else "🟢 Diversified"
        lines.append(f"\n> **Sector HHI:** {hhi:.3f} — {hhi_label} *(0=perfect diversification, 1=single sector)*")

        # ── Correlation & Diversification Analysis ────────────────────────
        lines.append("")
        lines.append("## 🔗 Correlation & Diversification")
        lines.append(corr_matrix_str)
        lines.append("")
        eff_n_label = "🔴 Low" if eff_n < 1.5 else "🟡 Moderate" if eff_n < 2.5 else "🟢 Good"
        avg_corr_label = "🔴 High" if avg_corr > 0.60 else "🟡 Moderate" if avg_corr > 0.35 else "🟢 Low"
        lines.append(f"| Metric | Value | Interpretation |")
        lines.append(f"|--------|-------|----------------|")
        lines.append(f"| Avg Pairwise Correlation | {avg_corr:.2f} | {avg_corr_label} co-movement between holdings |")
        lines.append(f"| Effective N (eigenvalue) | {eff_n:.1f} | {eff_n_label} — true independent bets in portfolio |")
        if high_corr:
            lines.append(f"\n**⚠️ High Correlation Pairs (>0.70):**")
            for hc in high_corr:
                lines.append(f"- {hc}")
            lines.append("> These pairs move together — holding both adds limited diversification. One may be redundant.")
        else:
            lines.append(f"\n> ⚠️ **Note:** Even without pairs >0.70, Effective N = {eff_n:.1f} indicates portfolio "
                         f"behaves like **{eff_n:.1f} independent assets** — not {len(valid_tickers)}. "
                         "Sector concentration (not just pairwise correlation) is the real risk driver here.")

        # ── Stress Tests (Historical + Simulated) ─────────────────────────
        lines.append("")
        lines.append("## 🧪 Stress Testing")
        lines.append("*Historical crises use actual observed returns + sector-adjusted amplification (NOT linear beta)*")
        lines.append("")
        lines.append("| Scenario | SPX Actual | Portfolio Est. | Notes |")
        lines.append("|----------|------------|----------------|-------|")
        for sl in scenario_lines:
            lines.append(sl)
        lines.append(f"\n> **Tech Weight:** {tech_weight*100:.0f}% of equity. "
                     f"In risk-off events, tech typically amplifies market moves by 1.2–1.8×. "
                     f"{'Linear beta alone (equity β=' + _fmt_beta(port_beta_equity) + ') understates true drawdown risk.' if _has_beta_equity else 'Equity beta unavailable (N/A) — avoid beta-based inference until data coverage improves.'}")

        # ── EisaX Risk Assessment ─────────────────────────────────────────
        lines.append("")
        lines.append("## 💡 EisaX Risk Assessment")
        _top_sector = max(sector_concentration, key=sector_concentration.get) if sector_concentration else "N/A"
        _top_sector_pct = sector_concentration.get(_top_sector, 0) * 100
        if _has_beta_total and port_beta_total > 1.5:
            lines.append(
                f"🔴 **Aggressive** — total β={_fmt_beta(port_beta_total)} (equity β={_fmt_beta(port_beta_equity)}), "
                f"CVaR={cvar_95:.2f}%/day, Eff.N={eff_n:.1f} bets. "
                f"{_top_sector_pct:.0f}% in {_top_sector}. "
                f"{('In a 2022-style selloff your equity sleeve would have lost ~' + str(round(abs(tech_weight * -0.33 + (1-tech_weight) * -0.195 * port_beta_equity)*100)) + '% ') if _has_beta_equity else ''}"
                f"vs SPX -19.5%. Diversification is urgently needed."
            )
        elif _has_beta_total and port_beta_total > 1.0:
            lines.append(
                f"🟡 **Moderate-Aggressive** — total β={_fmt_beta(port_beta_total)} (equity β={_fmt_beta(port_beta_equity)}), CVaR={cvar_95:.2f}%/day. "
                "Above-market sensitivity. Trim highest-beta names on strength; add one defensive sector."
            )
        elif _has_beta_total:
            lines.append(
                f"🟢 **Balanced** — total β={_fmt_beta(port_beta_total)} (equity β={_fmt_beta(port_beta_equity)}), "
                f"CVaR={cvar_95:.2f}%/day, Eff.N={eff_n:.1f}. Reasonable risk profile."
            )
        else:
            lines.append(
                f"⚪ **Risk Classification Limited** — beta data unavailable for enough holdings. CVaR={cvar_95:.2f}%/day, Eff.N={eff_n:.1f}. Add missing beta data before beta-based decisions."
            )
        lines.append("")

        # ══════════════════════════════════════════════════════════════════════
        # ── 1. PERFORMANCE ATTRIBUTION ────────────────────────────────────
        # Shows each holding's contribution to total return (Brinson model)
        # ══════════════════════════════════════════════════════════════════════
        if len(price_data) >= 1:
            lines.append("## 📐 Performance Attribution (1Y, Equity Sleeve)")
            lines.append("*Brinson-Hood-Beebower: each holding's contribution to equity-sleeve return*")
            lines.append("")
            lines.append("| Ticker | Weight (Equity Sleeve) | 1Y Return | Contribution | Attribution |")
            lines.append("|--------|--------|-----------|--------------|-------------|")
            attr_rows = []
            for t in valid_tickers:
                if t in price_data and len(price_data[t]) > 1:
                    t_ret = float((price_data[t].iloc[-1] / price_data[t].iloc[0]) - 1) * 100
                    contrib = t_ret * valid_weights.get(t, 0)
                    attr_rows.append((t, valid_weights.get(t, 0), t_ret, contrib))
            attr_rows.sort(key=lambda x: -x[3])  # sort by contribution descending
            for t, w, ret, contrib in attr_rows:
                bar = "🟢" if contrib > 0 else "🔴"
                lines.append(f"| {t} | {_fmt_safe(w*100)}% | {_fmt_safe(ret, '+.1f')}% | {_fmt_safe(contrib, '+.2f')}pp | {bar} |")
            # Residual from compounding + rounding differences
            explained = sum(c for _, _, _, c in attr_rows)
            residual = port_total_return - explained
            if abs(residual) > 0.05:
                lines.append(f"| Residual (model) | — | — | {_fmt_safe(residual, '+.2f')}pp | {'🔴' if residual < 0 else '⚪'} |")
            lines.append(f"| **TOTAL** | 100% | — | **{_fmt_safe(port_total_return, '+.2f')}pp** | |")
            lines.append("")
            _excluded = [t for t in valid_tickers if t not in price_data or len(price_data.get(t, [])) < 2]
            if _excluded:
                lines.append(f"\n> ⚠️ **Excluded from attribution:** {', '.join(_excluded)} (data unavailable)")
            # Alpha source: top contributor vs S&P
            if attr_rows and spx_return is not None and alpha is not None:
                top_t, top_w, top_ret, top_c = attr_rows[0]
                lines.append(f"> 🏆 **Alpha driver:** {top_t} contributed {_fmt_safe(top_c, '+.1f')}pp (+{_fmt_safe(top_ret)}% × {top_w*100:.0f}% weight). "
                             f"S&P 500 returned {_fmt_safe(spx_return * 100 if spx_return is not None else None, '+.1f')}% — alpha gap of {_fmt_safe(alpha, '+.1f')}pp.")
            lines.append("")

        # ══════════════════════════════════════════════════════════════════════
        # ── 2. FACTOR EXPOSURE (Fama-French Proxy) ───────────────────────────
        # Approximates factor tilts using beta, P/E, market cap, momentum
        # ══════════════════════════════════════════════════════════════════════
        lines.append("## 🧬 Factor Exposure Analysis")
        lines.append("*Fama-French proxy: factor tilts computed from fundamentals + price momentum*")
        lines.append("")

        factor_scores = {}
        factor_weights = {}
        _factor_order = ["Market (Beta)", "Growth (P/E tilt)", "Size (SMB proxy)", "Momentum (12M)"]
        for t in valid_tickers:
            info   = stock_info.get(t, {})
            w      = valid_weights.get(t, 0)
            beta   = info.get("beta")
            pe     = info.get("pe")
            mc     = info.get("mktcap")
            # 1Y momentum from price data
            mom = None
            if t in price_data and len(price_data[t]) > 20:
                _s = price_data[t]
                mom = float((_s.iloc[-1] / _s.iloc[max(0, len(_s)-252)]) - 1)

            # Factor scoring (institutional proxy)
            # Market beta exposure
            if isinstance(beta, (int, float)) and np.isfinite(beta):
                factor_scores["Market (Beta)"] = factor_scores.get("Market (Beta)", 0.0) + (beta * w)
                factor_weights["Market (Beta)"] = factor_weights.get("Market (Beta)", 0.0) + w
            # Growth (inverse P/E proxy — high PE = growth tilt)
            if isinstance(pe, (int, float)) and np.isfinite(pe):
                growth_score = min(max((pe - 15) / 50, -1), 1)  # normalize
                factor_scores["Growth (P/E tilt)"] = factor_scores.get("Growth (P/E tilt)", 0.0) + (growth_score * w)
                factor_weights["Growth (P/E tilt)"] = factor_weights.get("Growth (P/E tilt)", 0.0) + w
            # Size (log market cap — large = negative small-cap factor)
            if isinstance(mc, (int, float)) and np.isfinite(mc) and mc > 0:
                import math as _math
                size_score = -(_math.log10(mc) - 10) / 3  # large cap = negative SMB
                factor_scores["Size (SMB proxy)"] = factor_scores.get("Size (SMB proxy)", 0.0) + (size_score * w)
                factor_weights["Size (SMB proxy)"] = factor_weights.get("Size (SMB proxy)", 0.0) + w
            # Momentum (12-1 month)
            if isinstance(mom, (int, float)) and np.isfinite(mom):
                factor_scores["Momentum (12M)"] = factor_scores.get("Momentum (12M)", 0.0) + (mom * w)
                factor_weights["Momentum (12M)"] = factor_weights.get("Momentum (12M)", 0.0) + w

        lines.append("| Factor | Portfolio Exposure | Interpretation |")
        lines.append("|--------|--------------------|----------------|")
        _factor_labels = {
            "Market (Beta)":       lambda v: ("🔴 High market sensitivity" if v > 1.5 else "🟢 Low market sensitivity" if v < 0.8 else "🟡 Market-like"),
            "Growth (P/E tilt)":   lambda v: ("🔴 Strong growth tilt — high valuation risk" if v > 0.4 else "🟢 Value tilt" if v < -0.1 else "🟡 Blend"),
            "Size (SMB proxy)":    lambda v: ("🟢 Large-cap dominated (low SMB)" if v < -0.1 else "🔴 Small-cap tilt" if v > 0.2 else "🟡 Mid-cap blend"),
            "Momentum (12M)":      lambda v: ("🟢 Strong momentum" if v > 0.20 else "🔴 Negative momentum" if v < -0.10 else "🟡 Neutral momentum"),
        }
        for fname in _factor_order:
            _w_cov = factor_weights.get(fname, 0.0)
            if _w_cov <= 0:
                lines.append(f"| {fname} | **N/A** | ⚪ Insufficient data |")
                continue
            fval = factor_scores.get(fname, 0.0) / _w_cov
            label_fn = _factor_labels.get(fname, lambda v: "—")
            lines.append(f"| {fname} | **{fval:+.2f}** | {label_fn(fval)} |")
        lines.append("")
        lines.append("> *Factor exposures are approximated from fundamentals. "
                     "For precise Fama-French loadings, a 3-year return regression against HML/SMB/MKT factors is required.*")
        lines.append("")

        # ══════════════════════════════════════════════════════════════════════
        # ── 3. INSTITUTIONAL OPTIMIZATION ENGINE ─────────────────────────────
        # Constrained multi-objective: Maximise Sharpe & Minimise CVaR
        # with institutional mandate limits (single stock, sector, beta, eff N)
        # ══════════════════════════════════════════════════════════════════════
        if len(price_data) >= 2:
            lines.append("## ⚙️ Institutional Optimization Engine")
            lines.append("*Convex QP · CLARABEL solver · Zero-tolerance constraint enforcement*")
            lines.append("")
            try:
                import cvxpy as cp
                from scipy.optimize import minimize as _scimin

                _opt_tickers = [t for t in returns_df.columns if t in valid_weights]
                _n = len(_opt_tickers)
                if _n < 2:
                    raise ValueError("Need ≥2 equity holdings for optimization")

                _mu       = np.array([float(returns_df[t].mean() * 252) for t in _opt_tickers])
                _cov_raw  = returns_df[_opt_tickers].cov().values * 252
                _cov      = np.array(_cov_raw, dtype=float) + np.eye(_n) * 1e-8   # PSD regularisation
                _missing_beta = [t for t in _opt_tickers if stock_info.get(t, {}).get("beta") is None]
                if _missing_beta:
                    raise ValueError(f"Missing beta data for optimization: {', '.join(_missing_beta)}")
                _betas_v  = np.array([max(float(stock_info.get(t, {}).get("beta")), 0.01)
                                      for t in _opt_tickers])
                _MIN_W    = 0.01          # 1 % floor per holding
                _PORT_V   = 100_000       # $100k reference for dollar amounts

                # ── Sector matrix ──────────────────────────────────────────
                _tkr_sec   = {t: stock_info.get(t,{}).get("sector","Unknown") for t in _opt_tickers}
                _sec_uniq  = sorted(set(_tkr_sec.values()))
                _S_mat     = np.array([[1 if _tkr_sec[t]==s else 0 for t in _opt_tickers]
                                       for s in _sec_uniq])

                # Minimum achievable portfolio beta (max weight in lowest-beta stock)
                _bidx = np.argsort(_betas_v)
                _min_beta_possible = float(
                    _betas_v[_bidx[0]] * (1 - (_n-1)*_MIN_W)
                    + sum(_betas_v[_bidx[i]] * _MIN_W for i in range(1, _n))
                )

                # ── Helpers ────────────────────────────────────────────────
                def _port_metrics(w: np.ndarray):
                    w = np.clip(w, 0, 1); w /= w.sum()
                    ret  = float(_mu @ w)
                    vol  = float(np.sqrt(w @ _cov @ w))
                    sh   = round((ret - RF_RATE) / vol, 2) if vol > 0 else 0.0
                    beta = float(_betas_v @ w)
                    pr   = returns_df[_opt_tickers].values @ w
                    cvar = float(pr[pr <= np.percentile(pr, 5)].mean() * 100)
                    return ret, vol, sh, beta, cvar   # ret, vol, sharpe, beta, cvar

                # ── SECTION 1: Constraint Diagnostics ─────────────────────
                lines.append("### 🔍 Constraint Diagnostics")
                lines.append(f"*Reference: ${_PORT_V:,} portfolio · normalized weights*")
                lines.append("")
                lines.append("| Constraint | Limit | Current | Status | Required Fix |")
                lines.append("|------------|-------|---------|--------|--------------|")
                _has_breach = False

                for _t in _opt_tickers:
                    _cw = valid_weights.get(_t, 0)
                    if _cw > 0.20:
                        _has_breach = True
                        _exc = _cw - 0.20
                        lines.append(
                            f"| Single stock: **{_t}** | ≤20% | {_cw*100:.1f}% |"
                            f" 🔴 −{_exc*100:.1f}pp | Sell ~${_exc*_PORT_V:,.0f} of {_t} |"
                        )
                for _sec, _sw in sorted(sector_concentration.items(), key=lambda x: -x[1]):
                    if _sw > 0.40:
                        _has_breach = True
                        _exc = _sw - 0.40
                        lines.append(
                            f"| Sector: **{_sec}** | ≤40% | {_sw*100:.1f}% |"
                            f" 🔴 −{_exc*100:.1f}pp | Reduce by ~${_exc*_PORT_V:,.0f} / add non-{_sec} |"
                        )
                if port_beta > 1.30:
                    _has_breach = True
                    _exc = port_beta - 1.30
                    lines.append(
                        f"| Portfolio beta | ≤1.30 | {port_beta:.2f} |"
                        f" 🔴 +{_exc:.2f} | Add low-β assets: TLT, GLD, BRK-B, XLV |"
                    )
                if not _has_breach:
                    lines.append("| All constraints | — | — | ✅ No breaches | — |")

                _overweight_total = sum(max(0, valid_weights.get(_t,0) - 0.20) for _t in _opt_tickers)
                if _overweight_total > 0.005:
                    lines.append("")
                    lines.append(
                        f"> 💸 **Estimated rebalancing:** ~${_overweight_total*_PORT_V:,.0f} turnover "
                        f"({_overweight_total*100:.1f}% of portfolio) to reach basic mandate compliance"
                    )
                lines.append("")

                # ── Feasibility check ──────────────────────────────────────
                def _check_feas(max_s: float, max_b: float, max_sec: float):
                    """Returns (single_ok, beta_ok, sector_ok, issues[], suggestions[])"""
                    iss, sug = [], []
                    # 1. Single-stock: N * max_s must cover 100%
                    s_ok = (_n * max_s >= 1.0 - 0.005)
                    if not s_ok:
                        need = int(np.ceil(1.0 / max_s))
                        iss.append(
                            f"{_n} holdings × {max_s*100:.0f}% = {_n*max_s*100:.0f}% < 100% "
                            f"(need ≥{need} holdings for this mandate)"
                        )
                        sug.append(f"Add {need - _n}+ new assets OR use Aggressive mandate ({_n}×20%→80% min)")
                    # 2. Beta: minimum achievable ≤ limit
                    b_ok = (_min_beta_possible <= max_b + 0.005)
                    if not b_ok:
                        iss.append(
                            f"Min achievable β = {_min_beta_possible:.2f} exceeds cap β≤{max_b:.2f}"
                        )
                        sug.append("Add: TLT (β≈−0.5), GLD (β≈0.1), USMV (β≈0.7), BRK-B (β≈0.9)")
                    # 3. Sector: sum of per-sector max allocations must ≥ 100%
                    _sec_cap_sum = sum(
                        min(max_sec, sum(1 for _t in _opt_tickers if _tkr_sec[_t]==s) * max_s)
                        for s in _sec_uniq
                    )
                    sec_ok = (_sec_cap_sum >= 1.0 - 0.005)
                    if not sec_ok:
                        _dom = max(_sec_uniq, key=lambda s: sum(1 for _t in _opt_tickers if _tkr_sec[_t]==s))
                        _nd  = sum(1 for _t in _opt_tickers if _tkr_sec[_t]==_dom)
                        iss.append(
                            f"Sector '{_dom}': {_nd}/{_n} holdings → "
                            f"max total = {_sec_cap_sum*100:.0f}% < 100% with current assets"
                        )
                        sug.append("Add diversifying sectors: XLV (Healthcare), TLT (Bonds), GLD (Gold), BRK-B, XLP")
                    return s_ok, b_ok, sec_ok, iss, sug

                # ── QP solver: Max-Sharpe (Lasserre lifting) ──────────────
                def _qp_solve(max_s, max_b, max_sec,
                              s_ok, b_ok, sec_ok,
                              strict: bool = True):
                    """
                    Convex QP — two modes:

                    strict=True (default / Institutional Mode):
                        ALL feasible flags must be True. If any constraint cannot be
                        enforced → return (None, "❌ INFEASIBLE — [reason]").
                        Never silently relaxes a constraint. Never returns a
                        partial/relaxed solution labelled as compliant.

                    strict=False (Exploratory Mode — explicit opt-in):
                        Solves with whatever constraints ARE geometrically possible.
                        Result is ALWAYS labelled "🔍 EXPLORATORY MODE" so the user
                        knows constraints were adjusted. Never presented as compliant.
                    """
                    # ── Strict mode: hard gate ─────────────────────────────
                    if strict:
                        infeas_reasons = []
                        if not s_ok:
                            need = int(np.ceil(1.0 / max_s))
                            infeas_reasons.append(
                                f"single-stock cap {max_s*100:.0f}% requires ≥{need} holdings "
                                f"(current: {_n})"
                            )
                        if not b_ok:
                            infeas_reasons.append(
                                f"beta cap β≤{max_b:.2f} unachievable "
                                f"(min possible β={_min_beta_possible:.2f})"
                            )
                        if not sec_ok:
                            _dom = max(_sec_uniq,
                                       key=lambda s: sum(1 for _t in _opt_tickers
                                                         if _tkr_sec[_t] == s))
                            _nd  = sum(1 for _t in _opt_tickers if _tkr_sec[_t] == _dom)
                            infeas_reasons.append(
                                f"sector cap {max_sec*100:.0f}% unachievable: "
                                f"'{_dom}' has {_nd}/{_n} holdings"
                            )
                        if infeas_reasons:
                            return None, "❌ INFEASIBLE — " + "; ".join(infeas_reasons)

                    # ── Build constraint list ──────────────────────────────
                    try:
                        y = cp.Variable(_n, nonneg=True)
                        k = cp.Variable(nonneg=True)
                        constr = [
                            cp.quad_form(y, _cov) <= 1,
                            cp.sum(y) == k,
                            k >= 0,
                            y >= _MIN_W * k,
                        ]
                        _relaxed_tags: list[str] = []
                        # _active_s_cap: the cap actually enforced (used for post-solve clipping too)
                        if s_ok:
                            _active_s_cap = max_s
                            constr.append(y <= _active_s_cap * k)
                        else:
                            # Relaxed mode: use 1/n + 5% buffer (always geometrically feasible)
                            _active_s_cap = round(1.0 / _n + 0.05, 3)
                            constr.append(y <= _active_s_cap * k)
                            _relaxed_tags.append(
                                f"single-stock cap adjusted to {_active_s_cap*100:.0f}% "
                                f"(mandate {max_s*100:.0f}% infeasible with {_n} holdings)"
                            )
                        if b_ok:
                            constr.append(_betas_v @ y <= max_b * k)
                        else:
                            _relaxed_tags.append(f"beta cap lifted (was β≤{max_b:.2f}, min achievable β={_min_beta_possible:.2f})")
                        if sec_ok:
                            constr.append(_S_mat @ y <= max_sec * k)
                        else:
                            _relaxed_tags.append(f"sector cap lifted (was {max_sec*100:.0f}%, current holdings prevent compliance)")

                        prob = cp.Problem(cp.Maximize((_mu - RF_RATE) @ y), constr)

                        # Primary solver: CLARABEL (strict interior-point)
                        prob.solve(solver=cp.CLARABEL, verbose=False)

                        if prob.status == "optimal" and k.value is not None and float(k.value) > 1e-8:
                            raw   = np.maximum(np.array(y.value, dtype=float) / float(k.value), 0.0)
                            w_out = raw / raw.sum()
                            if s_ok:
                                w_out = np.clip(w_out, _MIN_W, max_s)
                                w_out /= w_out.sum()

                            # Hard post-solve verification (catch solver float drift)
                            if s_ok and float(np.max(w_out)) > max_s + 0.001:
                                return None, (f"❌ INFEASIBLE — solver returned "
                                              f"max={np.max(w_out)*100:.1f}% > {max_s*100:.0f}% cap "
                                              f"(numerical instability)")
                            if b_ok and float(_betas_v @ w_out) > max_b + 0.005:
                                return None, (f"❌ INFEASIBLE — solver beta "
                                              f"β={float(_betas_v@w_out):.3f} > cap β≤{max_b}")

                            if not strict and _relaxed_tags:
                                status = ("🔍 EXPLORATORY MODE — constraints adjusted: "
                                          + "; ".join(_relaxed_tags))
                            else:
                                status = "✅ Fully compliant"
                            return w_out, status

                        # SCS fallback (handles near-degenerate edge cases)
                        prob.solve(solver=cp.SCS, eps=1e-6, verbose=False)
                        if prob.status in ("optimal", "optimal_inaccurate") \
                                and k.value is not None and float(k.value) > 1e-8:
                            raw   = np.maximum(np.array(y.value, dtype=float) / float(k.value), 0.0)
                            w_out = raw / raw.sum()
                            if s_ok:
                                w_out = np.clip(w_out, _MIN_W, max_s)
                                w_out /= w_out.sum()
                            if not strict and _relaxed_tags:
                                status = ("🔍 EXPLORATORY MODE (approx) — "
                                          + "; ".join(_relaxed_tags))
                            else:
                                status = "✅ Fully compliant (approx)"
                            return w_out, status

                        return None, f"❌ INFEASIBLE — solver status: {prob.status}"
                    except Exception as _se:
                        return None, f"❌ Solver error: {_se}"

                # ── SECTION 2: Mandate Mode Feasibility Table ──────────────
                _MANDATES = [
                    # (display_name, max_single, max_beta, max_sector, profile)
                    ("🏛️ Conservative", 0.10, 1.00, 0.30, "Pension / Endowment"),
                    ("⚖️ Balanced",      0.15, 1.20, 0.35, "Family Office / Multi-asset"),
                    ("🚀 Aggressive",    0.20, 1.50, 0.40, "Institutional Growth"),
                ]

                _feas_data = []   # (nm, ms, mb, msec, prof, s_ok, b_ok, sec_ok, iss, sug)
                for _nm, _ms, _mb, _msec, _prof in _MANDATES:
                    _so, _bo, _seco, _iss, _sug = _check_feas(_ms, _mb, _msec)
                    _feas_data.append((_nm, _ms, _mb, _msec, _prof, _so, _bo, _seco, _iss, _sug))

                lines.append("### 📊 Mandate Mode Feasibility")
                lines.append("")
                lines.append("| | 🏛️ Conservative | ⚖️ Balanced | 🚀 Aggressive |")
                lines.append("|---|:---:|:---:|:---:|")
                lines.append("| **Max single stock** | 10% | 15% | 20% |")
                lines.append("| **Max portfolio β** | ≤1.00 | ≤1.20 | ≤1.50 |")
                lines.append("| **Max sector weight** | 30% | 35% | 40% |")
                lines.append("| Single-stock achievable | " + " | ".join(
                    "✅" if r[5] else f"❌ Need {int(np.ceil(1/r[1]))}+ holdings"
                    for r in _feas_data) + " |")
                lines.append("| Beta achievable | " + " | ".join(
                    "✅" if r[6] else f"❌ Min β={_min_beta_possible:.2f}"
                    for r in _feas_data) + " |")
                lines.append("| Sector cap achievable | " + " | ".join(
                    "✅" if r[7] else "⚠️ Current assets only"
                    for r in _feas_data) + " |")
                lines.append("")

                # Collect deduplicated issues / suggestions (keyed by constraint type)
                _iss_by_type: dict[str, str] = {}  # key=type, val=worst message
                _sug_set: set = set()
                for _fd in _feas_data:
                    _so2,_bo2,_seco2,_iss2,_sug2 = _fd[5],_fd[6],_fd[7],_fd[8],_fd[9]
                    for _i2 in _iss2:
                        # Bucket by type: single/beta/sector
                        _ikey = ("single" if "holdings" in _i2 else
                                 "beta"   if "β" in _i2 else "sector")
                        # Keep the most severe (Conservative = strictest = hardest to satisfy)
                        if _ikey not in _iss_by_type:
                            _iss_by_type[_ikey] = _i2
                    _sug_set.update(_sug2)

                if _iss_by_type:
                    lines.append("> **⚠️ Constraint violations — current holdings cannot satisfy any mandate:**")
                    for _ikey in ["single","beta","sector"]:
                        if _ikey in _iss_by_type:
                            lines.append(f"> - {_iss_by_type[_ikey]}")
                    lines.append(">")
                    lines.append("> **💡 To unlock full institutional compliance, add any of:**")
                    for _s in sorted(_sug_set):
                        lines.append(f"> - {_s}")
                    lines.append("")

                # ── Run optimizers (STRICT MODE) for each mandate ─────────
                # strict=True: if any constraint infeasible → ❌, never silently relax
                _m_results = []   # (nm, ms, mb, msec, prof, w_opt, status, iss, sug)
                for _nm, _ms, _mb, _msec, _prof, _so, _bo, _seco, _iss, _sug in _feas_data:
                    _wo, _stat = _qp_solve(_ms, _mb, _msec, _so, _bo, _seco, strict=True)
                    _m_results.append((_nm, _ms, _mb, _msec, _prof, _wo, _stat, _iss, _sug))

                # ── SECTION 3: Optimal Weight Comparison ──────────────────
                lines.append("### 📊 Optimal Weight Comparison — Strict Institutional Mode")
                lines.append("")
                lines.append("> 🔒 **Strict Mode:** Only fully-compliant solutions shown. "
                             "`❌ INFEASIBLE` = mandate cannot be achieved with current holdings — "
                             "see asset suggestions above.")
                lines.append("")
                _col_hdrs = " | ".join(r[0] for r in _m_results)
                lines.append(f"| Ticker | Current | {_col_hdrs} |")
                lines.append("|--------|---------|" + "|".join(["------"]*len(_m_results)) + "|")
                for _i, _t in enumerate(_opt_tickers):
                    _cw = valid_weights.get(_t, 0)
                    _row = f"| **{_t}** | {_cw*100:.1f}% |"
                    for *_, _wo, _stat, _iss2, _sug2 in _m_results:
                        _row += f" {_wo[_i]*100:.1f}% |" if _wo is not None else " ❌ N/A |"
                    lines.append(_row)

                # Metrics rows
                lines.append("|--------|---------|" + "|".join(["------"]*len(_m_results)) + "|")
                _m_metrics = [_port_metrics(r[5]) if r[5] is not None else None for r in _m_results]

                def _mfmt(m_list, idx, fmtfn):
                    return " | ".join(fmtfn(m[idx]) if m is not None else "❌" for m in m_list)

                lines.append(f"| **Sharpe** | {sharpe:.2f} | {_mfmt(_m_metrics,2,lambda x:f'{x:.2f}')} |")
                lines.append(f"| **Beta** | {port_beta:.2f} | {_mfmt(_m_metrics,3,lambda x:f'{x:.2f}')} |")
                lines.append(f"| **CVaR/day** | {cvar_95:.2f}% | {_mfmt(_m_metrics,4,lambda x:f'{x:.2f}%')} |")
                lines.append(f"| **Ann. Vol** | {ann_vol*100:.1f}% | {_mfmt(_m_metrics,1,lambda x:f'{x*100:.1f}%')} |")
                lines.append("| **Compliance** | Baseline | " + " | ".join(
                    "✅ Compliant" if r[5] is not None else "❌ INFEASIBLE"
                    for r in _m_results) + " |")
                lines.append("")

                # ── Exploratory Allocation (only if at least one mandate is infeasible) ──
                _any_infeasible = any(r[5] is None for r in _m_results)
                if _any_infeasible:
                    # Find the most lenient feasibility set (Aggressive mandate)
                    _agg = _feas_data[-1]  # Aggressive = last
                    _rnm, _rms, _rmb, _rmsec, _rprof, _rso, _rbo, _rseco, _riss, _rsug = _agg
                    _rwo, _rstat = _qp_solve(_rms, _rmb, _rmsec, _rso, _rbo, _rseco, strict=False)

                    lines.append("### 🔍 Exploratory Allocation — Pre-Asset Expansion")
                    lines.append("")
                    lines.append("> 🔍 **Exploratory Allocation (Non-Mandate Scenario)** — "
                                 "Shows the best achievable structure with current holdings "
                                 "before adding diversifying assets. "
                                 "Constraints adjusted where mathematically required. "
                                 "This is a planning scenario, not a mandate-compliant allocation.")
                    lines.append("")
                    if _rwo is not None:
                        lines.append("| Ticker | Current | 🔍 Exploratory |")
                        lines.append("|--------|---------|----------------|")
                        for _i, _t in enumerate(_opt_tickers):
                            _cw = valid_weights.get(_t, 0)
                            lines.append(f"| **{_t}** | {_cw*100:.1f}% | {_rwo[_i]*100:.1f}% |")
                        lines.append("|--------|---------|----------------|")
                        _rr, _rv, _rsh, _rb2, _rcv = _port_metrics(_rwo)
                        lines.append(f"| **Sharpe** | {sharpe:.2f} | {_rsh:.2f} |")
                        lines.append(f"| **Beta** | {port_beta:.2f} | {_rb2:.2f} |")
                        lines.append(f"| **CVaR/day** | {cvar_95:.2f}% | {_rcv:.2f}% |")
                        lines.append(f"| **Compliance** | ❌ Breach | 🔍 Exploratory — Pre-Expansion |")
                        lines.append("")
                        lines.append(f"> 🔍 *{_rstat}*")
                    else:
                        lines.append("> ❌ No exploratory solution found. "
                                     "The current portfolio composition prevents optimization. "
                                     "Add diversifying assets before running again.")
                    lines.append("")

                # ── SECTION 4: Execution Plan (strictly compliant mandates only) ──
                _best = next(
                    (r for r in _m_results if r[5] is not None and "✅" in r[6]),
                    None   # None = no compliant mandate found
                )
                # If no strictly compliant mandate found, use relaxed (with clear warning)
                if _best is None and _any_infeasible and '_rwo' in dir() and _rwo is not None:
                    _best = ("🔍 Exploratory Allocation (Non-Mandate Scenario)", _rms, _rmb, _rmsec,
                             "Pre-asset expansion — exploratory scenario", _rwo, _rstat, [], [])
                if _best is not None:
                    _bnm, _bms, _bmb, _bmsec, _bprof, _bw, _bstat, _biss, _bsug = _best
                    _br, _bv, _bsh, _bb, _bcv = _port_metrics(_bw)
                    _is_relaxed_plan = any(x in str(_bstat) for x in ("RELAXED", "EXPLORATORY")) or \
                                       any(x in str(_bnm) for x in ("🔧", "🔍"))

                    lines.append(f"### 📋 Execution Plan → *{_bnm}*")
                    lines.append(f"*{_bprof}*")
                    if _is_relaxed_plan:
                        lines.append("")
                        lines.append("> 🔍 **Exploratory Allocation** — These trades show the optimal "
                                     "structure achievable with current holdings before adding new assets. "
                                     "This is a planning scenario to guide asset expansion decisions — "
                                     "**not a mandate-compliant allocation.**")
                    lines.append("")
                    lines.append("| Action | Ticker | From | To | Δ | $ on $100k | Reason |")
                    lines.append("|--------|--------|------|----|---|------------|--------|")
                    for _i, _t in enumerate(_opt_tickers):
                        _cw = valid_weights.get(_t, 0)
                        _gw = _bw[_i]
                        _d  = _gw - _cw
                        _usd = abs(_d) * _PORT_V
                        if abs(_d) < 0.005:
                            _act, _ico = "HOLD  ", "⚪"
                        elif _d > 0:
                            _act, _ico = "BUY  ↑", "🟢"
                        else:
                            _act, _ico = "SELL ↓", "🔴"
                        _reason = (
                            f"Exceeds {_bms*100:.0f}% mandate cap"       if _cw > _bms + 0.005 and _d < 0 else
                            f"High β={_betas_v[_i]:.2f} — reduce beta"   if _betas_v[_i] > _bmb and _d < 0 else
                            "Underweight vs risk-optimal"                  if _d >  0.02 else
                            "Overweight vs risk-optimal"                   if _d < -0.02 else
                            "Mandate rebalance"
                        )
                        lines.append(
                            f"| {_ico} **{_act}** | {_t} | {_cw*100:.1f}% | {_gw*100:.1f}% |"
                            f" {_d*100:+.1f}pp | ${_usd:,.0f} | {_reason} |"
                        )
                    lines.append("")

                    # ── SECTION 5: Multi-Objective Trade-off ──────────────
                    lines.append("### 📐 Multi-Objective Trade-off Analysis")
                    lines.append("")
                    lines.append("| Strategy | Sharpe | CVaR/day | Beta | Ann.Vol | Status |")
                    lines.append("|----------|--------|----------|------|---------|--------|")
                    lines.append(
                        f"| **Current portfolio** | {sharpe:.2f} | {cvar_95:.2f}% |"
                        f" {port_beta:.2f} | {ann_vol*100:.1f}% | Baseline |"
                    )
                    for _nm2, _ms2, _mb2, _msec2, _prof2, _wo2, _stat2, _, _ in _m_results:
                        if _wo2 is None:
                            lines.append(f"| {_nm2} | — | — | — | — | {_stat2} |")
                            continue
                        _mr, _mv, _msh, _mb3, _mcv = _port_metrics(_wo2)
                        lines.append(
                            f"| {_nm2} | {_msh:.2f} ({_msh-sharpe:+.2f}) |"
                            f" {_mcv:.2f}% ({_mcv-cvar_95:+.2f}pp) |"
                            f" {_mb3:.2f} ({_mb3-port_beta:+.2f}) |"
                            f" {_mv*100:.1f}% | {_stat2} |"
                        )
                    lines.append("")

                    # ── SECTION 6: Post-Rebalance Simulation ──────────────
                    lines.append("### 🔮 Post-Rebalance Simulation")
                    lines.append(f"*Best mandate: {_bnm} · {_bprof}*")
                    lines.append("")
                    lines.append("| Metric | Before | After | Δ | Verdict |")
                    lines.append("|--------|--------|-------|---|---------|")
                    _curr_breach  = any(valid_weights.get(_t,0) > 0.20 for _t in _opt_tickers)
                    _after_breach = any(_bw[_i] > _bms + 0.002 for _i in range(_n))
                    _sim_rows = [
                        ("Sharpe Ratio",    f"{sharpe:.2f}",        f"{_bsh:.2f}",
                            _bsh-sharpe,      lambda d: "✅ Improved" if d > 0.05 else "🟡 Similar"),
                        ("Ann. Volatility", f"{ann_vol*100:.1f}%",  f"{_bv*100:.1f}%",
                            (_bv-ann_vol)*100, lambda d: "✅ Lower" if d < -1 else "🟡 Similar" if d < 2 else "🔴 Higher"),
                        ("Portfolio Beta",  f"{port_beta:.2f}",     f"{_bb:.2f}",
                            _bb-port_beta,     lambda d: "✅ Reduced" if d < -0.05 else "⚠️ Similar"),
                        ("CVaR 95%/day",    f"{cvar_95:.2f}%",      f"{_bcv:.2f}%",
                            _bcv-cvar_95,      lambda d: "✅ Safer tail" if d > 0.1 else "🟡 Similar"),
                        ("Single-stock cap",
                            "🔴 Breach" if _curr_breach else "✅ OK",
                            ("🔍 Exploratory" if _is_relaxed_plan else
                             "✅ Compliant" if not _after_breach else "🔴 Still Breach"),
                            0, lambda d: ("🔍 Exploratory scenario" if _is_relaxed_plan
                                          else "✅ Resolved" if not _after_breach else "🔴")),
                        ("Beta mandate",
                            "🔴 Breach" if port_beta > _bmb else "✅ OK",
                            ("🔍 Exploratory" if _is_relaxed_plan and _bb > _bmb + 0.01
                             else "✅ OK" if _bb <= _bmb + 0.01 else "⚠️ Near limit"),
                            0, lambda d: "🔍 Exploratory" if _is_relaxed_plan else "✅"),
                    ]
                    for _lbl, _bef, _aft, _dlt, _vfn in _sim_rows:
                        _dstr = f"{_dlt:+.2f}" if isinstance(_dlt, float) and abs(_dlt) > 0.005 else "—"
                        lines.append(f"| {_lbl} | {_bef} | {_aft} | {_dstr} | {_vfn(_dlt)} |")
                    lines.append("")

            except Exception as _opt_e:
                logger.warning("Optimization engine failed: %s", _opt_e, exc_info=True)
                lines.append(f"*Optimization engine error: {_opt_e}*")
                lines.append("")

        # ── DeepSeek CIO Analysis ──────────────────────────────────────────
        try:
            import requests as _rq, os as _os
            deepseek_key = _os.environ.get("DEEPSEEK_API_KEY", "")

            holdings_summary = ", ".join(
                [f"{t} ({portfolio.get(t,0)*100:.0f}% total / {valid_weights.get(t,0)*100:.0f}% equity)" for t in valid_tickers]
            )
            corr_note = "High correlations: " + ", ".join(high_corr) if high_corr else "No high-correlation pairs."
            benchmark_note = (f"S&P 500 returned {spx_return*100:+.1f}% over the same period (Alpha: {alpha:+.1f}%)"
                              if spx_return is not None else "Benchmark data unavailable.")
            div_note = (
                f"Portfolio dividend yield (total): {port_div_yield_total:.2f}% | "
                f"equity sleeve: {port_div_yield_equity:.2f}%"
            )
            scenario_summary = "; ".join([
                f"{lbl}: equity sleeve {(mkt*port_beta_equity*100):+.1f}%"
                for lbl, mkt, _, _ in _CRISIS_SCENARIOS
            ]) if _has_beta_equity else "N/A (beta data unavailable)"

            _sector_summary = "; ".join([f"{s}: {w*100:.0f}%" for s, w in sorted(sector_concentration.items(), key=lambda x: -x[1])])
            _stock_level_lines = []
            for t in valid_tickers:
                _si = stock_info.get(t, {})
                _b = _si.get("beta")
                _p = _si.get("pe")
                _d = _si.get("div_yield")
                _beta_txt = _fmt_beta(_b)
                _pe_txt = f"{_p:.1f}" if isinstance(_p, (int, float)) and np.isfinite(_p) else "N/A"
                _div_txt = f"{_d*100:.2f}%" if isinstance(_d, (int, float)) and np.isfinite(_d) else "N/A"
                _stock_level_lines.append(f"- {t}: {_si.get('sector','N/A')} | β={_beta_txt} | P/E={_pe_txt} | Div={_div_txt}")
            ds_prompt = f"""You are a CIO-level portfolio risk analyst at an institutional fund. Provide a rigorous, data-driven analysis.

PORTFOLIO: {holdings_summary}
Cash: {cash_pct:.1f}%

QUANTITATIVE METRICS (1Y, rf=4.5%, Historical Simulation):
- Equity Sleeve Return: {port_total_return:+.1f}%
- Estimated Total Portfolio Return (incl. cash): {port_total_return_total_est:+.1f}% | {benchmark_note}
- Annualized Volatility: {ann_vol*100:.1f}% | Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}
- Portfolio Beta (total): {_fmt_beta(port_beta_total)} | Portfolio Beta (equity): {_fmt_beta(port_beta_equity)}
- VaR (95%): {var_95:.2f}%/day | CVaR (95%): {cvar_95:.2f}%/day
- Max Drawdown: {max_dd:.1f}% (duration: {max_dd_duration} trading days)
- {rolling_sharpe_str if rolling_sharpe_str else "Rolling Sharpe: insufficient data"}
- {div_note}

DIVERSIFICATION:
- Sector HHI: {hhi:.3f} ({_sector_summary})
- Effective N (eigenvalue): {eff_n:.1f} true independent bets out of {len(valid_tickers)} holdings
- Avg pairwise correlation: {avg_corr:.2f}
- {corr_note}

STOCK-LEVEL DETAIL:
{chr(10).join(_stock_level_lines)}

HISTORICAL STRESS TESTS (sector-adjusted, NOT linear beta):
{chr(10).join([f"- {lbl}: SPX {spx*100:+.1f}% → equity sleeve {(tech_weight*spx*tm + (1-tech_weight)*spx*port_beta_equity)*100:+.1f}%" for lbl, spx, tm, _ in _CRISIS_SCENARIOS]) if _has_beta_equity else "- N/A (insufficient beta coverage)"}

Write a comprehensive institutional analysis with EXACTLY these 6 sections:
1. **📋 Executive Assessment** — Alpha quality, risk-adjusted performance, Sharpe trend (improving/declining?)
2. **🚨 Top 3 Risks** — Each with severity (Critical/High/Medium), quantified worst-case loss, and specific trigger
3. **🔄 Rebalancing Plan** — Exact target weights (must sum to 100%), rationale using CVaR and Eff.N metrics
4. **➕ Suggested Additions** — 3 specific tickers with expected Sharpe/Beta/correlation impact on portfolio
5. **📊 Tax & Income Note** — Capital gains exposure on highest-return holdings + income gap analysis
6. **✅ EisaX Final Verdict** — Rating: Conservative/Balanced/Aggressive/Speculative + one advisory suggested action
   - Include explicit lines: Confidence (%), Primary Uncertainty (2 drivers), No-Action Case, Alternative Scenario

CRITICAL:
- Do not invent or re-derive numeric values. Use the metrics exactly as provided above.
- If any metric is unavailable, state "N/A" explicitly. Do not backfill with assumptions.
- Keep weights consistent with the provided basis labels (total vs equity sleeve).
- Reference CVaR, Effective N, and rolling Sharpe trend in your analysis.
- Use advisory language only ("consider", "prefer", "may"). Avoid command language ("execute immediately", "must sell", "exit 100%").
- If risk profile is elevated, do NOT use phrases like "well-positioned" in the executive opening.
- Include this exact boundary line at the end of section 6:
  "Decision Boundary: This analysis provides strategic guidance, not execution instructions."
- Be institutional — no platitudes. Max 550 words."""

            if deepseek_key:
                ds_resp = _rq.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"},
                    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": ds_prompt}],
                          "max_tokens": 900, "temperature": 0.25},
                    timeout=40
                )
                if ds_resp.status_code == 200:
                    cio_analysis = (ds_resp.json().get("choices", [{}])[0]
                                    .get("message", {}).get("content", ""))
                    if cio_analysis:
                        cio_analysis = _soften_portfolio_advice(cio_analysis)
                        # Keep one confidence source in the final report (Decision Discipline Layer only).
                        _cio_lines = []
                        for _ln in cio_analysis.splitlines():
                            if _re.search(r'(?i)\bconfidence\b', _ln):
                                continue
                            _cio_lines.append(_ln)
                        cio_analysis = "\n".join(_cio_lines)
                        cio_analysis = _re.sub(r'\n{3,}', '\n\n', cio_analysis).strip()
                        if "Decision Boundary:" not in cio_analysis:
                            cio_analysis += (
                                "\n\n> Decision Boundary: This analysis provides strategic guidance, not execution instructions."
                            )
                        lines.append("## 🧠 CIO Deep Analysis (AI-Powered)")
                        lines.append(cio_analysis)
                else:
                    lines.append(f"*CIO analysis unavailable (API {ds_resp.status_code})*")
            else:
                lines.append("## 🧠 Portfolio Assessment")
                lines.append("**Risk Level: Aggressive** — High-beta tech concentration. "
                              "Reduce TSLA, add VIG/XLV/JPM for balance." if (_has_beta_total and port_beta_total > 1.5) else
                              "**Risk Level: Moderate** — Monitor correlation clusters and rebalance quarterly.")
        except Exception as _e:
            logger.warning("DeepSeek CIO analysis failed: %s", _e)
            lines.append("## 🧠 Portfolio Assessment")
            _risk_label = "Aggressive" if (_has_beta_total and port_beta_total > 1.5) else "Moderate-Aggressive" if (_has_beta_total and port_beta_total > 1.2) else "Moderate" if _has_beta_total else "Unknown"
            _top_holding = max(valid_weights, key=valid_weights.get) if valid_weights else "N/A"
            _top_w = valid_weights.get(_top_holding, 0) * 100
            lines.append(
                f"**Risk Profile: {_risk_label}** — total β={_fmt_beta(port_beta_total)} (equity β={_fmt_beta(port_beta_equity)}), CVaR={cvar_95:.2f}%/day, "
                f"Sharpe={sharpe:.2f}, Effective N={eff_n:.1f} independent bets. "
                f"Portfolio is {int(tech_weight*100)}% Technology with {_top_holding} as lead position ({_top_w:.0f}%). "
                + ("Concentration risk is the primary concern — reduce single-sector exposure and add uncorrelated assets (TLT, GLD, BRK-B) to improve Effective N toward ≥4."
                   if tech_weight > 0.6 else
                   "Risk profile is within institutional bounds. Monitor rolling Sharpe for momentum deterioration.")
            )

        decision_conf = _compute_portfolio_confidence(
            sharpe=sharpe,
            eff_n=eff_n,
            avg_corr=avg_corr,
            beta_total=port_beta_total,
            cvar_95=cvar_95,
            rolling_sharpe_now=rolling_sharpe_now,
        )
        lines.append(
            _build_portfolio_decision_layer(
                confidence=decision_conf,
                risk_label=("Aggressive" if (_has_beta_total and port_beta_total > 1.5) else "Moderate-Aggressive" if (_has_beta_total and port_beta_total > 1.2) else "Moderate" if _has_beta_total else "Unknown"),
                rolling_sharpe_now=rolling_sharpe_now,
                tech_weight=tech_weight,
                next_review_hint="next quarterly review",
            )
        )

        lines.append("")
        lines.append("*To analyze any individual stock: type `analyze TICKER`*")

        # ── Audit Trail ────────────────────────────────────────────────────
        import uuid as _uuid_mod, hashlib as _hl, datetime as _dt_mod
        _snap_id   = str(_uuid_mod.uuid4())
        _generated = _dt_mod.datetime.now(_dt_mod.timezone.utc)
        _gen_str   = _generated.strftime("%Y-%m-%d %H:%M:%S UTC")
        _price_asof = prices_df.index[-1].strftime("%Y-%m-%d") if len(price_data) >= 2 else "N/A"
        _period_start = prices_df.index[0].strftime("%Y-%m-%d") if len(price_data) >= 2 else "N/A"
        _tickers_str = ", ".join(_opt_tickers if 'price_data' in dir() and len(price_data) >= 2 else valid_tickers)

        # Build preliminary report for hashing (before audit section appended)
        _pre_report = "\n".join(lines)
        _report_hash = _hl.sha256(_pre_report.encode()).hexdigest()[:16]

        lines.append("")
        lines.append("---")
        lines.append("## 📋 Audit Trail")
        lines.append("*For compliance, reproducibility, and institutional trust*")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| **Report ID** | `{_snap_id}` |")
        lines.append(f"| **Generated** | {_gen_str} |")
        lines.append(f"| **Price Data As-Of** | {_price_asof} |")
        lines.append(f"| **Period Analysed** | {_period_start} → {_price_asof} (252 trading days) |")
        lines.append(f"| **Data Source** | Yahoo Finance (yfinance) — 15-min delayed |")
        lines.append(f"| **Tickers Fetched** | {_tickers_str} |")
        lines.append(f"| **Risk-Free Rate** | 4.50% (US 3-Month T-Bill) |")
        lines.append(f"| **Methodology** | Historical Simulation · Brinson Attribution · Fama-French proxy |")
        lines.append(f"| **Optimizer** | CLARABEL convex QP (cvxpy) — strict constraint enforcement |")
        lines.append(f"| **Report Hash** | `sha256:{_report_hash}...` |")
        lines.append("")
        lines.append(f"> 🔁 **Reproduce this report:** Save snapshot ID `{_snap_id}` → call `GET /v1/portfolio/snapshot/{_snap_id}`")
        lines.append("")

        report = "\n".join(lines)

        # ── Save to Portfolio Memory ───────────────────────────────────────
        try:
            from portfolio_memory import save_snapshot as _save_snap
            _metrics_for_mem = {
                "sharpe":       sharpe,
                "beta":         _safe_round(port_beta_total, 3),
                "beta_equity":  _safe_round(port_beta_equity, 3),
                "cvar_95":      round(cvar_95, 3),
                "ann_vol":      round(ann_vol * 100, 2),
                "total_return": round(port_total_return, 2),
                "total_return_total_est": round(port_total_return_total_est, 2),
                "sortino":      sortino,
                "max_dd":       round(max_dd, 2),
                "div_yield":    round(port_div_yield_total, 3),
                "div_yield_equity": round(port_div_yield_equity, 3),
            }
            _sources_for_mem = [{
                "source":       "Yahoo Finance (yfinance)",
                "tickers":      valid_tickers,
                "period":       f"{_period_start} → {_price_asof}",
                "fetched_at":   _gen_str,
                "price_as_of":  _price_asof,
                "risk_free":    "4.50% US T-Bill",
            }]
            _saved_snap_id = _save_snap(
                user_id=user_id,
                holdings={t: round(valid_weights.get(t, 0), 6) for t in valid_tickers},
                metrics=_metrics_for_mem,
                data_sources=_sources_for_mem,
                report_md=report,
            )
            logger.info("[PortfolioMemory] Snapshot saved: %s (user: %s)", _saved_snap_id, user_id)
        except Exception as _mem_err:
            logger.warning("[PortfolioMemory] Save failed: %s", _mem_err)

        return _clean_nan({
            "status":      "success",
            "snapshot_id": _snap_id,
            "portfolio":   portfolio,
            "tickers":     tickers,
            "analysis":    report,
            "audit": {
                "report_id":     _snap_id,
                "generated_utc": _gen_str,
                "price_as_of":   _price_asof,
                "period":        f"{_period_start} → {_price_asof}",
                "data_source":   "Yahoo Finance (yfinance)",
                "report_hash":   f"sha256:{_report_hash}",
                "optimizer":     "CLARABEL (cvxpy)",
            },
            "metrics": {
                "total_return_pct":  round(port_total_return, 2),
                "total_return_total_est_pct": round(port_total_return_total_est, 2),
                "spx_return_pct":    round(spx_return * 100, 2) if spx_return is not None else None,
                "alpha_pct":         round(alpha, 2) if spx_return is not None else None,
                "alpha_total_est_pct": round(alpha_total_est, 2) if spx_return is not None else None,
                "ann_return_pct":    round(ann_return * 100, 2),
                "ann_vol_pct":       round(ann_vol * 100, 2),
                "sharpe":            sharpe,
                "sortino":           sortino,
                "var_95_pct":        round(var_95, 3),
                "max_drawdown_pct":  round(max_dd, 2),
                "beta":              _safe_round(port_beta_total, 3),
                "beta_equity":       _safe_round(port_beta_equity, 3),
                "div_yield_pct":     round(port_div_yield_total, 3),
                "div_yield_equity_pct": round(port_div_yield_equity, 3),
                "equity_allocation_pct": round(equity_alloc_total * 100, 2),
                "cash_allocation_pct": round(cash_pct, 2),
                "risk_free_rate":    "4.5% (US T-Bill)",
                "method":            "Historical Simulation, 252 trading days",
            }
        })
    except Exception as e:
        logger.exception("[upload_portfolio] Unhandled error for file %s", getattr(file, 'filename', '?'))
        return {"error": str(e), "detail": "Portfolio analysis failed — see server logs for details"}


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO MEMORY API
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/v1/portfolio/history/{user_id}")
@limiter.limit("30/minute")
async def portfolio_history(
    request: Request,
    user_id: str,
    limit: int = 20,
    user: dict = Depends(_require_jwt),
):
    """
    Return portfolio snapshot history for a user.
    Shows how allocation + metrics evolved over time.
    """
    is_admin = user.get("role") == "admin" or user.get("user_id") == "admin"
    if not is_admin and user_id != str(user["sub"]):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        from portfolio_memory import get_user_snapshots, get_performance_history
        snapshots = get_user_snapshots(user_id, limit=limit)
        history   = get_performance_history(user_id, limit=limit)
        return {
            "user_id":    user_id,
            "count":      len(snapshots),
            "snapshots":  snapshots,
            "performance_history": history,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/v1/portfolio/snapshot/{snapshot_id}")
@limiter.limit("30/minute")
async def get_portfolio_snapshot(
    request: Request,
    snapshot_id: str,
    user: dict = Depends(_require_jwt),
):
    """
    Retrieve a specific portfolio snapshot by ID — full report + audit data.
    Enables report reproducibility.
    """
    try:
        from portfolio_memory import get_snapshot
        snap = get_snapshot(snapshot_id)
        if not snap:
            raise HTTPException(status_code=404, detail=f"Snapshot {snapshot_id} not found")
        is_admin = user.get("role") == "admin" or user.get("user_id") == "admin"
        if snap.get("user_id") and snap["user_id"] != str(user["sub"]) and not is_admin:
            raise HTTPException(status_code=403, detail="Access denied")
        return snap
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


@app.get("/v1/portfolio/compare")
@limiter.limit("20/minute")
async def compare_portfolio_snapshots(
    request: Request,
    snap_a: str,
    snap_b: str,
    user: dict = Depends(_require_jwt),
):
    """
    Compare two portfolio snapshots: allocation drift + metric changes.
    Use to track how a portfolio evolved between rebalances.
    """
    try:
        from portfolio_memory import compare_snapshots, get_snapshot
        is_admin = user.get("role") == "admin" or user.get("user_id") == "admin"
        snapshot_a = get_snapshot(snap_a)
        snapshot_b = get_snapshot(snap_b)
        if not snapshot_a or not snapshot_b:
            raise HTTPException(status_code=404, detail="One or both snapshot IDs not found")
        for snap in (snapshot_a, snapshot_b):
            if snap.get("user_id") and snap["user_id"] != str(user["sub"]) and not is_admin:
                raise HTTPException(status_code=403, detail="Access denied")
        diff = compare_snapshots(snap_a, snap_b)
        if not diff:
            raise HTTPException(status_code=404, detail="One or both snapshot IDs not found")

        # Build human-readable markdown summary
        md_lines = [
            f"## 📊 Portfolio Comparison",
            f"**{diff['date_a']}** → **{diff['date_b']}**",
            "",
            "### Allocation Changes",
            "| Ticker | Before | After | Δ | Direction |",
            "|--------|--------|-------|---|-----------|",
        ]
        for t, d in sorted(diff["allocation_diff"].items()):
            direction = "🟢 Added" if d["before"] == 0 else "🔴 Removed" if d["after"] == 0 else ("🔼 Increased" if d["delta"] > 0 else "🔽 Decreased" if d["delta"] < 0 else "⚪ Unchanged")
            md_lines.append(f"| **{t}** | {d['before']:.1f}% | {d['after']:.1f}% | {d['delta']:+.1f}pp | {direction} |")

        md_lines += ["", "### Metric Changes", "| Metric | Before | After | Δ | Better? |", "|--------|--------|-------|---|---------|"]
        for key, label, better_if in [
            ("sharpe",       "Sharpe Ratio",   "higher"),
            ("beta",         "Portfolio Beta", "lower"),
            ("cvar_95",      "CVaR 95%/day",   "higher"),
            ("ann_vol",      "Ann. Volatility","lower"),
            ("total_return", "1Y Total Return","higher"),
        ]:
            d = diff["metric_diff"].get(key)
            if d is None:
                continue
            improved = (d["delta"] > 0) == (better_if == "higher")
            icon = "✅" if improved else "🔴" if d["delta"] != 0 else "⚪"
            md_lines.append(f"| {label} | {d['before']:.2f} | {d['after']:.2f} | {d['delta']:+.2f} | {icon} |")

        diff["summary_md"] = "\n".join(md_lines)
        return diff
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


@app.post("/v1/global-allocate")
@limiter.limit("5/minute")
async def global_allocate(
    request: Request,
    user: dict = Depends(_require_jwt),
):
    """
    🌍 Global Allocation Engine — cross-market QP optimization.
    Allocates across US / GCC / Egypt / Crypto / Gold / Bonds.

    Body (JSON):
    {
        "profile":          "conservative" | "balanced" | "growth" | "aggressive",
        "region_include":   ["US","GCC","Gold"],      // optional: only these
        "region_exclude":   ["Crypto"],               // optional: exclude these
        "custom_caps":      {"Crypto": 0.10},         // optional: override caps
        "port_value_usd":   100000,                   // optional: $100k default
        "rf_rate":          0.045                     // optional: risk-free rate
    }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    profile          = body.get("profile", "balanced")
    region_include   = body.get("region_include")
    region_exclude   = body.get("region_exclude")
    custom_caps      = body.get("custom_caps")
    port_value_usd   = float(body.get("port_value_usd", 100_000))
    rf_rate          = float(body.get("rf_rate", 0.045))

    try:
        from global_allocator import allocate
        result = allocate(
            profile=profile,
            region_include=region_include,
            region_exclude=region_exclude,
            custom_caps=custom_caps,
            rf_rate=rf_rate,
            port_value_usd=port_value_usd,
        )
        return result
    except Exception as e:
        logger.error("Global allocator error: %s", e, exc_info=True)
        return {"error": str(e)}


@app.get("/v1/global-allocate/profiles")
@limiter.limit("60/minute")
async def global_allocate_profiles(
    request: Request,
    user: dict = Depends(_require_jwt),
):
    """List available risk profiles and regions for the Global Allocation Engine."""
    try:
        from global_allocator import _PROFILES, _UNIVERSE
        regions = sorted(set(a.region for a in _UNIVERSE))
        return {
            "profiles": {
                k: {"label": v["label"], "description": v["description"],
                    "max_beta": v["max_beta"], "max_vol": v["max_vol"]}
                for k, v in _PROFILES.items()
            },
            "regions":  regions,
            "assets":   [{"name": a.name, "region": a.region, "proxy": a.proxy,
                          "description": a.description} for a in _UNIVERSE],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/v1/portfolio/performance/{user_id}")
@limiter.limit("30/minute")
async def portfolio_performance_chart(
    request: Request,
    user_id: str,
    user: dict = Depends(_require_jwt),
):
    """
    Return time-series of key metrics across all snapshots for chart rendering.
    Frontend can plot Sharpe / Beta / CVaR evolution over time.
    """
    is_admin = user.get("role") == "admin" or user.get("user_id") == "admin"
    if not is_admin and user_id != str(user["sub"]):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        from portfolio_memory import get_performance_history
        history = get_performance_history(user_id, limit=50)
        return {
            "user_id": user_id,
            "data_points": len(history),
            "series": {
                "dates":        [h["date"] for h in history],
                "sharpe":       [h["sharpe"] for h in history],
                "beta":         [h["beta"] for h in history],
                "cvar_95":      [h["cvar_95"] for h in history],
                "total_return": [h["total_return"] for h in history],
                "ann_vol":      [h["ann_vol"] for h in history],
            }
        }
    except Exception as e:
        return {"error": str(e)}


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


@app.post("/v1/pilot-report")
@app.post("/v1/report")
@limiter.limit("10/minute")
async def pilot_report(
    payload: PilotReportPayload,
    request: Request,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
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
        session_user,
        ip=client_ip,
        user_agent=user_agent,
    )
    orchestrator.session_mgr.get_or_create_session(
        session_user,
        session_id=session_id,
        ip=client_ip,
        user_agent=user_agent,
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

    return JSONResponse(
        content={
            "html_report": html_report,
            "report_json": report_json,
        }
    )

@app.post("/v1/chat")
@limiter.limit("30/minute")
async def unified_chat(
    payload: MessagePayload,
    request: Request,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token")
):
    """نقطة الدخول الرئيسية للمحادثة - مع الحماية"""

    # Accept both X-API-Key and access-token headers (frontend uses access-token)
    _token = access_token or access_token_alt
    if _token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")

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
            headers=orchestrator.session_mgr.get_quota_header(payload.user_id),
        )

    message = payload.message
    resolution_payload = None
    active_file_id = None
    if payload.settings and isinstance(payload.settings, dict):
        active_file_id = payload.settings.get("active_file_id")

    if not payload.files and not active_file_id and _should_resolve_direct_analysis_request(message):
        from core.services.entity_resolution import EntityResolution, resolve_asset_entity

        # ── Fast pre-route via local ticker index ─────────────────────────────
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
    if resolution_payload:
        response_body["resolution"] = resolution_payload
    return JSONResponse(
        content=response_body,
        headers=orchestrator.session_mgr.get_quota_header(payload.user_id),
    )

# Backward-compatible aliases used by older UI pages (/chat and /api/chat).
@app.post("/chat")
@app.post("/api/chat")
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

# ── SSE Streaming Chat Endpoint ───────────────────────────────────────────────
@app.post("/v1/chat/stream")
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
            # stream_process_message already yields fully-formatted SSE lines
            async for sse_line in orchestrator.stream_process_message(
                user_id=payload.user_id,
                message=payload.message,
                session_id=session_id,
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
        },
    )


# --- TTS Endpoint ---

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

@app.post("/v1/tts")
@limiter.limit("20/minute")
async def text_to_speech(request: Request, tts_body: TTSRequest, access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
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
ADMIN_PASSPHRASE = os.getenv("ADMIN_PASSPHRASE", "") or os.getenv("ADMIN_TOKEN", "")
if not ADMIN_TOKEN:
    logger.warning("[STARTUP] ADMIN_TOKEN is not set — admin endpoints will be disabled")

class AdminAuthRequest(BaseModel):
    token: str = Field(..., min_length=1)

class AdminLoginRequest(BaseModel):
    password: str = Field(..., min_length=1)

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
    if ADMIN_TOKEN and token and orchestrator.session_mgr.verify_admin_password(token, ADMIN_TOKEN):
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

@app.post("/admin/login")
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

@app.post("/admin/logout")
@limiter.limit("10/minute")
async def admin_logout(request: Request):
    response = JSONResponse({"status": "ok"})
    response.delete_cookie("eisax_admin_session", path="/")
    return response


@app.get("/admin/sessions")
@limiter.limit("30/minute")
async def admin_sessions(request: Request, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
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

@app.get("/admin/session/{session_id}")
@limiter.limit("60/minute")
async def admin_session_detail(request: Request, session_id: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    return orchestrator.session_mgr.get_chat_history(session_id)

@app.get("/admin/stats")
@limiter.limit("30/minute")
async def admin_stats(request: Request, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    return orchestrator.session_mgr.get_admin_stats()

@app.post("/admin/user/{user_id}/block")
@limiter.limit("30/minute")
async def block_user(request: Request, user_id: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    orchestrator.session_mgr.set_user_blocked(user_id, True)
    orchestrator.session_mgr.log_admin_action("block_user", user_id)
    return {"status": "blocked", "user_id": user_id}

@app.post("/admin/user/{user_id}/unblock")
@limiter.limit("30/minute")
async def unblock_user(request: Request, user_id: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    orchestrator.session_mgr.set_user_blocked(user_id, False)
    orchestrator.session_mgr.log_admin_action("unblock_user", user_id)
    return {"status": "unblocked", "user_id": user_id}

@app.post("/admin/user/{user_id}/message")
@limiter.limit("20/minute")
async def send_admin_message(request: Request, user_id: str, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    content = body.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    orchestrator.session_mgr.queue_admin_message(user_id, content)
    orchestrator.session_mgr.log_admin_action("message_user", user_id, content[:80])
    return {"status": "queued", "user_id": user_id}

@app.get("/admin/messages")
@limiter.limit("30/minute")
async def get_admin_messages(request: Request, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    return orchestrator.session_mgr.get_admin_message_history()

@app.post("/admin/settings/password")
@limiter.limit("5/minute")
async def change_admin_password(request: Request, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    new_password = body.get("new_password", "").strip()
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    orchestrator.session_mgr.change_admin_password(new_password)
    return {"status": "password updated"}

@app.post("/admin/user/{user_id}/limit")
@limiter.limit("30/minute")
async def set_user_limit(request: Request, user_id: str, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    daily_limit = int(body.get("daily_limit", 0))
    if daily_limit < 0:
        raise HTTPException(status_code=400, detail="daily_limit must be >= 0")
    orchestrator.session_mgr.set_user_profile(user_id, daily_limit=daily_limit)
    orchestrator.session_mgr.log_admin_action("set_limit", user_id, str(daily_limit))
    return {"status": "ok", "user_id": user_id, "daily_limit": daily_limit}

@app.post("/admin/user/{user_id}/note")
@limiter.limit("30/minute")
async def set_user_note(request: Request, user_id: str, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    note = body.get("note", "")
    orchestrator.session_mgr.set_user_profile(user_id, note=note)
    orchestrator.session_mgr.log_admin_action("set_note", user_id, note[:60] if note else "cleared")
    return {"status": "ok", "user_id": user_id}

@app.post("/admin/user/{user_id}/tier")
@limiter.limit("30/minute")
async def set_user_tier(request: Request, user_id: str, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    tier = body.get("tier", "basic")
    if tier not in ("basic", "pro", "vip"):
        raise HTTPException(status_code=400, detail="tier must be basic, pro, or vip")
    orchestrator.session_mgr.set_user_profile(user_id, tier=tier)
    orchestrator.session_mgr.log_admin_action("set_tier", user_id, tier)
    return {"status": "ok", "user_id": user_id, "tier": tier}

@app.post("/admin/broadcast")
@limiter.limit("5/minute")
async def broadcast_message(request: Request, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    content = body.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    count = orchestrator.session_mgr.broadcast_admin_message(content)
    orchestrator.session_mgr.log_admin_action("broadcast", f"{count} users", content[:80])
    return {"status": "broadcast", "recipients": count}

@app.delete("/admin/user/{user_id}/sessions")
@limiter.limit("30/minute")
async def delete_user_sessions(request: Request, user_id: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    count = orchestrator.session_mgr.delete_user_sessions(user_id)
    orchestrator.session_mgr.log_admin_action("delete_sessions", user_id, f"{count} sessions deleted")
    return {"status": "deleted", "user_id": user_id, "sessions_deleted": count}

@app.post("/admin/ip/{ip}/block")
@limiter.limit("30/minute")
async def block_ip_endpoint(request: Request, ip: str, body: dict = {}, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    reason = (body or {}).get("reason", "")
    orchestrator.session_mgr.block_ip(ip, reason)
    orchestrator.session_mgr.log_admin_action("block_ip", ip, reason or "no reason")
    return {"status": "blocked", "ip": ip}

@app.post("/admin/ip/{ip}/unblock")
@limiter.limit("30/minute")
async def unblock_ip_endpoint(request: Request, ip: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    orchestrator.session_mgr.unblock_ip(ip)
    orchestrator.session_mgr.log_admin_action("unblock_ip", ip)
    return {"status": "unblocked", "ip": ip}

@app.get("/admin/blocked-ips")
@limiter.limit("30/minute")
async def get_blocked_ips(request: Request, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    return orchestrator.session_mgr.get_blocked_ips()

@app.get("/admin/audit-log")
@limiter.limit("30/minute")
async def get_audit_log(request: Request, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    return orchestrator.session_mgr.get_audit_log()

@app.get("/admin/notifications")
@limiter.limit("60/minute")
async def get_notifications(request: Request, since: str = "", access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token or "", request)
    if not since:
        from datetime import datetime, timedelta, timezone
        since = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    return orchestrator.session_mgr.get_new_activity(since)

@app.get("/admin/export/users")
@limiter.limit("10/minute")
async def export_users(request: Request, access_token: str = Header(None, alias="X-Admin-Key")):
    from fastapi.responses import StreamingResponse as SR
    import csv
    _check_admin(access_token or "", request)
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

@app.get("/api/history")
@limiter.limit("60/minute")
async def get_history(request: Request, user=Depends(_require_jwt)):
    # User can only see their own sessions; admin sees any user via query param
    if user.get("role") == "admin":
        target_uid = request.query_params.get("user_id") or user["sub"]
    else:
        target_uid = user["sub"]
    return orchestrator.session_mgr.get_user_sessions(str(target_uid))

@app.get("/api/history/{session_id}")
@limiter.limit("60/minute")
async def get_session_history(request: Request, session_id: str, user=Depends(_require_jwt)):
    history = orchestrator.session_mgr.get_chat_history(session_id)
    # Enforce ownership: verify the session belongs to the authenticated user
    sessions = orchestrator.session_mgr.get_user_sessions(str(user["sub"]))
    owned_ids = {s["session_id"] for s in sessions}
    if session_id not in owned_ids and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    return history

@app.delete("/api/history/{session_id}")
@limiter.limit("20/minute")
async def delete_session(request: Request, session_id: str, user=Depends(_require_jwt)):
    sessions = orchestrator.session_mgr.get_user_sessions(str(user["sub"]))
    owned_ids = {s["session_id"] for s in sessions}
    if session_id not in owned_ids and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    orchestrator.session_mgr.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}

@app.post("/v1/export")
@limiter.limit("10/minute")
async def export_chat(
    request: Request,
    user: dict = Depends(_require_jwt),
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
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
        download_token = _create_download_token(filename, str(user["sub"]))
        return {
            "success": True,
            "filename": filename,
            "download_url": f"/v1/download/{download_token}",
            "title": title,
            "format": fmt,
            "report_id": result.get("report_id"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/download/{token}")
@limiter.limit("60/minute")
async def download_file(request: Request, token: str, user=Depends(_require_jwt)):
    """Download exported file — requires authentication."""
    from fastapi.responses import FileResponse

    token_entry = _DOWNLOAD_TOKENS.get(token)
    if not token_entry or token_entry.get("expires", 0) <= _time.time():
        _DOWNLOAD_TOKENS.pop(token, None)
        raise HTTPException(status_code=404, detail="Download link expired or not found")

    if token_entry.get("user_id") != str(user["sub"]) and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access denied")

    filename = os.path.basename(token_entry["filename"])
    export_dir = str(EXPORTS_DIR)
    file_path = os.path.join(export_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=token_entry["filename"])

@app.get("/v1/autocomplete")
async def autocomplete_tickers(q: str = "", limit: int = 12):
    """
    Search pipeline cache for ticker/name matches.
    Returns up to `limit` results matching the query prefix.
    Used by the frontend autocomplete dropdown.
    """
    q = (q or "").strip()
    if not q or len(q) > 25:
        return {"results": []}
    try:
        import sys as _sys_ac, os as _os_ac
        _sys_ac.path.insert(0, _os_ac.path.dirname(_os_ac.path.abspath(__file__)))
        from pipeline import CacheManager as _ACCache
        _ac_cache = _ACCache()
        q_up = q.upper()

        _SUFFIX = {
            "uae": ".AE", "ksa": ".SR", "egypt": ".CA",
            "kuwait": ".KW", "qatar": ".QA", "bahrain": ".BH",
            "morocco": ".MA", "tunisia": ".TN",
            "america": "", "crypto": "-USD", "commodities": "",
        }

        seen_tickers: set = set()
        results: list = []

        def _extract_rows(df, mkt: str, mask) -> None:
            sfx = _SUFFIX.get(mkt, "")
            for _, row in df[mask].iterrows():
                bare = row["_bare"]
                full = f"{bare}{sfx}" if sfx else bare
                if full in seen_tickers:
                    continue
                seen_tickers.add(full)
                results.append({
                    "ticker": full,
                    "name":   str(row.get("name", bare)),
                    "market": mkt.upper(),
                    "price":  round(float(row.get("close", 0) or 0), 3),
                    "change": round(float(row.get("change", 0) or 0), 2),
                })

        _markets = ["uae", "ksa", "egypt", "kuwait", "qatar", "bahrain",
                    "morocco", "tunisia", "america", "crypto"]

        # ── Pass 1: prefix matches (startswith) ──────────────────────────────
        _dfs: dict = {}
        for mkt in _markets:
            if len(results) >= limit:
                break
            _df, _ = _ac_cache.get_latest(mkt)
            if _df is None or _df.empty:
                continue
            _df = _df.copy()
            _df["_bare"] = _df["ticker"].astype(str).str.split(":").str[-1].str.upper()
            _name_col   = _df["name"].astype(str).str.upper()
            _mask = _df["_bare"].str.startswith(q_up) | _name_col.str.startswith(q_up)
            _extract_rows(_df[_mask].head(5), mkt, slice(None))
            _dfs[mkt] = _df  # cache for pass 2

        # ── Pass 2: substring matches (contains) — fills remaining slots ─────
        if len(results) < limit:
            for mkt in _markets:
                if len(results) >= limit:
                    break
                _df = _dfs.get(mkt)
                if _df is None:
                    _df_raw, _ = _ac_cache.get_latest(mkt)
                    if _df_raw is None or _df_raw.empty:
                        continue
                    _df = _df_raw.copy()
                    _df["_bare"] = _df["ticker"].astype(str).str.split(":").str[-1].str.upper()
                _name_col = _df["name"].astype(str).str.upper()
                _mask2 = _df["_bare"].str.contains(q_up, na=False) | _name_col.str.contains(q_up, na=False)
                _extract_rows(_df[_mask2].head(4), mkt, slice(None))

        return {"results": results[:limit]}
    except Exception as _ace:
        return {"results": []}

@app.get("/v1/brain/status")
@limiter.limit("30/minute")
async def brain_status(request: Request, access_token: str = Header(None, alias="X-API-Key")):
    if access_token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    from learning_engine import get_engine
    return get_engine().status()

@app.get("/v1/brain/wisdom")
@limiter.limit("20/minute")
async def brain_wisdom(request: Request, access_token: str = Header(None, alias="X-API-Key")):
    if access_token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
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

@app.post("/v1/alerts")
@limiter.limit("20/minute")
async def create_alert(request: Request, access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN: raise HTTPException(403, "Unauthorized")
    body = await request.json()
    from core.price_alerts import add_alert
    alert_id = add_alert(body.get('user_id','anonymous'), body['ticker'], body['condition'], body['threshold'])
    return {'alert_id': alert_id, 'status': 'created'}

@app.get("/v1/alerts")
@limiter.limit("30/minute")
async def list_alerts(request: Request, user_id: str = 'anonymous', access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN: raise HTTPException(403, "Unauthorized")
    from core.price_alerts import get_user_alerts
    return get_user_alerts(user_id)

@app.delete("/v1/alerts/{alert_id}")
@limiter.limit("20/minute")
async def remove_alert(request: Request, alert_id: int, user_id: str = 'anonymous', access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN: raise HTTPException(403, "Unauthorized")
    from core.price_alerts import delete_alert
    delete_alert(alert_id, user_id)
    return {'status': 'deleted'}

@app.get('/v1/version')
@limiter.limit('60/minute')
async def app_version(request: Request):
    return {'status': 'ok'}

# ── HTML → PDF Export ──
class HtmlExportPayload(BaseModel):
    html: str
    filename: str = ""
    access_token: str = ""

@app.post("/v1/export/html")
@limiter.limit("10/minute")
async def export_html_to_pdf(
    request: Request,
    payload: HtmlExportPayload,
    access_token: str = Header(None, alias="X-API-Key")
):
    token = access_token or payload.access_token
    if token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
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
        download_token = _create_download_token(fname, "admin")
        return {"url": f"/v1/download/{download_token}", "download_url": f"/v1/download/{download_token}", "filename": fname}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/dashboard/{ticker}")
@limiter.limit("20/minute")
async def dashboard(request: Request, ticker: str, access_token: str = Header(None, alias="X-API-Key")):
    """Return all dashboard data for a ticker in one call — no LLM, runs concurrently."""
    if access_token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
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
                loop.run_in_executor(None, _fetch_tadawul_candles, tadawul_id),
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
                loop.run_in_executor(None, get_market_pulse),
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

@app.post("/v1/translate-ar")
@limiter.limit("20/minute")
async def translate_to_arabic(request: Request, payload: TranslatePayload, access_token: str = Header(None, alias="X-API-Key")):
    """Translate an English investment report to Arabic. Primary: DeepSeek. Fallback: GLM."""
    token = access_token or payload.access_token
    if token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")

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
                        "model": "deepseek-chat",
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


@app.post("/v1/export/html-pdf")
@limiter.limit("5/minute")
async def export_html_pdf(request: Request, payload: HtmlExportPayload, access_token: str = Header(None, alias="X-API-Key")):
    token = access_token or payload.access_token
    if token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
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
        download_token = _create_download_token(fname, "admin")
        return {"url": f"/v1/download/{download_token}", "download_url": f"/v1/download/{download_token}", "filename": fname}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  B2B AUTH  — /auth/*  and  /admin/*
# ══════════════════════════════════════════════════════════════════════════════
from core.auth    import hash_password, verify_password, create_token, decode_token, generate_temp_password, JWT_SECRET, JWT_ALGORITHM
from core.user_db import (init_users_table, create_user, get_user_by_email,
                           get_user_by_id, list_users, update_user, delete_user, record_login)

# Initialise users table on startup (idempotent)
init_users_table()

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


def _resolve_user_context(
    access_token: Optional[str] = None,
    access_token_alt: Optional[str] = None,
    authorization: Optional[str] = None,
) -> dict:
    bearer = (authorization or "").removeprefix("Bearer ").strip()
    if bearer and not bearer.startswith("eixa_") and bearer != SECURE_TOKEN:
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


# ── Pydantic models ───────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email:    str
    password: str

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

class CreateUserRequest(BaseModel):
    email:    str
    name:     str
    role:     str = "user"   # "user" | "admin"

class UpdateUserRequest(BaseModel):
    name:      Optional[str] = None
    role:      Optional[str] = None
    is_active: Optional[int] = None


class APIKeyCreateRequest(BaseModel):
    name: str = "Default"
    tier: str = "basic"
    daily_limit: int = 0


class APIKeyValidateRequest(BaseModel):
    key: Optional[str] = None


# ── Auth endpoints ─────────────────────────────────────────────────────────────
@app.post("/auth/login")
@limiter.limit("10/minute")
async def auth_login(request: Request, body: LoginRequest):
    user = get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account disabled")
    record_login(user["id"])
    token = create_token(
        user["id"], user["email"], user["role"],
        must_change=bool(user["must_change_pw"])
    )
    return {
        "token":       token,
        "must_change": bool(user["must_change_pw"]),
        "name":        user["name"],
        "role":        user["role"],
    }


@app.post("/auth/change-password")
@limiter.limit("5/minute")
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


@app.get("/auth/me")
@limiter.limit("60/minute")
async def auth_me(request: Request, payload: dict = Depends(_require_jwt)):
    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id":    user["id"],
        "email": user["email"],
        "name":  user["name"],
        "role":  user["role"],
    }


@app.post("/v1/keys")
@limiter.limit("20/minute")
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


@app.get("/v1/keys")
@limiter.limit("20/minute")
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


@app.delete("/v1/keys/{key_id}")
@limiter.limit("20/minute")
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


@app.post("/v1/keys/validate")
@limiter.limit("20/minute")
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


# ── Admin endpoints ────────────────────────────────────────────────────────────
@app.post("/admin/users")
@limiter.limit("10/minute")
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


@app.get("/admin/users")
@limiter.limit("30/minute")
async def admin_list_users(request: Request, _: dict = Depends(_require_admin)):
    return list_users()


@app.patch("/admin/users/{user_id}")
@limiter.limit("20/minute")
async def admin_update_user(request: Request, user_id: int, body: UpdateUserRequest, _: dict = Depends(_require_admin)):
    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    if not update_user(user_id, **changes):
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


@app.delete("/admin/users/{user_id}")
@limiter.limit("10/minute")
async def admin_delete_user(request: Request, user_id: int, _: dict = Depends(_require_admin)):
    if not delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


@app.post("/admin/users/{user_id}/reset-password")
@limiter.limit("10/minute")
async def admin_reset_password(request: Request, user_id: int, _: dict = Depends(_require_admin)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    temp_pw = generate_temp_password()
    update_user(user_id, password_hash=hash_password(temp_pw), must_change_pw=1)
    return {"temp_password": temp_pw}


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


# ── Session Cleanup ───────────────────────────────────────────────────────────

@app.post("/admin/cleanup")
@limiter.limit("5/minute")
async def run_cleanup(
    request: Request,
    days: int = 30,
    access_token: str = Header(None, alias="X-Admin-Key"),
):
    _check_admin(access_token or "", request)
    result = orchestrator.session_mgr.cleanup_old_sessions(days_to_keep=days)
    return result


# ── Logging Dashboard ─────────────────────────────────────────────────────────

@app.get("/admin/logs")
@limiter.limit("30/minute")
async def admin_logs_page(request: Request):
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / "admin_logs.html"))


@app.get("/admin/logs/stream")
@limiter.limit("10/minute")
async def admin_logs_stream(
    request: Request,
    access_token: str = Header(None, alias="X-Admin-Key"),
    access_token_alt: str = Header(None, alias="X-API-Key"),
    token: str = "",
):
    _check_admin(access_token or access_token_alt or token or "", request)
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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Analytics Dashboard ───────────────────────────────────────────────────────

@app.get("/admin/analytics")
@limiter.limit("30/minute")
async def admin_analytics_page(request: Request):
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / "admin_analytics.html"))


@app.get("/admin/analytics/data")
@limiter.limit("30/minute")
async def admin_analytics_data(
    request: Request,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
    token: str = "",
):
    _check_admin(access_token or access_token_alt or token or "", request)

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
                (day,),
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
            (today.isoformat(),),
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


@app.get("/v1/usage")
@limiter.limit("30/minute")
async def user_usage(
    request: Request,
    days: int = 30,
    user_id: Optional[str] = None,
    user: dict = Depends(_require_jwt),
):
    if user.get("role") == "admin" and user_id:
        effective_user_id = user_id
    else:
        effective_user_id = str(user["sub"])
    return orchestrator.session_mgr.get_user_usage_stats(effective_user_id, days=min(days, 90))


if __name__ == "__main__":
    uvicorn.run("api_bridge_v2:app", host="0.0.0.0", port=8000, workers=2)


# ── F-5: Redis Health ─────────────────────────────────────────────────────────

@app.get("/v1/redis/health")
@limiter.limit("30/minute")
async def redis_health(
    request: Request,
    access_token:     str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    from core.redis_store import redis_info
    return redis_info()


# ── F-6: Referral System ──────────────────────────────────────────────────────

@app.get("/v1/referral")
@limiter.limit("30/minute")
async def get_referral(
    request: Request,
    user_id: Optional[str] = None,
    user: dict = Depends(_require_jwt),
):
    if user.get("role") == "admin" and user_id:
        effective_user_id = user_id
    else:
        effective_user_id = str(user["sub"])
    from core.referrals import get_referral_stats
    return get_referral_stats(effective_user_id)


@app.post("/v1/referral/apply")
@limiter.limit("5/minute")
async def apply_referral_code(
    request: Request,
    user: dict = Depends(_require_jwt),
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    user_id = str(user["sub"])
    code = str(body.get("code", "")).strip()
    if not code:
        raise HTTPException(status_code=400, detail="Required: code")
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


@app.post("/v1/webhooks")
@limiter.limit("10/minute")
async def register_webhook(
    request: Request,
    body: WebhookConfig,
    user: dict = Depends(_require_jwt),
):
    effective_user_id = str(user["sub"])
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
        (effective_user_id, body.url, _json.dumps(body.events), body.secret))
    wid = cur.lastrowid; conn.commit(); conn.close()
    logger.info("[webhooks] registered %d for %s → %s", wid, effective_user_id, body.url)
    return {"webhook_id": wid, "status": "registered", "url": body.url}


@app.get("/v1/webhooks")
@limiter.limit("30/minute")
async def list_webhooks(
    request: Request,
    user_id: Optional[str] = None,
    user: dict = Depends(_require_jwt),
):
    if user.get("role") == "admin" and user_id:
        effective_user_id = user_id
    else:
        effective_user_id = str(user["sub"])
    import sqlite3 as _sl2
    conn = _sl2.connect(str(APP_DB)); conn.row_factory = _sl2.Row
    try:
        rows = conn.execute(
            "SELECT id,url,events,active,created_at FROM webhooks WHERE user_id=? ORDER BY created_at DESC",
            (effective_user_id,)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


@app.delete("/v1/webhooks/{webhook_id}")
@limiter.limit("20/minute")
async def delete_webhook(
    request: Request,
    webhook_id: int,
    user: dict = Depends(_require_jwt),
):
    user_id = str(user["sub"])
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


@app.post("/v1/billing/checkout")
@limiter.limit("10/minute")
async def create_checkout(
    request: Request,
    body: CheckoutRequest,
    user: dict = Depends(_require_jwt),
):
    effective_user_id = str(user["sub"])
    effective_email = user.get("email") or body.email
    try:
        from core.billing import StripeBilling
        url = StripeBilling().create_checkout_session(effective_user_id, effective_email, body.tier)
        return {"checkout_url": url}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("[billing/checkout] %s", exc)
        raise HTTPException(status_code=500, detail="Billing service error")


@app.post("/v1/billing/webhook")
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


@app.post("/v1/billing/portal")
@limiter.limit("10/minute")
async def billing_portal(
    request: Request,
    body: PortalRequest,
    user: dict = Depends(_require_jwt),
):
    effective_user_id = str(user["sub"])
    try:
        from core.billing import StripeBilling
        billing = StripeBilling()
        cid = billing.get_customer_id(effective_user_id)
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

@app.get("/v1/sentiment/{ticker}")
@limiter.limit("20/minute")
async def get_ticker_sentiment(
    request: Request,
    ticker: str,
    use_cache: bool = True,
    access_token:     str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    """VADER sentiment analysis on recent news for a single ticker."""
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        from core.sentiment import SentimentAnalyzer
        result = await asyncio.get_event_loop().run_in_executor(
            None, SentimentAnalyzer().analyze_ticker, ticker.upper(), use_cache
        )
        return result
    except Exception as exc:
        logger.error("[sentiment] ticker=%s %s", ticker, exc)
        raise HTTPException(status_code=500, detail=f"Sentiment error: {exc}")


@app.post("/v1/sentiment/batch")
@limiter.limit("5/minute")
async def get_batch_sentiment(
    request: Request,
    access_token:     str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    """Analyze sentiment for multiple tickers at once (max 10)."""
    _check_admin(access_token or access_token_alt or "", request)
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


@app.get("/v1/sentiment/market/overview")
@limiter.limit("10/minute")
async def get_market_sentiment(
    request: Request,
    use_cache: bool = True,
    access_token:     str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    """Aggregate market sentiment from major ETF news (SPY/QQQ/DIA)."""
    _check_admin(access_token or access_token_alt or "", request)
    try:
        from core.sentiment import SentimentAnalyzer
        result = await asyncio.get_event_loop().run_in_executor(
            None, SentimentAnalyzer().market_sentiment, use_cache
        )
        return result
    except Exception as exc:
        logger.error("[sentiment/market] %s", exc)
        raise HTTPException(status_code=500, detail=f"Sentiment error: {exc}")


@app.get("/v1/sentiment/{ticker}/trend")
@limiter.limit("20/minute")
async def get_sentiment_trend(
    request: Request,
    ticker: str,
    hours: int = 48,
    access_token:     str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    """Historical sentiment trend (hourly buckets) from local DB."""
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
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


@app.post('/v1/backtest')
@limiter.limit('10/minute')
async def run_backtest(
    request: Request,
    body: BacktestRequest,
    access_token: str = Header(None, alias='X-API-Key'),
    access_token_alt: str = Header(None, alias='access-token'),
):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(403, 'Unauthorized')
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
            body.initial_capital,
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


@app.post("/v1/screener")
@limiter.limit("5/minute")
async def stock_screener(
    request: Request,
    body: ScreenerRequest,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(403, "Unauthorized")
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
            sector=body.sector,
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

@app.get("/v1/forex")
@limiter.limit("20/minute")
async def get_forex(
    request: Request,
    category: str = "all",   # all | arab | major | em
    use_cache: bool = True,
    access_token:     str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    """Live FX rates — Arab pairs (AED/SAR/EGP/KWD/QAR/BHD) + major pairs."""
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
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


@app.get("/v1/forex/{symbol}")
@limiter.limit("30/minute")
async def get_forex_pair(
    request: Request,
    symbol: str,
    access_token:     str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    """Single FX pair — e.g. /v1/forex/USDAED=X or /v1/forex/EURUSD"""
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
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
