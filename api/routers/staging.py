"""
EisaX — Staging-API Router
Extracted from api_bridge_v2.py (lines 383-2179).

Routers exported:
  staging_router      → /staging-api/* (9 routes)
  guest_admin_router  → /v1/admin/guest-users* (5 routes)
"""

import os
import logging
import re as _re
import copy as _copy
import time as _time
import json as _json
import subprocess as _subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends, Form, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

# ── Limiter (own instance — decorators only; actual enforcement via app.state.limiter) ──
limiter = Limiter(key_func=get_remote_address)

# ── Logger (same name as api_bridge — logs merge correctly) ──────────────────
logger = logging.getLogger("api_bridge")

# ── Env-based globals (re-read from env — same source as api_bridge_v2) ─────
SECURE_TOKEN = os.getenv("SECURE_TOKEN", "")
_ENVIRONMENT = os.getenv("ENVIRONMENT", "production").strip().lower()
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
_STAGING_UPSTREAM_BASE = os.getenv("STAGING_UPSTREAM_BASE", "http://127.0.0.1:8000").rstrip("/")
_GUEST_LIMIT_MESSAGE = "This guest demo has reached its analysis limit. Please contact EisaX for extended access."
_HTPASSWD_PATH = Path(os.getenv("EISAX_HTPASSWD_PATH", "/etc/nginx/.htpasswd_eisax"))

try:
    _APP_VERSION = open("/home/ubuntu/investwise/version.txt").read().strip()
except Exception:
    _APP_VERSION = "2.0.0"

# ── Intent constants ─────────────────────────────────────────────────────────
_INTENT_ASSET_ANALYSIS = "asset_analysis"
_INTENT_PORTFOLIO_ANALYSIS = "portfolio_analysis"
_INTENT_PORTFOLIO_CONSTRUCTION = "portfolio_construction"
_INTENT_FIXED_INCOME = "fixed_income"


# ── Auth helpers ─────────────────────────────────────────────────────────────
def _require_admin_cookie(request: Request) -> dict:
    """Validate admin session cookie (JWT). Raises 401/403 on failure."""
    import jwt as _jwt
    from core.auth import decode_token
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


# ── Pydantic models ──────────────────────────────────────────────────────────
class StagingLeadPayload(BaseModel):
    email: str = Field(..., max_length=320)
    name: Optional[str] = Field(default=None, max_length=120)
    query: Optional[str] = Field(default=None, max_length=500)
    report_kind: Optional[str] = Field(default=None, max_length=40)


class GuestAccessUserRequest(BaseModel):
    username: str
    role: str = "guest"
    password: Optional[str] = None
    copy_password_from: Optional[str] = "alan.talib"
    active: bool = True
    analysis_limit: int = 6
    portfolio_limit: int = 0
    analyses_used: Optional[int] = None
    portfolios_used: Optional[int] = None


# ── Staging helper functions ──────────────────────────────────────────────────
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
            r'\bFundamental:\s*\*?\*?\s*(BUY|HOLD|SELL|REDUCE|AVOID|ACCUMULATE|WATCHLIST)\b',
            r'\bالأساسيات:\s*\*?\*?\s*(BUY|HOLD|SELL|REDUCE|AVOID|ACCUMULATE)\b',
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
    if result_type == "construction":
        overview = _staging_extract_section(markdown, ["Portfolio Overview", "Strategy Readiness"])
        for paragraph in _re.split(r"\n\s*\n", overview or ""):
            cleaned = _staging_clean_text(paragraph)
            if cleaned and len(cleaned) > 40 and not cleaned.startswith("|") and not cleaned.startswith("-") and not cleaned.startswith("#"):
                sentences = _re.findall(r"[^.!?]+[.!?]+", cleaned)
                return " ".join(sentences[:2]).strip() or cleaned[:280]
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
    if result_type == "construction":
        insights: list[str] = []
        ret_m = _re.search(r"Expected Annual Return[:\s]+([0-9.]+%[^.\n]{0,60})", markdown or "", _re.I)
        if ret_m:
            insights.append("Expected annual return: " + ret_m.group(1).strip().rstrip(","))
        vol_m = _re.search(r"Annual Volatility[:\s]+([0-9.]+%[^.\n]{0,80})", markdown or "", _re.I)
        shr_m = _re.search(r"Sharpe Ratio[:\s]+([0-9.]+[^.\n]{0,60})", markdown or "", _re.I)
        if vol_m and shr_m:
            insights.append(f"Volatility {vol_m.group(1).strip().rstrip(',')} · Sharpe {shr_m.group(1).strip()}")
        elif vol_m:
            insights.append("Annual volatility: " + vol_m.group(1).strip().rstrip(","))
        conc_m = _re.search(
            r"(Concentration[^.\n]{10,120}|correlation[^.\n]{10,120}|Effective N[^.\n]{10,120})",
            markdown or "", _re.I
        )
        if conc_m:
            insights.append(_staging_clean_text(conc_m.group(1)))
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


def _public_resolution_payload(resolution: dict) -> dict:
    allowed = {
        "query_raw", "normalized_query", "resolution_status", "symbol",
        "market", "asset_type", "currency", "name", "local_name", "exchange",
    }
    payload = {key: value for key, value in (resolution or {}).items() if key in allowed and value not in (None, "")}
    candidates = resolution.get("candidates") if isinstance(resolution, dict) else None
    if isinstance(candidates, list):
        safe_candidates = []
        for candidate in candidates[:8]:
            if not isinstance(candidate, dict):
                continue
            safe_candidates.append(
                {
                    key: candidate.get(key)
                    for key in ("symbol", "market", "asset_type", "currency", "name", "local_name", "exchange")
                    if candidate.get(key) not in (None, "")
                }
            )
        if safe_candidates:
            payload["candidates"] = safe_candidates
    return payload


def _resolution_error_response(resolution: dict, *, guest_visible: bool = False) -> JSONResponse:
    status = resolution.get("resolution_status") or "unresolved"
    normalized_query = resolution.get("normalized_query") or resolution.get("query_raw") or "this request"
    if status == "ambiguous":
        detail = f"'{normalized_query}' is ambiguous. Please specify the exact instrument."
    else:
        detail = f"Unable to resolve '{normalized_query}' safely. Please use the exact ticker."
    safe_resolution = _public_resolution_payload(resolution) if guest_visible else resolution
    payload = {"ok": False, "detail": detail, **safe_resolution}
    return JSONResponse(status_code=400, content=payload)


def _classify_request_intent(message: str, *, has_file: bool = False) -> str:
    raw = (message or "").strip()
    if has_file:
        return _INTENT_PORTFOLIO_ANALYSIS
    if not raw:
        return _INTENT_ASSET_ANALYSIS

    lowered = raw.lower()
    portfolio_analysis_keywords = [
        "upload", "csv", "xlsx", "xls", "my portfolio", "my holdings", "my positions",
        "analyze portfolio", "analyze my portfolio", "portfolio analysis", "portfolio review",
        "portfolio risk", "holdings analysis", "upload portfolio",
    ]
    portfolio_analysis_keywords_ar = [
        "محفظتي", "محفظتى", "حلل محفظتي", "حلل محفظتى",
        "تحليل محفظتي", "تحليل محفظتى", "ارفع", "رفع ملف", "csv", "اكسل", "إكسل",
    ]
    if any(keyword in lowered for keyword in portfolio_analysis_keywords):
        return _INTENT_PORTFOLIO_ANALYSIS
    if any(keyword in raw for keyword in portfolio_analysis_keywords_ar):
        return _INTENT_PORTFOLIO_ANALYSIS

    construction_keywords = [
        "portfolio", "build", "create", "construct", "design", "allocate",
        "rebalance", "re-balance", "allocation", "max down", "max drawdown",
        "risk tolerance", "drawdown", "aggressive", "conservative", "balanced",
        "moderate", "equities", "stocks", "saudi", "us equities", "american equities",
    ]
    construction_keywords_ar = [
        "محفظة", "محفظه", "ابني", "ابنى", "ابنِ", "كوّن", "كون", "خصص", "خصّص",
        "وزع", "وزّع", "أعد موازنة", "اعد موازنة", "موازنة", "عدواني", "عدوانيه",
        "محافظ", "متوازن", "مخاطرة", "مخاطر",
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


# ── Trial-user report sanitizer ───────────────────────────────────────────────
_TRIAL_SANITIZE_RULES: list[tuple] = [
    (
        _re.compile(
            r"\*Conviction:\s*(\d+)%\s*—\s*"
            r"Score\([^)]+\)\s*\+\s*Upside\([^)]+\)\s*\+"
            r"\s*Coverage\([^)]+\)\s*\+\s*Trend\([^)]+\)\s*\+"
            r"\s*ADX\([^)]+\)\s*→\s*Raw\([^)]+\)\s*→\s*Clamped\([^)]+\)\*"
        ),
        lambda m: f"*Conviction: {m.group(1)}% — derived from quality, upside, coverage, trend strength, and risk controls.*",
    ),
    (
        _re.compile(
            r"Score is 100% deterministic — computed from live market data using explicit "
            r"mathematical thresholds\. No LLM estimation\. Every point is traceable to a "
            r"specific data input\."
        ),
        "Score is generated from EisaX's proprietary market-data framework and validated against live inputs.",
    ),
    (
        _re.compile(
            r"Sizing: Score \d+/100 → \S+ tier \| Core: [^|]+ \| Max: [^\—–-]+[—–-] "
            r"deterministic table, not LLM judgment"
        ),
        "Position sizing derived from EisaX's proprietary risk framework.",
    ),
]


def _sanitize_trial_report(text: str) -> str:
    if not text:
        return text
    for pattern, replacement in _TRIAL_SANITIZE_RULES:
        if callable(replacement):
            text = pattern.sub(replacement, text)
        else:
            text = pattern.sub(replacement, text)
    return text


def _cleanup_report_text_artifacts(text: str) -> str:
    if not text:
        return text
    replacements = [
        (
            r"Position maintained\s+[—-]\s+momentum is bullish momentum\.\s+"
            r"Await clearer confirmation before adding or reducing exposure\.",
            "Position maintained — bullish momentum exists, but trend confirmation remains incomplete. "
            "Await clearer confirmation before adding or reducing exposure.",
        ),
        (r"\bmomentum is bullish momentum\b", "bullish momentum exists"),
        (r"\bmomentum is bearish momentum\b", "bearish momentum is present"),
        (r"\bmomentum is neutral momentum\b", "momentum remains neutral"),
        # Phrase repair — the LLM/structural-variation occasionally rewrites
        # the templated "avoid chasing" → "Sell chasing", which a reader
        # parses as a hidden verdict in a No-Action sentence. Runs here as
        # a final safety net so it catches reports served from either the
        # in-memory _REPORT_CACHE or the SQLite analysis_cache, regardless
        # of when they were generated.
        (r"\b[Ss]ell\s+chasing\b", "avoid chasing"),
        (r"\b[Ss]ell\s+(adding|entering|extending)\b", r"avoid \1"),
        # Reasoning / meta-instruction leaks — the LLM occasionally echoes
        # parenthetical guidance like "(pre-computed, exact copy)" or
        # "(exact copy from PRE-COMPUTED VALUES)" which it absorbed from
        # the system prompt. Strip these so the published report carries
        # no editorial scaffolding.
        # Any parenthetical that contains "pre-computed" / "precomputed"
        # is editorial scaffolding leaked from the system prompt — strip it
        # regardless of what follows ("exact copy", "scenarios", "values",
        # "do not recalculate", etc.).
        (r"\s*\([^)]*\bpre[- ]?computed\b[^)]*\)", ""),
        (r"\s*\([^)]*\bexact\s+(?:copy|values)\b[^)]*\)", ""),
        (r"\s*\(verbatim(?:\s+from\s+[^)]+)?\)", ""),
        (r"\s*\(do\s+not\s+recalculate\)", ""),
        (r"\s*\(copy\s+from\s+PRE-COMPUTED[^)]*\)", ""),
        # LLM-style first-person reasoning preambles that occasionally slip
        # through ("Wait — pre-computed values …", "Let me look at …",
        # "I need to …", "Actually, …", "Hmm, …", "Looking at the data, …").
        # Drop the leading clause up to the first sentence terminator.
        (
            r"(?m)^\s*(?:Wait[\s,\.—–\-]+|Hmm[\s,\.]+|Actually[\s,]+|"
            r"Let\s+me\s+(?:think|look|check|see|verify)[^\n.]*[\.—\-]\s*|"
            r"I(?:'ll| will)\s+(?:start by|need to|look at|check)[^\n.]*[\.—\-]\s*|"
            r"Looking\s+at\s+(?:the|this)[^\n.]*[\.—\-]\s*)",
            "",
        ),
        # Sometimes the LLM annotates table titles with "(used in the report)"
        # or "(below in the report)" — editorial scaffolding, strip it.
        (r"\s*\((?:as\s+)?(?:shown|used|displayed|provided)(?:\s+(?:above|below|here))?\s*(?:in\s+the\s+report)?\)", ""),

        # ── Fair Value math claim repair ──────────────────────────────────
        # The LLM sometimes prints "Forward EPS 0.20 د.إ × 17x sector P/E"
        # whose arithmetic does not equal the stated FV (0.20 × 17 = 3.40,
        # not the actual FV figure). Strip the derivation phrase so the
        # value stands on its own as a proprietary EisaX estimate without
        # false arithmetic. Engine output retains the figure; only the
        # misleading derivation tail is removed.
        # Patterns use `[^)\n]` (allowing '.') with bounded length so the
        # price decimal (0.20) doesn't break the class.
        (
            r",\s*derived\s+from\s+(?:a\s+)?[Ff]orward\s+EPS\s+(?:of\s+)?[^,\n]{1,90}?\d+\s*x\b[^,\n]{0,40}",
            "",
        ),
        (
            r"\s*\(\s*[Ff]orward\s+EPS[^)\n]{1,80}?\bx\s*sector[^)\n]{0,40}\)",
            " (proprietary EisaX methodology)",
        ),
        (
            r"derived\s+from\s+(?:a\s+)?[Ff]orward\s+EPS\s+(?:of\s+)?[^,\n]{1,80}?\d+\s*x[^,\n]{0,40}?(?:sector[- ]?(?:average\s+)?)?P/E",
            "derived from EisaX proprietary methodology (forward earnings × sector multiple, peer-benchmarked)",
        ),
        # Catch the explicit equation form:
        # "EisaX Fair Value Estimate is calculated as Forward EPS (X) × Y x sector P/E = Z"
        # The arithmetic rarely matches (X × Y ≠ Z because Z uses a different
        # peer-adjusted multiple). Strip the equation tail so only the
        # methodology label and the value remain.
        (
            r"(EisaX\s+Fair\s+Value\s+Estimate\s+is\s+)"
            r"calculated\s+as\s+[Ff]orward\s+EPS[^=\n]{0,140}=\s*",
            r"\1based on the EisaX proprietary methodology (peer-benchmarked forward earnings × sector multiple) — estimated at ",
        ),
        # Drop "= Z" tail when methodology is already labelled
        (
            r"((?:Forward|TTM)\s+EPS\s*\(?[^)\n]{0,60}\)?\s*[×x]\s*\d+\s*x?\s+(?:sector|peer)[- ]?(?:average\s+)?P/E)\s*=\s*\*{0,2}[د.إ﷼\$]?\s*[\d\.,]+\*{0,2}",
            r"\1",
        ),

        # ── Score label disambiguation ─────────────────────────────────────
        # An unlabeled "Score: 50/100 — Score reflects business quality" line
        # is the Fundamental Quality Score (different metric from EisaX Score).
        # Relabel so the reader doesn't conflate it with the top-card 71/100.
        # Handle leading `**` wrap (e.g. `**Score: 50/100 — Score reflects...**`).
        (
            r"(?m)^(\s*)\*{0,2}\s*Score:\s*\*{0,2}(\d{1,3})/100\*{0,2}\s*[—\-]\s*(?:Score\s+reflects|Reflects)\s+business\s+quality[^.\n]*\.?\*{0,2}",
            r"\1**Fundamental Quality Score: \2/100** — reflects business quality only (independent of the top-card EisaX Score, which is a blended composite).",
        ),

        # ── Analyst coverage wording ──────────────────────────────────────
        # Accept several variants the LLM has used: "exists", "found",
        # "was found", "available", "reported", "is reported", "is provided".
        # Normalise to one phrase.
        (
            r"\b[Nn]o\s+(?:formal\s+)?analyst\s+coverage\s+"
            r"(?:exists?|"
            r"(?:is\s+|was\s+|currently\s+)?(?:found|available|reported|provided|tracked|disclosed)"
            r")(?:\s+for\s+\S+)?",
            "No analyst coverage available in current dataset",
        ),

        # ── Position sizing — strip orphan "refer to" pointer ────────────
        # The report has a concrete "≤5% of portfolio" sizing earlier; this
        # later "refer to EisaX Score → Core Allocation" line is dangling
        # and adds no value. Remove the whole sentence/bullet.
        (
            r"[\s\-\*]*\*?\s*Full\s+position\s+sizing:\s*refer\s+to\s+EisaX\s+Score[^.\n]*\.\s*\*?",
            "",
        ),
        (
            r"\s*[Pp]osition\s+sizing[^.]{0,30}refer\s+to\s+EisaX\s+Score[^.\n]*\.",
            "",
        ),
    ]
    for pattern, replacement in replacements:
        text = _re.sub(pattern, replacement, text, flags=_re.IGNORECASE)

    # ── FV / Scenario disambiguation footnote ─────────────────────────────
    # When both the proprietary EisaX Fair Value Estimate and a multi-row
    # Valuation Range Table (Bear/Base/Bull) appear, append a one-line
    # footnote so the reader doesn't conflate "EisaX FV" with the Bull
    # scenario implied price (they use different methodologies).
    has_fv = bool(_re.search(r"EisaX\s+(?:Fair\s+Value|FV)", text, _re.IGNORECASE))
    # Accept either the "Implied Price" column header, the literal
    # "Valuation Range Table" / "Valuation Range:" heading, or the
    # presence of an emoji-tagged Bear/Base/Bull row.
    has_scenario = bool(_re.search(
        r"(?:Bear|Bull)[^\n]*\|[^\n]*[Ii]mplied\s+Price"
        r"|Valuation\s+Range(?:\s+Table)?\s*:?\s*\*?\*?"
        r"|\|\s*🚀\s*Bull[^\n]*\|"
        r"|\|\s*🐻\s*Bear[^\n]*\|",
        text,
        _re.IGNORECASE,
    ))
    footnote = (
        "\n\n> _Note on Fair Value vs Scenarios: the EisaX Fair Value Estimate "
        "is a single proprietary target derived from peer-benchmarked forward "
        "earnings × sector multiple (central case). The Bear / Base / Bull "
        "scenario implied prices apply stress-test P/E multiples to the same "
        "earnings stream and act as an uncertainty band around the FV — Base "
        "anchors to current price, Bull tracks the FV ceiling under expansion, "
        "and Bear sets the downside floor. FV typically sits between Base and "
        "Bull. The two views are complementary, not redundant._\n\n"
    )
    footnote_marker = "Note on Fair Value vs Scenarios"
    if has_fv and has_scenario and footnote_marker not in text:
        # Append after the Valuation Range Table if we can locate it.
        # Use underscore italics (_..._) so we never collide with adjacent
        # bold markers (**) the LLM emits in nearby notes.
        m_tbl = _re.search(
            r"(\|\s*🚀?\s*Bull[^|\n]*\|[^\n]*\|[^\n]*\|[^\n]*\|\s*\n)",
            text,
        )
        if m_tbl:
            text = text[: m_tbl.end()] + footnote + text[m_tbl.end():]

    # ── S/R Ladder: insert SMA200 as nearest support if missing ──────────
    # The LLM sometimes lists only 52W Low as support, ignoring SMA200 which
    # is typically a much closer dynamic support. Try to read SMA200 from
    # the body text; if absent, pull from the TradingView pipeline cache so
    # the ladder is always complete.
    _sr_table_marker = _re.search(
        r"(\|\s*Level\s*\|\s*Price\s*\|\s*Type\s*\|\s*Basis\s*\|[^\n]*\n"
        r"\|[\s\-:|]+\n"  # markdown separator row
        r"(?:\|[^\n]+\n){2,12}?)"
        r"(\|\s*S1\s*\|[^|\n]+\|\s*Support\s*\|[^|\n]+\|)",
        text,
    )
    if _sr_table_marker and "SMA200" not in _sr_table_marker.group(0):
        _sma200 = None
        # 1. Try body text first
        for _pat in (r"SMA[\s_(]*200[)\s]*[^\d\.]{0,10}([\d]+\.\d{2,4})",
                     r"SMA\(200\)[^\d\.]{0,10}([\d]+\.\d{2,4})"):
            _m = _re.search(_pat, text)
            if _m:
                try:
                    _sma200 = float(_m.group(1))
                    break
                except ValueError:
                    pass
        # 2. Fallback: TV pipeline cache by detecting the ticker from the
        #    first H1 heading "EisaX Intelligence Report: <TICKER>".
        if _sma200 is None:
            try:
                _tick_m = _re.search(
                    r"#\s*EisaX\s+Intelligence\s+Report:\s*([A-Z0-9.=\-]+)", text
                )
                if _tick_m:
                    _tk_norm = _tick_m.group(1).upper()
                    # market mapping
                    _mk_map = {".AE":"uae",".DU":"uae",".SR":"ksa",".CA":"egypt",
                               ".KW":"kuwait",".QA":"qatar",".BH":"bahrain",
                               ".MA":"morocco",".TN":"tunisia","=F":"commodities"}
                    _tv_mkt = next((v for k, v in _mk_map.items() if _tk_norm.endswith(k)), "america")
                    from core.data_layer import market_cache_adapter as _mca_sr
                    _df = _mca_sr.get_latest_snapshot(_tv_mkt)
                    if _df is not None and not _df.empty:
                        _bare = _tk_norm.split(".")[0]
                        _col = _df["ticker"].astype(str).str.upper()
                        _matches = _df[
                            _col.str.endswith(":" + _bare)
                            | (_col == _tk_norm)
                            | (_col == _bare)
                        ]
                        if not _matches.empty:
                            _v = _matches.iloc[0].get("SMA200")
                            if _v is not None:
                                try:
                                    _sma200 = float(_v)
                                except (TypeError, ValueError):
                                    pass
            except Exception:
                pass

        if _sma200 and _sma200 > 0:
            # Detect currency symbol used in the table from first price cell
            _sym_m = _re.search(r"\|\s*(?:Spot|R\d|S\d)\s*\|\s*(\D{0,4})\d", _sr_table_marker.group(0))
            _sym = _sym_m.group(1).strip() if _sym_m else ""
            new_row = f"| S2 | {_sym}{_sma200:.2f} | Support | SMA200 |\n"
            # Insert the new S2 row right before the existing S1 row
            text = text.replace(
                _sr_table_marker.group(2),
                new_row + _sr_table_marker.group(2),
                1,
            )

    # ── SMA200 / SMA50 numeric reconciliation ────────────────────────────
    # The LLM occasionally hallucinates SMA50/SMA200 values that disagree
    # with the S/R ladder and with TV cache. Pull the authoritative values
    # from TV cache and rewrite any "(<currency><number>)" attached to an
    # SMA50/SMA200 mention so the body, ladder, and any stop-loss reference
    # all align. Also rebuild the canonical "Price at X is Y% above SMA200
    # and Z% below SMA50" comparison sentence with computed percentages.
    try:
        _tick_sma = _re.search(
            r"#\s*EisaX\s+Intelligence\s+Report:\s*([A-Z0-9.=\-]+)", text
        )
        if _tick_sma:
            _tk = _tick_sma.group(1).upper()
            _mk_map = {".AE":"uae",".DU":"uae",".SR":"ksa",".CA":"egypt",
                       ".KW":"kuwait",".QA":"qatar",".BH":"bahrain",
                       "=F":"commodities"}
            _tv_mkt = next((v for k, v in _mk_map.items() if _tk.endswith(k)), "america")
            from core.data_layer import market_cache_adapter as _mca_sma
            _df = _mca_sma.get_latest_snapshot(_tv_mkt)
            _row = None
            if _df is not None and not _df.empty:
                _bare = _tk.split(".")[0]
                _col = _df["ticker"].astype(str).str.upper()
                _ms = _df[_col.str.endswith(":" + _bare) | (_col == _tk) | (_col == _bare)]
                if not _ms.empty:
                    _row = _ms.iloc[0]
            if _row is not None:
                _tv_sma200 = _row.get("SMA200")
                _tv_sma50  = _row.get("SMA50")
                _tv_price  = _row.get("close")
                try:
                    _tv_sma200 = float(_tv_sma200) if _tv_sma200 is not None else None
                except (TypeError, ValueError):
                    _tv_sma200 = None
                try:
                    _tv_sma50  = float(_tv_sma50)  if _tv_sma50  is not None else None
                except (TypeError, ValueError):
                    _tv_sma50 = None
                try:
                    _tv_price  = float(_tv_price)  if _tv_price  is not None else None
                except (TypeError, ValueError):
                    _tv_price = None

                # Detect currency prefix for replacement substitution
                _cur_pref_m = _re.search(
                    r"SMA(?:200|50)\s*\(\s*(د\.إ|﷼|ج\.م|ر\.ق|\$)",
                    text,
                )
                _cur_pref = _cur_pref_m.group(1) if _cur_pref_m else ""

                # 1. Normalize SMA200 "(price)" mentions
                if _tv_sma200 and _tv_sma200 > 0:
                    text = _re.sub(
                        r"SMA\s*200\s*\(\s*(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*[\d\.,]+\s*\)",
                        f"SMA200 ({_cur_pref}{_tv_sma200:.2f})",
                        text,
                    )
                # 2. Normalize SMA50 "(price)" mentions
                if _tv_sma50 and _tv_sma50 > 0:
                    text = _re.sub(
                        r"SMA\s*50\s*\(\s*(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*[\d\.,]+\s*\)",
                        f"SMA50 ({_cur_pref}{_tv_sma50:.2f})",
                        text,
                    )
                # 3. Also normalize the S/R table SMA200 row value if it disagrees
                if _tv_sma200 and _tv_sma200 > 0:
                    text = _re.sub(
                        r"(\|\s*S\d\s*\|\s*)(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*[\d\.,]+(\s*\|\s*Support\s*\|\s*SMA200\s*\|)",
                        rf"\1{_cur_pref}{_tv_sma200:.2f}\2",
                        text,
                    )

                # 4. Rebuild the comparison sentence with correct percentages
                if _tv_sma200 and _tv_sma200 > 0 and _tv_sma50 and _tv_sma50 > 0 and _tv_price:
                    _pct200 = (_tv_price - _tv_sma200) / _tv_sma200 * 100
                    _pct50  = (_tv_price - _tv_sma50)  / _tv_sma50  * 100
                    _pos200 = "above" if _pct200 >= 0 else "below"
                    _pos50  = "above" if _pct50  >= 0 else "below"
                    _new_sent = (
                        f"Price at {_cur_pref}{_tv_price:.2f} is {abs(_pct200):.1f}% "
                        f"{_pos200} SMA200 ({_cur_pref}{_tv_sma200:.2f}) and "
                        f"{abs(_pct50):.1f}% {_pos50} SMA50 ({_cur_pref}{_tv_sma50:.2f})"
                    )
                    text = _re.sub(
                        r"Price\s+at\s+(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?[\d\.,]+\s+is\s+\d+\.\d+%\s+"
                        r"(?:above|below)\s+SMA\s*200\s*\([^)]+\)\s+and\s+\d+\.\d+%\s+"
                        r"(?:above|below)\s+SMA\s*50\s*\([^)]+\)",
                        _new_sent,
                        text,
                    )

                # 4b. Trend-line variant: "Price is above SMA200 (X) by Y%" —
                # rewrite Y% using the TV cache so the body, ladder, and
                # methodology references all agree on a single number.
                if _tv_sma200 and _tv_sma200 > 0 and _tv_price:
                    _pct = (_tv_price - _tv_sma200) / _tv_sma200 * 100
                    _pos = "above" if _pct >= 0 else "below"
                    text = _re.sub(
                        r"Price\s+is\s+(?:above|below)\s+SMA\s*200\s*\("
                        r"(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?[\d\.,]+\s*\)\s+by\s+\d+\.\d+%",
                        f"Price is {_pos} SMA200 ({_cur_pref}{_tv_sma200:.2f}) by {abs(_pct):.1f}%",
                        text,
                    )
                if _tv_sma50 and _tv_sma50 > 0 and _tv_price:
                    _pct = (_tv_price - _tv_sma50) / _tv_sma50 * 100
                    _pos = "above" if _pct >= 0 else "below"
                    text = _re.sub(
                        r"Price\s+is\s+(?:above|below)\s+SMA\s*50\s*\("
                        r"(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?[\d\.,]+\s*\)\s+by\s+\d+\.\d+%",
                        f"Price is {_pos} SMA50 ({_cur_pref}{_tv_sma50:.2f}) by {abs(_pct):.1f}%",
                        text,
                    )

                # 5. Stop-loss SMA200 reference — if any line mentions a stop
                #    "below SMA200" with a numeric level, normalize the level
                #    to the TV SMA200 value.
                if _tv_sma200 and _tv_sma200 > 0:
                    text = _re.sub(
                        r"(stop[- ]?loss[^.\n]{0,80}?(?:below|under|at)\s+(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?)[\d\.,]+"
                        r"([^.\n]{0,80}?\bSMA[\s_(]*200\b[^.\n]*)",
                        rf"\g<1>{_tv_sma200:.2f}\g<2>",
                        text,
                        flags=_re.IGNORECASE,
                    )
    except Exception:
        pass

    # ── News filter: keep only ticker/sector-relevant items ──────────────
    # The news engine sometimes returns sector/region pulls that aren't
    # related to the analysed ticker (e.g. Lulu Retail in an ADNOCGAS
    # report). Drop list items whose anchor text contains none of the
    # asset-specific keywords. We only run this filter for the Latest News
    # bullet list — leaving other sections untouched.
    # Tightened: drop the generic UAE/Emirates/DFM/ADX tokens because they
    # let through items like "UAE Lottery winner", "UK envoy to UAE", "FDI
    # destinations" which have no bearing on ADNOC Gas. Require an
    # energy-domain keyword (oil/gas/LNG/etc.) or an exact ADNOC-Gas /
    # ADNOCGAS / pipeline / Hormuz reference. Generic "ADNOC" alone (e.g.
    # "ADNOC Distribution road trip") is no longer enough.
    _news_keywords_re = _re.compile(
        r"\b(?:ADNOC\s*Gas|ADNOCGAS|"
        r"LNG|crude|Brent|OPEC|hydrocarbon|petrochemical|refinery|pipeline|"
        r"Hormuz|"
        r"oil\s+(?:price|market|disruption|sanctions?|spike|drop|fall|swing|outlook|export|production)|"
        r"gas\s+(?:price|market|production|export|supply|contract|demand)|"
        r"energy\s+(?:price|market|disruption|sanctions?|sector|policy))\b",
        _re.IGNORECASE,
    )
    # Find the Latest News block boundaries
    _news_m = _re.search(
        r"(?ms)^📰\s*\*\*Latest News\*\*[^\n]*\n(?P<body>.+?)(?=^---|\n## |\Z)",
        text,
    )
    if _news_m:
        body = _news_m.group("body")
        # Only filter the lines inside the body that are bullet-list news
        # items (start with "- [" markdown link). Section headers and notes
        # remain untouched.
        new_lines = []
        for line in body.split("\n"):
            stripped = line.strip()
            is_news_item = (
                _re.match(r"^[-*]\s*(?:⚪\s*)?\[", stripped) is not None
            )
            if is_news_item:
                # Keep only if the anchor text or surrounding line contains
                # a relevant keyword. For ADNOCGAS this is asset-specific.
                if _news_keywords_re.search(stripped):
                    new_lines.append(line)
                # else: drop
            else:
                new_lines.append(line)
        new_body = "\n".join(new_lines)
        # If filtering removed all news items in a sub-section, replace the
        # section with a one-line note. Detect by no remaining bullets.
        if not _re.search(r"^[-*]\s*(?:⚪\s*)?\[", new_body, _re.MULTILINE):
            new_body = (
                "\n_No directly relevant news items in the current feed window._\n"
            )
        text = text[: _news_m.start("body")] + new_body + text[_news_m.end("body"):]

    # ── Sector classification override ─────────────────────────────────
    # TradingView's `sector` field is wrong for several Gulf real-estate
    # developers (EMAAR, EMAARDEV, ALDAR, DAMAC, UPP, DEYAAR — tagged
    # "Finance" instead of "Real Estate"). When the report's subject
    # ticker is in the override map, we rewrite the sector mentions, the
    # regional peer table, and the risk taxonomy.
    try:
        from core.sector_overrides import (
            get_corrected_sector,
            REAL_ESTATE_PEER_UNIVERSE,
            REAL_ESTATE_RISK_TAXONOMY_EN,
            REAL_ESTATE_RISK_TAXONOMY_AR,
        )
        _tick_m = _re.search(
            r"#\s*EisaX\s+Intelligence\s+Report:\s*([A-Z0-9.=\-]+)", text
        )
        _subject = _tick_m.group(1).upper() if _tick_m else None
        _override = get_corrected_sector(_subject) if _subject else None
    except Exception:
        _override = None

    if _override:
        _corr_sector, _corr_industry = _override
        # 1. Replace "Sector:" mentions in scorecard headers
        text = _re.sub(
            r"(\*\*Sector:\*\*\s*)Finance\b",
            rf"\1{_corr_sector}",
            text,
        )
        text = _re.sub(
            r"(Sector:\s*)Finance\b",
            rf"\1{_corr_sector}",
            text,
        )
        # 2. Sector Standing line
        text = _re.sub(
            r"(its\s+)Finance(\s+sector)",
            rf"\1{_corr_sector}\2",
            text,
        )
        text = _re.sub(
            r"(Sector\s+Avg\s+\()Finance(\))",
            rf"\1{_corr_sector}\2",
            text,
        )
        # 3. Regional Peer Comparison heading + Cross-Market sector tag
        text = _re.sub(
            r"(Regional Peer Comparison[^\n]*?\*?)Finance(\s+Sector)",
            rf"\1{_corr_sector}\2",
            text,
        )
        text = _re.sub(
            r"(\*\*)Finance(\*\*\s+sector\s+RSI)",
            rf"\1{_corr_sector}\2",
            text,
        )
        text = _re.sub(
            r"(Sector\s+Rotation:\*?\*?\s*[⚖️🔥📉]?\s*\*?\*?)Finance",
            rf"\1{_corr_sector}",
            text,
        )

        # 4. Replace the (bank-laden) Regional Peer Comparison table
        #    with a curated real-estate peer table queried from TV cache.
        if _corr_sector.lower().startswith("real estate"):
            try:
                from core.data_layer import market_cache_adapter as _mca_re
                _peer_rows = []
                _market_flag = {"uae":"🇦🇪 UAE", "ksa":"🇸🇦 KSA", "qatar":"🇶🇦 Qatar"}
                _exchange_for_market = {"uae":"DFM", "ksa":"TADAWUL", "qatar":"QSE"}
                for _mkt, _bare in REAL_ESTATE_PEER_UNIVERSE:
                    _df = _mca_re.get_latest_snapshot(_mkt)
                    if _df is None or _df.empty:
                        continue
                    _col = _df["ticker"].astype(str).str.upper()
                    _m = _df[
                        _col.str.endswith(":" + _bare) | (_col == _bare)
                    ]
                    if _m.empty:
                        continue
                    _r = _m.iloc[0]
                    _close = float(_r.get("close") or 0)
                    if _close <= 0:
                        continue
                    _change = float(_r.get("change") or 0)
                    _pe_raw = _r.get("price_earnings_ttm")
                    try:
                        _pe = float(_pe_raw)
                        _pe_str = f"{_pe:.1f}x" if _pe == _pe else "—"  # NaN check
                    except (TypeError, ValueError):
                        _pe_str = "—"
                    _rsi_raw = _r.get("RSI")
                    try:
                        _rsi_str = f"{float(_rsi_raw):.1f}"
                    except (TypeError, ValueError):
                        _rsi_str = "—"
                    _dy_raw = _r.get("dividend_yield_recent")
                    try:
                        _dy = float(_dy_raw)
                        _dy_str = f"{_dy:.2f}%" if _dy and _dy > 0 else "—"
                    except (TypeError, ValueError):
                        _dy_str = "—"
                    _mc_raw = _r.get("market_cap_basic")
                    try:
                        _mc = float(_mc_raw)
                        _mc_str = f"{_mc/1e9:.1f}B"
                    except (TypeError, ValueError):
                        _mc_str = "—"
                    _arrow = "▲" if _change > 0 else "▼" if _change < 0 else "—"
                    _ticker_disp = str(_r["ticker"])
                    _name = _bare
                    _peer_rows.append(
                        (float(_mc_raw or 0), _ticker_disp, _name, _market_flag.get(_mkt, _mkt.upper()),
                         _close, _change, _arrow, _pe_str, _rsi_str, _dy_str, _mc_str)
                    )
                # Sort by market cap desc
                _peer_rows.sort(key=lambda r: r[0], reverse=True)
                if _peer_rows:
                    _hdr = (
                        f"### 🌍 Regional Peer Comparison — *{_corr_sector} Sector Across Gulf Markets*\n\n"
                        "| # | Stock | Market | Price | Change | P/E | RSI | Div Yield | Mkt Cap |\n"
                        "|---|-------|--------|-------|--------|-----|-----|-----------|---------|\n"
                    )
                    _lines = [_hdr]
                    for i, (_mc, _td, _nm, _mk, _c, _ch, _ar, _pe, _rsi, _dy, _mcs) in enumerate(_peer_rows[:12], 1):
                        _lines.append(
                            f"| {i} | **{_nm}** `{_td}` | {_mk} | {_c:.2f} | "
                            f"{_ch:+.2f}% {_ar} | {_pe} | {_rsi} | {_dy} | {_mcs} |\n"
                        )
                    _new_peer_block = "".join(_lines)
                    # Replace the existing peer block (heading + table rows
                    # through the blank line after the last table row).
                    text = _re.sub(
                        r"### 🌍 Regional Peer Comparison[^\n]*\n"
                        r"(?:\s*\n)*"          # allow blank separator lines
                        r"(?:\|[^\n]*\n)+",     # the markdown table rows
                        _new_peer_block,
                        text,
                        count=1,
                    )
            except Exception:
                pass  # leave original peer table if curation fails

        # ── Bank-peer narrative scrub (Section 6 Peer Comparison) ────
        # Real-estate sector overrides need to also remove the LLM's
        # peer-narrative paragraph that compares EMAAR to banks like
        # EMIRATESNBD / FAB. Replace it with a neutral RE-focused note.
        if _corr_sector.lower().startswith("real estate"):
            _bank_names = (
                r"EMIRATESNBD|FAB|ADCB|QNBK|ADIB|EmiratesNBD|"
                r"Saudi National Bank|Al Rajhi|First Abu Dhabi"
            )
            # Remove entire paragraph starting with "vs <bank>:" or
            # "**vs <bank> (DFM:<bank>):**" — runs to the next double
            # newline or section heading. Handles bold opener BEFORE the
            # "vs" token and inline exchange-prefixed tickers.
            text = _re.sub(
                rf"(?ms)^\*{{0,2}}vs\s+\*{{0,2}}(?:{_bank_names})\b"
                rf"[^\n]*(?:\n(?!\n|#{{2,3}}\s|---)[^\n]+)*",
                ("_(EMAAR is a real-estate developer; bank peers are excluded "
                 "as they sit in a different sector universe. See the Regional "
                 "Peer Comparison table for real-estate peers.)_"),
                text,
            )
            # Sweep up any remaining sentence that still mentions a bank
            # peer by name within Section 6 — these are stray comparison
            # clauses the first scrub may have left in adjacent paragraphs.
            _sec6_m = _re.search(
                r"(?ms)(#{2,3}\s+\d+\.?\s*(?:⚔️\s*)?Peer\s+Comparison[^\n]*\n)(.*?)(?=\n#{2,3}\s)",
                text,
            )
            if _sec6_m:
                _sec6_body = _sec6_m.group(2)
                # Drop any sentence mentioning a bank ticker by name
                _clean_body = _re.sub(
                    rf"[^.\n]*\b(?:{_bank_names})\b[^.\n]*\.",
                    "",
                    _sec6_body,
                )
                # Collapse 3+ newlines back to 2
                _clean_body = _re.sub(r"\n{3,}", "\n\n", _clean_body)
                text = text[: _sec6_m.start(2)] + _clean_body + text[_sec6_m.end(2):]

        # 5. Risk taxonomy replacement — if the report's risk section
        #    contains oil/commodity boilerplate (which makes no sense for
        #    a real-estate developer), splice in the real-estate taxonomy.
        _commodity_risk_re = _re.compile(
            r"\b(?:oil[- ]price\s+sensitivity|cyclical\s+commodity\s+exposure|"
            r"OPEC|Brent|crude\s+oil|hydrocarbon)\b",
            _re.IGNORECASE,
        )
        # For real-estate tickers, strip any remaining oil/Brent/commodity
        # sentences in the narrative (they leak from boilerplate triggers).
        if _corr_sector.lower().startswith("real estate"):
            text = _re.sub(
                r"(?m)^[\-\*]?\s*\*?\*?(?:Brent\s+crude\s+spikes?|"
                r"Oil\s+price\s+sensitivity|Cyclical\s+commodity\s+exposure|"
                r"OPEC[- ]?\+?\s+(?:supply|production))[^\n]+\n?",
                "",
                text,
                flags=_re.IGNORECASE,
            )
        # Inject the RE risk taxonomy whenever the subject is a real-estate
        # ticker — independent of whether commodity boilerplate survived,
        # since the body may have been scrubbed earlier in this pass.
        if _corr_sector.lower().startswith("real estate"):
            # Detect language: prefer explicit Arabic section headings
            # (e.g. "## التحليل الأساسي"), not stray Arabic terms that may
            # leak into a primarily-English report.
            _is_ar = bool(_re.search(
                r"#{1,3}\s*[؀-ۿ]{3,}.*\n",  # H1-H3 heading in Arabic
                text,
            ))
            _taxonomy = REAL_ESTATE_RISK_TAXONOMY_AR if _is_ar else REAL_ESTATE_RISK_TAXONOMY_EN
            # Inject before the Key Risks heading body — keep heading,
            # replace the body. Match H2 (## 4. Key Risks) or H3 (### Key Risks).
            _risk_m = _re.search(
                r"(#{2,3}\s+\d+\.?\s*(?:⚠️\s*)?(?:Key\s+Risks?|Risks?|المخاطر[^\n]*)[^\n]*\n)"
                r"(.*?)"
                r"(?=\n#{2,3}\s|\Z)",
                text, _re.S,
            )
            if _risk_m:
                _hdr_line = _risk_m.group(1)
                text = text[:_risk_m.start(2)] + "\n" + _taxonomy + "\n" + text[_risk_m.end(2):]

        # ── Real-Estate final polish (last pass before currency fix) ──────
        # Covers leftovers that earlier scrubs may miss because they live
        # outside the Key Risks section: scorecard one-liners, executive
        # memo, peer-comparison prose, valuation methodology footers.
        if _corr_sector.lower().startswith("real estate"):
            # (A) Strip any remaining commodity / oil-cycle phrases anywhere
            #     in the report (memo, scorecard, summary, top-risk row).
            # Catch-all: for an RE ticker, ANY sentence mentioning Brent,
            # OPEC, crude-oil, or hydrocarbon is off-thesis and should be
            # scrubbed. The more-specific phrases below pick up bullet-only
            # variants ("Cyclical commodity exposure" with no surrounding
            # sentence, etc.).
            _re_commodity_phrases = [
                r"cyclical\s+commodity\s+exposure",
                r"commodity\s+price\s+cycle",
                r"oil[- ]price\s+sensitivity",
                r"oil\s+price\s+(?:decline|drop|fall|swing|move|spike)",
                r"hydrocarbon\s+(?:price\s+)?exposure",
                r"crude[- ]oil\s+(?:linked|exposure|sensitivity)",
                r"OPEC[+\- ]?\s*(?:supply|production|cuts?|decisions?)",
                r"\bOPEC\b",
                r"\bBrent\b",                    # any standalone Brent ref
                r"\bcrude\s+oil\b",
                r"\bhydrocarbon\b",
                r"energy[- ]price\s+(?:volatility|sensitivity)",
                r"correlation\s+to\s+oil\s+price",
                r"lower\s+crude\s+pressures",
            ]
            for _p in _re_commodity_phrases:
                # Sentence-level scrub
                text = _re.sub(
                    rf"[^.\n]*\b{_p}\b[^.\n]*\.\s*",
                    "",
                    text,
                    flags=_re.IGNORECASE,
                )
                # Bullet-level scrub (line starts with -/* and contains phrase)
                text = _re.sub(
                    rf"(?m)^[\-\*]\s*\*?\*?[^\n]*?\b{_p}\b[^\n]*\n",
                    "",
                    text,
                    flags=_re.IGNORECASE,
                )
                # Table-row scrub (pipes around a row mentioning the phrase)
                text = _re.sub(
                    rf"(?m)^\|[^\n]*\b{_p}\b[^\n]*\|\s*\n",
                    "",
                    text,
                    flags=_re.IGNORECASE,
                )

            # (B) Sector-average wording — RE peers, not Finance sector.
            text = _re.sub(
                r"\bFinance\s+sector\s+average\b",
                "Real-Estate peer average",
                text,
                flags=_re.IGNORECASE,
            )
            text = _re.sub(
                r"\b(?:the\s+)?sector[- ]average\s+P/E\b",
                "real-estate peer-average P/E",
                text,
                flags=_re.IGNORECASE,
            )
            text = _re.sub(
                r"\bsector\s+P/E\s+average\b",
                "real-estate peer P/E average",
                text,
                flags=_re.IGNORECASE,
            )
            text = _re.sub(
                r"\b(?:vs\.?|versus)\s+(?:the\s+)?Finance\s+sector\b",
                "vs. Real-Estate peers",
                text,
                flags=_re.IGNORECASE,
            )

            # (C) Valuation methodology unification.
            # If the narrative never anchors Forward EPS to a real source
            # (consensus, guidance, FY-tag, or 4-digit year), rewrite the
            # methodology label to TTM EPS × real-estate peer multiple so
            # the report doesn't imply forward-looking data we don't have.
            _has_real_fwd_eps = bool(_re.search(
                r"Forward\s+EPS[^.\n]{0,200}\b(?:consensus|guidance|FY\d{2,4}|20\d{2}|analyst[s]?\s+estimate)\b",
                text,
                _re.IGNORECASE,
            ))
            if not _has_real_fwd_eps:
                # Rewrite the FV-vs-Scenario disambiguation footnote that was
                # appended earlier in this function (uses "Forward EPS × sector-average P/E").
                text = text.replace(
                    "uses Forward EPS × sector-average P/E",
                    "uses TTM EPS × real-estate peer-average P/E",
                )
                # Drop any other "Forward EPS × Nx sector P/E" methodology line
                text = _re.sub(
                    r"\bForward\s+EPS\s*[×x]\s*\d+\s*x?\s+sector[- ]?(?:average\s+)?P/E\b",
                    "TTM EPS × real-estate peer-average P/E",
                    text,
                    flags=_re.IGNORECASE,
                )
                # Strip methodology-only Forward-EPS clauses ("based on Forward EPS …")
                text = _re.sub(
                    r"\b(?:based\s+on|using|applying)\s+Forward\s+EPS[^.\n]{0,120}\.",
                    "based on TTM EPS × real-estate peer-average P/E.",
                    text,
                    flags=_re.IGNORECASE,
                )

            # (D) SMA200 guard — if no S/R row was injected with SMA200
            #     basis (meaning TV cache had no value), strip any prose
            #     that asserts "price is below/above SMA200" because the
            #     level itself is unavailable to verify.
            _has_sma200_in_table = bool(_re.search(
                r"\|\s*S\d\s*\|[^|]*\|\s*Support\s*\|\s*SMA200\s*\|",
                text,
                _re.IGNORECASE,
            ))
            if not _has_sma200_in_table:
                text = _re.sub(
                    r"[^.\n]*\b(?:trading\s+|currently\s+|sits\s+|is\s+)?"
                    r"(?:below|above|near|crossing|under|over)\s+(?:its\s+|the\s+)?SMA[\s_(]*200[)\s]*[^.\n]*\.\s*",
                    "",
                    text,
                    flags=_re.IGNORECASE,
                )
                text = _re.sub(
                    r"[^.\n]*\b(?:trading\s+|currently\s+|sits\s+|is\s+)?"
                    r"(?:below|above|near|crossing|under|over)\s+(?:its\s+|the\s+)?200[- ]?day\s+(?:moving\s+average|MA|SMA)[^.\n]*\.\s*",
                    "",
                    text,
                    flags=_re.IGNORECASE,
                )
                # Also drop standalone bullets that only carry an SMA200 claim
                text = _re.sub(
                    r"(?m)^[\-\*]\s*\*?\*?[^\n]*?\bSMA[\s_(]*200[)\s]*[^\n]*\n",
                    "",
                    text,
                    flags=_re.IGNORECASE,
                )

            # (E) Tidy any blank-line cascades the scrubs leave behind
            text = _re.sub(r"\n{3,}", "\n\n", text)

    # ── Currency fix in fact-check ────────────────────────────────────
    # For local-market tickers (.AE/.DU/.SR/.CA/.KW/.QA), force the
    # fact-check Live Price row to render in the local currency. The
    # LLM sometimes emits "$11.78" when the surrounding report already
    # uses "د.إ" / "ر.س" / etc.
    try:
        _tick_m2 = _re.search(
            r"#\s*EisaX\s+Intelligence\s+Report:\s*([A-Z0-9.=\-]+)", text
        )
        _t = (_tick_m2.group(1).upper() if _tick_m2 else "")
        _cur_sym = None
        if _t.endswith((".AE", ".DU")):           _cur_sym = "د.إ"
        elif _t.endswith(".SR"):                   _cur_sym = "﷼"
        elif _t.endswith(".CA"):                   _cur_sym = "ج.م"
        elif _t.endswith(".KW"):                   _cur_sym = "ف"
        elif _t.endswith(".QA"):                   _cur_sym = "ر.ق"
        # Also catch bare symbols where the report already uses the
        # local symbol elsewhere
        if _cur_sym is None:
            for _sym_probe in ("د.إ", "﷼", "ج.م", "ف", "ر.ق"):
                if _sym_probe in text:
                    _cur_sym = _sym_probe
                    break
        if _cur_sym and _cur_sym != "$":
            # Replace "$X.YY" in the Live Price row of the fact-check table
            text = _re.sub(
                r"(\|\s*Live\s+Price\s*\|\s*)\$([\d,]+\.\d+)",
                rf"\1\2 {_cur_sym}",
                text,
            )
            text = _re.sub(
                r"(Live Price[:\s\|]+\*{0,2}\s*)\$([\d,]+\.\d+)(?!\s*(?:د\.إ|﷼|ج\.م|ر\.ق))",
                rf"\1\2 {_cur_sym}",
                text,
            )
    except Exception:
        pass

    return text


def _sanitize_guest_report_json(report_json: Optional[dict]) -> Optional[dict]:
    if not isinstance(report_json, dict):
        return report_json
    safe = _copy.deepcopy(report_json)
    system = safe.setdefault("system", {})
    system["environment"] = "demo"
    system["model_primary"] = "EisaX Intelligence Engine"
    compliance = safe.setdefault("compliance", {})
    compliance["pilot_status"] = "demo"
    return safe


def _sanitize_guest_resolution(resolution: Optional[dict]) -> Optional[dict]:
    if not isinstance(resolution, dict):
        return resolution
    return _public_resolution_payload(resolution)


def _replace_score(match, score: int) -> str:
    return f"{match.group(1)}{score}{match.group(2)}"


def _apply_report_meta_to_text(text: str, report_json: Optional[dict]) -> str:
    if not text or not isinstance(report_json, dict):
        return text
    meta = report_json.get("report_meta") or {}
    if not isinstance(meta, dict) or not meta:
        return text

    eisax_score = meta.get("eisax_score")
    blended_score = meta.get("blended_score")
    fundamental_score = meta.get("fundamental_quality_score")
    confidence_score = meta.get("confidence_score")
    conviction_score = meta.get("conviction_score")
    conviction_label = meta.get("conviction_label")
    overall_risk_label = meta.get("overall_risk_label") or meta.get("overall_risk_level")
    market_beta_risk = meta.get("market_beta_risk")
    risk_drivers = meta.get("risk_drivers") or []
    primary_driver = risk_drivers[0] if risk_drivers else "the listed risk drivers"

    text = _re.sub(r"\bFinal Technical Signal\b", "Technical Signal (Supporting)", text, flags=_re.IGNORECASE)
    text = _re.sub(r"fundamental data is largely unavailable", "fundamental data coverage is limited", text, flags=_re.IGNORECASE)
    text = _re.sub(r"data is unavailable", "data coverage is partial", text, flags=_re.IGNORECASE)

    if isinstance(eisax_score, int):
        text = _re.sub(
            r"(EisaX Score:\s*\*{0,2})\d+(/100\*{0,2})",
            lambda m: _replace_score(m, eisax_score),
            text, flags=_re.IGNORECASE,
        )
    if isinstance(blended_score, int):
        text = _re.sub(
            r"(Blended:\s*\*{0,2})\d+(/100\*{0,2})",
            lambda m: _replace_score(m, blended_score),
            text, flags=_re.IGNORECASE,
        )
    if isinstance(fundamental_score, int) and isinstance(eisax_score, int):
        text = _re.sub(
            r"(?mi)^(\s*(?:[-*]\s*)?)Score:\s*\*{0,2}\d+/100\*{0,2}\s*$",
            lambda m: (
                f"{m.group(1)}Fundamental Quality Score: {fundamental_score}/100 "
                f"(EisaX Score: {eisax_score}/100)"
            ),
            text,
        )

    if isinstance(confidence_score, int) and isinstance(conviction_score, int):
        conviction_text = f"{conviction_label}, {conviction_score}%" if conviction_label else f"{conviction_score}%"
        text = _re.sub(
            r"Verdict Confidence:\s*\*{0,2}\d+%\*{0,2}\s*(?:\(|\|)?\s*Conviction:\s*\*{0,2}(?:\d+%|Low|Medium|High)\*{0,2}\)?",
            f"Verdict Confidence: {confidence_score}% (Conviction: {conviction_text})",
            text, flags=_re.IGNORECASE,
        )

    if overall_risk_label:
        text = _re.sub(
            r"\blow risk profile\b",
            f"low market-beta risk profile; overall risk remains {overall_risk_label}",
            text, flags=_re.IGNORECASE,
        )

    if overall_risk_label and market_beta_risk:
        def _risk_profile_row(match):
            row = match.group(0)
            pct = _re.search(r"(\d{1,3})%\s*Risk", row, flags=_re.IGNORECASE)
            pct_text = f"{pct.group(1)}% Risk" if pct else f"{str(market_beta_risk).title()} market-beta risk"
            return (
                f"| Market/Beta Risk | {pct_text} | "
                f"{str(market_beta_risk).title()} market-beta risk; overall risk is "
                f"{overall_risk_label} due to {primary_driver}. |"
            )
        text = _re.sub(r"(?mi)^\|\s*Risk Profile\s*\|[^\n]+$", _risk_profile_row, text)

    return text


def _sanitize_guest_report_text(text: str, report_json: Optional[dict]) -> str:
    text = _cleanup_report_text_artifacts(text)
    text = _sanitize_trial_report(text)
    text = _apply_report_meta_to_text(text, report_json)
    text = _re.sub(r"\bDeepSeek(?:\s+V3)?\b", "EisaX Intelligence Engine", text, flags=_re.IGNORECASE)
    text = _re.sub(r"\b(?:OpenAI|Gemini|GLM)\b", "EisaX Intelligence Engine", text, flags=_re.IGNORECASE)
    text = _re.sub(r"/home/ubuntu/\S+", "[internal path removed]", text)
    return text


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
    report_id: Optional[str] = None,
    restrict_for_trial: bool = False,
    access_context: Optional[dict] = None,
    analysis_data: Optional[dict] = None,   # NEW: live_payload data for SSOT reconciler
    subject_ticker: Optional[str] = None,    # NEW: ticker for FactSheet construction
) -> dict:
    is_guest = _is_guest_access(access_context, restrict_for_trial=restrict_for_trial)
    report_text = _cleanup_report_text_artifacts(report_text)
    if html_report:
        html_report = _cleanup_report_text_artifacts(html_report)
    if isinstance(report_json, dict):
        report_text = _apply_report_meta_to_text(report_text, report_json)
        if html_report:
            html_report = _apply_report_meta_to_text(html_report, report_json)
        # _apply_report_meta_to_text re-injects fields from report_json
        # (Risk Profile row using primary_driver, etc) which can re-introduce
        # phrases the first cleanup pass already scrubbed. Re-run cleanup so
        # the meta-injection passes through the same filters. Idempotent.
        report_text = _cleanup_report_text_artifacts(report_text)
        if html_report:
            html_report = _cleanup_report_text_artifacts(html_report)

    # ── SSOT Reconciler — Phase C integration ───────────────────────────
    # FactSheet + Reconciler runs AFTER all legacy cleanup. It owns the
    # final consistency check (price, SMA200, verdict, score, news,
    # currency, sector-aware scrubs). Legacy regex patches above remain
    # as a safety net until we've validated the reconciler on enough
    # tickers — they're idempotent so no harm.
    _ssot_fs = None  # hoisted so payload builder can read SSOT verdict below
    try:
        _ssot_ticker = subject_ticker
        if not _ssot_ticker and report_text:
            _h1 = _re.search(
                r"#\s*EisaX\s+Intelligence\s+Report:\s*([A-Z0-9.=\-]+)",
                report_text,
            )
            if _h1:
                _ssot_ticker = _h1.group(1).upper()
        if _ssot_ticker and report_kind in ("stock", "asset", ""):
            from core.services.fact_sheet import build_fact_sheet
            from core.services.report_reconciler import reconcile_report
            _live_payload = {
                "data": analysis_data or {},
                "report_json": report_json or {},
            }
            _ssot_fs = build_fact_sheet(_ssot_ticker, live_payload=_live_payload)
            if _ssot_fs.blocking_errors:
                logger.error(
                    "[SSOT] %s blocked by FactSheet errors: %s",
                    _ssot_ticker, _ssot_fs.blocking_errors,
                )
                # Don't block production yet — log only until enough samples validated
            report_text, _audit = reconcile_report(report_text, _ssot_fs)
            if html_report:
                html_report, _html_audit = reconcile_report(html_report, _ssot_fs)
            logger.info(
                "[SSOT] %s reconciler %s",
                _ssot_ticker, _audit.summary(),
            )
            for c in _audit.corrections[:10]:
                logger.info(
                    "[SSOT-correction] %s %s: %s → %s (%s)",
                    _ssot_ticker, c.field_name, c.old, c.new, c.rule,
                )
            for w in _audit.warnings[:5]:
                logger.warning("[SSOT-warn] %s %s", _ssot_ticker, w)
    except Exception as _ssot_err:
        logger.warning(
            "[SSOT] reconciler failed (non-fatal) for %s: %s",
            subject_ticker or "?", _ssot_err,
        )
    if is_guest:
        report_text = _sanitize_guest_report_text(report_text, report_json)
        if html_report:
            html_report = _sanitize_guest_report_text(html_report, report_json)
        report_json = _sanitize_guest_report_json(report_json)
    summary = _staging_extract_summary(report_text, report_kind=report_kind, result_type=result_type)
    confidence = _staging_extract_confidence(report_text)
    _clean = html_report or report_text
    _meta = report_json.get("report_meta") if isinstance(report_json, dict) else {}
    _meta = _meta if isinstance(_meta, dict) else {}
    risk_level = _meta.get("overall_risk_label") or _staging_extract_risk_level(report_text)
    confidence = _meta.get("confidence_label") or confidence
    payload = {
        "ok": True,
        "mode": mode,
        "query": query,
        "report_kind": report_kind,
        "result_type": result_type or report_kind,
        "summary": summary,
        # SSOT verdict wins when FactSheet is available; legacy extractor is fallback only
        "verdict": (
            _ssot_fs.verdict
            if _ssot_fs and _ssot_fs.verdict
            else _staging_extract_verdict(report_text, report_kind=report_kind, result_type=result_type)
        ),
        "risk_level": risk_level,
        "confidence": confidence,
        "insights": _staging_extract_insights(report_text, report_kind=report_kind, summary=summary, result_type=result_type),
        "download_url": None if is_guest else download_url,
        "teaser": (report_text[:500] + "...") if len(report_text) > 500 else report_text,
        "full_report": _clean,
        "rule_based_clean_report": _clean,
        "report_json": report_json,
        "report_id": report_id,
        "access": {
            "role": "guest" if is_guest else (access_context or {}).get("role", "admin"),
            "username": (access_context or {}).get("username") if is_guest else None,
            "demo": bool(is_guest),
            "analysis_limit": (access_context or {}).get("analysis_limit") if is_guest else None,
            "analyses_remaining": (access_context or {}).get("analyses_remaining") if is_guest else None,
            "downloads_enabled": not is_guest,
        },
    }
    if resolution:
        payload["resolution"] = _sanitize_guest_resolution(resolution) if is_guest else resolution
    return payload


def _staging_fallback_payload(query: str, file_name: Optional[str] = None, access_context: Optional[dict] = None) -> dict:
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
        access_context=access_context,
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


# ── Guest trial helpers ───────────────────────────────────────────────────────
def _guest_trial_status(username: str) -> dict:
    import sqlite3 as _sq
    _DB = "/home/ubuntu/investwise/investwise.db"
    username = (username or "guest").strip()[:80]
    try:
        with _sq.connect(_DB) as _c:
            _c.row_factory = _sq.Row
            row = _c.execute(
                "SELECT analyses_used, max_analyses, portfolios_used, max_portfolios "
                "FROM guest_trial WHERE username = ?",
                (username,),
            ).fetchone()
            if not row:
                return {}
            return {
                "analysis_limit": int(row["max_analyses"]),
                "analyses_remaining": max(0, int(row["max_analyses"]) - int(row["analyses_used"])),
                "portfolio_limit": int(row["max_portfolios"]),
                "portfolios_remaining": max(0, int(row["max_portfolios"]) - int(row["portfolios_used"])),
            }
    except Exception as _e:
        logger.warning("[guest_trial] status failed: %s", _e)
        return {}


def _guest_trial_increment_success(username: str, is_portfolio: bool) -> dict:
    import sqlite3 as _sq
    _DB = "/home/ubuntu/investwise/investwise.db"
    username = (username or "guest").strip()[:80]
    field = "portfolios_used" if is_portfolio else "analyses_used"
    limit_field = "max_portfolios" if is_portfolio else "max_analyses"
    try:
        with _sq.connect(_DB) as _c:
            _c.execute(
                f"UPDATE guest_trial SET {field} = {field} + 1, last_used = datetime('now') "
                f"WHERE username = ? AND {field} < {limit_field}",
                (username,),
            )
            _c.commit()
    except Exception as _e:
        logger.warning("[guest_trial] increment failed: %s", _e)
    return _guest_trial_status(username)


def _guest_trial_check(username: str, is_portfolio: bool) -> tuple[bool, str]:
    import sqlite3 as _sq
    _DB = "/home/ubuntu/investwise/investwise.db"
    username = (username or "guest").strip()[:80]
    default_analyses = max(0, _safe_env_int("EISAX_GUEST_MAX_ANALYSES", 10))
    default_portfolios = max(0, _safe_env_int("EISAX_GUEST_MAX_PORTFOLIOS", 3))
    try:
        with _sq.connect(_DB) as _c:
            _c.row_factory = _sq.Row
            row = _c.execute(
                "SELECT * FROM guest_trial WHERE username = ?", (username,)
            ).fetchone()
            if row is None:
                _c.execute(
                    "INSERT INTO guest_trial "
                    "(username, analyses_used, portfolios_used, max_analyses, max_portfolios, created_at, last_used) "
                    "VALUES (?, 0, 0, ?, ?, datetime('now'), NULL)",
                    (username, default_analyses, default_portfolios),
                )
                _c.commit()
                row = _c.execute(
                    "SELECT * FROM guest_trial WHERE username = ?", (username,)
                ).fetchone()
            if is_portfolio:
                if row["portfolios_used"] >= row["max_portfolios"]:
                    return False, _GUEST_LIMIT_MESSAGE
                return True, ""
            if row["analyses_used"] >= row["max_analyses"]:
                return False, _GUEST_LIMIT_MESSAGE
            return True, ""
    except Exception as _e:
        logger.warning("[guest_trial] check failed: %s", _e)
        return True, ""   # on DB error, allow rather than block


# ── Guest admin helpers ───────────────────────────────────────────────────────
def _guest_access_source_ip(request: Request) -> str:
    forwarded = (request.headers.get("X-Forwarded-For") or "").split(",", 1)[0].strip()
    if forwarded:
        return forwarded[:80]
    return (getattr(request.client, "host", "") if request.client else "")[:80]


def _require_same_origin_admin_mutation(request: Request) -> None:
    from urllib.parse import urlparse as _urlparse
    host = (request.headers.get("host") or "").lower()
    if not host:
        raise HTTPException(status_code=403, detail="Forbidden")
    for header_name in ("origin", "referer"):
        value = (request.headers.get(header_name) or "").strip()
        if not value:
            continue
        parsed = _urlparse(value)
        origin_host = (parsed.netloc or "").lower()
        if origin_host and origin_host != host:
            raise HTTPException(status_code=403, detail="Forbidden")


def _require_guest_access_admin(request: Request, *, mutation: bool = False) -> dict:
    access = _resolve_staging_access(request)
    if (
        access.get("role") != "admin"
        or access.get("method") != "basic_auth"
        or (access.get("username") or "").lower() not in _STAGING_ADMIN_USERS
    ):
        raise HTTPException(status_code=403, detail="Admins only")
    if mutation:
        _require_same_origin_admin_mutation(request)
    return access


def _audit_guest_access_admin(request: Request, action: str, target_username: str, **extra) -> None:
    access = _resolve_staging_access(request)
    safe_extra = {key: value for key, value in extra.items() if key not in {"password", "token"}}
    logger.info(
        "[guest_access_admin] action=%s admin=%s target=%s ip=%s extra=%s",
        action,
        access.get("username") or "unknown",
        target_username,
        _guest_access_source_ip(request),
        safe_extra,
    )


def _safe_guest_username(username: str) -> str:
    username = (username or "").strip()
    if not _re.fullmatch(r"[A-Za-z0-9._@-]{2,80}", username):
        raise HTTPException(status_code=400, detail="Username must be 2-80 chars: letters, numbers, dot, dash, underscore, or @")
    return username


def _read_htpasswd_entries() -> dict[str, str]:
    if not _HTPASSWD_PATH.exists():
        return {}
    entries: dict[str, str] = {}
    for line in _HTPASSWD_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip() or ":" not in line:
            continue
        username, password_hash = line.split(":", 1)
        if username:
            entries[username] = password_hash
    return entries


def _write_htpasswd_entries(entries: dict[str, str]) -> None:
    body = "".join(f"{username}:{password_hash}\n" for username, password_hash in sorted(entries.items()))
    try:
        _HTPASSWD_PATH.write_text(body, encoding="utf-8")
    except PermissionError as exc:
        raise HTTPException(
            status_code=500,
            detail="Credential store is not writable. Ask ops to check admin access configuration.",
        ) from exc


def _hash_basic_auth_password(password: str) -> str:
    password = password or ""
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    try:
        result = _subprocess.run(
            ["openssl", "passwd", "-apr1", "-stdin"],
            input=password,
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
        password_hash = result.stdout.strip()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Password hashing is unavailable") from exc
    if not password_hash:
        raise HTTPException(status_code=500, detail="Password hashing failed")
    return password_hash


def _ensure_guest_trial_table() -> None:
    import sqlite3 as _sq
    with _sq.connect("/home/ubuntu/investwise/investwise.db") as _c:
        _c.execute(
            """
            CREATE TABLE IF NOT EXISTS guest_trial (
                username TEXT PRIMARY KEY,
                analyses_used INTEGER NOT NULL DEFAULT 0,
                portfolios_used INTEGER NOT NULL DEFAULT 0,
                max_analyses INTEGER NOT NULL DEFAULT 5,
                max_portfolios INTEGER NOT NULL DEFAULT 3,
                created_at TEXT NOT NULL,
                last_used TEXT
            )
            """
        )
        _c.commit()


def _upsert_guest_quota(
    username: str,
    *,
    max_analyses: int,
    max_portfolios: int,
    analyses_used: Optional[int] = None,
    portfolios_used: Optional[int] = None,
) -> dict:
    import sqlite3 as _sq
    username = _safe_guest_username(username)
    max_analyses = max(0, min(int(max_analyses), 10000))
    max_portfolios = max(0, min(int(max_portfolios), 10000))
    safe_analyses_used = None if analyses_used is None else max(0, min(int(analyses_used), max_analyses))
    safe_portfolios_used = None if portfolios_used is None else max(0, min(int(portfolios_used), max_portfolios))
    _ensure_guest_trial_table()
    with _sq.connect("/home/ubuntu/investwise/investwise.db") as _c:
        _c.row_factory = _sq.Row
        row = _c.execute("SELECT * FROM guest_trial WHERE username = ?", (username,)).fetchone()
        if row is None:
            _c.execute(
                "INSERT INTO guest_trial "
                "(username, analyses_used, portfolios_used, max_analyses, max_portfolios, created_at, last_used) "
                "VALUES (?, ?, ?, ?, ?, datetime('now'), NULL)",
                (
                    username,
                    safe_analyses_used or 0,
                    safe_portfolios_used or 0,
                    max_analyses,
                    max_portfolios,
                ),
            )
        else:
            updates = {
                "max_analyses": max_analyses,
                "max_portfolios": max_portfolios,
            }
            if safe_analyses_used is not None:
                updates["analyses_used"] = safe_analyses_used
            if safe_portfolios_used is not None:
                updates["portfolios_used"] = safe_portfolios_used
            parts = ", ".join(f"{key} = ?" for key in updates)
            _c.execute(f"UPDATE guest_trial SET {parts} WHERE username = ?", [*updates.values(), username])
        _c.commit()
        row = _c.execute("SELECT * FROM guest_trial WHERE username = ?", (username,)).fetchone()
        return dict(row)


def _list_guest_access_users() -> list[dict]:
    import sqlite3 as _sq
    _ensure_guest_trial_table()
    htpasswd = _read_htpasswd_entries()
    quota_rows: dict[str, dict] = {}
    with _sq.connect("/home/ubuntu/investwise/investwise.db") as _c:
        _c.row_factory = _sq.Row
        for row in _c.execute("SELECT * FROM guest_trial ORDER BY username").fetchall():
            quota_rows[row["username"]] = dict(row)
    usernames = sorted(set(htpasswd) | set(quota_rows), key=lambda value: value.lower())
    users: list[dict] = []
    for username in usernames:
        role = "admin" if username.lower() in _STAGING_ADMIN_USERS else "guest"
        quota = quota_rows.get(username) or {}
        max_analyses = int(quota.get("max_analyses") or 0)
        analyses_used = int(quota.get("analyses_used") or 0)
        max_portfolios = int(quota.get("max_portfolios") or 0)
        portfolios_used = int(quota.get("portfolios_used") or 0)
        users.append(
            {
                "username": username,
                "display_name": username,
                "role": role,
                "active": username in htpasswd,
                "analysis_limit": max_analyses,
                "analyses_used": analyses_used,
                "analyses_remaining": max(0, max_analyses - analyses_used),
                "portfolio_limit": max_portfolios,
                "portfolios_used": portfolios_used,
                "portfolios_remaining": max(0, max_portfolios - portfolios_used),
                "created_at": quota.get("created_at"),
                "last_used": quota.get("last_used"),
                "editable": role == "guest",
            }
        )
    return users


def _guest_access_user_payload(username: str) -> dict:
    username = _safe_guest_username(username)
    for user in _list_guest_access_users():
        if user["username"].lower() == username.lower():
            return user
    raise HTTPException(status_code=404, detail="Guest user not found")


# ── Routers ───────────────────────────────────────────────────────────────────
staging_router = APIRouter(prefix="/staging-api", tags=["staging"])
guest_admin_router = APIRouter(prefix="/v1/admin", tags=["admin"])


# ═══════════════════════════════════════════════════════════════════════════════
# Staging routes
# ═══════════════════════════════════════════════════════════════════════════════

@staging_router.get("/health")
@limiter.limit("30/minute")
async def staging_public_health(request: Request):
    _ensure_staging_public_enabled()
    return {"status": "ok"}


@staging_router.get("/updates")
@limiter.limit("60/minute")
async def api_updates_latest(request: Request):
    """Return latest daily + weekly updates with LinkedIn-formatted text."""
    from core.services.market_updates import get_latest_updates
    result = get_latest_updates()
    if not result:
        return JSONResponse({"daily": None, "weekly": None}, status_code=200)
    return result


@staging_router.get("/updates/daily")
@limiter.limit("60/minute")
async def api_updates_daily(request: Request):
    from core.services.market_updates import get_latest_updates
    result = get_latest_updates()
    daily = result.get("daily")
    if not daily:
        return JSONResponse({"detail": "No daily update available yet"}, status_code=404)
    return daily


@staging_router.get("/updates/weekly")
@limiter.limit("60/minute")
async def api_updates_weekly(request: Request):
    from core.services.market_updates import get_latest_updates
    result = get_latest_updates()
    weekly = result.get("weekly")
    if not weekly:
        return JSONResponse({"detail": "No weekly update available yet"}, status_code=404)
    return weekly


@staging_router.post("/updates/generate")
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


@staging_router.post("/analyze")
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

    access_context = _resolve_staging_access(request)
    _log_username = access_context.get("username", "unknown")
    _log_filename = file.filename if file else None
    logger.info(
        "[activity-log] user=%s query='%s' symbol='%s' file=%s ip=%s",
        _log_username,
        normalized_query[:200] if normalized_query else "",
        selected_symbol,
        _log_filename or "none",
        request.headers.get("X-Real-IP") or request.client.host,
    )

    intent = _classify_request_intent(normalized_query, has_file=file is not None)
    if not normalized_query and not file and not selected_symbol:
        raise HTTPException(status_code=400, detail="Enter a request or upload a portfolio file.")

    _trial_restrict = access_context.get("role") == "guest"
    user_id = (
        f"guest:{access_context.get('username', 'guest')}"
        if _trial_restrict
        else _staging_client_id(request)
    )

    if _trial_restrict:
        _is_portfolio = file is not None
        _allowed, _limit_msg = _guest_trial_check(access_context.get("username", "guest"), _is_portfolio)
        if not _allowed:
            raise HTTPException(status_code=429, detail=_limit_msg)
        access_context.update(_guest_trial_status(access_context.get("username", "guest")))

    def _shape_public_result(*, count_success: bool = False, is_portfolio_run: bool = False, **kwargs) -> dict:
        payload = _staging_shape_result(
            restrict_for_trial=_trial_restrict,
            access_context=access_context,
            **kwargs,
        )
        if _trial_restrict and count_success and payload.get("ok") and payload.get("mode") == "live":
            status = _guest_trial_increment_success(access_context.get("username", "guest"), is_portfolio_run)
            access_context.update(status)
            if isinstance(payload.get("access"), dict):
                payload["access"].update(
                    {
                        "analysis_limit": status.get("analysis_limit"),
                        "analyses_remaining": status.get("analyses_remaining"),
                    }
                )
        return payload

    # Import orchestrator lazily to avoid circular import at module level
    from api_bridge_v2 import orchestrator

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
            return _shape_public_result(
                count_success=True,
                is_portfolio_run=True,
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
                    },
                    guest_visible=_trial_restrict,
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
                return _shape_public_result(
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
            report_text = await run_in_threadpool(_detect_and_build, normalized_query, preferred_language)
            if not report_text:
                raise HTTPException(status_code=502, detail="Portfolio builder returned no content.")
            return _shape_public_result(
                count_success=True,
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
                return _shape_public_result(
                    count_success=True,
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
            _idx_resolution = None
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
            return _resolution_error_response(resolution_payload, guest_visible=_trial_restrict)

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
                # Propagate the user's preferred report language so the
                # FinancialAgent's prompt builder produces an Arabic
                # report when the dropdown is set to AR. Without this,
                # the analysis pipeline always falls back to English
                # because language detection on a normalized "analyze
                # 2222.SR" instruction always reads as English.
                "language": preferred_language,
                "report_language": preferred_language,
            },
        )
        report_text = live_payload.get("reply") or ""
        if not report_text:
            raise HTTPException(status_code=502, detail="Analysis returned no content.")

        # ── Phase G — Institutional single-asset parity wrapper ─────────────
        # Augments the LLM report with Section A (metadata + confidence + implementation)
        # and Section G (audit appendix + model constraints). Non-invasive — preserves
        # original report body, only prepends/appends + polishes retail emoji headers.
        # Toggle via INSTITUTIONAL_STOCK_WRAPPER=0 to disable.
        try:
            if os.getenv("INSTITUTIONAL_STOCK_WRAPPER", "1") not in ("0", "false", "False"):
                from core.services.institutional_stock_wrapper import wrap_stock_report
                _live_data = live_payload.get("data") or {}
                _fund_data = (_live_data.get("fundamentals") if isinstance(_live_data, dict) else {}) or {}
                _tech_data = (_live_data.get("technical")    if isinstance(_live_data, dict) else {}) or {}
                _realized_vol = (_tech_data.get("realized_vol_pct")
                                 or _tech_data.get("volatility_pct")
                                 or _fund_data.get("volatility_pct"))
                _beta_val     = _fund_data.get("beta") or _fund_data.get("beta_world")
                _adv          = _tech_data.get("avg_daily_volume")
                _has_fund     = bool(_fund_data and (
                    _fund_data.get("pe") is not None
                    or _fund_data.get("eps") is not None
                    or _fund_data.get("revenue") is not None
                ))
                report_text = wrap_stock_report(
                    report_text,
                    symbol=resolution.symbol,
                    market=resolution.market or "",
                    asset_type=resolution.asset_type or "equity",
                    language=preferred_language,
                    realized_vol_pct=_realized_vol if isinstance(_realized_vol, (int, float)) else None,
                    beta=_beta_val if isinstance(_beta_val, (int, float)) else None,
                    fundamentals_available=_has_fund,
                    avg_daily_volume=_adv if isinstance(_adv, (int, float)) else None,
                )
        except Exception as _ex:
            logger.warning("[institutional-wrapper] failed for %s: %s", resolution.symbol, _ex)

        generated_at = datetime.now().astimezone().isoformat()
        report_json = None
        try:
            from core.services.pilot_report_json import build_pilot_report_json
            report_json = build_pilot_report_json(
                symbol=resolution.symbol,
                market=resolution.market,
                language=preferred_language,
                report_text=report_text,
                analysis_data=live_payload.get("data") or {},
                system_version=_APP_VERSION,
                model_primary="EisaX Agent",
                generated_at=generated_at,
                data_as_of=generated_at,
                latency_seconds=max(0, int(round(_time.perf_counter() - started_at))),
            )
        except Exception as exc:
            logger.warning("[staging-api] pilot json unavailable for '%s': %s", normalized_query, exc)

        import hashlib as _hashlib
        _json_rid = (report_json or {}).get("report_id", "")
        if _json_rid:
            _report_id = _json_rid
            _id_source = "json_meta"
        else:
            _report_id = _hashlib.sha256(
                f"{resolution.symbol}|{generated_at}|{report_text}".encode("utf-8")
            ).hexdigest()[:16]
            _id_source = "sha256_fallback"

        logger.info(
            "[polish] report_id=%s source=%s ticker=%s",
            _report_id, _id_source, resolution.symbol,
        )

        try:
            from core.polish_cache import set as _pc_set
            _pc_set(f"rule:{_report_id}", report_text)
        except Exception as _pc_err:
            logger.debug("[polish_cache] store skipped: %s", _pc_err)

        logger.info(
            "[editorial] editorial_mode=rule_based_only raw_len=%d report_id=%s "
            "llm_skipped=true endpoint=/staging-api/analyze",
            len(report_text), _report_id,
        )

        return _shape_public_result(
            count_success=True,
            query=normalized_query or selected_symbol,
            report_text=report_text,
            report_kind="stock",
            mode="live",
            download_url=live_payload.get("download_url"),
            html_report=report_text,
            report_json=report_json,
            resolution=resolution_payload,
            report_id=_report_id,
            analysis_data=live_payload.get("data") or {},
            subject_ticker=resolution.symbol,
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
                access_context=access_context,
            ),
        )


@staging_router.get("/chart-data")
@limiter.limit("60/minute")
async def staging_chart_data(request: Request, ticker: str = ""):
    """
    Public price history for Chart.js rendering on the staging frontend.
    Returns {dates, prices, ticker} for the last 60 trading days.
    No auth required — rate-limited to 60/min per IP.
    """
    _ensure_staging_public_enabled()
    ticker = (ticker or "").strip().upper()
    if not ticker or len(ticker) > 20:
        raise HTTPException(status_code=400, detail="Invalid ticker")

    import yfinance as _yf
    import math as _m
    from datetime import datetime as _dt, timedelta as _td

    df = None
    try:
        _end   = _dt.now()
        _start = _end - _td(days=90)
        _df = await run_in_threadpool(
            lambda: _yf.download(ticker, start=_start, end=_end, progress=False, auto_adjust=True)
        )
        if _df is not None and not _df.empty:
            df = _df
    except Exception:
        pass

    # Fallback: investing.com for UAE/local market tickers (DFM/ADX, Saudi, EGX, etc.)
    if df is None or df.empty:
        try:
            from core.market_data_engine import UAE_INVESTING, _fetch_investing
            info = UAE_INVESTING.get(ticker)
            if info:
                _start_str = (_dt.now() - _td(days=90)).strftime("%Y-%m-%d")
                _df = await run_in_threadpool(_fetch_investing, ticker, info, _start_str)
                if _df is not None and not _df.empty:
                    df = _df
        except Exception:
            pass

    if df is None or df.empty:
        return JSONResponse({"error": "No historical data available", "ticker": ticker})

    tail = df.tail(60)
    _col = "Close" if "Close" in tail.columns else tail.columns[0]
    _dates_raw  = list(tail.index)
    _prices_raw = [float(v) for v in tail[_col].values]

    dates  = [d.strftime("%b %d") for d, p in zip(_dates_raw, _prices_raw)
              if not (_m.isnan(p) or _m.isinf(p))]
    prices = [round(p, 3)        for p in _prices_raw
              if not (_m.isnan(p) or _m.isinf(p))]

    if not prices:
        return JSONResponse({"error": "No valid price data", "ticker": ticker})

    return {"dates": dates, "prices": prices, "ticker": ticker}


@staging_router.get("/quick")
@limiter.limit("60/minute")
async def staging_quick_snapshot(request: Request, q: str = ""):
    """
    Fast pipeline-cache snapshot for a ticker — returns in <2s.
    Used by the frontend to show Quick View while full analysis loads.
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

        if rsi is not None:
            if rsi < 30:   rsi_signal = "oversold"
            elif rsi > 70: rsi_signal = "overbought"
            else:          rsi_signal = "neutral"
        else:
            rsi_signal = "unknown"

        if price and sma200:
            sma_signal = "above" if price > sma200 else "below"
        else:
            sma_signal = "unknown"

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


@staging_router.post("/lead")
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


# ═══════════════════════════════════════════════════════════════════════════════
# Guest admin routes  (/v1/admin/guest-users*)
# ═══════════════════════════════════════════════════════════════════════════════

@guest_admin_router.get("/guest-users")
@limiter.limit("30/minute")
async def guest_access_users_page(request: Request):
    _require_guest_access_admin(request)
    from fastapi.responses import FileResponse
    from core.config import STATIC_DIR
    return FileResponse(str(STATIC_DIR / "admin_guest_users.html"))


@guest_admin_router.get("/guest-users/data")
@limiter.limit("30/minute")
async def guest_access_users_data(request: Request):
    _require_guest_access_admin(request)
    return {"users": _list_guest_access_users()}


@guest_admin_router.post("/guest-users")
@limiter.limit("20/minute")
async def upsert_guest_access_user(request: Request, body: GuestAccessUserRequest):
    _require_guest_access_admin(request, mutation=True)
    username = _safe_guest_username(body.username)
    role = (body.role or "guest").strip().lower()
    if role != "guest" or username.lower() in _STAGING_ADMIN_USERS:
        raise HTTPException(status_code=400, detail="This dashboard can only manage guest demo users")

    entries = _read_htpasswd_entries()
    if body.active:
        if body.password:
            password_hash = _hash_basic_auth_password(body.password)
        elif username in entries:
            password_hash = entries[username]
        else:
            source_user = _safe_guest_username(body.copy_password_from or "")
            if source_user.lower() in _STAGING_ADMIN_USERS:
                raise HTTPException(status_code=400, detail="Cannot copy admin credentials")
            password_hash = entries.get(source_user)
            if not password_hash:
                raise HTTPException(status_code=400, detail="Password required for new active user")
        entries[username] = password_hash
    else:
        entries.pop(username, None)

    _write_htpasswd_entries(entries)
    _upsert_guest_quota(
        username,
        max_analyses=body.analysis_limit,
        max_portfolios=body.portfolio_limit,
        analyses_used=body.analyses_used,
        portfolios_used=body.portfolios_used,
    )
    _audit_guest_access_admin(
        request,
        "upsert_guest",
        username,
        active=bool(body.active),
        analysis_limit=body.analysis_limit,
        portfolio_limit=body.portfolio_limit,
    )
    return {"status": "ok", "user": _guest_access_user_payload(username)}


@guest_admin_router.post("/guest-users/{username}/reset")
@limiter.limit("20/minute")
async def reset_guest_access_user_quota(request: Request, username: str):
    _require_guest_access_admin(request, mutation=True)
    username = _safe_guest_username(username)
    if username.lower() in _STAGING_ADMIN_USERS:
        raise HTTPException(status_code=400, detail="Admin users are not managed here")
    current = _guest_access_user_payload(username)
    _upsert_guest_quota(
        username,
        max_analyses=current["analysis_limit"],
        max_portfolios=current["portfolio_limit"],
        analyses_used=0,
        portfolios_used=0,
    )
    _audit_guest_access_admin(request, "reset_quota", username)
    return {"status": "ok", "user": _guest_access_user_payload(username)}


@guest_admin_router.delete("/guest-users/{username}")
@limiter.limit("10/minute")
async def delete_guest_access_user(request: Request, username: str):
    _require_guest_access_admin(request, mutation=True)
    username = _safe_guest_username(username)
    if username.lower() in _STAGING_ADMIN_USERS:
        raise HTTPException(status_code=400, detail="Admin users are not managed here")

    entries = _read_htpasswd_entries()
    removed_auth = entries.pop(username, None) is not None
    _write_htpasswd_entries(entries)
    try:
        import sqlite3 as _sq
        with _sq.connect("/home/ubuntu/investwise/investwise.db") as _c:
            _c.execute("DELETE FROM guest_trial WHERE username = ?", (username,))
            _c.commit()
    except Exception as exc:
        logger.warning("[guest_access_admin] quota delete failed for %s: %s", username, exc)
    _audit_guest_access_admin(request, "delete_guest", username, removed_auth=removed_auth)
    return {"status": "ok", "removed_auth": removed_auth}
