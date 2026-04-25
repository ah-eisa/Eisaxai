"""
core/services/health_service.py
────────────────────────────────
Comprehensive health check for the EisaX system.

Checks all critical services in parallel and returns a structured report
suitable for monitoring dashboards or API endpoints.

Usage
─────
    from core.services.health_service import run_health_check
    report = await run_health_check(secure_token="...")
"""

import asyncio
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Module-level start time – fallback for environments without psutil
# ---------------------------------------------------------------------------
_START_TIME: float = time.monotonic()

# ---------------------------------------------------------------------------
# Version – read once at import time from version.txt if available
# ---------------------------------------------------------------------------
_VERSION_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "version.txt")

def _read_version() -> str:
    try:
        with open(_VERSION_FILE, "r") as fh:
            return fh.read().strip() or "unknown"
    except Exception:
        return "unknown"

_VERSION: str = _read_version()

# ---------------------------------------------------------------------------
# Type alias for a single service result
# ---------------------------------------------------------------------------
ServiceResult = dict  # {"status": str, "latency_ms": float, "detail": str}


# ---------------------------------------------------------------------------
# Individual check helpers
# ---------------------------------------------------------------------------

def _ok(detail: str, latency_ms: float) -> ServiceResult:
    return {"status": "ok", "latency_ms": round(latency_ms, 2), "detail": detail}


def _degraded(detail: str, latency_ms: float) -> ServiceResult:
    return {"status": "degraded", "latency_ms": round(latency_ms, 2), "detail": detail}


def _down(detail: str, latency_ms: float) -> ServiceResult:
    return {"status": "down", "latency_ms": round(latency_ms, 2), "detail": detail}


# ── Database ─────────────────────────────────────────────────────────────────

async def _check_database() -> ServiceResult:
    t0 = time.monotonic()
    try:
        # SessionManager is synchronous; run it in a thread to avoid blocking
        # the event loop during I/O.
        def _probe() -> str:
            from core.session_manager import SessionManager  # local import – avoids circular deps at module load
            sm = SessionManager()
            sm.get_chat_history("__health_probe__")
            return "query ok"

        detail = await asyncio.get_event_loop().run_in_executor(None, _probe)
        return _ok(detail, (time.monotonic() - t0) * 1000)
    except Exception as exc:
        return _down(f"{type(exc).__name__}: {exc}", (time.monotonic() - t0) * 1000)


# ── Gemini ────────────────────────────────────────────────────────────────────

async def _check_gemini() -> ServiceResult:
    t0 = time.monotonic()
    try:
        primary_key = os.getenv("GEMINI_API_KEY", "")
        backup_key = os.getenv("GEMINI_API_KEY_BACKUP", "")

        candidates: list[tuple[str, str]] = []
        if primary_key:
            candidates.append(("primary", primary_key))
        if backup_key and backup_key != primary_key:
            candidates.append(("backup", backup_key))

        if not candidates:
            return _down("No Gemini API key configured", (time.monotonic() - t0) * 1000)

        def _probe(api_key: str) -> str:
            from google import genai  # google-genai package
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents="hi",
            )
            text = resp.text
            if not text:
                raise ValueError("Empty response text")
            return f"response length {len(text)} chars"

        errors: list[str] = []
        loop = asyncio.get_event_loop()
        for label, api_key in candidates:
            try:
                detail = await loop.run_in_executor(None, _probe, api_key)
                return _ok(f"{label} key ok ({detail})", (time.monotonic() - t0) * 1000)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                if len(msg) > 220:
                    msg = msg[:217] + "..."
                errors.append(f"{label}={msg}")

        return _down("; ".join(errors), (time.monotonic() - t0) * 1000)
    except Exception as exc:
        return _down(f"{type(exc).__name__}: {exc}", (time.monotonic() - t0) * 1000)


# ── DeepSeek ─────────────────────────────────────────────────────────────────

async def _check_deepseek() -> ServiceResult:
    """Probe DeepSeek using the lightweight /models endpoint — no tokens consumed."""
    t0 = time.monotonic()
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            return _down("DEEPSEEK_API_KEY not set", (time.monotonic() - t0) * 1000)

        import httpx
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(
                "https://api.deepseek.com/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if r.status_code == 401:
                return _down("Invalid API key (401)", (time.monotonic() - t0) * 1000)
            if r.status_code == 429:
                return _degraded("Rate limited (429)", (time.monotonic() - t0) * 1000)
            r.raise_for_status()
            return _ok(f"HTTP {r.status_code}", (time.monotonic() - t0) * 1000)
    except Exception as exc:
        return _down(f"{type(exc).__name__}: {exc}", (time.monotonic() - t0) * 1000)


# ── Disk ──────────────────────────────────────────────────────────────────────

async def _check_disk() -> ServiceResult:
    t0 = time.monotonic()
    try:
        total, used, free = shutil.disk_usage("/home/ubuntu/investwise")
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        detail = f"{free_gb:.2f} GB free of {total_gb:.2f} GB total"
        latency_ms = (time.monotonic() - t0) * 1000
        if free_gb < 1.0:
            return _degraded(f"Low disk space – {detail}", latency_ms)
        return _ok(detail, latency_ms)
    except Exception as exc:
        return _degraded(f"{type(exc).__name__}: {exc}", (time.monotonic() - t0) * 1000)


# ── Memory ────────────────────────────────────────────────────────────────────

async def _check_memory() -> ServiceResult:
    t0 = time.monotonic()
    try:
        import psutil
        vm = psutil.virtual_memory()
        available_gb = vm.available / (1024 ** 3)
        total_gb = vm.total / (1024 ** 3)
        percent_used = vm.percent
        detail = (
            f"{available_gb:.2f} GB available of {total_gb:.2f} GB total "
            f"({percent_used:.1f}% used)"
        )
        latency_ms = (time.monotonic() - t0) * 1000
        if available_gb < 0.25:
            return _degraded(f"Low memory – {detail}", latency_ms)
        return _ok(detail, latency_ms)
    except ImportError:
        # Fall back to /proc/meminfo when psutil is unavailable
        try:
            meminfo: dict[str, int] = {}
            with open("/proc/meminfo", "r") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        meminfo[key] = int(parts[1])  # value in kB

            total_kb = meminfo.get("MemTotal", 0)
            available_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            available_gb = available_kb / (1024 ** 2)
            total_gb = total_kb / (1024 ** 2)
            percent_used = round((1 - available_kb / total_kb) * 100, 1) if total_kb else 0.0
            detail = (
                f"{available_gb:.2f} GB available of {total_gb:.2f} GB total "
                f"({percent_used:.1f}% used) [via /proc/meminfo]"
            )
            latency_ms = (time.monotonic() - t0) * 1000
            if available_gb < 0.25:
                return _degraded(f"Low memory – {detail}", latency_ms)
            return _ok(detail, latency_ms)
        except Exception as exc2:
            return _degraded(
                f"Could not read memory info: {type(exc2).__name__}: {exc2}",
                (time.monotonic() - t0) * 1000,
            )
    except Exception as exc:
        return _degraded(f"{type(exc).__name__}: {exc}", (time.monotonic() - t0) * 1000)


# ── LLM Fallback Chain Stats ──────────────────────────────────────────────────

def _get_llm_fallback_stats() -> dict:
    """
    Query the llm_fallback module for provider health status.

    Returns a dict with active provider availability and circuit-breaker
    failure counts, plus response-cache stats.  Never raises.
    """
    try:
        from core.llm_fallback import get_llm_health  # local import – avoid circular deps
        return get_llm_health()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# ── Worker uptime ─────────────────────────────────────────────────────────────

def _get_worker_uptime() -> float:
    """
    Return the number of seconds the current gunicorn worker process has been
    running, derived from the OS process creation time via psutil.

    Falls back to the module-level monotonic clock (_START_TIME) when psutil
    is unavailable.
    """
    try:
        import psutil
        create_time = psutil.Process(os.getpid()).create_time()
        return round(time.time() - create_time, 3)
    except Exception:
        # psutil unavailable – use the module import time as approximation
        return round(time.monotonic() - _START_TIME, 3)


# ── Last analysis timestamp ───────────────────────────────────────────────────

def _get_last_analysis_at() -> str | None:
    """
    Return the ISO-8601 UTC timestamp of the most recent chat message stored
    in the sessions DB, or None if the table is empty or the query fails.
    """
    try:
        from core.session_manager import SessionManager  # local import
        sm = SessionManager()
        from core.db import db
        with db.get_cursor() as (conn, c):
            c.execute("SELECT MAX(timestamp) FROM chat_history")
            row = c.fetchone()
        if row and row[0]:
            raw = row[0]
            # SQLite stores timestamps as strings; normalise to ISO-8601 UTC
            if isinstance(raw, str) and raw:
                # Append Z if no timezone designator present
                if raw.endswith("Z") or "+" in raw or (raw.count("-") > 2):
                    return raw
                return raw + "Z"
            return str(raw)
        return None
    except Exception as exc:
        return None  # silently degrade – health check must not raise


# ---------------------------------------------------------------------------
# Overall status aggregation
# ---------------------------------------------------------------------------

def _aggregate_status(services: dict[str, ServiceResult]) -> tuple[str, str, list[str]]:
    """
    Returns (overall_status, summary_string, optional_failures).

    Rules
    -----
    - database down                          → "down"
    - both LLMs (gemini, deepseek) down      → "down"
    - deepseek down while gemini up          → "degraded"
    - gemini down while deepseek up          → "ok" (optional dependency)
    - disk/memory degraded                   → "degraded"
    - everything ok                          → "ok"
    """
    db_status = services["database"]["status"]
    gemini_status = services["gemini"]["status"]
    deepseek_status = services["deepseek"]["status"]
    optional_failures: list[str] = []

    if db_status == "down":
        return "down", "Critical failure: database is unreachable", optional_failures

    if gemini_status == "down" and deepseek_status == "down":
        return "down", "Critical failure: all LLM services are down", optional_failures

    degraded_services = [
        name
        for name, result in services.items()
        if result["status"] in ("degraded", "down")
    ]

    if gemini_status in ("down", "degraded") and deepseek_status == "ok":
        optional_failures.append("gemini")
        degraded_services = [name for name in degraded_services if name != "gemini"]

    if degraded_services:
        count = len(degraded_services)
        names = ", ".join(degraded_services)
        return "degraded", f"{count} service(s) degraded: {names}", optional_failures

    if optional_failures:
        names = ", ".join(optional_failures)
        return "ok", f"All core systems operational; optional issue(s): {names}", optional_failures

    return "ok", "All systems operational", optional_failures


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_health_check(secure_token: str) -> dict[str, Any]:  # noqa: ARG001  – token reserved for auth
    """
    Run all service checks in parallel and return a structured health report.

    Parameters
    ----------
    secure_token:
        Reserved for caller authentication / rate-limiting.  Not currently
        validated inside this function; enforcement is the caller's
        responsibility.

    Returns
    -------
    dict with keys: status, timestamp, uptime_seconds, services, summary
    """
    results: list[Any] = await asyncio.gather(
        _check_database(),
        _check_gemini(),
        _check_deepseek(),
        _check_disk(),
        _check_memory(),
        return_exceptions=True,
    )

    service_names = ("database", "gemini", "deepseek", "disk", "memory")
    services: dict[str, ServiceResult] = {}

    for name, result in zip(service_names, results):
        if isinstance(result, BaseException):
            # asyncio.gather should not surface exceptions here because each
            # helper already catches them, but guard defensively.
            services[name] = _down(
                f"Unhandled exception: {type(result).__name__}: {result}",
                0.0,
            )
        else:
            services[name] = result

    overall_status, summary, optional_failures = _aggregate_status(services)

    # ── New extended fields (each wrapped defensively) ────────────────────
    llm_providers: dict = {}
    try:
        llm_providers = _get_llm_fallback_stats()
    except Exception as exc:
        llm_providers = {"error": f"{type(exc).__name__}: {exc}"}

    worker_uptime: float = 0.0
    try:
        worker_uptime = _get_worker_uptime()
    except Exception:
        worker_uptime = round(time.monotonic() - _START_TIME, 3)

    last_analysis_at: str | None = None
    try:
        last_analysis_at = _get_last_analysis_at()
    except Exception:
        last_analysis_at = None

    # ── Redis health ──────────────────────────────────────────────────────────
    redis_status: dict = {}
    try:
        from core.redis_store import redis_info
        redis_status = redis_info()
    except Exception as exc:
        redis_status = {"available": False, "error": str(exc)}

    return {
        "status": overall_status,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "uptime_seconds": round(time.monotonic() - _START_TIME, 3),
        "worker_uptime_seconds": worker_uptime,
        "version": _VERSION,
        "last_analysis_at": last_analysis_at,
        "llm_providers": llm_providers,
        "redis": redis_status,
        "services": services,
        "optional_failures": optional_failures,
        "summary": summary,
    }
