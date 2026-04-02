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
# Module-level start time – used to compute process uptime
# ---------------------------------------------------------------------------
_START_TIME: float = time.monotonic()

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
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return _down("GEMINI_API_KEY not set", (time.monotonic() - t0) * 1000)

        def _probe() -> str:
            from google import genai  # google-genai package
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="hi",
            )
            text = resp.text
            if not text:
                raise ValueError("Empty response text")
            return f"response length {len(text)} chars"

        detail = await asyncio.get_event_loop().run_in_executor(None, _probe)
        return _ok(detail, (time.monotonic() - t0) * 1000)
    except Exception as exc:
        return _down(f"{type(exc).__name__}: {exc}", (time.monotonic() - t0) * 1000)


# ── DeepSeek ─────────────────────────────────────────────────────────────────

async def _check_deepseek() -> ServiceResult:
    t0 = time.monotonic()
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            return _down("DEEPSEEK_API_KEY not set", (time.monotonic() - t0) * 1000)

        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                },
            )
            r.raise_for_status()
            return _ok(f"HTTP {r.status_code}", (time.monotonic() - t0) * 1000)
    except Exception as exc:
        return _down(f"{type(exc).__name__}: {exc}", (time.monotonic() - t0) * 1000)


# ── Kimi (Moonshot) ───────────────────────────────────────────────────────────

async def _check_kimi() -> ServiceResult:
    t0 = time.monotonic()
    try:
        api_key = os.getenv("MOONSHOT_API_KEY", "")
        if not api_key:
            return _down("MOONSHOT_API_KEY not set", (time.monotonic() - t0) * 1000)

        import httpx
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                "https://api.moonshot.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "kimi-k2.5",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                    "temperature": 1,
                },
            )
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


# ---------------------------------------------------------------------------
# Overall status aggregation
# ---------------------------------------------------------------------------

def _aggregate_status(services: dict[str, ServiceResult]) -> tuple[str, str]:
    """
    Returns (overall_status, summary_string).

    Rules
    -----
    - database down                          → "down"
    - all 3 LLMs (gemini, deepseek, kimi) down → "down"
    - 1-2 LLMs down OR disk/memory degraded → "degraded"
    - everything ok                          → "ok"
    """
    db_status = services["database"]["status"]
    llm_statuses = {
        name: services[name]["status"]
        for name in ("gemini", "deepseek", "kimi")
    }
    llms_down = [name for name, s in llm_statuses.items() if s == "down"]

    if db_status == "down":
        return "down", "Critical failure: database is unreachable"

    if len(llms_down) == 3:
        return "down", "Critical failure: all LLM services are down"

    degraded_services = [
        name
        for name, result in services.items()
        if result["status"] in ("degraded", "down")
    ]

    if degraded_services:
        count = len(degraded_services)
        names = ", ".join(degraded_services)
        return "degraded", f"{count} service(s) degraded: {names}"

    return "ok", "All systems operational"


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
        _check_kimi(),
        _check_disk(),
        _check_memory(),
        return_exceptions=True,
    )

    service_names = ("database", "gemini", "deepseek", "kimi", "disk", "memory")
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

    overall_status, summary = _aggregate_status(services)

    return {
        "status": overall_status,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "uptime_seconds": round(time.monotonic() - _START_TIME, 3),
        "services": services,
        "summary": summary,
    }
