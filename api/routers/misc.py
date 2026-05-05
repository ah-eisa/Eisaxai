"""
EisaX — Miscellaneous API Routers
Extracted from api_bridge_v2.py (lines 3772-4414).

Routers exported:
  admin_misc_router → /admin/cleanup, /admin/logs/*, /admin/analytics/*
  misc_router       → /v1/usage, /v1/redis/health, /v1/referral, /v1/webhooks,
                      /v1/billing/*, /v1/sentiment/*, /v1/backtest, /v1/screener,
                      /v1/forex, /v1/forex/{symbol}
"""

import os
import json as _json
import logging
import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException, Header, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

# ── Module-level globals ─────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger("api_bridge")

SECURE_TOKEN = os.getenv("SECURE_TOKEN", "")
ADMIN_TOKEN  = os.getenv("ADMIN_TOKEN", "")

# ── Paths from config ─────────────────────────────────────────────────────────
from core.config import APP_DB, BACKEND_LOG, STATIC_DIR

# ── JWT auth helpers ──────────────────────────────────────────────────────────
import jwt as _jwt

def _decode_token_lazy(token_str: str):
    """Lazily import and call decode_token from core.auth."""
    from core.auth import decode_token
    return decode_token(token_str)

_bearer = HTTPBearer(auto_error=False)


def _require_jwt(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    """FastAPI dependency — validates Bearer JWT, returns payload dict."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return _decode_token_lazy(credentials.credentials)
    except _jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except _jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def _decode_admin_session_token(token: str) -> Optional[dict]:
    if not token:
        return None
    try:
        payload = _decode_token_lazy(token)
    except (_jwt.ExpiredSignatureError, _jwt.InvalidTokenError):
        return None
    if payload.get("role") != "admin":
        return None
    return payload


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
    if ADMIN_TOKEN and token:
        from api_bridge_v2 import orchestrator as _orch
        if _orch.session_mgr.verify_admin_password(token, ADMIN_TOKEN):
            return
    if not SECURE_TOKEN and not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin access is not configured")
    raise HTTPException(status_code=403, detail="Forbidden")


# ── Pydantic models ───────────────────────────────────────────────────────────
class WebhookConfig(BaseModel):
    user_id:  str
    url:      str
    events:   list = Field(default_factory=lambda: ["analysis_complete"])
    secret:   str = ""


class CheckoutRequest(BaseModel):
    user_id: str
    email:   str
    tier:    str   # "pro" | "vip"


class PortalRequest(BaseModel):
    user_id: str


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
    include_sentiment: bool = False


# ── Routers ───────────────────────────────────────────────────────────────────
admin_misc_router = APIRouter(tags=["admin"])
misc_router = APIRouter(tags=["misc"])


# ═══════════════════════════════════════════════════════════════════════════════
# Admin misc routes
# ═══════════════════════════════════════════════════════════════════════════════

@admin_misc_router.post("/admin/cleanup")
@limiter.limit("5/minute")
async def run_cleanup(
    request: Request,
    days: int = 30,
    access_token: str = Header(None, alias="X-Admin-Key"),
):
    _check_admin(access_token or "", request)
    from api_bridge_v2 import orchestrator
    result = orchestrator.session_mgr.cleanup_old_sessions(days_to_keep=days)
    return result


@admin_misc_router.get("/admin/logs")
@limiter.limit("30/minute")
async def admin_logs_page(request: Request):
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / "admin_logs.html"))


@admin_misc_router.get("/admin/logs/stream")
@limiter.limit("10/minute")
async def admin_logs_stream(
    request: Request,
    access_token: str = Header(None, alias="X-Admin-Key"),
    access_token_alt: str = Header(None, alias="X-API-Key"),
    token: str = "",
):
    _check_admin(access_token or access_token_alt or token or "", request)

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
                        await asyncio.sleep(0.5)
        except Exception as exc:
            yield f"data: {_json.dumps({'line': f'[ERROR] {exc}'})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@admin_misc_router.get("/admin/analytics")
@limiter.limit("30/minute")
async def admin_analytics_page(request: Request):
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / "admin_analytics.html"))


@admin_misc_router.get("/admin/analytics/data")
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

    conn = sqlite3.connect(str(APP_DB))
    try:
        today = datetime.now(timezone.utc).date()
        days_list = [(today - timedelta(days=i)).isoformat() for i in range(13, -1, -1)]

        msgs_per_day = {}
        for day in days_list:
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


# ═══════════════════════════════════════════════════════════════════════════════
# Misc routes  (/v1/usage, /v1/redis/health, /v1/referral, webhooks, billing,
#               sentiment, backtest, screener, forex)
# ═══════════════════════════════════════════════════════════════════════════════

@misc_router.get("/v1/usage")
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
    from api_bridge_v2 import orchestrator
    return orchestrator.session_mgr.get_user_usage_stats(effective_user_id, days=min(days, 90))


@misc_router.get("/v1/redis/health")
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


@misc_router.get("/v1/referral")
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


@misc_router.post("/v1/referral/apply")
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


@misc_router.post("/v1/webhooks")
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


@misc_router.get("/v1/webhooks")
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


@misc_router.delete("/v1/webhooks/{webhook_id}")
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


@misc_router.post("/v1/billing/checkout")
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


@misc_router.post("/v1/billing/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        from core.billing import StripeBilling
        result = StripeBilling().handle_webhook(payload, sig)
        if result.get("tier") and result.get("user_id"):
            from api_bridge_v2 import orchestrator
            orchestrator.session_mgr.set_user_profile(result["user_id"], tier=result["tier"])
            logger.info("[billing] upgraded user %s to tier %s", result["user_id"], result["tier"])
        return {"received": True, "event": result.get("event")}
    except Exception as exc:
        logger.error("[billing/webhook] %s", exc)
        raise HTTPException(status_code=400, detail=f"Webhook error: {exc}")


@misc_router.post("/v1/billing/portal")
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


@misc_router.get("/v1/sentiment/{ticker}")
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


@misc_router.post("/v1/sentiment/batch")
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


@misc_router.get("/v1/sentiment/market/overview")
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


@misc_router.get("/v1/sentiment/{ticker}/trend")
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


@misc_router.post("/v1/backtest")
@limiter.limit("10/minute")
async def run_backtest(
    request: Request,
    body: BacktestRequest,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(403, "Unauthorized")
    try:
        from core.backtester import BacktestEngine, MACrossover, RSIStrategy, MACDStrategy
        strategies = {
            "ma_crossover": MACrossover(short=body.short_window, long=body.long_window),
            "rsi": RSIStrategy(period=body.rsi_period, oversold=body.rsi_oversold, overbought=body.rsi_overbought),
            "macd": MACDStrategy(),
        }
        if body.strategy not in strategies:
            raise HTTPException(400, f"Unknown strategy. Choose: {list(strategies.keys())}")
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
        logger.error("[backtest] %s", exc)
        raise HTTPException(500, f"Backtest error: {exc}")


@misc_router.post("/v1/screener")
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


@misc_router.get("/v1/forex")
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


@misc_router.get("/v1/forex/{symbol}")
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
