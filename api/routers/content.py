"""
api/routers/content.py
───────────────────────
Content / intelligence routes extracted from api_bridge_v2.py.

Router exported:
  content_router  — no prefix, mounts /api/history, /v1/export, /v1/download,
                    /v1/polish-report, /v1/autocomplete, /v1/quick, /v1/brain,
                    /v1/alerts, /v1/version, /v1/export/html,
                    /v1/dashboard, /v1/translate-ar, /v1/export/html-pdf
"""
from __future__ import annotations

import logging
import os
import time as _time

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.routes.auth import _require_jwt
from core.config import EXPORTS_DIR
from core.export_engine import export as export_engine
from core.news_aggregator import get_news as _get_aggregated_news

logger = logging.getLogger("api_bridge")
limiter = Limiter(key_func=get_remote_address)

SECURE_TOKEN = os.getenv("SECURE_TOKEN", "")

content_router = APIRouter()

# ── Models ────────────────────────────────────────────────────────────────────

class HtmlExportPayload(BaseModel):
    html: str
    filename: str = ""
    access_token: str = ""

class TranslatePayload(BaseModel):
    text: str
    access_token: str = ""

# ── Chat history ──────────────────────────────────────────────────────────────

@content_router.get("/api/history")
@limiter.limit("60/minute")
async def get_history(request: Request, user=Depends(_require_jwt)):
    from api_bridge_v2 import orchestrator
    if user.get("role") == "admin":
        target_uid = request.query_params.get("user_id") or user["sub"]
    else:
        target_uid = user["sub"]
    return orchestrator.session_mgr.get_user_sessions(str(target_uid))

@content_router.get("/api/history/{session_id}")
@limiter.limit("60/minute")
async def get_session_history(request: Request, session_id: str, user=Depends(_require_jwt)):
    from api_bridge_v2 import orchestrator
    history = orchestrator.session_mgr.get_chat_history(session_id)
    sessions = orchestrator.session_mgr.get_user_sessions(str(user["sub"]))
    owned_ids = {s["session_id"] for s in sessions}
    if session_id not in owned_ids and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    return history

@content_router.delete("/api/history/{session_id}")
@limiter.limit("20/minute")
async def delete_session(request: Request, session_id: str, user=Depends(_require_jwt)):
    from api_bridge_v2 import orchestrator
    sessions = orchestrator.session_mgr.get_user_sessions(str(user["sub"]))
    owned_ids = {s["session_id"] for s in sessions}
    if session_id not in owned_ids and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    orchestrator.session_mgr.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}

# ── Export ─────────────────────────────────────────────────────────────────────

@content_router.post("/v1/export")
@limiter.limit("10/minute")
async def export_chat(
    request: Request,
    user: dict = Depends(_require_jwt),
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    import re, shutil
    from api_bridge_v2 import _create_download_token
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    fmt = body.get("format", "pdf")
    messages = body.get("messages", [])
    title = body.get("title", "EisaX Report")
    _polish = body.get("polish", False)
    smart = [m for m in messages if m.get("role") == "assistant"
             and len(m.get("content", "")) > 200
             and not any(x in m.get("content", "") for x in ["Hello!", "Hi!", "How can I help", "مرحباً", "أهلاً"])]
    if not smart:
        smart = messages

    # === GLM FORMATTING LAYER ===
    try:
        from core.glm_client import GLMClient
        glm = GLMClient()
        combined = "\n\n---\n\n".join([m.get("content", "") for m in smart if m.get("content")])
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
        "—": "-", "‘": "'", "“": '"', "”": '"',
        "?": "-"
    }

    def clean_content(text):
        for emoji, replacement in emoji_map.items():
            text = text.replace(emoji, replacement)
        return text

    smart = [{"role": m["role"], "content": clean_content(m.get("content", ""))} for m in smart]

    for msg in smart:
        c = msg.get("content", "")
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

            if _polish:
                _pdf_report_id = body.get("report_id", "")
                _polished_from_cache = None
                if _pdf_report_id:
                    try:
                        from core.polish_cache import get as _pc_get, set as _pc_set
                        _polished_from_cache = _pc_get(f"llm:{_pdf_report_id}")
                    except Exception:
                        pass
                if _polished_from_cache:
                    combined = _polished_from_cache
                    logger.info("[editorial.full] PDF using cached polished report (%d chars)", len(combined))
                else:
                    try:
                        from core.editorial import full_editorial_pass as _editorial_full
                        _raw_len_pdf = len(combined)
                        combined = _editorial_full(combined)
                        logger.info(
                            "[editorial] editorial_mode=llm_full raw_len=%d rule_clean_len=%d "
                            "latency_ms=N/A endpoint=/v1/export",
                            _raw_len_pdf, len(combined),
                        )
                        if _pdf_report_id:
                            try:
                                _pc_set(f"llm:{_pdf_report_id}", combined)
                            except Exception:
                                pass
                    except Exception as _ed_err:
                        logger.warning("[editorial.full] PDF pass skipped: %s", _ed_err)

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
            raise HTTPException(status_code=500, detail=result.get("error", "Export failed"))
        filename = os.path.basename(result.get("filename", ""))
        src = result.get("filename", "")
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


@content_router.get("/v1/download/{token}")
@limiter.limit("60/minute")
async def download_file(request: Request, token: str, user=Depends(_require_jwt)):
    """Download exported file — requires authentication."""
    from fastapi.responses import FileResponse
    from api_bridge_v2 import _DOWNLOAD_TOKENS

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

# ── Polish report ──────────────────────────────────────────────────────────────

@content_router.post("/v1/polish-report")
@limiter.limit("6/minute")
async def polish_report(request: Request):
    """
    Condensed 13-section institutional report from a previously analysed report.

    Request body (JSON):
      { "report_id": "<id>", "ticker": "<ticker>", "verdict": "<official verdict>", "language": "en|ar" }

    Flow:
      1. Check polish_cache llm:{language}:{report_id} → return instantly (fallback=false).
      2. Retrieve rule-based text from polish_cache rule:{report_id}.
      3. Resolve official verdict: body.verdict → regex-extract from base_text.
      4. Call polish_condensed(base_text, ticker, verdict) — LLM produces condensed version
         with a hard verdict-lock rule injected into the system prompt.
      5. _polish_guard checks: length + verdict (official) + ticker + numbers.
         Pass  → cache llm:{language}:{report_id}, return {ok, fallback: false, polished_report}
         Fail  → do NOT cache, return {ok, fallback: true, reason, clean_report: null}
         Frontend must NOT replace visible report when fallback=true.

    Logs: [polish-report] rejected report_id=<id> reason=<reason> cached=false
    """
    from api.routers.staging import _resolve_staging_access
    if _resolve_staging_access(request).get("role") == "guest":
        raise HTTPException(status_code=403, detail="Not available in demo access")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON body required: {report_id}")

    report_id = (body.get("report_id") or "").strip()
    ticker     = (body.get("ticker") or "").strip().upper()
    verdict    = (body.get("verdict") or "").strip().upper()
    language   = (body.get("language") or "en").strip().lower()
    if language not in ("en", "ar"):
        language = "en"
    logger.info("[polish-report] language=%s report_id=%s", language, report_id or "(missing)")
    if not report_id or len(report_id) > 64:
        raise HTTPException(status_code=400, detail="report_id required (max 64 chars)")

    try:
        from core.polish_cache import get as _pc_get, set as _pc_set
    except Exception as _imp_err:
        raise HTTPException(status_code=500, detail=f"Polish cache unavailable: {_imp_err}")

    # ── 1. Cache hit ──────────────────────────────────────────────────────────
    _cache_key = f"llm:{language}:{report_id}"
    _cached_llm = _pc_get(_cache_key)
    logger.info("[polish-cache] lookup key=%s hit=%s", _cache_key, "true" if _cached_llm else "false")
    if _cached_llm:
        return {
            "ok": True,
            "report_id": report_id,
            "polished_report": _cached_llm,
            "fallback": False,
            "from_cache": True,
        }

    # ── 2. Load rule-based source ─────────────────────────────────────────────
    base_text = _pc_get(f"rule:{report_id}")
    if not base_text:
        raise HTTPException(
            status_code=404,
            detail="Report not found — run /staging-api/analyze first to generate report_id",
        )

    # ── 3. Resolve official verdict (body → regex fallback from cached text) ──
    if not verdict:
        import re as _re
        _vdict_re = _re.compile(
            r'\b(STRONG BUY|STRONG SELL|BUY|SELL|HOLD|REDUCE|ACCUMULATE|AVOID)\b', _re.I
        )
        _vm = _vdict_re.search(base_text)
        if _vm:
            verdict = _vm.group(0).upper()
            logger.debug("[polish-report] verdict inferred from base_text: %s", verdict)

    # ── 4. Condensed institutional pass ──────────────────────────────────────
    from core.editorial import polish_condensed as _polish_condensed
    polished, is_fallback, fallback_reason = await run_in_threadpool(
        _polish_condensed, base_text, ticker, verdict, language
    )

    # ── 5. Cache only on success; never cache a rejected/fallback output ──────
    if not is_fallback:
        logger.info("[polish-cache] store key=%s", _cache_key)
        _pc_set(_cache_key, polished)
        logger.info(
            "[polish-report] editorial_mode=polish_condensed report_id=%s "
            "polished_len=%d verdict=%s fallback=false",
            report_id, len(polished), verdict or "?",
        )
        return {
            "ok": True,
            "report_id": report_id,
            "polished_report": polished,
            "fallback": False,
            "from_cache": False,
        }
    else:
        logger.warning(
            "[polish-report] rejected report_id=%s reason=%s cached=false",
            report_id, fallback_reason,
        )
        return {
            "ok": True,
            "report_id": report_id,
            "fallback": True,
            "reason": fallback_reason,
            "clean_report": None,
            "from_cache": False,
        }

# ── Autocomplete ───────────────────────────────────────────────────────────────

@content_router.get("/v1/autocomplete")
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
            _dfs[mkt] = _df

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
    except Exception:
        return {"results": []}

# ── Quick snapshot ─────────────────────────────────────────────────────────────

@content_router.get("/v1/quick")
@limiter.limit("60/minute")
async def v1_quick_snapshot(request: Request, q: str = ""):
    """
    Fast pipeline-cache snapshot — public, no auth, rate-limited.
    Returns price/RSI/SMA/PE/yield/sector/signal in <2s.
    Used by ai.eisax.com Quick View card while full analysis loads.
    """
    q = (q or "").strip().upper()
    if not q or len(q) > 20:
        raise HTTPException(status_code=400, detail="Invalid ticker")

    try:
        from pipeline import CacheManager as _QCV
        import math as _qmv

        _MKTS_SFX = {
            "uae": ".AE", "ksa": ".SR", "egypt": ".CA",
            "kuwait": ".KW", "qatar": ".QA", "bahrain": ".BH",
            "morocco": ".MA", "tunisia": ".TN",
            "america": "", "crypto": "-USD",
        }

        def _vfv(v):
            try:
                f = float(v)
                return None if (_qmv.isnan(f) or _qmv.isinf(f) or f == 0) else f
            except (TypeError, ValueError):
                return None

        bare = q.split("-")[0]
        for sfx in (".AE", ".SR", ".CA", ".KW", ".QA", ".BH", ".MA", ".TN"):
            if bare.endswith(sfx):
                bare = bare[: -len(sfx)]
                break

        qcmv = _QCV()
        row = None
        mkt_found = None
        for mkt, sfx in _MKTS_SFX.items():
            df, _ = qcmv.get_latest(mkt)
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
            raise HTTPException(status_code=404, detail=f"Ticker '{q}' not found")

        sfx = _MKTS_SFX.get(mkt_found, "")
        full_ticker = bare + sfx

        price  = _vfv(row.get("close"))
        change = round(float(row.get("change") or 0), 2)
        rsi    = _vfv(row.get("RSI"))
        sma50  = _vfv(row.get("SMA50"))
        sma200 = _vfv(row.get("SMA200"))
        pe     = _vfv(row.get("price_earnings_ttm"))
        divy   = _vfv(row.get("dividend_yield_recent"))
        mc     = _vfv(row.get("market_cap_basic"))
        sector = str(row.get("sector") or "").strip() or None
        macd   = _vfv(row.get("MACD.macd"))
        macd_s = _vfv(row.get("MACD.signal"))

        if rsi is not None:
            rsi_signal = "oversold" if rsi < 30 else ("overbought" if rsi > 70 else "neutral")
        else:
            rsi_signal = "unknown"

        sma_signal = ("above" if price > sma200 else "below") if (price and sma200) else "unknown"

        _bull, _bear = 0, 0
        if rsi_signal == "oversold":     _bull += 1
        elif rsi_signal == "overbought": _bear += 1
        if sma_signal == "above":        _bull += 1
        elif sma_signal == "below":      _bear += 1
        if macd is not None and macd_s is not None:
            if macd > macd_s: _bull += 1
            else:             _bear += 1
        if change > 1.5:    _bull += 0.5
        elif change < -1.5: _bear += 0.5

        quick_signal = "BULLISH" if _bull >= 2 else ("BEARISH" if _bear >= 2 else "NEUTRAL")

        return {
            "ticker":       full_ticker,
            "name":         str(row.get("name") or bare),
            "market":       mkt_found.upper() if mkt_found else "UNKNOWN",
            "price":        round(price, 4) if price else None,
            "change":       change,
            "rsi":          round(rsi, 1) if rsi else None,
            "rsi_signal":   rsi_signal,
            "sma50":        round(sma50, 4) if sma50 else None,
            "sma200":       round(sma200, 4) if sma200 else None,
            "sma_signal":   sma_signal,
            "pe_ratio":     round(pe, 2) if pe else None,
            "div_yield":    round(divy, 2) if divy else None,
            "market_cap":   round(mc / 1e9, 2) if mc else None,
            "sector":       sector,
            "quick_signal": quick_signal,
        }
    except HTTPException:
        raise
    except Exception as _qve:
        logger.warning("[v1/quick] error for %s: %s", q, _qve)
        raise HTTPException(status_code=500, detail="Quick snapshot unavailable")

# ── Brain / alerts / version ───────────────────────────────────────────────────

@content_router.get("/v1/brain/status")
@limiter.limit("30/minute")
async def brain_status(request: Request, access_token: str = Header(None, alias="X-API-Key")):
    if access_token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    from learning_engine import get_engine
    return get_engine().status()

@content_router.get("/v1/brain/wisdom")
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
        "engine_stats": engine._stats,
    }

@content_router.post("/v1/alerts")
@limiter.limit("20/minute")
async def create_alert(request: Request, access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(403, "Unauthorized")
    body = await request.json()
    from core.price_alerts import add_alert
    alert_id = add_alert(body.get('user_id', 'anonymous'), body['ticker'], body['condition'], body['threshold'])
    return {'alert_id': alert_id, 'status': 'created'}

@content_router.get("/v1/alerts")
@limiter.limit("30/minute")
async def list_alerts(request: Request, user_id: str = 'anonymous', access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(403, "Unauthorized")
    from core.price_alerts import get_user_alerts
    return get_user_alerts(user_id)

@content_router.delete("/v1/alerts/{alert_id}")
@limiter.limit("20/minute")
async def remove_alert(request: Request, alert_id: int, user_id: str = 'anonymous', access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(403, "Unauthorized")
    from core.price_alerts import delete_alert
    delete_alert(alert_id, user_id)
    return {'status': 'deleted'}

@content_router.get('/v1/version')
@limiter.limit('60/minute')
async def app_version(request: Request):
    return {'status': 'ok'}

# ── HTML → PDF export ─────────────────────────────────────────────────────────

@content_router.post("/v1/export/html")
@limiter.limit("10/minute")
async def export_html_to_pdf(
    request: Request,
    payload: HtmlExportPayload,
    access_token: str = Header(None, alias="X-API-Key"),
):
    from api_bridge_v2 import _create_download_token
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
        return {
            "url": f"/v1/download/{download_token}",
            "download_url": f"/v1/download/{download_token}",
            "filename": fname,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Dashboard ──────────────────────────────────────────────────────────────────

@content_router.get("/v1/dashboard/{ticker}")
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

    is_saudi  = ticker.endswith(".SR")
    tadawul_id = ticker.replace(".SR", "") if is_saudi else None

    from core.rapid_data import get_tadawul_quote, get_tadawul_history, _fetch_tadawul_candles
    try:
        if is_saudi:
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
            tadawul_quote = get_tadawul_quote(tadawul_id)
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

    try:
        close_series = prices_df[ticker] if ticker in prices_df.columns else prices_df.iloc[:, 0]
        tech   = generate_technical_summary(ticker, close_series)
        beta   = float((profile.get("fundamentals") or {}).get("beta") or 1.0)
        stress = run_stress_test(close_series, beta=beta)
    except Exception:
        tech   = {}
        stress = {"scenarios": {}, "annual_vol": 0}

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

    dc = dc_data or {}
    dc_technicals = {
        "rsi":         _safe_float(dc.get("rsi")),
        "sma50":       _safe_float(dc.get("sma50")),
        "sma200":      _safe_float(dc.get("sma200")),
        "short_float": _safe_float(dc.get("short_float")),
        "avg_volume":  dc.get("avg_volume"),
        "perf_week":   _safe_float(dc.get("perf_week")),
        "perf_month":  _safe_float(dc.get("perf_month")),
        "perf_ytd":    _safe_float(dc.get("perf_ytd")),
    }
    for k, v in dc_technicals.items():
        if v is not None:
            tech[k] = v

    dc_fundamentals = {
        "analyst_rating":      dc.get("analyst_rating"),
        "analyst_buy":         dc.get("analyst_buy"),
        "analyst_hold":        dc.get("analyst_hold"),
        "analyst_sell":        dc.get("analyst_sell"),
        "price_target":        dc.get("price_target"),
        "price_target_mean":   dc.get("price_target_mean"),
        "price_target_low":    dc.get("price_target_low"),
        "price_target_high":   dc.get("price_target_high"),
        "price_target_median": dc.get("price_target_median"),
        "forward_pe":          _safe_float(dc.get("forward_pe")),
        "earnings_date":       dc.get("earnings_date"),
        "week_52_range":       dc.get("week_52_range"),
        "inst_own":            _safe_float(dc.get("inst_own")),
        "insider_own":         _safe_float(dc.get("insider_own")),
        "debt_equity":         _safe_float(dc.get("debt_equity")),
        "roe":                 _safe_float(dc.get("roe")),
        "roa":                 _safe_float(dc.get("roa")),
        "profit_margin":       _safe_float(dc.get("profit_margin")),
        "gross_margin":        dc.get("gross_margin"),
        "net_margin":          dc.get("net_margin_annual"),
        "free_cash_flow":      dc.get("free_cash_flow"),
    }

    dc_financials = {
        "revenue_history": dc.get("revenue_history") or {},
        "eps_history":     dc.get("eps_history")     or {},
    }

    base_fundamentals = _clean_dict(profile.get("fundamentals", {}))
    for k, v in dc_fundamentals.items():
        if v is not None and k not in base_fundamentals:
            base_fundamentals[k] = v

    ev = events_data or {}
    events_fields = {
        "earnings_date": ev.get("earnings_date"),
        "ex_div_date":   ev.get("ex_div_date"),
        "div_date":      ev.get("div_date"),
        "eps_est_avg":   ev.get("eps_est_avg"),
        "eps_est_high":  ev.get("eps_est_high"),
        "eps_est_low":   ev.get("eps_est_low"),
        "rev_est_avg":   ev.get("rev_est_avg"),
    }
    for k, v in events_fields.items():
        if v is not None and not base_fundamentals.get(k):
            base_fundamentals[k] = v

    mp = market_pulse or {}
    cnbc_news = _get_aggregated_news(ticker=ticker, limit=5)
    combined_news = (dc_news or []) + cnbc_news

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
        "ticker":        ticker,
        "quote":         _clean_dict(quote),
        "fundamentals":  base_fundamentals,
        "technicals":    _clean_dict(tech),
        "financials":    dc_financials,
        "stress":        {k: _clean_dict(v) for k, v in stress.get("scenarios", {}).items()},
        "annual_vol":    stress.get("annual_vol", 0),
        "news":          combined_news,
        "fear_greed":    mp.get("fear_greed") or {},
        "econ_calendar": mp.get("calendar")   or [],
        "dc_source":     dc.get("source", ""),
        "is_saudi":        is_saudi,
        "tadawul_intraday": tadawul_hist,
    }

# ── Translate to Arabic ────────────────────────────────────────────────────────

@content_router.post("/v1/translate-ar")
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
    text_in = payload.text[:8000]
    user_msg = f"ترجم هذا النص كاملاً مع الحفاظ على تنسيق Markdown:\n\n{text_in}"

    import httpx

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
                        "max_tokens": 4000,
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
                    "max_tokens": 8000,
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

# ── HTML-PDF export (alternate endpoint) ──────────────────────────────────────

@content_router.post("/v1/export/html-pdf")
@limiter.limit("5/minute")
async def export_html_pdf(request: Request, payload: HtmlExportPayload, access_token: str = Header(None, alias="X-API-Key")):
    from api_bridge_v2 import _create_download_token
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
        return {
            "url": f"/v1/download/{download_token}",
            "download_url": f"/v1/download/{download_token}",
            "filename": fname,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
