"""
core/services/market_route_handler.py
───────────────────────────────────────
Route handlers for STOCK_ANALYSIS, FINANCIAL (CIO/PORTFOLIO_OPTIMIZE),
PORTFOLIO CRUD and GENERAL (Gemini) — extracted from process_message.

Public API
──────────
    handle_stock_analysis(orchestrator, session_id, user_id,
                          message, instruction, user_ctx) -> dict

    handle_financial(orchestrator, session_id, user_id,
                     message, instruction, handler, user_ctx) -> dict | None
        Returns None if it falls through (caller continues to GENERAL).

    handle_portfolio(session_id, user_id, message, reply_saver) -> dict

    handle_general(orchestrator, session_id, user_id,
                   message, instruction, user_ctx) -> dict
"""

from __future__ import annotations

import logging
import os
import re as _re
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── STOCK_ANALYSIS ─────────────────────────────────────────────────────────────

async def handle_stock_analysis(
    orchestrator: Any,
    session_id:   str,
    user_id:      str,
    message:      str,
    instruction:  str,
    user_ctx:     dict,
) -> dict:
    """
    Run the full EisaX stock-analysis pipeline (FinancialAgent → Gemini fallback).
    """
    import asyncio
    _resolved_ticker = None

    # ── Analysis Cache check ─────────────────────────────────────────────────
    try:
        try:
            from core.ticker_resolver import resolve_ticker as _resolve
        except Exception:
            from core.tools.ticker_resolver import resolve_ticker as _resolve
        from core.analysis_cache import get as _ac_get

        _resolved_ticker = _resolve(message) or _resolve(instruction or "")
        if not _resolved_ticker:
            _combined = f"{message} {instruction or ''}".upper()
            _candidates = _re.findall(r"\b([A-Z]{2,6}(?:=[A-Z])?)\b", _combined)
            _skip = {"AND", "THE", "FOR", "WITH", "PRICE", "STOCK", "ANALYZE"}
            for _tk in _candidates:
                if _tk not in _skip:
                    _resolved_ticker = _tk
                    break
        if _resolved_ticker:
            _cached = _ac_get(_resolved_ticker)
            if _cached:
                age_min = _cached["cache_age"] // 60
                reply = _cached["reply"]
                # Add subtle cache indicator
                reply += f"\n\n*ℹ️ تحليل محدّث (منذ {age_min} دقيقة)*"
                orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
                orchestrator.session_mgr.save_message(session_id, user_id, "assistant", reply)
                return {
                    "reply": reply,
                    "session_id": session_id,
                    "agent_name": "EisaX Cache",
                    "model": _cached["model"],
                }
    except Exception as _ce:
        logger.debug(f"[AnalysisCache] cache check skipped: {_ce}")
        _resolved_ticker = None

    try:
        # ── DFM screening inside STOCK_ANALYSIS ───────────────────────────────
        try:
            from core.dfm_lookup import is_dfm_query, get_dfm_context, screen_dfm
            ml = message.lower()
            _screen_map = {
                "low_pe":   ["أرخص", "ارخص", "lowest pe", "low pe", "cheap", "undervalued", "value stocks"],
                "high_pe":  ["أغلى", "اغلى", "highest pe", "growth", "expensive"],
                "low_beta": ["أقل تذبذب", "stable", "low risk", "low beta", "defensive"],
            }
            screen_hit = None
            for crit, kws in _screen_map.items():
                if any(k in ml for k in kws) and any(x in ml for x in ["dfm", "سوق دبي", "dubai"]):
                    screen_hit = crit
                    break
            if screen_hit:
                logger.info("[DFM] Screening: %s", screen_hit)
                stocks = screen_dfm(screen_hit, top_n=10)
                text   = f"## DFM Screening: {screen_hit.replace('_', ' ').title()}\n\n"
                text  += "| # | Company | Ticker | P/E | Beta | Market Cap | Volume |\n"
                text  += "|---|---------|--------|-----|------|------------|--------|\n"
                for i, s in enumerate(stocks, 1):
                    text += (
                        f"| {i} | {s['name']} | {s['ticker'] or 'N/A'} | "
                        f"{s['pe_ratio'] or 'N/A'} | {s['beta'] or 'N/A'} | "
                        f"{s['market_cap']} | {s['avg_vol_3m']} |\n"
                    )
                dfm_reply = await orchestrator._handle_dfm_query(message, text)
                orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
                orchestrator.session_mgr.save_message(session_id, user_id, "assistant", dfm_reply)
                return {
                    "reply": dfm_reply, "session_id": session_id,
                    "agent_name": "EisaX DFM Screen", "model": "deepseek",
                }

            # ── Single DFM stock fast-path (only when no FinancialAgent) ──────
            if not orchestrator.financial_agent and is_dfm_query(message):
                dfm_ctx = get_dfm_context(message)
                if dfm_ctx:
                    logger.info("[DFM] Fast-path for: %s", message[:50])
                    dfm_reply = await orchestrator._handle_dfm_query(message, dfm_ctx)
                    orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
                    orchestrator.session_mgr.save_message(session_id, user_id, "assistant", dfm_reply)
                    return {
                        "reply": dfm_reply, "session_id": session_id,
                        "agent_name": "EisaX DFM", "model": "deepseek",
                    }
            elif is_dfm_query(message):
                logger.info("[DFM] Skipping fast-path → full EisaX pipeline for: %s", message[:50])
        except Exception as dfm_err:
            logger.warning("[DFM] Fast-path failed: %s", dfm_err)

        # ── EGX fast-path (only when no FinancialAgent) ───────────────────────
        try:
            from core.egx_lookup import is_egx_query, get_egx_context
            if not orchestrator.financial_agent and is_egx_query(message):
                egx_ctx = get_egx_context(message)
                if egx_ctx:
                    logger.info("[EGX] Fast-path for: %s", message[:50])
                    egx_reply = await orchestrator._handle_dfm_query(message, egx_ctx)
                    orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
                    orchestrator.session_mgr.save_message(session_id, user_id, "assistant", egx_reply)
                    return {
                        "reply": egx_reply, "session_id": session_id,
                        "agent_name": "EisaX EGX", "model": "deepseek",
                    }
            elif is_egx_query(message):
                logger.info("[EGX] Skipping fast-path → full EisaX pipeline for: %s", message[:50])
        except Exception as egx_err:
            logger.warning("[EGX] Fast-path failed: %s", egx_err)

        # ── FinancialAgent (primary path) ─────────────────────────────────────
        agent = orchestrator.financial_agent
        if agent:
            loop = asyncio.get_event_loop()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: agent._handle_analytics(
                            session_id,
                            {"user_ctx": user_ctx, "user_id": user_id},
                            instruction,
                        ),
                    ),
                    timeout=240,
                )
            except asyncio.TimeoutError:
                logger.warning("[STOCK_ANALYSIS] Agent timed out — falling back to DeepSeek")
                _fb = await orchestrator._handle_bond_query(instruction)
                result = {
                    "reply": _fb if isinstance(_fb, str)
                             else _fb.get("reply", "❌ Analysis timed out. Please try again.")
                }

            reply_text = result.get("reply", "")
            if reply_text:
                orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
                orchestrator.session_mgr.save_message(session_id, user_id, "assistant", reply_text)
                _track_memory(orchestrator, user_id, message, instruction, reply_text)
                # ── Save to analysis cache ───────────────────────────────────
                try:
                    if _resolved_ticker:
                        from core.analysis_cache import set as _ac_set
                        _ac_set(_resolved_ticker, reply_text, result.get("model", "deepseek"))
                except Exception as _ce:
                    logger.debug(f"[AnalysisCache] cache set skipped: {_ce}")
                return {
                    "reply":      reply_text,
                    "session_id": session_id,
                    "agent_name": "EisaX Financial Analyst",
                    "model":      "DeepSeek + yfinance",
                }
            else:
                raise ValueError(f"FinancialAgent returned empty reply for: {instruction[:60]}")

    except Exception as exc:
        logger.warning("FinancialAgent failed: %s — falling back to Gemini with live price", exc)

    # ── Gemini fallback with live price injection ─────────────────────────────
    result = await _stock_gemini_fallback(orchestrator, session_id, user_id, message, instruction)
    # ── Save to analysis cache ───────────────────────────────────────────────
    try:
        if _resolved_ticker and isinstance(result, dict) and result.get("reply"):
            from core.analysis_cache import set as _ac_set
            _ac_set(_resolved_ticker, result["reply"], result.get("model", "deepseek"))
    except Exception as _ce:
        logger.debug(f"[AnalysisCache] cache set skipped: {_ce}")
    return result


async def _stock_gemini_fallback(
    orchestrator: Any,
    session_id:   str,
    user_id:      str,
    message:      str,
    instruction:  str,
) -> dict:
    """Last-resort Gemini analysis when FinancialAgent fails."""
    from datetime import datetime as _dt
    try:
        today = _dt.now().strftime("%B %d, %Y")
        cached_price_str = ""
        try:
            from core.price_cache import get as _gcp
            raw_ticker = instruction.upper().split()[-1] if instruction else ""
            alias_map  = {
                "XAUUSD": "GC=F", "XAU/USD": "GC=F", "GOLD": "GC=F",
                "XAGUSD": "SI=F", "SILVER": "SI=F",
                "XPTUSD": "PL=F", "PLATINUM": "PL=F",
                "XPDUSD": "PA=F", "PALLADIUM": "PA=F",
                "XCUUSD": "HG=F", "COPPER": "HG=F",
                "OIL": "CL=F", "XTIUSD": "CL=F", "CRUDE": "CL=F",
            }
            lookup = alias_map.get(raw_ticker, raw_ticker)
            cp     = _gcp(lookup)
            if cp:
                cached_price_str = (
                    f"\n\n⚠️ LIVE PRICE AVAILABLE (from cache): "
                    f"{lookup} = ${cp:,.2f} — "
                    "YOU MUST USE THIS EXACT PRICE. Do NOT use any other price."
                )
        except Exception:
            pass

        system = (
            f"You are EisaX AI, institutional-grade investment intelligence built by Ahmed Eisa.\n"
            f"Today: {today}.\n"
            "CRITICAL: You are analyzing based on CURRENT 2026 market data. "
            "Do NOT use training-data prices — use ONLY the live price injected below if available. "
            f"If no live price is injected, explicitly state the price is unavailable and do not guess."
            f"{cached_price_str}"
        )
        fb_reply = orchestrator._gemini_generate(
            f"{system}\n\nUser: {instruction}\n\nAssistant:", label="STOCK_FB"
        ) or ""
        if fb_reply:
            orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
            orchestrator.session_mgr.save_message(session_id, user_id, "assistant", fb_reply)
            return {
                "reply":      fb_reply,
                "session_id": session_id,
                "agent_name": "EisaX Financial Analyst",
                "model":      "Gemini (fallback)",
            }
    except Exception as fb_err:
        logger.error("STOCK_ANALYSIS Gemini fallback also failed: %s", fb_err)

    err_reply = "⚠️ Live market data is temporarily unavailable. Please try again in a moment."
    orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
    orchestrator.session_mgr.save_message(session_id, user_id, "assistant", err_reply)
    return {"reply": err_reply, "session_id": session_id, "agent_name": "EisaX", "model": "error"}


def _track_memory(
    orchestrator: Any,
    user_id:      str,
    message:      str,
    instruction:  str,
    reply_text:   str,
) -> None:
    """Extract ticker facts and save analysis verdict to memory (fire-and-forget)."""
    try:
        from core.orchestrator import MEMORY_ENABLED
        if not MEMORY_ENABLED:
            return
        from core.memory import extract_and_save_user_facts, track_stock_interest
        from core.orchestrator import _save_analysis_to_memory
        extract_and_save_user_facts(user_id, message, reply_text[:600])
        mentioned = _re.findall(r"\b([A-Z]{2,5})\b", instruction.upper())
        skip      = {"AND", "OR", "THE", "FOR", "VS", "BUY", "SELL", "HOLD"}
        for tk in mentioned[:3]:
            if tk not in skip:
                track_stock_interest(user_id, tk)
                _save_analysis_to_memory(user_id, tk, reply_text)
    except Exception:
        pass


# ── FINANCIAL ──────────────────────────────────────────────────────────────────

async def handle_financial(
    orchestrator: Any,
    session_id:   str,
    user_id:      str,
    message:      str,
    instruction:  str,
    handler:      str,
    user_ctx:     dict,
    chat_history: list = None,
) -> dict | None:
    """
    Handle CIO_ANALYSIS and PORTFOLIO_OPTIMIZE routes.
    Returns None only if both sub-handlers fail (caller falls through to GENERAL).
    """
    from core.agents.finance import FinancialAgent

    # ── PORTFOLIO_OPTIMIZE: Gate — require min inputs before executing ────────
    # Checks current message AND recent conversation history combined.
    # This prevents the system from inventing a portfolio from a vague request.
    if handler == "PORTFOLIO_OPTIMIZE":
        import re as _re_po
        # Build accumulated context: current message + last 6 history messages
        _hist_text = " ".join(
            m.get("content", "") for m in (chat_history or [])[-6:]
            if m.get("role") in ("user", "assistant")
        )
        _full_ctx = (message + " " + _hist_text).lower()
        _has_amount  = bool(_re_po.search(
            r'\d[\d,\.]*\s*(k|m|b|الف|مليون|مليار|دولار|ريال|درهم|\$|usd|eur|aed|sar)',
            _full_ctx
        ))
        _has_risk    = any(w in _full_ctx for w in [
            'aggressive','عدوانية','عدواني','عدوانى','جريئة','جريء','جرئ','جريئ',
            'risk','مخاطرة','مخاطر','conservative','محافظ','متوازن',
            'balanced','moderate','low risk','high risk',
            'منخفض','متوسط','عالي','عالية','bold','جرئ',
        ])
        _has_market  = any(w in _full_ctx for w in [
            'uae','dubai','saudi','مصر','egypt','us','america','global',
            'gulf','خليج','سوق','stocks','bonds','crypto','gold','سندات',
            'اسهم','اسهم','ذهب','خام','تشفير','محلي','دولي',
            'امريكي','امريكية','سعودي','سعودية','اماراتي','اماراتية',
        ])
        _missing = []
        if not _has_amount:  _missing.append("budget")
        if not _has_risk:    _missing.append("risk_tolerance")
        if not _has_market:  _missing.append("preferred_markets")
        if _missing:
            _is_ar = bool(_re.search(r"[\u0600-\u06FF]", message))
            _lang_guard = (
                "User language is Arabic. You MUST reply fully in Arabic only.\n"
                if _is_ar else
                "User language is English. You MUST reply fully in English only.\n"
            )
            _gate_prompt = (
                "Reply in the same language as the user's message. If the user wrote in Arabic, respond entirely in Arabic. If in English, respond in English.\n\n"
                + _lang_guard + "\n"
                "You are EisaX Portfolio Advisor. The user asked for portfolio construction/optimization, "
                "but key inputs are missing.\n"
                f"User message: {message}\n"
                f"Recent context: {_hist_text[:1200]}\n"
                f"Missing required fields: {', '.join(_missing)}\n\n"
                "Field definitions:\n"
                "- budget: investable amount and currency.\n"
                "- risk_tolerance: conservative / balanced / aggressive.\n"
                "- preferred_markets: GCC / US / International / mixed.\n\n"
                "Also ask for optional details:\n"
                "- investment horizon (short/medium/long)\n"
                "- target return (optional)\n\n"
                "Write a concise, friendly clarification message with bullet points. "
                "Do not build a portfolio yet."
            )
            _clarify = (orchestrator._gemini_generate(_gate_prompt, label="PORTFOLIO_GATE") or "").strip()
            _clarify_has_ar = bool(_re.search(r"[\u0600-\u06FF]", _clarify or ""))
            if (not _clarify) or (_is_ar and not _clarify_has_ar):
                if _is_ar:
                    _clarify = (
                        "حتى أبني لك محفظة دقيقة، أحتاج المعلومات التالية:\n"
                        "• الميزانية والعملة\n"
                        "• مستوى المخاطرة (محافظ/متوازن/عدواني)\n"
                        "• الأسواق المفضلة (خليجي/أمريكي/دولي/مختلط)\n\n"
                        "اختياريًا: المدة الزمنية وهدف العائد."
                    )
                else:
                    _clarify = (
                        "To build an accurate portfolio, please share:\n"
                        "• Budget and currency\n"
                        "• Risk tolerance (conservative/balanced/aggressive)\n"
                        "• Preferred markets (GCC/US/international/mixed)\n\n"
                        "Optional: time horizon and target return."
                    )
            orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
            orchestrator.session_mgr.save_message(session_id, user_id, "assistant", _clarify)
            return {"reply": _clarify, "session_id": session_id, "agent_name": "EisaX Portfolio Advisor"}

    # ── CIO_ANALYSIS ──────────────────────────────────────────────────────────
    if handler == "CIO_ANALYSIS":
        logger.info("[Orchestrator] Router → CIO_ANALYSIS handler")
        try:
            result     = FinancialAgent()._handle_cio_analysis(message)
            reply_text = result.get("reply", "")
            if reply_text:
                orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
                orchestrator.session_mgr.save_message(session_id, user_id, "assistant", reply_text)
                return {
                    "reply": reply_text, "session_id": session_id,
                    "agent_name": "EisaX CIO", "model": "deepseek-cio",
                }
        except Exception as exc:
            logger.error("[Orchestrator] CIO handler failed: %s", exc)

        # ── CLEAN PIPELINE (Step 1-5 architecture) ────────────────────────────────
    try:
        import sys as _sys, os as _os
        _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from portfolio_pipeline import is_pipeline_request, run as pipeline_run
        if is_pipeline_request(message):
            import logging as _log
            _log.getLogger(__name__).info("[Pipeline] Routing to clean pipeline: %s", message[:60])
            _pipeline_report = pipeline_run(message)
            if _pipeline_report:
                orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
                orchestrator.session_mgr.save_message(session_id, user_id, "assistant", _pipeline_report)
                return {
                    "reply": _pipeline_report, "session_id": session_id,
                    "agent_name": "EisaX Portfolio Pipeline", "model": "pipeline+deepseek",
                }
    except Exception as _pipe_exc:
        import logging as _log
        _log.getLogger(__name__).warning("[Pipeline] Failed: %s — falling back to legacy", _pipe_exc)

        # ── GLOBAL ALLOCATOR (Portfolio Build) ────────────────────────────────────
    try:
        from portfolio_builder import detect_and_build
        alloc_reply = detect_and_build(message)
        if alloc_reply:
            import logging
            logging.getLogger(__name__).info("[GlobalAllocator] Built portfolio for: %s", message[:50])
            orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
            orchestrator.session_mgr.save_message(session_id, user_id, "assistant", alloc_reply)
            return {
                "reply": alloc_reply, "session_id": session_id,
                "agent_name": "EisaX Global Allocator", "model": "CLARABEL QP",
            }
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("[GlobalAllocator] Failed: %s", exc)

    # ── PORTFOLIO_OPTIMIZE ────────────────────────────────────────────────────
    try:
        agent = orchestrator.financial_agent
        if not agent:
            return None

        mem: dict = {}
        try:
            from core.orchestrator import MEMORY_ENABLED
            if MEMORY_ENABLED:
                from core.memory import get_user_memory
                mem = get_user_memory(user_id) or {}
        except Exception:
            pass

        if user_ctx:
            mem["user_ctx"] = user_ctx
            mem["user_id"]  = user_id
            if user_ctx.get("risk_profile") and "risk" not in mem:
                mem["risk"] = user_ctx["risk_profile"]

        result     = agent._handle_optimize("default", mem, instruction, {})
        reply_text = result.get("reply", "")
        if reply_text:
            orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
            orchestrator.session_mgr.save_message(session_id, user_id, "assistant", reply_text)
            try:
                from core.orchestrator import MEMORY_ENABLED
                if MEMORY_ENABLED:
                    from core.memory import extract_and_save_user_facts
                    extract_and_save_user_facts(user_id, message, reply_text)
            except Exception:
                pass
            return {
                "reply": reply_text, "session_id": session_id,
                "agent_name": "EisaX Financial Advisor",
                "model":      "Portfolio Optimizer",
            }

        # ── Normalize + agent.think() fallback ─────────────────────────────
        try:
            norm_prompt = (
                "You are a financial request normalizer.\n"
                "Convert this user request to a clear, direct English instruction "
                "for a financial AI agent.\n"
                "Keep all numbers, percentages, and financial terms.\n"
                "Return ONLY the normalized English instruction, nothing else.\n\n"
                f'User request: "{message}"\n\n'
                'Examples:\n'
                '"عايز محفظة عدوانية بـ 100,000 دولار" → "Build an aggressive multi-asset portfolio with $100,000 capital targeting high returns"\n'
                '"اعمل محفظة بـ 15% عائد" → "Build a portfolio targeting 15% annual return with diversified assets"\n'
                '"قارن بين NVDA و AMD" → "Compare NVDA vs AMD stocks with full analysis"\n'
            )
            norm_instruction = orchestrator._gemini_generate(norm_prompt, label="normalize")
            logger.info("[Normalize] '%s' → '%s'", message[:50], norm_instruction[:80])
        except Exception:
            norm_instruction = instruction

        result2    = agent.think(norm_instruction, {"session_id": session_id}, {})
        reply_text = result2.get("reply", "")
        if reply_text:
            orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
            orchestrator.session_mgr.save_message(session_id, user_id, "assistant", reply_text)
            try:
                from core.orchestrator import MEMORY_ENABLED
                if MEMORY_ENABLED:
                    from core.memory import extract_and_save_user_facts
                    extract_and_save_user_facts(user_id, message, reply_text)
            except Exception:
                pass
            return {
                "reply": reply_text, "session_id": session_id,
                "agent_name": "EisaX Portfolio Manager",
                "model":      "FinancialAgent",
            }
    except Exception as exc:
        logger.warning("FINANCIAL route failed: %s", exc)

    return None  # fall through to GENERAL


# ── PORTFOLIO CRUD ──────────────────────────────────────────────────────────────

async def handle_portfolio(
    session_id:  str,
    user_id:     str,
    message:     str,
    reply_saver: Callable[[str, str, str, str], None],
) -> dict:
    """
    Handle portfolio CRUD (add/remove/show positions).

    Parameters
    ──────────
    reply_saver : callable(session_id, user_id, role, text)
    """
    from core.portfolio_tracker import PortfolioTracker

    _NON_TICKERS = {
        "SHARE","SHARES","STOCK","STOCKS","FETCH","GIVE","CALC",
        "CALCU","SUGGE","PLEAS","APPRO","TOTAL","EACH","THEN",
        "WITH","FROM","FULL","PART","HOLD","SELL","SHOW","HELP",
        "COST","PRICE","MART","MARK","SUGG","CURR","AVER","YOUR",
        "INTO","CASE","THIS","THAT","THEY","WHEN","WHAT","SOME",
        "MORE","LESS","BEST","RISK","SAFE","HIGH","LOWM","MILD",
        "BEAR","MILD","SEVE","CRAS","ACRO","EXPE","COMP","INCL",
        "CLEA","DIRE","NUMER","MATE","OPPO","WEIG","POSI","REMO",
        "ADDI","POTE","REDU","INST","CALC","PERF","FULL","SIMP",
        "RUN","USE","GET","PUT","SET","LET","CUT","ADD","AND",
        "FOR","THE","ALL","ANY","NEW","OLD","TOP","LOW","MAX",
        "MIN","NET","PER","OFF","OUT","END","DAY","NOW","YES",
        "NOT","ALSO","BOTH","EACH","SUCH","JUST","VERY","ONLY",
    }

    _ANALYSIS_SIGNALS = [
        "stress test","p&l","cio","unrealized","احسب","profit/loss",
        "scenario","rebalance","توصية","recommend","fetch latest",
        "calculate","stress","current portfolio","محفظتي الحالية",
        "analyze my","analysis","hold","sell partial","buy more",
    ]

    try:
        import time as _time
        import re as _rep
        pt         = PortfolioTracker()
        ml         = message.lower()
        is_analysis = any(sig in ml for sig in _ANALYSIS_SIGNALS)

        raw_adds   = _rep.findall(r"(\d+\.?\d*)\s*([A-Z]{2,6})", message.upper())
        all_adds   = [(sh, tk) for sh, tk in raw_adds if tk not in _NON_TICKERS]
        price_hint = _rep.search(r"(?:at|@|بسعر)\s*\$?(\d+\.?\d*)", message, _rep.I)
        is_add     = (
            not is_analysis and
            bool(_rep.search(r"(?:^|\b)(?:add|buy|اشتر|أضف|اضف)\b", message, _rep.I))
        )
        remove_match = _rep.search(
            r"(?:remove|sell|احذف|بيع)\s*(?:(\d+\.?\d*)\s*)?([A-Z]{2,5})",
            message, _rep.I,
        )

        if is_add and all_adds:
            added = []
            from core.market_data import get_full_stock_profile
            for shares_str, ticker in all_adds:
                shares = float(shares_str)
                if price_hint:
                    price = float(price_hint.group(1))
                else:
                    try:
                        profile = get_full_stock_profile(ticker)
                        price   = profile.get("quote", {}).get("price", 0)
                    except Exception:
                        price = 0
                _time.sleep(0.3)
                pt.add_position(user_id, ticker, shares, price)
                added.append(f"{shares:.0f} {ticker} @ ${price:.2f}")
            reply_text = "✅ تم الإضافة:\n" + "\n".join(f"• {a}" for a in added)

        elif not is_analysis and (
            remove_match or _rep.search(r"(?:remove|sell|احذف|بيع)", message, _rep.I)
        ):
            raw_tickers  = _rep.findall(r"\b([A-Z]{2,6})\b", message.upper())
            all_tickers  = [t for t in raw_tickers if t not in _NON_TICKERS]
            removed = []
            for tkr in all_tickers:
                res = pt.remove_position(user_id, tkr)
                if res["success"]:
                    removed.append(tkr)
            reply_text = ("تم حذف: " + ", ".join(removed)) if removed else "لم يتم العثور على الاسهم"

        else:
            summary = pt.get_portfolio_summary(user_id)
            if not summary["success"] or not summary["positions"]:
                reply_text = 'محفظتك فاضية حالياً. أضف أسهم بـ: "أضف 5 AAPL"'
            else:
                s   = summary["summary"]
                pnl_emoji = "📈" if s["total_pnl"] >= 0 else "📉"
                lines_out = [f"📊 **محفظتك** | {s['position_count']} أسهم\n", ""]
                lines_out += [
                    "| السهم | الكمية | سعر الشراء | السعر الحالي | P&L |",
                    "|:-----:|:------:|:----------:|:------------:|:---:|",
                ]
                for p in summary["positions"]:
                    sign = "+" if p["pnl"] >= 0 else ""
                    lines_out.append(
                        f"| {p['ticker']} | {p['shares']} | ${p['purchase_price']:.2f} | "
                        f"${p['current_price']:.2f} | {sign}${p['pnl']:.2f} ({sign}{p['pnl_pct']:.2f}%) |"
                    )
                lines_out.append(
                    f"\n{pnl_emoji} **الإجمالي:** ${s['total_value']:,.2f} | "
                    f"P&L: {'+' if s['total_pnl']>=0 else ''}{s['total_pnl']:.2f} "
                    f"({s['total_pnl_pct']:.2f}%)"
                )
                reply_text = "\n".join(lines_out)

        reply_saver(session_id, user_id, "user", message)
        reply_saver(session_id, user_id, "assistant", reply_text)
        return {
            "reply":      reply_text,
            "session_id": session_id,
            "agent_name": "EisaX Portfolio",
            "model":      "live",
        }
    except Exception as exc:
        logger.warning("Portfolio handler failed: %s", exc)
        reply_text = "عذراً، حدث خطأ أثناء بناء المحفظة. حاول مرة أخرى."
        return {"reply": reply_text, "session_id": session_id, "agent_name": "EisaX", "model": "error"}


# ── GENERAL (Gemini) ────────────────────────────────────────────────────────────

async def handle_general(
    orchestrator: Any,
    session_id:   str,
    user_id:      str,
    message:      str,
    instruction:  str,
    user_ctx:     dict,
    chat_history: list = None,
) -> dict:
    """
    Handle general questions via Gemini with Playbook + optional memory/RAG context.
    """
    try:
        from core.orchestrator import MEMORY_ENABLED, GEMINI_MODEL
    except Exception:
        MEMORY_ENABLED = False
        GEMINI_MODEL   = "gemini-2.0-flash"

    _user_name = user_ctx.get("name") if user_ctx else None

    _greeting_words = {
        "مرحبا","ازيك","ازى","ازيكو","هاي","هالو","اهلا","أهلا","سلام","صباح","مساء",
        "hi","hello","hey","morning","evening","howdy","sup","yo",
    }
    is_greeting = (
        instruction.lower().strip() in ("greeting", "greet")
        or any(w in message.lower().split() for w in _greeting_words)
        or len(message.strip()) < 12
    )

    memory_context = ""
    rag_context    = ""

    if is_greeting:
        name_part     = f" يا {_user_name}" if _user_name else ""
        system_prompt = (
            "You are EisaX, an AI financial assistant built by Ahmed Eisa.\n"
            "You speak Arabic and English. Reply warmly and briefly in the same language as the user.\n"
            f"If the user greets you, greet them back naturally{name_part} — "
            "NO date, NO profile summary, NO financial context dump.\n"
            "Just a friendly, short greeting and offer to help."
        )
    else:
        name_hint = (
            f"User's name is {_user_name} — use it naturally once if relevant."
            if _user_name else ""
        )
        from core.config import PLAYBOOK_PATH as _cfg_pb
        pb_path = os.getenv("PLAYBOOK_PATH", str(_cfg_pb))
        pb_content = ""
        try:
            with open(pb_path, encoding="utf-8") as pbf:
                pb_content = pbf.read()
        except Exception:
            pass

        system_prompt = (
            "You are EisaX — an institutional-grade AI investment intelligence system "
            "built by Ahmed Eisa.\n"
            "You speak Arabic and English fluently. Reply in the same language as the user.\n"
            f"{name_hint}\n\n"
            "OPERATING RULES — follow these exactly:\n"
            + (pb_content if pb_content else "Be direct, numbers-first, never guess prices, never use filler phrases.")
        )

        if MEMORY_ENABLED:
            try:
                from core.memory import format_ctx_for_prompt, format_memory_for_prompt
                memory_context = (
                    format_ctx_for_prompt(user_ctx) if user_ctx
                    else format_memory_for_prompt(user_id)
                )
            except Exception:
                pass

        # RAG — skip for price/stock queries to avoid stale price hallucinations
        _price_kws = ["analyze","analysis","price","stock","gold","xauusd","xau",
                      "silver","oil","btc","bitcoin","crypto","حلل","سعر","تحليل"]
        if not any(w in message.lower() for w in _price_kws):
            try:
                from core.vector_memory import get_rag_context
                rag_context = get_rag_context(message, max_chars=800)
            except Exception:
                pass

    # Build conversation history block (last 6 messages = 3 turns)
    history_block = ""
    if chat_history:
        _hist_lines = []
        for m in chat_history[-6:]:
            _role = "المستخدم" if m.get("role") == "user" else "EisaX"
            _content = (m.get("content") or "")[:300]  # cap per message
            _hist_lines.append(f"{_role}: {_content}")
        if _hist_lines:
            history_block = "\n\n## سياق المحادثة السابقة:\n" + "\n".join(_hist_lines) + "\n"

    full_prompt = f"{system_prompt}\n\n{memory_context}\n{rag_context}{history_block}\nUser: {message}\n\nAssistant:"

    if not orchestrator.gemini_client:
        reply_text = "عذراً، خدمة الذكاء الاصطناعي غير متاحة حالياً."
    else:
        try:
            reply_text = (
                orchestrator._gemini_generate(full_prompt, label="GENERAL")
                or "عذراً، لم أستطع فهم السؤال."
            )
        except Exception as exc:
            logger.error("Gemini failed: %s", exc)
            reply_text = "عذراً، حدث خطأ مؤقت. حاول مرة أخرى."

    orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
    orchestrator.session_mgr.save_message(session_id, user_id, "assistant", reply_text)
    try:
        if MEMORY_ENABLED:
            from core.memory import extract_and_save_user_facts
            extract_and_save_user_facts(user_id, message, reply_text)
    except Exception:
        pass

    return {
        "reply":      reply_text,
        "response":   reply_text,
        "session_id": session_id,
        "agent_name": "EisaX AI",
        "model":      GEMINI_MODEL,
    }
