import os
import time as _time
import httpx
from dotenv import load_dotenv

load_dotenv()
import logging
import re
from typing import Dict, Optional
from core.session_manager import SessionManager
from datetime import datetime as _dt

logger = logging.getLogger(__name__)

# ── Admin Handler ─────────────────────────────────────────────────────────────
try:
    from core.admin_handler import (
        unlock_admin, is_admin_active, lock_admin,
        read_file, read_logs, append_playbook, write_file,
        store_pending_modification, get_pending_modification,
        clear_pending_modification, is_confirmation, is_rejection,
    )
    ADMIN_ENABLED = True
    logger.info("[Admin] admin_handler loaded OK")
except Exception as _adm_e:
    logger.warning("[Admin] admin_handler unavailable: %s — admin mode disabled", _adm_e)
    ADMIN_ENABLED = False


# Phase-1 refactor: _retry now lives in core.utils — import as alias for back-compat
from core.utils import retry as _retry  # noqa: F401

# Router prompt (was previously defined inline in old orchestrator; now lives in prompt_manager)
try:
    from core.prompt_manager import ROUTER_PROMPT
except Exception as _rp_e:
    logger.warning("[Orchestrator] prompt_manager unavailable (%s) — using fallback ROUTER_PROMPT", _rp_e)
    ROUTER_PROMPT = (
        "You are EisaX's routing layer. Classify the user message into one of: "
        "STOCK_ANALYSIS, PORTFOLIO, BOND, CRYPTO, MACRO, GENERAL.\n"
        "Respond with ONLY the category label.\n\nMessage: {message}"
    )

try:
    from core.memory_manager import (
        get_user_memory, format_memory_for_prompt,
        save_stock_analysis, extract_and_save_user_facts,
        get_rich_user_context, format_ctx_for_prompt, format_memory_for_prompt_rich,
        track_stock_interest
    )
    MEMORY_ENABLED = True
except Exception as e:
    logger.warning("Memory manager unavailable: %s", e)
    MEMORY_ENABLED = False

# ═══════════════════════════════════════════════════════════════════════════
# GEMINI Configuration — Primary & Backup Models for High Reliability
# ═══════════════════════════════════════════════════════════════════════════

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
"""Main API key for Gemini LLM. Read from .env file."""

GEMINI_API_KEY_BACKUP = os.getenv("GEMINI_API_KEY_BACKUP", "")
"""Backup API key for fallback scenarios if primary fails."""

# Primary model: Ultra-fast, economical, for routing and general queries
# Avg latency: ~200ms | Cost: Low | Use: Intent classification, general chat
GEMINI_MODEL = "gemini-2.5-flash"

# Backup model: More capable, for complex reasoning when primary fails
# Avg latency: ~800ms | Cost: Standard | Use: Fallback analysis, complex prompts
GEMINI_MODEL_BACKUP = "gemini-2.5-flash"

# ?? Kimi (Moonshot) Configuration - Primary Maestro ?????????????????????
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")
KIMI_MODEL = "kimi-k2.5"

# ═══════════════════════════════════════════════════════════════════════════
# GEMINI Configuration — Primary & Backup Models for High Reliability
# ═══════════════════════════════════════════════════════════════════════════


def _extract_verdict_from_reply(reply: str, ticker: str) -> dict:
    """
    #7 Fact-check: extract price + verdict from reply and validate against stock_memory.
    Returns dict with verdict, price, is_valid flag.
    """
    result = {"verdict": None, "price": None, "ticker": ticker, "is_valid": True}
    try:
        # Extract verdict
        verdict_match = re.search(r"\b(BUY|SELL|HOLD|REDUCE|STRONG BUY|STRONG SELL)\b", reply.upper())
        if verdict_match:
            result["verdict"] = verdict_match.group(1)

        # Extract price - look for patterns like $123.45 or 123.45 USD
        price_patterns = [
            r"\$([\d,]+\.?\d*)",
            r"([\d,]+\.?\d*)\s*(?:USD|\$)",
            r"(?:price|سعر|السعر)[^\d]*([\d,]+\.?\d*)",
        ]
        for pat in price_patterns:
            m = re.search(pat, reply, re.IGNORECASE)
            if m:
                try:
                    price_str = m.group(1).replace(",", "")
                    price = float(price_str)
                    if 0.01 < price < 1_000_000:  # sanity range
                        result["price"] = price
                        break
                except ValueError:
                    continue

        # Fact-check: compare extracted price vs stock_memory last price
        if result["price"] and MEMORY_ENABLED:
            try:
                from core.memory_manager import get_stock_memory
                mem = get_stock_memory(ticker)
                if mem and mem.get("last_price") and mem["last_price"] > 0:
                    last_known = float(mem["last_price"])
                    diff_pct = abs(result["price"] - last_known) / last_known * 100
                    if diff_pct > 20:
                        logger.warning(
                            "[FactCheck] %s price in reply $%.2f differs %.1f%% from last known $%.2f",
                            ticker, result["price"], diff_pct, last_known
                        )
                        result["is_valid"] = False
            except Exception:
                pass
    except Exception as e:
        logger.debug("[FactCheck] extraction failed: %s", e)
    return result


def _save_analysis_to_memory(user_id: str, ticker: str, reply: str):
    """
    #8 Memory: save stock analysis verdict+price to stock_memory after every analysis.
    Also saves ticker to user watchlist.
    """
    if not MEMORY_ENABLED or not ticker or ticker == "UNKNOWN":
        return
    try:
        fact = _extract_verdict_from_reply(reply, ticker)
        verdict = fact.get("verdict") or "HOLD"
        price = fact.get("price") or 0.0
        # Save summary = first 300 chars of reply (the conclusion section)
        summary = reply[:300].replace("\n", " ").strip()
        from core.memory_manager import save_stock_analysis
        # Pass user_id so per-user history is saved alongside global stock_memory
        save_stock_analysis(ticker.upper(), verdict, price, summary, user_id=user_id)
        logger.info("[Memory] Saved analysis: %s → %s @ $%.2f (user=%s)", ticker, verdict, price, user_id)

        # #9 Brain: persist prediction so the learning engine can track accuracy
        if price and price > 0 and verdict in (
            "BUY", "SELL", "HOLD", "ACCUMULATE", "REDUCE", "STRONG BUY", "STRONG SELL"
        ):
            try:
                from core.brain import save_prediction
                save_prediction(ticker=ticker.upper(), verdict=verdict, price=price, horizon=30)
                logger.info("[Brain] Prediction saved: %s verdict=%s price=%.2f", ticker, verdict, price)
            except Exception as _bp_e:
                logger.debug("[Brain] save_prediction failed: %s", _bp_e)
    except Exception as e:
        logger.debug("[Memory] save_analysis failed: %s", e)


class MultiAgentOrchestrator:
    def __init__(self, db_path: str = "investwise.db"):
        self.session_mgr = SessionManager(db_path)
        self.gemini_client = None
        self.gemini_client_backup = None
        try:
            from google import genai
            self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("Gemini primary client initialized: %s", GEMINI_MODEL)
            if GEMINI_API_KEY_BACKUP:
                self.gemini_client_backup = genai.Client(api_key=GEMINI_API_KEY_BACKUP)
                logger.info("Gemini backup client initialized: %s", GEMINI_MODEL_BACKUP)
        except Exception as e:
            logger.error("Gemini Client init failed: %s", e)

        self.kimi_client = None
        if MOONSHOT_API_KEY:
            try:
                from openai import OpenAI
                self.kimi_client = OpenAI(
                    api_key=MOONSHOT_API_KEY,
                    base_url="https://api.moonshot.ai/v1"
                )
                logger.info("[Kimi] Client initialized successfully with model: %s", KIMI_MODEL)
            except Exception as e:
                logger.error("[Kimi] Client init failed: %s", e)
                self.kimi_client = None

        # Full async I/O migration: shared AsyncClient for DFM and Bond handlers.
        self.httpx_client = httpx.AsyncClient()

        self._financial_agent = None

    async def aclose(self):
        client = getattr(self, "httpx_client", None)
        if client and not client.is_closed:
            await client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    def __del__(self):
        client = getattr(self, "httpx_client", None)
        if not client or client.is_closed:
            return
        try:
            import asyncio
            loop = asyncio.get_running_loop()
        except Exception:
            try:
                import asyncio
                asyncio.run(client.aclose())
            except Exception:
                pass
        else:
            try:
                loop.create_task(client.aclose())
            except Exception:
                pass

    @property
    def financial_agent(self):
        if self._financial_agent is None:
            try:
                from core.agents.finance import FinancialAgent
                self._financial_agent = FinancialAgent()
            except Exception as e:
                logger.warning("FinancialAgent init failed: %s", e)
        return self._financial_agent

    def _gemini_generate(self, contents: str, *, label: str = "") -> str:
        """
        Generate LLM response with automatic fallback mechanism.
        
        High-reliability pattern:
        1. Primary: gemini-3.1-flash-lite (fast, economical)
        2. Backup: gemini-2.5-flash (slower, more capable)
        3. Retry: Up to 2 attempts per model with exponential backoff
        
        Args:
            contents (str): Prompt/message to send to LLM
            label (str): Label for logging (e.g., "router", "admin"), optional
            
        Returns:
            str: LLM response text
            
        Raises:
            RuntimeError: If both primary and backup clients fail
            
        Example:
            >>> response = orch._gemini_generate("Say hello", label="greeting")
            >>> response
            "Hello! How can I help you?"
        """
        # ── Generation config: high output limit for long portfolio/analysis replies ──
        try:
            from google.genai import types as _gtypes
            _gen_cfg = _gtypes.GenerateContentConfig(max_output_tokens=8192, temperature=0.7)
        except Exception:
            _gen_cfg = None

        # ── ATTEMPT 1: Primary Model (Fast & Economical) ──
        try:
            resp = _retry(
                lambda: self.gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    **( {"config": _gen_cfg} if _gen_cfg else {} )
                ),
                max_attempts=2, base_delay=0.3
            )
            return (resp.text or "").strip()
        except Exception as e:
            logger.warning("[Gemini] Primary (%s) failed%s: %s — attempting backup",
                           GEMINI_MODEL, f" [{label}]" if label else "", e)

        # ── ATTEMPT 2: Backup Model (Slower but More Capable) ──
        if self.gemini_client_backup:
            try:
                resp = _retry(
                    lambda: self.gemini_client_backup.models.generate_content(
                        model=GEMINI_MODEL_BACKUP,
                        contents=contents,
                        **( {"config": _gen_cfg} if _gen_cfg else {} )
                    ),
                    max_attempts=2, base_delay=0.3
                )
                logger.info("[Gemini] Backup (%s) succeeded%s",
                            GEMINI_MODEL_BACKUP, f" [{label}]" if label else "")
                return (resp.text or "").strip()
            except Exception as e2:
                logger.error("[Gemini] Backup (%s) also failed%s: %s",
                             GEMINI_MODEL_BACKUP, f" [{label}]" if label else "", e2)
                raise
        raise RuntimeError(f"No Gemini backup available for [{label}]")

    def _maestro_route_generate(self, prompt: str) -> str:
        """Primary router: Kimi maestro, then DeepSeek, then Gemini fallback."""
        # Primary: Kimi K2.5
        if self.kimi_client:
            try:
                resp = _retry(
                    lambda: self.kimi_client.chat.completions.create(
                        model=KIMI_MODEL,
                        messages=[
                            {"role": "system", "content": "You are EisaX router maestro. Return strict JSON only."},
                            {"role": "user", "content": prompt},
                        ],
                        # Kimi K2.5 only supports temperature=1 — do not pass other values
                        temperature=1,
                        # Force JSON output — prevents prose wrapping around the JSON
                        response_format={"type": "json_object"},
                    ),
                    max_attempts=2,
                    base_delay=0.3,
                )
                content = ((resp.choices[0].message.content or "") if resp and resp.choices else "").strip()
                if content:
                    return content
            except Exception as e:
                logger.warning("[Router] Kimi failed: %s ? falling back to DeepSeek", e)

        # Fallback 1: DeepSeek (sync httpx; this function is sync by design)
        ds_key = os.getenv("DEEPSEEK_API_KEY", "")
        if ds_key:
            try:
                with httpx.Client(timeout=30.0) as client:
                    r = client.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                        json={
                            "model": "deepseek-chat",
                            "messages": [
                                {"role": "system", "content": "You are EisaX router maestro. Return strict JSON only."},
                                {"role": "user", "content": prompt},
                            ],
                            "max_tokens": 800,
                            "temperature": 0.1,
                        },
                    )
                r.raise_for_status()
                resp = r.json()
                if "choices" in resp:
                    content = (resp["choices"][0]["message"]["content"] or "").strip()
                    if content:
                        return content
            except Exception as e:
                logger.warning("[Router] DeepSeek fallback failed: %s ? falling back to Gemini", e)

        # Fallback 2: Gemini
        try:
            raw = self._gemini_generate(prompt, label="router-gemini-fallback")
            if raw:
                return raw
        except Exception as e:
            logger.error("[Router] Gemini fallback failed: %s", e)

        # All 3 LLMs failed — return safe fallback JSON instead of crashing
        logger.error("[Router] All models failed (Kimi/DeepSeek/Gemini) — using GENERAL fallback")
        return '{"route":"GENERAL","handler":"GENERAL","instruction":"' + \
               prompt[:100].replace('"', '') + '","clarification_question":""}'

    async def _run_sync_in_executor(self, fn, *args, **kwargs):
        """Temporary bridge for sync I/O until full migration to httpx.AsyncClient."""
        import asyncio
        import functools
        loop = asyncio.get_running_loop()
        bound = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(None, bound)

    # ═══════════════════════════════════════════════════════════════════════════
    # SSE STREAMING — yields SSE-formatted events for /v1/chat/stream endpoint
    # ═══════════════════════════════════════════════════════════════════════════

    async def stream_process_message(
        self, user_id: str, message: str, session_id: Optional[str] = None
    ):
        """
        Async generator for Server-Sent Events streaming.

        Yields SSE lines:
          data: {"type":"status","text":"..."}   ← progress indicator
          data: {"type":"token","text":"..."}    ← text chunk
          data: {"type":"done","meta":{...}}     ← final metadata

        Strategy:
          GENERAL route   → true Gemini streaming (token-by-token)
          All other routes → progress events while processing, then
                             natural-chunk streaming of the completed reply
        """
        import json as _j
        import asyncio as _aio
        import re as _re

        def _sse(type_: str, payload) -> str:
            return f"data: {_j.dumps({'type': type_, 'text': payload}, ensure_ascii=False)}\n\n"

        if not session_id:
            session_id = self.session_mgr.get_or_create_session(user_id)

        # ── Immediate acknowledgment ───────────────────────────────────────────
        yield _sse("status", "⚡ EisaX يفكر...")

        # ── Fast-path: GENERAL (greetings, capability questions) ──────────────
        try:
            from core.services.routing_service import is_greeting
            _is_greet = is_greeting(message)
        except Exception:
            _is_greet = len(message.strip()) < 12

        if _is_greet:
            yield _sse("status", "💬 ...")
            _reply_chunks = []
            try:
                # Run Gemini in executor (sync SDK), collect tokens
                _system = (
                    "You are EisaX, an AI financial assistant built by Ahmed Eisa. "
                    "Reply warmly and briefly in the same language as the user."
                )
                _prompt = f"{_system}\n\nUser: {message}\n\nAssistant:"
                _raw = await self._run_sync_in_executor(
                    self._gemini_generate, _prompt, label="stream-greeting"
                )
                _reply_chunks = [_raw] if _raw else []
            except Exception as _ge:
                logger.warning("[Stream] Gemini greeting failed: %s", _ge)

            _full_reply = "".join(_reply_chunks)
            # Stream word-by-word for greeting
            for _word in (_full_reply + " ").split(" "):
                if _word:
                    yield _sse("token", _word + " ")
                    await _aio.sleep(0.03)
            self.session_mgr.save_message(session_id, user_id, "user", message)
            self.session_mgr.save_message(session_id, user_id, "assistant", _full_reply)
            yield _sse("done", _j.dumps({"session_id": session_id, "agent": "EisaX AI"}, ensure_ascii=False))
            return

        # ── Complex routes: status heartbeat + background processing ──────────
        _status_sequence = [
            "🔍 تصنيف الطلب...",
            "📊 جلب البيانات...",
            "🧠 التحليل جارٍ...",
            "✍️ إعداد التقرير...",
        ]
        _status_idx = 0

        # Launch process_message as a background task
        _task = _aio.create_task(
            self.process_message(user_id, message, session_id)
        )

        # Send heartbeat status messages while waiting
        while not _task.done():
            await _aio.sleep(2.5)
            if _status_idx < len(_status_sequence):
                yield _sse("status", _status_sequence[_status_idx])
                _status_idx += 1

        # Retrieve result (re-raise if task raised exception)
        try:
            _result = _task.result()
        except Exception as _te:
            logger.error("[Stream] process_message task failed: %s", _te)
            yield _sse("error", "⚠️ حدث خطأ أثناء المعالجة. حاول مرة أخرى.")
            yield _sse("done", _j.dumps({"session_id": session_id}, ensure_ascii=False))
            return

        _reply = _result.get("reply", "")
        if not _reply:
            yield _sse("error", "⚠️ لم أتمكن من معالجة طلبك. حاول مرة أخرى.")
            yield _sse("done", _j.dumps({"session_id": session_id}, ensure_ascii=False))
            return

        yield _sse("status", "✅ جاهز")

        # ── Stream reply in natural sentence/line chunks ───────────────────────
        # Split on sentence boundaries while preserving markdown structure
        _chunks = _re.split(r'(?<=[\.\!\?؟\n])\s*', _reply)
        for _chunk in _chunks:
            if _chunk.strip():
                yield _sse("token", _chunk)
                # Brief pause between chunks — gives natural reading feel
                await _aio.sleep(0.015)

        # ── Done event with full metadata ──────────────────────────────────────
        yield _sse("done", _j.dumps({
            "session_id": session_id,
            "agent":      _result.get("agent_name", "EisaX"),
            "model":      _result.get("model", ""),
            "download_url": _result.get("download_url"),
        }, ensure_ascii=False))

    def _extract_ticker(self, message: str) -> str:
        """
        Extract stock ticker symbol from user message (AR/EN/Company names).
        
        Supports:
        - English: "Apple" → AAPL
        - Arabic: "نيفيديا" → NVDA  
        - Local markets: "ارامكو" → 2222.SR, "اعمار" → EMAAR.DU
        - Crypto: "بيتكوين" → BTC-USD
        
        Args:
            message (str): User message in Arabic or English
            
        Returns:
            str: Stock ticker (e.g., "AAPL") or "UNKNOWN" if not found
            
        Example:
            >>> orch._extract_ticker("حللى سهم ابل")
            "AAPL"
            >>> orch._extract_ticker("Analyze Tesla stock")
            "TSLA"
        """
        try:
            prompt = f"""Extract the stock ticker symbol from this message.
Return ONLY the ticker symbol (e.g. NVDA, AAPL, TSLA).
If it's a company name, return its ticker. If Arabic name, translate then return ticker.
If no specific stock mentioned, return UNKNOWN.

Message: {message}

Ticker:"""
            raw = self._gemini_generate(prompt, label="extract_ticker")
            parts = raw.upper().split()
            if not parts:
                return "UNKNOWN"
            import re
            ticker = re.sub(r'[^A-Z0-9\.]', '', parts[0])
            return ticker if len(ticker) >= 2 else "UNKNOWN"
        except Exception as e:
            logger.warning("Ticker extraction failed: %s", e)
            return "UNKNOWN"

    def _classify_intent(self, message: str, chat_history: list = None) -> tuple:
        """
        Classify user intent and route to appropriate handler.
        
        Understands Arabic and English messages and determines:
        - route: STOCK_ANALYSIS | FINANCIAL | PORTFOLIO | GENERAL | CLARIFY
        - handler: Specific agent to call (CIO_ANALYSIS, PORTFOLIO_OPTIMIZE, etc.)
        - instruction: Normalized English instruction
        - clarification: Question if intent is unclear
        
        Args:
            message (str): User message in any language
            
        Returns:
            tuple: (route, handler, instruction, clarification_question)
            
        Example:
            >>> route, handler, inst, _ = orch._classify_intent("عايز محفظة عدوانية")
            >>> route, handler
            ("FINANCIAL", "PORTFOLIO_OPTIMIZE")
        """
        import json, re
        try:
            # Build conversation context block for smarter routing
            _ctx_block = ""
            if chat_history:
                _recent = chat_history[-6:]
                _ctx_lines = []
                for m in _recent:
                    _r = "User" if m.get("role") == "user" else "AI"
                    _c = (m.get("content") or "")[:150]
                    _ctx_lines.append(f"{_r}: {_c}")
                if _ctx_lines:
                    _ctx_block = "\nRecent conversation:\n" + "\n".join(_ctx_lines) + "\n"

            prompt = ROUTER_PROMPT.format(message=(_ctx_block + "Current message: " + message))
            raw = self._maestro_route_generate(prompt)
            if not raw:
                return "GENERAL", "GENERAL", message, ""
            text = re.sub(r"```json|```", "", raw).strip()
            # Valid route + handler values
            _VALID_ROUTES   = {"STOCK_ANALYSIS", "FINANCIAL", "PORTFOLIO", "SCREENER", "GENERAL", "CLARIFY", "CRYPTO", "MACRO", "BOND"}
            _VALID_HANDLERS = {"STOCK_ANALYSIS", "FINANCIAL", "PORTFOLIO", "SCREENER", "GENERAL", "CLARIFY",
                               "CIO_ANALYSIS", "PORTFOLIO_OPTIMIZE", "BOND", "DFM", "CRYPTO", "MACRO"}
            # Route→handler default mapping when handler is missing or invalid
            _ROUTE_HANDLER_MAP = {
                "STOCK_ANALYSIS": "STOCK_ANALYSIS",
                "FINANCIAL":      "CIO_ANALYSIS",
                "PORTFOLIO":      "PORTFOLIO_OPTIMIZE",
                "SCREENER":       "SCREENER",
                "BOND":           "BOND",
                "CRYPTO":         "STOCK_ANALYSIS",
                "MACRO":          "GENERAL",
                "GENERAL":        "GENERAL",
                "CLARIFY":        "GENERAL",
            }
            try:
                result = json.loads(text)
                route = result.get("route", "GENERAL").upper()
                if route not in _VALID_ROUTES:
                    route = "GENERAL"
                raw_handler = result.get("handler", "").upper()
                # Use explicit handler if valid, else derive from route
                handler = raw_handler if raw_handler in _VALID_HANDLERS else _ROUTE_HANDLER_MAP.get(route, "GENERAL")
                instruction = result.get("instruction", message)
                clarification = result.get("clarification_question", "")
                logger.info("[Brain] route=%s | handler=%s | %s", route, handler, instruction[:80])
                return route, handler, instruction, clarification
            except Exception:
                # Non-JSON fallback: parse first word as label
                label = text.split()[0].upper() if text else "GENERAL"
                if label not in _VALID_ROUTES:
                    label = "GENERAL"
                return label, _ROUTE_HANDLER_MAP.get(label, "GENERAL"), message, ""
        except Exception as e:
            logger.warning("Brain failed: %s", e)
            return "GENERAL", "GENERAL", message, ""


    # Full async I/O migration: use httpx.AsyncClient first, keep requests+executor as safety fallback.
    async def _handle_dfm_query(self, message: str, dfm_context: str) -> str:
        """Fast-path DFM analysis using DeepSeek + local fundamentals. No blocking scraping."""
        import os, requests
        from datetime import datetime as _dt
        today = _dt.now().strftime("%B %d, %Y")

        system_prompt = (
            f"You are EisaX AI, Chief Investment Officer — built by Eng. Ahmed Eisa.\n"
            f"Today: {today}\n\n"
            "You are analyzing a Dubai Financial Market (DFM) listed stock.\n"
            "Use the provided fundamentals data in your analysis.\n"
            "Structure your response with:\n"
            "1. Company Overview & Key Metrics\n"
            "2. Valuation Analysis (P/E vs sector, market cap)\n"
            "3. Risk Assessment (Beta, liquidity via volume)\n"
            "4. Investment Verdict (Buy/Hold/Sell with reasoning)\n"
            "5. Key Risks\n"
            "Tone: Professional CIO-style, numbers-first, markdown format.\n\n"
            f"{dfm_context}"
        )

        ds_key = os.getenv("DEEPSEEK_API_KEY", "")
        if ds_key:
            try:
                try:
                    if not self.httpx_client:
                        raise RuntimeError("httpx AsyncClient is unavailable")
                    r = await self.httpx_client.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                        json={
                            "model": "deepseek-chat",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": message}
                            ],
                            "max_tokens": 3000,
                            "temperature": 0.3
                        },
                        timeout=60,
                    )
                except Exception as _httpx_e:
                    logger.warning("[DFMHandler] httpx failed: %s; using requests fallback", _httpx_e)
                    r = await self._run_sync_in_executor(
                        requests.post,
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                        json={
                            "model": "deepseek-chat",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": message}
                            ],
                            "max_tokens": 3000,
                            "temperature": 0.3
                        },
                        timeout=60,
                    )
                resp = r.json()
                if "choices" in resp:
                    return resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"[DFMHandler] DeepSeek failed: {e}")

        # Gemini fallback (primary → backup automatic)
        try:
            return self._gemini_generate(f"{system_prompt}\n\nUser: {message}", label="DFM")
        except Exception as e:
            logger.warning(f"[DFMHandler] Gemini failed: {e}")
            return dfm_context  # Return raw data as last resort

    # Full async I/O migration: use httpx.AsyncClient first, keep requests+executor as safety fallback.
    async def _handle_bond_query(self, message: str) -> str:
        """
        CIO-grade fixed income analysis.
        - If message contains an ISIN → delegate to FinancialAgent._handle_fixed_income()
          which uses OpenFIGI, CDS spread, live rating, YTM, and the new scoring model.
        - Otherwise → existing sovereign yield-curve analysis path.
        """
        import os, requests
        from datetime import datetime as _dt

        # ── ISIN fast-path: delegate to the new Fixed Income Engine ─────────
        try:
            from core.fixed_income import extract_isin, is_fixed_income_query
            isin = extract_isin(message)
            if isin:
                logger.info("[BondHandler] ISIN detected (%s) → FixedIncomeEngine", isin)
                from core.agents.finance import FinancialAgent
                _fi_agent = FinancialAgent()
                result = _fi_agent._handle_fixed_income(message, {})
                return result.get("reply", "")
        except Exception as _fi_err:
            logger.warning("[BondHandler] FixedIncomeEngine delegation failed: %s", _fi_err)
        # ── End ISIN fast-path ────────────────────────────────────────────────

        today = _dt.now().strftime("%B %d, %Y")

        country_hints = {
            "egypt": ["egypt", "egyptian", "مصر", "مصرية", "egp", "cbe"],
            "usa": ["us treasury", "treasuries", "t-bill", "fed", "federal reserve"],
            "uae": ["uae", "emirates", "درهم"],
            "saudi": ["saudi", "ksa", "سعودية", "ساما"],
        }
        detected_country = "general"
        msg_lower = message.lower()
        for country, hints in country_hints.items():
            if any(h in msg_lower for h in hints):
                detected_country = country
                break

        country_context = {
            "egypt": (
                "Focus on Egypt-specific context:\n"
                "- CBE rate decisions and EGP inflation\n"
                "- EGP/USD FX risk and devaluation history\n"
                "- IMF program impact on sovereign credibility\n"
                "- Compare: EGP T-bills (short, ~25-28% yield) vs USD Eurobonds\n"
                "- Moody's/S&P/Fitch credit rating for Egypt"
            ),
            "usa": (
                "Focus on US Treasuries:\n"
                "- Fed funds rate and yield curve shape\n"
                "- T-bills vs T-notes vs T-bonds duration tradeoff\n"
                "- Real yield after CPI inflation"
            ),
            "uae": (
                "Focus on UAE fixed income:\n"
                "- UAE sovereign bonds and Sukuk\n"
                "- AED/USD peg — minimal FX risk\n"
                "- Abu Dhabi vs Dubai issuances"
            ),
            "general": (
                "Provide a comparative fixed income overview:\n"
                "- Sovereign vs corporate bonds\n"
                "- Short vs long duration tradeoffs\n"
                "- Credit rating impact on yield\n"
                "- FX risk for foreign-currency bonds"
            ),
        }.get(detected_country, "")

        # ─── Fetch live bond data ───
        try:
            from core.bond_data_fetcher import get_bond_data
            bond_data = get_bond_data(message)
            live_data_block = bond_data.get("prompt_block", "")
            logger.info(f"[BondHandler] {bond_data['country_name']} | {bond_data['source']} | {len(bond_data.get('yields',[]))} maturities")
        except Exception as _bde:
            live_data_block = ""
            logger.warning(f"[BondHandler] Data fetch failed: {_bde}")

        system_prompt = (
            f"You are EisaX AI, Chief Investment Officer — built by Eng. Ahmed Eisa.\n"
            f"Today: {today}\n\n"
            "You are a fixed income specialist. When asked about bonds:\n"
            "- Analyze objectively: yield, duration, credit risk, FX risk, liquidity\n"
            "- Compare options clearly (short-term bills vs long-term bonds)\n"
            "- Give a CIO-style verdict: which is more suitable and why\n"
            "- End with a brief risk disclaimer\n"
            "- NEVER refuse bond questions — this is institutional-grade educational analysis\n\n"
            f"{country_context}\n\n"
            "Tone: Professional, direct, numbers-first. Use markdown."
        )

        ds_key = os.getenv("DEEPSEEK_API_KEY", "")
        if ds_key:
            try:
                try:
                    if not self.httpx_client:
                        raise RuntimeError("httpx AsyncClient is unavailable")
                    r = await self.httpx_client.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                        json={
                            "model": "deepseek-chat",
                            "messages": [
                                {"role": "system", "content": system_prompt + ("\n\n" + live_data_block if live_data_block else "")},
                                {"role": "user", "content": message}
                            ],
                            "max_tokens": 8000,
                            "temperature": 0.3
                        },
                        timeout=60,
                    )
                except Exception as _httpx_e:
                    logger.warning("[BondHandler] httpx failed: %s; using requests fallback", _httpx_e)
                    r = await self._run_sync_in_executor(
                        requests.post,
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                        json={
                            "model": "deepseek-chat",
                            "messages": [
                                {"role": "system", "content": system_prompt + ("\n\n" + live_data_block if live_data_block else "")},
                                {"role": "user", "content": message}
                            ],
                            "max_tokens": 8000,
                            "temperature": 0.3
                        },
                        timeout=60,
                    )
                resp = r.json()
                if "choices" in resp:
                    return resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"[BondHandler] DeepSeek failed: {e}")

        try:
            bond_contents = f"{system_prompt}{chr(10)+chr(10)+live_data_block if live_data_block else ''}\n\nUser: {message}"
            return self._gemini_generate(bond_contents, label="Bond")
        except Exception as e:
            logger.error(f"[BondHandler] Gemini fallback failed: {e}")
            return "Unable to retrieve bond analysis at this time. Please try again."

    def _gemini_think(self, message: str) -> dict:
        """Gemini classifies intent before routing."""
        try:
            from datetime import datetime as _dt
            import json, re
            today = _dt.now().strftime("%B %d, %Y")
            think_prompt = f"""Classify this message intent for EisaX financial assistant.
Today: {today}
Message: "{message}"
Return ONLY JSON (no markdown):
{{"intent": "stock_analysis|portfolio|export|general", "ticker": "NVDA or null", "confidence": 0.95, "reasoning": "one sentence"}}
Rules:
- stock_analysis: any question about a specific stock/crypto price/analysis/outlook
- portfolio: multi-asset allocation, diversification, portfolio building
- export: download/save/export report
- general: greetings, explanations, other
Arabic examples: "فين سعر ابل"→stock_analysis/AAPL, "حلل NVDA"→stock_analysis/NVDA"""
            raw = self._gemini_generate(think_prompt, label="think")
            if not raw:
                return {"intent": "general", "ticker": None, "confidence": 0.5}
            text = re.sub(r"```json|```", "", raw).strip()
            result = json.loads(text)
            logger.info("[Think] intent=%s, ticker=%s, conf=%s", result.get('intent'), result.get('ticker'), result.get('confidence'))
            return result
        except Exception as e:
            logger.warning("[Think] failed: %s", e)
            return {"intent": "general", "ticker": None, "confidence": 0.5}

    async def process_message(self, user_id: str, message: str, session_id: Optional[str] = None) -> Dict:
        """
        Main message dispatcher — thin coordinator that delegates to service modules.

        Routing order
        ─────────────
        1. Admin mode          (admin_orchestrator)
        2. Export fast-path    (export_service)
        3. File analysis       (routing_service)
        4. DFM screen          (routing_service)
        5. Bond detection      (routing_service)
        6. Greeting fast-path  (routing_service → GENERAL)
        7. Arabic ticker       (routing_service → STOCK_ANALYSIS)
        8. Gemini router       (_classify_intent)
        9. Route handlers      (market_route_handler)
        """
        from core.services.admin_orchestrator import handle_admin_mode as _handle_admin
        from core.services.export_service      import handle_export     as _handle_export
        from core.services.routing_service     import (
            is_export_request, is_file_analysis, is_bond_request, is_greeting,
            detect_arabic_ticker, detect_dfm_screen, handle_file_analysis,
        )
        from core.services.market_route_handler import (
            handle_stock_analysis as _handle_stock,
            handle_financial      as _handle_financial,
            handle_screening      as _handle_screening,
            handle_portfolio      as _handle_portfolio,
            handle_general        as _handle_general,
        )

        try:
            if not session_id:
                session_id = self.session_mgr.get_or_create_session(user_id)

            # ── 1. Admin Mode ──────────────────────────────────────────────────
            if ADMIN_ENABLED:
                _adm_resp = await _handle_admin(self, session_id, user_id, message)
                if _adm_resp is not None:
                    return _adm_resp

            # ── 2. Export fast-path ────────────────────────────────────────────
            if is_export_request(message):
                history = self.session_mgr.get_chat_history(session_id)
                return await _handle_export(message, history, session_id)

            # ── 3. File Analysis ───────────────────────────────────────────────
            if is_file_analysis(message):
                reply, label = handle_file_analysis(
                    message,
                    financial_agent=self.financial_agent,
                    gemini_client=self.gemini_client,
                    gemini_model=GEMINI_MODEL,
                    gemini_api_key=GEMINI_API_KEY,
                    user_id=user_id,
                    session_id=session_id,
                )
                self.session_mgr.save_message(session_id, user_id, "user", message[:500])
                self.session_mgr.save_message(session_id, user_id, "assistant", reply)
                return {"reply": reply, "session_id": session_id, "agent_name": label}

            # ── 4. DFM Screening ───────────────────────────────────────────────
            _dfm_screen = detect_dfm_screen(message)
            if _dfm_screen:
                _criterion, _screen_text = _dfm_screen
                logger.info("[DFM] Screening pre-route: %s", _criterion)
                _screen_reply = await self._handle_dfm_query(message, _screen_text)
                self.session_mgr.save_message(session_id, user_id, "user", message)
                self.session_mgr.save_message(session_id, user_id, "assistant", _screen_reply)
                return {
                    "reply": _screen_reply, "session_id": session_id,
                    "agent_name": "EisaX DFM Screen", "model": "deepseek",
                }

            # ── 5a. Clean Portfolio Pipeline (MUST be before bond check) ─────────
            try:
                import sys as _sys, os as _os, re as _re_pp
                _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                if _root not in _sys.path:
                    _sys.path.insert(0, _root)
                from portfolio_pipeline import is_pipeline_request, run as pipeline_run
                if is_pipeline_request(message):
                    # Gate: require min inputs (amount OR risk OR markets) before executing.
                    # Vague requests like "عايز محفظة" must be clarified first.
                    _pp_low = message.lower()
                    _pp_has_amount = bool(_re_pp.search(
                        r'\d[\d,\.]*\s*(k|m|b|الف|مليون|مليار|دولار|ريال|درهم|\$|usd|eur|aed|sar)', _pp_low))
                    _pp_has_risk   = any(w in _pp_low for w in [
                        'aggressive','عدوانية','عدواني','risk','مخاطرة','مخاطر','conservative',
                        'محافظ','متوازن','balanced','moderate','منخفض','متوسط','عالي','عالية'])
                    _pp_has_market = any(w in _pp_low for w in [
                        'uae','dubai','saudi','مصر','egypt','us ','america','global','gulf','خليج',
                        'stocks','bonds','crypto','gold','سندات','اسهم','ذهب','دولي','محلي'])
                    if _pp_has_amount or _pp_has_risk or _pp_has_market:
                        logger.info("[Pipeline] Routing to clean pipeline: %s", message[:60])
                        _pipe_report = pipeline_run(message)
                        if _pipe_report:
                            self.session_mgr.save_message(session_id, user_id, "user", message)
                            self.session_mgr.save_message(session_id, user_id, "assistant", _pipe_report)
                            return {
                                "reply": _pipe_report, "session_id": session_id,
                                "agent_name": "EisaX Portfolio Pipeline", "model": "pipeline+deepseek",
                            }
                    else:
                        logger.info("[Pipeline] Vague portfolio request — skipping pipeline, will clarify")
            except Exception as _pipe_exc:
                logger.warning("[Pipeline] Failed: %s — continuing", _pipe_exc)

            # ── 5. Bond / Fixed Income ─────────────────────────────────────────
            if is_bond_request(message):
                reply_text = await self._handle_bond_query(message)
                self.session_mgr.save_message(session_id, user_id, "user", message)
                self.session_mgr.save_message(session_id, user_id, "assistant", reply_text)
                return {
                    "reply": reply_text, "session_id": session_id,
                    "agent_name": "EisaX Fixed Income Analyst",
                }

            # ── Load recent conversation history for context ───────────────────
            _recent_history: list = []
            try:
                _recent_history = self.session_mgr.get_chat_history(session_id)[-10:] or []
            except Exception:
                pass

            # ── 6 & 7. Greeting / Arabic-ticker fast-paths + Gemini router ────
            _user_ctx: dict = {}
            if is_greeting(message):
                route, handler, instruction, clarification = "GENERAL", "GENERAL", message, ""
                logger.info("[Router] greeting fast-path — skipping classify_intent")
            else:
                try:
                    if MEMORY_ENABLED:
                        _user_ctx = get_rich_user_context(user_id) or {}
                except Exception:
                    pass

                _fast_ticker = detect_arabic_ticker(message)
                if _fast_ticker:
                    route       = "STOCK_ANALYSIS"
                    handler     = "STOCK_ANALYSIS"
                    instruction = f"analyze {_fast_ticker}"
                    clarification = ""
                    logger.info("[Router] Arabic stock fast-path: '%s' → %s", message.strip(), _fast_ticker)
                else:
                    route, handler, instruction, clarification = self._classify_intent(message, chat_history=_recent_history)
                    logger.info("[Router] route=%s | handler=%s | %s", route, handler, instruction[:60])

            # CLARIFY override when ticker is detectable
            if route == "CLARIFY":
                try:
                    from core.intent_classifier import IntentClassifier as _IC
                    _tks = _IC.extract_tickers(message)
                    if _tks:
                        route       = "STOCK_ANALYSIS"
                        handler     = "STOCK_ANALYSIS"
                        instruction = f"analyze {' and '.join(_tks)}"
                        clarification = ""
                        logger.info("[Router] CLARIFY overridden → STOCK_ANALYSIS for tickers: %s", _tks)
                except Exception:
                    pass

            if route == "CLARIFY" and clarification:
                return {"reply": clarification, "session_id": session_id}

            # ── Bond (router-decided) ──────────────────────────────────────────
            if handler == "BOND":
                reply_text = await self._handle_bond_query(message)
                self.session_mgr.save_message(session_id, user_id, "user", message)
                self.session_mgr.save_message(session_id, user_id, "assistant", reply_text)
                return {"reply": reply_text, "session_id": session_id, "agent_name": "EisaX Fixed Income"}

            # ── DFM (router-decided) ───────────────────────────────────────────
            if handler == "DFM":
                _analysis_kw = [
                    "حلل", "تحليل", "analyze", "analysis", "technical", "تقني",
                    "scorecard", "verdict", "invest", "buy", "sell", "hold",
                    "سعر", "price", "target", "هدف", "توصية", "recommendation",
                ]
                if any(kw in message.lower() for kw in _analysis_kw) and self.financial_agent:
                    try:
                        import asyncio as _asyncio
                        _result = await _asyncio.wait_for(
                            self._run_sync_in_executor(
                                self.financial_agent._handle_analytics,
                                session_id, {"user_ctx": _user_ctx, "user_id": user_id}, message,
                            ),
                            timeout=120,
                        )
                        _rt = _result.get("reply", "")
                        if _rt and len(_rt) > 200:
                            self.session_mgr.save_message(session_id, user_id, "user", message)
                            self.session_mgr.save_message(session_id, user_id, "assistant", _rt)
                            return {
                                "reply": _rt, "session_id": session_id,
                                "agent_name": "EisaX Financial Analyst", "model": "DeepSeek + yfinance",
                            }
                    except Exception as _dfm_fa_e:
                        logger.warning("[DFM→FinancialAgent] Failed: %s — falling back to DFM handler", _dfm_fa_e)
                try:
                    from core.dfm_lookup import get_dfm_context
                    _dfm_ctx = get_dfm_context(message) or ""
                    reply_text = await self._handle_dfm_query(message, _dfm_ctx)
                    self.session_mgr.save_message(session_id, user_id, "user", message)
                    self.session_mgr.save_message(session_id, user_id, "assistant", reply_text)
                    return {"reply": reply_text, "session_id": session_id, "agent_name": "EisaX DFM"}
                except Exception as _dfm_e:
                    logger.warning("[DFM handler] failed: %s", _dfm_e)

            # ── 8. Route handlers ──────────────────────────────────────────────
            if route == "SCREENER" or handler == "SCREENER":
                return await _handle_screening(
                    message=message,
                    session_id=session_id,
                    user_id=user_id,
                    orchestrator=self,
                    instruction=instruction,
                    user_ctx=_user_ctx,
                    chat_history=_recent_history,
                    route=route,
                    handler=handler,
                )

            if route == "STOCK_ANALYSIS":
                return await _handle_stock(self, session_id, user_id, message, instruction, _user_ctx)

            elif route == "FINANCIAL":
                _fin_resp = await _handle_financial(
                    self, session_id, user_id, message, instruction, handler, _user_ctx,
                    chat_history=_recent_history,
                )
                if _fin_resp is not None:
                    return _fin_resp
                # Fall through to GENERAL if both CIO and optimize fail

            elif route == "PORTFOLIO":
                return await _handle_portfolio(
                    session_id, user_id, message,
                    reply_saver=self.session_mgr.save_message,
                )

            # ── 9. GENERAL — Agent Loop for tool-requiring queries, Gemini for chat ──
            # Use the agent loop when the question likely needs live data tools.
            # Pure conversation/greetings go directly to Gemini (faster).
            _tool_signals = [
                "price", "سعر", "analyze", "حلل", "news", "أخبار", "screen",
                "portfolio", "محفظة", "compare", "قارن", "fundamentals",
                "recommend", "best stock", "أفضل سهم", "buy", "sell",
                "اشتري", "يبيع", "dividend", "earnings", "ربح", "return",
            ]
            _needs_tools = any(sig in message.lower() for sig in _tool_signals)
            if _needs_tools:
                try:
                    from core.agent_loop import run_agent as _run_agent
                    _agent_result = await _run_agent(
                        message=instruction or message,
                        user_ctx=_user_ctx,
                        user_id=user_id,
                        session_id=session_id,
                    )
                    _agent_reply = _agent_result.get("reply", "")
                    if _agent_reply:
                        self.session_mgr.save_message(session_id, user_id, "user", message)
                        self.session_mgr.save_message(session_id, user_id, "assistant", _agent_reply)
                        logger.info("[AgentLoop] Answered via %d tool(s) in %d iter(s)",
                                    len(_agent_result.get("tools_used", [])),
                                    _agent_result.get("iterations", 0))
                        return {
                            "reply":      _agent_reply,
                            "session_id": session_id,
                            "agent_name": "EisaX Agent",
                            "model":      _agent_result.get("model", "deepseek-agent"),
                        }
                except Exception as _ale:
                    logger.warning("[AgentLoop] Failed: %s — falling back to Gemini", _ale)

            return await _handle_general(self, session_id, user_id, message, instruction, _user_ctx,
                                         chat_history=_recent_history)

        except Exception as e:
            logger.error("process_message error: %s", e)
            return {
                "reply":      "عذراً، حدث خطأ غير متوقع. فريق EisaX يتابع.",
                "session_id": session_id or "error",
                "agent_name": "ErrorHandler",
                "model":      "error",
            }


    # ══════════════════════════════════════════════════════════════════════════
    # STREAMING — process_message_stream
    # Async generator yielding SSE-style event dicts for the /v1/chat/stream
    # endpoint.  Event schema:
    #   {"type": "status",  "text": "..."}  — loader/progress message
    #   {"type": "token",   "text": "..."}  — LLM content chunk
    #   {"type": "done",    "session_id": "...", "agent": "...", "model": "..."}
    #   {"type": "error",   "text": "..."}
    # ══════════════════════════════════════════════════════════════════════════
    async def process_message_stream(
        self,
        user_id: str,
        message: str,
        session_id: str | None = None,
    ):
        """
        Streaming version of process_message.
        - GENERAL / BOND / simple routes: true token-by-token streaming via Gemini/DeepSeek.
        - STOCK_ANALYSIS: status events during heavy data-fetch, then streams the final reply.
        - Everything else: falls back to non-streaming with a single 'token' flush.
        """
        import asyncio
        from core.streaming import status, token, done, error as _err
        from core.services.routing_service import (
            is_export_request, is_file_analysis, is_bond_request, is_greeting,
            detect_arabic_ticker, detect_dfm_screen,
        )

        try:
            if not session_id:
                session_id = self.session_mgr.get_or_create_session(user_id)

            # ── Export / file — no streaming benefit, fall through ────────────
            if is_export_request(message) or is_file_analysis(message):
                yield status("⏳ جارٍ المعالجة...")
                result = await self.process_message(user_id, message, session_id)
                reply = result.get("reply", "")
                for i in range(0, len(reply), 24):
                    yield token(reply[i:i+24])
                yield done(session_id=session_id, agent=result.get("agent_name","EisaX"), model=result.get("model"))
                return

            # ── Greeting fast-path — stream Gemini directly ───────────────────
            if is_greeting(message):
                yield status("...")
                from core.streaming import stream_gemini
                from state import SYSTEM_PROMPTS
                sys_p = SYSTEM_PROMPTS.get("assistant", "You are EisaX AI.")
                async for evt in stream_gemini(
                    f"{sys_p}\n\nUser: {message}\nEisaX:",
                    model=GEMINI_MODEL_BACKUP,
                    max_tokens=300,
                    temperature=0.7,
                ):
                    if evt["type"] in ("token", "error"):
                        yield evt
                self.session_mgr.save_message(session_id, user_id, "user", message[:500])
                yield done(session_id=session_id, agent="EisaX", model=GEMINI_MODEL_BACKUP)
                return

            # ── Route classification ─────────────────────────────────────────
            yield status("🔍 فهم الطلب...")
            _fast_ticker = detect_arabic_ticker(message)
            if _fast_ticker:
                route, handler, instruction = "STOCK_ANALYSIS", "STOCK_ANALYSIS", f"analyze {_fast_ticker}"
            else:
                route, handler, instruction, _ = self._classify_intent(message)

            # ── STOCK_ANALYSIS — ToolAgent streaming ────────────────────────
            if route == "STOCK_ANALYSIS":
                from core.tool_agent import ToolAgent
                _ta = ToolAgent(user_id=user_id)
                _reply_buf = []
                async for evt in _ta.stream(message):
                    if evt["type"] == "status":
                        yield status(evt["text"])
                    elif evt["type"] == "token":
                        _reply_buf.append(evt["text"])
                        yield token(evt["text"])
                        await asyncio.sleep(0)
                    elif evt["type"] == "error":
                        # ToolAgent failed — fall back to legacy pipeline
                        logger.warning("[StreamOrch] ToolAgent error: %s — falling back", evt["text"])
                        yield status("📊 تحليل متعمق...")
                        _fa_result = await asyncio.wait_for(
                            self._run_sync_in_executor(
                                self.financial_agent._handle_analytics,
                                session_id,
                                {"user_id": user_id},
                                message,
                            ),
                            timeout=150,
                        )
                        _fb_reply = _fa_result.get("reply", "")
                        yield status("")
                        for i in range(0, len(_fb_reply), 32):
                            yield token(_fb_reply[i:i+32])
                            await asyncio.sleep(0)
                        _reply_buf = [_fb_reply]
                        break

                _full_reply = "".join(_reply_buf)
                if _full_reply:
                    self.session_mgr.save_message(session_id, user_id, "user", message)
                    self.session_mgr.save_message(session_id, user_id, "assistant", _full_reply)
                yield done(session_id=session_id, agent="EisaX ToolAgent", model="DeepSeek+tools")
                return

            # ── GENERAL — true streaming via Gemini ──────────────────────────
            if route == "GENERAL":
                from core.streaming import stream_gemini
                from state import SYSTEM_PROMPTS
                from core.agents.general import GeneralAgent
                _user_ctx: dict = {}
                try:
                    if MEMORY_ENABLED:
                        _user_ctx = get_rich_user_context(user_id) or {}
                except Exception:
                    pass

                sys_p = SYSTEM_PROMPTS.get("cio", SYSTEM_PROMPTS.get("assistant", ""))
                _ctx_block = ""
                try:
                    if _user_ctx:
                        from core.memory_manager import format_ctx_for_prompt
                        _ctx_block = format_ctx_for_prompt(_user_ctx)
                except Exception:
                    pass

                full_prompt = f"{sys_p}\n{_ctx_block}\n\nUser: {message}\nEisaX:"
                _reply_buf = []
                async for evt in stream_gemini(
                    full_prompt,
                    model=GEMINI_MODEL_BACKUP,
                    max_tokens=1500,
                    temperature=0.7,
                ):
                    if evt["type"] == "token":
                        _reply_buf.append(evt["text"])
                        yield evt
                    elif evt["type"] == "error":
                        yield evt

                _full_reply = "".join(_reply_buf)
                self.session_mgr.save_message(session_id, user_id, "user", message[:500])
                self.session_mgr.save_message(session_id, user_id, "assistant", _full_reply)
                yield done(session_id=session_id, agent="EisaX", model=GEMINI_MODEL_BACKUP)
                return

            # ── All other routes — non-streaming fallback ────────────────────
            yield status("⏳ جارٍ التحليل...")
            result = await self.process_message(user_id, message, session_id)
            reply = result.get("reply", "")
            chunk_size = 32
            for i in range(0, len(reply), chunk_size):
                yield token(reply[i:i+chunk_size])
                await asyncio.sleep(0)
            yield done(
                session_id=session_id,
                agent=result.get("agent_name", "EisaX"),
                model=result.get("model"),
            )

        except Exception as e:
            logger.error("[StreamingOrchestrator] error: %s", e)
            yield _err(str(e))
            yield done(session_id=session_id or "error", agent="ErrorHandler", model=None)

    # BUG-03 FIX: Only these path prefixes may be modified via admin chat.
    _ALLOWED_WRITE_PREFIXES = (
        "core/agents/",
        "core/prompts/",
        "core/prompt_manager",
        "core/utils",
        "prompts/",
        "playbook/",
        "eisax_playbook.md",
        "core/admin_handler.py",
        "core/orchestrator.py",
        "core/glm_client.py",
        "core/grok_client.py",
        "core/scorecard.py",
        "core/etf_intelligence.py",
    )

    async def _apply_pending_modification(self, proposal: str) -> str:
        """Parse a Gemini-proposed change and apply it to the target file with backup."""
        import re as _re_adm
        from pathlib import Path as _Path
        try:
            file_match = _re_adm.search(r'FILE:\s*(.+)', proposal)
            if not file_match:
                return "❌ Could not parse file path from proposal."
            file_path = file_match.group(1).strip()

            # BUG-03 FIX: enforce path whitelist — resolve to relative path first
            from core.config import BASE_DIR as _cfg_base
            _BASE = str(_cfg_base) + "/"
            _rel  = file_path.replace(_BASE, "").lstrip("/")
            if not any(_rel.startswith(pfx) for pfx in self._ALLOWED_WRITE_PREFIXES):
                return (
                    f"⛔ **Security block:** `{_rel}` is not in the allowed write whitelist.\n"
                    f"Allowed prefixes: `{', '.join(self._ALLOWED_WRITE_PREFIXES)}`"
                )

            proposed_match = _re_adm.search(
                r'PROPOSED CODE:\s*```(?:python)?\n(.*?)```',
                proposal, _re_adm.DOTALL
            )
            if not proposed_match:
                return "❌ Could not parse the PROPOSED CODE block."
            proposed_code = proposed_match.group(1).strip()

            current_match = _re_adm.search(
                r'CURRENT CODE:\s*```(?:python)?\n(.*?)```',
                proposal, _re_adm.DOTALL
            )

            existing = read_file(file_path)
            if existing.startswith("ERROR"):
                return f"❌ {existing}"

            if current_match:
                current_code = current_match.group(1).strip()
                new_content  = existing.replace(current_code, proposed_code, 1)
                if new_content == existing:
                    return "❌ Could not find CURRENT CODE in the file. Nothing changed."
            else:
                new_content = proposed_code

            reason_match = _re_adm.search(r'REASON:\s*(.+)', proposal)
            reason = reason_match.group(1).strip() if reason_match else "Admin modification via chat"

            result = write_file(file_path, new_content, reason=reason)
            if result["success"]:
                return f"✅ **Applied.**\n📁 `{file_path}`\n💾 Backup: `{result['backup_path']}`"
            else:
                return f"❌ Write failed: {result['error']}"
        except Exception as e:
            return f"❌ _apply_pending_modification error: {e}"


# ── Singleton ──
_orchestrator_instance = MultiAgentOrchestrator()

async def think(message: str, settings: dict = {}, history: list = []) -> dict:
    user_id = settings.get("user_id", settings.get("session_id", "default"))
    session_id = settings.get("session_id", None)
    return await _orchestrator_instance.process_message(
        user_id=user_id,
        message=message,
        session_id=session_id
    )
