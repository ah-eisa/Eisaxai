import logging
import requests
import time as _time
from typing import Any, Dict, Optional
import os
import re
from datetime import datetime
import config
import state
from core.llm import get_client
from core.agents.base import BaseAgent
import core.portfolio_manager as pm
from core.broker import BrokerClient

# Phase-1 refactor: TTLCache now lives in core.utils
from core.utils import TTLCache as _TTLCache  # noqa: F401

logger = logging.getLogger(__name__)

import threading as _threading  # kept for any legacy direct threading.Lock() usage
import functools as _functools

# ── Utility helpers (extracted to finance_helpers for independent testability) ─
from core.agents.finance_helpers import (   # noqa: E402
    _VERDICT_TIERS,
    _safe_div_yield,
    _consensus_divergence,
    _fetch_btc_etf_flows,
    _compute_decision_confidence,
    _soften_execution_language,
    _round_scenario_prices,
    _fetch_onchain,
)
from core.services.decision_policy import (  # noqa: E402
    apply_language_locks,
    classify_data_coverage_level,
    compact_low_data_generation_inputs,
    count_valid_fundamental_fields,
)

# ── yfinance ADX guard (monkey-patch) ────────────────────────────────────────
# Yahoo Finance has NO data for Abu Dhabi ADX stocks (.AE suffix).
# Every yf.Ticker("ADNOCGAS.AE").info call returns HTTP 404 and triggers
# exponential retries (3× per call site × many call sites = 3+ min wasted).
# Patch yf.Ticker globally so .AE tickers return empty results immediately,
# allowing the system to fall through to local cache / pipeline data.
try:
    import yfinance as _yf_module
    import pandas as _pd

    _OrigTicker = _yf_module.Ticker

    class _ADXSafeTicker(_OrigTicker):
        """Drop-in replacement that short-circuits Yahoo for ADX (.AE) stocks."""
        _ADX_SUFFIXES = ('.AE',)

        def __init__(self, ticker, *args, **kwargs):
            super().__init__(ticker, *args, **kwargs)
            self._is_adx_skip = str(ticker).upper().endswith(self._ADX_SUFFIXES)

        @property
        def info(self):
            if self._is_adx_skip:
                return {}
            return super().info

        @property
        def fast_info(self):
            if self._is_adx_skip:
                return type('_EmptyFastInfo', (), {'last_price': None, 'market_cap': None})()
            return super().fast_info

        def history(self, *args, **kwargs):
            if self._is_adx_skip:
                return _pd.DataFrame()
            return super().history(*args, **kwargs)

        @property
        def calendar(self):
            if self._is_adx_skip:
                return {}
            return super().calendar

    _yf_module.Ticker = _ADXSafeTicker
    logger.info("[yf_adx_guard] Monkey-patched yf.Ticker — .AE tickers will skip Yahoo Finance")
except Exception as _yf_patch_err:
    logger.warning("[yf_adx_guard] Patch failed: %s", _yf_patch_err)

# ── Report Cache (TTL: 10 min) ─────────────────────────────────────────────
_REPORT_CACHE: dict = {}
_REPORT_CACHE_TTL = 600  # seconds

# Per-instance caches (not global singletons — avoids cross-request pollution)
_div_yield_cache    = _TTLCache(ttl_seconds=3600)   # dividend yields → 1h TTL
_fundamentals_cache = _TTLCache(ttl_seconds=600)    # fundamentals   → 10min TTL

# Suffixes that are always regional equity exchanges — skip ETF detection entirely.
# MENA markets do not issue ETFs under these suffixes; misclassification causes
# wrong scorecard weights (40/60 instead of 60/40) and wrong analysis path.
_ETF_EQUITY_ONLY_SUFFIXES = (
    ".CA",   # Egypt (EGX)
    ".AE",   # UAE (DFM/ADX)
    ".DU",   # UAE (DFM alternate)
    ".AD",   # UAE (ADX alternate)
    ".SR",   # Saudi Arabia (Tadawul)
    ".KW",   # Kuwait (BK)
    ".QA",   # Qatar (QSE)
    ".BH",   # Bahrain (BHX)
    ".MA",   # Morocco (Casablanca)
    ".TN",   # Tunisia (BVMT)
)


# _safe_div_yield, _VERDICT_TIERS, _consensus_divergence, _fetch_btc_etf_flows
# → imported from core.agents.finance_helpers above


def _yf_with_retry(ticker: str, max_attempts: int = 3, base_delay: float = 1.5):
    """
    Create a yfinance Ticker and fetch .info with exponential backoff.
    Returns (ticker_obj, info_dict). Raises on all attempts failing.
    """
    import yfinance as yf

    # Yahoo Finance does NOT carry ADX (Abu Dhabi .AE) stocks — they 404 every
    # time, burning 2-3 minutes on retries. Return empty immediately so the
    # caller falls through to local-cache / pipeline data.
    _YF_UNAVAILABLE = ('.AE', '.BH', '.MA', '.TN')
    if any(ticker.upper().endswith(s) for s in _YF_UNAVAILABLE):
        logger.debug("[yf_retry] %s: skipping yfinance (ADX .AE not on Yahoo) — using local cache", ticker)
        return yf.Ticker(ticker), {}

    last_exc = None
    for attempt in range(max_attempts):
        try:
            t = yf.Ticker(ticker)
            info = t.info  # triggers the network call
            return t, info
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = base_delay * (2 ** attempt)
                logger.warning("[yf_retry] %s attempt %d/%d failed: %s — retrying in %.1fs",
                               ticker, attempt + 1, max_attempts, exc, wait)
                _time.sleep(wait)
    raise last_exc


try:
    from core.realtime_data import get_live_news, deepcrawl_stock
except Exception as _realtime_import_err:
    logger.debug("[finance] core.realtime_data unavailable: %s — live news/deepcrawl disabled", _realtime_import_err)
    get_live_news = None      # type: ignore[assignment]
    deepcrawl_stock = None    # type: ignore[assignment]
from core.intent_classifier import IntentClassifier
from core.ticker_resolver import TickerResolver
from core.local_tickers import SUPPORTED_CURRENCIES, get_all_tickers_flat, get_ticker_currency
from core.egypt_bonds import is_egypt_bond_query, get_egypt_bond_data, format_egypt_bonds_for_prompt
from core.fixed_income import (
    is_fixed_income_query, extract_isin,
    get_instrument_data, compute_fi_score, format_fi_for_prompt,
    detect_sukuk_query_language,
)

# Module-level resolver instance
_ticker_resolver = TickerResolver()

from core.institutional import (
    detect_output_mode,
    get_output_mode_instruction
)
from core.portfolio_tracker import PortfolioTracker


# ── Mixin imports (extracted handlers) ────────────────────────────────────
from core.agents.handlers.cio import CIOMixin
from core.agents.handlers.fixed_income import FixedIncomeMixin
from core.agents.handlers.export_handler import ExportMixin
from core.agents.handlers.scorecard import ScorecardMixin
from core.agents.handlers.analytics import AnalyticsMixin
from core.agents.handlers.trade import TradeMixin
from core.agents.handlers.portfolio import PortfolioMixin

class FinancialAgent(BaseAgent, CIOMixin, FixedIncomeMixin, ExportMixin, ScorecardMixin, AnalyticsMixin, TradeMixin, PortfolioMixin):
    def __init__(self):
        super().__init__(name="FinancialAgent")
        self.client_factory = get_client
        self.web_search_enabled = False
        self.portfolio_tracker = PortfolioTracker()
        try:
            # Check if web_search tool is available
            import importlib
            if importlib.util.find_spec("anthropic"):
                self.web_search_enabled = True
                logger.info("[EisaX] Web search capability: ENABLED")
        except Exception as _e:
            logger.warning("[EisaX] Web search capability: DISABLED")
        self._setup_web_search()

    def _setup_web_search(self):
        """Setup web search capability if available"""
        try:
            from core.web_tools import web_search
            self._web_search = web_search
            logger.info("✅ [EisaX] Web search enabled")
        except Exception as _e:
            self._web_search = None
            logger.warning("⚠️ [EisaX] Web search not available")

    def _fetch_missing_scorecard_data(self, ticker: str, existing_data: dict) -> dict:
        """Fetch missing data from Yahoo Finance for scorecard"""
        try:
            _stock, info = _yf_with_retry(ticker)
            info = info or {}
            if not existing_data.get('quality'):
                existing_data['quality'] = info.get('fundamentalScore') or info.get('overallRisk') or 50
            if not existing_data.get('net_margin'):
                nm = info.get('netMargins') or info.get('profitMargins')
                if nm: existing_data['net_margin'] = float(nm) * 100 if nm < 1 else float(nm)
            if not existing_data.get('beta'):
                _b = info.get('beta')
                if _b and float(_b) != 0:
                    existing_data['beta'] = round(float(_b), 2)
                # Never default to 1.0 — leave None if missing
            if not existing_data.get('mc'):
                existing_data['mc'] = info.get('marketCap') or 0
        except Exception as e:
            logger.error(f"⚠️ Error in yfinance after retries: {e}")
        return existing_data

    def _execute_web_search_direct(self, query: str) -> dict:
        """Execute web search using Serper API"""
        import os, requests
        try:
            api_key = os.getenv("SERPER_API_KEY")
            if not api_key:
                return {"error": "SERPER_API_KEY not found"}
            
            response = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": 5},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get("organic", [])[:5]:
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": item.get("link", "")
                    })
                return {"success": True, "results": results, "query": query}
            else:
                return {"error": f"Search failed: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def think(self, 
              message: str, 
              context: Dict[str, Any], 
              settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        
        s = settings or {}
        mem = context.get("memory", {})
        sid = context.get("session_id", "default")
        model = s.get("model") or os.getenv("MODEL_NAME", config.DEFAULT_MODEL)
        temperature = s.get("temperature", 0.3) # Lower temp for finance
        
        # Check specific intents
        # We assume the Orchestrator might have passed a hint, but we can re-check logic here
        # or expose specific methods. For now, we implement the main logic from Orchestrator.
        
        primary_intent = IntentClassifier.detect_primary_intent(message, mem)
        
        # Greeks detection (Custom check)
        is_greeks = any(x in message.lower() for x in ["delta", "theta", "rho", "vega", "gamma", "black-scholes"])
        
        # Dispatch with error isolation
        try:
            # ── Clean Pipeline — intercept build-portfolio requests first ─────────
            try:
                import sys as _sys, os as _os
                _proj_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
                if _proj_root not in _sys.path:
                    _sys.path.insert(0, _proj_root)
                from portfolio_pipeline import is_pipeline_request, run as pipeline_run
                _pipeline_check = is_pipeline_request(message)
                logger.info("[Dispatch] Pipeline check=%s for msg: %s", _pipeline_check, message[:60])
                if _pipeline_check:
                    logger.info("[Dispatch] Pipeline request detected — routing to clean pipeline")
                    report = pipeline_run(message)
                    return {
                        "type": "chat.reply",
                        "reply": report,
                        "data": {"agent": "finance", "analysis_type": "pipeline_report"},
                    }
            except Exception as _ppe:
                logger.warning("[Dispatch] Pipeline check failed: %s — continuing normal dispatch", _ppe)

            # ── Fixed Income / Sukuk / ISIN — check first (specific beats generic) ──
            if primary_intent == "fixed_income" or is_fixed_income_query(message):
                return self._handle_fixed_income(message, s)

            # Egyptian bonds — check before generic analytics so it gets dedicated handling
            if is_egypt_bond_query(message):
                return self._handle_egypt_bonds(message, s)

            if primary_intent in ["optimize", "portfolio_optimize"]:
                return self._handle_optimize(sid, mem, message, s)

            if primary_intent in ["report", "portfolio_report"]:
                return self._handle_report(sid, mem, message)

            if primary_intent in ["analyze", "technical_analysis", "risk_analysis"] or "analyze" in message.lower():
                return self._handle_analytics(sid, mem, message)
    
            if primary_intent in ["forecast", "simulate", "project"] or any(x in message.lower() for x in ["forecast", "simulate", "prediction"]):
                return self._handle_forecast(sid, mem, message)
                
            if is_greeks:
                return self._handle_greeks(sid, message)
            if any(x in message.lower() for x in ["portfolio", "positions", "balance", "buying power"]):
                return self._handle_account_display()
            if primary_intent == "trade_execution":
                return self._handle_trade(sid, mem, message)
        except Exception as e:
            logger.error(f"[FinancialAgent] Handler failed: {e}. Falling back to default chat.")
            # Fall through to default chat logic below

        # ── Portfolio Commands ───────────────────────────────────────────
        portfolio_keywords = {
            "add": ["add", "buy", "purchase", "bought"],
            "remove": ["sell", "sold", "remove", "close"],
            "show": ["portfolio", "holdings", "positions", "show my"]
        }
        
        # Detect portfolio intent
        msg_lower = message.lower()
        
        # ADD position: "add 10 shares NVDA at $175"
        if any(kw in msg_lower for kw in portfolio_keywords["add"]) and ("share" in msg_lower or "stock" in msg_lower):
            return self._handle_portfolio_add(sid, mem, message)
        
        # REMOVE position: "sell 5 shares AAPL"
        if any(kw in msg_lower for kw in portfolio_keywords["remove"]) and ("share" in msg_lower or "stock" in msg_lower):
            return self._handle_portfolio_remove(sid, mem, message)
        
        # SHOW portfolio: "show my portfolio"
        if any(kw in msg_lower for kw in portfolio_keywords["show"]):
            return self._handle_portfolio_show(sid, mem, message)
        
        # Default Financial Chat
        system_prompt = state.SYSTEM_PROMPTS.get("investment", "")
        output_mode = detect_output_mode(message)
        mode_instruction = get_output_mode_instruction(output_mode)
        enhanced_prompt = f"{system_prompt}\n\nOUTPUT MODE FOR THIS RESPONSE: {mode_instruction}"
        
        # Check EXPORT intent explicitly
        if primary_intent == "report_export":
            return self._handle_export(sid, mem, message)
        # BUG-01 FIX: initialize positioning vars — they're only set inside
        # _handle_analytics (different scope). Default to "N/A" for fallback chat.
        pre_entry  = "N/A"
        pre_stop   = "N/A"
        pre_target = "N/A"

        try:
            client = self.client_factory()
            
            # Replace placeholders with actual positioning values
            enhanced_prompt = enhanced_prompt.replace("PLACEHOLDER_ENTRY", pre_entry)
            enhanced_prompt = enhanced_prompt.replace("PLACEHOLDER_TARGET", pre_target)
            enhanced_prompt = enhanced_prompt.replace("PLACEHOLDER_STOP", pre_stop)
            
            response = client.create_completion(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": enhanced_prompt},
                    {"role": "user", "content": message},
                ],
            )
            reply_content = response.choices[0].message.content
            
            # Post-process (The Orchestrator used to do this, now we do it here for finance)
            if len(reply_content) > 300 and "EISAX INSIGHT" not in reply_content:
                if not reply_content.strip().endswith("---"):
                    reply_content += "\n\n---"
                reply_content += "\n**EISAX INSIGHT:** Strategic clarity emerges from structured analysis."
            
            # Update memory with last reply if needed (usually orchestrated by caller, but we can return data)
            return {
                "type": "chat.reply",
                "reply": reply_content,
                "data": {"agent": "finance", "last_reply": reply_content}
            }

        except Exception as e:
            return {"type": "error", "reply": f"Financial Agent Error: {e}"}

    def _resolve_ticker(self, msg: str) -> str:
        """يحوّل اسم الشركة أو الاسم العربي لـ ticker صح — supports local Arab markets."""
        # ── Step 1: Try TickerResolver (covers Saudi, Egypt, UAE) ──
        result = _ticker_resolver.resolve_single(msg)
        if result:
            return result
        
        # ── Step 2: Try resolving individual words ──
        for word in msg.split():
            result = _ticker_resolver.resolve_single(word)
            if result:
                return result
        
        # ── Step 3: Fallback to hardcoded US/Crypto mapping ──
        KNOWN = {
            "نيفيديا": "NVDA", "nvidia": "NVDA", "انفيديا": "NVDA",
            "ابل": "AAPL", "apple": "AAPL", "أبل": "AAPL",
            "مايكروسوفت": "MSFT", "microsoft": "MSFT",
            "امازون": "AMZN", "amazon": "AMZN", "أمازون": "AMZN",
            "جوجل": "GOOGL", "google": "GOOGL", "alphabet": "GOOGL",
            "ميتا": "META", "meta": "META", "فيسبوك": "META",
            "تسلا": "TSLA", "tesla": "TSLA",
            "amd": "AMD", "ايه ام دي": "AMD",
            "intel": "INTC", "انتل": "INTC",
            "aramco": "2222.SR", "ارامكو": "2222.SR", "أرامكو": "2222.SR",
            "sabic": "2010.SR", "سابك": "2010.SR",
            # UAE energy
            "adnoc": "ADNOCDIST.AE", "أدنوك": "ADNOCDIST.AE", "ادنوك": "ADNOCDIST.AE",
            "adnoc distribution": "ADNOCDIST.AE", "أدنوك للتوزيع": "ADNOCDIST.AE",
            "adnoc gas": "ADNOCGAS.AE", "أدنوك للغاز": "ADNOCGAS.AE",
            "adnoc drilling": "ADNOCDRILL.AE", "أدنوك للحفر": "ADNOCDRILL.AE",
            "taqa": "TAQA.AE", "طاقة": "TAQA.AE",
            # UAE general
            "emaar": "EMAAR.DU", "اعمار": "EMAAR.DU", "إعمار": "EMAAR.DU",
            "dewa": "DEWA.DU", "ديوا": "DEWA.DU",
            "enbd": "ENBD.DU", "الإمارات دبي الوطني": "ENBD.DU",
            "air arabia": "AIRARABIA.DU", "العربية للطيران": "AIRARABIA.DU",
            # Kuwait
            "kfh": "KFH.KW", "بيت التمويل الكويتي": "KFH.KW", "بيتك": "KFH.KW",
            "kuwait finance house": "KFH.KW",
            "nbk": "NBK.KW", "بنك الكويت الوطني": "NBK.KW",
            "national bank of kuwait": "NBK.KW",
            "zain": "ZAIN.KW", "زين": "ZAIN.KW", "زين الكويت": "ZAIN.KW",
            "mobile telecom": "ZAIN.KW",
            "boubyan": "BOUBYAN.KW", "بنك بوبيان": "BOUBYAN.KW",
            "burgan": "BURGAN.KW", "بنك برقان": "BURGAN.KW",
            "ahli bank kuwait": "ABK.KW", "البنك الأهلي الكويتي": "ABK.KW",
            "gulf bank": "GULFBANK.KW", "بنك الخليج": "GULFBANK.KW",
            "kpc": "KPC.KW", "بترو الكويت": "KPC.KW",
            "humansoft": "HUMANSOFT.KW", "هيومانسوفت": "HUMANSOFT.KW",
            "agility": "AGLTY.KW", "أجيليتي": "AGLTY.KW",
            # Qatar
            "qnb": "QNBK.QA", "بنك قطر الوطني": "QNBK.QA", "قطر الوطني": "QNBK.QA",
            "qatar national bank": "QNBK.QA",
            "industries qatar": "IQCD.QA", "قطر للصناعات": "IQCD.QA",
            "qatar industries": "IQCD.QA",
            "qatargas": "QATARGAS.QA",
            "ooredoo": "ORDS.QA", "أوريدو": "ORDS.QA",
            "qatar airways": "QATR.QA",
            "commercial bank qatar": "CBQK.QA", "البنك التجاري قطر": "CBQK.QA",
            "masraf al rayan": "MARK.QA", "مصرف الريان": "MARK.QA",
            "qatar islamic bank": "QIBK.QA", "بنك قطر الإسلامي": "QIBK.QA",
            "qib": "QIBK.QA",
            "milaha": "QNNS.QA", "ميلاها": "QNNS.QA", "ملاحة": "QNNS.QA",
            "woqod": "WDAM.QA", "وقود": "WDAM.QA",
            # Crypto
            "bitcoin": "BTC-USD", "btc": "BTC-USD", "بيتكوين": "BTC-USD", "بتكوين": "BTC-USD",
            "ethereum": "ETH-USD", "eth": "ETH-USD", "ايثيريوم": "ETH-USD", "اثيريوم": "ETH-USD",
            "solana": "SOL-USD", "sol": "SOL-USD", "سولانا": "SOL-USD",
            "xrp": "XRP-USD", "ريبل": "XRP-USD", "ripple": "XRP-USD",
            "bnb": "BNB-USD", "binance coin": "BNB-USD",
            "dogecoin": "DOGE-USD", "doge": "DOGE-USD", "دوج": "DOGE-USD",
            "cardano": "ADA-USD", "ada": "ADA-USD",
            "avalanche": "AVAX-USD", "avax": "AVAX-USD",
            "chainlink": "LINK-USD", "link": "LINK-USD",
            "polkadot": "DOT-USD", "dot": "DOT-USD",
        }
        low = msg.lower()
        for name, ticker in KNOWN.items():
            if name in low:
                return ticker
        return None

    def _format_local_price(self, price: float, ticker: str) -> str:
        """Format price with correct local currency symbol."""
        currency = get_ticker_currency(ticker)
        currency_info = SUPPORTED_CURRENCIES.get(currency, {})
        symbol = currency_info.get("symbol", "$")
        if currency in ("SAR", "AED", "EGP", "KWF", "QAR"):
            return f"{price:,.2f} {symbol}"
        return f"${price:,.2f}"

    def _get_local_display_name(self, ticker: str, lang: str = "ar") -> str:
        """Get display name for a ticker in Arabic or English."""
        info = _ticker_resolver.get_ticker_info(ticker)
        if not info:
            return ticker
        if lang == "ar":
            return info.get("name_ar", info.get("name_en", ticker))
        return info.get("name_en", ticker)

    def _is_local_ticker(self, ticker: str) -> bool:
        """Check if ticker is from a local Arab market."""
        return any(ticker.endswith(s) for s in [".SR", ".CA", ".AE", ".DU", ".KW", ".QA"])

    def _get_brain_context(self, ticker: str) -> str:
        """جيب الـ history السابق للسهم من الـ Brain"""
        try:
            import sqlite3
            from core.config import APP_DB as _cfg_app_db
            conn = sqlite3.connect(str(_cfg_app_db))
            cur = conn.cursor()

            cur.execute("""
                SELECT verdict, price_at_prediction, target_price, prediction_date
                FROM predictions
                WHERE ticker = ? AND price_at_prediction > 0
                ORDER BY prediction_date DESC LIMIT 3
            """, (ticker,))
            rows = cur.fetchall()

            cur.execute("SELECT analysis_count, last_verdict FROM stock_knowledge WHERE ticker=?", (ticker,))
            sk = cur.fetchone()
            conn.close()

            if not rows:
                return ""

            lines = [f"\n\n## EisaX Brain — {ticker} Historical Context"]
            lines.append(f"Times analyzed: {sk[0] if sk else len(rows)}")
            lines.append("\nPrevious verdicts:")
            for verdict, price, target, date in rows:
                target_str = f" → target ${target:.2f}" if target else ""
                date_str = str(date)[:10]
                lines.append(f"  • {date_str}: {verdict} @ ${price:.2f}{target_str}")

            if len(rows) >= 2:
                last = rows[0]
                prev = rows[1]
                change = ((last[1] - prev[1]) / prev[1] * 100) if prev[1] else 0
                direction = "📈 UP" if change > 0 else "📉 DOWN"
                lines.append(f"\nPrice movement since last analysis: {direction} {abs(change):.1f}%")
                if prev[0] == "BUY" and change > 0:
                    lines.append("✅ Previous BUY call was CORRECT")
                elif prev[0] in ["SELL", "REDUCE"] and change < 0:
                    lines.append("✅ Previous SELL call was CORRECT")
                elif prev[0] == "BUY" and change < 0:
                    lines.append("⚠️ Previous BUY call moved AGAINST prediction")

            lines.append("\nUse this context to refine your current analysis.")
            return "\n".join(lines)
        except Exception as e:
            return ""

