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

class FinancialAgent(BaseAgent):
    """
    Specialized agent for financial analysis, portfolio optimization, and reporting.
    """
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

    # Fallback direct Serper implementation (not used when core.web_tools loads)
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


    def _handle_cio_analysis(self, msg: str, sid: str = "default") -> Dict[str, Any]:
        """
        Direct CIO portfolio analysis — P&L, stress test, recommendation.
        Bypasses the full optimizer. Uses yfinance for live prices.
        """
        import yfinance as yf
        import re
        import os, requests as _req
        from datetime import datetime

        # ── Parse holdings from message ───────────────────────────────────────
        # Pattern: "TICKER: N shares @ average cost $PRICE" or "TICKER: N shares @ $PRICE"
        holdings = {}

        # Robust line-by-line parser — handles English, Arabic, Saudi tickers
        import re as _re2
        _ticker_pat = _re2.compile(r'([A-Z]{2,5}(?:\.[A-Z]{2,3})?|[0-9]{3,4}\.[A-Z]{2,3})')
        _number_pat = _re2.compile(r'[0-9]+(?:[,.][0-9]+)*')
        _skip_words = {'AT','OR','IN','OF','TO','BY','VS','AND','THE','FOR','SAR','USD','AED','EGP','SR'}

        for line in msg.replace(';','\n').split('\n'):
            line = line.strip().lstrip('-*\u2022 ')
            tks = [t for t in _ticker_pat.findall(line) if t not in _skip_words]
            if len(tks) == 1 and tks[0] not in holdings:
                ticker = tks[0]
                # Remove ticker from line before extracting numbers
                line_no_ticker = _ticker_pat.sub(' ', line)
                nums = [float(n.replace(',','')) for n in _number_pat.findall(line_no_ticker)]
                if len(nums) >= 2:
                    holdings[ticker] = {'shares': nums[0], 'avg_cost': nums[1]}
            elif len(tks) > 1:
                # Multiple tickers on one line — skip, handled by fallback
                pass

        # Fallback: whole-message scan for missed tickers
        if not holdings:
            _p = _re2.compile(
                r'([A-Z]{2,5}(?:\.[A-Z]{2,3})?|[0-9]{3,4}\.[A-Z]{2,3})'
                r'[^0-9]{0,20}([0-9]+(?:,[0-9]+)?)[^0-9]{0,30}([0-9]+(?:[.,][0-9]+)*)',
                _re2.IGNORECASE)
            _skip_words2 = {'AT','OR','IN','OF','TO','BY','VS','AND','THE','FOR','SAR','USD','AED','EGP','SR'}
            for m in _p.finditer(msg):
                t = m.group(1).upper()
                if t not in _skip_words2 and t not in holdings:
                    try:
                        holdings[t] = {'shares': float(m.group(2).replace(',','')),
                                       'avg_cost': float(m.group(3).replace(',',''))}
                    except Exception:
                        pass


        # ── Fetch live prices ─────────────────────────────────────────────────
        # Normalize crypto tickers — yfinance requires "BTC-USD" not "BTC"
        _CRYPTO_MAP = {
            'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD',
            'BNB': 'BNB-USD', 'ADA': 'ADA-USD', 'DOGE': 'DOGE-USD',
            'XRP': 'XRP-USD', 'DOT': 'DOT-USD', 'AVAX': 'AVAX-USD',
            'MATIC': 'MATIC-USD', 'LINK': 'LINK-USD', 'LTC': 'LTC-USD',
        }
        # UAE tickers need market_data_engine (yfinance returns 404 for .AE/.DU)
        _uae_tickers = {t for t in holdings if t.endswith('.DU') or t.endswith('.AE')}

        # Build original_ticker → yfinance_ticker map (None = skip yfinance)
        _yf_ticker_map = {}
        for t in holdings:
            if t in _CRYPTO_MAP:
                _yf_ticker_map[t] = _CRYPTO_MAP[t]
            elif t in _uae_tickers:
                _yf_ticker_map[t] = None   # handled by market_data_engine below
            else:
                _yf_ticker_map[t] = t

        # Batch-fetch all yfinance-eligible tickers
        _yf_tickers = [v for v in _yf_ticker_map.values() if v is not None]
        prices = {}
        try:
            if _yf_tickers:
                tickers_str = ' '.join(_yf_tickers)
                raw = yf.download(tickers_str, period='5d', auto_adjust=True,
                                  group_by='ticker', progress=False, threads=False)
                if isinstance(raw.columns, __import__('pandas').MultiIndex):
                    for t, yf_t in _yf_ticker_map.items():
                        if yf_t is None:
                            continue
                        try:
                            col = raw.xs(yf_t, axis=1, level=0)
                            prices[t] = float(col['Close'].dropna().iloc[-1])
                        except Exception:
                            pass
                else:
                    # Single-ticker download has flat columns
                    if 'Close' in raw.columns:
                        single_px = float(raw['Close'].dropna().iloc[-1])
                        for t, yf_t in _yf_ticker_map.items():
                            if yf_t is not None:
                                prices[t] = single_px
        except Exception as e:
            logger.error(f"[CIO] yfinance batch fetch failed: {e}")

        # Slow individual fallback for missed non-UAE tickers
        for t, yf_t in _yf_ticker_map.items():
            if yf_t is None or t in prices:
                continue
            try:
                info = yf.Ticker(yf_t).fast_info
                prices[t] = float(getattr(info, 'last_price', 0) or 0)
            except Exception:
                prices[t] = 0.0

        # UAE stocks: use market_data_engine (StockAnalysis / RapidAPI backed)
        for t in _uae_tickers:
            if prices.get(t, 0.0) > 0:
                continue
            try:
                from core.market_data_engine import get_latest_price as _get_uae_px
                _uae_res = _get_uae_px(t, 'AE')
                if _uae_res and _uae_res.get('close', 0) > 0:
                    prices[t] = float(_uae_res['close'])
                    logger.info(f"[CIO] UAE price for {t}: {prices[t]} AED")
            except Exception as _ue:
                logger.warning(f"[CIO] UAE price fetch failed for {t}: {_ue}")
            if t not in prices:
                prices[t] = 0.0

        # ── Currency detection & normalization ──────────────────────────────
        # Detect currency per ticker and normalize cost basis to match market price currency
        def _get_currency(ticker: str) -> tuple:
            """Returns (currency_code, symbol, usd_rate)
            usd_rate: multiply local price by this to get USD (1.0 for USD assets)
            """
            t = ticker.upper()
            if t.endswith('.SR'):
                return ('SAR', 'SAR', 1/3.75)   # 1 SAR = 0.2667 USD
            elif t.endswith('.AE') or t.endswith('.DU'):
                return ('AED', 'AED', 1/3.6725) # 1 AED = 0.2723 USD
            elif t.endswith('.CA'):
                return ('EGP', 'EGP', 1/50.0)   # approximate EGP rate
            elif t.endswith('.KW'):
                return ('KWF', 'KWF', 1/3070.0) # 1 KWD = 3.27 USD; 1000 fils = 1 KWD
            elif t.endswith('.QA'):
                return ('QAR', 'QAR', 1/3.64)   # 1 QAR = 0.2747 USD
            elif t.endswith('-USD') or t.endswith('BTC') or t.endswith('ETH'):
                return ('USD', '$', 1.0)
            else:
                return ('USD', '$', 1.0)         # US stocks default USD

        # ── Upgrade crypto prices via CoinGecko (real-time, no key needed) ─────
        _COINGECKO_IDS = {
            'BTC': 'bitcoin',    'ETH': 'ethereum',   'SOL': 'solana',
            'BNB': 'binancecoin','ADA': 'cardano',     'DOGE': 'dogecoin',
            'XRP': 'ripple',     'DOT': 'polkadot',    'AVAX': 'avalanche-2',
            'MATIC': 'matic-network', 'LINK': 'chainlink', 'LTC': 'litecoin',
        }
        _crypto_in_portfolio = [t for t in holdings if t in _COINGECKO_IDS]
        if _crypto_in_portfolio:
            try:
                _cg_ids = ','.join(_COINGECKO_IDS[t] for t in _crypto_in_portfolio)
                _cg_res = _req.get(
                    f"https://api.coingecko.com/api/v3/simple/price?ids={_cg_ids}&vs_currencies=usd",
                    timeout=8
                )
                if _cg_res.status_code == 200:
                    _cg_data = _cg_res.json()
                    for t in _crypto_in_portfolio:
                        _cg_px = _cg_data.get(_COINGECKO_IDS[t], {}).get('usd', 0)
                        if _cg_px and float(_cg_px) > 0:
                            prices[t] = float(_cg_px)
                            logger.info(f"[CIO] CoinGecko live price {t}: ${_cg_px:,.2f}")
            except Exception as _cge:
                logger.warning(f"[CIO] CoinGecko fetch failed — keeping yfinance prices: {_cge}")

        # ── Compute P&L ───────────────────────────────────────────────────────
        today = datetime.now().strftime('%B %d, %Y')
        rows = []
        total_cost   = 0.0
        total_value  = 0.0
        table_lines  = ["| Ticker | Shares | Avg Cost | Current Price | Position Value (USD) | Unrealized P&L | Return |",
                        "|--------|--------|----------|---------------|----------------------|----------------|--------|"]

        for t, h in holdings.items():
            shares    = h['shares']
            avg_cost  = h['avg_cost']   # in local currency as user entered
            curr_px   = prices.get(t, 0.0)  # in local currency from yfinance
            currency, sym, usd_rate = _get_currency(t)

            # Normalize to USD for portfolio-level aggregation
            avg_cost_usd = avg_cost * usd_rate
            curr_px_usd  = curr_px  * usd_rate

            pos_cost  = shares * avg_cost_usd
            pos_val   = shares * curr_px_usd
            pnl       = pos_val - pos_cost
            ret_pct   = (pnl / pos_cost * 100) if pos_cost > 0 else 0.0
            total_cost  += pos_cost
            total_value += pos_val
            emoji = "📈" if pnl >= 0 else "📉"

            # Display in original currency for clarity
            if currency == 'USD':
                cost_display = f"${avg_cost:.2f}"
                px_display   = f"${curr_px:.2f}"
            else:
                cost_display = f"{avg_cost:.2f} {sym}"
                px_display   = f"{curr_px:.2f} {sym}"

            # Show fractional shares properly (e.g. 0.5 BTC, not "0")
            shares_display = f"{shares:g}" if shares != int(shares) else f"{int(shares):,}"
            table_lines.append(
                f"| **{t}** | {shares_display} | {cost_display} | {px_display} | "
                f"${pos_val:,.0f} | {emoji} ${pnl:+,.0f} | {ret_pct:+.1f}% |"
            )
            rows.append({'ticker': t, 'shares': shares, 'avg_cost': avg_cost_usd,
                        'curr_px': curr_px_usd, 'pos_cost': pos_cost,
                        'pos_val': pos_val, 'pnl': pnl, 'ret_pct': ret_pct,
                        'currency': currency})

        total_pnl     = total_value - total_cost
        total_ret_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

        # ── Stress tests ──────────────────────────────────────────────────────
        scenarios = [(-0.15, "Mild Correction (-15%)"),
                     (-0.25, "Moderate Bear (-25%)"),
                     (-0.40, "Severe Crash (-40%)")]
        stress_lines = ["| Scenario | Portfolio Value | vs Cost Basis | P&L |",
                        "|----------|----------------|----------------|-----|"]
        for drop, label in scenarios:
            stressed_val = total_value * (1 + drop)
            vs_cost      = stressed_val - total_cost
            stress_lines.append(
                f"| {label} | ${stressed_val:,.0f} | "
                f"{'📉' if vs_cost < 0 else '📈'} ${vs_cost:+,.0f} | "
                f"{(vs_cost/total_cost*100):+.1f}% |"
            )

        # ── Build report body (profile card inserted later after region compute) ──
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        _report_pnl_block = f"""## 📊 Portfolio P&L Summary

{chr(10).join(table_lines)}

**Total Cost Basis:** ${total_cost:,.0f}
**Current Portfolio Value:** ${total_value:,.0f}
**Unrealized P&L: {pnl_emoji} ${total_pnl:+,.0f} ({total_ret_pct:+.1f}%)**

---

## 🧪 Stress Test Scenarios

{chr(10).join(stress_lines)}

---
"""

        # ── Fetch CURRENT dividend yields — parallel + cached ─────────────────
        def _fetch_one_yield(ticker_str: str) -> float:
            """Fetch current market dividend yield (NOT yield-on-cost). Cached 1h."""
            cached = _div_yield_cache.get(f"dy_{ticker_str}")
            if cached is not None:
                return cached
            try:
                _t_info = yf.Ticker(ticker_str).info
                # trailingAnnualDividendYield is reliable decimal (0.004 = 0.4%)
                # dividendYield returns % value (0.4 for 0.4%) — avoid multiplying it
                _raw = float(_t_info.get("trailingAnnualDividendYield") or 0)
                result = min(max(_raw * 100, 0.0), 15.0)  # decimal→pct, clamp [0%,15%]
            except Exception:
                result = 0.0
            _div_yield_cache.set(f"dy_{ticker_str}", result)
            return result

        from concurrent.futures import ThreadPoolExecutor as _TPE
        _dy_tickers = [r['ticker'] for r in rows]
        with _TPE(max_workers=min(8, max(1, len(_dy_tickers)))) as _pool:
            _dy_vals = list(_pool.map(_fetch_one_yield, _dy_tickers))
        live_div_yields = dict(zip(_dy_tickers, _dy_vals))

        port_div_yield_current = sum(
            live_div_yields.get(r['ticker'], 0) * (r['pos_val'] / total_value)
            for r in rows
        ) if total_value > 0 else 0.0

        # ── Region classification ──────────────────────────────────────────────
        def _get_region(ticker: str) -> str:
            t = ticker.upper()
            if t in _CRYPTO_MAP or t.endswith('-USD'):
                return 'Crypto'
            elif t.endswith('.DU') or t.endswith('.AE'):
                return 'UAE'
            elif t.endswith('.SR'):
                return 'Saudi Arabia'
            elif t.endswith('.KW'):
                return 'Kuwait'
            elif t.endswith('.QA'):
                return 'Qatar'
            elif t.endswith('.CA'):
                return 'Egypt'
            elif t.endswith('.L'):
                return 'UK'
            elif t.endswith('.PA') or t.endswith('.DE') or t.endswith('.MI'):
                return 'Europe'
            elif t.endswith('.T') or t.endswith('.HK'):
                return 'Asia'
            else:
                return 'US'

        # Map tickers to regions and compute regional weights
        _ticker_regions = {r['ticker']: _get_region(r['ticker']) for r in rows}
        _region_value = {}
        for r in rows:
            _rg = _ticker_regions[r['ticker']]
            _region_value[_rg] = _region_value.get(_rg, 0.0) + r['pos_val']

        _region_weights_str = " | ".join(
            f"{rg} {(val/total_value*100):.1f}%"
            for rg, val in sorted(_region_value.items(), key=lambda x: -x[1])
        ) if total_value > 0 else ""

        # ── Investor profile: detect home region & MENA intentionality ────────
        _mena_regions = {'UAE', 'Saudi Arabia', 'Kuwait', 'Qatar', 'Egypt'}
        _gcc_regions  = {'UAE', 'Saudi Arabia', 'Kuwait', 'Qatar'}
        _mena_weight  = sum(_region_value.get(rg, 0) for rg in _mena_regions) / total_value * 100 if total_value > 0 else 0
        _gcc_weight   = sum(_region_value.get(rg, 0) for rg in _gcc_regions)  / total_value * 100 if total_value > 0 else 0
        _is_gcc_investor = _gcc_weight >= 20

        if _mena_weight >= 40:
            _investor_profile      = f"GCC/MENA-focused investor ({_mena_weight:.0f}% in MENA, {_gcc_weight:.0f}% in GCC). This regional exposure is intentional — respect it."
            _investor_profile_icon = "🌍 GCC/MENA-Focused"
        elif _mena_weight >= 20:
            _investor_profile      = f"Diversified investor with meaningful GCC/MENA exposure ({_mena_weight:.0f}% in MENA). Respect this intentional regional allocation."
            _investor_profile_icon = "🌍 Diversified + MENA"
        else:
            _investor_profile      = "Global investor with limited MENA exposure."
            _investor_profile_icon = "🌐 Global"

        # ── Compute specific calendar dates for execution timeline ─────────────
        from datetime import timedelta
        _today_dt     = datetime.now()
        # Skip weekends for trading days
        def _next_trading_date(dt, days_ahead):
            d = dt + timedelta(days=days_ahead)
            while d.weekday() >= 5:   # Saturday=5, Sunday=6
                d += timedelta(days=1)
            return d.strftime('%A, %b %d')

        _date_immediate  = _next_trading_date(_today_dt, 1)   # next trading day
        _date_this_week  = _next_trading_date(_today_dt, 4)   # ~end of week
        _date_next_review = (_today_dt + timedelta(days=30)).strftime('%B %d, %Y')

        # ── Replacement universe ───────────────────────────────────────────────
        # GCC-investor rule: within GCC markets (UAE/SA/KW/QA), cross-exchange
        # replacements are acceptable — they're all home-region for this investor.
        _GCC_COMBINED = (
            'EMAAR.DU, FAB.DU, ADNOCGAS.DU, ADNOCDIST.DU, DIB.DU, ENBD.DU, ALDAR.DU, DEWA.DU, TAQA.DU | '
            '2222.SR (Aramco), 1120.SR (Al-Rajhi), 2010.SR (SABIC), 2380.SR (Petrochem) | '
            'QNBK.QA, ORDS.QA | NBK.KW, KFH.KW'
        )
        _REGION_UNIVERSE = {
            'US':           'AAPL, MSFT, GOOGL, AMZN, NVDA, META, BRK-B, JPM, V, JNJ, XOM, UNH, LLY, HD, PG',
            'UAE':          _GCC_COMBINED if _is_gcc_investor else 'EMAAR.DU, FAB.DU, ADNOCGAS.DU, ADNOCDIST.DU, DIB.DU, ENBD.DU, ALDAR.DU, DEWA.DU, TAQA.DU',
            'Saudi Arabia': _GCC_COMBINED if _is_gcc_investor else '2222.SR (Aramco), 1120.SR (Al-Rajhi), 2010.SR (SABIC), 2380.SR (Petrochem)',
            'Kuwait':       _GCC_COMBINED if _is_gcc_investor else 'NBK.KW, ZAIN.KW, KFH.KW, HUMANSOFT.KW, AGILITY.KW',
            'Qatar':        _GCC_COMBINED if _is_gcc_investor else 'QNBK.QA, ORDS.QA, QIIB.QA, MARK.QA',
            'Egypt':        'COMI.CA, HRHO.CA, EAST.CA, SWDY.CA, EKHO.CA, ABUK.CA',
            'Crypto':       'BTC, ETH, SOL, BNB, AVAX, LINK',
            'UK':           'BP.L, SHEL.L, AZN.L, HSBA.L, ULVR.L',
            'Europe':       'ASML.AS, SAP.DE, LVMH.PA, NESN.SW, ROG.SW',
            'Asia':         '9984.T (SoftBank), 7203.T (Toyota), 0700.HK (Tencent), 9988.HK (Alibaba)',
        }

        # Per-ticker region + universe line for prompt
        _ticker_region_lines = "\n".join(
            f"  {r['ticker']} → {_ticker_regions[r['ticker']]} "
            f"(weight {r['pos_val']/total_value*100:.1f}%)"
            for r in sorted(rows, key=lambda x: -x['pos_val'])
        ) if total_value > 0 else ""

        _universe_lines = "\n".join(
            f"  {rg}: {names}"
            for rg, names in _REGION_UNIVERSE.items()
            if rg in _ticker_regions.values()
        )

        # ── Investor profile card (shown in the report itself) ────────────────
        _profile_card = f"""\n## 👤 Investor Profile\n\n| Field | Value |\n|-------|-------|\n| Profile | {_investor_profile_icon} |\n| MENA Exposure | {_mena_weight:.0f}% of portfolio |\n| GCC Exposure | {_gcc_weight:.0f}% of portfolio |\n| Cross-GCC replacements | {'✅ Allowed (same home region)' if _is_gcc_investor else '⛔ Not applicable'} |\n\n---\n"""

        # ── Send to DeepSeek for CIO recommendation ────────────────────────────
        ds_key = os.getenv("DEEPSEEK_API_KEY", "")
        cio_section = ""
        if ds_key:
            # Build holdings summary WITHOUT cost basis in a way that prevents yield-on-cost calc
            def _fmt_shares(n):
                return f"{n:g}" if n != int(n) else f"{int(n):,}"
            holdings_summary = "\n".join(
                f"- {r['ticker']} [{_ticker_regions[r['ticker']]}]: {_fmt_shares(r['shares'])} shares | "
                f"current price ${r['curr_px']:.2f} | "
                f"return {r['ret_pct']:+.1f}% | "
                f"CURRENT dividend yield {live_div_yields.get(r['ticker'], 0):.2f}% (= annual div ÷ market price)"
                for r in rows
            )
            weights_summary = " | ".join(
                f"{r['ticker']} {r['pos_val']/total_value*100:.1f}%"
                for r in sorted(rows, key=lambda x: -x['pos_val'])
            ) if total_value > 0 else ""

            prompt = f"""You are EisaX, a CIO-level portfolio strategist. Today is {today}.

⛔ CRITICAL RULES — follow all exactly:
1. NEVER compute or reference "yield on cost". The ONLY valid dividend yield is CURRENT yield (annual dividend ÷ CURRENT market price), already provided below.
2. REGIONAL DISCIPLINE: When recommending to REDUCE, TRIM, or REPLACE any position, you MUST suggest an alternative from the approved universe for that region. Preserve the client's original regional allocation ratio.
3. Do NOT recommend moving money cross-region unless structurally critical — flag explicitly as "⚠️ Regional Shift" with a clear reason.
4. INVESTOR PROFILE: {_investor_profile} Honour this profile. For a GCC investor, UAE and Saudi/Kuwait/Qatar replacements are interchangeable within the same GCC home region.

CLIENT PORTFOLIO:
{holdings_summary}

PORTFOLIO METRICS:
- Total cost basis: ${total_cost:,.0f}
- Current value: ${total_value:,.0f}
- Unrealized P&L: ${total_pnl:+,.0f} ({total_ret_pct:+.1f}%)
- Current weights: {weights_summary}
- Regional allocation: {_region_weights_str}
- Current portfolio dividend yield: {port_div_yield_current:.2f}%

REGIONAL MAP:
{_ticker_region_lines}

APPROVED REPLACEMENT UNIVERSE (suggest ONLY from the matching region below):
{_universe_lines}

STRESS TEST:
- Mild correction (-15%): ${total_value*0.85:,.0f} (${total_value*0.85-total_cost:+,.0f} vs cost)
- Moderate bear (-25%): ${total_value*0.75:,.0f} (${total_value*0.75-total_cost:+,.0f} vs cost)
- Severe crash (-40%): ${total_value*0.60:,.0f} (${total_value*0.60-total_cost:+,.0f} vs cost)

Provide a CIO-grade analysis with EXACTLY these four sections:

## 4. 🎯 CIO Recommendation
Give a clear verdict: HOLD / PARTIAL SELL / BUY MORE / REBALANCE
State target weights for each position after rebalancing.
End this section with one line: **"Projected portfolio yield after rebalancing: X.XX%"** — compute this from the suggested new weights × the yields provided above.

## 5. 💡 Strategic Adjustments
For each change, use this exact format:
"Trim [X]% of [TICKER] → Rotate into [REPLACEMENT-TICKER] — [one-line rationale]"
• Include 2222.SR (Aramco) or other GCC names if the client has GCC exposure and they add diversification value.
• Never suggest cross-region rotation without ⚠️ Regional Shift flag.

## 6. 📅 Execution Plan
List trades in priority order (highest urgency first). For each trade specify:
- **Priority**: Immediate (by {_date_immediate}) / This Week (by {_date_this_week}) / Next Review (by {_date_next_review})
- **Order type**: Limit or Market — and why
- **Timing**: Which part of the trading session (e.g. "first 30 min after open", "avoid last 15 min")

## 7. ⚠️ Risk Flags
2–4 bullet points on concentration, liquidity, or correlation risks.

Be direct, numbers-first, institutional CIO tone. Max 750 words total."""

            try:
                r = _req.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                    json={"model": "deepseek-chat",
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 1200, "temperature": 0},
                    timeout=120
                )
                data = r.json()
                cio_section = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if cio_section:
                    logger.info("[CIO] DeepSeek recommendation generated (%d chars)", len(cio_section))
                else:
                    cio_section = "*(CIO recommendation unavailable — DeepSeek returned empty response)*"
            except Exception as e:
                logger.error(f"[CIO] DeepSeek call failed: {e}")
                cio_section = f"*(CIO recommendation unavailable: {e})*"
        else:
            cio_section = "*(CIO recommendation unavailable — DEEPSEEK_API_KEY not set)*"

        _disclaimer = """
---

> ⚠️ **Disclaimer:** This analysis is based on provided cost basis data and live market prices fetched at the time of this request. All prices, returns, and recommendations are for informational purposes only and do not constitute financial advice. Verify all prices independently before execution. Past performance is not indicative of future results.
"""
        # Assemble full report: title → investor profile card → P&L → CIO → disclaimer
        _report_title = f"# 🎯 EisaX CIO Analysis — {today}\n"
        full_reply = _report_title + _profile_card + _report_pnl_block + cio_section + _disclaimer

        # Save as artifact
        state.set_artifact(sid, {
            "type": "cio_analysis",
            "content": full_reply,
            "source": "cio_direct",
            "exportable": True,
            "timestamp": datetime.now()
        })

        return {"type": "chat.reply", "reply": full_reply,
                "data": {"agent": "finance", "analysis_type": "cio_direct"}}

    def _handle_optimize(self, sid: str, mem: Dict[str, Any], msg: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Wraps _handle_optimize_inner with a 240-second timeout to prevent hangs."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TE
        try:
            with ThreadPoolExecutor(max_workers=1) as _ex:
                _fut = _ex.submit(self._handle_optimize_inner, sid, mem, msg, settings)
                return _fut.result(timeout=240)
        except _TE:
            logger.warning("[_handle_optimize] Timed out after 240s — returning fallback")
            return {
                "type": "chat.reply",
                "reply": (
                    "⏱️ Portfolio optimization timed out (>4 min). This usually means:\n"
                "- Market data APIs are slow right now\n"
                "- Too many simultaneous requests\n\n"
                "**Please try again in 30 seconds.** If this keeps happening, try a simpler request like: 'optimize NVDA MSFT AAPL GOOGL'\n"
                    "**Quick suggestion while we optimize:**\n"
                    "- For aggressive growth: QQQ (35%), NVDA (20%), MSFT (15%), AMZN (15%), TSLA (15%)\n"
                    "- For balanced: SPY (40%), QQQ (20%), BND (20%), GLD (10%), VNQ (10%)\n"
                    "- For conservative: BND (50%), SPY (30%), GLD (10%), VYM (10%)\n\n"
                    "Try again in a moment for a full optimized analysis."
                ),
                "data": {}
            }
        except Exception as e:
            logger.error("[_handle_optimize] Error: %s", e)
            return {"type": "error", "reply": f"Portfolio optimization error: {e}", "data": {}}

    # ── Egyptian Bonds Handler ────────────────────────────────────────────────

    def _handle_egypt_bonds(self, message: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch live Egyptian government bond yield curve + macro context,
        then ask the LLM to produce a CIO-style analysis.
        """
        try:
            bond_data = get_egypt_bond_data()
            bond_context = format_egypt_bonds_for_prompt(bond_data)

            model = settings.get("model") or os.getenv("MODEL_NAME", config.DEFAULT_MODEL)

            system_prompt = (
                "You are EisaX, an institutional CIO specialising in emerging-market fixed income. "
                "You have been provided with live data for Egyptian government bonds and T-bills. "
                "Produce a concise, data-driven analysis covering:\n"
                "1. Current yield curve shape (normal / inverted / flat) and what it signals\n"
                "2. Real yield vs. inflation context (if data available)\n"
                "3. Relative value: short-end vs. long-end opportunity\n"
                "4. EGP currency risk for foreign investors\n"
                "5. Key risks and catalysts (CBE rate path, IMF programme, FX reserves)\n"
                "6. Investment verdict: who should buy, in what maturity, and why\n\n"
                "Use markdown with headers. Be specific with numbers. "
                "Today's date: " + datetime.now().strftime("%B %d, %Y")
            )

            client = self.client_factory()
            user_message = f"{bond_context}\n\nUser question: {message}"
            response = client.create_completion(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ]
            )
            reply = response.choices[0].message.content if hasattr(response, "choices") else str(response)

            # Attach raw data for potential export
            return {
                "type": "chat.reply",
                "reply": reply,
                "data": {
                    "egypt_bonds": bond_data,
                    "source": bond_data.get("source"),
                    "fetched_at": bond_data.get("fetched_at"),
                }
            }

        except Exception as e:
            logger.error("[_handle_egypt_bonds] Error: %s", e)
            return {
                "type": "chat.reply",
                "reply": (
                    "⚠️ Could not fetch live Egyptian bond data right now. "
                    "Common data points:\n\n"
                    "- **CBE Overnight Deposit Rate**: ~27.25% (as of early 2025)\n"
                    "- **91-day T-bill**: ~27–28%\n"
                    "- **1-year T-bill**: ~28–29%\n"
                    "- **5-year bond**: ~27.5–29%\n"
                    "- **10-year bond**: ~27–28.5%\n\n"
                    "Please check [CBE](https://www.cbe.org.eg) or "
                    "[Investing.com Egypt Bonds](https://www.investing.com/rates-bonds/egypt-government-bonds) "
                    "for the latest figures."
                ),
                "data": {}
            }

    # ── Fixed Income / Sukuk / ISIN Handler ──────────────────────────────────

    def _handle_fixed_income(self, message: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full Sukuk & Bond analysis by ISIN.
        1. Extract ISIN from message
        2. Fetch metadata via OpenFIGI + FMP + FRED benchmarks
        3. Compute EisaX Fixed Income Score
        4. Generate CIO-style report via LLM
        Falls back gracefully when no ISIN is found (general fixed-income question).
        """
        try:
            model = settings.get("model") or os.getenv("MODEL_NAME", config.DEFAULT_MODEL)
            lang  = detect_sukuk_query_language(message)
            isin  = extract_isin(message)

            # ── Case A: specific ISIN provided ───────────────────────────────
            if isin:
                logger.info("[fixed_income] ISIN detected: %s", isin)
                data  = get_instrument_data(isin, hint_text=message)
                score = compute_fi_score(data)
                fi_context = format_fi_for_prompt(data, score)

                is_sukuk   = data.get("is_sukuk", False)
                name_str   = data.get("name") or isin

                # ── Detect Bond ETF — fundamentally different methodology ──
                sec_type_raw = (data.get("security_type") or "").lower()
                is_bond_etf  = any(kw in sec_type_raw for kw in ("etf", "fund", "exchange traded"))

                if is_bond_etf:
                    # Bond ETF — NO seniority, covenants, YTM-in-isolation analysis
                    # Focus on: fund mechanics, expense ratio, duration, credit spread index, peers
                    is_hy_etf = any(kw in (name_str + sec_type_raw).lower()
                                    for kw in ("high yield", "junk", "hyg", "jnk", "faln", "angl", "bb", "b rated"))
                    peer_block = (
                        "\n### 4. Peer Comparison (HY Bond ETF Universe)\n"
                        "   Compare against primary HY ETF peers:\n"
                        "   - HYG (iShares iBoxx $ High Yield): largest, most liquid, ~8yr duration\n"
                        "   - JNK (SPDR Bloomberg HY Bond): similar but slightly higher yield / lower quality tilt\n"
                        "   - FALN (iShares Fallen Angels): higher quality bias (BB, recently downgraded)\n"
                        "   - ANGL (VanEck Fallen Angel): similar fallen angel approach, different index\n"
                        "   Discuss: yield spread vs peers, AUM, expense ratio, index methodology differences\n\n"
                    ) if is_hy_etf else (
                        "\n### 4. Peer Comparison (Bond ETF Universe)\n"
                        "   Compare against relevant IG bond ETF peers:\n"
                        "   - LQD (iShares IG Corp): broad IG corporate benchmark\n"
                        "   - VCIT (Vanguard IG Corp): lower expense ratio alternative\n"
                        "   - IGIB (iShares Intermediate IG): intermediate duration focus\n"
                        "   Discuss: yield spread vs peers, duration difference, expense ratio, index coverage\n\n"
                    )

                    system_prompt = (
                        f"You are EisaX, an institutional CIO specialising in fixed income ETF analysis.\n"
                        f"You have been given live data for **{name_str}**, which is a Bond ETF — NOT an individual bond.\n\n"
                        f"⚠️ IMPORTANT METHODOLOGY NOTE:\n"
                        f"  This is a Bond ETF (a diversified fund). Do NOT apply individual bond analysis:\n"
                        f"  - Do NOT discuss seniority or covenants (those apply to individual bonds)\n"
                        f"  - Do NOT apply YTM the same way (ETFs hold hundreds of bonds; use SEC 30-day yield)\n"
                        f"  - DO focus on: expense ratio, duration, index tracked, AUM/liquidity, credit spread index\n\n"
                        f"Produce a CIO-grade Bond ETF report with EXACTLY these sections:\n\n"
                        f"## 📊 Bond ETF Analysis — {name_str}\n\n"
                        f"### 1. Fund Overview\n"
                        f"   - Fund type (HY / IG / Treasury / TIPS / Fallen Angel etc.)\n"
                        f"   - Index tracked, number of holdings, AUM, expense ratio\n"
                        f"   - Exchange listing and daily volume (liquidity)\n\n"
                        f"### 2. Yield & Income Analysis\n"
                        f"   - SEC 30-day yield vs distribution yield (not YTM)\n"
                        f"   - Credit spread of underlying index vs benchmark (US10Y or IG index)\n"
                        f"   - Income attractiveness relative to risk-free rate\n\n"
                        f"### 3. Duration & Rate Risk\n"
                        f"   - Effective duration of the fund (not maturity of individual bonds)\n"
                        f"   - Price sensitivity: estimated NAV impact per 1% rate move\n"
                        f"   - Positioning in current rate environment\n\n"
                        f"{peer_block}"
                        f"### 5. Credit Quality & Default Risk\n"
                        f"   - Weighted average credit rating and HY/IG split\n"
                        f"   - Default rate sensitivity in recession vs base case\n"
                        f"   - Spread widening risk in a credit crunch\n\n"
                        f"### 6. Investment Verdict\n"
                        f"   - Clear BUY / HOLD / AVOID with conviction level\n"
                        f"   - Target investor profile (income, tactical, institutional)\n"
                        f"   - Entry conditions and key risk triggers\n\n"
                        f"## 🎯 EisaX Fixed Income Score: {score['total']}/100  {score['verdict_label']} {score['verdict']}\n\n"
                        f"Copy the scorecard table from the data block EXACTLY — do not add a weighted column.\n\n"
                        f"Use clear markdown. Be specific with numbers from the data. "
                        f"Today: {datetime.now().strftime('%B %d, %Y')}"
                    )

                else:
                    # ── Individual Bond / Sukuk ──────────────────────────────
                    instrument_type = "Sukuk" if is_sukuk else "Bond"

                    system_prompt = (
                        f"You are EisaX, an institutional CIO specialising in fixed income and Islamic finance.\n"
                        f"You have been given live instrument data for {name_str}.\n\n"
                        f"Produce a professional CIO-grade report with EXACTLY these sections:\n\n"
                        f"## 📊 {instrument_type} Analysis — {name_str}\n\n"
                        f"### 1. Instrument Overview\n"
                        f"   - Summarise key terms (ISIN, issuer, coupon, maturity, currency, exchange)\n"
                        f"   - Security type and market sector\n\n"
                        f"### 2. Yield Analysis\n"
                        f"   - Current coupon rate vs benchmark yields provided\n"
                        f"   - Spread in basis points (bps) and what it implies\n"
                        f"   - Estimated YTM commentary (based on time to maturity)\n\n"
                        f"### 3. Credit Risk Assessment\n"
                        f"   - Issuer credit profile and sovereign/corporate context\n"
                        f"   - Country rating — NOTE: if the rating has a staleness warning in the data, explicitly flag it\n"
                        f"   - Key downside risks (default, FX, liquidity)\n\n"
                    )

                    if is_sukuk:
                        system_prompt += (
                            f"### 4. Sukuk Structure\n"
                            f"   - Structure type (Ijara / Murabaha / Wakala / Mudarabah / Musharaka)\n"
                            f"   - Asset backing and SPV mechanics (if inferable)\n"
                            f"   - Sharia compliance confidence level\n"
                            f"   - How periodic distributions compare to conventional coupon\n\n"
                        )
                    else:
                        system_prompt += (
                            f"### 4. Bond Structure\n"
                            f"   - Seniority (Senior Unsecured / Subordinated / Secured)\n"
                            f"   - Covenant and call/put features if available in the data\n"
                            f"   - If seniority is not in the data, state 'Not available from ISIN lookup'\n\n"
                        )

                    system_prompt += (
                        f"### 5. FX & Currency Risk\n"
                        f"   - Issue currency vs likely investor base currency\n"
                        f"   - FX peg status (for AED/SAR/QAR etc.)\n"
                        f"   - Hedging context\n\n"
                        f"### 6. Liquidity Assessment\n"
                        f"   - Exchange listing and secondary market depth\n"
                        f"   - Typical bid-ask spread estimate\n"
                        f"   - Suitable investor type (retail / institutional / HNWI)\n\n"
                        f"### 7. Investment Verdict\n"
                        f"   - Clear BUY / HOLD / AVOID recommendation\n"
                        f"   - Who should invest (income seekers, Islamic funds, GCC investors, etc.)\n"
                        f"   - Entry conditions and risk limits\n\n"
                        f"## 🎯 EisaX Fixed Income Score: {score['total']}/100  {score['verdict_label']} {score['verdict']}\n\n"
                        f"Copy the scorecard table from the data block EXACTLY — do not add a weighted column.\n"
                        f"If any factor shows N/A, note that it was excluded from scoring and the total was rescaled.\n\n"
                        f"Use clear markdown. Be specific with numbers from the data. "
                        f"Today: {datetime.now().strftime('%B %d, %Y')}"
                    )

                user_msg = f"{fi_context}\n\nUser question: {message}"

            # ── Case B: general fixed-income question (no ISIN) ──────────────
            else:
                logger.info("[fixed_income] General fixed-income query (no ISIN)")
                system_prompt = (
                    "You are EisaX, an institutional CIO specialising in fixed income, "
                    "Sukuk, government bonds, and Islamic finance products.\n\n"
                    "Answer the user's question with CIO-level depth. Include:\n"
                    "- Relevant yield context for the GCC/MENA region\n"
                    "- Sukuk structures (Ijara, Wakala, Murabaha, etc.) where relevant\n"
                    "- Credit quality and ratings context\n"
                    "- Investment suitability (retail, institutional, Islamic funds)\n\n"
                    "For specific ISIN analysis, ask the user to provide the ISIN "
                    "(e.g. XS1234567890 — 12-character code starting with 2 letters).\n\n"
                    "Use markdown with headers. Be concise and data-driven. "
                    f"Today: {datetime.now().strftime('%B %d, %Y')}"
                )
                user_msg = message
                data  = {}
                score = {}
                fi_context = ""

            # ── LLM call ─────────────────────────────────────────────────────
            client = self.client_factory()
            response = client.create_completion(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
            )
            reply = response.choices[0].message.content if hasattr(response, "choices") else str(response)

            return {
                "type": "chat.reply",
                "reply": reply,
                "data": {
                    "instrument": data,
                    "fi_score":   score,
                    "isin":       isin,
                },
            }

        except Exception as e:
            logger.error("[_handle_fixed_income] Error: %s", e, exc_info=True)
            isin_hint = extract_isin(message) or ""
            return {
                "type": "chat.reply",
                "reply": (
                    f"⚠️ Could not complete fixed income analysis"
                    f"{f' for **{isin_hint}**' if isin_hint else ''}.\n\n"
                    f"**Error:** {e}\n\n"
                    f"**For Sukuk/Bond analysis, please provide:**\n"
                    f"- The 12-character ISIN (e.g. `XS1234567890`, `AE000A1RKDU1`)\n"
                    f"- Found on the term sheet, prospectus, or your broker platform\n\n"
                    f"**Free ISIN lookups:**\n"
                    f"- [OpenFIGI](https://www.openfigi.com/search)\n"
                    f"- [ISIN.net](https://www.isin.net)\n"
                    f"- [Nasdaq Dubai](https://www.nasdaqdubai.com)\n"
                ),
                "data": {},
            }

    def _handle_optimize_inner(self, sid: str, mem: Dict[str, Any], msg: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        # ── Clean Pipeline (Step 1-5 architecture) ────────────────────────────
        try:
            from portfolio_pipeline import is_pipeline_request, run as pipeline_run
            if is_pipeline_request(msg):
                logger.info("[Portfolio] Pipeline request detected — routing to clean pipeline")
                report = pipeline_run(msg)
                return {
                    "type": "chat.reply",
                    "reply": report,
                    "data": {"agent": "finance", "analysis_type": "pipeline_report"},
                }
        except Exception as _pe:
            logger.warning("[Portfolio] Pipeline routing failed: %s — falling back to legacy path", _pe)

        # Logic copied/adapted from Orchestrator._handle_optimize

        tickers = IntentClassifier.extract_tickers(msg)
        # "Fresh Start" detection — English + Arabic (ابنى/انشئ/اعمل/جديد)
        fresh_start = any(v in msg.lower() for v in [
            "build", "create", "make", "generate", "new", "start",
            "ابنى", "ابني", "ابن ", "انشئ", "اعمل", "جديد", "بناء", "انشاء", "أنشئ"
        ])
        
        if not tickers and not fresh_start:
            # Only use memory tickers if they match the same market context
            mem_tickers = mem.get("tickers", [])
            if mem_tickers:
                # Detect market of message vs memory tickers
                msg_has_local = any(x in msg.upper() for x in [".SR", ".CA", ".DU", ".AE", ".KW", ".QA", "ARAMCO", "SABIC", "CIB", "EMAAR"])
                mem_has_local = any(t.upper().endswith((".SR", ".CA", ".DU", ".AE", ".KW", ".QA")) for t in mem_tickers)
                # Only use memory if market context matches
                if msg_has_local == mem_has_local:
                    tickers = mem_tickers
                else:
                    logger.info("[Portfolio] Skipping memory tickers — market context mismatch (msg=%s, mem=%s)", 
                                "local" if msg_has_local else "US", "local" if mem_has_local else "US")
        
        # Parse explicit constraints from message
        from core.portfolio import parse_constraints
        constraints = parse_constraints(msg)
        target_return = constraints.get("target_return")
        max_drawdown_val = constraints.get("max_drawdown")

        if not tickers and not state.get_artifact(sid):
            rp = pm.detect_risk_pref(msg) or mem.get("risk") or "medium"
            
            if rp == "high" or "aggressive" in msg.lower():
                tickers = pm.recommend_etfs("high")
                method = "max_sharpe"
            elif rp == "low" or "conservative" in msg.lower():
                tickers = pm.recommend_etfs("low")
                method = "min_vol"
            else:
                tickers = pm.recommend_etfs("medium")
                method = "max_sharpe"
            
            start = str(pm.get_param(mem, msg, "start", config.DEFAULT_START))
            w_raw, perf = pm.optimize_and_get_data(
                tickers=tickers, start=start, end=None, method=method,
                min_w=0.0, max_w=0.20, min_assets=4, seed_w=config.DEFAULT_SEED_W, rf=config.DEFAULT_RF,
                target_return=target_return, max_drawdown=max_drawdown_val
            )
            
            # Use RICH STRATEGY GUIDE for generic requests
            guide_md = pm.generate_strategy_guide_llm(
                risk_profile=rp,
                tickers=tickers,
                weights=w_raw,
                performance=perf
            )
            
            # Fix date
            import re as _re
            from datetime import datetime as _dt
            _correct = _dt.now().strftime("%B %d, %Y")
            guide_md = _re.sub(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2}', _correct, guide_md)
            guide_md = _re.sub(r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}', _correct, guide_md)
            guide_md = _re.sub(r'20\d{2}-\d{2}-\d{2}', _dt.now().strftime('%Y-%m-%d'), guide_md)

            # SAVE ARTIFACT
            state.set_artifact(sid, {
                "type": "strategy",
                "content": guide_md,
                "source": "self_generated",
                "exportable": True,
                "timestamp": datetime.now()
            })
            
            extra_mem = {
                "tickers": tickers, "method": method, "start": start, "end": None,
                "weights": w_raw, "performance": perf,
                "risk": rp,
                "min_w": 0.0, "max_w": 0.35
            }
            
            return {
                "type": "chat.reply",
                "reply": guide_md,
                "data": extra_mem
            }

        # Standard optimization
        start = str(pm.get_param(mem, msg, "start", config.DEFAULT_START))
        end = pm.get_param(mem, msg, "end", None)
        end = None if end in (None, "", "none", "null") else str(end)
        method = str(pm.get_param(mem, msg, "method", mem.get("method") or "max_sharpe")).lower()
        min_w = pm.parse_float(pm.get_param(mem, msg, "min_w", config.DEFAULT_MIN_W), config.DEFAULT_MIN_W)
        max_w = pm.parse_float(pm.get_param(mem, msg, "max_w", config.DEFAULT_MAX_W), config.DEFAULT_MAX_W)
        min_assets = pm.parse_int(pm.get_param(mem, msg, "min_assets", config.DEFAULT_MIN_ASSETS), config.DEFAULT_MIN_ASSETS)
        seed_w = pm.parse_float(pm.get_param(mem, msg, "seed_w", config.DEFAULT_SEED_W), config.DEFAULT_SEED_W)
        rf = pm.parse_float(pm.get_param(mem, msg, "rf", config.DEFAULT_RF), config.DEFAULT_RF)

        # Only call smart_expand when we have fewer than 3 explicit tickers
        if len(tickers) < 3:
            tickers = pm.smart_expand_tickers(msg, tickers)

        # Last-resort fallback: if still < 2 tickers, use risk-based ETF list
        if len(tickers) < 2:
            rp_fb = pm.detect_risk_pref(msg) or mem.get("risk") or "medium"
            tickers = pm.recommend_etfs(rp_fb)
            logger.info(f"[Optimize] Ticker fallback triggered → using recommend_etfs({rp_fb}): {tickers}")

        w_raw, perf = pm.optimize_and_get_data(
            tickers=tickers, start=start, end=end, method=method,
            min_w=min_w, max_w=max_w, min_assets=min_assets, seed_w=seed_w, rf=rf,
            target_return=target_return, max_drawdown=max_drawdown_val,
        )
        rp = pm.detect_risk_pref(msg) or mem.get("risk") or "medium"
        reply_text = pm.generate_strategy_guide_llm(
            risk_profile=rp,
            tickers=tickers,
            weights=w_raw,
            performance=perf,
            target_return=target_return,
            max_drawdown=max_drawdown_val,
        )
        
        # SAVE ARTIFACT
        state.set_artifact(sid, {
            "type": "portfolio",
            "content": reply_text,
            "source": "self_generated",
            "exportable": True,
            "timestamp": datetime.now()
        })

        extra_mem = {
            "tickers": tickers, "method": method, "start": start, "end": end,
            "weights": w_raw, "performance": perf,
            "metrics": {},
            "min_w": min_w, "max_w": max_w
        }
            
        return {
            "type": "chat.reply", 
            "reply": reply_text,
            "data": extra_mem
        }

    def _handle_report(self, sid: str, mem: Dict[str, Any], msg: str) -> Dict[str, Any]:
        """Legacy report handler (generates fresh). Redirects to export if explicit."""
        if "export" in msg.lower() or "pdf" in msg.lower():
            return self._handle_export(sid, mem, msg)
            
        # ... logic to generate report body ...
        # Reuse existing logic but ensuring we populate last_artifact too
        return self._handle_export(sid, mem, msg, force_refresh=True)

    def _handle_export(self, sid: str, mem: Dict[str, Any], msg: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Handles explicit export requests.
        Prioritizes state.last_artifact.
        Falls back to generating a report from memory tokens.
        """
        from core.report_engine import ReportEngine

        # ── Hard Rejection Gate ────────────────────────────────────────────────
        _perf_export = mem.get("performance") or {}
        _exp_ret_ex  = float(_perf_export.get("expected_return", 0) or 0)
        _sharpe_ex   = float(_perf_export.get("sharpe", 0) or 0)
        if _perf_export and (_exp_ret_ex < 0.045 or _sharpe_ex < 0):
            _w_ex  = mem.get("weights") or mem.get("weights_raw") or {}
            _top3e = sorted(_w_ex.items(), key=lambda x: -x[1])[:3] if _w_ex else []
            _fix_lines = []
            if _exp_ret_ex < 0.045:
                _fix_lines.append("أضف أسهم US أو Gold أو Bonds لرفع العائد المتوقع فوق معدل الخطر الصفري (4.5%)")
            if _sharpe_ex < 0:
                _fix_lines.append("العائد المتوقع أقل من معدل الخطر الصفري — المحفظة لا تُعوّض المستثمر عن المخاطرة")
            _fixes_md = "\n".join(f"- {f}" for f in _fix_lines) if _fix_lines else "- راجع مكونات المحفظة"
            _rejection = (
                "# ❌ Portfolio Rejected — Strategy Invalid\n\n"
                "**لا يمكن تنفيذ هذه المحفظة — المعايير الأساسية غير مستوفاة.**\n\n"
                "| المؤشر | القيمة | المطلوب |\n"
                "|--------|--------|---------|\n"
                f"| العائد المتوقع | **{_exp_ret_ex*100:.2f}%** | > 4.5% |\n"
                f"| Sharpe Ratio | **{_sharpe_ex:.2f}** | > 0 |\n\n"
                "## السبب\n\n"
                "المحفظة المقترحة **تخسر قيمتها** أو لا تُعوّض عن مخاطرها. "
                "تنفيذها سيضر بالمستثمر بدلاً من مساعدته.\n\n"
                "## الإصلاحات المقترحة\n\n"
                f"{_fixes_md}\n\n"
                "## جرّب بدلاً من ذلك\n\n"
                "> ابني محفظة **balanced** باستخدام **US + GCC + Gold** لضمان عائد إيجابي ومتوازن.\n"
            )
            return {"type": "chat.reply", "reply": _rejection}
        # ── End Rejection Gate ─────────────────────────────────────────────────

        # ── Normalize stale tickers in session memory ─────────────────────────
        # Old sessions may have "UAE", "SAUDI" etc. in mem — fix them in-place
        # before any report path runs.
        try:
            if mem.get("tickers"):
                mem["tickers"] = pm._normalize_tickers(mem["tickers"])
            if mem.get("weights"):
                _old_w = mem["weights"]
                _new_w = {}
                for _t, _w in _old_w.items():
                    _mapped = pm._TICKER_MAP.get(_t.upper(), _t)
                    if _mapped:
                        _new_w[_mapped] = _new_w.get(_mapped, 0) + _w
                mem["weights"] = _new_w
        except Exception:
            pass

        # ── Placeholder gate (also covers stale cached artifacts) ─────────────
        _w_for_gate = mem.get("weights") or mem.get("weights_raw") or {}
        _ph = pm.has_placeholder_tickers(_w_for_gate)
        if _ph:
            _block_msg = (
                "# ⛔ Report Blocked — Unverified Assets\n\n"
                f"**Placeholder tickers detected:** `{'`, `'.join(_ph)}`\n\n"
                "EisaX cannot produce a client-facing report with unidentified securities.\n\n"
                "**Fix:** Ask EisaX to rebuild the portfolio — "
                "the optimizer will select verified assets from the live market library.\n\n"
                "> **Rule:** Every asset must be verified by ticker, name, and market.\n"
            )
            return {"type": "chat.reply", "reply": _block_msg}
        # ─────────────────────────────────────────────────────────────────────

        content_to_export = ""
        title = f"EISAX Report {datetime.now().strftime('%Y-%m-%d')}"

        # 1. Check valid artifact — but invalidate cache if it contains placeholders
        _artifact_cached = state.get_artifact(sid)
        _cached_content = _artifact_cached.get("content", "") if _artifact_cached else ""
        _cache_has_placeholder = any(
            f"`{p}`" in _cached_content or f" {p} " in _cached_content or f"/{p}" in _cached_content
            for p in pm._ALL_FAKE_TICKERS
        ) if _cached_content else False

        if not force_refresh and _artifact_cached and _artifact_cached.get("exportable") and not _cache_has_placeholder:
            content_to_export = _artifact_cached["content"]
            # Try to derive title from content header?
            first_line = content_to_export.strip().split('\n')[0]
            if first_line.startswith("# "):
                title = first_line[2:].strip()
        else:
            # 2. Fallback: Generate from Memory (Tickers)
            tickers = mem.get("tickers", [])
            if not tickers and not state.get_artifact(sid):
                return {"type": "chat.reply", "reply": "I don't have a generated report or portfolio to export yet. Try 'Optimize my portfolio' first."}
            
            # Generate fresh report
            base_report_md = f"# Investment Report\n\nPortfolio: {', '.join(tickers)}\n\n"
            base_report_md += pm.build_portfolio_report_body(mem)
            content_to_export = pm.generate_executive_report_llm(
                model=config.DEFAULT_MODEL, temperature=0.2, mem=mem, base_report_md=base_report_md
            )
            
            # SAVE ARTIFACT (so prompt loop stops)
            state.set_artifact(sid, {
                "type": "report",
                "content": content_to_export,
                "source": "self_generated",
                "exportable": True,
                "timestamp": datetime.now()
            })

        # 3. Generate PDF
        try:
            engine = ReportEngine()
            pdf_path = engine.generate_pdf(title, content_to_export)
            filename = os.path.basename(pdf_path)
            
            logger.info(f"[Finance] EXPORT SUCCESS: {filename} | Content length: {len(content_to_export)}")
            return {
                "type": "report.export",
                # DIRECT LINK, NO QUESTIONS
                "reply": f"Here is your PDF document: **{title}**\n\n[Download PDF](/static/reports/{filename})", 
                "data": {
                    "format": "pdf", 
                    "printable": True, 
                    "url": f"/static/reports/{filename}",
                    "download_url": f"/static/reports/{filename}",
                    "filename": filename,
                    "title": title
                }
            }
        except Exception as e:
             return {"type": "error", "reply": f"Export failed: {e}"}

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

    # ══════════════════════════════════════════════════════════════════════
    # Helper methods – extracted from _handle_analytics to remove duplication
    # ══════════════════════════════════════════════════════════════════════

    # ── Static helpers re-exported from core.agents.finance_helpers ─────────
    # These aliases keep all existing self._method() / FinancialAgent._method()
    # call-sites working without change.
    _compute_decision_confidence = staticmethod(_compute_decision_confidence)

    def _build_decision_framework_block(self,
                                        *,
                                        verdict: str,
                                        confidence: int,
                                        conviction: str,
                                        conviction_note: str = "",
                                        beta: float,
                                        current_price: float,
                                        entry_price: float,
                                        sma50: float,
                                        next_earnings: Optional[str],
                                        currency_sym: str,
                                        is_local_mkt: bool,
                                        is_arabic: bool,
                                        is_crypto: bool = False,
                                        is_etf: bool = False,
                                        is_commodity: bool = False,
                                        is_reit: bool = False) -> str:
        """Build a compact advisory layer: confidence, uncertainty, horizon, and no-action case."""
        _verdict = str(verdict or "HOLD").upper()
        _beta = float(beta or 1.0)
        _cp = float(current_price or 0.0)
        _ep = float(entry_price or 0.0)
        _sma50 = float(sma50 or 0.0)
        _conviction_note_line = f"{conviction_note}\n" if conviction_note else ""

        def _fmt_price(v: float) -> str:
            if not v:
                return "N/A"
            return f"{v:,.2f} {currency_sym}" if is_local_mkt else f"${v:,.2f}"

        if _verdict in ("SELL", "REDUCE", "AVOID"):
            no_action_en = (
                "If price remains below SMA50 and no catalyst quality improves, keep exposure unchanged and await trend repair confirmation."
            )
            no_action_ar = "لو السعر فضل تحت SMA50 ومفيش تحسن واضح في الكاتاليست، الأفضل تثبيت المراكز والانتظار لحد تأكيد إصلاح الاتجاه."
        elif _verdict == "HOLD":
            if is_crypto:
                no_action_en = (
                    "If price stays in a range with no directional catalyst or on-chain signal shift, keep allocation unchanged and await a higher-conviction setup."
                )
                no_action_ar = "لو السعر في نطاق جانبي بدون محفز اتجاهي أو تحول في إشارات الأونشين، يفضل الإبقاء على التوزيع الحالي وانتظار إشارة أوضح."
            else:
                no_action_en = (
                    "If price stays in a range with no catalyst surprise, keep allocation unchanged and monitor entry conditions before adding."
                )
                no_action_ar = "لو السعر في نطاق جانبي بدون مفاجآت محفزة، يفضل الإبقاء على التوزيع الحالي ومراقبة شروط الدخول قبل أي إضافة."
        elif _cp and _ep and _cp > (_ep * 1.02):
            no_action_en = f"If price stays above the preferred entry zone ({_fmt_price(_ep)}), avoid chasing and await pullback confirmation."
            no_action_ar = f"لو السعر استمر أعلى من منطقة الدخول المفضلة ({_fmt_price(_ep)})، الأفضل عدم المطاردة وانتظار تأكيد البولباك."
        else:
            no_action_en = (
                "If confirmation above SMA50 fails, avoid adding and keep allocation unchanged."
            )
            no_action_ar = "لو فشل تأكيد الإغلاق أعلى SMA50، الأفضل عدم زيادة المراكز والإبقاء على التوزيع الحالي."

        # ── Uncertainty Driver 1: asset-type-specific ────────────────────
        if is_crypto:
            earn_u_en = "On-chain signals (MVRV, exchange outflows) and macro liquidity cycle shifts can rapidly reprice the asset."
            earn_u_ar = "إشارات الأونشين (MVRV، تدفق البورصات) وتحولات دورة السيولة الكلية قد تعيد التسعير بسرعة."
        elif is_etf or is_commodity:
            earn_u_en = "Supply/demand balance shifts, OPEC+ decisions, or macro regime change may materially move this asset."
            earn_u_ar = "تحولات ميزان العرض والطلب أو قرارات أوبك+ أو تغيير النظام الكلي قد تؤثر جوهرياً على هذا الأصل."
        elif is_reit:
            earn_u_en = "Rate path and occupancy cycle data are the primary re-rating triggers for this REIT."
            earn_u_ar = "مسار أسعار الفائدة ودورة الإشغال هما المحركان الأساسيان لإعادة تقييم صندوق العقارات هذا."
        else:
            earn_u_en = f"Upcoming earnings on {next_earnings} may materially reset guidance." if next_earnings else "Upcoming earnings/guidance timing may reset the thesis."
            earn_u_ar = f"نتائج الأرباح القادمة في {next_earnings} قد تعيد ضبط الفرضية بالكامل." if next_earnings else "توقيت الأرباح/التوجيهات القادمة قد يعيد تشكيل الفرضية."

        # ── Uncertainty Driver 2: beta / volatility ──────────────────────
        if is_crypto:
            macro_u_en = "Crypto vol is structurally elevated (annualised ~60-90%); position sizing should reflect this — not standard equity beta."
            macro_u_ar = "تذبذب العملات الرقمية مرتفع هيكلياً (~60-90% سنوياً)؛ يجب أن يعكس حجم المركز هذا الواقع لا معامل بيتا التقليدي."
        elif is_reit:
            macro_u_en = "Rate sensitivity is the dominant risk factor; a 50bps move in the 10Y can shift NAV by 5-10%."
            macro_u_ar = "حساسية أسعار الفائدة هي عامل المخاطر الأساسي؛ حركة 50 نقطة أساس في العشر سنوات قد تحرك NAV بنسبة 5-10%."
        elif _beta >= 1.5:
            macro_u_en = f"High beta sensitivity ({_beta:.2f}x) can amplify market drawdowns."
            macro_u_ar = f"حساسية بيتا مرتفعة ({_beta:.2f}x) وقد تضخم أي هبوط سوقي."
        elif _beta >= 1.1:
            macro_u_en = f"Moderate beta/rate sensitivity ({_beta:.2f}x) can shift risk/reward quickly."
            macro_u_ar = f"حساسية بيتا/الفائدة متوسطة ({_beta:.2f}x) وقد تغير معادلة المخاطرة بسرعة."
        else:
            macro_u_en = "Macro beta sensitivity is limited; thesis risk is more company-specific."
            macro_u_ar = "حساسية البيتا الكلية محدودة، والمخاطر الأكبر مرتبطة بتنفيذ الشركة نفسها."

        if is_arabic:
            return (
                "\n\n---\n"
                "## إطار القرار (Advisory Layer)\n"
                f"- **ثقة القرار:** {confidence}%\n"
                f"{_conviction_note_line}"
                "- **الأفق الزمني:** تكتيكي 1-3 أشهر | استراتيجي 12-36 شهر\n"
                f"- **حالة عدم اتخاذ إجراء:** {no_action_ar}\n"
                f"- **عوامل عدم اليقين:** 1) {earn_u_ar} 2) {macro_u_ar}\n"
                "> الهدف من هذا الإطار هو دعم القرار وليس إصدار أوامر تنفيذية مباشرة."
            )

        return (
            "\n\n---\n"
            "## Decision Framework (Advisory Layer)\n"
            f"- **Verdict Confidence:** {confidence}% (Conviction: {conviction})\n"
            f"{_conviction_note_line}"
            "- **Time Horizon:** Tactical 1-3 months | Strategic 12-36 months\n"
            f"- **No-Action Case:** {no_action_en}\n"
            f"- **Primary Uncertainty Drivers:** 1) {earn_u_en} 2) {macro_u_en}\n"
            "> This layer is advisory and supports decision quality; it is not an execution command."
        )

    _soften_execution_language = staticmethod(_soften_execution_language)

    _round_scenario_prices = staticmethod(_round_scenario_prices)

    _fetch_onchain = staticmethod(_fetch_onchain)

    def _compute_rolling_beta(self, ticker: str) -> float:
        """Compute 90-day rolling beta vs SPY for any ticker. Returns float."""
        try:
            import numpy as _np
            from core.data import get_prices as _gp
            _prices = _gp([ticker, "SPY"], start="2025-01-01", end=None, force_refresh=False)
            if ticker in _prices.columns and "SPY" in _prices.columns:
                _cr = _prices[ticker].pct_change().dropna()
                _sr = _prices["SPY"].pct_change().dropna()
                _common = _cr.index.intersection(_sr.index)
                if len(_common) > 30:
                    _cv = _np.cov(_cr.loc[_common].values, _sr.loc[_common].values)
                    beta = round(float(_cv[0, 1] / _cv[1, 1]) if _cv[1, 1] != 0 else 1.0, 2)
                    return max(0.3, min(beta, 4.0))
        except Exception as _e:
            logger.warning(f"[RollingBeta] Failed for {ticker}: {_e}")
        return 1.5 if ticker.endswith('-USD') else 1.0

    def _build_scorecard_md(self, target, real_price, analyst_target, fund, summary, dc_data, forward_pe, fg_data=None, onchain=None, effective_beta=None, display_target=None, target_is_estimate=False, target_is_sma=False, analyst_consensus=None, change_pct=0):
        """Build the EisaX Proprietary Score Card markdown block."""
        try:
            from core.scorecard import calculate_score, get_verdict
            # Use display_target (analyst OR EisaX FV) for upside/score calculations
            _display_target = display_target if display_target is not None else analyst_target
            _target_is_estimate = target_is_estimate

            # ── Use pre-computed beta or sector-appropriate default (NOT 1.0) ──
            _is_crypto_t = target.endswith('-USD') and any(c in target for c in ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'ADA', 'AVAX'])
            _beta_raw = effective_beta or float(fund.get('beta') or dc_data.get('beta') or summary.get('beta') or 0)
            if _beta_raw and _beta_raw > 0:
                _sc_beta = _beta_raw
            else:
                # Sector-appropriate default — NOT 1.0 which overstates risk for defensive stocks
                _s_for_beta = (fund.get('sector', '') or '').lower()
                _sc_beta = (1.5 if _is_crypto_t
                            else 0.4 if any(x in _s_for_beta for x in ('energy', 'oil', 'gas', 'utilities'))
                            else 0.7 if any(x in _s_for_beta for x in ('real estate', 'financials', 'banks'))
                            else 1.1)  # tech/general default

            # ── On-chain data for crypto ──
            _onchain = onchain or {}
            _real_ath = _onchain.get('ath') or float(fund.get('year_high') or 0)

            quality_score = (int(fund.get('fundamental_score')) if fund.get('fundamental_score') else None)
            if quality_score is not None:
                quality_score = min(quality_score, 95)

            sc_data = {
                'price': real_price or 0,
                'target': (_display_target or 0),          # analyst target OR EisaX FV estimate
                'target_is_estimate': _target_is_estimate, # flag: True = EisaX FV, not analyst
                'target_is_sma': target_is_sma,            # flag: True = SMA technical target
                'beta': _sc_beta,
                'mc': fund.get('market_cap') or 0,
                # Pass None for missing quality fields — scorecard normalises against available data
                'quality': quality_score,
                'forward_pe': float(dc_data.get('forward_pe') or forward_pe or 0),
                'ttm_pe': float(fund.get('pe_ratio') or 0),
                'net_margin': (float(fund.get('net_margin')) if fund.get('net_margin') else None),
                'gross_margin': (float(fund.get('gross_margin')) if fund.get('gross_margin') else None),
                'rev_growth': (float(fund.get('revenue_growth')) if fund.get('revenue_growth') else None),
                'roe': (float(fund.get('roe')) if fund.get('roe') else None),
                'debt_equity': (float(fund.get('debt_equity')) if fund.get('debt_equity') is not None and str(fund.get('debt_equity')) not in ('N/A', '', 'None') else None),
                'roic': (float(fund.get('roic')) if fund.get('roic') else None),
                'rsi': float(summary.get('rsi') or 50),
                'adx': float(summary.get('adx') or 20),
                'sma200': float(summary.get('sma_200') or fund.get('sma200') or summary.get('sma200') or 0),
                'year_high': _real_ath,
                'fear_greed': int((fg_data or {}).get('score', 50) or 50),
                'is_tech': fund.get('sector', '') in ['Technology', 'Semiconductors', 'Software', 'Communication Services'],
                'risk_count': sum(1 for r in ['concentration risk','valuation risk','multiple compression','cyclical','competition','regulatory','liquidity risk','high beta','interest rate'] if r in str(fund).lower() + str(summary).lower()),
                'is_bearish': summary.get('momentum') == 'Bearish',
                'ticker': target,
                'sector': fund.get('sector', 'Unknown'),
                # Real Wall Street analyst consensus — primary signal for Analyst Sentiment factor
                'analyst_consensus': analyst_consensus or '',
                'analyst_count': (int(fund.get('analyst_count') or 0) if fund.get('analyst_count') else 0),
                # Dividend yield (as decimal, e.g. 0.05 = 5%) — used in low-upside verdict override
                # yfinance may return '5.00%', '509.00%', 0.05, or 5.0 — normalise all to decimal
                'dividend_yield': _safe_div_yield(fund.get('dividend_yield') or dc_data.get('dividend_yield') or 0),
                'llm_verdict': (
                    'SELL' if (summary.get('trend') == 'Bearish' and summary.get('momentum') == 'Bearish')
                    else 'REDUCE' if (summary.get('trend') == 'Bearish' and summary.get('momentum') not in ('Bullish',) and not (_display_target and real_price and (_display_target - real_price) / real_price > 0.30))
                    # Override HOLD → BUY when upside > 30% AND analyst consensus = Strong Buy
                    else 'BUY' if (
                        summary.get('trend') == 'Bullish'
                        or (
                            _display_target and real_price
                            and ((_display_target - real_price) / real_price) > 0.30
                            and "strong buy" in (analyst_consensus or "").lower()
                        )
                    )
                    else 'HOLD'
                ),
                'is_crypto': bool(is_crypto if 'is_crypto' in dir() else False),
                'is_etf':    bool(_etf_meta_early is not None if '_etf_meta_early' in dir() else False),
                'avg_volume': float(fund.get('volume_avg90d') or fund.get('avg_volume') or 0),
                'annual_vol': float(summary.get('annual_vol', 0) or 0),
            }

            # Moat signals — guard against None values
            moat_count = 2 if (sc_data.get('quality') or 0) >= 90 else 1
            if (sc_data.get('net_margin') or 0) >= 30: moat_count += 1
            if (sc_data.get('rev_growth') or 0) >= 30: moat_count += 1
            # Regional market leader bonus — UAE/Saudi/Egypt dominant companies
            if any(str(target).upper().endswith(x) for x in ('.AE', '.DU', '.AD', '.SR', '.KW', '.CA')):
                moat_count += 1  # regional dominance premium
            sc_data['moat_signals'] = moat_count
            sc_data['has_moat'] = moat_count >= 2
            sc_data['daily_change_pct'] = change_pct or 0

            # ── Richer bullish/bearish signal counting ──
            _trend = summary.get('trend', '')
            _momentum = summary.get('momentum', '')
            _rsi_v = sc_data.get('rsi', 50)
            _sma200_v = sc_data.get('sma200', 0)
            _price_v = sc_data.get('price', 0)

            _bullish_c = 0
            if _trend == 'Bullish':         _bullish_c += 2
            if _momentum == 'Bullish':      _bullish_c += 1
            if _price_v and _sma200_v and _price_v > _sma200_v * 1.02:
                _bullish_c += 1                              # price above SMA200
            if 45 <= _rsi_v <= 65:          _bullish_c += 1  # healthy RSI range
            if summary.get('macd', 0) > summary.get('macd_signal', 0): _bullish_c += 1  # MACD bullish crossover

            _bearish_c = 0
            if _trend == 'Bearish':         _bearish_c += 2
            if _momentum == 'Bearish':      _bearish_c += 1
            if _price_v and _sma200_v and _price_v < _sma200_v * 0.95:
                _bearish_c += 1                              # price well below SMA200
            if _rsi_v < 30:                 _bearish_c += 1  # deeply oversold
            if _rsi_v > 80:                 _bearish_c += 1  # overbought → bearish risk

            sc_data['bullish_count'] = _bullish_c
            sc_data['bearish_count'] = _bearish_c
            result = calculate_score(sc_data)
            if not result:
                return ""

            factors, final, sc_data = result  # capture updated sc_data (has _score_capped, etc.)
            verdict_sc, emoji, conviction = get_verdict(final, sc_data)

            # ── Decision Engine binding layer (Week 4) ─────────────────────
            # Build interpretation labels from sc_data (ADX/RSI are deterministic).
            # Then apply hard constraints via build_decision — the result overrides
            # verdict_sc so the displayed verdict is NEVER contradicted by signals.
            try:
                from core.services.decision_engine import (
                    build_decision as _build_decision,
                    classify_decision_type as _classify_decision_type,
                )
                from core.services.interpretation_engine import (
                    build_interpretation_labels as _build_interp_labels_de,
                )
                _de_price   = float(sc_data.get('price') or 0)
                _de_sma200  = float(sc_data.get('sma200') or 0)
                _de_avgvol  = float(sc_data.get('avg_volume') or 0)
                _de_labels = _build_interp_labels_de(
                    adx=float(sc_data.get('adx') or 0),
                    rsi=float(sc_data.get('rsi') or 50),
                    price=_de_price,
                    entry_price=_de_sma200 or None,    # SMA200 as entry-zone reference
                    div_yield=float(sc_data.get('dividend_yield') or 0) or None,
                    volume_today=_de_avgvol or None,   # proxy; gives "normal" conviction
                    volume_avg=_de_avgvol or None,
                )
                _upside_for_de = (
                    (sc_data['target'] - sc_data['price']) / sc_data['price'] * 100
                    if sc_data.get('target') and sc_data.get('price')
                    else 0.0
                )
                _de_result = _build_decision(
                    interpretation_labels=_de_labels,
                    score_data={**sc_data, 'upside_pct': _upside_for_de,
                                'scorecard_verdict': verdict_sc,
                                'eisax_score': final},  # EisaX Score for Rule 8A
                )
                # Override verdict; keep emoji/conviction from scorecard
                verdict_sc    = _de_result['verdict']
                decision_type = _classify_decision_type(verdict_sc, _de_labels)

                # ── RULE 8A — Final enforcement in _build_scorecard_md ────────
                # Even after decision engine, if score ≥ 75 AND upside ≥ 20%:
                # Fundamental Verdict = BUY. Weak technicals = Entry Timing only.
                _upside_r8a = (
                    (sc_data['target'] - sc_data['price']) / sc_data['price'] * 100
                    if sc_data.get('target') and sc_data.get('price') else 0.0
                )
                if final >= 75 and _upside_r8a >= 20.0 and verdict_sc not in ('BUY', 'STRONG BUY'):
                    verdict_sc = 'BUY'
                    conviction = 'High' if final >= 80 else 'Medium'
                    emoji      = '🟢'
                    sc_data['_rule8a_applied'] = True
                    logger.info(
                        f"[Rule8A] {target}: Score={final}, Upside={_upside_r8a:.1f}%"
                        f" → Fundamental=BUY (was {_de_result['verdict']})"
                    )
            except Exception as _de_err:
                import logging as _de_log
                _de_log.getLogger(__name__).warning(
                    "[DecisionEngine] binding failed for %s: %s", target, _de_err
                )
                # Fallback: ADX-aware classification (no LLM trend_state)
                _de_adx = float(sc_data.get('adx') or 0)
                if verdict_sc in ("BUY", "STRONG BUY"):
                    if _de_adx >= 25:
                        decision_type = "trend_confirmed"
                    elif _de_adx >= 20:
                        decision_type = "early_reversal"
                    else:
                        decision_type = "contrarian_early"
                elif verdict_sc == "HOLD":
                    decision_type = "wait_for_confirmation"
                else:
                    decision_type = "trend_failure"
            # ── End Decision Engine ────────────────────────────────────────

            _div_info = _consensus_divergence(
                verdict_sc, analyst_consensus or '',
                adx=float(sc_data.get('adx') or 20),
                beta=float(sc_data.get('beta') or 1.0),
            )
            from core.scorecard import compute_entry_quality as _ceq
            _eq_score, _eq_label, _eq_note = _ceq(sc_data)
            _entry_quality_block = (
                f'\n**Entry Quality: {_eq_score}/100 — {_eq_label}**\n'
                f'*{_eq_note}*\n'
            )
            upside = ((sc_data['target'] - sc_data['price']) / sc_data['price'] * 100) if sc_data['target'] else 0

            # Display mapping — internal codes → institutional-grade client-facing labels
            _VERDICT_DISPLAY = {
                "REDUCE": "Positioning: Underweight",
                "SELL":   "Risk Stance: Avoid",
                "BUY":    "BUY",
                "HOLD":   "HOLD",
                "AVOID":  "Risk Stance: Avoid",
            }
            _DECISION_TYPE_LABELS = {
                "contrarian_early": "Contrarian Early",
                "early_reversal": "Early Reversal",
                "trend_confirmed": "Trend Confirmed",
                "wait_for_confirmation": "Wait For Confirmation",
                "trend_failure": "Trend Failure",
            }
            _decision_type_label = _DECISION_TYPE_LABELS.get(
                decision_type, decision_type.replace('_', ' ').title()
            )
            verdict_display = _VERDICT_DISPLAY.get(verdict_sc, verdict_sc)
            if verdict_sc == "BUY":
                verdict_display = f"Tactical BUY — {_decision_type_label}"

            # ── Entry Timing (from scorecard Rule 8A or get_verdict) ─────────
            _entry_timing = sc_data.get('entry_timing', '')
            # If entry_timing not yet set (no Rule8A path), derive from technicals
            if not _entry_timing:
                _adx_et = float(sc_data.get('adx', 0) or 0)
                _rsi_et = float(sc_data.get('rsi', 50) or 50)
                if verdict_sc in ('BUY', 'STRONG BUY'):
                    if _rsi_et > 70:
                        _entry_timing = 'WAIT — RSI overbought, await pullback'
                    elif _adx_et < 20:
                        _entry_timing = 'WAIT — trend not confirmed (ADX < 20)'
                    elif _adx_et < 25:
                        _entry_timing = 'ADD ON DIP — await ADX > 25'
                    else:
                        _entry_timing = 'BUY NOW — trend confirmed'
                elif verdict_sc in ('REDUCE', 'SELL', 'AVOID'):
                    _entry_timing = 'REDUCE INTO STRENGTH'
                else:
                    _entry_timing = 'WAIT'

            # ── English timing preserved before Arabic translation ─────────────
            _entry_timing_en = _entry_timing  # always English; needed for decision logic

            # Arabic timing labels (user-facing Quick View only; English kept in prompt)
            # _is_arabic_request lives in _handle_analytics scope — guard against NameError here
            _is_ar_sc = False
            try:
                _is_ar_sc = bool(_is_arabic_request)
            except NameError:
                pass
            if _is_ar_sc:
                _TIMING_AR = {
                    'WAIT — RSI overbought, await pullback': 'انتظر — مؤشر RSI في منطقة التشبع، انتظر تراجعًا',
                    'WAIT — trend not confirmed (ADX < 20)': 'انتظر — الاتجاه غير مؤكد (ADX أقل من 20)',
                    'ADD ON DIP — await ADX > 25': 'شراء تدريجي عند التراجع — انتظر ADX فوق 25',
                    'BUY NOW — trend confirmed': 'شراء الآن — الاتجاه مؤكد',
                    'REDUCE INTO STRENGTH': 'خفّف مع الارتفاع',
                    'WAIT': 'انتظر تأكيدًا',
                }
                _entry_timing = _TIMING_AR.get(_entry_timing, _entry_timing)

            # ── Persist decision data for _handle_analytics (no regex needed) ──
            self._last_scorecard_decision = {
                'verdict':     verdict_sc,
                'timing_en':   _entry_timing_en,   # English; used for WAIT/BUY logic
                'timing':      _entry_timing,       # Display form (may be translated)
                'score':       final,
                'conviction':  conviction,
                'emoji':       emoji,
            }

            if _is_crypto_t:
                # ── Crypto-specific scorecard display ──
                # ATH: priority chain → CoinGecko real ATH → sc_data year_high → fund year_high
                _ath = (
                    float(_onchain.get('ath') or 0)
                    or float(sc_data.get('year_high') or 0)
                    or float(fund.get('year_high') or 0)
                )
                _ath_dist = ((sc_data['price'] - _ath) / _ath * 100) if _ath and _ath > 0 else 0
                _fg = sc_data.get('fear_greed', 50)
                _fg_label = "Extreme Fear" if _fg <= 20 else "Fear" if _fg <= 40 else "Neutral" if _fg <= 60 else "Greed" if _fg <= 80 else "Extreme Greed"
                _fg_emoji = "🔴" if _fg <= 25 else "🟠" if _fg <= 45 else "🟡" if _fg <= 55 else "🟢" if _fg <= 75 else "🔴"
                _sma200 = sc_data.get('sma200', 0)
                _sma_dist = ((sc_data['price'] - _sma200) / _sma200 * 100) if _sma200 else 0
                # ATH from CoinGecko (real ATH, not just 52w high)
                _ath_date = _onchain.get('ath_date', '')
                _circ = _onchain.get('circulating_supply', 0)
                _max_s = _onchain.get('max_supply', 0)
                _supply_pct = _onchain.get('supply_ratio', 0)
                _vol_24h = _onchain.get('total_volume_24h', 0)
                _hash_eh = _onchain.get('hash_rate_eh', 0)
                _active_addr = _onchain.get('active_addresses', 0)
                _n_tx = _onchain.get('n_tx_24h', 0)
                # Market Cap Rank: hardcoded fallback for known assets
                _mc_rank_raw = _onchain.get('mc_rank', 0) or 0
                _CRYPTO_RANK_FALLBACK = {'BTC-USD': 1, 'ETH-USD': 2, 'BNB-USD': 3, 'SOL-USD': 4, 'XRP-USD': 5, 'DOGE-USD': 8, 'ADA-USD': 9, 'AVAX-USD': 10}
                _mc_rank = _mc_rank_raw if _mc_rank_raw and _mc_rank_raw > 0 else _CRYPTO_RANK_FALLBACK.get(target.upper(), None)
                # Format ATH display
                _ath_display = (f"{self._format_local_price(_ath, target)} ({_ath_dist:+.1f}%)" + (f" 📅 {_ath_date}" if _ath_date else "")) if _ath and _ath > 0 else "N/A"
                _rank_display = f"#{_mc_rank}" if _mc_rank else "N/A"

                sc_md = f"""
---

## 🎯 EisaX Crypto Score Card
**{target}** | Fundamental: **{verdict_display} {emoji}** | Timing: **{_entry_timing}** | Conviction: **{conviction}** | EisaX Score: **{final}/100** | Blended: **{sc_data.get('blended_score', final)}/100**

*Crypto-specific scoring: Network Dominance, SMA200 Valuation, ATH Recovery, On-Chain Metrics*

| Metric | Value |
|--------|-------|
| Live Price | {self._format_local_price(sc_data['price'], target)} |
| Beta (90d vs SPY) | {sc_data['beta']:.2f} {'⚠️ High Vol' if sc_data['beta'] > 2 else '🟡 Moderate' if sc_data['beta'] > 1.3 else '🟢'} |
| All-Time High | {_ath_display} |
| Price vs SMA200 | {_sma_dist:+.1f}% {'🔴 Below' if _sma_dist < -10 else '🟡 Near' if _sma_dist < 10 else '🟢 Above'} |
| Fear & Greed Index | {_fg}/100 {_fg_emoji} {_fg_label} |
| Market Cap Rank | {_rank_display} |"""
                # ── On-Chain Metrics section ──
                if _circ or _hash_eh or _active_addr:
                    sc_md += f"""

**⛓️ On-Chain Metrics:**

| Metric | Value |
|--------|-------|"""
                    if _circ and _max_s:
                        sc_md += f"\n| Supply (Circulating / Max) | {_circ:,.0f} / {_max_s:,.0f} ({_supply_pct}%) |"
                    elif _circ:
                        sc_md += f"\n| Circulating Supply | {_circ:,.0f} |"
                    if _vol_24h:
                        sc_md += f"\n| 24h Volume | ${_vol_24h/1e9:.1f}B |"
                    if _hash_eh:
                        sc_md += f"\n| Hash Rate | {_hash_eh:.0f} EH/s |"
                    if _active_addr:
                        sc_md += f"\n| Active Addresses (24h) | {_active_addr:,} |"
                    if _n_tx:
                        sc_md += f"\n| Transactions (24h) | {_n_tx:,} |"
            else:
                sc_md = f"""
---

## 🎯 EisaX Proprietary Score Card
**{target}** | Fundamental: **{verdict_display} {emoji}** | Timing: **{_entry_timing}** | Conviction: **{conviction}** | EisaX Score: **{final}/100** | Blended: **{sc_data.get('blended_score', final)}/100**

*Conviction driven by: {", ".join(filter(None, [
    # Upside — only show as positive driver when conviction is Medium/High (final >= 60)
    (f"strong upside potential (+{upside:.0f}%)" if upside > 20 else f"moderate upside (+{upside:.0f}%)" if upside > 10 else f"modest upside (+{upside:.0f}%)")
        if (upside > 5 and verdict_sc not in ("REDUCE", "SELL", "AVOID") and final >= 60) else None,
    # For low-score stocks with upside, note the conflict
    f"upside (+{upside:.0f}%) constrained by weak fundamentals/data gaps" if (upside > 15 and final < 60) else None,
    "upside limited by bearish technicals" if (verdict_sc in ("REDUCE", "SELL", "AVOID") and upside > 10) else None,
    "attractive valuation" if (factors.get("Valuation", (0,1))[0] / factors.get("Valuation", (0,1))[1]) >= 0.75 and verdict_sc not in ("REDUCE", "SELL", "AVOID") and final >= 60 else None,
    "strong quality fundamentals" if (factors.get("Quality Score", (0,1))[0] / factors.get("Quality Score", (0,1))[1]) >= 0.65 else None,
    # Technical — oversold is a caution signal for low-conviction stocks, not a driver
    "oversold — potential bounce but trend bearish" if ((sc_data.get('rsi') or 50) < 30 and final < 60) else
    "bullish technical momentum" if (factors.get("Technical Momentum", (0,1))[0] / factors.get("Technical Momentum", (0,1))[1]) >= 0.75 else None,
    "low risk profile" if (factors.get("Risk Profile", (0,1))[0] / factors.get("Risk Profile", (0,1))[1]) >= 0.80 and verdict_sc not in ("REDUCE", "SELL", "AVOID") else None,
    "strong market position" if (factors.get("Market Position", (0,1))[0] / factors.get("Market Position", (0,1))[1]) >= 0.75 and verdict_sc not in ("REDUCE", "SELL", "AVOID") else None,
    "fundamental data gaps limit conviction" if (sc_data.get('quality') is None or (sc_data.get('net_margin') is None and sc_data.get('roe') is None)) and final < 60 else None,
    "limited upside vs risk" if upside <= 2 and final < 60 else None,
    "bearish primary trend" if summary.get('trend') == 'Bearish' else None,
    "price below SMA200" if (sc_data.get('sma200') and sc_data['price'] < sc_data['sma200'] * 0.95) else None,
    "no analyst coverage — EisaX FV estimate only" if sc_data.get('target_is_estimate') and final >= 60 else
    "no analyst coverage + data gaps — speculative" if sc_data.get('target_is_estimate') and final < 60 else None,
])) or "balanced risk-reward profile"}*

| Metric | Value |
|--------|-------|
| Live Price | {self._format_local_price(sc_data['price'], target)} |
| Price Target | {(self._format_local_price(sc_data['target'], target) + f" (+{upside:.1f}%)" + (" *[SMA Tech.]*" if sc_data.get('target_is_sma') else " *[EisaX FV Est.]*" if sc_data.get('target_is_estimate') else (f" *[Sell-side consensus — {sc_data['analyst_count']} analysts]*" if sc_data.get('analyst_count') and sc_data['analyst_count'] > 0 else " *[Analyst-derived]*"))) if sc_data['target'] else "N/A"} |
| Beta | {f"{sc_data['beta']:.2f}" if sc_data.get('beta') else "N/A"} {'⚠️ High Risk' if (sc_data.get('beta') or 0) > 2 else ''} |
| Forward P/E | {f"{sc_data['forward_pe']:.1f}x" if sc_data.get('forward_pe') else 'N/A'} {'🟢 Reasonable' if 0 < (sc_data.get('forward_pe') or 0) < 30 else '🟡 High' if (sc_data.get('forward_pe') or 0) >= 30 else ''} |"""

            sc_md += """

**Factor Analysis:**

| Factor | Score | Bar |
|--------|-------|-----|"""
            _tm_note = ""
            for fname, (val, max_v) in factors.items():
                if fname == "Risk Profile":
                    # ── Display as RISK LEVEL % (higher = riskier, more intuitive) ──
                    _risk_raw = 100 - int((val / max_v) * 100)
                    # Priority: method param → sc_data → fund beta → 1.0 fallback
                    _eff_beta_rp = float(
                        effective_beta if (effective_beta and float(effective_beta) > 0) else
                        sc_data.get('beta') if (sc_data.get('beta') and float(sc_data.get('beta')) > 0) else
                        fund.get('beta') if (fund and fund.get('beta') and float(fund.get('beta')) > 0) else
                        1.0
                    )
                    _beta_floor = (
                        65 if _eff_beta_rp > 2.0 else
                        50 if _eff_beta_rp > 1.5 else
                        35 if _eff_beta_rp > 1.0 else
                        20 if _eff_beta_rp > 0.5 else 0
                    )
                    # Crypto annual-vol floor (high vol even with low market beta)
                    _ann_vol_rp = float(sc_data.get('annual_vol', 0) or 0)
                    if _ann_vol_rp > 0.60:
                        _beta_floor = max(_beta_floor, 50)
                    pct = max(_risk_raw, _beta_floor)
                    filled = int(pct / 10)
                    bar = "█" * filled + "░" * (10 - filled)
                    f_emoji = "🔴" if pct >= 65 else ("🟡" if pct >= 35 else "🟢")
                    sc_md += f"\n| {fname} | {pct}% Risk | {f_emoji} `{bar}` |"
                else:
                    pct = int((val / max_v) * 100)
                    filled = int((val / max_v) * 10)
                    bar = "█" * filled + "░" * (10 - filled)
                    f_emoji = "🟢" if pct >= 75 else ("🟡" if pct >= 50 else "🔴")
                    sc_md += f"\n| {fname} | {pct}% | {f_emoji} `{bar}` |"
                    if (
                        fname == "Technical Momentum"
                        and pct <= 0
                        and str(verdict_sc or "").upper() not in ("SELL", "AVOID")
                    ):
                        _tm_note = (
                            "\n\n*0% reflects current bearish price trend — not a fundamental deficiency. "
                            "Reversion toward SMA50 would recover this component.*"
                        )
            if _tm_note:
                sc_md += _tm_note

            # ── Pillar Breakdown — scoring methodology transparency ────────────
            # Groups factors into 3 economic pillars so every point is traceable.
            if not _is_crypto_t:
                _PILLAR_MAP = {
                    "🏦 Fundamentals":      {
                        "keys": ["Quality Score", "Valuation", "Market Position"],
                        "desc": "Quality • Valuation • Market Position",
                    },
                    "📈 Technical & Risk":  {
                        "keys": ["Price Upside", "Risk Profile", "Technical Momentum"],
                        "desc": "Upside Potential • Risk Profile • Momentum",
                    },
                    "💬 Analyst Sentiment": {
                        "keys": ["Analyst Sentiment"],
                        "desc": "Wall St Consensus",
                    },
                }
            else:
                _PILLAR_MAP = {
                    "🌐 Network & Dominance": {
                        "keys": ["Quality Score", "Network Dominance"],
                        "desc": "Market Cap Tier • Network Dominance",
                    },
                    "📈 Price & Technical":   {
                        "keys": ["Valuation", "ATH Recovery Potential", "Technical Momentum"],
                        "desc": "Price vs SMA • ATH Recovery • Momentum",
                    },
                    "⚡ Risk & Sentiment":    {
                        "keys": ["Risk Profile", "Analyst Sentiment"],
                        "desc": "Volatility Risk • Fear & Greed",
                    },
                }

            _pillar_rows = ""
            _ptotal_max  = 0
            _ptotal_earn = 0
            _analyst_na  = result[2].get('_analyst_na', False) if result else False
            for _pname, _pinfo in _PILLAR_MAP.items():
                _p_earned = sum(factors[k][0] for k in _pinfo["keys"] if k in factors)
                _p_max    = sum(factors[k][1] for k in _pinfo["keys"] if k in factors)
                if _p_max == 0:
                    continue
                _ptotal_earn += _p_earned
                _ptotal_max  += _p_max
                # Special label for Analyst Sentiment when N/A
                _is_analyst_pillar = ("Analyst Sentiment" in _pinfo["keys"])
                if _is_analyst_pillar and _analyst_na:
                    _pillar_rows += (
                        f"\n| {_pname} | No analyst coverage | {_p_max} | 0 | "
                        f"⚪ `░░░░░░░░░░` *(N/A)* |"
                    )
                else:
                    _p_pct  = int(_p_earned / _p_max * 100)
                    _p_bar  = "█" * int(_p_pct / 10) + "░" * (10 - int(_p_pct / 10))
                    _p_icon = "🟢" if _p_pct >= 70 else ("🟡" if _p_pct >= 50 else "🔴")
                    _pillar_rows += (
                        f"\n| {_pname} | {_pinfo['desc']} | {_p_max} | {_p_earned} | "
                        f"{_p_icon} `{_p_bar}` |"
                    )

            # No rescaling note needed — max is always 100 now
            _rescale_note = ""
            sc_md += f"""

**📊 Score Breakdown:**

| Pillar | Factors | Max | Earned | Score |
|--------|---------|-----|--------|-------|{_pillar_rows}
| **TOTAL** | *(all pillars)* | **{_ptotal_max}** | **{_ptotal_earn}** | **{f"Raw: {sc_data['_raw_score']} → Capped: {final} (below SMA200)" if sc_data.get('_score_capped') and sc_data.get('_raw_score') and sc_data['_raw_score'] != final else f"{final}/100"}** |

> *Score is 100% deterministic — computed from live market data using explicit mathematical thresholds. No LLM estimation. Every point is traceable to a specific data input.*"""

            filled_big = int((final / 100) * 20)
            big_bar = "█" * filled_big + "░" * (20 - filled_big)
            # Show cap explanation if score was reduced
            _cap_note = ""
            if sc_data.get('_score_capped') and sc_data.get('_raw_score') and sc_data['_raw_score'] != final:
                _raw = sc_data['_raw_score']
                _cap_note = f"\n\n> ⚠️ **Technical Override:** Raw score was **{_raw}/100** → capped to **{final}/100** because price is below SMA200. Upgrade to BUY requires reclaiming SMA200 (${sc_data.get('sma200', 0):,.2f})."
            sc_md += f"""

**Overall: `{big_bar}` {final}/100**

> *EisaX Proprietary Score | Abu Dhabi*{_cap_note}
---"""
            return sc_md
        except Exception as e:
            logger.error(f"[Scorecard] failed: {e}")
            return ""

    def _build_factcheck_block(self, real_price, fund, summary, dc_data, forward_pe,
                               next_earnings=None, fg_data=None, ticker="", effective_beta=None):
        """Build the fact-check verification block with earnings urgency flag + Fear & Greed."""
        try:
            from datetime import datetime as _dt2
            _today   = _dt2.now().strftime("%b %d, %Y")
            _is_sar  = str(ticker).upper().endswith(".SR")
            _is_aed  = str(ticker).upper().endswith((".AE", ".DU"))
            _is_egp  = str(ticker).upper().endswith(".CA")
            _is_kwf  = str(ticker).upper().endswith(".KW")
            _is_qar  = str(ticker).upper().endswith(".QA")
            _sym     = ("﷼" if _is_sar else "د.إ" if _is_aed else "ج.م" if _is_egp else
                        "ف" if _is_kwf else "ر.ق" if _is_qar else "$")
            _is_local_price = _is_sar or _is_aed or _is_egp or _is_kwf or _is_qar
            _fp2 = real_price or summary.get('price', 0)
            _live_price = (f"{_fp2:,.2f} {_sym}" if _is_local_price and _fp2
                          else f"${_fp2:,.2f}" if _fp2 else "N/A")

            # ── Beta: single source of truth = effective_beta (pre-validated,
            #    same value as scorecard). Raw yfinance is not used directly
            #    because it can return garbage like -0.01 for GCC stocks.
            _beta_eff_fc = float(effective_beta) if effective_beta else 0.0
            if _beta_eff_fc < 0:
                _beta_live = "Not reliable"       # negative beta is garbage data
            elif _beta_eff_fc > 5:
                _beta_live = "Not reliable"       # absurdly high — don't display
            elif _beta_eff_fc > 0:
                _is_crypto_fc = str(ticker).upper().endswith('-USD')
                _beta_note = " *(rolling)*" if _is_crypto_fc else ""
                _beta_live = f"{_beta_eff_fc:.2f}{_beta_note}"
            else:
                # effective_beta is 0 (unavailable) — try dc_data only (StockAnalysis)
                _beta_dc_fc = float(dc_data.get('beta') or 0)
                if 0 < _beta_dc_fc <= 5:
                    _beta_live = f"{_beta_dc_fc:.2f}"
                elif _beta_dc_fc < 0 or _beta_dc_fc > 5:
                    _beta_live = "Not reliable"
                else:
                    _beta_live = 'N/A'
            # ── P/E sanity: values ≤ 0 or > 200 are not meaningful to display ──
            _pe_raw    = fund.get('pe_ratio') or (
                float(dc_data.get('pe_ratio', 0)) if dc_data.get('pe_ratio') else 0)
            _pe_float  = float(_pe_raw) if _pe_raw else 0.0
            _pe_live   = ("Not reliable" if _pe_float > 200
                          else f"{_pe_float:.1f}x" if _pe_float > 0
                          else 'N/A')
            _fpe_raw   = float(dc_data.get('forward_pe') or forward_pe or 0)
            _fpe_live  = ("Not reliable" if _fpe_raw > 200
                          else f"{_fpe_raw:.1f}x" if _fpe_raw > 0
                          else 'N/A')
            # Crypto: inject market cap from CoinGecko
            if not dc_data.get('market_cap') and str(ticker).upper().endswith(('-USD', '-USDT')):
                try:
                    import requests as _rq2
                    _cg_map = {'BTC-USD':'bitcoin','ETH-USD':'ethereum','SOL-USD':'solana',
                               'XRP-USD':'ripple','BNB-USD':'binancecoin','DOGE-USD':'dogecoin'}
                    _cg_id = _cg_map.get(ticker.upper())
                    if _cg_id:
                        _r = _rq2.get(f'https://api.coingecko.com/api/v3/simple/price'
                                      f'?ids={_cg_id}&vs_currencies=usd&include_market_cap=true',
                                      timeout=5)
                        if _r.status_code == 200:
                            _mc = _r.json().get(_cg_id, {}).get('usd_market_cap', 0)
                            if _mc > 0:
                                dc_data['market_cap'] = (f"${_mc/1e12:.2f}T" if _mc >= 1e12
                                                         else f"${_mc/1e9:.0f}B")
                except Exception:
                    pass
            # Stocks: جيب market cap و 52W من yfinance لو مش موجود
            if not dc_data.get('market_cap') or not dc_data.get('week_52_range'):
                try:
                    import yfinance as _yfc3
                    _fi = _yfc3.Ticker(ticker).fast_info
                    if not dc_data.get('market_cap'):
                        _mc = getattr(_fi, 'market_cap', None)
                        if _mc and _mc > 0:
                            # Local currency display for regional stocks
                            _is_local_mc = str(ticker).upper().endswith(('.AE', '.DU', '.AD', '.SR', '.KW', '.CA'))
                            if str(ticker).upper().endswith(('.AE', '.DU', '.AD')):
                                _mc_sym, _mc_sfx = 'AED ', ''
                            elif str(ticker).upper().endswith('.SR'):
                                _mc_sym, _mc_sfx = 'SAR ', ''
                            elif str(ticker).upper().endswith('.KW'):
                                _mc_sym, _mc_sfx = 'KWD ', ''
                            elif str(ticker).upper().endswith('.CA'):
                                _mc_sym, _mc_sfx = 'EGP ', ''
                            else:
                                _mc_sym, _mc_sfx = '$', ''
                            dc_data['market_cap'] = (f"{_mc_sym}{_mc/1e12:.2f}T" if _mc >= 1e12
                                                     else f"{_mc_sym}{_mc/1e9:.2f}B" if _mc >= 1e9
                                                     else f"{_mc_sym}{_mc/1e6:.0f}M")
                    if not dc_data.get('week_52_range'):
                        _lo = getattr(_fi, 'year_low', None)
                        _hi = getattr(_fi, 'year_high', None)
                        if _lo and _hi:
                            # Use local currency symbol for regional stocks, $ for US
                            _52w_sym = (_sym if (_is_sar or _is_aed or _is_egp) else '$')
                            dc_data['week_52_range'] = f"{_52w_sym}{_lo:,.2f} - {_52w_sym}{_hi:,.2f}"
                except Exception:
                    pass
            # ── DB fallback for Market Cap + 52W (covers UAE/Saudi/Egypt) ──────
            if not dc_data.get('market_cap') or not dc_data.get('week_52_range'):
                try:
                    import sqlite3 as _sq3
                    from core.config import CORE_DB as _cfg_core_db
                    _dbc = _sq3.connect(str(_cfg_core_db))
                    _dbr = _dbc.execute(
                        "SELECT market_cap, week_52_high, week_52_low FROM uae_fundamentals WHERE ticker=? LIMIT 1",
                        (str(ticker).upper(),)
                    ).fetchone()
                    _dbc.close()
                    if _dbr:
                        _db_mc, _db_hi, _db_lo = _dbr
                        if not dc_data.get('market_cap') and _db_mc and _db_mc > 0:
                            _sym_mc = ('AED ' if str(ticker).upper().endswith(('.AE','.DU'))
                                       else 'SAR ' if str(ticker).upper().endswith('.SR')
                                       else 'EGP ' if str(ticker).upper().endswith('.CA')
                                       else '$')
                            dc_data['market_cap'] = (f"{_sym_mc}{_db_mc/1e12:.2f}T" if _db_mc >= 1e12
                                                     else f"{_sym_mc}{_db_mc/1e9:.2f}B" if _db_mc >= 1e9
                                                     else f"{_sym_mc}{_db_mc/1e6:.0f}M")
                        if not dc_data.get('week_52_range') and _db_hi and _db_lo:
                            _52w_sym_db = (_sym if (_is_sar or _is_aed or _is_egp) else '$')
                            dc_data['week_52_range'] = f"{_52w_sym_db}{_db_lo:,.2f} – {_52w_sym_db}{_db_hi:,.2f}"
                except Exception:
                    pass
            _mc_live   = dc_data.get('market_cap') or 'N/A'
            _range_live = dc_data.get('week_52_range') or 'N/A'

            # LOCAL PRICE row (SAR / AED / EGP)
            _local_lbl = "SAR" if _is_sar else ("AED" if _is_aed else ("EGP" if _is_egp else ""))
            _local_row = (f"| Local Price ({_local_lbl}) | — | {real_price:,.2f} {_sym} | ➕ |\n"
                          if (_is_sar or _is_aed or _is_egp) and real_price else "")

            # ── Earnings date: skip past dates, label recent ones ──
            _earnings_raw  = next_earnings or dc_data.get('earnings_date') or fund.get('next_earnings_date') or 'N/A'
            _earnings_live = _earnings_raw
            _earnings_flag = ""
            try:
                from datetime import datetime as _dt3
                for _fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
                    try:
                        _earn_dt  = _dt3.strptime(str(_earnings_raw).split("T")[0].strip(), _fmt)
                        _days_to  = (_earn_dt - _dt3.now()).days
                        if _days_to < 0:
                            # Earnings already happened — label it as "recently reported"
                            _earnings_live = f"{_earnings_raw} *(recently reported — awaiting next date)*"
                        elif 0 <= _days_to <= 3:
                            _earnings_flag = f"\n\n> ⚠️ **URGENT CATALYST:** Earnings in **{_days_to} day(s)** ({_earnings_raw}). High volatility expected."
                        elif _days_to <= 14:
                            _earnings_flag = f"\n\n> 📅 **NEAR-TERM CATALYST:** Earnings in {_days_to} days ({_earnings_raw})."
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

            # Fear & Greed row
            _fg = fg_data or {}
            _fg_score  = _fg.get('score')
            _fg_rating = _fg.get('rating', '')
            _fg_label  = _fg.get('label_ar', '')
            _fg_emoji  = (
                "🔴" if _fg_score is not None and _fg_score < 25 else
                "🟠" if _fg_score is not None and _fg_score < 45 else
                "🟡" if _fg_score is not None and _fg_score < 55 else
                "🟢" if _fg_score is not None and _fg_score < 75 else
                "💹" if _fg_score is not None else "—"
            )
            _fg_row = (f"| Fear & Greed | — | {_fg_emoji} {int(_fg_score)} — {_fg_rating} | ➕ |\n"
                       if _fg_score is not None else "")

            return f"""\n\n---
🔍 **FACT-CHECK** *(Verified {_today})*

| Metric | Report | Live | Status |
|--------|--------|------|--------|
| Price | {_live_price} | {_live_price} | ✅ |
{_local_row}| Beta | — | {_beta_live} | ➕ |
| P/E (TTM) | — | {_pe_live} | ➕ |
| Forward P/E | — | {_fpe_live} | ➕ |
| Market Cap | — | {_mc_live} | ➕ |
| 52W Range | — | {_range_live} | ➕ |
{_fg_row}📅 **Next Earnings:** {_earnings_live}{_earnings_flag}

*Source: Yahoo Finance + StockAnalysis + CNN Fear&Greed — live at time of query*"""
        except Exception as e:
            logger.error(f"[FactCheck] build failed: {e}")
            return ""

    def _save_to_brain(self, target, reply_text, real_price, analyst_target, fund, news_sent):
        """Save analysis verdict and stock knowledge to the Brain DB."""
        try:
            from learning_engine import get_engine
            _ru = reply_text.upper()
            _bv = "SELL" if ("SELL" in _ru or "REDUCE" in _ru) else "HOLD" if "HOLD" in _ru else "BUY"
            if real_price and real_price > 0:
                _bc = get_engine()._get_conn()
                _bc.execute(
                    "INSERT INTO predictions (ticker, prediction_date, verdict, price_at_prediction, target_price, horizon_days) VALUES (?, date('now'), ?, ?, ?, 30)",
                    (target, _bv, real_price, analyst_target or None)
                )
                _bc.execute(
                    "INSERT INTO stock_knowledge (ticker, company_name, sector, summary, last_price, last_verdict, last_sentiment, analysis_count, first_seen, last_updated, tags) VALUES (?, ?, ?, ?, ?, ?, ?, 1, date('now'), datetime('now'), '[]') ON CONFLICT(ticker) DO UPDATE SET last_price=excluded.last_price, last_verdict=excluded.last_verdict, last_updated=excluded.last_updated, analysis_count=analysis_count+1",
                    (target, fund.get('company_name', target), fund.get('sector', 'Unknown'), f"{_bv} @ ${real_price:.2f}", real_price, _bv, news_sent or 'Neutral')
                )
                _bc.commit()
                _bc.close()
                logger.info(f"[Brain] Saved: {target} {_bv} @ ${real_price:.2f}")
        except Exception as _be:
            logger.warning(f"[Brain] Warning: {_be}")

    @staticmethod
    def _precompute_report_data(
        real_price, forward_pe, analyst_target, fund, summary, dc_data,
        currency_sym="$",
        is_crypto: bool = False,
        is_etf: bool = False,
    ) -> dict:
        """
        Pre-compute ALL numerical values for the report.
        LLM receives finished numbers — never computes anything itself.
        N/A only appears when source data is genuinely absent.
        """
        def _to_float(v) -> float:
            try:
                if v in (None, "", "N/A", "None"):
                    return 0.0
                return float(v)
            except Exception:
                return 0.0

        _ = currency_sym  # reserved for future currency-specific formatting
        p = _to_float(real_price)
        fpe = _to_float(forward_pe or (dc_data or {}).get("forward_pe"))
        eps_ttm = _to_float((fund or {}).get("eps") or (dc_data or {}).get("eps"))
        at = _to_float(analyst_target)
        sma50 = _to_float((summary or {}).get("sma_50"))
        sma200 = _to_float((summary or {}).get("sma_200"))
        beta = _to_float((dc_data or {}).get("beta") or (fund or {}).get("beta") or 1.0)

        out = {"beta": beta}

        # ── Forward EPS ──────────────────────────────────────────────────────
        if fpe > 0 and p > 0:
            out["forward_eps"] = round(p / fpe, 2)
            out["forward_eps_source"] = "price/fwd_pe"
        elif eps_ttm > 0:
            out["forward_eps"] = round(eps_ttm, 2)
            out["forward_eps_source"] = "ttm_eps_approx"
        else:
            out["forward_eps"] = None
            out["forward_eps_source"] = "unavailable"

        # ── Valuation Scenarios (Bear/Base/Bull) ─────────────────────────────
        fwd_eps = out["forward_eps"]
        if fwd_eps and fpe > 0:
            out["val_bear_pe"] = round(fpe * 0.70, 1)
            out["val_base_pe"] = round(fpe, 1)
            out["val_bull_pe"] = round(fpe * 1.40, 1)
            out["val_bear_price"] = round(fpe * 0.70 * fwd_eps, 0)
            out["val_base_price"] = round(fpe * fwd_eps, 0)
            out["val_bull_price"] = round(fpe * 1.40 * fwd_eps, 0)
            out["val_bear_updown"] = round((out["val_bear_price"] - p) / p * 100, 1) if p else None
            out["val_base_updown"] = round((out["val_base_price"] - p) / p * 100, 1) if p else None
            out["val_bull_updown"] = round((out["val_bull_price"] - p) / p * 100, 1) if p else None
        else:
            out["val_bear_price"] = out["val_base_price"] = out["val_bull_price"] = None
            out["val_bear_pe"] = out["val_base_pe"] = out["val_bull_pe"] = None
            out["val_bear_updown"] = out["val_base_updown"] = out["val_bull_updown"] = None

            # price-based fallback for assets with no P/E
            if is_crypto and p > 0:
                _sma200 = _to_float((summary or {}).get("sma_200"))
                _52wk_high = _to_float((dc_data or {}).get("52wk_high") or (dc_data or {}).get("fiftyTwoWeekHigh"))
                out["val_bear_price"] = round(p * 0.60, 0)           # -40% bear cycle
                out["val_base_price"] = round(_sma200, 0) if _sma200 > p * 0.50 else round(p * 1.20, 0)
                out["val_bull_price"] = round(_52wk_high, 0) if _52wk_high > p else round(p * 1.70, 0)
                out["val_bear_updown"] = round((out["val_bear_price"] - p) / p * 100, 1)
                out["val_base_updown"] = round((out["val_base_price"] - p) / p * 100, 1)
                out["val_bull_updown"] = round((out["val_bull_price"] - p) / p * 100, 1)
                out["val_bear_pe"] = "Bear cycle"
                out["val_base_pe"] = "SMA200 reversion"
                out["val_bull_pe"] = "ATH retest"
            elif is_etf and p > 0:
                _sma200 = _to_float((summary or {}).get("sma_200"))
                out["val_bear_price"] = round(p * 0.80, 0)
                out["val_base_price"] = round(_sma200, 0) if _sma200 > p * 0.70 else round(p * 1.08, 0)
                out["val_bull_price"] = round(p * 1.20, 0)
                out["val_bear_updown"] = round((out["val_bear_price"] - p) / p * 100, 1)
                out["val_base_updown"] = round((out["val_base_price"] - p) / p * 100, 1)
                out["val_bull_updown"] = round((out["val_bull_price"] - p) / p * 100, 1)
                out["val_bear_pe"] = "Stress scenario"
                out["val_base_pe"] = "SMA200 reversion"
                out["val_bull_pe"] = "Breakout scenario"

        # ── Upside to Target ─────────────────────────────────────────────────
        out["upside_to_target"] = round((at - p) / p * 100, 1) if at and p else None

        # ── Price vs SMAs ────────────────────────────────────────────────────
        out["pct_vs_sma50"] = round((p - sma50) / sma50 * 100, 1) if sma50 else None
        out["pct_vs_sma200"] = round((p - sma200) / sma200 * 100, 1) if sma200 else None

        # ── Entry Zone / Pullback Distance ───────────────────────────────────
        entry = sma50 if sma50 and p > sma50 * 1.02 else (sma200 if sma200 else None)
        if entry and p:
            out["entry_zone"] = round(entry, 2)
            out["pct_above_entry"] = round((p - entry) / p * 100, 1)
        else:
            out["entry_zone"] = None
            out["pct_above_entry"] = None

        # Technical S/R Ladder (S1/S2/S3 and R1/R2/R3)
        # Priority order: nearest SMA -> nearest Fibonacci -> recent swing -> 52W boundary.
        h52 = _to_float(
            (fund or {}).get("week52_high") or
            (fund or {}).get("year_high") or
            (dc_data or {}).get("fiftyTwoWeekHigh") or
            (dc_data or {}).get("52wk_high") or 0
        )
        l52 = _to_float(
            (fund or {}).get("week52_low") or
            (fund or {}).get("year_low") or
            (dc_data or {}).get("fiftyTwoWeekLow") or
            (dc_data or {}).get("52wk_low") or 0
        )

        _fib_levels = {}
        if h52 and l52 and h52 > l52:
            _rng = h52 - l52
            _fib_levels = {
                "23.6%": round(l52 + _rng * 0.236, 3),
                "38.2%": round(l52 + _rng * 0.382, 3),
                "50.0%": round((h52 + l52) / 2, 3),
                "61.8%": round(l52 + _rng * 0.618, 3),
                "78.6%": round(l52 + _rng * 0.786, 3),
            }

        _nearest_sma_label, _nearest_sma_val = None, None
        _sma_candidates = []
        if sma50:
            _sma_candidates.append(("SMA50", round(sma50, 3)))
        if sma200:
            _sma_candidates.append(("SMA200", round(sma200, 3)))
        if p > 0 and _sma_candidates:
            _nearest_sma_label, _nearest_sma_val = min(_sma_candidates, key=lambda x: abs(x[1] - p))

        _nearest_fib_label, _nearest_fib_val = None, None
        if p > 0 and _fib_levels:
            _fib_candidates = [(k, v) for k, v in _fib_levels.items() if abs(v - p) / max(p, 1.0) > 0.001]
            if _fib_candidates:
                _nearest_fib_label, _nearest_fib_val = min(_fib_candidates, key=lambda x: abs(x[1] - p))

        _swing_high = _to_float(
            (summary or {}).get("swing_high") or
            (summary or {}).get("recent_swing_high") or
            (summary or {}).get("high_20d") or
            (dc_data or {}).get("swing_high") or
            (dc_data or {}).get("recent_swing_high") or 0
        )
        _swing_low = _to_float(
            (summary or {}).get("swing_low") or
            (summary or {}).get("recent_swing_low") or
            (summary or {}).get("low_20d") or
            (dc_data or {}).get("swing_low") or
            (dc_data or {}).get("recent_swing_low") or 0
        )

        _levels = []

        def _push_level(price_v: float, level_type: str, basis: str, priority: int):
            if not (p > 0 and price_v):
                return
            _dist = abs(price_v - p) / p
            if _dist <= 0.0005:
                return
            _levels.append({
                "price": round(price_v, 3),
                "type": level_type,
                "basis": basis,
                "priority": priority,
                "distance": _dist,
            })

        if _nearest_sma_val:
            _push_level(_nearest_sma_val, "Resistance" if _nearest_sma_val > p else "Support", _nearest_sma_label, 1)

        if _nearest_fib_val:
            _push_level(_nearest_fib_val, "Resistance" if _nearest_fib_val > p else "Support", f"Fib {_nearest_fib_label}", 2)

        if _swing_high:
            _push_level(_swing_high, "Resistance" if _swing_high > p else "Support", "Recent Swing High", 3)
        if _swing_low:
            _push_level(_swing_low, "Resistance" if _swing_low > p else "Support", "Recent Swing Low", 3)

        if h52:
            _push_level(h52, "Resistance" if h52 > p else "Support", "52W High", 4)
        if l52:
            _push_level(l52, "Support" if l52 < p else "Resistance", "52W Low", 4)

        _dedup = {}
        for _lv in _levels:
            _k = (_lv["type"], round(_lv["price"], 3))
            if _k not in _dedup:
                _dedup[_k] = _lv
            else:
                _cur = _dedup[_k]
                if (_lv["priority"], _lv["distance"]) < (_cur["priority"], _cur["distance"]):
                    _dedup[_k] = _lv
        _levels = list(_dedup.values())

        _above = sorted([x for x in _levels if x["price"] > p], key=lambda x: (x["distance"], x["priority"], x["price"]))[:3]
        _below = sorted([x for x in _levels if x["price"] < p], key=lambda x: (x["distance"], x["priority"], -x["price"]))[:3]

        for i, _lv in enumerate(_above, 1):
            _lv["level"] = f"R{i}"
        for i, _lv in enumerate(_below, 1):
            _lv["level"] = f"S{i}"

        out["sr_levels_above"] = _above
        out["sr_levels_below"] = _below

        out["fib_resistance"] = round(_above[0]["price"], 3) if _above else None
        out["fib_resistance_pct"] = round((out["fib_resistance"] - p) / p * 100, 1) if out["fib_resistance"] else None
        out["fib_resistance_label"] = _above[0]["basis"] if _above else None
        out["fib_support"] = round(_below[0]["price"], 3) if _below else None
        out["fib_support_pct"] = round((out["fib_support"] - p) / p * 100, 1) if out["fib_support"] else None
        out["fib_key_support"] = round(l52, 3) if l52 else None
        out["fib_52w_high"] = round(h52, 3) if h52 else None
        out["fib_52w_low"] = round(l52, 3) if l52 else None
        out["fib_above_52w_high"] = bool(h52 and p > h52)

        _sr_rows = [
            "| Level | Price | Type | Basis |",
            "|-------|-------|------|-------|",
        ]
        for _lv in reversed(_above):
            _sr_rows.append(f"| {_lv['level']} | {currency_sym}{_lv['price']:,.2f} | {_lv['type']} | {_lv['basis']} |")
        _sr_rows.append(f"| Spot | {currency_sym}{p:,.2f} | Current | Live |" if p else "| Spot | N/A | Current | Live |")
        for _lv in _below:
            _sr_rows.append(f"| {_lv['level']} | {currency_sym}{_lv['price']:,.2f} | {_lv['type']} | {_lv['basis']} |")
        out["sr_levels_table"] = "\n".join(_sr_rows)

        return out
    # ══════════════════════════════════════════════════════════════════════
    # Main analytics handler (refactored)
    # ══════════════════════════════════════════════════════════════════════

    def _handle_analytics(
        self,
        sid: str,
        mem: dict,
        msg: str,
        _no_multi: bool = False,
        mode: str = "full",
    ) -> dict:
        import core.analytics as ca
        from core.data import get_prices
        import os, requests
        from datetime import datetime

        # === DETECT LANGUAGE (for full Arabic report) ===
        _arabic_chars = sum(1 for c in msg if '\u0600' <= c <= '\u06FF')
        _is_arabic_request = _arabic_chars >= 2  # 2+ Arabic characters = Arabic request
        _analysis_mode = (mode or "full").strip().lower()
        if _analysis_mode not in {"quick", "full", "cio"}:
            _analysis_mode = "full"

        # === EXTRACT TICKERS FIRST ===
        tickers = IntentClassifier.extract_tickers(msg)
        # ── Report Cache check ───────────────────────────────────────────────
        import time as _tc
        _cache_key = msg.strip().lower()[:80]
        _cached = _REPORT_CACHE.get(_cache_key)
        if _cached:
            _age = _tc.time() - _cached[0]
            if _age < _REPORT_CACHE_TTL:
                logger.info(f"[ReportCache] HIT ({_age:.0f}s old)")
                return _cached[1]
        logger.info(f"[FA] tickers extracted: {tickers}")
        if not tickers:
            tickers = mem.get("tickers", [])
        # Fallback: long commodity names (>6 chars) are missed by TICKER_RE — catch them here
        if not tickers:
            _kw_fallback = {
                "PLATINUM": "PLATINUM", "PALLADIUM": "PALLADIUM",
                "NATURAL GAS": "NG=F", "BRENT OIL": "BZ=F", "BRENT": "BZ=F",
                "ETHEREUM": "ETH-USD", "BITCOIN": "BTC-USD",
            }
            _msg_up0 = msg.upper()
            for _kw, _sym in _kw_fallback.items():
                if _kw in _msg_up0:
                    tickers = [_sym]
                    break
        if not tickers:
            return {"type": "chat.reply", "reply": "Please specify a ticker to analyze (e.g. 'analyze NVDA')."}

        # === DEDUP: collapse alias-equivalent tickers & remove spurious local matches ===
        _DEDUP_MAP = {
            # Futures roots (extract_tickers strips "=F" suffix from e.g. "HG=F" → "HG")
            "GC": "GC=F", "SI": "SI=F", "CL": "CL=F", "NG": "NG=F",
            "PL": "PL=F", "PA": "PA=F", "HG": "HG=F", "BZ": "BZ=F",
            # Commodity name aliases
            "GOLD": "GC=F", "XAUUSD": "GC=F", "XAU": "GC=F",
            "SILVER": "SI=F", "XAGUSD": "SI=F", "XAG": "SI=F",
            "COPPER": "HG=F", "XCUUSD": "HG=F",
            "OIL": "CL=F", "WTIUSD": "CL=F", "CRUDE": "CL=F", "XTIUSD": "CL=F",
            "PLATINUM": "PL=F", "XPTUSD": "PL=F",
            "PALLADIUM": "PA=F", "XPDUSD": "PA=F",
            # Crypto
            "BTC": "BTC-USD", "BITCOIN": "BTC-USD", "BTCUSD": "BTC-USD",
            "ETH": "ETH-USD", "ETHEREUM": "ETH-USD", "ETHUSD": "ETH-USD",
            "SOL": "SOL-USD", "XRP": "XRP-USD", "BNB": "BNB-USD",
        }
        _seen_res = set()
        _deduped = []
        for _tk in tickers:
            _r = _DEDUP_MAP.get(_tk.upper(), _tk.upper())
            if _r not in _seen_res:
                _seen_res.add(_r)
                _deduped.append(_tk)
        tickers = _deduped
        # Remove spurious local-market tickers injected by the resolver when the user
        # didn't explicitly mention them.  e.g. "analyze COPPER" → resolver adds
        # ETISALAT.AE; "analyze AAPL and MSFT" → resolver adds DEWA.DU.
        # We keep a local ticker only if its root (before the dot) appears literally
        # in the message.
        _msg_up = msg.upper()
        _local_sfx = (".AE", ".DU", ".SR", ".CA", ".KW", ".QA")
        def _explicitly_in_msg(tk: str) -> bool:
            root = tk.upper().split(".")[0]   # "ETISALAT.AE" → "ETISALAT"
            full = tk.upper()
            return root in _msg_up or full in _msg_up
        tickers_clean = [_tk for _tk in tickers
                         if not any(_tk.upper().endswith(s) for s in _local_sfx)
                         or _explicitly_in_msg(_tk)]
        tickers = tickers_clean if tickers_clean else tickers
        logger.info(f"[FA] tickers after dedup: {tickers}")

        # === WEB RESEARCH ===
        research_context = ""
        try:
            if hasattr(self, '_web_search') and self._web_search and tickers:
                _ticker = tickers[0].upper() if tickers else ""
                r1 = self._web_search(f"{_ticker} stock analysis outlook 2026")
                r2 = self._web_search(f"{_ticker} earnings forecast analyst target price")
                snippets = []
                for r in [r1, r2]:
                    if isinstance(r, dict):
                        for item in r.get("organic", [])[:3]:
                            s = item.get("snippet", "")
                            t = item.get("title", "")
                            if s:
                                snippets.append(f"- {t}: {s}")
                if snippets:
                    research_context = "\nRECENT WEB RESEARCH:\n" + "\n".join(snippets[:6])
                    logger.debug(f"[EisaX Research] Found {len(snippets)} sources")
        except Exception as e:
            logger.error(f"[EisaX Research] failed: {e}")

        # ── Multi-ticker handler ─────────────────────────────────────────────
        if len(tickers) > 1 and not _no_multi:
            logger.info(f"[EisaX] Multi-ticker: {tickers}")
            reports = []
            _skipped = []
            for _t in tickers[:4]:
                if _t in {"VS", "AND", "OR", "THE", "FOR"}:
                    continue
                try:
                    _r = self._handle_analytics(
                        "default",
                        mem,
                        f"analyze {_t}",
                        _no_multi=True,
                        mode=_analysis_mode,
                    )
                    if _r.get("type") == "error":
                        logger.warning(f"[EisaX] {_t} skipped in comparison — {_r.get('reply','')[:80]}")
                        _skipped.append(_t)
                    elif _r.get("reply"):
                        reports.append(_r["reply"])
                except Exception as _e:
                    logger.error(f"[EisaX] {_t} failed in comparison: {_e}")
                    _skipped.append(_t)
            if not reports:
                _all_bad = ", ".join(_skipped) if _skipped else ", ".join(tickers[:4])
                return {
                    "type": "error",
                    "reply": (
                        f"⚠️ Could not retrieve market data for: **{_all_bad}**.\n"
                        f"Verify the ticker symbols and try again."
                    ),
                }
            if reports:
                try:
                    import requests as _req
                    from dotenv import load_dotenv, find_dotenv as _find_dotenv
                    load_dotenv(_find_dotenv(usecwd=True) or "/home/ubuntu/investwise/.env")
                    _ds_key = os.getenv("DEEPSEEK_API_KEY","")
                    _names = [t for t in tickers[:4] if t not in {"VS","AND","OR"}]
                    _r2 = _req.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {_ds_key}", "Content-Type": "application/json"},
                        json={"model": "deepseek-chat",
                              "messages": [{"role": "user", "content": f"Compare {' vs '.join(_names)} in a markdown table with: Verdict, Score, Upside, Risk, Best For. Be concise."}],
                              "max_tokens": 400, "temperature": 0},
                        timeout=30
                    )
                    _summary = _r2.json()["choices"][0]["message"]["content"].strip()
                except Exception as _e:
                    _summary = ""
                _combined = "# EisaX Comparison: " + " | ".join(_names) + "\n\n"
                if _skipped:
                    _combined += (
                        f"> ⚠️ **Note:** Insufficient market data for "
                        f"**{', '.join(_skipped)}** — excluded from comparison.\n\n"
                    )
                if _summary:
                    _combined += "## Head-to-Head Summary\n" + _summary + "\n\n---\n\n"

                _combined += "\n\n---\n\n".join(reports)
                return {"type": "chat.reply", "reply": _combined, "data": {"agent": "finance"}}

        target = tickers[0].upper()

        # ── Ticker Aliases ──────────────────────────────────────────────────
        _TICKER_ALIASES = {
            # Spot gold/silver → ETF equivalents (yfinance doesn't support XAUUSD)
            "XAUUSD": "GC=F", "XAU/USD": "GC=F", "GOLD": "GC=F", "XAUUSD=X": "GC=F",
            "XAGUSD": "SI=F", "XAG/USD": "SI=F", "SILVER": "SI=F",
            "XPTUSD": "PL=F", "XPT/USD": "PL=F", "PLATINUM": "PL=F",
            "XPDUSD": "PA=F", "XPD/USD": "PA=F", "PALLADIUM": "PA=F",
            "XCUUSD": "HG=F", "XCU/USD": "HG=F", "COPPER": "HG=F",
            "XTIUSD": "CL=F", "OIL": "CL=F", "WTIUSD": "CL=F", "CRUDE": "CL=F",
            # Crypto — bare symbols need -USD suffix for yfinance
            "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
            "XRP": "XRP-USD", "BNB": "BNB-USD", "DOGE": "DOGE-USD",
            "ADA": "ADA-USD", "AVAX": "AVAX-USD", "DOT": "DOT-USD",
            "LINK": "LINK-USD", "MATIC": "MATIC-USD", "ATOM": "ATOM-USD",
            "LTC": "LTC-USD", "UNI": "UNI-USD", "SHIB": "SHIB-USD",
            "TON": "TON-USD", "SUI": "SUI-USD", "TRX": "TRX-USD",
            # UAE company aliases
            "ETISALAT": "EAND.AE", "ETISALAT.AE": "EAND.AE",
            "ETISALAT.DU": "EAND.DU",
            "ADNOC": "ADNOCGAS.AE", "ARAMCO": "2222.SR",
        }
        if target in _TICKER_ALIASES:
            _original_target = target
            target = _TICKER_ALIASES[target]
            logger.info(f"[Alias] {_original_target} → {target}")

        # ── Local Market Enrichment ──
        _local_data_injection = ""
        try:
            from core.local_market_enricher import build_local_prompt_injection, is_local_ticker
            if is_local_ticker(target):
                _local_data_injection = build_local_prompt_injection(target)
        except Exception as _le:
            logger.debug(f"[EisaX] Local enricher: {_le}")
        # ── Brain Context ────────────────────────────────────────────
        _brain_ctx = self._get_brain_context(target)

        # ── 1-3. PARALLEL DATA FETCH ──────────────────────────────────────────
        # All 7 network calls run concurrently → reduces fetch time from ~15s to ~5s
        import re as _re
        from concurrent.futures import ThreadPoolExecutor as _TpEx
        from core.market_data import get_full_stock_profile as _get_profile
        from core.fundamental_engine import get_fundamentals as _get_fund
        from core.rapid_data import get_fear_greed as _get_fg, get_events_calendar as _get_events

        def _safe(v):
            try: return str(round(float(v), 2))
            except: return str(v) if v else "N/A"

        # Submit all network calls simultaneously (news engine runs in parallel too)
        from core.news_engine_client import get_ticker_news as _get_engine_news
        with _TpEx(max_workers=8) as _exe:
            _f_profile    = _exe.submit(_get_profile, target)
            _f_fund       = _exe.submit(_get_fund, target)
            _f_dc         = _exe.submit(deepcrawl_stock, target)
            _f_yf         = _exe.submit(_yf_with_retry, target)
            _f_prices     = _exe.submit(get_prices, [target], "2023-01-01", None)
            _f_fg         = _exe.submit(_get_fg)
            _f_events     = _exe.submit(_get_events, target)
            _f_eng_news   = _exe.submit(_get_engine_news, target)   # ← EisaX news engine
            # ── Grok disabled — was adding 12s to every report ──────────────
            _f_grok       = None

            # ── collect: Live Price + Macro ──────────────────────────────────
            real_price = None; change_pct = 0.0
            t10y = fed = unemp = inflation = gdp = "N/A"
            news_sent = "N/A"; news_score = 0; sentiment = {}

            # ── Market Cache lookup (UAE/KSA/Egypt/Qatar tickers) ────────────
            # Try before yfinance — cache has live TradingView data every 15min
            _cache_row = None
            try:
                _target_up = target.upper()
                # Determine which market cache to search
                _cache_markets = []
                if _target_up.endswith(".AE") or _target_up.endswith(".DU"):
                    _cache_markets = ["uae"]
                elif _target_up.endswith(".SR"):
                    _cache_markets = ["ksa"]
                elif _target_up.endswith(".CA"):
                    _cache_markets = ["egypt"]
                elif _target_up.endswith(".KW"):
                    _cache_markets = ["kuwait"]
                elif _target_up.endswith(".QA"):
                    _cache_markets = ["qatar"]
                elif _target_up.endswith(".BH"):
                    _cache_markets = ["bahrain"]
                elif _target_up.endswith(".MA"):
                    _cache_markets = ["morocco"]
                elif _target_up.endswith(".TN"):
                    _cache_markets = ["tunisia"]
                else:
                    # Try all regional caches for bare tickers like "ADNOCGAS"
                    _cache_markets = ["uae", "ksa", "egypt", "kuwait", "qatar"]

                import os as _os, json as _json
                _cache_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "market_cache")
                _idx_path = _os.path.join(_cache_dir, "index.json")
                if _os.path.exists(_idx_path):
                    import pandas as _pd
                    with open(_idx_path) as _f:
                        _idx = _json.load(_f)
                    for _mkt in _cache_markets:
                        if _mkt not in _idx:
                            continue
                        _entries = _idx[_mkt]
                        if isinstance(_entries, list):
                            _entries = sorted(_entries, key=lambda x: x.get("timestamp",""), reverse=True)
                            _latest = _entries[0] if _entries else None
                        else:
                            _latest = _entries
                        if not _latest:
                            continue
                        _fpath = _os.path.join(_cache_dir, _latest["filename"])
                        _df = _pd.read_parquet(_fpath)
                        # Match by yfinance suffix format (ADNOCGAS.AE → ADX:ADNOCGAS)
                        # or by bare symbol
                        _bare = _target_up.split(".")[0]
                        _match = _df[
                            _df["ticker"].str.upper().str.endswith(":" + _bare) |
                            (_df["ticker"].str.upper() == _target_up)
                        ]
                        if not _match.empty:
                            _cache_row = _match.iloc[0].to_dict()
                            logger.info("[MarketCache] Found %s in %s cache: price=%.2f", target, _mkt, float(_cache_row.get("close",0) or 0))
                            break
            except Exception as _ce:
                logger.debug("[MarketCache] Lookup failed for %s: %s", target, _ce)

            try:
                profile    = _f_profile.result(timeout=25)
                quote      = profile.get("quote", {})
                sentiment  = profile.get("sentiment", {})
                macro      = profile.get("macro", {})
                real_price = quote.get("price")
                change_pct = quote.get("change_pct", 0) or 0
                t10y      = _safe(macro.get("treasury_10y", {}).get("value", "N/A"))
                fed       = _safe(macro.get("fed_funds",    {}).get("value", "N/A"))
                unemp     = _safe(macro.get("unemployment", {}).get("value", "N/A"))
                inflation = _safe(macro.get("inflation",    {}).get("value", "N/A"))
                gdp       = _safe(macro.get("gdp_growth",   {}).get("value", "N/A"))
                news_sent  = sentiment.get("sentiment", "N/A")
                news_score = sentiment.get("score", 0)
            except Exception as e:
                logger.warning(f"[Analytics] market_data failed (non-fatal): {e}")
                profile = {}
                quote = {}
                sentiment = {}
                macro = {}

            # ── Inject Market Cache data if yfinance returned no price ────────
            if _cache_row and not real_price:
                try:
                    real_price = float(_cache_row.get("close") or 0) or None
                    change_pct = float(_cache_row.get("change") or 0)
                    logger.info("[MarketCache] Injected price=%.2f change=%.2f%% for %s",
                                real_price or 0, change_pct, target)
                except Exception as _cij:
                    logger.debug("[MarketCache] Price inject failed: %s", _cij)

            # ── collect: Fundamentals (resilient waterfall) ──────────────────
            fund = {}
            _fund_source = "none"
            try:
                fund = _f_fund.result(timeout=15) or {}
                if fund:
                    _fund_source = "yfinance/fundamental_engine"
            except Exception as e:
                logger.error(f"[Analytics] Fundamentals failed: {e}")

            # Waterfall: if primary failed or sparse, try DB cache then RapidAPI
            _fund_useful = sum(1 for k in ("pe_ratio","beta","market_cap","eps","revenue") if fund.get(k))
            if _fund_useful < 2:
                # Try DB cache (uae_fundamentals — covers UAE/Saudi/Egypt)
                try:
                    import sqlite3 as _sq
                    from core.config import CORE_DB as _cfg_core_db
                    _db = _sq.connect(str(_cfg_core_db))
                    _row = _db.execute(
                        "SELECT pe_ratio,beta,market_cap,eps,div_yield,revenue,net_margin,forward_pe,sector,company_name,"
                        "week_52_high,week_52_low,roe,gross_margin,revenue_growth,earnings_growth,net_income "
                        "FROM uae_fundamentals WHERE ticker=? LIMIT 1", (target.upper(),)
                    ).fetchone()
                    _db.close()
                    if _row and any(v is not None for v in _row[:8]):
                        _cols = ["pe_ratio","beta","market_cap","eps","div_yield","revenue","net_margin","forward_pe",
                                 "sector","company_name","week52_high","week52_low","roe","gross_margin",
                                 "revenue_growth","earnings_growth","net_income"]
                        _db_fund = {k: v for k, v in zip(_cols, _row) if v is not None}
                        fund = {**_db_fund, **fund}   # DB fills gaps, live data takes priority
                        _fund_source = "db_cache+yfinance"
                        logger.info(f"[Fund/DB] {target}: filled {len(_db_fund)} fields from DB cache")
                except Exception as _dbe:
                    logger.warning(f"[Fund/DB] {target}: {_dbe}")

            if _fund_useful < 2:
                # Try RapidAPI (Investing.com) as last resort
                try:
                    from core.rapidapi_client import get_fundamentals as _rapi_fund
                    _rapi_data = _rapi_fund(target) or {}
                    if _rapi_data:
                        fund = {**_rapi_data, **fund}
                        _fund_source = "rapidapi"
                        logger.info(f"[Fund/RapidAPI] {target}: filled {len(_rapi_data)} fields")
                except Exception as _rape:
                    logger.debug(f"[Fund/RapidAPI] {target}: {_rape}")

            # ── Inject Market Cache fundamentals (fills gaps for UAE/KSA/Egypt) ─
            if _cache_row:
                try:
                    _cache_fund = {}
                    import math as _math_fc
                    def _valid_cache_num(v):
                        """Return True if v is a non-NaN, non-inf, non-zero number."""
                        try:
                            f = float(v)
                            return not (_math_fc.isnan(f) or _math_fc.isinf(f)) and f != 0
                        except (TypeError, ValueError):
                            return False
                    _pe_raw = _cache_row.get("price_earnings_ttm")
                    if _valid_cache_num(_pe_raw) and not fund.get("pe_ratio"):
                        _cache_fund["pe_ratio"] = float(_pe_raw)
                    _eps_raw = _cache_row.get("earnings_per_share_diluted_ttm")
                    if _valid_cache_num(_eps_raw) and not fund.get("eps"):
                        _cache_fund["eps"] = float(_eps_raw)
                    if _cache_row.get("market_cap_basic") and not fund.get("market_cap"):
                        _cache_fund["market_cap"] = float(_cache_row["market_cap_basic"])
                    if _cache_row.get("sector") and not fund.get("sector"):
                        _cache_fund["sector"] = str(_cache_row["sector"])
                    if _cache_row.get("dividend_yield_recent") is not None and not fund.get("div_yield"):
                        _cache_fund["div_yield"] = float(_cache_row["dividend_yield_recent"] or 0)
                    # Inject 52W range from TradingView cache if not already populated
                    _cache_52h = float(_cache_row.get("high_52_week") or _cache_row.get("week52_high") or 0)
                    _cache_52l = float(_cache_row.get("low_52_week") or _cache_row.get("week52_low") or 0)
                    if _cache_52h and not fund.get("week52_high"):
                        _cache_fund["week52_high"] = _cache_52h
                    if _cache_52l and not fund.get("week52_low"):
                        _cache_fund["week52_low"] = _cache_52l
                    # Inject TradingView technicals directly
                    _cache_fund["rsi"]        = round(float(_cache_row.get("RSI") or 0), 2)
                    _cache_fund["macd"]       = round(float(_cache_row.get("MACD.macd") or 0), 4)
                    _cache_fund["macd_signal"]= round(float(_cache_row.get("MACD.signal") or 0), 4)
                    _cache_fund["sma50"]      = round(float(_cache_row.get("SMA50") or 0), 4)
                    _cache_fund["sma200"]     = round(float(_cache_row.get("SMA200") or 0), 4)
                    _cache_fund["atr"]        = round(float(_cache_row.get("ATR") or 0), 4)
                    _cache_fund["stoch_k"]    = round(float(_cache_row.get("Stoch.K") or 0), 2)
                    _cache_fund["volume"]     = int(_cache_row.get("volume") or 0)
                    _cache_fund["data_source"] = "TradingView Live Cache"
                    fund = {**_cache_fund, **fund}   # cache fills gaps, live data takes priority
                    if _fund_source == "none":
                        _fund_source = "tradingview_cache"
                    logger.info("[MarketCache] Injected %d fundamental fields for %s (P/E=%.1f, RSI=%.1f)",
                                len(_cache_fund), target,
                                (float(_cache_row.get("price_earnings_ttm") or 0) if _valid_cache_num(_cache_row.get("price_earnings_ttm")) else 0.0),
                                float(_cache_row.get("RSI") or 0))
                except Exception as _cfe:
                    logger.debug("[MarketCache] Fund inject failed: %s", _cfe)

            logger.info(f"[Fund] {target}: source={_fund_source}, fields={_fund_useful}")

            # Data coverage level drives compact low-data report behavior.
            _data_coverage_count = count_valid_fundamental_fields(fund)
            _data_coverage_level = classify_data_coverage_level(_data_coverage_count)
            _low_data_compact_mode = _data_coverage_level in ("technical_only", "low")

            # ── collect: Analyst Consensus (DeepCrawl primary, yfinance fill) ─
            # Pre-seed from fund dict (get_fundamentals runs its own sequential yfinance)
            analyst_target = fund.get('analyst_target') or None
            analyst_consensus = fund.get('analyst_consensus') or None
            analyst_count = fund.get('analyst_count') or None
            forward_pe = None
            dividend_yield = None; news_links = []; earnings_date = None
            dc_data = {}
            # ── EisaX News Engine — collected in parallel, resolved first ────────
            _engine_news_data = {}
            try:
                _engine_news_data = _f_eng_news.result(timeout=4) or {}
            except Exception as _ene:
                logger.debug(f"[NewsEngine] result failed for {target}: {_ene}")

            # ── Grok disabled — _x_data stays empty ──────────────────────────
            _x_data: dict = {}
            try:
                dc_data = _f_dc.result(timeout=15) or {}
                if dc_data.get("price_target"):
                    pt_m = _re.search(r"([\d.]+)", dc_data["price_target"])
                    if pt_m:
                        analyst_target = float(pt_m.group(1))
                analyst_consensus = dc_data.get("analyst_rating", "")
                forward_pe = float(dc_data.get("forward_pe", 0)) or None
                earnings_date = dc_data.get("earnings_date", "")
                # DeepCrawl "dividend" = annual dollar amount ($1.04), NOT yield %
                # Convert to yield: $1.04 / $254.23 = 0.0041 (0.41%)
                _dc_div_dollar = float(dc_data.get("dividend", 0) or 0)
                _dc_price = float(dc_data.get("price", 0) or 0) or (real_price or 0)
                if _dc_div_dollar > 0 and _dc_price > 0:
                    dividend_yield = _dc_div_dollar / _dc_price  # dollar → decimal yield
                    if dividend_yield > 0.20:  # > 20% yield = data error
                        dividend_yield = None
                else:
                    dividend_yield = None
                logger.info(f"[Analytics] DeepCrawl OK: price={dc_data.get('price')}, target={analyst_target}")
            except Exception as e:
                logger.error(f"[Analytics] DeepCrawl failed: {e}")
            try:
                yt, info = _f_yf.result(timeout=15)
                if not analyst_target:
                    analyst_target = info.get("targetMeanPrice") or info.get("targetMedianPrice")
                if not analyst_consensus:
                    analyst_consensus = info.get("recommendationKey", "").replace("_", " ").title()
                analyst_count = info.get("numberOfAnalystOpinions")
                if not forward_pe:
                    _fpe_raw = info.get("forwardPE")
                    forward_pe = float(_fpe_raw) if (_fpe_raw and float(_fpe_raw) > 0) else None
                if not dividend_yield:
                    # trailingAnnualDividendYield is decimal (0.006 = 0.6%)
                    _trail_dy = float(info.get("trailingAnnualDividendYield") or 0)
                    # Sanity cap: if > 0.50 (50%) it's data garbage — discard
                    # Consistent with yfinance occasionally returning % instead of decimal
                    if _trail_dy > 0.50:
                        _trail_dy = _trail_dy / 100  # treat as already-percentage
                    if _trail_dy > 0.50:
                        _trail_dy = 0  # still absurd → discard entirely
                    dividend_yield = _trail_dy if _trail_dy > 0 else None
                # ── Volume + 52W range (for Technical Outlook) ───────────────
                _vol_today = info.get("volume") or info.get("regularMarketVolume") or 0
                _vol_avg   = info.get("averageVolume") or 0
                _vol_10d   = info.get("averageVolume10days") or 0
                _52w_high  = info.get("fiftyTwoWeekHigh") or 0
                _52w_low   = info.get("fiftyTwoWeekLow") or 0
                # Store for later use in data_block
                if _vol_today: fund['volume_today'] = int(_vol_today)
                if _vol_avg:   fund['volume_avg90d'] = int(_vol_avg)
                if _vol_10d:   fund['volume_avg10d'] = int(_vol_10d)
                if _52w_high:  fund['week52_high'] = float(_52w_high)
                if _52w_low:   fund['week52_low']  = float(_52w_low)

                raw_news = yt.news or []
                for n in raw_news[:4]:
                    try:
                        c = n.get("content", {})
                        title = c.get("title", "") or n.get("title", "")
                        link = (c.get("canonicalUrl", {}).get("url", "") or
                                c.get("clickThroughUrl", {}).get("url", "") or
                                n.get("link", "") or n.get("url", ""))
                        if title and link:
                            news_links.append({"title": title[:120], "url": link})
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"[Analytics] yfinance analyst failed: {e}")

            # ── Fund dict fallback: covers yfinance rate-limit failures ──────────
            # get_fundamentals() runs sequentially inside its own thread — higher success rate
            if not analyst_target:
                analyst_target = fund.get('analyst_target')
                if analyst_target:
                    logger.info(f"[Analytics] analyst_target from fund dict: {analyst_target}")
            if not analyst_consensus:
                analyst_consensus = fund.get('analyst_consensus', '')
            if not analyst_count:
                analyst_count = fund.get('analyst_count')

            _data_coverage_count = count_valid_fundamental_fields(
                fund,
                dc_data,
                analyst_target=analyst_target,
                forward_pe=forward_pe,
            )
            _data_coverage_level = classify_data_coverage_level(_data_coverage_count)
            _low_data_compact_mode = _data_coverage_level in ("technical_only", "low")

            # ── EisaX News Engine: inject as PRIMARY source (highest quality) ────
            # The engine has curated GCC/MENA + global financial news updated every 15min.
            # We add engine news BEFORE FMP/Serper to give them priority in the display.
            if _engine_news_data:
                from core.news_engine_client import format_news_links as _fmt_eng_links
                _eng_links = _fmt_eng_links(_engine_news_data)
                _seen_eng  = {n["url"] for n in news_links}
                for _el in _eng_links:
                    if _el["url"] not in _seen_eng:
                        news_links.append(_el)
                        _seen_eng.add(_el["url"])
                logger.info(f"[NewsEngine] {target}: injected {len(_eng_links)} articles into news_links")

            if not news_links:
                try:
                    fmp_news = get_live_news(target, limit=4)
                    for n in fmp_news:
                        if n.get("headline") and n.get("url"):
                            news_links.append({"title": n["headline"][:120], "url": n["url"]})
                except Exception as e:
                    logger.error(f"[Analytics] FMP news failed: {e}")

            # ── Regional energy stocks: supplement with geo/sector context news ──
            # UAE/Saudi energy tickers rarely appear in yfinance news — fetch
            # regional energy/geopolitical context separately via NewsAPI.
            _t_upper_news = target.upper()
            _is_regional_energy = (
                _t_upper_news.endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
                and (_is_energy if '_is_energy' in dir() else
                     any(k in _t_upper_news for k in ("ADNOC", "ARAMCO", "2222", "TAQA", "DANA", "GAS", "OIL", "ENERG")))
            )
            if _is_regional_energy and len(news_links) < 3:
                try:
                    from core.realtime_data import get_live_news as _gln
                    _sector_ctx = fund.get('sector', 'Energy').lower()
                    _region_q = (
                        "Gulf oil energy OPEC Middle East geopolitical risk 2026"
                        if _t_upper_news.endswith((".AE", ".DU"))
                        else "Saudi Aramco oil energy OPEC 2026"
                        if _t_upper_news.endswith(".SR")
                        else "oil energy OPEC Middle East 2026"
                    )
                    _geo_news = _gln(target, company_name=_region_q, limit=5)
                    for n in _geo_news:
                        h = n.get("headline", "")
                        u = n.get("url", "")
                        if h and u and not any(x["title"] == h for x in news_links):
                            news_links.append({"title": h[:120], "url": u})
                    logger.info(f"[RegionalNews] {target}: supplemented with {len(_geo_news)} regional items")
                except Exception as _rne:
                    logger.warning(f"[RegionalNews] supplement failed: {_rne}")

            # ── Local non-energy: fetch company + market news via NewsAPI ──────
            _is_local_ticker = target.upper().endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
            if _is_local_ticker and len(news_links) < 2:
                try:
                    from core.realtime_data import get_live_news as _gln2
                    # Build smart query: company name + market context
                    _co_name = fund.get('company_name') or target.split('.')[0]
                    # Specific company query first — no generic market terms
                    _mkt_ctx = (
                        "UAE" if target.upper().endswith((".AE", ".DU"))
                        else "Saudi Arabia" if target.upper().endswith(".SR")
                        else "Egypt" if target.upper().endswith(".CA")
                        else "Kuwait" if target.upper().endswith(".KW")
                        else "Qatar"
                    )
                    # Try specific company name first
                    _ticker_base = target.split('.')[0]
                    _local_news = _gln2(target, company_name=f"{_co_name}", limit=5)
                    # If few results, try ticker base
                    if len(_local_news) < 2:
                        _local_news = _gln2(target, company_name=f"{_ticker_base} {_mkt_ctx}", limit=5)
                    for n in _local_news:
                        h = n.get("headline", "")
                        u = n.get("url", "")
                        if h and u and not any(x["title"] == h for x in news_links):
                            news_links.append({"title": h[:120], "url": u})
                    # If still empty, fetch sector+market news
                    if len(news_links) < 2:
                        _sector = fund.get('sector','') or 'investment'
                        _mkt_news = _gln2(target, company_name=f"{_sector} {_mkt_ctx} market 2026", limit=4)
                        for n in _mkt_news:
                            h = n.get("headline", "")
                            u = n.get("url", "")
                            if h and u and not any(x["title"] == h for x in news_links):
                                news_links.append({"title": h[:120], "url": u})
                    logger.info(f"[LocalNews] {target}: {len(news_links)} news items fetched")
                except Exception as _lne:
                    logger.warning(f"[LocalNews] {target} failed: {_lne}")

            # ── Last-resort: Serper web search for news ────────────────────
            if len(news_links) < 2:
                try:
                    _serper_key = os.getenv("SERPER_API_KEY", "")
                    if _serper_key:
                        import requests as _req_serper
                        _ticker_base_serper = target.split('.')[0]
                        _co_name_serper = (fund.get('company_name') or dc_data.get('company_name')
                                           or _ticker_base_serper)
                        _is_gulf_ticker = target.upper().endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
                        # For commodity futures tickers, use the commodity name as query term
                        _commodity_name_map = {
                            "GC": "gold price", "SI": "silver price", "CL": "crude oil WTI",
                            "NG": "natural gas price", "PL": "platinum price", "PA": "palladium price",
                            "HG": "copper price", "BZ": "brent oil price",
                            "GC=F": "gold price", "SI=F": "silver price", "CL=F": "crude oil WTI",
                            "PL=F": "platinum price", "PA=F": "palladium price",
                            "HG=F": "copper price", "BZ=F": "brent oil price",
                            "GLD": "gold ETF price", "IAU": "gold ETF price", "SGOL": "gold ETF",
                            "GLDM": "gold ETF", "SLV": "silver ETF price", "SIVR": "silver ETF",
                            "USO": "crude oil ETF", "BNO": "brent oil ETF",
                            "PPLT": "platinum ETF", "PALL": "palladium ETF", "CPER": "copper ETF",
                        }
                        _serper_commodity = _commodity_name_map.get(
                            _ticker_base_serper.upper(), _commodity_name_map.get(target.upper(), "")
                        )
                        if _is_gulf_ticker:
                            _sq = (f'"{_co_name_serper}" OR "{_ticker_base_serper}" أخبار stock news '
                                   f'site:zawya.com OR site:gulfnews.com OR site:arabianbusiness.com')
                        elif _serper_commodity:
                            _sq = f'{_serper_commodity} market news 2026'
                        else:
                            _sq = f'"{_co_name_serper}" stock news {(fund.get("sector","") or "")}'
                        _sr = _req_serper.post(
                            "https://google.serper.dev/news",
                            headers={"X-API-KEY": _serper_key, "Content-Type": "application/json"},
                            json={"q": _sq, "num": 6},
                            timeout=8
                        )
                        if _sr.status_code == 200:
                            for _sn in _sr.json().get("news", []):
                                _sh = _sn.get("title", "")
                                _su = _sn.get("link", "")
                                if _sh and _su and not any(x["title"] == _sh for x in news_links):
                                    news_links.append({"title": _sh[:120], "url": _su})
                            logger.info(f"[NewsSerper] {target}: got {len(news_links)} items via Serper")
                except Exception as _sne:
                    logger.warning(f"[NewsSerper] {target} failed: {_sne}")

            # ── EisaX News Aggregator — final fallback ────────────────────────
            if len(news_links) < 2:
                try:
                    from core.news_aggregator import get_news as _agg_news
                    _agg = _agg_news(ticker=(_original_target if "_original_target" in dir() else target), limit=5)
                    for _an in _agg:
                        _at = _an.get("title", "")
                        _au = _an.get("url", "")
                        if _at and _au and not any(x["title"] == _at for x in news_links):
                            news_links.append({"title": _at[:120], "url": _au})
                    logger.info(f"[Aggregator] {target}: got {len(news_links)} items")
                except Exception as _age:
                    logger.warning(f"[Aggregator] {target} failed: {_age}")
            # ── News relevance filter ──────────────────────────────────────────
            # Remove articles that are clearly about unrelated companies/topics.
            # Applies to ALL news collected above.
            def _is_relevant_news(title: str, ticker_str: str, company: str) -> bool:
                """Return True if the headline is relevant to this stock/sector."""
                if not title:
                    return False
                t_low   = title.lower()
                tk_low  = ticker_str.lower().split('.')[0]  # base ticker, e.g. "adnocgas" from "ADNOCGAS.DU"
                co_low  = (company or "").lower()

                # ── Arabic title guard for MENA tickers ──────────────────────
                # If >40% of title chars are Arabic AND ticker is a MENA stock,
                # require the Arabic company name or English ticker to appear.
                # This blocks "ارباح القمم ..." from slipping in for DAMAC, etc.
                _mena_ticker = ticker_str.upper().endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
                _arabic_char_count = sum(1 for c in title if '\u0600' <= c <= '\u06FF')
                _is_arabic_title = _arabic_char_count > len(title) * 0.4
                if _is_arabic_title and _mena_ticker:
                    # Map known tickers to their Arabic company name fragments
                    _ar_name_map = {
                        "damac":       ["داماك"],
                        "emaar":       ["إعمار", "اعمار"],
                        "aldar":       ["الدار"],
                        "deyaar":      ["ديار"],
                        "adnoc":       ["أدنوك", "ادنوك", "adnoc"],
                        "adnocgas":    ["أدنوك", "ادنوك", "adnoc"],
                        "taqa":        ["طاقة", "taqa"],
                        "adx":         ["adx"],
                        "enbd":        ["الإمارات", "دبي الوطني"],
                        "fab":         ["أبوظبي الأول", "الأول"],
                        "dib":         ["الإسلامي"],
                        "emiratesnbd": ["الإمارات", "دبي"],
                        "aramco":      ["أرامكو", "ارامكو"],
                        "sabic":       ["سابك"],
                        "stc":         ["الاتصالات", "stc"],
                        "etisalat":    ["اتصالات", "e&"],
                        "du":          ["دو"],
                    }
                    _ar_names = _ar_name_map.get(tk_low, [])
                    # Also try the English ticker itself in the Arabic title
                    _has_match = (
                        (tk_low in t_low)  # English ticker in Arabic-heavy title
                        or any(ar in title for ar in _ar_names)
                    )
                    if not _has_match:
                        return False  # Arabic article about a different company → reject

                # Explicit known noise sources
                _noise_sources = [
                    "wallstreetbets", "reddit", "r/stocks", "memestocks",
                    "mcdonald's", "mcdonalds", "coca-cola", "coca cola",
                    "unrelated_company"
                ]
                if any(n in t_low for n in _noise_sources):
                    return False

                # Check for company name or ticker match
                # Strip futures suffixes like =F (e.g. "gc=f" → "gc") — too short to match,
                # but also try the commodity keyword directly
                _tk_clean = tk_low.split('=')[0]  # "gc=f" → "gc", "si=f" → "si"
                if tk_low and len(tk_low) > 2 and tk_low in t_low:
                    return True
                # For commodity futures tickers (GC=F, SI=F, CL=F), use commodity keywords
                _commodity_kw_map = {
                    "gc": ["gold", "xau", "bullion", "precious metal"],
                    "si": ["silver", "xag", "precious metal"],
                    "cl": ["crude", "oil", "wti", "petroleum"],
                    "ng": ["natural gas", "lng"],
                    "pl": ["platinum", "pgm", "precious metal"],
                    "pa": ["palladium", "pgm", "precious metal"],
                    "hg": ["copper", "base metal", "industrial metal"],
                    "pplt": ["platinum", "precious metal"],
                    "pall": ["palladium", "precious metal"],
                    "cper": ["copper", "base metal"],
                    # Gold ETFs — map their tickers to gold keywords
                    "gld": ["gold", "xau", "bullion", "precious metal"],
                    "iau": ["gold", "xau", "bullion", "precious metal"],
                    "sgol": ["gold", "xau", "bullion"],
                    "gldm": ["gold", "xau", "bullion"],
                    "slv": ["silver", "xag", "precious metal"],
                    "sivr": ["silver", "xag"],
                    "uso": ["crude", "oil", "wti", "petroleum"],
                    "bno": ["brent", "oil", "crude"],
                }
                if _tk_clean in _commodity_kw_map:
                    if any(k in t_low for k in _commodity_kw_map[_tk_clean]):
                        return True
                if co_low and len(co_low) > 3:
                    # Match first word of company name (e.g., "Microsoft" in "microsoft...")
                    first_word = co_low.split()[0]
                    if len(first_word) > 3 and first_word in t_low:
                        return True

                # Sector/macro keywords are always relevant
                _macro_ok = [
                    "oil", "opec", "brent", "crude", "energy", "gas",
                    "fed", "rate", "inflation", "gdp", "earnings", "market",
                    "uae", "dubai", "abu dhabi", "gulf", "iran", "hormuz",
                    "saudi", "aramco", "tech", "ai", "semiconductor",
                    "microsoft", "apple", "nvidia", "google", "alphabet",
                    "real estate", "property", "reit",
                    "bitcoin", "crypto", "btc", "ethereum",
                    "gold", "xau", "bullion", "precious metal", "silver", "xag",
                    "platinum", "palladium", "copper", "pgm", "base metal",
                    "commodity", "commodities",
                ]
                # For sector-relevant macro news — accept if sector matches
                _t_sector = (fund.get('sector') or '').lower()
                _sector_keys = {
                    'energy':      ['oil', 'opec', 'brent', 'crude', 'gas', 'lng', 'iran', 'hormuz'],
                    'technology':  ['ai', 'semiconductor', 'tech', 'chip', 'cloud', 'software'],
                    'real estate': ['real estate', 'property', 'reit', 'mortgage', 'housing'],
                    'financials':  ['bank', 'lending', 'fed', 'rate', 'credit', 'loan'],
                    'crypto':      ['bitcoin', 'btc', 'crypto', 'ethereum', 'blockchain'],
                    'commodit':    ['gold', 'xau', 'bullion', 'silver', 'precious metal', 'oil', 'brent', 'crude', 'commodity'],
                    'precious':    ['gold', 'xau', 'bullion', 'silver', 'platinum', 'palladium', 'precious metal'],
                }
                for sec, keys in _sector_keys.items():
                    if sec in _t_sector:
                        if any(k in t_low for k in keys):
                            return True

                # Broader market keywords — ONLY pass if the title also contains the
                # ticker or company name (prevents ETF/generic articles slipping through)
                _broad_ok = ['earnings', 'revenue', 'ipo', 'dividend', 'buyback',
                             'forecast', 'outlook', 'guidance', 'acquisition', 'merger']
                if any(k in t_low for k in _broad_ok):
                    # Require company/ticker anchor to avoid off-topic articles
                    # e.g. "JP Morgan Dividend ETF" passes 'dividend' but has no MSFT anchor
                    if tk_low and tk_low in t_low:
                        return True
                    if co_low and len(co_low.split()[0]) > 3 and co_low.split()[0] in t_low:
                        return True
                    # Fall through — broad keyword alone is NOT enough

                return False  # couldn't confirm relevance → filter out

            _co_name_for_filter = fund.get('company_name', target)
            _orig_count = len(news_links)
            news_links = [
                n for n in news_links
                if _is_relevant_news(n.get('title', ''), target, _co_name_for_filter)
            ]
            if len(news_links) < _orig_count:
                logger.info(f"[NewsFilter] {target}: filtered {_orig_count - len(news_links)} irrelevant articles, kept {len(news_links)}")

            # ── Post-filter Serper rescue — if filter killed all news, try Serper ──
            if len(news_links) == 0:
                try:
                    _serper_key2 = os.getenv("SERPER_API_KEY", "")
                    if _serper_key2:
                        import requests as _req_s2
                        _tb2 = target.split('.')[0]
                        _cn2 = fund.get('company_name') or dc_data.get('company_name') or _tb2
                        _gulf2 = target.upper().endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
                        if _gulf2:
                            _sq2 = f'"{_cn2}" OR "{_tb2}" stock news zawya arabianbusiness 2026'
                        else:
                            _sq2 = f'"{_cn2}" stock news 2026'
                        _sr2 = _req_s2.post(
                            "https://google.serper.dev/news",
                            headers={"X-API-KEY": _serper_key2, "Content-Type": "application/json"},
                            json={"q": _sq2, "num": 6}, timeout=8
                        )
                        if _sr2.status_code == 200:
                            for _sn2 in _sr2.json().get("news", []):
                                _sh2 = _sn2.get("title", "")
                                _su2 = _sn2.get("link", "")
                                # Apply same relevance filter — rescue doesn't bypass it
                                if _sh2 and _su2 and _is_relevant_news(_sh2, target, _co_name_for_filter):
                                    news_links.append({"title": _sh2[:120], "url": _su2})
                            logger.info(f"[NewsSerper/Rescue] {target}: {len(news_links)} items after rescue")
                except Exception as _sne2:
                    logger.debug(f"[NewsSerper/Rescue] {target}: {_sne2}")

            # ── EisaX Aggregator post-filter rescue ──────────────────────
            if len(news_links) == 0:
                try:
                    from core.news_aggregator import get_news as _agg_news2
                    _agg2 = _agg_news2(ticker=(_original_target if "_original_target" in dir() else target), limit=5)
                    for _an2 in _agg2:
                        _at2 = _an2.get("title", "")
                        _au2 = _an2.get("url", "")
                        # Apply same relevance filter — rescue doesn't bypass it
                        if _at2 and _au2 and _is_relevant_news(_at2, target, _co_name_for_filter):
                            news_links.append({"title": _at2[:120], "url": _au2})
                    logger.info(f"[Aggregator/Rescue] {target}: {len(news_links)} items")
                except Exception as _age2:
                    logger.warning(f"[Aggregator/Rescue] {target} failed: {_age2}")
            # ── Local/UAE: merge StockAnalysis dc_data into fund ─────────────
            # yfinance returns Unknown/0/1.0 defaults for regional stocks.
            # dc_data (from _stockanalysis_uae) has real values — merge them in.
            _LOCAL_SUFFIXES = (".AE", ".DU", ".SR", ".CA", ".KW", ".QA")
            if dc_data and target.upper().endswith(_LOCAL_SUFFIXES):
                def _dc_f(key):
                    v = dc_data.get(key)
                    try:
                        return float(str(v).strip()) if v not in (None, "", "N/A") else None
                    except Exception:
                        return None

                def _dc_size(key):
                    """Parse "19.23B AED" or "250M AED" → float bytes."""
                    v = str(dc_data.get(key, "") or "")
                    try:
                        if 'T' in v: return float(v.split('T')[0]) * 1e12
                        if 'B' in v: return float(v.split('B')[0]) * 1e9
                        if 'M' in v: return float(v.split('M')[0]) * 1e6
                    except Exception:
                        pass
                    return None

                def _dc_pct(key):
                    """Parse "+12.5%" or "-3.0%" → float."""
                    v = str(dc_data.get(key, "") or "")
                    try:
                        return float(v.strip().rstrip('%'))
                    except Exception:
                        return None

                # Beta: yfinance uses 1.0 as default — always prefer StockAnalysis
                _db = _dc_f('beta')
                if _db is not None and (not fund.get('beta') or abs(float(fund.get('beta', 1.0)) - 1.0) < 0.01):
                    fund['beta'] = _db

                # P/E TTM
                _dp = _dc_f('pe_ratio')
                if _dp and not fund.get('pe_ratio'):
                    fund['pe_ratio'] = _dp

                # Forward P/E
                _dfpe = _dc_f('forward_pe')
                if _dfpe and not forward_pe:
                    forward_pe = _dfpe

                # EPS
                _de = _dc_f('eps')
                if _de and not fund.get('eps'):
                    fund['eps'] = _de

                # Revenue
                _dr = _dc_size('revenue')
                if _dr and not fund.get('revenue'):
                    fund['revenue'] = _dr

                # Net Income
                _dni = _dc_size('net_income')
                if _dni and not fund.get('net_income'):
                    fund['net_income'] = _dni

                # Market Cap (raw billions from StockAnalysis)
                _mc_raw = dc_data.get('market_cap_raw')
                if _mc_raw and not fund.get('market_cap'):
                    fund['market_cap'] = (_mc_raw * 1e9 if _mc_raw < 1e6 else _mc_raw)

                # Revenue growth / EPS growth
                _drg = _dc_pct('rev_growth')
                if _drg is not None and not fund.get('revenue_growth'):
                    fund['revenue_growth'] = _drg
                _deg = _dc_pct('earnings_growth')
                if _deg is not None and not fund.get('eps_growth'):
                    fund['eps_growth'] = _deg

                # Dividend yield
                if dc_data.get('dividend_yield') and not dividend_yield:
                    try:
                        _dy_str = str(dc_data['dividend_yield']).strip().rstrip('%')
                        _dy = float(_dy_str) / 100
                        if _dy > 0:
                            dividend_yield = _dy
                    except Exception:
                        pass

                logger.info(f"[LocalMerge] {target}: beta={fund.get('beta')}, "
                            f"pe={fund.get('pe_ratio')}, rev={fund.get('revenue')}, "
                            f"mc={fund.get('market_cap')}")

            # ── collect: Fear & Greed + Events ──────────────────────────────
            fg_data = {}; ev_out = {}
            next_earnings = earnings_date
            try:
                fg_data = _f_fg.result(timeout=10) or {}
            except Exception:
                pass
            try:
                ev_out = _f_events.result(timeout=10) or {}
                if ev_out.get("earnings_date"):
                    next_earnings = ev_out["earnings_date"]
            except Exception:
                pass
            logger.info(f"[Analytics] FearGreed={fg_data.get('score','?')} NextEarnings={next_earnings}")

            # ── collect: Technical Analysis ──────────────────────────────────
            _is_local_market = target.upper().endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA", ".BH", ".MA", ".TN"))
            try:
                prices = _f_prices.result(timeout=15)
                if prices.empty:
                    if _is_local_market:
                        raise ValueError("UAE ticker — try local engine")
                    logger.warning(f"[Analytics] No price data returned for {target}")
                    return {
                        "type": "error",
                        "reply": (
                            f"⚠️ Insufficient market data for **{target}** — "
                            f"technical analysis unavailable.\n"
                            f"Verify the ticker symbol and try again."
                        ),
                    }
                series  = prices[target]
                summary = ca.generate_technical_summary(target, series)
                returns = series.pct_change().dropna()
                var_95  = ca.calculate_var(returns)
                max_dd  = ca.calculate_max_drawdown(series)
                # Inject annualised volatility into summary for Risk Profile floor
                summary['annual_vol'] = float(returns.std() * (252 ** 0.5)) if not returns.empty else 0.0
            except Exception as _price_e:
                if _is_local_market:
                    # ── UAE fallback: use local market data engine ────────────────
                    logger.info(f"[UAE Fallback] yfinance failed for {target}, trying local engine")
                    _local_enriched = {}
                    summary = {"price": 0, "trend": "N/A", "momentum": "N/A", "condition": "N/A", 
                               "rsi": 50.0, "sma_50": 0.0, "sma_200": 0.0, "adx": 0.0, "atr": 0.0, 
                               "macd": 0.0, "macd_signal": 0.0}
                    var_95 = 0.02; max_dd = 0.20
                    import pandas as _pd; series = _pd.Series(dtype=float)
                    
                    # 1️⃣ Try direct load from Parquet cache
                    _df_cache = None  # BUG-02 FIX: initialize before try block
                    try:
                        from core.market_data_engine import get_stock_data as _get_mde
                        _mkt = ("AE" if target.upper().endswith((".AE", ".DU"))
                                else "SA" if target.upper().endswith(".SR")
                                else "EG" if target.upper().endswith(".CA")
                                else "KW" if target.upper().endswith(".KW")
                                else "QA" if target.upper().endswith(".QA")
                                else "BH" if target.upper().endswith(".BH")
                                else "MA" if target.upper().endswith(".MA")
                                else "TN" if target.upper().endswith(".TN")
                                else None)
                        if _mkt:
                            _df_cache = _get_mde(target, _mkt, period="5y", force_refresh=False)
                            if _df_cache is not None and not _df_cache.empty and "Close" in _df_cache.columns:
                                series = _df_cache["Close"].copy()
                                logger.info(f"[UAE Fallback] Loaded {len(series)} rows from Parquet cache")
                    except Exception as _cache_e:
                        logger.warning(f"[UAE Fallback] Parquet load failed: {_cache_e}")

                    # 2️⃣ If we have historical data, calculate REAL technical indicators
                    if not series.empty and len(series) > 30:
                        try:
                            # Pass full OHLCV DataFrame if available (needed for real ADX/ATR)
                            _tech_input = _df_cache if (
                                _df_cache is not None          # BUG-02 FIX: no longer needs dir() check
                                and not _df_cache.empty
                                and all(c in _df_cache.columns for c in ("High", "Low", "Close"))
                            ) else series
                            summary = ca.generate_technical_summary(target, _tech_input)
                            returns = series.pct_change().dropna()
                            var_95 = ca.calculate_var(returns)
                            max_dd = ca.calculate_max_drawdown(series)
                            logger.info(f"[UAE Fallback] ✅ Calculated from {len(series)} data points: "
                                        f"RSI={summary.get('rsi','N/A')}, SMA50/200={summary.get('sma_50','N/A')}")
                        except Exception as _calc_e:
                            logger.warning(f"[UAE Fallback] Technical calc failed: {_calc_e}")
                    
                    # 3️⃣ Enrich with fundamentals (DFM/sector data)
                    try:
                        from core.local_market_enricher import enrich_local_analysis
                        _local_enriched = enrich_local_analysis(target)
                        # Populate real_price from local data
                        if not real_price and _local_enriched.get("price"):
                            real_price = float(_local_enriched["price"])
                            change_pct = float(_local_enriched.get("change_pct") or 0)
                        # Enrich fund dict with local fundamentals
                        _local_fund = _local_enriched.get("fundamentals", {})
                        if _local_fund:
                            if not fund.get("market_cap") and _local_fund.get("market_cap"):
                                fund["market_cap"] = _local_fund["market_cap"]
                            if not fund.get("pe_ratio") and _local_fund.get("pe_ratio"):
                                fund["pe_ratio"] = _local_fund["pe_ratio"]
                            if not fund.get("beta") and _local_fund.get("beta"):
                                fund["beta"] = _local_fund["beta"]
                        # Set known info from ticker info
                        _tk_info = _ticker_resolver.get_ticker_info(target) or {}
                        if not fund.get("sector") or fund.get("sector") in ("Unknown", "N/A", ""):
                            # Priority: dc_data (StockAnalysis) → Excel lookup → ticker_resolver → fallback
                            _dc_sector = dc_data.get("sector", "") if dc_data else ""
                            try:
                                from core.excel_stock_lookup import get_sector as _xl_sector
                                _excel_sector = _xl_sector(target, default="")
                            except Exception:
                                _excel_sector = ""
                            from core.fundamental_engine import _classify_sector as _clf_sec
                            _fallback_sector = _clf_sec(target) or "N/A"
                            fund["sector"] = (_dc_sector or _excel_sector or _tk_info.get("sector") or _fallback_sector)
                        if not fund.get("industry") or fund.get("industry") in ("Unknown", "N/A", ""):
                            try:
                                from core.excel_stock_lookup import get_industry as _xl_ind
                                _excel_ind = _xl_ind(target, default="")
                                if _excel_ind:
                                    fund["industry"] = _excel_ind
                            except Exception:
                                pass
                        if not fund.get("company_name") or fund.get("company_name") == target:
                            try:
                                from core.excel_stock_lookup import get_company_name as _xl_name
                                _excel_name = _xl_name(target, default="")
                                if _excel_name:
                                    fund["company_name"] = _excel_name
                            except Exception:
                                pass
                        if not fund.get("company_name"):
                            fund["company_name"] = _tk_info.get("name_en", target)
                        # Historical context
                        _local_hist = _local_enriched.get("historical", {})
                        if _local_hist.get("high_52w"):
                            fund["year_high"] = _local_hist["high_52w"]
                        if _local_hist.get("low_52w"):
                            fund["year_low"]  = _local_hist["low_52w"]
                    except Exception as _le:
                        logger.warning(f"[UAE Fallback] Local enrich failed: {_le}")
                    
                    logger.info(f"[UAE Fallback] Final: RSI={summary.get('rsi','N/A')}, "
                                f"Price={real_price}, DataPoints={len(series)}")
                else:
                    logger.error(f"[Analytics] Technical analysis failed for {target}: {_price_e}")
                    return {
                        "type": "error",
                        "reply": (
                            f"⚠️ Insufficient market data for **{target}** — "
                            f"technical analysis unavailable.\n"
                            f"Verify the ticker symbol and try again."
                        ),
                    }

        # ── 3a-fix. Price re-validation: fill real_price from fund/summary if concurrent fetch failed ──
        if not real_price:
            real_price = (
                float(fund.get('price') or 0) or
                float(summary.get('price') or 0) or
                float(dc_data.get('price') or 0) if dc_data else 0
            ) or None
            if real_price:
                logger.info(f"[Analytics] real_price recovered from fallback: {real_price}")

        # ── 3b. Sequential analyst target fetch (after concurrent pool exits) ──
        # All concurrent yfinance calls are done — now we can safely make one clean call.
        if not analyst_target and real_price:
            try:
                import yfinance as _yf_seq, time as _t_seq
                _seq_info = _yf_seq.Ticker(target).info or {}
                _at_seq = _seq_info.get("targetMeanPrice") or _seq_info.get("targetMedianPrice")
                if _at_seq:
                    analyst_target = float(_at_seq)
                    if not analyst_consensus:
                        analyst_consensus = _seq_info.get("recommendationKey", "").replace("_", " ").title()
                    if not analyst_count:
                        analyst_count = _seq_info.get("numberOfAnalystOpinions")
                    logger.info(f"[Analytics] analyst_target (sequential): {analyst_target}, consensus: {analyst_consensus}")
            except Exception as _seq_e:
                logger.debug(f"[Analytics] sequential analyst fetch failed: {_seq_e}")

        # ── 3b-fix. Sequential fundamentals re-fetch if concurrent pool returned sparse data ──
        # Concurrent yfinance calls often invalidate each other's session crumbs.
        # If key fields are missing, one clean sequential call recovers them.
        # Sparse if ANY 2 of the 3 key metrics missing
        _missing_count = sum(1 for k in ["net_margin", "roe", "revenue_growth"] if not fund.get(k))
        _fund_sparse = _missing_count >= 2
        if _fund_sparse:
            try:
                import yfinance as _yf_fund_seq, time as _t_seq
                _t_seq.sleep(1.5)  # let yfinance rate limit reset
                _fi_seq = _yf_fund_seq.Ticker(target).info or {}
                if _fi_seq.get("profitMargins"):
                    fund["net_margin"] = round(_fi_seq["profitMargins"] * 100, 1)
                if _fi_seq.get("returnOnEquity"):
                    fund["roe"] = round(_fi_seq["returnOnEquity"] * 100, 2)
                if _fi_seq.get("revenueGrowth"):
                    fund["revenue_growth"] = round(_fi_seq["revenueGrowth"] * 100, 1)
                if _fi_seq.get("earningsGrowth"):
                    fund["eps_growth"] = round(_fi_seq["earningsGrowth"] * 100, 1)
                if _fi_seq.get("grossMargins"):
                    fund["gross_margin"] = round(_fi_seq["grossMargins"] * 100, 1)
                if _fi_seq.get("operatingMargins"):
                    fund["operating_margin"] = round(_fi_seq["operatingMargins"] * 100, 1)
                if not fund.get("pe_ratio") and _fi_seq.get("trailingPE"):
                    fund["pe_ratio"] = round(_fi_seq["trailingPE"], 1)
                if not fund.get("current_ratio") and _fi_seq.get("currentRatio"):
                    fund["current_ratio"] = round(_fi_seq["currentRatio"], 2)
                if not fund.get("beta") and _fi_seq.get("beta"):
                    fund["beta"] = round(_fi_seq["beta"], 2)
                if not fund.get("eps") and _fi_seq.get("trailingEps"):
                    fund["eps"] = round(_fi_seq["trailingEps"], 2)
                if not fund.get("market_cap") and _fi_seq.get("marketCap"):
                    fund["market_cap"] = _fi_seq["marketCap"]
                logger.info(f"[FundFix] {target}: sequential re-fetch recovered nm={fund.get('net_margin')}, roe={fund.get('roe')}, rg={fund.get('revenue_growth')}")
            except Exception as _ff_e:
                logger.debug(f"[FundFix] sequential re-fetch failed: {_ff_e}")

        # ── 3b/c/d. Regional DB enrichment (Phase-2 refactor → regional_handler) ──
        from core.services.regional_handler import merge_regional_data as _merge_regional
        fund = _merge_regional(target, fund)

        # ── 3e. Extreme price move detection (crash / halt investigation) ─────
        _is_crash = abs(change_pct) >= 20  # ≥20% single-day move
        _crash_direction = "CRASH 📉" if change_pct <= -20 else "CIRCUIT BREAKER RALLY 📈" if change_pct >= 20 else ""

        # ── 4. Format helpers ─────────────────────────────────────────────────
        def _B(n):
            try:
                if not n: return "N/A"
                v = float(n)
                if _currency_lbl != "USD":
                    if v >= 1e12: return f"{v/1e12:.2f}T {_currency_sym}"
                    if v >= 1e9:  return f"{v/1e9:.1f}B {_currency_sym}"
                    if v >= 1e6:  return f"{v/1e6:.0f}M {_currency_sym}"
                    return f"{v:,.0f} {_currency_sym}"
                return f"${v/1e9:.1f}B" if v >= 1e9 else f"${v/1e6:.0f}M"
            except: return "N/A"
        def _P(n): return f"{n:.1f}%" if n else "N/A"
        def _X(n): return f"{n:.1f}x" if n else "N/A"

        # Currency symbol — use correct symbol for each market (Phase-2 → regional_handler)
        from core.services.regional_handler import detect_currency as _detect_currency
        _currency_sym, _currency_lbl = _detect_currency(target)
        _t_upper = target.upper()
        _fallback_price = real_price or summary.get('price', 0)
        _is_local_mkt = _currency_lbl != "USD"
        _is_local_currency = _currency_lbl in ("SAR", "AED", "EGP", "KWF", "QAR")
        price_str = (f"{_fallback_price:,.2f} {_currency_sym} ({change_pct:+.2f}%)"
                     if _fallback_price and _is_local_mkt and change_pct
                     else f"{_fallback_price:,.2f} {_currency_sym}"
                     if _fallback_price and _is_local_mkt
                     else f"${_fallback_price:,.2f} ({change_pct:+.2f}%)"
                     if _fallback_price and change_pct
                     else f"${_fallback_price:,.2f}"
                     if _fallback_price else "N/A")

        # ── 5. Rolling beta (DRY — one computation for all uses) ─────────────
        _is_crypto_asset = target.endswith('-USD') and any(c in target for c in ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'ADA', 'AVAX'])
        if _is_crypto_asset:
            _effective_beta = self._compute_rolling_beta(target)
            logger.info(f"[Crypto Beta] {target} rolling beta = {_effective_beta}")
        else:
            # Priority: dc_data (StockAnalysis) > fund (yfinance) > rolling > sector default
            # Reject yfinance default of exactly 1.0 for regional stocks (always a placeholder)
            _dc_beta_v = float(dc_data.get('beta') or 0)
            _yf_beta_v = float(fund.get('beta') or 0)
            _is_local_stock = any(target.upper().endswith(sfx) for sfx in ('.AE', '.DU', '.SR', '.CA', '.KW', '.QA'))
            # Reject yfinance garbage for regional stocks:
            # (a) exactly 1.0 → placeholder default  (b) ≤ 0 → calculation artifact / no data
            if _is_local_stock and (abs(_yf_beta_v - 1.0) < 0.005 or _yf_beta_v <= 0):
                _yf_beta_v = 0  # discard suspicious default
            _effective_beta = _dc_beta_v or _yf_beta_v or 0
            if not _effective_beta:
                # Sector-appropriate default when no real beta available
                _s_eb = (fund.get('sector', '') or '').lower()
                _effective_beta = (0.3 if any(x in _s_eb for x in ('energy', 'oil', 'gas', 'utilities'))
                                   else 0.7 if any(x in _s_eb for x in ('real estate', 'financials', 'banks'))
                                   else 1.1)

        # ── Sanitize summary: replace NaN/inf with safe defaults ─────────────
        import math as _math_san
        _summary_defaults = {"rsi": 50.0, "sma_50": 0.0, "sma_200": 0.0,
                              "adx": 0.0, "atr": 0.0, "macd": 0.0, "macd_signal": 0.0, "price": 0.0}
        for _sk, _sd in _summary_defaults.items():
            _sv = summary.get(_sk, _sd)
            try:
                _svf = float(_sv or 0)
                if _math_san.isnan(_svf) or _math_san.isinf(_svf):
                    summary[_sk] = _sd
                else:
                    summary[_sk] = _svf
            except Exception:
                summary[_sk] = _sd

        # ── Sanitize fund dict: replace NaN/inf in numeric fields ──────────────
        import math as _math_fund
        _FUND_NUMERIC_FIELDS = ["pe_ratio", "eps", "revenue_growth", "roe", "roic",
                                "net_margin", "gross_margin", "debt_equity", "market_cap",
                                "div_yield", "dividend_yield", "beta", "forward_pe",
                                "week52_high", "week52_low", "volume_avg90d"]
        for _fk in _FUND_NUMERIC_FIELDS:
            _fv = fund.get(_fk)
            if _fv is not None:
                try:
                    _ff = float(_fv)
                    if _math_fund.isnan(_ff) or _math_fund.isinf(_ff):
                        fund[_fk] = None
                except (TypeError, ValueError):
                    pass

        # ── 5a. Fetch on-chain data for crypto (parallel-safe) ───────────────
        _onchain_data = {}
        _btc_etf_signal = ""
        if _is_crypto_asset:
            try:
                _onchain_data = self._fetch_onchain(target)
                logger.info(f"[OnChain] {target}: ATH=${_onchain_data.get('ath',0)}, HashRate={_onchain_data.get('hash_rate_eh',0)}EH/s, ActiveAddr={_onchain_data.get('active_addresses',0)}")
            except Exception as _oc_e:
                logger.warning(f"[OnChain] Failed for {target}: {_oc_e}")
            # BTC-only: ETF institutional flow signal (IBIT volume vs 90-day avg)
            if target == "BTC-USD":
                try:
                    _btc_etf_signal = _fetch_btc_etf_flows()
                except Exception as _etf_e:
                    logger.debug(f"[BTC ETF] skipped: {_etf_e}")

        # ── 5a-ii. Detect energy sector + fetch oil price ─────────────────────
        _ENERGY_SECTORS = {"energy", "oil & gas", "oil", "petroleum", "integrated oil", "gas"}
        _ENERGY_PREFIXES = ("ADNOC", "2222", "2030", "2010", "TAQA", "DANA", "ARAMCO")
        _t_base = target.split('.')[0].upper()
        _is_energy = (
            fund.get('sector', '').lower() in _ENERGY_SECTORS
            or fund.get('industry', '').lower() in {"oil & gas integrated", "oil & gas e&p",
                "oil & gas refining & marketing", "oil & gas equipment & services"}
            or any(_t_base.startswith(pfx) for pfx in _ENERGY_PREFIXES)
            or "GAS" in _t_base or "OIL" in _t_base or "PETRO" in _t_base or "ENERG" in _t_base
        )
        _oil_data = {}
        if _is_energy:
            try:
                import yfinance as _yf_oil
                # fast_info: ~3x faster than .info — price + prev_close is all we need here
                _brent    = _yf_oil.Ticker("BZ=F")
                _oil_fi   = _brent.fast_info
                _oil_price = float(getattr(_oil_fi, "last_price",     None) or 0) or None
                _prev      = float(getattr(_oil_fi, "previous_close", None) or 0) or None
                _oil_change = 0
                if _oil_price and _prev:
                    _oil_change = ((_oil_price - _prev) / _prev) * 100
                _oil_data = {"price": _oil_price, "change_pct": round(_oil_change, 2), "name": "Brent Crude"}
                logger.info(f"[Oil] Brent=${_oil_price:.2f} ({_oil_change:+.1f}%)")
            except Exception as _oil_e:
                logger.warning(f"[Oil] Brent fetch failed: {_oil_e}")

        # ── 5b-pre. Fair Value estimate (computed here so data_block can include it) ──
        _fv_estimate = None
        _fv_label = "Analyst consensus"
        _valuation_pe = None
        if not analyst_target and real_price:
            try:
                _eps_ttm = float(fund.get('eps') or dc_data.get('eps') or 0)
                _eg_raw = fund.get('eps_growth') or str(dc_data.get('earnings_growth', '0')).strip('%+')
                _eg = float(_eg_raw) if _eg_raw else 0
                _fpe_val = float(forward_pe or 0)
                _valuation_pe = int(_fpe_val) if _fpe_val > 0 else None
                # STRICT DATA MODE: no synthetic sector multiple fallback.
                # Compute FV only when forward P/E is available from data.
                if _eps_ttm > 0 and _valuation_pe:
                    _fwd_eps = _eps_ttm * (1 + _eg / 100)
                    _fv_estimate = round(_fwd_eps * _valuation_pe, 3)
                    # Do NOT override analyst_target — keep them separate.
                    # analyst_target = None means "no real analyst coverage"
                    _fv_label = f"EisaX Fair Value (EPS×{_valuation_pe}x)"
                    logger.info(f"[FairValue] {target}: FwdEPS={_fwd_eps:.3f} × PE={_valuation_pe} = {_fv_estimate}")
            except Exception as _fve:
                logger.debug(f"[FairValue] calc failed: {_fve}")

        # _display_target: real analyst target OR EisaX fair-value estimate (clearly labelled)
        _display_target = analyst_target or _fv_estimate  # for display/scorecard upside only
        _target_is_estimate = (analyst_target is None and _fv_estimate is not None)

        # SMA technical target fallback — used in scorecard when no analyst/FV target
        _sma50_sc  = float(summary.get('sma_50', 0) or 0)
        _sma200_sc = float(summary.get('sma_200', 0) or 0)
        import math as _math_tgt
        if _sma50_sc and _math_tgt.isnan(_sma50_sc): _sma50_sc = 0.0
        if _sma200_sc and _math_tgt.isnan(_sma200_sc): _sma200_sc = 0.0
        _sma_tech_target = None
        if not _display_target and real_price:
            if _sma200_sc and real_price < _sma200_sc:
                _sma_tech_target = round(_sma200_sc, 3)
            elif _sma50_sc and real_price < _sma50_sc:
                _sma_tech_target = round(_sma50_sc, 3)
        # Pass to scorecard as display_target only when no real target
        _scorecard_target = _display_target or _sma_tech_target

        # ⚠️ DO NOT change is_etf=False below — _etf_meta_early is NOT yet defined at this point.
        # It is assigned ~350 lines later. ETFs get their own scenario table from _build_etf_sc().
        _precomputed = FinancialAgent._precompute_report_data(
            real_price=real_price,
            forward_pe=forward_pe,
            analyst_target=analyst_target,
            fund=fund,
            summary=summary,
            dc_data=dc_data,
            currency_sym=_currency_sym,
            is_crypto=_is_crypto_asset,
            is_etf=False,  # ← MUST stay False — _etf_meta_early not defined yet here
        )

        # ── Scenario Probabilities (computed from live Fear&Greed + technicals) ─
        _fg_sc  = int((fg_data or {}).get('score', 50) or 50)
        _macd_v = float((summary or {}).get('macd', 0) or 0)
        _macd_s = float((summary or {}).get('macd_signal', 0) or 0)
        _macd_bull = _macd_v > _macd_s
        _p_vs_sma50_pos = (
            float((summary or {}).get('price', 0) or 0) >
            float((summary or {}).get('sma_50', 0) or 0)
        )
        if _is_crypto_asset:
            _sc_bull  = 20 + (5 if _fg_sc < 20 else 0) + (5 if _p_vs_sma50_pos else 0)
            _sc_base  = 35
            _sc_shock = 15
        else:
            _sc_bull  = 25 + (5 if _fg_sc < 20 else 0) + (10 if _macd_bull else 0)
            _sc_base  = 40
            _sc_shock = 10
        _sc_bear = max(100 - _sc_bull - _sc_base - _sc_shock, 5)
        # Re-normalise to exactly 100
        _total_sc = _sc_bull + _sc_base + _sc_bear + _sc_shock
        if _total_sc != 100:
            _sc_bear += (100 - _total_sc)
        _precomputed['sc_prob_bull']  = _sc_bull
        _precomputed['sc_prob_base']  = _sc_base
        _precomputed['sc_prob_bear']  = _sc_bear
        _precomputed['sc_prob_shock'] = _sc_shock
        # Expected Value: weighted avg of scenario returns (shock = -25% default)
        _shock_return = -25.0
        _ev_num = (
            (_precomputed.get('val_bull_updown') or 0) * _sc_bull +
            (_precomputed.get('val_base_updown') or 0) * _sc_base +
            (_precomputed.get('val_bear_updown') or 0) * _sc_bear +
            _shock_return * _sc_shock
        ) / 100
        _precomputed['scenario_ev'] = round(_ev_num, 1)

        # US peer comparison table (for US equities only).
        _us_peer_map = {
            "MSFT": ["GOOGL", "AAPL", "AMZN", "META"],
            "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
            "GOOGL": ["MSFT", "META", "AMZN", "AAPL"],
            "META": ["GOOGL", "SNAP", "PINS", "MSFT"],
            "AMZN": ["MSFT", "GOOGL", "SHOP", "WMT"],
            "NVDA": ["AMD", "INTC", "QCOM", "AVGO"],
            "AMD": ["NVDA", "INTC", "QCOM", "AVGO"],
        }
        _peer_table_str = "No peer data available"
        _peer_rows = []  # always initialized — populated below if US stock with known peers
        _non_us_suffixes = (".SR", ".AE", ".DU", ".CA", ".KW", ".QA", ".BH")
        _is_us_stock_for_peers = not target.upper().endswith(_non_us_suffixes)
        _us_peers = [] if _low_data_compact_mode else (_us_peer_map.get(target.upper(), []) if _is_us_stock_for_peers else [])
        if _us_peers:
            try:
                import yfinance as yf
                _peer_rows = []
                for _pt in _us_peers[:4]:
                    try:
                        _pi = yf.Ticker(_pt).info or {}
                        _mkt_cap_raw = _pi.get("marketCap") or 0
                        _mkt_cap_val = int(_mkt_cap_raw) if _mkt_cap_raw else 0
                        _peer_rows.append({
                            "ticker": _pt,
                            "name": str(_pi.get("shortName", _pt))[:20],
                            "fwd_pe": round(float(_pi.get("forwardPE") or 0), 1),
                            "mkt_cap": _mkt_cap_val,
                            "div_yield": round(_safe_div_yield(_pi.get("dividendYield") or 0) * 100, 2),
                            "rev_growth": round(float(_pi.get("revenueGrowth") or 0) * 100, 1),
                            "gross_margin": round(float(_pi.get("grossMargins") or 0) * 100, 1),
                        })
                    except Exception:
                        continue
                if _peer_rows:
                    _peer_table_str = "| Ticker | Fwd P/E | Mkt Cap | Div Yield | Rev Growth | Gross Margin |\\n"
                    _peer_table_str += "|--------|---------|---------|-----------|------------|---------------|\\n"
                    for _pr in _peer_rows:
                        _mkt_cap_val = _pr["mkt_cap"] or 0
                        if _mkt_cap_val >= 1e12:
                            _mc = f"${_mkt_cap_val / 1e12:.1f}T"
                        elif _mkt_cap_val >= 1e9:
                            _mc = f"${_mkt_cap_val / 1e9:.0f}B"
                        else:
                            _mc = "N/A"
                        _peer_table_str += (
                            f"| {_pr['ticker']} | {_pr['fwd_pe'] or 'N/A'}x | {_mc} | "
                            f"{_pr['div_yield'] or 'N/A'}% | {_pr['rev_growth'] or 'N/A'}% | "
                            f"{_pr['gross_margin'] or 'N/A'}% |\\n"
                        )
            except Exception as _peer_e:
                logger.debug(f"[USPeers] Peer table build skipped: {_peer_e}")

        # ── 5b. Build data block for DeepSeek ─────────────────────────────────
        _cache_source_note = (
            "\n⚡ DATA SOURCE: TradingView Live Cache (updated every 15 min) — price and technicals are REAL and LIVE."
            "\n⛔ DO NOT write 'CRITICAL DATA NOTE' or 'no live price injected' — the price below IS the live price."
        ) if fund.get("data_source") == "TradingView Live Cache" or (_cache_row is not None and price_str != "N/A") else ""

        data_block = f"""
TICKER: {_original_target if "_original_target" in dir() else target} (resolved: {target})
COMPANY: {fund.get('company_name') or (_original_target if '_original_target' in dir() else target)}
SECTOR: {fund.get('sector', 'N/A')} | INDUSTRY: {fund.get('industry', 'N/A')}
CURRENCY: {_currency_lbl} (use {_currency_sym} symbol in ALL price references){chr(10) + "IMPORTANT: This is an Egyptian stock (EGX). Market Cap, prices and all monetary values are in EGP (Egyptian Pound ج.م). Do NOT convert to USD or display in USD." if _t_upper.endswith(".CA") else ""}{_cache_source_note}
LIVE PRICE: {price_str}
MARKET CAP: {_B(fund.get('market_cap'))}
QUALITY SCORE: {fund.get('fundamental_score', 'N/A')}/100

NEWS SENTIMENT: {news_sent} (score: {news_score})

MACRO: 10Y Treasury: {t10y}% | Fed Funds: {fed}% | Unemployment: {unemp}% | CPI YoY: {inflation}% | GDP Growth: {gdp}%

GROWTH:
- Revenue Growth YoY: {_P(fund.get('revenue_growth'))}
- EPS Growth YoY: {_P(fund.get('eps_growth'))}
- Revenue (TTM): {_B(fund.get('revenue'))}
- EPS (TTM): ${fund.get('eps', 'N/A')}

PROFITABILITY:
- Gross Margin: {_P(fund.get('gross_margin'))}
- Operating Margin: {_P(fund.get('operating_margin'))}
- Net Margin: {_P(fund.get('net_margin'))}
- ROE: {_P(fund.get('roe'))}
- ROIC: {_P(fund.get('roic'))}

VALUATION:
- P/E (TTM): {_X(fund.get('pe_ratio'))}
- Forward P/E: {_X(float(dc_data.get("forward_pe") or 0) or forward_pe)}
=== PRE-COMPUTED VALUES (use these exact numbers — do NOT recalculate) ===
Forward EPS: {f"{_currency_sym}{_precomputed['forward_eps']:.2f} [{_precomputed['forward_eps_source']}]" if _precomputed['forward_eps'] else "N/A — source data absent"}

VALUATION SCENARIOS (mandatory table — copy these numbers exactly, including Probability column):
| Scenario | Probability | Multiple | Implied Price | vs Current |
|----------|-------------|----------|---------------|------------|
| 🐻 Bear  | {_precomputed['sc_prob_bear']}% | {f"{_precomputed['val_bear_pe']}{'x' if isinstance(_precomputed['val_bear_pe'], (int,float)) else ''}" if _precomputed['val_bear_pe'] is not None else 'N/A'} | {f"{_currency_sym}{_precomputed['val_bear_price']:,.0f}" if _precomputed['val_bear_price'] else 'N/A'} | {f"{_precomputed['val_bear_updown']:+.1f}%" if _precomputed['val_bear_updown'] is not None else 'N/A'} |
| ⚖️ Base  | {_precomputed['sc_prob_base']}% | {f"{_precomputed['val_base_pe']}{'x' if isinstance(_precomputed['val_base_pe'], (int,float)) else ''}" if _precomputed['val_base_pe'] is not None else 'N/A'} | {f"{_currency_sym}{_precomputed['val_base_price']:,.0f}" if _precomputed['val_base_price'] else 'N/A'} | {f"{_precomputed['val_base_updown']:+.1f}%" if _precomputed['val_base_updown'] is not None else 'N/A'} |
| 🚀 Bull  | {_precomputed['sc_prob_bull']}% | {f"{_precomputed['val_bull_pe']}{'x' if isinstance(_precomputed['val_bull_pe'], (int,float)) else ''}" if _precomputed['val_bull_pe'] is not None else 'N/A'} | {f"{_currency_sym}{_precomputed['val_bull_price']:,.0f}" if _precomputed['val_bull_price'] else 'N/A'} | {f"{_precomputed['val_bull_updown']:+.1f}%" if _precomputed['val_bull_updown'] is not None else 'N/A'} |
| 💥 Macro Shock | {_precomputed['sc_prob_shock']}% | — | — | -25% est. |

Expected Value: {_precomputed['scenario_ev']:+.1f}% (Bull×{_precomputed['sc_prob_bull']}% + Base×{_precomputed['sc_prob_base']}% + Bear×{_precomputed['sc_prob_bear']}% + Shock×{_precomputed['sc_prob_shock']}%)

Upside to Analyst Target: {f"{_precomputed['upside_to_target']:+.1f}%" if _precomputed['upside_to_target'] is not None else 'N/A'}
Price vs SMA50:  {f"{_precomputed['pct_vs_sma50']:+.1f}%" if _precomputed['pct_vs_sma50'] is not None else 'N/A'}
Price vs SMA200: {f"{_precomputed['pct_vs_sma200']:+.1f}%" if _precomputed['pct_vs_sma200'] is not None else 'N/A'}
Entry Zone: {f"{_currency_sym}{_precomputed['entry_zone']:,.2f} (price is {_precomputed['pct_above_entry']:.1f}% above)" if _precomputed['entry_zone'] else "At or below entry — zone active"}
=== END PRE-COMPUTED VALUES ===
- P/S (TTM): {_X(fund.get('ps_ratio'))}
- EV/EBITDA: {_X(fund.get('ev_ebitda'))}
- Beta: {_effective_beta}
- Gross Margin: {_P(fund.get('gross_margin'))}{" (Non-GAAP; GAAP may vary ~2-3%)" if fund.get('gross_margin') else ""}
- Dividend Yield: {f"{dividend_yield*100:.2f}%" if dividend_yield and dividend_yield > 0.001 else "Minimal (<0.1%)"}

ANALYST CONSENSUS:
- Recommendation: {analyst_consensus or 'N/A'} ({analyst_count or 'N/A'} analysts)
- Price Target (Mean): {((_currency_sym if _is_local_currency else "$") + str(round(_display_target, 2))) if _display_target else 'N/A'}{" [" + _fv_label + "]" if _target_is_estimate else ""}
- Upside Potential: {f"{((_display_target/real_price)-1)*100:.1f}%" if _display_target and real_price else 'N/A'}
{"- NOTE: No analyst coverage found. Target shown is EisaX Fair Value Estimate (Forward EPS × " + str(_valuation_pe) + "x sector P/E). Present as 'EisaX Fair Value Estimate' in section 5, NOT as analyst consensus. Do NOT use SMA200 as a price target." if _target_is_estimate else ""}

US PEER COMPARISON TABLE:
{_peer_table_str}

BALANCE SHEET:
- Cash: {_B(fund.get('cash'))}
- Total Debt: {_B(fund.get('total_debt'))}
- Debt/Equity: {fund.get('debt_equity', 'N/A')}
- Current Ratio: {fund.get('current_ratio') or 'N/A'}

EARNINGS:
- Last Earnings Date: {fund.get('last_earnings_date', 'N/A')}
- NEXT EARNINGS DATE: {next_earnings or 'N/A'}
- EPS Actual vs Est (last): ${fund.get('last_eps_actual', 'N/A')} vs ${fund.get('last_eps_estimate', 'N/A')}
- Earnings Surprise: {fund.get('earnings_surprise_pct', 'N/A')}%
- Next Quarter EPS Estimate: ${ev_out.get('eps_est_avg', 'N/A')} (range: ${ev_out.get('eps_est_low','?')} – ${ev_out.get('eps_est_high','?')})
- Next Quarter Revenue Estimate: {f"${ev_out['rev_est_avg']/1e9:.1f}B" if ev_out.get('rev_est_avg') else 'N/A'} (range: {f"${ev_out['rev_est_low']/1e9:.1f}B" if ev_out.get('rev_est_low') else '?'} – {f"${ev_out['rev_est_high']/1e9:.1f}B" if ev_out.get('rev_est_high') else '?'})

MARKET SENTIMENT (Fear & Greed Index):
- Score: {fg_data.get('score', 'N/A')} / 100
- Rating: {fg_data.get('rating', 'N/A')} ({fg_data.get('label_ar', '')})
- Implication: {"Extreme fear — historically a contrarian buy signal; staged entries become more favorable" if (fg_data.get('score') or 50) < 25 else "Fear zone — market is risk-off; tighter stop losses advised" if (fg_data.get('score') or 50) < 45 else "Neutral sentiment" if (fg_data.get('score') or 50) < 55 else "Greed — market momentum favors bulls, but watch for complacency" if (fg_data.get('score') or 50) < 75 else "Extreme greed — elevated risk of correction; use caution on new entries"}

TECHNICALS:
- Trend: {summary['trend']} (Price vs SMA200)
- Momentum: {summary['momentum']} (MACD)
- RSI: {summary['rsi']:.1f} → {summary['condition']}
- MACD: {summary.get('macd', 0):.2f} | Signal: {summary.get('macd_signal', 0):.2f} | {"Bullish crossover" if summary.get('macd', 0) > summary.get('macd_signal', 0) else "Bearish crossover"}
- SMA50: {_currency_sym}{summary['sma_50']:,.2f} | SMA200: {_currency_sym}{summary['sma_200']:,.2f}
- Price vs SMA50: {f"{((real_price - summary['sma_50']) / summary['sma_50'] * 100):+.1f}%" if real_price and summary.get('sma_50') and float(summary.get('sma_50',0)) != 0 else "N/A"} | vs SMA200: {f"{((real_price - summary['sma_200']) / summary['sma_200'] * 100):+.1f}%" if real_price and summary.get('sma_200') and float(summary.get('sma_200',0)) != 0 else "N/A"}
- ADX: {summary.get('adx', 0):.1f} ({"Strong trend" if summary.get('adx', 0) >= 30 else "Confirmed trend" if summary.get('adx', 0) >= 25 else "Emerging trend" if summary.get('adx', 0) >= 20 else "Weak trend"}) | ATR: {summary.get('atr', 0):.2f}
{("- ⚠️ Technical Note: Momentum is improving, but ADX still maps to a weak trend regime, so directional conviction remains limited." if (summary.get('adx', 0) < 20 and (summary.get('macd', 0) > 0 or summary.get('rsi', 0) > 55)) else "- ⚠️ Technical Note: Momentum is improving, but ADX still maps to an emerging trend regime, so the move should be treated as early-stage rather than fully validated." if (summary.get('adx', 0) < 25 and (summary.get('macd', 0) > 0 or summary.get('rsi', 0) > 55)) else "")}
{(lambda v_t, v_a: f"""
VOLUME:
- Today: {v_t/1e6:.1f}M vs 90-day avg {v_a/1e6:.1f}M → {"🔴 LOW volume ({:.0f}% of avg) — weak conviction in move".format(v_t/v_a*100) if v_a and v_t/v_a < 0.75 else "🟢 HIGH volume ({:.0f}% of avg) — strong conviction".format(v_t/v_a*100) if v_a and v_t/v_a > 1.25 else "⚪ Normal volume ({:.0f}% of avg)".format(v_t/v_a*100) if v_a else "N/A"}
""" if v_a else "")(
    fund.get('volume_today', 0) or 0,
    fund.get('volume_avg90d', 0) or 0,
)}
{(lambda _fh, _fl, _fp: (lambda _rng: f"""
FIBONACCI LEVELS — current price {_currency_sym}{_fp:,.2f} | 52W range {_currency_sym}{_fl:,.2f}–{_currency_sym}{_fh:,.2f}
{"⚡ Price ABOVE 52W High — all retracement levels are SUPPORT; use extension levels for resistance." if _fp > _fh else ""}

RESISTANCE LEVELS (above current price {_currency_sym}{_fp:,.2f} only):
{chr(10).join(
    f"  {k}: {_currency_sym}{v:,.2f} ({(v-_fp)/_fp*100:+.1f}%)"
    for k, v in sorted([
        ("127.2% ext", round(_fl+_rng*1.272,2)),
        ("161.8% ext", round(_fl+_rng*1.618,2)),
        ("78.6%",      round(_fl+_rng*0.786,2)),
        ("61.8%",      round(_fl+_rng*0.618,2)),
        ("50.0%",      round((_fh+_fl)/2,2)),
        ("38.2%",      round(_fl+_rng*0.382,2)),
        ("23.6%",      round(_fl+_rng*0.236,2)),
    ], key=lambda x: x[1])
    if v > _fp * 1.001
) or "  (none within range — use 127.2% and 161.8% extension levels above)"}

SUPPORT LEVELS (below current price {_currency_sym}{_fp:,.2f} only):
{chr(10).join(
    f"  {k}: {_currency_sym}{v:,.2f} ({(v-_fp)/_fp*100:+.1f}%)"
    for k, v in sorted([
        ("61.8%",   round(_fl+_rng*0.618,2)),
        ("50.0%",   round((_fh+_fl)/2,2)),
        ("38.2%",   round(_fl+_rng*0.382,2)),
        ("23.6%",   round(_fl+_rng*0.236,2)),
        ("52W Low", round(_fl,2)),
    ], key=lambda x: x[1], reverse=True)
    if v < _fp * 0.999
) or "  N/A"}

TECHNICAL LEVELS TABLE (deterministic S/R ladder - use this exact table in Section 3):
{_precomputed.get('sr_levels_table', 'N/A')}
WARNING: A level is resistance ONLY if it is ABOVE {_currency_sym}{_fp:,.2f}. A level is support ONLY if it is BELOW {_currency_sym}{_fp:,.2f}. Never label a level as resistance if it is below the current price.
""")(_fh - _fl) if _fh and _fl else "FIBONACCI LEVELS: N/A — 52-week range data unavailable\n")(
    fund.get('week52_high') or fund.get('year_high') or 0,
    fund.get('week52_low')  or fund.get('year_low')  or 0,
    real_price or 0,
)}
RISK:
- VaR (95%, daily): {var_95*100:.2f}%
- Max Historical Drawdown: {max_dd*100:.2f}%
{"" if not _onchain_data else f"""
ON-CHAIN METRICS (LIVE):
- All-Time High: ${(_onchain_data.get('ath') or 0):,.0f} (ATH change: {(_onchain_data.get('ath_change_pct') or 0):.1f}%, date: {_onchain_data.get('ath_date', 'N/A')})
- Supply: {(_onchain_data.get('circulating_supply') or 0):,.0f} / {(_onchain_data.get('max_supply') or 0):,.0f} ({_onchain_data.get('supply_ratio', 0)}% mined)
- 24h Volume: ${(_onchain_data.get('total_volume_24h') or 0)/1e9:.1f}B
- Market Cap Rank: #{_onchain_data.get('mc_rank', 'N/A')}
{f'- Hash Rate: {_onchain_data["hash_rate_eh"]:.0f} EH/s' if _onchain_data.get('hash_rate_eh') else ''}
{f'- Active Addresses (24h): {_onchain_data["active_addresses"]:,}' if _onchain_data.get('active_addresses') else ''}
{f'- Transactions (24h): {_onchain_data["n_tx_24h"]:,}' if _onchain_data.get('n_tx_24h') else ''}
{('- ' + _btc_etf_signal) if _btc_etf_signal else ''}
IMPORTANT: Use these on-chain metrics in your analysis. Discuss supply scarcity, network activity, and hash rate health.
"""}
{"" if not _oil_data.get('price') else f"""
OIL PRICE DATA (LIVE):
- Brent Crude: ${_oil_data['price']:.2f}/bbl ({_oil_data['change_pct']:+.1f}%)
IMPORTANT: This is an ENERGY SECTOR stock. Oil prices are the #1 driver of revenue and valuation.
Include an Oil Price Sensitivity Analysis table in your report showing impact at $50, $60, $70, $80, $90/bbl.
Discuss OPEC+ dynamics and energy transition risks.

OIL PRICE SENSITIVITY (pre-computed):
| Oil Price (Brent) | Change from Current | Est. Revenue Impact | Est. Stock Price |
|-------------------|--------------------|--------------------|-----------------|
| ${_oil_data['price']:.0f}/bbl (current) | — | Base | {_currency_sym}{real_price or 0:,.2f} |
| $90/bbl | {((90 - _oil_data['price']) / _oil_data['price'] * 100):+.0f}% | {((90 - _oil_data['price']) / _oil_data['price'] * 70):+.0f}% | {_currency_sym}{(real_price or 0) * (1 + (90 - _oil_data['price']) / _oil_data['price'] * 0.55):,.2f} |
| $80/bbl | {((80 - _oil_data['price']) / _oil_data['price'] * 100):+.0f}% | {((80 - _oil_data['price']) / _oil_data['price'] * 70):+.0f}% | {_currency_sym}{(real_price or 0) * (1 + (80 - _oil_data['price']) / _oil_data['price'] * 0.55):,.2f} |
| $70/bbl | {((70 - _oil_data['price']) / _oil_data['price'] * 100):+.0f}% | {((70 - _oil_data['price']) / _oil_data['price'] * 70):+.0f}% | {_currency_sym}{(real_price or 0) * (1 + (70 - _oil_data['price']) / _oil_data['price'] * 0.55):,.2f} |
| $60/bbl | {((60 - _oil_data['price']) / _oil_data['price'] * 100):+.0f}% | {((60 - _oil_data['price']) / _oil_data['price'] * 70):+.0f}% | {_currency_sym}{(real_price or 0) * (1 + (60 - _oil_data['price']) / _oil_data['price'] * 0.55):,.2f} |
| $50/bbl | {((50 - _oil_data['price']) / _oil_data['price'] * 100):+.0f}% | {((50 - _oil_data['price']) / _oil_data['price'] * 70):+.0f}% | {_currency_sym}{(real_price or 0) * (1 + (50 - _oil_data['price']) / _oil_data['price'] * 0.55):,.2f} |
"""}
{f"""SCENARIO ANALYSIS (Energy-Sector — Oil-Price-Adjusted):
Note: Impact already pre-calculated using 0.55x oil sensitivity. Copy EXACTLY — do NOT add extra columns.
| Scenario | Impact | Implied Price | Suggested Hedge |
|----------|--------|---------------|-----------------|
| 🚀 Oil Spike $150+/bbl | +{((((150 - _oil_data.get('price',80)) / _oil_data.get('price',80)) * 55)):.1f}% | {_currency_sym}{(real_price or 0) * (1 + (((150 - _oil_data.get('price',80)) / _oil_data.get('price',80)) * 0.55)):,.2f} | Hold / partial profit |
| 🛢️ Oil Crash to $50/bbl | {(-(((_oil_data.get('price',80)-50)/_oil_data.get('price',80))*55)):.1f}% | {_currency_sym}{(real_price or 0) * (1 + (-(((_oil_data.get('price',80)-50)/_oil_data.get('price',80))*55))/100):,.2f} | Gold + Tech |
| 📉 OPEC+ Production Surge | {(-18 * 0.55):.1f}% | {_currency_sym}{(real_price or 0) * (1 + (-18 * 0.55)/100):,.2f} | Diversified equities |
| 🌱 Energy Transition (long-term) | {(-30 * 0.55 * 0.75):.1f}% | {_currency_sym}{(real_price or 0) * (1 + (-30 * 0.55 * 0.75)/100):,.2f} | Clean energy + Tech |
| 🏦 Fed Rate Shock +2% | {((-8 * max(float(_effective_beta), 0.4)) + (-5 * 0.55)):.1f}% | {_currency_sym}{(real_price or 0) * (1 + ((-8 * max(float(_effective_beta), 0.4)) + (-5 * 0.55))/100):,.2f} | Treasuries + Cash |
""" if _is_energy else (f"""SCENARIO ANALYSIS (UAE Real Estate — Geopolitical + Rate Sensitive):
Note: Dubai real estate reacts to regional geopolitics AND global rates, not just market beta ({_effective_beta}).
Use -20% to -30% for geopolitical scenarios regardless of low beta — tourist/investor sentiment collapses in conflict.
| Scenario | Impact Driver | Est. Price Impact | Implied Price ({_currency_sym}) | Suggested Hedge |
|----------|--------------|------------------|--------------------------|-----------------|
| 🚀 Dubai Tourism Boom | +35% tourism surge | +{(35 * 0.40):.1f}% | {_currency_sym}{(real_price or 0) * (1 + (35 * 0.40)/100):,.2f} | Hold / add on dips |
| 🌍 Geopolitical Risk Escalation (Middle East) | Gulf security crisis — energy markets & liquidity impact | -{(28):.1f}% | {_currency_sym}{(real_price or 0) * (1 - 28/100):,.2f} | Gold + global REITs |
| 📉 Dubai Bear Market | -30% DFM correction | -{(30 * 0.85):.1f}% | {_currency_sym}{(real_price or 0) * (1 - 30 * 0.85/100):,.2f} | Cash + Bonds |
| 🏦 Fed Rate Shock +2% | Higher financing cost | -{(18 * max(float(_effective_beta), 0.35)):.1f}% | {_currency_sym}{(real_price or 0) * (1 - 18 * max(float(_effective_beta), 0.35)/100):,.2f} | US Treasuries |
| 🌱 Expo/Infrastructure Catalyst | Mega-project boost | +{(20 * 0.50):.1f}% | {_currency_sym}{(real_price or 0) * (1 + 20 * 0.50/100):,.2f} | Hold / add |
""" if (
    any(x in (fund.get('sector','') or '').lower() for x in ('real estate', 'property', 'reits'))
    and target.upper().endswith(('.DU', '.AE'))
) else (f"""SCENARIO ANALYSIS (Crash-Recovery — Post -39%+ Event):
⚠️ This stock experienced a severe single-day crash. Beta-adjusted scenarios are NOT meaningful here.
Use event-driven scenarios instead (corporate action, mean-reversion, or further collapse).
| Scenario | Trigger | Price Impact | Implied Price ({_currency_sym}) | Suggested Action |
|----------|---------|-------------|--------------------------|-----------------|
| ✅ Corporate Action Clarified | Rights issue priced in — stock normalises | +{(45):.0f}% | {_currency_sym}{(real_price or 0) * 1.45:,.2f} | BUY on confirmed clarity |
| 🔄 Partial Mean Reversion | Stock recovers 50% of crash | +{(25):.0f}% | {_currency_sym}{(real_price or 0) * 1.25:,.2f} | Hold / add gradually |
| ⚠️ Fundamental Impairment | Crash = real earnings deterioration | -{(30):.0f}% | {_currency_sym}{(real_price or 0) * 0.70:,.2f} | STOP LOSS immediately |
| 📉 Continued Selling / Forced Liquidation | No buyers for 1-2 weeks | -{(20):.0f}% | {_currency_sym}{(real_price or 0) * 0.80:,.2f} | Volume confirmation pending |
| 🏦 EM Currency Devaluation | Local currency weakens -15% | -{(15):.0f}% | {_currency_sym}{(real_price or 0) * 0.85:,.2f} | Hedge with USD exposure |
CRITICAL INSTRUCTION: In section 8, present THESE crash-recovery scenarios instead of generic beta-adjusted ones.
The #1 question investors need answered is: WHY did the stock crash -39%? Address this directly.
""" if abs(change_pct or 0) >= 20 else f"""SCENARIO ANALYSIS (Beta-Adjusted — use these in section 9 of your report):
Note: Beta = {_effective_beta}. Impact already pre-calculated (Market_Move × Beta). Copy EXACTLY — do NOT add extra columns.
REQUIREMENT: Show at least 2 BULLISH rows (🚀💡📈) and at least 2 BEARISH rows (📉🏦🤖⚠️).
| Scenario | Impact | Implied Price | Suggested Hedge |
|----------|--------|---------------|-----------------|
| 🚀 Bull Market Rally (+20%) | {(20 * float(_effective_beta)):.1f}% | ${(real_price or 0) * (1 + (20 * float(_effective_beta))/100):.2f} | Hold / add on dips |
| 💡 Fed Pivot / Rate Cut (+15%) | {(15 * float(_effective_beta)):.1f}% | ${(real_price or 0) * (1 + (15 * float(_effective_beta))/100):.2f} | Growth + Tech |
| 📉 AI/Tech Slowdown (-20%) | {(-20 * float(_effective_beta)):.1f}% | ${(real_price or 0) * (1 + (-20 * float(_effective_beta))/100):.2f} | Healthcare + Staples |
| 🏦 Fed Rate Shock +2% (-18%) | {(-18 * float(_effective_beta)):.1f}% | ${(real_price or 0) * (1 + (-18 * float(_effective_beta))/100):.2f} | Value stocks + Cash |
"""))}
{(lambda: (
    # ── Rich news context block: engine (3-bucket) + fallback (flat list) ──
    __import__('core.news_engine_client', fromlist=['build_news_prompt_block'])
    .build_news_prompt_block(_engine_news_data, target)
    if _engine_news_data and (_engine_news_data.get('direct') or _engine_news_data.get('sector') or _engine_news_data.get('country'))
    else (
        (chr(10) + "LATEST NEWS (LIVE — integrate into Section 4 Risks and Section 7 Why Now):" + chr(10)
         + chr(10).join(f"- {n['title']}" for n in news_links[:5]) + chr(10)
         + "INSTRUCTION: Reference at least 1-2 of these headlines in Section 4 Key Risks and/or Section 7 Why Now.")
        if news_links else ""
    )
)())}"""

        # ── X Sentiment Block (Grok) — appended to data_block if available ────
        # Gives DeepSeek real investor sentiment from X/Twitter in the last 48h.
        # If Grok call failed, _x_data is empty → block is skipped silently.
        _x_block = ""
        if not _low_data_compact_mode and _x_data and _x_data.get("sentiment") and _x_data.get("source") != "grok-unavailable":
            _xs   = _x_data.get("sentiment", "")
            _xsc  = _x_data.get("score", 0.0)
            _xsum = _x_data.get("x_summary", "")
            _xbrk = _x_data.get("breaking")
            _xthm = _x_data.get("themes", [])
            _xpst = _x_data.get("top_posts", [])

            _x_block = f"\n\n--- X/Twitter Sentiment (Grok Live · last 48h) ---\n"
            _x_block += f"Overall: {_xs} (score: {_xsc:+.2f})\n"
            if _xsum:
                _x_block += f"Summary: {_xsum}\n"
            if _xbrk:
                _x_block += f"⚡ BREAKING: {_xbrk}\n"
            if _xthm:
                _x_block += f"Key Themes: {' · '.join(_xthm)}\n"
            if _xpst:
                _x_block += "Top Posts from X:\n"
                for _p in _xpst[:4]:
                    _lk  = f" ({_p.get('likes',0):,} likes)" if _p.get('likes') else ""
                    _src = _p.get('source', '')
                    _txt = _p.get('text', '')[:160]
                    _dt  = _p.get('date', '')
                    _imp = _p.get('impact', 'Neutral')
                    _ico = "🟢" if _imp == "Positive" else "🔴" if _imp == "Negative" else "⚪"
                    _x_block += f"  {_ico} {_src}{_lk} ({_dt}): \"{_txt}\"\n"

            _x_block += (
                "INSTRUCTION: Use this X sentiment data in Section 8 (Why Now?) under a "
                "'📱 X Sentiment' bullet. If there is BREAKING news, mention it in Section 4 "
                "(Key Risks). ONLY cite sources that appear in the Top Posts above."
            )
            data_block += _x_block
            logger.info(f"[Grok] X sentiment injected for {target}: {_xs} ({_xsc:+.2f})")


        # ── EisaX Cache: Gulf Peer Data → Section 6 ──────────────────────────
        # Inject real live cache data for peer comparison so DeepSeek uses
        # actual Gulf market numbers instead of training knowledge.
        try:
            import sys as _sys2
            from core.config import BASE_DIR as _BD2
            _r2 = str(_BD2)
            if _r2 not in _sys2.path:
                _sys2.path.insert(0, _r2)
            from pipeline import cache as _pc, fetcher as _pf
            from query_engine import QueryEngine as _QE
            _qe2 = _QE(_pc, _pf)
            _peer_sector = fund.get("sector", "")
            if not _low_data_compact_mode and _peer_sector and target.upper().endswith((".SR", ".AE", ".DU", ".CA", ".KW", ".QA", ".BH")):
                _peer_df = _qe2.cross_market(_peer_sector)
                if _peer_df is not None and not _peer_df.empty:
                    if "market_cap_basic" in _peer_df.columns:
                        _peer_df = _peer_df.dropna(subset=["market_cap_basic"]).nlargest(8, "market_cap_basic")
                    _peer_rows = []
                    for _pr in _peer_df.itertuples():
                        _pt = getattr(_pr, "ticker", "")
                        _pnm = getattr(_pr, "name", _pt)
                        _pclose = getattr(_pr, "close", None)
                        _ppe = getattr(_pr, "price_earnings_ttm", None)
                        _prsi = getattr(_pr, "RSI", None)
                        _pchg = getattr(_pr, "change", None)
                        _pmc = getattr(_pr, "market_cap_basic", None)
                        try:
                            import math as _m
                            _ppe_s  = f"{float(_ppe):.1f}x"  if _ppe  and not _m.isnan(float(_ppe))  else "N/A"
                            _prsi_s = f"{float(_prsi):.1f}"  if _prsi and not _m.isnan(float(_prsi)) else "N/A"
                            _pchg_s = f"{float(_pchg):+.2f}%"if _pchg and not _m.isnan(float(_pchg)) else "N/A"
                            _pmc_s  = f"{float(_pmc)/1e9:.0f}B" if _pmc and not _m.isnan(float(_pmc)) else "N/A"
                            _pclose_s = f"{float(_pclose):.2f}" if _pclose and not _m.isnan(float(_pclose)) else "N/A"
                        except Exception:
                            _ppe_s = _prsi_s = _pchg_s = _pmc_s = _pclose_s = "N/A"
                        _peer_rows.append(f"  {_pnm} ({_pt}): price={_pclose_s}, change={_pchg_s}, P/E={_ppe_s}, RSI={_prsi_s}, mktcap={_pmc_s}")
                    if _peer_rows:
                        data_block += (
                            "\n\nGULF PEER COMPARISON DATA (LIVE — from EisaX 15-min cache):\n"
                            f"Sector: {_peer_sector} | Top peers by market cap:\n"
                            + "\n".join(_peer_rows)
                            + "\nINSTRUCTION: Use these EXACT live numbers in Section 6 (⚔️ Peer Comparison). "
                            "Compare P/E, RSI momentum, and market cap vs the target stock. "
                            "Do NOT use training data — these are real-time Gulf market values."
                        )
                        logger.info("[EisaX] Injected %d Gulf peers into LLM prompt for %s", len(_peer_rows), target)
        except Exception as _pe2:
            logger.debug("[EisaX] Gulf peer injection skipped: %s", _pe2)

        # ── ETF data_block override ───────────────────────────────────────────
        # If this is an ETF, REPLACE the stock data_block with ETF-specific one.
        # ETF detection runs later (after sector fill) so we patch here.
        _etf_meta_early = None
        try:
            from core.etf_intelligence import detect_etf as _detect_etf_early
            # Use profile._yf_raw for best ETF detection (has quoteType field)
            _etf_early_yf_raw = (
                profile.get("_yf_raw", {}) if "profile" in dir() and profile else {}
            ) or fund or {}
            if str(target).upper().endswith(_ETF_EQUITY_ONLY_SUFFIXES):
                logger.debug("[ETF] %s: skipped early ETF detection for equity-only suffix", target)
            else:
                _etf_meta_early = _detect_etf_early(target, _etf_early_yf_raw)
            if _etf_meta_early:
                from core.etf_intelligence import build_etf_data_block as _build_etf_db, build_etf_scenarios as _build_etf_sc
                from core.macro_intelligence import get_live_macro as _etf_glm
                _etf_macro_live = {}
                try: _etf_macro_live = _etf_glm()
                except Exception: pass
                _etf_db = _build_etf_db(
                    _etf_meta_early, target, real_price or 0, change_pct or 0,
                    summary, fg_data, macro=_etf_macro_live, var_95=var_95, max_dd=max_dd
                )
                if _low_data_compact_mode:
                    _etf_scenarios = "ETF scenarios disabled in low-data compact mode."
                else:
                    _etf_scenarios = _build_etf_sc(_etf_meta_early["etf_type"], real_price or 100, _etf_macro_live)
                data_block = _etf_db + "\n\n" + _etf_scenarios
                logger.info(f"[ETF] {target}: replaced data_block with ETF-specific version ({_etf_meta_early['etf_type']})")
                # Set sector for ETF if missing — ensures news filter and report show correct sector
                if not fund.get("sector") or fund.get("sector") in ("Unknown", "N/A", ""):
                    _is_futures_ticker = target.upper().endswith("=F") or target.upper() in (
                        "GC=F", "SI=F", "CL=F", "NG=F", "PL=F", "PA=F", "HG=F", "BZ=F"
                    )
                    _etf_sector_map = {
                        "commodity_gold":      "Commodities - Precious Metals" if _is_futures_ticker else "ETF - Precious Metals",
                        "commodity_silver":    "Commodities - Precious Metals" if _is_futures_ticker else "ETF - Precious Metals",
                        "commodity_platinum":  "Commodities - Precious Metals" if _is_futures_ticker else "ETF - Precious Metals",
                        "commodity_palladium": "Commodities - Precious Metals" if _is_futures_ticker else "ETF - Precious Metals",
                        "commodity_copper":    "Commodities - Industrial Metals" if _is_futures_ticker else "ETF - Industrial Metals",
                        "commodity_oil":       "Commodities - Energy" if _is_futures_ticker else "ETF - Energy",
                        "commodity_other":     "Commodities" if _is_futures_ticker else "ETF - Commodities",
                        "bond_treasury":    "Fixed Income",
                        "bond_corporate":   "Fixed Income",
                        "bond_tips":        "Fixed Income",
                        "equity_index_us":  "Equities - US Index",
                        "equity_index_intl":"Equities - International",
                        "equity_sector":    "Equities - Sector",
                        "reit_etf":         "Real Estate",
                        "leveraged":        "Leveraged ETF",
                        "dividend":         "Equities - Dividend",
                    }
                    fund["sector"] = _etf_sector_map.get(_etf_meta_early["etf_type"], "ETF")
        except Exception as _etf_db_e:
            logger.debug(f"[ETF] data_block override skipped: {_etf_db_e}")

        data_block = compact_low_data_generation_inputs(
            data_block,
            {
                "coverage_count": _data_coverage_count,
                "coverage_level": _data_coverage_level,
                "low_data_mode": _low_data_compact_mode,
            },
        )

        # ── 5b. Pre-calculate Positioning (used in prompt) ──────────────────
        import math as _math_ep
        def _ep_clean(v):
            try:
                f = float(v or 0)
                return 0.0 if (_math_ep.isnan(f) or _math_ep.isinf(f)) else f
            except Exception:
                return 0.0
        sma50_v  = _ep_clean(summary.get('sma_50', 0))
        sma200_v = _ep_clean(summary.get('sma_200', 0))
        _rp_ref  = _ep_clean(real_price or _fallback_price or 0)

        # ── Entry: prefer nearest Fibonacci support BELOW current price ──────
        _h52 = _ep_clean(fund.get('week52_high', 0))
        _l52 = _ep_clean(fund.get('week52_low', 0))
        _fib_ep = None
        if _h52 and _l52 and _rp_ref and _h52 > _l52:
            # Fibonacci retracement levels (from 52W low)
            _fib_levels = [
                _l52 + (_h52 - _l52) * 0.382,  # 38.2%
                _l52 + (_h52 - _l52) * 0.500,  # 50.0%
                _l52 + (_h52 - _l52) * 0.618,  # 61.8%
                _l52 + (_h52 - _l52) * 0.236,  # 23.6%
            ]
            # Nearest Fibonacci level that is BELOW current price (best support entry)
            _fib_below = [f for f in _fib_levels if f < _rp_ref * 0.995]
            if _fib_below:
                _fib_ep = max(_fib_below)  # closest support below price

        # Use Fibonacci entry if it's a meaningful pullback (1-15% below current)
        if _fib_ep and _rp_ref and 0.85 <= (_fib_ep / _rp_ref) <= 0.99:
            ep = _fib_ep
        elif sma200_v:
            ep = sma200_v * 1.02
        else:
            ep = None

        from core.services.report_snapshot import ReportSnapshot as _ReportSnapshot

        _trust_audit_log = []
        _trust_visible_warnings = []
        _report_snapshot = None
        _report_classification = "SAFE"

        _atr_val = float(summary.get('atr', 0) or fund.get('atr', 0) or 0)

        def _atr_stop(ref_price, atr, mult=2.0, fallback_pct=0.09):
            if atr and atr > 0 and ref_price and ref_price > 0 and atr < ref_price * 0.25:
                return round(ref_price - (mult * atr), 4)
            return round(ref_price * (1 - fallback_pct), 4) if ref_price else None

        if _rp_ref and sma200_v:
            _pct_from_sma = (_rp_ref - sma200_v) / sma200_v
            if _pct_from_sma < -0.10:
                ep = _rp_ref * 0.97
                sp = _atr_stop(_rp_ref, _atr_val, fallback_pct=0.09)
            elif _pct_from_sma < 0:
                ep = sma200_v * 0.98
                sp = _atr_stop(sma200_v, _atr_val, fallback_pct=0.08)
            else:
                ep = ep if ep else sma200_v * 1.01
                sp = _atr_stop(_rp_ref, _atr_val, mult=2.0, fallback_pct=0.09)
        else:
            ep = ep if ep else (_rp_ref * 0.96 if _rp_ref else None)
            sp = _atr_stop(_rp_ref, _atr_val, fallback_pct=0.09)

        _snapshot_target = _display_target or None
        _trust_target_is_sma = False
        _trust_sma_used = "SMA50" if (sma50_v and not sma200_v) else "SMA200"
        if not _snapshot_target and sma200_v and _rp_ref:
            _is_crypto_tgt = bool(
                str(target).upper().endswith(('-USD', '-BTC', '-ETH'))
                or 'BTC' in str(target).upper()
                or 'ETH' in str(target).upper()
                or 'crypto' in str(fund.get('sector', '')).lower()
            )
            if _rp_ref < sma200_v:
                _snapshot_target = sma200_v if _is_crypto_tgt else sma200_v * 1.15
                _trust_sma_used = "SMA200"
            elif sma50_v and _rp_ref < sma50_v:
                _snapshot_target = sma50_v
                _trust_sma_used = "SMA50"
            else:
                _snapshot_target = sma200_v * 1.15
                _trust_sma_used = "SMA200"
            _trust_target_is_sma = True
        elif not _snapshot_target and sma50_v and _rp_ref:
            _snapshot_target = sma50_v if _rp_ref < sma50_v else sma50_v * 1.05
            _trust_target_is_sma = True
            _trust_sma_used = "SMA50"

        def _fmt_positioning_price(p):
            if not p:
                return "N/A"
            return f"{p:,.2f} {_currency_sym}" if _is_local_currency else f"${p:,.2f}"

        _trust_target_label = (
            f"{_trust_sma_used} Technical Target"
            if _trust_target_is_sma
            else _fv_label
            if _target_is_estimate
            else "Analyst Target"
        )
        pre_entry = _fmt_positioning_price(ep)
        if ep and _rp_ref and ep < _rp_ref * 0.985:
            pre_entry += " *(Limit Order - wait for pullback)*"
        pre_stop = _fmt_positioning_price(sp)
        if _snapshot_target and _rp_ref:
            _snapshot_upside = ((_snapshot_target / _rp_ref) - 1) * 100
            pre_target = (
                f"{_fmt_positioning_price(_snapshot_target)} ({_snapshot_upside:+.1f}%) - *{_trust_target_label}*"
            )
        else:
            pre_target = "N/A"

        _snapshot_ts = datetime.now().isoformat()
        _price_source = "realtime" if real_price else "cache" if _fallback_price else "fallback"
        _price_delay = 15 if _price_source == "cache" else None
        _interpretation_labels = {}
        _approved_phrase_map = {}
        _interpretation_block = ""
        _approved_phrase_block = ""
        _interpretation_context = {}

        def _safe_positive_float(value):
            try:
                numeric = float(value or 0)
            except Exception:
                return None
            return numeric if numeric > 0 else None

        def _nearest_support_level(current_price, *candidates):
            if not current_price:
                return None
            valid = []
            for candidate in candidates:
                numeric = _safe_positive_float(candidate)
                if numeric and numeric < current_price:
                    valid.append(numeric)
            return max(valid) if valid else None

        def _nearest_resistance_level(current_price, *candidates):
            if not current_price:
                return None
            valid = []
            for candidate in candidates:
                numeric = _safe_positive_float(candidate)
                if numeric and numeric > current_price:
                    valid.append(numeric)
            return min(valid) if valid else None

        try:
            from core.services.interpretation_engine import (
                build_interpretation_labels as _build_interpretation_labels,
                format_interpretation_block as _format_interpretation_block,
            )
            from core.services.phrase_builder import (
                build_approved_phrase_map as _build_approved_phrase_map,
                format_approved_phrase_block as _format_approved_phrase_block,
            )

            _interp_price = _safe_positive_float(_rp_ref)
            _interp_support = _nearest_support_level(
                _interp_price,
                (summary or {}).get("fib_support"),
                (summary or {}).get("support"),
                (summary or {}).get("fib_key_support"),
                (dc_data or {}).get("support"),
                sma50_v,
                sma200_v,
                _l52,
            )
            _interp_resistance = _nearest_resistance_level(
                _interp_price,
                (summary or {}).get("fib_resistance"),
                (summary or {}).get("resistance"),
                (dc_data or {}).get("resistance"),
                sma50_v,
                sma200_v,
                _h52,
            )
            _interp_div_yield = (
                dividend_yield
                if "dividend_yield" in dir() and dividend_yield is not None
                else fund.get("dividend_yield")
                or fund.get("trailingAnnualDividendYield")
            )
            _interp_entry = _safe_positive_float(ep)
            _interp_volume_today = _safe_positive_float(
                fund.get("volume_today") or (summary or {}).get("volume")
            )
            _interp_volume_avg = _safe_positive_float(
                fund.get("volume_avg90d") or fund.get("avg_volume")
            )
            _trend_text = str((summary or {}).get("trend", "") or "").lower()
            if "bear" in _trend_text or "below sma200" in _trend_text:
                _primary_trend = "bearish"
            elif "bull" in _trend_text or "above sma200" in _trend_text:
                _primary_trend = "bullish"
            else:
                _primary_trend = "neutral"

            _interpretation_labels = _build_interpretation_labels(
                adx=float((summary or {}).get("adx", 0) or 0),
                rsi=float((summary or {}).get("rsi", 50) or 50),
                price=_interp_price or 0,
                support=_interp_support or 0,
                resistance=_interp_resistance or 0,
                div_yield=_interp_div_yield,
                entry_price=_interp_entry,
                volume_today=_interp_volume_today,
                volume_avg=_interp_volume_avg,
            )
            _approved_phrase_map = _build_approved_phrase_map(
                _interpretation_labels,
                primary_trend=_primary_trend,
            )
            _interpretation_block = _format_interpretation_block(_interpretation_labels)
            _approved_phrase_block = _format_approved_phrase_block(_approved_phrase_map)
            _interpretation_context = {
                "adx": float((summary or {}).get("adx", 0) or 0),
                "rsi": float((summary or {}).get("rsi", 50) or 50),
                "price": _interp_price or 0,
                "support": _interp_support or 0,
                "resistance": _interp_resistance or 0,
                "div_yield": _interp_div_yield,
                "entry_price": _interp_entry or 0,
                "volume_today": _interp_volume_today or 0,
                "volume_avg": _interp_volume_avg or 0,
                "primary_trend": _primary_trend,
                "labels": dict(_interpretation_labels),
                "phrases": dict(_approved_phrase_map),
            }
        except Exception as _interp_err:
            logger.warning("[InterpretationLayer] Initialization failed for %s: %s", target, _interp_err)
            _trust_audit_log.append({
                "event": "interpretation_layer_initialization_failed",
                "timestamp": _snapshot_ts,
                "error": str(_interp_err),
            })
            _report_classification = "PARTIAL"
        _trust_raw_snapshot = {
            "ticker": {"value": (_original_target if "_original_target" in dir() and _original_target != target else target), "source": "fallback", "timestamp": _snapshot_ts},
            "price": {"value": _rp_ref or None, "source": _price_source, "timestamp": _snapshot_ts, "delay_minutes": _price_delay},
            "entry": {"value": ep, "source": "calculated", "timestamp": _snapshot_ts},
            "stop": {"value": sp, "source": "calculated", "timestamp": _snapshot_ts},
            "target": {"value": _snapshot_target, "source": "calculated" if (_trust_target_is_sma or _target_is_estimate) else "fallback", "timestamp": _snapshot_ts},
            "beta": {"value": locals().get("_effective_beta") or fund.get('beta') or 1.0, "source": "cache" if fund.get('beta') else "fallback", "timestamp": _snapshot_ts},
            "pe": {"value": fund.get('pe_ratio'), "source": "cache" if fund.get('pe_ratio') else "fallback", "timestamp": _snapshot_ts},
            "forward_pe": {"value": forward_pe, "source": "cache" if forward_pe else "fallback", "timestamp": _snapshot_ts},
            "sma50": {"value": sma50_v or None, "source": "calculated", "timestamp": _snapshot_ts},
            "sma200": {"value": sma200_v or None, "source": "calculated", "timestamp": _snapshot_ts},
            "week52_high": {"value": _h52 or None, "source": "cache" if fund.get('week52_high') else "fallback", "timestamp": _snapshot_ts},
            "week52_low": {"value": _l52 or None, "source": "cache" if fund.get('week52_low') else "fallback", "timestamp": _snapshot_ts},
            "market_cap": {"value": fund.get('market_cap'), "source": "cache" if fund.get('market_cap') else "fallback", "timestamp": _snapshot_ts},
            "div_yield": {"value": fund.get('dividend_yield') or fund.get('trailingAnnualDividendYield'), "source": "cache" if (fund.get('dividend_yield') or fund.get('trailingAnnualDividendYield')) else "fallback", "timestamp": _snapshot_ts},
        }
        try:
            _report_snapshot = _ReportSnapshot(_trust_raw_snapshot)
            _report_snapshot.set("_interpretation_context", {"value": _interpretation_context, "source": "calculated", "timestamp": _snapshot_ts})
            _report_snapshot.set("_interpretation_labels", {"value": dict(_interpretation_labels), "source": "calculated", "timestamp": _snapshot_ts})
            _report_snapshot.set("_interpretation_block", {"value": _interpretation_block, "source": "calculated", "timestamp": _snapshot_ts})
            _report_snapshot.set("_approved_phrase_map", {"value": dict(_approved_phrase_map), "source": "calculated", "timestamp": _snapshot_ts})
            _report_snapshot.freeze()
            _trust_audit_log.extend(_report_snapshot.get_audit_log())
        except Exception as _snapshot_err:
            logger.warning("[TrustLayer] Snapshot initialization failed for %s: %s", target, _snapshot_err)
            _trust_visible_warnings.append("Data validation layer unavailable — report generated with fallback safeguards.")
            _trust_audit_log.append({
                "event": "snapshot_initialization_failed",
                "timestamp": _snapshot_ts,
                "error": str(_snapshot_err),
            })
            _report_classification = "PARTIAL"
        _is_local_currency = _currency_lbl in ("SAR", "AED", "EGP", "KWF", "QAR")
        # ── ETF Detection ────────────────────────────────────────────────────
        _etf_meta = None
        try:
            from core.etf_intelligence import detect_etf as _detect_etf
            # _yf_raw is in profile (from get_full_stock_profile), not in fund
            _yf_info_for_etf = (
                profile.get("_yf_raw", {}) if "profile" in dir() and profile else {}
            ) or fund.get("_yf_raw", {}) or {}
            if str(target).upper().endswith(_ETF_EQUITY_ONLY_SUFFIXES):
                logger.debug("[ETF] %s: skipped ETF detection for equity-only suffix", target)
            else:
                _etf_meta = _detect_etf(target, _yf_info_for_etf)
            if _etf_meta:
                logger.info(f"[ETF] {target} detected as {_etf_meta['etf_type']} — {_etf_meta['etf_label']}")
        except Exception as _etf_e:
            logger.debug(f"[ETF] detection skipped: {_etf_e}")

        # ── Pre-compute Scorecard (ONE computation — reused for both hint + display) ──
        # Build the full scorecard markdown ONCE here, BEFORE the DeepSeek prompt.
        # Extract verdict from it → guaranteed identical to what appears in the report.
        try:
            if _etf_meta:
                # ETF path — use ETF-specific scorecard
                from core.etf_intelligence import (
                    calculate_etf_score as _calc_etf_score,
                    build_etf_scorecard_md as _build_etf_sc_md,
                )
                _live_macro = {}
                try:
                    from core.macro_intelligence import get_live_macro as _glm
                    _live_macro = _glm()
                except Exception: pass
                _etf_score_result = _calc_etf_score(
                    _etf_meta, summary, fg_data,
                    var_95=var_95, macro=_live_macro
                )
                _sc_display_ticker = (_original_target if "_original_target" in dir() and _original_target != target else target)
                _pre_scorecard_md = _build_etf_sc_md(
                    _sc_display_ticker, _etf_meta, real_price, _etf_score_result, summary,
                    resolved_ticker=target
                )
                _etf_conv = ('High' if _etf_score_result['score'] >= 75
                             else 'Medium' if _etf_score_result['score'] >= 60 else 'Low')
                scorecard_verdict_hint = f"{_etf_score_result['verdict']} {_etf_score_result['emoji']} (Conviction: {_etf_conv})"
                logger.info(f"[ETF Scorecard] {target}: {scorecard_verdict_hint} score={_etf_score_result['score']}")
                # Mirror _last_scorecard_decision so _handle_analytics has structured data
                _etf_v = _etf_score_result['verdict']
                _etf_et = ('REDUCE INTO STRENGTH' if _etf_v in ('REDUCE', 'SELL', 'AVOID')
                           else 'BUY NOW — trend confirmed' if _etf_v == 'BUY' else 'WAIT')
                self._last_scorecard_decision = {
                    'verdict':   _etf_v,
                    'timing_en': _etf_et,
                    'timing':    _etf_et,
                    'score':     _etf_score_result['score'],
                    'conviction': _etf_conv,
                    'emoji':     _etf_score_result['emoji'],
                }
            else:
                # ── Stock path (original) ─────────────────────────────────────
                _pre_scorecard_md = self._build_scorecard_md(
                    target, real_price, analyst_target, fund, summary, dc_data, forward_pe,
                    fg_data=fg_data, onchain=_onchain_data, effective_beta=_effective_beta,
                    display_target=_scorecard_target, target_is_estimate=_target_is_estimate,
                    target_is_sma=(_sma_tech_target is not None and _display_target is None),
                    analyst_consensus=analyst_consensus,
                    change_pct=change_pct
                )
                import re as _re_hint
                # Extract verdict from scorecard markdown: "MSFT | **HOLD 🟡** | Conviction: **Low**"
                _vh_m = _re_hint.search(r'\|\s*\*\*([A-Z]+)\s*([^\*]*)\*\*\s*\|\s*Conviction:\s*\*\*([^\*]+)\*\*', _pre_scorecard_md)
                if _vh_m:
                    _sc_v, _sc_e, _sc_c = _vh_m.group(1).strip(), _vh_m.group(2).strip(), _vh_m.group(3).strip()
                    scorecard_verdict_hint = f'{_sc_v} {_sc_e} (Conviction: {_sc_c})'
                else:
                    # Primary regex failed — try broader extraction before giving up
                    _vh_m2 = _re_hint.search(r'\b(BUY|HOLD|SELL|REDUCE|ACCUMULATE|UNDERWEIGHT|AVOID)\b', _pre_scorecard_md)
                    _vc_m2 = _re_hint.search(r'Conviction[\s:*|]+?(High|Medium|Low)', _pre_scorecard_md, _re_hint.IGNORECASE)
                    if _vh_m2:
                        _sc_c2 = _vc_m2.group(1).strip() if _vc_m2 else 'Medium'
                        scorecard_verdict_hint = f'{_vh_m2.group(1)} (Conviction: {_sc_c2})'
                        logger.warning(f"[ScorecardHint] Primary regex failed for {target}, broad fallback: {scorecard_verdict_hint}")
                    else:
                        scorecard_verdict_hint = None
                        logger.warning(f"[ScorecardHint] Could not extract verdict for {target} — pre-verdict omitted from prompt")
                logger.info(f"[ScorecardHint] {target}: verdict={scorecard_verdict_hint}")
        except Exception as _sve:
            logger.warning(f"[ScorecardHint] exception for {target}: {_sve}")
            _pre_scorecard_md = ""
            scorecard_verdict_hint = None  # Do not default to HOLD on error

        # ── Read structured decision from scorecard (set by _build_scorecard_md) ──
        # This is the single source of truth — no regex on markdown needed downstream.
        _scorecard_decision = getattr(self, '_last_scorecard_decision', {})

        _scv_parts = (scorecard_verdict_hint or '').split()
        _scorecard_verdict = (_scv_parts[0].upper() if _scv_parts else '') or 'UNKNOWN'

        import re as _re_cv2
        _scorecard_conviction_level = "Medium"  # safe default
        if scorecard_verdict_hint:
            _scv_conv_m = _re_cv2.search(r'Conviction:\s*(High|Medium|Low)', scorecard_verdict_hint, _re_cv2.IGNORECASE)
            if _scv_conv_m:
                _scorecard_conviction_level = _scv_conv_m.group(1).capitalize()

        _trend_state = str((summary or {}).get('trend') or '').strip().lower()
        if _scorecard_verdict in ("BUY", "ACCUMULATE") and _trend_state == "bearish":
            _decision_type = "contrarian_early"
        elif _scorecard_verdict in ("BUY", "ACCUMULATE") and _trend_state == "neutral":
            _decision_type = "early_reversal"
        elif _scorecard_verdict in ("BUY", "ACCUMULATE") and _trend_state == "bullish":
            _decision_type = "trend_confirmed"
        elif _scorecard_verdict == "HOLD":
            _decision_type = "wait_for_confirmation"
        elif _scorecard_verdict in ("REDUCE", "SELL", "UNDERWEIGHT", "AVOID"):
            _decision_type = "trend_failure"
        else:  # UNKNOWN — scorecard could not be computed
            _decision_type = "open"

        _decision_type_label_map = {
            "contrarian_early": "Contrarian Early",
            "early_reversal": "Early Reversal",
            "trend_confirmed": "Trend Confirmed",
            "wait_for_confirmation": "Wait For Confirmation",
            "trend_failure": "Trend Failure",
            "open": "Open — Reason Independently",
        }
        _decision_type_label = _decision_type_label_map.get(
            _decision_type, _decision_type.replace('_', ' ').title()
        )
        _contrarian_section8b_rules = (
            "   - If Decision Type = contrarian_early, Section 8b MUST include these exact fields:\n"
            "     why_now: [timing edge right now]\n"
            "     what_confirms: [specific confirmation trigger]\n"
            "     what_invalidates: [specific invalidation trigger]"
        ) if _decision_type == "contrarian_early" else ""

        # ── 5c. Web Research (EisaX competitive advantage) ─────────────────────
        research_context = ""
        try:
            from datetime import datetime as _dt
            # Search for current market outlook
            logger.debug(f"[EisaX Research] Searching for {target} 2026 outlook...")
            # This would require web_search tool - placeholder for now
            research_context = f"""
RESEARCH CONTEXT ({_dt.now().strftime("%B %Y")}):
- Market analysts project strong tech sector performance in 2026
- AI infrastructure spending remains elevated
- Federal Reserve maintaining accommodative policy
            """
        except Exception as e:
            logger.debug(f"[Research] Web search unavailable: {e}")
        # ── 5c. Market Research (EisaX Competitive Advantage) ─────────────────
        research_summary = ""
        try:
            logger.debug(f"[EisaX Research] Searching for {target} market context...")
            _research_q_map = {
                "GC=F": "gold price 2026 outlook Goldman Sachs forecast",
                "SI=F": "silver price 2026 outlook forecast market",
                "CL=F": "crude oil price 2026 outlook Goldman Sachs OPEC",
                "PL=F": "platinum price 2026 outlook forecast market",
                "PA=F": "palladium price 2026 outlook forecast market",
                "HG=F": "copper price 2026 outlook Goldman Sachs forecast electrification",
                "NG=F": "natural gas price 2026 outlook forecast market",
                "BZ=F": "brent crude oil price 2026 outlook forecast",
            }
            _research_query = _research_q_map.get(
                target.upper(),
                f"{target} stock 2026 analyst forecast Goldman Sachs Morgan Stanley"
            )
            search_result = self._web_search(_research_query)
            
            if search_result.get("success"):
                from datetime import datetime as _dt_rs
                research_summary = f"\n\n=== LIVE MARKET RESEARCH ({_dt_rs.now().strftime('%B %Y')}) ===\n"
                research_summary += "CRITICAL: Cite these sources using format: 'According to [Source Name]...'\n\n"
                for idx, result in enumerate(search_result.get("results", [])[:3], 1):
                    # Extract source name from title or domain
                    title = result['title']
                    link = result.get('link', '')
                    source_name = "market research"
                    if 'goldman' in title.lower() or 'goldman' in link.lower():
                        source_name = "Goldman Sachs"
                    elif 'morgan' in title.lower() or 'morgan' in link.lower():
                        source_name = "Morgan Stanley"
                    elif 'cnbc' in link.lower():
                        source_name = "CNBC"
                    elif 'fool' in link.lower():
                        source_name = "The Motley Fool"
                    
                    research_summary += f"{idx}. [{source_name}] {title}\n"
                    research_summary += f"   {result['snippet']}\n\n"
                logger.info(f"[EisaX Research] ✅ Found {len(search_result.get('results', []))} sources")
                logger.debug(f"[EisaX Research] Summary length: {len(research_summary)} chars")
                logger.debug(f"[EisaX Research] Preview: {research_summary[:200]}...")
            else:
                logger.error(f"[EisaX Research] ❌ Search failed: {search_result.get('error')}")
                research_summary = ""
        except Exception as e:
            logger.error(f"[EisaX Research] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            research_summary = ""
        
        # ── 5d. User Context (personalization) ──────────────────────────────
        _user_ctx_block = ""
        try:
            _user_ctx = mem.get("user_ctx", {}) if isinstance(mem, dict) else {}
            if _user_ctx:
                from core.memory_manager import format_ctx_for_prompt
                _user_ctx_block = format_ctx_for_prompt(_user_ctx, target_ticker=target)
        except Exception as _uce:
            logger.error(f"[EisaX Memory] User context inject failed: {_uce}")
        # ── 5e. Global Macro Intelligence ────────────────────────────────────
        _macro_prompt_block = ""
        try:
            from core.macro_intelligence import get_macro_context as _get_macro_ctx
            _ticker_sector = (fund.get("sector") or dc_data.get("sector") if dc_data else "") or ""
            _macro_ctx = _get_macro_ctx(
                ticker=target,
                sector=_ticker_sector,
                news_headlines=[n.get("title","") for n in (news_links or [])],
            )
            _macro_prompt_block = _macro_ctx.get("prompt_block", "")
            logger.info(f"[Macro] context built — {len(_macro_ctx.get('linkages',[]))} linkages, "
                        f"{len(_macro_ctx.get('macro_news',[]))} headlines")
        except Exception as _mce:
            logger.warning(f"[Macro] context failed (non-fatal): {_mce}")

        # ── 6. DeepSeek CIO Synthesis ─────────────────────────────────────────
        deepseek_reply = ""
        try:
            from dotenv import load_dotenv, find_dotenv as _find_dotenv
            load_dotenv(_find_dotenv(usecwd=True) or "/home/ubuntu/investwise/.env")
            ds_key = os.getenv("DEEPSEEK_API_KEY", "")
            logger.debug(f"[DeepSeek] key found: {bool(ds_key)}, length: {len(ds_key)}")
            if ds_key:
                from datetime import datetime as _dt
                if _etf_meta_early:
                    _peer_comp_instruction = (
                        "   ETF mode: name 2 direct alternative funds. Compare expense ratio, yield/return, and AUM. "
                        "Format: \"vs [FUND]: [difference]. [why an investor would choose this one over it].\""
                    )
                elif _is_us_stock_for_peers and _us_peers:
                    _peer_comp_instruction = (
                        "   Stock mode (US): Use the US PEER COMPARISON TABLE from the data to build a full peer table in Section 6. "
                        "Show Fwd P/E, Market Cap, Div Yield, Rev Growth, Gross Margin for each peer. "
                        "State the premium/discount vs each. DO NOT write only 2 sentences — use the full table.\n"
                        "   ⛔ DATA LOCK: Copy Div Yield values EXACTLY as shown in the peer table above (e.g. '2.03%'). "
                        "NEVER recalculate or use training knowledge for yields — the table is Python-computed and authoritative. "
                        "If a yield looks unusual, copy it anyway and add a footnote, do NOT replace it."
                    )
                else:
                    _peer_comp_instruction = (
                        "   Stock mode — compare to the single closest DIRECT competitor in the same sub-industry:\n"
                        "   • Sentence 1 (Valuation): state both forward P/E (or EV/EBITDA, P/S for growth) values and the % premium or discount.\n"
                        "   • Sentence 2 (Edge): where does this company lead or lag vs the peer? (growth rate, margin, market share, moat, product pipeline)\n"
                        "   Format: \"vs [PEER_TICKER]: [valuation sentence]. [competitive position sentence].\"\n"
                        "   Example: \"vs NVDA: AMD trades at 24x fwd P/E vs NVDA's 35x — a 31% discount. AMD leads in CPU market share but lags NVDA's data center GPU dominance (NVDA holds ~80% market share vs AMD ~15%).\"\n"
                        "   ⛔ Do NOT write more than 2 sentences. ⛔ Do NOT include any rating or recommendation.\n"
                        "   ✅ If peer valuation inputs are missing, explicitly state 'N/A' or 'data coverage is partial'. Never invent peer numbers.\n"
                        "   ⛔ If you truly cannot compare, name the peer and compare qualitatively (margins, growth, market share).\n"
                        "   ⚡ PEER SELECTION: Choose the MOST RELEVANT competitor — for cloud/software companies this may be AMZN (AWS) or META, not necessarily GOOGL. For UAE/Saudi companies compare to the closest regional peer."
                    )

                # ── Data mode block: compact report when fundamentals are limited ────────
                _data_mode_block = ""
                if '_data_coverage_level' in dir() and _data_coverage_level in ("technical_only", "low"):
                    _data_mode_block = (
                        "\n🔴 DATA COVERAGE ALERT: FUNDAMENTAL DATA LIMITED\n"
                        "⛔ COMPACT REPORT MODE — MANDATORY:\n"
                        "- Executive Summary MUST open with: \"⚠️ Fundamental data coverage is limited — this analysis relies primarily on price behavior.\"\n"
                        "- Section 2 (Fundamental Analysis): Write 2-3 sentences MAX. State which metrics ARE available. Then: \"Fundamental visibility is limited; analysis relies primarily on price behavior.\"\n"
                        "- Section 5: Write \"Analyst consensus and valuation scenarios are disabled in low-data mode.\" Do not create valuation tables.\n"
                        "- Section 6 (Peer Comparison): Write \"Peer comparison is disabled in low-data mode because fundamental coverage is limited.\" Do not create peer tables.\n"
                        "- Section 9: Write one concise scenario-sensitivity sentence only. Do not create scenario tables.\n"
                        "- This overrides any later instruction asking for valuation ranges, peer comparison, or scenario tables.\n"
                        "- Avoid strong BUY/SELL wording; describe technical moves as positive or negative momentum that requires confirmation.\n"
                        "- Total memo body: maximum 600 words. Be concise.\n"
                    )

                # ── Conviction anchor: cascade Low conviction to all sections ──────────
                _conviction_anchor_block = ""
                _scorecard_conviction_level_safe = locals().get("_scorecard_conviction_level", "Medium")
                # Parse score from _pre_scorecard_md (already built) — _eisax_score is
                # only defined AFTER this try block, so we cannot reference it here.
                _eisax_score_int = 0
                try:
                    import re as _re_sc_early
                    _esc_m = _re_sc_early.search(r'EisaX Score:\s*\*\*(\d+)/100\*\*', _pre_scorecard_md)
                    if _esc_m:
                        _eisax_score_int = int(_esc_m.group(1))
                except Exception:
                    pass
                if _scorecard_conviction_level_safe == "Low" or _eisax_score_int < 60:
                    _conviction_anchor_block = (
                        f"\n⚠️ LOW CONVICTION SIGNAL: EisaX Score={_eisax_score_int}/100, Conviction={_scorecard_conviction_level_safe}\n"
                        "This MUST cascade through the entire memo:\n"
                        "- Every section: use hedged language (\"suggests\", \"may indicate\", \"limited evidence for\") rather than confident assertions.\n"
                        "- Avoid specific price targets — use ranges with explicit uncertainty (e.g., \"estimated range 20–25, low confidence\").\n"
                        "- Section 8b Conviction: MUST be Low for both Fundamental and Timing dimensions.\n"
                        "- Do NOT write a high-confidence Executive Summary when conviction is Low.\n"
                    )

                # ── Verdict tone lock: prevent tone contradictions ──────────────────────
                _verdict_tone_lock = ""
                _locked_verdict = locals().get("_scorecard_verdict", "")
                if _locked_verdict in ("REDUCE", "SELL", "AVOID"):
                    _verdict_tone_lock = (
                        f"\n🔒 VERDICT TONE LOCK: {_locked_verdict}\n"
                        "⛔ BANNED PHRASES across ALL sections (any of these = hard violation):\n"
                        "  \"attractive entry\", \"compelling opportunity\", \"buy the dip\", \"accumulate\", \"add to position\",\n"
                        "  \"strong momentum\", \"poised for upside\", \"bullish setup\", \"upside potential looks significant\"\n"
                        "✅ REQUIRED TONE: Cautious, risk-first framing. Every section must justify the REDUCE/SELL verdict.\n"
                    )
                elif _locked_verdict in ("BUY", "STRONG BUY", "ACCUMULATE"):
                    _verdict_tone_lock = (
                        f"\n🔒 VERDICT TONE LOCK: {_locked_verdict}\n"
                        "⛔ BANNED PHRASES that contradict a BUY verdict:\n"
                        "  \"avoid\", \"high risk\", \"not recommended\", \"too expensive\", \"overvalued\",\n"
                        "  \"limited upside\", \"unattractive\"\n"
                        "(Exception: if citing specific analyst concerns with data, you may mention them — but frame them as risks to monitor, not reasons to avoid.)\n"
                        "✅ REQUIRED TONE: Constructive. Timing caveats (WAIT, ADD ON DIP) belong ONLY in Section 8b Entry Timing — NOT in Executive Summary.\n"
                    )

                prompt = f"""You are EisaX, Chief Investment Officer - built by Eng. Ahmed Eisa.

🚨 CRITICAL: Today's date is {_dt.now().strftime("%B %d, %Y")}.{research_summary}
   - You MUST use this EXACT date in your memo header
   - Any historical data reference must be clearly labeled as "historical"
   - All analysis must reflect current 2026 market conditions
   - MEMO SUBJECT LINE: In the memo header, the "Re:" line MUST use the ticker exactly as the user typed it: **{_original_target if "_original_target" in dir() else target}** — NOT the resolved symbol. E.g. if user typed "XAUUSD", write "Re: Analysis of XAUUSD" not "Re: Analysis of GC=F".

Your advantage over general AI assistants:
- You are a SPECIALIZED financial analyst with 20+ years CIO experience
- You have access to LIVE market data (not training data)
- You provide institutional-grade analysis with specific entry/exit levels

{(f'🎯 SCORECARD PRE-VERDICT (computed before this memo): **{scorecard_verdict_hint}**') if scorecard_verdict_hint else '⚠️ Scorecard pre-verdict unavailable — reason independently from the data below.'}
{_verdict_tone_lock}{_data_mode_block}{_conviction_anchor_block}
DECISION TYPE (deterministic): **{_decision_type} — {_decision_type_label}**

⛔ MANDATORY DECISION STRUCTURE — output this exact block in Section 8b of every report (no exceptions):
  Fundamental Verdict: BUY / HOLD / REDUCE / SELL
  Entry Timing: BUY NOW / WAIT / ADD ON DIP / REDUCE INTO STRENGTH
  Conviction — Fundamental: High / Medium / Low  |  Timing: High / Medium / Low
  Score: [N]/100 — Score reflects business quality, not short-term return potential.

🔴 RULE 8A — FORCED BUY (HARD RULE, NO EXCEPTIONS):
   If Score ≥ 75 AND Upside ≥ 20% → Fundamental Verdict MUST be BUY.
   RSI overbought does NOT override this. ADX weak does NOT override this.
   Weak technicals → set Entry Timing = WAIT. They do NOT change Fundamental Verdict to HOLD.
   When Fundamental = BUY and Timing = WAIT, you MUST include this sentence:
   "BUY conditions met, but entry delayed due to [specific technical reason]."

🔴 RULE 8B — HOLD IS RESTRICTED:
   HOLD for Fundamental Verdict is only valid when ALL THREE are true:
   (1) Score between 60–74, (2) Upside < 20%, (3) Bear case downside < -15%.
   If Score ≥ 75 + Upside ≥ 20% and you write HOLD → this is a rule violation.
   "Tactical HOLD" as a default is banned. Use it only for a named, time-bound reason.

⛔ TONE ALIGNMENT RULE:
Your tone MUST follow Fundamental Verdict — not Entry Timing:
- Fundamental = REDUCE or SELL → Cautious tone. No "compelling entry" or buy language.
- Fundamental = HOLD → Balanced tone. Acknowledge both upside potential and risks equally.
- Fundamental = BUY → Constructive tone. Even when Entry Timing = WAIT, do NOT write a HOLD-toned memo.
- ⛔ NEVER collapse BUY + WAIT into HOLD. They are separate, distinct outputs.
- ⛔ NEVER write a bullish Executive Summary when Fundamental Verdict = REDUCE/SELL.

🔴 LANGUAGE QUALITY RULES:
- ⛔ NEVER use boilerplate phrases like "according to recent analyst data", "market observers note", "analysts suggest", or "industry experts believe" — these are empty filler. Use the ACTUAL data provided or state explicitly that data coverage is partial.
- ⛔ NEVER cite a news source that is NOT in the LATEST NEWS section of the data below. Do NOT reference "The Times of India", "Hindustan Times", regional newspapers, blogs, or any outlet from your training knowledge. If you cite a source, it MUST appear verbatim in the LATEST NEWS section.
- ⛔ NEVER invent or paraphrase headlines not present in the LATEST NEWS data. If no relevant news exists, say "No relevant headlines at time of analysis."
- ⛔ BE CONSISTENT on valuation: if the Scorecard labels Forward P/E as "🟢 Reasonable", do NOT describe the same P/E as "elevated" in the memo body. Use the same label throughout.
- ⛔ EARNINGS DATE: Use ONLY the exact date from the data. NEVER combine a fiscal quarter label from one year with a date from another year (e.g. "Q1 2027 on April 29, 2026" is wrong). If unsure of the fiscal quarter label, just say "next earnings report on [date]".
- ✅ Peer comparisons in Section 6 should include actual numbers WHEN available in the provided data. If unavailable, state "N/A" explicitly.
- ✅ If EPS growth estimate is available in the data, include the YoY % in Section 2.

Analyze the following data and write an institutional-grade investment memorandum.
{_user_ctx_block}
{data_block}
{_interpretation_block}
{_approved_phrase_block}

INTERPRETATION CONTROL RULES:
- Any sentence making a technical, timing, support/resistance, volume, or yield claim must remain consistent with the locked interpretation block.
- Executive Summary must ground strength, main risk, and timing posture in the locked phrases.
- Technical Outlook must use the locked trend, RSI, support/resistance, and volume labels without reinterpretation.
- Why Now must use the locked entry-quality and trend-confirmation language.
- Portfolio Role must use the locked yield description and tactical versus strategic framing.
- If the locked labels are cautious, your wording must remain cautious.
- ⚠️ RULE 8A EXCEPTION (takes precedence over the line above): When the SCORECARD PRE-VERDICT = BUY, cautious technical labels apply ONLY to the Entry Timing description in Section 8b. The Executive Summary, thesis, and overall memo tone MUST be constructive (BUY-aligned). Do NOT write a cautious or HOLD-toned Executive Summary for a BUY verdict. Entry timing caveats (RSI overbought, ADX weak) belong in Section 8b — not in the opening thesis.

{(f"""
⚠️ ETF ANALYSIS MODE — {_etf_meta_early['etf_label'] if _etf_meta_early else ''}
This is an ETF, NOT a stock. Follow ETF-specific rules:
- Section 2 = "{"Commodity Analysis" if _etf_meta_early and _etf_meta_early.get("etf_type","").startswith("commodity") else "Fund Analysis"}" (NOT Fundamental Analysis): Discuss what the fund/contract tracks, expense ratio cost drag, AUM liquidity, and how the underlying asset/index is valued. NO EPS, Revenue, ROE, ROIC, or corporate metrics.
- Section 5 = "Market Catalysts": No analyst consensus. Discuss macro catalysts that drive this fund (rate moves, commodity shifts, sector rotation, etc.).
- Section 6 = "⚔️ Peer Comparison": name 2 direct alternative funds (by ticker). Compare expense ratio, yield/return profile, and AUM in exactly 2 sentences. No corporate competitors — funds only.
- Section 7 = "EisaX Outlook": Compare to ALTERNATIVE investments (e.g., for GLD: compare to TLT, T-bills, TIPS; for TLT: compare to HYG, cash, SPY). Include one specific number and one risk/reward statement.
- Section 9 = Use the ETF-SPECIFIC scenario table provided in the data.
- Do NOT mention P/E ratio, EPS, Revenue, ROE, ROIC, analyst price targets, or earnings dates.
""") if _etf_meta_early else ""}
Structure your response with these sections (ALL sections are MANDATORY — do NOT skip any):
CONSISTENCY RULES (MANDATORY):
- ALL 9 sections below MUST appear in order (1 → 9). Missing any section is a hard failure.
- If any required metric/field is unavailable after using provided data, write **N/A** explicitly.
- Never fabricate or estimate missing values from prior knowledge; use **N/A** + brief data limitation note.
 - Keep all numbers internally consistent across sections (same price, beta, target logic, dates).
 - If section content is partially unavailable, keep the section and mark unavailable rows/items as **N/A**.
 - ⛔ CONSISTENCY RULE: Every report must have identical section structure. Missing data = show the section with 'Data unavailable: [specific field]'. Never omit a section. Never vary structure between tickers.
**ANALYTICAL WEIGHTING — Asset-Specific Factor Priorities:**
{"For this CRYPTO asset: weight LIQUIDITY & ON-CHAIN signals at 50%, technical momentum at 35%, macro/sentiment at 15%. Fundamental metrics (P/E, EPS) are not applicable. Lead your analysis with on-chain context, exchange flows, and macro liquidity cycle." if _is_crypto_asset else
 "For this ETF/COMMODITY: weight MACRO REGIME & UNDERLYING ASSET at 55%, technical trend at 30%, fund structure at 15%. Lead with macro drivers, not equity fundamentals." if _etf_meta_early else
 "For this EQUITY: weight FUNDAMENTALS (valuation, growth, profitability) at 55%, technical trend at 30%, macro/sentiment at 15%. Lead with valuation anchor and earnings quality."}
1. **Executive Summary** (3-4 sentences):
   - Sentence 1 — **THESIS KILL SHOT** *(mandatory)*: Write ONE sharp sentence (≤18 words) that captures the core market narrative — what the market is currently mis-pricing or correctly pricing. This is NOT a valuation recitation — it is a conviction statement. Examples: "The market is repricing AI growth premium as rate risk resurfaces." | "Near-peak oil cycle leaves limited upside despite dominant franchise." | "Cyclical re-rating risk outweighs near-term earnings strength." Think: what is the single most important thing an informed investor must know right now?
   - Sentence 2: Biggest strength AND biggest risk in ONE sentence, each with a specific number
   - Sentence 3: The verdict — and if it's a TACTICAL verdict (timing-based) vs FUNDAMENTAL verdict, say so explicitly. Example: "Our **Tactical Underweight** reflects negative momentum and geopolitical risk premium — NOT fundamental weakness; the underlying business case remains [compelling/solid]."
   - Optional Sentence 4: Key catalyst to watch for thesis change
2. **{"Commodity Analysis" if _etf_meta_early and _etf_meta_early.get("etf_type","").startswith("commodity") else "Fund Analysis" if _etf_meta_early else "Fundamental Analysis"}** ({"macro drivers, real yield sensitivity, USD relationship, central bank demand, and supply/demand dynamics for the underlying commodity. Do NOT use ETF/fund language — this is a commodity futures contract." if _etf_meta_early and _etf_meta_early.get("etf_type","").startswith("commodity") else "what the fund tracks, expense ratio drag, AUM size, macro drivers of the underlying asset" if _etf_meta_early else "growth quality, profitability, valuation - mention Forward P/E and Gross Margin GAAP note"})
3. **Technical Outlook** (MANDATORY — you MUST include ALL of the following from the TECHNICALS data):
   - SMA50, SMA200, RSI, MACD, ADX values with trend direction and momentum condition
   - CRITICAL: use the exact RSI condition label from data — e.g. "RSI: 32.2 (Near Oversold)" not your own label
   - Volume vs average: state if volume is LOW/NORMAL/HIGH vs 90-day avg and what this means for conviction
   - Technical S/R Ladder: reproduce the deterministic TECHNICAL LEVELS TABLE exactly (R3->R1, Spot, S1->S3) and explain the nearest R1/S1 in one sentence
   - ⚠️ Technical Note: Momentum indicators (MACD/RSI) reflect price-driven buying pressure, while ADX measures trend strength independently of direction. A bullish momentum reading alongside a weak ADX (< 25) indicates early-stage or range-bound price action — not a confirmed trend. Treat momentum signals with reduced confidence until ADX sustains above 25.
   - ⛔ Do NOT repeat these technical facts in Section 8 (Why Now) — Section 8 focuses on TIMING and CATALYSTS only
4. **Key Risks** (top 2-3 BUSINESS risks with severity rating):
   ⛔ DATA GAPS ARE NOT RISKS: If fundamental metrics (ROE, ROIC, Net Margin, etc.) are unavailable, note this ONCE in Section 2 as a data limitation. Do NOT list "Weak Fundamental Metrics" or "Data Unavailability" as a Key Risk in Section 4.
   ✅ Section 4 must contain only genuine business, macro, commodity, regulatory, or market risks.
   **INSTITUTIONAL LABEL RULE (MANDATORY):** Every risk MUST begin with its **Institutional Concept Label** in bold — a precise 2-4 word term used by CFA/buy-side analysts. Follow immediately with (Severity: X). Examples of correct labels:
   - **Cyclical Commodity Exposure** (Severity: High)
   - **AI Growth Multiple Compression** (Severity: High)
   - **Geopolitical Risk Premium** (Severity: Medium-High)
   - **Rate Sensitivity / Duration Risk** (Severity: Medium)
   - **Regulatory Overhang** (Severity: Medium)
   - **Execution & Competitive Moat Risk** (Severity: Medium)
   - **Sovereign Concentration Risk** (Severity: High)
   - **Liquidity Discount** (Severity: Low)
   Do NOT describe the risk with a vague phrase — name it with the institutional concept label FIRST, then explain.
   MANDATORY: If LATEST NEWS appears in the data, reference at least one relevant headline here as a named risk.
   ⛔ Do NOT name specific countries in conflict scenarios — use region-level framing: "Middle East tensions", "Gulf security risk", "Geopolitical Risk Premium".
5. **Analyst Consensus & Catalysts**
   - State the consensus target or EisaX FV estimate with % upside, upcoming earnings catalyst
   - **MANDATORY: Include a Valuation Range table** with 3 scenarios:
     | Scenario | Multiple | Implied Price | Upside/Downside |
     |----------|----------|---------------|-----------------|
     | 🐻 Bear | [sector floor]x | [price] | [%] |
     | ⚖️ Base | [normalized fair multiple]x | [price] | [%] |
     | 🚀 Bull | [sector ceiling]x | [price] | [%] |
     ⛔ CRITICAL: The Base Case MUST use the SECTOR AVERAGE P/E (not the current stock P/E) — a stock trading at a DISCOUNT to sector average means Base Case > current price. If current TTM P/E = 3.8x but sector average = 7x, the Base multiple = 7x and Base implied price > current price.
     ⛔ ALL numerical values are pre-computed in the PRE-COMPUTED VALUES section above. Copy them exactly. NEVER recalculate. NEVER show N/A for any value that appears in PRE-COMPUTED VALUES — those are guaranteed computed by Python. N/A only appears when PRE-COMPUTED VALUES itself shows N/A, which means source data was genuinely absent.
     Sector floor = sector avg P/E × 0.70; ceiling = sector avg P/E × 1.40.
     If no EPS/PB/NAV inputs are available, keep valuation rows as "N/A" and state data limitation clearly.
6. **⚔️ Peer Comparison** (MANDATORY — do NOT skip):
{_peer_comp_instruction}
"\n⛔ BANNED PHRASES — never write these regardless of verdict:\n"
"- \"bullish trends are expected in 2026\" or any variation\n"
"- \"diversification is recommended, aligning with our balanced view\"\n"
"- Any forward-looking phrase not directly supported by the data block above.\n"

7. **EisaX Outlook** — Write 3 sub-sections:

   **a) Return Outlook (2 sentences):**
   - One specific number (implied return %, EV/EBITDA vs peers, FCF yield, or total return = upside + dividend)
   - One clear risk/reward statement
   - ⛔ DO NOT include any verdict, buy/sell/hold rating, or recommendation

   **b) 💼 Portfolio Role** — 3 bullet points explaining WHY this stock belongs in a portfolio:
   - What TYPE of exposure it provides (e.g. "Value exposure at deep discount", "High income via X% yield", "Regional real estate beta", "AI infrastructure play")
   - What PORTFOLIO FUNCTION it serves (e.g. "Defensive income anchor", "Cyclical recovery play", "Diversifier vs US tech")
   - What INVESTOR PROFILE it suits (e.g. "Income-focused long-term investor", "Contrarian value investor", "GCC equity allocator")

   **c) 🔗 Correlation Context** — 3 bullet points (be specific with correlation direction):
   - Correlation to GCC/regional equities (High/Medium/Low + direction explanation)
   - Correlation to oil price or key macro driver (if energy/materials) OR to global rates (if real estate/financials)
   - Correlation to US/global equities in risk-off environments (does it decouple or sell off in tandem?)
   - ⛔ DO NOT write any score or scorecard

8. **⏰ Why Now?** (MANDATORY — focus on TIMING and CATALYSTS, not technical analysis which belongs in Section 3):

   **a) Timing Signals (bullet format):**
   • Market Sentiment: Fear & Greed at {fg_data.get('score','N/A')} ({fg_data.get('rating','N/A')}) — what extreme reading means for entry timing RIGHT NOW
   • Upcoming Catalyst: next earnings date, product launch, regulatory event, or sector-specific driver — cite LATEST NEWS if relevant; explain WHY this catalyst matters NOW
   • Risk/Timing: one specific risk to the entry timing (NOT a repeat of Section 4 risks — frame it as timing risk)
   {"• Oil Price: Brent at $" + str(round(_oil_data.get('price',0),2)) + "/bbl — impact on revenue and margins" if _is_energy else ""}
   {("• 📱 X Sentiment: sentiment is **" + str(_x_data.get('sentiment','')) + "** (score: " + f"{_x_data.get('score',0):+.2f}" + "). Key themes: " + ", ".join(_x_data.get('themes',[])[:2]) + ".") if _x_data and _x_data.get("sentiment") else ""}

   **b) 📋 Verdict Clarification** (MANDATORY when verdict is REDUCE/Underweight/HOLD despite strong fundamentals):
   - If the stock has strong fundamentals (P/E < 10x, upside > 20%, or yield > 5%) BUT the verdict is cautious, you MUST explicitly state:
     "**Verdict Type: Tactical [Underweight/Reduce]** — based on timing/momentum, NOT fundamental weakness."
     Then explain: "Fundamental case is [strong/compelling] — the underweight reflects [specific timing risk, e.g. bearish trend, geopolitical premium, pending catalyst]."
   - If verdict is BUY/Strong Buy and decision type is not contrarian_early: confirm it is both fundamentally AND technically supported.
   {_contrarian_section8b_rules}
   - Add one explicit line: **Primary uncertainty:** [2 concrete uncertainty drivers].
   - Add one explicit line: **No-Action Case:** when HOLD/no trade is preferable.

   **c) 📋 Entry Considerations (if investor chooses to act)**
   The following are data-driven observations, not investment instructions.
   - Stage 1: Describe the market condition or price zone where risk/reward improves
   - Stage 2: Describe what confirmation signal would indicate stronger trend reliability
   - Stage 3: Describe what thesis-validation event would improve allocation confidence
   - Full position sizing: refer to EisaX Score → Core Allocation range from the Scorecard
   ⛔ TONE RULES: Use ONLY observational language grounded in data and thresholds.
   ⛔ FORBIDDEN phrases: "Do not chase", "Initiate", "Execute", "Must", "Immediately reduce", "You should", "We recommend".
   ⛔ DO NOT repeat Entry/Stop/Target price levels (those are in the auto-generated Positioning Guide below)

   **d) ⚠️ Risk Action Plan** — Scenario-linked risk observations when risks materialize:
   The following are data-driven observations, not investment instructions.
   - Observation 1: "If [specific measurable trigger], then [expected impact on thesis/risk profile]"
   - Observation 2: "If [geopolitical/macro trigger], then [expected portfolio risk asymmetry]"
   - Observation 3: "If [thesis validation trigger], then [evidence that bull case is strengthening]"
   Format: "• [Trigger]: [Observation]"
   - Phrase all statements as conditional observations, not execution commands.
   **e) ❓ What Would Make Me Wrong?** — Thesis Invalidation Conditions (MANDATORY — 2 specific triggers):
   State the 2 most concrete conditions that would INVALIDATE the primary thesis. Each must be specific and measurable — not generic.
   - Format: "If [specific measurable event or price level], the [bull/bear] thesis breaks because [reason]."
   - Example BUY thesis break: "If RSI drops below 30 AND price closes below SMA200 for 3 consecutive days, the bullish thesis breaks — momentum failure signals distribution, not accumulation."
   - Example SELL thesis break: "If EPS growth exceeds 25% QoQ AND ADX rises above 30, the bearish thesis breaks — fundamental reacceleration with trend confirmation."
   ⛔ Do NOT use vague phrases like "market deterioration" or "unexpected news" — name the specific trigger.

9. **🌍 Advanced Scenario Analysis**
   {"Include the Oil Price Sensitivity table AND the Energy-Sector scenario table from the data. Show how different oil prices ($50-$90/bbl) affect this stock." if _is_energy else "Include a markdown table of 4 beta-adjusted scenarios from the SCENARIO ANALYSIS section in the data. REQUIREMENT: At least 2 scenarios must be BULLISH (upside cases) and at least 1 must be BEARISH. Do NOT generate all-bearish or all-downside scenarios — this is for institutional investors who need balanced upside and downside analysis."}
   Format:
   Emoji rule: 🚀📈💡 for BULLISH rows · 📉🏦🤖⚠️ for BEARISH rows. NEVER use 📉 on a positive-impact row.
   ⛔ The SCENARIO ANALYSIS data already has exactly 4 columns: Scenario | Impact | Implied Price | Suggested Hedge. Copy this table EXACTLY — do NOT add a Market Move column or split any cell. Use "Expected Price" as the header for the price column.
   ADD a 5th column: **Trigger** — one specific, measurable event that would activate this scenario (e.g. "Brent breaks $80", "Price closes below SMA200", "Fed hikes +50bps"). This must be a concrete observable condition, NOT a vague description.
   ⛔ PRECISION RULE: Expected Price values MUST be rounded ranges, NOT exact decimals. Write "~24.5–25.5 SAR" not "24.96 SAR". Exact decimal prices create false precision and mislead investors. Round to nearest 0.5 or whole number and use a ±5% range format.
   | Scenario | Impact | Expected Price | Trigger | Suggested Hedge |
   |----------|--------|----------------|---------|-----------------|

{"10. **🛢️ Oil Price Sensitivity** (MANDATORY for energy stocks): Include the full Oil Price Sensitivity table from the data showing revenue impact at $50, $60, $70, $80, $90/bbl. Discuss the breakeven oil price and OPEC+ production outlook." if _is_energy else ""}

Use actual numbers. Be specific. Institutional tone.
{"CRITICAL: This is an ENERGY sector stock. Oil prices are the PRIMARY driver. You MUST discuss oil price impact throughout the report, include the sensitivity table, and reference Brent crude at $" + str(round(_oil_data.get('price',0),2)) + "/bbl." if _is_energy else ""}
{"CURRENCY: Use " + _currency_sym + " (" + _currency_lbl + ") for ALL price references — NOT USD." if _currency_lbl != "USD" else ""}
{"LANGUAGE: The user's request was in Arabic. Write the FULL report in Arabic. IMPORTANT: Use the SAME number of sections, SAME level of detail, and ALL 9 sections — do NOT simplify or shorten because it is in Arabic. Arabic and English reports must be identical in depth and structure. Section 6 (Peer Comparison) must still be exactly 2 sentences with competitor ticker and valuation numbers in Arabic. USE THESE EXACT ARABIC SECTION HEADINGS — no variations: ### 1. الملخص التنفيذي | ### 2. أطروحة الاستثمار | ### 3. التحليل الفني | ### 4. إشارات المخاطر | ### 5. التقييم والسعر المستهدف | ### 6. المقارنة مع الأقران | ### 7. القرار والتوقيت | ### 8. ما الذي يغيّر هذا القرار | ### 9. ما الذي قد يثبت خطأ هذا الرأي. Verdict labels in Arabic: شراء / احتفاظ / تخفيف / بيع. Timing labels: شراء الآن / انتظر تأكيدًا / شراء تدريجي عند التراجع / خفّف مع الارتفاع." if _is_arabic_request else "LANGUAGE: Write in English."}
{"🚨 EXTREME PRICE MOVE ALERT — " + _crash_direction + " (" + f"{change_pct:+.2f}%" + " single-day move detected): This MUST be the FIRST thing addressed in Section 1 (Executive Summary). In Section 4 (Key Risks), you MUST investigate and explain the likely cause: check if this is an ex-dividend drop, rights issue (capital increase), trading halt lifted, forced selling, major news event, or circuit-breaker trigger. State the most probable cause based on available data. Do NOT treat this as a normal trading day — this is an exceptional event requiring forensic analysis." if _is_crash else ""}
IMPORTANT RULES:
- Do NOT mention dividend yield unless above 0.5%
- Entry zone must ALWAYS be BELOW the current live price
- Stop loss: one consistent value only
- Analyst count: use the EXACT number from the data. Do NOT round or cap it.
- ⛔ EARNINGS DATE RULE: The NEXT EARNINGS DATE in the data is the ONLY date to use. Do NOT derive or guess fiscal quarter labels (Q1/Q2/Q3/Q4) from calendar dates — the fiscal year varies by company. Use the date as-is (e.g. "April 29, 2026") and say "next earnings" not "Q1 FY2027".
- ⛔ NEVER write "Score: XX/100" in sections 1-8. That appears ONLY in the Scorecard.
- ⛔ DO NOT create any scorecard table, score breakdown, scoring methodology, or positioning section in your response. NO "Growth: X/30", "Valuation: X/20", "Score: XX/100", "Confidence Score", "Entry Zone", "Stop Loss", "Target" sections. The EisaX Proprietary Scorecard AND Positioning Guide are automatically appended below your memo — ANY duplication causes critical display errors and will be rejected.
- ⛔ Your response MUST end after section 9. Do NOT add any additional sections, tables, or blocks after section 9.
- ALL 9 sections above are MANDATORY. Do NOT skip Technical Outlook, Why Now, or Advanced Scenario Analysis.
- ⛔ NEWS INTEGRATION RULE: If FRESH NEWS CONTEXT is provided in the data, you MUST cite at least 1 specific headline by name in Section 1 (Executive Summary) AND at least 1 in Section 4 (Key Risks). Do NOT generically mention "recent news" — quote or paraphrase the actual headline title. Failing to integrate news is a critical quality failure.
- ⛔ SCENARIO TABLE RULE: Section 9 table MUST have 5 columns: Scenario | Impact | Expected Price | Trigger | Suggested Hedge. The Trigger column is MANDATORY — each row must have a specific measurable trigger (e.g. "Brent breaks $80/bbl", "Price closes below SMA200", "Fed hikes +50bps", "Foreign outflows accelerate"). If you omit the Trigger column your response will be rejected.
- ⛔ CONSISTENCY RULE: Section 8 (Why Now) must be CONSISTENT with the Scorecard verdict. If the verdict is REDUCE or SELL, do NOT frame the analysis as a "contrarian opportunity" or suggest it is a good entry point. Instead, explain what would need to change for the thesis to improve. If the verdict is HOLD/BUY, you may describe constructive entry timing.
- ⛔ UPSIDE LANGUAGE RULE: Only use "strong upside" when upside potential is genuinely >20%. For <10% upside use "modest upside" or "limited upside". For 10-20% upside use "moderate upside". Never call +3% to +5% returns "strong upside" — that misleads investors.
- ⛔ ADVISORY LANGUAGE RULE: Avoid hard-command phrasing ("buy now", "exit 100%", "must do"). Use probability-aware language ("data indicates", "preferred setup", "if/then risk case").
- ⛔ CORRELATION RULE: Never state a specific correlation coefficient (e.g. ">0.8", "0.85 correlation") unless it is explicitly provided in the data. Use qualitative language instead: "High positive correlation (historically strong relationship)", "Moderate positive correlation", "Low correlation". Stating unverified correlation numbers damages credibility.
- ⛔ STRICT DATA INTEGRITY RULE: Never fabricate numeric values. If a metric is missing after using provided data, output 'N/A' and mention the data gap briefly.
Do NOT include a standalone Positioning section.{_brain_ctx}
{_macro_prompt_block}
"""

                # Replace placeholders with pre-calculated values
                prompt = prompt.replace("PLACEHOLDER_ENTRY", pre_entry)
                prompt = prompt.replace("PLACEHOLDER_TARGET", pre_target)
                prompt = prompt.replace("PLACEHOLDER_STOP", pre_stop)
                prompt += "\n\n🚨 MANDATORY: Entry=" + pre_entry + " | Stop=" + pre_stop + " | Target=" + pre_target + " — USE THESE EXACT LEVELS."
                if research_context:
                    prompt += "\n\n" + research_context
                
                # Add Local Market Data
                prompt += _local_data_injection

                # ── Mode-based prompt adjustment ──────────────────────────────
                if _analysis_mode == "quick":
                    _max_tokens = 1500
                    _mode_instruction = """
🎯 QUICK MODE: Write a condensed analysis with ONLY these 3 sections:
1. Executive Summary (4 sentences max)
2. Key Verdict + Scorecard (2 sentences: what the score means + why)
3. Entry/Risk levels (bullet points only: Entry zone, Stop, Target, 1 key risk)

Skip sections 3,4,5,6,8,9. Total response: max 400 words. Be direct and actionable.
"""
                elif _analysis_mode == "cio":
                    _max_tokens = 3000
                    _mode_instruction = """
🎯 CIO MEMO MODE: Write a formal institutional investment memorandum.
- Formal prose only — NO markdown tables, NO bullet lists, NO emojis
- Sections: Executive Summary → Thesis → Risk Assessment → Recommendation
- Tone: Board-room level, measured, cite specific data points
- Length: 600-800 words maximum
"""
                else:
                    _max_tokens = 4500
                    _mode_instruction = ""

                if _low_data_compact_mode:
                    _max_tokens = min(_max_tokens, 1600)

                if _mode_instruction:
                    prompt = _mode_instruction + "\n\n" + prompt

                r = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {ds_key}",
                             "Content-Type": "application/json"},
                    json={"model": "deepseek-chat",
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": _max_tokens,
                          "temperature": 0},
                    timeout=150
                )
                logger.debug(f"[DeepSeek] status: {r.status_code}, response keys: {list(r.json().keys())}")
                resp_json = r.json()
                if "choices" in resp_json:
                    deepseek_reply = resp_json["choices"][0]["message"]["content"].strip()
                    logger.debug(f"[DeepSeek] got reply length: {len(deepseek_reply)}")
                else:
                    logger.debug(f"[DeepSeek] unexpected response: {resp_json}")
                # Force correct date in response (DeepSeek often ignores prompt date)
                from datetime import datetime as _dt
                correct_date = _dt.now().strftime("%B %d, %Y")
                import re
                # Replace any date pattern like "May 7, 2024" or "Date: May 7, 2024"
                import re as _re
                # Fix all date formats: **Date:**, **DATE:**, **date:**
                deepseek_reply = _re.sub(r'\*\*[Dd][Aa][Tt][Ee]:\*\*\s*[^\n]*', '**Date:** ' + correct_date, deepseek_reply)
                # Remove vague boilerplate citations entirely (replace with nothing)
                # These phrases carry zero information: "According to recent analyst data, X"
                # becomes just "X" — same meaning, no fake source.
                deepseek_reply = _re.sub(
                    r'According to (?:\[market research\]|market research|\[.*?\]|recent (?:sector |analyst )?(?:analysis|data|outlook|reports?)|(?:the )?[Mm]arket [Oo]utlook \d{4}|(?:the )?[Ii]ndustry [Aa]nalysts?|(?:the )?[Aa]nalyst [Cc]onsensus|(?:the )?[Mm]arket [Oo]bservers?)'
                    r'(?:,?\s*(?:for \d{4}|in \d{4}|as of [A-Za-z]+ \d{4}|from [A-Za-z]+ \d{4}))?'
                    r',?\s*(?:\([A-Za-z]+\.?\s+\d{4}\))?,?\s*',
                    '',   # ← delete entirely — carries zero information
                    deepseek_reply
                )
                # Also strip any remaining bare [market research] / [source] placeholders inline
                deepseek_reply = _re.sub(r'\[(?:market research|source|data|research|citation needed)[^\]]{0,30}\]', '', deepseek_reply, flags=_re.IGNORECASE)
                # Remove standalone trailing date artifacts like ", (Feb 2026)" or "(Feb 2026)"
                deepseek_reply = _re.sub(
                    r',?\s*\([A-Za-z]{3,9}\.?\s+20\d{2}\)',
                    '',
                    deepseek_reply
                )
                # Fix memo-style DATE: February 1, 2026
                deepseek_reply = _re.sub(
                    r'(\*\*DATE:\*\*\s*)(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2}',
                    r'\g<1>' + correct_date, deepseek_reply
                )
                if deepseek_reply:
                    deepseek_reply = apply_language_locks(
                        deepseek_reply,
                        {
                            "coverage_count": _data_coverage_count,
                            "coverage_level": _data_coverage_level,
                            "low_data_mode": _low_data_compact_mode,
                            "recommendation": _scorecard_verdict,
                            "final_action": (
                                "WAIT / NO ACTION"
                                if _low_data_compact_mode and _scorecard_verdict == "HOLD"
                                else "WATCHLIST / WAIT FOR ENTRY"
                                if _low_data_compact_mode and _scorecard_verdict == "BUY"
                                else _scorecard_verdict
                            ),
                        },
                    )
        except Exception as e:
            import traceback
            logger.error(f"[Analytics] DeepSeek failed: {e}")
            traceback.print_exc()

        # ── 7. Scorecard already computed above (pre-DeepSeek) — extract score ───
        # _pre_scorecard_md was built before the DeepSeek prompt to get verdict hint.
        # Reuse it here — no second computation needed.
        import re as _re_sc
        _eisax_score_match = _re_sc.search(r'EisaX Score:\s*\*\*(\d+)/100\*\*', _pre_scorecard_md)
        _eisax_score = _eisax_score_match.group(1) if _eisax_score_match else 'N/A'

        _exch_label = (
            "🇸🇦 Tadawul · SAR" if _t_upper.endswith(".SR") else
            "🇦🇪 ADX/DFM · AED" if _t_upper.endswith((".AE", ".DU")) else
            "🇪🇬 EGX · EGP" if _t_upper.endswith(".CA") else
            "🇰🇼 Boursa Kuwait · KWF" if _t_upper.endswith(".KW") else
            "🇶🇦 Qatar Exchange · QAR" if _t_upper.endswith(".QA") else ""
        )
        _oil_badge = f" | **🛢️ Brent: ${_oil_data.get('price',0):.2f}**" if _is_energy and _oil_data.get('price') else ""
        _display_ticker = (_original_target if "_original_target" in dir() and _original_target != target else target)
        _price_header_label = "Cached Price" if (_report_snapshot and _report_snapshot.is_cached("price")) else "Live Price"
        header = (
            f"# EisaX Intelligence Report: {_display_ticker}\n\n"
            f"**🔴 {_price_header_label}:** {price_str} | "
            f"**Sector:** {fund.get('sector', 'N/A')} | "
            f"**EisaX Score:** {_eisax_score}/100"
            + (f" | **{_exch_label}**" if _exch_label else "")
            + _oil_badge
            + "\n\n---\n\n"
        )

        # ── ASCII Price Chart ──────────────────────────────────────────────────
        def _ascii_chart(series, width=40, height=6):
            try:
                s = series.dropna().tail(60).values
                mn, mx = s.min(), s.max()
                if mx == mn: return ""
                rows = []
                for h in range(height, 0, -1):
                    row = ""
                    threshold = mn + (mx - mn) * h / height
                    for v in s[::max(1, len(s)//width)]:
                        row += "█" if v >= threshold else "░"
                    label = f"${mn+(mx-mn)*h/height:.0f}" if h in [height, height//2, 1] else "    "
                    rows.append(f"{label:>8} |{row}")
                rows.append(f"{'':>8} └{'─'*len(rows[0].split('|')[1])}")
                rows.append(f"{'':>8}  60 days ago {'→':>{len(rows[0].split('|')[1])-14}} Today")
                return "\n".join(rows)
            except Exception as _e:
                return ""

        chart_str = _ascii_chart(series) if series is not None and len(series) > 10 else ""

        # ── News Links Block — rendered from engine data (3 buckets) + fallback ─
        _eng_direct  = _engine_news_data.get("direct",  []) if _engine_news_data else []
        _eng_sector  = _engine_news_data.get("sector",  []) if _engine_news_data else []
        _eng_country = _engine_news_data.get("country", []) if _engine_news_data else []
        _eng_related = _engine_news_data.get("related", []) if _engine_news_data else []
        _eng_meta    = _engine_news_data.get("meta",    {}) if _engine_news_data else {}

        # ── Dedup + GLM relevance filter (Phase-2 → news_filter service) ──────
        from core.services.news_filter import filter_all_buckets as _filter_all_buckets
        _filtered_buckets = _filter_all_buckets(
            _eng_direct, _eng_sector, _eng_country, _eng_related,
            asset_name  = fund.get('company_name') or target,
            ticker      = target,
            sector_name = fund.get('sector', 'General') or 'General',
            asset_type  = (
                _etf_meta_early.get('etf_type', 'etf') if _etf_meta_early
                else ('crypto' if target.endswith('-USD') else 'stock')
            ),
            etf_meta    = _etf_meta_early,
        )
        _eng_direct  = _filtered_buckets["direct"]
        _eng_sector  = _filtered_buckets["sector"]
        _eng_country = _filtered_buckets["country"]
        _eng_related = _filtered_buckets["related"]

        _has_engine_news = bool(_eng_direct or _eng_sector or _eng_country or _eng_related)

        if _has_engine_news:
            # Rich 3-bucket layout from the news engine
            news_block = "\n\n---\n📰 **Latest News** *(EisaX live news engine)*\n"

            def _sentiment_icon(s: str) -> str:
                return {"bullish": "🟢", "bearish": "🔴"}.get(s, "⚪")

            # Sector relevance keyword sets — must match ≥1 to appear in sector news.
            # IMPORTANT: use multi-word or spaced terms for short words to prevent
            # substring false-positives (e.g. 'gas' in 'madagascar', 'ai' in 'airstrike').
            _SECTOR_KEYWORDS: dict[str, list[str]] = {
                'technology':    [' ai ', ' ai-', 'artificial intelligence', 'machine learning',
                                  'generative', 'tech', 'chip', 'semiconductor', 'cloud',
                                  'software', 'cyber', 'data center', 'microsoft', 'apple',
                                  'google', 'amazon', 'nvidia', 'meta', 'openai', 'copilot',
                                  'azure', 'startup', 'ipo', 'saas', 'chatgpt', 'llm', 'model'],
                'energy':        ['crude oil', 'oil price', 'oil market', 'oil supply',
                                  'oil output', ' oil ', 'brent', 'opec', 'natural gas',
                                  ' lng ', 'petroleum', 'pipeline', 'refin', 'energy sector',
                                  'energy market', 'energy price', 'renewabl', 'solar energy',
                                  'wind power', 'oil company', 'oil producer', 'aramco',
                                  'adnoc', 'exxon', 'chevron', 'bp ', 'shell '],
                'financials':    ['bank', 'fed ', 'federal reserve', 'inflation', 'credit',
                                  ' loan', 'fintech', 'lending', 'interest rate', 'ecb',
                                  'monetary', 'bond yield', 'treasury'],
                'real estate':   ['real estate', 'property', 'reit', 'housing', 'mortgage',
                                  'construction', 'commercial real'],
                'healthcare':    ['health', 'pharma', 'drug ', 'fda', 'clinical trial',
                                  'biotech', 'medical', 'vaccine', 'hospital'],
                'consumer':      ['retail', 'consumer', 'spending', 'e-commerce', 'amazon',
                                  'walmart', 'brand', 'supply chain'],
                'communication': ['media', 'streaming', 'telecom', 'broadband', 'social media',
                                  '5g', 'advertising'],
                'industrials':   ['manufacturing', 'aerospace', 'defense', 'logistic',
                                  'infrastructure', 'supply chain', 'freight'],
                'materials':     ['mining', 'steel', 'copper', 'aluminum', 'chemical',
                                  'commodity', 'lithium'],
                'utilities':     ['utility', 'electric grid', 'power grid', 'water utility',
                                  'natural gas distribution'],
                'crypto':        ['bitcoin', 'btc', 'crypto', 'ethereum', 'blockchain',
                                  'defi', 'web3', 'nft'],
            }
            # Country/region relevance keywords
            _COUNTRY_KEYWORDS: dict[str, list[str]] = {
                'usa':         ['fed', 's&p', 'dow', 'nasdaq', 'wall street', 'trump', 'congress', 'dollar', 'gdp', 'recession', 'inflation', 'us market', 'american'],
                'uae':         ['uae', 'dubai', 'abu dhabi', 'adx', 'dfm', 'difc', 'mena', 'gulf', 'emirati'],
                'saudi':       ['saudi', 'aramco', 'tadawul', 'riyadh', 'vision 2030', 'pif', 'neom'],
                'gcc':         ['gcc', 'gulf', 'opec', 'oil price', 'crude', 'mena', 'middle east market'],
                'global':      ['global market', 'world economy', 'imf', 'world bank', 'trade war', 'tariff', 'g7', 'g20'],
            }

            # Non-financial noise — blocked regardless of sector or country
            _HARD_NOISE = [
                # Military / conflict
                'airstrike', 'air strike', 'military strike', 'troops killed', 'soldiers killed',
                'soldiers wounded', 'bombing', 'mortar attack', 'drone strike',
                'security capabilities assessment', 'launches security', 'security assessment',
                'military exercise', 'naval exercise', 'military operation',
                'peacekeeping', 'coup', 'civil war', 'insurgent',
                # Natural disasters / weather
                'rain alert', 'rainfall alert', 'flood warning', 'flash flood',
                'earthquake', 'tsunami', 'hurricane', 'tornado warning', 'typhoon',
                'uae weather', 'weather alert', 'weather:', 'weather forecast',
                'work remotely on friday', 'employees to work remotely',
                # Social / non-financial government
                'warm moment', 'casual restaurant', 'restaurant visit', 'family visit',
                'president visits', 'royal visit', 'official visit to',
                # Sports / entertainment
                'cricket score', 'football match', 'soccer match', 'olympics', 'world cup',
                'celebrity', 'recipe', 'fashion week', 'movie review', 'tv show',
                'horoscope', 'dating', 'workout tips', 'music album',
                # Non-financial Arabic/regional content
                'يتعافى الإنسان', 'ولحافه', 'بسريره',
            ]

            def _sector_relevant(title: str, sector: str) -> bool:
                """Return True if title is relevant to the given sector."""
                t = title.lower()
                # Hard noise filter first — block non-financial topics regardless of sector
                if any(n in t for n in _HARD_NOISE):
                    return False
                for sec_key, kws in _SECTOR_KEYWORDS.items():
                    if sec_key in sector.lower():
                        return any(kw in t for kw in kws)
                # Unknown sector — accept anything that passed the noise filter
                return True

            def _country_relevant(title: str, country: str) -> bool:
                """Return True if title is relevant to the given country/region."""
                t = title.lower()
                country_l = country.lower()
                # Hard noise filter first
                if any(n in t for n in _HARD_NOISE):
                    return False
                for ckey, kws in _COUNTRY_KEYWORDS.items():
                    if ckey in country_l:
                        return any(kw in t for kw in kws)
                return True

            if _eng_direct:
                _co_label = target.split(".")[0]
                news_block += f"\n**📌 {_co_label} — Company News**\n"
                for _a in _eng_direct[:5]:
                    _ico  = _sentiment_icon(_a.get("sentiment", "neutral"))
                    _src  = f" *({_a['source']})*" if _a.get("source") else ""
                    _url  = _a.get("url", "")
                    _ttl  = _a.get("title", "")
                    news_block += (
                        f"- {_ico} [{_ttl}]({_url}){_src}\n" if _url
                        else f"- {_ico} {_ttl}{_src}\n"
                    )

            if _eng_sector:
                _sec_label = _eng_meta.get("inferred_sector", "Sector")
                _stock_sector = fund.get('sector', '') or _sec_label
                # Filter: only keep articles that are relevant to this sector
                _filtered_sector = [
                    _a for _a in _eng_sector
                    if _sector_relevant(_a.get("title", ""), _stock_sector)
                ]
                if _filtered_sector:
                    news_block += f"\n**🏭 {_sec_label} — Sector News**\n"
                    for _a in _filtered_sector[:3]:
                        _url = _a.get("url", "")
                        _ttl = _a.get("title", "")
                        _src = f" *({_a['source']})*" if _a.get("source") else ""
                        news_block += (
                            f"- [{_ttl}]({_url}){_src}\n" if _url else f"- {_ttl}{_src}\n"
                        )

            if _eng_country:
                _cntry_label = _eng_meta.get("inferred_country", "Region")
                # Filter: only keep articles relevant to this country/region
                _filtered_country = [
                    _a for _a in _eng_country
                    if _country_relevant(_a.get("title", ""), _cntry_label)
                ]
                if _filtered_country:
                    news_block += f"\n**🌍 {_cntry_label} — Market News**\n"
                    for _a in _filtered_country[:3]:
                        _url = _a.get("url", "")
                        _ttl = _a.get("title", "")
                        _src = f" *({_a['source']})*" if _a.get("source") else ""
                        news_block += (
                            f"- [{_ttl}]({_url}){_src}\n" if _url else f"- {_ttl}{_src}\n"
                        )

            # Show related/recent when direct+sector+country are all empty
            if _eng_related and not (_eng_direct or _eng_sector or _eng_country):
                news_block += "\n**📡 Related & Recent News**\n"
                for _a in _eng_related[:5]:
                    _ico  = _sentiment_icon(_a.get("sentiment", "neutral"))
                    _url  = _a.get("url", "")
                    _ttl  = _a.get("title", "")
                    _src  = f" *({_a['source']})*" if _a.get("source") else ""
                    news_block += (
                        f"- {_ico} [{_ttl}]({_url}){_src}\n" if _url
                        else f"- {_ico} {_ttl}{_src}\n"
                    )

        elif news_links:
            # Fallback: flat list from yfinance/FMP/Serper — apply GLM filter here too
            try:
                from core.glm_client import GLMClient as _GLMClient2
                _glm2 = _GLMClient2()
                _glm_name2    = fund.get('company_name') or target
                _glm_sector2  = fund.get('sector', 'General') or 'General'
                _glm_type2    = 'crypto' if target.endswith('-USD') else 'stock'
                news_links = _glm2.filter_news_relevance(
                    news_links, _glm_name2, target, _glm_sector2, _glm_type2)
            except Exception:
                pass  # keep original news_links on any failure
            news_block = "\n\n---\n📰 **Latest News** *(live at time of query)*\n"
            for n in news_links:
                _url = n.get("url", "")
                _ttl = n.get("title", "")
                news_block += (
                    f"- [{_ttl}]({_url})\n" if _url else f"- {_ttl}\n"
                )
        else:
            # Mandatory fallback — never skip news section
            if _is_regional_energy:
                _news_fallback_msg = (
                    "No live news fetched. Monitor: "
                    "**Argaam**, **Mubasher**, **Reuters Energy**, "
                    "and OPEC+ statements for real-time catalysts."
                )
            elif _is_local_ticker:
                _news_fallback_msg = (
                    "No live news fetched. Check **Argaam**, **The National**, "
                    "or the issuer's investor relations page for latest updates."
                )
            else:
                _news_fallback_msg = (
                    "No live news fetched. Check **Bloomberg**, **Reuters**, "
                    "or **Seeking Alpha** for the latest updates."
                )
            news_block = f"\n\n---\n📰 **Latest News**\n> ⚠️ {_news_fallback_msg}\n"

        # ── X / Twitter Posts Block (Grok Live) ───────────────────────────────
        # Rendered directly — not dependent on LLM compliance.
        # Appended after news section when Grok returned valid top_posts.
        _x_posts_block = ""
        _xp_list = _x_data.get("top_posts", []) if _x_data else []
        if _xp_list and _x_data.get("source") == "grok-live":
            _xs_label  = _x_data.get("sentiment", "")
            _xs_score  = _x_data.get("score", 0.0)
            _xs_score_str = f"{_xs_score:+.2f}"
            _xbrk2     = _x_data.get("breaking")
            _xthm2     = _x_data.get("themes", [])

            _x_posts_block  = "\n\n---\n"
            _x_posts_block += f"📱 **X / Twitter Sentiment** *(Grok live · last 48h)*\n"
            if _xs_label:
                _s_ico = {"bullish": "🟢", "bearish": "🔴", "mixed": "🟡", "neutral": "⚪"}.get(
                    _xs_label.lower(), "🟢" if _xs_score >= 0.3 else "🔴" if _xs_score <= -0.3 else "⚪"
                )
                _x_posts_block += f"> {_s_ico} **{_xs_label}** (score: {_xs_score_str})"
                if _xthm2:
                    _x_posts_block += f" · {' · '.join(_xthm2[:3])}"
                _x_posts_block += "\n"
            if _xbrk2:
                _x_posts_block += f"> ⚡ **BREAKING:** {_xbrk2}\n"
            _x_posts_block += "\n"
            for _xp in _xp_list[:4]:
                _p_ico  = "🟢" if _xp.get("impact") == "Positive" else "🔴" if _xp.get("impact") == "Negative" else "⚪"
                _p_src  = _xp.get("source", "")
                _p_lk   = f" *({_xp['likes']:,} likes)*" if _xp.get("likes") else ""
                _p_dt   = f" · {_xp.get('date','')}" if _xp.get("date") else ""
                _p_txt  = _xp.get("text", "")[:180]
                _x_posts_block += f"- {_p_ico} **{_p_src}**{_p_lk}{_p_dt}: \"{_p_txt}\"\n"
            news_block += _x_posts_block

        # ── Positioning Block ──────────────────────────────────────────────────
        import math as _math_pos
        def _clean(v, d=0.0):
            try:
                f = float(v or 0)
                return d if (_math_pos.isnan(f) or _math_pos.isinf(f)) else f
            except Exception:
                return d
        sma50  = _clean(summary.get('sma_50', 0))
        sma200 = _clean(summary.get('sma_200', 0))
        _fp_ref = _clean(real_price or _fallback_price or 0)
        def _fmt_price(p):
            if not p:
                return "N/A"
            return f"{p:,.2f} {_currency_sym}" if _is_local_mkt else f"${p:,.2f}"
        if _report_snapshot:
            entry_price = _report_snapshot.get("entry")
            stop_price = _report_snapshot.get("stop")
            _pos_target = _report_snapshot.get("target")
            _rp_pos = _report_snapshot.get("price")
        else:
            entry_price = ep
            stop_price = sp
            _pos_target = _snapshot_target
            _rp_pos = _fp_ref

        entry_level = _fmt_price(entry_price)
        stop_level = _fmt_price(stop_price)
        if _pos_target and _rp_pos:
            upside = ((_pos_target / _rp_pos) - 1) * 100
            target_level = f"{_pos_target:,.2f} {_currency_sym} ({upside:+.1f}%)" if _is_local_mkt else f"${_pos_target:,.2f} ({upside:+.1f}%)"
        else:
            target_level = "N/A"

        if _trust_target_is_sma:
            _is_crypto_local = bool(
                fund.get('asset_type') == 'crypto'
                or str(target).upper().endswith(('-USD', '-BTC', '-ETH'))
                or 'BTC' in str(target).upper()
                or 'ETH' in str(target).upper()
                or 'crypto' in str(fund.get('sector', '')).lower()
            )
            if _is_crypto_local and _rp_pos and sma200 and _rp_pos < sma200:
                _target_rationale = f'⚠️ Base case target: {_trust_sma_used} mean reversion (price below {_trust_sma_used}, no analyst coverage)'
            elif _is_crypto_local:
                _target_rationale = f'⚠️ Crypto technical target: {_trust_sma_used} × 1.15 extension (no analyst coverage)'
            else:
                _target_rationale = f'⚠️ Technical target ({_trust_sma_used} mean-reversion) — not analyst'
        elif _target_is_estimate:
            _target_rationale = '⚠️ EisaX FV Estimate (no analyst coverage)'
        else:
            _target_rationale = 'Analyst consensus target'
        # Smart stop rationale — determine dynamically from stop_price vs reference
        _rp_pos2 = real_price or _fallback_price or 0
        _stop_pct = round((1 - stop_price / _fp_ref) * 100, 1) if (stop_price and _fp_ref and _fp_ref > 0) else 9.0
        if _atr_val and _atr_val > 0 and stop_price and _fp_ref:
            _expected_atr_stop = round(_fp_ref - 2 * _atr_val, 2)
            _expected_sma_stop = round(sma200 - 2 * _atr_val, 2) if sma200 else None
            _atr_dist = min(
                abs(stop_price - _expected_atr_stop) / _fp_ref if _fp_ref else 1,
                abs(stop_price - _expected_sma_stop) / _fp_ref if (_expected_sma_stop and _fp_ref) else 1,
            )
            _stop_rationale = f"ATR-based stop (2×ATR={_atr_val:.2f}, -{_stop_pct:.1f}%)"
        elif stop_price and sma200 and abs(stop_price - sma200 * 0.95) / max(sma200 * 0.95, 1) < 0.04:
            _stop_rationale = f"Below SMA200 support (-{_stop_pct:.1f}%)"
        else:
            _stop_rationale = f"Trailing stop (-{_stop_pct:.1f}% from current)"
        # ── Pullback status annotation ─────────────────────────────────────────
        _rp_pos3 = real_price or _fallback_price or 0
        if entry_price and _rp_pos3 and _rp_pos3 > entry_price * 1.02:
            _pct_to_entry = ((_rp_pos3 - entry_price) / _rp_pos3) * 100
            _entry_note = (
                f"\n\n> ⏳ **Awaiting Pullback** — Current price "
                f"({_fmt_price(_rp_pos3)}) is **{_pct_to_entry:.1f}% above** the entry level. "
                f"Current price ({_fmt_price(_rp_pos3)}) is **{_pct_to_entry:.1f}% above** the identified entry zone of {_fmt_price(entry_price)}, which reduces the margin of safety relative to the defined risk parameters."
            )
        else:
            _entry_note = ""   # price already at or below entry — no note needed

        _entry_rationale = (
            'Near SMA50 support'
            if entry_price and sma50 and abs(entry_price - sma50) / sma50 < 0.03
            else 'Near SMA200 support'
            if entry_price and sma200 and abs(entry_price - sma200) / sma200 < 0.05
            else 'Pullback entry — below current price'
            if entry_price and _rp_pos3 and entry_price < _rp_pos3 * 0.98
            else 'At current price — entry zone active'
        )

        from core.services.positioning_validator import validate_positioning as _trust_validate_positioning
        _positioning_validation = _trust_validate_positioning(entry_price, stop_price, _pos_target, side="long")
        _trust_audit_log.append(_positioning_validation.audit)
        if _positioning_validation.suppressed:
            _trust_visible_warnings.append("Positioning section unavailable pending validation.")

        # ── Position Size Block ────────────────────────────────────────────────
        # Safe fallbacks: sc_data/verdict_sc/final/conviction are local to _build_scorecard_md.
        # Extract real score from _pre_scorecard_md string first (most accurate source).
        import re as _re_sc2
        _sc_score_extracted = None
        _sc_blended_extracted = None
        if '_pre_scorecard_md' in dir() and _pre_scorecard_md:
            _sc_m = _re_sc2.search(r'EisaX Score[:\s*]*\*\*(\d+)/100\*\*', _pre_scorecard_md)
            if _sc_m:
                _sc_score_extracted = int(_sc_m.group(1))
            # Extract blended score from scorecard headline e.g. "Blended: **61/100**"
            _bl_m = _re_sc2.search(r'Blended[:\s*]*\*\*(\d+)/100\*\*', _pre_scorecard_md)
            if _bl_m:
                _sc_blended_extracted = int(_bl_m.group(1))
            # Also extract tech score row: "| Tech Signal Score | 48/100 |"
            _ts_m = _re_sc2.search(r'Tech[^\|]*Score\s*\|\s*(\d+)/100', _pre_scorecard_md)
            _sc_tech_extracted = int(_ts_m.group(1)) if _ts_m else None
            # Also extract conviction from scorecard
            _cv_m = _re_sc2.search(r'Conviction:\s*\*\*([^*]+)\*\*', _pre_scorecard_md)
            if _cv_m and 'conviction' not in dir():
                conviction = _cv_m.group(1).strip()

        if 'sc_data' not in dir():
            sc_data = {'beta': float(_effective_beta or 1.0), 'price': real_price or 0}
        # Inject extracted scores back into sc_data so logging has correct values
        if _sc_blended_extracted is not None:
            sc_data['blended_score'] = _sc_blended_extracted
        if '_sc_tech_extracted' in dir() and _sc_tech_extracted is not None:
            sc_data['tech_score'] = _sc_tech_extracted
        if '_div_info' not in dir():
            _div_info = {'diverges': False, 'gap': 0, 'message': ''}
        if 'verdict_sc' not in dir():
            _vh = (scorecard_verdict_hint or 'HOLD').split()[0].upper()
            verdict_sc = _vh if _vh in ('BUY', 'HOLD', 'SELL', 'REDUCE') else 'HOLD'
        if 'final' not in dir():
            _sh = scorecard_verdict_hint or ''
            final = 75 if 'BUY' in _sh.upper() else 45 if 'SELL' in _sh.upper() or 'REDUCE' in _sh.upper() else 55
        if 'conviction' not in dir():
            _hint_up = (scorecard_verdict_hint or '').upper()
            conviction = 'High' if 'STRONG BUY' in _hint_up else 'Medium' if 'BUY' in _hint_up else 'Low'
        if not _div_info.get('message'):
            _div_info = _consensus_divergence(
                verdict_sc, analyst_consensus or '',
                adx=float(sc_data.get('adx') or (summary or {}).get('adx') or 20),
                beta=float(sc_data.get('beta') or _effective_beta or 1.0),
            )

        _beta_ps  = float(sc_data.get('beta') or 1.0)
        _vrd_lower = (verdict_sc or '').lower()
        # Use score extracted from scorecard markdown for perfect consistency
        _score_ps = _sc_score_extracted if _sc_score_extracted is not None else (final if isinstance(final, (int, float)) else 50)

        # ── Deterministic score-based sizing table ─────────────────────────────
        _SIZING_TABLE = [
            (85, 100, "7–10%", "12%",  "High Conviction"),
            (70,  84, "5–8%",  "10%",  "Medium-High"),
            (55,  69, "3–5%",  "7%",   "Medium"),
            (0,   54, "1–3%",  "5%",   "Low Conviction"),
        ]
        _alloc_core, _alloc_max, _sizing_label = "1–3%", "5%", "Low Conviction"
        for _lo, _hi, _core, _max, _lbl in _SIZING_TABLE:
            if _lo <= _score_ps <= _hi:
                _alloc_core, _alloc_max, _sizing_label = _core, _max, _lbl
                break

        _beta_warn   = (f"\n- ⚠️ High Beta ({_beta_ps:.1f}x) — reduce size by ~30% vs baseline" if _beta_ps > 2.0
                        else f"\n- ⚠️ Elevated Beta ({_beta_ps:.1f}x) — size conservatively" if _beta_ps > 1.5
                        else "")
        _sector_warn = ("\n- ⚠️ High oil-price dependency — cap total Energy sector exposure at 15% of portfolio"
                        if _is_regional_energy else "")

        _position_size_block = (
            f"\n\n**💼 Suggested Position Size**\n"
            f"| | Guidance |\n"
            f"|---|---|\n"
            f"| Core Allocation | {_alloc_core} of portfolio |\n"
            f"| Add on Pullback | {entry_level} |\n"
            f"| Max Exposure | {_alloc_max} |\n"
            f"> *Sizing: Score {_score_ps}/100 → {_sizing_label} tier | Core: {_alloc_core} | Max: {_alloc_max} — deterministic table, not LLM judgment*"
            f"{_beta_warn}{_sector_warn}"
        )
        _bullish_count = int(sc_data.get('bullish_count') or 0)
        _bearish_count = int(sc_data.get('bearish_count') or 0)
        _decision_conf = self._compute_decision_confidence(
            score=_score_ps,
            bullish_count=_bullish_count,
            bearish_count=_bearish_count,
            beta=_beta_ps,
            verdict=verdict_sc,
        )
        # ── Deterministic conviction formula (fully traceable) ─────────────────
        _upside_val = float(_precomputed.get('upside_to_target') or 0)
        _has_coverage = bool(analyst_target and float(analyst_target or 0) > 0)
        _adx_val = float((summary or {}).get('adx', 0))
        _trend_bear = (verdict_sc or '').upper() in ('SELL', 'REDUCE', 'AVOID')

        _conv_base = round(_score_ps * 0.5, 1)
        _conv_upside = round(min(_upside_val / 2, 15), 1)
        _conv_coverage = 10.0 if _has_coverage else 0.0
        _conv_trend = -10.0 if (_trend_bear and _adx_val > 25) else 0.0
        _conv_adx = round(min(_adx_val / 4, 12.5), 1)
        _conv_raw = _conv_base + _conv_upside + _conv_coverage + _conv_trend + _conv_adx
        _conv_pct = int(min(max(round(_conv_raw), 30), 85))

        _conviction_note = (
            f"*Conviction: {_conv_pct}% — "
            f"Score({_conv_base}) + Upside({_conv_upside}) + "
            f"Coverage({_conv_coverage:+.0f}) + Trend({_conv_trend:+.0f}) + ADX({_conv_adx:+.1f}) "
            f"→ Raw({_conv_raw:.1f}) → Clamped(30–85%)*"
        )
        _decision_framework_block = self._build_decision_framework_block(
            verdict=verdict_sc,
            confidence=_decision_conf,
            conviction=conviction,
            conviction_note=_conviction_note,
            beta=_beta_ps,
            current_price=_rp_pos3,
            entry_price=entry_price,
            sma50=sma50,
            next_earnings=next_earnings,
            currency_sym=_currency_sym,
            is_local_mkt=_is_local_mkt,
            is_arabic=_is_arabic_request,
            is_crypto=_is_crypto_asset,
            is_etf=bool(_etf_meta_early),
            is_commodity=bool(_etf_meta_early and _etf_meta_early.get("etf_type", "").startswith("commodity")),
            is_reit=bool((fund or {}).get("sector", "").lower() in ("real estate", "reits")),
        )

        # ── Entry Quality Score ───────────────────────────────────────────────
        try:
            from core.scorecard import compute_entry_quality as _ceq2
            _eq_sc_data = {
                'rsi': float((summary or {}).get('rsi', 50) or 50),
                'adx': float((summary or {}).get('adx', 20) or 20),
                'price': float(real_price or _fallback_price or 0),
                'sma200': float((summary or {}).get('sma_200', 0) or 0),
                'fear_greed': int((fg_data or {}).get('score', 50) or 50),
                'volume': float(fund.get('volume_today', 0) or 0),
                'avg_volume': float(fund.get('volume_avg90d', 0) or fund.get('avg_volume', 0) or 0),
                'trend': str((summary or {}).get('trend', '') or ''),
            }
            _eq_score2, _eq_label2, _eq_note2 = _ceq2(_eq_sc_data)
            # ── Context-aware cap + dynamic caption ──────────────────────────────
            _rp_eq  = float(real_price or _fallback_price or 0)
            _ep_eq  = float(entry_price or 0)
            _vrd_eq = str(verdict_sc or verdict or "HOLD").upper()

            # Determine price-vs-entry relationship
            _above_2pct  = bool(_ep_eq and _rp_eq and _rp_eq > _ep_eq * 1.02)   # >2% above entry
            _above_lt2   = bool(_ep_eq and _rp_eq and _ep_eq * 1.0 < _rp_eq <= _ep_eq * 1.02)  # 0-2% above
            _at_or_below = bool(not _above_2pct and not _above_lt2)              # at or below entry

            # Apply cap based on position vs entry
            if _above_2pct:
                _eq_score2 = min(_eq_score2, 50)   # >2% above entry → max 50
            elif _above_lt2:
                _eq_score2 = min(_eq_score2, 75)   # <2% above entry → max 75
            # No cap when at/below entry — score computed normally

            # Dynamic caption: derived from score + position + verdict (5-case table)
            _hold_like = _vrd_eq in ("HOLD", "REDUCE", "AVOID", "SELL")

            if _eq_score2 >= 80 and _at_or_below:
                _eq_label2 = "Good Timing ✅"
                _eq_note2  = "Strong setup — price at or below entry zone."
            elif _eq_score2 >= 60 and _at_or_below:
                _eq_label2 = "Fair ✅"
                _eq_note2  = "Fair setup — within entry zone."
            elif _eq_score2 >= 60 and not _at_or_below:
                _eq_label2 = "Caution ⚠️"
                _eq_note2  = "Price above entry zone — await pullback before sizing in."
            elif _eq_score2 < 60 and _above_2pct:
                # >2% above entry zone is the strongest negative signal — takes priority over HOLD
                _eq_label2 = "Poor Timing ❌"
                _eq_note2  = "Poor timing — price extended above entry zone. Wait for pullback."
            elif _eq_score2 < 60 and _hold_like:
                _eq_label2 = "Caution ⚠️"
                _eq_note2  = "Caution — entry not confirmed, await signal before acting."
            else:
                # fallback: map by score only
                if _eq_score2 >= 80:
                    _eq_label2 = "Good Timing ✅"
                    _eq_note2  = "Entry conditions are favorable — risk/reward is well-positioned."
                elif _eq_score2 >= 60:
                    _eq_label2 = "Fair ✅"
                    _eq_note2  = "Acceptable setup — proceed with standard position sizing."
                elif _eq_score2 >= 40:
                    _eq_label2 = "Caution ⚠️"
                    _eq_note2  = "Timing is suboptimal — consider scaling in gradually."
                else:
                    _eq_label2 = "Poor Timing ❌"
                    _eq_note2  = "Entry conditions are unfavorable — wait for a better setup."
        except Exception as _eq_ex:
            logger.debug(f"[EntryQuality] failed: {_eq_ex}")
            _eq_score2, _eq_label2, _eq_note2 = 50, 'N/A', ''

        # ── Technical Signal (Supporting) ─────────────────────────────────────
        from core.services.scorecard_engine import classify_adx as _sc_classify_adx
        _trend_bull  = bool(real_price and sma200 and real_price > sma200)
        _macd_bull   = float(summary.get('macd', 0) or 0) > float(summary.get('macd_signal', 0) or 0)
        _adx_val_sc  = float(summary.get('adx', 0) or 0)
        _adx_strong  = _adx_val_sc > 25
        _trend_lbl   = "Bullish Trend" if _trend_bull  else "Bearish Trend"
        _macd_lbl    = "Bullish Momentum" if _macd_bull  else "Bearish Momentum"
        _adx_short_sc, _ = _sc_classify_adx(_adx_val_sc)
        _adx_lbl     = f"{_adx_short_sc} ADX"
        if _trend_bull and _macd_bull and _adx_strong:
            _final_sig, _final_sig_emoji = "Strong Buy",   "✅"
        elif _trend_bull and _macd_bull and not _adx_strong:
            _final_sig, _final_sig_emoji = "Weak Buy",     "⚠️"
        elif _trend_bull and not _macd_bull and _adx_strong:
            _final_sig, _final_sig_emoji = "Hold/Caution", "⚠️"
        elif _trend_bull and not _macd_bull and not _adx_strong:
            _final_sig, _final_sig_emoji = "Neutral",      "⚪"
        elif not _trend_bull and _macd_bull and _adx_strong:
            _final_sig, _final_sig_emoji = "Hold/Caution", "⚠️"
        elif not _trend_bull and _macd_bull and not _adx_strong:
            _final_sig, _final_sig_emoji = "Neutral",      "⚪"
        elif not _trend_bull and not _macd_bull and _adx_strong:
            _final_sig, _final_sig_emoji = "Strong Sell",  "🔴"
        else:
            _final_sig, _final_sig_emoji = "Weak Sell",    "⚠️"
        if _low_data_compact_mode:
            if "Buy" in _final_sig:
                _final_sig, _final_sig_emoji = "Positive momentum (low-data reliability)", "⚠️"
            elif "Sell" in _final_sig:
                _final_sig, _final_sig_emoji = "Negative momentum (low-data reliability)", "⚠️"

        # ── Market Regime Label ───────────────────────────────────────────────
        _fg_score_r  = int((fg_data or {}).get('score', 50) or 50)
        _trend_bull_r = _trend_bull  # already computed above
        if _fg_score_r <= 30 and not _trend_bull_r:
            _regime, _regime_emoji, _regime_color = "RISK-OFF",    "🔴", "red"
        elif _fg_score_r >= 70 and _trend_bull_r:
            _regime, _regime_emoji, _regime_color = "RISK-ON",     "🟢", "green"
        elif _fg_score_r <= 45 or not _trend_bull_r:
            _regime, _regime_emoji, _regime_color = "CAUTIOUS",    "🟡", "orange"
        else:
            _regime, _regime_emoji, _regime_color = "NEUTRAL",     "⚪", "gray"
        _fg_lbl_r = "Extreme Fear" if _fg_score_r <= 20 else "Fear" if _fg_score_r <= 40 else "Neutral" if _fg_score_r <= 60 else "Greed" if _fg_score_r <= 80 else "Extreme Greed"
        _regime_block = (
            f"\n\n---\n"
            f"{_regime_emoji} **Market Regime: {_regime}**\n"
            f"*(Fear & Greed: {_fg_score_r} — {_fg_lbl_r} | Trend: {'Bullish' if _trend_bull_r else 'Bearish'})*\n"
        )
        _final_tech_block = (
            f"\n\n---\n"
            f"📡 **Technical Signal (Supporting): {_final_sig} {_final_sig_emoji}**\n"
            f"*({_trend_lbl} + {_macd_lbl} + {_adx_lbl})*\n"
        )

        if _positioning_validation.suppressed:
            positioning_block = ""
        else:
            positioning_block = (
                f"\n\n---\n"
                f"📊 **Positioning Guide**\n"
                f"> ⏱️ **Entry Quality: {_eq_score2}/100 — {_eq_label2}** | {_eq_note2}\n\n"
                f"| | Level | Rationale |\n"
                f"|---|---|---|\n"
                f"| 🟢 Entry | {entry_level} | {_entry_rationale} |\n"
                f"| 🎯 Target | {target_level} | {_target_rationale} |\n"
                f"| 🔴 Stop | {stop_level} | {_stop_rationale} |\n"
                f"{_entry_note}"
                f"{_position_size_block}"
            )

        _trust_warning_block = ""
        if _trust_visible_warnings:
            _trust_warning_block = "\n\n---\n" + "\n".join(f"> {warning}" for warning in _trust_visible_warnings)

        _ascii_section = (
            "\n" + "\n".join(f"> {l}" for l in chart_str.split("\n")) + "\n"
            if chart_str else ""
        )
        chart_block = (
            f"\n\n---\n📈 **Price Chart (60 days)**\n"
            f"<div class=\"eisax-chart\" data-ticker=\"{target}\"></div>"
            + _ascii_section
        )

        _analysis_disclaimer = (
            "\n\n---\n"
            "> ⚠️ **Disclaimer:** This report is generated by EisaX AI and is for informational purposes only. "
            "It does not constitute financial advice, investment recommendation, or an offer to buy or sell any security. "
            "All prices and data are fetched live at the time of the query and may not reflect real-time market conditions. "
            "Past performance is not indicative of future results. Always verify data independently and consult a licensed financial advisor before making investment decisions."
        )

        logger.debug(f"[DEBUG] deepseek_reply length before if: {len(deepseek_reply)}, preview: {deepseek_reply[:100]}")
        # ── Post-process: fix RE line to show original ticker, not resolved alias ──
        if deepseek_reply:
            import re as _re_fix
            # Fix 1: correct RE: subject line — use original ticker the user typed
            # The LLM often wraps RE: in **bold** markers: "**Re:** Analysis of GC=F"
            if "_original_target" in dir() and _original_target != target:
                deepseek_reply = _re_fix.sub(
                    rf'(?i)(\*{{0,2}}RE:\*{{0,2}}\s+Analysis\s+of\s+){_re_fix.escape(target)}',
                    rf'\g<1>{_original_target}',
                    deepseek_reply
                )
            # Fix 1b: replace remaining resolved ticker (e.g. "GC=F") in body text
            # with a human-readable commodity name when user typed an alias.
            # Only apply for commodity futures aliases to avoid breaking stock tickers.
            _commodity_display_map = {
                "GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil",
                "NG=F": "Natural Gas", "PL=F": "Platinum", "PA=F": "Palladium",
                "HG=F": "Copper", "BZ=F": "Brent Oil",
            }
            if target in _commodity_display_map:
                _friendly = _commodity_display_map[target]
                deepseek_reply = deepseek_reply.replace(target, _friendly)
            # Fix 2: correct RSI condition label — Gemini ignores the prompt instruction
            # and labels RSI as "Neutral" when it should reflect the computed condition.
            _rsi_val = summary.get('rsi', 50)
            _correct_condition = (
                "Overbought" if _rsi_val > 70 else
                "Near Overbought" if _rsi_val >= 60 else
                "Near Oversold" if _rsi_val <= 40 else
                "Oversold" if _rsi_val <= 30 else
                "Neutral"
            )
            if _correct_condition != "Neutral":
                _rsi_str = _re_fix.escape(f"{_rsi_val:.1f}")
                # Single broad pattern: RSI ... <value> ... Neutral (within same clause)
                # Handles: "RSI is 35.6 (Neutral)", "RSI: 35.6 (Neutral)", "RSI at 35.6 is Neutral",
                #           "RSI at 35.6 is **Neutral**", etc.
                deepseek_reply = _re_fix.sub(
                    rf'(?i)((?:RSI)\b[^.;]*?{_rsi_str}[^.;]*?)\*{{0,2}}Neutral\*{{0,2}}',
                    rf'\g<1>{_correct_condition}',
                    deepseek_reply
                )
            # ── Peer Table Data Lock: override DeepSeek's div yields with Python values ──
            if _peer_rows:
                try:
                    import re as _re_peer
                    for _pr in _peer_rows:
                        _ptk = _re_peer.escape(str(_pr['ticker']))
                        _correct_yield = f"{_pr['div_yield']}%" if _pr.get('div_yield') else "N/A%"
                        # Match table row for this ticker and replace 4th column (Div Yield)
                        deepseek_reply = _re_peer.sub(
                            rf'(\|\s*\*{{0,2}}{_ptk}\*{{0,2}}\s*\|[^|]+\|[^|]+\|)\s*[^|]*?(%|N/A)\s*(\|)',
                            rf'\g<1> {_correct_yield} \g<3>',
                            deepseek_reply
                        )
                except Exception as _peer_fix_e:
                    logger.debug(f"[PeerFix] skipped: {_peer_fix_e}")
            # ── Smart Compression: remove Section 8 sentences that repeat Section 4 risks ──
            try:
                import re as _re_compress
                # Extract Section 4 content
                _s4_match = _re_compress.search(
                    r'(?:^|\n)#+\s*4[.\s]*Key Risks?(.*?)(?=\n#+\s*5[.\s])',
                    deepseek_reply, _re_compress.DOTALL | _re_compress.IGNORECASE
                )
                _s4_text = _s4_match.group(1) if _s4_match else ""
                if _s4_text:
                    # Extract key noun phrases from Section 4 (2-4 word sequences from risk labels)
                    _s4_phrases = set(_re_compress.findall(
                        r'\*\*([A-Z][A-Za-z\s/&]{3,30})\*\*', _s4_text
                    ))
                    # In Section 8 timing block, replace sentences that purely restate S4 risks
                    def _compress_s8(m):
                        _s8_block = m.group(0)
                        for _phrase in _s4_phrases:
                            # If an entire bullet/sentence in S8 is just restating the S4 risk label
                            # with no new timing/catalyst info → strip to summary form
                            _escaped = _re_compress.escape(_phrase[:20])
                            _s8_block = _re_compress.sub(
                                rf'(^[•\-\*]\s+\*\*{_escaped}[^:]*:\*\*\s+)([^•\-\*\n]{{20,150}}\n)',
                                lambda mm: mm.group(1) + "*(see Section 4)*\n",
                                _s8_block,
                                flags=_re_compress.MULTILINE
                            )
                        return _s8_block
                    deepseek_reply = _re_compress.sub(
                        r'(?:^|\n)#+\s*8[.\s].*?(?=\n#+\s*9[.\s]|\Z)',
                        _compress_s8,
                        deepseek_reply,
                        flags=_re_compress.DOTALL | _re_compress.IGNORECASE
                    )
            except Exception as _comp_e:
                logger.debug(f"[Compress] skipped: {_comp_e}")
            deepseek_reply = self._soften_execution_language(deepseek_reply)
            deepseek_reply = self._round_scenario_prices(deepseek_reply, _currency_sym)
            # ── TG1/TG6: Deduplicate repeated sentences across the full report ──────
            try:
                _dedup_lines = []
                _seen_line_keys = set()
                for _dline in deepseek_reply.split('\n'):
                    _dk = _dline.strip().lower()
                    # Always keep headings, table rows, empty lines, and short lines
                    if not _dk or _dk.startswith('#') or _dk.startswith('|') or len(_dk) < 40:
                        _dedup_lines.append(_dline)
                        continue
                    # For prose lines, deduplicate on first 80 chars
                    _dk80 = _dk[:80]
                    if _dk80 not in _seen_line_keys:
                        _seen_line_keys.add(_dk80)
                        _dedup_lines.append(_dline)
                    # else: silently drop the duplicate prose line
                deepseek_reply = '\n'.join(_dedup_lines)
            except Exception as _dd_e:
                logger.debug("[Dedup] skipped: %s", _dd_e)
            # ── Quick-mode reply trimmer: strip CIO boilerplate + cap at 3 sections ──
            if _analysis_mode == "quick":
                import re as _re2
                # Strip CIO memo header (MEMORANDUM / To: / From: / Date: / Re:)
                deepseek_reply = _re2.sub(
                    r'\*\*MEMORANDUM\*\*.*?^---\n?',
                    '', deepseek_reply, flags=_re2.DOTALL | _re2.MULTILINE
                ).strip()
                # Cap to first 3 markdown sections (###) — skip the intro if any
                _sections = _re2.split(r'(?=^#{1,3} )', deepseek_reply, flags=_re2.MULTILINE)
                _kept = [_sections[0]] if _sections[0].strip() else []
                _sec_count = 0
                for _sec in _sections[1:]:
                    if _sec_count < 3:
                        _kept.append(_sec)
                        _sec_count += 1
                    else:
                        break
                deepseek_reply = "".join(_kept).strip()

            def _enforce_verdict_consistency(text: str, verdict: str) -> str:
                """
                Strip / relabel banned phrases that contradict the locked verdict.
                Applied once after LLM output, before Quick View rendering.
                Returns cleaned text (never raises).
                """
                import re as _re_ev
                v = (verdict or 'HOLD').upper()

                # Phrase → safe replacement (preserves tone, kills contradiction)
                _BUY_PHRASES = [
                    (r'\bstrong(?:ly)?\s+buy\b',          'consider accumulating'),
                    (r'\baggressive(?:ly)?\s+entr[yies]+\b','measured entry'),
                    (r'\baccumulate\s+aggressively\b',     'accumulate gradually'),
                    (r'\badd\s+(?:more\s+)?exposure\b',    'maintain exposure'),
                    (r'\bupside\s+breakout\b',             'technical improvement'),
                    (r'\blong\s+position(?:ing)?\b',       'position monitoring'),
                ]
                _SELL_PHRASES = [
                    (r'\bstrong(?:ly)?\s+(?:bullish|uptrend)\b', 'consolidating'),
                    (r'\bupside\s+(?:target|breakout|momentum)\b','recovery potential'),
                    (r'\bbullish\s+momentum\b',                   'momentum shift'),
                    (r'\badd\s+(?:to\s+)?(?:position|exposure)\b','monitor closely'),
                ]

                try:
                    if v in ('HOLD', 'WAIT'):
                        for pat, repl in _BUY_PHRASES:
                            text = _re_ev.sub(pat, repl, text, flags=_re_ev.IGNORECASE)
                        for pat, repl in _SELL_PHRASES[:1]:   # only worst offender for HOLD
                            text = _re_ev.sub(pat, repl, text, flags=_re_ev.IGNORECASE)
                    elif v in ('REDUCE', 'SELL', 'AVOID'):
                        for pat, repl in _SELL_PHRASES:
                            text = _re_ev.sub(pat, repl, text, flags=_re_ev.IGNORECASE)
                        for pat, repl in _BUY_PHRASES:
                            text = _re_ev.sub(pat, repl, text, flags=_re_ev.IGNORECASE)
                except Exception:
                    pass  # never corrupt the report
                return text

            def _build_quick_view(
                full_report: str,
                ticker: str,
                scorecard_md: str = "",
                final_action_line: str = "",
                decision_data: dict = None,
                is_arabic: bool = False,
            ) -> str:
                """Compact snapshot — verdict · Final Action · deterministic insight · one risk."""
                import re as _re_qv
                _ar = is_arabic

                _lbl_fundamental = "الأساسيات:" if _ar else "Fundamental:"
                _lbl_timing      = "التوقيت:"   if _ar else "Timing:"
                _lbl_conviction  = "الثقة:"     if _ar else "Conviction:"
                _lbl_score       = "درجة EisaX:" if _ar else "EisaX Score:"

                # ── Line 1: Verdict — built from structured decision_data ─────────
                # decision_data is passed by the caller from self._last_scorecard_decision;
                # NO closure dependency on _build_scorecard_md locals.
                dd = decision_data or {}
                if dd:
                    _verdict_display = (
                        f"**{ticker}"
                        f" | {_lbl_fundamental} {dd.get('verdict','HOLD')} {dd.get('emoji','')}"
                        f" | {_lbl_timing} {dd.get('timing','WAIT')}"
                        f" | {_lbl_conviction} {dd.get('conviction','Medium')}"
                        f" | {_lbl_score} {dd.get('score',0)}/100**"
                    )
                else:
                    # No structured data — minimal informative fallback (never "Analysis Complete")
                    _vl = ""
                    if scorecard_md:
                        _vm = _re_qv.search(
                            r'\*\*' + _re_qv.escape(ticker) + r'\*\*[^*]*EisaX Score.*?\d+/100',
                            scorecard_md
                        )
                        if _vm:
                            _vl = _re_qv.sub(r'[*`]', '', _vm.group(0)).strip()
                    if not _vl:
                        # Last resort: show ticker + score unavailable (no "Analysis Complete")
                        _vl = f"{ticker} | Score: Unavailable — displaying core metrics only"
                    _verdict_display = f"**{_vl}**"

                # ── Line 2: Deterministic quick insight from interpretation labels ──
                _qv_verdict = dd.get('verdict', 'HOLD') if dd else 'HOLD'
                try:
                    from core.services.phrase_builder import build_quick_insight
                    _qv_decision = {
                        'verdict':      _qv_verdict,
                        'verdict_type': 'Tactical',
                        'constraints':  getattr(_de_result, 'get', lambda k, d=None: d)('constraints', [])
                                        if '_de_result' in dir() else [],
                    }
                    _insight = build_quick_insight({"ticker": ticker}, _interpretation_labels or {}, _qv_decision)
                except Exception as _qv_err:
                    logger.debug("[QuickView] deterministic insight failed: %s", _qv_err)
                    _insight = ""
                    _clean = _re_qv.sub(
                        r'MEMORANDUM.*?(?:^---\s*$|\n---\s*\n)',
                        '', full_report[:3000], flags=_re_qv.DOTALL | _re_qv.MULTILINE
                    )
                    _s1 = _re_qv.search(
                        r'(?:^|\n)#+\s*1[.\s]*Executive Summary\s*\n(.*?)(?=\n#+\s*2[.\s])',
                        _clean, _re_qv.DOTALL | _re_qv.IGNORECASE
                    )
                    if _s1:
                        _s1_text = _re_qv.sub(r'[#*`>]', '', _s1.group(1)).strip()
                        _sents = _re_qv.split(r'(?<=[.!?])\s+', _s1_text)
                        _insight = _sents[0] if _sents else ""
                    if not _insight:
                        _plain = _re_qv.sub(r'[#*`>]', '', _clean)
                        _sents = _re_qv.split(r'(?<=[.!?])\s+', _plain.strip())
                        # Never produce "analysis complete" — show data-tied note instead
                        _insight = _sents[0] if _sents else f"Core metrics displayed for {ticker}."

                # ── Line 3: Top risk label from Section 4 ────────────────────────
                _risk_patterns = [
                    r'(?:Key Risks?|إشارات المخاطر|مخاطر رئيسية)[^\n]*\n+([^\n]{20,200})',
                ]
                _top_risk = ""
                for _rp in _risk_patterns:
                    _rm = _re_qv.search(_rp, full_report, _re_qv.IGNORECASE)
                    if _rm:
                        _top_risk = _rm.group(1).strip()
                        break
                if not _top_risk:
                    _s4 = _re_qv.search(
                        r'(?:^|\n)#+\s*4[.\s]*(?:Key Risks?|إشارات المخاطر|مخاطر رئيسية)(.*?)(?=\n#+\s*5[.\s])',
                        full_report, _re_qv.DOTALL | _re_qv.IGNORECASE
                    )
                    if _s4:
                        for _l in _s4.group(1).split('\n'):
                            _ls = _l.strip()
                            if _re_qv.match(r'^[\*\-•]|^\d+\.', _ls) and len(_ls) > 15:
                                _lbl = _re_qv.search(r'\*\*([^*]+)\*\*\s*\(Severity[^)]+\)', _ls)
                                if _lbl:
                                    _top_risk = _lbl.group(0)
                                elif len(_ls) < 120:
                                    _top_risk = _re_qv.sub(r'[*`]', '', _ls)[:100]
                                break

                # Strip accidental leading numbering from insight and risk
                _insight = _re_qv.sub(r'^\d+\.\s*', '', _insight).strip()
                _top_risk = _re_qv.sub(r'^\d+\.\s*', '', _top_risk).strip()

                # Flatten embedded newlines so insight stays on one line
                # (prevents "...weak.\n⚠️ Top Risk" collision)
                _insight = ' '.join(_insight.splitlines()).strip()
                _top_risk = ' '.join(_top_risk.splitlines()).strip()

                # ── Final Action label — passed in from outer scope ────────────
                # Computed in the calling scope where verdict_sc / _entry_timing
                # are definitively available; passed as `final_action_line` param.
                _final_action_line = final_action_line

                # ── Contradiction guard: relabel insight if it conflicts verdict ──
                try:
                    _buy_re = _re_qv.compile(
                        r'\b(strong buy|buy now|accumulate|add to position|tactical buy|long position)\b',
                        _re_qv.IGNORECASE,
                    )
                    _red_re = _re_qv.compile(
                        r'\b(reduce|sell|trim|underweight|exit|short)\b',
                        _re_qv.IGNORECASE,
                    )
                    _conflict = False
                    if _qv_verdict in ('HOLD', 'WAIT') and (_buy_re.search(_insight) or _red_re.search(_insight)):
                        _conflict = True
                    if _qv_verdict == 'BUY' and _red_re.search(_insight):
                        _conflict = True
                    if _qv_verdict in ('REDUCE', 'SELL', 'AVOID') and _buy_re.search(_insight):
                        _conflict = True
                    if _conflict:
                        _ts_label = 'إشارة تقنية (داعمة)' if _ar else 'Technical Signal (Supporting)'
                        _insight = f"[{_ts_label}] {_insight}"
                except Exception:
                    pass

                _lines = [_verdict_display]
                if _final_action_line:
                    _lines.append(_final_action_line)
                if _insight:
                    _lines.append(f"💡 {_insight}")
                if _top_risk:
                    _lines.append(f"⚠️ {'أبرز مخاطر' if _ar else 'Top Risk'}: {_top_risk}")

                _qv_trailer = "\n\n---\n📄 *التقرير الكامل أدناه*\n" if _ar else "\n\n---\n📄 *Full report below*\n"
                return (
                    f"## ⚡ {'نظرة سريعة' if _ar else 'Quick View'} — {ticker}\n\n"
                    + "\n\n".join(_lines)
                    + _qv_trailer
                )

            # ── Fix 3: Verdict consistency enforcer (post-LLM, pre-Quick View) ─
            try:
                _ev_verdict = (_scorecard_decision.get('verdict') or verdict_sc or 'HOLD')
                deepseek_reply = _enforce_verdict_consistency(deepseek_reply, _ev_verdict)
            except Exception as _ev_err:
                logger.debug("[VerdictEnforcer] skipped: %s", _ev_err)

            # ── Apply interpretation guard before rendering Quick View ────────
            try:
                if _interpretation_labels:
                    from core.services.interpretation_guard import InterpretationGuard

                    _guard = InterpretationGuard()
                    _guard_result = _guard.audit_and_sanitize(deepseek_reply, _interpretation_labels)
                    if _guard_result.replacements_made > 0:
                        deepseek_reply = _guard_result.text
                        _trust_audit_log.extend(_guard_result.audit_log)
                        _report_classification = "PARTIAL"
                        _override_warning = "Technical language aligned with confirmed data signals."
                        if _override_warning not in _trust_visible_warnings:
                            _trust_visible_warnings.append(_override_warning)
                        logger.info(
                            "[QuickView] interpretation guard replaced %d conflicting claim(s)",
                            _guard_result.replacements_made,
                        )
            except Exception as _guard_err:
                logger.debug("[QuickView] interpretation guard skipped: %s", _guard_err)

            # ── Compute Final Action from structured _scorecard_decision ─────
            # No regex — data comes directly from self._last_scorecard_decision.
            try:
                _fa_v  = (_scorecard_decision.get('verdict') or verdict_sc or 'HOLD').upper()
                _fa_et = (_scorecard_decision.get('timing_en') or 'WAIT').upper()
                if _fa_v in ('REDUCE', 'SELL', 'AVOID'):
                    _outer_fa = '🔴 REDUCE / RISK CONTROL'
                elif _fa_v == 'BUY' and 'WAIT' in _fa_et:
                    _outer_fa = '🟡 WATCHLIST / WAIT FOR ENTRY'
                elif _fa_v == 'BUY':
                    _outer_fa = '🟢 BUY — Entry Confirmed'
                elif _fa_v == 'HOLD' and 'WAIT' in _fa_et:
                    _outer_fa = '⚪ WAIT / NO ACTION'
                else:
                    _outer_fa = '⚪ HOLD — Monitor'
                if _is_arabic_request:
                    _outer_fa = {
                        '🔴 REDUCE / RISK CONTROL':      '🔴 تخفيض / إدارة مخاطر',
                        '🟡 WATCHLIST / WAIT FOR ENTRY': '🟡 قائمة مراقبة / انتظر نقطة دخول',
                        '🟢 BUY — Entry Confirmed':      '🟢 شراء — نقطة دخول مؤكدة',
                        '⚪ WAIT / NO ACTION':           '⚪ انتظر / لا إجراء',
                        '⚪ HOLD — Monitor':             '⚪ احتفظ — مراقبة',
                    }.get(_outer_fa, _outer_fa)
                _lbl_fa = 'القرار النهائي' if _is_arabic_request else 'Final Action'
                _outer_final_action_line = f"**{_lbl_fa}:** {_outer_fa}"
                logger.debug("[QuickView] Final Action: v=%s et=%s → %s", _fa_v, _fa_et, _outer_fa)
            except Exception as _ofa_err:
                logger.debug("[QuickView] Final Action compute failed: %s", _ofa_err)
                _outer_final_action_line = ""

            quick_view = _build_quick_view(
                deepseek_reply,
                target,
                decision_data=_scorecard_decision,
                final_action_line=_outer_final_action_line,
                is_arabic=_is_arabic_request,
            )
            final_reply = quick_view + "\n\n---\n## 📋 Full Report\n\n" + deepseek_reply
        # ── 7. Build Final Reply ───────────────────────────────────────────────
        if deepseek_reply:
            try:
                _vel_note = ""
                # ── Prediction Tracker + Smart Signals ───────────────────────
                _heatmap_block = ""
                _trend_chart_block = ""
                _alert_block = ""
                try:
                    from prediction_tracker import (
                        log_prediction as _log_pred,
                        check_due_predictions as _check_preds,
                        log_score as _log_score,
                        get_score_velocity as _get_velocity,
                        get_portfolio_heatmap as _get_heatmap,
                        get_score_trend_chart as _get_trend,
                        check_verdict_upgrade as _check_upgrade,
                        get_accuracy_stats as _get_acc_stats,
                    )
                    _pred_price = float(real_price or _fallback_price or 0)
                    _pred_target_raw = _display_target or analyst_target or 0
                    _pred_target = float(_pred_target_raw) if _pred_target_raw else 0
                    _pred_verdict = str(verdict_sc or verdict or "HOLD").upper()
                    if _pred_price > 0 and target:
                        _log_pred(target, _pred_verdict, _pred_price, _pred_target)
                    _score_result = locals().get("result")
                    _sc_fund = int(getattr(_score_result, "__getitem__", lambda i: [None, 0, {}])(1) if _score_result else 0) if isinstance(_score_result, (list, tuple)) and len(_score_result) > 1 else 0
                    _sc_blend = int(sc_data.get("blended_score", 0) or 0)
                    _sc_tech = int(sc_data.get("tech_score", 0) or 0)
                    _log_score(target, _sc_fund, _sc_tech, _sc_blend, _pred_verdict)

                    # ── #1 Score Velocity ──────────────────────────────────────
                    _velocity = _get_velocity(target)
                    if _velocity.get("change") and abs(_velocity["change"]) >= 5:
                        _vel_icon = "📈" if _velocity["arrow"] == "↑" else "📉"
                        _vel_signed = _velocity["change"]  # keep sign: +8 or -6
                        _vel_note = (
                            f"\n\n> {_vel_icon} **Score Velocity:** Blended score {_velocity['arrow']} "
                            f"{_vel_signed:+d} pts vs last analysis "
                            f"({_velocity.get('prev_score', '?')!s} → {_velocity.get('current_score', '?')!s}) "
                            f"— trend is **{_velocity['direction']}**\n"
                        )
                    else:
                        _vel_note = ""

                    # ── #4 Portfolio Heat Map ──────────────────────────────────
                    _sector = (fund or {}).get("sector", "") or ""
                    _hmap = _get_heatmap(target, _sector)
                    if _hmap.get("message"):
                        _heatmap_block = f"\n\n> {_hmap['message']}\n"

                    # ── #5 Blended Score Trend Chart ───────────────────────────
                    _trend = _get_trend(target)
                    if _trend.get("message"):
                        _trend_chart_block = f"\n\n> {_trend['message']}\n"

                    # ── #6 Auto-Alert: Verdict Upgrade/Downgrade ───────────────
                    _prev_v = _velocity.get("prev_verdict") or ""
                    _upgrade = _check_upgrade(target, _prev_v, _pred_verdict, _sc_blend)
                    if _upgrade.get("message"):
                        _alert_block = f"\n\n> {_upgrade['message']}\n"

                    _check_preds()  # resolve any due predictions (non-blocking)

                    # ── #2 Prediction Accuracy Badge ──────────────────────────
                    _acc = _get_acc_stats(days=90)
                    if _acc.get("total", 0) >= 5:
                        _acc_pct = _acc["accuracy"]
                        _acc_icon = "🎯" if _acc_pct >= 65 else ("⚡" if _acc_pct >= 50 else "📊")
                        _acc_block = (
                            f"\n\n> {_acc_icon} **Prediction Accuracy (90d):** "
                            f"{_acc.get('hits', 0)}/{_acc['total']} correct "
                            f"(**{_acc_pct}%**) — tracked across all EisaX analyses\n"
                        )
                    else:
                        _acc_block = ""
                except Exception as _pt_e:
                    logger.debug(f"[PredTracker] skipped: {_pt_e}")
                    _acc_block = ""

                factcheck_block = self._build_factcheck_block(
                    real_price, fund, summary, dc_data, forward_pe,
                    next_earnings=next_earnings, fg_data=fg_data,
                    ticker=target, effective_beta=_effective_beta
                )
                _acc_block = locals().get("_acc_block", "")
                if _analysis_mode == "quick":
                    _div_block = ('\n\n' + _div_info['message'] + '\n') if _div_info.get('diverges') else ''
                    reply = (header + _regime_block + _vel_note + _trend_chart_block
                             + _alert_block + _acc_block + _div_block + final_reply
                             + _decision_framework_block + _final_tech_block
                             + _heatmap_block + _analysis_disclaimer)
                elif _analysis_mode == "cio":
                    _div_block = ('\n\n' + _div_info['message'] + '\n') if _div_info.get('diverges') else ''
                    reply = (header + _regime_block + _vel_note + _trend_chart_block
                             + _alert_block + _acc_block + _div_block + final_reply
                             + _decision_framework_block + _final_tech_block
                             + _heatmap_block + _analysis_disclaimer)
                else:
                    _div_block = ('\n\n' + _div_info['message'] + '\n') if _div_info.get('diverges') else ''
                    reply = (
                        header
                        + _regime_block
                        + _vel_note
                        + _trend_chart_block
                        + _alert_block
                        + _acc_block
                        + _div_block
                        + final_reply
                        + _decision_framework_block
                        + _final_tech_block
                        + factcheck_block
                        + news_block
                        + _trust_warning_block
                        + positioning_block
                        + _heatmap_block
                        + _pre_scorecard_md
                        + chart_block
                        + _analysis_disclaimer
                    )

                _trust_layer_data = {
                    "classification": _report_classification,
                    "warnings": list(_trust_visible_warnings),
                    "errors": [],
                    "audit": list(_trust_audit_log),
                }
                if _report_snapshot:
                    from core.services.report_lint import ReportSection as _ReportSection
                    from core.services.report_lint import RenderedReport as _RenderedReport
                    from core.services.report_lint import lint_report as _lint_report

                    if _analysis_mode in ("quick", "cio"):
                        _report_sections = [_ReportSection("Memo", reply)]
                    else:
                        _report_sections = [
                            _ReportSection("Memo", final_reply),
                            _ReportSection("Fact Check", factcheck_block),
                            _ReportSection("News", news_block),
                            _ReportSection("Trust Warnings", _trust_warning_block),
                            _ReportSection("Positioning Guide", positioning_block),
                            _ReportSection("Heatmap", _heatmap_block),
                            _ReportSection("Scorecard", _pre_scorecard_md),
                            _ReportSection("Chart", chart_block),
                            _ReportSection("Disclaimer", _analysis_disclaimer),
                        ]

                    _render_candidate = _RenderedReport(
                        ticker=_display_ticker,
                        full_text=reply,
                        sections=_report_sections,
                        entry=entry_price,
                        stop=stop_price,
                        target=_pos_target,
                        warnings=list(_trust_visible_warnings),
                        audit_log=list(_trust_audit_log),
                        observed_prices=[
                            _report_snapshot.get("price"),
                            real_price or _fallback_price or 0,
                        ],
                    )
                    _lint = _lint_report(
                        _render_candidate,
                        _report_snapshot,
                        decision=locals().get('_de_result'),
                        interpretation_labels=locals().get('_de_labels'),
                    )
                    _trust_audit_log.extend(_lint.audit)
                    _trust_layer_data = {
                        "classification": (
                            "FLAGGED" if not _lint.safe_to_render
                            else "PARTIAL" if (_lint.warnings or _trust_visible_warnings or _report_classification == "PARTIAL")
                            else "SAFE"
                        ),
                        "warnings": _lint.warnings + list(_trust_visible_warnings),
                        "errors": _lint.errors,
                        "audit": list(_trust_audit_log),
                    }

                    if not _lint.safe_to_render:
                        _blocked_reasons = "\n".join(f"- {err}" for err in _lint.errors)
                        reply = (
                            header
                            + "> Warning: Trust layer blocked this report before render.\n\n"
                            + _blocked_reasons
                        )
                        state.set_artifact(sid, {
                            "type": "analysis", "content": reply, "source": "self_generated",
                            "exportable": False, "timestamp": datetime.now()
                        })
                        _REPORT_CACHE[_cache_key] = (_tc.time(), {
                            "type": "chat.reply",
                            "reply": reply,
                            "data": {
                                "agent": "finance",
                                "analytics": summary,
                                "fundamentals": fund,
                                "trust_layer": _trust_layer_data,
                            },
                        })
                        return {
                            "type": "chat.reply",
                            "reply": reply,
                            "data": {
                                "agent": "finance",
                                "analytics": summary,
                                "fundamentals": fund,
                                "trust_layer": _trust_layer_data,
                            },
                        }

                    if _analysis_mode not in ("quick", "cio"):
                        _section_map = {section.name: section for section in _render_candidate.sections}
                        factcheck_block = _section_map["Fact Check"].content
                        news_block = _section_map["News"].content
                        _trust_warning_block = _section_map["Trust Warnings"].content
                        positioning_block = _section_map["Positioning Guide"].content
                        _heatmap_block = _section_map["Heatmap"].content
                        _pre_scorecard_md = _section_map["Scorecard"].content
                        chart_block = _section_map["Chart"].content
                        _analysis_disclaimer = _section_map["Disclaimer"].content
                        reply = (
                            header
                            + _regime_block
                            + _vel_note
                            + _trend_chart_block
                            + _alert_block
                            + _acc_block
                            + _div_block
                            + final_reply
                            + _decision_framework_block
                            + _final_tech_block
                            + factcheck_block
                            + news_block
                            + _trust_warning_block
                            + positioning_block
                            + _heatmap_block
                            + _pre_scorecard_md
                            + chart_block
                            + _analysis_disclaimer
                        )

                # ── EisaX Cache Enhancement ────────────────────────────────
                try:
                    import sys as _sys
                    from core.config import BASE_DIR as _BASE_DIR
                    _root = str(_BASE_DIR)
                    if _root not in _sys.path:
                        _sys.path.insert(0, _root)
                    from report_enhancer import ReportEnhancer
                    from pipeline import cache as _cache, fetcher as _fetcher
                    from query_engine import QueryEngine
                    _qe = QueryEngine(_cache, _fetcher)
                    reply = ReportEnhancer(_qe).enhance(reply, ticker=target)
                    logger.info("[EisaX] Enhancer applied to %s", target)
                except Exception as _enh_err:
                    logger.warning("[EisaX] Enhancer skipped for %s: %s", target, _enh_err)

                # Save artifact
                state.set_artifact(sid, {
                    "type": "analysis", "content": reply, "source": "self_generated",
                    "exportable": True, "timestamp": datetime.now()
                })

                # Save to brain
                self._save_to_brain(target, reply, real_price, analyst_target, fund, news_sent)

                _REPORT_CACHE[_cache_key] = (_tc.time(), {"type": "chat.reply", "reply": reply, "data": {"agent": "finance", "analytics": summary, "fundamentals": fund, "trust_layer": _trust_layer_data}})
                return {"type": "chat.reply", "reply": reply, "data": {"agent": "finance", "analytics": summary, "fundamentals": fund, "trust_layer": _trust_layer_data}}
            except Exception as _e:
                logger.error(f"[Analytics] Reply build failed: {_e}")
                return {"type": "chat.reply", "reply": final_reply, "data": {"agent": "finance"}}

        # ── 8. Fallback: structured reply without DeepSeek ─────────────────────
        # Use scorecard decision if available, else derive from summary signals
        _fb_sd = getattr(self, '_last_scorecard_decision', {})
        verdict = (_fb_sd.get('verdict') or
                   ("ACCUMULATE" if summary['trend'] == "Bullish" and summary['momentum'] == "Bullish"
                    else "REDUCE" if summary['trend'] == "Bearish" and summary['momentum'] == "Bearish"
                    else "HOLD"))
        _fb_score     = _fb_sd.get('score', 0)
        _fb_timing_en = _fb_sd.get('timing_en', 'WAIT')
        _fb_emoji     = _fb_sd.get('emoji', '🟡')
        _fb_conv      = _fb_sd.get('conviction', 'Medium')

        # Build Final Action for fallback
        _fb_v_up = verdict.upper()
        _fb_et_up = _fb_timing_en.upper()
        if _fb_v_up in ('REDUCE', 'SELL', 'AVOID'):
            _fb_fa = '🔴 REDUCE / RISK CONTROL'
        elif _fb_v_up == 'BUY' and 'WAIT' in _fb_et_up:
            _fb_fa = '🟡 WATCHLIST / WAIT FOR ENTRY'
        elif _fb_v_up == 'BUY':
            _fb_fa = '🟢 BUY — Entry Confirmed'
        elif 'WAIT' in _fb_et_up:
            _fb_fa = '⚪ WAIT / NO ACTION'
        else:
            _fb_fa = '⚪ HOLD — Monitor'

        # Quick View block — always present, even in fallback
        _fb_qv = (
            f"## ⚡ Quick View — {target}\n\n"
            f"**{target} | Fundamental: {verdict} {_fb_emoji}"
            f" | Timing: {_fb_timing_en}"
            f" | Conviction: {_fb_conv}"
            f" | EisaX Score: {_fb_score}/100**\n\n"
            f"**Final Action:** {_fb_fa}\n\n"
            f"💡 Full analysis unavailable — displaying core metrics only.\n\n"
            f"---\n📄 *Full report below*\n\n"
        )

        reply = (
            header + _fb_qv +
            f"## Core Metrics\n\n"
            f"### Fundamentals\n"
            f"- Revenue Growth: {_P(fund.get('revenue_growth'))} | EPS Growth: {_P(fund.get('eps_growth'))}\n"
            f"- Net Margin: {_P(fund.get('net_margin'))} | ROE: {_P(fund.get('roe'))}\n"
            f"- P/E: {_X(fund.get('pe_ratio'))} | EV/EBITDA: {_X(fund.get('ev_ebitda'))}\n"
            f"- Market Cap: {_B(fund.get('market_cap'))} | Cash: {_B(fund.get('cash'))}\n\n"
            f"### Technicals\n"
            f"- Trend: {summary['trend']} | RSI: {summary['rsi']:.1f} | MACD: {summary['momentum']}\n"
            f"- VaR(95%): {var_95*100:.2f}% | Max DD: {max_dd*100:.2f}%"
        )

        # Fact-check block (fallback)
        try:
            from core.fact_checker import FactChecker
            fact_data = {**summary, "price": real_price or summary.get("price")}
            fact_block = FactChecker().verify_analysis(target, fact_data)
            reply += "\n\n" + fact_block
        except Exception as e:
            logger.error(f"[Analytics] FactChecker failed: {e}")

        # ── EisaX Cache Enhancement (fallback path) ───────────────────────
        try:
            import sys as _sys
            from core.config import BASE_DIR as _BASE_DIR
            _root = str(_BASE_DIR)
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from report_enhancer import ReportEnhancer
            from pipeline import cache as _cache, fetcher as _fetcher
            from query_engine import QueryEngine
            _qe = QueryEngine(_cache, _fetcher)
            reply = ReportEnhancer(_qe).enhance(reply, ticker=target)
            logger.info("[EisaX] Enhancer applied to %s (fallback)", target)
        except Exception as _enh_err:
            logger.warning("[EisaX] Enhancer skipped for %s: %s", target, _enh_err)

        # Save artifact
        state.set_artifact(sid, {
            "type": "analysis", "content": reply, "source": "self_generated",
            "exportable": True, "timestamp": datetime.now()
        })

        return {
            "type": "chat.reply",
            "reply": reply,
            "data": {"agent": "finance", "analytics": summary, "fundamentals": fund}
        }

    def _handle_forecast(self, sid: str, mem: Dict[str, Any], msg: str) -> Dict[str, Any]:
        # ... (imports) ...
        import core.analytics as ca
        import numpy as np
        from core.data import get_prices
        
        tickers = IntentClassifier.extract_tickers(msg)
        if not tickers and not state.get_artifact(sid):
            tickers = mem.get("tickers", [])
            
        if not tickers and not state.get_artifact(sid):
            return {"type": "chat.reply", "reply": "Please specify a ticker to forecast."}
            
        target = tickers[0]
        try:
            prices = get_prices([target], start="2020-01-01", end=None)
            if prices.empty:
                return {"type": "error", "reply": f"Could not fetch data for {target}."}
                
            series = prices[target]
            sim_days = 252 * 1 # 1 Year default
            if "5 year" in msg.lower(): sim_days = 252 * 5
            if "10 year" in msg.lower(): sim_days = 252 * 10
            
            paths = ca.calculate_monte_carlo(series, days=sim_days)
            stats = ca.get_simulation_stats(paths)
            
            current_price = series.iloc[-1]
            p50_ret = (stats['p50'] / current_price) - 1
            
            reply = (
                f"# Monte Carlo Forecast: {target}\n\n"
                f"**Horizon:** {sim_days/252:.1f} Years\n"
                f"**Simulations:** 1,000 Paths\n"
                f"**Current Price:** ${current_price:.2f}\n\n"
                f"## Projected Outcomes\n"
                f"- **Bear Case (P10):** ${stats['p10']:.2f}\n"
                f"- **Base Case (P50):** ${stats['p50']:.2f} ({p50_ret*100:+.1f}%)\n"
                f"- **Bull Case (P90):** ${stats['p90']:.2f}\n\n"
                f"**Analysis:** Based on historical volatility of {series.pct_change().std()*np.sqrt(252)*100:.1f}%. "
                f"The range of outcomes indicates the inherent uncertainty in long-term projections."
            )
            
            # SAVE ARTIFACT
            state.set_artifact(sid, {
                "type": "forecast",
                "content": reply,
                "source": "self_generated",
                "exportable": True,
                "timestamp": datetime.now()
            })
            
            return {
                "type": "chat.reply",
                "reply": reply,
                "data": {"agent": "finance", "forecast": stats}
            }
            
        except Exception as e:
            return {"type": "error", "reply": f"Forecast failed for {target}: {e}"}

    def _handle_trade(self, sid: str, mem: Dict[str, Any], msg: str) -> Dict[str, Any]:
        """
        Executes paper trades via the BrokerClient.
        """
        from core.broker import BrokerClient
        
        # 1. Initialize Broker
        broker = BrokerClient()
        if not broker.is_active():
             return {"type": "error", "reply": "Broker connection failed. Please check ALPACA_API_KEY and ALPACA_SECRET_KEY."}

        # 2. Parse Intent
        tickers = IntentClassifier.extract_tickers(msg)
        if not tickers and not state.get_artifact(sid):
            return {"type": "chat.reply", "reply": "Please specify a ticker to trade (e.g., 'Buy 10 AAPL')."}
        
        symbol = tickers[0]
        side = "buy" if "buy" in msg.lower() else "sell" if "sell" in msg.lower() else None
        
        if not side:
             return {"type": "chat.reply", "reply": "Please specify 'buy' or 'sell'."}
             
        # Simple quantity parser: look for first number
        import re
        # Find integers or floats
        qty_match = re.search(r'\b\d+\b', msg)
        qty = float(qty_match.group(0)) if qty_match else 1.0
        
        # 3. Execute
        try:
            order = broker.submit_order(symbol, qty, side, "market", "day")
            
            if "error" in order:
                return {"type": "error", "reply": f"Trade rejected: {order['error']}"}
            
            reply = (
                f"# Trade Submitted: {side.upper()} {symbol}\n"
                f"**Qty:** {order['qty']}\n"
                f"**Status:** {order['status'].upper()}\n"
                f"**Order ID:** `{order['id']}`\n"
            )
            return {
                "type": "chat.reply", 
                "reply": reply,
                "data": {"agent": "finance", "trade_id": order['id']}
            }
            
        except Exception as e:
             return {"type": "error", "reply": f"Trade execution failed: {e}"}

    def _handle_greeks(self, sid: str, msg: str) -> Dict[str, Any]:
        """Calculates Option Greeks using Black-Scholes."""
        import core.analytics as ca
        import re
        
        # 1. Parameter Extraction (Defaults)
        S = 100.0; K = 100.0; T = 0.25; r = 0.05; sigma = 0.20
        option_type = "call" if "call" in msg.lower() else "put"
        
        # Try to find specific values via regex
        spot_match = re.search(r"(?:spot|price|current)\s*(?:is|at|=)?\s*\$?(\d+\.?\d*)", msg, re.I)
        if spot_match: S = float(spot_match.group(1))
        
        strike_match = re.search(r"(?:strike)\s*(?:is|at|=)?\s*\$?(\d+\.?\d*)", msg, re.I)
        if strike_match: K = float(strike_match.group(1))
        
        iv_match = re.search(r"(?:iv|vol|volatility)\s*(?:is|at|=)?\s*(\d+\.?\d*)", msg, re.I)
        if iv_match: 
            val = float(iv_match.group(1))
            sigma = val / 100.0 if val > 1.0 else val
            
        rate_match = re.search(r"(?:rate|rf)\s*(?:is|at|=)?\s*(\d+\.?\d*)", msg, re.I)
        if rate_match:
            val = float(rate_match.group(1))
            r = val / 100.0 if val > 1.0 else val
            
        months_match = re.search(r"(\d+)\s*month", msg, re.I)
        if months_match: T = float(months_match.group(1)) / 12.0
        
        # 2. Calculate
        try:
            res = ca.calculate_black_scholes(S, K, T, r, sigma, option_type)
            
            reply = (
                f"# Strategic Greeks Analysis: {option_type.upper()}\n\n"
                f"**Engine:** Black-Scholes-Merton Model\n\n"
                f"### Input Parameters\n"
                f"- **Spot:** ${S:.2f}\n"
                f"- **Strike:** ${K:.2f} ({((K/S)-1)*100:+.1f}% from spot)\n"
                f"- **Volatility (IV):** {sigma*100:.1f}%\n"
                f"- **Time to Expiry:** {T*12:.1f} months\n"
                f"- **Risk-free Rate:** {r*100:.2f}%\n\n"
                f"### Derived Greeks\n"
                f"| Metric | Value | Interpretation |\n"
                f"|---|---|---|\n"
                f"| **Delta** | {res['delta']:.4f} | Probabilistic exposure to price move |\n"
                f"| **Theta** | {res['theta']:.4f} | Daily time decay (value loss) |\n"
                f"| **Theory Price** | ${res['price']:.2f} | Fair value projection |\n\n"
                f"**EISAX Operational Note:** Theta decay accelerates sharply in the final 30 days. Plan your entries accordingly."
            )
            
            # SAVE ARTIFACT
            state.set_artifact(sid, {
                "type": "greeks",
                "content": reply,
                "source": "self_generated",
                "exportable": True,
                "timestamp": datetime.now()
            })
                
            return {
                "type": "chat.reply", 
                "reply": reply, 
                "data": {"agent": "finance", "greeks": res}
            }
        except Exception as e:
            # Re-raise so the try-except in think() can log it and fallback
            raise ValueError(f"Greeks calculation failed: {e}")
            
    def _handle_portfolio_show(self, sid: str = None, mem: dict = None, msg: str = None) -> dict:
        """جلب وعرض بيانات المحفظة الحقيقية من Alpaca"""
        broker = BrokerClient()
        if not broker.is_active():
            return {"type": "chat.reply", "reply": "❌ لا يمكن الاتصال بالوسيط. تأكد من إعداد المفاتيح في ملف .env"}
        
        acct = broker.get_account()
        pos = broker.get_positions()
        
        reply = "## 📊 ملخص المحفظة\n\n"
        reply += f"**حالة الحساب:** {acct.get('status', 'N/A').upper()}\n"
        reply += f"**إجمالي القيمة (Equity):** ${acct.get('equity', 0):,.2f}\n"
        reply += f"**القوة الشرائية:** ${acct.get('buying_power', 0):,.2f}\n\n"
        
        if pos:
            reply += "## 📈 الصفقات المفتوحة\n"
            for p in pos:
                reply += f"- **{p['symbol']}**: {p['qty']} سهم | القيمة: ${p['market_value']:,.2f} | الربح/الخسارة: {p['unrealized_plpc']*100:+.2f}%\n"
        else:
            reply += "*لا توجد صفقات مفتوحة حالياً.*"
            
        return {"type": "chat.reply", "reply": reply}

    def _handle_account_display(self) -> dict:
        """عرض ملخص الحساب والمحفظة — wrapper لـ _handle_portfolio_show"""
        return self._handle_portfolio_show()

    def _handle_portfolio_add(self, sid: str, mem: dict, msg: str) -> dict:
        """إضافة صفقة للمحفظة المحلية — يدعم الأسهم المحلية"""
        import re
        tickers = IntentClassifier.extract_tickers(msg)
        if not tickers:
            return {"type": "chat.reply", "reply": "يرجى تحديد السهم المراد إضافته. مثال: 'add 10 shares NVDA at $130' أو 'أضف 10 أسهم أرامكو'"}
        
        ticker = tickers[0].upper()
        
        # Try to resolve via local ticker resolver if not a known format
        if not any(ticker.endswith(s) for s in ['.SR', '.CA', '.AE', '.DU', '.KW', '.QA', '-USD']):
            local = _ticker_resolver.resolve_single(ticker)
            if local:
                ticker = local

        # Parse quantity
        qty_match = re.search(r'(\d+\.?\d*)\s*(?:share|سهم|stock)', msg.lower())
        qty = float(qty_match.group(1)) if qty_match else 1.0
        
        # Parse price
        price_match = re.search(r'(?:at|@|بسعر|price)\s*\$?(\d+\.?\d*)', msg.lower())
        if price_match:
            price = float(price_match.group(1))
        else:
            # Try to get live price — fast_info is ~3x faster than .info for price-only lookup
            try:
                import yfinance as yf
                _fi = yf.Ticker(ticker).fast_info
                price = float(getattr(_fi, "last_price", None) or 0)
            except Exception as _e:
                price = 0
        
        try:
            self.portfolio_tracker.add_position(ticker, qty, price)
            price_str = self._format_local_price(price, ticker)
            total_str = self._format_local_price(qty * price, ticker)
            name = self._get_local_display_name(ticker)
            return {
                "type": "chat.reply",
                "reply": f"✅ تم إضافة **{qty:.0f} سهم {name} ({ticker})** بسعر **{price_str}** للمحفظة.\n\nالقيمة الإجمالية: **{total_str}**"
            }
        except Exception as e:
            return {"type": "error", "reply": f"فشل إضافة الصفقة: {e}"}

    def _handle_portfolio_remove(self, sid: str, mem: dict, msg: str) -> dict:
        """إزالة/بيع صفقة من المحفظة المحلية"""
        import re
        tickers = IntentClassifier.extract_tickers(msg)
        if not tickers:
            return {"type": "chat.reply", "reply": "يرجى تحديد السهم المراد بيعه. مثال: 'sell 5 shares AAPL' أو 'بيع 5 أسهم أرامكو'"}
        
        ticker = tickers[0].upper()
        
        # Resolve local tickers
        if not any(ticker.endswith(s) for s in ['.SR', '.CA', '.AE', '.DU', '.KW', '.QA', '-USD']):
            local = _ticker_resolver.resolve_single(ticker)
            if local:
                ticker = local
        
        # Parse quantity
        qty_match = re.search(r'(\d+\.?\d*)\s*(?:share|سهم|stock)', msg.lower())
        qty = float(qty_match.group(1)) if qty_match else None
        
        try:
            self.portfolio_tracker.remove_position(ticker, qty)
            name = self._get_local_display_name(ticker)
            qty_str = f"{qty:.0f} سهم من " if qty else "كل أسهم "
            return {
                "type": "chat.reply",
                "reply": f"✅ تم بيع {qty_str}**{name} ({ticker})** من المحفظة."
            }
        except Exception as e:
            return {"type": "error", "reply": f"فشل إزالة الصفقة: {e}"}
