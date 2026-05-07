# Auto-extracted mixin — do not edit directly; source of truth is git history.
from __future__ import annotations
from typing import Any, Dict, Optional
import logging
from core.intent_classifier import IntentClassifier
logger = logging.getLogger(__name__)


class AnalyticsMixin:
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
        import os, requests, state
        from datetime import datetime
        # Lazy import — finance.py imports this mixin at class-body, so a
        # module-top "from core.agents.finance import ..." would be circular.
        # By the time _handle_analytics runs the parent module is fully loaded.
        from core.agents.finance import (
            FinancialAgent,
            _REPORT_CACHE, _REPORT_CACHE_TTL,
            _ETF_EQUITY_ONLY_SUFFIXES,
            _consensus_divergence,
            _fetch_btc_etf_flows,
            _safe_div_yield,
            _yf_with_retry,
            _ticker_resolver,
            _pd,
            deepcrawl_stock,
            get_live_news,
            apply_language_locks,
            classify_data_coverage_level,
            compact_low_data_generation_inputs,
            count_valid_fundamental_fields,
        )

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

            # ── MENA Pipeline Override ────────────────────────────────────────
            # TradingView cache provides real fundamentals for regional stocks.
            # Read directly from _cache_row (avoids None-override bug in fund merge).
            # If ≥4 valid TV fields present → force "medium" coverage.
            if _cache_row and target.upper().endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA", ".BH", ".MA", ".TN")) and _data_coverage_level in ("technical_only", "low"):
                import math as _math_mpo
                def _mpo_valid(v):
                    try:
                        f = float(v)
                        return not (_math_mpo.isnan(f) or _math_mpo.isinf(f)) and f != 0
                    except (TypeError, ValueError):
                        return False
                _cr_sector = str(_cache_row.get("sector") or "").lower()
                _tv_valid = sum(1 for v in [
                    _cache_row.get("price_earnings_ttm")   if _mpo_valid(_cache_row.get("price_earnings_ttm")) else None,
                    _cache_row.get("earnings_per_share_diluted_ttm") if _mpo_valid(_cache_row.get("earnings_per_share_diluted_ttm")) else None,
                    _cache_row.get("market_cap_basic")     if _mpo_valid(_cache_row.get("market_cap_basic")) else None,
                    _cr_sector                             if _cr_sector not in ("", "unknown", "n/a") else None,
                    _cache_row.get("dividend_yield_recent") if _mpo_valid(_cache_row.get("dividend_yield_recent")) else None,
                    _cache_row.get("SMA200")               if _mpo_valid(_cache_row.get("SMA200")) else None,
                    _cache_row.get("RSI")                  if _mpo_valid(_cache_row.get("RSI")) else None,
                ] if v is not None)
                if _tv_valid >= 4:
                    _data_coverage_level = "medium"
                    _low_data_compact_mode = False
                    logger.info("[DataCoverage] %s: pipeline cache %d fields → medium coverage", target, _tv_valid)

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

            # ── MENA Pipeline Override ────────────────────────────────────────
            # TradingView cache provides real fundamentals for regional stocks.
            # Read directly from _cache_row (avoids None-override bug in fund merge).
            # If ≥4 valid TV fields present → force "medium" coverage.
            if _cache_row and target.upper().endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA", ".BH", ".MA", ".TN")) and _data_coverage_level in ("technical_only", "low"):
                import math as _math_mpo2
                def _mpo2_valid(v):
                    try:
                        f = float(v)
                        return not (_math_mpo2.isnan(f) or _math_mpo2.isinf(f)) and f != 0
                    except (TypeError, ValueError):
                        return False
                _cr2_sector = str(_cache_row.get("sector") or "").lower()
                _tv_valid2 = sum(1 for v in [
                    _cache_row.get("price_earnings_ttm")   if _mpo2_valid(_cache_row.get("price_earnings_ttm")) else None,
                    _cache_row.get("earnings_per_share_diluted_ttm") if _mpo2_valid(_cache_row.get("earnings_per_share_diluted_ttm")) else None,
                    _cache_row.get("market_cap_basic")     if _mpo2_valid(_cache_row.get("market_cap_basic")) else None,
                    _cr2_sector                            if _cr2_sector not in ("", "unknown", "n/a") else None,
                    _cache_row.get("dividend_yield_recent") if _mpo2_valid(_cache_row.get("dividend_yield_recent")) else None,
                    _cache_row.get("SMA200")               if _mpo2_valid(_cache_row.get("SMA200")) else None,
                    _cache_row.get("RSI")                  if _mpo2_valid(_cache_row.get("RSI")) else None,
                ] if v is not None)
                if _tv_valid2 >= 4:
                    _data_coverage_level = "medium"
                    _low_data_compact_mode = False
                    logger.info("[DataCoverage] %s: pipeline cache %d fields → medium coverage", target, _tv_valid2)

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
                from core.agents.finance import _fetch_onchain as _onchain_fn
                _onchain_data = _onchain_fn(target)
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
| 🚀 Bull Market Rally (+20%) | {(20 * float(_effective_beta)):.1f}% | {_currency_sym}{(real_price or 0) * (1 + (20 * float(_effective_beta))/100):.2f} | Hold / add on dips |
| 💡 Fed Pivot / Rate Cut (+15%) | {(15 * float(_effective_beta)):.1f}% | {_currency_sym}{(real_price or 0) * (1 + (15 * float(_effective_beta))/100):.2f} | Growth + Tech |
| 📉 AI/Tech Slowdown (-20%) | {(-20 * float(_effective_beta)):.1f}% | {_currency_sym}{(real_price or 0) * (1 + (-20 * float(_effective_beta))/100):.2f} | Healthcare + Staples |
| 🏦 Fed Rate Shock +2% (-18%) | {(-18 * float(_effective_beta)):.1f}% | {_currency_sym}{(real_price or 0) * (1 + (-18 * float(_effective_beta))/100):.2f} | Value stocks + Cash |
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
        from core.agents.finance import _compute_decision_confidence as _cdc_fn
        _decision_conf = _cdc_fn(
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
            from core.agents.finance import (
                _soften_execution_language as _soften_fn,
                _round_scenario_prices as _round_fn,
            )
            deepseek_reply = _soften_fn(deepseek_reply)
            deepseek_reply = _round_fn(deepseek_reply, _currency_sym)
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
                    # Use the canonical EisaX score that the scorecard
                    # already computed (`final`) instead of digging it back
                    # out of the `result` tuple — the previous nested-getattr
                    # expression silently fell through to 0 on every run, so
                    # score_history was full of zero rows that broke the
                    # velocity / trend chart calculations.
                    _sc_fund  = int(final) if 'final' in dir() and final else 0
                    _sc_blend = int(sc_data.get("blended_score") or final or 0)
                    _sc_tech  = int(sc_data.get("tech_score") or 0)
                    if _sc_fund > 0 or _sc_blend > 0:
                        _log_score(target, _sc_fund, _sc_tech, _sc_blend, _pred_verdict)
                    else:
                        logger.debug(f"[PredTracker] skipping log_score for {target}: zero scores")

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

                # Save to brain (raw — Layer 2, source of truth)
                self._save_to_brain(
                    target, reply, real_price, analyst_target, fund, news_sent,
                    verdict=verdict_sc if 'verdict_sc' in dir() else None,
                    currency_sym=_currency_sym if '_currency_sym' in dir() else "$",
                )

                # Rule-based editorial — instant, no LLM, safe for main response
                try:
                    from core.editorial import rule_based_clean as _editorial_rule
                    _raw_len = len(reply)
                    reply = _editorial_rule(reply)
                    logger.info(
                        "[editorial] editorial_mode=rule_based_only raw_len=%d "
                        "rule_clean_len=%d delta=%d llm_skipped=true endpoint=_handle_analytics",
                        _raw_len, len(reply), _raw_len - len(reply),
                    )
                except Exception as _ed_err:
                    logger.debug("[editorial.rule] skipped: %s", _ed_err)

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


