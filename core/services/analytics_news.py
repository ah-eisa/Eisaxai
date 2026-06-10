"""
core/services/analytics_builder.py
────────────────────────────────────
Complex logic extracted from FinanceAgent._handle_analytics.

Public API
──────────
    enrich_after_fetch(target, fr) -> dict
        Derives analyst targets, beta, energy/crypto flags, fair value, etc.

    collect_news_waterfall(target, fr, dc_data, fund) -> tuple[list, str, float]
        Full news collection with 8+ fallback levels.

    build_data_block(target, fr, ctx, original_target=None) -> str
        Builds the structured text block for the LLM.

    build_analytics_prompt(target, data_block, ctx, scorecard_verdict_hint,
                           is_arabic, brain_ctx, local_injection,
                           research_summary, original_target=None,
                           macro_block="") -> str
        Builds the full DeepSeek prompt.

    assemble_report(target, fr, ctx, deepseek_reply, news_block, pos,
                    pre_scorecard_md, original_target=None) -> str
        Assembles the final markdown report.
"""

from __future__ import annotations

import logging
import math
import os
import re as _re

from core.services.data_fetcher import FetchResult

logger = logging.getLogger(__name__)


# ── A. enrich_after_fetch ─────────────────────────────────────────────────────

def collect_news_waterfall(
    target: str,
    fr: FetchResult,
    dc_data: dict,
    fund: dict,
) -> tuple:
    """
    Full news collection pipeline with 8+ fallback levels.
    Returns (news_links, news_sent, news_score).
    """
    news_links: list = list(fr.news_links or [])
    _engine_news_data = fr.engine_news or {}

    news_sent  = fund.get("news_sentiment", "N/A")
    news_score = float(fund.get("news_score", 0.0) or 0.0)

    # ── Seed from engine news (inject at FRONT) ───────────────────────────────
    if _engine_news_data:
        try:
            from core.news_engine_client import format_news_links as _fmt_eng_links
            _eng_links = _fmt_eng_links(_engine_news_data)
            _seen_eng  = {n["url"] for n in news_links}
            _injected  = []
            for _el in _eng_links:
                if _el["url"] not in _seen_eng:
                    _injected.append(_el)
                    _seen_eng.add(_el["url"])
            news_links = _injected + news_links  # engine links at FRONT
            logger.info("[newsfall] %s: injected %d engine links", target, len(_injected))
        except Exception as _ene:
            logger.debug("[newsfall] engine news format failed: %s", _ene)

    # ── FMP fallback ─────────────────────────────────────────────────────────
    if not news_links:
        try:
            from core.realtime_data import get_live_news
            fmp_news = get_live_news(target, limit=4)
            for n in fmp_news:
                if n.get("headline") and n.get("url"):
                    news_links.append({"title": n["headline"][:120], "url": n["url"]})
        except Exception as _fmpe:
            logger.error("[newsfall] FMP news failed: %s", _fmpe)

    # ── Regional energy supplement ────────────────────────────────────────────
    _t_upper_news = target.upper()
    _ENERGY_PREFIXES2 = ("ADNOC", "ARAMCO", "2222", "TAQA", "DANA", "GAS", "OIL", "ENERG")
    _is_regional_energy = (
        _t_upper_news.endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
        and any(k in _t_upper_news for k in _ENERGY_PREFIXES2)
    )
    if _is_regional_energy and len(news_links) < 3:
        try:
            from core.realtime_data import get_live_news as _gln
            _region_q = (
                "Gulf oil energy OPEC Iran war 2026"
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
            logger.info("[newsfall] %s: supplemented with %d regional items", target, len(_geo_news))
        except Exception as _rne:
            logger.warning("[newsfall] regional supplement failed: %s", _rne)

    # ── Local non-energy: NewsAPI ─────────────────────────────────────────────
    _is_local_ticker = _t_upper_news.endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
    if _is_local_ticker and len(news_links) < 2:
        try:
            from core.realtime_data import get_live_news as _gln2
            _co_name = fund.get("company_name") or target.split(".")[0]
            _mkt_ctx = (
                "UAE" if _t_upper_news.endswith((".AE", ".DU"))
                else "Saudi Arabia" if _t_upper_news.endswith(".SR")
                else "Egypt" if _t_upper_news.endswith(".CA")
                else "Kuwait" if _t_upper_news.endswith(".KW")
                else "Qatar"
            )
            _ticker_base = target.split(".")[0]
            _local_news = _gln2(target, company_name=f"{_co_name}", limit=5)
            if len(_local_news) < 2:
                _local_news = _gln2(target, company_name=f"{_ticker_base} {_mkt_ctx}", limit=5)
            for n in _local_news:
                h = n.get("headline", "")
                u = n.get("url", "")
                if h and u and not any(x["title"] == h for x in news_links):
                    news_links.append({"title": h[:120], "url": u})
            if len(news_links) < 2:
                _sector = fund.get("sector", "") or "investment"
                _mkt_news = _gln2(target, company_name=f"{_sector} {_mkt_ctx} market 2026", limit=4)
                for n in _mkt_news:
                    h = n.get("headline", "")
                    u = n.get("url", "")
                    if h and u and not any(x["title"] == h for x in news_links):
                        news_links.append({"title": h[:120], "url": u})
            logger.info("[newsfall] %s: %d local news items", target, len(news_links))
        except Exception as _lne:
            logger.warning("[newsfall] local news failed: %s", _lne)

    # ── Serper last-resort ────────────────────────────────────────────────────
    if len(news_links) < 2:
        try:
            _serper_key = os.getenv("SERPER_API_KEY", "")
            if _serper_key:
                import requests as _req_serper
                _ticker_base_serper = target.split(".")[0]
                _co_name_serper = (
                    fund.get("company_name") or dc_data.get("company_name") or _ticker_base_serper
                )
                _is_gulf_ticker = _t_upper_news.endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
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
                    _ticker_base_serper.upper(),
                    _commodity_name_map.get(target.upper(), ""),
                )
                if _is_gulf_ticker:
                    _sq = (
                        f'"{_co_name_serper}" OR "{_ticker_base_serper}" أخبار stock news '
                        f'site:zawya.com OR site:gulfnews.com OR site:arabianbusiness.com'
                    )
                elif _serper_commodity:
                    _sq = f"{_serper_commodity} market news 2026"
                else:
                    _sq = f'"{_co_name_serper}" stock news {(fund.get("sector", "") or "")}'
                _sr = _req_serper.post(
                    "https://google.serper.dev/news",
                    headers={"X-API-KEY": _serper_key, "Content-Type": "application/json"},
                    json={"q": _sq, "num": 6},
                    timeout=8,
                )
                if _sr.status_code == 200:
                    for _sn in _sr.json().get("news", []):
                        _sh = _sn.get("title", "")
                        _su = _sn.get("link", "")
                        if _sh and _su and not any(x["title"] == _sh for x in news_links):
                            news_links.append({"title": _sh[:120], "url": _su})
                    logger.info("[newsfall] %s: Serper got %d items", target, len(news_links))
        except Exception as _sne:
            logger.warning("[newsfall] Serper failed: %s", _sne)

    # ── EisaX Aggregator final fallback ───────────────────────────────────────
    if len(news_links) < 2:
        try:
            from core.news_aggregator import get_news as _agg_news
            _agg = _agg_news(ticker=target, limit=5)
            for _an in _agg:
                _at = _an.get("title", "")
                _au = _an.get("url", "")
                if _at and _au and not any(x["title"] == _at for x in news_links):
                    news_links.append({"title": _at[:120], "url": _au})
            logger.info("[newsfall] %s: aggregator got %d items", target, len(news_links))
        except Exception as _age:
            logger.warning("[newsfall] aggregator failed: %s", _age)

    # ── Relevance filter ──────────────────────────────────────────────────────
    def _is_relevant_news(title: str, ticker_str: str, company: str) -> bool:
        if not title:
            return False
        t_low  = title.lower()
        tk_low = ticker_str.lower().split(".")[0]
        co_low = (company or "").lower()

        _noise_sources = [
            "wallstreetbets", "reddit", "r/stocks", "memestocks",
            "mcdonald's", "mcdonalds", "coca-cola", "coca cola",
            "unrelated_company",
        ]
        if any(n in t_low for n in _noise_sources):
            return False

        _tk_clean = tk_low.split("=")[0]
        if tk_low and len(tk_low) > 2 and tk_low in t_low:
            return True
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
            first_word = co_low.split()[0]
            if len(first_word) > 3 and first_word in t_low:
                return True

        _t_sector = (fund.get("sector") or "").lower()
        _sector_keys = {
            "energy":      ["oil", "opec", "brent", "crude", "gas", "lng", "iran", "hormuz"],
            "technology":  ["ai", "semiconductor", "tech", "chip", "cloud", "software"],
            "real estate": ["real estate", "property", "reit", "mortgage", "housing"],
            "financials":  ["bank", "lending", "fed", "rate", "credit", "loan"],
            "crypto":      ["bitcoin", "btc", "crypto", "ethereum", "blockchain"],
            "commodit":    ["gold", "xau", "bullion", "silver", "precious metal", "oil", "brent", "crude", "commodity"],
            "precious":    ["gold", "xau", "bullion", "silver", "platinum", "palladium", "precious metal"],
        }
        for sec, keys in _sector_keys.items():
            if sec in _t_sector:
                if any(k in t_low for k in keys):
                    return True

        _broad_ok = ["earnings", "revenue", "ipo", "dividend", "buyback",
                     "forecast", "outlook", "guidance", "acquisition", "merger"]
        if any(k in t_low for k in _broad_ok):
            if tk_low and tk_low in t_low:
                return True
            if co_low and len(co_low.split()[0]) > 3 and co_low.split()[0] in t_low:
                return True

        return False

    _co_name_for_filter = fund.get("company_name", target)
    _orig_count = len(news_links)
    news_links = [
        n for n in news_links
        if _is_relevant_news(n.get("title", ""), target, _co_name_for_filter)
    ]
    if len(news_links) < _orig_count:
        logger.info("[newsfall] %s: filtered %d irrelevant, kept %d",
                    target, _orig_count - len(news_links), len(news_links))

    # ── Post-filter Serper rescue ─────────────────────────────────────────────
    if len(news_links) == 0:
        try:
            _serper_key2 = os.getenv("SERPER_API_KEY", "")
            if _serper_key2:
                import requests as _req_s2
                _tb2  = target.split(".")[0]
                _cn2  = fund.get("company_name") or dc_data.get("company_name") or _tb2
                _gulf2 = _t_upper_news.endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA"))
                _sq2 = (
                    f'"{_cn2}" OR "{_tb2}" stock news zawya arabianbusiness 2026'
                    if _gulf2
                    else f'"{_cn2}" stock news 2026'
                )
                _sr2 = _req_s2.post(
                    "https://google.serper.dev/news",
                    headers={"X-API-KEY": _serper_key2, "Content-Type": "application/json"},
                    json={"q": _sq2, "num": 6},
                    timeout=8,
                )
                if _sr2.status_code == 200:
                    for _sn2 in _sr2.json().get("news", []):
                        _sh2 = _sn2.get("title", "")
                        _su2 = _sn2.get("link", "")
                        if _sh2 and _su2:
                            news_links.append({"title": _sh2[:120], "url": _su2})
                    logger.info("[newsfall] %s: post-filter Serper rescue: %d items", target, len(news_links))
        except Exception as _sne2:
            logger.debug("[newsfall] post-filter Serper rescue: %s", _sne2)

    # ── EisaX Aggregator post-filter rescue ───────────────────────────────────
    if len(news_links) == 0:
        try:
            from core.news_aggregator import get_news as _agg_news2
            _agg2 = _agg_news2(ticker=target, limit=5)
            for _an2 in _agg2:
                _at2 = _an2.get("title", "")
                _au2 = _an2.get("url", "")
                if _at2 and _au2:
                    news_links.append({"title": _at2[:120], "url": _au2})
            logger.info("[newsfall] %s: aggregator rescue: %d items", target, len(news_links))
        except Exception as _age2:
            logger.warning("[newsfall] aggregator rescue failed: %s", _age2)

    return (news_links, news_sent, news_score)

