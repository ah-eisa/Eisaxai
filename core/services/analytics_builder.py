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

from core.services.analytics_enricher import enrich_after_fetch
from core.services.analytics_news import collect_news_waterfall
from core.services.analytics_data import build_data_block, build_analytics_prompt

__all__ = ['enrich_after_fetch','collect_news_waterfall','build_data_block',
           'build_analytics_prompt','assemble_report','_post_render_cleanup']

def _post_render_cleanup(report: str) -> str:
    """
    Fix common markdown rendering issues in the assembled report.

    1. Section header spacing: "### 1.Executive" -> "### 1. Executive"
    2. Suppress empty emoji bullets: lines containing only a lone emoji bullet
    3. Collapse duplicate --- separators (3+ consecutive -> 2)
    """
    import re as _re_cl

    # Fix 1: add space after digit-period in section headers (e.g. "### 1.Executive")
    report = _re_cl.sub(r'(#{1,6}\s*\d+\.)(\S)', r'\1 \2', report)

    # Fix 2: suppress lines that are just an emoji bullet with no text
    # Matches lines that are only whitespace + emoji + optional whitespace
    report = _re_cl.sub(
        r'(?m)^[ \t]*[*\-]?\s*\U0001f4a1\s*$\n?',
        '',
        report,
    )

    # Fix 3: collapse 3+ consecutive --- separator lines into exactly 2
    report = _re_cl.sub(r'(\n---){3,}', '\n---\n---', report)

    # Fix 4: ensure blank line before markdown section headers (## / ###)
    # so that sections never run into the preceding paragraph's last line.
    # Only adds \n when the preceding line is non-empty (not already separated).
    report = _re_cl.sub(r'(?m)([^\n])(\n)(#{1,6} )', r'\1\n\n\3', report)

    return report

def assemble_report(
    target: str,
    fr: FetchResult,
    ctx: dict,
    deepseek_reply: str,
    news_block: str,
    pos: dict,
    pre_scorecard_md: str,
    original_target: str | None = None,
) -> str:
    """
    Assembles the final markdown report from all pieces.
    Extracted from _handle_analytics lines 3862–4403.
    Does NOT set state.last_artifact or call _save_to_brain — caller is responsible.
    """
    import math as _math_pos
    import re as _re_rep

    fund       = fr.fund or {}
    summary    = fr.summary or {}
    var_95     = fr.var_95 or 0.02
    max_dd     = fr.max_dd or 0.20
    real_price = fr.real_price
    change_pct = fr.change_pct or 0.0

    currency_sym = ctx.get("currency_sym", "$")
    currency_lbl = ctx.get("currency_lbl", "USD")
    _is_local_mkt = currency_lbl != "USD"
    _is_local_currency = currency_lbl in ("SAR", "AED", "EGP", "KWF", "QAR")
    is_energy    = ctx.get("is_energy", False)
    oil_data     = ctx.get("oil_data", {}) or {}
    is_crash     = ctx.get("is_crash", False)
    effective_beta = ctx.get("effective_beta", 1.0)
    display_target = ctx.get("display_target")
    target_is_estimate = ctx.get("target_is_estimate", True)
    _is_regional_energy = ctx.get("is_regional_energy", False)
    _is_local_ticker    = ctx.get("is_local_ticker", False)
    x_data       = fr.x_data or {}
    _engine_news_data = fr.engine_news or {}

    _fallback_price = real_price or summary.get("price", 0)
    price_str = (
        f"{_fallback_price:,.2f} {currency_sym} ({change_pct:+.2f}%)"
        if _fallback_price and _is_local_mkt and change_pct
        else f"{_fallback_price:,.2f} {currency_sym}"
        if _fallback_price and _is_local_mkt
        else f"${_fallback_price:,.2f} ({change_pct:+.2f}%)"
        if _fallback_price and change_pct
        else f"${_fallback_price:,.2f}"
        if _fallback_price else "N/A"
    )
    _t_upper = target.upper()

    # ── EisaX score from scorecard markdown ──────────────────────────────────
    _eisax_score_match = _re_rep.search(r"EisaX Score:\s*\*\*(\d+)/100\*\*", pre_scorecard_md)
    _eisax_score = _eisax_score_match.group(1) if _eisax_score_match else "N/A"

    _exch_label = (
        "🇸🇦 Tadawul · SAR" if _t_upper.endswith(".SR") else
        "🇦🇪 ADX/DFM · AED" if _t_upper.endswith((".AE", ".DU")) else
        "🇪🇬 EGX · EGP" if _t_upper.endswith(".CA") else
        "🇰🇼 Boursa Kuwait · KWF" if _t_upper.endswith(".KW") else
        "🇶🇦 Qatar Exchange · QAR" if _t_upper.endswith(".QA") else ""
    )
    _oil_badge = f" | **🛢️ Brent: ${oil_data.get('price',0):.2f}**" if is_energy and oil_data.get("price") else ""
    _display_ticker = original_target if (original_target and original_target != target) else target

    header = (
        f"# EisaX Intelligence Report: {_display_ticker}\n\n"
        f"**🔴 Live Price:** {price_str} | "
        f"**Sector:** {fund.get('sector', 'N/A')} | "
        f"**EisaX Score:** {_eisax_score}/100"
        + (f" | **{_exch_label}**" if _exch_label else "")
        + _oil_badge
        + "\n\n---\n\n"
    )

    # ── Chart placeholder ─────────────────────────────────────────────────────
    chart_block = (
        f'\n\n---\n📈 **Price Chart (60 days)**\n'
        f'<div class="eisax-chart" data-ticker="{target}"></div>'
    )

    # ── Disclaimer ────────────────────────────────────────────────────────────
    _analysis_disclaimer = (
        "\n\n---\n"
        "> ⚠️ **Disclaimer:** This report is generated by EisaX AI and is for informational purposes only. "
        "It does not constitute financial advice, investment recommendation, or an offer to buy or sell any security. "
        "All prices and data are fetched live at the time of the query and may not reflect real-time market conditions. "
        "Past performance is not indicative of future results. Always verify data independently and consult a licensed financial advisor before making investment decisions."
    )

    # ── Positioning block ─────────────────────────────────────────────────────
    def _clean(v, d=0.0):
        try:
            f = float(v or 0)
            return d if (_math_pos.isnan(f) or _math_pos.isinf(f)) else f
        except Exception:
            logger.debug("[positioning] _clean: cannot coerce %r to float — using default %r", v, d, exc_info=True)
            return d

    sma50  = _clean(summary.get("sma_50", 0))
    sma200 = _clean(summary.get("sma_200", 0))
    ep     = pos.get("ep")
    sp     = pos.get("sp")

    _fp_ref = _clean(real_price or _fallback_price or 0)
    if _fp_ref and sma200:
        _pct_from_sma = (_fp_ref - sma200) / sma200
        if _pct_from_sma < -0.10:
            entry_price = _fp_ref * 0.97
            stop_price  = _fp_ref * 0.91
        elif _pct_from_sma < 0:
            entry_price = sma200 * 0.98
            stop_price  = sma200 * 0.92
        else:
            entry_price = ep if ep else sma200 * 1.01
            stop_price  = sp if sp else sma200 * 0.95
    else:
        entry_price = ep if ep else (_fp_ref * 0.96 if _fp_ref else None)
        stop_price  = sp if sp else (_fp_ref * 0.91 if _fp_ref else None)

    if _fp_ref and entry_price and entry_price >= _fp_ref:
        entry_price = _fp_ref * 0.97
        stop_price  = _fp_ref * 0.91

    # Validate: stop must be below entry for a long trade
    from core.services.scorecard_engine import validate_positioning as _validate_pos
    if entry_price and stop_price:
        entry_price, stop_price, _pos_fixed, _pos_note = _validate_pos(
            entry_price, stop_price, _fp_ref
        )

    def _fmt_price(p):
        if not p:
            return "N/A"
        return f"{p:,.2f} {currency_sym}" if _is_local_mkt else f"${p:,.2f}"

    entry_level = _fmt_price(entry_price)
    stop_level  = _fmt_price(stop_price)
    _pos_target = display_target
    _rp_pos = real_price or _fallback_price or 0

    _target_is_sma = False
    if _pos_target and _rp_pos:
        upside = ((_pos_target / _rp_pos) - 1) * 100
        target_level = (
            f"{_pos_target:,.2f} {currency_sym} ({upside:+.1f}%)"
            if _is_local_mkt
            else f"${_pos_target:,.2f} ({upside:+.1f}%)"
        )
    elif sma200 and _rp_pos:
        if _rp_pos < sma200:
            _tech_tgt = sma200 * 1.15
        elif sma50 and _rp_pos < sma50:
            _tech_tgt = sma50
        else:
            _tech_tgt = sma200 * 1.15
        _sma_used = "SMA50" if (sma50 and _rp_pos < sma50) else "SMA200"
        _tech_up  = ((_tech_tgt / _rp_pos) - 1) * 100
        target_level = (
            f"{_tech_tgt:,.2f} {currency_sym} ({_tech_up:+.1f}%)"
            if _is_local_mkt
            else f"${_tech_tgt:,.2f} ({_tech_up:+.1f}%)"
        )
        _target_is_sma = True
    elif sma50 and _rp_pos:
        _tech_tgt = sma50 if _rp_pos < sma50 else sma50 * 1.05
        _tech_up  = ((_tech_tgt / _rp_pos) - 1) * 100
        _sma_used = "SMA50"
        target_level = (
            f"{_tech_tgt:,.2f} {currency_sym} ({_tech_up:+.1f}%)"
            if _is_local_mkt
            else f"${_tech_tgt:,.2f} ({_tech_up:+.1f}%)"
        )
        _target_is_sma = True
    else:
        target_level = "N/A"

    if "_sma_used" not in dir():
        _sma_used = "SMA50" if (sma50 and not sma200) else "SMA200"

    _target_rationale = (
        f"⚠️ Technical target ({_sma_used} mean-reversion) — not analyst" if _target_is_sma
        else "⚠️ EisaX FV Estimate (no analyst coverage)" if target_is_estimate
        else "Analyst consensus target"
    )
    _rp_pos2 = real_price or _fallback_price or 0
    _stop_rationale = (
        "Below SMA200 (-5%)"
        if stop_price and sma200 and abs(stop_price - sma200 * 0.95) / (sma200 * 0.95) < 0.03
        else "Trailing stop (-9% from current)"
        if _rp_pos2 and stop_price and stop_price >= _rp_pos2 * 0.88
        else "Key support level (-9% from current)"
    )
    _rp_pos3 = real_price or _fallback_price or 0
    if entry_price and _rp_pos3 and _rp_pos3 > entry_price * 1.02:
        _pct_to_entry = ((_rp_pos3 - entry_price) / _rp_pos3) * 100
        _entry_note = (
            f"\n\n> ⏳ **Awaiting Pullback** — Current price "
            f"({_fmt_price(_rp_pos3)}) is **{_pct_to_entry:.1f}% above** the entry zone "
            f"({_fmt_price(entry_price)}), which reduces the margin of safety relative to the defined risk parameters."
        )
    else:
        _entry_note = ""

    _entry_rationale = (
        "Near SMA200 support"
        if entry_price and sma200 and abs(entry_price - sma200) / sma200 < 0.05
        else "Pullback entry — below current price"
        if entry_price and _rp_pos3 and entry_price < _rp_pos3 * 0.98
        else "At current price — entry zone active"
    )

    positioning_block = (
        f"\n\n---\n"
        f"📊 **Positioning Guide**\n"
        f"| | Level | Rationale |\n"
        f"|---|---|---|\n"
        f"| 🟢 Entry | {entry_level} | {_entry_rationale} |\n"
        f"| 🎯 Target | {target_level} | {_target_rationale} |\n"
        f"| 🔴 Stop | {stop_level} | {_stop_rationale} |\n"
        f"{_entry_note}"
    )

    # ── Assemble with DeepSeek reply ──────────────────────────────────────────
    if deepseek_reply:
        try:
            from core.fact_checker import FactChecker
            fact_data = {**summary, "price": real_price or summary.get("price")}
            factcheck_block = FactChecker().verify_analysis(target, fact_data)
        except Exception as _fce:
            logger.error("[assemble] FactChecker failed: %s", _fce)
            factcheck_block = ""
        _report = (
            header
            + deepseek_reply
            + factcheck_block
            + news_block
            + positioning_block
            + pre_scorecard_md
            + chart_block
            + _analysis_disclaimer
        )
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
            _report = ReportEnhancer(_qe).enhance(_report, ticker=target)
            logger.info("[assemble] Enhancer applied to %s", target)
        except Exception as _enh_err:
            logger.warning("[assemble] Enhancer skipped for %s: %s", target, _enh_err)
        return _post_render_cleanup(_report)

    # ── Fallback: structured reply without DeepSeek ───────────────────────────
    def _P(n):
        return f"{n:.1f}%" if n else "N/A"

    def _X(n):
        return f"{n:.1f}x" if n else "N/A"

    def _B(n):
        try:
            if not n: return "N/A"
            v = float(n)
            if currency_lbl != "USD":
                if v >= 1e9:  return f"{v/1e9:.1f}B {currency_sym}"
                if v >= 1e6:  return f"{v/1e6:.0f}M {currency_sym}"
                return f"{v:,.0f} {currency_sym}"
            return f"${v/1e9:.1f}B" if v >= 1e9 else f"${v/1e6:.0f}M"
        except Exception:
            logger.debug("[fallback_report] _B: format error for value %r", n, exc_info=True)
            return "N/A"

    verdict = (
        "ACCUMULATE" if summary.get("trend") == "Bullish" and summary.get("momentum") == "Bullish"
        else "REDUCE" if summary.get("trend") == "Bearish" and summary.get("momentum") == "Bearish"
        else "HOLD"
    )
    reply = (
        header
        + f"## Fundamentals\n"
        f"- Revenue Growth: {_P(fund.get('revenue_growth'))} | EPS Growth: {_P(fund.get('eps_growth'))}\n"
        f"- Net Margin: {_P(fund.get('net_margin'))} | ROE: {_P(fund.get('roe'))}\n"
        f"- P/E: {_X(fund.get('pe_ratio'))} | EV/EBITDA: {_X(fund.get('ev_ebitda'))}\n"
        f"- Market Cap: {_B(fund.get('market_cap'))} | Cash: {_B(fund.get('cash'))}\n\n"
        f"## Technicals\n"
        f"- Trend: {summary.get('trend','N/A')} | RSI: {summary.get('rsi',50):.1f} | MACD: {summary.get('momentum','N/A')}\n"
        f"- VaR(95%): {var_95*100:.2f}% | Max DD: {max_dd*100:.2f}%\n\n"
        f"**EisaX Verdict: {verdict}**"
    )
    try:
        from core.fact_checker import FactChecker
        fact_data = {**summary, "price": real_price or summary.get("price")}
        fact_block = FactChecker().verify_analysis(target, fact_data)
        reply += "\n\n" + fact_block
    except Exception as _fce2:
        logger.error("[assemble] FactChecker (fallback) failed: %s", _fce2)
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
        logger.info("[assemble] Enhancer applied to %s", target)
    except Exception as _enh_err:
        logger.warning("[assemble] Enhancer skipped for %s: %s", target, _enh_err)
    return _post_render_cleanup(reply)

