"""
core/services/market_updates.py
────────────────────────────────
EisaX Market Updates — institutional-grade daily pulse + weekly strategy brief.

Public API (unchanged signatures):
    generate_daily_update()  -> dict
    generate_weekly_update() -> dict
    get_latest_updates()     -> dict
    format_for_linkedin(update_json: dict) -> str

New helpers (internal):
    build_eisax_stance(moves, regime, fg)   -> dict
    build_invalidation_logic(moves, regime) -> list
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path("/home/ubuntu/investwise/data/market_updates.db")
_MARKET_SNAPSHOT_PATH = Path("/home/ubuntu/investwise/data/market_updates_snapshot.json")
_MARKET_CACHE_TTL = timedelta(minutes=15)
_MARKET_CACHE: dict[str, Any] = {
    "lookback_days": None,
    "fetched_at": None,
    "data_timestamp": None,
    "data": None,
}
_LAST_MARKET_DATA_TIMESTAMP: Optional[str] = None

# ── Benchmarks ────────────────────────────────────────────────────────────────

_BENCHMARKS = {
    "SPY":      "S&P 500",
    "QQQ":      "Nasdaq 100",
    "^VIX":     "VIX",
    "GLD":      "Gold",
    "SLV":      "Silver",
    "USO":      "Oil (WTI)",
    "BTC-USD":  "Bitcoin",
    "^TNX":     "10Y Treasury Yield",
    "UUP":      "US Dollar (DXY)",
    "^TASI":    "Saudi Market Composite",
    "^DFMGI":   "UAE Market Composite",
    "EGX30.CA": "Egypt Market Composite",
}

_PIPELINE_REGIONAL_BENCHMARKS = {
    "^TASI": {"market": "ksa", "label": "Saudi Market Composite"},
    "^DFMGI": {"market": "uae", "label": "UAE Market Composite"},
    "EGX30.CA": {"market": "egypt", "label": "Egypt Market Composite"},
}

_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")


# ── Storage ───────────────────────────────────────────────────────────────────

# Re-export sub-modules (explicit — supports underscore names)
from core.services.market_db import (
    _init_db, _save_update, _get_latest, _utc_now_iso, _set_market_cache, _get_cached_market_data,
    _persist_last_good_snapshot, _load_last_good_snapshot, _get_market_data_timestamp,
)

from core.services.market_collector import (
    _weighted_average, _load_pipeline_regional_moves, _collect_market_data, _get_fear_greed, _get_recent_sentiment_summary, _determine_regime,
    _determine_regime_confidence, build_eisax_stance, _build_asset_allocation_view, build_invalidation_logic, _build_cross_asset_snapshot, _call_openai,
    _call_openai_text, _call_gemini, _generate_insight,
)

from core.services.market_report import (
    _daily_decision_type, _daily_positioning_mode, _daily_confidence_score, _daily_market_state, _generate_full_report_text, _generate_cio_daily_report_text,
    _generate_linkedin_text_ai, _trim_to_words, _fit_word_window, _clean_text, _clean_text_list, _merge_text_list_with_fallback,
    _normalize_key_moves, _normalize_winners_losers, _normalize_regional_view, _snapshot_brief, _format_report_number, _trigger_hierarchy_lines,
    _best_expression_line, _best_hedge_line, _weekly_why_now_lines, _market_view_lines, _enrich_full_report, _allocation_summary,
    _as_number, _daily_linkedin_hook, _weekly_linkedin_hook, _weekly_stance_label, _daily_positioning_line, _weekly_positioning_line,
    _daily_linkedin_insight_lines, _weekly_linkedin_insight_lines, _weekly_focus_text, _build_web_version,
)

from core.services.market_regional import (
    _load_pipeline_market_frame, _market_top_sector, _market_top_movers, _build_daily_regional_internals, _load_pipeline_snapshot_series, _dfm_real_estate_weighted_change,
    _commodities_wti_change, _compute_gcc_decoupling_signal, _translate_risk_trigger_ar, _translate_phrase_ar, _translate_catalyst_ar, _ordered_daily_catalysts,
    _spy_range_levels, _format_internal_line_en, _format_internal_line_ar, _build_full_report_fallback, _build_cio_daily_report_fallback,
)

from core.services.market_arabic import (
    _build_linkedin_text, _build_linkedin_text_v2, _deterministic_daily, _deterministic_weekly, _enforce_cio_daily_language, _weekly_decision_type,
    _weekly_positioning_mode, _weekly_confidence_score, _weekly_market_state, _enforce_cio_weekly_language, _build_cio_weekly_report_fallback, _ar_label,
    _ar_num, _build_cio_daily_report_ar, _build_cio_daily_report_fallback_v2, _build_cio_daily_report_ar_v2, _build_cio_weekly_report_ar, _apply_daily_consistency,
    _apply_weekly_consistency, _generate_arabic_report, _daily_snapshot_internal_lines_en, _daily_snapshot_internal_lines_ar, _daily_snapshot_pairs_en, _daily_snapshot_pairs_ar,
    _build_cio_daily_report_fallback_v3, _build_cio_daily_report_ar_v3, _finalize_daily_update, _finalize_weekly_update,
)

def generate_daily_update() -> dict:
    """Generate today's EisaX Daily Market Pulse. Saves to DB and returns structured JSON."""
    logger.info("[market_updates] Generating daily update")
    moves    = _collect_market_data(lookback_days=10)
    fg       = _get_fear_greed()
    senti    = _get_recent_sentiment_summary()
    regime   = _determine_regime(moves)
    conf     = _determine_regime_confidence(moves, regime)
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stance   = build_eisax_stance(moves, regime, fg)
    invali   = build_invalidation_logic(moves, regime)
    snapshot = _build_cross_asset_snapshot(moves)
    fallback = _deterministic_daily(moves, regime, fg)
    fallback["cross_asset_snapshot"] = snapshot

    moves_summary = {
        k: {"label": v["label"], "d1_pct": v["d1_pct"], "d5_pct": v["d5_pct"], "price": v.get("price")}
        for k, v in moves.items()
    }

    prompt = f"""You are EisaX — institutional AI investment intelligence used by portfolio managers.
Generate a Daily Market Pulse as valid JSON only. Institutional tone. Direct. No generic phrases.

Market data ({today}):
{json.dumps(moves_summary, indent=2)}

Fear & Greed: {fg.get('score', 50)} ({fg.get('rating', 'Neutral')})
Sentiment (24h): {json.dumps(senti)}
Regime: {regime} (confidence: {conf})
Pre-computed stance: {json.dumps(stance)}
Pre-computed invalidation: {json.dumps(invali)}

Return ONLY this JSON (no markdown fences):
{{
  "date": "{today}",
  "market_regime": "{regime}",
  "regime_confidence": "{conf}",
  "what_matters_now": [
    "<institutional interpretation — NOT news — cross-asset relationship that changes positioning>",
    "<what the regime signal means for risk allocation right now>",
    "<one specific observation that retail misses but institutions act on>"
  ],
  "key_moves": [
    {{"asset": "<name>", "move": "<\u00b1X.X% (1d)>", "reason": "<WHY it moved — specific causation, not description>"}},
    "... 3-5 most significant moves"
  ],
  "eisax_view": {{
    "stance": "<Tactical BUY|HOLD|REDUCE RISK>",
    "overweight_assets": ["<asset1>", "<asset2>"],
    "underweight_assets": ["<asset1>"],
    "neutral_assets": ["<asset1>", "<asset2>"],
    "focus": "<primary focus in 4-6 words>",
    "horizon": "<short-term|tactical|swing|defensive>"
  }},
  "why_now": "<2 sentences MAX: the SPECIFIC setup making this moment actionable. Zero filler.>",
  "what_invalidates": [
    "<price level or macro trigger that proves this view wrong — be specific>",
    "<second condition with number/level>",
    "<third — macro data point or Fed action>"
  ],
  "tactical_positioning": "<1-2 lines: what to DO with a portfolio today — name actual asset classes>",
  "next_triggers": ["<specific event/level/date>", "<specific event>", "<specific level>"],
  "fear_greed_index": {fg.get('score', 50)}
}}

Hard rules:
- NEVER write: "markets showed resilience" / "investor confidence increased" / "amid uncertainty"
- what_matters_now: INTERPRETATION not description — what does this MEAN for portfolios?
- why_now: explain the SPECIFIC setup, not the regime in general
- what_invalidates: include actual price numbers
- 120-180 words total across all text fields
"""

    ai_result = _generate_insight(prompt, max_tokens=1000)

    required = ("key_moves", "what_matters_now", "why_now", "what_invalidates", "tactical_positioning")
    if ai_result and all(k in ai_result for k in required):
        update = {
            "date":                 today,
            "market_regime":        regime,
            "regime_confidence":    ai_result.get("regime_confidence", conf),
            "what_matters_now":     ai_result.get("what_matters_now", [])[:3],
            "key_moves":            ai_result.get("key_moves", [])[:5],
            "cross_asset_snapshot": snapshot,
            "eisax_view":           stance,
            "why_now":              ai_result.get("why_now", ""),
            "what_invalidates":     invali[:4],
            "tactical_positioning": ai_result.get("tactical_positioning", ""),
            "next_triggers":        ai_result.get("next_triggers", [])[:3],
            "fear_greed_index":     fg.get("score"),
        }
    else:
        logger.warning("[market_updates] AI daily generation failed — using deterministic fallback")
        update = dict(fallback)

    update["eisax_view"] = stance
    update["what_invalidates"] = invali[:4]
    update = _apply_daily_consistency(update, fallback, snapshot)
    update = _enforce_cio_daily_language(update, moves, fg)
    update = _finalize_daily_update(update, moves_summary, fg)

    _save_update("daily", update)
    ev = update.get("eisax_view", {})
    logger.info("[market_updates] Daily saved: regime=%s stance=%s conf=%s",
                regime, ev.get("stance") if isinstance(ev, dict) else ev, conf)
    return update

def generate_weekly_update() -> dict:
    """Generate this week's EisaX Weekly Strategy Brief. Saves to DB and returns structured JSON."""
    logger.info("[market_updates] Generating weekly update")
    moves  = _collect_market_data(lookback_days=10)
    fg     = _get_fear_greed()
    senti  = _get_recent_sentiment_summary()
    regime = _determine_regime(moves)
    conf   = _determine_regime_confidence(moves, regime)

    now        = datetime.now(timezone.utc)
    week_start = (now - timedelta(days=7)).strftime("%b %d")
    week_end   = now.strftime("%b %d, %Y")
    week_range = f"{week_start}–{week_end}"

    stance = build_eisax_stance(moves, regime, fg)
    invali = build_invalidation_logic(moves, regime)
    fallback = _deterministic_weekly(moves, regime, fg)

    moves_summary = {
        k: {"label": v["label"], "d1_pct": v["d1_pct"], "d5_pct": v["d5_pct"], "price": v.get("price")}
        for k, v in moves.items()
    }

    prompt = f"""You are EisaX — institutional AI investment intelligence. Style: Goldman/Bridgewater strategy note. Decisive. No fluff.
Generate a Weekly Strategy Brief as valid JSON only.

Market data (week: {week_range}):
{json.dumps(moves_summary, indent=2)}

Fear & Greed: {fg.get('score', 50)} ({fg.get('rating', 'Neutral')})
Sentiment: {json.dumps(senti)}
Regime: {regime} (confidence: {conf})
Pre-computed stance: {json.dumps(stance)}
Pre-computed invalidation: {json.dumps(invali)}

Return ONLY this JSON (no markdown fences):
{{
  "week_range": "{week_range}",
  "market_summary": "<2-3 sharp sentences: what DROVE markets. Require cross-asset context. Institutional voice.>",
  "positioning": "<How portfolios should be positioned NOW — name overweight/underweight asset classes explicitly>",
  "asset_allocation_view": {{
    "equities": "<Overweight|Neutral|Underweight>",
    "crypto": "<Overweight|Neutral|Underweight>",
    "metals": "<Overweight|Neutral|Underweight>",
    "commodities": "<Overweight|Neutral|Underweight>",
    "cash": "<Overweight|Neutral|Underweight>"
  }},
  "regional_view": {{
    "US": "<sharp 1-sentence view on US equities — with stance>",
    "GCC": "<1-sentence view: oil linkage + TASI/DFM direction>",
    "Egypt": "<1-sentence EM view: dollar + commodity context>"
  }},
  "winners_losers": {{
    "winners": ["<asset> <\u00b1X.X%>", "..."],
    "losers":  ["<asset> <\u00b1X.X%>", "..."]
  }},
  "highest_conviction_opportunity": "<ONE specific trade idea with reasoning, entry context, and time horizon>",
  "key_risks": [
    "<specific risk + WHY it matters NOW — include data or levels>",
    "<second risk>",
    "<third risk>"
  ],
  "what_changes_this_view": ["<price/macro trigger with specific level>", "<second trigger>"],
  "portfolio_angle": "<2-3 sentences: what to DO across the book — use allocation language>",
  "eisax_verdict": "<1 sentence: sharp and actionable. Tell the reader exactly what to do.>"
}}

Rules:
- 250-350 words total across text fields
- NEVER use "markets showed resilience" or "investor confidence increased"
- highest_conviction_opportunity: SPECIFIC trade with SPECIFIC reasoning
- eisax_verdict: action verb first — "Reduce", "Add", "Hold", "Rotate"
- what_changes_this_view: must include price levels or specific data releases
- Present triggers as hierarchy: Primary trigger, Secondary, Tertiary
- Detailed market-by-market coverage will follow, so avoid repeating GCC/Egypt language in earlier sections
"""

    ai_result = _generate_insight(prompt, max_tokens=1400)

    required = ("market_summary", "eisax_verdict", "positioning", "highest_conviction_opportunity")
    if ai_result and all(k in ai_result for k in required):
        update = {
            "week_range":                     week_range,
            "market_summary":                 ai_result.get("market_summary", ""),
            "positioning":                    ai_result.get("positioning", ""),
            "asset_allocation_view":          _build_asset_allocation_view(regime),
            "regional_view":                  ai_result.get("regional_view", {}),
            "winners_losers":                 ai_result.get("winners_losers", {"winners": [], "losers": []}),
            "highest_conviction_opportunity": ai_result.get("highest_conviction_opportunity", ""),
            "key_risks":                      ai_result.get("key_risks", [])[:3],
            "what_changes_this_view":         invali[:4],
            "portfolio_angle":                ai_result.get("portfolio_angle", ""),
            "eisax_verdict":                  ai_result.get("eisax_verdict", ""),
            "fear_greed_index":               fg.get("score"),
        }
    else:
        logger.warning("[market_updates] AI weekly generation failed — using deterministic fallback")
        update = dict(fallback)

    update["asset_allocation_view"] = _build_asset_allocation_view(regime)
    update["what_changes_this_view"] = invali[:4]
    update = _apply_weekly_consistency(update, fallback)
    update["market_regime"] = regime
    update["regime_confidence"] = conf
    update["cross_asset_snapshot"] = update.get("cross_asset_snapshot") or fallback.get("cross_asset_snapshot", {})
    update = _enforce_cio_weekly_language(update, moves, fg)
    update = _finalize_weekly_update(update)

    _save_update("weekly", update)
    logger.info("[market_updates] Weekly saved: verdict=%s", update.get("eisax_verdict", "")[:60])
    return update

def format_for_linkedin(update: dict) -> str:
    return _build_linkedin_text_v2(update) if update else ""

def _linkedin_daily(u: dict) -> str:
    regime  = u.get("market_regime", "Cautious")
    date    = u.get("date", "")
    view    = u.get("eisax_view", {})
    stance  = view.get("stance", "HOLD")  if isinstance(view, dict) else str(view)
    focus   = view.get("focus", "")       if isinstance(view, dict) else ""
    horizon = view.get("horizon", "")     if isinstance(view, dict) else ""

    lines = [
        f"EisaX Daily Market Pulse — {regime}",
        date,
        "",
    ]

    for bullet in (u.get("what_matters_now") or [])[:2]:
        lines.append(f"• {bullet}")
    lines.append("")

    lines.append(f"Stance: {stance}" + (f" | Focus: {focus}" if focus else "") + (f" | Horizon: {horizon}" if horizon else ""))
    lines.append("")

    why = u.get("why_now", "")
    if why:
        lines += [why, ""]

    invali = u.get("what_invalidates") or []
    if invali:
        lines += [f"Risk: {invali[0]}", ""]

    tactical = u.get("tactical_positioning", "")
    if tactical:
        lines += [f"Positioning: {tactical}", ""]

    lines.append("#EisaX #MarketIntelligence #Investing #AlternativeData")
    return "\n".join(lines)

def _linkedin_weekly(u: dict) -> str:
    lines = [
        f"EisaX Weekly Strategy Brief — {u.get('week_range', '')}",
        "",
        u.get("market_summary", ""),
        "",
    ]

    positioning = u.get("positioning", "")
    if positioning:
        lines += [positioning, ""]

    conviction = u.get("highest_conviction_opportunity", "")
    if conviction:
        lines += [f"Highest Conviction: {conviction}", ""]

    risks = u.get("key_risks") or []
    if risks:
        lines += [f"Key Risk: {risks[0]}", ""]

    changes = u.get("what_changes_this_view") or []
    if changes:
        lines += [f"What changes this view: {changes[0]}", ""]

    verdict = u.get("eisax_verdict", "")
    if verdict:
        lines += [f"EisaX Verdict: {verdict}", ""]

    lines.append("#EisaX #WeeklyStrategy #MarketIntelligence #Investing")
    return "\n".join(lines)

def get_latest_updates() -> dict:
    """Return latest daily + weekly with LinkedIn-formatted versions."""
    _init_db()
    daily  = _get_latest("daily")
    weekly = _get_latest("weekly")
    result = {}
    if daily:
        # Prefer AI-generated LinkedIn text; fall back to template
        linkedin = daily.pop("linkedin_text", None) or format_for_linkedin(daily)
        full_report = daily.pop("full_report", None)
        ar_full_report = daily.pop("ar_full_report", None)
        result["daily"] = {
            "data":         daily,
            "linkedin":     linkedin,
            "full_report":  full_report,
            "ar_full_report": ar_full_report,
            "generated_at": daily.pop("_generated_at", None),
        }
    if weekly:
        linkedin_w = weekly.pop("linkedin_text", None) or format_for_linkedin(weekly)
        full_report_w = weekly.pop("full_report", None)
        ar_full_report_w = weekly.pop("ar_full_report", None)
        result["weekly"] = {
            "data":         weekly,
            "linkedin":     linkedin_w,
            "full_report":  full_report_w,
            "ar_full_report": ar_full_report_w,
            "generated_at": weekly.pop("_generated_at", None),
        }
    return result

