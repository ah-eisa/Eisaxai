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

from core.services.market_report import (
    _clean_text, _clean_text_list, _merge_text_list_with_fallback,
    _normalize_key_moves, _normalize_winners_losers, _normalize_regional_view,
    _format_report_number, _as_number, _allocation_summary, _snapshot_brief,
    _fit_word_window, _daily_decision_type, _daily_positioning_mode,
    _daily_confidence_score, _daily_market_state, _daily_positioning_line,
    _daily_linkedin_hook, _daily_linkedin_insight_lines,
    _weekly_linkedin_hook, _weekly_stance_label, _weekly_positioning_line,
    _weekly_linkedin_insight_lines, _weekly_focus_text, _build_web_version,
)
from core.services.market_collector import (
    build_eisax_stance, build_invalidation_logic,
    _build_asset_allocation_view, _build_cross_asset_snapshot,
    _determine_regime, _determine_regime_confidence,
    _call_openai_text, _call_gemini,
)
from core.services.market_regional import (
    _build_daily_regional_internals, _compute_gcc_decoupling_signal,
    _format_internal_line_en, _format_internal_line_ar,
    _ordered_daily_catalysts, _spy_range_levels,
    _translate_phrase_ar, _translate_risk_trigger_ar, _translate_catalyst_ar,
)
from core.services.market_db import _get_market_data_timestamp

def _build_linkedin_text(update: dict) -> str:
    if "date" in update:
        view = update.get("eisax_view", {}) if isinstance(update.get("eisax_view"), dict) else {}
        stance = _clean_text(view.get("stance")) or "HOLD"
        focus = _clean_text(view.get("focus")) or "selective quality"
        horizon = _clean_text(view.get("horizon")) or "tactical"
        matters = _clean_text_list(update.get("what_matters_now"), 3)
        risk_lines = _clean_text_list(update.get("what_invalidates"), 3)
        positioning = _clean_text(update.get("tactical_positioning"))
        snapshot_line = _snapshot_brief(update.get("cross_asset_snapshot") or {})
        matter_1 = matters[0].rstrip(".") if matters else ""
        matter_2 = matters[1].rstrip(".") if len(matters) > 1 else ""
        base = (
            f"Today’s EisaX market pulse stays {update.get('market_regime', 'Cautious').lower()}, with a {stance.lower()} stance and a {horizon} horizon. "
            f"The focus remains {focus.lower()}, not broad beta. "
            f"{snapshot_line}. "
            f"{(matter_1 + '. ') if matter_1 else ''}"
            f"{(matter_2 + '. ') if matter_2 else ''}"
            f"What matters for positioning is simple: {_clean_text(update.get('why_now')).rstrip('.')}. "
            f"Portfolio action stays explicit — {positioning.rstrip('.')}. "
            f"The view only changes if {(risk_lines[0] if risk_lines else 'price breaks the current support structure').rstrip('.')} "
            f"or if {(risk_lines[1] if len(risk_lines) > 1 else 'volatility and yields reprice higher').rstrip('.')}."
        )
        return _fit_word_window(base, 120, 180)

    risks = _clean_text_list(update.get("key_risks"), 3)
    changes = _clean_text_list(update.get("what_changes_this_view"), 3)
    allocation = _allocation_summary(update.get("asset_allocation_view") or {}).replace("Allocation: ", "")
    base = (
        f"EisaX ends the week with a clear portfolio map rather than a macro recap. "
        f"{_clean_text(update.get('market_summary')).rstrip('.')}. "
        f"Allocation stays {allocation}. "
        f"Portfolio posture is {_clean_text(update.get('positioning')).rstrip('.')}. "
        f"The highest-conviction expression remains {_clean_text(update.get('highest_conviction_opportunity')).rstrip('.')}. "
        f"The first risk on the desk is {(risks[0] if risks else 'macro repricing across rates and volatility').rstrip('.')}. "
        f"The framework changes if {(changes[0] if changes else 'price, volatility, and rates break the current regime').rstrip('.')}. "
        f"That leaves the weekly verdict disciplined: {_clean_text(update.get('eisax_verdict')).rstrip('.')}."
    )
    return _fit_word_window(base, 120, 180)

def _build_linkedin_text_v2(update: dict) -> str:
    if "date" in update:
        view = update.get("eisax_view", {}) if isinstance(update.get("eisax_view"), dict) else {}
        stance = _clean_text(view.get("stance")) or "HOLD"
        focus = _clean_text(view.get("focus")) or "selective quality"
        risk_lines = _clean_text_list(update.get("what_invalidates"), 3)
        insight_lines = _daily_linkedin_insight_lines(update)
        risk_1 = (risk_lines[0] if risk_lines else "price breaks the current support structure").rstrip(".")
        risk_2 = (risk_lines[1] if len(risk_lines) > 1 else "volatility and yields reprice higher").rstrip(".")
        lines = [
            _daily_linkedin_hook(update),
            f"{insight_lines[0].rstrip('.')}.",
            f"{insight_lines[1].rstrip('.')}.",
            f"EisaX View: {stance} — focus on {focus}.",
            f"Risk: {risk_1} or {risk_2}.",
            _daily_positioning_line(update),
            "#EisaX #MarketIntelligence #CrossAsset #PortfolioManagement",
        ]
        return "\n\n".join(line for line in lines if line).strip()

    risks = _clean_text_list(update.get("key_risks"), 3)
    changes = _clean_text_list(update.get("what_changes_this_view"), 3)
    insight_lines = _weekly_linkedin_insight_lines(update)
    idea = _clean_text(update.get("highest_conviction_opportunity")).rstrip(".")
    risk_1 = (changes[0] if changes else "price, volatility, and rates break the current regime").rstrip(".")
    risk_2 = (risks[0] if risks else "macro repricing tightens the risk budget").rstrip(".")
    lines = [
        _weekly_linkedin_hook(update),
        f"{insight_lines[0].rstrip('.')}.",
        f"{insight_lines[1].rstrip('.')}.",
        f"{insight_lines[2].rstrip('.')}.",
        f"EisaX View: {_weekly_stance_label(update)} — focus on {_weekly_focus_text(update)}.",
        f"Risk: {risk_1} or {risk_2}.",
        _weekly_positioning_line(update),
        "#EisaX #MarketStrategy #CrossAsset #PortfolioManagement",
    ]
    return "\n\n".join(line for line in lines if line).strip()

def _deterministic_daily(moves: dict, regime: str, fg: dict) -> dict:
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    vix_val  = (moves.get("^VIX")    or {}).get("price", 20)
    spy_d1   = (moves.get("SPY")     or {}).get("d1_pct", 0)
    spy_d5   = (moves.get("SPY")     or {}).get("d5_pct", 0)
    fg_score = fg.get("score", 50)
    conf     = _determine_regime_confidence(moves, regime)
    stance   = build_eisax_stance(moves, regime, fg)
    invali   = build_invalidation_logic(moves, regime)
    snapshot = _build_cross_asset_snapshot(moves)

    key_moves = []
    for ticker, data in moves.items():
        if ticker in ("^VIX", "^TNX", "UUP"):
            continue
        if abs(data.get("d1_pct", 0)) >= 0.4:
            pct = data["d1_pct"]
            key_moves.append({
                "asset":  data["label"],
                "move":   f"{pct:+.1f}% (1d)",
                "reason": (
                    "Momentum extending on elevated volume" if abs(pct) > 1.5
                    else "Price confirming directional bias"
                ),
            })
    key_moves = sorted(key_moves,
                       key=lambda x: abs(float(x["move"].split("%")[0])),
                       reverse=True)[:5]
    if not key_moves:
        key_moves = [{
            "asset":  "S&P 500",
            "move":   f"{spy_d1:+.1f}% (1d)",
            "reason": "Range-bound session; no clear directional catalyst",
        }]

    if vix_val > 20:
        vix_note = f"elevated risk premium — position sizing takes priority over returns"
    else:
        vix_note = f"contained volatility — conditions support selective risk-taking"

    if fg_score > 70:
        fg_note = "complacency at elevated levels — contrarian caution warranted"
    elif fg_score < 30:
        fg_note = "fear elevated — contrarian opportunity forming in quality names"
    else:
        fg_note = "balanced sentiment — no extreme signal, regime data is the guide"

    return {
        "date":              today,
        "market_regime":     regime,
        "regime_confidence": conf,
        "what_matters_now": [
            f"VIX at {vix_val:.1f} — {vix_note}",
            f"S&P 500 {'up' if spy_d5 > 0 else 'down'} {abs(spy_d5):.1f}% on the week — "
            f"{'trend intact, bias remains constructive' if 0 < spy_d5 < 3 else 'significant move in play, confirm breadth' if abs(spy_d5) >= 3 else 'price action inconclusive, await catalyst'}",
            f"Fear & Greed at {fg_score}/100 ({fg.get('rating', 'Neutral')}) — {fg_note}",
        ],
        "key_moves":             key_moves,
        "cross_asset_snapshot":  snapshot,
        "eisax_view":            stance,
        "why_now": (
            f"Risk assets are under distribution pressure; VIX at {vix_val:.1f} confirms elevated hedging demand. "
            f"Price action diverging from macro expectations — protect capital first."
            if regime == "Bearish"
            else
            f"Momentum is constructive across the equity complex with VIX suppressed at {vix_val:.1f}. "
            f"Breadth improving — breadth confirms the move rather than leading it."
            if regime == "Bullish"
            else
            f"Cross-asset signals are mixed. VIX at {vix_val:.1f} with indecisive price action. "
            f"Conviction is low — sizing small and waiting for confirmation is rational."
        ),
        "what_invalidates":     invali,
        "tactical_positioning": (
            f"Reduce equity beta; raise cash/gold allocation. Size positions for a down 3–5% scenario."
            if regime == "Bearish"
            else
            f"Maintain core holdings. Deploy fresh capital on any 1–2% pullback to key support."
            if regime == "Cautious"
            else
            f"Add to leaders on pullbacks. Avoid chasing extended moves; use limit orders near VWAP."
        ),
        "next_triggers":    ["Fed communications", "CPI / PCE data release",
                             f"VIX {'<' if regime == 'Bullish' else '>'}{round(vix_val * 0.8 if regime == 'Bullish' else vix_val * 1.25)} level break"],
        "fear_greed_index": fg_score,
    }

def _deterministic_weekly(moves: dict, regime: str, fg: dict) -> dict:
    now        = datetime.now(timezone.utc)
    week_start = (now - timedelta(days=7)).strftime("%b %d")
    week_end   = now.strftime("%b %d, %Y")
    week_range = f"{week_start}–{week_end}"

    spy5     = (moves.get("SPY")     or {}).get("d5_pct", 0)
    qqq5     = (moves.get("QQQ")     or {}).get("d5_pct", 0)
    btc5     = (moves.get("BTC-USD") or {}).get("d5_pct", 0)
    gld5     = (moves.get("GLD")     or {}).get("d5_pct", 0)
    uso5     = (moves.get("USO")     or {}).get("d5_pct", 0)
    vix      = (moves.get("^VIX")   or {}).get("price", 20)
    tnx      = (moves.get("^TNX")   or {}).get("price", 4.25)
    dxy5     = (moves.get("UUP")    or {}).get("d5_pct", 0)
    fg_score = fg.get("score", 50)
    stance   = build_eisax_stance(moves, regime, fg)
    invali   = build_invalidation_logic(moves, regime)
    snapshot = _build_cross_asset_snapshot(moves)

    sorted_mv = sorted(moves.items(), key=lambda x: x[1].get("d5_pct", 0), reverse=True)
    winners = [f"{v['label']} {v['d5_pct']:+.1f}%" for _, v in sorted_mv[:3] if v.get("d5_pct", 0) > 0]
    losers  = [f"{v['label']} {v['d5_pct']:+.1f}%" for _, v in sorted_mv[-3:] if v.get("d5_pct", 0) < 0]

    alloc = _build_asset_allocation_view(regime)

    tasi_data = moves.get("^TASI")
    gcc_view  = (
        f"Saudi market composite {tasi_data['d5_pct']:+.1f}% over five sessions. Oil {uso5:+.1f}% {'supports' if uso5 > 0 else 'pressures'} GCC earnings visibility."
        if tasi_data
        else f"GCC data limited — oil at {(moves.get('USO') or {}).get('price', 0):.1f} remains the primary earnings driver for the region."
    )

    return {
        "week_range":    week_range,
        "market_summary": (
            f"US equities {'advanced' if spy5 > 0 else 'retreated'} {spy5:+.1f}% as "
            f"{'risk appetite held firm despite macro headwinds' if spy5 > 0 else 'macro uncertainty and rate sensitivity weighed on sentiment'}. "
            f"Nasdaq {'outperformed' if qqq5 > spy5 else 'underperformed'} at {qqq5:+.1f}%. "
            f"VIX settled at {vix:.1f} — {'volatility regime remains elevated; hedging demand sustained' if vix > 22 else 'fear premium stayed contained, trend confirmation intact'}. "
            f"Bitcoin {('rallied' if btc5 > 5 else 'declined' if btc5 < -5 else 'traded range-bound')} {btc5:+.1f}%, "
            f"{'tracking equity risk sentiment' if btc5 * spy5 > 0 else 'diverging from equity direction — watch for cross-asset dislocation'}."
        ),
        "positioning": (
            f"Reduce beta exposure; shift meaningfully into gold and short-duration bonds. "
            f"Target 30–40% defensive allocation until VIX recedes below {max(vix * 0.75, 18):.0f}."
            if regime == "Bearish"
            else
            f"Hold core allocations; deploy cash selectively on weakness in quality names. "
            f"Avoid adding to extended positions — wait for a clean 2–3% pullback."
            if regime == "Cautious"
            else
            f"Maintain equity overweight with focus on earnings quality over multiple expansion. "
            f"Size positions to allow adding on any 2–3% pullback toward key support levels."
        ),
        "asset_allocation_view": alloc,
        "cross_asset_snapshot": snapshot,
        "regional_view": {
            "US":    f"S&P 500 {spy5:+.1f}% on the week. {'Regime support intact; focus on quality earnings.' if spy5 > 0 and vix < 20 else 'Caution warranted — breadth needs to confirm any rebound before re-engagement.'}",
            "GCC":   gcc_view,
            "Egypt": f"Egypt market composite data remain driven by local breadth. DXY {dxy5:+.1f}% on week creates {'headwinds for EM including Egypt via capital outflow pressure' if dxy5 > 0 else 'some relief for EM assets including Egypt via easing outflow pressure'}.",
        },
        "winners_losers":    {
            "winners": winners or ["No meaningful outperformers this week"],
            "losers":  losers  or ["No meaningful underperformers this week"],
        },
        "highest_conviction_opportunity": (
            f"Gold (GLD) — {gld5:+.1f}% on week. Inflation re-acceleration + dollar pressure creates asymmetric setup: "
            f"downside protected by central bank demand; upside driven by de-dollarization and real-rate compression."
            if gld5 > 0 and regime != "Bullish"
            else
            f"Quality US Technology on pullbacks — Nasdaq {qqq5:+.1f}% creates tactical re-entry if VIX stays below {vix + 4:.0f}. "
            f"AI capex cycle structurally intact; use weakness as add opportunity, not exit signal."
            if regime == "Bullish"
            else
            f"Short-duration Treasuries — 10Y at {tnx:.2f}% provides carry with limited duration risk. "
            f"Capital preservation with income in uncertain regime outperforms cash on a risk-adjusted basis."
        ),
        "key_risks": [
            f"Rate trajectory — 10Y at {tnx:.2f}%; any move above {tnx + 0.4:.2f}% reprices equity multiples and amplifies credit stress",
            "Geopolitical escalation feeding into energy prices — oil spike above $95 disrupts both consumer spending and EM stability",
            (
                "Earnings guidance cuts given elevated sell-side expectations — even minor misses risk multiple compression"
                if regime == "Bullish"
                else
                "Credit tightening accelerating — watch high-yield spreads for leading signal of broader funding stress"
            ),
        ],
        "what_changes_this_view": invali,
        "portfolio_angle": (
            f"With {regime.lower()} conditions and VIX at {vix:.1f}, "
            + (
                "priority is alpha preservation over return generation. Cut high-beta positions, hold quality franchises, "
                f"and size gold allocation at 15–20% of portfolio. Cash is a position."
                if regime == "Bearish"
                else
                "selective deployment is rational. Focus on companies with pricing power and visible earnings. "
                "Avoid multiple expansion plays — the environment rewards quality, not momentum."
                if regime == "Cautious"
                else
                "maintain exposure but avoid chasing. Rotate into leaders that are pulling back to support, "
                "not into laggards hoping for catch-up. Quality > beta in this phase."
            )
        ),
        "eisax_verdict": (
            f"Reduce risk; protect capital. The setup does not reward beta — it punishes it."
            if regime == "Bearish"
            else
            f"Hold positions; add selectively on 2–3% pullbacks. Await a decisive catalyst before increasing allocation."
            if regime == "Cautious"
            else
            f"Stay long quality names. Trim speculative positions on strength and reinvest into leaders on weakness."
        ),
        "fear_greed_index": fg_score,
    }

def _enforce_cio_daily_language(update: dict, moves: dict, fg: dict) -> dict:
    snapshot = update.get("cross_asset_snapshot") or {}
    regime = update.get("market_regime", "Cautious")
    fg_score = update.get("fear_greed_index")
    if fg_score is None:
        fg_score = fg.get("score", 50)

    vix_val = _as_number((snapshot.get("volatility") or {}).get("price"))
    tnx_val = _as_number((snapshot.get("rates") or {}).get("price"))
    oil_val = _as_number((snapshot.get("commodities") or {}).get("price"))

    default_matters = [
        f"Rates remain the first variable: 10Y Treasury at {tnx_val:.2f}% is still setting the liquidity budget ahead of the equity headline." if tnx_val is not None else "Rates remain the first variable because liquidity is still governing the risk budget.",
        f"Oil at {oil_val:.1f} and VIX at {vix_val:.1f} keep inflation pressure and hedging demand alive at the same time, which limits broad beta appetite." if oil_val is not None and vix_val is not None else "Oil and volatility are still active constraints on broad beta appetite.",
        f"Fear & Greed at {fg_score}/100 matters less than whether price can hold while liquidity stays this tight.",
    ]

    update["what_matters_now"] = default_matters

    if regime == "Bearish":
        why_now = f"Risk assets are under distribution pressure; VIX at {vix_val:.1f} confirms elevated hedging demand. Price action is not earning the right to carry broad beta." if vix_val is not None else "Risk assets are under distribution pressure, so capital preservation takes priority over broad beta."
    elif regime == "Bullish":
        why_now = f"Risk can still be owned, but only selectively, because rates at {tnx_val:.2f}% are not giving the market a clean liquidity tailwind. The tape stays usable if leadership holds, but the bar for broad beta expansion remains high." if tnx_val is not None else "Risk can still be owned, but only selectively, because liquidity is not giving the market a clean all-clear."
    else:
        why_now = f"This is a conflicted regime: volatility at {vix_val:.1f} and rates at {tnx_val:.2f}% are still too restrictive for passive risk-taking. Capital should stay active, but constrained, until price and liquidity resolve together." if vix_val is not None and tnx_val is not None else "This is a conflicted regime, so capital should stay active but constrained until price and liquidity resolve together."
    update["why_now"] = why_now

    if regime == "Bearish":
        tactical = "Reduce equity beta; raise cash and gold allocation. Size positions for defense first."
    elif regime == "Bullish":
        tactical = "Add to leaders on pullbacks, not on extension. Let rates and volatility confirm the move before increasing gross exposure."
    else:
        tactical = "Maintain core holdings, but add only on controlled pullbacks into support and keep new risk concentrated in quality leaders."
    update["tactical_positioning"] = tactical

    return update

def _weekly_decision_type(update: dict) -> str:
    alloc = update.get("asset_allocation_view") or {}
    equities = _clean_text(alloc.get("equities"))
    cash = _clean_text(alloc.get("cash"))
    if equities == "Underweight" or cash == "Overweight":
        return "REDUCE"
    if equities == "Overweight" and cash == "Underweight":
        return "BUY_SELECTIVE"
    return "HOLD_ACTIVE_CONSTRAINT"

def _weekly_positioning_mode(update: dict) -> str:
    decision_type = update.get("weekly_decision_type") or _weekly_decision_type(update)
    if decision_type == "REDUCE":
        return "Defensive"
    if decision_type == "BUY_SELECTIVE":
        return "Selective"
    return "Active Constraint"

def _weekly_confidence_score(regime: str, confidence: str, update: dict) -> int:
    base = {"Low": 56, "Medium": 70, "High": 82}.get(confidence, 60)
    decision_type = update.get("weekly_decision_type") or _weekly_decision_type(update)
    if regime == "Bullish":
        base += 4
    elif regime == "Bearish":
        base += 2
    elif regime == "Cautious":
        base -= 5
    if decision_type == "HOLD_ACTIVE_CONSTRAINT":
        base -= 3
    return max(35, min(95, int(round(base))))

def _weekly_market_state(update: dict) -> str:
    decision_type = update.get("weekly_decision_type") or _weekly_decision_type(update)
    regime = update.get("market_regime", "Cautious")
    if decision_type == "REDUCE" or regime == "Bearish":
        return "transition"
    if decision_type == "BUY_SELECTIVE":
        return "positioning"
    return "conflicted"

def _enforce_cio_weekly_language(update: dict, moves: dict, fg: dict) -> dict:
    snapshot = update.get("cross_asset_snapshot") or {}
    regime = update.get("market_regime", "Cautious")
    alloc = update.get("asset_allocation_view") or {}

    spy5 = _as_number((snapshot.get("us_equities") or {}).get("d5_pct"))
    qqq5 = _as_number((moves.get("QQQ") or {}).get("d5_pct"))
    btc5 = _as_number((snapshot.get("crypto") or {}).get("d5_pct"))
    oil5 = _as_number((snapshot.get("commodities") or {}).get("d5_pct"))
    oil_px = _as_number((snapshot.get("commodities") or {}).get("price"))
    vix = _as_number((snapshot.get("volatility") or {}).get("price"))
    tnx = _as_number((snapshot.get("rates") or {}).get("price"))
    dxy5 = _as_number((moves.get("UUP") or {}).get("d5_pct"))

    if regime == "Bearish":
        update["market_summary"] = (
            f"Rates ended the week as the dominant variable, with the 10Y Treasury at {tnx:.2f}% keeping liquidity tight across risk assets. "
            f"SPY at {spy5:+.1f}% and Nasdaq at {qqq5:+.1f}% did not earn enough sponsorship to justify broad beta, while oil at {oil_px:.1f} kept the inflation channel alive. "
            f"This was not a rebound week - it was a capital-protection week."
            if tnx is not None and spy5 is not None and qqq5 is not None and oil_px is not None
            else "Rates and liquidity dominated the week, and the tape did not justify broad beta expansion."
        )
        update["positioning"] = "Reduce beta, keep cash and gold elevated, and treat any rally as tactical until rates and volatility improve together."
        update["highest_conviction_opportunity"] = (
            f"Gold on weakness remains the highest-conviction expression while yields hold near {tnx:.2f}% and oil keeps inflation risk alive."
            if tnx is not None
            else "Gold remains the highest-conviction expression while liquidity stays restrictive."
        )
        update["portfolio_angle"] = "Keep the book defensive, own liquid quality only, and let cash and gold absorb uncertainty before re-risking."
        update["eisax_verdict"] = "Reduce beta and preserve capital until liquidity improves."
    elif regime == "Bullish":
        update["market_summary"] = (
            f"Equities held the weekly lead with SPY at {spy5:+.1f}% and Nasdaq at {qqq5:+.1f}%, but the real question was whether rates at {tnx:.2f}% would allow the move to broaden. "
            f"Oil at {oil_px:.1f} and VIX at {vix:.1f} kept inflation and hedging costs relevant, so upside stayed selective rather than indiscriminate. "
            f"This was a leadership week, not a blind risk-on week."
            if spy5 is not None and qqq5 is not None and tnx is not None and oil_px is not None and vix is not None
            else "Equity leadership held, but rates still kept the upside selective rather than indiscriminate."
        )
        update["positioning"] = "Add selectively to quality leadership, keep cash efficient rather than idle, and avoid chasing second-tier beta."
        update["highest_conviction_opportunity"] = (
            f"Quality US technology on pullbacks remains the cleanest add while Nasdaq leadership holds and the 10Y stays near {tnx:.2f}%."
            if tnx is not None
            else "Quality US technology on pullbacks remains the cleanest add while leadership holds."
        )
        update["portfolio_angle"] = "Stay long quality, fund adds from laggards rather than cash alone, and keep the book ready to buy pullbacks instead of upside extension."
        update["eisax_verdict"] = "Add selectively to leadership, but do not confuse a usable tape with a broad-beta all-clear."
    else:
        update["market_summary"] = (
            f"Rates still governed the weekly tape, with the 10Y at {tnx:.2f}% keeping liquidity tight even as SPY finished {spy5:+.1f}% and Nasdaq {qqq5:+.1f}% over five sessions. "
            f"Oil at {oil_px:.1f} and VIX at {vix:.1f} prevented the market from graduating into a clean risk-on regime, while Bitcoin at {btc5:+.1f}% confirmed only selective liquidity. "
            f"This was a positioning week, not a broad allocation week."
            if tnx is not None and spy5 is not None and qqq5 is not None and oil_px is not None and vix is not None and btc5 is not None
            else "Rates, oil, and volatility kept the week in a positioning regime rather than a broad allocation regime."
        )
        update["positioning"] = "Hold active constraint: keep core quality exposure, deploy on weakness only, and keep speculative sleeves underweight."
        update["highest_conviction_opportunity"] = (
            f"Short-duration Treasuries remain the cleanest carry sleeve while the 10Y trades near {tnx:.2f}% and broad risk still lacks a clean liquidity tailwind."
            if tnx is not None
            else "Short-duration Treasuries remain the cleanest carry sleeve while broad risk still lacks a clean liquidity tailwind."
        )
        update["portfolio_angle"] = "Run the book with controlled gross exposure, recycle capital into quality and defense, and wait for rates and volatility to confirm before pressing beta."
        update["eisax_verdict"] = "Hold with active constraint and add only where liquidity and leadership align."

    update["regional_view"] = {
        "US": (
            f"US equities stayed tradable, but {tnx:.2f}% in 10Y yields still capped how much multiple expansion the tape could absorb."
            if tnx is not None
            else "US equities stayed tradable, but rates still capped how much multiple expansion the tape could absorb."
        ),
        "GCC": (
            f"GCC direction still runs through oil at {oil_px:.1f}; that supports regional liquidity and banks, but it also keeps the inflation channel alive."
            if oil_px is not None
            else "GCC direction still runs through oil and regional liquidity rather than global beta alone."
        ),
        "Egypt": (
            f"Egypt remains tied to dollar and rate pressure, with DXY at {dxy5:+.1f}% on the week keeping imported inflation and funding conditions in focus."
            if dxy5 is not None
            else "Egypt remains tied to dollar strength, rates, and imported inflation more than headline US equity performance."
        ),
    }

    update["key_risks"] = [
        f"Rates: a further move higher from {tnx:.2f}% in the 10Y would tighten liquidity and pressure both valuation and EM funding." if tnx is not None else "Rates remain the first weekly risk because tighter liquidity still governs the book.",
        f"Oil: crude near {oil_px:.1f} is supportive for GCC cash flow, but another leg higher would re-open inflation pressure across the global book." if oil_px is not None else "Oil remains a two-sided risk because it supports GCC cash flow while threatening inflation.",
        "Positioning: if leadership narrows further, portfolios that are too broad will underperform even without an index breakdown.",
    ]

    return update

def _build_cio_weekly_report_fallback(update: dict) -> str:
    alloc = update.get("asset_allocation_view") or {}
    snapshot = update.get("cross_asset_snapshot") or {}
    regime = update.get("market_regime", "Cautious")
    confidence = update.get("regime_confidence", "Low")
    decision_type = update.get("weekly_decision_type") or _weekly_decision_type(update)
    mode = update.get("weekly_positioning_mode") or _weekly_positioning_mode(update)
    confidence_score = update.get("weekly_confidence_score") or _weekly_confidence_score(regime, confidence, update)
    changes = _clean_text_list(update.get("what_changes_this_view"), 4)
    state = _weekly_market_state(update)

    def _num(value: Any, pct: bool = False, rates: bool = False, crypto: bool = False) -> str:
        if isinstance(value, str):
            return value
        if not isinstance(value, (int, float)):
            return "Market Closed"
        if crypto:
            return f"{value:,.0f}"
        if rates:
            return f"{value:.2f}%"
        if pct:
            return f"{value:+.2f}%"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.2f}"

    us = snapshot.get("us_equities") or {}
    rates = snapshot.get("rates") or {}
    oil = snapshot.get("commodities") or {}
    vix = snapshot.get("volatility") or {}
    btc = snapshot.get("crypto") or {}

    maintain = []
    for key in ("equities", "metals", "commodities"):
        value = alloc.get(key)
        if value == "Overweight":
            maintain.append(key)
    maintain_text = ", ".join(maintain) if maintain else "quality and liquidity"
    avoid_text = "broad beta" if decision_type != "REDUCE" else "high beta and unfunded cyclicals"

    lines = [
        "EisaX Weekly Strategy Brief",
        f"Week: {update.get('week_range', '')}",
        f"Regime: {regime}",
        f"Confidence: {confidence}",
        f"Decision Type: {decision_type}",
        f"Confidence Score: {confidence_score}",
        "",
        "## Executive Summary",
        f"- {_clean_text(update.get('market_summary'))}",
        f"- {_clean_text(update.get('positioning'))}",
        f"- {_clean_text(update.get('portfolio_angle'))}",
        f"- Weekly verdict: {_clean_text(update.get('eisax_verdict'))}",
        "",
        "## Cross-Asset Reality",
        f"Equities -> SPY finished the week at {_num(us.get('price'))} with {_num(us.get('d5_pct'), pct=True)} over five sessions; leadership held, but only where liquidity allowed it.",
        f"Rates -> 10Y Treasury ended near {_num(rates.get('price'), rates=True)}; that remained the weekly gatekeeper for multiple expansion and overall risk budget.",
        f"Oil -> WTI proxy at {_num(oil.get('price'))} and {_num(oil.get('d5_pct'), pct=True)} on the week supported GCC cash flow but kept inflation risk active.",
        f"Volatility -> VIX at {_num(vix.get('price'))} did not signal panic, but it stayed high enough to keep hedging and sizing discipline relevant.",
        f"Crypto -> Bitcoin at {_num(btc.get('price'), crypto=True)} with {_num(btc.get('d5_pct'), pct=True)} on the week remained a secondary liquidity tell, not the primary allocation anchor.",
        "This is not a broad beta week.",
        f"This is a {state} week.",
        "",
        "## Regional Read (GCC + Egypt)",
        f"- {_clean_text((update.get('regional_view') or {}).get('GCC'))}",
        f"- {_clean_text((update.get('regional_view') or {}).get('Egypt'))}",
        "- GCC still absorbs higher oil more cleanly than Egypt absorbs higher rates and dollar pressure.",
        "",
        "## Allocation Decision",
        f"Stance: {_clean_text(update.get('eisax_verdict'))}",
        f"Mode: {mode}",
        "Execution:",
        f"- Rule 1: Maintain exposure where allocation already favors {maintain_text}.",
        "- Rule 2: Add only on weakness that holds structure, not on upside extension.",
        f"- Rule 3: Avoid {avoid_text} until rates, volatility, and price confirm together.",
        "",
        "## Risk Framework",
        f"- SPY level: {changes[0] if len(changes) > 0 else 'SPY must hold the current weekly structure or positioning has to tighten.'}",
        f"- VIX level: {changes[1] if len(changes) > 1 else 'A renewed volatility spike would force the book back into defense.'}",
        f"- 10Y level: {changes[2] if len(changes) > 2 else 'A further move higher in the 10Y would reprice liquidity and cap risk appetite.'}",
        "- Regime shift: weekly conviction changes only if price, volatility, and rates move together rather than in isolation.",
        "",
        "## Tactical Playbook",
        f"- Maintain: {maintain_text}.",
        f"- Add: {_clean_text(update.get('highest_conviction_opportunity'))}",
        f"- Avoid: {avoid_text}.",
        f"- Focus: {_clean_text(update.get('portfolio_angle'))}",
        "",
        "## Catalysts",
        "- Fed communication and rate repricing",
        "- CPI / PCE and inflation persistence",
        "- Oil and its read-through into GCC liquidity",
        "- Treasury yields and dollar pressure on regional funding conditions",
        "",
        "## Final Line",
        "The weekly book should follow liquidity and leadership, not index-level comfort.",
    ]
    return "\n".join(lines).strip()

def _ar_label(text: str, mapping: dict, default: str = "") -> str:
    return mapping.get(text, default or text)

def _ar_num(value: Any, pct: bool = False, rates: bool = False, crypto: bool = False) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, (int, float)):
        return "غير متاح"
    if crypto:
        return f"{value:,.0f}"
    if rates:
        return f"{value:.2f}%"
    if pct:
        return f"{value:+.2f}%"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.2f}"

def _build_cio_daily_report_ar(update: dict) -> str:
    view = update.get("eisax_view", {}) if isinstance(update.get("eisax_view"), dict) else {}
    snapshot = update.get("cross_asset_snapshot") or {}
    regime = update.get("market_regime", "Cautious")
    confidence = update.get("regime_confidence", "Low")
    decision_type = update.get("decision_type") or _daily_decision_type(view)
    confidence_score = update.get("confidence_score") or _daily_confidence_score(regime, confidence, update.get("fear_greed_index"))
    mode = update.get("positioning_mode") or _daily_positioning_mode(view)
    market_state = _daily_market_state(snapshot, regime)
    invalidates = _clean_text_list(update.get("what_invalidates"), 4)
    triggers = _clean_text_list(update.get("next_triggers"), 4)

    regime_ar = _ar_label(regime, {"Bullish": "صعودي", "Bearish": "هبوطي", "Cautious": "حذر", "Conflicted": "متضارب"}, "حذر")
    conf_ar = _ar_label(confidence, {"High": "عالية", "Medium": "متوسطة", "Low": "منخفضة"}, "منخفضة")
    decision_ar = _ar_label(decision_type, {
        "HOLD_ACTIVE_CONSTRAINT": "احتفاظ بقيود نشطة",
        "BUY_SELECTIVE": "شراء انتقائي",
        "REDUCE": "خفض المخاطر",
    }, decision_type)
    mode_ar = _ar_label(mode, {"Active Constraint": "قيود نشطة", "Selective": "انتقائي", "Defensive": "دفاعي"}, mode)
    stance_ar = _ar_label(_clean_text(view.get("stance")), {
        "HOLD": "احتفاظ",
        "Tactical BUY": "شراء تكتيكي",
        "REDUCE RISK": "خفض المخاطر",
    }, _clean_text(view.get("stance")))
    state_ar = _ar_label(market_state, {"conflicted": "تموضع متضارب", "transition": "انتقالية", "positioning": "تموضع"}, market_state)
    state_line_ar = "هذا نظام متضارب." if market_state == "conflicted" else f"هذا سوق {state_ar}."
    state_line_ar = "هذا نظام متضارب." if market_state == "conflicted" else f"هذا سوق {state_ar}."

    us = snapshot.get("us_equities") or {}
    rates = snapshot.get("rates") or {}
    oil = snapshot.get("commodities") or {}
    vix = snapshot.get("volatility") or {}
    btc = snapshot.get("crypto") or {}

    lines = [
        "EisaX Daily Market Pulse",
        f"Date: {update.get('date', '')}",
        f"Regime: {regime_ar}",
        f"Confidence: {conf_ar}",
        f"Decision Type: {decision_ar}",
        f"Confidence Score: {confidence_score}",
        "",
        "## الملخص التنفيذي",
        "• السوق قابل للاستثمار، لكن فقط عبر انضباط في الحجم وقيود نشطة على المخاطر.",
        f"• عوائد الخزانة قرب {_ar_num(rates.get('price'), rates=True)} ما زالت تضبط ميزانية السيولة قبل أي ارتياح في الأسهم.",
        f"• النفط عند {_ar_num(oil.get('price'))} والتذبذب عند {_ar_num(vix.get('price'))} يبقيان مسار التضخم والتحوط مفتوحًا، لذلك لا توجد أرضية لبيتا واسعة.",
        f"• الموقف الحالي هو {stance_ar} ضمن وضع {mode_ar} حتى ينكسر إطار المخاطر بوضوح.",
        "",
        "## واقع الأصول المتقاطعة",
        f"• الأسهم -> مؤشر SPY أنهى الجلسة عند {_ar_num(us.get('price'))} مع {_ar_num(us.get('d1_pct'), pct=True)} يوميًا و{_ar_num(us.get('d5_pct'), pct=True)} خلال خمسة أيام؛ السعر متماسك، لكن السيولة لا تسمح بعد بتوسيع المخاطرة على نطاق واسع.",
        f"• العوائد -> سندات الخزانة الأميركية 10 سنوات عند {_ar_num(rates.get('price'), rates=True)}؛ وهذا يعني أن السيولة ما زالت أهم من عنوان السوق نفسه.",
        f"• النفط -> خام WTI proxy عند {_ar_num(oil.get('price'))} مع {_ar_num(oil.get('d1_pct'), pct=True)}؛ وهذا مفيد لتدفقات الخليج لكنه يعيد فتح قناة التضخم.",
        f"• VIX -> التذبذب عند {_ar_num(vix.get('price'))}؛ الخوف ليس حادًا، لكنه ليس منخفضًا بما يكفي لتبرير شراء بيتا بلا تمييز.",
        f"• الكريبتو -> بيتكوين عند {_ar_num(btc.get('price'), crypto=True)} ويُقرأ كمؤشر سيولة ثانوي لا كمرتكز التخصيص الأساسي.",
        "هذا ليس سوقًا اتجاهيًا.",
        f"هذا سوق {state_ar}.",
        "",
        "## القراءة الإقليمية (الخليج + مصر)",
        f"• الخليج ما زال مربوطًا بالنفط عند {_ar_num(oil.get('price'))}؛ وهذا يدعم السيولة والبنوك والطاقة، لكنه لا يلغي أثر التضخم العالمي.",
        f"• عوائد 10 سنوات عند {_ar_num(rates.get('price'), rates=True)} أهم لمصر من عنوان الأسهم الأميركية، لأن ضغط الدولار والتمويل الخارجي ما زال حاضرًا.",
        "• التدفقات القطاعية تميل إلى بنوك وطاقة الخليج، بينما تحتاج مصر إلى شركات دفاعية وميزانيات قادرة على امتصاص ضغط الدولار.",
        "",
        "## التموضع",
        f"الموقف: {stance_ar}",
        f"الوضع: {mode_ar}",
        "التنفيذ:",
        "• حافظ على المراكز الأساسية عالية الجودة، لكن لا توسع الانكشاف إلا على تراجعات منضبطة.",
        "• أضف فقط عندما يثبت السعر ويهدأ ضغط العوائد والتذبذب معًا.",
        "• أعد تدوير رأس المال من البيتا المضاربية إلى الجودة والسيولة والدفاعيات.",
        "",
        "## إطار المخاطر",
        f"• SPY: {invalidates[0] if len(invalidates) > 0 else 'كسر البنية الحالية في SPY يعني أن السوق يفقد الرعاية السعرية.'}",
        f"• VIX: {invalidates[1] if len(invalidates) > 1 else 'أي ارتفاع واضح في VIX يعني تضييقًا مباشرًا في ميزانية المخاطر.'}",
        f"• 10Y: {invalidates[2] if len(invalidates) > 2 else 'أي ارتفاع إضافي في عائد 10 سنوات سيضغط على التقييمات والسيولة.'}",
        "• تغير النظام: القرار يتغير فقط عندما تتحرك الأسعار والعوائد والتذبذب معًا، لا عندما يتحرك عامل واحد منفردًا.",
        "",
        "## الدليل التكتيكي",
        f"• حافظ على: {', '.join(view.get('overweight_assets') or view.get('neutral_assets') or ['Quality Equities'])}.",
        f"• انشر رأس المال عند: {_clean_text(update.get('tactical_positioning'))}.",
        f"• تجنب: {', '.join(view.get('underweight_assets') or ['البيتا المضاربية'])}.",
        f"• ركز على: {_clean_text(view.get('focus')) or 'الجودة الانتقائية'}.",
        "",
        "## المحفزات",
    ]
    for item in (triggers or ["خطاب الفيدرالي", "بيانات CPI / PCE", "تحرك النفط", "العوائد الأميركية"])[:4]:
        lines.append(f"• {item}")
    lines += [
        "",
        "## الخلاصة",
        "• رأس المال يجب أن يتبع السيولة لا العناوين؛ وحتى تؤكد العوائد والتذبذب المرحلة التالية، يبقى الهجوم الصحيح هو الهجوم الانتقائي فقط.",
    ]
    return "\n".join(lines).strip()

def _build_cio_daily_report_fallback_v2(update: dict) -> str:
    view = update.get("eisax_view", {}) if isinstance(update.get("eisax_view"), dict) else {}
    snapshot = update.get("cross_asset_snapshot") or {}
    regime = update.get("market_regime", "Cautious")
    confidence = update.get("regime_confidence", "Low")
    fg_score = update.get("fear_greed_index", 50)
    decision_type = update.get("decision_type") or _daily_decision_type(view)
    confidence_score = update.get("confidence_score") or _daily_confidence_score(regime, confidence, fg_score)
    mode = update.get("positioning_mode") or _daily_positioning_mode(view)
    invalidates = _clean_text_list(update.get("what_invalidates"), 4)
    triggers = _clean_text_list(update.get("next_triggers"), 4)
    catalysts = _ordered_daily_catalysts(triggers)
    market_state = _daily_market_state(snapshot, regime)
    market_levels = update.get("market_levels") or {}
    regional_internals = update.get("regional_internals") or {}
    decoupling = update.get("gcc_decoupling") or {}

    def _entry(key: str) -> dict:
        return snapshot.get(key) or {}

    def _num(value: Any, pct: bool = False, rates: bool = False, crypto: bool = False) -> str:
        if isinstance(value, str):
            return value
        if not isinstance(value, (int, float)):
            return "Market Closed"
        if crypto:
            return f"{value:,.0f}"
        if rates:
            return f"{value:.2f}%"
        if pct:
            return f"{value:+.2f}%"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.2f}"

    us = _entry("us_equities")
    rates = _entry("rates")
    oil = _entry("commodities")
    vix = _entry("volatility")
    btc = _entry("crypto")
    gcc = _entry("gcc")
    egypt = _entry("egypt")
    qqq = market_levels.get("nasdaq") or {}
    gold = market_levels.get("gold") or {}
    dxy = market_levels.get("dxy") or {}

    us_d5 = _as_number(us.get("d5_pct"))
    rate_d1 = _as_number(rates.get("d1_pct"))
    oil_d5 = _as_number(oil.get("d5_pct"))
    btc_d5 = _as_number(btc.get("d5_pct"))
    rate_level = _num(rates.get("price"), rates=True)

    if regime == "Bearish":
        lead = "Liquidity is tightening faster than equities can absorb, so the regime stays decisively defensive."
        stance_line = "Capital allocation should prioritize protection and dry powder over return maximization."
    elif regime == "Bullish":
        lead = "Risk can still be owned, but only where rates are not re-tightening the discount rate faster than earnings can offset it."
        stance_line = "The book can stay constructive, but only through leadership rather than broad beta."
    else:
        lead = "The market is investable, but only through disciplined sizing and active constraint."
        stance_line = "This is a hold-with-intent setup, not a passive hold and not a launch pad for fresh beta."

    if us_d5 is not None and rate_d1 is not None and us_d5 > 0 and rate_d1 > 0:
        contradiction = "Equities are advancing into tighter rates, which means liquidity is lagging price and upside should be treated as narrower than the index suggests."
    elif us_d5 is not None and rate_d1 is not None and us_d5 < 0 and rate_d1 < 0:
        contradiction = "Rates are offering some relief, but equities are not monetizing it yet, which tells you confidence in growth is still incomplete."
    elif oil_d5 is not None and oil_d5 > 0:
        contradiction = "Firmer oil supports regional cash flow, but it also reopens inflation pressure and limits how far duration-sensitive risk can run."
    else:
        contradiction = "Rates, volatility, and oil still matter more than the latest equity headline."

    tactical_avoid = ", ".join(view.get("underweight_assets") or []) or "speculative beta"
    tactical_focus = view.get("focus") or "selective quality"
    maintain_line = ", ".join(view.get("overweight_assets") or view.get("neutral_assets") or ["Quality Equities"])
    deploy_line = update.get("tactical_positioning") or "Deploy fresh capital only when pullbacks hold and rates stop tightening."
    risk_shift = (
        "A break in any two of the three core triggers forces a regime review and a fresh position-sizing decision."
        if len(invalidates) >= 3
        else "A clear break in price, volatility, or rates is enough to force a new allocation stance."
    )

    lines = [
        "EisaX Daily Market Pulse",
        f"Date: {update.get('date', '')}",
        f"Regime: {regime}",
        f"Confidence: {confidence}",
        f"Decision Type: {decision_type}",
        f"Confidence Score: {confidence_score}",
        "",
        "## Executive Summary",
        f"• {lead}",
        f"• {contradiction}",
        f"• Rates at {rate_level}, VIX at {_num(vix.get('price'))}, oil at {_num(oil.get('price'))}, and Fear & Greed at {fg_score}/100 keep the tape tradable but constrained.",
        f"• {stance_line}",
        f"• Stance stays {view.get('stance', 'HOLD')} in {mode.lower()} mode until the risk framework breaks.",
        "",
        "## Cross-Asset Reality",
        f"• Equities -> SPY is at {_num(us.get('price'))} with {_num(us.get('d1_pct'), pct=True)} on the day and {_num(us.get('d5_pct'), pct=True)} over five sessions, while Nasdaq proxy QQQ is at {_num(qqq.get('price'))} with {_num(qqq.get('d1_pct'), pct=True)}.",
        f"• Rates -> 10Y Treasury is at {rate_level} with {_num(rates.get('d1_pct'), pct=True)} on the day, while DXY proxy UUP is {_num(dxy.get('d1_pct'), pct=True)} on the day and {_num(dxy.get('d5_pct'), pct=True)} over five sessions.",
        f"• Oil -> WTI proxy via USO ETF, not spot barrel price, sits at {_num(oil.get('price'))} with {_num(oil.get('d1_pct'), pct=True)} on the day and {_num(oil.get('d5_pct'), pct=True)} over five sessions; Gold proxy GLD at {_num(gold.get('price'))} and {_num(gold.get('d1_pct'), pct=True)} shows defense has not been abandoned.",
        f"• VIX -> Volatility is at {_num(vix.get('price'))}; hedging demand remains {'elevated' if isinstance(vix.get('price'), (int, float)) and vix.get('price') > 20 else 'contained but not cheap enough for indiscriminate exposure'}.",
        f"• Crypto -> Bitcoin at {_num(btc.get('price'), crypto=True)} is {'confirming' if isinstance(btc_d5, (int, float)) and isinstance(us_d5, (int, float)) and btc_d5 * us_d5 >= 0 else 'not confirming'} the broader risk tape; treat it as a liquidity tell, not a primary signal.",
        "This is not a directional market.",
        f"This is a {market_state} market.",
        "",
        "## Regional Read (GCC + Egypt)",
        f"• GCC stays anchored to oil at {_num(oil.get('price'))}; Saudi market composite is {_num(gcc.get('d1_pct'), pct=True)} on the day and {_num(gcc.get('d5_pct'), pct=True)} over five sessions, while Egypt market composite is {_num(egypt.get('d1_pct'), pct=True)} on the day and {_num(egypt.get('d5_pct'), pct=True)} over five sessions.",
        f"• US rates at {rate_level} matter more than headline equity strength for regional liquidity: GCC can absorb that pressure better than Egypt, while Egypt remains exposed to imported inflation and funding costs.",
    ]
    for key in ("ksa", "uae", "egypt"):
        internal_line = _format_internal_line_en(regional_internals.get(key) or {})
        if internal_line:
            lines.append(f"• {internal_line}")
    if decoupling:
        if decoupling.get("method") == "correlation":
            lines.append(
                f"• GCC Oil Beta Decoupling -> DFM real-estate correlation to the WTI-linked proxy now {decoupling.get('correlation', 0):+.2f} "
                f"over the last {decoupling.get('sample_size', 0)} snapshots; decoupling score {decoupling.get('score', 0)}/100 ({decoupling.get('signal', 'No Signal')})."
            )
        else:
            lines.append(
                f"• GCC Oil Beta Decoupling -> score {decoupling.get('score', 0)}/100. "
                f"DFM real-estate basket {decoupling.get('latest_dfm_re_change', 0):+.2f}% versus WTI-linked proxy {decoupling.get('latest_wti_change', 0):+.2f}% "
                f"across the latest {decoupling.get('sample_size', 0)} snapshots ({decoupling.get('signal', 'No Signal')})."
            )
    lines += [
        "• GCC still favors banks and energy while oil stays firm; Egypt still needs hard-currency balance sheets, exporters, and defensives.",
        "",
        "## Positioning",
        f"Stance: {view.get('stance', 'HOLD')}",
        f"Mode: {mode}",
        "Execution:",
        "• Maintain core quality exposure, but keep gross risk capped until price and liquidity confirm together.",
        "• Deploy only on weakness that holds structure, not on emotional upside extension.",
        "• Recycle capital from speculative beta into quality, defense, and liquid optionality.",
        "",
        "## Risk Framework",
        f"• SPY level: {invalidates[0] if len(invalidates) > 0 else 'SPY must hold the current range or the tape loses sponsorship.'}",
        f"• VIX level: {invalidates[1] if len(invalidates) > 1 else 'VIX needs to stay contained; a renewed volatility spike would tighten the risk budget immediately.'}",
        f"• 10Y level: {invalidates[2] if len(invalidates) > 2 else 'A higher 10Y yield would reprice liquidity and cap equity upside.'}",
        f"• Regime shift: {risk_shift}",
        "",
        "## Tactical Playbook",
        f"• Maintain: keep exposure centered on {maintain_line}; quality US leaders, liquid GCC banks and energy, and defense that still earns carry.",
        f"• Deploy: {deploy_line}",
        f"• Avoid: avoid adding to {tactical_avoid} while rates and volatility remain the gating variables.",
        f"• Focus: focus on {tactical_focus}, GCC banks and energy when oil confirms, UAE real estate only when breadth stabilizes, and Egypt only through balance-sheet quality.",
        "",
        "## Catalysts",
    ]
    for item in (triggers or ["Fed communications", "CPI / PCE data release", "Oil and rate repricing"])[:4]:
        lines.append(f"• {item}")
    lines += [
        "",
        "## Final Line",
        "Capital should follow liquidity, not headlines; until rates and volatility confirm the next leg, the only winning aggression is selective aggression.",
    ]
    return "\n".join(lines).strip()

def _build_cio_daily_report_ar_v2(update: dict) -> str:
    view = update.get("eisax_view", {}) if isinstance(update.get("eisax_view"), dict) else {}
    snapshot = update.get("cross_asset_snapshot") or {}
    regime = update.get("market_regime", "Cautious")
    confidence = update.get("regime_confidence", "Low")
    decision_type = update.get("decision_type") or _daily_decision_type(view)
    confidence_score = update.get("confidence_score") or _daily_confidence_score(regime, confidence, update.get("fear_greed_index"))
    mode = update.get("positioning_mode") or _daily_positioning_mode(view)
    market_state = _daily_market_state(snapshot, regime)
    invalidates = _clean_text_list(update.get("what_invalidates"), 4)
    triggers = _clean_text_list(update.get("next_triggers"), 4)
    market_levels = update.get("market_levels") or {}
    regional_internals = update.get("regional_internals") or {}
    decoupling = update.get("gcc_decoupling") or {}

    regime_ar = _ar_label(regime, {"Bullish": "صعودي", "Bearish": "هبوطي", "Cautious": "حذر", "Conflicted": "متضارب"}, "حذر")
    conf_ar = _ar_label(confidence, {"High": "عالية", "Medium": "متوسطة", "Low": "منخفضة"}, "منخفضة")
    decision_ar = _ar_label(decision_type, {
        "HOLD_ACTIVE_CONSTRAINT": "احتفاظ بقيود نشطة",
        "BUY_SELECTIVE": "شراء انتقائي",
        "REDUCE": "خفض المخاطر",
    }, decision_type)
    mode_ar = _ar_label(mode, {"Active Constraint": "قيود نشطة", "Selective": "انتقائي", "Defensive": "دفاعي"}, mode)
    stance_ar = _ar_label(_clean_text(view.get("stance")), {
        "HOLD": "احتفاظ",
        "Tactical BUY": "شراء تكتيكي",
        "REDUCE RISK": "خفض المخاطر",
    }, _clean_text(view.get("stance")))
    state_ar = _ar_label(market_state, {"conflicted": "تموضع متضارب", "transition": "انتقالية", "positioning": "تموضع"}, market_state)
    state_line_ar = "هذا نظام متضارب." if market_state == "conflicted" else f"هذا سوق {state_ar}."

    us = snapshot.get("us_equities") or {}
    rates = snapshot.get("rates") or {}
    oil = snapshot.get("commodities") or {}
    vix = snapshot.get("volatility") or {}
    btc = snapshot.get("crypto") or {}
    gcc = snapshot.get("gcc") or {}
    egypt = snapshot.get("egypt") or {}
    qqq = market_levels.get("nasdaq") or {}
    gold = market_levels.get("gold") or {}
    dxy = market_levels.get("dxy") or {}
    fg_score = update.get("fear_greed_index", 50)

    maintain_line = _translate_phrase_ar(", ".join(view.get("overweight_assets") or view.get("neutral_assets") or ["Quality Equities"]))
    avoid_line = _translate_phrase_ar(", ".join(view.get("underweight_assets") or ["High Beta", "Speculative Crypto"]))
    focus_line = _translate_phrase_ar(_clean_text(view.get("focus")) or "الجودة الانتقائية")

    lines = [
        "EisaX Daily Market Pulse",
        f"Date: {update.get('date', '')}",
        f"Regime: {regime_ar}",
        f"Confidence: {conf_ar}",
        f"Decision Type: {decision_ar}",
        f"Confidence Score: {confidence_score}",
        "",
        "## الملخص التنفيذي",
        "• السوق قابل للاستثمار، لكن فقط عبر انضباط في الحجم وقيود نشطة على المخاطر.",
        f"• عوائد الخزانة عند {_ar_num(rates.get('price'), rates=True)} ما زالت تضبط ميزانية السيولة قبل أي ارتياح حقيقي في الأسهم.",
        f"• النفط عند {_ar_num(oil.get('price'))}، وVIX عند {_ar_num(vix.get('price'))}، ومؤشر الخوف والطمع عند {fg_score}/100؛ لذلك السوق قابل للتداول لكنه غير مؤهل لبيتا واسعة.",
        f"• الموقف الحالي هو {stance_ar} ضمن وضع {mode_ar}، لا لأن السوق ضعيف فقط، بل لأن السيولة ما زالت أغلى من السردية.",
        "",
        "## واقع الأصول المتقاطعة",
        f"• الأسهم -> SPY عند {_ar_num(us.get('price'))} مع {_ar_num(us.get('d1_pct'), pct=True)} يوميًا و{_ar_num(us.get('d5_pct'), pct=True)} خلال خمسة أيام، بينما ناسداك عبر QQQ عند {_ar_num(qqq.get('price'))} مع {_ar_num(qqq.get('d1_pct'), pct=True)}. السعر متماسك، لكن اتساع المخاطرة ما زال محدودًا.",
        f"• العوائد -> عائد 10 سنوات عند {_ar_num(rates.get('price'), rates=True)} مع تحرك يومي {_ar_num(rates.get('d1_pct'), pct=True)}، بينما الدولار عبر UUP عند {_ar_num(dxy.get('d1_pct'), pct=True)} يوميًا و{_ar_num(dxy.get('d5_pct'), pct=True)} خلال خمسة أيام؛ السيولة ما زالت أهم من عنوان السوق نفسه.",
        f"• النفط -> خام WTI proxy عبر صندوق USO ETF وليس سعر البرميل الفوري عند {_ar_num(oil.get('price'))} مع {_ar_num(oil.get('d1_pct'), pct=True)} يوميًا و{_ar_num(oil.get('d5_pct'), pct=True)} خلال خمسة أيام؛ وهو داعم لتدفقات الخليج، لكنه يبقي قناة التضخم مفتوحة. الذهب عبر GLD عند {_ar_num(gold.get('price'))} مع {_ar_num(gold.get('d1_pct'), pct=True)} يؤكد أن التحوط لم يختفِ.",
        f"• VIX -> التذبذب عند {_ar_num(vix.get('price'))}؛ الخوف ليس حادًا، لكنه ليس منخفضًا بما يكفي لتبرير شراء بيتا بلا تمييز.",
        f"• الكريبتو -> بيتكوين عند {_ar_num(btc.get('price'), crypto=True)} ويُقرأ كمؤشر سيولة ثانوي لا كمرتكز التخصيص الأساسي.",
        "هذا ليس سوقًا اتجاهيًا.",
        f"هذا سوق {state_ar}.",
        "",
        "## القراءة الإقليمية (الخليج + مصر)",
        f"• الخليج ما زال مربوطًا بالنفط عند {_ar_num(oil.get('price'))}؛ المركب السعودي عند {_ar_num(gcc.get('d1_pct'), pct=True)} يوميًا و{_ar_num(gcc.get('d5_pct'), pct=True)} خلال خمسة أيام، بينما المركب المصري عند {_ar_num(egypt.get('d1_pct'), pct=True)} يوميًا و{_ar_num(egypt.get('d5_pct'), pct=True)} خلال خمسة أيام.",
        f"• عوائد 10 سنوات عند {_ar_num(rates.get('price'), rates=True)} أهم لمصر من عنوان الأسهم الأميركية، لأن ضغط الدولار والتمويل الخارجي ما زال حاضرًا.",
    ]
    for key, title in (("ksa", "السعودية"), ("uae", "الإمارات"), ("egypt", "مصر")):
        internal_line = _format_internal_line_ar(regional_internals.get(key) or {}, title)
        if internal_line:
            lines.append(f"• {internal_line}")
    if decoupling:
        if decoupling.get("method") == "correlation":
            lines.append(
                f"• انفصال بيتا النفط في الخليج -> ارتباط سلة عقارات دبي مع proxy النفط المرتبط بـ WTI بلغ الآن {decoupling.get('correlation', 0):+.2f} "
                f"عبر آخر {decoupling.get('sample_size', 0)} لقطات، مع درجة انفصال {decoupling.get('score', 0)}/100 "
                f"({_translate_phrase_ar(decoupling.get('signal', 'إشارة غير متاحة')) or decoupling.get('signal', '')})."
            )
        else:
            lines.append(
                f"• انفصال بيتا النفط في الخليج -> درجة الانفصال الآن {decoupling.get('score', 0)}/100. "
                f"سلة عقارات دبي عند {decoupling.get('latest_dfm_re_change', 0):+.2f}% مقابل proxy النفط المرتبط بـ WTI عند {decoupling.get('latest_wti_change', 0):+.2f}% "
                f"عبر آخر {decoupling.get('sample_size', 0)} لقطات ({_translate_phrase_ar(decoupling.get('signal', 'إشارة غير متاحة')) or decoupling.get('signal', '')})."
            )
    lines += [
        "• التدفقات القطاعية ما زالت تميل إلى بنوك وطاقة الخليج، بينما تحتاج مصر إلى ميزانيات قوية ومصادر عملة صعبة ودفاعيات تشغيلية.",
        "",
        "## التموضع",
        f"الموقف: {stance_ar}",
        f"الوضع: {mode_ar}",
        "التنفيذ:",
        "• حافظ على المراكز الأساسية عالية الجودة، لكن لا توسع الانكشاف إلا على تراجعات منضبطة.",
        "• أضف فقط عندما يثبت السعر ويهدأ ضغط العوائد والتذبذب معًا، لا عندما يصعد الشريط وحده.",
        "• أعد تدوير رأس المال من البيتا المضاربية إلى الجودة والسيولة والدفاعيات.",
        "",
        "## إطار المخاطر",
        f"• SPY: {_translate_risk_trigger_ar(invalidates[0]) if len(invalidates) > 0 else 'أي كسر واضح في البنية الحالية لـ SPY يعني أن السوق يفقد الرعاية السعرية.'}",
        f"• VIX: {_translate_risk_trigger_ar(invalidates[1]) if len(invalidates) > 1 else 'أي ارتفاع واضح في VIX يعني تضييقًا مباشرًا في ميزانية المخاطر.'}",
        f"• 10Y: {_translate_risk_trigger_ar(invalidates[2]) if len(invalidates) > 2 else 'أي ارتفاع إضافي في عائد 10 سنوات سيضغط على التقييمات والسيولة.'}",
        "• تغير النظام: القرار يتغير فقط عندما تتحرك الأسعار والعوائد والتذبذب معًا، لا عندما يتحرك عامل واحد منفردًا.",
        "",
        "## الدليل التكتيكي",
        f"• حافظ على: {maintain_line}.",
        "• انشر رأس المال عند: تراجعات منضبطة تثبت فوق الدعم مع هدوء متزامن في العوائد وVIX.",
        f"• تجنب: {avoid_line}.",
        f"• ركز على: {focus_line}، مع بنوك وطاقة الخليج حين يؤكد النفط، والإمارات فقط عندما يستقر اتساع السوق، ومصر عبر الميزانيات الأقوى.",
        "",
        "## المحفزات",
    ]
    for item in (triggers or ["خطاب الفيدرالي", "بيانات CPI / PCE", "تحرك النفط", "العوائد الأميركية"])[:4]:
        lines.append(f"• {_translate_catalyst_ar(item)}")
    lines += [
        "",
        "## الخلاصة",
        "• رأس المال يجب أن يتبع السيولة لا العناوين؛ وحتى تؤكد العوائد والتذبذب المرحلة التالية، يبقى الهجوم الصحيح هو الهجوم الانتقائي فقط.",
    ]
    return "\n".join(lines).strip()

def _build_cio_weekly_report_ar(update: dict) -> str:
    alloc = update.get("asset_allocation_view") or {}
    snapshot = update.get("cross_asset_snapshot") or {}
    regime = update.get("market_regime", "Cautious")
    confidence = update.get("regime_confidence", "Low")
    decision_type = update.get("weekly_decision_type") or _weekly_decision_type(update)
    confidence_score = update.get("weekly_confidence_score") or _weekly_confidence_score(regime, confidence, update)
    mode = update.get("weekly_positioning_mode") or _weekly_positioning_mode(update)
    changes = _clean_text_list(update.get("what_changes_this_view"), 4)
    state = _weekly_market_state(update)

    regime_ar = _ar_label(regime, {"Bullish": "صعودي", "Bearish": "هبوطي", "Cautious": "حذر"}, "حذر")
    conf_ar = _ar_label(confidence, {"High": "عالية", "Medium": "متوسطة", "Low": "منخفضة"}, "منخفضة")
    decision_ar = _ar_label(decision_type, {
        "HOLD_ACTIVE_CONSTRAINT": "احتفاظ بقيود نشطة",
        "BUY_SELECTIVE": "شراء انتقائي",
        "REDUCE": "خفض المخاطر",
    }, decision_type)
    mode_ar = _ar_label(mode, {"Active Constraint": "قيود نشطة", "Selective": "انتقائي", "Defensive": "دفاعي"}, mode)
    state_ar = _ar_label(state, {"conflicted": "تموضع متضارب", "transition": "انتقالية", "positioning": "تموضع"}, state)

    us = snapshot.get("us_equities") or {}
    rates = snapshot.get("rates") or {}
    oil = snapshot.get("commodities") or {}
    vix = snapshot.get("volatility") or {}
    btc = snapshot.get("crypto") or {}

    overweight = [k for k, v in alloc.items() if v == "Overweight"]
    maintain_text = ", ".join(overweight) if overweight else "الجودة والسيولة"

    if regime == "Bearish":
        summary_ar = (
            f"العوائد بقيت المحرك الأول للأسبوع، مع 10 سنوات عند {_ar_num(rates.get('price'), rates=True)}، وهو ما أبقى السيولة ضيقة عبر معظم الأصول. "
            f"أداء SPY عند {_ar_num(us.get('d5_pct'), pct=True)} لم يكن كافيًا لتبرير بيتا واسعة، بينما النفط عند {_ar_num(oil.get('price'))} أعاد إبقاء التضخم داخل الصورة. "
            "كان هذا أسبوع حماية رأس المال لا أسبوع مطاردة للعائد."
        )
        positioning_ar = "خفض المخاطر ورفع الوزن الدفاعي في النقد والذهب والمدة القصيرة، مع التعامل مع أي ارتداد على أنه تكتيكي لا هيكلي."
        angle_ar = "أدر الدفتر بعقلية دفاعية، واحتفظ فقط بالجودة القادرة على امتصاص تشديد السيولة، ولا توسع الانكشاف قبل هدوء العوائد والتذبذب معًا."
        verdict_ar = "خفّض المخاطر واحمِ رأس المال حتى تتحسن السيولة بوضوح."
        conviction_ar = "الذهب عند التراجعات يظل التعبير الأعلى قناعة طالما بقيت العوائد مرتفعة ومسار التضخم مفتوحًا."
    elif regime == "Bullish":
        summary_ar = (
            f"القيادة بقيت مع الأسهم الأميركية عالية الجودة، إذ أنهى SPY الأسبوع عند {_ar_num(us.get('d5_pct'), pct=True)} واحتفظت ناسداك بميزة القيادة. "
            f"لكن 10 سنوات عند {_ar_num(rates.get('price'), rates=True)} والنفط عند {_ar_num(oil.get('price'))} أبقيا الصعود انتقائيًا لا شاملًا. "
            "كان هذا أسبوع قيادة نوعية لا أسبوع شراء أعمى للمخاطرة."
        )
        positioning_ar = "أضف انتقائيًا إلى القيادة عالية الجودة على التراجعات، مع إبقاء الانكشاف الجديد ممولًا من تدوير المراكز الأضعف لا من مطاردة السوق."
        angle_ar = "ابقَ طويلًا في الجودة، وموّل الإضافات من المراكز الضعيفة، وكن جاهزًا لشراء الضعف لا لشراء التمدد السعري."
        verdict_ar = "أضف انتقائيًا إلى القيادة، لكن لا تخلط بين شريط قابل للاستثمار وإشارة شراء شاملة."
        conviction_ar = "التكنولوجيا الأميركية عالية الجودة على التراجعات تظل أفضل فرصة ما دام ضغط العوائد لا يتسارع."
    else:
        summary_ar = (
            f"العوائد بقيت حاكمة للشريط الأسبوعي، مع 10 سنوات عند {_ar_num(rates.get('price'), rates=True)}، رغم أن SPY أنهى الأسبوع عند {_ar_num(us.get('d5_pct'), pct=True)}. "
            f"النفط عند {_ar_num(oil.get('price'))} وVIX عند {_ar_num(vix.get('price'))} منعا السوق من التحول إلى نظام صريح للمخاطرة، بينما بيتكوين أكد فقط سيولة انتقائية. "
            "كان هذا أسبوع تموضع لا أسبوع تخصيص واسع."
        )
        positioning_ar = "احتفاظ بقيود نشطة: أبقِ الانكشاف الأساسي في الجودة، وانشر رأس المال فقط على الضعف المنضبط، وابقِ الأذرع المضاربية تحت الوزن."
        angle_ar = "أدر الدفتر بانكشاف إجمالي مضبوط، وأعد تدوير رأس المال إلى الجودة والدفاعيات، وانتظر تأكيد العوائد والتذبذب قبل الضغط على البيتا."
        verdict_ar = "احتفظ بقيود نشطة، ولا تضف إلا حيث تتوافق السيولة مع القيادة."
        conviction_ar = "السندات الأميركية قصيرة الأجل تظل التعبير الأنظف عن العائد إلى أن تحصل الأصول الخطرة على ذيل سيولة أوضح."

    lines = [
        "EisaX Weekly Strategy Brief",
        f"Week: {update.get('week_range', '')}",
        f"Regime: {regime_ar}",
        f"Confidence: {conf_ar}",
        f"Decision Type: {decision_ar}",
        f"Confidence Score: {confidence_score}",
        "",
        "## الملخص التنفيذي",
        f"• {summary_ar}",
        f"• {positioning_ar}",
        f"• {angle_ar}",
        f"• الحكم الأسبوعي: {verdict_ar}",
        "",
        "## واقع الأصول المتقاطعة",
        f"• الأسهم -> SPY أغلق الأسبوع عند {_ar_num(us.get('price'))} مع {_ar_num(us.get('d5_pct'), pct=True)} خلال خمسة أيام؛ القيادة موجودة، لكنها ليست واسعة بما يكفي لتبرير بيتا شاملة.",
        f"• العوائد -> عائد 10 سنوات أنهى الأسبوع قرب {_ar_num(rates.get('price'), rates=True)}؛ وما زال هو بوابة التوسّع في المضاعفات والمخاطرة.",
        f"• النفط -> النفط عند {_ar_num(oil.get('price'))} مع {_ar_num(oil.get('d5_pct'), pct=True)} أسبوعيًا؛ يدعم تدفقات الخليج لكنه يبقي التضخم حاضرًا داخل المحافظ.",
        f"• التذبذب -> VIX عند {_ar_num(vix.get('price'))}؛ لا يوجد ذعر، لكن التحوط والانضباط في الحجم ما زالا ضروريين.",
        f"• الكريبتو -> بيتكوين عند {_ar_num(btc.get('price'), crypto=True)} مع {_ar_num(btc.get('d5_pct'), pct=True)} أسبوعيًا؛ مؤشر سيولة ثانوي لا مرساة تخصيص رئيسية.",
        "هذا ليس أسبوع بيتا واسعة.",
        f"هذا أسبوع {state_ar}.",
        "",
        "## القراءة الإقليمية (الخليج + مصر)",
        f"• {_clean_text((update.get('regional_view') or {}).get('GCC'))}",
        f"• {_clean_text((update.get('regional_view') or {}).get('Egypt'))}",
        "• الخليج يستوعب النفط المرتفع أفضل بكثير من استيعاب مصر للعوائد المرتفعة وضغط الدولار.",
        "",
        "## قرار التخصيص",
        f"الموقف: {decision_ar}",
        f"الوضع: {mode_ar}",
        "التنفيذ:",
        f"• حافظ على الانكشاف حيث يظل التخصيص مائلاً إلى {maintain_text}.",
        "• زد فقط على الضعف المنضبط، لا على الصعود المتأخر.",
        "• لا تطارد بيتا عريضة ما لم تؤكد العوائد والتذبذب والسعر الاتجاه نفسه معًا.",
        "",
        "## إطار المخاطر",
        f"• SPY: {changes[0] if len(changes) > 0 else 'كسر البنية الأسبوعية في SPY يفرض تشديد التموضع.'}",
        f"• VIX: {changes[1] if len(changes) > 1 else 'عودة VIX للصعود الواضح تعني العودة الفورية إلى الدفاع.'}",
        f"• 10Y: {changes[2] if len(changes) > 2 else 'أي ارتفاع جديد في عائد 10 سنوات يعيد تسعير السيولة ويضغط على الشهية للمخاطرة.'}",
        "• تغير النظام: القناعة الأسبوعية تتغير فقط إذا تحرك السعر والتذبذب والعوائد في الاتجاه نفسه.",
        "",
        "## الدليل التكتيكي",
        f"• حافظ على: {maintain_text}.",
        f"• أضف عبر: {conviction_ar}",
        "• تجنب: البيتا الواسعة والأسهم التي تعتمد على توسع المضاعفات وحده.",
        f"• ركز على: {angle_ar}",
        "",
        "## المحفزات",
        "• خطاب الفيدرالي وإعادة تسعير الفائدة",
        "• CPI / PCE واستمرار التضخم",
        "• النفط وأثره على سيولة الخليج",
        "• العوائد الأميركية والدولار وتأثيرهما على التمويل الإقليمي",
        "",
        "## الخلاصة",
        "• دفتر الأسبوع يجب أن يتبع السيولة والقيادة، لا راحة المؤشرات الرئيسية وحدها.",
    ]
    return "\n".join(lines).strip()

def _apply_daily_consistency(update: dict, fallback: dict, snapshot: dict) -> dict:
    update["what_matters_now"] = _merge_text_list_with_fallback(update.get("what_matters_now"), fallback.get("what_matters_now", []), 3)
    update["key_moves"] = _normalize_key_moves(update.get("key_moves"), fallback.get("key_moves", []))
    update["cross_asset_snapshot"] = snapshot or fallback.get("cross_asset_snapshot", {})
    update["why_now"] = _clean_text(update.get("why_now")) or _clean_text(fallback.get("why_now"))
    update["tactical_positioning"] = _clean_text(update.get("tactical_positioning")) or _clean_text(fallback.get("tactical_positioning"))
    update["next_triggers"] = _merge_text_list_with_fallback(update.get("next_triggers"), fallback.get("next_triggers", []), 3)
    if update.get("fear_greed_index") is None:
        update["fear_greed_index"] = fallback.get("fear_greed_index")
    return update

def _apply_weekly_consistency(update: dict, fallback: dict) -> dict:
    update["market_summary"] = _clean_text(update.get("market_summary")) or _clean_text(fallback.get("market_summary"))
    update["positioning"] = _clean_text(update.get("positioning")) or _clean_text(fallback.get("positioning"))
    update["cross_asset_snapshot"] = update.get("cross_asset_snapshot") or fallback.get("cross_asset_snapshot", {})
    update["regional_view"] = _normalize_regional_view(update.get("regional_view"), fallback.get("regional_view", {}))
    update["winners_losers"] = _normalize_winners_losers(update.get("winners_losers"), fallback.get("winners_losers", {}))
    update["highest_conviction_opportunity"] = _clean_text(update.get("highest_conviction_opportunity")) or _clean_text(fallback.get("highest_conviction_opportunity"))
    update["key_risks"] = _merge_text_list_with_fallback(update.get("key_risks"), fallback.get("key_risks", []), 3)
    update["portfolio_angle"] = _clean_text(update.get("portfolio_angle")) or _clean_text(fallback.get("portfolio_angle"))
    update["eisax_verdict"] = _clean_text(update.get("eisax_verdict")) or _clean_text(fallback.get("eisax_verdict"))
    if update.get("fear_greed_index") is None:
        update["fear_greed_index"] = fallback.get("fear_greed_index")
    return update

def _generate_arabic_report(update: dict, is_weekly: bool = False) -> str:
    """Generate an Arabic-language intelligence report. Uses LLM with deterministic fallback."""
    import re as _re

    def _ar_daily_fallback(u):
        snap = u.get("cross_asset_snapshot", {})
        def fn(v, p):
            try:
                f = float(v)
                return "N/A" if f == 0 else f"{f:.{p}f}"
            except:
                return "N/A"
        spy = fn(snap.get("us_equities", {}).get("price", 0), 0)
        vix = fn(snap.get("volatility", {}).get("price", 0), 1)
        tnx = fn(snap.get("rates", {}).get("price", 0), 2)
        oil = fn(snap.get("commodities", {}).get("price", 0), 1)
        btc_raw = snap.get("crypto", {}).get("price", 0)
        try:
            btc = f"${float(btc_raw)/1000:.1f}k" if float(btc_raw) > 1000 else f"${float(btc_raw):.0f}"
        except:
            btc = "N/A"
        view = u.get("eisax_view", {}) if isinstance(u.get("eisax_view"), dict) else {}
        stance_map = {"Tactical BUY": "شراء تكتيكي", "HOLD": "احتفاظ", "REDUCE RISK": "تخفيض المخاطر"}
        stance = stance_map.get(view.get("stance", "HOLD"), view.get("stance", "احتفاظ"))
        regime = u.get("market_regime", "متحفظ")
        regime_map = {"Bullish": "صعودي", "Bearish": "هبوطي", "Cautious": "متحفظ", "Neutral": "محايد"}
        regime_ar = regime_map.get(regime, regime)
        conf_map = {"High": "عالية", "Medium": "متوسطة", "Low": "منخفضة"}
        conf = conf_map.get(u.get("regime_confidence", "Low"), "منخفضة")
        why = u.get("why_now", "الشريط قابل للاستثمار ولكن الاقتناع لا يزال انتقائياً.")
        lines = [
            f"## تقرير إيساكس الاستخباراتي — الأسواق العالمية",
            f"**التاريخ:** {u.get('date', '')} | **النظام السائد:** {regime_ar} ({conf} الاقتناع)",
            "",
            "### 1. الملخص التنفيذي",
            f"مؤشر S&P 500 عند {spy} ومؤشر VIX قرب {vix} يدعمان هيكلاً {regime_ar}. النفط الخام WTI عند ${oil} يحدد التحيز الافتتاحي للخليج، مبقياً قطاع الطاقة الإقليمي نشطاً في حين تضغط عوائد السندات الأمريكية لـ10 سنوات عند {tnx}% على سيولة الأسواق الناشئة.",
            f"**لماذا الآن:** {why}",
            "",
            "### 2. نظرة عامة على الأسواق العالمية",
            f"يتمسك S&P بمستوى {spy} بينما تتتبع عوائد الخزانة الأمريكية لـ10 سنوات مستوى {tnx}%. يعني انضغاط VIX عند {vix} أن أسواق الخيارات لا تسعّر مخاطر هبوطية وشيكة. النفط WTI عند ${oil} وبيتكوين قرب {btc} يدفعان رأس المال نحو الأسهم الأمريكية / النمو عالي الجودة.",
            "",
            "### 3. قراءة الأسواق الإقليمية (الخليج + مصر)",
            f"النفط فوق ${oil} يوفر أرضية اتجاهية صلبة لمؤشر تداول السعودي وأوامر الافتتاح الإماراتية. تمتص أسهم البنوك والطاقة تدفقات السيولة اليومية. في مصر، تُلزم معدلات الفائدة الأمريكية عند {tnx}% مؤشر EGX30 بالتنقل في رياح السياسة النقدية الهيكلية — مفضلةً الميزانيات الدفاعية على بيتا السلع الاستهلاكية.",
        ]
        matters = u.get("what_matters_now") or []
        matter_ar_map = {
            "Equities advancing into fear": "تقدم الأسهم رغم الخوف — الموقف لا يزال غير ممتلئ — يدعم الاستمرارية",
            "Energy deflation": "تراجع الطاقة إلى جانب معدلات هادئة — يزيل الضغط الكلي الصامت — يبقي نمو الأرباح المحدد الرئيسي",
            "Volatility": f"VIX يتماسك قرب {vix} — الطلب على التحوط لا يزال قائماً — التحرك انتقائي لا واسع",
        }
        lines.append("")
        lines.append("### 4. ما يقود الشريط")
        for m in matters[:3]:
            matched = next((v for k, v in matter_ar_map.items() if k.lower() in m.lower()), m)
            lines.append(f"- {matched}")
        lines += [
            "",
            "### 5. التموضع والاستراتيجية",
            f"تتبنى إيساكس موقف **{stance}** على أفق تأرجح. بناء المحافظ في الأسهم الأمريكية / النمو عالي الجودة عالمياً. إقليمياً، زيادة الوزن في طاقة الخليج والبنوك المرتبطة بخام ${oil}. تجنب بدائل الأسواق الناشئة عالية البيتا حتى تنضغط العوائد دون {tnx}%.",
            "",
            "### 6. إطار المخاطر",
            "يظل هذا الرأي الهيكلي قائماً ما لم تُطلق الأسعار نقاط إبطال محددة:",
        ]
        for inv in (u.get("what_invalidates") or [])[:4]:
            lines.append(f"- {inv}")
        lines += [
            "",
            "### 7. دليل التكتيك",
            f"- {u.get('tactical_positioning', 'الحفاظ على التعرض الطويل، التنفيذ حول مستويات الإبطال المحددة.')}",
        ]
        for trig in (u.get("next_triggers") or [])[:3]:
            lines.append(f"- {trig}")
        lines += [
            "",
            "### 8. ما ينبغي مراقبته",
            f"المتحدثون من الفيدرالي الأمريكي، بيانات مؤشر أسعار المستهلك القادمة، وبيانات مخزون النفط من إدارة معلومات الطاقة. مراقبة عوائد الخزانة الأمريكية لـ10 سنوات — أي ارتفاع فوق {tnx}% يُكبّل فوراً صعود بنوك الأسواق الناشئة والخليج.",
            "",
            "### 9. الخلاصة",
            f"ثق بالشريط. S&P عند {spy} وVIX عند {vix} يُملي التنقل في الاتجاه عبر الأسهم الأمريكية / النمو عالي الجودة. التنفيذ الدقيق حول نقاط الإبطال وتحديد حجم التعرض غير المحوط.",
        ]
        return "\n".join(lines).strip()

    def _ar_weekly_fallback(u):
        alloc = u.get("asset_allocation_view") or {}
        regime = u.get("market_regime", "متحفظ")
        regime_map = {"Bullish": "صعودي", "Bearish": "هبوطي", "Cautious": "متحفظ", "Neutral": "محايد"}
        regime_ar = regime_map.get(regime, regime)
        conf_map = {"High": "عالية", "Medium": "متوسطة", "Low": "منخفضة"}
        conf = conf_map.get(u.get("regime_confidence", "Low"), "منخفضة")
        snap = u.get("cross_asset_snapshot", {})
        def fn(v, p):
            try:
                f = float(v)
                return "N/A" if f == 0 else f"{f:.{p}f}"
            except:
                return "N/A"
        oil = fn(snap.get("commodities", {}).get("price", 0), 1)
        tnx = fn(snap.get("rates", {}).get("price", 0), 2)
        vix = fn(snap.get("volatility", {}).get("price", 0), 1)
        spy = fn(snap.get("us_equities", {}).get("price", 0), 0)
        alloc_map = {"Overweight": "وزن زائد", "Underweight": "وزن منقوص", "Neutral": "محايد"}
        eq_ar = alloc_map.get(alloc.get("equities", "Neutral"), "محايد")
        metals_ar = alloc_map.get(alloc.get("metals", "Neutral"), "محايد")
        cash_ar = alloc_map.get(alloc.get("cash", "Neutral"), "محايد")
        conviction = u.get("highest_conviction_opportunity", "التكنولوجيا الأمريكية عالية الجودة عند التراجعات.")
        verdict = u.get("eisax_verdict", "احتفاظ مع إضافة انتقائية.")
        changes = u.get("what_changes_this_view") or ["كسر SPY تحت $694", "VIX يغلق فوق 25", "ارتفاع عائد الخزانة 10 سنوات فوق 4.5%"]
        lines = [
            f"## الموجز الاستراتيجي الأسبوعي من إيساكس — الاقتصاد الكلي العالمي والخليج",
            f"**الفترة:** {u.get('week_range', '')} | **النظام السائد:** {regime_ar} ({conf} الاقتناع)",
            "",
            "### 1. ملخص الاستراتيجية التنفيذية",
            f"تحافظ التوقعات الأسبوعية على موقف {regime_ar}. يتمحور التركيز الاستراتيجي حول الحفاظ على رأس المال في الأسواق الناشئة إلى جانب التعامل الانتقائي مع الخليج. النفط عند ${oil} والعائد الأمريكي عند {tnx}% يرسمان الإطار الكلي الإقليمي والعالمي.",
            "",
            "### 2. توزيع الأصول العالمية",
            f"الأسهم: {eq_ar} | المعادن: {metals_ar} | النقد: {cash_ar}.",
            "توزيع الأصول يعكس الموقف الاتجاهي مع الحفاظ على هامش للمرونة على مستوى القطاعات.",
            "",
            "### 3. القراءة الاستراتيجية الإقليمية (الخليج + مصر)",
            f"يرتبط مزاج الخليج بالنفط عند ${oil}. تظل سيولة البنوك السعودية والإماراتية متينة، وتمتص قطاعا الطاقة والمصارف التدفقات اليومية. تواصل مصر التعامل مع الضغوط النقدية الهيكلية في ظل عوائد أمريكية عند {tnx}% — مما يُفضّل الميزانيات الدفاعية على التعرض للنمو.",
            "",
            "### 4. المذكرة الاقتناعية",
            f"{conviction}",
            "",
            "### 5. إطار السيولة والعوائد",
            f"مع عائد الخزانة لـ10 سنوات عند {tnx}%، تُفضّل نسبة المخاطرة إلى العائد التداولات التناوبية الداخلية على الرهانات السوقية الواسعة. مؤشر VIX عند {vix} يبقي الإطار الانتقائي قائماً.",
            "",
            "### 6. التموضع في المحفظة",
            f"الحكم: **{verdict}**",
            f"تركيز المحفظة: أسهم أمريكية / نمو عالي الجودة، مع زيادة وزن قطاعي الطاقة والبنوك في الخليج.",
            "",
            "### 7. إطار المخاطر",
            f"يظل التوزيع الأسبوعي ثابتاً ما لم تتحقق المحفزات التالية: {'; '.join(changes[:3])}. إغلاق حاسم واحد عبر أي من هذه المستويات يستدعي إعادة التقييم الفوري.",
            "",
            "### 8. قائمة المراقبة الاستراتيجية",
            f"مراقبة اتصالات الفيدرالي الأمريكي لأي إشارة تحول. بيانات مخزون النفط من إدارة معلومات الطاقة ستؤكد أو تُقوّض أرضية ${oil}+. بيانات مؤشر أسعار المستهلكين القادمة هي الاختبار الكلي الحاسم — أي مفاجأة صعودية تُعيد تسعير مخاطر الخليج والأسواق الناشئة.",
            "",
            "### 9. الخلاصة الاستراتيجية",
            f"الشريط {regime_ar} يكافئ الانضباط. التنفيذ نحو القادة عند الضعف. التخفيض عند الارتفاعات غير المبررة. الحفاظ على الاقتناع الهيكلي: النفط عند ${oil} والخزانة لـ10 سنوات عند {tnx}% يُحددان المظروف الكلي الإقليمي والعالمي.",
        ]
        rpt = "\n".join(lines).strip()
        import re as _r
        rpt = _r.sub(r'\.{2,}', '.', rpt)
        return rpt

    eng_report = update.get("full_report", "")
    if not eng_report:
        return _ar_weekly_fallback(update) if is_weekly else _ar_daily_fallback(update)

    if is_weekly:
        instruction = "أنت محلل استثماري متخصص في الأسواق الخليجية. ترجم الموجز الاستراتيجي الأسبوعي التالي إلى العربية الاحترافية مع الحفاظ على: جميع الأرقام كما هي، بنية الأقسام التسعة، الأسلوب المؤسسي الحاد. نصوص عربية فقط — لا إنجليزية إلا الرموز والأرقام.\n\n"
    else:
        instruction = "أنت محلل استثماري متخصص في الأسواق الخليجية. ترجم التقرير اليومي التالي إلى العربية الاحترافية مع الحفاظ على: جميع الأرقام كما هي، بنية الأقسام التسعة، الأسلوب المؤسسي الحاد. نصوص عربية فقط — لا إنجليزية إلا الرموز والأرقام.\n\n"

    prompt = instruction + eng_report
    raw = _call_openai_text(prompt, max_tokens=1800) or _call_gemini(prompt)
    if raw:
        return _re.sub(r"^```\w*\s*|```$", "", raw.strip(), flags=_re.MULTILINE)

    # LLM failed — use deterministic fallback
    logger.warning("[market_updates] Arabic LLM translation failed — using deterministic Arabic fallback")
    return _ar_weekly_fallback(update) if is_weekly else _ar_daily_fallback(update)

def _daily_snapshot_internal_lines_en(label: str, internal: dict) -> list[tuple[str, str]]:
    if not isinstance(internal, dict) or not internal:
        return []
    lines: list[tuple[str, str]] = []
    adv = internal.get("advancers", 0)
    dec = internal.get("decliners", 0)
    flat = internal.get("unchanged", 0)
    lines.append((f"{label} Breadth", f"{adv} up / {dec} down / {flat} flat"))
    weighted = internal.get("weighted_change")
    if isinstance(weighted, (int, float)):
        lines.append((f"{label} Cap-Weighted Move", f"{weighted:+.2f}%"))
    movers = ", ".join((internal.get("top_gainers") or [])[:2])
    laggards = ", ".join((internal.get("top_losers") or [])[:2])
    if movers:
        lines.append((f"{label} Movers", movers))
    if laggards:
        lines.append((f"{label} Laggards", laggards))
    return lines

def _daily_snapshot_internal_lines_ar(label: str, internal: dict) -> list[tuple[str, str]]:
    if not isinstance(internal, dict) or not internal:
        return []
    lines: list[tuple[str, str]] = []
    adv = internal.get("advancers", 0)
    dec = internal.get("decliners", 0)
    flat = internal.get("unchanged", 0)
    lines.append((f"اتساع {label}", f"{adv} صاعد / {dec} هابط / {flat} دون تغير"))
    weighted = internal.get("weighted_change")
    if isinstance(weighted, (int, float)):
        lines.append((f"الحركة الموزونة {label}", f"{weighted:+.2f}%"))
    movers = "، ".join((internal.get("top_gainers") or [])[:2])
    laggards = "، ".join((internal.get("top_losers") or [])[:2])
    if movers:
        lines.append((f"أبرز صاعدين {label}", movers))
    if laggards:
        lines.append((f"أبرز ضاغطين {label}", laggards))
    return lines

def _daily_snapshot_pairs_en(update: dict) -> list[tuple[str, str]]:
    snapshot = update.get("cross_asset_snapshot") or {}
    market_levels = update.get("market_levels") or {}
    regional_internals = update.get("regional_internals") or {}
    decoupling = update.get("gcc_decoupling") or {}

    def _entry(key: str) -> dict:
        return snapshot.get(key) or {}

    def _num(value: Any, pct: bool = False, rates: bool = False, crypto: bool = False) -> str:
        if isinstance(value, str):
            return value
        if not isinstance(value, (int, float)):
            return "Market Closed"
        if crypto:
            return f"{value:,.0f}"
        if rates:
            return f"{value:.2f}%"
        if pct:
            return f"{value:+.2f}%"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.2f}"

    us = _entry("us_equities")
    rates = _entry("rates")
    oil = _entry("commodities")
    vix = _entry("volatility")
    btc = _entry("crypto")
    gcc = _entry("gcc")
    egypt = _entry("egypt")
    qqq = market_levels.get("nasdaq") or {}
    gold = market_levels.get("gold") or {}
    dxy = market_levels.get("dxy") or {}
    fg_score = update.get("fear_greed_index")

    pairs: list[tuple[str, str]] = [
        ("SPY", f"{_num(us.get('price'))} | Day {_num(us.get('d1_pct'), pct=True)} | 5D {_num(us.get('d5_pct'), pct=True)}"),
        ("Nasdaq Proxy (QQQ)", f"{_num(qqq.get('price'))} | Day {_num(qqq.get('d1_pct'), pct=True)}"),
        ("VIX", _num(vix.get("price"))),
        ("10Y", f"{_num(rates.get('price'), rates=True)} | Day {_num(rates.get('d1_pct'), pct=True)}"),
        ("WTI Proxy (USO)", f"{_num(oil.get('price'))} | Day {_num(oil.get('d1_pct'), pct=True)} | 5D {_num(oil.get('d5_pct'), pct=True)}"),
        ("Gold Proxy (GLD)", f"{_num(gold.get('price'))} | Day {_num(gold.get('d1_pct'), pct=True)}"),
        ("DXY Proxy (UUP)", f"{_num(dxy.get('price'))} | Day {_num(dxy.get('d1_pct'), pct=True)} | 5D {_num(dxy.get('d5_pct'), pct=True)}"),
        ("BTC", f"{_num(btc.get('price'), crypto=True)} | 5D {_num(btc.get('d5_pct'), pct=True)}"),
        ("Fear & Greed", f"{float(fg_score):.1f}/100" if isinstance(fg_score, (int, float)) else str(fg_score or "N/A")),
        ("Saudi Composite", f"Day {_num(gcc.get('d1_pct'), pct=True)} | 5D {_num(gcc.get('d5_pct'), pct=True)}"),
        ("Egypt Composite", f"Day {_num(egypt.get('d1_pct'), pct=True)} | 5D {_num(egypt.get('d5_pct'), pct=True)}"),
    ]
    for key, label in (("ksa", "Saudi"), ("uae", "UAE"), ("egypt", "Egypt")):
        pairs.extend(_daily_snapshot_internal_lines_en(label, regional_internals.get(key) or {}))
    if decoupling:
        if decoupling.get("method") == "correlation":
            pairs.append((
                "GCC Oil Beta Decoupling",
                f"{decoupling.get('score', 0)}/100 | Correlation {decoupling.get('correlation', 0):+.2f} | Sample {decoupling.get('sample_size', 0)} snapshots | {decoupling.get('signal', 'No Signal')}",
            ))
        else:
            pairs.append((
                "GCC Oil Beta Decoupling",
                f"{decoupling.get('score', 0)}/100 | DFM Real Estate {decoupling.get('latest_dfm_re_change', 0):+.2f}% | WTI-linked Proxy {decoupling.get('latest_wti_change', 0):+.2f}% | Sample {decoupling.get('sample_size', 0)} snapshots | {decoupling.get('signal', 'No Signal')}",
            ))
    return [(label, value) for label, value in pairs if _clean_text(value)]

def _daily_snapshot_pairs_ar(update: dict) -> list[tuple[str, str]]:
    snapshot = update.get("cross_asset_snapshot") or {}
    market_levels = update.get("market_levels") or {}
    regional_internals = update.get("regional_internals") or {}
    decoupling = update.get("gcc_decoupling") or {}

    us = snapshot.get("us_equities") or {}
    rates = snapshot.get("rates") or {}
    oil = snapshot.get("commodities") or {}
    vix = snapshot.get("volatility") or {}
    btc = snapshot.get("crypto") or {}
    gcc = snapshot.get("gcc") or {}
    egypt = snapshot.get("egypt") or {}
    qqq = market_levels.get("nasdaq") or {}
    gold = market_levels.get("gold") or {}
    dxy = market_levels.get("dxy") or {}
    fg_score = update.get("fear_greed_index")

    pairs: list[tuple[str, str]] = [
        ("SPY", f"{_ar_num(us.get('price'))} | يومي {_ar_num(us.get('d1_pct'), pct=True)} | 5 أيام {_ar_num(us.get('d5_pct'), pct=True)}"),
        ("Nasdaq Proxy (QQQ)", f"{_ar_num(qqq.get('price'))} | يومي {_ar_num(qqq.get('d1_pct'), pct=True)}"),
        ("VIX", _ar_num(vix.get("price"))),
        ("10Y", f"{_ar_num(rates.get('price'), rates=True)} | يومي {_ar_num(rates.get('d1_pct'), pct=True)}"),
        ("WTI Proxy (USO)", f"{_ar_num(oil.get('price'))} | يومي {_ar_num(oil.get('d1_pct'), pct=True)} | 5 أيام {_ar_num(oil.get('d5_pct'), pct=True)}"),
        ("Gold Proxy (GLD)", f"{_ar_num(gold.get('price'))} | يومي {_ar_num(gold.get('d1_pct'), pct=True)}"),
        ("DXY Proxy (UUP)", f"{_ar_num(dxy.get('price'))} | يومي {_ar_num(dxy.get('d1_pct'), pct=True)} | 5 أيام {_ar_num(dxy.get('d5_pct'), pct=True)}"),
        ("بيتكوين", f"{_ar_num(btc.get('price'), crypto=True)} | 5 أيام {_ar_num(btc.get('d5_pct'), pct=True)}"),
        ("الخوف والطمع", f"{float(fg_score):.1f}/100" if isinstance(fg_score, (int, float)) else str(fg_score or "غير متاح")),
        ("المركب السعودي", f"يومي {_ar_num(gcc.get('d1_pct'), pct=True)} | 5 أيام {_ar_num(gcc.get('d5_pct'), pct=True)}"),
        ("المركب المصري", f"يومي {_ar_num(egypt.get('d1_pct'), pct=True)} | 5 أيام {_ar_num(egypt.get('d5_pct'), pct=True)}"),
    ]
    for key, label in (("ksa", "السعودية"), ("uae", "الإمارات"), ("egypt", "مصر")):
        pairs.extend(_daily_snapshot_internal_lines_ar(label, regional_internals.get(key) or {}))
    if decoupling:
        signal_ar = _translate_phrase_ar(decoupling.get("signal", "إشارة غير متاحة")) or decoupling.get("signal", "")
        if decoupling.get("method") == "correlation":
            pairs.append((
                "انفصال بيتا النفط في الخليج",
                f"{decoupling.get('score', 0)}/100 | الارتباط {decoupling.get('correlation', 0):+.2f} | العينة {decoupling.get('sample_size', 0)} لقطات | {signal_ar}",
            ))
        else:
            pairs.append((
                "انفصال بيتا النفط في الخليج",
                f"{decoupling.get('score', 0)}/100 | عقارات دبي {decoupling.get('latest_dfm_re_change', 0):+.2f}% | proxy النفط المرتبط بـ WTI {decoupling.get('latest_wti_change', 0):+.2f}% | العينة {decoupling.get('sample_size', 0)} لقطات | {signal_ar}",
            ))
    return [(label, value) for label, value in pairs if _clean_text(value)]

def _build_cio_daily_report_fallback_v3(update: dict) -> str:
    view = update.get("eisax_view", {}) if isinstance(update.get("eisax_view"), dict) else {}
    snapshot = update.get("cross_asset_snapshot") or {}
    regime = update.get("market_regime", "Cautious")
    confidence = update.get("regime_confidence", "Low")
    fg_score = update.get("fear_greed_index", 50)
    decision_type = update.get("decision_type") or _daily_decision_type(view)
    confidence_score = update.get("confidence_score") or _daily_confidence_score(regime, confidence, fg_score)
    mode = update.get("positioning_mode") or _daily_positioning_mode(view)
    invalidates = _clean_text_list(update.get("what_invalidates"), 4)
    triggers = _clean_text_list(update.get("next_triggers"), 4)
    catalysts = _ordered_daily_catalysts(triggers)
    market_state = _daily_market_state(snapshot, regime)
    market_levels = update.get("market_levels") or {}
    regional_internals = update.get("regional_internals") or {}
    decoupling = update.get("gcc_decoupling") or {}

    def _entry(key: str) -> dict:
        return snapshot.get(key) or {}

    def _near(value: Any, decimals: int = 0, pct: bool = False) -> str:
        if not isinstance(value, (int, float)):
            return "current levels"
        if pct:
            return f"{value:.1f}%"
        if decimals > 0:
            return f"{value:.{decimals}f}"
        return f"{round(value):,}"

    def _breadth_state(internal: dict) -> str:
        adv = int(internal.get("advancers") or 0)
        dec = int(internal.get("decliners") or 0)
        if adv and dec:
            if adv > dec * 1.1:
                return "broadening"
            if dec > adv * 1.1:
                return "weak"
        return "balanced"

    us = _entry("us_equities")
    rates = _entry("rates")
    oil = _entry("commodities")
    vix = _entry("volatility")
    btc = _entry("crypto")
    qqq = market_levels.get("nasdaq") or {}
    gold = market_levels.get("gold") or {}

    us_price = _as_number(us.get("price"))
    us_d5 = _as_number(us.get("d5_pct"))
    rate_d1 = _as_number(rates.get("d1_pct"))
    btc_d5 = _as_number(btc.get("d5_pct"))
    ksa_state = _breadth_state(regional_internals.get("ksa") or {})
    uae_state = _breadth_state(regional_internals.get("uae") or {})
    egypt_weighted = _as_number((regional_internals.get("egypt") or {}).get("weighted_change"))
    fg_bucket = f"the low {int(float(fg_score) // 10) * 10}s" if isinstance(fg_score, (int, float)) else "depressed territory"

    if regime == "Bearish":
        lead = "Liquidity is tightening faster than risk assets can absorb, so the regime stays decisively defensive."
        stance_line = "The book should prioritize protection, cash optionality, and patience over return maximization."
    elif regime == "Bullish":
        lead = "Risk can still be owned, but only where leadership is strong enough to outrun any renewed tightening in rates."
        stance_line = "The book can stay constructive, but only through leadership rather than broad beta."
    else:
        lead = "The market is investable, but only through disciplined sizing and active constraint."
        stance_line = "This is a hold-with-intent setup, not a passive hold and not a launch pad for fresh beta."

    if us_d5 is not None and rate_d1 is not None and us_d5 > 0 and rate_d1 > 0:
        contradiction = "Equities are pressing higher while rates stay firm, which means price is leading liquidity and upside should be treated as structurally narrow."
    elif us_d5 is not None and rate_d1 is not None and us_d5 < 0 and rate_d1 < 0:
        contradiction = "Rates are offering some relief, but equities are still not converting it into conviction, which keeps confidence in growth incomplete."
    else:
        contradiction = "Rates, volatility, and oil still matter more than the latest equity headline."

    if ksa_state == "weak" and uae_state == "weak":
        gcc_breadth_line = "Saudi and UAE breadth remain weak despite firm oil, which confirms internal participation is still narrow."
    elif ksa_state == "weak" or uae_state == "weak":
        gcc_breadth_line = "Oil is supporting GCC cash flow, but regional breadth is not broad enough yet to justify an indiscriminate chase."
    else:
        gcc_breadth_line = "Oil still underwrites GCC liquidity, but breadth is only improving selectively rather than confirming a full regional risk-on."

    if isinstance(egypt_weighted, (int, float)) and egypt_weighted > 0:
        egypt_line = "Egypt is relatively firmer, but the move still reads as tactical rather than structural while dollar liquidity remains constrained."
    else:
        egypt_line = "Egypt remains liquidity-constrained; higher US rates still favor exporters, hard-currency earners, and defensive balance sheets."

    dec_signal = decoupling.get("signal", "No Signal")
    if dec_signal == "Active Decoupling Signal":
        decoupling_line = "Dubai real estate is not confirming the oil move, so the GCC decoupling signal remains active."
    elif dec_signal == "Partial Decoupling":
        decoupling_line = "Dubai real estate is only partially confirming oil strength, so GCC cyclicals still need stricter selection."
    else:
        decoupling_line = "Oil sensitivity is still flowing through the region, which means GCC cyclicals should be sized through sector discipline rather than index beta."

    tactical_avoid = ", ".join(view.get("underweight_assets") or []) or "speculative beta"
    tactical_focus = view.get("focus") or "selective quality"
    maintain_line = ", ".join(view.get("overweight_assets") or view.get("neutral_assets") or ["Quality Equities"])
    state_line = "This is a conflicted regime." if market_state == "conflicted" else f"This is a {market_state} market."
    spy_low, spy_high = _spy_range_levels(invalidates)
    if isinstance(us_price, (int, float)) and us_price > spy_high:
        equities_line = "• Equities -> SPY is testing a breakout above the recent range, with QQQ still carrying leadership; breakout needs confirmation as liquidity remains constrained."
        spy_risk_line = f"Holding above ${spy_high:,.0f} confirms upside continuation; failure back inside range invalidates breakout."
    elif isinstance(us_price, (int, float)) and us_price >= spy_high - 1:
        equities_line = f"• Equities -> SPY is testing the upper boundary of its recent range near {_near(spy_high)}, with QQQ still carrying leadership; upside requires confirmation as liquidity remains constrained."
        spy_risk_line = f"A sustained break above ${spy_high:,.0f} confirms upside continuation; rejection keeps the range intact."
    elif isinstance(us_price, (int, float)) and us_price < spy_low:
        equities_line = "• Equities -> SPY is testing a breakdown below the recent range, with QQQ leadership no longer enough to stabilize the tape; breakdown needs confirmation while liquidity remains constrained."
        spy_risk_line = f"Holding below ${spy_low:,.0f} confirms downside continuation; recovery back inside range invalidates breakdown."
    else:
        equities_line = "• Equities -> SPY remains inside its recent range, with QQQ still carrying leadership; upside still requires confirmation as liquidity remains constrained."
        spy_risk_line = f"A sustained break above ${spy_high:,.0f} confirms upside continuation; rejection keeps the range intact."
    risk_shift = (
        "A break in any two of the three core triggers forces a regime review and a fresh position-sizing decision."
        if len(invalidates) >= 3
        else "A clear break in price, volatility, or rates is enough to force a new allocation stance."
    )
    snapshot_pairs = _daily_snapshot_pairs_en(update)

    lines = [
        "EisaX Daily Market Pulse",
        f"Date: {update.get('date', '')}",
        f"Regime: {regime}",
        f"Confidence: {confidence}",
        f"Decision Type: {decision_type}",
        f"Confidence Score: {confidence_score}",
        "",
        "## Market Snapshot",
    ]
    for label, value in snapshot_pairs:
        lines.append(f"{label}: {value}")
    lines += [
        "",
        "## Executive Summary",
        f"• {lead}",
        f"• {contradiction}",
        f"• Rates are still near {_near(rates.get('price'), decimals=1, pct=True)}, VIX is near {_near(vix.get('price'))}, oil proxy is near {_near(oil.get('price'))}, and Fear & Greed remains stuck in {fg_bucket}; the tape is tradable but constrained.",
        f"• {stance_line}",
        f"• Stance stays {view.get('stance', 'HOLD')} in {mode.lower()} mode until the risk framework breaks.",
        "",
        "## Cross-Asset Reality",
        equities_line,
        f"• Rates -> Treasury yields remain near {_near(rates.get('price'), decimals=1, pct=True)}; that level still caps the risk budget and keeps funding conditions tight.",
        f"• Oil -> WTI proxy remains elevated near {_near(oil.get('price'))}; supports GCC cash flow but keeps inflation pressure active.",
        "• Gold -> Holding bid confirms defense is not being fully unwound.",
        f"• VIX -> Volatility near {_near(vix.get('price'))} says hedging demand is contained, not cheap enough to justify indiscriminate gross exposure.",
        f"• Crypto -> Bitcoin is {'confirming' if isinstance(btc_d5, (int, float)) and isinstance(us_d5, (int, float)) and btc_d5 * us_d5 >= 0 else 'not confirming'} the broader tape, but it remains a secondary liquidity tell rather than a primary allocation anchor.",
        "This is not a directional market.",
        state_line,
        "",
        "## Regional Read (GCC + Egypt)",
        f"• {gcc_breadth_line}",
        "• GCC still favors banks and energy, but leadership is too narrow to convert firmer oil into broad regional beta.",
        f"• {decoupling_line}",
        f"• {egypt_line}",
        "",
        "## Positioning",
        f"Stance: {view.get('stance', 'HOLD')}",
        f"Mode: {mode}",
        "Execution:",
        "• Maintain core quality exposure; cap gross risk until price and liquidity confirm together.",
        "• Add only on controlled pullbacks that hold structure and keep new risk strictly inside leadership.",
        "• Fund new risk by trimming speculative beta into quality, defense, and liquid optionality.",
        "",
        "## Risk Framework",
        f"• SPY level: {spy_risk_line}",
        f"• VIX level: {invalidates[1] if len(invalidates) > 1 else 'VIX needs to stay contained; a renewed volatility spike would tighten the risk budget immediately.'}",
        f"• 10Y level: {invalidates[2] if len(invalidates) > 2 else 'A higher 10Y yield would reprice liquidity and cap equity upside.'}",
        f"• Regime shift: {risk_shift}",
        "",
        "## Tactical Playbook",
        f"• Maintain: Core exposure ({maintain_line} + quality US leaders + GCC banks and energy).",
        "• Deploy: add only on controlled pullbacks into leadership.",
        f"• Avoid: fresh {tactical_avoid} while rates and volatility remain the gating variables.",
        f"• Focus: {tactical_focus}, oil-confirmed GCC leadership, and Egypt only through hard-currency balance-sheet quality.",
        "",
        "## Catalysts",
    ]
    for item in (catalysts or ["Fed communications", "CPI / PCE data release", "10Y yield break"])[:4]:
        lines.append(f"• {item}")
    lines += [
        "",
        "## Final Line",
        "Liquidity is the constraint. Until it expands, aggression must stay selective.",
    ]
    return "\n".join(lines).strip()

def _build_cio_daily_report_ar_v3(update: dict) -> str:
    view = update.get("eisax_view", {}) if isinstance(update.get("eisax_view"), dict) else {}
    snapshot = update.get("cross_asset_snapshot") or {}
    regime = update.get("market_regime", "Cautious")
    confidence = update.get("regime_confidence", "Low")
    decision_type = update.get("decision_type") or _daily_decision_type(view)
    confidence_score = update.get("confidence_score") or _daily_confidence_score(regime, confidence, update.get("fear_greed_index"))
    mode = update.get("positioning_mode") or _daily_positioning_mode(view)
    market_state = _daily_market_state(snapshot, regime)
    invalidates = _clean_text_list(update.get("what_invalidates"), 4)
    triggers = _clean_text_list(update.get("next_triggers"), 4)
    catalysts = _ordered_daily_catalysts(triggers)
    regional_internals = update.get("regional_internals") or {}
    decoupling = update.get("gcc_decoupling") or {}

    def _near(value: Any, decimals: int = 0, pct: bool = False) -> str:
        if not isinstance(value, (int, float)):
            return "المستويات الحالية"
        if pct:
            return f"{value:.1f}%"
        if decimals > 0:
            return f"{value:.{decimals}f}"
        return f"{round(value):,}"

    def _breadth_state(internal: dict) -> str:
        adv = int(internal.get("advancers") or 0)
        dec = int(internal.get("decliners") or 0)
        if adv and dec:
            if adv > dec * 1.1:
                return "broadening"
            if dec > adv * 1.1:
                return "weak"
        return "balanced"

    def _risk_ar(text: str) -> str:
        out = _translate_risk_trigger_ar(text)
        out = out.replace("خروج SPY خارج ", "خروج SPY خارج النطاق ")
        out = out.replace("تحرك عائد 10 سنوات خارج نطاق ", "تحرك عائد 10 سنوات خارج النطاق ")
        out = out.replace(" النطاق الأخير —", " —")
        return out

    regime_ar = _ar_label(regime, {"Bullish": "صعودي", "Bearish": "هبوطي", "Cautious": "حذر", "Conflicted": "متضارب"}, "حذر")
    conf_ar = _ar_label(confidence, {"High": "عالية", "Medium": "متوسطة", "Low": "منخفضة"}, "منخفضة")
    decision_ar = _ar_label(decision_type, {
        "HOLD_ACTIVE_CONSTRAINT": "احتفاظ بقيود نشطة",
        "BUY_SELECTIVE": "شراء انتقائي",
        "REDUCE": "خفض المخاطر",
    }, decision_type)
    mode_ar = _ar_label(mode, {"Active Constraint": "قيود نشطة", "Selective": "انتقائي", "Defensive": "دفاعي"}, mode)
    stance_ar = _ar_label(_clean_text(view.get("stance")), {
        "HOLD": "احتفاظ",
        "Tactical BUY": "شراء تكتيكي",
        "REDUCE RISK": "خفض المخاطر",
    }, _clean_text(view.get("stance")))
    state_ar = _ar_label(market_state, {"conflicted": "تموضع متضارب", "transition": "انتقالية", "positioning": "تموضع"}, market_state)
    state_line_ar = "هذا نظام متضارب." if market_state == "conflicted" else f"هذا سوق {state_ar}."

    us = snapshot.get("us_equities") or {}
    rates = snapshot.get("rates") or {}
    oil = snapshot.get("commodities") or {}
    vix = snapshot.get("volatility") or {}
    btc = snapshot.get("crypto") or {}
    fg_score = update.get("fear_greed_index", 50)
    us_d5 = _as_number(us.get("d5_pct"))
    us_price = _as_number(us.get("price"))
    btc_d5 = _as_number(btc.get("d5_pct"))
    ksa_state = _breadth_state(regional_internals.get("ksa") or {})
    uae_state = _breadth_state(regional_internals.get("uae") or {})
    egypt_weighted = _as_number((regional_internals.get("egypt") or {}).get("weighted_change"))
    fg_bucket = "الثلاثينيات المنخفضة" if isinstance(fg_score, (int, float)) else "منطقة خوف واضحة"

    if regime == "Bearish":
        lead = "السيولة تنكمش أسرع من قدرة الأصول الخطرة على الاستيعاب، لذلك يبقى النظام دفاعيًا بوضوح."
        stance_line = "الأولوية هنا لحفظ رأس المال والسيولة الاختيارية لا لتعظيم العائد."
    elif regime == "Bullish":
        lead = "يمكن امتلاك المخاطرة، لكن فقط حيث القيادة أقوى من أي إعادة تشديد في العوائد."
        stance_line = "يمكن إبقاء الدفتر بنبرة بناءة، لكن عبر القيادة لا عبر بيتا واسعة."
    else:
        lead = "السوق ما زال قابلًا للاستثمار، لكن فقط عبر انضباط في الحجم وقيود نشطة على المخاطر."
        stance_line = "هذا احتفاظ مقصود، لا احتفاظ سلبي ولا منصة لإضافة بيتا جديدة."

    if ksa_state == "weak" and uae_state == "weak":
        gcc_breadth_line = "النفط ما زال يدعم تدفقات الخليج، لكن اتساع السعودية والإمارات يظل ضعيفًا، ما يؤكد أن المشاركة الداخلية ما زالت ضيقة."
    elif ksa_state == "weak" or uae_state == "weak":
        gcc_breadth_line = "النفط يدعم التدفقات الخليجية، لكن اتساع السوق الإقليمي لا يزال غير كافٍ لتبرير مطاردة واسعة للمخاطرة."
    else:
        gcc_breadth_line = "النفط ما زال يسند سيولة الخليج، لكن التحسن في الاتساع انتقائي أكثر من كونه تأكيدًا لموجة مخاطرة كاملة."

    if isinstance(egypt_weighted, (int, float)) and egypt_weighted > 0:
        egypt_line = "مصر أكثر ثباتًا نسبيًا، لكن الحركة تبقى تكتيكية لا هيكلية ما دامت سيولة الدولار مقيدة."
    else:
        egypt_line = "مصر ما زالت سوقًا مقيدًا بالسيولة؛ ارتفاع العوائد الأميركية يفرض تفضيل المصدرين والدخل الدولاري والميزانيات الدفاعية."

    dec_signal = decoupling.get("signal", "No Signal")
    if dec_signal == "Active Decoupling Signal":
        decoupling_line = "عقارات دبي لا تؤكد حركة النفط، لذلك تبقى إشارة انفصال بيتا النفط نشطة."
    elif dec_signal == "Partial Decoupling":
        decoupling_line = "عقارات دبي تؤكد النفط جزئيًا فقط، لذلك تبقى الحاجة عالية للانتقاء داخل دورات الخليج."
    else:
        decoupling_line = "حساسية النفط ما زالت تمر عبر المنطقة، ما يعني أن التعرض الدوري يجب أن يُبنى بالقطاع لا ببيتا المؤشر."

    maintain_line = (_translate_phrase_ar(", ".join(view.get("overweight_assets") or view.get("neutral_assets") or ["Quality Equities"])) or "").replace(",", "،")
    avoid_line = (_translate_phrase_ar(", ".join(view.get("underweight_assets") or ["High Beta", "Speculative Crypto"])) or "").replace(",", "،")
    focus_line = _translate_phrase_ar(_clean_text(view.get("focus")) or "Selective Quality")
    spy_low, spy_high = _spy_range_levels(invalidates)
    if isinstance(us_price, (int, float)) and us_price > spy_high:
        equities_line = "• الأسهم -> SPY يختبر اختراقًا أعلى النطاق الأخير، وقيادة QQQ ما زالت موجودة؛ لكن الاختراق يحتاج تأكيدًا لأن السيولة ما زالت مقيدة."
        spy_risk_line = f"الثبات فوق ${spy_high:,.0f} يؤكد استمرار الصعود؛ والعودة داخل النطاق تُبطل الاختراق."
    elif isinstance(us_price, (int, float)) and us_price >= spy_high - 1:
        equities_line = f"• الأسهم -> SPY يختبر الحد العلوي من نطاقه الأخير قرب {_near(spy_high)}، وقيادة QQQ ما زالت موجودة؛ لكن الصعود يحتاج تأكيدًا لأن السيولة ما زالت مقيدة."
        spy_risk_line = f"اختراق مستدام فوق ${spy_high:,.0f} يؤكد استمرار الصعود؛ والرفض يبقي النطاق قائمًا."
    elif isinstance(us_price, (int, float)) and us_price < spy_low:
        equities_line = "• الأسهم -> SPY يختبر كسرًا أدنى النطاق الأخير، وقيادة QQQ لم تعد كافية لتثبيت الشريط؛ والكسر يحتاج تأكيدًا ما دامت السيولة مقيدة."
        spy_risk_line = f"الثبات دون ${spy_low:,.0f} يؤكد استمرار الهبوط؛ والعودة داخل النطاق تُبطل الكسر."
    else:
        equities_line = "• الأسهم -> SPY ما زال داخل نطاقه الأخير، وقيادة QQQ ما زالت موجودة؛ لكن الصعود ما زال يحتاج تأكيدًا لأن السيولة مقيدة."
        spy_risk_line = f"اختراق مستدام فوق ${spy_high:,.0f} يؤكد استمرار الصعود؛ والرفض يبقي النطاق قائمًا."
    snapshot_pairs = _daily_snapshot_pairs_ar(update)

    lines = [
        "EisaX Daily Market Pulse",
        f"Date: {update.get('date', '')}",
        f"Regime: {regime_ar}",
        f"Confidence: {conf_ar}",
        f"Decision Type: {decision_ar}",
        f"Confidence Score: {confidence_score}",
        "",
        "## لقطة السوق",
    ]
    for label, value in snapshot_pairs:
        lines.append(f"{label}: {value}")
    lines += [
        "",
        "## الملخص التنفيذي",
        f"• {lead}",
        f"• الأسهم تقترب من الحد العلوي لنطاقها الأخير بينما العوائد ما زالت قرب {_near(rates.get('price'), decimals=1, pct=True)}، ما يعني أن السيولة أضعف من حركة السعر.",
        f"• النفط proxy ما زال مرتفعًا قرب {_near(oil.get('price'))}، وVIX قرب {_near(vix.get('price'))}، ومؤشر الخوف والطمع ما يزال في {fg_bucket}؛ لذلك الشريط قابل للتداول لكنه غير مناسب لبيتا واسعة.",
        f"• {stance_line}",
        f"• الموقف يظل {stance_ar} ضمن وضع {mode_ar} حتى يكسر إطار المخاطر الحالي.",
        "",
        "## واقع الأصول المتقاطعة",
        equities_line,
        f"• العوائد -> عوائد 10 سنوات ما زالت قرب {_near(rates.get('price'), decimals=1, pct=True)}؛ وهذا المستوى ما زال يفرض سقفًا على ميزانية المخاطر ويُبقي التمويل مشدودًا.",
        f"• النفط -> WTI proxy ما زال مرتفعًا قرب {_near(oil.get('price'))}؛ يدعم تدفقات الخليج لكنه يبقي ضغط التضخم نشطًا.",
        "• الذهب -> استمرار الطلب عليه يؤكد أن التحوط لم يُفك بالكامل.",
        f"• VIX -> التذبذب قرب {_near(vix.get('price'))} يعني أن الطلب على الحماية منضبط، لكنه ليس رخيصًا بما يكفي لتبرير انكشاف عشوائي.",
        f"• الكريبتو -> بيتكوين {'يؤكد' if isinstance(btc_d5, (int, float)) and isinstance(us_d5, (int, float)) and btc_d5 * us_d5 >= 0 else 'لا يؤكد'} نبرة الشريط الأوسع، لكنه يبقى إشارة سيولة ثانوية لا مرساة تخصيص أساسية.",
        "هذا ليس سوقًا اتجاهيًا.",
        state_line_ar,
        "",
        "## القراءة الإقليمية (الخليج + مصر)",
        f"• {gcc_breadth_line}",
        "• الخليج ما زال يفضل البنوك والطاقة، لكن القيادة أضيق من أن تتحول إلى شراء إقليمي شامل.",
        f"• {decoupling_line}",
        f"• {egypt_line}",
        "",
        "## التموضع",
        f"الموقف: {stance_ar}",
        f"الوضع: {mode_ar}",
        "التنفيذ:",
        "• حافظ على الانكشاف الأساسي عالي الجودة مع إبقاء إجمالي المخاطر مقيدًا حتى يؤكد السعر والسيولة معًا.",
        "• أضف فقط على تراجعات منضبطة تحافظ على البنية، واجعل المخاطرة الجديدة محصورة بصرامة داخل القيادة فقط.",
        "• موّل أي إضافة جديدة عبر خفض البيتا المضاربية لصالح الجودة والدفاعيات والسيولة الاختيارية.",
        "",
        "## إطار المخاطر",
        f"• SPY: {spy_risk_line}",
        f"• VIX: {_risk_ar(invalidates[1]) if len(invalidates) > 1 else 'أي ارتفاع واضح في VIX يعني تضييقًا مباشرًا في ميزانية المخاطر.'}",
        f"• 10Y: {_risk_ar(invalidates[2]) if len(invalidates) > 2 else 'أي ارتفاع إضافي في عائد 10 سنوات سيضغط على التقييمات والسيولة.'}",
        "• تغير النظام: القرار يتغير فقط عندما ينكسر اثنان من المحركات الثلاثة معًا، لا عندما يتحرك عامل واحد منفردًا.",
        "",
        "## الدليل التكتيكي",
        f"• حافظ على: الانكشاف الأساسي ({maintain_line} + جودة أميركية قيادية + بنوك وطاقة خليجية).",
        "• انشر رأس المال عند: تراجعات منضبطة داخل القيادة فقط.",
        f"• تجنب: {avoid_line} ما دامت العوائد والتذبذب هما القيد الفعلي.",
        f"• ركز على: {focus_line}، وقيادة الخليج المؤكدة بالنفط، ومصر فقط عبر جودة الميزانية والعملة الصعبة.",
        "",
        "## المحفزات",
    ]
    for item in (catalysts or ["Fed communications", "CPI / PCE data release", "10Y yield break"])[:4]:
        lines.append(f"• {_translate_catalyst_ar(item)}")
    lines += [
        "",
        "## الخلاصة",
        "السيولة هي القيد. وحتى تتسع، يجب أن يبقى الهجوم انتقائيًا.",
    ]
    return "\n".join(lines).strip()

def _finalize_daily_update(update: dict, moves_summary: dict, fg: dict) -> dict:
    update["data_timestamp"] = _get_market_data_timestamp()
    update["web_version"] = _build_web_version(update)
    view = update.get("eisax_view", {}) if isinstance(update.get("eisax_view"), dict) else {}
    update["decision_type"] = _daily_decision_type(view)
    update["positioning_mode"] = _daily_positioning_mode(view)
    update["confidence_score"] = _daily_confidence_score(
        update.get("market_regime", "Cautious"),
        update.get("regime_confidence", "Low"),
        update.get("fear_greed_index"),
    )
    update["market_levels"] = {
        "nasdaq": moves_summary.get("QQQ") or {},
        "gold": moves_summary.get("GLD") or {},
        "dxy": moves_summary.get("UUP") or {},
        "silver": moves_summary.get("SLV") or {},
    }
    update["regional_internals"] = _build_daily_regional_internals()
    update["gcc_decoupling"] = _compute_gcc_decoupling_signal()
    full_report = _build_cio_daily_report_fallback_v3(update)
    update["full_report"] = (full_report or "").strip()
    update["ar_full_report"] = _build_cio_daily_report_ar_v3(update)
    update["linkedin_text"] = _build_linkedin_text_v2(update)
    return update

def _finalize_weekly_update(update: dict) -> dict:
    update["data_timestamp"] = _get_market_data_timestamp()
    update["web_version"] = _build_web_version(update)
    update["weekly_decision_type"] = _weekly_decision_type(update)
    update["weekly_positioning_mode"] = _weekly_positioning_mode(update)
    update["weekly_confidence_score"] = _weekly_confidence_score(
        update.get("market_regime", "Cautious"),
        update.get("regime_confidence", "Low"),
        update,
    )
    update["full_report"] = _build_cio_weekly_report_fallback(update)
    update["ar_full_report"] = _build_cio_weekly_report_ar(update)
    update["linkedin_text"] = _build_linkedin_text_v2(update)
    return update

