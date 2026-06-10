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


def _daily_decision_type(view: dict) -> str:
    stance = _clean_text((view or {}).get("stance")).upper()
    if "REDUCE" in stance:
        return "REDUCE"
    if "BUY" in stance:
        return "BUY_SELECTIVE"
    return "HOLD_ACTIVE_CONSTRAINT"

def _daily_positioning_mode(view: dict) -> str:
    stance = _clean_text((view or {}).get("stance")).upper()
    if "REDUCE" in stance:
        return "Defensive"
    if "BUY" in stance:
        return "Selective"
    return "Active Constraint"

def _daily_confidence_score(regime: str, confidence: str, fg_score: Any) -> int:
    base = {"Low": 58, "Medium": 72, "High": 84}.get(confidence, 60)
    if regime == "Bullish":
        base += 3
    elif regime == "Bearish":
        base += 2
    elif regime == "Cautious":
        base -= 6
    elif regime == "Conflicted":
        base -= 8

    fg_val = _as_number(fg_score)
    if fg_val is not None and (fg_val <= 20 or fg_val >= 80):
        base += 2

    return max(35, min(95, int(round(base))))

def _daily_market_state(snapshot: dict, regime: str) -> str:
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    eq_d1 = _as_number((snapshot.get("us_equities") or {}).get("d1_pct"))
    rate_d1 = _as_number((snapshot.get("rates") or {}).get("d1_pct"))
    oil_d1 = _as_number((snapshot.get("commodities") or {}).get("d1_pct"))

    if regime == "Cautious":
        return "conflicted"
    if regime == "Bearish":
        return "transition"
    if eq_d1 is not None and eq_d1 > 0 and ((rate_d1 is not None and rate_d1 > 0) or (oil_d1 is not None and oil_d1 > 0)):
        return "positioning"
    return "positioning"

def _generate_full_report_text(daily_data: dict, moves_summary: dict, fg: dict) -> str:
    """
    Generate the complete institutional Daily Market Intelligence Brief
    (Internal Full Version) as a formatted markdown string.
    Returns empty string on failure — caller falls back gracefully.
    """
    regime   = daily_data.get("market_regime", "Cautious")
    conf     = daily_data.get("regime_confidence", "Low")
    date     = daily_data.get("date", "")
    fg_score = fg.get("score", 50)
    fg_label = fg.get("rating", "Neutral")
    view     = daily_data.get("eisax_view", {})
    stance   = view.get("stance", "HOLD") if isinstance(view, dict) else str(view)

    prompt = f"""You are an institutional macro strategist writing the EisaX Daily Market Intelligence Brief.
Use ONLY the data provided below. Do NOT hallucinate numbers. Do NOT describe events — INTERPRET them.

MARKET DATA ({date}):
{json.dumps(moves_summary, indent=2)}

COMPUTED ANALYSIS:
Regime: {regime} | Confidence: {conf}
Fear & Greed: {fg_score}/100 ({fg_label})
EisaX Stance: {json.dumps(view)}
Invalidation triggers: {json.dumps(daily_data.get("what_invalidates", []))}
Cross-asset snapshot: {json.dumps(daily_data.get("cross_asset_snapshot", {}))}

Write the brief using EXACTLY this structure (use ## for section headers, • for bullets):

## EisaX Daily Market Pulse — {regime}
Date: {date} | Confidence: {conf}

---

## What Matters Now
• [Cross-asset interpretation — connects 2+ asset classes — NOT a data point, an INSIGHT]
• [Why the regime matters for positioning RIGHT NOW]
• [What retail misses that institutions are acting on]

## Cross-Asset Interpretation
[2 tight paragraphs: explain flows and divergences between equities, rates, volatility, commodities, crypto. Interpret, don't list. No generic phrases.]

## EisaX View
Stance: {stance}
Focus: [specific assets/sectors]
Horizon: [short-term / swing / defensive]

Overweight: [assets]
Neutral: [assets]
Underweight: [assets]
Cash: [recommended % with context]

## Why Now
[2-3 sentences MAX. Explain the SPECIFIC setup — momentum, liquidity, sentiment divergence, or macro catalyst. Zero filler.]

## Market-by-Market View
### US Equities
• [1 line with numbers and positioning meaning]
• [1 line with interpretation]
### Volatility
• [1 line with VIX level and move]
• [1 line with what that means for hedging/risk budget]
### Rates
• [1 line with 10Y level and move]
• [1 line with what that means for valuations]
### Crypto
• [1 line with Bitcoin move and level]
• [1 line on whether crypto confirms or diverges from the risk tape]
### Metals
• [1 line with Gold move and level]
• [1 line on whether defense is being bid]
### Oil
• [1 line with Oil move and level]
• [1 line on what it says about growth/inflation]
### GCC
• [1 line with local market state or "Market Closed"]
• [1 line on oil/liquidity read-through]
### Egypt
• [1 line with local market state or "Market Closed"]
• [1 line on dollar/liquidity implications]

## What Invalidates This View
• Primary trigger: [Price trigger: specific instrument + specific level]
• Secondary: [Volatility trigger: VIX level + what it implies]
• Tertiary: [Macro trigger: yield level or data release]
• Additional condition if warranted

## Tactical Playbook
[Concrete, actionable instructions. Name specific instruments. No vague advice. 3-5 bullet points.]

## What to Watch — Next 24–72h
• [Specific event, level, or release]
• [Specific event, level, or release]
• [Specific event, level, or release]

---

HARD RULES:
- NEVER write: "markets showed resilience" / "investor confidence increased" / "amid uncertainty"
- Every bullet must contain decision-relevant information
- Tone: CIO-level, confident, measured — NOT dramatic, NOT retail
- Length: 500–700 words total
- No repetition across sections
- Detailed market-by-market coverage will follow, so do not repeat GCC/Egypt lines in summary sections
"""
    raw = _call_openai_text(prompt, max_tokens=1600) or _call_gemini(prompt)
    return (raw or "").strip()

def _generate_cio_daily_report_text(daily_data: dict, moves_summary: dict, fg: dict) -> str:
    regime = daily_data.get("market_regime", "Cautious")
    conf = daily_data.get("regime_confidence", "Low")
    date = daily_data.get("date", "")
    fg_score = fg.get("score", 50)
    fg_label = fg.get("rating", "Neutral")
    view = daily_data.get("eisax_view", {}) if isinstance(daily_data.get("eisax_view"), dict) else {}
    decision_type = daily_data.get("decision_type") or _daily_decision_type(view)
    confidence_score = daily_data.get("confidence_score") or _daily_confidence_score(regime, conf, fg_score)
    mode = daily_data.get("positioning_mode") or _daily_positioning_mode(view)
    snapshot = daily_data.get("cross_asset_snapshot", {})
    market_state = _daily_market_state(snapshot, regime)

    def _mv_text(symbol: str) -> str:
        entry = moves_summary.get(symbol) or {}
        price = entry.get("price")
        if not isinstance(price, (int, float)):
            return "N/A"
        if symbol == "BTC-USD":
            return f"{price:,.0f}"
        if symbol == "^TNX":
            return f"{price:.2f}%"
        if price >= 1000:
            return f"{price:,.0f}"
        if price >= 100:
            return f"{price:.2f}"
        return f"{price:.2f}"

    prompt = f"""You are EisaX, an institutional-grade Chief Investment Officer (CIO) AI.

Your role is to generate a DAILY MARKET PULSE for professional investors managing multi-asset portfolios.

You do NOT behave like a retail analyst.
You do NOT summarize news.
You do NOT produce generic insights.
You think and communicate like a CIO responsible for capital allocation across equities, rates, commodities, crypto, and regional markets (GCC + Egypt + global).

INPUT DATA:
S&P 500: {_mv_text("SPY")}
VIX: {_mv_text("^VIX")}
10Y Treasury Yield: {_mv_text("^TNX")}
Oil (WTI): {_mv_text("USO")}
Bitcoin: {_mv_text("BTC-USD")}
Fear & Greed Index: {fg_score} ({fg_label})
Nasdaq: {_mv_text("QQQ")}
Gold: {_mv_text("GLD")}
DXY: {_mv_text("UUP")}

Computed context:
Date: {date}
Regime: {regime}
Confidence: {conf}
Decision Type: {decision_type}
Confidence Score: {confidence_score}
Positioning mode: {mode}
Pre-computed stance: {json.dumps(view)}
Invalidation triggers: {json.dumps(daily_data.get("what_invalidates", []))}
Cross-asset snapshot: {json.dumps(snapshot)}
Market state label for the closing lines: {market_state}

CORE OBJECTIVE:
Transform cross-asset data into:
- a clear market regime
- a resolved interpretation
- a precise positioning decision
- a structured tactical plan

MANDATORY STRUCTURE (NO DEVIATION):
1. HEADER
2. Executive Summary
3. Cross-Asset Reality
4. Regional Read (GCC + Egypt)
5. Positioning
6. Risk Framework
7. Tactical Playbook
8. Catalysts
9. Final Line

OUTPUT FORMAT:
EisaX Daily Market Pulse
Date: {date}
Regime: {regime}
Confidence: {conf}
Decision Type: {decision_type}
Confidence Score: {confidence_score}

## Executive Summary
â€¢ [line 1]
â€¢ [line 2]
â€¢ [line 3]
â€¢ [line 4]

## Cross-Asset Reality
â€¢ Equities -> [interpretation]
â€¢ Rates -> [liquidity impact]
â€¢ Oil -> [inflation vs growth]
â€¢ VIX -> [risk behavior]
â€¢ Crypto -> [optional only if it changes the read]
This is not a directional market.
This is a {market_state} market.

## Regional Read (GCC + Egypt)
â€¢ [oil anchor for GCC]
â€¢ [US rates into GCC/Egypt liquidity]
â€¢ [sector flows]

## Positioning
Stance: {view.get("stance", "HOLD")}
Mode: {mode}
Execution:
â€¢ Rule 1
â€¢ Rule 2
â€¢ Rule 3

## Risk Framework
â€¢ SPY level: [specific level logic]
â€¢ VIX level: [specific level logic]
â€¢ 10Y level: [specific level logic]
â€¢ Regime shift: [clear statement]

## Tactical Playbook
â€¢ Maintain: [what to maintain]
â€¢ Deploy: [when to deploy]
â€¢ Avoid: [what to avoid]
â€¢ Focus: [where to focus]

## Catalysts
â€¢ [Fed]
â€¢ [CPI / PCE]
â€¢ [Oil]
â€¢ [Rates]

## Final Line
[One sharp CIO-level sentence]

LOGIC RULES:
- NEVER say "mixed signals"; use "conflicted regime"
- NEVER treat oil as purely bullish; evaluate inflation impact
- ALWAYS prioritize Rates > Liquidity > Equities
- HOLD must be active, not passive
- ALWAYS resolve contradictions
- No fluff, no repetition, no market recap language
- Do not add any section beyond the structure above
- Use only the provided data and triggers
"""
    raw = _call_openai_text(prompt, max_tokens=1600) or _call_gemini(prompt)
    return (raw or "").strip()

def _generate_linkedin_text_ai(daily_data: dict) -> str:
    """Return a clean, scannable LinkedIn-ready version with no headers."""
    return _build_linkedin_text(daily_data)

def _trim_to_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(" ,;:.") + "."

def _fit_word_window(text: str, min_words: int = 120, max_words: int = 180) -> str:
    cleaned = " ".join((text or "").split()).strip()
    words = cleaned.split()
    if len(words) > max_words:
        return _trim_to_words(cleaned, max_words)
    if len(words) >= min_words:
        return cleaned
    filler = (
        " That keeps the decision anchored to price, volatility, and rates, "
        "instead of letting narrative drift override disciplined portfolio action."
    )
    while len(words) < min_words:
        cleaned = (cleaned + filler).strip()
        words = cleaned.split()
    return _trim_to_words(cleaned, max_words)

def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()

def _clean_text_list(values: Any, limit: int) -> list:
    if not isinstance(values, list):
        return []
    cleaned = [_clean_text(v) for v in values]
    return [v for v in cleaned if v][:limit]

def _merge_text_list_with_fallback(values: Any, fallback: Any, limit: int) -> list:
    primary = _clean_text_list(values, limit)
    backup = _clean_text_list(fallback, limit)
    merged = []
    for item in primary + backup:
        if item and item not in merged:
            merged.append(item)
        if len(merged) >= limit:
            break
    return merged

def _normalize_key_moves(values: Any, fallback: list) -> list:
    if not isinstance(values, list):
        return fallback[:5]
    normalized = []
    for item in values:
        if not isinstance(item, dict):
            continue
        asset = _clean_text(item.get("asset"))
        move = _clean_text(item.get("move"))
        reason = _clean_text(item.get("reason"))
        if asset and move and reason:
            normalized.append({"asset": asset, "move": move, "reason": reason})
    return normalized[:5] or fallback[:5]

def _normalize_winners_losers(values: Any, fallback: dict) -> dict:
    values = values if isinstance(values, dict) else {}
    winners = _clean_text_list(values.get("winners"), 3) or fallback.get("winners", [])
    losers = _clean_text_list(values.get("losers"), 3) or fallback.get("losers", [])
    return {"winners": winners[:3], "losers": losers[:3]}

def _normalize_regional_view(values: Any, fallback: dict) -> dict:
    values = values if isinstance(values, dict) else {}
    out = {}
    for key in ("US", "GCC", "Egypt"):
        value = _clean_text(values.get(key))
        if len(value.split()) < 4:
            value = ""
        out[key] = value or _clean_text(fallback.get(key))
    return out

def _snapshot_brief(snapshot: dict) -> str:
    if not isinstance(snapshot, dict):
        return ""

    def _fmt_pct(entry_key: str) -> str:
        val = (snapshot.get(entry_key) or {}).get("d1_pct")
        if isinstance(val, str):
            return val
        if isinstance(val, (int, float)):
            return f"{val:+.1f}%"
        return "Market Closed"

    spy_move = _fmt_pct("us_equities")
    btc_move = _fmt_pct("crypto")
    vix_price = (snapshot.get("volatility") or {}).get("price")
    tnx_price = (snapshot.get("rates") or {}).get("price")
    vix_text = vix_price if isinstance(vix_price, str) else (f"{vix_price:.1f}" if isinstance(vix_price, (int, float)) else "Market Closed")
    tnx_text = tnx_price if isinstance(tnx_price, str) else (f"{tnx_price:.2f}%" if isinstance(tnx_price, (int, float)) else "Market Closed")
    return f"Cross-asset: SPY {spy_move} | VIX {vix_text} | 10Y {tnx_text} | BTC {btc_move}"

def _format_report_number(value: Any, pct: bool = False) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, (int, float)):
        return "Market Closed"
    if pct:
        return f"{value:+.2f}%"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:.2f}"

def _trigger_hierarchy_lines(items: list, title: str) -> list:
    labels = ["Primary trigger", "Secondary", "Tertiary", "Fourth"]
    cleaned = _clean_text_list(items, 4)
    if not cleaned:
        return []
    lines = [title]
    for idx, item in enumerate(cleaned):
        label = labels[idx] if idx < len(labels) else f"Trigger {idx + 1}"
        lines.append(f"• {label}: {item}")
    return lines

def _best_expression_line(update: dict) -> str:
    if "date" in update:
        view = update.get("eisax_view", {}) if isinstance(update.get("eisax_view"), dict) else {}
        stance = _clean_text(view.get("stance")).upper()
        focus = _clean_text(view.get("focus")) or "quality equities"
        if "REDUCE RISK" in stance:
            return "Best expression: raise cash, keep gold and short duration as the first line of defense."
        if "BUY" in stance:
            return f"Best expression: own {focus.lower()} on controlled pullbacks, not on emotional extension."
        return "Best expression: keep exposure centered on quality and wait for cleaner confirmation before pressing risk."

    alloc = update.get("asset_allocation_view") or {}
    if alloc.get("equities") == "Overweight":
        return "Best expression: stay with quality leadership and add only where pullbacks hold structure."
    if alloc.get("equities") == "Underweight" or alloc.get("cash") == "Overweight":
        return "Best expression: hold a defensive core and let cash and gold do part of the risk management."
    return "Best expression: keep gross balanced and upgrade only the highest-quality parts of the book."

def _best_hedge_line(update: dict) -> str:
    if "date" in update:
        view = update.get("eisax_view", {}) if isinstance(update.get("eisax_view"), dict) else {}
        underweights = " ".join(str(x).lower() for x in (view.get("underweight_assets") or []))
        if "crypto" in underweights or "high beta" in underweights:
            return "Best hedge: do not pay up for broad protection too early; reduce high-beta exposure first."
        return "Best hedge: use gold, cash, and disciplined sizing before reaching for expensive index protection."

    alloc = update.get("asset_allocation_view") or {}
    if alloc.get("metals") == "Overweight":
        return "Best hedge: gold remains the cleaner hedge than chasing late defensive beta."
    if alloc.get("cash") == "Overweight":
        return "Best hedge: cash is still a position when volatility is not paying you to be fully invested."
    return "Best hedge: hedge through quality and liquidity first, then add explicit protection only if triggers break."

def _weekly_why_now_lines(update: dict) -> list:
    alloc = update.get("asset_allocation_view") or {}
    snapshot = update.get("cross_asset_snapshot") or {}
    vix = _as_number((snapshot.get("volatility") or {}).get("price"))
    tnx = _as_number((snapshot.get("rates") or {}).get("price"))

    if alloc.get("equities") == "Overweight":
        first = "Markets remain constructive but not crowded."
        second = (
            f"Volatility contained at {vix:.1f} and rates stable near {tnx:.2f}% keep the bullish structure intact."
            if vix is not None and tnx is not None
            else "Volatility contained and rates stable keep the bullish structure intact."
        )
        return [first, second]
    if alloc.get("equities") == "Underweight" or alloc.get("cash") == "Overweight":
        first = "The market is still trading defensively, even if headline moves look calmer."
        second = (
            f"Volatility at {vix:.1f} and rates near {tnx:.2f}% do not justify broad risk expansion yet."
            if vix is not None and tnx is not None
            else "Volatility and rates do not justify broad risk expansion yet."
        )
        return [first, second]
    first = "The tape is investable, but confirmation is still incomplete."
    second = (
        f"Volatility near {vix:.1f} and rates around {tnx:.2f}% keep the setup balanced rather than fully risk-on."
        if vix is not None and tnx is not None
        else "Volatility and rates keep the setup balanced rather than fully risk-on."
    )
    return [first, second]

def _market_view_lines(snapshot: dict, regional_view: Optional[dict] = None) -> list:
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    regional_view = regional_view if isinstance(regional_view, dict) else {}

    def _entry(key: str) -> dict:
        return snapshot.get(key) or {}

    def _closed(entry: dict) -> bool:
        return entry.get("price") == "Market Closed"

    us = _entry("us_equities")
    vix = _entry("volatility")
    btc = _entry("crypto")
    metals = _entry("metals")
    oil = _entry("commodities")
    rates = _entry("rates")
    gcc = _entry("gcc")
    egypt = _entry("egypt")

    lines = ["## Market-by-Market View"]

    if not _closed(us):
        us_d1 = us.get("d1_pct")
        us_d5 = us.get("d5_pct")
        lines += [
            f"### US Equities",
            f"• {us.get('label', 'US Equities')} { _format_report_number(us_d1, pct=True) } on the day and { _format_report_number(us_d5, pct=True) } over five sessions at { _format_report_number(us.get('price')) }.",
            f"• {'Leadership is still constructive, but it should be owned through quality rather than broad beta.' if isinstance(us_d5, (int, float)) and us_d5 > 0 else 'The tape is losing sponsorship, so risk should stay selective and tightly sized.'}",
        ]
    else:
        lines += ["### US Equities", "• Market Closed.", "• Re-open with fresh price confirmation before changing exposure."]

    if not _closed(vix):
        vix_price = vix.get("price")
        lines += [
            f"### Volatility",
            f"• {vix.get('label', 'VIX')} sits at { _format_report_number(vix_price) } with a { _format_report_number(vix.get('d1_pct'), pct=True) } daily move.",
            f"• {'Fear premium is still elevated, so hedges remain expensive but relevant.' if isinstance(vix_price, (int, float)) and vix_price > 20 else 'Fear premium is contained, which keeps the risk budget open for selective adds.'}",
        ]
    else:
        lines += ["### Volatility", "• Market Closed.", "• Use prior session hedging levels as the reference until fresh volatility pricing prints."]

    if not _closed(rates):
        rate = rates.get("price")
        lines += [
            f"### Rates",
            f"• {rates.get('label', '10Y Treasury')} is at { _format_report_number(rate) }% with a { _format_report_number(rates.get('d1_pct'), pct=True) } daily move.",
            f"• {'Rates are tightening the valuation window again, which caps multiple expansion.' if isinstance(rates.get('d1_pct'), (int, float)) and rates.get('d1_pct') > 0 else 'Rates are giving equities duration relief, which supports leadership if breadth confirms.'}",
        ]
    else:
        lines += ["### Rates", "• Market Closed.", "• Use the prior yield regime as the macro anchor until the Treasury market reopens."]

    if not _closed(btc):
        btc_d5 = btc.get("d5_pct")
        lines += [
            f"### Crypto",
            f"• {btc.get('label', 'Bitcoin')} is { _format_report_number(btc.get('d1_pct'), pct=True) } on the day and { _format_report_number(btc_d5, pct=True) } over five sessions at { _format_report_number(btc.get('price')) }.",
            f"• {'Crypto is confirming the risk bid rather than fighting it.' if isinstance(btc_d5, (int, float)) and btc_d5 >= 0 else 'Crypto is withholding confirmation, so this is not a clean all-risk-on tape.'}",
        ]
    else:
        lines += ["### Crypto", "• Market Closed.", "• Treat crypto as an unconfirmed sleeve until fresh liquidity and price data return."]

    if not _closed(metals):
        metals_d5 = metals.get("d5_pct")
        lines += [
            f"### Metals",
            f"• {metals.get('label', 'Gold')} is { _format_report_number(metals.get('d1_pct'), pct=True) } on the day and { _format_report_number(metals_d5, pct=True) } over five sessions at { _format_report_number(metals.get('price')) }.",
            f"• {'Gold still holding demand means defense has not been abandoned.' if isinstance(metals_d5, (int, float)) and metals_d5 >= 0 else 'Soft metals performance suggests portfolios are not paying up for protection today.'}",
        ]
    else:
        lines += ["### Metals", "• Market Closed.", "• Prior metals pricing remains the best read on defensive demand until the market reopens."]

    if not _closed(oil):
        oil_d5 = oil.get("d5_pct")
        lines += [
            f"### Oil",
            f"• {oil.get('label', 'Oil')} is { _format_report_number(oil.get('d1_pct'), pct=True) } on the day and { _format_report_number(oil_d5, pct=True) } over five sessions at { _format_report_number(oil.get('price')) }.",
            f"• {'Oil is reinforcing the growth read and can reopen inflation pressure if the move persists.' if isinstance(oil_d5, (int, float)) and oil_d5 > 0 else 'Recent pullback in oil tempers the global growth signal without fully breaking it.'}",
        ]
    else:
        lines += ["### Oil", "• Market Closed.", "• Keep the last traded oil regime in mind for inflation and GCC sensitivity."]

    if not _closed(gcc):
        lines += [
            f"### GCC",
            f"• {gcc.get('label', 'Saudi TASI')} is { _format_report_number(gcc.get('d1_pct'), pct=True) } on the day and { _format_report_number(gcc.get('d5_pct'), pct=True) } over five sessions at { _format_report_number(gcc.get('price')) }.",
            f"• {regional_view.get('GCC') or 'GCC direction still runs through oil and domestic liquidity rather than global beta alone.'}",
        ]
    else:
        lines += [
            "### GCC",
            "• Market Closed.",
            f"• {regional_view.get('GCC') or 'Keep oil and domestic liquidity as the primary GCC read-through until fresh local pricing returns.'}",
        ]

    if not _closed(egypt):
        lines += [
            f"### Egypt",
            f"• {egypt.get('label', 'Egypt EGX30')} is { _format_report_number(egypt.get('d1_pct'), pct=True) } on the day and { _format_report_number(egypt.get('d5_pct'), pct=True) } over five sessions at { _format_report_number(egypt.get('price')) }.",
            f"• {regional_view.get('Egypt') or 'Egypt remains more sensitive to dollar liquidity and imported inflation than to headline US equity strength.'}",
        ]
    else:
        lines += [
            "### Egypt",
            "• Market Closed.",
            f"• {regional_view.get('Egypt') or 'Egypt should still be read through the dollar, rates, and imported inflation channel until the market reopens.'}",
        ]

    return lines

def _enrich_full_report(report: str, update: dict) -> str:
    report = (report or "").strip()
    if "Portfolio Translation" not in report:
        translation_block = "\n".join([
            "## Portfolio Translation",
            f"• {_best_expression_line(update)}",
            f"• {_best_hedge_line(update)}",
        ]).strip()
        report = f"{report}\n\n{translation_block}".strip()

    trigger_source = update.get("what_invalidates") if "date" in update else update.get("what_changes_this_view")
    if "Primary trigger:" not in report and trigger_source:
        trigger_block = ["## Trigger Hierarchy"]
        trigger_block.extend(_trigger_hierarchy_lines(trigger_source, "")[1:])
        report = f"{report}\n\n{chr(10).join(trigger_block).strip()}".strip()

    if "date" not in update and "Why Now:" not in report:
        why_block = "\n".join(["## Why Now"] + [f"• {item}" for item in _weekly_why_now_lines(update)]).strip()
        report = f"{report}\n\n{why_block}".strip()

    market_view_lines = _market_view_lines(update.get("cross_asset_snapshot") or {}, update.get("regional_view") or {})
    market_view_block = "\n".join(market_view_lines).strip()
    if "## Market-by-Market View" not in report:
        report = f"{report}\n\n{market_view_block}".strip()
    return report

def _allocation_summary(allocation: dict) -> str:
    if not isinstance(allocation, dict):
        return ""
    order = [
        ("equities", "Eq"),
        ("crypto", "Crypto"),
        ("metals", "Metals"),
        ("commodities", "Cmdty"),
        ("cash", "Cash"),
    ]
    compact = {"Overweight": "OW", "Neutral": "N", "Underweight": "UW"}
    pieces = []
    for key, label in order:
        val = allocation.get(key)
        if val:
            pieces.append(f"{label} {compact.get(val, val)}")
    return "Allocation: " + " | ".join(pieces) if pieces else ""

def _as_number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None

def _daily_linkedin_hook(update: dict) -> str:
    snapshot = update.get("cross_asset_snapshot") or {}
    spy = _as_number((snapshot.get("us_equities") or {}).get("d1_pct"))
    oil = _as_number((snapshot.get("commodities") or {}).get("d1_pct"))
    gold = _as_number((snapshot.get("metals") or {}).get("d1_pct"))
    btc = _as_number((snapshot.get("crypto") or {}).get("d1_pct"))
    vix = _as_number((snapshot.get("volatility") or {}).get("price"))
    tnx = _as_number((snapshot.get("rates") or {}).get("price"))

    if spy and spy > 0.25 and oil and oil < -0.5:
        return "Equities are trading a lower-rate impulse, not a growth impulse."
    if spy and spy > 0.25 and gold and gold > 0 and tnx is not None and tnx <= 4.35:
        return "Risk is being added without the market giving an all-clear."
    if spy and spy < -0.25 and gold and gold > 0:
        return "Protection is outperforming before the tape fully admits the slowdown."
    if spy is not None and spy >= 0 and vix and vix > 20:
        return "Index strength is holding, but the hedge is still expensive."
    if btc and btc > 1 and tnx is not None and tnx < 4.35:
        return "Beta is working, but only where rates are doing the lifting."
    return "The tape is moving, but breadth is still too narrow to trust blindly."

def _weekly_linkedin_hook(update: dict) -> str:
    winners = " ".join((update.get("winners_losers") or {}).get("winners", [])).lower()
    losers = " ".join((update.get("winners_losers") or {}).get("losers", [])).lower()
    allocation = update.get("asset_allocation_view") or {}

    if any(x in winners for x in ("qqq", "spy")) and any(x in losers for x in ("uso", "oil")):
        return "Leadership is staying with duration while oil refuses the growth message."
    if any(x in winners for x in ("gld", "gold")) and any(x in losers for x in ("btc", "crypto")):
        return "Defensives are getting paid while speculative beta is not earning a wider leash."
    if allocation.get("equities") == "Underweight" or allocation.get("cash") == "Overweight":
        return "Defense is working, but the tape is not rewarding blind de-risking."
    if allocation.get("equities") == "Neutral" and allocation.get("metals") == "Overweight":
        return "This is a holding pattern, not a launch pad."
    return "The tape is investable, but only through selectivity."

def _weekly_stance_label(update: dict) -> str:
    allocation = update.get("asset_allocation_view") or {}
    equities = allocation.get("equities")
    cash = allocation.get("cash")
    if equities == "Overweight" and cash == "Underweight":
        return "Tactical BUY"
    if equities == "Underweight" or cash == "Overweight":
        return "REDUCE RISK"
    return "HOLD"

def _daily_positioning_line(update: dict) -> str:
    positioning = _clean_text(update.get("tactical_positioning")).rstrip(".")
    view = update.get("eisax_view", {}) if isinstance(update.get("eisax_view"), dict) else {}
    underweights = [str(x).lower() for x in (view.get("underweight_assets") or [])]
    avoid = "broad beta and weak cyclicals"
    if any("speculative" in item for item in underweights):
        avoid = "speculative beta"
    elif any("crypto" in item for item in underweights):
        avoid = "unfunded crypto beta"
    elif any("equities" in item for item in underweights):
        avoid = "indiscriminate equity exposure"
    return f"Positioning: {positioning}. Avoid {avoid}."

def _weekly_positioning_line(update: dict) -> str:
    positioning = _clean_text(update.get("positioning")).rstrip(".")
    angle = _clean_text(update.get("portfolio_angle")).rstrip(".")
    avoid = "weak cyclicals and late beta"
    if "gold" in angle.lower():
        avoid = "chasing rebounds against the regime"
    return f"Positioning: {positioning}. Avoid {avoid}."

def _daily_linkedin_insight_lines(update: dict) -> list:
    snapshot = update.get("cross_asset_snapshot") or {}
    spy = _as_number((snapshot.get("us_equities") or {}).get("d1_pct"))
    oil = _as_number((snapshot.get("commodities") or {}).get("d1_pct"))
    gold = _as_number((snapshot.get("metals") or {}).get("d1_pct"))
    btc = _as_number((snapshot.get("crypto") or {}).get("d1_pct"))
    tnx = _as_number((snapshot.get("rates") or {}).get("price"))
    lines = []
    if tnx is not None and tnx <= 4.35 and btc is not None and btc > 0:
        lines.append("Lower yields and participating crypto keep pressure on underweight books, but the move is still concentrated.")
    if spy is not None and spy > 0.25 and oil is not None and oil < -0.5:
        lines.append("Oil failing to confirm the equity move argues against a broad cyclical read-through.")
    elif spy is not None and spy > 0.25 and gold is not None and gold > 0:
        lines.append("Gold holding in while equities rise tells you portfolios are adding risk without abandoning defense.")
    defaults = [
        "This is a quality expression, not an all-clear for indiscriminate beta.",
        "The trade still needs selectivity because leadership is tighter than the index suggests.",
    ]
    for default in defaults:
        if len(lines) >= 2:
            break
        lines.append(default)
    return lines[:2]

def _weekly_linkedin_insight_lines(update: dict) -> list:
    winners = " ".join((update.get("winners_losers") or {}).get("winners", [])).lower()
    losers = " ".join((update.get("winners_losers") or {}).get("losers", [])).lower()
    allocation = update.get("asset_allocation_view") or {}
    lines = []
    if any(x in winners for x in ("gld", "gold")) and any(x in losers for x in ("uso", "oil", "btc")):
        lines.append("Gold holding leadership while oil and speculative beta lag keeps the macro message mixed.")
    if allocation.get("equities") == "Neutral" and allocation.get("metals") == "Overweight":
        lines.append("That argues for tighter entry standards and selective adds, not broader gross expansion.")
    elif allocation.get("equities") == "Overweight":
        lines.append("The tape still rewards leadership, but only where balance sheets and earnings carry the trade.")
    else:
        lines.append("Capital is still rewarding selectivity over size, which matters more than the headline tape.")
    if "gold" in winners:
        lines.append("The cleaner hedge is still gold, not a blind reach for cyclical rebound.")
    else:
        lines.append("The book should still be financed through selectivity, not optimism.")
    defaults = [
        "The market is still paying up for balance-sheet quality over operating leverage.",
        "Risk should be deployed through confirmed leaders, not broad market hope.",
        "The cleaner hedge is still quality and carry, not narrative beta.",
    ]
    for default in defaults:
        if len(lines) >= 3:
            break
        lines.append(default)
    return lines[:3]

def _weekly_focus_text(update: dict) -> str:
    allocation = update.get("asset_allocation_view") or {}
    if allocation.get("equities") == "Overweight" and allocation.get("cash") == "Underweight":
        return "quality equities"
    if allocation.get("equities") == "Underweight" or allocation.get("cash") == "Overweight":
        return "defense and capital preservation"
    if allocation.get("metals") == "Overweight":
        return "selective quality with hedges"
    return "selective quality, not broad beta"

def _build_web_version(update: dict) -> str:
    if "date" in update:
        view = update.get("eisax_view", {}) if isinstance(update.get("eisax_view"), dict) else {}
        stance = _clean_text(view.get("stance")) or "HOLD"
        focus = _clean_text(view.get("focus"))
        why_now = _clean_text(update.get("why_now"))
        tactical = _clean_text(update.get("tactical_positioning"))
        risk = (_clean_text_list(update.get("what_invalidates"), 1) or [""])[0]
        snapshot_line = _snapshot_brief(update.get("cross_asset_snapshot") or {})
        parts = [
            f"{update.get('date', '')} | {update.get('market_regime', 'Cautious')} | {stance}",
            f"Focus: {focus}" if focus else "",
            snapshot_line,
            f"Why now: {why_now}" if why_now else "",
            f"Positioning: {tactical}" if tactical else "",
            f"Invalidation: {risk}" if risk else "",
        ]
        return "\n".join(p for p in parts if p).strip()

    verdict = _clean_text(update.get("eisax_verdict"))
    summary = _clean_text(update.get("market_summary"))
    positioning = _clean_text(update.get("positioning"))
    risk = (_clean_text_list(update.get("what_changes_this_view"), 1) or [""])[0]
    allocation = _allocation_summary(update.get("asset_allocation_view") or {})
    idea = _clean_text(update.get("highest_conviction_opportunity"))
    parts = [
        f"Weekly Strategy | {update.get('week_range', '')}",
        allocation,
        f"Positioning: {_trim_to_words(positioning, 18)}" if positioning else "",
        f"Best idea: {_trim_to_words(idea, 18)}" if idea else "",
        f"Verdict: {_trim_to_words(verdict, 16)}" if verdict else "",
        f"Invalidation: {_trim_to_words(risk, 18)}" if risk else "",
        _trim_to_words(summary, 22) if summary else "",
    ]
    return "\n".join(p for p in parts if p).strip()

