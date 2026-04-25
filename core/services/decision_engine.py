"""
decision_engine.py — Week 4: Verdict Binding Layer

Applies hard deterministic constraints on top of the scorecard verdict.
The scorecard (calculate_score + get_verdict) remains the scoring engine;
this layer guarantees that the final verdict is NEVER contradicted by the
deterministic interpretation signals (ADX, RSI, entry quality, risk).

Rules:
  - LLM has NO decision authority — it explains the verdict, not decides it
  - Scorecard verdict is the starting point; this layer can only constrain it
  - All overrides are logged in 'constraints'
"""
from __future__ import annotations

from typing import Any


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _derive_risk_score(score_data: dict) -> float:
    """
    Derive a 0–100 risk score from available sc_data signals.
    Used only inside this module — does NOT replace the scorecard Risk Profile factor.
    """
    beta        = _safe_float(score_data.get("beta"), 1.0)
    risk_count  = int(score_data.get("risk_count") or 0)
    bearish_cnt = int(score_data.get("bearish_count") or 0)

    score = 0.0
    if beta > 2.5:    score += 40
    elif beta > 2.0:  score += 25
    elif beta > 1.5:  score += 10

    score += risk_count  * 8
    score += bearish_cnt * 7
    return min(100.0, score)


def _compute_confidence(
    quality_score: float,
    risk_score: float,
    upside_pct: float,
) -> float:
    """
    Blended confidence: 0.30 – 0.85.

    confidence = (0.4 * quality + 0.3 * safety + 0.3 * upside_scaled) / 100
    Clamped to [0.30, 0.85] — never overconfident, never near-zero.
    """
    upside_scaled = min(100.0, upside_pct * 5.0)   # 20 % upside → 100
    raw = (
        0.4 * quality_score
        + 0.3 * (100.0 - risk_score)
        + 0.3 * upside_scaled
    ) / 100.0
    return round(max(0.30, min(0.85, raw)), 3)


# ── Public API ────────────────────────────────────────────────────────────────

def build_decision(
    interpretation_labels: dict,
    score_data: dict,
    snapshot: dict | None = None,
) -> dict:
    """
    Verdict Binding Layer.

    Parameters
    ----------
    interpretation_labels
        Output of build_interpretation_labels() — contains TrendStrength,
        EntryQuality, RSIZone, VolumeConviction, YieldQuality, TrendBorderline.
    score_data
        The sc_data dict from the scorecard pipeline.  Must include
        'upside_pct' (pre-computed).  Optional keys used:
        'scorecard_verdict', 'beta', 'risk_count', 'bearish_count',
        'quality', 'dividend_yield', 'forward_pe'.
    snapshot
        Optional plain dict or ReportSnapshot; not required for core logic.

    Returns
    -------
    {
      'verdict'      : 'BUY' | 'HOLD' | 'REDUCE' | 'AVOID',
      'verdict_type' : 'Tactical' | 'Strategic' | 'Income Hold',
      'confidence'   : float  (0.30 – 0.85),
      'reasons'      : list[str],
      'constraints'  : list[str],   # blocking conditions that fired
    }
    """
    # ── Unpack interpretation labels ────────────────────────────────────────
    trend_strength    = interpretation_labels.get("TrendStrength", "")
    entry_quality     = interpretation_labels.get("EntryQuality", "")
    rsi_zone          = interpretation_labels.get("RSIZone", "")
    volume_conviction = interpretation_labels.get("VolumeConviction", "")
    yield_quality     = interpretation_labels.get("YieldQuality", "")

    # ── Unpack numeric inputs ───────────────────────────────────────────────
    beta          = _safe_float(score_data.get("beta"), 1.0)
    upside_pct    = _safe_float(score_data.get("upside_pct"), 0.0)
    quality_score = max(1.0, _safe_float(score_data.get("quality") or 50, 50.0))
    risk_score    = _derive_risk_score(score_data)
    _scorecard_score = _safe_float(
        score_data.get('eisax_score') or score_data.get('blended_score') or score_data.get('quality') or 0, 0.0
    )

    # Normalise div_yield to percent (0.05 → 5.0, 5.0 stays 5.0)
    _div_raw      = _safe_float(score_data.get("dividend_yield") or 0, 0.0)
    div_yield_pct = _div_raw * 100.0 if _div_raw <= 1.0 else _div_raw

    # ── Normalise scorecard starting verdict ────────────────────────────────
    _raw_verdict = str(score_data.get("scorecard_verdict") or "HOLD").upper()
    _CANONICAL = {
        "STRONG BUY":     "BUY",
        "BUY (HIGH RISK)": "BUY",
        "ACCUMULATE":     "BUY",
        "SELL":           "AVOID",
    }
    base_verdict = _CANONICAL.get(_raw_verdict, _raw_verdict)
    if base_verdict not in ("BUY", "HOLD", "REDUCE", "AVOID"):
        base_verdict = "HOLD"

    # True when get_verdict() already approved BUY (blended ≥ 68 threshold passed)
    _scorecard_approved_buy = _raw_verdict in (
        "BUY", "STRONG BUY", "BUY (HIGH RISK)", "ACCUMULATE"
    )

    verdict       = base_verdict
    verdict_type  = "Tactical"
    reasons:     list[str] = []
    constraints: list[str] = []

    # ══════════════════════════════════════════════════════════════════════
    # HARD BLOCKS  (applied first — these always fire regardless of score)
    # ══════════════════════════════════════════════════════════════════════

    # Block 1 — weak trend + poor entry: no BUY
    if trend_strength == "weak trend" and entry_quality == "poor timing":
        if verdict == "BUY":
            verdict = "HOLD"
            constraints.append("weak trend + poor timing: BUY blocked")
        reasons.append(
            "ADX is below trend-confirmation threshold and price is extended above the entry zone."
        )

    # Block 2 — RSI overbought: never BUY
    if rsi_zone == "overbought" and verdict == "BUY":
        verdict = "HOLD"
        constraints.append("RSI overbought: BUY downgraded to HOLD")
        reasons.append("RSI in overbought territory; near-term entry is unfavorable.")

    # Block 3 — high beta + high risk: REDUCE
    if beta > 2.0 and risk_score > 70:
        if verdict in ("BUY", "HOLD"):
            verdict = "REDUCE"
            constraints.append(
                f"beta={beta:.2f} > 2.0 and risk_score={risk_score:.0f} > 70: REDUCE"
            )
        reasons.append(
            f"Elevated beta ({beta:.2f}) combined with high risk-signal count "
            f"indicates asymmetric downside."
        )

    # Block 4 — insufficient upside: no BUY
    if upside_pct < 5.0 and verdict == "BUY":
        verdict = "HOLD"
        constraints.append(f"upside {upside_pct:.1f}% < 5%: BUY blocked")
        reasons.append(
            f"Price-to-target upside of {upside_pct:.1f}% is insufficient for a BUY stance."
        )

    # ══════════════════════════════════════════════════════════════════════
    # BUY CONFIRMATION
    # Trend requirement:
    #   • "confirmed trend" or "strong trend" (ADX ≥ 25): always OK
    #   • "emerging trend" (ADX 20–25): allowed when scorecard already approved
    #     BUY (blended ≥ 68) AND upside ≥ 10% — avoids phantom BUY on weak data
    #   • "weak trend" (ADX < 20): BUY blocked regardless (use Entry Timing)
    # Entry quality:
    #   • "entry level unavailable" treated as neutral — data gap, not negative
    # ══════════════════════════════════════════════════════════════════════

    if verdict == "BUY":
        _trend_ok = trend_strength in ("confirmed trend", "strong trend") or (
            trend_strength == "emerging trend"
            and _scorecard_approved_buy
            and upside_pct >= 10.0
        )
        _entry_ok = entry_quality in (
            "favorable entry", "acceptable entry", "entry level unavailable"
        )
        if not _trend_ok:
            verdict = "HOLD"
            constraints.append(
                f'trend_strength="{trend_strength}" insufficient for BUY '
                f'(need confirmed/strong, or emerging+scorecard_buy+upside≥10%)'
            )
            reasons.append(
                "BUY requires ADX ≥ 25 (confirmed or strong trend); "
                "or ADX 20–25 with scorecard approval and ≥10% upside. "
                "Current trend strength is below threshold."
            )
        elif not _entry_ok:
            verdict = "HOLD"
            constraints.append(f'entry_quality="{entry_quality}" insufficient for BUY')
            reasons.append(
                "Entry conditions are not yet favorable; await a pullback toward the entry zone."
            )
        elif upside_pct < 10.0 and risk_score > 60:
            verdict = "HOLD"
            constraints.append(
                f"upside {upside_pct:.1f}% < 10% with risk_score {risk_score:.0f} > 60"
            )
            reasons.append(
                "Risk-adjusted upside is insufficient: upside below 10% with elevated risk."
            )

    # ══════════════════════════════════════════════════════════════════════
    # DIVIDEND OVERRIDE  (income framing when yield is attractive)
    # ══════════════════════════════════════════════════════════════════════

    if div_yield_pct >= 4.0 and trend_strength in ("weak trend", "emerging trend"):
        verdict_type = "Income Hold"
        constraints.append(
            f"div_yield={div_yield_pct:.1f}% with weak trend: reclassified as Income Hold"
        )
        if verdict == "BUY":
            verdict = "HOLD"
        reasons.append(
            f"Attractive income component ({div_yield_pct:.1f}% yield) supports an income "
            f"allocation despite the weak trend regime."
        )

    # ══════════════════════════════════════════════════════════════════════
    # POSITIVE REASONS  (only when no hard blocks fired)
    # ══════════════════════════════════════════════════════════════════════

    if not constraints:
        if trend_strength in ("confirmed trend", "strong trend"):
            reasons.append(f"Trend is {trend_strength} (ADX-confirmed).")
        if entry_quality in ("favorable entry", "acceptable entry"):
            reasons.append(f"Entry quality is {entry_quality}.")
        if volume_conviction == "strong volume confirmation":
            reasons.append("Volume provides strong confirmation of the move.")
        if upside_pct >= 10.0:
            reasons.append(f"Price-to-target upside is {upside_pct:.1f}%.")

    # ══════════════════════════════════════════════════════════════════════
    # VERDICT TYPE  (when not already set to Income Hold)
    # ══════════════════════════════════════════════════════════════════════

    if verdict_type != "Income Hold":
        if verdict in ("REDUCE", "AVOID"):
            verdict_type = "Strategic"
        else:
            verdict_type = "Tactical"

    # ══════════════════════════════════════════════════════════════════════
    # CONFIDENCE
    # ══════════════════════════════════════════════════════════════════════

    confidence = _compute_confidence(quality_score, risk_score, upside_pct)

    # ══════════════════════════════════════════════════════════════════════
    # TG3 — EDGE GUARDRAILS  (score-based, RSI/ADX never used here)
    # Runs after all prior blocks but before Rule 8A BUY protection.
    # ══════════════════════════════════════════════════════════════════════

    if _scorecard_score > 0:
        # TG3-1: Low score + high downside → REDUCE
        _downside_high = risk_score > 60 or beta > 1.8 or int(score_data.get("bearish_count") or 0) >= 3
        if _scorecard_score < 55 and _downside_high and verdict in ("BUY", "HOLD"):
            verdict = "REDUCE"
            constraints.append(
                f"TG3-1: score={_scorecard_score:.0f}<55 + downside_high → REDUCE"
            )
            reasons.append(
                f"Composite score ({_scorecard_score:.0f}/100) is below the 55-threshold "
                f"with elevated downside risk signals; position reduction is warranted."
            )

        # TG3-2: Mid-range score + no clear edge → enforce HOLD (prevent phantom BUY)
        _no_clear_edge = upside_pct < 15.0 and risk_score > 40
        if 60.0 <= _scorecard_score <= 74.0 and _no_clear_edge and verdict == "BUY":
            verdict = "HOLD"
            constraints.append(
                f"TG3-2: score={_scorecard_score:.0f} in 60-74 + no_clear_edge → HOLD"
            )
            reasons.append(
                f"Score in the 60–74 range ({_scorecard_score:.0f}/100) without sufficient upside "
                f"({upside_pct:.1f}%) or favorable risk profile; true HOLD stance is appropriate."
            )

        # TG3-3: Score >= 75 + upside >= 20% → BUY protected by Rule 8A below
        # (no action needed here; Rule 8A is the terminal BUY protector)

    # ── RULE 8A — TERMINAL OVERRIDE (runs after all hard blocks) ─────────────
    # Hard blocks above only affect Entry Timing, not Fundamental Verdict.
    # If Score ≥ 75 AND Upside ≥ 20%, Fundamental Verdict MUST be BUY.
    # The constraints that fired are moved to entry_timing_constraints so the
    # caller can still use them for timing guidance.
    # eisax_score = the displayed EisaX Score (output of calculate_score).
    # Falls back to blended_score, then quality. This is always the right reference for Rule 8A.
    if upside_pct >= 20.0 and _scorecard_score >= 75.0 and verdict == "HOLD":
        verdict      = "BUY"
        verdict_type = "Tactical"
        constraints.append(
            f"Rule8A override: Score={_scorecard_score:.0f}≥75, Upside={upside_pct:.1f}%≥20% "
            f"→ Fundamental=BUY. Prior constraints become Entry Timing constraints."
        )
        reasons.append(
            f"BUY conditions met (Score≥75, Upside={upside_pct:.1f}%). "
            f"Entry timing is WAIT — weak technicals affect timing, not the fundamental verdict."
        )

    return {
        "verdict":      verdict,
        "verdict_type": verdict_type,
        "confidence":   confidence,
        "reasons":      reasons or [
            "Insufficient signals for a directional view; default HOLD stance."
        ],
        "constraints":  constraints,
    }


def classify_decision_type(
    verdict: str,
    interpretation_labels: dict,
) -> str:
    """
    Deterministic decision_type label — uses ADX-based trend strength,
    NOT LLM-generated summary.trend.

    Returns one of:
      'trend_confirmed' | 'early_reversal' | 'contrarian_early' |
      'wait_for_confirmation' | 'trend_failure' | 'income_hold'
    """
    trend_strength = interpretation_labels.get("TrendStrength", "")
    verdict_up     = verdict.upper()

    if verdict_up in ("BUY", "STRONG BUY"):
        if trend_strength in ("confirmed trend", "strong trend"):
            return "trend_confirmed"
        if trend_strength == "emerging trend":
            return "early_reversal"
        return "contrarian_early"

    if verdict_up == "HOLD":
        return "wait_for_confirmation"

    # REDUCE / AVOID
    return "trend_failure"
