"""
test_week4_decision_engine.py — Week 4: Decision Engine (Verdict Binding Layer)

Tests:
  1.  ADX weak + poor entry              → HOLD
  2.  Confirmed trend + good entry       → BUY
  3.  Overbought RSI                     → no BUY allowed
  4.  High dividend + weak trend         → Income Hold (HOLD + Income Hold type)
  5.  Low upside (<5 %)                  → HOLD
  6.  High beta + high risk              → REDUCE
  7.  Weak trend alone (no bad entry)    → BUY blocked (need confirmed trend)
  8.  BUY with upside 8% + risk_score>60 → HOLD
  9.  classify_decision_type: confirmed  → trend_confirmed
  10. classify_decision_type: weak BUY   → contrarian_early
  11. lint_verdict_consistency: BUY+weak → ERROR
  12. lint_verdict_consistency: "Trend Confirmed" text + weak ADX → ERROR
  13. lint_verdict_consistency: overbought + BUY → ERROR
  14. lint_verdict_consistency: clean BUY + confirmed trend → PASS
  15. build_quick_insight with decision HOLD + constraint
  16. build_quick_insight with decision BUY + confirmed trend
  17. confidence is always clamped to [0.30, 0.85]
  18. lint_report blocks render when verdict contradicts trend (pipeline integration)
  19. lint_report passes clean verdict without blocking
  20. lint_report SKIPs verdict check when decision not provided
"""
from __future__ import annotations

import pytest

from core.services.decision_engine import build_decision, classify_decision_type
from core.services.phrase_builder import build_quick_insight
from core.services.report_lint import (
    RenderedReport,
    ReportSection,
    lint_report,
    lint_verdict_consistency,
)


# ── shared helpers ────────────────────────────────────────────────────────────

def _labels(
    trend_strength="weak trend",
    entry_quality="poor timing",
    rsi_zone="neutral momentum",
    volume="normal volume conviction",
    yield_quality="moderate yield",
    borderline=False,
) -> dict:
    return {
        "TrendStrength":    trend_strength,
        "EntryQuality":     entry_quality,
        "RSIZone":          rsi_zone,
        "VolumeConviction": volume,
        "YieldQuality":     yield_quality,
        "TrendBorderline":  borderline,
    }


def _score(
    verdict="BUY",
    upside_pct=15.0,
    beta=1.0,
    risk_count=1,
    bearish_count=1,
    quality=65,
    dividend_yield=0.01,
) -> dict:
    return {
        "scorecard_verdict": verdict,
        "upside_pct":        upside_pct,
        "beta":              beta,
        "risk_count":        risk_count,
        "bearish_count":     bearish_count,
        "quality":           quality,
        "dividend_yield":    dividend_yield,
    }


def _report(text: str) -> RenderedReport:
    return RenderedReport(
        ticker="TEST",
        full_text=text,
        sections=[ReportSection("Memo", text)],
        observed_prices=[100.0],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# build_decision — HARD BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════

def test_weak_trend_poor_entry_is_hold():
    """Block 1: weak trend + poor entry → verdict=HOLD regardless of scorecard BUY."""
    result = build_decision(
        _labels(trend_strength="weak trend", entry_quality="poor timing"),
        _score(verdict="BUY", upside_pct=20.0),
    )
    assert result["verdict"] == "HOLD"
    assert any("weak trend + poor timing" in c for c in result["constraints"])


def test_confirmed_trend_good_entry_allows_buy():
    """BUY confirmation: confirmed trend + acceptable entry + upside>=10% → BUY."""
    result = build_decision(
        _labels(
            trend_strength="confirmed trend",
            entry_quality="acceptable entry",
            rsi_zone="bullish momentum",
        ),
        _score(verdict="BUY", upside_pct=18.0, risk_count=0, bearish_count=0, beta=0.9),
    )
    assert result["verdict"] == "BUY"
    assert not result["constraints"]


def test_overbought_rsi_blocks_buy():
    """Block 2: RSI overbought → BUY downgraded to HOLD."""
    result = build_decision(
        _labels(
            trend_strength="confirmed trend",
            entry_quality="acceptable entry",
            rsi_zone="overbought",
        ),
        _score(verdict="BUY", upside_pct=20.0),
    )
    assert result["verdict"] == "HOLD"
    assert any("RSI overbought" in c for c in result["constraints"])


def test_high_dividend_weak_trend_income_hold():
    """Dividend override: div_yield>=4% + weak trend → Income Hold."""
    result = build_decision(
        _labels(trend_strength="weak trend", entry_quality="acceptable entry"),
        _score(verdict="HOLD", upside_pct=8.0, dividend_yield=0.055),  # 5.5%
    )
    assert result["verdict"] == "HOLD"
    assert result["verdict_type"] == "Income Hold"
    assert any("Income Hold" in c for c in result["constraints"])


def test_low_upside_blocks_buy():
    """Block 4: upside < 5% → HOLD."""
    result = build_decision(
        _labels(trend_strength="confirmed trend", entry_quality="favorable entry"),
        _score(verdict="BUY", upside_pct=3.5),
    )
    assert result["verdict"] == "HOLD"
    assert any("< 5%" in c for c in result["constraints"])


def test_high_beta_high_risk_reduces():
    """Block 3: beta > 2.0 AND risk_score > 70 → REDUCE."""
    result = build_decision(
        _labels(trend_strength="confirmed trend", entry_quality="acceptable entry"),
        _score(verdict="BUY", upside_pct=20.0, beta=2.5, risk_count=5, bearish_count=5),
    )
    assert result["verdict"] == "REDUCE"
    assert any("beta" in c for c in result["constraints"])


# ═══════════════════════════════════════════════════════════════════════════════
# build_decision — BUY CONFIRMATION RULES
# ═══════════════════════════════════════════════════════════════════════════════

def test_weak_trend_alone_blocks_buy():
    """Weak trend with good entry still blocks BUY (need confirmed/strong trend)."""
    result = build_decision(
        _labels(trend_strength="weak trend", entry_quality="favorable entry",
                rsi_zone="bullish momentum"),
        _score(verdict="BUY", upside_pct=20.0, beta=0.8, risk_count=0, bearish_count=0),
    )
    assert result["verdict"] == "HOLD"
    assert any("insufficient for BUY" in c for c in result["constraints"])


def test_buy_low_upside_high_risk_is_hold():
    """upside 8% + risk_score > 60 → HOLD even if trend confirmed.
    beta=1.8, risk_count=4, bearish_count=4 → risk_score=10+32+28=70 > 60."""
    result = build_decision(
        _labels(trend_strength="confirmed trend", entry_quality="acceptable entry"),
        _score(verdict="BUY", upside_pct=8.0, beta=1.8, risk_count=4, bearish_count=4),
    )
    assert result["verdict"] == "HOLD"
    assert any("upside" in c and "risk_score" in c for c in result["constraints"])


# ═══════════════════════════════════════════════════════════════════════════════
# classify_decision_type
# ═══════════════════════════════════════════════════════════════════════════════

def test_classify_decision_type_confirmed_trend():
    labels = _labels(trend_strength="confirmed trend")
    assert classify_decision_type("BUY", labels) == "trend_confirmed"


def test_classify_decision_type_strong_trend():
    labels = _labels(trend_strength="strong trend")
    assert classify_decision_type("BUY", labels) == "trend_confirmed"


def test_classify_decision_type_weak_trend_buy():
    labels = _labels(trend_strength="weak trend")
    assert classify_decision_type("BUY", labels) == "contrarian_early"


def test_classify_decision_type_emerging_trend_buy():
    labels = _labels(trend_strength="emerging trend")
    assert classify_decision_type("BUY", labels) == "early_reversal"


def test_classify_decision_type_hold():
    labels = _labels(trend_strength="weak trend")
    assert classify_decision_type("HOLD", labels) == "wait_for_confirmation"


def test_classify_decision_type_reduce():
    labels = _labels(trend_strength="weak trend")
    assert classify_decision_type("REDUCE", labels) == "trend_failure"


# ═══════════════════════════════════════════════════════════════════════════════
# lint_verdict_consistency
# ═══════════════════════════════════════════════════════════════════════════════

def test_lint_verdict_buy_weak_trend_is_error():
    decision = {"verdict": "BUY", "verdict_type": "Tactical", "constraints": []}
    result = lint_verdict_consistency(
        _report("Setup looks bullish."),
        _labels(trend_strength="weak trend"),
        decision,
    )
    assert result["result"] == "ERROR"
    assert any("VERDICT_TREND_CONFLICT" in e for e in result["errors"])


def test_lint_verdict_trend_confirmed_text_with_weak_adx_is_error():
    decision = {"verdict": "HOLD", "verdict_type": "Tactical", "constraints": []}
    result = lint_verdict_consistency(
        _report("The stock is now in a Trend Confirmed phase."),
        _labels(trend_strength="weak trend"),
        decision,
    )
    assert result["result"] == "ERROR"
    assert any("TREND_CONFIRMED_LANGUAGE" in e for e in result["errors"])


def test_lint_verdict_overbought_buy_is_error():
    decision = {"verdict": "BUY", "verdict_type": "Tactical", "constraints": []}
    result = lint_verdict_consistency(
        _report("BUY signal active."),
        _labels(trend_strength="confirmed trend", rsi_zone="overbought"),
        decision,
    )
    assert result["result"] == "ERROR"
    assert any("RSI_OVERBOUGHT_BUY" in e for e in result["errors"])


def test_lint_verdict_clean_buy_confirmed_trend_passes():
    decision = {"verdict": "BUY", "verdict_type": "Tactical", "constraints": []}
    result = lint_verdict_consistency(
        _report("ADX is confirmed; entry conditions are favorable."),
        _labels(trend_strength="confirmed trend", rsi_zone="bullish momentum"),
        decision,
    )
    assert result["result"] == "PASS"


# ═══════════════════════════════════════════════════════════════════════════════
# build_quick_insight — verdict-aware phrases
# ═══════════════════════════════════════════════════════════════════════════════

def test_quick_insight_hold_with_constraint():
    decision = {
        "verdict":      "HOLD",
        "verdict_type": "Tactical",
        "constraints":  ["weak trend + poor timing: BUY blocked"],
    }
    insight = build_quick_insight({"ticker": "MSFT"}, _labels(), decision)
    assert "HOLD" in insight
    assert len(insight) > 20


def test_quick_insight_buy_confirmed_trend():
    decision = {
        "verdict":      "BUY",
        "verdict_type": "Tactical",
        "constraints":  [],
    }
    insight = build_quick_insight(
        {"ticker": "MSFT"},
        _labels(trend_strength="confirmed trend", entry_quality="acceptable entry"),
        decision,
    )
    assert "BUY" in insight
    assert len(insight) > 20


def test_quick_insight_income_hold():
    decision = {
        "verdict":      "HOLD",
        "verdict_type": "Income Hold",
        "constraints":  ["div_yield=5.0% with weak trend: reclassified as Income Hold"],
    }
    insight = build_quick_insight({"ticker": "2222.SR"}, _labels(), decision)
    assert "Income" in insight or "income" in insight
    assert len(insight) > 20


# ═══════════════════════════════════════════════════════════════════════════════
# confidence clamping
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("upside,quality,risk_count", [
    (0.0, 0,   10),   # worst case
    (100.0, 95, 0),   # best case
    (15.0, 65,  2),   # normal
])
def test_confidence_always_clamped(upside, quality, risk_count):
    result = build_decision(
        _labels(trend_strength="confirmed trend", entry_quality="acceptable entry"),
        _score(verdict="BUY", upside_pct=upside, quality=quality, risk_count=risk_count),
    )
    assert 0.30 <= result["confidence"] <= 0.85


# ═══════════════════════════════════════════════════════════════════════════════
# lint_report pipeline integration (Step 5 of wiring task)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_pipeline_snapshot():
    """Minimal frozen ReportSnapshot for pipeline tests."""
    from datetime import datetime
    from core.services.report_snapshot import ReportSnapshot
    ts = datetime.now().isoformat()
    snap = ReportSnapshot({
        "ticker":      {"value": "MSFT",  "source": "fallback",  "timestamp": ts},
        "price":       {"value": 395.0,   "source": "realtime",  "timestamp": ts},
        "entry":       {"value": 382.0,   "source": "calc",      "timestamp": ts},
        "stop":        {"value": 375.0,   "source": "calc",      "timestamp": ts},
        "target":      {"value": 430.0,   "source": "calc",      "timestamp": ts},
        "beta":        {"value": 1.1,     "source": "cache",     "timestamp": ts},
        "pe":          {"value": 28.4,    "source": "cache",     "timestamp": ts},
        "forward_pe":  {"value": 26.2,    "source": "cache",     "timestamp": ts},
        "sma50":       {"value": 401.0,   "source": "calc",      "timestamp": ts},
        "sma200":      {"value": 404.0,   "source": "calc",      "timestamp": ts},
        "week52_high": {"value": 468.0,   "source": "cache",     "timestamp": ts},
        "week52_low":  {"value": 344.0,   "source": "cache",     "timestamp": ts},
        "market_cap":  {"value": 3.1e12,  "source": "cache",     "timestamp": ts},
        "div_yield":   {"value": 0.008,   "source": "cache",     "timestamp": ts},
    })
    snap.freeze()
    return snap


def test_lint_report_blocks_on_verdict_conflict():
    """
    Full pipeline: lint_report with decision=BUY + trend=weak → ERROR → safe_to_render=False.
    This proves lint_verdict_consistency is globally enforced at render time.
    """
    snap = _make_pipeline_snapshot()
    bad_decision = {"verdict": "BUY", "verdict_type": "Tactical", "constraints": []}
    weak_labels  = _labels(trend_strength="weak trend", rsi_zone="neutral momentum")

    report = RenderedReport(
        ticker="MSFT",
        full_text="Setup looks constructive for a BUY.",
        sections=[ReportSection("Memo", "Setup looks constructive for a BUY.")],
        observed_prices=[395.0],
    )
    result = lint_report(
        report, snap,
        decision=bad_decision,
        interpretation_labels=weak_labels,
    )
    assert result.safe_to_render is False, "Render should be blocked on verdict conflict"
    assert any("VERDICT_TREND_CONFLICT" in e for e in result.errors)


def test_lint_report_passes_clean_verdict():
    """
    Full pipeline: lint_report with decision=HOLD + trend=weak → PASS (no block).
    """
    snap = _make_pipeline_snapshot()
    good_decision = {"verdict": "HOLD", "verdict_type": "Tactical", "constraints": []}
    weak_labels   = _labels(trend_strength="weak trend", rsi_zone="neutral momentum")

    report = RenderedReport(
        ticker="MSFT",
        full_text="Awaiting trend confirmation before entry.",
        sections=[ReportSection("Memo", "Awaiting trend confirmation.")],
        observed_prices=[395.0],
    )
    result = lint_report(
        report, snap,
        decision=good_decision,
        interpretation_labels=weak_labels,
    )
    # verdict_consistency passes; other checks may warn but should not block
    verdict_check = next(
        (a for a in result.audit if a.get("check") == "verdict_consistency"), None
    )
    assert verdict_check is not None
    assert verdict_check["result"] == "PASS"


def test_lint_report_skips_verdict_check_without_decision():
    """
    Backward-compat: lint_report called without decision skips verdict check (no crash).
    """
    snap = _make_pipeline_snapshot()
    report = RenderedReport(
        ticker="MSFT",
        full_text="Standard memo.",
        sections=[ReportSection("Memo", "Standard memo.")],
        observed_prices=[395.0],
    )
    result = lint_report(report, snap)  # no decision / labels passed
    skip_check = next(
        (a for a in result.audit if a.get("check") == "verdict_consistency"), None
    )
    assert skip_check is not None
    assert skip_check["result"] == "SKIP"
