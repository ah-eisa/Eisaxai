"""
test_portfolio_analytics.py — Institutional portfolio diagnostics tests.

Covers:
  1. Effective N correctness (known-weight cases)
  2. Worst-case drawdown extraction
  3. Conditional-approval gate trigger
  4. Economic buckets sum to ~100 %
  5. Bucket concentration warning
  6. Diversification label thresholds
  7. Sharpe context note — triggers only when both conditions hit
  8. Consistent diversification phrase — replaces mis-matched language
"""
from __future__ import annotations

import math

import pytest

from core.services.portfolio_analytics import (
    BUCKET_ORDER,
    bucket_concentration_warning,
    classify_bucket,
    compute_economic_buckets,
    compute_effective_n,
    compute_worst_case_drawdown,
    consistent_diversification_phrase,
    diversification_label,
    diversification_soft_suggestion,
    estimate_worst_case_from_vol,
    readiness_with_drawdown,
    sharpe_context_note,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Effective N
# ─────────────────────────────────────────────────────────────────────────────

class TestEffectiveN:

    def test_equal_weights_10_positions(self):
        weights = [0.1] * 10
        assert compute_effective_n(weights) == pytest.approx(10.0, rel=1e-6)

    def test_equal_weights_5_positions(self):
        weights = [0.2] * 5
        assert compute_effective_n(weights) == pytest.approx(5.0, rel=1e-6)

    def test_single_position_returns_one(self):
        assert compute_effective_n([1.0]) == pytest.approx(1.0)

    def test_concentrated_portfolio(self):
        # 80% in one name, 5% each in 4 others → very concentrated
        weights = [0.80, 0.05, 0.05, 0.05, 0.05]
        n = compute_effective_n(weights)
        assert n < 2.0

    def test_empty_weights_returns_zero(self):
        assert compute_effective_n([]) == 0.0

    def test_self_normalises_percents(self):
        # Passing percents (sum = 100) must yield the same N as fractions
        n_frac = compute_effective_n([0.25, 0.25, 0.25, 0.25])
        n_pct  = compute_effective_n([25, 25, 25, 25])
        assert n_frac == pytest.approx(n_pct, rel=1e-6)

    def test_ignores_zero_and_none(self):
        assert compute_effective_n([0.5, 0.5, 0, None]) == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Worst-case drawdown
# ─────────────────────────────────────────────────────────────────────────────

class TestWorstCase:

    def test_returns_minimum_of_scenarios(self):
        # -30, -25, -40 → -40 (most negative)
        assert compute_worst_case_drawdown([-0.30, -0.25, -0.40]) == pytest.approx(-0.40)

    def test_empty_input_returns_none(self):
        assert compute_worst_case_drawdown([]) is None

    def test_handles_mixed_positive_negative(self):
        assert compute_worst_case_drawdown([0.05, -0.10, 0.20, -0.35]) == pytest.approx(-0.35)

    def test_ignores_none_entries(self):
        assert compute_worst_case_drawdown([None, -0.10, None]) == pytest.approx(-0.10)

    def test_ignores_non_numeric(self):
        assert compute_worst_case_drawdown(["oops", -0.15]) == pytest.approx(-0.15)

    def test_vol_based_estimate_negative(self):
        # 20% vol with 2-σ fallback → -40%
        assert estimate_worst_case_from_vol(0.20, z=2.0) == pytest.approx(-0.40)

    def test_vol_based_estimate_zero_vol(self):
        assert estimate_worst_case_from_vol(0.0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Conditional-approval gate
# ─────────────────────────────────────────────────────────────────────────────

class TestReadinessGate:

    def test_worst_below_target_preserves_status(self):
        v = readiness_with_drawdown("✅ APPROVED", worst_case=-0.20, target_drawdown=0.25)
        assert v.status == "✅ APPROVED"
        assert v.breaches_drawdown is False
        assert v.note is None

    def test_worst_above_target_flags_conditional(self):
        v = readiness_with_drawdown("✅ APPROVED", worst_case=-0.40, target_drawdown=0.25)
        assert v.status == "⚠️ CONDITIONAL APPROVAL"
        assert v.breaches_drawdown is True
        assert v.note is not None
        assert "exceeds" in v.note.lower()

    def test_exact_target_does_not_trigger(self):
        v = readiness_with_drawdown("✅ APPROVED", worst_case=-0.25, target_drawdown=0.25)
        assert v.breaches_drawdown is False

    def test_none_worst_case_preserves(self):
        v = readiness_with_drawdown("⚠️ CONDITIONAL", worst_case=None, target_drawdown=0.25)
        assert v.status == "⚠️ CONDITIONAL"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Economic buckets
# ─────────────────────────────────────────────────────────────────────────────

class TestEconomicBuckets:

    def test_classify_growth_beta_by_ticker(self):
        assert classify_bucket("NVDA") == "Growth Beta"
        assert classify_bucket("BTC-USD") == "Growth Beta"

    def test_classify_commodity_cycle(self):
        assert classify_bucket("GSG") == "Commodity Cycle"
        assert classify_bucket("XOM") == "Commodity Cycle"

    def test_classify_defensive(self):
        assert classify_bucket("GLD") == "Defensive"
        assert classify_bucket("TLT") == "Defensive"

    def test_classify_regional_beta_via_suffix(self):
        assert classify_bucket("2222.SR") == "Commodity Cycle"  # explicit ticker wins
        assert classify_bucket("EMAAR.AE") == "Regional Beta"

    def test_classify_falls_back_to_other(self):
        assert classify_bucket("XYZ999") == "Other"

    def test_buckets_sum_to_100(self):
        positions = [
            {"ticker": "NVDA", "weight": 0.25},
            {"ticker": "GLD",  "weight": 0.25},
            {"ticker": "GSG",  "weight": 0.25},
            {"ticker": "EMAAR.AE", "weight": 0.25},
        ]
        buckets = compute_economic_buckets(positions)
        assert math.isclose(sum(buckets.values()), 100.0, abs_tol=0.5)
        assert "Growth Beta" in buckets
        assert "Defensive" in buckets
        assert "Commodity Cycle" in buckets
        assert "Regional Beta" in buckets

    def test_buckets_normalise_non_unit_sum(self):
        positions = [
            {"ticker": "NVDA", "weight": 50},   # percent inputs
            {"ticker": "GLD",  "weight": 50},
        ]
        buckets = compute_economic_buckets(positions)
        assert math.isclose(sum(buckets.values()), 100.0, abs_tol=0.5)

    def test_concentration_warning_triggers_above_50(self):
        buckets = {"Growth Beta": 65.0, "Defensive": 35.0}
        warn = bucket_concentration_warning(buckets, threshold=50.0)
        assert warn is not None
        assert "Growth Beta" in warn

    def test_concentration_warning_quiet_when_balanced(self):
        buckets = {"Growth Beta": 40.0, "Defensive": 30.0, "Commodity Cycle": 30.0}
        assert bucket_concentration_warning(buckets) is None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Diversification label thresholds
# ─────────────────────────────────────────────────────────────────────────────

class TestDiversificationLabel:

    def test_under_5_is_concentrated(self):
        assert diversification_label(4.9) == "Highly concentrated"
        assert diversification_label(1.0) == "Highly concentrated"

    def test_between_5_and_10_is_moderate(self):
        assert diversification_label(5.0) == "Moderately diversified"
        assert diversification_label(7.5) == "Moderately diversified"
        assert diversification_label(10.0) == "Moderately diversified"

    def test_above_10_is_well(self):
        assert diversification_label(10.1) == "Well diversified"
        assert diversification_label(15.0) == "Well diversified"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Sharpe context note
# ─────────────────────────────────────────────────────────────────────────────

class TestSharpeContext:

    def test_low_sharpe_high_vol_triggers(self):
        note = sharpe_context_note(sharpe=0.3, volatility=0.25)
        assert note is not None
        assert "Sharpe" in note

    def test_high_sharpe_silent(self):
        assert sharpe_context_note(sharpe=1.2, volatility=0.25) is None

    def test_low_vol_silent(self):
        assert sharpe_context_note(sharpe=0.3, volatility=0.10) is None

    def test_none_inputs_silent(self):
        assert sharpe_context_note(None, 0.25) is None
        assert sharpe_context_note(0.3, None) is None


# ─────────────────────────────────────────────────────────────────────────────
# 7. Consistent diversification phrase
# ─────────────────────────────────────────────────────────────────────────────

class TestConsistentPhrase:

    def test_rewrites_well_diversified_when_n_low(self):
        out = consistent_diversification_phrase(3.0, "portfolio is well diversified across sectors")
        assert "well diversified" not in out.lower()
        assert "concentrated high-conviction" in out.lower()

    def test_preserves_when_n_adequate(self):
        src = "portfolio is well diversified"
        assert consistent_diversification_phrase(12.0, src) == src

    def test_preserves_unrelated_phrase(self):
        src = "moderately diversified"
        assert consistent_diversification_phrase(3.0, src) == src


# ─────────────────────────────────────────────────────────────────────────────
# 8. Soft diversification suggestion
# ─────────────────────────────────────────────────────────────────────────────

class TestSoftSuggestion:

    def test_triggers_on_low_effective_n(self):
        note = diversification_soft_suggestion(3.0, {"Growth Beta": 40, "Defensive": 60})
        assert note is not None
        assert "suggestion" in note.lower()

    def test_triggers_on_high_bucket(self):
        note = diversification_soft_suggestion(8.0, {"Growth Beta": 70, "Defensive": 30})
        assert note is not None

    def test_silent_when_healthy(self):
        note = diversification_soft_suggestion(
            10.0, {"Growth Beta": 40, "Defensive": 30, "Commodity Cycle": 30},
        )
        assert note is None


# ─────────────────────────────────────────────────────────────────────────────
# 9. BUCKET_ORDER invariant
# ─────────────────────────────────────────────────────────────────────────────

def test_bucket_order_contains_all_known_buckets():
    assert "Growth Beta" in BUCKET_ORDER
    assert "Commodity Cycle" in BUCKET_ORDER
    assert "Defensive" in BUCKET_ORDER
    assert "Regional Beta" in BUCKET_ORDER
    assert "Other" in BUCKET_ORDER
