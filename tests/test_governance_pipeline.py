"""
Governance Pipeline Test Suite (Phase 3+4+5 hardening).

Guards against regression of the layered architecture:

    TRUTH       → data_guard (read-only)               [not tested here]
    DECISION    → decision_state.DecisionState         [frozen authority]
    GATE        → evidence_router.SectionAllowList
    INTEGRITY   → protected_blocks (protect+restore)
    RENDER      → LLM (not tested here)
    STYLISTIC   → editorial, tone governor, variations  (tone-only mutation)
    OBSERVER    → field_validator, contradiction_scanner verdict-rules (no mutation)
    AUDIT       → ReconciliationAudit (integration log)

Each test is a contract assertion — fast, deterministic, no network/LLM/DB.
"""
from __future__ import annotations

import pytest


# ══════════════════════════════════════════════════════════════════════════════
#  TRUTH / DECISION layer: DecisionState
# ══════════════════════════════════════════════════════════════════════════════
class TestDecisionState:
    def test_canonical_buy_state(self):
        from core.decision_state import build_decision_state
        ds = build_decision_state(
            {"tax_verdict": "Buy", "tax_timing": "Attractive",
             "tax_evidence": "Strong", "tax_risk": "Low",
             "tax_execution": "Scale In", "score": 78, "emoji": "🟢"},
            ticker="NVDA",
        )
        assert ds.verdict == "Buy"
        assert ds.action == "Scale In"
        assert ds.risk == "Low"
        assert ds.evidence == "Strong"
        assert ds.score == 78

    def test_legacy_accumulate_normalizes_to_buy(self):
        from core.decision_state import build_decision_state
        ds = build_decision_state(
            {"verdict": "ACCUMULATE", "conviction": "Medium",
             "score": 65, "emoji": "🟡"},
            ticker="X",
        )
        assert ds.verdict == "Buy"

    def test_frozen_state_is_immutable(self):
        from core.decision_state import build_decision_state
        ds = build_decision_state(
            {"tax_verdict": "Hold", "tax_evidence": "Limited", "score": 50},
            ticker="X",
        )
        with pytest.raises(Exception):
            ds.verdict = "Buy"  # frozen dataclass must reject mutation

    def test_score_clamped(self):
        from core.decision_state import build_decision_state
        ds = build_decision_state(
            {"tax_verdict": "Hold", "score": 9999, "emoji": "🟡"},
            ticker="X",
        )
        assert 0 <= ds.score <= 100


# ══════════════════════════════════════════════════════════════════════════════
#  GATE: evidence_router
# ══════════════════════════════════════════════════════════════════════════════
class TestEvidenceRouter:
    def test_rich_data_enables_premium_sections(self):
        from core.evidence_router import route_evidence
        fund = {
            "pe_ratio": 25, "eps": 5, "beta": 1.1, "revenue": 100e9,
            "net_margin": 20, "gross_margin": 45, "roe": 18, "debt_equity": 40,
            "ebitda": 30e9, "free_cash_flow": 15e9,
            "forward_eps": 6.0, "analyst_target": 200.0,
            "revenue_growth": 12, "earnings_growth": 14,
            "div_yield": 1.2, "payout_ratio": 28, "5y": True,
        }
        peers = [{"ticker": f"P{i}", "pe_ratio": 20+i} for i in range(5)]
        analyst = {"analyst_count": 18, "dc_consensus": "Buy",
                   "next_earnings": "2026-07-01"}
        allow = route_evidence(
            fund=fund, scorecard={}, summary={"adx": 28},
            peers=peers, analyst_data=analyst, ticker="NVDA",
        )
        assert allow.full_fundamental is True
        assert allow.peer_comparison is True
        assert allow.dcf_valuation is True

    def test_adx_low_data_disables_everything(self):
        from core.evidence_router import route_evidence
        allow = route_evidence(
            fund={"pe_ratio": 9.8, "beta": 0.8, "eps": 1.4, "sector": "Utilities"},
            scorecard={}, summary={}, peers=[], analyst_data={},
            ticker="UTIL.AE",
        )
        assert allow.full_fundamental is False
        assert allow.peer_comparison is False
        assert allow.valuation_scenarios is False
        assert allow.dcf_valuation is False
        # reasons must be populated for diagnostics
        assert allow.reasons.get("peer_comparison")

    def test_dcf_requires_fcf_and_growth(self):
        from core.evidence_router import route_evidence
        fund = {"free_cash_flow": 1e9}   # has FCF but no revenue_growth
        allow = route_evidence(
            fund=fund, scorecard={}, summary={}, peers=[],
            analyst_data={}, ticker="X",
        )
        assert allow.dcf_valuation is False


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRITY: protected_blocks
# ══════════════════════════════════════════════════════════════════════════════
class TestProtectedBlocks:
    def test_disclaimer_buy_or_sell_preserved(self):
        from core.protected_blocks import protect, restore
        text = (
            "Some report body.\n\n"
            "> ⚠️ **Disclaimer:** This is not an offer to buy or sell any security.\n\n"
            "Action: Buy candidate.\n"
        )
        protected, spans = protect(text)
        # disclaimer must be sentinelized — "buy or sell" should NOT appear in protected text
        assert "buy or sell any security" not in protected
        # but unprotected body must remain
        assert "Action: Buy candidate." in protected
        # restore round-trip
        restored = restore(protected, spans)
        assert "buy or sell any security" in restored

    def test_url_preserved_through_substitution(self):
        from core.protected_blocks import protect, restore
        import re
        text = "See [link](https://example.com?rating=Buy)"
        protected, spans = protect(text)
        # mid-pipeline substitution that would corrupt URL
        protected = re.sub(r"\bBuy\b", "TAMPERED", protected)
        restored = restore(protected, spans)
        # URL came back intact
        assert "https://example.com?rating=Buy" in restored

    def test_empty_input(self):
        from core.protected_blocks import protect, restore
        out, spans = protect("")
        assert out == ""
        assert spans == []
        assert restore("", spans) == ""


# ══════════════════════════════════════════════════════════════════════════════
#  STYLISTIC: evidence_tone_governor
# ══════════════════════════════════════════════════════════════════════════════
class TestToneGovernor:
    def test_theatrical_phrasing_always_stripped(self):
        from core.evidence_tone_governor import govern_tone
        for evidence in ("Strong", "Moderate", "Limited"):
            res = govern_tone(
                "Thesis Kill Shot: this could destroy the position.",
                evidence=evidence,
            )
            assert "Thesis Kill Shot" not in res.text
            assert res.edits_made >= 1

    def test_limited_strips_precise_target(self):
        from core.evidence_tone_governor import govern_tone
        res = govern_tone(
            "DCF suggests fair value of $4.50 here.",
            evidence="Limited",
        )
        assert "$4.50" not in res.text
        assert "DCF suggests" not in res.text

    def test_limited_downgrades_correctly_pricing(self):
        from core.evidence_tone_governor import govern_tone
        res = govern_tone(
            "The market is correctly pricing peak gas cycle gains.",
            evidence="Limited",
        )
        assert "correctly pricing" not in res.text.lower()

    def test_strong_evidence_keeps_precise_target(self):
        from core.evidence_tone_governor import govern_tone
        res = govern_tone(
            "DCF suggests fair value of $4.50.",
            evidence="Strong",
        )
        # strong evidence may keep precision
        assert "$4.50" in res.text


# ══════════════════════════════════════════════════════════════════════════════
#  OBSERVER: field_validator (Phase 5 consolidation — observer-only)
# ══════════════════════════════════════════════════════════════════════════════
class TestFieldValidator:
    def test_observer_never_mutates_text_even_with_inconsistency(self):
        from core.field_validator import validate_fields
        report = "- **Current Ratio:** unavailable\n- **ROE:** 21.2%\n"
        fund = {"current_ratio": 1.47, "roe": 21.2}
        res = validate_fields(report, fund)
        # text returned UNCHANGED — silent mutation forbidden
        assert res.text == report
        # but inconsistency is detected
        assert res.detected == 1
        # and corrected is always 0 in observer mode
        assert res.corrected == 0
        # fix object exists with the would-be value
        assert res.fixes[0].field == "current_ratio"
        assert res.fixes[0].actual_value == 1.47

    def test_observer_zero_detections_on_legitimate_na(self):
        from core.field_validator import validate_fields
        report = "- **Debt/Equity:** N/A\n"
        fund = {"debt_equity": None}
        res = validate_fields(report, fund)
        # legitimate N/A — no inconsistency recorded
        assert res.text == report
        # detected counts claims, but no fix recorded
        assert len(res.fixes) == 0


# ══════════════════════════════════════════════════════════════════════════════
#  OBSERVER: contradiction_scanner V2 (verdict-rules demoted to flag-only)
# ══════════════════════════════════════════════════════════════════════════════
class TestContradictionScannerObserver:
    def test_reduce_in_hold_report_flag_only_no_mutation(self):
        from core.contradiction_scanner import scan
        report = "Verdict: Hold\nFundamental Verdict: Reduce\nAction: Hold."
        res = scan(report, decision_data={"tax_verdict": "Hold"})
        # rule fires
        rule_ids = {c.rule_id for c in res.contradictions}
        assert "L1_REDUCE_VERDICT_IN_HOLD_REPORT" in rule_ids
        # but body text is UNCHANGED — observer-only
        for c in res.contradictions:
            if c.rule_id == "L1_REDUCE_VERDICT_IN_HOLD_REPORT":
                assert c.auto_fix is None
                assert c.severity == "warn"

    def test_buy_in_hold_report_flag_only_no_mutation(self):
        from core.contradiction_scanner import scan
        report = "Verdict: Hold\nFinal Action: Buy.\nMaintain caution."
        res = scan(report, decision_data={"tax_verdict": "Hold"})
        rule_ids = {c.rule_id for c in res.contradictions}
        assert "L1_BUY_VERDICT_IN_HOLD_REPORT" in rule_ids
        for c in res.contradictions:
            if c.rule_id == "L1_BUY_VERDICT_IN_HOLD_REPORT":
                assert c.auto_fix is None
                assert c.severity == "warn"

    def test_disclaimer_buy_survives_scan(self):
        """The most critical Phase 5 invariant: disclaimer must not be mutated."""
        from core.contradiction_scanner import scan
        report = (
            "Verdict: Hold\n\n"
            "> Disclaimer: This is not an offer to buy or sell any security.\n"
        )
        res = scan(report, decision_data={"tax_verdict": "Hold"})
        # disclaimer body must survive in fixed_text
        assert "buy or sell any security" in res.fixed_text

    def test_high_risk_defensive_style_autofix_preserved(self):
        """Layer 1 STYLE auto-fixes (not semantic) must still apply."""
        from core.contradiction_scanner import scan
        report = "This is a defensive core holding despite elevated volatility."
        res = scan(report, decision_data={"tax_risk": "High"})
        # style auto-fix runs — "defensive" replaced when Risk=High
        assert "defensive" not in res.fixed_text.lower()


# ══════════════════════════════════════════════════════════════════════════════
#  STYLISTIC: editorial.rule_based_clean (integration)
# ══════════════════════════════════════════════════════════════════════════════
class TestEditorial:
    def test_legacy_conviction_label_stripped(self):
        from core.editorial import rule_based_clean
        text = "Verdict: Hold\nConviction: Low. " * 30   # long enough to trigger pass
        out = rule_based_clean(text, ticker="X")
        assert "Conviction: Low" not in out

    def test_uppercase_verdict_normalized(self):
        from core.editorial import rule_based_clean
        text = ("The HOLD recommendation reflects mixed signals. " * 10
                + "ACCUMULATE on dips. " * 10)
        out = rule_based_clean(text, ticker="X")
        # All-caps verdicts get title-cased
        assert "HOLD" not in out
        assert "ACCUMULATE" not in out

    def test_disclaimer_word_buy_preserved_through_pipeline(self):
        """Integration test: full pipeline preserves disclaimer 'buy or sell'."""
        from core.protected_blocks import protect, restore
        from core.editorial import rule_based_clean
        from core.contradiction_scanner import scan
        text = (
            "Verdict: Hold\nAction: Hold Steady.\n"
            "> Disclaimer: This is not an offer to buy or sell any security."
            + ("\n\nNarrative paragraph. " * 30)
        )
        protected, spans = protect(text)
        cleaned = rule_based_clean(protected, ticker="X")
        scan_res = scan(cleaned, decision_data={"tax_verdict": "Hold"})
        restored = restore(scan_res.fixed_text, spans)
        # The disclaimer's "buy or sell" must survive the full pipeline
        assert "buy or sell any security" in restored


# ══════════════════════════════════════════════════════════════════════════════
#  STYLISTIC: sentence_variation + structural_variation (deterministic)
# ══════════════════════════════════════════════════════════════════════════════
class TestVariation:
    def test_sentence_variation_deterministic(self):
        from core.sentence_variation import apply_controlled_variability
        text = "Maintain position — await confirmation. " * 10
        a1 = apply_controlled_variability(text, "X")
        a2 = apply_controlled_variability(text, "X")
        # same ticker → same output (deterministic)
        assert a1 == a2

    def test_structural_variation_idempotent(self):
        from core.structural_variation import apply_structural_variation
        text = (
            "### Executive Summary\n"
            "Strong cash flow remains visible, but the cycle is mature. "
            "Buy because cash is durable. "
            "Strong backlog coverage improves visibility. "
            "Risk remains that capex pressures margins.\n"
        )
        a1 = apply_structural_variation(text, "NVDA")
        a2 = apply_structural_variation(a1, "NVDA")   # idempotent
        assert a1 == a2


# ══════════════════════════════════════════════════════════════════════════════
#  TAXONOMY: canonical mapping (decision_policy)
# ══════════════════════════════════════════════════════════════════════════════
class TestTaxonomy:
    def test_canonical_verdict_normalization(self):
        from core.services.decision_policy import canonical_verdict
        assert canonical_verdict("ACCUMULATE") == "Buy"
        assert canonical_verdict("AVOID") == "Sell"
        assert canonical_verdict("HOLD") == "Hold"

    def test_canonical_evidence_normalization(self):
        from core.services.decision_policy import canonical_evidence
        assert canonical_evidence("Low") == "Limited"
        assert canonical_evidence("Medium") == "Moderate"
        assert canonical_evidence("High") == "Strong"

    def test_canonical_execution_derivation(self):
        from core.services.decision_policy import canonical_execution
        assert canonical_execution("Buy", "Attractive") == "Scale In"
        assert canonical_execution("Hold", "Neutral") == "Hold Steady"
        assert canonical_execution("Reduce", "Extended") == "Reduce Exposure"
        assert canonical_execution("Sell", "Extended") == "Reduce Exposure"
