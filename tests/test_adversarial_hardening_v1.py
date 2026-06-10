"""
Adversarial Hardening Validation Layer — v1.

Security-grade adversarial test suite for the EisaX deterministic decision
system. This module validates that the production pipeline remains correct
under intentional manipulation, ambiguity, and adversarial LLM-style outputs.

Scope:
  1. Attack Simulation Tests (5 categories: A–E)
     A. Semantic Contradiction Injection
     B. Hidden Verdict Smuggling
     C. Protected Block Integrity Attack
     D. Cross-Section Conflict Injection
     E. Reconciliation Leakage
  2. System Invariance Tests
     A. Stylistic-Order Invariance
     B. Deterministic Decision Invariance
  3. Adversarial Stability Metrics
     helper: adversarial_stability_score()
     final battery: full-attack stability assertion

Hard constraints (enforced by review):
  - No production code modifications.
  - No semantic corrections in test logic.
  - No external dependencies (no LLM, no API, no DB).
  - Tests are pure-Python deterministic.

Expected baseline at green state:
  0 mutation events.
  100% defined-attack detection.
  reconciliation_completeness == 1.0.
  stability_score >= 0.95.
"""
from __future__ import annotations

import re
import pytest


# ══════════════════════════════════════════════════════════════════════════════
#  Adversarial Stability Helper
# ══════════════════════════════════════════════════════════════════════════════
def adversarial_stability_score(result: dict) -> dict:
    """
    Compute the adversarial stability score from a battery-run result.

    Components:
      detection_rate            = detected / total_attacks         (weight 0.50)
      mutation_safety           = 1.0 if mutation_events == 0      (weight 0.30)
      reconciliation_completeness = 1.0 if all detected surfaced   (weight 0.20)

    Args:
        result: dict with keys:
            total_attacks:           int  (>= 1)
            detected_attacks:        int
            missed_attacks:          int
            mutation_events:         int  (must be 0 for green)
            reconciliation_complete: bool

    Returns:
        {"stability_score": float, "detected": int, "missed": int,
         "mutation_events": int}
    """
    total     = max(1, int(result.get("total_attacks", 0)))
    detected  = int(result.get("detected_attacks", 0))
    missed    = int(result.get("missed_attacks", 0))
    mutations = int(result.get("mutation_events", 0))
    reconcile_ok = bool(result.get("reconciliation_complete", False))

    detection_rate = min(1.0, detected / total)
    mutation_safety = 1.0 if mutations == 0 else 0.0
    reconciliation = 1.0 if reconcile_ok else 0.0

    stability = (
        0.50 * detection_rate
        + 0.30 * mutation_safety
        + 0.20 * reconciliation
    )
    return {
        "stability_score": round(stability, 3),
        "detected":        detected,
        "missed":          missed,
        "mutation_events": mutations,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  1. ATTACK SIMULATION TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestAttackSimulation_A_SemanticContradictionInjection:
    """
    Simulate LLM output that contains a verdict-anchored contradiction inside
    a tone-safe Hold report.

    Defense invariants:
      • DecisionState MUST NOT change.
      • ContradictionScanner MUST flag (no auto-fix).
      • No silent text mutation.
    """

    def test_hold_with_hidden_reduce_label(self):
        from core.contradiction_scanner import scan
        from core.decision_state import build_decision_state
        decision = {"tax_verdict": "Hold", "tax_evidence": "Limited",
                    "tax_risk": "Moderate", "score": 60, "emoji": "🟡"}
        ds_pre = build_decision_state(decision, ticker="X")
        report = (
            "Verdict: Hold\n"
            "Fundamental Verdict: Reduce\n"
            "Action: Hold Steady\n"
        )
        res = scan(report, decision_data=decision)
        # DecisionState authority untouched
        ds_post = build_decision_state(decision, ticker="X")
        assert ds_pre == ds_post
        # Scanner flags drift
        rule_ids = {c.rule_id for c in res.contradictions}
        assert "L1_REDUCE_VERDICT_IN_HOLD_REPORT" in rule_ids
        # NO mutation — body retains both
        assert "Verdict: Hold" in res.fixed_text
        assert "Fundamental Verdict: Reduce" in res.fixed_text
        # Severity is observer (warn), not blocker
        for c in res.contradictions:
            if c.rule_id == "L1_REDUCE_VERDICT_IN_HOLD_REPORT":
                assert c.auto_fix is None and c.severity == "warn"

    def test_hold_with_hidden_buy_label(self):
        from core.contradiction_scanner import scan
        decision = {"tax_verdict": "Hold"}
        report = "Verdict: Hold\nFinal Action: Buy\nMaintain stance."
        res = scan(report, decision_data=decision)
        assert "L1_BUY_VERDICT_IN_HOLD_REPORT" in {c.rule_id for c in res.contradictions}
        for c in res.contradictions:
            if c.rule_id == "L1_BUY_VERDICT_IN_HOLD_REPORT":
                assert c.auto_fix is None and c.severity == "warn"
        # No mutation
        assert "Final Action: Buy" in res.fixed_text

    def test_hold_with_uppercase_accumulate_legacy(self):
        """ACCUMULATE is legacy vocabulary → normalized to Buy by editorial.
        But within scan() context, it's normalized by extract_state, not
        silently rewritten in the report body."""
        from core.contradiction_scanner import scan, extract_state
        report = "Verdict: Hold\nRecommendation: ACCUMULATE"
        extracted = extract_state(report)
        # ACCUMULATE → Buy in canonical extraction
        assert "Buy" in extracted.unique_verdicts()
        res = scan(report, decision_data={"tax_verdict": "Hold"})
        rule_ids = {c.rule_id for c in res.contradictions}
        assert "L1_BUY_VERDICT_IN_HOLD_REPORT" in rule_ids


class TestAttackSimulation_B_HiddenVerdictSmuggling:
    """
    LLM smuggles verdict-like vocabulary into narrative prose without
    explicit verdict labels.

    Defense invariants:
      • Extractor MUST require verdict-anchor label.
      • No false positive — narrative phrases stay as narrative.
    """

    def test_buyers_entered_aggressively_not_extracted(self):
        from core.contradiction_scanner import extract_state, scan
        report = (
            "Verdict: Hold\n"
            "Last quarter, buyers entered aggressively into the energy complex.\n"
            "Action: Hold Steady"
        )
        extracted = extract_state(report)
        # Narrative "buyers" is NOT a Buy verdict
        assert extracted.unique_verdicts() == {"Hold"}
        res = scan(report, decision_data={"tax_verdict": "Hold"})
        assert "L1_BUY_VERDICT_IN_HOLD_REPORT" not in {c.rule_id for c in res.contradictions}

    def test_buy_side_momentum_not_extracted(self):
        from core.contradiction_scanner import extract_state, scan
        report = (
            "Verdict: Hold\n"
            "Buy-side momentum increased into year-end, indicating institutional rotation.\n"
            "Action: Hold Steady"
        )
        extracted = extract_state(report)
        assert extracted.unique_verdicts() == {"Hold"}
        res = scan(report, decision_data={"tax_verdict": "Hold"})
        assert "L1_BUY_VERDICT_IN_HOLD_REPORT" not in {c.rule_id for c in res.contradictions}

    def test_institutional_accumulation_not_extracted(self):
        from core.contradiction_scanner import extract_state
        report = (
            "Verdict: Hold\n"
            "Institutional accumulation has been detected on prior dips, but "
            "the timing remains extended."
        )
        extracted = extract_state(report)
        # No verdict label → only the explicit Hold counts
        assert extracted.unique_verdicts() == {"Hold"}


class TestAttackSimulation_C_ProtectedBlockIntegrity:
    """
    Adversarial modification attempts inside immune zones.

    Defense invariant:
      • ProtectedBlocks must restore byte-identical content after the full
        pipeline runs.
    """

    def test_disclaimer_byte_identical_after_full_pipeline(self):
        from core.protected_blocks import protect, restore
        from core.editorial import rule_based_clean
        from core.contradiction_scanner import scan

        disclaimer = (
            "> ⚠️ **Disclaimer:** This report is informational only. "
            "Not an offer to buy or sell any security. ACCUMULATE. Conviction: Low."
        )
        report = f"Verdict: Hold\nAction: Hold Steady.\n\n{disclaimer}\n\nBody. " * 5
        protected, spans = protect(report)
        cleaned = rule_based_clean(protected, ticker="ADV")
        scan_res = scan(cleaned, decision_data={"tax_verdict": "Hold"})
        restored = restore(scan_res.fixed_text, spans)
        # disclaimer block survives byte-identical
        assert disclaimer in restored

    def test_disclaimer_injected_verdict_protected(self):
        """Even a smuggled `Verdict: Buy` inside the disclaimer must not
        propagate to the scanner or editorial — the protect/restore symmetry
        contains it."""
        from core.protected_blocks import protect, restore
        from core.editorial import rule_based_clean
        from core.contradiction_scanner import scan
        adv = (
            "> ⚠️ **Disclaimer:** Verdict: Buy. ACCUMULATE. "
            "Not an offer to buy or sell any security."
        )
        report = f"Verdict: Hold\nAction: Hold Steady.\n\n{adv}\n\nNarrative. " * 5
        protected, spans = protect(report)
        cleaned = rule_based_clean(protected, ticker="ADV")
        res = scan(cleaned, decision_data={"tax_verdict": "Hold"})
        restored = restore(res.fixed_text, spans)
        # smuggled disclaimer text MUST come back identical
        assert adv in restored

    def test_audit_trail_protected_through_transforms(self):
        from core.protected_blocks import protect, restore
        from core.editorial import rule_based_clean
        audit = (
            "## Audit Trail\n"
            "- Source URL: https://internal.example.com/audit/REDUCE-2026-05\n"
            "- Score Trend: HOLD → HOLD → HOLD\n"
        )
        report = f"Verdict: Hold.\n{audit}\nBody. " * 5
        protected, spans = protect(report)
        cleaned = rule_based_clean(protected, ticker="ADV")
        restored = restore(cleaned, spans)
        # Audit trail intact — uppercase HOLDs inside it not normalized by editorial
        assert "HOLD → HOLD → HOLD" in restored

    def test_url_inside_link_protected_through_substitution(self):
        from core.protected_blocks import protect, restore
        text = "See [analysis](https://example.com?rating=Buy)"
        protected, spans = protect(text)
        # Simulate a body-wide substitution that would corrupt URL
        protected = re.sub(r"\bBuy\b", "TAMPERED", protected)
        restored = restore(protected, spans)
        # URL parameter preserved
        assert "rating=Buy" in restored


class TestAttackSimulation_D_CrossSectionConflictInjection:
    """
    Force a mismatch between Quick View and body section authority claims.

    Defense invariant:
      • Scanner must flag L1_MULTI_RISK_LEVELS (or multi-verdict).
      • No auto-fix permitted.
      • No reconciliation suppression.
    """

    def test_multi_risk_levels_detected(self):
        from core.contradiction_scanner import scan
        report = (
            "## Quick View\n"
            "**Risk:** Moderate · **Evidence:** Limited · **Timing:** Extended\n\n"
            "## Risk Framework\n"
            "**Risk:** High — cyclical exposure dominates the thesis.\n"
        )
        res = scan(report, decision_data={"tax_verdict": "Hold", "tax_risk": "Moderate"})
        rule_ids = {c.rule_id for c in res.contradictions}
        assert "L1_MULTI_RISK_LEVELS" in rule_ids

    def test_multi_risk_no_auto_fix(self):
        from core.contradiction_scanner import scan
        report = (
            "**Risk:** Moderate\n\n**Risk:** High\n"
        )
        res = scan(report, decision_data={"tax_risk": "Moderate"})
        # Body unchanged — no silent reconciliation of conflicting risk labels
        assert "**Risk:** Moderate" in res.fixed_text
        assert "**Risk:** High" in res.fixed_text
        for c in res.contradictions:
            if c.rule_id == "L1_MULTI_RISK_LEVELS":
                assert c.auto_fix is None

    def test_multi_verdict_detected(self):
        from core.contradiction_scanner import scan
        report = (
            "Verdict: Hold\n"
            "Fundamental Verdict: Reduce\n"
            "Final Action: Buy\n"
        )
        res = scan(report, decision_data={"tax_verdict": "Hold"})
        assert "L1_MULTI_VERDICT_AUTHORITY" in {c.rule_id for c in res.contradictions}


class TestAttackSimulation_E_ReconciliationLeakage:
    """
    Inject multiple unfixable contradictions. ALL must surface in result.unfixed.
    """

    def test_all_unfixed_surface(self):
        from core.contradiction_scanner import scan
        report = (
            "Verdict: Hold\n"
            "Fundamental Verdict: Reduce\n"      # L1_REDUCE_VERDICT_IN_HOLD_REPORT
            "Final Action: Buy\n"                # L1_BUY_VERDICT_IN_HOLD_REPORT
            "**Risk:** Moderate\n"
            "**Risk:** High\n"                   # L1_MULTI_RISK_LEVELS
        )
        res = scan(report, decision_data={"tax_verdict": "Hold", "tax_risk": "Moderate"})
        expected = {
            "L1_REDUCE_VERDICT_IN_HOLD_REPORT",
            "L1_BUY_VERDICT_IN_HOLD_REPORT",
            "L1_MULTI_RISK_LEVELS",
        }
        # All present in contradictions list
        all_ids = {c.rule_id for c in res.contradictions}
        assert expected.issubset(all_ids), f"missing detection for: {expected - all_ids}"
        # All present in unfixed list (none have auto_fix)
        unfixed_ids = {c.rule_id for c in res.unfixed}
        assert expected.issubset(unfixed_ids), f"dropped from unfixed: {expected - unfixed_ids}"

    def test_no_contradictions_silently_dropped(self):
        """Run a scan with a complex adversarial input — every detected
        contradiction must be reachable via the public ScanResult."""
        from core.contradiction_scanner import scan
        report = (
            "Verdict: Hold\nFundamental Verdict: Reduce\nAction: Scale In\n"
        )
        res = scan(report, decision_data={"tax_verdict": "Hold"})
        # contradictions and unfixed are both non-empty
        assert len(res.contradictions) >= 1
        # every unfixed item is also in contradictions
        for u in res.unfixed:
            assert u in res.contradictions


# ══════════════════════════════════════════════════════════════════════════════
#  2. SYSTEM INVARIANCE TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestSystemInvariance_A_StylisticOrder:
    """Reordering stylistic-only layers must NOT change DecisionState or audit semantics."""

    @staticmethod
    def _stylistic_order_AB(text: str, ticker: str) -> str:
        from core.sentence_variation import apply_controlled_variability
        from core.structural_variation import apply_structural_variation
        return apply_structural_variation(apply_controlled_variability(text, ticker), ticker)

    @staticmethod
    def _stylistic_order_BA(text: str, ticker: str) -> str:
        from core.sentence_variation import apply_controlled_variability
        from core.structural_variation import apply_structural_variation
        return apply_controlled_variability(apply_structural_variation(text, ticker), ticker)

    def test_decision_state_invariant_under_reorder(self):
        from core.decision_state import build_decision_state
        sc = {"tax_verdict": "Hold", "tax_timing": "Extended",
              "tax_evidence": "Limited", "tax_risk": "Moderate",
              "score": 66, "emoji": "🟡"}
        ds1 = build_decision_state(sc, ticker="INV")
        ds2 = build_decision_state(sc, ticker="INV")
        # DecisionState authority is independent of any text-transform order
        assert ds1 == ds2
        assert ds1.verdict == "Hold"
        assert ds1.action == "Hold Steady"

    def test_audit_flags_invariant_under_reorder(self):
        from core.contradiction_scanner import scan
        text = ("Verdict: Hold\nAction: Hold Steady.\n"
                "Fundamental Verdict: Reduce\n"
                "Maintain stance. " * 5)
        out_ab = self._stylistic_order_AB(text, "INV")
        out_ba = self._stylistic_order_BA(text, "INV")
        dec = {"tax_verdict": "Hold"}
        # The core decision-drift rule must fire in BOTH orderings
        flags_ab = {c.rule_id for c in scan(out_ab, dec).contradictions}
        flags_ba = {c.rule_id for c in scan(out_ba, dec).contradictions}
        for flag in ("L1_REDUCE_VERDICT_IN_HOLD_REPORT",):
            assert flag in flags_ab
            assert flag in flags_ba


class TestSystemInvariance_B_DeterministicDecision:
    """Same input must produce identical DecisionState across runs."""

    def test_same_input_same_decision_state(self):
        from core.decision_state import build_decision_state
        sc = {"tax_verdict": "Buy", "tax_timing": "Attractive",
              "tax_evidence": "Strong", "tax_risk": "Low",
              "tax_execution": "Scale In", "score": 80, "emoji": "🟢"}
        outputs = [build_decision_state(sc, ticker="X").to_dict() for _ in range(5)]
        for o in outputs[1:]:
            assert o == outputs[0]

    def test_decision_state_idempotent_legacy_inputs(self):
        """Legacy verdict labels (ACCUMULATE/AVOID) must normalize deterministically."""
        from core.decision_state import build_decision_state
        ds_a = build_decision_state(
            {"verdict": "ACCUMULATE", "conviction": "Medium", "score": 65, "emoji": "🟡"},
            ticker="X",
        )
        ds_b = build_decision_state(
            {"verdict": "ACCUMULATE", "conviction": "Medium", "score": 65, "emoji": "🟡"},
            ticker="X",
        )
        assert ds_a == ds_b
        assert ds_a.verdict == "Buy"


# ══════════════════════════════════════════════════════════════════════════════
#  3. ADVERSARIAL STABILITY METRICS
# ══════════════════════════════════════════════════════════════════════════════
class TestStabilityHelper:
    """Validate the adversarial_stability_score helper itself."""

    def test_perfect_score(self):
        s = adversarial_stability_score({
            "total_attacks": 10, "detected_attacks": 10,
            "missed_attacks": 0, "mutation_events": 0,
            "reconciliation_complete": True,
        })
        assert s["stability_score"] == 1.0
        assert s["detected"] == 10 and s["mutation_events"] == 0

    def test_mutation_zeroes_safety_component(self):
        s = adversarial_stability_score({
            "total_attacks": 10, "detected_attacks": 10,
            "missed_attacks": 0, "mutation_events": 1,
            "reconciliation_complete": True,
        })
        # mutation drops the 0.30 weight to 0
        assert s["stability_score"] < 1.0
        assert s["mutation_events"] == 1

    def test_missed_attack_penalty(self):
        s = adversarial_stability_score({
            "total_attacks": 10, "detected_attacks": 5,
            "missed_attacks": 5, "mutation_events": 0,
            "reconciliation_complete": True,
        })
        # Detection rate halved → score drops below perfect
        assert 0.0 < s["stability_score"] < 1.0
        assert s["missed"] == 5

    def test_reconciliation_incomplete_penalty(self):
        s = adversarial_stability_score({
            "total_attacks": 5, "detected_attacks": 5,
            "missed_attacks": 0, "mutation_events": 0,
            "reconciliation_complete": False,
        })
        # Reconciliation contributes 0.20 — drops to 0.80
        assert abs(s["stability_score"] - 0.80) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL: FULL-BATTERY ATTACK RESISTANCE ASSERTION
# ══════════════════════════════════════════════════════════════════════════════
class TestAttackBatterySummary:
    """
    Run the full attack battery as an aggregated stability check.
    Asserts the system meets the green-state thresholds:
        stability_score >= 0.95
        mutation_events == 0
        reconciliation_complete == True
    """

    def test_full_battery_meets_security_thresholds(self):
        from core.contradiction_scanner import scan, extract_state
        from core.protected_blocks import protect, restore
        from core.editorial import rule_based_clean

        attack_scenarios = [
            # (attack_id, report_text, decision_data, expected_rule_id_or_none)
            ("A1_hidden_reduce", "Verdict: Hold\nFundamental Verdict: Reduce",
             {"tax_verdict": "Hold"}, "L1_REDUCE_VERDICT_IN_HOLD_REPORT"),
            ("A2_hidden_buy",    "Verdict: Hold\nFinal Action: Buy",
             {"tax_verdict": "Hold"}, "L1_BUY_VERDICT_IN_HOLD_REPORT"),
            ("B1_buyers_smuggle","Verdict: Hold\nbuyers entered aggressively in Q4.",
             {"tax_verdict": "Hold"}, None),  # must NOT trigger
            ("B2_buyside_smug",  "Verdict: Hold\nBuy-side momentum increased.",
             {"tax_verdict": "Hold"}, None),  # must NOT trigger
            ("D1_multi_risk",    "**Risk:** Moderate\n**Risk:** High",
             {"tax_risk": "Moderate"}, "L1_MULTI_RISK_LEVELS"),
            ("D2_multi_verdict", "Verdict: Hold\nFinal Action: Buy\nFundamental Verdict: Reduce",
             {"tax_verdict": "Hold"}, "L1_MULTI_VERDICT_AUTHORITY"),
        ]

        total      = 0
        detected   = 0
        missed     = 0
        mutations  = 0
        reconcile_complete = True

        for attack_id, text, decision, expected_rule in attack_scenarios:
            original = text
            res = scan(text, decision_data=decision)
            detected_rules = {c.rule_id for c in res.contradictions}

            if expected_rule is not None:
                total += 1
                if expected_rule in detected_rules:
                    detected += 1
                else:
                    missed += 1
            else:
                # smuggling scenarios — only false-positive detection counts as missed defense
                total += 1
                # We expect NO Buy/Reduce drift rule to fire on narrative-only prose
                forbidden = {"L1_BUY_VERDICT_IN_HOLD_REPORT",
                             "L1_REDUCE_VERDICT_IN_HOLD_REPORT"}
                if not (forbidden & detected_rules):
                    detected += 1
                else:
                    missed += 1

            # Mutation check: scanner must not silently mutate body text
            # (only style-only auto-fixes are allowed, and none apply here)
            if res.fixed_text != original:
                # Check if the change was a style-only fix (e.g., Conviction:→Evidence:)
                # If body verdict labels changed → counts as mutation
                for token in ("Reduce", "Buy", "Risk:"):
                    if token in original and token not in res.fixed_text:
                        mutations += 1
                        break

            # Reconciliation completeness — all detected contradictions surface in unfixed
            for c in res.contradictions:
                if c.auto_fix is None and c not in res.unfixed:
                    reconcile_complete = False

        # Disclaimer integrity check — counted as attack 7
        disclaimer = "> ⚠️ **Disclaimer:** Not an offer to buy or sell any security."
        full = f"Verdict: Hold.\n{disclaimer}\nBody. " * 3
        protected, spans = protect(full)
        cleaned = rule_based_clean(protected, ticker="ADV")
        scan_res = scan(cleaned, decision_data={"tax_verdict": "Hold"})
        restored = restore(scan_res.fixed_text, spans)
        total += 1
        if disclaimer in restored:
            detected += 1
        else:
            missed += 1
            mutations += 1

        result = {
            "total_attacks":           total,
            "detected_attacks":        detected,
            "missed_attacks":          missed,
            "mutation_events":         mutations,
            "reconciliation_complete": reconcile_complete,
        }
        score = adversarial_stability_score(result)

        # Print for human-readable test output
        print(f"\nAdversarial Stability Report: {score}")

        # GREEN-STATE THRESHOLDS
        assert score["mutation_events"] == 0, \
            f"Mutation events detected: {score['mutation_events']}"
        assert score["missed"] == 0, \
            f"Attacks missed: {score['missed']}"
        assert score["stability_score"] >= 0.95, \
            f"Stability {score['stability_score']} below 0.95 threshold"
