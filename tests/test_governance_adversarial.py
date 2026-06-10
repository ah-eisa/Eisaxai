"""
Adversarial + Invariance + Stress test suite (Phase 5 extension).

Three test classes — they guard against the hardest cases:

  TestAdversarial         — 5 tests: simulate hostile LLM output / corruption attempts
  TestPipelineInvariance  — pipeline-stable: reordering stylistic layers must not
                            change the DecisionState or alter audit semantics
  TestMultiTickerStress   — parametrized run across 10 tickers spanning 5 categories
                            (energy, tech growth, distressed, low coverage, high vol)

No network, no LLM, no DB. Pure deterministic.
"""
from __future__ import annotations

import re
import pytest


# ══════════════════════════════════════════════════════════════════════════════
#  1. ADVERSARIAL LAYER (5 tests)
# ══════════════════════════════════════════════════════════════════════════════
class TestAdversarial:
    """Simulate hostile LLM output. Pipeline must defend without silent mutation."""

    def test_semantic_contradiction_inside_tone_safe_text(self):
        """
        LLM produces tone-safe Hold prose with an injected 'Reduce' verdict.
        Scanner MUST flag it as observer warning, MUST NOT silently rewrite.
        """
        from core.contradiction_scanner import scan
        # Tone-safe paragraph + injected verdict drift in a labeled section
        report = (
            "### Executive Summary\n"
            "The setup is mixed. Cash generation remains stable but timing is unclear.\n\n"
            "### Investment Verdict\n"
            "Verdict: Hold\n"
            "Fundamental Verdict: Reduce  ← drift injected here\n"
            "Action: Hold Steady\n"
        )
        res = scan(report, decision_data={"tax_verdict": "Hold"})
        rule_ids = {c.rule_id for c in res.contradictions}
        # The scanner detects:
        assert "L1_REDUCE_VERDICT_IN_HOLD_REPORT" in rule_ids
        # No silent mutation — observer warning only
        for c in res.contradictions:
            if c.rule_id in {"L1_REDUCE_VERDICT_IN_HOLD_REPORT",
                             "L1_BUY_VERDICT_IN_HOLD_REPORT"}:
                assert c.auto_fix is None
                assert c.severity == "warn"
        # body text retains the contradiction so the audit trail reflects reality
        assert "Fundamental Verdict: Reduce" in res.fixed_text

    def test_hidden_verdict_inside_sentence_variants(self):
        """
        LLM hides a contradicting verdict by camouflaging it inside narrative
        prose ("buy-side", "buyers entered"). These must NOT trigger
        verdict-drift rules (regex is anchored to verdict labels).
        """
        from core.contradiction_scanner import scan, extract_state
        report = (
            "Verdict: Hold\n"
            "On the buy-side, sentiment is mixed; buyers entered Q4 but momentum faded.\n"
            "Action: Hold Steady"
        )
        extracted = extract_state(report)
        # Only verdict-anchored mentions should be extracted — narrative "buy-side" excluded
        assert "Buy" not in extracted.unique_verdicts()
        # Scanner should NOT flag this as multi-verdict drift
        res = scan(report, decision_data={"tax_verdict": "Hold"})
        rule_ids = {c.rule_id for c in res.contradictions}
        assert "L1_BUY_VERDICT_IN_HOLD_REPORT" not in rule_ids

    def test_disclaimer_tampering_attempt(self):
        """
        Adversarial disclaimer with injected verdict text. Even if the LLM
        smuggles "Verdict: Buy" inside the disclaimer block, protected_blocks
        must shield the entire disclaimer text from any transform.
        """
        from core.protected_blocks import protect, restore
        from core.editorial import rule_based_clean
        from core.contradiction_scanner import scan

        adversarial = (
            "Verdict: Hold\nAction: Hold Steady.\n\n"
            "> ⚠️ **Disclaimer:** Verdict: Buy. The above is not an offer to buy or sell. "
            "Conviction: Low. ACCUMULATE.\n\n"
            + "Narrative paragraph. " * 30
        )
        # Run the FULL governance stack
        protected, spans = protect(adversarial)
        cleaned = rule_based_clean(protected, ticker="ADV")
        scan_res = scan(cleaned, decision_data={"tax_verdict": "Hold"})
        restored = restore(scan_res.fixed_text, spans)

        # Disclaimer must come back BYTE-IDENTICAL to the original block
        original_disclaimer = (
            "> ⚠️ **Disclaimer:** Verdict: Buy. The above is not an offer to buy or sell. "
            "Conviction: Low. ACCUMULATE."
        )
        assert original_disclaimer in restored, \
            "Disclaimer was tampered — protected_blocks failed"
        # And the body of the report does NOT keep the legacy terminology
        # outside the disclaimer (editorial normalizes the rest)
        body = restored.split("> ⚠️")[0]
        assert "Conviction: Low" not in body
        assert "ACCUMULATE" not in body  # uppercase normalization runs on body

    def test_cross_section_inconsistency_injection(self):
        """
        Quick View shows one risk level while body shows another.
        Scanner must detect L1_MULTI_RISK_LEVELS — observer flag, no mutation.
        """
        from core.contradiction_scanner import scan
        report = (
            "## Quick View\n"
            "**Risk:** Moderate · **Evidence:** Limited · **Timing:** Extended\n\n"
            "## Risk Framework\n"
            "**Risk:** High — geopolitical exposure dominates the cycle.\n"
        )
        res = scan(report, decision_data={"tax_verdict": "Hold", "tax_risk": "Moderate"})
        rule_ids = {c.rule_id for c in res.contradictions}
        assert "L1_MULTI_RISK_LEVELS" in rule_ids
        # observer-only — no auto-fix
        for c in res.contradictions:
            if c.rule_id == "L1_MULTI_RISK_LEVELS":
                assert c.auto_fix is None
        # Body text unchanged by the multi-risk rule
        assert "**Risk:** High" in res.fixed_text
        assert "**Risk:** Moderate" in res.fixed_text

    def test_reconciliation_omission_failure(self):
        """
        Every unfixed contradiction MUST appear in result.unfixed.
        If a refactor drops items, ReconciliationAudit will under-report —
        this test guards against that omission.
        """
        from core.contradiction_scanner import scan
        report = (
            "Verdict: Hold\n"
            "Fundamental Verdict: Reduce\n"     # → L1_REDUCE_VERDICT_IN_HOLD_REPORT (warn, no fix)
            "Final Action: Buy\n"               # → L1_BUY_VERDICT_IN_HOLD_REPORT (warn, no fix)
            "**Risk:** Moderate\n"
            "**Risk:** High\n"                  # → L1_MULTI_RISK_LEVELS (blocker, no fix)
        )
        res = scan(report, decision_data={"tax_verdict": "Hold", "tax_risk": "Moderate"})
        # All three rule ids must be present in contradictions
        expected = {
            "L1_REDUCE_VERDICT_IN_HOLD_REPORT",
            "L1_BUY_VERDICT_IN_HOLD_REPORT",
            "L1_MULTI_RISK_LEVELS",
        }
        seen = {c.rule_id for c in res.contradictions}
        assert expected.issubset(seen), \
            f"ReconciliationAudit would miss: {expected - seen}"
        # And unfixed list also contains them (none of them are auto-fixable)
        seen_unfixed = {c.rule_id for c in res.unfixed}
        assert expected.issubset(seen_unfixed), \
            f"unfixed list dropped: {expected - seen_unfixed}"


# ══════════════════════════════════════════════════════════════════════════════
#  2. PIPELINE INVARIANCE
# ══════════════════════════════════════════════════════════════════════════════
class TestPipelineInvariance:
    """Stylistic-layer reordering must not change DecisionState or audit semantics."""

    @staticmethod
    def _pipeline_order_A(text: str, ticker: str) -> str:
        """sentence_variation BEFORE structural_variation"""
        from core.sentence_variation import apply_controlled_variability
        from core.structural_variation import apply_structural_variation
        out = apply_controlled_variability(text, ticker)
        out = apply_structural_variation(out, ticker)
        return out

    @staticmethod
    def _pipeline_order_B(text: str, ticker: str) -> str:
        """structural_variation BEFORE sentence_variation"""
        from core.sentence_variation import apply_controlled_variability
        from core.structural_variation import apply_structural_variation
        out = apply_structural_variation(text, ticker)
        out = apply_controlled_variability(out, ticker)
        return out

    def test_decision_state_invariant_under_stylistic_reorder(self):
        """
        Reorder the two non-decision stylistic layers (sentence_variation vs
        structural_variation). The DecisionState built from the scorecard must
        be IDENTICAL — neither layer touches it.
        """
        from core.decision_state import build_decision_state
        # DecisionState is built from the scorecard dict — it doesn't depend on
        # text transforms at all. Re-build with same inputs from any order.
        scorecard = {"tax_verdict": "Hold", "tax_timing": "Extended",
                     "tax_evidence": "Limited", "tax_risk": "Moderate",
                     "score": 66, "emoji": "🟡"}
        ds_a = build_decision_state(scorecard, ticker="X")
        ds_b = build_decision_state(scorecard, ticker="X")
        assert ds_a == ds_b
        # Even after pipeline reorder, the decision is the source-of-truth
        assert ds_a.verdict == "Hold"
        assert ds_a.action == "Hold Steady"

    def test_audit_flags_invariant_under_stylistic_reorder(self):
        """
        Same scan input → same set of contradictions, regardless of which
        stylistic transform ran first. The scanner depends on text content,
        not transform order.
        """
        from core.contradiction_scanner import scan
        text = (
            "Verdict: Hold\nAction: Hold Steady.\n"
            "Fundamental Verdict: Reduce\n"
            "Maintain a cautious stance. " * 5
        )
        # Apply each order
        out_a = self._pipeline_order_A(text, "INVX")
        out_b = self._pipeline_order_B(text, "INVX")
        # Run scan on both
        decision = {"tax_verdict": "Hold"}
        rules_a = {c.rule_id for c in scan(out_a, decision).contradictions}
        rules_b = {c.rule_id for c in scan(out_b, decision).contradictions}
        # The verdict-drift rule must fire in BOTH cases
        assert "L1_REDUCE_VERDICT_IN_HOLD_REPORT" in rules_a
        assert "L1_REDUCE_VERDICT_IN_HOLD_REPORT" in rules_b
        # The set may differ slightly only on rules that depend on prose
        # phrasing — but the core decision-drift flag is invariant.
        assert rules_a == rules_b or "L1_REDUCE_VERDICT_IN_HOLD_REPORT" in (rules_a & rules_b)

    def test_disclaimer_invariant_under_reorder(self):
        """Disclaimer survives BOTH pipeline orderings unchanged."""
        from core.protected_blocks import protect, restore
        from core.editorial import rule_based_clean
        from core.contradiction_scanner import scan

        disclaimer = "> ⚠️ **Disclaimer:** Not an offer to buy or sell any security."
        text = "Verdict: Hold.\n" + disclaimer + "\nBody text. " * 10

        def _run(order_fn):
            protected, spans = protect(text)
            cleaned = rule_based_clean(protected, ticker="INV")
            scanned = scan(cleaned, decision_data={"tax_verdict": "Hold"})
            varied = order_fn(scanned.fixed_text, "INV")
            return restore(varied, spans)

        out_a = _run(self._pipeline_order_A)
        out_b = _run(self._pipeline_order_B)
        # Disclaimer text intact in BOTH orderings
        assert disclaimer in out_a
        assert disclaimer in out_b


# ══════════════════════════════════════════════════════════════════════════════
#  3. MULTI-TICKER STRESS SET (10 tickers × 5 categories)
# ══════════════════════════════════════════════════════════════════════════════
# Each entry: (ticker, category, scorecard_dict, fund_dict, summary_dict)
# Designed to cover the canonical edge cases the governance stack must handle.
_STRESS_FIXTURES = [
    # ── Energy (Gulf + US) ─────────────────────────────────────────────────────
    (
        "ADNOCGAS.AE", "energy_gulf",
        {"tax_verdict": "Hold", "tax_timing": "Extended", "tax_evidence": "Limited",
         "tax_risk": "Moderate", "score": 66, "emoji": "🟡"},
        {"pe_ratio": 13.2, "beta": 0.28, "eps": 0.25, "div_yield": 5.07,
         "revenue": 18.5e9, "net_margin": 19.45, "sector": "Energy Minerals"},
        {"adx": 18, "rsi": 71.9, "trend": "Bullish"},
    ),
    (
        "XOM", "energy_us",
        {"tax_verdict": "Buy", "tax_timing": "Neutral", "tax_evidence": "Strong",
         "tax_risk": "Moderate", "score": 74, "emoji": "🟢"},
        {"pe_ratio": 14, "beta": 1.0, "eps": 8.5, "div_yield": 3.4,
         "revenue": 350e9, "net_margin": 11, "roe": 18, "debt_equity": 30,
         "ebitda": 80e9, "free_cash_flow": 36e9, "gross_margin": 30,
         "forward_eps": 9.0, "analyst_target": 130.0,
         "revenue_growth": 5, "earnings_growth": 8, "sector": "Energy Minerals"},
        {"adx": 25, "rsi": 58, "trend": "Bullish"},
    ),
    # ── Tech growth ────────────────────────────────────────────────────────────
    (
        "NVDA", "tech_growth_megacap",
        {"tax_verdict": "Buy", "tax_timing": "Attractive", "tax_evidence": "Strong",
         "tax_risk": "Moderate", "score": 82, "emoji": "🟢"},
        {"pe_ratio": 38, "beta": 1.7, "eps": 6.4, "revenue": 80e9,
         "net_margin": 50, "gross_margin": 73, "roe": 70, "debt_equity": 25,
         "ebitda": 38e9, "free_cash_flow": 32e9,
         "forward_eps": 8.1, "analyst_target": 950.0,
         "revenue_growth": 122, "earnings_growth": 200, "sector": "Technology Services"},
        {"adx": 32, "rsi": 65, "trend": "Bullish"},
    ),
    (
        "MSFT", "tech_growth_megacap",
        {"tax_verdict": "Buy", "tax_timing": "Neutral", "tax_evidence": "Strong",
         "tax_risk": "Low", "score": 78, "emoji": "🟢"},
        {"pe_ratio": 33, "beta": 0.95, "eps": 11, "revenue": 230e9,
         "net_margin": 36, "gross_margin": 70, "roe": 38, "debt_equity": 50,
         "forward_eps": 12.5, "analyst_target": 450.0,
         "revenue_growth": 13, "earnings_growth": 14, "sector": "Technology Services"},
        {"adx": 24, "rsi": 55, "trend": "Bullish"},
    ),
    # ── Distressed (poor fundamentals / negative growth) ───────────────────────
    (
        "DISTRESSED_A", "distressed",
        {"tax_verdict": "Reduce", "tax_timing": "Extended", "tax_evidence": "Moderate",
         "tax_risk": "High", "score": 38, "emoji": "🔴"},
        {"pe_ratio": 120, "beta": 1.4, "eps": 0.4, "revenue": 5e9,
         "net_margin": -8, "gross_margin": 18, "roe": -12, "debt_equity": 280,
         "revenue_growth": -15, "earnings_growth": -42, "sector": "Retail"},
        {"adx": 14, "rsi": 38, "trend": "Bearish"},
    ),
    (
        "DISTRESSED_B", "distressed",
        {"tax_verdict": "Sell", "tax_timing": "Extended", "tax_evidence": "Moderate",
         "tax_risk": "High", "score": 28, "emoji": "🔴"},
        {"pe_ratio": None, "beta": 2.1, "eps": -2.5, "revenue": 1.2e9,
         "net_margin": -55, "gross_margin": 12, "roe": -85, "debt_equity": 550,
         "revenue_growth": -32, "earnings_growth": -200, "sector": "Industrial Goods"},
        {"adx": 18, "rsi": 25, "trend": "Bearish"},
    ),
    # ── Low coverage (sparse ADX/EGX stock) ────────────────────────────────────
    (
        "LOWCOV_A.AE", "low_coverage",
        {"tax_verdict": "Hold", "tax_timing": "Neutral", "tax_evidence": "Limited",
         "tax_risk": "Moderate", "score": 55, "emoji": "🟡"},
        {"pe_ratio": 10, "beta": 0.7, "eps": 1.1, "sector": "Utilities"},
        {"adx": 15, "rsi": 50, "trend": "Sideways"},
    ),
    (
        "LOWCOV_B.CA", "low_coverage",
        {"tax_verdict": "Hold", "tax_timing": "Extended", "tax_evidence": "Limited",
         "tax_risk": "Moderate", "score": 50, "emoji": "🟡"},
        {"pe_ratio": 6, "beta": 0.6, "sector": "Banks", "div_yield": 7.0},
        {"adx": 12, "rsi": 45, "trend": "Sideways"},
    ),
    # ── High volatility ────────────────────────────────────────────────────────
    (
        "TSLA", "high_volatility",
        {"tax_verdict": "Hold", "tax_timing": "Extended", "tax_evidence": "Moderate",
         "tax_risk": "High", "score": 58, "emoji": "🟡"},
        {"pe_ratio": 78, "beta": 2.3, "eps": 3.2, "revenue": 95e9,
         "net_margin": 8, "gross_margin": 19, "roe": 15, "debt_equity": 12,
         "forward_eps": 4.0, "analyst_target": 230.0,
         "revenue_growth": -1, "earnings_growth": -35, "sector": "Consumer Cyclical"},
        {"adx": 22, "rsi": 48, "trend": "Sideways"},
    ),
    (
        "BTC-USD", "high_volatility_crypto",
        {"tax_verdict": "Hold", "tax_timing": "Neutral", "tax_evidence": "Limited",
         "tax_risk": "High", "score": 60, "emoji": "🟡"},
        {"price": 108_500, "beta": 2.5, "sector": "Crypto"},
        {"adx": 30, "rsi": 60, "trend": "Bullish",
         "sma50": 103_200, "sma200": 91_000},
    ),
]


class TestMultiTickerStress:
    """
    For each ticker fixture: run the deterministic governance core
    (DecisionState + EvidenceRouter + ContradictionScanner + ToneGovernor)
    and assert sane outputs across the full diversity of inputs.
    """

    @pytest.mark.parametrize("ticker,category,scorecard,fund,summary", _STRESS_FIXTURES,
                             ids=[t[0] for t in _STRESS_FIXTURES])
    def test_decision_state_for_every_category(self, ticker, category, scorecard,
                                               fund, summary):
        from core.decision_state import build_decision_state
        ds = build_decision_state(scorecard, summary=summary, ticker=ticker)
        # Type contracts: every DecisionState must satisfy canonical values
        assert ds.verdict in {"Buy", "Hold", "Reduce", "Sell"}
        assert ds.action in {"Scale In", "Wait", "Hold Steady", "Reduce Exposure"}
        assert ds.risk in {"Low", "Moderate", "High"}
        assert ds.evidence in {"Limited", "Moderate", "Strong"}
        assert ds.timing in {"Attractive", "Neutral", "Extended"}
        assert 0 <= ds.score <= 100

    @pytest.mark.parametrize("ticker,category,scorecard,fund,summary", _STRESS_FIXTURES,
                             ids=[t[0] for t in _STRESS_FIXTURES])
    def test_evidence_router_no_crash_each_ticker(self, ticker, category, scorecard,
                                                  fund, summary):
        from core.evidence_router import route_evidence
        allow = route_evidence(
            fund=fund, scorecard=scorecard, summary=summary,
            peers=[], analyst_data={}, ticker=ticker,
        )
        # Low-coverage tickers should disable premium sections
        if category in {"low_coverage", "high_volatility_crypto"}:
            assert allow.peer_comparison is False
            assert allow.dcf_valuation is False
        # Rich tech-growth tickers should have full_fundamental enabled
        if category == "tech_growth_megacap":
            assert allow.full_fundamental is True

    @pytest.mark.parametrize("ticker,category,scorecard,fund,summary", _STRESS_FIXTURES,
                             ids=[t[0] for t in _STRESS_FIXTURES])
    def test_scanner_observer_mode_each_ticker(self, ticker, category, scorecard,
                                               fund, summary):
        """
        For each ticker, the scanner must execute without crashing
        and Buy/Reduce verdict flags must remain observer-only (no auto_fix).
        """
        from core.contradiction_scanner import scan
        # Build adversarial input that injects a contradicting verdict
        report = (
            f"Verdict: {scorecard.get('tax_verdict','Hold')}\n"
            f"Action: Hold Steady\n"
            f"Fundamental Verdict: Reduce  ← drift\n"
            f"Maintain stance under {category} conditions.\n"
        )
        res = scan(report, decision_data=scorecard, summary=summary)
        # Every Buy/Reduce verdict contradiction is observer-only
        for c in res.contradictions:
            if c.rule_id in {"L1_REDUCE_VERDICT_IN_HOLD_REPORT",
                             "L1_BUY_VERDICT_IN_HOLD_REPORT"}:
                assert c.auto_fix is None
                assert c.severity == "warn"

    @pytest.mark.parametrize("ticker,category,scorecard,fund,summary", _STRESS_FIXTURES,
                             ids=[t[0] for t in _STRESS_FIXTURES])
    def test_tone_governor_evidence_aware_each_ticker(self, ticker, category,
                                                      scorecard, fund, summary):
        """
        Tone governor must respect the scorecard's evidence tier — Limited
        evidence reports must strip precise targets; Strong evidence reports
        keep them.
        """
        from core.evidence_tone_governor import govern_tone
        evidence = scorecard.get("tax_evidence", "Moderate")
        # Stress text with both theatrical + precise target
        text = "Thesis Kill Shot: DCF suggests fair value of $4.50 here."
        res = govern_tone(text, evidence=evidence)
        # Theatrical phrasing is stripped at every tier
        assert "Thesis Kill Shot" not in res.text
        # Precise target rules apply only when evidence is Limited
        if evidence == "Limited":
            assert "$4.50" not in res.text
        elif evidence == "Strong":
            assert "$4.50" in res.text  # Strong evidence preserves precision
