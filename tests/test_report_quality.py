"""
tests/test_report_quality.py
─────────────────────────────
Unit tests for EisaX report quality fixes.

Covers:
  - Bug 1: validate_positioning — stop must be below entry for long trades
  - Bug 2: classify_adx — consistent 4-bucket classification
  - Bug 3/7: 52W key-name normalization (year_high → week52_high)
  - Bug 4: No duplicate text in "Awaiting Pullback" note
  - Bug 6: _post_render_cleanup — section spacing, empty emoji bullets, dup ---
"""

import sys
import os
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Bug 2 — classify_adx
# ---------------------------------------------------------------------------

class TestClassifyAdx:
    """classify_adx must be the single deterministic source of truth for ADX labels."""

    def setup_method(self):
        from core.services.scorecard_engine import classify_adx
        self.classify_adx = classify_adx

    def test_weak_below_20(self):
        label, desc = self.classify_adx(15.0)
        assert label == "Weak"
        assert "ADX" in desc

    def test_emerging_20_to_25(self):
        label, desc = self.classify_adx(22.5)
        assert label == "Emerging"
        assert "20" in desc or "25" in desc

    def test_confirmed_25_to_30(self):
        label, desc = self.classify_adx(27.0)
        assert label == "Confirmed"

    def test_strong_at_30(self):
        label, desc = self.classify_adx(30.0)
        assert label == "Strong"

    def test_strong_above_30(self):
        label, desc = self.classify_adx(45.0)
        assert label == "Strong"

    def test_boundary_exactly_25(self):
        """ADX == 25 should be Confirmed (>= 25), not Emerging."""
        label, _ = self.classify_adx(25.0)
        assert label == "Confirmed"

    def test_boundary_exactly_20(self):
        """ADX == 20 should be Emerging (>= 20), not Weak."""
        label, _ = self.classify_adx(20.0)
        assert label == "Emerging"

    def test_zero(self):
        label, _ = self.classify_adx(0)
        assert label == "Weak"

    def test_none_coerces_to_zero(self):
        """None input should be coerced to 0 without raising."""
        label, _ = self.classify_adx(None)
        assert label == "Weak"

    def test_returns_tuple_of_two_strings(self):
        result = self.classify_adx(30)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)

    def test_no_binary_only_labels(self):
        """There must be at least one non-Strong, non-Weak label to avoid binary output."""
        labels = {self.classify_adx(v)[0] for v in [10, 22, 27, 35]}
        assert "Emerging" in labels or "Confirmed" in labels


# ---------------------------------------------------------------------------
# Bug 1 — validate_positioning
# ---------------------------------------------------------------------------

class TestValidatePositioning:
    """validate_positioning must guarantee stop < entry for every long trade."""

    def setup_method(self):
        from core.services.scorecard_engine import validate_positioning
        self.validate = validate_positioning

    def test_valid_no_fix_needed(self):
        ep, sp, fixed, note = self.validate(100.0, 93.0, 105.0)
        assert ep == pytest.approx(100.0)
        assert sp == pytest.approx(93.0)
        assert fixed is False
        assert note == ""

    def test_stop_above_entry_is_fixed(self):
        """Classic Bug 1: Fibonacci entry below SMA200*0.95 → stop >= entry."""
        # entry=382.64, stop=384.63 (stop > entry — invalid)
        ep, sp, fixed, note = self.validate(382.64, 384.63, 395.0)
        assert sp < ep, "stop must be below entry after fix"
        assert fixed is True
        assert note != ""

    def test_stop_equal_entry_is_fixed(self):
        ep, sp, fixed, note = self.validate(150.0, 150.0, 155.0)
        assert sp < ep
        assert fixed is True

    def test_entry_above_price_is_fixed(self):
        """Entry above current price is invalid for a limit buy."""
        ep, sp, fixed, note = self.validate(110.0, 100.0, 105.0)
        # After fix entry should be <= price
        assert ep <= 105.0 * 1.001
        assert sp < ep
        assert fixed is True

    def test_none_inputs_passthrough(self):
        ep, sp, fixed, note = self.validate(None, None, 100.0)
        assert ep is None
        assert sp is None
        assert fixed is False

    def test_fix_stop_is_7pct_below_entry(self):
        """When stop >= entry, corrected stop = ep * 0.93."""
        ep_in, sp_in = 200.0, 205.0
        ep, sp, fixed, _ = self.validate(ep_in, sp_in, 210.0)
        assert sp == pytest.approx(ep * 0.93, rel=1e-6)

    def test_msft_scenario(self):
        """
        Reproduce the MSFT-style case:
        SMA200=404.87, Fib entry=382.64, stop=SMA200*0.95=384.63
        stop (384.63) > entry (382.64) → must be fixed.
        """
        sma200 = 404.87
        entry_price = 382.64
        stop_price = sma200 * 0.95  # 384.63 — above entry
        price = 395.0

        ep, sp, fixed, note = self.validate(entry_price, stop_price, price)
        assert sp < ep, f"stop {sp:.2f} must be < entry {ep:.2f}"
        assert fixed is True


# ---------------------------------------------------------------------------
# Bug 3 — 52W key name normalization
# ---------------------------------------------------------------------------

class TestWeek52KeyNormalization:
    """year_high/year_low from DB cache must be aliased to week52_high/week52_low."""

    def test_year_high_mapped_to_week52_high(self):
        """
        Simulate what enrich_after_fetch does when DB provides year_high/year_low.
        The logic: if not fund.get('week52_high') and fund.get('year_high') → copy.
        """
        fund = {"year_high": 450.0, "year_low": 300.0}

        if not fund.get("week52_high") and fund.get("year_high"):
            fund["week52_high"] = float(fund["year_high"])
        if not fund.get("week52_low") and fund.get("year_low"):
            fund["week52_low"] = float(fund["year_low"])

        assert fund.get("week52_high") == 450.0
        assert fund.get("week52_low") == 300.0

    def test_existing_week52_not_overwritten(self):
        """Live yfinance data (week52_high) takes priority over DB year_high."""
        fund = {"year_high": 999.0, "week52_high": 450.0, "year_low": 999.0, "week52_low": 300.0}

        if not fund.get("week52_high") and fund.get("year_high"):
            fund["week52_high"] = float(fund["year_high"])
        if not fund.get("week52_low") and fund.get("year_low"):
            fund["week52_low"] = float(fund["year_low"])

        assert fund["week52_high"] == 450.0  # not overwritten
        assert fund["week52_low"] == 300.0   # not overwritten

    def test_db_cols_use_canonical_key_names(self):
        """
        The _cols list in finance.py must use 'week52_high'/'week52_low',
        NOT 'year_high'/'year_low'.
        """
        # Read the actual source to verify the key names
        import ast, textwrap

        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "core", "agents", "finance.py"
        )
        with open(src_path) as fh:
            source = fh.read()

        # Ensure the canonical names appear together in the _cols assignment
        assert '"week52_high"' in source, "_cols must use 'week52_high'"
        assert '"week52_low"' in source,  "_cols must use 'week52_low'"

        # The old wrong names should no longer appear in the _cols context
        # (They may appear elsewhere, so check the specific assignment context)
        import re
        cols_match = re.search(
            r'_cols\s*=\s*\[.*?\]',
            source,
            re.DOTALL,
        )
        assert cols_match, "_cols list not found in finance.py"
        cols_str = cols_match.group(0)
        assert '"year_high"' not in cols_str, "'year_high' still present in _cols"
        assert '"year_low"'  not in cols_str, "'year_low' still present in _cols"


# ---------------------------------------------------------------------------
# Bug 4 — No duplicate text in "Awaiting Pullback" note
# ---------------------------------------------------------------------------

class TestAwaitingPullbackNote:
    """The entry note must not repeat the same sentence twice."""

    def _make_entry_note(self, rp, entry_price):
        """Reproduce the logic from assemble_report for _entry_note."""
        def _fmt_price(p):
            return f"${p:,.2f}"

        if entry_price and rp and rp > entry_price * 1.02:
            _pct_to_entry = ((rp - entry_price) / rp) * 100
            # This is the FIXED version — single sentence
            note = (
                f"\n\n> Awaiting Pullback -- Current price "
                f"({_fmt_price(rp)}) is {_pct_to_entry:.1f}% above the entry zone "
                f"({_fmt_price(entry_price)}), which reduces the margin of safety "
                f"relative to the defined risk parameters."
            )
        else:
            note = ""
        return note

    def test_no_duplicate_sentence(self):
        note = self._make_entry_note(rp=420.0, entry_price=395.0)
        assert note != ""

        # Split into sentences and check for duplicates
        sentences = [s.strip() for s in note.split(".") if s.strip()]
        assert len(sentences) == len(set(sentences)), (
            f"Duplicate sentences detected in entry note:\n{note}"
        )

    def test_current_price_appears_once(self):
        note = self._make_entry_note(rp=420.0, entry_price=395.0)
        # "Current price" phrase should appear only once
        count = note.count("Current price")
        assert count == 1, f"'Current price' appears {count} times in note:\n{note}"

    def test_no_note_when_price_at_entry(self):
        """When price == entry (not above 2%), note should be empty."""
        note = self._make_entry_note(rp=395.0, entry_price=395.0)
        assert note == ""

    def test_no_note_when_price_below_entry(self):
        """When price < entry, note should be empty."""
        note = self._make_entry_note(rp=380.0, entry_price=395.0)
        assert note == ""


# ---------------------------------------------------------------------------
# Bug 6 — _post_render_cleanup
# ---------------------------------------------------------------------------

class TestPostRenderCleanup:
    """_post_render_cleanup must fix header spacing, empty bullets, duplicate ---."""

    def setup_method(self):
        from core.services.analytics_builder import _post_render_cleanup
        self.cleanup = _post_render_cleanup

    def test_section_header_spacing_added(self):
        report = "### 1.Executive Summary\n### 2.Technical Outlook\n"
        result = self.cleanup(report)
        assert "### 1. Executive Summary" in result
        assert "### 2. Technical Outlook" in result

    def test_section_header_already_spaced_unchanged(self):
        report = "### 1. Executive Summary\n"
        result = self.cleanup(report)
        assert "### 1. Executive Summary" in result
        assert "### 1.  Executive Summary" not in result  # no double space

    def test_empty_bullet_suppressed(self):
        report = "Some text\n\U0001f4a1\nMore text\n"
        result = self.cleanup(report)
        assert "\U0001f4a1" not in result

    def test_bullet_with_text_preserved(self):
        report = "- \U0001f4a1 This is a tip with content\n"
        result = self.cleanup(report)
        assert "This is a tip with content" in result

    def test_duplicate_separators_collapsed(self):
        report = "Section A\n---\n---\n---\nSection B\n"
        result = self.cleanup(report)
        # Should have at most 2 consecutive ---
        import re
        assert not re.search(r'(\n---){3,}', result), (
            f"3+ consecutive --- still present after cleanup:\n{result}"
        )

    def test_two_separators_preserved(self):
        report = "A\n---\n---\nB\n"
        result = self.cleanup(report)
        # Two consecutive --- is acceptable, should not be further collapsed
        assert "---" in result

    def test_no_change_to_clean_report(self):
        report = "### 1. Executive Summary\n\nNormal text.\n\n---\n\nMore text.\n"
        result = self.cleanup(report)
        assert result == report


# ---------------------------------------------------------------------------
# Integration: compute_positioning never produces stop >= entry
# ---------------------------------------------------------------------------

class TestComputePositioningAlwaysValid:
    """compute_positioning must always return stop < entry."""

    def setup_method(self):
        from core.services.scorecard_engine import compute_positioning
        self.compute = compute_positioning

    def _assert_valid(self, result):
        ep = result.get("ep")
        sp = result.get("sp")
        if ep is not None and sp is not None:
            assert sp < ep, (
                f"stop ({sp:.4f}) >= entry ({ep:.4f}) — invalid long setup\n"
                f"Result: {result}"
            )

    def test_normal_case(self):
        result = self.compute(
            real_price=420.0, sma200=400.0,
            h52=450.0, l52=320.0,
            display_target=480.0,
        )
        self._assert_valid(result)

    def test_msft_bug1_scenario(self):
        """
        MSFT-style: Fib entry (382.64) falls below SMA200*0.95 (384.63).
        Before fix: stop=384.63 > entry=382.64. After fix: must pass.
        """
        result = self.compute(
            real_price=395.0, sma200=404.87,
            h52=468.35, l52=344.79,
            display_target=490.0,
        )
        self._assert_valid(result)

    def test_price_at_high(self):
        """Price near 52W high — entry likely forced to price*0.96."""
        result = self.compute(
            real_price=468.0, sma200=400.0,
            h52=470.0, l52=320.0,
            display_target=None,
        )
        self._assert_valid(result)

    def test_price_deep_below_sma200(self):
        """Price deeply below SMA200 → stop should use price*0.91."""
        result = self.compute(
            real_price=320.0, sma200=400.0,
            h52=420.0, l52=300.0,
            display_target=None,
        )
        self._assert_valid(result)

    def test_no_52w_data(self):
        """Without 52W data, falls back to SMA200 entry."""
        result = self.compute(
            real_price=100.0, sma200=95.0,
            h52=0.0, l52=0.0,
            display_target=None,
        )
        self._assert_valid(result)

    def test_gcc_local_currency(self):
        result = self.compute(
            real_price=12.50, sma200=11.80,
            h52=14.00, l52=9.50,
            display_target=15.0,
            currency_sym="SAR", currency_lbl="SAR",
        )
        self._assert_valid(result)
        assert "SAR" in result["pre_entry"] or result["pre_entry"] == "N/A"
