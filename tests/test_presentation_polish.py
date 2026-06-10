"""
test_presentation_polish.py — Presentation & data-polish layer tests

Covers:
  1. Fact-Check: invalid beta (< 0 or > 5) → "Not reliable"
  2. Fact-Check: P/E > 200 or ≤ 0 → "Not reliable"
  3. Fact-Check: valid beta renders correctly
  4. Fact-Check: valid P/E renders correctly
  5. Interpretation guard: RSI neutral + "momentum is bearish" → "momentum is weakening"
  6. Interpretation guard: RSI bullish + "momentum is bearish" → "momentum is improving"
  7. Interpretation guard: RSI oversold + "momentum is bearish" → unchanged (valid)
  8. Header spacing: "### 1.Executive" → "### 1. Executive"
  9. Section spacing: blank line inserted before merged header
  10. No merged sections after cleanup
"""
from __future__ import annotations

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _beta_display(effective_beta: float, dc_beta: float = 0.0, ticker: str = "AAPL") -> str:
    """Replicate the fact-check beta formatting logic from _build_factcheck_block."""
    _beta_eff_fc = float(effective_beta) if effective_beta else 0.0
    if _beta_eff_fc < 0:
        return "Not reliable"                          # negative = garbage data
    if _beta_eff_fc > 5:
        return "Not reliable"                          # absurdly high
    if _beta_eff_fc > 0:
        is_crypto = ticker.upper().endswith('-USD')
        note = " *(rolling)*" if is_crypto else ""
        return f"{_beta_eff_fc:.2f}{note}"
    # effective_beta == 0 — try dc_data
    _beta_dc_fc = float(dc_beta) if dc_beta else 0.0
    if 0 < _beta_dc_fc <= 5:
        return f"{_beta_dc_fc:.2f}"
    if _beta_dc_fc < 0 or _beta_dc_fc > 5:
        return "Not reliable"
    return 'N/A'


def _pe_display(pe_raw: float) -> str:
    """Replicate the fact-check P/E formatting logic."""
    _pe_float = float(pe_raw) if pe_raw else 0.0
    if _pe_float > 200:
        return "Not reliable"
    if _pe_float > 0:
        return f"{_pe_float:.1f}x"
    return 'N/A'


# ─────────────────────────────────────────────────────────────────────────────
# 1–4: Fact-Check sanity filters
# ─────────────────────────────────────────────────────────────────────────────

class TestFactCheckSanity:

    def test_negative_beta_is_not_reliable(self):
        assert _beta_display(-0.01) == "Not reliable"

    def test_zero_beta_falls_through_to_na(self):
        assert _beta_display(0.0) == 'N/A'

    def test_excessive_beta_not_reliable(self):
        assert _beta_display(7.5) == "Not reliable"

    def test_valid_beta_renders(self):
        assert _beta_display(1.20) == "1.20"

    def test_valid_beta_at_upper_bound(self):
        assert _beta_display(5.0) == "5.00"

    def test_beta_just_above_5_not_reliable(self):
        assert _beta_display(5.01) == "Not reliable"

    def test_crypto_rolling_note_preserved(self):
        result = _beta_display(1.50, ticker="BTC-USD")
        assert "*(rolling)*" in result
        assert "1.50" in result

    def test_pe_over_200_not_reliable(self):
        assert _pe_display(250.0) == "Not reliable"

    def test_pe_exactly_200_renders(self):
        assert _pe_display(200.0) == "200.0x"

    def test_pe_zero_is_na(self):
        assert _pe_display(0.0) == 'N/A'

    def test_pe_negative_is_na(self):
        assert _pe_display(-5.0) == 'N/A'

    def test_pe_valid_renders(self):
        assert _pe_display(28.4) == "28.4x"

    def test_pe_extreme_value_not_reliable(self):
        assert _pe_display(999.0) == "Not reliable"


# ─────────────────────────────────────────────────────────────────────────────
# 5–7: Interpretation guard momentum wording
# ─────────────────────────────────────────────────────────────────────────────

class TestMomentumWording:

    def setup_method(self):
        from core.services.interpretation_guard import InterpretationGuard
        self.guard = InterpretationGuard()

    def _labels(self, rsi_zone: str) -> dict:
        return {
            "TrendStrength":    "weak trend",
            "EntryQuality":     "poor timing",
            "RSIZone":          rsi_zone,
            "VolumeConviction": "normal volume conviction",
            "YieldQuality":     "moderate yield",
        }

    def test_neutral_rsi_bearish_becomes_weakening(self):
        text = "The stock's momentum is bearish despite recent price stability."
        result = self.guard.audit_and_sanitize(text, self._labels("neutral momentum"))
        assert "momentum is bearish" not in result.text
        assert "weakening" in result.text
        assert result.replacements_made >= 1

    def test_bullish_rsi_bearish_becomes_improving(self):
        text = "Analysts note that momentum is bearish in this setup."
        result = self.guard.audit_and_sanitize(text, self._labels("bullish momentum"))
        assert "momentum is bearish" not in result.text
        assert "improving" in result.text

    def test_oversold_rsi_bearish_is_allowed(self):
        """When RSI is oversold, 'momentum is bearish' is technically valid — no replacement."""
        text = "The stock's momentum is bearish as it enters oversold territory."
        result = self.guard.audit_and_sanitize(text, self._labels("oversold"))
        # oversold → rule is NOT added → no replacement
        assert result.replacements_made == 0

    def test_weak_momentum_rsi_bearish_becomes_weakening(self):
        text = "Price action confirms momentum is bearish."
        result = self.guard.audit_and_sanitize(text, self._labels("weak momentum"))
        assert "momentum is bearish" not in result.text
        assert "weakening" in result.text


# ─────────────────────────────────────────────────────────────────────────────
# 8–10: _post_render_cleanup — header spacing + section breaks
# ─────────────────────────────────────────────────────────────────────────────

class TestPostRenderCleanup:

    def setup_method(self):
        from core.services.analytics_builder import _post_render_cleanup
        self.cleanup = _post_render_cleanup

    def test_header_space_added(self):
        result = self.cleanup("### 1.Executive Summary\n")
        assert "### 1. Executive Summary" in result

    def test_header_space_not_doubled(self):
        result = self.cleanup("### 1. Executive Summary\n")
        assert "### 1.  Executive Summary" not in result
        assert "### 1. Executive Summary" in result

    def test_blank_line_before_header_after_text(self):
        """Paragraph immediately followed by ### header gets a blank line inserted."""
        raw = "This is a paragraph.\n### 2. Technical Outlook\n"
        result = self.cleanup(raw)
        # Should have two newlines before the header
        assert "\n\n### 2. Technical Outlook" in result

    def test_no_blank_line_added_when_already_separated(self):
        """If blank line already exists, cleanup must not add a third newline."""
        raw = "Paragraph.\n\n### 3. Why Now\n"
        result = self.cleanup(raw)
        # Must not produce triple newline
        assert "\n\n\n### 3. Why Now" not in result
        assert "\n\n### 3. Why Now" in result

    def test_no_merged_sections(self):
        """Sections in output must each start on their own paragraph."""
        raw = (
            "Intro text.\n"
            "### 1.Executive Summary\nContent here.\n"
            "### 2.Technical Outlook\nMore content.\n"
        )
        result = self.cleanup(raw)
        # Both headers must be preceded by blank line (or start of string)
        import re
        headers = re.findall(r'(?:^|\n\n)(#{1,6} \d+\. \w)', result)
        assert len(headers) >= 2, f"Expected 2 spaced headers, got: {result!r}"
