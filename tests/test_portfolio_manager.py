"""
tests/test_portfolio_manager.py
================================
Unit tests for core/portfolio_manager.py — Phase C-4.

Tests only the pure functions that require no network access:
  - parse_float / parse_int
  - detect_risk_pref (English + Arabic)
  - recommend_etfs / method_from_risk
  - render_weights
  - _tv_to_yfinance
  - has_placeholder_tickers
  - compute_risk_score
"""

import pytest
from core.portfolio_manager import (
    parse_float,
    parse_int,
    detect_risk_pref,
    recommend_etfs,
    method_from_risk,
    render_weights,
    has_placeholder_tickers,
    compute_risk_score,
)

# _tv_to_yfinance is not public, but we can import it directly
from core.portfolio_manager import _tv_to_yfinance


# ══════════════════════════════════════════════════════════════════════
# parse_float / parse_int
# ══════════════════════════════════════════════════════════════════════

class TestParsers:

    def test_parse_float_int_input(self):
        assert parse_float(5, 0.0) == pytest.approx(5.0)

    def test_parse_float_string(self):
        assert parse_float("3.14", 0.0) == pytest.approx(3.14)

    def test_parse_float_invalid_uses_default(self):
        assert parse_float("abc", 1.5) == pytest.approx(1.5)

    def test_parse_float_none_uses_default(self):
        assert parse_float(None, 2.0) == pytest.approx(2.0)

    def test_parse_int_normal(self):
        assert parse_int(7, 0) == 7

    def test_parse_int_float_string(self):
        assert parse_int("3.9", 0) == 3

    def test_parse_int_invalid_uses_default(self):
        assert parse_int("xyz", 10) == 10


# ══════════════════════════════════════════════════════════════════════
# detect_risk_pref
# ══════════════════════════════════════════════════════════════════════

class TestDetectRiskPref:

    # ── English ───────────────────────────────────────────────────────
    def test_low_risk_english(self):
        assert detect_risk_pref("I want a conservative portfolio") == "low"

    def test_low_risk_safer(self):
        assert detect_risk_pref("something safer please") == "low"

    def test_high_risk_english(self):
        assert detect_risk_pref("I want high risk investments") == "high"

    def test_aggressive_english(self):
        assert detect_risk_pref("Be aggressive with the allocation") == "high"

    def test_max_return_english(self):
        assert detect_risk_pref("I want maximum return") == "high"

    def test_neutral_returns_none(self):
        assert detect_risk_pref("optimize my portfolio") is None

    def test_empty_string(self):
        assert detect_risk_pref("") is None

    def test_none_input(self):
        assert detect_risk_pref(None) is None

    # ── Arabic ────────────────────────────────────────────────────────
    def test_low_risk_arabic_conservative(self):
        assert detect_risk_pref("أريد محفظة محافظ") == "low"

    def test_low_risk_arabic_safe(self):
        assert detect_risk_pref("شيء آمن من فضلك") == "low"

    def test_high_risk_arabic(self):
        assert detect_risk_pref("أريد مخاطره عاليه") == "high"

    def test_high_risk_arabic_max_gain(self):
        assert detect_risk_pref("اقصى ربح ممكن") == "high"


# ══════════════════════════════════════════════════════════════════════
# recommend_etfs / method_from_risk
# ══════════════════════════════════════════════════════════════════════

class TestRecommendEtfs:

    def test_high_risk_includes_qqq(self):
        etfs = recommend_etfs("high")
        assert "QQQ" in etfs

    def test_low_risk_includes_bnd(self):
        etfs = recommend_etfs("low")
        assert "BND" in etfs

    def test_none_returns_balanced(self):
        etfs = recommend_etfs(None)
        assert len(etfs) > 0
        # Balanced should have VTI
        assert "VTI" in etfs

    def test_method_low_risk(self):
        assert method_from_risk("low") == "min_vol"

    def test_method_high_risk(self):
        assert method_from_risk("high") == "max_sharpe"

    def test_method_none(self):
        assert method_from_risk(None) is None


# ══════════════════════════════════════════════════════════════════════
# render_weights
# ══════════════════════════════════════════════════════════════════════

class TestRenderWeights:

    def test_renders_known_ticker_with_name(self):
        result = render_weights({"SPY": 0.60, "GLD": 0.40})
        assert "S&P 500" in result or "SPY" in result
        assert "Gold" in result or "GLD" in result

    def test_renders_percentages(self):
        result = render_weights({"SPY": 0.5, "BND": 0.5})
        assert "50.00%" in result

    def test_sorts_by_weight_descending(self):
        result = render_weights({"BND": 0.2, "SPY": 0.8})
        lines = result.strip().split("\n")
        # First line should be SPY (0.8 > 0.2)
        assert "SPY" in lines[0]

    def test_empty_weights(self):
        result = render_weights({})
        assert result == ""


# ══════════════════════════════════════════════════════════════════════
# _tv_to_yfinance
# ══════════════════════════════════════════════════════════════════════

class TestTvToYfinance:

    def test_tadawul_conversion(self):
        assert _tv_to_yfinance("TADAWUL:2222") == "2222.SR"

    def test_dfm_conversion(self):
        assert _tv_to_yfinance("DFM:EMAAR") == "EMAAR.DU"

    def test_adx_conversion(self):
        assert _tv_to_yfinance("ADX:FAB") == "FAB.AE"

    def test_no_colon_passthrough(self):
        assert _tv_to_yfinance("AAPL") == "AAPL"

    def test_unknown_exchange_passthrough(self):
        result = _tv_to_yfinance("UNKNOWN:XYZ")
        assert result == "XYZ"


# ══════════════════════════════════════════════════════════════════════
# has_placeholder_tickers
# ══════════════════════════════════════════════════════════════════════

class TestHasPlaceholderTickers:

    def test_detects_fake_ticker(self):
        # "TICKER" and "PLACEHOLDER" are in _ALL_FAKE_TICKERS
        weights = {"AAPL": 0.5, "TICKER": 0.3, "MSFT": 0.2}
        fakes = has_placeholder_tickers(weights)
        assert "TICKER" in fakes

    def test_detects_placeholder(self):
        weights = {"SPY": 0.6, "PLACEHOLDER": 0.4}
        fakes = has_placeholder_tickers(weights)
        assert "PLACEHOLDER" in fakes

    def test_no_fakes_in_clean_portfolio(self):
        weights = {"AAPL": 0.4, "MSFT": 0.3, "SPY": 0.3}
        assert has_placeholder_tickers(weights) == []

    def test_empty_weights(self):
        assert has_placeholder_tickers({}) == []


# ══════════════════════════════════════════════════════════════════════
# compute_risk_score
# ══════════════════════════════════════════════════════════════════════

class TestComputeRiskScore:

    def test_very_conservative_portfolio(self):
        result = compute_risk_score({"volatility": 0.05, "max_drawdown": 0.06})
        assert result["score"] <= 2
        assert "Conservative" in result["label"] or result["score"] <= 2

    def test_moderate_portfolio(self):
        result = compute_risk_score({"volatility": 0.15, "max_drawdown": 0.20})
        assert 3 <= result["score"] <= 7

    def test_aggressive_portfolio(self):
        result = compute_risk_score({"volatility": 0.45, "max_drawdown": 0.65})
        assert result["score"] >= 8

    def test_score_in_range(self):
        for vol in [0.05, 0.10, 0.20, 0.35, 0.50]:
            result = compute_risk_score({"volatility": vol, "max_drawdown": vol * 2})
            assert 1 <= result["score"] <= 10

    def test_var_monthly_is_positive(self):
        result = compute_risk_score({"volatility": 0.20})
        assert result["var_monthly"] > 0

    def test_returns_required_keys(self):
        result = compute_risk_score({"volatility": 0.18})
        for key in ("score", "label", "emoji", "var_monthly"):
            assert key in result

    def test_missing_drawdown_uses_estimate(self):
        # Should not crash when max_drawdown is absent
        result = compute_risk_score({"volatility": 0.15})
        assert isinstance(result["score"], int)
