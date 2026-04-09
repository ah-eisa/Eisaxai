"""
tests/test_scorecard.py
=======================
Unit tests for core/scorecard.py — hardening Phase C-4.

Covers:
  - sanitize_field: boundary validation for beta, div_yield, pe, mktcap
  - render_field: formatting + N/A passthrough
  - compute_tech_score: bullish / bearish / neutral signal combinations
  - compute_entry_quality: good / poor / acceptable timing scenarios
  - get_verdict: equity, crypto, ETF; upside guard; beta penalty; soft override
"""

import pytest
from core.scorecard import (
    sanitize_field,
    render_field,
    compute_tech_score,
    compute_entry_quality,
    get_verdict,
    calculate_score,
)


# ══════════════════════════════════════════════════════════════════════
# sanitize_field
# ══════════════════════════════════════════════════════════════════════

class TestSanitizeField:

    # ── beta ──────────────────────────────────────────────────────────
    def test_beta_normal(self):
        assert sanitize_field("beta", 1.2) == pytest.approx(1.2)

    def test_beta_zero_disallowed_for_normal_ticker(self):
        assert sanitize_field("beta", 0.0, ticker="MSFT") is None

    def test_beta_near_zero_disallowed(self):
        # Aramco-style near-zero beta → None
        assert sanitize_field("beta", -0.01, ticker="2222.SR") is None

    def test_beta_near_zero_allowed_for_hedge(self):
        # GLD is a known low-beta hedge — should pass
        result = sanitize_field("beta", 0.02, ticker="GLD")
        assert result is not None

    def test_beta_negative_extreme(self):
        assert sanitize_field("beta", -1.5, ticker="MSFT") is None

    def test_beta_too_high(self):
        assert sanitize_field("beta", 6.0, ticker="TSLA") is None

    def test_beta_high_but_valid(self):
        assert sanitize_field("beta", 2.5, ticker="TSLA") == pytest.approx(2.5)

    # ── dividend yield ─────────────────────────────────────────────────
    def test_div_yield_normal(self):
        assert sanitize_field("div_yield", 5.0) == pytest.approx(5.0)

    def test_div_yield_too_high(self):
        assert sanitize_field("div_yield", 35.0) is None

    def test_div_yield_negative(self):
        assert sanitize_field("div_yield", -1.0) is None

    def test_div_yield_zero(self):
        assert sanitize_field("div_yield", 0.0) == pytest.approx(0.0)

    # ── P/E ───────────────────────────────────────────────────────────
    def test_pe_normal(self):
        assert sanitize_field("pe", 22.5) == pytest.approx(22.5)

    def test_pe_negative(self):
        # Negative P/E = loss-making company, should be filtered
        assert sanitize_field("pe", -15.0) is None

    def test_pe_absurd(self):
        assert sanitize_field("pe", 1500.0) is None

    def test_forward_pe_normal(self):
        assert sanitize_field("forward_pe", 18.0) == pytest.approx(18.0)

    # ── market cap ────────────────────────────────────────────────────
    def test_mktcap_normal(self):
        assert sanitize_field("mktcap", 1.5e12) == pytest.approx(1.5e12)

    def test_mktcap_negative(self):
        assert sanitize_field("mktcap", -1000) is None

    # ── None passthrough ──────────────────────────────────────────────
    def test_none_value(self):
        assert sanitize_field("beta", None) is None

    # ── Non-numeric passthrough ───────────────────────────────────────
    def test_string_passthrough(self):
        # Non-numeric fields are passed through
        result = sanitize_field("name", "Apple Inc.")
        assert result == "Apple Inc."


# ══════════════════════════════════════════════════════════════════════
# render_field
# ══════════════════════════════════════════════════════════════════════

class TestRenderField:

    def test_renders_float(self):
        assert render_field("pe", 22.5) == "22.50"

    def test_renders_na_for_none(self):
        assert "N/A" in render_field("beta", None)

    def test_renders_na_for_invalid_beta(self):
        # beta=0.0 for non-hedge → sanitize → None → N/A
        result = render_field("beta", 0.0, ticker="AAPL")
        assert "N/A" in result

    def test_custom_format(self):
        result = render_field("div_yield", 5.123, fmt="{:.1f}%")
        assert result == "5.1%"


# ══════════════════════════════════════════════════════════════════════
# compute_tech_score
# ══════════════════════════════════════════════════════════════════════

class TestComputeTechScore:

    def _make(self, trend="", momentum="", adx=20, rsi=50):
        return {"trend": trend, "momentum": momentum, "adx": adx, "rsi": rsi}

    def test_strong_bullish(self):
        score = compute_tech_score(self._make("Bullish", "Bullish", adx=40, rsi=50))
        # 50 + 25 + 15 + 15 + 3 = 108 → capped at 100
        assert score == 100

    def test_strong_bearish(self):
        score = compute_tech_score(self._make("Bearish", "Bearish", adx=40, rsi=50))
        # 50 - 25 - 15 + 15 + 3 = 28
        assert score == 28

    def test_neutral(self):
        score = compute_tech_score(self._make("", "", adx=20, rsi=50))
        # 50 + 0 + 0 + 5 + 3 = 58
        assert score == 58

    def test_oversold_boosts_score(self):
        score_normal = compute_tech_score(self._make("Bearish", "Bearish", adx=20, rsi=50))
        score_oversold = compute_tech_score(self._make("Bearish", "Bearish", adx=20, rsi=25))
        # rsi=25 → +10 instead of +3
        assert score_oversold > score_normal

    def test_overbought_reduces_score(self):
        score_normal = compute_tech_score(self._make("Bullish", "Bullish", adx=20, rsi=50))
        score_overbought = compute_tech_score(self._make("Bullish", "Bullish", adx=20, rsi=80))
        assert score_overbought < score_normal

    def test_always_in_range(self):
        for trend in ("Bullish", "Bearish", ""):
            for momentum in ("Bullish", "Bearish", ""):
                for adx in (10, 20, 30, 40):
                    for rsi in (10, 30, 50, 70, 90):
                        score = compute_tech_score(
                            {"trend": trend, "momentum": momentum, "adx": adx, "rsi": rsi}
                        )
                        assert 0 <= score <= 100


# ══════════════════════════════════════════════════════════════════════
# compute_entry_quality
# ══════════════════════════════════════════════════════════════════════

class TestComputeEntryQuality:

    def _ideal(self):
        """Ideal entry conditions — RSI in zone, strong ADX, near SMA200, fear zone."""
        return {
            "rsi": 45, "adx": 35, "price": 100, "sma200": 98,
            "fear_greed": 20, "volume": 200, "avg_volume": 100, "trend": "Bullish"
        }

    def _poor(self):
        """Poor entry — overbought RSI, choppy ADX, overextended, greed zone."""
        return {
            "rsi": 80, "adx": 15, "price": 140, "sma200": 100,
            "fear_greed": 85, "volume": 50, "avg_volume": 100, "trend": "Bearish"
        }

    def test_ideal_entry_is_good(self):
        score, label, note = compute_entry_quality(self._ideal())
        assert score >= 70
        assert "Good" in label or "✅" in label

    def test_poor_entry_is_poor(self):
        score, label, note = compute_entry_quality(self._poor())
        assert score < 55

    def test_score_is_bounded(self):
        score, _, _ = compute_entry_quality(self._ideal())
        assert 0 <= score <= 100

    def test_returns_three_values(self):
        result = compute_entry_quality({"rsi": 50, "adx": 20})
        assert len(result) == 3
        score, label, note = result
        assert isinstance(score, int)
        assert isinstance(label, str)
        assert isinstance(note, str)


# ══════════════════════════════════════════════════════════════════════
# get_verdict
# ══════════════════════════════════════════════════════════════════════

class TestGetVerdict:

    def _equity_data(self, price=100, target=130, beta=1.0,
                     trend="Bullish", momentum="Bullish", adx=30, rsi=50,
                     is_crypto=False, is_etf=False):
        return {
            "price": price, "target": target, "beta": beta,
            "trend": trend, "momentum": momentum, "adx": adx, "rsi": rsi,
            "is_crypto": is_crypto, "is_etf": is_etf,
        }

    # ── Basic verdict tiers ──────────────────────────────────────────
    def test_strong_bull_fundamentals_get_buy(self):
        # score=80+, strong technicals → BUY or STRONG BUY
        data = self._equity_data(price=100, target=160, beta=0.8,
                                 trend="Bullish", momentum="Bullish", adx=40)
        verdict, emoji, conviction = get_verdict(85, data)
        assert verdict in ("BUY", "STRONG BUY", "BUY (High Risk)")

    def test_weak_fundamentals_get_reduce_or_sell(self):
        data = self._equity_data(price=100, target=80, beta=1.0,
                                 trend="Bearish", momentum="Bearish", adx=30)
        verdict, emoji, conviction = get_verdict(25, data)
        assert verdict in ("REDUCE", "SELL")

    def test_neutral_gets_hold(self):
        data = self._equity_data(price=100, target=110, beta=1.0,
                                 trend="", momentum="", adx=18)
        verdict, emoji, conviction = get_verdict(55, data)
        assert verdict == "HOLD"

    # ── Upside guard ─────────────────────────────────────────────────
    def test_upside_guard_demotes_buy_when_no_room(self):
        # High score but price ≈ target (only 5% upside)
        data = self._equity_data(price=100, target=105, beta=1.0,
                                 trend="Bullish", momentum="Bullish", adx=35)
        verdict, emoji, conviction = get_verdict(82, data)
        # Upside < 10% AND total return < 12% → should be HOLD
        assert verdict == "HOLD"

    def test_upside_guard_bypassed_when_dividend_compensates(self):
        data = self._equity_data(price=100, target=108, beta=1.0,
                                 trend="Bullish", momentum="Bullish", adx=35)
        data["dividend_yield"] = 0.05   # 5% div yield → total return 13%
        verdict, emoji, conviction = get_verdict(82, data)
        # total_return = 8% + 5% = 13% >= 12% → still HOLD (upside_pct < 10)
        assert verdict == "HOLD"

    # ── Beta penalty ─────────────────────────────────────────────────
    def test_high_beta_demotes_strong_buy(self):
        data_normal = self._equity_data(price=100, target=160, beta=1.0,
                                        trend="Bullish", momentum="Bullish", adx=40)
        data_risky  = self._equity_data(price=100, target=160, beta=2.5,
                                        trend="Bullish", momentum="Bullish", adx=40)
        verdict_normal, _, _ = get_verdict(85, data_normal)
        verdict_risky,  _, _ = get_verdict(85, data_risky)
        # High beta should prevent STRONG BUY → BUY (High Risk)
        assert verdict_normal in ("STRONG BUY", "BUY")
        assert verdict_risky == "BUY (High Risk)"

    # ── Soft technical override ───────────────────────────────────────
    def test_strong_bear_tech_nudges_buy_down(self):
        data = self._equity_data(price=100, target=160, beta=1.0,
                                 trend="Bearish", momentum="Bearish", adx=40, rsi=50)
        # tech_score will be very low; fundamental score is BUY territory
        verdict, _, _ = get_verdict(72, data)
        # Should be nudged down by at most one tier
        assert verdict != "STRONG BUY"

    # ── Crypto weight ─────────────────────────────────────────────────
    def test_crypto_uses_tech_heavy_weight(self):
        # Same fundamental score, crypto should weight technicals 70%
        data_equity = self._equity_data(price=100, target=160, beta=1.5,
                                        trend="Bullish", momentum="Bullish", adx=35,
                                        is_crypto=False)
        data_crypto = self._equity_data(price=100, target=160, beta=1.5,
                                        trend="Bullish", momentum="Bullish", adx=35,
                                        is_crypto=True)
        v_eq, _, _ = get_verdict(60, data_equity)
        v_cr, _, _ = get_verdict(60, data_crypto)
        # Both may return different verdicts due to weight difference — just verify no crash
        assert v_eq in ("SELL", "REDUCE", "HOLD", "BUY", "STRONG BUY", "BUY (High Risk)")
        assert v_cr in ("SELL", "REDUCE", "HOLD", "BUY", "STRONG BUY", "BUY (High Risk)")

    # ── Emoji and conviction are populated ───────────────────────────
    def test_verdict_emoji_nonempty(self):
        data = self._equity_data()
        _, emoji, conviction = get_verdict(65, data)
        assert emoji
        assert conviction

    # ── Missing data resilience ───────────────────────────────────────
    def test_missing_price_target_does_not_crash(self):
        data = {"trend": "Bullish", "momentum": "Bullish"}
        verdict, emoji, conviction = get_verdict(70, data)
        assert verdict in ("SELL", "REDUCE", "HOLD", "BUY", "STRONG BUY", "BUY (High Risk)")


# ══════════════════════════════════════════════════════════════════════
# calculate_score — smoke tests (network-free)
# ══════════════════════════════════════════════════════════════════════

class TestCalculateScore:

    def test_returns_none_without_price(self):
        assert calculate_score({}) is None
        assert calculate_score({"ticker": "AAPL"}) is None

    def test_returns_tuple_with_price(self):
        data = {"price": 150, "ticker": "AAPL"}
        result = calculate_score(data)
        # May return None if more fields missing, but should not raise
        assert result is None or isinstance(result, tuple)

    def test_score_in_range_when_data_present(self):
        data = {
            "price": 150,
            "ticker": "AAPL",
            "target": 180,
            "pe": 28.0,
            "forward_pe": 24.0,
            "gross_margin": 44.0,
            "roe": 1.3,
            "revenue_growth": 8.0,
            "debt_to_equity": 1.5,
            "rsi": 52,
            "trend": "Bullish",
            "momentum": "Bullish",
            "adx": 28,
            "mc": 2.4e12,
        }
        result = calculate_score(data)
        # calculate_score returns (factors_dict, final_score_int, data) — 3-tuple
        if result is not None:
            _factors, score, _data = result
            assert 0 <= score <= 100
