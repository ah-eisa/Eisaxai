"""
test_week3_stabilization.py — Week 3 Stabilization regression suite

Covers:
  1. ADX 24.9 → no block (borderline emerging, WARNING not ERROR)
  2. ADX 18   → weak trend label
  3. Peer self-row exact match enforced
  4. Dividend mismatch (>0.5 pp) fails lint
  5. Quick View is never empty
  6. Invalid beta handled (excluded, not crashed)
"""
from __future__ import annotations

import pytest
from datetime import datetime

from core.services.interpretation_engine import classify_trend_strength
from core.services.phrase_builder import build_quick_insight
from core.services.report_lint import (
    RenderedReport,
    ReportSection,
    lint_beta_sanity,
    lint_dividend_consistency,
    lint_peer_self_row,
    lint_trend_language,
)
from core.services.report_snapshot import ReportSnapshot


# ── shared fixture helpers ────────────────────────────────────────────────────

def _make_snapshot(**overrides):
    ts = datetime.now().isoformat()
    base = {
        "ticker":      {"value": "MSFT",   "source": "fallback",  "timestamp": ts},
        "price":       {"value": 395.0,    "source": "realtime",  "timestamp": ts},
        "entry":       {"value": 382.0,    "source": "calc",      "timestamp": ts},
        "stop":        {"value": 375.0,    "source": "calc",      "timestamp": ts},
        "target":      {"value": 430.0,    "source": "calc",      "timestamp": ts},
        "beta":        {"value": 1.1,      "source": "cache",     "timestamp": ts},
        "pe":          {"value": 28.4,     "source": "cache",     "timestamp": ts},
        "forward_pe":  {"value": 26.2,     "source": "cache",     "timestamp": ts},
        "sma50":       {"value": 401.0,    "source": "calc",      "timestamp": ts},
        "sma200":      {"value": 404.0,    "source": "calc",      "timestamp": ts},
        "week52_high": {"value": 468.0,    "source": "cache",     "timestamp": ts},
        "week52_low":  {"value": 344.0,    "source": "cache",     "timestamp": ts},
        "market_cap":  {"value": 3.1e12,   "source": "cache",     "timestamp": ts},
        "div_yield":   {"value": 0.008,    "source": "cache",     "timestamp": ts},
    }
    for k, v in overrides.items():
        base[k] = {"value": v, "source": "override", "timestamp": ts}
    snap = ReportSnapshot(base)
    snap.freeze()
    return snap


def _snap_with_adx(adx: float):
    """Snapshot with _interpretation_context carrying the given ADX."""
    snap = _make_snapshot()
    # We need to set _interpretation_context before freeze — rebuild unlocked
    ts = datetime.now().isoformat()
    base = {
        "ticker":      {"value": "MSFT",  "source": "fallback", "timestamp": ts},
        "price":       {"value": 395.0,   "source": "realtime", "timestamp": ts},
        "entry":       {"value": 382.0,   "source": "calc",     "timestamp": ts},
        "stop":        {"value": 375.0,   "source": "calc",     "timestamp": ts},
        "target":      {"value": 430.0,   "source": "calc",     "timestamp": ts},
        "beta":        {"value": 1.1,     "source": "cache",    "timestamp": ts},
        "pe":          {"value": 28.4,    "source": "cache",    "timestamp": ts},
        "forward_pe":  {"value": 26.2,    "source": "cache",    "timestamp": ts},
        "sma50":       {"value": 401.0,   "source": "calc",     "timestamp": ts},
        "sma200":      {"value": 404.0,   "source": "calc",     "timestamp": ts},
        "week52_high": {"value": 468.0,   "source": "cache",    "timestamp": ts},
        "week52_low":  {"value": 344.0,   "source": "cache",    "timestamp": ts},
        "market_cap":  {"value": 3.1e12,  "source": "cache",    "timestamp": ts},
        "div_yield":   {"value": 0.008,   "source": "cache",    "timestamp": ts},
        "_interpretation_context": {
            "value": {"adx": adx, "price": 395.0, "support": 380.0,
                      "div_yield": 0.008, "entry_price": 382.0,
                      "volume_today": 700_000, "volume_avg": 1_000_000},
            "source": "calculated",
            "timestamp": ts,
        },
    }
    s = ReportSnapshot(base)
    s.freeze()
    return s


def _report(text: str, ticker: str = "MSFT") -> RenderedReport:
    return RenderedReport(
        ticker=ticker,
        full_text=text,
        sections=[ReportSection("Memo", text)],
        observed_prices=[395.0],
    )


# ── Test 1: ADX 24.9 → no block ──────────────────────────────────────────────

def test_adx_24_9_weak_phrase_is_acceptable():
    """ADX 24.9 borderline-emerging: 'weak trend' language is ACCEPTABLE (PASS, not blocked)."""
    snap = _snap_with_adx(24.9)
    result = lint_trend_language(_report("The stock is in a weak trend."), snap)
    assert result["result"] == "PASS", (
        f"ADX 24.9 borderline: 'weak trend' phrase should be PASS (not blocked), got {result['result']}"
    )


def test_adx_24_9_wrong_label_is_warning_not_error():
    """ADX 24.9 borderline-emerging: a completely wrong label (e.g. 'strong trend') produces WARNING not ERROR."""
    snap = _snap_with_adx(24.9)
    result = lint_trend_language(_report("The stock shows a confirmed trend."), snap)
    assert result["result"] == "WARNING", (
        f"ADX 24.9 should produce WARNING (borderline), got {result['result']}"
    )
    assert result.get("event") == "trend_tolerance_override"


def test_adx_24_9_borderline_flag():
    detail = classify_trend_strength(24.9)
    assert detail["label"] == "emerging trend"
    assert detail["borderline"] is True


# ── Test 2: ADX 18 → weak trend ──────────────────────────────────────────────

def test_adx_18_is_weak_trend():
    detail = classify_trend_strength(18)
    assert detail["label"] == "weak trend"
    assert detail["borderline"] is False


def test_adx_18_wrong_phrase_is_error():
    snap = _snap_with_adx(18)
    result = lint_trend_language(_report("The stock shows a strong trend."), snap)
    assert result["result"] == "ERROR"
    assert result.get("borderline") is False


# ── Test 3: Peer self-row exact match ────────────────────────────────────────

def test_peer_self_row_exact_match_passes():
    from core.services.peer_table_builder import build_peer_table
    snap = _make_snapshot(ticker="MSFT", price=395.0)
    result = build_peer_table(snap, [
        {"ticker": "MSFT", "price": 999.0, "pe": 50},   # LLM hallucination — must be dropped
        {"ticker": "AAPL", "price": 185.0, "pe": 30},
    ])
    assert result[0]["ticker"] == "MSFT"
    assert result[0]["price"] == 395.0, "Self-row must use snapshot price, not LLM value"
    assert len(result) == 2  # self + AAPL


def test_peer_self_row_lint_blocks_on_mismatch():
    snap = _make_snapshot(ticker="MSFT", price=395.0)
    # Report contains a peer table row for MSFT with wrong price
    bad_report = _report("| MSFT | Software | 450.00 |", ticker="MSFT")
    result = lint_peer_self_row(bad_report, snap)
    assert result["result"] == "ERROR"


# ── Test 4: Dividend mismatch (>0.5 pp) fails lint ───────────────────────────

def test_dividend_mismatch_over_half_pp_fails():
    snap = _make_snapshot(div_yield=0.03)  # 3.0%
    report = _report("Dividend Yield: 4.00%.")
    result = lint_dividend_consistency(report, snap)
    assert result["result"] == "MISMATCH"


def test_dividend_within_tolerance_passes():
    snap = _make_snapshot(div_yield=0.03)  # 3.0%
    report = _report("Dividend Yield: 3.00%.")
    result = lint_dividend_consistency(report, snap)
    assert result["result"] == "PASS"


# ── Test 5: Quick View never empty ───────────────────────────────────────────

@pytest.mark.parametrize("labels", [
    {},
    {"TrendStrength": "weak trend", "EntryQuality": "poor timing", "YieldQuality": "moderate yield", "RSIZone": "neutral momentum"},
    {"TrendStrength": "emerging trend", "EntryQuality": "acceptable entry", "YieldQuality": "attractive yield", "RSIZone": "bullish momentum"},
    {"TrendStrength": "weak trend", "RSIZone": "overbought", "EntryQuality": "acceptable entry", "YieldQuality": "low yield"},
    {"TrendStrength": "confirmed trend", "RSIZone": "bullish momentum", "EntryQuality": "favorable entry", "YieldQuality": "minimal yield"},
])
def test_quick_view_never_empty(labels):
    insight = build_quick_insight({"ticker": "TEST"}, labels)
    assert insight, f"Quick View was empty for labels: {labels}"
    assert len(insight) > 10


def test_quick_view_overbought_weak_trend():
    insight = build_quick_insight(
        {"ticker": "XYZ"},
        {"TrendStrength": "weak trend", "RSIZone": "overbought",
         "EntryQuality": "acceptable entry", "YieldQuality": "low yield"},
    )
    assert "overbought" in insight.lower()
    assert "weak trend" in insight.lower()


# ── Test 6: Invalid beta handled ────────────────────────────────────────────

@pytest.mark.parametrize("beta_val", [-0.5, -1, 5.1, 10, 99])
def test_invalid_beta_is_flagged(beta_val):
    snap = _make_snapshot(beta=beta_val)
    result = lint_beta_sanity(snap)
    assert result["result"] == "INVALID", f"beta={beta_val} should be INVALID"
    assert result["event"] == "invalid_beta"
    assert result["action"] == "exclude_from_narrative"


@pytest.mark.parametrize("beta_val", [0.0, 0.5, 1.0, 2.5, 5.0])
def test_valid_beta_passes(beta_val):
    snap = _make_snapshot(beta=beta_val)
    result = lint_beta_sanity(snap)
    assert result["result"] == "PASS"
