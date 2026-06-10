"""
test_week2_trust.py — Week 2 Trust Expansion test suite

Tests:
  1. Peer self-row always equals snapshot (not LLM-generated value)
  2. ADNOC price must be identical in peer table and snapshot
  3. Dividend yield mismatch detected by linter
  4. 52W availability conflict detected when snapshot has data
  5. Snapshot canonical accessor prevents non-canonical field access
"""
from __future__ import annotations

import pytest
from datetime import datetime


# ── shared fixture ────────────────────────────────────────────────────────────

def _make_snapshot(
    ticker="ADNOCGAS.AE",
    price=3.85,
    div_yield=0.053,
    week52_high=4.30,
    week52_low=3.20,
    pe=12.4,
    forward_pe=11.2,
    market_cap=5.2e10,
):
    from core.services.report_snapshot import ReportSnapshot

    ts = datetime.now().isoformat()
    snap = ReportSnapshot(
        {
            "ticker":      {"value": ticker,         "source": "fallback",  "timestamp": ts},
            "price":       {"value": price,          "source": "realtime",  "timestamp": ts},
            "entry":       {"value": price * 0.96,   "source": "calc",      "timestamp": ts},
            "stop":        {"value": price * 0.92,   "source": "calc",      "timestamp": ts},
            "target":      {"value": price * 1.10,   "source": "calc",      "timestamp": ts},
            "beta":        {"value": 0.9,            "source": "cache",     "timestamp": ts},
            "pe":          {"value": pe,             "source": "cache",     "timestamp": ts},
            "forward_pe":  {"value": forward_pe,     "source": "cache",     "timestamp": ts},
            "sma50":       {"value": price * 0.98,   "source": "calc",      "timestamp": ts},
            "sma200":      {"value": price * 0.99,   "source": "calc",      "timestamp": ts},
            "week52_high": {"value": week52_high,    "source": "cache",     "timestamp": ts},
            "week52_low":  {"value": week52_low,     "source": "cache",     "timestamp": ts},
            "market_cap":  {"value": market_cap,     "source": "cache",     "timestamp": ts},
            "div_yield":   {"value": div_yield,      "source": "cache",     "timestamp": ts},
        }
    )
    snap.freeze()
    return snap


# ── Test 1: Peer self-row always equals snapshot ──────────────────────────────

def test_peer_self_row_always_equals_snapshot():
    """
    LLM provides a row for the subject ticker with hallucinated values.
    build_peer_table must discard it and inject the correct snapshot row.
    """
    from core.services.peer_table_builder import build_peer_table

    snap = _make_snapshot()
    llm_rows = [
        # LLM hallucinated price 4.20 and div_yield 0.0529 for the subject ticker
        {"ticker": "ADNOCGAS.AE", "price": 4.20, "pe": 999, "div_yield": 0.0529},
        {"ticker": "TAQA.AE",     "price": 2.30, "pe": 18.0},
    ]
    result = build_peer_table(snap, llm_rows)

    assert result[0]["ticker"]    == "ADNOCGAS.AE", "self-row ticker mismatch"
    assert result[0]["price"]     == 3.85,          "self-row price must be snapshot value"
    assert result[0]["pe"]        == 12.4,          "self-row PE must be snapshot value"
    assert result[0]["div_yield"] == 0.053,         "self-row div_yield must be snapshot value"
    # Peer row preserved unchanged
    assert result[1]["ticker"] == "TAQA.AE"


# ── Test 2: ADNOC price consistent across snapshot and peer table ─────────────

def test_adnoc_price_identical_in_peer_table_and_snapshot():
    """
    build_peer_table must use snapshot price — identical to snapshot.get("price").
    This is the ADNOC-specific regression test for the LLM price mismatch bug.
    """
    from core.services.peer_table_builder import build_peer_table

    snap = _make_snapshot(ticker="ADNOCGAS.AE", price=3.85)
    result = build_peer_table(snap, [])

    peer_self_price = result[0]["price"]
    snapshot_price  = snap.get("price")

    assert peer_self_price == snapshot_price, (
        f"Price mismatch: peer table has {peer_self_price}, snapshot has {snapshot_price}"
    )


# ── Test 3: Dividend yield mismatch detected by linter ───────────────────────

def test_dividend_yield_mismatch_detected():
    """
    ADNOC regression: snapshot holds div_yield=0.01 (1.0%) but the LLM
    recalled 5.29% from training data — a 4.29 percentage point hallucination.
    lint_dividend_consistency must flag MISMATCH (delta >> 0.5% tolerance).
    """
    from core.services.report_lint import lint_dividend_consistency, RenderedReport, ReportSection

    snap = _make_snapshot(div_yield=0.01)  # 1.0% actual
    report = RenderedReport(
        ticker="ADNOCGAS.AE",
        full_text=(
            "The company offers a Dividend Yield: 5.29%. "
            "Investors note the attractive dividend yield of 5.29%."
        ),
        sections=[ReportSection("Fundamentals", "Dividend Yield: 5.29%")],
    )
    result = lint_dividend_consistency(report, snap)

    assert result["result"] == "MISMATCH", f"Expected MISMATCH, got: {result}"
    assert 5.29 in result["report_values"]
    assert result["snapshot_pct"] == pytest.approx(1.0, abs=0.01)


# ── Test 4: 52W conflict detected when snapshot has data ─────────────────────

def test_52w_conflict_when_snapshot_has_data_but_report_claims_unavailable():
    """
    Snapshot has week52_high=4.30, week52_low=3.20.
    Report claims '52W range: unavailable'.
    lint_52w_consistency must flag CONFLICT.
    """
    from core.services.report_lint import lint_52w_consistency, RenderedReport, ReportSection

    snap = _make_snapshot(week52_high=4.30, week52_low=3.20)
    report = RenderedReport(
        ticker="ADNOCGAS.AE",
        full_text="52-week range: N/A (unavailable). RSI at 42 suggests neutral momentum.",
        sections=[ReportSection("Technicals", "52W range unavailable.")],
    )
    result = lint_52w_consistency(report, snap)

    assert result["result"] == "CONFLICT", f"Expected CONFLICT, got: {result}"
    assert result["week52_high"] == 4.30


# ── Test 5: Snapshot canonical accessor enforces field lock ──────────────────

def test_snapshot_canonical_fields_locked_against_non_canonical_access():
    """
    get_canonical() must return correct values for canonical fields and raise
    KeyError for non-canonical fields (e.g. sma50, sma200, beta).
    This proves the LLM cannot override numeric integrity via the accessor.
    """
    snap = _make_snapshot(price=3.85, div_yield=0.053, forward_pe=11.2)

    # Canonical fields return snapshot-sourced values
    assert snap.get_canonical("price")["value"]      == 3.85
    assert snap.get_canonical("div_yield")["value"]  == 0.053
    assert snap.get_canonical("forward_pe")["value"] == 11.2
    assert snap.get_canonical("week52_high")["value"] == 4.30
    assert snap.get_canonical("pe")["value"]          == 12.4

    # Provenance is tracked
    assert snap.get_canonical("price")["source"] == "realtime"

    # Non-canonical fields raise KeyError — LLM cannot use this accessor for them
    with pytest.raises(KeyError):
        snap.get_canonical("sma50")

    with pytest.raises(KeyError):
        snap.get_canonical("beta")
