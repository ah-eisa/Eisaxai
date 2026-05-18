"""
Phase H4 factor model tests.

Run with:
    cd /home/ubuntu/investwise && python -m phase_h.tests.test_factor_model
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from phase_h.factor_model import (
    compute_factor_decomposition,
    render_factor_decomposition_md,
)
from phase_h.testing.assertions import assert_envelope_valid
from phase_h.tone_guard import audit_block


def _factor_panel(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(42)
    t = np.arange(n, dtype=float)
    mkt = 0.008 + 0.025 * np.sin(t / 4.0) + rng.normal(0.0, 0.006, n)
    return pd.DataFrame(
        {
            "MKT": mkt,
            "SMB": rng.normal(0.0, 0.018, n),
            "HML": rng.normal(0.0, 0.016, n),
            "MOM": rng.normal(0.0, 0.017, n),
            "RMW": rng.normal(0.0, 0.014, n),
            "CMA": rng.normal(0.0, 0.013, n),
            "RF": 0.0,
        },
        index=idx,
    )


def test_synthetic_pure_market_loading() -> None:
    factors = _factor_panel(60)
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        {"SPY": 0.95 * factors["MKT"] + rng.normal(0.0, 0.0008, len(factors))},
        index=factors.index,
    )
    payload = compute_factor_decomposition({"SPY": 1.0}, returns, factors, model="Carhart")
    assert abs(payload["loadings"]["MKT"] - 0.95) < 0.08
    assert abs(payload["loadings"]["SMB"]) < 0.12
    assert abs(payload["loadings"]["HML"]) < 0.12
    assert abs(payload["loadings"]["MOM"]) < 0.12
    assert_envelope_valid(payload["_envelope"], expected_engine="factor_decomp")


def test_carhart_returns_4_factors_and_ff3_ff5_counts() -> None:
    factors = _factor_panel(60)
    returns = pd.DataFrame({"SPY": factors["MKT"]}, index=factors.index)
    assert len(compute_factor_decomposition({"SPY": 1.0}, returns, factors, model="FF3")["loadings"]) == 3
    assert len(compute_factor_decomposition({"SPY": 1.0}, returns, factors, model="Carhart")["loadings"]) == 4
    assert len(compute_factor_decomposition({"SPY": 1.0}, returns, factors, model="FF5")["loadings"]) == 5
    assert len(compute_factor_decomposition({"SPY": 1.0}, returns, factors, model="FF5_QMJ")["loadings"]) == 6
    assert len(compute_factor_decomposition({"SPY": 1.0}, returns, factors, model="LowVol")["loadings"]) == 1


def test_warning_emitted_for_high_beta() -> None:
    factors = _factor_panel(60)
    returns = pd.DataFrame({"SPY": 2.05 * factors["MKT"]}, index=factors.index)
    payload = compute_factor_decomposition({"SPY": 1.0}, returns, factors, model="FF3")
    assert any("exceeds 1.8" in warning for warning in payload["warnings"])


def test_render_bilingual_complete() -> None:
    factors = _factor_panel(60)
    returns = pd.DataFrame({"SPY": factors["MKT"]}, index=factors.index)
    payload = compute_factor_decomposition({"SPY": 1.0}, returns, factors, model="Carhart")
    md_en = render_factor_decomposition_md(payload, "en")
    md_ar = render_factor_decomposition_md(payload, "ar")
    for label in ("Factor Risk Decomposition", "Loading", "t-stat", "R²", "Stability", "Window"):
        assert label in md_en
    for label in ("تحليل المخاطر بحسب العوامل", "التحميل", "إحصاء t", "الثبات", "النافذة"):
        assert label in md_ar


def test_no_forbidden_phrases() -> None:
    factors = _factor_panel(60)
    returns = pd.DataFrame({"SPY": factors["MKT"]}, index=factors.index)
    md = render_factor_decomposition_md(
        compute_factor_decomposition({"SPY": 1.0}, returns, factors, model="Carhart"),
        "en",
    )
    assert audit_block(md) == {}


def test_short_history_demotes_tier() -> None:
    factors = _factor_panel(12)
    returns = pd.DataFrame({"SPY": factors["MKT"]}, index=factors.index)
    payload = compute_factor_decomposition({"SPY": 1.0}, returns, factors, model="Carhart")
    assert payload["reliability_tier"] == "Indicative"
    assert payload["window_months"] == 12


def test_no_data_returns_degenerate_not_raise() -> None:
    payload = compute_factor_decomposition({"SPY": 1.0}, returns_panel=pd.DataFrame(), factor_panel=pd.DataFrame())
    assert payload["reliability_tier"] == "Indicative"
    assert all(value == 0.0 for value in payload["loadings"].values())
    assert any("zero loadings" in note for note in payload["notes"])


def test_sparse_gcc_applies_james_stein_shrinkage() -> None:
    factors = _factor_panel(18)
    returns = pd.DataFrame({"KSA": 0.90 * factors["MKT"]}, index=factors.index)
    payload = compute_factor_decomposition({"KSA": 1.0}, returns, factors, model="Carhart")
    diag = payload["asset_diagnostics"]["KSA"]
    assert diag["shrinkage_applied"] is True
    assert diag["tau"] == 18.0
    assert payload["shrinkage_applied"] is True
    assert payload["reliability_tier"] == "Institutional-Lite"
    assert payload["loadings"]["MKT"] < 0.90


def main() -> int:
    tests = [
        test_synthetic_pure_market_loading,
        test_carhart_returns_4_factors_and_ff3_ff5_counts,
        test_warning_emitted_for_high_beta,
        test_render_bilingual_complete,
        test_no_forbidden_phrases,
        test_short_history_demotes_tier,
        test_no_data_returns_degenerate_not_raise,
        test_sparse_gcc_applies_james_stein_shrinkage,
    ]
    failures: list[str] = []
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - test runner reports all failures
            failures.append(f"{test.__name__}: {exc!r}")
    if failures:
        print("FAILURES: " + "; ".join(failures))
        return 1
    print(f"OK: phase_h.tests.test_factor_model ({len(tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
