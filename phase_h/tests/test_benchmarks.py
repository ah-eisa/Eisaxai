"""
Phase H1 benchmark engine tests.

Run with:
    cd /home/ubuntu/investwise && python -m phase_h.tests.test_benchmarks
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from phase_h.benchmarks import (
    compute_benchmark_relative,
    pick_benchmark,
    render_benchmark_relative_md,
)
from phase_h.orchestrator import augment_result
from phase_h.report_helpers import L
from phase_h.tone_guard import audit_block


def _synthetic_panel(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2021-01-31", periods=n, freq="ME")
    t = np.arange(n, dtype=float)
    spy = 0.008 + 0.025 * np.sin(t / 4.0)
    tlt = 0.003 + 0.55 * spy
    agg = 0.002 + 0.35 * spy
    return pd.DataFrame({"SPY": spy, "TLT": tlt, "AGG": agg, "ACWI": 0.95 * spy}, index=idx)


def _ann(s: pd.Series) -> float:
    return float((1.0 + s).prod() ** (12.0 / len(s)) - 1.0)


def test_pick_benchmark_policy() -> None:
    assert pick_benchmark(None, "crypto") == "BTC-USD"
    assert pick_benchmark({"GCC": 0.55, "US": 0.30}, None) == "^TASI"
    assert pick_benchmark({"US": 0.60, "GCC": 0.20}, None) == "SPY"
    assert pick_benchmark({"US": 0.35, "GCC": 0.35, "Bonds": 0.30}, None) == "URTH"


def test_compute_with_synthetic_panel() -> None:
    panel = _synthetic_panel(60)
    payload = compute_benchmark_relative(
        weights={"SPY": 0.6, "TLT": 0.4},
        returns_panel=panel,
        benchmark_ticker="SPY",
    )
    expected = 0.4 * (_ann(panel["TLT"].tail(36)) - _ann(panel["SPY"].tail(36))) * 100.0
    decomp_sum = sum(float(v) for v in payload["excess_decomp"].values())
    assert payload["tracking_error_pct"] > 0
    assert abs(payload["active_return_pct"] - expected) < 1.0
    assert abs(decomp_sum - payload["active_return_pct"]) < 0.5
    assert payload["reliability_tier"] == "Institutional"


def test_short_history_demotes_tier() -> None:
    panel = _synthetic_panel(6)
    payload = compute_benchmark_relative(
        weights={"SPY": 0.6, "TLT": 0.4},
        returns_panel=panel,
        benchmark_ticker="SPY",
    )
    assert payload["reliability_tier"] == "Indicative"


def test_render_includes_all_metric_labels_en_and_ar() -> None:
    payload = compute_benchmark_relative(
        weights={"SPY": 0.6, "TLT": 0.4},
        returns_panel=_synthetic_panel(60),
        benchmark_ticker="SPY",
    )
    required = [
        "active_return",
        "tracking_error",
        "information_ratio",
        "rolling_alpha",
        "rolling_beta",
        "relative_drawdown",
        "upside_capture",
        "downside_capture",
        "relative_volatility",
        "active_share",
        "style_drift",
        "allocation_effect",
        "selection_effect",
        "factor_effect",
        "concentration_effect",
    ]
    for language in ("en", "ar"):
        md = render_benchmark_relative_md(payload, language)
        for key in required:
            assert L(key, language) in md


def test_no_forbidden_phrases_after_scrub() -> None:
    payload = compute_benchmark_relative(
        weights={"SPY": 0.6, "TLT": 0.4},
        returns_panel=_synthetic_panel(60),
        benchmark_ticker="SPY",
    )
    md = render_benchmark_relative_md(payload, "en")
    assert audit_block(md) == {}


def test_feasibility_failure_skipped_by_orchestrator() -> None:
    result = {
        "weights": {"SPY": 0.6, "TLT": 0.4},
        "returns_panel": _synthetic_panel(60),
        "feasibility": {"status": "infeasible"},
        "report_md": "## C. Risk Diagnostics\n\nbody.\n\n## G. Audit Appendix\n\nbody.\n",
    }
    out = augment_result(result, language="en")
    assert "benchmark_relative" not in out


def main() -> int:
    tests = [
        test_pick_benchmark_policy,
        test_compute_with_synthetic_panel,
        test_short_history_demotes_tier,
        test_render_includes_all_metric_labels_en_and_ar,
        test_no_forbidden_phrases_after_scrub,
        test_feasibility_failure_skipped_by_orchestrator,
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
    print(f"OK: phase_h.tests.test_benchmarks ({len(tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
