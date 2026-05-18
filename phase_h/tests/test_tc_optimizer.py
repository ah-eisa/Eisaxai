"""
Phase H2 transaction-cost optimizer tests.

Run with:
    cd /home/ubuntu/investwise && python -m phase_h.tests.test_tc_optimizer
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Iterator

import cvxpy as cp

from phase_h.contracts import unwrap
from phase_h.tc_optimizer import build_turnover_terms, estimate_execution, render_execution_md
from phase_h.tone_guard import audit_block
from phase_h.testing import assert_envelope_valid


@contextmanager
def _env(name: str, value: str) -> Iterator[None]:
    saved = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved


def test_turnover_zero_when_no_prior() -> None:
    w = cp.Variable(2)
    lin, quad, pers = build_turnover_terms(cp, w, None, linear_lambda=1.0, quadratic_lambda=1.0, persistence_lambda=1.0)
    assert float(lin.value) == 0.0
    assert float(quad.value) == 0.0
    assert float(pers.value) == 0.0

    payload = estimate_execution({"SPY": 0.6, "TLT": 0.4}, w_prev=None)
    assert_envelope_valid(payload, expected_engine="execution_diagnostics")
    assert unwrap(payload)["turnover_pct"] == 100.0


def test_turnover_matches_l1() -> None:
    payload = estimate_execution({"SPY": 0.5, "TLT": 0.5}, {"SPY": 0.6, "TLT": 0.4})
    assert unwrap(payload)["turnover_pct"] == 10.0


def test_slippage_increases_with_participation() -> None:
    weights = {"BTC-USD": 0.55, "ETH-USD": 0.45}
    prev = {"BTC-USD": 0.45, "ETH-USD": 0.55}
    meta = {
        "BTC-USD": {"vol": 0.75, "spread_bp": 0.0},
        "ETH-USD": {"vol": 0.90, "spread_bp": 0.0},
    }
    with _env("EISAX_TC_ADV_PARTICIPATION", "0.05"):
        base = unwrap(estimate_execution(weights, prev, meta))["slippage_bp"]
    with _env("EISAX_TC_ADV_PARTICIPATION", "0.10"):
        higher = unwrap(estimate_execution(weights, prev, meta))["slippage_bp"]
    ratio = higher / base
    assert 1.35 <= ratio <= 1.48


def test_render_bilingual_complete() -> None:
    payload = estimate_execution(
        {"Saudi Equities ETF": 0.12, "SPY": 0.58, "TLT": 0.30},
        {"Saudi Equities ETF": 0.10, "SPY": 0.60, "TLT": 0.30},
    )
    for language, heading, turnover_label in (
        ("en", "Execution Efficiency Diagnostics", "Turnover"),
        ("ar", "تشخيصات كفاءة التنفيذ", "معدل الدوران"),
    ):
        md = render_execution_md(payload, language)
        assert heading in md
        assert turnover_label in md
        assert "Implementation" in md or "فجوة التنفيذ" in md
        assert "Market Impact" in md or "أثر السوق" in md
        assert "Estimated Slippage" in md or "الانزلاق المقدر" in md
        assert "Rebalance" in md or "إعادة التوازن" in md


def test_no_forbidden_phrases() -> None:
    payload = estimate_execution({"SPY": 0.6, "TLT": 0.4}, {"SPY": 0.5, "TLT": 0.5})
    assert audit_block(render_execution_md(payload, "en")) == {}


def test_infeasibility_fallback() -> None:
    from global_allocator import allocate

    with _env("EISAX_TC_LINEAR_LAMBDA", "1e309"):
        result = allocate(
            profile="balanced",
            w_prev={"SPY": 0.40, "TLT": 0.30, "GLD": 0.30},
            max_drawdown=1.0,
        )
    assert not result.get("error"), result.get("error")
    assert result.get("weights")
    diag = result.get("execution_diag") or {}
    notes = unwrap(diag).get("notes", []) if diag else []
    assert any("turnover penalty relaxed" in str(note) for note in notes)


def main() -> int:
    tests = [
        test_turnover_zero_when_no_prior,
        test_turnover_matches_l1,
        test_slippage_increases_with_participation,
        test_render_bilingual_complete,
        test_no_forbidden_phrases,
        test_infeasibility_fallback,
    ]
    failures: list[str] = []
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report all failures
            failures.append(f"{test.__name__}: {exc!r}")
    if failures:
        print("FAILURES: " + "; ".join(failures))
        return 1
    print(f"OK: phase_h.tests.test_tc_optimizer ({len(tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
