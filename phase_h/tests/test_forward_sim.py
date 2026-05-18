"""
Phase H3 forward simulation tests.

Run with:
    python -m phase_h.tests.test_forward_sim
"""

from __future__ import annotations

import math
import os
import sys
from typing import Callable, List

from phase_h.cache import clear
from phase_h.forward_sim import (
    DEFAULT_SCENARIO_PRIORS,
    DETERMINISTIC_PATTERNS,
    _probabilistic_lint,
    render_forward_scenario_md,
    run_forward_simulation,
)
from phase_h.tone_guard import audit_block


WEIGHTS = {
    "US S&P 500 Broad": 45.0,
    "US Treasuries (LT)": 25.0,
    "Gold": 10.0,
    "US Large Cap Tech": 10.0,
    "Cash / T-Bills": 10.0,
}


def _run(**kwargs):
    clear()
    params = dict(weights=WEIGHTS, horizon_years=5, paths_per_scenario=350, seed=42)
    params.update(kwargs)
    return run_forward_simulation(**params)


def _payload(result):
    return result.get("payload") if isinstance(result.get("payload"), dict) else result


def test_seed_determinism() -> None:
    a = _payload(_run())
    b = _payload(_run())
    assert a["aggregate"] == b["aggregate"]


def test_scenario_probabilities_sum_to_one() -> None:
    assert math.isclose(sum(DEFAULT_SCENARIO_PRIORS.values()), 1.0, rel_tol=0, abs_tol=1e-12)
    p = _payload(_run())
    total = sum(sc["probability"] for sc in p["scenarios"].values())
    assert math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-12)


def test_recession_terminal_below_soft_landing() -> None:
    p = _payload(_run(paths_per_scenario=650))
    assert p["scenarios"]["recession"]["terminal_p50"] < p["scenarios"]["soft_landing"]["terminal_p50"]


def test_contributions_increase_p50() -> None:
    no_flow = _payload(_run(contributions_per_year=0.0))
    with_flow = _payload(_run(contributions_per_year=10000.0))
    assert with_flow["aggregate"]["terminal_p50"] > no_flow["aggregate"]["terminal_p50"]


def test_render_bilingual_complete() -> None:
    p = _payload(_run())
    md_en = render_forward_scenario_md(p, language="en")
    md_ar = render_forward_scenario_md(p, language="ar")
    assert "## H. Forward Scenario Distribution" in md_en
    assert "Aggregate (probability-weighted)" in md_en
    assert "Probability of loss" in md_en
    assert "## H. توزيع السيناريوهات المستقبلية" in md_ar
    assert "الإجمالي المرجح بالاحتمالات" in md_ar
    assert "احتمال الخسارة" in md_ar


def test_no_deterministic_language() -> None:
    p = _payload(_run())
    scrubbed = _probabilistic_lint(render_forward_scenario_md(p, language="en"))
    hits = [rx.pattern for rx in DETERMINISTIC_PATTERNS if rx.search(scrubbed)]
    assert not hits, hits


def test_no_forbidden_phrases() -> None:
    p = _payload(_run())
    md = render_forward_scenario_md(p, language="en")
    assert audit_block(md) == {}


def test_seed_propagated_to_audit() -> None:
    saved = os.environ.get("EISAX_PHASE_H_DETERMINISTIC_SEED")
    try:
        os.environ.pop("EISAX_PHASE_H_DETERMINISTIC_SEED", None)
        clear()
        p = _payload(run_forward_simulation(weights=WEIGHTS, horizon_years=5, paths_per_scenario=100))
        assert p["seed"] == 42
    finally:
        if saved is None:
            os.environ.pop("EISAX_PHASE_H_DETERMINISTIC_SEED", None)
        else:
            os.environ["EISAX_PHASE_H_DETERMINISTIC_SEED"] = saved


def main() -> int:
    tests: List[Callable[[], None]] = [
        test_seed_determinism,
        test_scenario_probabilities_sum_to_one,
        test_recession_terminal_below_soft_landing,
        test_contributions_increase_p50,
        test_render_bilingual_complete,
        test_no_deterministic_language,
        test_no_forbidden_phrases,
        test_seed_propagated_to_audit,
    ]
    failures: List[str] = []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - aggregate failures for CLI use
            failures.append(f"{fn.__name__}: {exc!r}")
            print(f"  FAIL  {fn.__name__}: {exc!r}")
    if failures:
        print("FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("OK: phase_h forward simulation tests passed (8/8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
