"""
Phase H5 committee mode tests.

Run with:
    cd /home/ubuntu/investwise && python -m phase_h.tests.test_committee
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Iterator

from phase_h.committee import (
    SUPPORTED_MODES,
    build_committee_brief,
    build_objections,
    render_committee_brief_md,
)
from phase_h.orchestrator import inject_phase_h_sections
from phase_h.testing.assertions import assert_envelope_valid
from phase_h.tone_guard import audit_block


def _base_report() -> str:
    return (
        "## A. Executive Summary\n\nbody.\n\n"
        "## B. Mandate Feasibility\n\nbody.\n\n"
        "## C. Risk Diagnostics\n\nbody.\n\n"
        "## D. Allocation Logic\n\nbody.\n\n"
        "## E. Rebalancing Plan\n\nbody.\n\n"
        "## F. AI Commentary Layer\n\nbody.\n\n"
        "## G. Audit Appendix\n\nbody.\n"
    )


def _synthetic_result() -> dict:
    return {
        "weights": {
            "SPY": 0.48,
            "QQQ": 0.18,
            "KSA": 0.16,
            "TLT": 0.13,
            "GLD": 0.05,
        },
        "metrics": {"profile": "balanced", "volatility_pct": 11.2, "sharpe": 0.62},
        "feasibility": {"status": "feasible", "binding_constraints": ["equity cap", "drawdown budget"]},
        "confidence": {"reliability_tier": "Institutional"},
        "benchmark_relative": {
            "tracking_error_pct": 5.4,
            "information_ratio": 0.18,
            "active_return_pct": 0.9,
            "downside_capture": 1.08,
            "active_share_pct": 42.0,
            "regime_behavior": {"outperform_envs": ["inflation"], "lag_envs": ["recession"]},
            "reliability_tier": "Institutional",
            "notes": ["active share may require attribution review"],
        },
        "execution_diag": {
            "turnover_pct": 22.5,
            "implementation_shortfall_bp": 18.0,
            "complexity_tier": "moderate",
            "liquidity_stress": "elevated",
        },
        "forward_scenario": {
            "scenarios": {
                "soft_landing": {"terminal_p10": -7.0, "max_dd_p50_pct": -8.0},
                "recession": {"terminal_p10": -24.0, "max_dd_p50_pct": -18.0},
                "liquidity_crisis": {"terminal_p10": -31.0, "max_dd_p50_pct": -23.0},
                "inflation": {"terminal_p10": -14.0, "max_dd_p50_pct": -12.0},
            },
            "aggregate": {"prob_loss": 27.0},
        },
        "factor_decomp": {
            "loadings": {"MKT": 1.12, "HML": -0.52, "SMB": 0.18, "MOM": 0.34},
            "r_squared": 0.81,
            "rolling_stability": 0.74,
            "warnings": ["MKT beta may exceed policy review threshold"],
            "reliability_tier": "Institutional",
        },
        "asset_meta": {"KSA": {"region": "GCC"}},
        "report_md": _base_report(),
    }


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


def test_all_supported_modes_build_without_error() -> None:
    result = _synthetic_result()
    for mode in SUPPORTED_MODES:
        payload = build_committee_brief(result, mode=mode, language="en")
        assert payload["mode"] == mode
        assert payload["headline"]
        assert_envelope_valid(payload, expected_engine="committee_brief")
        assert_envelope_valid(payload["_envelope"], expected_engine="committee_brief")
        md = render_committee_brief_md(payload, "en")
        assert "## I. Investment Committee Brief" in md


def test_1pager_within_size_budget() -> None:
    payload = build_committee_brief(_synthetic_result(), mode="1pager", language="en")
    assert len(render_committee_brief_md(payload, "en")) < 4000


def test_defend_vs_bear_distinct_content() -> None:
    result = _synthetic_result()
    defend = render_committee_brief_md(build_committee_brief(result, mode="defend"), "en")
    bear = render_committee_brief_md(build_committee_brief(result, mode="bear"), "en")
    assert "**Stance: Defend may apply.**" in defend
    assert "**Stance: Challenge may apply.**" in bear
    assert "Liquidity stress may screen" in bear
    assert defend != bear


def test_no_forbidden_phrases() -> None:
    for mode in SUPPORTED_MODES:
        md = render_committee_brief_md(build_committee_brief(_synthetic_result(), mode=mode), "en")
        assert audit_block(md) == {}


def test_no_deterministic_language() -> None:
    forbidden = ("will", "guarantees", "guarantee", "always")
    for mode in SUPPORTED_MODES:
        md = render_committee_brief_md(build_committee_brief(_synthetic_result(), mode=mode), "en")
        lower = md.lower()
        assert not any(word in lower for word in forbidden), mode


def test_bilingual_complete() -> None:
    payload_en = build_committee_brief(_synthetic_result(), mode="hostile", language="en")
    payload_ar = build_committee_brief(_synthetic_result(), mode="hostile", language="ar")
    md_en = render_committee_brief_md(payload_en, "en")
    md_ar = render_committee_brief_md(payload_ar, "ar")
    assert "Investment Committee Brief" in md_en
    assert "Committee Objections" in md_en
    assert "Thesis Fragility" in md_en
    assert "ملخص لجنة الاستثمار" in md_ar
    assert "اعتراضات اللجنة" in md_ar
    assert "هشاشة الفرضية" in md_ar


def test_objection_schema_and_fragility() -> None:
    payload = build_committee_brief(_synthetic_result(), mode="hostile")
    objections = payload["objections"]
    assert 5 <= len(objections) <= 8
    for objection in objections:
        assert set(["category", "claim", "evidence_ref", "severity", "counter"]).issubset(objection)
    assert 0.0 <= float(payload["thesis_fragility_score"]) <= 100.0
    assert payload["cio_defensibility_verdict"] in {"defensible", "requires-justification", "weak-thesis"}
    macro = build_objections(_synthetic_result(), category="macro")
    assert len(macro) == 1
    assert macro[0]["category"] == "macro"


def test_section_appears_before_G() -> None:
    result = _synthetic_result()
    result["committee_brief"] = build_committee_brief(result, mode="1pager")
    result["phase_h_meta"] = {}
    md = inject_phase_h_sections(result, _base_report(), language="en")
    assert "## I." in md
    assert md.rfind("## G.") > md.rfind("## I.")


def test_default_mode_from_registry_env() -> None:
    with _env("EISAX_COMMITTEE_MODE", "challenge_liquidity"):
        payload = build_committee_brief(_synthetic_result(), mode=None)
    assert payload["mode"] == "challenge_liquidity"
    assert payload["objections"][0]["category"] == "liquidity"


def main() -> int:
    tests = [
        test_all_supported_modes_build_without_error,
        test_1pager_within_size_budget,
        test_defend_vs_bear_distinct_content,
        test_no_forbidden_phrases,
        test_no_deterministic_language,
        test_bilingual_complete,
        test_objection_schema_and_fragility,
        test_section_appears_before_G,
        test_default_mode_from_registry_env,
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
    print(f"OK: phase_h.tests.test_committee ({len(tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
