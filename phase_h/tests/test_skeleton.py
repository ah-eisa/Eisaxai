"""
Phase H skeleton smoke tests.

Verify:
- package imports
- orchestrator runs on a synthetic result dict without crashing
- flag-off path returns input unchanged
- bilingual labels resolve
- audit meta builds and hashes
- tone-guard scrubs forbidden phrases

Run with:
    cd /home/ubuntu/investwise && python -m phase_h.tests.test_skeleton
"""

from __future__ import annotations

import os
import sys


def _make_result(language: str = "en") -> dict:
    if language == "ar":
        report = (
            "## A. الملخص التنفيذي\n\nنص.\n\n"
            "## B. جدوى التفويض\n\nنص.\n\n"
            "## C. تشخيصات المخاطر\n\nنص.\n\n"
            "## D. منطق التخصيص\n\nنص.\n\n"
            "## E. خطة إعادة التوازن\n\nنص.\n\n"
            "## F. طبقة التعليق الذكي\n\nنص.\n\n"
            "## G. ملحق المراجعة\n\nنص.\n"
        )
    else:
        report = (
            "## A. Executive Summary\n\nbody.\n\n"
            "## B. Mandate Feasibility\n\nbody.\n\n"
            "## C. Risk Diagnostics\n\nbody.\n\n"
            "## D. Allocation Logic\n\nbody.\n\n"
            "## E. Rebalancing Plan\n\nbody.\n\n"
            "## F. AI Commentary Layer\n\nbody.\n\n"
            "## G. Audit Appendix\n\nbody.\n"
        )
    return {
        "weights": {"SPY": 0.6, "TLT": 0.3, "GLD": 0.1},
        "metrics": {"profile": "balanced"},
        "feasibility": {"status": "feasible"},
        "confidence": {"reliability_tier": "Institutional"},
        "report_md": report,
    }


def main() -> int:
    failures: list[str] = []

    # 1. package import
    try:
        import phase_h  # noqa: F401
        from phase_h import schemas, feature_flags, tone_guard, report_helpers, audit_extensions
        from phase_h.orchestrator import augment_result
    except Exception as exc:
        print(f"FAIL: import phase_h — {exc!r}")
        return 1

    # 2. flag-off path
    saved = os.environ.get("EISAX_PHASE_H_ENABLED")
    os.environ["EISAX_PHASE_H_ENABLED"] = "0"
    import importlib
    importlib.reload(feature_flags)
    from phase_h import orchestrator as orch_off
    importlib.reload(orch_off)
    r_off = orch_off.augment_result(_make_result(), language="en")
    if "benchmark_relative" in r_off or "phase_h_meta" in r_off:
        failures.append("flag-off path leaked Phase H output")

    # 3. flag-on path EN
    if saved is None:
        os.environ.pop("EISAX_PHASE_H_ENABLED", None)
    else:
        os.environ["EISAX_PHASE_H_ENABLED"] = saved
    importlib.reload(feature_flags)
    from phase_h import orchestrator as orch_on
    importlib.reload(orch_on)
    r_en = orch_on.augment_result(_make_result(), language="en")
    if "phase_h_meta" not in r_en:
        failures.append("EN path missing phase_h_meta")
    if "Benchmark Relative Diagnostics" not in r_en["report_md"]:
        failures.append("EN report missing H1 subsection")
    if "Forward Scenario Distribution" not in r_en["report_md"]:
        failures.append("EN report missing H3 top-level")
    if r_en["report_md"].find("## G.") < r_en["report_md"].find("## H."):
        failures.append("Audit Appendix (G) is no longer last")

    # 4. flag-on path AR
    r_ar = orch_on.augment_result(_make_result(language="ar"), language="ar")
    if "تشخيصات الأداء النسبي مقابل المؤشر" not in r_ar["report_md"]:
        failures.append("AR report missing H1 Arabic heading")

    # 5. tone-guard
    scrubbed = tone_guard.scrub_text("This is a no-brainer moonshot 🚀 — strong timing.")
    if "no-brainer" in scrubbed or "moonshot" in scrubbed or "strong timing" in scrubbed:
        failures.append(f"tone-guard failed to scrub: {scrubbed!r}")

    # 6. audit meta build
    meta = audit_extensions.build_meta(
        engines_ran=["x"], engine_versions={"x": "1.0"}, engine_payloads={"x": {"k": 1}}
    )
    if not meta.get("audit_hashes", {}).get("x"):
        failures.append("audit meta failed to hash payload")

    # 7. feasibility-failure passthrough
    inf = _make_result()
    inf["feasibility"] = {"status": "infeasible"}
    r_inf = orch_on.augment_result(inf, language="en")
    if "phase_h_meta" in r_inf:
        failures.append("infeasible path leaked Phase H output")

    # 8. uploaded path (w_prev provided, no optimizer)
    r_up = orch_on.augment_result(_make_result(), language="en",
                                  w_prev={"SPY": 0.4, "TLT": 0.5, "GLD": 0.1})
    if r_up.get("execution_diag", {}).get("turnover_pct", 0) <= 0:
        failures.append("uploaded path turnover not computed")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK: phase_h skeleton smoke tests passed (8/8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
