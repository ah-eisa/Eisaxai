"""
Phase H orchestrator.

`augment_result()` is the single integration call used by
global_allocator.allocate(), portfolio_upload.upload_portfolio(),
and institutional_stock_wrapper. It:

1. Runs every enabled engine.
2. Attaches typed payloads to the result dict (additively).
3. Augments the report_md with the new sub/super sections in
   the correct position (preserving A-G hierarchy).
4. Attaches PhaseHMeta to the result for the audit appendix.

Flag-off path: returns the result dict unchanged.

Engines are imported here only (not in __init__) so that a broken
sub-engine does not break the package import.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from . import PHASE_H_VERSION
from .audit_extensions import build_meta, render_audit_rows
from .feature_flags import (
    PHASE_H_BENCHMARK,
    PHASE_H_COMMITTEE,
    PHASE_H_ENABLED,
    PHASE_H_FACTOR_MODEL,
    PHASE_H_FORWARD_SIM,
    PHASE_H_TC_OPTIMIZER,
)
from .report_helpers import inject_subsection, insert_top_level_before, finalize

# Engines (imported here so package import never fails on engine bug)
from . import benchmarks as _bench
from . import tc_optimizer as _tc
from . import forward_sim as _fwd
from . import factor_model as _fac
from . import committee as _com


# Anchor markdown markers (must match strings produced upstream)
_ANCHOR_C_EN = "## C. Risk Diagnostics"
_ANCHOR_C_AR = "## C. التشخيصات الكمية للمخاطر"
_ANCHOR_C_AR_ALT = "## C. تشخيصات المخاطر"
_ANCHOR_E_EN = "## E. Rebalancing Plan"
_ANCHOR_E_AR = "## E. خطة إعادة التوازن"
_ANCHOR_G_EN = "## G. Audit Appendix"
_ANCHOR_G_AR = "## G. ملحق المراجعة"


def _anchor_c(report_md: str, language: str) -> str:
    if language == "ar":
        for cand in (_ANCHOR_C_AR, _ANCHOR_C_AR_ALT):
            if cand in report_md:
                return cand
        return _ANCHOR_C_AR
    return _ANCHOR_C_EN


def _anchor_e(report_md: str, language: str) -> str:
    return _ANCHOR_E_AR if language == "ar" else _ANCHOR_E_EN


def _anchor_g(report_md: str, language: str) -> str:
    return _ANCHOR_G_AR if language == "ar" else _ANCHOR_G_EN


def augment_result(
    result: Dict[str, Any],
    *,
    language: str = "en",
    rebalance_frequency: str = "quarterly",
    committee_mode: Optional[str] = None,
    horizon_years: float = 5.0,
    asset_kind: Optional[str] = None,
    region_tilt: Optional[str] = None,
    benchmark_ticker: Optional[str] = None,
    w_prev: Optional[Dict[str, float]] = None,
    inject_into_report: bool = True,
) -> Dict[str, Any]:
    """
    Run Phase H engines and augment `result`. Returns the same dict
    instance (mutated additively) for caller convenience.

    If PHASE_H_ENABLED is false, returns `result` unchanged.

    `inject_into_report=False` runs the engines and attaches typed
    payloads, but leaves `result["report_md"]` untouched. Use this when
    the report is still being assembled by an upstream wrapper (e.g.
    portfolio_builder._run_allocator) — then call
    `inject_phase_h_sections(result, full_md, language)` once the full
    A-G report is built, to perform the injection in the correct order.
    """
    if not PHASE_H_ENABLED or not isinstance(result, dict):
        return result

    # Don't touch results that came back as feasibility failures —
    # the existing failure markdown is authoritative. Support both the
    # dict form ({"status": "..."}) and the legacy string form used by
    # global_allocator.allocate() ("All constraints satisfied" /
    # "Approximate solution" / "Infeasible: ...").
    if result.get("error"):
        return result
    feas = result.get("feasibility")
    if isinstance(feas, dict):
        if feas.get("status") == "infeasible":
            return result
    elif isinstance(feas, str):
        low = feas.lower()
        if "infeas" in low or "no solution" in low or "fail" in low:
            return result

    weights = result.get("weights") or {}
    report_md = result.get("report_md") or ""

    engines_ran = []
    engine_versions: Dict[str, str] = {}
    engine_payloads: Dict[str, Any] = {}

    # ── H1 — Benchmark Relative ────────────────────────────────────
    if PHASE_H_BENCHMARK and weights:
        try:
            bench_payload = _bench.compute_benchmark_relative(
                weights=weights,
                returns_panel=result.get("returns_panel"),
                benchmark_ticker=benchmark_ticker,
                region_tilt=region_tilt,
                asset_kind=asset_kind,
                language=language,
            )
            if bench_payload:
                result["benchmark_relative"] = bench_payload
                engines_ran.append("benchmark_relative")
                engine_versions["benchmark_relative"] = _bench.ENGINE_VERSION
                engine_payloads["benchmark_relative"] = bench_payload
                md = finalize(_bench.render_benchmark_relative_md(bench_payload, language))
                if md.strip() and inject_into_report:
                    report_md = inject_subsection(report_md, _anchor_c(report_md, language), md)
        except Exception as exc:  # noqa: BLE001 — engine isolation
            result.setdefault("_phase_h_errors", []).append(f"H1: {exc!r}")

    # ── H2 — Execution Diagnostics ─────────────────────────────────
    if PHASE_H_TC_OPTIMIZER and weights:
        try:
            exec_payload = _tc.estimate_execution(
                weights=weights,
                w_prev=w_prev,
                asset_meta=result.get("asset_meta"),
                rebalance_frequency=rebalance_frequency,
                language=language,
            )
            if exec_payload:
                result["execution_diag"] = exec_payload
                engines_ran.append("execution_diag")
                engine_versions["execution_diag"] = _tc.ENGINE_VERSION
                engine_payloads["execution_diag"] = exec_payload
                md = finalize(_tc.render_execution_md(exec_payload, language))
                if md.strip() and inject_into_report:
                    report_md = inject_subsection(report_md, _anchor_e(report_md, language), md)
        except Exception as exc:
            result.setdefault("_phase_h_errors", []).append(f"H2: {exc!r}")

    # ── H4 — Factor Decomposition ──────────────────────────────────
    if PHASE_H_FACTOR_MODEL and weights:
        try:
            fac_payload = _fac.compute_factor_decomposition(
                weights=weights,
                returns_panel=result.get("returns_panel"),
                factor_panel=result.get("factor_panel"),
                language=language,
            )
            if fac_payload:
                result["factor_decomp"] = fac_payload
                engines_ran.append("factor_decomp")
                engine_versions["factor_decomp"] = _fac.ENGINE_VERSION
                engine_payloads["factor_decomp"] = fac_payload
                md = finalize(_fac.render_factor_decomposition_md(fac_payload, language))
                if md.strip() and inject_into_report:
                    report_md = inject_subsection(report_md, _anchor_c(report_md, language), md)
        except Exception as exc:
            result.setdefault("_phase_h_errors", []).append(f"H4: {exc!r}")

    # ── H3 — Forward Scenario Distribution (top-level section H) ───
    if PHASE_H_FORWARD_SIM and weights:
        try:
            fwd_payload = _fwd.run_forward_simulation(
                weights=weights,
                expected_returns=result.get("expected_returns"),
                cov_matrix=result.get("cov_matrix"),
                horizon_years=horizon_years,
                language=language,
            )
            if fwd_payload:
                result["forward_scenario"] = fwd_payload
                engines_ran.append("forward_scenario")
                engine_versions["forward_scenario"] = _fwd.ENGINE_VERSION
                engine_payloads["forward_scenario"] = fwd_payload
                md = finalize(_fwd.render_forward_scenario_md(fwd_payload, language))
                if md.strip() and inject_into_report:
                    report_md = insert_top_level_before(report_md, _anchor_g(report_md, language), md)
        except Exception as exc:
            result.setdefault("_phase_h_errors", []).append(f"H3: {exc!r}")

    # ── H5 — Committee Brief (top-level section I) ─────────────────
    if PHASE_H_COMMITTEE and (committee_mode or os.environ.get("EISAX_COMMITTEE_MODE")):
        try:
            mode = committee_mode or os.environ.get("EISAX_COMMITTEE_MODE", "1pager")
            com_payload = _com.build_committee_brief(result, mode=mode, language=language)
            if com_payload:
                result["committee_brief"] = com_payload
                engines_ran.append("committee_brief")
                engine_versions["committee_brief"] = _com.ENGINE_VERSION
                engine_payloads["committee_brief"] = com_payload
                md = finalize(_com.render_committee_brief_md(com_payload, language))
                if md.strip() and inject_into_report:
                    report_md = insert_top_level_before(report_md, _anchor_g(report_md, language), md)
        except Exception as exc:
            result.setdefault("_phase_h_errors", []).append(f"H5: {exc!r}")

    # ── Audit appendix extension ───────────────────────────────────
    meta = build_meta(
        engines_ran=engines_ran,
        engine_versions=engine_versions,
        engine_payloads=engine_payloads,
    )
    result["phase_h_meta"] = meta

    # Append a small reproducibility block INSIDE audit appendix.
    if inject_into_report:
        g_anchor = _anchor_g(report_md, language)
        if g_anchor in report_md:
            audit_md = render_audit_rows(meta, language)
            if audit_md.strip():
                report_md = report_md.rstrip() + "\n" + audit_md.rstrip() + "\n"
        result["report_md"] = report_md

    return result


def inject_phase_h_sections(
    result: Dict[str, Any],
    full_report_md: str,
    language: str = "en",
) -> str:
    """
    Inject already-computed Phase H sections into a complete A-G report.

    Use this from `portfolio_builder._run_allocator()` after the full
    A-G markdown has been assembled. Engines are NOT re-run; we read
    payloads from `result` and render them into the correct anchors.

    Returns the augmented markdown string.
    """
    if not PHASE_H_ENABLED or not isinstance(result, dict) or not full_report_md:
        return full_report_md

    md = full_report_md

    # H1 under C
    bench = result.get("benchmark_relative")
    if bench:
        chunk = finalize(_bench.render_benchmark_relative_md(bench, language))
        if chunk.strip():
            md = inject_subsection(md, _anchor_c(md, language), chunk)

    # H4 under C (after H1 so it appears right after)
    fac = result.get("factor_decomp")
    if fac:
        chunk = finalize(_fac.render_factor_decomposition_md(fac, language))
        if chunk.strip():
            md = inject_subsection(md, _anchor_c(md, language), chunk)

    # H2 under E
    exe = result.get("execution_diag")
    if exe:
        chunk = finalize(_tc.render_execution_md(exe, language))
        if chunk.strip():
            md = inject_subsection(md, _anchor_e(md, language), chunk)

    # H3 top-level before G
    fwd = result.get("forward_scenario")
    if fwd:
        chunk = finalize(_fwd.render_forward_scenario_md(fwd, language))
        if chunk.strip():
            md = insert_top_level_before(md, _anchor_g(md, language), chunk)

    # H5 top-level before G (placed AFTER H3 by ordering)
    com = result.get("committee_brief")
    if com:
        chunk = finalize(_com.render_committee_brief_md(com, language))
        if chunk.strip():
            md = insert_top_level_before(md, _anchor_g(md, language), chunk)

    # Audit appendix reproducibility block — append at very end (G is last).
    meta = result.get("phase_h_meta") or {}
    if meta:
        g_anchor = _anchor_g(md, language)
        if g_anchor in md:
            audit_md = render_audit_rows(meta, language)
            if audit_md.strip():
                md = md.rstrip() + "\n" + audit_md.rstrip() + "\n"

    return md


__all__ = ["augment_result", "inject_phase_h_sections", "PHASE_H_VERSION"]
