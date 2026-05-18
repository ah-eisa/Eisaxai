"""
Phase H — numerical safety module.

Centralised hard-validators for the institutional pipeline. Every
engine that touches a covariance matrix, weight vector, return panel,
or optimizer assertion routes through here.

Failure modes:
- `validate_*` returns a typed `ValidationResult`; never raises by
  default so the report degrades gracefully into Indicative tier.
- `assert_*` raises `NumericalSafetyError` for the optimizer path
  where a silent-bad-PSD matrix would produce wrong-but-valid weights.

Contract: every check writes one structured row into the
`validation_log` returned by `validate_pipeline()`, which the
audit appendix renders into Section G for full reproducibility.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


logger = logging.getLogger("phase_h.numerics")


# ──────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────

class NumericalSafetyError(AssertionError):
    """Raised by `assert_*` helpers when a hard-stop violation occurs."""


@dataclass
class ValidationFinding:
    check: str
    severity: str        # "ok" | "warn" | "fail"
    detail: str
    metric: Optional[float] = None


@dataclass
class ValidationResult:
    ok: bool
    findings: List[ValidationFinding] = field(default_factory=list)

    def add(self, check: str, severity: str, detail: str, metric: Optional[float] = None) -> None:
        self.findings.append(ValidationFinding(check, severity, detail, metric))
        if severity == "fail":
            self.ok = False

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        self.findings.extend(other.findings)
        if not other.ok:
            self.ok = False
        return self

    def as_audit_rows(self) -> List[List[str]]:
        rows: List[List[str]] = []
        for f in self.findings:
            metric_str = f"{f.metric:.6g}" if f.metric is not None else "—"
            rows.append([f.check, f.severity, f.detail, metric_str])
        return rows


# ──────────────────────────────────────────────────────────────────────
# Tunables (env-overrideable)
# ──────────────────────────────────────────────────────────────────────

PSD_MIN_EIGENVALUE         = 1e-9
PSD_CONDITION_NUMBER_LIMIT = 1e12
WEIGHT_SUM_TOLERANCE       = 1e-4
NAN_DETECTION_ATOL         = 1e-12


# ──────────────────────────────────────────────────────────────────────
# Dimension & shape checks
# ──────────────────────────────────────────────────────────────────────

def validate_universe_dims(
    *,
    universe_size: int,
    cov: Optional[np.ndarray] = None,
    corr: Optional[np.ndarray] = None,
    expected_returns: Optional[Sequence[float]] = None,
    weights: Optional[Sequence[float]] = None,
    asset_names: Optional[Sequence[str]] = None,
) -> ValidationResult:
    """
    Check that every per-asset structure matches `universe_size`.
    Catches the `_CORR` 17 vs `_UNIVERSE` 20 class of bugs at boot,
    not at runtime via IndexError.
    """
    r = ValidationResult(ok=True)
    if universe_size <= 0:
        r.add("universe_size", "fail", "universe is empty", float(universe_size))
        return r
    r.add("universe_size", "ok", f"n={universe_size}", float(universe_size))

    if cov is not None:
        if cov.ndim != 2 or cov.shape[0] != universe_size or cov.shape[1] != universe_size:
            r.add("cov_shape", "fail",
                  f"cov shape {cov.shape} vs universe n={universe_size}", float(cov.size))
        else:
            r.add("cov_shape", "ok", f"cov {cov.shape} matches universe")
    if corr is not None:
        if corr.ndim != 2 or corr.shape[0] != universe_size or corr.shape[1] != universe_size:
            r.add("corr_shape", "fail",
                  f"corr shape {corr.shape} vs universe n={universe_size}", float(corr.size))
        else:
            r.add("corr_shape", "ok", f"corr {corr.shape} matches universe")
    if expected_returns is not None:
        n = len(list(expected_returns))
        sev = "ok" if n == universe_size else "fail"
        r.add("expected_returns_len", sev,
              f"len={n} vs universe n={universe_size}", float(n))
    if weights is not None:
        n = len(list(weights))
        sev = "ok" if n == universe_size else "fail"
        r.add("weights_len", sev,
              f"len={n} vs universe n={universe_size}", float(n))
    if asset_names is not None:
        n = len(list(asset_names))
        sev = "ok" if n == universe_size else "fail"
        r.add("asset_names_len", sev,
              f"len={n} vs universe n={universe_size}", float(n))
    return r


# ──────────────────────────────────────────────────────────────────────
# Numerical-quality checks
# ──────────────────────────────────────────────────────────────────────

def has_nan_or_inf(arr: Any) -> bool:
    try:
        a = np.asarray(arr, dtype=float)
    except (TypeError, ValueError):
        return False
    if a.size == 0:
        return False
    return bool(np.isnan(a).any() or np.isinf(a).any())


def validate_psd(
    cov: np.ndarray,
    *,
    label: str = "cov",
    fix_if_violation: bool = False,
) -> Tuple[ValidationResult, np.ndarray]:
    """
    Validate covariance PSD-ness. If `fix_if_violation`, eigenvalue-clip
    and return the sanitised matrix; otherwise return the input matrix
    untouched. Always populates findings — caller decides hard-stop policy.
    """
    r = ValidationResult(ok=True)
    out = cov

    if cov is None:
        r.add(f"{label}_psd", "fail", "covariance is None")
        return r, np.empty((0, 0))

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        r.add(f"{label}_psd", "fail", f"not square: {cov.shape}", float(cov.size))
        return r, out

    if has_nan_or_inf(cov):
        r.add(f"{label}_nan", "fail", "NaN/Inf detected in covariance")
        if fix_if_violation:
            out = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            r.add(f"{label}_nan_fix", "warn", "NaN/Inf coerced to 0")

    # Symmetrise (catch numerical drift)
    asym = float(np.max(np.abs(out - out.T))) if out.size else 0.0
    if asym > 1e-6:
        r.add(f"{label}_asymmetry", "warn", f"asymmetry={asym:.3e}", asym)
        if fix_if_violation:
            out = 0.5 * (out + out.T)

    try:
        eigvals = np.linalg.eigvalsh(out)
        min_eig = float(eigvals.min()) if eigvals.size else float("nan")
        cond = float(eigvals.max() / max(eigvals.min(), PSD_MIN_EIGENVALUE)) if eigvals.size else float("nan")
        r.add(f"{label}_min_eigenvalue", "ok" if min_eig >= -PSD_MIN_EIGENVALUE else "fail",
              f"min eig={min_eig:.3e}", min_eig)
        r.add(f"{label}_condition_number", "ok" if cond <= PSD_CONDITION_NUMBER_LIMIT else "warn",
              f"cond={cond:.3e}", cond)
        if fix_if_violation and min_eig < PSD_MIN_EIGENVALUE:
            eigvals_clipped = np.maximum(eigvals, PSD_MIN_EIGENVALUE)
            eigvecs = np.linalg.eigh(out)[1]
            out = (eigvecs * eigvals_clipped) @ eigvecs.T
            r.add(f"{label}_psd_fix", "warn", "eigenvalues clipped to PSD")
    except np.linalg.LinAlgError as exc:
        r.add(f"{label}_eig_failed", "fail", repr(exc))

    return r, out


def validate_weights(
    w: Sequence[float],
    *,
    label: str = "weights",
    allow_short: bool = False,
) -> ValidationResult:
    """
    Validate a weight vector: finite, summing to ~1, in [0, 1] unless shorting.
    """
    r = ValidationResult(ok=True)
    arr = np.asarray(list(w), dtype=float)
    if arr.size == 0:
        r.add(f"{label}_size", "fail", "empty weight vector")
        return r
    if has_nan_or_inf(arr):
        r.add(f"{label}_nan", "fail", "NaN/Inf in weights")
        return r
    s = float(arr.sum())
    if abs(s - 1.0) > WEIGHT_SUM_TOLERANCE:
        r.add(f"{label}_sum", "fail",
              f"sum={s:.6f} (expected 1.0 ±{WEIGHT_SUM_TOLERANCE})", s)
    else:
        r.add(f"{label}_sum", "ok", f"sum={s:.6f}", s)
    if not allow_short and (arr < -1e-9).any():
        r.add(f"{label}_negative", "fail",
              f"negative weight present (min={float(arr.min()):.3e})", float(arr.min()))
    if (arr > 1.0 + 1e-9).any():
        r.add(f"{label}_oversize", "warn",
              f"weight > 1.0 (max={float(arr.max()):.6f})", float(arr.max()))
    return r


def validate_returns_panel(
    panel: Any,
    *,
    min_observations: int = 12,
    label: str = "returns",
) -> ValidationResult:
    """
    Validate a returns DataFrame / 2-D ndarray. Catches NaN columns,
    near-constant series, and insufficient history.
    """
    r = ValidationResult(ok=True)
    if panel is None:
        r.add(f"{label}_present", "fail", "no returns panel supplied")
        return r
    try:
        a = np.asarray(panel, dtype=float)
    except (TypeError, ValueError) as exc:
        r.add(f"{label}_coerce", "fail", repr(exc))
        return r
    if a.ndim != 2:
        r.add(f"{label}_shape", "fail", f"expected 2-D, got ndim={a.ndim}")
        return r
    n_obs, n_assets = a.shape
    r.add(f"{label}_shape", "ok", f"obs={n_obs} assets={n_assets}", float(n_obs))
    if n_obs < min_observations:
        r.add(f"{label}_history", "warn",
              f"obs={n_obs} < min {min_observations}", float(n_obs))
    if has_nan_or_inf(a):
        nan_cols = int(np.isnan(a).any(axis=0).sum())
        r.add(f"{label}_nan_columns", "warn",
              f"{nan_cols} column(s) contain NaN", float(nan_cols))
    std_per_col = np.nanstd(a, axis=0)
    near_const = int((std_per_col < 1e-10).sum())
    if near_const:
        r.add(f"{label}_constant_columns", "warn",
              f"{near_const} near-constant series detected", float(near_const))
    return r


def validate_benchmark_alignment(
    portfolio_returns: Any,
    benchmark_returns: Any,
    *,
    min_overlap: int = 12,
) -> ValidationResult:
    """
    Ensure portfolio + benchmark series can be safely compared:
    same length AND meaningful overlap window AND no all-NaN overlap.
    """
    r = ValidationResult(ok=True)
    p = np.asarray(portfolio_returns, dtype=float)
    b = np.asarray(benchmark_returns, dtype=float)
    if p.ndim != 1 or b.ndim != 1:
        r.add("benchmark_align_shape", "fail",
              f"portfolio ndim={p.ndim}, benchmark ndim={b.ndim}")
        return r
    if p.size != b.size:
        r.add("benchmark_align_length", "fail",
              f"portfolio={p.size} vs benchmark={b.size}", float(p.size))
    else:
        r.add("benchmark_align_length", "ok", f"len={p.size}", float(p.size))
    mask = np.isfinite(p) & np.isfinite(b)
    overlap = int(mask.sum())
    sev = "ok" if overlap >= min_overlap else "fail"
    r.add("benchmark_align_overlap", sev,
          f"overlap={overlap} min={min_overlap}", float(overlap))
    if overlap == 0:
        r.add("benchmark_align_overlap_zero", "fail", "no finite overlap")
    return r


# ──────────────────────────────────────────────────────────────────────
# Hard-stop assertions for the optimizer path
# ──────────────────────────────────────────────────────────────────────

def assert_psd(cov: np.ndarray, *, label: str = "cov") -> None:
    """
    Raise NumericalSafetyError if `cov` is not PSD enough to feed CVXPY.
    Used inside allocate() right before solver calls; bypasses the
    soft-degrade path.
    """
    res, _ = validate_psd(cov, label=label, fix_if_violation=False)
    if not res.ok:
        bad = [f for f in res.findings if f.severity == "fail"]
        raise NumericalSafetyError(
            f"PSD assertion failed for {label}: " + "; ".join(f"{f.check}={f.detail}" for f in bad)
        )


def assert_universe_synchronised(
    *,
    universe_size: int,
    cov: Optional[np.ndarray] = None,
    expected_returns: Optional[Sequence[float]] = None,
    asset_names: Optional[Sequence[str]] = None,
) -> None:
    """Raise on dimension mismatch between universe and per-asset structures."""
    res = validate_universe_dims(
        universe_size=universe_size,
        cov=cov,
        expected_returns=expected_returns,
        asset_names=asset_names,
    )
    if not res.ok:
        bad = [f for f in res.findings if f.severity == "fail"]
        raise NumericalSafetyError(
            "Universe dimension mismatch: " + "; ".join(f"{f.check}={f.detail}" for f in bad)
        )


# ──────────────────────────────────────────────────────────────────────
# Aggregated pipeline validator
# ──────────────────────────────────────────────────────────────────────

def validate_pipeline(
    *,
    universe_size: int,
    cov: Optional[np.ndarray] = None,
    corr: Optional[np.ndarray] = None,
    expected_returns: Optional[Sequence[float]] = None,
    weights: Optional[Sequence[float]] = None,
    asset_names: Optional[Sequence[str]] = None,
    returns_panel: Optional[Any] = None,
    benchmark_pair: Optional[Tuple[Any, Any]] = None,
) -> ValidationResult:
    """Run every applicable check and return a single ValidationResult."""
    r = ValidationResult(ok=True)
    r.merge(validate_universe_dims(
        universe_size=universe_size, cov=cov, corr=corr,
        expected_returns=expected_returns, weights=weights, asset_names=asset_names,
    ))
    if cov is not None:
        psd_res, _ = validate_psd(cov)
        r.merge(psd_res)
    if weights is not None:
        r.merge(validate_weights(weights))
    if returns_panel is not None:
        r.merge(validate_returns_panel(returns_panel))
    if benchmark_pair is not None:
        r.merge(validate_benchmark_alignment(benchmark_pair[0], benchmark_pair[1]))
    return r


__all__ = [
    "NumericalSafetyError",
    "ValidationFinding",
    "ValidationResult",
    "has_nan_or_inf",
    "validate_universe_dims",
    "validate_psd",
    "validate_weights",
    "validate_returns_panel",
    "validate_benchmark_alignment",
    "assert_psd",
    "assert_universe_synchronised",
    "validate_pipeline",
]
