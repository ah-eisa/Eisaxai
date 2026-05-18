"""
Phase H3 — Multi-Period Forward Simulation Engine.

Vectorised monthly Monte Carlo simulation with regime priors, stochastic
volatility, unstable correlations, and fat-tail jump risk.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .cache import memoize
from .contracts import make_envelope, unwrap
from .numerics import ValidationResult, validate_psd
from .registry import FeatureRegistry
from .report_helpers import LABELS, L, fmt_num, fmt_pct, md_table
from .schemas import ForwardScenario, ScenarioOutcome
from .tone_guard import scrub_block

ENGINE_VERSION = "0.2.0"


SCENARIO_NAMES = [
    "soft_landing",
    "recession",
    "stagflation",
    "ai_productivity_boom",
    "energy_shock",
    "liquidity_crisis",
]

DEFAULT_SCENARIO_PRIORS = {
    "soft_landing": 0.30,
    "recession": 0.20,
    "stagflation": 0.10,
    "ai_productivity_boom": 0.10,
    "energy_shock": 0.15,
    "liquidity_crisis": 0.15,
}

SCENARIO_SPECS = {
    "soft_landing": {"sigma": 0.95, "stress": 0.10, "equity": 0.005, "bond": 0.000, "commodity": 0.000},
    "recession": {"sigma": 1.20, "stress": 0.50, "equity": -0.040, "bond": 0.020, "commodity": 0.000},
    "stagflation": {"sigma": 1.30, "stress": 0.40, "equity": -0.020, "bond": -0.020, "commodity": 0.050},
    "ai_productivity_boom": {"sigma": 1.10, "stress": 0.25, "equity": 0.015, "tech": 0.060, "bond": 0.000, "commodity": 0.000},
    "energy_shock": {"sigma": 1.25, "stress": 0.35, "equity": -0.030, "bond": 0.000, "commodity": 0.080},
    "liquidity_crisis": {"sigma": 1.50, "stress": 0.80, "equity": -0.020, "bond": -0.010, "commodity": 0.000},
}

LOCAL_LABELS = {
    "aggregate_block_title": {
        "en": "Aggregate (probability-weighted)",
        "ar": "الإجمالي المرجح بالاحتمالات",
    },
    "expected_terminal_range": {
        "en": "Expected terminal value range (real)",
        "ar": "نطاق القيمة النهائية المتوقع (حقيقي)",
    },
    "prob_loss": {"en": "Probability of loss (real)", "ar": "احتمال الخسارة (حقيقي)"},
    "prob_target": {
        "en": "Probability of target (>=4% real ann.)",
        "ar": "احتمال بلوغ الهدف (>=4% حقيقي سنويا)",
    },
    "worst_decile": {"en": "Worst-decile terminal", "ar": "القيمة النهائية للعشر الأدنى"},
    "expected_dd_range": {"en": "Expected drawdown range", "ar": "نطاق الانخفاض المتوقع"},
    "recovery_duration_median": {
        "en": "Recovery duration (median)",
        "ar": "مدة التعافي (الوسيط)",
    },
    "distributional_disclaimer": {
        "en": "Distributional framing only - outcomes reflect modelled assumptions; not a forecast.",
        "ar": "إطار توزيعي فقط - تعكس النتائج افتراضات نموذجية وليست توقعا قطعيا.",
    },
}
LABELS.update({k: v for k, v in LOCAL_LABELS.items() if k not in LABELS})

DETERMINISTIC_PATTERNS = (
    re.compile(r"\bwill\b", re.IGNORECASE),
    re.compile(r"\bguarantee(?:s|d)?\b", re.IGNORECASE),
    re.compile(r"\bensure(?:s|d)?\b", re.IGNORECASE),
    re.compile(r"\bcertain(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\bexpect a return of\b", re.IGNORECASE),
)


def _enabled() -> bool:
    return FeatureRegistry.is_enabled("phase_h_forward_sim")


def _probabilistic_lint(text: str) -> str:
    """Scrub deterministic phrasing from rendered distribution text."""
    if not text:
        return text
    replacements = [
        (r"\bwill\b", "is modelled to"),
        (r"\bguarantee(?:s|d)?\b", "model-dependent indication"),
        (r"\bensure(?:s|d)?\b", "is designed to support"),
        (r"\bcertain(?:ly)?\b", "model-dependent"),
        (r"\bexpect a return of\b", "median modelled return:"),
    ]
    out = text
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def _weights_array(weights: Mapping[str, float]) -> Tuple[List[str], np.ndarray]:
    items = [(str(k), float(v)) for k, v in (weights or {}).items() if v is not None and float(v) > 0]
    if not items:
        return [], np.empty(0, dtype=float)
    names = [k for k, _ in items]
    arr = np.asarray([v for _, v in items], dtype=float)
    if arr.sum() > 1.5:
        arr = arr / 100.0
    total = float(arr.sum())
    if total <= 0:
        return [], np.empty(0, dtype=float)
    return names, arr / total


def _asset_bucket(name: str) -> str:
    low = name.lower()
    if any(x in low for x in ("cash", "t-bill", "tbill", "bil")):
        return "cash"
    if any(x in low for x in ("bitcoin", "ethereum", "crypto", "btc", "eth")):
        return "crypto"
    if any(x in low for x in ("gold", "silver", "oil", "crude", "copper", "commodity", "commodities", "gld", "slv", "uso", "cper")):
        return "commodity"
    if any(x in low for x in ("bond", "treasur", "duration", "tlt", "emb", "shy")):
        return "short_bond" if any(x in low for x in ("short", "shy")) else "bond"
    if any(x in low for x in ("tech", "nasdaq", "qqq", "ai", "software", "semiconductor")):
        return "tech_equity"
    if any(x in low for x in ("equity", "equities", "s&p", "spy", "mid-cap", "dividend", "value", "healthcare", "utilities", "ksa", "emaar")):
        return "equity"
    return "equity"


def _fallback_return(bucket: str) -> float:
    return {
        "cash": 0.045,
        "short_bond": 0.048,
        "bond": 0.055,
        "commodity": 0.075,
        "crypto": 0.250,
        "tech_equity": 0.135,
        "equity": 0.095,
    }.get(bucket, 0.085)


def _fallback_vol(bucket: str) -> float:
    return {
        "cash": 0.005,
        "short_bond": 0.030,
        "bond": 0.135,
        "commodity": 0.260,
        "crypto": 0.800,
        "tech_equity": 0.220,
        "equity": 0.165,
    }.get(bucket, 0.170)


def _coerce_return(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if abs(out) > 1.0:
        out /= 100.0
    return out


def _expected_returns(names: List[str], buckets: List[str], expected_returns: Optional[Mapping[str, float]]) -> np.ndarray:
    out: List[float] = []
    source = expected_returns or {}
    for name, bucket in zip(names, buckets):
        val = _coerce_return(source.get(name)) if isinstance(source, Mapping) else None
        out.append(_fallback_return(bucket) if val is None else val)
    return np.asarray(out, dtype=float)


def _fallback_corr(buckets: List[str]) -> np.ndarray:
    n = len(buckets)
    corr = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = buckets[i], buckets[j]
            pair = {a, b}
            if a == b:
                val = {
                    "cash": 0.10,
                    "short_bond": 0.45,
                    "bond": 0.50,
                    "commodity": 0.55,
                    "crypto": 0.82,
                    "tech_equity": 0.82,
                    "equity": 0.78,
                }.get(a, 0.35)
            elif pair <= {"equity", "tech_equity"}:
                val = 0.82
            elif "cash" in pair:
                val = 0.00
            elif ("bond" in pair or "short_bond" in pair) and ("equity" in pair or "tech_equity" in pair):
                val = -0.15
            elif ("commodity" in pair) and ("equity" in pair or "tech_equity" in pair):
                val = 0.18
            elif "crypto" in pair and ("equity" in pair or "tech_equity" in pair):
                val = 0.25
            elif "commodity" in pair and ("bond" in pair or "short_bond" in pair):
                val = 0.05
            elif "crypto" in pair:
                val = 0.10
            else:
                val = 0.20
            corr[i, j] = corr[j, i] = val
    return corr


def _cov_and_corr(
    cov_matrix: Optional[Any],
    buckets: List[str],
) -> Tuple[np.ndarray, np.ndarray, ValidationResult, bool]:
    n = len(buckets)
    fallback_used = False
    validation = ValidationResult(ok=True)

    cov = None
    if cov_matrix is not None:
        try:
            candidate = np.asarray(cov_matrix, dtype=float)
            if candidate.shape == (n, n):
                validation, cov = validate_psd(candidate, label="forward_sim_cov", fix_if_violation=True)
            else:
                validation.add("forward_sim_cov_shape", "warn", f"cov shape {candidate.shape} incompatible with n={n}")
        except Exception as exc:  # noqa: BLE001 - fallback is safer than aborting the report
            validation.add("forward_sim_cov_parse", "warn", repr(exc))

    if cov is None or cov.shape != (n, n) or not np.isfinite(cov).all():
        fallback_used = True
        vols = np.asarray([_fallback_vol(b) for b in buckets], dtype=float)
        corr = _fallback_corr(buckets)
        cov = np.outer(vols, vols) * corr
        validation, cov = validate_psd(cov, label="forward_sim_fallback_cov", fix_if_violation=True)

    vols = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.maximum(np.outer(vols, vols), 1e-12)
    corr = np.clip(corr, -0.99, 0.99)
    np.fill_diagonal(corr, 1.0)
    corr_validation, corr = validate_psd(corr, label="forward_sim_corr", fix_if_violation=True)
    validation.findings.extend(corr_validation.findings)
    return cov, corr, validation, fallback_used


def _scenario_priors() -> Dict[str, float]:
    priors = dict(DEFAULT_SCENARIO_PRIORS)
    raw = os.environ.get("EISAX_PHASE_H_SCENARIO_PRIORS")
    if raw:
        try:
            override = json.loads(raw)
            if isinstance(override, Mapping):
                for name in SCENARIO_NAMES:
                    if name in override:
                        priors[name] = max(0.0, float(override[name]))
        except Exception:
            priors = dict(DEFAULT_SCENARIO_PRIORS)
    total = sum(priors.get(name, 0.0) for name in SCENARIO_NAMES)
    if total <= 0:
        return dict(DEFAULT_SCENARIO_PRIORS)
    return {name: priors.get(name, 0.0) / total for name in SCENARIO_NAMES}


def _chol_psd(matrix: np.ndarray) -> np.ndarray:
    jitter = 1e-10
    for _ in range(5):
        try:
            return np.linalg.cholesky(matrix + np.eye(matrix.shape[0]) * jitter)
        except np.linalg.LinAlgError:
            jitter *= 10
    _, fixed = validate_psd(matrix, label="forward_sim_chol", fix_if_violation=True)
    return np.linalg.cholesky(fixed + np.eye(matrix.shape[0]) * jitter)


def _scenario_adjusted_mu(mu: np.ndarray, buckets: List[str], scenario: str) -> np.ndarray:
    spec = SCENARIO_SPECS[scenario]
    adj = np.zeros_like(mu)
    for i, bucket in enumerate(buckets):
        if bucket == "tech_equity":
            adj[i] = float(spec.get("tech", spec.get("equity", 0.0)))
        elif bucket == "equity":
            adj[i] = float(spec.get("equity", 0.0))
        elif bucket in {"bond", "short_bond"}:
            adj[i] = float(spec.get("bond", 0.0))
        elif bucket == "commodity":
            adj[i] = float(spec.get("commodity", 0.0))
    return mu + adj


def _path_outcomes(
    values: np.ndarray,
    *,
    start_value: float,
    horizon_years: float,
    inflation_assumption_pct: float,
    target_real_return: float,
) -> Tuple[ScenarioOutcome, Dict[str, np.ndarray]]:
    real_discount = (1.0 + inflation_assumption_pct / 100.0) ** max(horizon_years, 0.0)
    terminal_real = values[:, -1] / max(real_discount, 1e-12)
    terminal_gain = (terminal_real / max(start_value, 1e-12) - 1.0) * 100.0

    peaks = np.maximum.accumulate(values, axis=1)
    drawdowns = values / np.maximum(peaks, 1e-12) - 1.0
    max_dd = np.min(drawdowns, axis=1) * 100.0
    trough_idx = np.argmin(drawdowns, axis=1)
    peak_at_trough = peaks[np.arange(values.shape[0]), trough_idx]
    month_idx = np.arange(values.shape[1])[None, :]
    recovered = (month_idx > trough_idx[:, None]) & (values >= peak_at_trough[:, None])
    has_recovery = recovered.any(axis=1)
    first_recovery = np.argmax(recovered, axis=1)
    recovery_months = np.where(has_recovery, first_recovery - trough_idx, 0).astype(float)

    target_gain = ((1.0 + target_real_return) ** horizon_years - 1.0) * 100.0
    outcome = ScenarioOutcome(
        terminal_p10=float(np.percentile(terminal_gain, 10)),
        terminal_p50=float(np.percentile(terminal_gain, 50)),
        terminal_p90=float(np.percentile(terminal_gain, 90)),
        max_dd_p50_pct=float(np.percentile(max_dd, 50)),
        recovery_months_p50=float(np.percentile(recovery_months, 50)),
        prob_loss=float(np.mean(terminal_real < start_value)),
        prob_target=float(np.mean(terminal_gain >= target_gain)),
    )
    arrays = {
        "terminal_gain": terminal_gain,
        "terminal_real": terminal_real,
        "max_dd": max_dd,
        "recovery_months": recovery_months,
    }
    return outcome, arrays


def _simulate_one_scenario(
    *,
    rng: np.random.Generator,
    weights_arr: np.ndarray,
    mu_annual: np.ndarray,
    vols_annual: np.ndarray,
    corr: np.ndarray,
    buckets: List[str],
    scenario: str,
    horizon_years: float,
    paths: int,
    start_value: float,
    contributions_per_year: float,
    withdrawal_per_year: float,
    inflation_assumption_pct: float,
    target_real_return: float,
) -> Tuple[ScenarioOutcome, Dict[str, np.ndarray]]:
    steps = max(1, int(round(horizon_years * 12)))
    n = weights_arr.size
    spec = SCENARIO_SPECS[scenario]

    stress = np.full((n, n), 0.85, dtype=float)
    if scenario == "liquidity_crisis":
        stress[:] = 0.98
    np.fill_diagonal(stress, 1.0)
    base_chol = _chol_psd(corr)
    stress_chol = _chol_psd(stress)

    scenario_mu = _scenario_adjusted_mu(mu_annual, buckets, scenario)
    monthly_mu = scenario_mu / 12.0
    sigma_long = np.maximum(vols_annual * float(spec["sigma"]), 0.001)
    sigma_t = np.broadcast_to(sigma_long, (paths, n)).copy()

    values = np.empty((paths, steps + 1), dtype=float)
    values[:, 0] = start_value
    monthly_flow = (contributions_per_year - withdrawal_per_year) / 12.0
    current = np.full(paths, start_value, dtype=float)

    jump_scale = np.zeros(n, dtype=float)
    jump_mean = np.full(n, -0.05, dtype=float)
    for i, bucket in enumerate(buckets):
        if bucket in {"cash", "short_bond", "bond"}:
            jump_scale[i] = 0.0
        elif bucket in {"crypto", "commodity"}:
            jump_scale[i] = 0.05
            jump_mean[i] = -0.10
        else:
            jump_scale[i] = 0.03

    for step in range(1, steps + 1):
        if monthly_flow:
            current = np.maximum(current + monthly_flow, 1e-9)

        base_norm = rng.standard_normal((paths, n)) @ base_chol.T
        stress_norm = rng.standard_normal((paths, n)) @ stress_chol.T
        beta_w = np.clip(rng.beta(2.0, 18.0, size=(paths, 1)) + float(spec["stress"]), 0.0, 1.0)
        shocks = np.sqrt(1.0 - beta_w) * base_norm + np.sqrt(beta_w) * stress_norm
        chi = rng.chisquare(6.0, size=(paths, 1))
        shocks = shocks * np.sqrt(6.0 / np.maximum(chi, 1e-12)) * math.sqrt((6.0 - 2.0) / 6.0)

        vol_z = rng.standard_normal((paths, n))
        sigma_t = np.clip(
            sigma_t * np.exp(0.05 * (1.0 - sigma_t / sigma_long) + 0.10 * vol_z),
            0.4 * sigma_long,
            3.0 * sigma_long,
        )
        asset_returns = monthly_mu + (sigma_t / math.sqrt(12.0)) * shocks

        if np.any(jump_scale > 0):
            mask = rng.poisson(0.02, size=(paths, n)) > 0
            jump_mag = rng.normal(jump_mean, jump_scale, size=(paths, n))
            asset_returns = asset_returns + np.where(mask, jump_mag, 0.0)

        port_ret = asset_returns @ weights_arr
        current = np.maximum(current * (1.0 + port_ret), 1e-9)
        values[:, step] = current

    return _path_outcomes(
        values,
        start_value=start_value,
        horizon_years=horizon_years,
        inflation_assumption_pct=inflation_assumption_pct,
        target_real_return=target_real_return,
    )


@memoize("forward_scenario")
def _simulate_forward_payload(
    *,
    weights: Dict[str, float],
    expected_returns: Optional[Dict[str, float]],
    cov_matrix: Optional[Any],
    horizon_years: float,
    contributions_per_year: float,
    withdrawal_per_year: float,
    inflation_assumption_pct: float,
    paths_per_scenario: int,
    seed: int,
    start_value: float,
) -> Dict[str, Any]:
    names, weights_arr = _weights_array(weights)
    if weights_arr.size == 0:
        return {"payload": ForwardScenario(), "validation": ValidationResult(ok=False), "fallback_used": True}

    buckets = [_asset_bucket(name) for name in names]
    mu = _expected_returns(names, buckets, expected_returns)
    cov, corr, validation, fallback_used = _cov_and_corr(cov_matrix, buckets)
    vols = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    priors = _scenario_priors()
    target_real_return = float(os.environ.get("EISAX_PHASE_H_TARGET_REAL_RETURN", "0.04"))
    rng = np.random.default_rng(seed)

    scenarios: Dict[str, ScenarioOutcome] = {}
    pooled: Dict[str, List[np.ndarray]] = {
        "terminal_gain": [],
        "terminal_real": [],
        "max_dd": [],
        "recovery_months": [],
    }
    for name in SCENARIO_NAMES:
        outcome, arrays = _simulate_one_scenario(
            rng=rng,
            weights_arr=weights_arr,
            mu_annual=mu,
            vols_annual=vols,
            corr=corr,
            buckets=buckets,
            scenario=name,
            horizon_years=horizon_years,
            paths=paths_per_scenario,
            start_value=start_value,
            contributions_per_year=contributions_per_year,
            withdrawal_per_year=withdrawal_per_year,
            inflation_assumption_pct=inflation_assumption_pct,
            target_real_return=target_real_return,
        )
        outcome["name"] = name
        outcome["probability"] = priors[name]
        scenarios[name] = outcome
        for key, arr in arrays.items():
            pooled[key].append(arr)

    pooled_arrays = {key: np.concatenate(parts) for key, parts in pooled.items()}
    path_weights = np.concatenate([
        np.full(paths_per_scenario, priors[name] / max(paths_per_scenario, 1), dtype=float)
        for name in SCENARIO_NAMES
    ])
    path_weights = path_weights / path_weights.sum()
    choice = rng.choice(pooled_arrays["terminal_gain"].size, size=pooled_arrays["terminal_gain"].size, replace=True, p=path_weights)
    terminal_gain = pooled_arrays["terminal_gain"][choice]
    terminal_real = pooled_arrays["terminal_real"][choice]
    max_dd = pooled_arrays["max_dd"][choice]
    recovery_months = pooled_arrays["recovery_months"][choice]
    target_gain = ((1.0 + target_real_return) ** horizon_years - 1.0) * 100.0
    aggregate = ScenarioOutcome(
        name="aggregate",
        probability=1.0,
        terminal_p10=float(np.percentile(terminal_gain, 10)),
        terminal_p50=float(np.percentile(terminal_gain, 50)),
        terminal_p90=float(np.percentile(terminal_gain, 90)),
        max_dd_p50_pct=float(np.percentile(max_dd, 50)),
        recovery_months_p50=float(np.percentile(recovery_months, 50)),
        prob_loss=float(np.mean(terminal_real < start_value)),
        prob_target=float(np.mean(terminal_gain >= target_gain)),
    )
    dd_p10 = float(np.percentile(max_dd, 10))
    dd_p90 = float(np.percentile(max_dd, 90))

    payload = ForwardScenario(
        horizon_years=float(horizon_years),
        contributions_per_year=float(contributions_per_year),
        withdrawal_per_year=float(withdrawal_per_year),
        inflation_assumption_pct=float(inflation_assumption_pct),
        scenarios=scenarios,
        aggregate=aggregate,
        seed=int(seed),
        paths_simulated=int(paths_per_scenario * len(SCENARIO_NAMES)),
        notes=[
            "Forward simulation uses Student-t monthly shocks, stochastic volatility, regime priors, and jump terms.",
        ],
    )
    payload["aggregate_details"] = {
        "terminal_value_p10_multiple": float(1.0 + aggregate["terminal_p10"] / 100.0),
        "terminal_value_p50_multiple": float(1.0 + aggregate["terminal_p50"] / 100.0),
        "terminal_value_p90_multiple": float(1.0 + aggregate["terminal_p90"] / 100.0),
        "drawdown_p10_pct": dd_p10,
        "drawdown_p90_pct": dd_p90,
        "target_real_return": target_real_return,
    }
    return {"payload": payload, "validation": validation, "fallback_used": fallback_used}


def run_forward_simulation(
    weights: Dict[str, float],
    expected_returns: Optional[Dict[str, float]] = None,
    cov_matrix: Optional[Any] = None,
    horizon_years: float = 5.0,
    contributions_per_year: float = 0.0,
    withdrawal_per_year: float = 0.0,
    inflation_assumption_pct: float = 2.0,
    paths_per_scenario: int = 2000,
    seed: Optional[int] = None,
    language: str = "en",
    port_value_usd: float = 100000.0,
) -> Dict[str, Any]:
    """Run the H3 Monte Carlo engine and return a versioned Phase H envelope."""
    if not _enabled():
        return {}

    used_seed = int(seed if seed is not None else FeatureRegistry.get("phase_h_seed"))
    start_value = max(float(port_value_usd or 100000.0), 1.0)
    sim = _simulate_forward_payload(
        weights=dict(weights or {}),
        expected_returns=dict(expected_returns or {}) if isinstance(expected_returns, Mapping) else None,
        cov_matrix=cov_matrix,
        horizon_years=float(horizon_years or 5.0),
        contributions_per_year=float(contributions_per_year or 0.0),
        withdrawal_per_year=float(withdrawal_per_year or 0.0),
        inflation_assumption_pct=float(inflation_assumption_pct or 2.0),
        paths_per_scenario=max(1, int(paths_per_scenario or 2000)),
        seed=used_seed,
        start_value=start_value,
    )
    payload = sim.get("payload") or {}
    notes = list(payload.get("notes", []) if isinstance(payload, Mapping) else [])
    envelope = make_envelope(
        "forward_scenario",
        payload,
        validation=sim.get("validation"),
        fallback_used=bool(sim.get("fallback_used")),
        notes=notes,
    )
    if isinstance(payload, Mapping):
        envelope.update(payload)
    return envelope


def _signed_pct(value: Any, decimals: int = 0) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "-"
    sign = "+" if f > 0 else ""
    return f"{sign}{f:.{decimals}f}%"


def _prob_pct(value: Any, decimals: int = 0) -> str:
    try:
        return f"{float(value) * 100.0:.{decimals}f}%"
    except (TypeError, ValueError):
        return "-"


def _money_multiple(value: Any) -> str:
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "$-"


def _range_word(language: str) -> str:
    return "إلى" if language == "ar" else "to"


def _months_label(value: Any, language: str) -> str:
    months = fmt_num(value, 0)
    return f"{months} أشهر" if language == "ar" else f"{months} months"


def render_forward_scenario_md(payload: ForwardScenario, language: str = "en") -> str:
    if not _enabled() or not payload:
        return ""
    data = unwrap(payload)
    if not data:
        data = dict(payload)

    headers = [
        L("scenario", language),
        L("probability", language),
        L("terminal_p10", language),
        L("terminal_p50", language),
        L("terminal_p90", language),
        L("max_drawdown", language),
        L("recovery_months", language),
    ]
    rows: List[List[str]] = []
    for name in SCENARIO_NAMES:
        sc = (data.get("scenarios") or {}).get(name, {})
        rows.append([
            name.replace("_", " "),
            _prob_pct(sc.get("probability"), 1),
            _signed_pct(sc.get("terminal_p10")),
            _signed_pct(sc.get("terminal_p50")),
            _signed_pct(sc.get("terminal_p90")),
            _signed_pct(sc.get("max_dd_p50_pct")),
            fmt_num(sc.get("recovery_months_p50"), 1),
        ])
    table = md_table(headers, rows)

    horizon = fmt_num(data.get("horizon_years"), 1)
    inflation = fmt_pct(data.get("inflation_assumption_pct"), 1)
    if language == "ar":
        horizon_line = (
            f"*الأفق: {horizon} سنوات · التضخم: {inflation} · "
            f"البذرة: {data.get('seed', '-')} · المسارات: {data.get('paths_simulated', '-')}*"
        )
    else:
        horizon_line = (
            f"*Horizon: {horizon}y · Inflation: {inflation} · "
            f"Seed: {data.get('seed', '-')} · Paths: {data.get('paths_simulated', '-')}*"
        )

    aggregate = data.get("aggregate") or {}
    details = data.get("aggregate_details") or {}
    target = float(details.get("target_real_return", os.environ.get("EISAX_PHASE_H_TARGET_REAL_RETURN", "0.04"))) * 100.0
    target_label = L("prob_target", language).replace("4%", f"{target:.0f}%")
    dd_low = aggregate.get("max_dd_p50_pct")
    dd_high = details.get("drawdown_p10_pct", dd_low)
    agg_rows = [
        [
            L("expected_terminal_range", language),
            f"{_money_multiple(details.get('terminal_value_p10_multiple', 1 + float(aggregate.get('terminal_p10', 0) or 0) / 100.0))} - "
            f"{_money_multiple(details.get('terminal_value_p90_multiple', 1 + float(aggregate.get('terminal_p90', 0) or 0) / 100.0))} (P10-P90)",
        ],
        [L("prob_loss", language), _prob_pct(aggregate.get("prob_loss"))],
        [target_label, _prob_pct(aggregate.get("prob_target"))],
        [L("worst_decile", language), _money_multiple(details.get("terminal_value_p10_multiple", 1 + float(aggregate.get("terminal_p10", 0) or 0) / 100.0))],
        [L("expected_dd_range", language), f"{_signed_pct(dd_low)} {_range_word(language)} {_signed_pct(dd_high)}"],
        [L("recovery_duration_median", language), _months_label(aggregate.get("recovery_months_p50"), language)],
    ]
    aggregate_table = md_table([L("metric", language), L("value", language)], agg_rows)
    disclaimer = L("distributional_disclaimer", language)

    out = (
        f"## H. {L('forward_scenario', language)}\n\n"
        f"{horizon_line}\n\n"
        f"{table}\n\n"
        f"**{L('aggregate_block_title', language)}**\n\n"
        f"{aggregate_table}\n\n"
        f"*{disclaimer}*\n"
    )
    return scrub_block(_probabilistic_lint(out))


__all__ = [
    "ENGINE_VERSION",
    "SCENARIO_NAMES",
    "DEFAULT_SCENARIO_PRIORS",
    "DETERMINISTIC_PATTERNS",
    "run_forward_simulation",
    "render_forward_scenario_md",
    "_probabilistic_lint",
]
