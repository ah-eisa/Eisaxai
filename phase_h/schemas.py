"""
Phase H typed schemas.

Plain TypedDicts (Python 3.10+) to keep zero runtime overhead and
preserve JSON-serialisability for the audit appendix. Engines should
construct these via dict literals and rely on type checking only at
development time.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class BenchmarkRelative(TypedDict, total=False):
    benchmark_ticker: str
    benchmark_label: str
    active_return_pct: float
    tracking_error_pct: float
    information_ratio: float
    rolling_alpha_12m_pct: float
    rolling_beta_12m: float
    relative_drawdown_pct: float
    upside_capture: float
    downside_capture: float
    relative_volatility: float
    active_share_pct: float
    style_drift: str
    excess_decomp: Dict[str, float]      # allocation / selection / factor / concentration
    regime_behavior: Dict[str, List[str]]  # outperform_envs, lag_envs
    reliability_tier: str
    notes: List[str]


class ExecutionDiagnostics(TypedDict, total=False):
    turnover_pct: float
    implementation_shortfall_bp: float
    market_impact_bp: float
    slippage_bp: float
    complexity_tier: str
    liquidity_stress: str
    tax_note: str
    rebalance_frequency: str
    turnover_penalty_applied: bool
    quadratic_penalty_applied: bool
    persistence_preference_pct: float
    notes: List[str]


class ScenarioOutcome(TypedDict, total=False):
    name: str
    probability: float
    terminal_p10: float
    terminal_p50: float
    terminal_p90: float
    max_dd_p50_pct: float
    recovery_months_p50: float
    prob_loss: float
    prob_target: float


class ForwardScenario(TypedDict, total=False):
    horizon_years: float
    contributions_per_year: float
    withdrawal_per_year: float
    inflation_assumption_pct: float
    scenarios: Dict[str, ScenarioOutcome]
    aggregate: ScenarioOutcome
    seed: int
    paths_simulated: int
    notes: List[str]


class FactorDecomp(TypedDict, total=False):
    model: str
    loadings: Dict[str, float]
    t_stats: Dict[str, float]
    contribution_return: Dict[str, float]
    contribution_vol: Dict[str, float]
    contribution_drawdown: Dict[str, float]
    r_squared: float
    rolling_stability: float
    warnings: List[str]
    reliability_tier: str
    notes: List[str]


class CommitteeExhibit(TypedDict, total=False):
    number: int
    title: str
    payload_ref: str
    rendered_md: str


class CommitteeBrief(TypedDict, total=False):
    mode: str
    headline: str
    key_decision: str
    key_risks: List[str]
    positioning: str
    implementation_notes: str
    mandate_summary: str
    top_vulnerabilities: List[str]
    challenge_scenarios: List[str]
    exhibits: List[CommitteeExhibit]
    notes: List[str]


class PhaseHMeta(TypedDict, total=False):
    version: str
    flags: Dict[str, Any]
    seed: int
    engines_ran: List[str]
    engine_versions: Dict[str, str]
    audit_hashes: Dict[str, str]
