"""
EisaX Phase H — Institutional Portfolio Intelligence Expansion.

Public API surface. All sub-engines are imported lazily so that
flag-off paths or partial deployments do not crash the existing
A-G pipeline.

See PHASE_H_SPEC.md at repo root for the full architecture contract.
"""

from .feature_flags import (
    PHASE_H_ENABLED,
    PHASE_H_BENCHMARK,
    PHASE_H_TC_OPTIMIZER,
    PHASE_H_FORWARD_SIM,
    PHASE_H_FACTOR_MODEL,
    PHASE_H_COMMITTEE,
    PHASE_H_TONE_GUARD,
    PHASE_H_DETERMINISTIC_SEED,
    flag_state_snapshot,
)

from .schemas import (
    BenchmarkRelative,
    ExecutionDiagnostics,
    ForwardScenario,
    FactorDecomp,
    CommitteeBrief,
    PhaseHMeta,
)

PHASE_H_VERSION = "0.1.0"

__all__ = [
    "PHASE_H_VERSION",
    "PHASE_H_ENABLED",
    "PHASE_H_BENCHMARK",
    "PHASE_H_TC_OPTIMIZER",
    "PHASE_H_FORWARD_SIM",
    "PHASE_H_FACTOR_MODEL",
    "PHASE_H_COMMITTEE",
    "PHASE_H_TONE_GUARD",
    "PHASE_H_DETERMINISTIC_SEED",
    "flag_state_snapshot",
    "BenchmarkRelative",
    "ExecutionDiagnostics",
    "ForwardScenario",
    "FactorDecomp",
    "CommitteeBrief",
    "PhaseHMeta",
]
