"""
Phase H feature flags.

All flags read from environment at import time and are picklable
constants for downstream consumers. Override at runtime by reloading
this module if absolutely necessary.

Master switch: EISAX_PHASE_H_ENABLED. If 0/false, every engine no-ops
and the existing Phase A-G output is returned unchanged.
"""

from __future__ import annotations

import os
from typing import Dict


def _flag(name: str, default: str = "1") -> bool:
    raw = os.environ.get(name, default).strip().lower()
    return raw in {"1", "true", "yes", "on", "y", "t"}


def _int_flag(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except (TypeError, ValueError):
        return default


PHASE_H_ENABLED            = _flag("EISAX_PHASE_H_ENABLED", "1")
PHASE_H_BENCHMARK          = _flag("EISAX_PHASE_H_BENCHMARK", "1")
PHASE_H_TC_OPTIMIZER       = _flag("EISAX_PHASE_H_TC_OPTIMIZER", "1")
PHASE_H_FORWARD_SIM        = _flag("EISAX_PHASE_H_FORWARD_SIM", "1")
PHASE_H_FACTOR_MODEL       = _flag("EISAX_PHASE_H_FACTOR_MODEL", "1")
PHASE_H_COMMITTEE          = _flag("EISAX_PHASE_H_COMMITTEE", "1")
PHASE_H_TONE_GUARD         = _flag("EISAX_PHASE_H_TONE_GUARD", "1")
PHASE_H_DETERMINISTIC_SEED = _int_flag("EISAX_PHASE_H_DETERMINISTIC_SEED", 42)


def flag_state_snapshot() -> Dict[str, object]:
    """Return all Phase H flag states for inclusion in the audit appendix."""
    return {
        "EISAX_PHASE_H_ENABLED":            PHASE_H_ENABLED,
        "EISAX_PHASE_H_BENCHMARK":          PHASE_H_BENCHMARK,
        "EISAX_PHASE_H_TC_OPTIMIZER":       PHASE_H_TC_OPTIMIZER,
        "EISAX_PHASE_H_FORWARD_SIM":        PHASE_H_FORWARD_SIM,
        "EISAX_PHASE_H_FACTOR_MODEL":       PHASE_H_FACTOR_MODEL,
        "EISAX_PHASE_H_COMMITTEE":          PHASE_H_COMMITTEE,
        "EISAX_PHASE_H_TONE_GUARD":         PHASE_H_TONE_GUARD,
        "EISAX_PHASE_H_DETERMINISTIC_SEED": PHASE_H_DETERMINISTIC_SEED,
    }
