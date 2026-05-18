"""
Phase H — centralized Feature Registry.

Replaces scattered env-var lookups with a single, queryable registry.
Env vars still drive defaults so the existing ops surface keeps working,
but downstream code consults the registry instead of repeatedly calling
`os.environ.get(...)`. This:

- gives a single audit-able snapshot of every flag state
- lets tests / committee-mode overrides toggle features without env hacks
- centralises naming so engines don't drift apart on flag conventions
- supports versioned defaults (future migrations)

Usage:
    from phase_h.registry import FeatureRegistry
    if FeatureRegistry.is_enabled("phase_h_benchmark"):
        ...

For audit:
    FeatureRegistry.snapshot()  # dict suitable for the audit appendix
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip()


@dataclass
class Feature:
    """A single named feature flag."""
    key: str
    description: str
    kind: str                                       # "bool" | "int" | "float" | "str"
    default: Any
    env_name: Optional[str] = None
    loader: Optional[Callable[[Any], Any]] = None   # custom coercion when needed
    category: str = "phase_h"


# ──────────────────────────────────────────────────────────────────────
# Catalog — single source of truth
# ──────────────────────────────────────────────────────────────────────

_CATALOG: Dict[str, Feature] = {}


def _register(feature: Feature) -> None:
    if feature.key in _CATALOG:
        raise ValueError(f"duplicate feature key: {feature.key}")
    _CATALOG[feature.key] = feature


# Phase H — master + engines
_register(Feature(
    key="phase_h_enabled", description="Master switch for Phase H pipeline",
    kind="bool", default=True, env_name="EISAX_PHASE_H_ENABLED",
))
_register(Feature(
    key="phase_h_benchmark", description="H1 — Native Benchmark Analytics",
    kind="bool", default=True, env_name="EISAX_PHASE_H_BENCHMARK",
))
_register(Feature(
    key="phase_h_tc_optimizer", description="H2 — Transaction-Cost-Aware Optimizer",
    kind="bool", default=True, env_name="EISAX_PHASE_H_TC_OPTIMIZER",
))
_register(Feature(
    key="phase_h_forward_sim", description="H3 — Forward Multi-Period Simulation",
    kind="bool", default=True, env_name="EISAX_PHASE_H_FORWARD_SIM",
))
_register(Feature(
    key="phase_h_factor_model", description="H4 — True Factor Model Engine",
    kind="bool", default=True, env_name="EISAX_PHASE_H_FACTOR_MODEL",
))
_register(Feature(
    key="phase_h_committee", description="H5 — Investment Committee Mode",
    kind="bool", default=True, env_name="EISAX_PHASE_H_COMMITTEE",
))
_register(Feature(
    key="phase_h_tone_guard", description="Tone-guard scrubber (forbidden phrases + emoji)",
    kind="bool", default=True, env_name="EISAX_PHASE_H_TONE_GUARD",
))
_register(Feature(
    key="phase_h_seed", description="Deterministic Monte Carlo seed",
    kind="int", default=42, env_name="EISAX_PHASE_H_DETERMINISTIC_SEED",
))

# Numerical safety
_register(Feature(
    key="numerics_hard_assert", description="Fail-loud on PSD / dimension violation",
    kind="bool", default=True, env_name="EISAX_NUMERICS_HARD_ASSERT", category="numerics",
))
_register(Feature(
    key="numerics_psd_fix", description="Auto-clip eigenvalues to PSD if soft path",
    kind="bool", default=True, env_name="EISAX_NUMERICS_PSD_FIX", category="numerics",
))

# Latency / cache
_register(Feature(
    key="cache_enabled", description="Enable phase_h.cache memoization layer",
    kind="bool", default=True, env_name="EISAX_PHASE_H_CACHE", category="latency",
))
_register(Feature(
    key="cache_ttl_seconds", description="Default TTL for cached engine outputs",
    kind="int", default=900, env_name="EISAX_PHASE_H_CACHE_TTL", category="latency",
))

# Committee mode (string)
_register(Feature(
    key="committee_mode", description="Active committee brief mode",
    kind="str", default="", env_name="EISAX_COMMITTEE_MODE", category="phase_h",
))


# ──────────────────────────────────────────────────────────────────────
# Registry API
# ──────────────────────────────────────────────────────────────────────

class _Registry:
    """Thread-safe singleton interface."""
    _lock = threading.RLock()
    _overrides: Dict[str, Any] = {}

    @classmethod
    def get(cls, key: str) -> Any:
        with cls._lock:
            if key in cls._overrides:
                return cls._overrides[key]
            feat = _CATALOG.get(key)
            if feat is None:
                raise KeyError(f"unknown feature: {key}")
            env = feat.env_name
            if feat.kind == "bool":
                return _env_bool(env, bool(feat.default)) if env else bool(feat.default)
            if feat.kind == "int":
                return _env_int(env, int(feat.default)) if env else int(feat.default)
            if feat.kind == "float":
                return _env_float(env, float(feat.default)) if env else float(feat.default)
            if feat.kind == "str":
                return _env_str(env, str(feat.default)) if env else str(feat.default)
            return feat.default

    @classmethod
    def is_enabled(cls, key: str) -> bool:
        val = cls.get(key)
        if isinstance(val, bool):
            return val
        # non-bool keys default to truthy
        if isinstance(val, (int, float)):
            return val != 0
        if isinstance(val, str):
            return bool(val)
        return bool(val)

    @classmethod
    def override(cls, key: str, value: Any) -> None:
        """Test-only / committee-mode runtime override. Use sparingly."""
        with cls._lock:
            if key not in _CATALOG:
                raise KeyError(f"unknown feature: {key}")
            cls._overrides[key] = value

    @classmethod
    def reset_override(cls, key: Optional[str] = None) -> None:
        with cls._lock:
            if key is None:
                cls._overrides.clear()
            else:
                cls._overrides.pop(key, None)

    @classmethod
    def snapshot(cls) -> Dict[str, Any]:
        """All current values, keyed by feature key. For audit appendix."""
        with cls._lock:
            return {k: cls.get(k) for k in _CATALOG}

    @classmethod
    def catalog(cls) -> Dict[str, Feature]:
        return dict(_CATALOG)

    @classmethod
    def by_category(cls, category: str) -> Dict[str, Any]:
        return {k: cls.get(k) for k, f in _CATALOG.items() if f.category == category}


FeatureRegistry = _Registry()


__all__ = ["Feature", "FeatureRegistry"]
