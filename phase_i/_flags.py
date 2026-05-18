"""
phase_i._flags — register Phase I feature flags into the existing
FeatureRegistry under category "phase_i". Idempotent.
"""

from __future__ import annotations

from phase_h.registry import Feature  # type: ignore[attr-defined]


def _register_if_absent(feat: Feature) -> None:
    from phase_h import registry as _registry  # late binding (survives reloads)
    cat = _registry._CATALOG  # type: ignore[attr-defined]
    if feat.key not in cat:
        cat[feat.key] = feat


_FLAGS = [
    Feature(
        key="phase_i_enabled",
        description="Master switch for the Phase I context graph",
        kind="bool", default=True,
        env_name="EISAX_PHASE_I_ENABLED", category="phase_i",
    ),
    Feature(
        key="phase_i_strict_review",
        description="When True, queries hide edges with review_status != 'approved'",
        kind="bool", default=False,
        env_name="EISAX_PHASE_I_STRICT_REVIEW", category="phase_i",
    ),
    Feature(
        key="phase_i_min_tier",
        description="Minimum provenance tier (1-4) for edges returned to consumers",
        kind="int", default=4,
        env_name="EISAX_PHASE_I_MIN_TIER", category="phase_i",
    ),
]


def register() -> None:
    for f in _FLAGS:
        _register_if_absent(f)


register()


__all__ = ["register"]
