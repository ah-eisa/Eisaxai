"""
core.data_layer._flags — registers data-layer feature flags into the
existing Phase H FeatureRegistry at import time.

This is additive: we never overwrite a Phase H key, and we never modify
phase_h/*. The flags live under category "data_layer" so they can be
listed via FeatureRegistry.by_category("data_layer") for audit.
"""

from __future__ import annotations

from phase_h.registry import Feature, FeatureRegistry  # type: ignore[attr-defined]


def _register_if_absent(feat: Feature) -> None:
    # Re-import _CATALOG every call — phase_h.registry may be reloaded
    # during test runs (committee case does importlib.reload), which
    # produces a fresh _CATALOG that an import-time binding would miss.
    from phase_h import registry as _registry  # type: ignore[import-not-found]
    cat = _registry._CATALOG  # type: ignore[attr-defined]
    if feat.key in cat:
        return
    cat[feat.key] = feat


_FLAGS = [
    Feature(
        key="data_layer_enabled",
        description="Master switch for the institutional data layer",
        kind="bool", default=True,
        env_name="EISAX_DATA_LAYER_ENABLED", category="data_layer",
    ),
    Feature(
        key="data_layer_cache_ttl",
        description="Default TTL (seconds) for data-layer memoised reads",
        kind="int", default=900,
        env_name="EISAX_DATA_LAYER_CACHE_TTL", category="data_layer",
    ),
    Feature(
        key="data_layer_strict_stale",
        description="Hard-fail when latest snapshot is older than stale limit",
        kind="bool", default=False,
        env_name="EISAX_DATA_LAYER_STRICT_STALE", category="data_layer",
    ),
    Feature(
        key="data_layer_gcc_metadata",
        description="Expose enriched GCC metadata (government_ownership, oil_beta, shariah, …)",
        kind="bool", default=True,
        env_name="EISAX_DATA_LAYER_GCC_METADATA", category="data_layer",
    ),
    Feature(
        key="data_layer_liquidity_profile",
        description="Expose tier-aware liquidity profiles + slippage estimator",
        kind="bool", default=True,
        env_name="EISAX_DATA_LAYER_LIQUIDITY", category="data_layer",
    ),
    Feature(
        key="data_layer_factor_panels",
        description="Expose FF3 / FF5 / Carhart factor panels (read-only)",
        kind="bool", default=True,
        env_name="EISAX_DATA_LAYER_FACTOR", category="data_layer",
    ),
    Feature(
        key="data_layer_calendars",
        description="Expose GCC Sun–Thu and US Mon–Fri trading calendars",
        kind="bool", default=True,
        env_name="EISAX_DATA_LAYER_CALENDARS", category="data_layer",
    ),
    Feature(
        key="data_layer_macro_series",
        description="Expose macro series (DXY, brent, sofr, …) — placeholder",
        kind="bool", default=True,
        env_name="EISAX_DATA_LAYER_MACRO", category="data_layer",
    ),
]


def register() -> None:
    for f in _FLAGS:
        _register_if_absent(f)


# Register on import — idempotent.
register()


__all__ = ["register", "FeatureRegistry"]
