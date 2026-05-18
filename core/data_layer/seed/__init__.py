"""
core.data_layer.seed — curated GCC + Egypt metadata seed modules.

Each market lives in its own module so reviewers can sign off per-country
sets independently. Modules expose `ENTRIES: Dict[str, Dict[str, Any]]`
mapping `EXCHANGE:SYMBOL` to a provenance-aware metadata payload built
via `core.data_layer.gcc_metadata._entry`.

`gcc_metadata.GCC_METADATA` merges every seed module at import time.
Adding a new market = creating a new module here and listing it in
`SEED_MODULES` below.
"""

from __future__ import annotations

from typing import Dict, Any, Iterable

from . import ksa, uae, kuwait, qatar  # noqa: F401 — side-effect modules

SEED_MODULES = (ksa, uae, kuwait, qatar)


def build_registry() -> Dict[str, Dict[str, Any]]:
    """Merge every seed module's ENTRIES into a single registry dict."""
    out: Dict[str, Dict[str, Any]] = {}
    for mod in SEED_MODULES:
        entries = getattr(mod, "ENTRIES", {})
        for k, v in entries.items():
            if k in out:
                raise ValueError(f"duplicate seed ticker {k!r} across seed modules")
            out[k] = v
    return out


def coverage_summary() -> Dict[str, int]:
    """Per-market entry counts. Used by `test_seed_coverage`."""
    return {mod.__name__.rsplit(".", 1)[-1]: len(getattr(mod, "ENTRIES", {}))
            for mod in SEED_MODULES}


__all__ = ["SEED_MODULES", "build_registry", "coverage_summary"]
