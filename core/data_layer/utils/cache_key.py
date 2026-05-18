"""
Stable cache-key construction for data-layer reads.

Delegates to phase_h.cache.cache_key so the data layer and engine layer
share one key namespace — keys are content-addressed SHA256 hashes that
remain stable under dict re-ordering.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from phase_h.cache import cache_key as _phase_h_cache_key


def stable_cache_key(namespace: str, **inputs: Any) -> str:
    """Build a stable SHA256 cache key for the given namespace + kwargs."""
    return _phase_h_cache_key(namespace, **inputs)


def normalise_inputs(items: Iterable[Any]) -> tuple:
    """Return a tuple suitable for use as a cache-key input (sorted, hashable)."""
    out = []
    for x in items:
        if isinstance(x, Mapping):
            out.append(tuple(sorted((str(k), str(v)) for k, v in x.items())))
        elif isinstance(x, (list, tuple, set)):
            out.append(tuple(sorted(str(v) for v in x)))
        else:
            out.append(str(x))
    return tuple(sorted(out))


__all__ = ["stable_cache_key", "normalise_inputs"]
