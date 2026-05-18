"""
core.data_layer.base — shared exceptions, constants, and meta helpers.

Cache-path constants live in `market_cache_adapter.py` (rule: the literal
"market_cache" path appears only inside the single adapter module).
"""

from __future__ import annotations

import time
from typing import Any, Dict

DEFAULT_SNAPSHOT_TTL_SECONDS = 900  # 15-minute cache cadence
STALE_SNAPSHOT_HARD_LIMIT_SECONDS = 6 * 3600  # 6h — anything older is unusable

GCC_MARKETS = ("uae", "ksa", "egypt", "kuwait", "qatar", "bahrain", "morocco", "tunisia")
DEVELOPED_MARKETS = ("america",)
ALTERNATIVE_MARKETS = ("crypto", "commodities")
ALL_MARKETS = GCC_MARKETS + DEVELOPED_MARKETS + ALTERNATIVE_MARKETS

# Deterministic sentinel timestamp used in place of "now" in envelopes
# produced by the data layer. The layer is a pure projection over a
# read-only cache so its outputs must not vary with wall-clock time.
DETERMINISTIC_PRODUCED_AT = "data_layer:deterministic"


class DataLayerError(Exception):
    """Top-level exception for data layer faults."""


class RecordNotFound(DataLayerError):
    """Requested ticker / benchmark / panel does not exist in the cache."""


class StaleSnapshotError(DataLayerError):
    """Latest snapshot is older than the hard-stale limit."""


class MalformedPayloadError(DataLayerError):
    """Underlying cache row exists but cannot be coerced into the expected shape."""


def get_layer_meta() -> Dict[str, Any]:
    """Return a small dict describing the layer state — used by audit appendix."""
    from . import DATA_LAYER_VERSION  # local import to avoid cycle
    return {
        "version": DATA_LAYER_VERSION,
        "snapshot_ttl_seconds": DEFAULT_SNAPSHOT_TTL_SECONDS,
        "stale_limit_seconds": STALE_SNAPSHOT_HARD_LIMIT_SECONDS,
        "markets": list(ALL_MARKETS),
        "observed_at": int(time.time()),
    }


__all__ = [
    "DEFAULT_SNAPSHOT_TTL_SECONDS",
    "STALE_SNAPSHOT_HARD_LIMIT_SECONDS",
    "GCC_MARKETS",
    "DEVELOPED_MARKETS",
    "ALTERNATIVE_MARKETS",
    "ALL_MARKETS",
    "DETERMINISTIC_PRODUCED_AT",
    "DataLayerError",
    "RecordNotFound",
    "StaleSnapshotError",
    "MalformedPayloadError",
    "get_layer_meta",
]
