"""
core.data_layer — Institutional Data Layer for EisaX.

Single canonical source for every market-data, fundamental, factor,
liquidity, calendar, sector, GCC-metadata and macro lookup used by
Phase H engines (H1 benchmarks, H2 TC optimizer, H3 forward sim,
H4 factor model, H5 committee mode) and any future engine.

Design rules:
- All readers wrap the existing 15-minute parquet cache at
  /home/ubuntu/investwise/market_cache/ (never re-fetch from source).
- Every reader returns versioned envelopes via phase_h.contracts.make_envelope.
- Every reader honours FeatureRegistry gating (`data_layer.*` category).
- Heavy reads memoised via phase_h.cache.memoize (15-min TTL by default).
- Validators delegate to phase_h.numerics.
- No retail tone — phrasing scrubbed where the layer emits human strings.

Public façade kept intentionally narrow so engines depend on stable names.
"""

from __future__ import annotations

DATA_LAYER_VERSION = "0.1.0"

from .base import DataLayerError, RecordNotFound, StaleSnapshotError, get_layer_meta
from .market_cache_adapter import (
    MarketCacheAdapter,
    get_latest_snapshot,
    get_ticker_row,
    get_universe_panel,
    list_markets,
    snapshot_timestamp,
)
from .benchmarks import (
    BENCHMARK_CATALOG,
    get_benchmark,
    get_benchmark_series,
    list_benchmarks,
)
from .liquidity_profiles import (
    LIQUIDITY_TIER_1,
    LIQUIDITY_TIER_2,
    LIQUIDITY_TIER_3,
    get_liquidity_profile,
    get_adv,
    estimate_slippage_bps,
    tier_of,
)
from .trading_calendars import (
    is_trading_day,
    next_trading_day,
    previous_trading_day,
    trading_days_between,
    market_calendar,
)
from .gcc_metadata import (
    get_gcc_metadata,
    list_gcc_tickers,
    GCC_METADATA,
)
from .factor_premia import (
    get_factor_panel,
    list_factor_models,
)
from .macro_series import (
    get_macro_series,
    list_macro_keys,
)
from .sectors import (
    get_sector,
    list_sectors,
    sector_classification,
)

__all__ = [
    "DATA_LAYER_VERSION",
    "DataLayerError",
    "RecordNotFound",
    "StaleSnapshotError",
    "get_layer_meta",
    # market cache
    "MarketCacheAdapter",
    "get_latest_snapshot",
    "get_ticker_row",
    "get_universe_panel",
    "list_markets",
    "snapshot_timestamp",
    # benchmarks
    "BENCHMARK_CATALOG",
    "get_benchmark",
    "get_benchmark_series",
    "list_benchmarks",
    # liquidity
    "LIQUIDITY_TIER_1",
    "LIQUIDITY_TIER_2",
    "LIQUIDITY_TIER_3",
    "get_liquidity_profile",
    "get_adv",
    "estimate_slippage_bps",
    "tier_of",
    # calendars
    "is_trading_day",
    "next_trading_day",
    "previous_trading_day",
    "trading_days_between",
    "market_calendar",
    # GCC metadata
    "get_gcc_metadata",
    "list_gcc_tickers",
    "GCC_METADATA",
    # factor premia
    "get_factor_panel",
    "list_factor_models",
    # macro
    "get_macro_series",
    "list_macro_keys",
    # sectors
    "get_sector",
    "list_sectors",
    "sector_classification",
]
