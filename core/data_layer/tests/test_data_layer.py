"""
core.data_layer.tests.test_data_layer — smoke + contract suite.

Runs without external network. Verifies:
    1. Package imports cleanly + version constant present.
    2. Feature flags register into FeatureRegistry under category "data_layer".
    3. MarketCacheAdapter lists markets + returns snapshots from the cache.
    4. get_universe_panel returns the requested columns for a known ticker.
    5. Benchmarks catalog ≥ 15 entries, each shape-valid.
    6. Trading calendars accept Sun-Thu for GCC, Mon-Fri for US.
    7. Liquidity profiles: tier_of bands + slippage estimator non-negative.
    8. GCC metadata: every curated entry passes validate_entry().
    9. Factor panels metadata roundtrip.
   10. Sector classification normalises a few aliases.
"""

from __future__ import annotations

import datetime as dt
import sys
import traceback
from typing import List, Tuple


def _run(name: str, fn) -> Tuple[str, bool, str]:
    try:
        fn()
        return (name, True, "")
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().splitlines()[-2:]
        return (name, False, f"{exc} :: {' | '.join(tb)}")


# 1
def test_package_imports():
    import core.data_layer as dl
    assert hasattr(dl, "DATA_LAYER_VERSION")
    assert dl.DATA_LAYER_VERSION


# 2
def test_feature_registry_registration():
    from core.data_layer import _flags  # noqa: F401 — import side-effect
    from phase_h.registry import FeatureRegistry
    cat = FeatureRegistry.by_category("data_layer")
    assert "data_layer_enabled" in cat, f"missing master flag, got {list(cat)}"
    assert "data_layer_gcc_metadata" in cat
    assert "data_layer_liquidity_profile" in cat
    assert FeatureRegistry.is_enabled("data_layer_enabled") is True


# 3
def test_market_cache_adapter_lists_markets():
    from core.data_layer import list_markets, get_latest_snapshot, snapshot_timestamp
    markets = list_markets()
    assert isinstance(markets, tuple)
    assert len(markets) > 0, "expected ≥ 1 market in cache"
    sample = markets[0]
    df = get_latest_snapshot(sample)
    assert df is not None and len(df) > 0
    ts = snapshot_timestamp(sample)
    assert ts is not None


# 4
def test_universe_panel_columns():
    from core.data_layer import get_universe_panel, list_markets, get_latest_snapshot
    markets = list_markets()
    assert markets, "no markets to derive a ticker from"
    df = get_latest_snapshot(markets[0])
    sample_tickers = list(df["ticker"].head(3))
    panel = get_universe_panel(sample_tickers, columns=["ticker", "close", "_market"])
    assert panel is not None
    assert set(panel.columns) >= {"ticker", "close", "_market"}
    assert len(panel) >= 1


# 5
def test_benchmark_catalog_shape():
    from core.data_layer import BENCHMARK_CATALOG, get_benchmark, get_benchmark_series
    assert len(BENCHMARK_CATALOG) >= 15
    for tk, meta in BENCHMARK_CATALOG.items():
        for k in ("label", "region", "asset_class", "source_market"):
            assert k in meta, f"{tk} missing {k}"
    spy = get_benchmark("SPY")
    assert spy and spy["region"] == "US"
    series = get_benchmark_series("SPY")
    assert series["ticker"] == "SPY"


# 6
def test_trading_calendar_basics():
    from core.data_layer import is_trading_day, market_calendar
    # 2026-05-17 is a Sunday — trading day for GCC, weekend for US.
    sunday = dt.date(2026, 5, 17)
    assert is_trading_day(sunday, "KSA") is True
    assert is_trading_day(sunday, "US") is False
    cal = market_calendar("US")
    assert "weekdays" in cal


# 7
def test_liquidity_profiles():
    from core.data_layer import (
        tier_of, LIQUIDITY_TIER_1, LIQUIDITY_TIER_2, LIQUIDITY_TIER_3,
        get_liquidity_profile, estimate_slippage_bps,
    )
    assert tier_of(60_000_000)["code"] == LIQUIDITY_TIER_1["code"]
    assert tier_of(20_000_000)["code"] == LIQUIDITY_TIER_2["code"]
    assert tier_of(1_000_000)["code"] == LIQUIDITY_TIER_3["code"]
    assert tier_of(None)["code"] == LIQUIDITY_TIER_3["code"]
    profile = get_liquidity_profile(ticker="SPY")
    assert "tier" in profile and "region_multiplier" in profile
    est = estimate_slippage_bps("SPY", 250_000)
    assert est["slippage_bps"] >= 0.0
    assert "components" in est


# 8
def test_gcc_metadata_curated_entries():
    from core.data_layer import GCC_METADATA, get_gcc_metadata, list_gcc_tickers
    from core.data_layer.gcc_metadata import validate_entry, provenance_summary
    assert len(GCC_METADATA) >= 12, f"expected ≥ 12 curated entries, got {len(GCC_METADATA)}"
    for tk, payload in GCC_METADATA.items():
        missing = validate_entry(payload)
        assert not missing, f"{tk} missing fields {missing}"
    aramco = get_gcc_metadata("TADAWUL:2222")
    # Provenance-aware schema: every field is
    #   {value, as_of_date, source_type, confidence, data_quality, methodology, fallback_used}
    assert aramco["country"]["value"] == "KSA"
    assert aramco["country"]["data_quality"] == "verified"
    assert aramco["country"]["source_type"] in {"issuer", "exchange", "regulator"}
    assert aramco["country"]["as_of_date"]  # ISO date present
    assert aramco["source"] == "curated"
    # Quantitative fields without an authoritative source remain "missing".
    assert aramco["government_ownership_pct"]["data_quality"] == "missing"
    assert aramco["government_ownership_pct"]["source_type"] == "missing"
    assert aramco["government_ownership_pct"]["fallback_used"] is True
    # Bare-match path
    aramco_bare = get_gcc_metadata("2222")
    assert aramco_bare["source"].startswith("curated")
    # Fallback path never raises and produces a fully-missing entry
    missing_entry = get_gcc_metadata("DOES:NOT_EXIST")
    assert missing_entry["source"] in {"fallback", "feature_disabled"}
    assert missing_entry["country"]["data_quality"] == "missing"
    # KSA filter works against the new nested schema
    ksa_entries = [GCC_METADATA[t] for t in list_gcc_tickers("KSA")]
    assert all(e["country"]["value"] == "KSA" for e in ksa_entries)
    # Provenance summary returns counts across the new enum vocabulary
    summary = provenance_summary(aramco)
    assert summary["verified"] >= 1
    assert summary["missing"] >= 1
    assert summary["tier_1"] >= 1  # at least one Tier-1 verified record


# 9
def test_factor_panels():
    from core.data_layer import get_factor_panel, list_factor_models
    from core.data_layer.factor_premia import panel_metadata
    assert "ff5" in list_factor_models()
    meta = panel_metadata("ff5")
    assert "rows" in meta


# 10
def test_sector_normalisation():
    from core.data_layer import sector_classification
    assert sector_classification("Finance") == "Financials"
    assert sector_classification("Health Technology") == "Health Care"
    assert sector_classification(None) == "Unknown"


CASES = [
    ("package_imports",              test_package_imports),
    ("feature_registry_registration",test_feature_registry_registration),
    ("market_cache_adapter",         test_market_cache_adapter_lists_markets),
    ("universe_panel_columns",       test_universe_panel_columns),
    ("benchmark_catalog_shape",      test_benchmark_catalog_shape),
    ("trading_calendar_basics",      test_trading_calendar_basics),
    ("liquidity_profiles",           test_liquidity_profiles),
    ("gcc_metadata_curated_entries", test_gcc_metadata_curated_entries),
    ("factor_panels",                test_factor_panels),
    ("sector_normalisation",         test_sector_normalisation),
]


def main() -> int:
    results = [_run(n, fn) for n, fn in CASES]
    fails = [(n, msg) for n, ok, msg in results if not ok]
    for name, ok, msg in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' :: ' + msg) if msg else ''}")
    print()
    if fails:
        print(f"data_layer tests: {len(fails)}/{len(results)} FAILED")
        return 1
    print(f"data_layer tests: {len(results)}/{len(results)} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
