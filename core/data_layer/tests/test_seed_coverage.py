"""
core.data_layer.tests.test_seed_coverage — gates Phase I implementation.

Enforces gcc_ingestion_spec.md §4 coverage thresholds and §3 provenance
discipline:

    1. Per-market counts: ≥25 KSA, ≥25 UAE, ≥5 KW, ≥5 QA   (60 total)
    2. Every entry has ≥3 Tier-1 fields (country / exchange / sector minimum)
    3. No `data_quality="estimated"` claim without a `methodology` string
    4. seed_audit returns ok=True (schema findings empty)
    5. Every entry passes MetadataField enum validation per field
"""

from __future__ import annotations

import sys
import traceback
from typing import List, Tuple


def _run(name, fn):
    try:
        fn()
        return (name, True, "")
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().splitlines()[-2:]
        return (name, False, f"{exc} :: {' | '.join(tb)}")


# 1 — per-market coverage
def test_per_market_counts():
    from core.data_layer.seed import coverage_summary
    cov = coverage_summary()
    assert cov["ksa"]    >= 25, f"KSA seed count {cov['ksa']} below 25"
    assert cov["uae"]    >= 25, f"UAE seed count {cov['uae']} below 25"
    assert cov["kuwait"] >= 5,  f"Kuwait seed count {cov['kuwait']} below 5"
    assert cov["qatar"]  >= 5,  f"Qatar seed count {cov['qatar']} below 5"
    assert sum(cov.values()) >= 60


# 2 — tier-1 floor for every entry
def test_tier1_floor_per_entry():
    from core.data_layer.gcc_metadata import GCC_METADATA, provenance_summary
    weak = []
    for tk, entry in GCC_METADATA.items():
        ps = provenance_summary(entry)
        if ps["tier_1"] < 3:
            weak.append((tk, ps["tier_1"]))
    assert not weak, f"entries below Tier-1 floor: {weak[:5]} ..."


# 3 — every estimated claim must carry a methodology
def test_estimated_claims_have_methodology():
    from core.data_layer.gcc_metadata import GCC_METADATA, SCHEMA_FIELDS
    bad = []
    for tk, entry in GCC_METADATA.items():
        for f in SCHEMA_FIELDS:
            v = entry.get(f)
            if not isinstance(v, dict):
                continue
            if v.get("data_quality") == "estimated" and not v.get("methodology"):
                bad.append((tk, f))
    assert not bad, f"estimated claims without methodology: {bad}"


# 4 — seed_audit returns ok
def test_seed_audit_passes():
    from core.data_layer.ingestion.runners.seed_audit import run_audit
    report = run_audit()
    assert report["ok"], f"seed_audit failed: {report['schema_findings'][:5]}"
    assert report["summary"]["registry_size"] >= 60


# 5 — strict enum validation per field
def test_strict_enum_validation():
    from core.data_layer.gcc_metadata import (
        GCC_METADATA, SOURCE_TYPES, DATA_QUALITY_LEVELS, SCHEMA_FIELDS,
    )
    violations = []
    for tk, entry in GCC_METADATA.items():
        for f in SCHEMA_FIELDS:
            v = entry.get(f)
            if not isinstance(v, dict):
                continue
            if v.get("source_type") not in SOURCE_TYPES:
                violations.append((tk, f, "source_type", v.get("source_type")))
            if v.get("data_quality") not in DATA_QUALITY_LEVELS:
                violations.append((tk, f, "data_quality", v.get("data_quality")))
    assert not violations, f"enum violations: {violations[:5]}"


# 6 — manifest snapshot loader round-trip
def test_manifest_loader_roundtrip():
    from core.data_layer.ingestion.sources.manifest_loader import load_all
    from core.data_layer.ingestion import record_to_metadata_field
    records = load_all("issuer")
    assert len(records) >= 1
    # Every record must convert to a valid MetadataField
    for r in records:
        mf = record_to_metadata_field(r)
        assert mf.data_quality in {"verified", "derived", "estimated", "missing"}
        assert mf.fallback_used is False


CASES = [
    ("per_market_counts",            test_per_market_counts),
    ("tier1_floor_per_entry",        test_tier1_floor_per_entry),
    ("estimated_claims_methodology", test_estimated_claims_have_methodology),
    ("seed_audit_passes",            test_seed_audit_passes),
    ("strict_enum_validation",       test_strict_enum_validation),
    ("manifest_loader_roundtrip",    test_manifest_loader_roundtrip),
]


def main() -> int:
    results = [_run(n, fn) for n, fn in CASES]
    fails = [(n, msg) for n, ok, msg in results if not ok]
    for name, ok, msg in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' :: ' + msg) if msg else ''}")
    print()
    if fails:
        print(f"seed coverage: {len(fails)}/{len(results)} FAILED")
        return 1
    print(f"seed coverage: {len(results)}/{len(results)} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
