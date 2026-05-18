"""
core.data_layer.ingestion.runners.seed_audit — validate the seed registry
against committed source manifests.

Two passes:
    1. Schema pass: every entry in `core.data_layer.seed.*` is shape-valid
       (validate_entry returns empty list) and uses only allowed enum values.
    2. Provenance pass: every Tier-1 / Tier-2 field that claims a non-
       fallback source must have a matching IngestionRecord in
       `ingestion/snapshots/<source>/`. Missing manifests are reported.

Run modes:
    --dry-run (default): prints results, no writes.
    --json: emits the audit report as JSON to stdout.

Invocation:
    /home/ubuntu/investwise/venv/bin/python3 \
        -m core.data_layer.ingestion.runners.seed_audit
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

from ..base import audit_run, new_run, IngestionRecord
from ..sources.manifest_loader import load_all, SNAPSHOTS_ROOT
from ...gcc_metadata import (
    GCC_METADATA, SCHEMA_FIELDS, validate_entry, provenance_summary,
)


def _index_manifests() -> Dict[str, List[IngestionRecord]]:
    """ticker → list of records discovered across all source dirs."""
    out: Dict[str, List[IngestionRecord]] = {}
    if not os.path.isdir(SNAPSHOTS_ROOT):
        return out
    for d in sorted(os.listdir(SNAPSHOTS_ROOT)):
        sub = os.path.join(SNAPSHOTS_ROOT, d)
        if not os.path.isdir(sub):
            continue
        for rec in load_all(d):
            out.setdefault(rec.ticker, []).append(rec)
    return out


def run_audit() -> Dict[str, object]:
    """Run both passes. Returns a structured audit report."""
    schema_findings: List[str] = []
    coverage_findings: List[Dict[str, object]] = []
    summary: Dict[str, int] = {
        "registry_size":   len(GCC_METADATA),
        "tier_1_total":    0,
        "tier_4_total":    0,
        "verified_total":  0,
        "missing_total":   0,
    }

    for ticker, entry in GCC_METADATA.items():
        missing_keys = validate_entry(entry)
        if missing_keys:
            schema_findings.append(f"{ticker}: missing schema fields {missing_keys}")
        ps = provenance_summary(entry)
        summary["tier_1_total"]   += ps["tier_1"]
        summary["tier_4_total"]   += ps["tier_4"]
        summary["verified_total"] += ps["verified"]
        summary["missing_total"]  += ps["missing"]

    manifests = _index_manifests()
    summary["manifest_records"] = sum(len(v) for v in manifests.values())
    summary["manifest_tickers"] = len(manifests)

    # Every Tier-1/Tier-2 field with `fallback_used=False` should ideally
    # have a manifest record. We REPORT (not fail) — the seed is itself a
    # provenance source for the country/exchange/sector fields.
    for ticker, entry in GCC_METADATA.items():
        for field_name in SCHEMA_FIELDS:
            field_val = entry.get(field_name)
            if not isinstance(field_val, dict):
                continue
            if field_val.get("fallback_used", True):
                continue
            if field_val.get("data_quality") not in {"verified", "derived"}:
                continue
            recs = [r for r in manifests.get(ticker, []) if r.field == field_name]
            if not recs:
                coverage_findings.append({
                    "ticker": ticker,
                    "field": field_name,
                    "issue": "no_manifest_record",
                    "source_type": field_val.get("source_type"),
                    "as_of_date": field_val.get("as_of_date"),
                })

    return {
        "ok": not schema_findings,
        "summary": summary,
        "schema_findings": schema_findings,
        "coverage_findings": coverage_findings,
    }


def main(argv: List[str]) -> int:
    as_json = "--json" in argv
    report = run_audit()
    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        s = report["summary"]
        print("SEED AUDIT")
        print(f"  registry size   : {s['registry_size']}")
        print(f"  verified fields : {s['verified_total']}")
        print(f"  missing fields  : {s['missing_total']}")
        print(f"  tier 1 fields   : {s['tier_1_total']}")
        print(f"  tier 4 fields   : {s['tier_4_total']}")
        print(f"  manifest records: {s['manifest_records']} ({s['manifest_tickers']} tickers)")
        if report["schema_findings"]:
            print()
            print("SCHEMA FINDINGS:")
            for f in report["schema_findings"][:20]:
                print(f"  - {f}")
        if report["coverage_findings"]:
            print()
            print(f"COVERAGE FINDINGS: {len(report['coverage_findings'])} fields without manifest records")
            print("  (first 5 shown — these are informational, not failures)")
            for f in report["coverage_findings"][:5]:
                print(f"  - {f['ticker']}.{f['field']}  source_type={f['source_type']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
