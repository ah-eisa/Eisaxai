"""
Generic JSON-manifest loader.

A manifest file describes one ticker's verified fields, sourced from one
or more public documents. Schema:

    {
      "ticker": "TADAWUL:2222",
      "reviewer": "ah",
      "fields": [
        {
          "field": "government_ownership_pct",
          "value": 90.0,
          "source_type": "issuer",
          "source_url": "https://www.aramco.com/.../prospectus.pdf",
          "document_id": "aramco_ipo_2019_p124",
          "captured_at": "2026-05-18",
          "captured_by": "ah",
          "confidence": 0.95,
          "methodology": "ipo_prospectus_p124_table_a3",
          "data_quality": "verified"
        }
      ]
    }

The loader walks every JSON file under
`core/data_layer/ingestion/snapshots/<source>/` and returns
`IngestionRecord` entries.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List

from ..base import IngestionRecord, IngestionError, validate_record


SNAPSHOTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "snapshots"
)


def _walk_json(path: str) -> Iterable[str]:
    if not os.path.isdir(path):
        return
    for root, _dirs, files in os.walk(path):
        for fn in files:
            if fn.endswith(".json"):
                yield os.path.join(root, fn)


def load_manifest(path: str) -> List[IngestionRecord]:
    """Parse one manifest JSON into a list of validated records."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    ticker = data.get("ticker")
    if not ticker:
        raise IngestionError(f"{path}: manifest missing 'ticker'")
    records: List[IngestionRecord] = []
    for raw in data.get("fields", []):
        rec = IngestionRecord(
            ticker=ticker,
            field=raw["field"],
            value=raw["value"],
            source_type=raw["source_type"],
            source_url=raw["source_url"],
            document_id=raw["document_id"],
            captured_at=raw["captured_at"],
            captured_by=raw["captured_by"],
            confidence=float(raw["confidence"]),
            methodology=raw["methodology"],
            data_quality=raw.get("data_quality", "verified"),
        )
        validate_record(rec)
        records.append(rec)
    return records


def load_all(source_dir: str = "issuer") -> List[IngestionRecord]:
    """Load every manifest under snapshots/<source_dir>/."""
    base = os.path.join(SNAPSHOTS_ROOT, source_dir)
    out: List[IngestionRecord] = []
    for p in _walk_json(base):
        out.extend(load_manifest(p))
    return out


__all__ = ["SNAPSHOTS_ROOT", "load_manifest", "load_all"]
