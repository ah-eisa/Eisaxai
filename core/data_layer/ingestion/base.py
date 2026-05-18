"""
core.data_layer.ingestion.base — ingestion contract primitives.

Each `IngestionRecord` represents one (ticker, field) value sourced from
exactly one document. Records are grouped into an `IngestionRun` so the
audit trail captures who/when/why a batch of changes was proposed.

The validators here mirror the strict enum vocabulary used by
`gcc_metadata.MetadataField` — anything else is rejected with
`IngestionError`.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..gcc_metadata import (
    SOURCE_TYPES,
    DATA_QUALITY_LEVELS,
    MetadataField,
)


class IngestionError(ValueError):
    """Raised when an `IngestionRecord` fails contract validation."""


@dataclass(frozen=True)
class IngestionRecord:
    """One ticker × one field, sourced from exactly one document."""
    ticker: str
    field: str
    value: Any
    source_type: str            # must be one of SOURCE_TYPES
    source_url: str             # canonical reference URL
    document_id: str            # filing id / page hash
    captured_at: str            # ISO date the source was captured
    captured_by: str            # reviewer initials or "automated:<job>"
    confidence: float           # 0..1
    methodology: str            # human-readable derivation
    data_quality: str = "verified"  # must be one of DATA_QUALITY_LEVELS

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IngestionRun:
    """A batch of records produced by a single ingestion job."""
    run_id: str
    started_at: str
    reviewer: str
    records: List[IngestionRecord] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add(self, rec: IngestionRecord) -> None:
        validate_record(rec)
        self.records.append(rec)

    def __len__(self) -> int:
        return len(self.records)


def validate_record(rec: IngestionRecord) -> None:
    """Reject records that violate the enum / range contract."""
    if rec.source_type not in SOURCE_TYPES:
        raise IngestionError(f"invalid source_type {rec.source_type!r}")
    if rec.data_quality not in DATA_QUALITY_LEVELS:
        raise IngestionError(f"invalid data_quality {rec.data_quality!r}")
    if not (0.0 <= float(rec.confidence) <= 1.0):
        raise IngestionError(f"confidence out of range: {rec.confidence}")
    if not rec.ticker or ":" not in rec.ticker:
        raise IngestionError(f"ticker must be EXCHANGE:SYMBOL, got {rec.ticker!r}")
    if not rec.field:
        raise IngestionError("field is required")
    if not rec.source_url:
        raise IngestionError("source_url is required for auditability")
    if not rec.captured_at or not rec.captured_by:
        raise IngestionError("captured_at and captured_by must be set")


def record_to_metadata_field(rec: IngestionRecord) -> MetadataField:
    """Convert a validated record into a MetadataField for the registry."""
    validate_record(rec)
    return MetadataField(
        value=rec.value,
        as_of_date=rec.captured_at,
        source_type=rec.source_type,
        confidence=float(rec.confidence),
        data_quality=rec.data_quality,
        methodology=rec.methodology,
        fallback_used=False,
    )


def audit_run(run: IngestionRun) -> Dict[str, Any]:
    """
    Produce a deterministic audit report for a run.

    The report digest is a SHA256 of the canonical JSON form so reviewers
    can sign/verify each ingestion batch independently.
    """
    payload = {
        "run_id": run.run_id,
        "started_at": run.started_at,
        "reviewer": run.reviewer,
        "record_count": len(run.records),
        "records": [r.as_dict() for r in run.records],
        "notes": list(run.notes),
    }
    canon = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    by_source: Dict[str, int] = {}
    by_quality: Dict[str, int] = {}
    for r in run.records:
        by_source[r.source_type] = by_source.get(r.source_type, 0) + 1
        by_quality[r.data_quality] = by_quality.get(r.data_quality, 0) + 1
    return {
        "payload": payload,
        "digest": digest,
        "summary": {
            "records":   len(run.records),
            "by_source": by_source,
            "by_quality": by_quality,
        },
    }


def new_run(reviewer: str) -> IngestionRun:
    """Convenience: open a new run stamped with the current ISO timestamp."""
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return IngestionRun(
        run_id=hashlib.sha1(f"{reviewer}:{started}".encode("utf-8")).hexdigest()[:16],
        started_at=started,
        reviewer=reviewer,
    )


__all__ = [
    "IngestionError",
    "IngestionRecord",
    "IngestionRun",
    "validate_record",
    "record_to_metadata_field",
    "audit_run",
    "new_run",
]
