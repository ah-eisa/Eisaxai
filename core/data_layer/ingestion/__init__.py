"""
core.data_layer.ingestion — provenance-audited GCC metadata ingestion.

Spec: /home/ubuntu/investwise/gcc_ingestion_spec.md

This package never writes into market_cache/. It produces, validates,
and audits provenance-tagged metadata records that target
`core.data_layer.seed.*` modules. The seed registry is the authoritative
source; the ingestion layer is the auditable producer for it.

Public surface kept narrow:
    - IngestionRecord     — one ticker × one field × one provenance source
    - IngestionRun        — a batch of records with a single run-id
    - validate_record     — schema validation against the source-type enum
    - audit_run           — produces a deterministic audit report
"""

from __future__ import annotations

from .base import (
    IngestionRecord,
    IngestionRun,
    IngestionError,
    validate_record,
    record_to_metadata_field,
    audit_run,
)

__all__ = [
    "IngestionRecord",
    "IngestionRun",
    "IngestionError",
    "validate_record",
    "record_to_metadata_field",
    "audit_run",
]
