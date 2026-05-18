"""
Deterministic, envelope-compatible wrapper for data-layer outputs.

Phase H's `make_envelope` stamps `produced_at` with the current UTC time.
Because the data layer is a pure read-only projection over a snapshotted
cache, its outputs must NOT vary with wall-clock time — otherwise:
    1. Cache-key derived hashes would drift between processes.
    2. Snapshot regression tests would never match.
    3. Audit reproducibility would silently break.

Strategy: call `make_envelope` to get the canonical shape, then overwrite
`produced_at` with one of:
    - the underlying snapshot timestamp (`snapshot_ts`), when known; or
    - the literal sentinel `data_layer:deterministic`.

Either replacement is independent of wall-clock time, satisfying the
"deterministic and envelope-compatible" rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from phase_h.contracts import make_envelope

from ..base import DETERMINISTIC_PRODUCED_AT


@dataclass(frozen=True)
class VersionedRecord:
    """Lightweight container — wraps payload + provenance fields."""
    source: str
    version: str
    payload: Mapping[str, Any]
    snapshot_ts: Optional[str] = None
    notes: tuple = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "version": self.version,
            "payload": dict(self.payload),
            "snapshot_ts": self.snapshot_ts,
            "notes": list(self.notes),
        }


def embed_version(
    *,
    engine: str,
    payload: Mapping[str, Any],
    data_layer_version: str,
    notes: Optional[List[str]] = None,
    fallback_used: bool = False,
    snapshot_ts: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a Phase H envelope around `payload` and normalise `produced_at`
    so the result is deterministic.

    `snapshot_ts`, when provided, is preferred over the sentinel — this
    preserves a useful audit trail (the report can show "data snapshot:
    2026-05-17T12:58Z" instead of an opaque placeholder).
    """
    enriched_notes = list(notes or []) + [f"data_layer_version={data_layer_version}"]
    envelope = make_envelope(
        engine=engine,
        payload=payload,
        validation=None,
        fallback_used=fallback_used,
        notes=enriched_notes,
    )
    # Hard rule: never expose a wall-clock timestamp from the data layer.
    envelope["produced_at"] = snapshot_ts or DETERMINISTIC_PRODUCED_AT
    envelope["deterministic"] = True
    return envelope


def is_deterministic_envelope(envelope: Mapping[str, Any]) -> bool:
    """True iff `envelope` was minted by this module (or carries an equivalent sentinel)."""
    if not isinstance(envelope, Mapping):
        return False
    if envelope.get("deterministic") is True:
        return True
    pa = envelope.get("produced_at")
    return pa == DETERMINISTIC_PRODUCED_AT or (isinstance(pa, str) and pa.startswith("data_layer:"))


__all__ = ["VersionedRecord", "embed_version", "is_deterministic_envelope"]
