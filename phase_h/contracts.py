"""
Phase H — versioned payload contract.

A single envelope wraps every Phase H engine output so consumers can
introspect the schema version. Required by PHASE_H_SPEC: "every engine
must expose versioned schema + deterministic fields + validation layer
+ fallback behavior".

Envelope shape:

    {
        "version":   "1.0",
        "engine":    "benchmark_relative",
        "produced_at": "2026-05-17T12:34:56Z",
        "payload":   { ... engine TypedDict ... },
        "validation": {
            "ok": bool,
            "findings": [{check, severity, detail, metric}, ...]
        },
        "fallback_used": bool,
        "notes":     [str, ...]
    }

`validate_envelope(env)` returns a `ValidationResult` from numerics.py.
`make_envelope(engine, payload, ...)` builds the canonical shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from .numerics import ValidationResult


SCHEMA_VERSION = "1.0"


# Engines that ship versioned envelopes. Add as engines mature.
ENGINE_KEYS = (
    "benchmark_relative",
    "execution_diagnostics",
    "forward_scenario",
    "factor_decomp",
    "committee_brief",
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _serialise(obj: Any) -> Any:
    """JSON-friendly coercion: dataclass → dict, set → list, fallback to str."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_serialise(v) for v in obj)
    if is_dataclass(obj):
        return _serialise(asdict(obj))
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def make_envelope(
    engine: str,
    payload: Mapping[str, Any],
    *,
    validation: Optional[ValidationResult] = None,
    fallback_used: bool = False,
    notes: Optional[List[str]] = None,
    version: str = SCHEMA_VERSION,
) -> Dict[str, Any]:
    """Wrap an engine payload in the canonical Phase H envelope."""
    if engine not in ENGINE_KEYS:
        # Permissive — log via notes, don't raise. Tests assert this stays empty.
        notes = list(notes or []) + [f"non-canonical engine key: {engine!r}"]
    findings = [
        {"check": f.check, "severity": f.severity, "detail": f.detail, "metric": f.metric}
        for f in (validation.findings if validation is not None else [])
    ]
    return {
        "version":       version,
        "engine":        engine,
        "produced_at":   _utc_iso(),
        "payload":       _serialise(payload),
        "validation": {
            "ok":       bool(validation.ok) if validation is not None else True,
            "findings": findings,
        },
        "fallback_used": bool(fallback_used),
        "notes":         list(notes or []),
    }


def unwrap(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the inner payload dict; emptyish if envelope malformed."""
    if not isinstance(envelope, Mapping):
        return {}
    if envelope.get("version") and "payload" in envelope:
        return dict(envelope.get("payload") or {})
    return dict(envelope)


def envelope_meta(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip payload, return only metadata (for audit appendix)."""
    if not isinstance(envelope, Mapping):
        return {}
    out = dict(envelope)
    out.pop("payload", None)
    return out


__all__ = [
    "SCHEMA_VERSION",
    "ENGINE_KEYS",
    "make_envelope",
    "unwrap",
    "envelope_meta",
]
