"""
Phase H — audit appendix extensions.

Builds the additional rows / hashes / seeds that should appear inside
the existing "## G. Audit Appendix" section so reproducibility holds.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional

from . import PHASE_H_VERSION
from .feature_flags import (
    PHASE_H_DETERMINISTIC_SEED,
    flag_state_snapshot,
)
from .report_helpers import L, md_table


def _hash_payload(payload: Any) -> str:
    """Stable hash for any JSON-serialisable payload (falls back to repr)."""
    try:
        raw = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        raw = repr(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_meta(
    engines_ran: Iterable[str],
    engine_versions: Optional[Dict[str, str]] = None,
    engine_payloads: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the PhaseHMeta payload to attach to the allocate() result dict.

    `engine_payloads` is a dict of {engine_name: payload} used to compute
    a deterministic hash per engine for the audit appendix.
    """
    versions = dict(engine_versions or {})
    versions.setdefault("phase_h", PHASE_H_VERSION)

    hashes: Dict[str, str] = {}
    for name, payload in (engine_payloads or {}).items():
        hashes[name] = _hash_payload(payload)

    return {
        "version": PHASE_H_VERSION,
        "flags": flag_state_snapshot(),
        "seed": PHASE_H_DETERMINISTIC_SEED,
        "engines_ran": list(engines_ran),
        "engine_versions": versions,
        "audit_hashes": hashes,
    }


def render_audit_rows(meta: Dict[str, Any], language: str = "en") -> str:
    """
    Return a small markdown table of Phase H reproducibility metadata
    to be appended INSIDE the existing audit appendix section.
    """
    if not meta:
        return ""

    flag_str = ", ".join(
        f"{k.replace('EISAX_PHASE_H_', '').lower()}={'on' if v is True else ('off' if v is False else v)}"
        for k, v in (meta.get("flags") or {}).items()
    )

    rows: List[List[str]] = [
        ["Phase H version", str(meta.get("version", PHASE_H_VERSION))],
        ["Seed",            str(meta.get("seed", PHASE_H_DETERMINISTIC_SEED))],
        ["Engines ran",     ", ".join(meta.get("engines_ran", [])) or "—"],
        ["Flags",           flag_str or "—"],
    ]
    for name, h in (meta.get("audit_hashes") or {}).items():
        rows.append([f"{name} hash", h])
    for name, v in (meta.get("engine_versions") or {}).items():
        rows.append([f"{name} version", v])

    headers = [L("metric", language), L("value", language)]
    title = "Phase H Reproducibility" if language == "en" else "إعادة إنتاج المرحلة H"
    return f"\n**{title}**\n\n" + md_table(headers, rows) + "\n"


__all__ = ["build_meta", "render_audit_rows"]
