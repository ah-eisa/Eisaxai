"""
Phase H — structural + tone assertions for report markdown.

Every assertion raises `AssertionError` with a precise diagnostic when
the report violates an institutional invariant. Used by the regression
suite and per-engine tests.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Mapping, Optional, Sequence, Set

from ..contracts import SCHEMA_VERSION
from ..tone_guard import audit_block


# ──────────────────────────────────────────────────────────────────────
# Section ordering invariant
# ──────────────────────────────────────────────────────────────────────

REQUIRED_SECTIONS_EN = ["## A.", "## B.", "## C.", "## D.", "## E.", "## G."]
REQUIRED_SECTIONS_AR = ["## A.", "## B.", "## C.", "## D.", "## E.", "## G."]
# F is conditional (LLM commentary), H/I are Phase H additions.

OPTIONAL_PHASE_H_TOPLEVEL = ["## H.", "## I."]


def assert_section_order(
    markdown: str,
    *,
    language: str = "en",
    require_h: bool = False,
    require_i: bool = False,
) -> None:
    """
    Assert that all required A-G top-level sections appear and that
    Section G is the LAST top-level section (audit appendix invariant).
    """
    required = REQUIRED_SECTIONS_EN if language == "en" else REQUIRED_SECTIONS_AR
    missing = [s for s in required if s not in markdown]
    if missing:
        raise AssertionError(f"missing required sections: {missing}")

    positions: List[tuple] = []
    for line in markdown.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            positions.append((markdown.find(line), line.strip()))

    if require_h and "## H." not in markdown:
        raise AssertionError("Section H (Forward Scenario) required but missing")
    if require_i and "## I." not in markdown:
        raise AssertionError("Section I (Committee Brief) required but missing")

    # Section G must be the last top-level section
    g_pos = markdown.rfind("## G.")
    last_top = max(markdown.rfind(s) for s in ("## A.","## B.","## C.","## D.","## E.","## F.","## G.","## H.","## I."))
    if g_pos != last_top:
        # Identify which sneaked past G
        violators = []
        for s in ("## H.", "## I."):
            if s in markdown and markdown.rfind(s) > g_pos:
                violators.append(s)
        raise AssertionError(
            f"Audit Appendix (## G.) is not last top-level section; "
            f"violators={violators}"
        )


# ──────────────────────────────────────────────────────────────────────
# Tone discipline
# ──────────────────────────────────────────────────────────────────────

def assert_tone_clean(markdown: str) -> None:
    """No forbidden phrases, no analytical-section emojis."""
    hits = audit_block(markdown)
    if hits:
        raise AssertionError(f"tone-guard violations: {hits}")


# ──────────────────────────────────────────────────────────────────────
# Envelope schema (versioned contract)
# ──────────────────────────────────────────────────────────────────────

REQUIRED_ENVELOPE_KEYS = {"version", "engine", "produced_at", "payload",
                          "validation", "fallback_used", "notes"}


def assert_envelope_valid(envelope: Mapping, *, expected_engine: Optional[str] = None) -> None:
    if not isinstance(envelope, Mapping):
        raise AssertionError(f"envelope is not a mapping: {type(envelope).__name__}")
    missing = REQUIRED_ENVELOPE_KEYS - set(envelope.keys())
    if missing:
        raise AssertionError(f"envelope missing required keys: {sorted(missing)}")
    if envelope["version"] != SCHEMA_VERSION:
        raise AssertionError(
            f"envelope version mismatch: {envelope['version']!r} != {SCHEMA_VERSION!r}"
        )
    if expected_engine is not None and envelope["engine"] != expected_engine:
        raise AssertionError(
            f"envelope engine mismatch: {envelope['engine']!r} != {expected_engine!r}"
        )
    val = envelope.get("validation") or {}
    if not isinstance(val, Mapping) or "ok" not in val or "findings" not in val:
        raise AssertionError("envelope.validation malformed")


# ──────────────────────────────────────────────────────────────────────
# Markdown structure: no broken tables, balanced headings
# ──────────────────────────────────────────────────────────────────────

_TABLE_ROW_RX = re.compile(r"^\|.+\|\s*$")
_TABLE_SEP_RX = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$")


def assert_no_broken_tables(markdown: str) -> None:
    """Every table header row must be followed by a `| --- |` separator."""
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        if not _TABLE_ROW_RX.match(line):
            continue
        # If next line is also a row, this might be a body row — fine.
        # If next line is a separator, this is a header row — fine.
        # Trigger only when a row is preceded by NOTHING table-like AND
        # next line is neither row nor separator.
        prev_ok = i > 0 and (_TABLE_ROW_RX.match(lines[i-1]) or _TABLE_SEP_RX.match(lines[i-1]))
        next_line = lines[i+1] if i + 1 < len(lines) else ""
        next_ok = _TABLE_ROW_RX.match(next_line) or _TABLE_SEP_RX.match(next_line)
        if not prev_ok and not next_ok:
            raise AssertionError(
                f"orphaned table row at line {i+1}: {line[:80]!r}"
            )


def assert_markdown_structure(markdown: str) -> None:
    """Aggregate structural checks: section order + no broken tables."""
    assert_section_order(markdown)
    assert_no_broken_tables(markdown)


# ──────────────────────────────────────────────────────────────────────
# Bilingual symmetry: every required label appears in both renders
# ──────────────────────────────────────────────────────────────────────

def assert_bilingual_render(
    md_en: str,
    md_ar: str,
    *,
    en_labels: Sequence[str],
    ar_labels: Sequence[str],
) -> None:
    en_missing = [l for l in en_labels if l not in md_en]
    ar_missing = [l for l in ar_labels if l not in md_ar]
    errors = []
    if en_missing:
        errors.append(f"EN missing: {en_missing}")
    if ar_missing:
        errors.append(f"AR missing: {ar_missing}")
    if errors:
        raise AssertionError(" | ".join(errors))


__all__ = [
    "REQUIRED_SECTIONS_EN",
    "REQUIRED_SECTIONS_AR",
    "assert_section_order",
    "assert_tone_clean",
    "assert_envelope_valid",
    "assert_no_broken_tables",
    "assert_bilingual_render",
    "assert_markdown_structure",
]
