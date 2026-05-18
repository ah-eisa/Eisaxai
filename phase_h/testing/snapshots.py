"""
Phase H — snapshot / golden-report comparison.

Snapshots are normalised markdown / JSON dumps stored under
`phase_h/testing/goldens/<name>.md` (or .json). The test runner
diff-compares the current output against the stored snapshot;
on mismatch it emits a unified diff to aid review.

Update flow: set EISAX_UPDATE_SNAPSHOTS=1 in env and rerun. The
runner OVERWRITES the snapshot files. Always review the diff in
git before committing the new snapshot.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


_GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
_GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

UPDATE_ENV = "EISAX_UPDATE_SNAPSHOTS"


# ──────────────────────────────────────────────────────────────────────
# Normalisation — strip volatile values so snapshots stay deterministic
# ──────────────────────────────────────────────────────────────────────

# Dates / timestamps / hashes / sizes / paths
_VOLATILE = [
    (re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"), "<TIMESTAMP>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}"), "<DATE>"),
    (re.compile(r"sha256:[0-9a-f]{4,}"), "sha256:<HASH>"),
    (re.compile(r"`[0-9a-f]{12,}`"), "`<HASH>`"),
    (re.compile(r"snapshot_id[\":\s]*[`\"]?[0-9a-f]{8,}"), "snapshot_id: <HASH>"),
    # Audit-appendix engine hashes — change with envelope timestamps so are
    # intrinsically volatile across runs; their stability is enforced by
    # determinism tests, not snapshot tests.
    (re.compile(r"\|\s*([a-z_]+)\s+hash\s*\|\s*[0-9a-f]{8,}\s*\|"), "| \\1 hash | <HASH> |"),
    (re.compile(r"len=\d+"), "len=<N>"),
    (re.compile(r"paths_simulated[\":\s]*\d+"), "paths_simulated: <N>"),
    (re.compile(r"\(\d+\.\d{2,4} bp\)"), "(<BP>)"),
]


def normalise(text: str) -> str:
    """Replace volatile substrings before snapshot comparison."""
    out = text
    for rx, repl in _VOLATILE:
        out = rx.sub(repl, out)
    # Collapse trailing whitespace + trailing newlines
    out = re.sub(r"[ \t]+$", "", out, flags=re.MULTILINE)
    return out.rstrip() + "\n"


# ──────────────────────────────────────────────────────────────────────
# Save / compare
# ──────────────────────────────────────────────────────────────────────

def _path(name: str, ext: str = "md") -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return _GOLDEN_DIR / f"{safe}.{ext}"


def save_snapshot(name: str, content: str | Mapping, *, ext: str = "md") -> Path:
    p = _path(name, ext)
    if ext == "json":
        p.write_text(json.dumps(content, indent=2, sort_keys=True, default=str), encoding="utf-8")
    else:
        p.write_text(normalise(content if isinstance(content, str) else str(content)),
                     encoding="utf-8")
    return p


def snapshot_compare(
    name: str,
    actual: str | Mapping,
    *,
    ext: str = "md",
    update_env: str = UPDATE_ENV,
) -> None:
    p = _path(name, ext)
    if ext == "json":
        actual_text = json.dumps(actual, indent=2, sort_keys=True, default=str)
    else:
        actual_text = normalise(actual if isinstance(actual, str) else str(actual))

    if os.environ.get(update_env, "").strip() in {"1", "true", "yes", "on"}:
        save_snapshot(name, actual_text, ext=ext)
        return

    if not p.exists():
        save_snapshot(name, actual_text, ext=ext)
        raise AssertionError(
            f"snapshot {name!r} did not exist — created on disk; "
            f"re-run to confirm. Path: {p}"
        )

    expected_text = p.read_text(encoding="utf-8")
    if expected_text == actual_text:
        return

    diff = "".join(difflib.unified_diff(
        expected_text.splitlines(keepends=True),
        actual_text.splitlines(keepends=True),
        fromfile=f"golden/{name}",
        tofile=f"actual/{name}",
        n=2,
    ))
    raise AssertionError(
        f"snapshot mismatch for {name!r}. "
        f"Set {update_env}=1 to refresh.\n{diff[:8000]}"
    )


__all__ = ["snapshot_compare", "save_snapshot", "normalise"]
