"""
Phase H — testing harness.

Public helpers for snapshot tests, golden-report comparisons, markdown
structure assertions, schema validation, and the regression-suite runner.

Use from any test:

    from phase_h.testing import (
        assert_section_order,
        assert_tone_clean,
        assert_envelope_valid,
        snapshot_compare,
    )
"""

from .assertions import (
    assert_section_order,
    assert_tone_clean,
    assert_envelope_valid,
    assert_no_broken_tables,
    assert_bilingual_render,
    assert_markdown_structure,
)
from .snapshots import snapshot_compare, save_snapshot
from .runner import run_regression_suite

__all__ = [
    "assert_section_order",
    "assert_tone_clean",
    "assert_envelope_valid",
    "assert_no_broken_tables",
    "assert_bilingual_render",
    "assert_markdown_structure",
    "snapshot_compare",
    "save_snapshot",
    "run_regression_suite",
]
