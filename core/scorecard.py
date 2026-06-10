"""scorecard.py -- thin re-export shim."""
from core.scorecard_parser import sanitize_field, render_field, parse_report
from core.scorecard_engine import calculate_score, generate_scorecard_markdown
from core.scorecard_verdict import (
    compute_entry_quality, compute_tech_score, get_verdict, compute_decision_type,
)

__all__ = [
    'sanitize_field', 'render_field', 'parse_report',
    'calculate_score', 'generate_scorecard_markdown',
    'compute_entry_quality', 'compute_tech_score',
    'get_verdict', 'compute_decision_type',
]
