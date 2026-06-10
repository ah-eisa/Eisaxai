"""
peer_table_builder.py — Component 1: Peer Self-Row Lock

Week 2 Trust Expansion: guarantees the subject ticker's row in the peer
comparison table is always sourced from the frozen ReportSnapshot, never
from LLM-generated data that may contain hallucinated prices or metrics.

Week 3 additions:
- div_yield garbage rejection: values > 15% are set to None unless whitelisted
- All regional tables route through this single build_peer_table function
"""
from __future__ import annotations

from typing import Any

from core.services.report_snapshot import ReportSnapshot

# Tickers whose high dividend yield is known-legitimate and should not be scrubbed
_DIV_YIELD_WHITELIST: frozenset[str] = frozenset()

_DIV_YIELD_GARBAGE_THRESHOLD = 0.15  # 15%


def _sanitize_div_yield(value: Any, ticker: str = "") -> Any:
    """Return None if div_yield looks like garbage (>15%), unless whitelisted."""
    if value is None:
        return None
    try:
        pct = float(value)
        # Normalize: if stored as a decimal fraction (e.g. 0.05 = 5%), convert
        # to percent for the threshold check, then restore to original scale
        pct_normalized = pct if pct > 1 else pct * 100
        if pct_normalized > _DIV_YIELD_GARBAGE_THRESHOLD * 100:
            if str(ticker).upper() in _DIV_YIELD_WHITELIST:
                return value
            return None
        return value
    except (TypeError, ValueError):
        return None


def build_peer_table(
    snapshot: ReportSnapshot,
    peer_rows_llm: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build the final peer comparison table.

    Parameters
    ----------
    snapshot      : frozen ReportSnapshot — the single source of truth.
    peer_rows_llm : LLM-generated peer rows (may contain a wrong self-row).

    Returns
    -------
    list of dicts, self-row guaranteed at index 0 with snapshot values.

    Rules (non-negotiable):
    - self-row MUST always equal snapshot values
    - no merge logic, no fallback, no LLM override
    - any LLM row whose ticker matches the subject ticker is removed
    - div_yield > 15% in peer rows is rejected (set to None) unless whitelisted
    """
    self_ticker = snapshot.get("ticker")

    self_row: dict[str, Any] = {
        "ticker":     self_ticker,
        "price":      snapshot.get("price"),
        "pe":         snapshot.get("pe"),
        "forward_pe": snapshot.get("forward_pe"),
        "div_yield":  snapshot.get("div_yield"),
        "market_cap": snapshot.get("market_cap"),
    }

    # Remove any LLM row that shadows the subject ticker (case-insensitive)
    filtered: list[dict[str, Any]] = [
        r for r in peer_rows_llm
        if str(r.get("ticker", "")).upper() != str(self_ticker or "").upper()
    ]

    # Sanitize garbage div_yield values in peer rows
    peer_rows: list[dict[str, Any]] = []
    for row in filtered:
        row = dict(row)
        row["div_yield"] = _sanitize_div_yield(row.get("div_yield"), row.get("ticker", ""))
        peer_rows.append(row)

    return [self_row] + peer_rows
