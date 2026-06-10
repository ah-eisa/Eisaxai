"""
fact_check_builder.py — Component 3: Fact-Check Hard Lock

Week 2 Trust Expansion: generates the deterministic fact-check block
entirely from the frozen ReportSnapshot. No LLM involvement allowed.
All values are 100% from verified data sources.
"""
from __future__ import annotations

from typing import Any

from core.services.report_snapshot import ReportSnapshot


def build_fact_check(snapshot: ReportSnapshot) -> list[dict[str, Any]]:
    """
    Build the fact-check reference block from snapshot.

    Returns a list of {metric, value} dicts — one per canonical field.
    Every value comes directly from the frozen snapshot; no LLM is called.

    This block is used to:
    - Surface data provenance in the UI
    - Cross-check LLM-generated sections for numeric consistency
    - Provide the lint engine with ground-truth values
    """
    _52h = snapshot.get("week52_high")
    _52l = snapshot.get("week52_low")

    return [
        {
            "metric": "Price",
            "value":  snapshot.get("price"),
            "source": snapshot.get_record("price").source,
        },
        {
            "metric": "Trailing PE",
            "value":  snapshot.get("pe"),
            "source": snapshot.get_record("pe").source,
        },
        {
            "metric": "Forward PE",
            "value":  snapshot.get("forward_pe"),
            "source": snapshot.get_record("forward_pe").source,
        },
        {
            "metric": "Dividend Yield",
            "value":  snapshot.get("div_yield"),
            "source": snapshot.get_record("div_yield").source,
        },
        {
            "metric": "Market Cap",
            "value":  snapshot.get("market_cap"),
            "source": snapshot.get_record("market_cap").source,
        },
        {
            "metric": "52W Range",
            "value":  (_52l, _52h),
            "source": snapshot.get_record("week52_low").source,
        },
    ]


def check_data_availability(snapshot: ReportSnapshot, field: str) -> bool:
    """
    Return True if snapshot has a non-None value for field.

    Used by technical builder to avoid claiming data is unavailable
    when the snapshot already holds the value.
    """
    try:
        return snapshot.get(field) is not None
    except KeyError:
        return False
