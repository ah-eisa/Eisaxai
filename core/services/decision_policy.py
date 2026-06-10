from __future__ import annotations

import math
import re
from typing import Any


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "N/A", "None", "-", "--"):
        return None
    try:
        if isinstance(value, str):
            value = value.replace("%", "").replace("x", "").replace(",", "").strip()
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    except Exception:
        return None


def _has_valid_metric(value: Any) -> bool:
    numeric = _safe_float(value)
    if numeric is not None:
        return abs(numeric) > 0
    text = _clean_text(value)
    return bool(text) and text.lower() not in {"n/a", "none", "null", "nan", "-", "--", "0"}


# ─────────────────────────────────────────────────────────────────────────────
# EisaX Decision Taxonomy — 4 unified axes
# ─────────────────────────────────────────────────────────────────────────────
# Verdict   : Buy / Hold / Reduce / Sell
# Timing    : Attractive / Neutral / Extended
# Execution : Scale In / Wait / Reduce Exposure / Hold Steady
# Evidence  : Limited / Moderate / Strong

_VERDICT_MAP = {
    "BUY":        "Buy",
    "ACCUMULATE": "Buy",
    "HOLD":       "Hold",
    "REDUCE":     "Reduce",
    "SELL":       "Sell",
    "AVOID":      "Sell",
}

_TIMING_MAP = {
    "BUY NOW":    "Attractive",
    "ATTRACTIVE": "Attractive",
    "ACCUMULATE": "Neutral",
    "NEUTRAL":    "Neutral",
    "WAIT":       "Extended",
    "WATCHLIST":  "Extended",
    "EXTENDED":   "Extended",
    "LOW":        "Extended",
}

_EVIDENCE_MAP = {
    "LOW":      "Limited",
    "MEDIUM":   "Moderate",
    "MED":      "Moderate",
    "MODERATE": "Moderate",
    "HIGH":     "Strong",
    "STRONG":   "Strong",
}


def canonical_verdict(raw: Any) -> str:
    """Normalize any verdict string to the 4 canonical values."""
    return _VERDICT_MAP.get(str(raw or "").strip().upper(), str(raw or "Hold"))


def canonical_timing(raw: Any) -> str:
    """Normalize any timing string to Attractive / Neutral / Extended."""
    return _TIMING_MAP.get(str(raw or "").strip().upper(), str(raw or "Neutral"))


def canonical_evidence(raw: Any) -> str:
    """Normalize any conviction/confidence string to Limited / Moderate / Strong."""
    return _EVIDENCE_MAP.get(str(raw or "").strip().upper(), str(raw or "Moderate"))


def canonical_execution(verdict: str, timing: str) -> str:
    """Derive execution recommendation from canonical verdict + timing."""
    v = canonical_verdict(verdict)
    t = canonical_timing(timing)
    if v == "Sell":
        return "Reduce Exposure"
    if v == "Reduce":
        return "Reduce Exposure"
    if v == "Buy" and t == "Attractive":
        return "Scale In"
    if v == "Buy":
        return "Wait"
    return "Hold Steady"


def count_valid_fundamental_fields(
    fundamentals: dict[str, Any] | None,
    dc_data: dict[str, Any] | None = None,
    *,
    analyst_target: Any = None,
    forward_pe: Any = None,
) -> int:
    fundamentals = fundamentals or {}
    dc_data = dc_data or {}
    candidates = [
        fundamentals.get("fundamental_score"),
        fundamentals.get("revenue_growth"),
        fundamentals.get("eps_growth"),
        fundamentals.get("gross_margin"),
        fundamentals.get("operating_margin"),
        fundamentals.get("net_margin"),
        fundamentals.get("roe"),
        fundamentals.get("roic"),
        fundamentals.get("pe_ratio"),
        forward_pe if forward_pe is not None else (dc_data.get("forward_pe") or fundamentals.get("forward_pe")),
        fundamentals.get("ev_ebitda"),
        fundamentals.get("current_ratio"),
        fundamentals.get("debt_equity"),
        analyst_target if analyst_target is not None else fundamentals.get("analyst_target"),
    ]
    return sum(1 for value in candidates if _has_valid_metric(value))


def classify_data_coverage_level(valid_fields: int) -> str:
    if valid_fields <= 1:
        return "technical_only"
    if valid_fields <= 3:
        return "low"
    if valid_fields <= 6:
        return "medium"
    return "high"


def _replace_section_body(
    text: str,
    section_number: int,
    next_section_number: int | None,
    replacement: str,
) -> str:
    next_boundary = rf"\n#+\s*{next_section_number}[.\s]" if next_section_number is not None else r"\Z"
    pattern = rf"((?:^|\n)#+\s*{section_number}[.\s][^\n]*\n)(.*?)(?={next_boundary}|\Z)"
    return re.sub(
        pattern,
        lambda match: match.group(1) + replacement.strip() + "\n",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def _apply_data_messaging_locks(text: str, coverage_count: int) -> str:
    locked = str(text or "")
    if coverage_count <= 0:
        return locked
    replacements = [
        (r"(?i)\bfundamental data is largely unavailable\b", "fundamental data coverage is limited"),
        (r"(?i)\bfundamental data is unavailable\b", "fundamental data coverage is limited"),
        (r"(?i)\bfundamental data unavailable\b", "fundamental data coverage is limited"),
        (r"(?i)\bdata is unavailable\b", "data coverage is partial"),
        (r"(?i)\bdata unavailable\b", "data coverage is partial"),
    ]
    for pattern, replacement in replacements:
        locked = re.sub(pattern, replacement, locked)
    return locked


def compact_low_data_generation_inputs(
    data_block: str,
    decision_context: dict[str, Any] | None,
) -> str:
    decision_context = decision_context or {}
    coverage_count = int(decision_context.get("coverage_count") or 0)
    coverage_level = _clean_text(decision_context.get("coverage_level")).lower()
    low_data_mode = bool(decision_context.get("low_data_mode")) or coverage_level in {"technical_only", "low"}
    locked = _apply_data_messaging_locks(data_block, coverage_count)
    if not low_data_mode:
        return locked

    note = (
        "LOW-DATA COMPACT MODE:\n"
        "Fundamental visibility is limited; analysis relies primarily on price behavior.\n"
        "Analyst consensus, valuation scenario tables, peer comparison, and extended outlook are disabled.\n"
    )
    locked = re.sub(
        r"\nVALUATION SCENARIOS \(.*?Expected Value:.*?\n",
        "\nVALUATION SCENARIOS: Disabled in low-data compact mode.\n",
        locked,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )
    locked = re.sub(
        r"\nUS PEER COMPARISON TABLE:\n.*?(?=\nBALANCE SHEET:|\Z)",
        "\nUS PEER COMPARISON TABLE: Disabled in low-data compact mode.\n",
        locked,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )
    locked = re.sub(
        r"\nGULF PEER COMPARISON DATA \(LIVE.*?(?=\n[A-Z][A-Z /&()'-]+:|\Z)",
        "\nGULF PEER COMPARISON DATA: Disabled in low-data compact mode.\n",
        locked,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )
    locked = re.sub(
        r"\nOIL PRICE SENSITIVITY \(pre-computed\):.*?(?=\nSCENARIO ANALYSIS \(|\nLATEST NEWS|\Z)",
        "\nOIL PRICE SENSITIVITY: Disabled in low-data compact mode.\n",
        locked,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )
    locked = re.sub(
        r"\nSCENARIO ANALYSIS \([^\n]*\):.*?(?=\nLATEST NEWS|\Z)",
        "\nSCENARIO ANALYSIS: Disabled in low-data compact mode.\n",
        locked,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if "LOW-DATA COMPACT MODE:" not in locked:
        locked = note + "\n" + locked
    return locked


def apply_language_locks(report_text: str, decision_context: dict[str, Any] | None) -> str:
    locked = str(report_text or "")
    decision_context = decision_context or {}
    recommendation = _clean_text(decision_context.get("recommendation")).upper() or "HOLD"
    final_action = _clean_text(decision_context.get("final_action"))
    low_data_mode = bool(decision_context.get("low_data_mode"))
    coverage_count = int(decision_context.get("coverage_count") or 0)

    replacements: list[tuple[str, str]] = []
    if recommendation == "HOLD":
        replacements.extend(
            [
                (r"(?i)\bhigh conviction buy\b", "measured hold"),
                (r"(?i)\baggressive entry\b", "disciplined patience"),
            ]
        )
        if final_action in {"WAIT / NO ACTION", "WATCHLIST / WAIT FOR ENTRY"}:
            replacements.append((r"(?i)\bstrong buy\b", "positive momentum"))

    if recommendation in {"REDUCE", "SELL", "AVOID"}:
        replacements.extend(
            [
                (r"(?i)\bstrong buy\b", "counter-trend momentum"),
                (r"(?i)\bweak buy\b", "short-term momentum"),
                (r"(?i)\bupside potential\b", "risk/reward profile"),
                (r"(?i)\bbullish case dominant\b", "bull case is not dominant"),
                (r"(?i)\bcompelling entry\b", "setup requires improvement"),
                (r"(?i)\battractive opportunity\b", "mixed setup"),
                (r"(?i)\bconstructive entry\b", "entry remains constrained"),
            ]
        )

    if low_data_mode:
        replacements.extend(
            [
                (r"(?i)\bhigh conviction\b", "low conviction"),
                (r"(?i)\bstrong buy\b", "positive momentum"),
                (r"(?i)\bstrong sell\b", "negative momentum"),
            ]
        )
        visibility_statement = "Fundamental visibility is limited; analysis relies primarily on price behavior."
        if visibility_statement.lower() not in locked.lower():
            locked, inserted = re.subn(
                r"((?:^|\n)#+\s*2[.\s][^\n]*\n)",
                r"\1" + visibility_statement + "\n\n",
                locked,
                count=1,
                flags=re.IGNORECASE,
            )
            if inserted == 0:
                locked = visibility_statement + "\n\n" + locked
        locked = _replace_section_body(
            locked,
            5,
            6,
            "Fundamental visibility is limited; analysis relies primarily on price behavior. Analyst consensus and valuation scenarios are disabled in low-data mode.",
        )
        locked = _replace_section_body(
            locked,
            6,
            7,
            "Peer comparison is disabled in low-data mode because fundamental coverage is limited.",
        )
        if _word_count(locked) > 600:
            locked = _replace_section_body(
                locked,
                7,
                8,
                "Outlook remains conditional on price confirmation and improved fundamental disclosure.",
            )
            locked = _replace_section_body(
                locked,
                8,
                9,
                "Timing remains uncertain; wait for confirmation before acting.",
            )
            locked = _replace_section_body(
                locked,
                9,
                None,
                "Maintain a low-conviction hold until visibility improves.",
            )
        if not any(token in locked.lower() for token in ("limited visibility", "uncertain", "requires confirmation")):
            hedge_sentence = "The setup is uncertain and requires confirmation."
            locked, inserted = re.subn(
                r"((?:^|\n)#+\s*1[.\s][^\n]*\n)",
                r"\1" + hedge_sentence + "\n\n",
                locked,
                count=1,
                flags=re.IGNORECASE,
            )
            if inserted == 0:
                locked = hedge_sentence + "\n\n" + locked

    for pattern, replacement in replacements:
        locked = re.sub(pattern, replacement, locked)
    return _apply_data_messaging_locks(locked, coverage_count)
