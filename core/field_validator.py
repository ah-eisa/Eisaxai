from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Match


logger = logging.getLogger("eisax.field_validator")


@dataclass
class FieldFix:
    field: str
    claimed: str
    actual_value: float | str
    fix_applied: str


@dataclass
class ValidatorResult:
    text: str
    fixes: list[FieldFix]
    detected: int
    corrected: int


# Match common phrasings:
#   "Current ratio: unavailable"
#   "**Current Ratio:** N/A"
#   "current ratio is unavailable"
#   "Gross margin: not available"
#   "ROE: —"
_CLAIM_PATTERNS = [
    re.compile(
        r"(?i)\*{0,2}[ \t]*([A-Za-z][A-Za-z0-9/. \t-]*?)[ \t]*(?:[:=][ \t]*\*{0,2}|\*{0,2}[ \t]*[:=][ \t]*\*{0,2})[ \t]*(?:unavailable|not[ \t]+available|N/?A|—|--)\*{0,2}\.?"
    ),
    re.compile(
        r"(?i)\b([A-Za-z][A-Za-z0-9/. \t-]*?)[ \t]+is[ \t]+(?:unavailable|not[ \t]+available|currently[ \t]+unavailable)\b"
    ),
]

_LABEL_MAP = {
    "current ratio": "current_ratio",
    "gross margin": "gross_margin",
    "operating margin": "operating_margin",
    "net margin": "net_margin",
    "roe": "roe",
    "return on equity": "roe",
    "roa": "roa",
    "return on assets": "roa",
    "debt/equity": "debt_equity",
    "debt to equity": "debt_equity",
    "d/e": "debt_equity",
    "ebitda": "ebitda",
    "free cash flow": "free_cash_flow",
    "fcf": "free_cash_flow",
    "p/b": "price_book",
    "price/book": "price_book",
    "price-to-book": "price_book",
    "price to book": "price_book",
    "p/e": "pe_ratio",
    "price/earnings": "pe_ratio",
    "price earnings": "pe_ratio",
    "revenue growth": "revenue_growth",
    "earnings growth": "earnings_growth",
    "dividend yield": "div_yield",
    "beta": "beta",
    "eps": "eps",
    "revenue": "revenue",
    "net income": "net_income",
}

_CANONICAL_LABELS = {
    "current_ratio": "Current Ratio",
    "gross_margin": "Gross Margin",
    "operating_margin": "Operating Margin",
    "net_margin": "Net Margin",
    "roe": "ROE",
    "roa": "ROA",
    "debt_equity": "Debt/Equity",
    "ebitda": "EBITDA",
    "free_cash_flow": "Free Cash Flow",
    "price_book": "P/B",
    "pe_ratio": "P/E",
    "revenue_growth": "Revenue Growth",
    "earnings_growth": "Earnings Growth",
    "div_yield": "Dividend Yield",
    "beta": "Beta",
    "eps": "EPS",
    "revenue": "Revenue",
    "net_income": "Net Income",
}

_PERCENT_FIELDS = {
    "gross_margin",
    "operating_margin",
    "net_margin",
    "roe",
    "roa",
    "div_yield",
    "revenue_growth",
    "earnings_growth",
}
_RATIO_FIELDS = {"current_ratio", "debt_equity", "price_book", "pe_ratio"}
_CURRENCY_FIELDS = {"revenue", "ebitda", "free_cash_flow", "net_income"}
_PLAIN_NUMBER_FIELDS = {"eps", "beta"}
_UNAVAILABLE_STRINGS = {"", "none", "null", "n/a", "na", "unavailable", "not available", "—", "--"}


@dataclass(frozen=True)
class _Claim:
    start: int
    end: int
    match: Match[str]
    field: str


def _clean_label(label: str) -> str:
    cleaned = str(label or "").strip().strip("*")
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s*/\s*", "/", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" :.=*").lower()


def _canonical_field(label: str) -> str | None:
    cleaned = _clean_label(label)
    if cleaned in _LABEL_MAP:
        return _LABEL_MAP[cleaned]

    spaced = cleaned.replace("/", " ")
    if spaced in _LABEL_MAP:
        return _LABEL_MAP[spaced]

    # Common prose can include leading words, e.g. "the current ratio".
    for known_label, field in sorted(_LABEL_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        if cleaned.endswith(known_label) or spaced.endswith(known_label.replace("/", " ")):
            return field
    return None


def _has_available_value(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in _UNAVAILABLE_STRINGS
    return True


def _to_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _format_currency(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1e9:
        return f"{value / 1e9:.2f}B"
    if absolute >= 1e6:
        return f"{value / 1e6:.1f}M"
    return f"{value:.2f}"


def _format_value(field: str, value: object) -> str:
    number = _to_float(value)
    if number is None:
        return str(value)
    if field in _PERCENT_FIELDS:
        return f"{number:.2f}%"
    if field in _RATIO_FIELDS:
        return f"{number:.2f}x"
    if field in _CURRENCY_FIELDS:
        return _format_currency(number)
    if field in _PLAIN_NUMBER_FIELDS:
        return f"{number:.2f}"
    return f"{number:.2f}"


def _replacement_label(field: str) -> str:
    return _CANONICAL_LABELS.get(field, field.replace("_", " ").title())


def _collect_claims(text: str) -> list[_Claim]:
    claims: list[_Claim] = []
    occupied: list[tuple[int, int]] = []

    for pattern in _CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < seen_end and end > seen_start for seen_start, seen_end in occupied):
                continue
            field = _canonical_field(match.group(1))
            if field is None:
                continue
            claims.append(_Claim(start=start, end=end, match=match, field=field))
            occupied.append((start, end))

    return sorted(claims, key=lambda claim: claim.start)


def validate_fields(text: str, fund: dict) -> ValidatorResult:
    """
    OBSERVER mode (Phase 5 consolidation):
    Walk the report. For each "X is unavailable" claim where the fund dict
    actually HAS the value, record an inconsistency. DO NOT mutate the text.

    Silent semantic mutation is forbidden in this layer — the LLM/render
    layer owns the prose. This observer only flags discrepancies so an
    upstream layer (or the reconciliation audit) can act on them.

    Returns ValidatorResult with:
        text:      input text UNCHANGED
        fixes:     list of FieldFix detailing each inconsistency
                   (fix_applied is the value we COULD have substituted)
        detected:  number of "unavailable" claims found
        corrected: 0 (this layer never auto-corrects anymore)
    """
    source_text = text if isinstance(text, str) else str(text or "")
    fund_data = fund if isinstance(fund, dict) else {}
    inconsistencies: list[FieldFix] = []

    claims = _collect_claims(source_text)
    for claim in claims:
        value = fund_data.get(claim.field)
        if not _has_available_value(value):
            continue
        # Record discrepancy but DO NOT modify the report text.
        formatted_value = _format_value(claim.field, value)
        would_be = f"{_replacement_label(claim.field)}: {formatted_value}"
        inconsistencies.append(
            FieldFix(
                field=claim.field,
                claimed=claim.match.group(0),
                actual_value=value,
                fix_applied=would_be,   # what we WOULD have written if we mutated
            )
        )

    result = ValidatorResult(
        text=source_text,             # UNCHANGED — observer-only
        fixes=inconsistencies,
        detected=len(claims),
        corrected=0,                  # observer mode never auto-corrects
    )
    logger.info(
        "[FieldValidator] detected=%d corrected=%d fields=%s",
        result.detected,
        result.corrected,
        [fix.field for fix in result.fixes],
    )
    return result


if __name__ == "__main__":
    scenarios = [
        (
            "Real contradiction",
            "...\n- **Current Ratio:** unavailable\n- **ROE:** 21.2%\n...",
            {"current_ratio": 1.47, "roe": 21.2},
        ),
        (
            "Legitimate N/A",
            "- **Debt/Equity:** N/A\n",
            {"debt_equity": None},
        ),
        (
            "No claims",
            "- Current Ratio: 1.47x\n- ROE: 21.20%\n",
            {"current_ratio": 1.47, "roe": 21.2},
        ),
    ]

    for name, report, fund in scenarios:
        result = validate_fields(report, fund)
        print(f"--- {name} ---")
        print(result.text)
        print(len(result.fixes))
        print(f"detected={result.detected} corrected={result.corrected}")
