"""
Phase H — shared bilingual markdown helpers.

Every Phase H engine emits markdown by calling helpers here so that:
- tone discipline is enforced in one place (via tone_guard)
- bilingual EN/AR mapping is centralised
- severity tags follow the existing institutional muted convention
- table formatting matches the rest of the report (no broken pipes)

This module is engine-agnostic. It does not import the engines.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from .tone_guard import scrub_text

# ────────────────────────────────────────────────────────────────────
# Bilingual label dictionary
# Keep keys English; AR values are the institutional translation.
# Extend here as new sections are added — never inline AR strings
# in engine code.
# ────────────────────────────────────────────────────────────────────
LABELS = {
    "benchmark_relative_diagnostics": {
        "en": "Benchmark Relative Diagnostics",
        "ar": "تشخيصات الأداء النسبي مقابل المؤشر",
    },
    "execution_efficiency": {
        "en": "Execution Efficiency Diagnostics",
        "ar": "تشخيصات كفاءة التنفيذ",
    },
    "forward_scenario": {
        "en": "Forward Scenario Distribution",
        "ar": "توزيع السيناريوهات المستقبلية",
    },
    "factor_decomposition": {
        "en": "Factor Risk Decomposition",
        "ar": "تحليل المخاطر بحسب العوامل",
    },
    "committee_brief": {
        "en": "Investment Committee Brief",
        "ar": "ملخص لجنة الاستثمار",
    },
    "active_return": {"en": "Active Return", "ar": "العائد النسبي"},
    "tracking_error": {"en": "Tracking Error", "ar": "خطأ التتبع"},
    "information_ratio": {"en": "Information Ratio", "ar": "نسبة المعلومات"},
    "rolling_alpha": {"en": "Rolling Alpha (12m)", "ar": "ألفا المتجدد (12 شهر)"},
    "rolling_beta": {"en": "Rolling Beta (12m)", "ar": "بيتا المتجدد (12 شهر)"},
    "relative_drawdown": {"en": "Relative Drawdown", "ar": "الانخفاض النسبي"},
    "upside_capture": {"en": "Upside Capture", "ar": "احتواء الصعود"},
    "downside_capture": {"en": "Downside Capture", "ar": "احتواء الهبوط"},
    "relative_volatility": {"en": "Relative Volatility", "ar": "التذبذب النسبي"},
    "active_share": {"en": "Active Share", "ar": "الحصة النشطة"},
    "style_drift": {"en": "Style Drift", "ar": "انحراف الأسلوب"},
    "metric": {"en": "Metric", "ar": "المقياس"},
    "value": {"en": "Value", "ar": "القيمة"},
    "tag": {"en": "Tag", "ar": "التصنيف"},
    "turnover": {"en": "Turnover", "ar": "معدل الدوران"},
    "implementation_shortfall": {
        "en": "Implementation Shortfall",
        "ar": "فجوة التنفيذ",
    },
    "market_impact": {"en": "Market Impact", "ar": "أثر السوق"},
    "slippage": {"en": "Estimated Slippage", "ar": "الانزلاق المقدر"},
    "complexity": {"en": "Execution Complexity", "ar": "تعقيد التنفيذ"},
    "liquidity_stress": {"en": "Liquidity Stress", "ar": "ضغط السيولة"},
    "rebalance_freq": {"en": "Rebalance Frequency", "ar": "تكرار إعادة التوازن"},
    "scenario": {"en": "Scenario", "ar": "السيناريو"},
    "probability": {"en": "Probability", "ar": "الاحتمال"},
    "terminal_p10": {"en": "Terminal P10", "ar": "القيمة النهائية (P10)"},
    "terminal_p50": {"en": "Terminal P50", "ar": "القيمة النهائية (P50)"},
    "terminal_p90": {"en": "Terminal P90", "ar": "القيمة النهائية (P90)"},
    "max_drawdown": {"en": "Max Drawdown (P50)", "ar": "أقصى انخفاض (P50)"},
    "recovery_months": {"en": "Recovery (months, P50)", "ar": "مدة التعافي (أشهر، P50)"},
    "factor": {"en": "Factor", "ar": "العامل"},
    "loading": {"en": "Loading", "ar": "التحميل"},
    "t_stat": {"en": "t-stat", "ar": "إحصاء t"},
    "contribution": {"en": "Contribution", "ar": "المساهمة"},
    "no_data": {"en": "Insufficient data", "ar": "بيانات غير كافية"},
    "reliability": {"en": "Reliability", "ar": "الموثوقية"},
}


def L(key: str, language: str = "en") -> str:
    """Look up a bilingual label. Falls back to the key itself if missing."""
    entry = LABELS.get(key)
    if not entry:
        return key
    return entry.get(language, entry.get("en", key))


# ────────────────────────────────────────────────────────────────────
# Severity tags — muted, parenthetical, lowercase.
# Matches the existing _impl_tag style used in portfolio_builder.
# ────────────────────────────────────────────────────────────────────
SEVERITY_TAGS = {
    "low":      {"en": "(low)",      "ar": "(منخفض)"},
    "moderate": {"en": "(moderate)", "ar": "(متوسط)"},
    "elevated": {"en": "(elevated)", "ar": "(مرتفع)"},
    "high":     {"en": "(high)",     "ar": "(عالٍ)"},
    "neutral":  {"en": "(neutral)",  "ar": "(محايد)"},
}


def severity_tag(level: str, language: str = "en") -> str:
    entry = SEVERITY_TAGS.get(level.lower(), SEVERITY_TAGS["neutral"])
    return entry.get(language, entry["en"])


# ────────────────────────────────────────────────────────────────────
# Markdown table builder — defensive against None / NaN
# ────────────────────────────────────────────────────────────────────
def md_table(headers: List[str], rows: Iterable[List[str]]) -> str:
    """Build a clean markdown table. Empty cells render as em-dash."""
    head = "| " + " | ".join(h or "—" for h in headers) + " |"
    sep  = "| " + " | ".join("---" for _ in headers) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join((c if (c is not None and c != "") else "—") for c in row) + " |")
    return "\n".join([head, sep, *body])


def fmt_pct(x: Optional[float], decimals: int = 2) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_num(x: Optional[float], decimals: int = 2) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def fmt_bp(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.1f} bp"
    except (TypeError, ValueError):
        return "—"


def section_heading(level: int, key: str, language: str = "en") -> str:
    """Render `## Title` / `### Title` etc. with bilingual label."""
    hashes = "#" * max(1, min(6, level))
    return f"{hashes} {L(key, language)}"


def inject_subsection(report_md: str, parent_marker: str, subsection_md: str) -> str:
    """
    Insert a subsection BEFORE the next top-level (## ) heading
    following `parent_marker` (e.g. "## C. Risk Diagnostics").

    If parent_marker is not found, append subsection at the end.

    This is the safe append strategy used by every engine that
    attaches under an existing A-G section.
    """
    if not parent_marker or parent_marker not in report_md:
        return report_md.rstrip() + "\n\n" + subsection_md.strip() + "\n"

    head, _, tail = report_md.partition(parent_marker)
    # find the next top-level heading after the parent
    cursor = 0
    next_top = -1
    lines = tail.splitlines(keepends=True)
    consumed = 0
    for i, line in enumerate(lines):
        if i == 0:
            consumed += len(line)
            continue
        # match new top-level "## X." but NOT "### "
        stripped = line.lstrip()
        if stripped.startswith("## ") and not stripped.startswith("### "):
            next_top = consumed
            break
        consumed += len(line)
    if next_top == -1:
        # no further top-level; append at end of tail
        return head + parent_marker + tail.rstrip() + "\n\n" + subsection_md.strip() + "\n"
    return (
        head
        + parent_marker
        + tail[:next_top].rstrip()
        + "\n\n"
        + subsection_md.strip()
        + "\n\n"
        + tail[next_top:]
    )


def insert_top_level_before(report_md: str, anchor_marker: str, block_md: str) -> str:
    """
    Insert `block_md` as a top-level section just before `anchor_marker`
    (e.g. before "## G. Audit Appendix"). Used for the new "## H. ..."
    section so Audit Appendix remains last.
    """
    if not anchor_marker or anchor_marker not in report_md:
        return report_md.rstrip() + "\n\n" + block_md.strip() + "\n"
    head, _, tail = report_md.partition(anchor_marker)
    return head.rstrip() + "\n\n" + block_md.strip() + "\n\n" + anchor_marker + tail


def finalize(text: str) -> str:
    """Apply tone-guard scrubbing before returning markdown to the report."""
    return scrub_text(text)


__all__ = [
    "LABELS",
    "L",
    "severity_tag",
    "md_table",
    "fmt_pct",
    "fmt_num",
    "fmt_bp",
    "section_heading",
    "inject_subsection",
    "insert_top_level_before",
    "finalize",
]
