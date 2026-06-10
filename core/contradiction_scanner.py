from __future__ import annotations

import logging
import re
from pprint import pprint
from typing import Callable
from dataclasses import asdict, dataclass, field


logger = logging.getLogger("eisax.contradiction_scanner")


try:
    from core.services.decision_policy import canonical_verdict as _policy_canonical_verdict
except Exception:  # pragma: no cover - optional dependency fallback
    _policy_canonical_verdict = None


@dataclass
class Contradiction:
    rule_id: str
    layer: int
    severity: str
    summary: str
    evidence: list[str]
    auto_fix: str | None


@dataclass
class ScanResult:
    contradictions: list[Contradiction] = field(default_factory=list)
    fixed_text: str = ""
    unfixed: list[Contradiction] = field(default_factory=list)

    def has_blockers(self) -> bool:
        return any(item.severity == "blocker" for item in self.contradictions)

    def to_dict(self) -> dict:
        return {
            "contradictions": [asdict(item) for item in self.contradictions],
            "fixed_text": self.fixed_text,
            "unfixed": [asdict(item) for item in self.unfixed],
            "has_blockers": self.has_blockers(),
        }


@dataclass
class ExtractedReportState:
    verdicts: list[str]
    risks: list[str]
    actions: list[str]
    sections: list[str]
    sentences_with_verdict: list[str]

    def unique_verdicts(self) -> set[str]:
        return set(self.verdicts)

    def unique_risks(self) -> set[str]:
        return set(self.risks)

    def unique_actions(self) -> set[str]:
        return set(self.actions)


@dataclass(frozen=True)
class HardRule:
    id: str
    severity: str
    summary: str
    pattern: re.Pattern[str]
    replacement: str | None
    condition_check: Callable[[dict, dict, dict], bool]
    apply_fix: Callable[[str, re.Pattern[str], str | None], str] | None = None


@dataclass(frozen=True)
class StatefulHardRule:
    id: str
    severity: str
    summary: Callable[[dict, dict, dict, ExtractedReportState], str]
    condition_check: Callable[[dict, dict, dict, ExtractedReportState], bool]
    evidence: Callable[[str, dict, dict, dict, ExtractedReportState], list[str]]
    replacement: str | None
    apply_fix: Callable[[str, ExtractedReportState], str] | None = None


@dataclass(frozen=True)
class SemanticRule:
    id: str
    summary: str
    condition_check: Callable[[dict, dict, dict], bool]
    justification_present: Callable[[str], bool]
    failure_excerpt: Callable[[str], str]


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _safe_float(value, default: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("%", "")
        if not cleaned:
            return default
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def _canonical_verdict_value(raw: str) -> str:
    raw_text = str(raw or "").strip()
    if not raw_text:
        return ""

    candidate = raw_text
    if _policy_canonical_verdict is not None:
        try:
            candidate = str(_policy_canonical_verdict(raw_text) or raw_text).strip()
        except Exception:
            candidate = raw_text

    mapping = {
        "buy": "Buy",
        "accumulate": "Buy",
        "overweight": "Buy",
        "hold": "Hold",
        "maintain": "Hold",
        "reduce": "Reduce",
        "underweight": "Reduce",
        "sell": "Sell",
        "avoid": "Sell",
    }
    return mapping.get(candidate.lower(), mapping.get(raw_text.lower(), candidate.title()))


def _canonical_verdict(decision_data: dict) -> str:
    data = _as_dict(decision_data)
    raw = str(data.get("tax_verdict") or data.get("verdict") or "").strip()
    return _canonical_verdict_value(raw)


def _canonical_risk_value(raw: str) -> str:
    mapping = {
        "low": "Low",
        "moderate": "Moderate",
        "high": "High",
        "elevated": "High",
    }
    return mapping.get(str(raw or "").strip().lower(), str(raw or "").strip().title())


def _canonical_action_value(raw: str) -> str:
    mapping = {
        "scale in": "Scale In",
        "wait": "Wait",
        "wait for entry": "Wait",
        "hold steady": "Hold Steady",
        "reduce exposure": "Reduce Exposure",
    }
    return mapping.get(str(raw or "").strip().lower(), str(raw or "").strip().title())


VERDICT_ANCHOR_PATTERN = re.compile(
    r"(?i)(?:Verdict|Action|Recommendation|Fundamental|Final Action|Decision)[:\s\*]+"
    r"(Buy|Hold|Reduce|Sell|Accumulate|Maintain|Underweight|Overweight)"
)
VERDICT_MENTION_PATTERN = re.compile(r"(?i)(?:^|\s|\*|\W)(Buy|Hold|Reduce|Sell)(?=\W|\s|$)")
VERDICT_CONTEXT_PATTERN = re.compile(r"(?i)\b(?:Verdict|Action|Recommendation|Fundamental|Final Action|Decision)\b")
RISK_ANCHOR_PATTERN = re.compile(r"(?i)\bRisk[:\s\*]+(High|Moderate|Low|Elevated)")
ACTION_ANCHOR_PATTERN = re.compile(
    r"(?i)\bAction[:\s\*]+(Reduce Exposure|Hold Steady|Wait for Entry|Scale In|Wait)"
)
SECTION_HEADER_PATTERN = re.compile(r"(?:^|\n)#{1,3}\s+(.+?)$", re.MULTILINE)
SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group(0).strip()) for match in SENTENCE_PATTERN.finditer(text or "")]


def extract_state(report_text: str) -> ExtractedReportState:
    """
    Walk the report and extract all verdict/risk/action mentions.
    Normalizes case (Hold/HOLD -> "Hold") via canonical taxonomy.
    """
    text = report_text if isinstance(report_text, str) else str(report_text or "")
    verdicts: list[str] = []
    risks: list[str] = []
    actions: list[str] = []
    sections = [match.group(1).strip() for match in SECTION_HEADER_PATTERN.finditer(text)]
    sentences_with_verdict: list[str] = []
    verdict_spans: set[tuple[int, int]] = set()
    sentence_indexes_with_verdict: set[int] = set()
    sentences = _sentence_spans(text)

    def sentence_index_for_span(start: int, end: int) -> int | None:
        for index, (sentence_start, sentence_end, _sentence) in enumerate(sentences):
            if sentence_start <= start and end <= sentence_end:
                return index
        return None

    for match in VERDICT_ANCHOR_PATTERN.finditer(text):
        span = match.span(1)
        verdict_spans.add(span)
        verdicts.append(_canonical_verdict_value(match.group(1)))
        sentence_index = sentence_index_for_span(*span)
        if sentence_index is not None:
            sentence_indexes_with_verdict.add(sentence_index)

    for index, (sentence_start, _sentence_end, sentence) in enumerate(sentences):
        if not VERDICT_CONTEXT_PATTERN.search(sentence):
            continue
        for match in VERDICT_MENTION_PATTERN.finditer(sentence):
            span = (sentence_start + match.start(1), sentence_start + match.end(1))
            if span in verdict_spans:
                continue
            verdict_spans.add(span)
            verdicts.append(_canonical_verdict_value(match.group(1)))
            sentence_indexes_with_verdict.add(index)

    for match in RISK_ANCHOR_PATTERN.finditer(text):
        risks.append(_canonical_risk_value(match.group(1)))

    for match in ACTION_ANCHOR_PATTERN.finditer(text):
        actions.append(_canonical_action_value(match.group(1)))

    for index in sorted(sentence_indexes_with_verdict):
        sentence = sentences[index][2]
        if sentence:
            sentences_with_verdict.append(sentence)

    return ExtractedReportState(
        verdicts=verdicts,
        risks=risks,
        actions=actions,
        sections=sections,
        sentences_with_verdict=sentences_with_verdict,
    )


def _text_excerpt(text: str, match: re.Match[str] | None = None, width: int = 140) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    if match is None:
        return cleaned[:width]

    source = text or ""
    start = max(match.start() - width // 2, 0)
    end = min(match.end() + width // 2, len(source))
    snippet = re.sub(r"\s+", " ", source[start:end]).strip()
    return snippet[: max(width, len(match.group(0)))]


def _contains_dividend_yield_over(text: str, threshold: float) -> bool:
    for match in re.finditer(
        r"(?i)\b(?:dividend(?:\s+yield)?|yield(?:\s+play)?)\b[^%\n]{0,32}?(\d+(?:\.\d+)?)\s*%",
        text or "",
    ):
        value = _safe_float(match.group(1))
        if value is not None and value > threshold:
            return True
    return False


def _replace_high_risk_defensive(match: re.Match[str]) -> str:
    source = match.string or ""
    window_start = max(match.start() - 80, 0)
    window_end = min(match.end() + 80, len(source))
    context = source[window_start:window_end].lower()

    if re.search(r"\b(?:utilities?|reits?|real estate|property|telecom(?:s)?|communication services)\b", context):
        return "rate-sensitive"
    if re.search(
        r"\b(?:energy|materials?|industrials?|consumer discretionary|autos?|semiconductors?|banks?|financials?)\b",
        context,
    ):
        return "cyclical"
    return "elevated-risk"


def _drop_dcf_sections(text: str, pattern: re.Pattern[str]) -> tuple[str, list[str]]:
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return text, []

    evidence = [re.sub(r"\s+", " ", match.group("heading")).strip() for match in matches[:3]]
    updated = pattern.sub("", text or "")
    updated = re.sub(r"\n{3,}", "\n\n", updated).strip("\n")
    return updated, evidence


def _simple_substitution(text: str, pattern: re.Pattern[str], replacement) -> str:
    return pattern.sub(replacement, text)


def _normalize_indefinite_articles(text: str) -> str:
    text = re.sub(r"\b([Aa])\s+(elevated-risk)\b", lambda m: ("An" if m.group(1) == "A" else "an") + f" {m.group(2)}", text)
    return text


def _failure_excerpt_for_pattern(text: str, pattern: re.Pattern[str], fallback: str) -> str:
    match = pattern.search(text or "")
    if match:
        return _text_excerpt(text, match)
    return _text_excerpt(fallback or text or "")


def _condition_high_risk(decision_data: dict, summary: dict, evidence_data: dict) -> bool:
    return str(_as_dict(decision_data).get("tax_risk") or "").strip() == "High"


def _condition_buy_low_upside(decision_data: dict, summary: dict, evidence_data: dict) -> bool:
    upside = _safe_float(_as_dict(summary).get("upside_pct"), 100.0)
    return _canonical_verdict(decision_data) == "Buy" and upside is not None and upside < 10


def _condition_weak_adx(decision_data: dict, summary: dict, evidence_data: dict) -> bool:
    adx = _safe_float(_as_dict(summary).get("adx"), 0.0)
    return adx is not None and adx < 20


def _condition_limited_evidence(decision_data: dict, summary: dict, evidence_data: dict) -> bool:
    return str(_as_dict(decision_data).get("tax_evidence") or "").strip() == "Limited"


def _condition_hold(decision_data: dict, summary: dict, evidence_data: dict) -> bool:
    return _canonical_verdict(decision_data) == "Hold"


def _condition_reduce(decision_data: dict, summary: dict, evidence_data: dict) -> bool:
    return _canonical_verdict(decision_data) == "Reduce"


def _condition_low_data_dcf(decision_data: dict, summary: dict, evidence_data: dict) -> bool:
    data = _as_dict(evidence_data)
    return data.get("dcf_valuation") is False


def _decision_has_tax_verdict(decision_data: dict) -> bool:
    return bool(str(_as_dict(decision_data).get("tax_verdict") or "").strip())


def _format_values(values: set[str]) -> str:
    return "{" + ", ".join(sorted(values)) + "}"


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _state_evidence_from_values(values: list[str]) -> list[str]:
    return _unique_in_order(values)[:3]


def _state_verdict_evidence(
    text: str,
    decision_data: dict,
    summary: dict,
    evidence_data: dict,
    extracted: ExtractedReportState,
) -> list[str]:
    return extracted.sentences_with_verdict[:3] or _state_evidence_from_values(extracted.verdicts)


def _state_risk_evidence(
    text: str,
    decision_data: dict,
    summary: dict,
    evidence_data: dict,
    extracted: ExtractedReportState,
) -> list[str]:
    return _state_evidence_from_values(extracted.risks)


def _misaligned_actions(decision_data: dict, extracted: ExtractedReportState) -> list[str]:
    verdict = _canonical_verdict(decision_data)
    actions = _unique_in_order(extracted.actions)
    if verdict == "Hold":
        return [action for action in actions if action not in {"Hold Steady", "Wait"}]
    if verdict == "Buy":
        return [action for action in actions if action == "Reduce Exposure"]
    if verdict in {"Reduce", "Sell"}:
        return [action for action in actions if action == "Scale In"]
    return []


def _state_action_evidence(
    text: str,
    decision_data: dict,
    summary: dict,
    evidence_data: dict,
    extracted: ExtractedReportState,
) -> list[str]:
    return _misaligned_actions(decision_data, extracted)[:3]


def _summary_multi_verdict(
    decision_data: dict,
    summary: dict,
    evidence_data: dict,
    extracted: ExtractedReportState,
) -> str:
    return (
        f"Multiple verdicts in same report: {_format_values(extracted.unique_verdicts())}. "
        f"Authoritative verdict is {_canonical_verdict(decision_data)}."
    )


def _summary_multi_risk(
    decision_data: dict,
    summary: dict,
    evidence_data: dict,
    extracted: ExtractedReportState,
) -> str:
    return f"Multiple risk levels in same report: {_format_values(extracted.unique_risks())}."


def _summary_action_misaligned(
    decision_data: dict,
    summary: dict,
    evidence_data: dict,
    extracted: ExtractedReportState,
) -> str:
    action = (_misaligned_actions(decision_data, extracted) or [""])[0]
    return f"Action {action} misaligned with Verdict {_canonical_verdict(decision_data)}."


def _replace_reduce_verdict_in_hold(text: str, extracted: ExtractedReportState) -> str:
    # Only replace verdict-anchored "Reduce" (preceded by Verdict:/Action:/Fundamental:/Recommendation:)
    # AND case-sensitive to avoid touching "reduce" in narrative prose.
    return re.sub(
        r"((?:Verdict|Action|Fundamental(?:\s+Verdict)?|Final\s+Action|Recommendation|Decision)[:\s\*]+)Reduce\b(?!\s+Exposure\b)",
        r"\1trim positioning",
        text or "",
    )


def _replace_buy_verdict_in_hold(text: str, extracted: ExtractedReportState) -> str:
    # Only replace verdict-anchored "Buy" — NEVER touch "buy" in disclaimers/prose.
    # Must match: "Verdict: Buy", "Action: Buy", "Fundamental: Buy", etc.
    # Must NOT match: "offer to buy or sell", "buyers entered", "buy-side".
    return re.sub(
        r"((?:Verdict|Action|Fundamental(?:\s+Verdict)?|Final\s+Action|Recommendation|Decision)[:\s\*]+)Buy\b",
        r"\1accumulation candidate",
        text or "",
    )


def _justification_buy_low_upside(text: str) -> bool:
    phrases = (
        r"(?i)\bincome thesis\b",
        r"(?i)\bstrategic allocation\b",
        r"(?i)\bmoat\b",
        r"(?i)\blong-duration\b",
        r"(?i)\bdividend-led\b",
        r"(?i)\byield play\b",
    )
    if any(re.search(pattern, text or "") for pattern in phrases):
        return True
    return _contains_dividend_yield_over(text, 4.0)


def _justification_buy_risk_high(text: str) -> bool:
    phrases = (
        r"(?i)\bsizing constraint\b",
        r"(?i)\bsmaller position\b",
        r"(?i)\bphased entry\b",
        r"(?i)\bscale in\b",
    )
    return any(re.search(pattern, text or "") for pattern in phrases)


def _justification_sell_replacement(text: str) -> bool:
    phrases = (
        r"(?i)\bcatalyst failure\b",
        r"(?i)\bvaluation premium\b",
        r"(?i)\bfundamental deterioration\b",
        r"(?i)\bdeteriorating fundamentals?\b",
        r"(?i)\bpremium valuation\b",
    )
    return any(re.search(pattern, text or "") for pattern in phrases)


def _no_justification(_text: str) -> bool:
    return False


L1_HIGH_RISK_DEFENSIVE_PATTERN = re.compile(r"(?i)\bdefensive\b(?!\s+sectors?\b)")
L1_LOW_UPSIDE_AGGRESSIVE_BUY_PATTERN = re.compile(r"(?i)\baggressive(?:ly)?\s+(?:buy|accumulate|add)\b")
L1_WEAK_ADX_CONFIRMED_TREND_PATTERN = re.compile(r"(?i)\b(?:confirmed\s+trend|trend\s+confirmed)\b")
L1_LIMITED_EVIDENCE_HIGH_CONVICTION_PATTERN = re.compile(
    r"(?i)\b(?:high[- ]conviction|strong\s+conviction)\b"
)
L1_LIMITED_EVIDENCE_PRECISE_TARGET_PATTERN = re.compile(r"\$\d+\.\d{2}")
L1_HOLD_PARALLEL_BUY_PATTERN = re.compile(
    r"(?i)\b(?:buy|strong buy|buy rating|recommend buy|accumulate|add)\b"
)
L1_REDUCE_PARALLEL_ACCUMULATE_PATTERN = re.compile(r"(?i)\baccumulate\b")
L1_LOW_DATA_DCF_PATTERN = re.compile(
    r"(?ms)^(?P<heading>#{1,6}[ \t]*.*(?:DCF|Discounted Cash Flow).*\n)(?P<body>.*?)(?=^#{1,6}[ \t]+|\Z)"
)

L2_DEFENSIVE_PATTERN = re.compile(r"(?i)\bdefensive\b")
L2_SELL_REASON_PATTERN = re.compile(
    r"(?i)\b(?:catalyst failure|valuation premium|fundamental deterioration|deteriorating fundamentals?|premium valuation)\b"
)


LAYER_1_RULES: tuple[HardRule | StatefulHardRule, ...] = (
    HardRule(
        id="L1_HIGH_RISK_DEFENSIVE",
        severity="blocker",
        summary='High-risk names cannot be described as "defensive".',
        pattern=L1_HIGH_RISK_DEFENSIVE_PATTERN,
        replacement="elevated-risk",
        condition_check=_condition_high_risk,
        apply_fix=lambda text, pattern, _replacement: _normalize_indefinite_articles(
            _simple_substitution(text, pattern, _replace_high_risk_defensive)
        ),
    ),
    HardRule(
        id="L1_LOW_UPSIDE_AGGRESSIVE_BUY",
        severity="blocker",
        summary="Low-upside Buy reports cannot use aggressive accumulation language.",
        pattern=L1_LOW_UPSIDE_AGGRESSIVE_BUY_PATTERN,
        replacement="measured accumulation",
        condition_check=_condition_buy_low_upside,
        apply_fix=_simple_substitution,
    ),
    HardRule(
        id="L1_WEAK_ADX_CONFIRMED_TREND",
        severity="blocker",
        summary="Weak ADX cannot support confirmed-trend language.",
        pattern=L1_WEAK_ADX_CONFIRMED_TREND_PATTERN,
        replacement="trend lacks confirmation",
        condition_check=_condition_weak_adx,
        apply_fix=_simple_substitution,
    ),
    HardRule(
        id="L1_LIMITED_EVIDENCE_HIGH_CONVICTION",
        severity="blocker",
        summary="Limited evidence cannot be framed as high conviction.",
        pattern=L1_LIMITED_EVIDENCE_HIGH_CONVICTION_PATTERN,
        replacement="calibrated allocation",
        condition_check=_condition_limited_evidence,
        apply_fix=_simple_substitution,
    ),
    HardRule(
        id="L1_LIMITED_EVIDENCE_PRECISE_TARGET",
        severity="blocker",
        summary="Limited evidence cannot support a precise two-decimal price target.",
        pattern=L1_LIMITED_EVIDENCE_PRECISE_TARGET_PATTERN,
        replacement=None,
        condition_check=_condition_limited_evidence,
        apply_fix=None,
    ),
    HardRule(
        id="L1_HOLD_PARALLEL_BUY",
        severity="blocker",
        summary="Hold verdict conflicts with explicit Buy language in the body.",
        pattern=L1_HOLD_PARALLEL_BUY_PATTERN,
        replacement=None,
        condition_check=_condition_hold,
        apply_fix=None,
    ),
    HardRule(
        id="L1_REDUCE_PARALLEL_ACCUMULATE",
        severity="blocker",
        summary="Reduce verdict cannot recommend accumulation.",
        pattern=L1_REDUCE_PARALLEL_ACCUMULATE_PATTERN,
        replacement="trim exposure",
        condition_check=_condition_reduce,
        apply_fix=_simple_substitution,
    ),
    HardRule(
        id="L1_LOW_DATA_DCF",
        severity="blocker",
        summary="DCF section must be removed when DCF valuation evidence is disabled.",
        pattern=L1_LOW_DATA_DCF_PATTERN,
        replacement="drop section",
        condition_check=_condition_low_data_dcf,
        apply_fix=lambda text, pattern, _replacement: _drop_dcf_sections(text, pattern)[0],
    ),
    StatefulHardRule(
        id="L1_MULTI_VERDICT_AUTHORITY",
        severity="blocker",
        summary=_summary_multi_verdict,
        condition_check=lambda decision_data, summary, evidence_data, extracted: (
            len(extracted.unique_verdicts()) > 1 and _decision_has_tax_verdict(decision_data)
        ),
        evidence=_state_verdict_evidence,
        replacement=None,
        apply_fix=None,
    ),
    StatefulHardRule(
        id="L1_MULTI_RISK_LEVELS",
        severity="blocker",
        summary=_summary_multi_risk,
        condition_check=lambda decision_data, summary, evidence_data, extracted: len(extracted.unique_risks()) > 1,
        evidence=_state_risk_evidence,
        replacement=None,
        apply_fix=None,
    ),
    StatefulHardRule(
        id="L1_VERDICT_ACTION_MISALIGNED",
        severity="blocker",
        summary=_summary_action_misaligned,
        condition_check=lambda decision_data, summary, evidence_data, extracted: bool(
            _misaligned_actions(decision_data, extracted)
        ),
        evidence=_state_action_evidence,
        replacement=None,
        apply_fix=None,
    ),
    # Phase 5 consolidation: verdict-level rules are OBSERVER-ONLY (no silent text mutation).
    # Body verdict references that conflict with authoritative DecisionState
    # get flagged for the reconciliation audit instead of being auto-rewritten —
    # an auto-rewrite can change report semantics in unexpected ways (e.g.
    # turning "no Buy thesis" into "no accumulation candidate thesis").
    StatefulHardRule(
        id="L1_REDUCE_VERDICT_IN_HOLD_REPORT",
        severity="warn",                 # demoted from "blocker" → "warn" (observer)
        summary=lambda decision_data, summary, evidence_data, extracted: (
            "Body mentions Reduce verdict while top-level is Hold."
        ),
        condition_check=lambda decision_data, summary, evidence_data, extracted: (
            _canonical_verdict(decision_data) == "Hold" and "Reduce" in extracted.unique_verdicts()
        ),
        evidence=_state_verdict_evidence,
        replacement=None,                # no auto-fix — observer only
        apply_fix=None,
    ),
    StatefulHardRule(
        id="L1_BUY_VERDICT_IN_HOLD_REPORT",
        severity="warn",                 # demoted from "blocker" → "warn" (observer)
        summary=lambda decision_data, summary, evidence_data, extracted: (
            "Body mentions Buy verdict while top-level is Hold."
        ),
        condition_check=lambda decision_data, summary, evidence_data, extracted: (
            _canonical_verdict(decision_data) == "Hold" and "Buy" in extracted.unique_verdicts()
        ),
        evidence=_state_verdict_evidence,
        replacement=None,                # no auto-fix — observer only
        apply_fix=None,
    ),
)


LAYER_2_RULES: tuple[SemanticRule, ...] = (
    SemanticRule(
        id="L2_BUY_LOW_UPSIDE_NO_JUSTIFICATION",
        summary="Buy verdict with sub-10% upside requires an income or strategic-allocation rationale.",
        condition_check=_condition_buy_low_upside,
        justification_present=_justification_buy_low_upside,
        failure_excerpt=lambda text: _text_excerpt(text),
    ),
    SemanticRule(
        id="L2_BUY_RISK_HIGH_NO_HEDGE",
        summary="High-risk Buy reports must include sizing or phased-entry guidance.",
        condition_check=lambda decision_data, summary, evidence_data: (
            _canonical_verdict(decision_data) == "Buy" and _condition_high_risk(decision_data, summary, evidence_data)
        ),
        justification_present=_justification_buy_risk_high,
        failure_excerpt=lambda text: _text_excerpt(text),
    ),
    SemanticRule(
        id="L2_SELL_NO_REPLACEMENT_THESIS",
        summary="Sell verdict must explain the exit thesis.",
        condition_check=lambda decision_data, summary, evidence_data: _canonical_verdict(decision_data) == "Sell",
        justification_present=_justification_sell_replacement,
        failure_excerpt=lambda text: _failure_excerpt_for_pattern(text, L2_SELL_REASON_PATTERN, text),
    ),
    SemanticRule(
        id="L2_DEFENSIVE_HIGH_BETA",
        summary='Using "defensive" conflicts with beta >= 1.3.',
        condition_check=lambda decision_data, summary, evidence_data: (
            (_safe_float(_as_dict(summary).get("beta"), 0.0) or 0.0) >= 1.3
        ),
        justification_present=_no_justification,
        failure_excerpt=lambda text: _failure_excerpt_for_pattern(text, L2_DEFENSIVE_PATTERN, text),
    ),
)


def scan(
    report_text: str,
    decision_data: dict,
    evidence_data: dict | None = None,
    summary: dict | None = None,
) -> ScanResult:
    """
    Run all Layer 1 + Layer 2 rules. Apply auto-fixes where deterministic.
    Returns ScanResult with original contradictions, fixed text, and unfixable list.
    """
    text = report_text if isinstance(report_text, str) else str(report_text or "")
    decision_dict = _as_dict(decision_data)
    evidence_dict = _as_dict(evidence_data)
    summary_dict = _as_dict(summary)
    extracted = extract_state(text)

    contradictions: list[Contradiction] = []
    fixed = text

    for rule in LAYER_1_RULES:
        if isinstance(rule, StatefulHardRule):
            try:
                if not rule.condition_check(decision_dict, summary_dict, evidence_dict, extracted):
                    continue
            except Exception:
                continue

            contradiction = Contradiction(
                rule_id=rule.id,
                layer=1,
                severity=rule.severity,
                summary=rule.summary(decision_dict, summary_dict, evidence_dict, extracted),
                evidence=rule.evidence(fixed, decision_dict, summary_dict, evidence_dict, extracted),
                auto_fix=rule.replacement,
            )
            contradictions.append(contradiction)

            if rule.replacement is not None and rule.apply_fix is not None:
                fixed = rule.apply_fix(fixed, extracted)
            continue

        try:
            if not rule.condition_check(decision_dict, summary_dict, evidence_dict):
                continue
        except Exception:
            continue

        matches = list(rule.pattern.finditer(fixed))
        if not matches:
            continue

        snippets = [re.sub(r"\s+", " ", match.group(0)).strip() for match in matches[:3]]
        if rule.id == "L1_LOW_DATA_DCF":
            snippets = [re.sub(r"\s+", " ", match.group("heading")).strip() for match in matches[:3]]

        contradiction = Contradiction(
            rule_id=rule.id,
            layer=1,
            severity=rule.severity,
            summary=rule.summary,
            evidence=snippets,
            auto_fix=rule.replacement,
        )
        contradictions.append(contradiction)

        if rule.replacement is not None and rule.apply_fix is not None:
            fixed = rule.apply_fix(fixed, rule.pattern, rule.replacement)

    for rule in LAYER_2_RULES:
        try:
            condition_met = rule.condition_check(decision_dict, summary_dict, evidence_dict)
        except Exception:
            condition_met = False

        if rule.id == "L2_DEFENSIVE_HIGH_BETA":
            beta = _safe_float(summary_dict.get("beta"), 0.0) or 0.0
            condition_met = beta >= 1.3 and bool(L2_DEFENSIVE_PATTERN.search(fixed))

        if not condition_met:
            continue

        try:
            justified = rule.justification_present(fixed)
        except Exception:
            justified = False

        if justified:
            continue

        contradictions.append(
            Contradiction(
                rule_id=rule.id,
                layer=2,
                severity="warn",
                summary=rule.summary,
                evidence=[rule.failure_excerpt(fixed)],
                auto_fix=None,
            )
        )

    unfixed = [item for item in contradictions if item.auto_fix is None]
    result = ScanResult(contradictions=contradictions, fixed_text=fixed, unfixed=unfixed)

    ticker_or_id = (
        decision_dict.get("ticker")
        or decision_dict.get("symbol")
        or decision_dict.get("id")
        or decision_dict.get("name")
        or "UNKNOWN"
    )
    fixed_ones = [item for item in contradictions if item.auto_fix is not None]
    logger.info(
        "[ContradictionScanner] %s: detected=%d fixed=%d unfixed=%d blockers=%d",
        ticker_or_id,
        len(contradictions),
        len(fixed_ones),
        len(unfixed),
        sum(1 for item in contradictions if item.severity == "blocker"),
    )
    return result


if __name__ == "__main__":
    scenarios = [
        {
            "name": "Clean report",
            "report_text": "Maintain steady allocation.",
            "decision_data": {"tax_verdict": "Hold", "tax_evidence": "Moderate", "tax_risk": "Moderate"},
            "summary": {},
            "evidence_data": None,
        },
        {
            "name": "High-risk defensive",
            "report_text": "This remains a defensive core holding despite elevated volatility.",
            "decision_data": {"tax_risk": "High"},
            "summary": {},
            "evidence_data": None,
        },
        {
            "name": "Buy + low upside, no justification",
            "report_text": "Buy the stock, but near-term upside is limited and timing remains neutral.",
            "decision_data": {"tax_verdict": "Buy", "tax_risk": "Moderate"},
            "summary": {"upside_pct": 5},
            "evidence_data": None,
        },
        {
            "name": "Limited evidence + conviction wording",
            "report_text": "This is a high-conviction buy with target $123.45.",
            "decision_data": {"tax_verdict": "Buy", "tax_evidence": "Limited"},
            "summary": {},
            "evidence_data": None,
        },
        {
            "name": "Multi-verdict authority conflict",
            "report_text": "Verdict: Hold. Fundamental Verdict: Reduce. Final Action: Hold.",
            "decision_data": {"tax_verdict": "Hold"},
            "summary": {},
            "evidence_data": None,
        },
        {
            "name": "Action misaligned",
            "report_text": "Verdict: Hold. Action: Scale In Gradually. Maintain position.",
            "decision_data": {"tax_verdict": "Hold"},
            "summary": {},
            "evidence_data": None,
        },
    ]

    for scenario in scenarios:
        print(f"=== {scenario['name']} ===")
        result = scan(
            report_text=scenario["report_text"],
            decision_data=scenario["decision_data"],
            evidence_data=scenario["evidence_data"],
            summary=scenario["summary"],
        )
        pprint(result.to_dict())
