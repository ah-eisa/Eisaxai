from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from core.services.phrase_builder import (
    build_support_phrase,
    build_timing_phrase,
    build_trend_phrase,
    build_volume_phrase,
    build_yield_phrase,
)


@dataclass
class SanitizedResult:
    text: str
    audit_log: list[dict] = field(default_factory=list)
    replacements_made: int = 0


def _sentence_chunks(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [part.strip() for part in parts if part and part.strip()]


def _replacement_for(field_name: str, labels: dict[str, str]) -> str:
    if field_name == "trend_strength":
        return build_trend_phrase(
            labels.get("TrendStrength", ""),
            labels.get("RSIZone", ""),
        )
    if field_name == "yield_quality":
        return build_yield_phrase(labels.get("YieldQuality", ""))
    if field_name == "entry_quality":
        return build_timing_phrase(labels.get("EntryQuality", ""))
    if field_name == "volume_conviction":
        return build_volume_phrase(labels.get("VolumeConviction", ""))
    if field_name == "support_proximity":
        return build_support_phrase(labels.get("SupportProximity", ""))
    if field_name == "momentum_state":
        rsi_zone = labels.get("RSIZone", "")
        if rsi_zone == "bullish momentum":
            return "momentum is improving"
        if rsi_zone == "overbought":
            return "momentum is overbought"
        # neutral momentum / weak momentum / anything else:
        return "momentum is weakening"
    return ""


def _conflict_rules(labels: dict[str, str]) -> list[tuple[str, re.Pattern[str], str]]:
    rules: list[tuple[str, re.Pattern[str], str]] = []
    trend_label = labels.get("TrendStrength", "")
    if trend_label != "strong trend":
        rules.append(("trend_strength", re.compile(r"(?i)\bstrong trend\b"), "strong trend"))
    if trend_label != "confirmed trend":
        rules.append(("trend_strength", re.compile(r"(?i)\bconfirmed trend\b"), "confirmed trend"))
    if trend_label not in {"weak trend"}:
        rules.append(("trend_strength", re.compile(r"(?i)\b(?:weak trend|no trend)\b"), "weak trend"))

    yield_label = labels.get("YieldQuality", "")
    if yield_label == "attractive yield":
        rules.append(("yield_quality", re.compile(r"(?i)\battractive yield\b"), "attractive yield"))
    if yield_label == "high yield":
        rules.append(("yield_quality", re.compile(r"(?i)\bhigh yield\b"), "high yield"))

    entry_label = labels.get("EntryQuality", "")
    if entry_label != "favorable entry":
        rules.append(("entry_quality", re.compile(r"(?i)\b(?:good entry|favorable entry)\b"), "favorable entry"))
    if entry_label != "poor timing":
        rules.append(("entry_quality", re.compile(r"(?i)\bpoor timing\b"), "poor timing"))

    volume_label = labels.get("VolumeConviction", "")
    if volume_label != "strong volume confirmation":
        rules.append(("volume_conviction", re.compile(r"(?i)\bstrong volume(?: confirmation)?\b"), "strong volume confirmation"))

    support_label = labels.get("SupportProximity", "")
    if support_label != "near support":
        rules.append(("support_proximity", re.compile(r"(?i)\bnear support\b"), "near support"))
    if support_label != "extended above support":
        rules.append(("support_proximity", re.compile(r"(?i)\bextended above support\b"), "extended above support"))

    # ── Momentum wording: "momentum is bearish" is only valid when RSI is
    #    clearly bearish (oversold / weak momentum).  Neutral or bullish RSI
    #    must use softened language ("weakening" / "improving").
    rsi_zone = labels.get("RSIZone", "")
    if rsi_zone not in ("oversold",):
        # Any RSI zone other than oversold → "momentum is bearish" is too strong
        rules.append((
            "momentum_state",
            re.compile(r"(?i)\bmomentum\s+is\s+bearish\b"),
            "momentum is bearish",
        ))

    return rules


class InterpretationGuard:
    def audit_and_sanitize(
        self,
        report_text: str,
        interpretation_labels: dict[str, str],
    ) -> SanitizedResult:
        audit_log: list[dict] = []
        replacements = 0
        rewritten_chunks: list[str] = []

        for chunk in _sentence_chunks(report_text):
            replacement = None
            replaced_field = None
            for field_name, pattern, conflicting_label in _conflict_rules(interpretation_labels):
                if not pattern.search(chunk):
                    continue
                expected_text = _replacement_for(field_name, interpretation_labels)
                if not expected_text or conflicting_label == interpretation_labels.get({
                    "trend_strength":    "TrendStrength",
                    "yield_quality":     "YieldQuality",
                    "entry_quality":     "EntryQuality",
                    "volume_conviction": "VolumeConviction",
                    "support_proximity": "SupportProximity",
                    # momentum_state maps to RSIZone; conflicting_label ("momentum is bearish")
                    # will never equal a raw RSIZone value, so the guard never skips this rule.
                    "momentum_state":    "RSIZone",
                }.get(field_name, "_no_match_"), ""):
                    continue
                replacement = expected_text
                replaced_field = field_name
                break

            if replacement and replaced_field:
                replacements += 1
                rewritten_chunks.append(replacement)
                audit_log.append(
                    {
                        "event": "interpretation_override",
                        "field": replaced_field,
                        "llm_text": chunk,
                        "replacement": replacement,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            else:
                rewritten_chunks.append(chunk)

        cleaned_text = "\n\n".join(rewritten_chunks).strip() or report_text
        return SanitizedResult(
            text=cleaned_text,
            audit_log=audit_log,
            replacements_made=replacements,
        )
