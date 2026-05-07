from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from core.services.decision_policy import (
    classify_data_coverage_level,
    count_valid_fundamental_fields,
)


ENUMS = {
    "recommendation": {"BUY", "HOLD", "REDUCE", "SELL"},
    "decision_type": {
        "trend_following",
        "contrarian_early",
        "range_hold",
        "risk_off",
        "event_driven",
    },
    "conviction_level": {"low", "medium", "high"},
    "trigger_type": {
        "technical_breakout",
        "trend_confirmation",
        "fundamental_shift",
        "risk_event",
    },
    "severity": {"low", "medium", "high"},
    "tracking_status": {"active", "paused", "closed"},
    "review_cycle": {"daily", "weekly", "monthly"},
    "pilot_status": {"live_pilot", "demo", "archived"},
    "asset_type": {"equity", "crypto", "commodity", "etf", "index"},
    "technical_trend": {"improving", "stable", "deteriorating"},
    "macd_signal": {"bullish_crossover", "bearish_crossover", "neutral"},
    "environment": {"pilot", "production", "demo"},
    "deterministic_scoring": {"deterministic", "probabilistic"},
    "decision_layer": {"llm_assisted", "rule_based", "hybrid"},
}

_SEVERITY_SCORES = {
    "low": 30,
    "medium": 55,
    "medium-high": 70,
    "high": 85,
}

# Simple in-process score cache for delta computation (resets on server restart).
# Key: symbol.upper(), Value: last eisax_score int
_SCORE_CACHE: dict[str, int] = {}


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, "", "N/A", "None"):
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    value_f = _safe_float(value)
    if value_f is None:
        return None
    return int(round(value_f))


def _clamp_int(value: Any, low: int, high: int) -> int:
    value_i = _safe_int(value)
    if value_i is None:
        return low
    return max(low, min(high, value_i))


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _ensure_tz_iso(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.now(timezone.utc).astimezone().tzinfo)
        return parsed.isoformat()
    except Exception:
        return fallback


def _clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def derive_conviction_level(conviction_score: int) -> str:
    if conviction_score <= 39:
        return "low"
    if conviction_score <= 69:
        return "medium"
    return "high"


def derive_fundamental_conviction(eisax_score: int) -> str:
    """Score-based fundamental conviction: HIGH / MEDIUM / LOW."""
    if eisax_score >= 80:
        return "HIGH"
    if eisax_score >= 65:
        return "MEDIUM"
    return "LOW"


def derive_timing_confidence(rsi: float, adx: float) -> str:
    """Technical timing confidence: HIGH / MEDIUM / LOW.
    RSI and ADX only affect timing — never the fundamental verdict.
    """
    if rsi > 70 or adx < 20:
        return "LOW"
    if adx < 25 or rsi > 60:
        return "MEDIUM"
    return "HIGH"  # ADX >= 25 AND RSI <= 60


def _normalize_recommendation(value: str) -> str:
    raw = _clean_text(value).upper()
    if raw in {"STRONG BUY", "TACTICAL BUY", "ACCUMULATE"}:
        return "BUY"
    if raw == "AVOID":
        return "SELL"
    if raw.startswith("BUY"):
        return "BUY"
    if raw.startswith("REDUCE"):
        return "REDUCE"
    if raw.startswith("SELL"):
        return "SELL"
    if raw.startswith("HOLD"):
        return "HOLD"
    raise ValueError(f"Unsupported recommendation: {value}")


def _map_asset_type(symbol: str, market: str, fundamentals: dict[str, Any]) -> str:
    symbol_u = (symbol or "").upper()
    sector = _clean_text(fundamentals.get("sector")).lower()
    industry = _clean_text(fundamentals.get("industry")).lower()
    market_u = (market or "").upper()
    if market_u == "CRYPTO" or symbol_u.endswith("-USD"):
        return "crypto"
    if symbol_u.endswith("=F") or "commodity" in sector or "commodity" in industry:
        return "commodity"
    if "etf" in sector or "etf" in industry or "fund" in industry:
        return "etf"
    return "equity"


def _map_currency(symbol: str, market: str) -> str:
    symbol_u = (symbol or "").upper()
    market_u = (market or "").upper()
    if symbol_u.endswith(".SR") or market_u in {"SAU", "KSA", "SA"}:
        return "SAR"
    if symbol_u.endswith((".AE", ".AD", ".DU")) or market_u in {"UAE", "AE", "DFM", "ADX"}:
        return "AED"
    if symbol_u.endswith(".CA") or market_u in {"EGY", "EGX"}:
        return "EGP"
    if symbol_u.endswith(".KW") or market_u in {"KWT", "KUWAIT"}:
        return "KWD"
    if symbol_u.endswith(".QA") or market_u in {"QAT", "QATAR"}:
        return "QAR"
    return "USD"


def _map_decision_type(
    recommendation: str,
    adx: float | None,
    next_earnings_text: str = "",
) -> str:
    next_earnings_text = _clean_text(next_earnings_text)
    adx = _safe_float(adx) or 0.0
    if next_earnings_text and any(token.isdigit() for token in next_earnings_text):
        try:
            candidate = datetime.fromisoformat(next_earnings_text.split()[0])
            days_to_event = (candidate.date() - datetime.now(candidate.tzinfo or timezone.utc).date()).days
            if 0 <= days_to_event <= 10:
                return "event_driven"
        except Exception:
            pass
    if recommendation == "BUY":
        return "trend_following" if adx >= 25 else "contrarian_early"
    if recommendation == "HOLD":
        return "range_hold"
    return "risk_off"


def _parse_recommendation_from_report(report_text: str) -> str:
    # 1. New format: "Fundamental: **Tactical BUY..." — most authoritative, check first
    match = re.search(
        r'Fundamental[:\s]+\*{0,2}(Tactical\s+BUY|BUY|HOLD|REDUCE|SELL|AVOID)',
        report_text, flags=re.IGNORECASE
    )
    if match:
        return _normalize_recommendation(match.group(1))

    # 2. Old pipe format: "| BUY" / "| HOLD" — skip if preceded by "Last verdict:"
    for m in re.finditer(r'\|\s*(BUY|HOLD|REDUCE|SELL|AVOID)\b', report_text, flags=re.IGNORECASE):
        pre = report_text[max(0, m.start() - 40): m.start()].lower()
        if 'last verdict' not in pre:
            return _normalize_recommendation(m.group(1))

    # 3. Word-boundary fallback — skip "Last verdict:" and "verdict:" context lines
    for m in re.finditer(r'\b(Tactical\s+BUY|BUY|HOLD|REDUCE|SELL|AVOID)\b', report_text, flags=re.IGNORECASE):
        pre = report_text[max(0, m.start() - 40): m.start()].lower()
        if 'last verdict' not in pre and 'verdict:' not in pre:
            return _normalize_recommendation(m.group(1))

    raise ValueError("Could not parse recommendation from report output")


def _parse_score_from_report(report_text: str) -> int:
    # Tolerant pattern: handles all markdown variants seen in live reports
    #   "EisaX Score: 59/100"
    #   "EisaX Score:** 59/100"
    #   "EisaX Score: **59/100**"
    #   "EisaX Score:**59/100**"
    match = re.search(
        r"EisaX Score:\s*\*{0,2}\s*(\d{1,3})\s*/\s*100",
        report_text,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Could not parse EisaX score from report output")
    return _clamp_int(match.group(1), 0, 100)


def _parse_score_components(report_text: str, fallback_score: int) -> dict[str, int]:
    """Return canonical score fields while preserving legacy report text."""
    scorecard = re.search(
        r"EisaX Score:\s*\*{0,2}(\d+)/100\*{0,2}\s*\|\s*Blended:\s*\*{0,2}(\d+)/100",
        report_text,
        flags=re.IGNORECASE,
    )
    if scorecard:
        eisax_score = _clamp_int(scorecard.group(1), 0, 100)
        blended_score = _clamp_int(scorecard.group(2), 0, 100)
    else:
        eisax_score = fallback_score
        blended_score = fallback_score

    fundamental_score = eisax_score
    clarification = re.search(
        r"Verdict Clarification.*?\bScore:\s*\*{0,2}(\d+)/100",
        report_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if clarification:
        fundamental_score = _clamp_int(clarification.group(1), 0, 100)
    else:
        quality = re.search(
            r"Fundamental Quality Score:\s*\*{0,2}(\d+)/100",
            report_text,
            flags=re.IGNORECASE,
        )
        if quality:
            fundamental_score = _clamp_int(quality.group(1), 0, 100)

    return {
        "eisax_score": eisax_score,
        "blended_score": blended_score,
        "fundamental_quality_score": fundamental_score,
    }


def _parse_entry_timing(report_text: str, recommendation: str) -> str:
    for pattern in (
        r"Entry Timing:\s*\*{0,2}([A-Za-z][A-Za-z /-]{1,40})",
        r"Timing:\s*\*{0,2}([A-Za-z][A-Za-z /-]{1,40})",
    ):
        match = re.search(pattern, report_text, flags=re.IGNORECASE)
        if match:
            raw = _clean_text(match.group(1)).upper()
            raw = re.split(r"\s*(?:\||\n|$)", raw)[0].strip()
            if raw:
                if raw.startswith("WAIT"):
                    return "WAIT"
                if raw.startswith("BUY"):
                    return "BUY NOW"
                if "DIP" in raw:
                    return "ADD ON DIP"
                return raw[:40]
    return "WAIT" if recommendation == "HOLD" else "CONFIRM"


def _parse_percent_after_label(report_text: str, label: str) -> int | None:
    match = re.search(
        rf"\b{re.escape(label)}:\s*\*{{0,2}}(\d{{1,3}})%",
        report_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return _clamp_int(match.group(1), 0, 100)


def _parse_level_label(report_text: str, label: str) -> str | None:
    match = re.search(
        rf"\b{re.escape(label)}:\s*\*{{0,2}}(Low|Medium|High)\b",
        report_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).title()


def _report_label_from_score(score: int) -> str:
    if score <= 59:
        return "Low"
    if score <= 74:
        return "Medium"
    return "High"


def _risk_label_from_level(level: str) -> str:
    value = (level or "").strip().upper()
    if value == "HIGH":
        return "High"
    if value in {"MEDIUM", "MODERATE"}:
        return "Medium"
    return "Low"


def _market_beta_risk(fundamentals: dict[str, Any]) -> str:
    beta = _safe_float(fundamentals.get("beta"))
    if beta is None:
        return "LOW"
    if beta >= 1.6:
        return "HIGH"
    if beta >= 1.2:
        return "MEDIUM"
    return "LOW"


def _commodity_cycle_risk(report_text: str, fundamentals: dict[str, Any], risk_map: list[dict[str, Any]]) -> str:
    blob = " ".join(
        [
            report_text or "",
            _clean_text(fundamentals.get("sector")),
            _clean_text(fundamentals.get("industry")),
            " ".join(_clean_text(item.get("risk")) for item in risk_map),
        ]
    ).lower()
    if not any(token in blob for token in ("oil", "commodity", "cyclical", "energy")):
        return "LOW"
    for item in risk_map:
        name = _clean_text(item.get("risk")).lower()
        if any(token in name for token in ("oil", "commodity", "cyclical", "energy")):
            return str(item.get("severity") or "medium").upper()
    return "HIGH"


def _overall_risk_level(risk_map: list[dict[str, Any]]) -> str:
    severities = {str(item.get("severity") or "").lower() for item in risk_map}
    if "high" in severities:
        return "HIGH"
    if "medium" in severities:
        return "MEDIUM"
    return "LOW"


def _build_report_meta(
    *,
    report_text: str,
    score_components: dict[str, int],
    risk_map: list[dict[str, Any]],
    fundamentals: dict[str, Any],
    recommendation: str,
    conviction_score: int,
    low_data_mode: bool,
) -> dict[str, Any]:
    confidence_score = _parse_percent_after_label(report_text, "Verdict Confidence")
    if confidence_score is None:
        confidence_score = conviction_score
    parsed_conviction_score = _parse_percent_after_label(report_text, "Conviction")
    canonical_conviction_score = parsed_conviction_score if parsed_conviction_score is not None else conviction_score
    confidence_label = _parse_level_label(report_text, "Confidence") or _report_label_from_score(confidence_score)
    conviction_label = _parse_level_label(report_text, "Conviction") or _report_label_from_score(canonical_conviction_score)
    if low_data_mode:
        confidence_label = "Low"
        conviction_label = "Low"

    commodity_level = _commodity_cycle_risk(report_text, fundamentals, risk_map)
    overall_level = _overall_risk_level(risk_map)
    if commodity_level == "HIGH":
        overall_level = "HIGH"
    drivers = [_clean_text(item.get("risk")) for item in risk_map if _clean_text(item.get("risk"))][:3]
    if not drivers:
        drivers = ["Execution and monitoring risk"]
    if commodity_level == "HIGH" and not any(
        any(token in driver.lower() for token in ("oil", "commodity", "cyclical", "energy"))
        for driver in drivers
    ):
        drivers = ["Cyclical commodity exposure"] + drivers
    drivers = drivers[:3]

    return {
        "eisax_score": score_components["eisax_score"],
        "blended_score": score_components["blended_score"],
        "fundamental_quality_score": score_components["fundamental_quality_score"],
        "overall_risk_level": overall_level,
        "overall_risk_label": _risk_label_from_level(overall_level),
        "risk_drivers": drivers,
        "market_beta_risk": _market_beta_risk(fundamentals),
        "commodity_cycle_risk": commodity_level,
        "confidence_label": confidence_label,
        "confidence_score": confidence_score,
        "conviction_label": conviction_label,
        "conviction_score": canonical_conviction_score,
        "verdict": recommendation,
        "fundamental_verdict": recommendation,
        "entry_timing": _parse_entry_timing(report_text, recommendation),
    }


def _parse_market_regime(report_text: str) -> tuple[str | None, int | None]:
    match = re.search(
        r"Market Regime:\s*([A-Z-]+).*?Fear\s*&\s*Greed:\s*(\d+)",
        report_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None, None
    return match.group(1).lower(), _safe_int(match.group(2))


def _parse_first_sentence_of_section(report_text: str, heading: str) -> str:
    pattern = rf"{re.escape(heading)}\s*(.*?)(?:\n###|\n##|\n---|\Z)"
    match = re.search(pattern, report_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    section_text = _clean_text(match.group(1))
    if not section_text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", section_text)
    return _clean_text(sentences[0])


def _parse_risk_map(report_text: str) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    section = re.search(
        r"Key Risks\s*(.*?)(?:\n###\s*5\.|\n##|\Z)",
        report_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not section:
        return risks
    for line in section.group(1).splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        match = re.search(
            r"\*\*(.+?)\*\*\s+\(Severity:\s*([A-Za-z-]+)\):"
            r"|\*\*(.+?)\s+\(Severity:\s*([A-Za-z-]+)\):\*\*",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        risk_name = match.group(1) or match.group(3)
        severity_raw = (match.group(2) or match.group(4)).strip().lower()
        severity = "high" if severity_raw in {"high", "medium-high"} else "medium" if severity_raw == "medium" else "low"
        risks.append(
            {
                "risk": _clean_text(risk_name),
                "severity": severity,
                "severity_score": _SEVERITY_SCORES.get(severity_raw, _SEVERITY_SCORES[severity]),
            }
        )
    return risks


def _parse_scenarios(report_text: str) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    marker = re.search(r"Valuation Scenarios\s*\(Probability-Weighted\)", report_text, flags=re.IGNORECASE)
    if not marker:
        return scenarios
    tail = report_text[marker.end():]
    for line in tail.splitlines():
        if line.strip().startswith("*Expected Value:"):
            break
        if line.strip().startswith("### ") or line.strip().startswith("## "):
            break
        if "|" not in line or "Scenario" in line or "---" in line:
            continue
        parts = [_clean_text(part) for part in line.strip().strip("|").split("|")]
        if len(parts) < 5:
            continue
        scenario_name = re.sub(r"^[^\w]+", "", parts[0]).strip() or parts[0]
        weight_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", parts[1])
        target_match = re.search(r"(-?\d+(?:\.\d+)?)", parts[3].replace(",", ""))
        return_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", parts[4])
        weight = _safe_int(weight_match.group(1) if weight_match else None)
        return_pct = _safe_float(return_match.group(1) if return_match else None)
        target_price = _safe_float(target_match.group(1) if target_match else None)
        if weight is None or weight <= 0:
            continue
        scenarios.append(
            {
                "scenario": scenario_name,
                "weight": weight,
                "target_price": round(target_price, 2) if target_price is not None else None,
                "return_pct": round(return_pct, 1) if return_pct is not None else None,
                "summary": _scenario_summary_from_name(scenario_name),
            }
        )
    return scenarios


def _scenario_summary_from_name(name: str) -> str:
    name_l = _clean_text(name).lower()
    if "bear" in name_l:
        return "Downside case driven by weaker sentiment or valuation pressure."
    if "bull" in name_l:
        return "Upside case driven by stronger execution and trend confirmation."
    if "shock" in name_l or "macro" in name_l:
        return "Stress scenario reflecting macro or market-wide risk-off conditions."
    return "Base case reflecting current expectations and steady execution."


