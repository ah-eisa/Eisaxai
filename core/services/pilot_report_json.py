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
    match = re.search(r"EisaX Score:\s*(\d+)/100", report_text, flags=re.IGNORECASE)
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


def normalize_scenarios(scenarios: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(scenarios) < 2:
        raise ValueError("scenario_analysis requires at least 2 scenarios")
    weights = []
    for scenario in scenarios:
        weight = _safe_int(scenario.get("weight"))
        if weight is None or weight <= 0:
            raise ValueError("scenario_analysis weights must be positive integers")
        scenario["weight"] = weight
        weights.append(weight)
    total = sum(weights)
    if total <= 0:
        raise ValueError("scenario_analysis total weight must be positive")
    probs = [round(weight / total, 4) for weight in weights]
    validation = {
        "scenario_weights_sum": total,
        "scenario_probabilities_normalized": True,
        "normalized_probabilities": probs,
    }
    return scenarios, validation


def _build_fallback_scenarios(
    summary: dict[str, Any],
    fundamentals: dict[str, Any],
) -> list[dict[str, Any]]:
    price = _safe_float(summary.get("price"))
    analyst_target = _safe_float(fundamentals.get("analyst_target"))
    week52_high = _safe_float(fundamentals.get("week52_high"))
    week52_low = _safe_float(fundamentals.get("week52_low"))
    if price is None:
        return []

    bear_target = week52_low or round(price * 0.85, 2)
    base_target = analyst_target or round(price * 1.05, 2)
    bull_target = max(
        value for value in (
            week52_high or 0,
            round(base_target * 1.15, 2),
            round(price * 1.20, 2),
        )
    )
    raw = [
        ("Bear", 25, bear_target, "Downside case reflecting weaker execution or de-rating."),
        ("Base", 50, base_target, "Base case reflecting current consensus and steady execution."),
        ("Bull", 25, bull_target, "Upside case reflecting stronger execution and confirmation."),
    ]
    scenarios = []
    for name, weight, target_price, summary_text in raw:
        if target_price is None:
            continue
        return_pct = round(((target_price - price) / price) * 100, 1)
        scenarios.append(
            {
                "scenario": name,
                "weight": weight,
                "target_price": round(target_price, 2),
                "return_pct": return_pct,
                "summary": summary_text,
            }
        )
    return scenarios


def _build_why_this_decision(
    recommendation: str,
    summary: dict[str, Any],
    fundamentals: dict[str, Any],
    language: str = "en",
) -> list[str]:
    _is_ar = language.lower().startswith("ar")
    rec = (recommendation or "").upper()
    adx  = _safe_float(summary.get("adx"))  or 0.0
    rsi  = _safe_float(summary.get("rsi"))  or 50.0
    beta = _safe_float(fundamentals.get("beta")) or 1.0

    if _is_ar:
        if rec == "BUY":
            reasons = ["الاتجاه الفني مؤكد ويدعم موقفًا إيجابيًا."]
            if adx >= 25:
                reasons.append(f"ADX عند {adx:.0f} يؤكد قوة الاتجاه الصاعد.")
            if rsi < 65:
                reasons.append("مؤشر الزخم (RSI) في نطاق صحي بعيدًا عن التشبع.")
            if beta < 1.5:
                reasons.append("مستوى المخاطر مقبول نسبيًا مقارنةً بالسوق.")
            reasons.append("الأساسيات تدعم استمرار الزخم الإيجابي.")
            return reasons[:5]
        if rec == "HOLD":
            reasons = ["التوقيت الحالي يستوجب الانتظار قبل أي إضافة."]
            if adx < 25:
                reasons.append("مؤشر ADX لم يصل بعد لمستوى تأكيد الاتجاه (25).")
            if rsi > 65:
                reasons.append("مؤشر RSI قريب من منطقة التشبع الشرائي — يُفضّل الانتظار.")
            reasons.append("احتفظ بالموقف الحالي وراقب مستويات الدعم والمحفزات القادمة.")
            return reasons[:5]
        if rec in ("REDUCE", "SELL"):
            reasons = ["إشارات المخاطر الحالية تستدعي تخفيف الانكشاف."]
            if beta > 1.5:
                reasons.append(f"معامل بيتا المرتفع ({beta:.1f}) يزيد من تقلبات المحفظة.")
            if adx < 20:
                reasons.append("ضعف الاتجاه الفني يرفع احتمالية الانعكاس.")
            reasons.append("راجع مستويات الوقف وخفّف التعرض تدريجيًا.")
            return reasons[:5]
        return ["لا توجد إشارات واضحة بما يكفي لتحديد الموقف في الوقت الحالي."]

    # English path — keep original logic exactly as-is
    reasons: list[str] = []
    revenue_growth = _safe_float(fundamentals.get("revenue_growth"))
    gross_margin = _safe_float(fundamentals.get("gross_margin"))
    forward_pe = _safe_float(fundamentals.get("forward_pe"))
    trend = _clean_text(summary.get("trend")).lower()
    momentum = _clean_text(summary.get("momentum")).lower()

    if revenue_growth is not None and gross_margin is not None:
        reasons.append(
            f"Fundamental quality remains supported by revenue growth of {revenue_growth:.1f}% and gross margin of {gross_margin:.1f}%."
        )
    elif revenue_growth is not None:
        reasons.append(f"Fundamental growth remains positive with revenue growth of {revenue_growth:.1f}%.")

    if recommendation == "BUY":
        reasons.append(
            "Technical posture supports upside participation, but execution still depends on confirmation holding."
            if adx < 25 else
            "Trend conditions are sufficiently constructive to justify active upside participation."
        )
    elif recommendation == "HOLD":
        reasons.append(
            "The asset remains fundamentally credible, but the current setup does not justify a higher-action stance."
        )
    else:
        reasons.append(
            "Risk-adjusted downside remains more important than upside until the technical structure stabilizes."
        )

    if forward_pe is not None:
        reasons.append(
            f"Forward valuation at {forward_pe:.1f}x still requires disciplined execution against trend and risk."
        )

    if adx:
        reasons.append(
            f"ADX at {adx:.1f} indicates {'weak' if adx < 20 else 'developing' if adx < 25 else 'confirmed'} trend strength, while RSI is {rsi:.1f}."
        )
    elif trend or momentum:
        reasons.append(f"Trend is {trend or 'mixed'} while momentum is {momentum or 'mixed'} at current levels.")

    cleaned = [_clean_text(item) for item in reasons if _clean_text(item)]
    if len(cleaned) < 2:
        cleaned.append("Current evidence supports discipline over aggressive positioning.")
    return cleaned[:5]


def _build_fallback_risk_map(summary: dict[str, Any], fundamentals: dict[str, Any], language: str = "en") -> list[dict[str, Any]]:
    if language.lower().startswith("ar"):
        risks = [
            {"label": "مخاطر التقلب", "description": "ارتفاع معامل بيتا يزيد من تقلبات الأصل مقارنةً بالسوق.", "severity": "medium", "severity_score": 55},
            {"label": "مخاطر التقييم", "description": "ضغط على مضاعفات التقييم في حال تراجع الأرباح أو رفع الفائدة.", "severity": "medium", "severity_score": 50},
            {"label": "ضعف التأكيد الفني", "description": "مؤشر ADX لم يصل لمستوى تأكيد الاتجاه — الزخم هش.", "severity": "low", "severity_score": 35},
            {"label": "مخاطر التنفيذ والمراقبة", "description": "مراقبة المحفزات والمستويات التقنية ضرورية لإدارة الموقف.", "severity": "low", "severity_score": 30},
        ]
        return risks
    risk_map: list[dict[str, Any]] = []
    beta = _safe_float(fundamentals.get("beta")) or 1.0
    forward_pe = _safe_float(fundamentals.get("forward_pe")) or 0.0
    adx = _safe_float(summary.get("adx")) or 0.0

    if beta >= 1.5:
        risk_map.append(
            {
                "risk": "Volatility and beta sensitivity",
                "severity": "high" if beta >= 2 else "medium",
                "severity_score": 85 if beta >= 2 else 60,
            }
        )
    if forward_pe >= 20:
        risk_map.append(
            {
                "risk": "Valuation compression risk",
                "severity": "high" if forward_pe >= 30 else "medium",
                "severity_score": 80 if forward_pe >= 30 else 55,
            }
        )
    if adx < 20:
        risk_map.append(
            {
                "risk": "Weak trend confirmation",
                "severity": "medium",
                "severity_score": 55,
            }
        )
    return risk_map or [
        {
            "risk": "Execution and monitoring risk",
            "severity": "medium",
            "severity_score": 50,
        }
    ]


def _build_triggers(
    recommendation: str,
    summary: dict[str, Any],
    fundamentals: dict[str, Any],
    language: str = "en",
) -> dict[str, Any]:
    if language.lower().startswith("ar"):
        _tgt = _safe_float(fundamentals.get("analyst_target_price")) or _safe_float(fundamentals.get("fair_value"))
        _stop = _safe_float(summary.get("sma200")) or _safe_float(fundamentals.get("stop_price"))
        _tgt_s  = f"{_tgt:.2f}" if _tgt else "مستوى الهدف"
        _stop_s = f"{_stop:.2f}" if _stop else "مستوى الوقف"
        return {
            "upgrade_trigger": {
                "type": "technical_breakout",
                "condition_text": f"تجاوز {_tgt_s} مع حجم تداول مرتفع وتأكيد الاتجاه",
                "action": "رفع التوصية إلى شراء",
            },
            "downgrade_trigger": {
                "type": "risk_event",
                "condition_text": f"كسر مستوى {_stop_s} أو تدهور الأساسيات",
                "action": "تخفيف التوصية أو الخروج الجزئي",
            },
            "thesis_break": {
                "type": "fundamental_shift",
                "condition_text": "تراجع جوهري في الأرباح أو تغيّر في البيئة الكلية",
                "action": "مراجعة كاملة للموقف والخروج عند تأكيد الانعكاس",
            },
        }
    price = _safe_float(summary.get("price")) or 0.0
    sma50 = _safe_float(summary.get("sma_50"))
    sma200 = _safe_float(summary.get("sma_200"))
    analyst_target = _safe_float(fundamentals.get("analyst_target"))
    week52_high = _safe_float(fundamentals.get("week52_high"))
    week52_low = _safe_float(fundamentals.get("week52_low"))
    resistance = next((value for value in (week52_high, analyst_target, sma50) if value and value > price), price * 1.05 if price else 0.0)
    support = next((value for value in (sma50, sma200, week52_low) if value and (not price or value < price * 1.02)), week52_low or sma200 or price * 0.95 if price else 0.0)
    upgrade_action = "Upgrade to BUY" if recommendation != "BUY" else "Maintain BUY with higher conviction"
    downgrade_action = "Downgrade to REDUCE" if recommendation == "HOLD" else "Downgrade to SELL"
    return {
        "upgrade_trigger": {
            "type": "technical_breakout",
            "condition_text": f"Close above {resistance:,.2f} with ADX above 20 and volume confirmation." if resistance else "Trend breakout with stronger breadth and volume confirmation.",
            "action": upgrade_action,
        },
        "downgrade_trigger": {
            "type": "risk_event",
            "condition_text": f"Break below {support:,.2f} on elevated volume." if support else "Support failure on elevated risk conditions.",
            "action": downgrade_action,
        },
        "thesis_break": {
            "type": "fundamental_shift",
            "condition_text": "Sustained price weakness below SMA200 together with a material deterioration in growth or profitability.",
            "action": "Invalidate current thesis",
        },
    }


def _build_what_would_make_me_wrong(
    recommendation: str,
    summary: dict[str, Any],
    fundamentals: dict[str, Any],
    language: str = "en",
) -> list[str]:
    if language.lower().startswith("ar"):
        rec = (recommendation or "").upper()
        if rec == "BUY":
            return [
                "تراجع حاد في الأرباح أو تخفيض التوقعات الرسمية.",
                "كسر مستوى الدعم الرئيسي مع حجم بيع مرتفع.",
                "تصاعد المخاطر الكلية أو تغيّر حاد في سياسة الفائدة.",
            ]
        if rec in ("HOLD", "REDUCE"):
            return [
                "اختراق واضح فوق مستوى المقاومة مع تأكيد الزخم.",
                "مفاجأة إيجابية في الأرباح أو إعادة تقييم الأساسيات.",
                "تحسّن حاد في بيئة المخاطر الكلية.",
            ]
        return [
            "أي تحسّن جوهري في الأساسيات أو البيئة الكلية قد يغيّر هذا الرأي.",
        ]
    analyst_target = _safe_float(fundamentals.get("analyst_target"))
    sma200 = _safe_float(summary.get("sma_200"))
    items = []
    if recommendation in {"HOLD", "REDUCE", "SELL"} and analyst_target:
        items.append(
            f"A sustained move toward {analyst_target:,.2f} with improving ADX would prove the current caution too conservative."
        )
    if recommendation == "BUY" and sma200:
        items.append(
            f"A decisive breakdown below SMA200 at {sma200:,.2f} would invalidate the constructive stance."
        )
    items.append("A material deterioration in execution quality or forward growth would invalidate the current thesis.")
    return [_clean_text(item) for item in items if _clean_text(item)]


def _build_status_summary(report_text: str, recommendation: str, language: str = "en") -> str:
    _is_ar = language.lower().startswith("ar")
    # Try English heading first, then Arabic heading
    _patterns = [
        r'###\s*1[\.:]?\s*Executive Summary\s*\n+([\s\S]{30,400}?)(?=\n###|\Z)',
        r'###\s*1[\.:]?\s*الملخص التنفيذي\s*\n+([\s\S]{30,400}?)(?=\n###|\Z)',
    ]
    for _pat in _patterns:
        _m = re.search(_pat, report_text, re.IGNORECASE)
        if _m:
            _text = _m.group(1).strip()
            _first = re.split(r'(?<=[.!?])\s+', _text)
            if _first and len(_first[0]) > 15:
                return _clean_text(_first[0])
    rec = (recommendation or "").upper()
    if _is_ar:
        if rec == "BUY":
            return "جودة الأصل والتوقيت يدعمان موقفًا إيجابيًا."
        if rec in ("HOLD", "REDUCE"):
            return "الأساسيات إيجابية، لكن التوقيت لا يزال يحتاج تأكيدًا."
        return "المخاطر تطغى حاليًا على الفرصة."
    if rec == "BUY":
        return "Asset quality and timing support a constructive stance."
    if rec in ("HOLD", "REDUCE"):
        return "Asset quality is constructive, but timing remains mixed."
    return "Risk conditions currently outweigh the upside case."


def _map_technical_trend(summary: dict[str, Any]) -> str:
    trend = _clean_text(summary.get("trend")).lower()
    momentum = _clean_text(summary.get("momentum")).lower()
    if trend == "bullish" or momentum == "bullish":
        return "improving"
    if trend == "bearish" and momentum == "bearish":
        return "deteriorating"
    return "stable"


def _map_macd_signal(summary: dict[str, Any]) -> str:
    macd = _safe_float(summary.get("macd"))
    signal = _safe_float(summary.get("macd_signal"))
    if macd is None or signal is None:
        return "neutral"
    if macd > signal:
        return "bullish_crossover"
    if macd < signal:
        return "bearish_crossover"
    return "neutral"


def _build_market_snapshot(
    summary: dict[str, Any],
    fundamentals: dict[str, Any],
    timestamp_iso: str,
) -> dict[str, Any] | None:
    snapshot: dict[str, Any] = {}
    price = _safe_float(summary.get("price"))
    market_cap = _safe_float(fundamentals.get("market_cap"))
    volume = _safe_float(fundamentals.get("volume_today"))
    if price is not None:
        snapshot["live_price"] = round(price, 2)
        snapshot["snapshot_time"] = timestamp_iso
    if market_cap is not None:
        snapshot["market_cap"] = round(market_cap)
    if volume is not None:
        snapshot["volume"] = round(volume)
    return snapshot or None


def _build_technical_view(summary: dict[str, Any], fundamentals: dict[str, Any]) -> dict[str, Any] | None:
    technical_view: dict[str, Any] = {
        "trend": _map_technical_trend(summary),
        "macd_signal": _map_macd_signal(summary),
    }
    rsi = _safe_float(summary.get("rsi"))
    adx = _safe_float(summary.get("adx"))
    if rsi is not None:
        technical_view["rsi"] = round(rsi, 1)
    if adx is not None:
        technical_view["adx"] = round(adx, 1)

    support_levels = [
        level
        for level in (
            _safe_float(summary.get("sma_50")),
            _safe_float(summary.get("sma_200")),
            _safe_float(fundamentals.get("week52_low")),
        )
        if level is not None
    ]
    resistance_levels = [
        level
        for level in (
            _safe_float(fundamentals.get("week52_high")),
            _safe_float(fundamentals.get("analyst_target")),
        )
        if level is not None
    ]
    if support_levels:
        technical_view["support_levels"] = [round(level, 2) for level in support_levels]
    if resistance_levels:
        technical_view["resistance_levels"] = [round(level, 2) for level in resistance_levels]
    return technical_view if len(technical_view) > 2 else None


def _build_fundamental_view(fundamentals: dict[str, Any]) -> dict[str, Any] | None:
    fundamental_view: dict[str, Any] = {}
    mapping = {
        "revenue_growth_yoy_pct": fundamentals.get("revenue_growth"),
        "gross_margin_pct": fundamentals.get("gross_margin"),
        "roe_pct": fundamentals.get("roe"),
    }
    for key, value in mapping.items():
        value_f = _safe_float(value)
        if value_f is not None:
            fundamental_view[key] = round(value_f, 1)
    forward_pe = _safe_float(fundamentals.get("forward_pe"))
    if forward_pe is not None:
        if forward_pe >= 25:
            comment = "Premium valuation leaves less room for disappointment."
        elif forward_pe >= 15:
            comment = "Valuation looks balanced relative to the current execution profile."
        else:
            comment = "Valuation appears undemanding relative to current fundamentals."
        fundamental_view["valuation_commentary"] = comment
    return fundamental_view or None


def _build_macro_context(report_text: str) -> dict[str, Any] | None:
    regime, fear_greed = _parse_market_regime(report_text)
    if regime is None and fear_greed is None:
        return None
    macro_context: dict[str, Any] = {}
    if regime is not None:
        macro_context["market_regime"] = regime
        regime_map = {
            "risk-on": "Risk appetite is constructive and supportive of upside participation.",
            "cautious": "Risk appetite remains selective and favors disciplined entry timing.",
            "risk-off": "Macro conditions are defensive and elevate downside sensitivity.",
            "neutral": "Macro conditions are balanced with no clear directional tailwind.",
        }
        macro_context["macro_summary"] = regime_map.get(regime, "Macro conditions are being monitored for directional confirmation.")
    if fear_greed is not None:
        macro_context["fear_greed_index"] = fear_greed
    return macro_context


def _strip_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {
            key: _strip_nulls(item)
            for key, item in value.items()
            if item is not None
        }
        return {key: item for key, item in cleaned.items() if item not in ({}, [], None)}
    if isinstance(value, list):
        cleaned_list = [_strip_nulls(item) for item in value]
        return [item for item in cleaned_list if item not in ({}, [], None)]
    return value


def validate_report_json(report_json: dict[str, Any]) -> dict[str, Any]:
    required_top = {
        "report_id",
        "generated_at",
        "system",
        "data_context",
        "asset",
        "headline_view",
        "decision_framework",
        "triggers",
        "risk_map",
        "what_would_make_me_wrong",
        "monitoring",
        "compliance",
    }
    missing = [key for key in sorted(required_top) if key not in report_json]
    if missing:
        raise ValueError(f"Missing required top-level keys: {', '.join(missing)}")

    if not _clean_text(report_json["report_id"]):
        raise ValueError("report_id must be a non-empty string")
    for path in (
        report_json["generated_at"],
        report_json["data_context"].get("data_as_of"),
    ):
        try:
            parsed = datetime.fromisoformat(str(path))
            if parsed.tzinfo is None:
                raise ValueError
        except Exception as exc:
            raise ValueError(f"Invalid timezone-aware ISO8601 timestamp: {path}") from exc

    headline = report_json["headline_view"]
    if headline.get("recommendation") not in ENUMS["recommendation"]:
        raise ValueError("recommendation is outside the allowed enum")
    if headline.get("decision_type") not in ENUMS["decision_type"]:
        raise ValueError("decision_type is outside the allowed enum")
    conviction_score = _clamp_int(headline.get("conviction_score"), 0, 100)
    headline["conviction_score"] = conviction_score
    headline["conviction_level"] = derive_conviction_level(conviction_score)
    headline["eisax_score"] = _clamp_int(headline.get("eisax_score"), 0, 100)

    system = report_json["system"]
    if system.get("environment") not in ENUMS["environment"]:
        raise ValueError("system.environment is outside the allowed enum")
    determinism = system.get("determinism") or {}
    if determinism.get("scoring") not in ENUMS["deterministic_scoring"]:
        raise ValueError("system.determinism.scoring is outside the allowed enum")
    if determinism.get("decision_layer") not in ENUMS["decision_layer"]:
        raise ValueError("system.determinism.decision_layer is outside the allowed enum")

    asset = report_json["asset"]
    if asset.get("asset_type") not in ENUMS["asset_type"]:
        raise ValueError("asset.asset_type is outside the allowed enum")

    why_items = report_json["decision_framework"].get("why_this_decision") or []
    why_items = [_clean_text(item) for item in why_items if _clean_text(item)]
    if len(why_items) < 2:
        raise ValueError("why_this_decision must contain at least 2 items")
    report_json["decision_framework"]["why_this_decision"] = why_items[:5]

    triggers = report_json["triggers"]
    for key in ("upgrade_trigger", "downgrade_trigger", "thesis_break"):
        trigger = triggers.get(key)
        if not isinstance(trigger, dict):
            raise ValueError(f"triggers.{key} is missing")
        if trigger.get("type") not in ENUMS["trigger_type"]:
            raise ValueError(f"triggers.{key}.type is outside the allowed enum")
        for field in ("condition_text", "action"):
            if not _clean_text(trigger.get(field)):
                raise ValueError(f"triggers.{key}.{field} must be non-empty")

    risk_map = report_json.get("risk_map") or []
    if not risk_map:
        raise ValueError("risk_map must contain at least 1 item")
    for risk in risk_map:
        if risk.get("severity") not in ENUMS["severity"]:
            raise ValueError("risk_map severity is outside the allowed enum")
        risk["severity_score"] = _clamp_int(risk.get("severity_score"), 0, 100)

    wrong_items = report_json.get("what_would_make_me_wrong") or []
    wrong_items = [_clean_text(item) for item in wrong_items if _clean_text(item)]
    if not wrong_items:
        raise ValueError("what_would_make_me_wrong must contain at least 1 item")
    report_json["what_would_make_me_wrong"] = wrong_items

    monitoring = report_json["monitoring"]
    if monitoring.get("tracking_status") not in ENUMS["tracking_status"]:
        raise ValueError("monitoring.tracking_status is outside the allowed enum")
    if monitoring.get("next_review_cycle") not in ENUMS["review_cycle"]:
        raise ValueError("monitoring.next_review_cycle is outside the allowed enum")

    compliance = report_json["compliance"]
    if compliance.get("pilot_status") not in ENUMS["pilot_status"]:
        raise ValueError("compliance.pilot_status is outside the allowed enum")

    technical_view = report_json.get("technical_view")
    if technical_view:
        if technical_view.get("trend") not in ENUMS["technical_trend"]:
            raise ValueError("technical_view.trend is outside the allowed enum")
        if technical_view.get("macd_signal") not in ENUMS["macd_signal"]:
            raise ValueError("technical_view.macd_signal is outside the allowed enum")

    scenario_analysis = report_json.get("scenario_analysis")
    if scenario_analysis:
        _, validation = normalize_scenarios(scenario_analysis)
        report_json["validation"] = validation
        total_probability = sum(validation["normalized_probabilities"])
        if abs(total_probability - 1.0) > 0.001:
            raise ValueError("normalized scenario probabilities must sum to 1.0 +/- 0.001")

    return _strip_nulls(report_json)


def build_pilot_report_json(
    *,
    symbol: str,
    market: str,
    language: str,
    report_text: str,
    analysis_data: dict[str, Any] | None,
    system_version: str,
    model_primary: str = "DeepSeek V3",
    generated_at: str | None = None,
    data_as_of: str | None = None,
    latency_seconds: int = 0,
) -> dict[str, Any]:
    analysis_data = analysis_data or {}
    summary = analysis_data.get("analytics") or {}
    fundamentals = analysis_data.get("fundamentals") or {}
    trust_layer = analysis_data.get("trust_layer") or {}
    coverage_count = count_valid_fundamental_fields(fundamentals)
    report_text_lower = str(report_text or "").lower()
    has_low_data_marker = any(
        marker in report_text_lower
        for marker in (
            "low-data mode",
            "fundamental data coverage is limited",
            "fundamental visibility is limited",
            "peer comparison is disabled",
            "valuation scenarios are disabled",
        )
    )
    has_high_data_evidence = bool(
        re.search(
            r"\|\s*Scenario\s*\|\s*Multiple\s*\||\|\s*Ticker\s*\|.*Fwd P/E|"
            r"Analyst consensus is|mean price target|Forward P/E|Gross Margin|Revenue Growth",
            str(report_text or ""),
            re.IGNORECASE | re.DOTALL,
        )
    )
    if coverage_count <= 3 and has_high_data_evidence and not has_low_data_marker:
        coverage_count = 7
    coverage_level = classify_data_coverage_level(coverage_count)
    low_data_mode = coverage_level in {"technical_only", "low"}
    now_iso = _iso_now()
    generated_at = _ensure_tz_iso(generated_at, now_iso)
    data_as_of = _ensure_tz_iso(data_as_of, generated_at)

    recommendation = _parse_recommendation_from_report(report_text)
    eisax_score = _parse_score_from_report(report_text)
    score_components = _parse_score_components(report_text, eisax_score)
    eisax_score = score_components["eisax_score"]
    # Score delta — compare to last cached score for this symbol
    _sym_key = _clean_text(symbol).upper()
    _prev_score = _SCORE_CACHE.get(_sym_key)
    score_delta: int | None = (eisax_score - _prev_score) if _prev_score is not None else None
    _SCORE_CACHE[_sym_key] = eisax_score
    _rsi_val = _safe_float(summary.get("rsi")) or 50.0
    _adx_val = _safe_float(summary.get("adx")) or 20.0
    fundamental_conviction = derive_fundamental_conviction(eisax_score)
    timing_confidence = derive_timing_confidence(_rsi_val, _adx_val)
    if low_data_mode:
        fundamental_conviction = "LOW"
        timing_confidence = "LOW"
    if language.lower().startswith("ar"):
        _LEVEL_AR = {"HIGH": "مرتفعة", "MEDIUM": "متوسطة", "LOW": "منخفضة"}
        fundamental_conviction = _LEVEL_AR.get(fundamental_conviction, fundamental_conviction)
        timing_confidence = _LEVEL_AR.get(timing_confidence, timing_confidence)
    conviction_score = _clamp_int(
        0.55 * eisax_score
        + 0.35 * ((_safe_float(summary.get("adx")) or 0) * 2.5)
        + 0.10 * (100 - min(abs((_safe_float(fundamentals.get("beta")) or 1.0) - 1.0) * 20, 40)),
        0,
        100,
    )
    if low_data_mode:
        conviction_score = min(conviction_score, 39)
    parsed_conviction_score = _parse_percent_after_label(report_text, "Conviction")
    if parsed_conviction_score is not None and not low_data_mode:
        conviction_score = parsed_conviction_score
    decision_type = _map_decision_type(
        recommendation=recommendation,
        adx=_safe_float(summary.get("adx")),
        next_earnings_text=_clean_text(fundamentals.get("last_earnings_date")),
    )
    status_summary = _build_status_summary(report_text, recommendation, language=language)
    why_this_decision = _build_why_this_decision(recommendation, summary, fundamentals, language=language)
    if low_data_mode:
        status_summary = (
            "تغطية البيانات الأساسية محدودة؛ يعتمد التحليل أساسًا على سلوك السعر."
            if language.lower().startswith("ar") else
            "Fundamental data coverage is limited; analysis relies primarily on price behavior."
        )
        limited_reason = (
            "الرؤية الأساسية محدودة، لذلك يتطلب القرار تأكيدًا إضافيًا من السعر."
            if language.lower().startswith("ar") else
            "Fundamental visibility is limited, so price confirmation is required."
        )
        why_this_decision = [limited_reason] + [
            item for item in why_this_decision if _clean_text(item) != _clean_text(limited_reason)
        ]
    risk_map = _parse_risk_map(report_text) or _build_fallback_risk_map(summary, fundamentals, language=language)
    report_meta = _build_report_meta(
        report_text=report_text,
        score_components=score_components,
        risk_map=risk_map,
        fundamentals=fundamentals,
        recommendation=recommendation,
        conviction_score=conviction_score,
        low_data_mode=low_data_mode,
    )

    report_json: dict[str, Any] = {
        "report_id": str(uuid4()),
        "generated_at": generated_at,
        "system": {
            "name": "EisaX",
            "environment": "pilot",
            "version": _clean_text(system_version) or "v1.0",
            "model_primary": _clean_text(model_primary) or "DeepSeek V3",
            "language": _clean_text(language) or "en",
            "determinism": {
                "scoring": "deterministic",
                "decision_layer": "llm_assisted",
            },
        },
        "data_context": {
            "data_as_of": data_as_of,
            "latency_seconds": _clamp_int(latency_seconds, 0, 3600),
        },
        "asset": {
            "symbol": _clean_text(symbol).upper(),
            "name": _clean_text(fundamentals.get("company_name")) or _clean_text(symbol).upper(),
            "asset_type": _map_asset_type(symbol, market, fundamentals),
            "market": _clean_text(market).upper(),
            "currency": _map_currency(symbol, market),
        },
        "headline_view": {
            "recommendation": recommendation,
            "decision_type": decision_type,
            "conviction_level": derive_conviction_level(conviction_score),
            "conviction_score": conviction_score,
            "eisax_score": eisax_score,
            "fundamental_conviction": fundamental_conviction,
            "timing_confidence": timing_confidence,
            "score_delta": score_delta,
            "status_summary": status_summary,
        },
        "report_meta": report_meta,
        "decision_framework": {
            "why_this_decision": why_this_decision,
            "no_action_case": (
                "حافظ على الموقف الحالي طالما السعر يتداول ضمن النطاق القائم ودون تغيير في المحفزات."
                if language.lower().startswith("ar") else
                "Maintain the current stance if price remains inside the existing range and no trigger condition is met."
            ),
        },
        "triggers": _build_triggers(recommendation, summary, fundamentals, language=language),
        "risk_map": risk_map,
        "what_would_make_me_wrong": _build_what_would_make_me_wrong(recommendation, summary, fundamentals, language=language),
        "monitoring": {
            "tracking_status": "active",
            "next_review_cycle": "weekly",
            "alert_eligible": trust_layer.get("classification") != "FLAGGED",
        },
        "compliance": {
            "disclaimer": (
                "هذا التقرير للأغراض المعلوماتية فقط ولا يُعدّ توصية استثمارية."
                if language.lower().startswith("ar") else
                "This report is for informational purposes only and does not constitute investment advice."
            ),
            "pilot_status": "live_pilot",
            "simulated": False,
        },
    }

    market_snapshot = _build_market_snapshot(summary, fundamentals, data_as_of)
    if market_snapshot:
        report_json["market_snapshot"] = market_snapshot

    scenarios = [] if low_data_mode else (_parse_scenarios(report_text) or _build_fallback_scenarios(summary, fundamentals))
    if scenarios:
        scenario_analysis, validation = normalize_scenarios(scenarios)
        report_json["scenario_analysis"] = scenario_analysis
        report_json["validation"] = validation
        expected_return = 0.0
        has_expected_value = False
        for scenario, probability in zip(scenario_analysis, validation["normalized_probabilities"]):
            return_pct = _safe_float(scenario.get("return_pct"))
            if return_pct is None:
                continue
            has_expected_value = True
            expected_return += return_pct * probability
        if has_expected_value:
            report_json["expected_value"] = {
                "expected_return_pct": round(expected_return, 1),
                "method": "probability_weighted_scenarios",
            }

    technical_view = _build_technical_view(summary, fundamentals)
    if technical_view:
        report_json["technical_view"] = technical_view

    fundamental_view = _build_fundamental_view(fundamentals)
    if fundamental_view:
        report_json["fundamental_view"] = fundamental_view

    macro_context = _build_macro_context(report_text)
    if macro_context:
        report_json["macro_context"] = macro_context

    return validate_report_json(report_json)
