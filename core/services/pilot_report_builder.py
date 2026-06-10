"""pilot_report_builder.py -- validate_report_json + build_pilot_report_json."""
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
from core.services.pilot_report_parsers import (
    ENUMS, _SCORE_CACHE, _safe_float, _safe_int, _clamp_int, _iso_now, _ensure_tz_iso,
    _clean_text, derive_conviction_level, derive_fundamental_conviction,
    derive_timing_confidence, _normalize_recommendation, _map_asset_type,
    _map_currency, _map_decision_type, _parse_recommendation_from_report,
    _parse_score_from_report, _parse_score_components, _parse_entry_timing,
    _parse_percent_after_label, _parse_level_label, _report_label_from_score,
    _risk_label_from_level, _market_beta_risk, _commodity_cycle_risk,
    _overall_risk_level, _build_report_meta, _parse_market_regime,
    _parse_first_sentence_of_section, _parse_risk_map, _parse_scenarios,
    _scenario_summary_from_name,
)
from core.services.pilot_report_builders import (
    normalize_scenarios, _build_fallback_scenarios,
    _build_why_this_decision, _build_fallback_risk_map, _build_triggers,
    _build_what_would_make_me_wrong, _build_status_summary, _map_technical_trend,
    _map_macd_signal, _build_market_snapshot, _build_technical_view,
    _build_fundamental_view, _build_macro_context, _strip_nulls,
)

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
    model_primary: str = "EisaX Agent",
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
            "model_primary": _clean_text(model_primary) or "EisaX Agent",
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
            "pilot_status": "institutional_pipeline",
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
