"""pilot_report_builders.py -- section builders for pilot report JSON."""
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
    ENUMS, _safe_float, _safe_int, _clamp_int, _iso_now, _ensure_tz_iso,
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
        # Prefer the TV snapshot timestamp for GCC tickers so downstream
        # sections can verify they share one point in time.
        snapshot["snapshot_time"] = (
            summary.get("snapshot_ts")
            or fundamentals.get("snapshot_ts")
            or timestamp_iso
        )
    if market_cap is not None:
        snapshot["market_cap"] = round(market_cap)
    if volume is not None:
        snapshot["volume"] = round(volume)
    _src = summary.get("data_source") or fundamentals.get("data_source")
    if _src:
        snapshot["data_source"] = _src
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


