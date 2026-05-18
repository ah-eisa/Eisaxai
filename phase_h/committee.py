"""
Phase H5 - Investment Committee Mode.

Builds a deterministic committee brief from existing result payloads only.
No Phase H engine is re-run here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, TypedDict

from .contracts import make_envelope, unwrap
from .registry import FeatureRegistry
from .report_helpers import L, fmt_num, fmt_pct, md_table
from .schemas import CommitteeBrief, CommitteeExhibit

ENGINE_VERSION = "0.2.0"


SUPPORTED_MODES = (
    "1pager",
    "cio_memo",
    "executive_memo",
    "defend",
    "bear",
    "stress",
    "hostile",
    "challenge_macro",
    "challenge_concentration",
    "challenge_liquidity",
    "challenge_valuation",
    "challenge_geopolitical",
)

CHALLENGE_CATEGORIES = ("macro", "concentration", "liquidity", "valuation", "geopolitical", "factor")
_CHALLENGE_MODE_TO_CATEGORY = {
    "challenge_macro": "macro",
    "challenge_concentration": "concentration",
    "challenge_liquidity": "liquidity",
    "challenge_valuation": "valuation",
    "challenge_geopolitical": "geopolitical",
}


class Objection(TypedDict, total=False):
    category: str
    claim: str
    evidence_ref: str
    severity: str
    counter: Optional[str]


_LABELS = {
    "en": {
        "mode": "Mode",
        "distribution": "Distribution",
        "pagebreak": "Page break may be recommended below",
        "headline": "Headline",
        "field": "Field",
        "detail": "Detail",
        "key_decision": "Key Decision",
        "positioning": "Positioning",
        "implementation": "Implementation",
        "mandate": "Mandate",
        "fragility": "Thesis Fragility",
        "verdict": "CIO Defensibility Verdict",
        "key_risks": "Key Risks",
        "top_vulnerabilities": "Top Vulnerabilities",
        "challenge_scenarios": "Challenge Scenarios",
        "exhibits": "Exhibits",
        "objections": "Committee Objections",
        "category": "Category",
        "claim": "Claim",
        "evidence": "Evidence",
        "severity": "Severity",
        "counter": "Counter",
        "unaddressed": "Unaddressed Objections",
        "payload_ref": "payload reference",
        "investment_committee": "Investment Committee",
        "stance_defend": "Stance: Defend may apply.",
        "stance_bear": "Stance: Challenge may apply.",
        "stance_hostile": "Stance: Hostile committee simulation may apply.",
    },
    "ar": {
        "mode": "الوضع",
        "distribution": "التوزيع",
        "pagebreak": "قد يوصى بفاصل صفحة أدناه",
        "headline": "العنوان",
        "field": "البند",
        "detail": "التفصيل",
        "key_decision": "القرار الرئيسي",
        "positioning": "التموضع",
        "implementation": "التنفيذ",
        "mandate": "التفويض",
        "fragility": "هشاشة الفرضية",
        "verdict": "حكم قابلية الدفاع من مدير الاستثمار",
        "key_risks": "المخاطر الرئيسية",
        "top_vulnerabilities": "نقاط الضعف الرئيسية",
        "challenge_scenarios": "سيناريوهات التحدي",
        "exhibits": "الملاحق",
        "objections": "اعتراضات اللجنة",
        "category": "الفئة",
        "claim": "الادعاء",
        "evidence": "الدليل",
        "severity": "الشدة",
        "counter": "الرد",
        "unaddressed": "اعتراضات غير معالجة",
        "payload_ref": "مرجع الحمولة",
        "investment_committee": "لجنة الاستثمار",
        "stance_defend": "الموقف: قد ينطبق الدفاع.",
        "stance_bear": "الموقف: قد ينطبق التحدي.",
        "stance_hostile": "الموقف: قد تنطبق محاكاة لجنة معارضة.",
    },
}


def _ll(key: str, language: str = "en") -> str:
    return _LABELS.get(language, _LABELS["en"]).get(key, _LABELS["en"].get(key, key))


def _enabled() -> bool:
    try:
        return bool(FeatureRegistry.is_enabled("phase_h_committee"))
    except Exception:
        return True


def _payload(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    if value.get("version") and value.get("engine"):
        return unwrap(value)
    return dict(value)


def _mode_or_default(mode: Optional[str]) -> str:
    candidate = str(mode or "").strip()
    if not candidate:
        try:
            candidate = str(FeatureRegistry.get("committee_mode") or "").strip()
        except Exception:
            candidate = ""
    return candidate if candidate in SUPPORTED_MODES else "1pager"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_weights(weights: Any) -> Dict[str, float]:
    if not isinstance(weights, Mapping):
        return {}
    parsed: Dict[str, float] = {}
    for key, value in weights.items():
        weight = _as_float(value)
        if weight > 0:
            parsed[str(key)] = weight
    total = sum(parsed.values())
    if total <= 0:
        return {}
    if total > 1.5:
        parsed = {k: v / 100.0 for k, v in parsed.items()}
        total = sum(parsed.values())
    return {k: v / total for k, v in parsed.items()}


def _region_for_asset(asset: str, meta: Mapping[str, Any]) -> str:
    raw = str(asset or "")
    upper = raw.upper()
    details = meta.get(raw) or meta.get(upper) or {}
    if isinstance(details, Mapping):
        for key in ("region", "market", "country"):
            if details.get(key):
                return str(details[key])
    if any(token in upper for token in ("KSA", "SAUDI", "TASI", "UAE", "QATAR", "KUWAIT", "GCC", "EFID", ".CA")):
        return "GCC"
    if any(token in upper for token in ("TLT", "IEF", "SHY", "AGG", "BND", "BOND", "TREAS")):
        return "Bonds"
    if any(token in upper for token in ("BTC", "ETH", "CRYPTO")):
        return "Crypto"
    if upper in {"SPY", "QQQ", "VTI", "VOO", "IWM", "MDY", "VIG"} or "US " in upper:
        return "US"
    if upper in {"ACWI", "URTH", "VT", "VEA", "EFA", "EEM"}:
        return "Global"
    return "Other"


def _region_weights(weights: Mapping[str, float], asset_meta: Any) -> Dict[str, float]:
    meta = asset_meta if isinstance(asset_meta, Mapping) else {}
    out: Dict[str, float] = {}
    for asset, weight in weights.items():
        region = _region_for_asset(asset, meta)
        out[region] = out.get(region, 0.0) + float(weight)
    return out


def _tracking_class(benchmark: Mapping[str, Any]) -> str:
    te = _as_float(benchmark.get("tracking_error_pct"))
    active_share = _as_float(benchmark.get("active_share_pct"))
    if te <= 3.0 and active_share <= 35.0:
        return "close"
    if te <= 7.5:
        return "moderate"
    return "active"


def _severity_rank(level: str) -> int:
    return {"low": 0, "moderate": 1, "elevated": 2, "high": 3}.get(str(level).lower(), 0)


def _scenario_tail_value(scenario: Mapping[str, Any]) -> float:
    return _as_float(
        scenario.get("terminal_p10", scenario.get("terminal_p10_pct", scenario.get("terminal_p10_return_pct"))),
        0.0,
    )


def _scenario_dd_value(scenario: Mapping[str, Any]) -> float:
    return _as_float(scenario.get("max_dd_p50_pct", scenario.get("max_drawdown_p50", 0.0)), 0.0)


def _fmt_signed_pct(value: Any) -> str:
    val = _as_float(value)
    return f"{val:+.1f}%"


def _english_sentence(text: str, language: str) -> str:
    if language == "ar":
        return text
    return text


def _counter_or_none(payload: Mapping[str, Any], sentence: str) -> Optional[str]:
    return sentence if payload else None


def _concentration_objection(result: Mapping[str, Any], language: str) -> Objection:
    weights = _normalise_weights(result.get("weights"))
    benchmark = _payload(result.get("benchmark_relative"))
    if weights:
        top_asset, top_weight = max(weights.items(), key=lambda item: item[1])
    else:
        top_asset, top_weight = "largest position", 0.0
    severity = "elevated" if top_weight > 0.15 else "low"
    if language == "ar":
        claim = f"تركيز {top_asset} قد يمثل {top_weight * 100:.1f}% من المحفظة."
        counter = _counter_or_none(benchmark, f"خطأ التتبع قد يكون {fmt_pct(benchmark.get('tracking_error_pct'), 1)} ونسبة المعلومات قد تكون {fmt_num(benchmark.get('information_ratio'), 2)}.")
    else:
        claim = f"{top_asset} concentration may represent {top_weight * 100:.1f}% of portfolio risk budget."
        counter = _counter_or_none(benchmark, f"Benchmark diagnostics may show tracking error of {fmt_pct(benchmark.get('tracking_error_pct'), 1)} and IR {fmt_num(benchmark.get('information_ratio'), 2)}.")
    return Objection(category="concentration", claim=claim, evidence_ref="weights", severity=severity, counter=counter)


def _liquidity_objection(result: Mapping[str, Any], language: str) -> Objection:
    execution = _payload(result.get("execution_diag"))
    stress = str(execution.get("liquidity_stress", "unknown")).lower()
    severity = "elevated" if stress in {"elevated", "high"} else "moderate" if stress == "moderate" else "low"
    if language == "ar":
        claim = f"ضغط السيولة قد يظهر بتصنيف {stress or 'unknown'}."
        counter = _counter_or_none(execution, f"فجوة التنفيذ قد تكون {fmt_num(execution.get('implementation_shortfall_bp'), 1)} نقطة أساس مع دوران {fmt_pct(execution.get('turnover_pct'), 1)}.")
    else:
        claim = f"Liquidity stress may screen as {stress or 'unknown'} under the execution payload."
        counter = _counter_or_none(execution, f"Execution diagnostics may show shortfall of {fmt_num(execution.get('implementation_shortfall_bp'), 1)} bp with turnover of {fmt_pct(execution.get('turnover_pct'), 1)}.")
    return Objection(category="liquidity", claim=claim, evidence_ref="execution_diag", severity=severity, counter=counter)


def _macro_objection(result: Mapping[str, Any], language: str) -> Objection:
    forward = _payload(result.get("forward_scenario"))
    scenarios = forward.get("scenarios") if isinstance(forward.get("scenarios"), Mapping) else {}
    recession = scenarios.get("recession") if isinstance(scenarios.get("recession"), Mapping) else {}
    p10 = _scenario_tail_value(recession)
    severity = "elevated" if p10 < -20.0 else "moderate" if p10 < -10.0 else "low"
    if language == "ar":
        claim = f"سيناريو الركود قد يسجل ذيلا عند {p10:+.1f}% في P10."
        counter = _counter_or_none(forward, f"التوزيع المجمع قد يكون مدللا لاحتمال خسارة {fmt_pct((forward.get('aggregate') or {}).get('prob_loss'), 1)}.")
    else:
        claim = f"Recession scenario may model terminal P10 at {p10:+.1f}%."
        counter = _counter_or_none(forward, f"Aggregate scenario output may model loss probability at {fmt_pct((forward.get('aggregate') or {}).get('prob_loss'), 1)}.")
    return Objection(category="macro", claim=claim, evidence_ref="forward_scenario.scenarios.recession", severity=severity, counter=counter)


def _valuation_objection(result: Mapping[str, Any], language: str) -> Objection:
    factor = _payload(result.get("factor_decomp"))
    loadings = factor.get("loadings") if isinstance(factor.get("loadings"), Mapping) else {}
    hml = _as_float(loadings.get("HML", loadings.get("Value", 0.0)))
    severity = "elevated" if hml < -0.4 else "moderate" if hml < -0.2 else "low"
    if language == "ar":
        claim = f"تعرض القيمة HML قد يكون {hml:+.2f} وقد يشير إلى ميل نمو."
        counter = _counter_or_none(factor, f"نموذج العوامل قد يعرض R2 عند {fmt_num(factor.get('r_squared'), 2)} وثباتا عند {fmt_num(factor.get('rolling_stability'), 2)}.")
    else:
        claim = f"HML loading may be {hml:+.2f} and may indicate growth-tilted valuation sensitivity."
        counter = _counter_or_none(factor, f"Factor model may show R2 of {fmt_num(factor.get('r_squared'), 2)} and stability of {fmt_num(factor.get('rolling_stability'), 2)}.")
    return Objection(category="valuation", claim=claim, evidence_ref="factor_decomp.loadings.HML", severity=severity, counter=counter)


def _geopolitical_objection(result: Mapping[str, Any], language: str) -> Objection:
    weights = _normalise_weights(result.get("weights"))
    regions = _region_weights(weights, result.get("asset_meta"))
    gcc_weight = regions.get("GCC", 0.0)
    severity = "elevated" if gcc_weight > 0.25 else "low"
    benchmark = _payload(result.get("benchmark_relative"))
    if language == "ar":
        claim = f"وزن الخليج قد يبلغ {gcc_weight * 100:.1f}% وقد يضيف حساسية جيوسياسية."
        counter = _counter_or_none(benchmark, f"سلوك الأنظمة قد يوفر سياقا عبر {len((benchmark.get('regime_behavior') or {}).get('outperform_envs', []))} بيئات تفوق محتملة.")
    else:
        claim = f"GCC exposure may reach {gcc_weight * 100:.1f}% and may add geopolitical sensitivity."
        counter = _counter_or_none(benchmark, f"Regime behavior may provide context across {len((benchmark.get('regime_behavior') or {}).get('outperform_envs', []))} modelled outperform environments.")
    return Objection(category="geopolitical", claim=claim, evidence_ref="weights.region_map", severity=severity, counter=counter)


def _factor_objection(result: Mapping[str, Any], language: str) -> Objection:
    factor = _payload(result.get("factor_decomp"))
    warnings = [str(w) for w in factor.get("warnings", []) if w] if isinstance(factor.get("warnings"), list) else []
    severity = "elevated" if warnings else "low"
    warning = warnings[0] if warnings else "factor crowding screens may be limited by available history"
    if language == "ar":
        claim = f"تحذيرات العوامل قد تشير إلى {warning}."
        counter = _counter_or_none(factor, f"اعتمادية نموذج العوامل قد تكون {factor.get('reliability_tier', 'Indicative')}.")
    else:
        claim = f"Factor diagnostics may indicate {warning}."
        counter = _counter_or_none(factor, f"Factor reliability may be {factor.get('reliability_tier', 'Indicative')}.")
    return Objection(category="factor", claim=claim, evidence_ref="factor_decomp.warnings", severity=severity, counter=counter)


def build_objections(
    result: Dict[str, Any],
    category: Optional[str] = None,
    language: str = "en",
) -> List[Objection]:
    """Build deterministic committee objections from existing result payloads."""
    if not _enabled() or not isinstance(result, Mapping):
        return []
    builders = {
        "macro": _macro_objection,
        "concentration": _concentration_objection,
        "liquidity": _liquidity_objection,
        "valuation": _valuation_objection,
        "geopolitical": _geopolitical_objection,
        "factor": _factor_objection,
    }
    cats: Sequence[str]
    if category and category in builders:
        cats = (category,)
    else:
        cats = CHALLENGE_CATEGORIES
    objections = [builders[cat](result, language) for cat in cats]
    objections.sort(key=lambda item: (-_severity_rank(item.get("severity", "low")), item.get("category", "")))
    return objections[:8]


def _top_vulnerabilities(factor: Mapping[str, Any], limit: int = 3, language: str = "en") -> List[str]:
    loadings = factor.get("loadings") if isinstance(factor.get("loadings"), Mapping) else {}
    warnings = [str(w) for w in factor.get("warnings", []) if w] if isinstance(factor.get("warnings"), list) else []
    top = sorted(loadings.items(), key=lambda item: abs(_as_float(item[1])), reverse=True)[:limit]
    out: List[str] = []
    for name, beta in top:
        related = next((w for w in warnings if str(name).lower() in w.lower()), None)
        if language == "ar":
            sentence = f"{name} قد يحمل بيتا {float(beta):+.2f}"
            if related:
                sentence += f" وقد يقترن بتحذير: {related}"
            out.append(sentence + ".")
        else:
            sentence = f"{name} may carry beta {float(beta):+.2f}"
            if related:
                sentence += f" and may pair with warning: {related}"
            out.append(sentence + ".")
    if not out:
        if language == "ar":
            out.append("تعرضات العوامل قد تكون غير كافية لتحديد نقاط ضعف مستقلة.")
        else:
            out.append("Factor exposures may be insufficient to identify standalone vulnerabilities.")
    return out


def _challenge_scenarios(forward: Mapping[str, Any], limit: int = 3, language: str = "en") -> List[str]:
    scenarios = forward.get("scenarios") if isinstance(forward.get("scenarios"), Mapping) else {}
    rows = []
    for name, scenario in scenarios.items():
        if isinstance(scenario, Mapping):
            rows.append((str(name), scenario, _scenario_tail_value(scenario)))
    rows.sort(key=lambda item: item[2])
    out: List[str] = []
    for name, scenario, p10 in rows[:limit]:
        dd = _scenario_dd_value(scenario)
        label = name.replace("_", " ")
        if language == "ar":
            out.append(f"{label} قد يسجل P10 عند {p10:+.1f}% وانخفاضا وسيطا عند {dd:+.1f}%.")
        else:
            out.append(f"{label} may model P10 at {p10:+.1f}% and median drawdown at {dd:+.1f}%.")
    if not out:
        if language == "ar":
            out.append("مخرجات السيناريو قد تكون غير كافية لترتيب ضغوط الذيل.")
        else:
            out.append("Scenario output may be insufficient to rank tail stress.")
    return out


def _key_risks(
    benchmark: Mapping[str, Any],
    factor: Mapping[str, Any],
    forward: Mapping[str, Any],
    *,
    limit: int,
    language: str,
) -> List[str]:
    risks: List[str] = []
    warnings = [str(w) for w in factor.get("warnings", []) if w] if isinstance(factor.get("warnings"), list) else []
    notes = [str(n) for n in benchmark.get("notes", []) if n] if isinstance(benchmark.get("notes"), list) else []
    for warning in warnings[:2]:
        risks.append(f"Factor warning may require committee review: {warning}." if language != "ar" else f"تحذير العوامل قد يتطلب مراجعة اللجنة: {warning}.")
    for note in notes[:2]:
        risks.append(f"Benchmark note may require attribution context: {note}." if language != "ar" else f"ملاحظة المؤشر قد تتطلب سياق إسناد: {note}.")
    risks.extend(_challenge_scenarios(forward, limit=2, language=language))
    if not risks:
        risks.append("Primary risk evidence may be limited by available Phase H payloads." if language != "ar" else "أدلة المخاطر الرئيسية قد تكون محدودة بحمولات المرحلة H المتاحة.")
    return risks[:limit]


def _mandate_summary(feasibility: Any, language: str) -> str:
    if isinstance(feasibility, Mapping):
        status = str(feasibility.get("status", feasibility.get("feasible", "unknown")))
        constraints = feasibility.get("binding_constraints") or feasibility.get("constraints") or []
    else:
        status = str(feasibility or "unknown")
        constraints = []
    if isinstance(constraints, str):
        constraints_text = constraints
    elif isinstance(constraints, Sequence):
        constraints_text = ", ".join(str(c) for c in constraints[:3]) if constraints else "none flagged"
    else:
        constraints_text = "none flagged"
    if language == "ar":
        return f"حالة التفويض قد تكون {status} مع قيود ملزمة قد تشمل {constraints_text}."
    return f"Mandate status may be {status} with binding constraints likely to include {constraints_text}."


def _positioning_sentence(benchmark: Mapping[str, Any], language: str) -> str:
    tracking = _tracking_class(benchmark)
    ir = fmt_num(benchmark.get("information_ratio"), 2)
    active = fmt_pct(benchmark.get("active_return_pct"), 1)
    downside = fmt_num(benchmark.get("downside_capture"), 2)
    if language == "ar":
        return f"التموضع النسبي قد يكون {tracking} مع عائد نشط {active} ونسبة معلومات {ir} واحتواء هبوط {downside}."
    return f"Benchmark-relative positioning may be {tracking} with active return {active}, IR {ir}, and downside capture {downside}."


def _implementation_sentence(execution: Mapping[str, Any], language: str) -> str:
    turnover = fmt_pct(execution.get("turnover_pct"), 1)
    complexity = execution.get("complexity_tier", "unknown")
    liquidity = execution.get("liquidity_stress", "unknown")
    shortfall = fmt_num(execution.get("implementation_shortfall_bp"), 1)
    if language == "ar":
        return f"التنفيذ قد يتضمن دوران {turnover} وتعقيد {complexity} وضغط سيولة {liquidity} وفجوة {shortfall} نقطة أساس."
    return f"Implementation may involve {turnover} turnover, {complexity} complexity, {liquidity} liquidity stress, and {shortfall} bp shortfall."


def _headline(
    result: Mapping[str, Any],
    weights: Mapping[str, float],
    benchmark: Mapping[str, Any],
    regions: Mapping[str, float],
    language: str,
) -> str:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
    profile = metrics.get("profile") or result.get("profile") or "balanced"
    confidence = result.get("confidence") if isinstance(result.get("confidence"), Mapping) else {}
    tier = confidence.get("reliability_tier") or benchmark.get("reliability_tier") or "Indicative"
    tracking = _tracking_class(benchmark)
    ir = fmt_num(benchmark.get("information_ratio"), 2)
    if language == "ar":
        return f"ملف {profile} قد يضم {len(weights)} مراكز عبر {len(regions) or 1} مناطق؛ الاعتمادية قد تكون {tier}؛ التتبع النسبي قد يكون {tracking} مع IR {ir}."
    return f"Profile {profile} may include {len(weights)} positions across {len(regions) or 1} regions; reliability may be {tier}; benchmark-relative tracking may be {tracking}, IR {ir}."


def _key_decision(mode: str, benchmark: Mapping[str, Any], forward: Mapping[str, Any], language: str) -> str:
    downside = _as_float(benchmark.get("downside_capture"), 1.0)
    agg = forward.get("aggregate") if isinstance(forward.get("aggregate"), Mapping) else {}
    prob_loss = _as_float(agg.get("prob_loss"), 0.0)
    if mode in {"bear", "stress", "hostile"} or mode.startswith("challenge_"):
        if language == "ar":
            return f"اللجنة قد تطلب تبريرا إضافيا لأن احتواء الهبوط قد يكون {downside:.2f} واحتمال الخسارة قد يكون {prob_loss:.1f}%."
        return f"Committee review may require added justification because downside capture may be {downside:.2f} and loss probability may be {prob_loss:.1f}%."
    if language == "ar":
        return "التخصيص قد يبقى ضمن التفويض إذا ظلت أخطاء التتبع وتعريضات العوامل ضمن الحدود المعتمدة."
    return "Current allocation may remain within mandate if tracking error and factor exposures remain within approved thresholds."


def _exhibits(mode: str) -> List[CommitteeExhibit]:
    base = [
        CommitteeExhibit(number=1, title="Benchmark-relative diagnostics", payload_ref="benchmark_relative"),
        CommitteeExhibit(number=2, title="Forward scenario distribution", payload_ref="forward_scenario"),
        CommitteeExhibit(number=3, title="Factor risk decomposition", payload_ref="factor_decomp"),
        CommitteeExhibit(number=4, title="Execution diagnostics", payload_ref="execution_diag"),
    ]
    limit = 2 if mode == "1pager" else 3 if mode == "stress" else 4
    if mode in {"hostile"} or mode.startswith("challenge_"):
        base.insert(0, CommitteeExhibit(number=1, title="Committee objections", payload_ref="committee_brief.objections"))
        for idx, exhibit in enumerate(base, start=1):
            exhibit["number"] = idx
    return base[:limit]


def _fragility(objections: Sequence[Mapping[str, Any]]) -> tuple[float, str, List[Mapping[str, Any]]]:
    if not objections:
        return 0.0, "defensible", []
    unaddressed = [obj for obj in objections if not obj.get("counter")]
    score = round(100.0 * len(unaddressed) / max(len(objections), 1), 1)
    if score <= 25.0:
        verdict = "defensible"
    elif score <= 60.0:
        verdict = "requires-justification"
    else:
        verdict = "weak-thesis"
    return score, verdict, unaddressed


def _attach_envelope(payload: CommitteeBrief, notes: Sequence[str]) -> None:
    envelope_payload = {k: v for k, v in payload.items() if k != "_envelope"}
    envelope = make_envelope("committee_brief", envelope_payload, notes=list(notes))
    payload["_envelope"] = envelope
    payload.update(envelope)


def build_committee_brief(
    result: Dict[str, Any],
    mode: Optional[str] = None,
    language: str = "en",
) -> CommitteeBrief:
    """Assemble the committee brief from already-computed result payloads."""
    if not _enabled() or not isinstance(result, Mapping):
        return CommitteeBrief()

    selected_mode = _mode_or_default(mode)
    weights = _normalise_weights(result.get("weights"))
    benchmark = _payload(result.get("benchmark_relative"))
    execution = _payload(result.get("execution_diag"))
    forward = _payload(result.get("forward_scenario"))
    factor = _payload(result.get("factor_decomp"))
    regions = _region_weights(weights, result.get("asset_meta"))

    objection_category = _CHALLENGE_MODE_TO_CATEGORY.get(selected_mode)
    challenge_objections = build_objections(dict(result), category=objection_category, language=language) if selected_mode == "hostile" or objection_category else []
    risk_objections = (
        challenge_objections
        if challenge_objections
        else build_objections(dict(result), language=language)
        if selected_mode in {"bear", "stress"}
        else []
    )
    objections = challenge_objections
    fragility, verdict, unaddressed = _fragility(objections)

    risk_limit = 3 if selected_mode == "1pager" else 5
    if selected_mode in {"bear", "stress", "hostile"} or selected_mode.startswith("challenge_"):
        risks = [obj["claim"] for obj in risk_objections[:risk_limit]] if risk_objections else _key_risks(benchmark, factor, forward, limit=risk_limit, language=language)
    else:
        risks = _key_risks(benchmark, factor, forward, limit=risk_limit, language=language)

    if selected_mode == "stress":
        scenarios = [s for s in _challenge_scenarios(forward, limit=5, language=language) if "recession" in s.lower() or "liquidity" in s.lower()]
        if len(scenarios) < 3:
            scenarios = _challenge_scenarios(forward, limit=3, language=language)
    else:
        scenarios = _challenge_scenarios(forward, limit=3, language=language)

    notes = ["committee brief built from existing result payloads only"]
    payload = CommitteeBrief(
        mode=selected_mode,
        headline=_headline(result, weights, benchmark, regions, language),
        key_decision=_key_decision(selected_mode, benchmark, forward, language),
        key_risks=risks,
        positioning=_positioning_sentence(benchmark, language),
        implementation_notes=_implementation_sentence(execution, language),
        mandate_summary=_mandate_summary(result.get("feasibility"), language),
        top_vulnerabilities=_top_vulnerabilities(factor, limit=3, language=language),
        challenge_scenarios=scenarios,
        exhibits=_exhibits(selected_mode),
        notes=notes,
        objections=objections,
        unaddressed_objections=[dict(obj) for obj in unaddressed],
        thesis_fragility_score=fragility,
        cio_defensibility_verdict=verdict,
    )
    _attach_envelope(payload, notes)
    return payload


def _stance_line(mode: str, language: str) -> str:
    if mode == "defend":
        if language == "ar":
            return f"**{_ll('stance_defend', language)}** التخصيص قد يبقى متوافقا مع التفويض وعتبات المؤشر."
        return f"**{_ll('stance_defend', language)}** The allocation may remain compliant with mandate and benchmark-relative thresholds."
    if mode == "bear":
        if language == "ar":
            return f"**{_ll('stance_bear', language)}** نقاط الضعف المادية قد تظهر تحت أنظمة الضغط."
        return f"**{_ll('stance_bear', language)}** Material vulnerabilities may surface under stress regimes."
    if mode == "hostile" or mode.startswith("challenge_"):
        if language == "ar":
            return f"**{_ll('stance_hostile', language)}** الاعتراضات قد تختبر قابلية الدفاع عن الفرضية."
        return f"**{_ll('stance_hostile', language)}** Objections may test thesis defensibility."
    return ""


def _list_block(title: str, items: Sequence[str], ordered: bool = False) -> str:
    if not items:
        return ""
    lines = [f"**{title}**"]
    for idx, item in enumerate(items, start=1):
        prefix = f"{idx}." if ordered else "-"
        lines.append(f"{prefix} {item}")
    return "\n".join(lines)


def _render_exhibits(exhibits: Sequence[Mapping[str, Any]], language: str) -> str:
    if not exhibits:
        return ""
    lines = [f"**{_ll('exhibits', language)}**"]
    for exhibit in exhibits:
        number = exhibit.get("number", "")
        title = exhibit.get("title", "")
        ref = exhibit.get("payload_ref", "")
        lines.append(f"- Exhibit {number}: {title} - {_ll('payload_ref', language)}: {ref}")
    return "\n".join(lines)


def _render_objections(payload: Mapping[str, Any], language: str) -> str:
    objections = payload.get("objections") if isinstance(payload.get("objections"), list) else []
    if not objections:
        return ""
    rows: List[List[str]] = []
    for obj in objections:
        if not isinstance(obj, Mapping):
            continue
        rows.append([
            str(obj.get("category", "")),
            str(obj.get("claim", "")),
            str(obj.get("evidence_ref", "")),
            str(obj.get("severity", "")),
            str(obj.get("counter") or ("Unaddressed may require additional evidence." if language != "ar" else "قد يتطلب دليلا إضافيا.")),
        ])
    table = md_table(
        [_ll("category", language), _ll("claim", language), _ll("evidence", language), _ll("severity", language), _ll("counter", language)],
        rows,
    )
    unaddressed = payload.get("unaddressed_objections") if isinstance(payload.get("unaddressed_objections"), list) else []
    lines = [f"**{_ll('objections', language)}**", table]
    if unaddressed:
        claims = [str(obj.get("claim", "")) for obj in unaddressed if isinstance(obj, Mapping)]
        lines.extend(["", _list_block(_ll("unaddressed", language), claims)])
    return "\n".join(line for line in lines if line)


def render_committee_brief_md(payload: CommitteeBrief, language: str = "en") -> str:
    if not _enabled() or not payload:
        return ""
    data = _payload(payload)
    if not data:
        data = dict(payload)

    mode = str(data.get("mode", "1pager"))
    heading = f"## I. {L('committee_brief', language)}"
    meta = f"*{_ll('mode', language)}: {mode} · {_ll('distribution', language)}: {_ll('investment_committee', language)} · {_ll('pagebreak', language)}.*"
    stance = _stance_line(mode, language)
    fragility = data.get("thesis_fragility_score", 0.0)
    verdict = data.get("cio_defensibility_verdict", "defensible")

    blocks: List[str] = [heading, meta]
    if stance:
        blocks.extend(["---", stance])

    blocks.extend(["---", f"**{_ll('headline', language)}** {data.get('headline', '')}"])

    if mode == "executive_memo":
        paragraphs = [
            f"**{_ll('positioning', language)}** {data.get('positioning', '')}",
            f"**{_ll('implementation', language)}** {data.get('implementation_notes', '')} {data.get('key_decision', '')}",
            f"**{_ll('key_risks', language)}** {' '.join(str(r) for r in data.get('key_risks', [])[:3])}",
        ]
        blocks.extend(["---", "\n\n".join(paragraphs)])
    else:
        rows: List[List[str]] = [
            [_ll("key_decision", language), str(data.get("key_decision", ""))],
            [_ll("positioning", language), str(data.get("positioning", ""))],
            [_ll("implementation", language), str(data.get("implementation_notes", ""))],
            [_ll("mandate", language), str(data.get("mandate_summary", ""))],
            [_ll("fragility", language), f"{_as_float(fragility):.1f}%"],
            [_ll("verdict", language), str(verdict)],
        ]
        blocks.extend(["---", md_table([_ll("field", language), _ll("detail", language)], rows)])

    blocks.extend([
        "---",
        _list_block(_ll("key_risks", language), [str(r) for r in data.get("key_risks", [])], ordered=True),
        "---",
        _list_block(_ll("top_vulnerabilities", language), [str(v) for v in data.get("top_vulnerabilities", [])]),
        "---",
        _list_block(_ll("challenge_scenarios", language), [str(c) for c in data.get("challenge_scenarios", [])]),
    ])

    objections = _render_objections(data, language)
    if objections:
        blocks.extend(["---", objections])

    exhibits = _render_exhibits(data.get("exhibits", []), language)
    if exhibits:
        blocks.extend(["---", exhibits])

    blocks.append("<!-- pagebreak -->")
    return "\n\n".join(block for block in blocks if block).rstrip() + "\n"


__all__ = [
    "ENGINE_VERSION",
    "SUPPORTED_MODES",
    "Objection",
    "build_objections",
    "build_committee_brief",
    "render_committee_brief_md",
]
