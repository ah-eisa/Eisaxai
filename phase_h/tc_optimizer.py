"""
Phase H2 — Transaction-Cost-Aware Optimizer.

Public surface:
- build_turnover_terms(cp, w, w_prev, ...) -> (linear, quadratic, persistence)
- estimate_execution(weights, w_prev, asset_meta, ...) -> versioned execution diagnostics
- render_execution_md(payload, language) -> markdown subsection
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .contracts import make_envelope, unwrap
from .registry import FeatureRegistry
from .report_helpers import L, fmt_bp, fmt_pct, md_table, section_heading, severity_tag
from .schemas import ExecutionDiagnostics

ENGINE_VERSION = "0.2.0"

_VALID_REBALANCE = {"monthly", "quarterly", "semiannual", "annual"}
_TRADING_DAYS = 252.0
_TAX_NOTE = {
    "en": "Tax-aware execution requires account-level lot data; placeholder pending integration with broker tax-lot feed.",
    "ar": "التنفيذ الواعي بالضرائب يتطلب بيانات الحصص على مستوى الحساب؛ مكان مخصص بانتظار التكامل مع تغذية الحصص الضريبية من الوسيط.",
}
_LOCAL_LABELS = {
    "turnover_penalty": {
        "en": "Turnover penalty",
        "ar": "عقوبة معدل الدوران",
    },
    "applied": {"en": "applied", "ar": "مطبقة"},
    "not_applied": {"en": "not applied", "ar": "غير مطبقة"},
    "gcc_calendar_note": {
        "en": "GCC liquidity note",
        "ar": "ملاحظة سيولة أسواق الخليج",
    },
    "notes": {"en": "Notes", "ar": "ملاحظات"},
}


def _enabled() -> bool:
    return FeatureRegistry.is_enabled("phase_h_tc_optimizer")


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _ll(key: str, language: str = "en") -> str:
    entry = _LOCAL_LABELS.get(key)
    if not entry:
        return key
    return entry.get(language, entry.get("en", key))


def build_turnover_terms(
    cp: Any,
    w: Any,
    w_prev: Optional[Any],
    *,
    linear_lambda: float = 0.0010,
    quadratic_lambda: float = 0.0005,
    persistence_lambda: float = 0.0002,
) -> Tuple[Any, Any, Any]:
    """
    Build CVXPY turnover penalty expressions.

    The returned expressions are positive penalty terms. Callers using a
    maximization objective should subtract them from the base utility.
    """
    if not _enabled() or cp is None or w_prev is None:
        zero = 0 if cp is None else cp.Constant(0.0)
        return zero, zero, zero
    linear_lambda = _env_float("EISAX_TC_LINEAR_LAMBDA", str(linear_lambda))
    quadratic_lambda = _env_float("EISAX_TC_QUADRATIC_LAMBDA", str(quadratic_lambda))
    persistence_lambda = _env_float("EISAX_TC_PERSISTENCE_LAMBDA", str(persistence_lambda))
    delta = w - w_prev
    linear = float(linear_lambda) * cp.norm1(delta)
    quadratic = float(quadratic_lambda) * cp.sum_squares(delta)
    persistence = float(persistence_lambda) * cp.norm1(delta)
    return linear, quadratic, persistence


def _normalise_weights(weights: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    if not weights:
        return parsed
    for key, val in weights.items():
        try:
            w = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(w) and w > 0:
            parsed[str(key)] = w
    total = sum(parsed.values())
    if total <= 0:
        return {}
    if total > 1.5:
        parsed = {k: v / 100.0 for k, v in parsed.items()}
        total = sum(parsed.values())
    return {k: v / total for k, v in parsed.items()} if total > 0 else {}


def _meta_for(asset: str, asset_meta: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not asset_meta:
        return {}
    raw = asset_meta.get(asset) if isinstance(asset_meta, Mapping) else None
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(asset_meta, Mapping) and any(k in asset_meta for k in ("vol", "volatility", "spread_bp", "kind", "region")):
        return dict(asset_meta)
    return {}


def _align_previous_keys(
    prev: Dict[str, float],
    weights: Mapping[str, float],
    asset_meta: Optional[Mapping[str, Any]],
) -> Dict[str, float]:
    if not prev or not asset_meta:
        return prev
    aliases: Dict[str, str] = {str(k).upper(): str(k) for k in weights}
    for asset, raw_meta in asset_meta.items():
        if not isinstance(raw_meta, Mapping):
            continue
        target = str(asset)
        aliases[target.upper()] = target
        for key in ("proxy", "ticker", "symbol"):
            val = raw_meta.get(key)
            if val:
                aliases[str(val).upper()] = target
    out: Dict[str, float] = {}
    for key, val in prev.items():
        target = aliases.get(str(key).upper(), str(key))
        out[target] = out.get(target, 0.0) + float(val)
    return out


def _daily_vol_pct(asset: str, meta: Mapping[str, Any]) -> float:
    for key in ("daily_vol_pct", "vol_daily_pct", "sigma_daily_pct"):
        if key in meta:
            try:
                val = float(meta[key])
                return max(0.05, val)
            except (TypeError, ValueError):
                pass
    for key in ("vol", "volatility", "vol_annual", "annual_vol"):
        if key in meta:
            try:
                val = float(meta[key])
                if val > 1.0:
                    return max(0.05, val / math.sqrt(_TRADING_DAYS))
                return max(0.05, (val / math.sqrt(_TRADING_DAYS)) * 100.0)
            except (TypeError, ValueError):
                pass
    return _default_annual_vol(asset) / math.sqrt(_TRADING_DAYS) * 100.0


def _tokens(asset: str, meta: Mapping[str, Any]) -> str:
    parts = [asset, str(meta.get("ticker", "")), str(meta.get("proxy", "")), str(meta.get("region", "")), str(meta.get("kind", ""))]
    return " ".join(parts).upper()


def _default_annual_vol(asset: str) -> float:
    t = asset.upper()
    if "BTC" in t:
        return 0.75
    if "ETH" in t or "CRYPTO" in t:
        return 0.90
    if any(x in t for x in ("EGYPT", "EFID")):
        return 0.28
    if any(x in t for x in ("USO", "CRUDE", "OIL")):
        return 0.33
    if any(x in t for x in ("SLV", "SILVER", "CPER", "COPPER")):
        return 0.28
    if any(x in t for x in ("GCC", "SAUDI", "KSA", "UAE", "EMAAR", "QETF")):
        return 0.19
    if any(x in t for x in ("TLT", "EMB", "AGG")):
        return 0.14
    if any(x in t for x in ("SHY", "BIL", "CASH")):
        return 0.03
    if any(x in t for x in ("GLD", "GOLD")):
        return 0.145
    return 0.16


def _asset_bucket(asset: str, meta: Mapping[str, Any]) -> str:
    t = _tokens(asset, meta)
    if any(x in t for x in ("BTC", "ETH", "CRYPTO")):
        return "crypto"
    if any(x in t for x in ("EFID", "EGYPT", ".CA")):
        return "egypt_equity"
    if any(x in t for x in ("KSA", "SAUDI", "UAE", "GCC", "EMAAR", "QETF", ".DU", ".QA")):
        return "gcc_equity"
    if "EMB" in t or "EM BOND" in t:
        return "em_bonds"
    if any(x in t for x in ("TLT", "SHY", "BIL", "AGG", "TREASUR", "T-BILL", "CASH")):
        return "us_bonds"
    if any(x in t for x in ("GLD", "GOLD", "SLV", "SILVER", "USO", "CPER", "OIL", "COPPER", "COMMOD")):
        return "commodity"
    if any(x in t for x in ("SPY", "QQQ", "MDY", "VIG", "XLV", "XLU", "S&P", "NASDAQ", "MID-CAP", "DIVIDEND", "HEALTHCARE", "UTILITIES")):
        return "us_equity"
    return "large_cap_equity"


def _spread_bp(bucket: str, meta: Mapping[str, Any]) -> float:
    try:
        return float(meta["spread_bp"])
    except (KeyError, TypeError, ValueError):
        pass
    return {
        "us_equity": 5.0,
        "large_cap_equity": 5.0,
        "small_cap_equity": 25.0,
        "em_equity": 30.0,
        "gcc_equity": 30.0,
        "egypt_equity": 30.0,
        "us_bonds": 8.0,
        "em_bonds": 8.0,
        "commodity": 6.0,
        "crypto": 15.0,
    }.get(bucket, 12.0)


def _liquidity_multiplier(bucket: str) -> float:
    return {
        "us_equity": 1.0,
        "large_cap_equity": 1.0,
        "us_bonds": 1.0,
        "commodity": 1.2,
        "gcc_equity": 1.8,
        "egypt_equity": 2.4,
        "em_bonds": 1.4,
        "crypto": 1.0,
    }.get(bucket, 1.0)


def _crypto_extra_bp(notional: float) -> Tuple[float, bool]:
    if notional <= 250_000:
        return 0.0, False
    if notional <= 2_000_000:
        return 20.0, True
    if notional <= 10_000_000:
        return 60.0, True
    return 150.0, True


def _stress_tier(participation: float) -> str:
    if participation < 0.02:
        return "low"
    if participation <= 0.10:
        return "moderate"
    if participation <= 0.25:
        return "elevated"
    return "high"


def _worst_tier(tiers: List[str]) -> str:
    order = {"low": 0, "moderate": 1, "elevated": 2, "high": 3}
    return max(tiers or ["low"], key=lambda t: order.get(t, 0))


def _turnover_tag(turnover_pct: float) -> str:
    if turnover_pct < 10:
        return "low"
    if turnover_pct <= 25:
        return "moderate"
    if turnover_pct <= 50:
        return "elevated"
    return "high"


def _shortfall_tag(bp: float) -> str:
    if bp < 15:
        return "low"
    if bp <= 40:
        return "moderate"
    if bp <= 100:
        return "elevated"
    return "high"


def _impact_tag(bp: float) -> str:
    if bp < 5:
        return "low"
    if bp <= 25:
        return "moderate"
    return "elevated"


def _slippage_tag(bp: float) -> str:
    if bp < 10:
        return "low"
    if bp <= 30:
        return "moderate"
    return "elevated"


def _make_execution_envelope(payload: ExecutionDiagnostics, notes: Optional[List[str]] = None) -> Dict[str, Any]:
    envelope = make_envelope("execution_diagnostics", payload, notes=notes or payload.get("notes", []))
    # Compatibility with H1 skeleton consumers that still read result["execution_diag"]["turnover_pct"].
    envelope.update(payload)
    return envelope


def estimate_execution(
    weights: Dict[str, float],
    w_prev: Optional[Dict[str, float]] = None,
    asset_meta: Optional[Dict[str, Dict[str, float]]] = None,
    rebalance_frequency: str = "quarterly",
    language: str = "en",
) -> Dict[str, Any]:
    """Estimate one-way turnover and execution cost diagnostics."""
    if not _enabled():
        return {}

    notes: List[str] = []
    freq = str(rebalance_frequency or "quarterly").strip().lower()
    if freq not in _VALID_REBALANCE:
        freq = "quarterly"
        notes.append("invalid rebalance frequency coerced to quarterly")

    w = _normalise_weights(weights)
    prev = _normalise_weights(w_prev) if w_prev is not None else None
    if prev is not None:
        prev = _align_previous_keys(prev, w, asset_meta)
    all_assets = set(w)
    if prev is not None:
        all_assets |= set(prev)

    if prev is None:
        deltas = {asset: abs(w.get(asset, 0.0)) for asset in all_assets}
        turnover_pct = sum(deltas.values()) * 100.0
    else:
        deltas = {asset: abs(w.get(asset, 0.0) - prev.get(asset, 0.0)) for asset in all_assets}
        turnover_pct = 0.5 * sum(deltas.values()) * 100.0

    participation = max(0.0, _env_float("EISAX_TC_ADV_PARTICIPATION", "0.05"))
    impact_coef = _env_float("EISAX_TC_IMPACT_COEF", "10")
    capital = max(0.0, _env_float("EISAX_TC_PORT_VALUE_USD", "100000"))

    slippage_bp = 0.0
    market_impact_bp = 0.0
    changed = 0
    stress_tiers: List[str] = []
    gcc_weight = 0.0
    discontinuity = False

    for asset in sorted(all_assets):
        delta = float(deltas.get(asset, 0.0))
        if delta <= 1e-10:
            continue
        changed += 1
        meta = _meta_for(asset, asset_meta)
        bucket = _asset_bucket(asset, meta)
        sigma_daily_pct = _daily_vol_pct(asset, meta)
        spread = _spread_bp(bucket, meta)
        impact = impact_coef * sigma_daily_pct * math.sqrt(participation)
        asset_slippage = (spread / 2.0 + impact) * _liquidity_multiplier(bucket)
        if bucket == "crypto":
            extra, triggered = _crypto_extra_bp(delta * capital)
            asset_slippage += extra
            discontinuity = discontinuity or triggered
        slippage_bp += delta * asset_slippage
        market_impact_bp += delta * 10.0 * sigma_daily_pct * math.sqrt(participation)
        stress_tiers.append(_stress_tier(participation))
        if bucket == "gcc_equity":
            gcc_weight += w.get(asset, 0.0)

    if turnover_pct < 5 and changed <= 3:
        complexity = "low"
    elif turnover_pct < 15 and changed <= 8:
        complexity = "moderate"
    elif turnover_pct < 35:
        complexity = "elevated"
    else:
        complexity = "high"

    lin_l = _env_float("EISAX_TC_LINEAR_LAMBDA", "0.0010")
    quad_l = _env_float("EISAX_TC_QUADRATIC_LAMBDA", "0.0005")
    pers_l = _env_float("EISAX_TC_PERSISTENCE_LAMBDA", "0.0002")
    max_observed = None
    if isinstance(asset_meta, Mapping):
        try:
            max_observed = float(asset_meta.get("max_observed_turnover_in_lookback"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            max_observed = None
    persistence_pref = 0.0
    if max_observed and max_observed > 0:
        persistence_pref = max(0.0, min(100.0, 100.0 * (1.0 - (turnover_pct / 100.0) / max_observed)))

    payload: ExecutionDiagnostics = {
        "turnover_pct": round(turnover_pct, 2),
        "implementation_shortfall_bp": round(slippage_bp + 2.0, 2),
        "market_impact_bp": round(market_impact_bp, 2),
        "slippage_bp": round(slippage_bp, 2),
        "complexity_tier": complexity,
        "liquidity_stress": _worst_tier(stress_tiers),
        "tax_note": _TAX_NOTE.get(language, _TAX_NOTE["en"]),
        "rebalance_frequency": freq,
        "turnover_penalty_applied": bool(prev is not None and lin_l > 0),
        "quadratic_penalty_applied": bool(quad_l > 0),
        "persistence_preference_pct": round(persistence_pref, 2),
        "liquidity_discontinuity_triggered": discontinuity,
        "notes": notes,
    }
    if gcc_weight >= 0.10:
        payload["gcc_calendar_note"] = "GCC legs use a Sun-Thu trading calendar; execution windows should account for the four-day local trading week."
    payload["linear_lambda"] = round(lin_l, 6)
    payload["quadratic_lambda"] = round(quad_l, 6)
    payload["persistence_lambda"] = round(pers_l, 6)
    return _make_execution_envelope(payload, notes)


def render_execution_md(payload: Mapping[str, Any], language: str = "en") -> str:
    if not _enabled() or not payload:
        return ""
    data = unwrap(payload)
    if not data and isinstance(payload, Mapping):
        data = dict(payload)

    turnover = float(data.get("turnover_pct", 0.0) or 0.0)
    shortfall = float(data.get("implementation_shortfall_bp", 0.0) or 0.0)
    impact = float(data.get("market_impact_bp", 0.0) or 0.0)
    slippage = float(data.get("slippage_bp", 0.0) or 0.0)

    rows: List[List[str]] = [
        [L("turnover", language), fmt_pct(turnover, 1), severity_tag(_turnover_tag(turnover), language)],
        [L("implementation_shortfall", language), fmt_bp(shortfall), severity_tag(_shortfall_tag(shortfall), language)],
        [L("market_impact", language), fmt_bp(impact), severity_tag(_impact_tag(impact), language)],
        [L("slippage", language), fmt_bp(slippage), severity_tag(_slippage_tag(slippage), language)],
        [L("complexity", language), str(data.get("complexity_tier", "—")), "—"],
        [L("liquidity_stress", language), str(data.get("liquidity_stress", "—")), "—"],
        [L("rebalance_freq", language), str(data.get("rebalance_frequency", "—")), "—"],
    ]
    heading = section_heading(3, "execution_efficiency", language)
    table = md_table([L("metric", language), L("value", language), L("tag", language)], rows)
    applied = _ll("applied" if data.get("turnover_penalty_applied") else "not_applied", language)
    lin = float(data.get("linear_lambda", _env_float("EISAX_TC_LINEAR_LAMBDA", "0.0010")) or 0.0)
    quad = float(data.get("quadratic_lambda", _env_float("EISAX_TC_QUADRATIC_LAMBDA", "0.0005")) or 0.0)
    penalty = f"*{_ll('turnover_penalty', language)}: {applied} (linear λ={lin:.4f}, quadratic λ={quad:.4f})*"
    lines = [heading, "", table, "", penalty, "", f"*{data.get('tax_note') or _TAX_NOTE.get(language, _TAX_NOTE['en'])}*"]
    if data.get("gcc_calendar_note"):
        lines.extend(["", f"*{_ll('gcc_calendar_note', language)}: {data['gcc_calendar_note']}*"])
    notes = [str(n) for n in data.get("notes", []) if n]
    if notes:
        lines.extend(["", f"*{_ll('notes', language)}: {'; '.join(notes)}*"])
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ENGINE_VERSION",
    "build_turnover_terms",
    "estimate_execution",
    "render_execution_md",
]
