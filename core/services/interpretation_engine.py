from __future__ import annotations


def _normalize_yield_percent(div_yield: float | None) -> float | None:
    if div_yield is None:
        return None
    value = float(div_yield)
    if value < 0:
        return value
    return value if value > 1 else value * 100


def classify_trend_strength(adx: float) -> dict:
    """Return {'label': str, 'borderline': bool}.

    ADX ranges:
      <20      → weak      (never borderline)
      20–25    → emerging  (borderline if ADX ≥ 24 — close to confirmed territory)
      25–30    → confirmed (borderline if ADX < 26 — close to emerging territory)
      ≥30      → strong    (never borderline)
    """
    value = float(adx or 0)
    if value < 20:
        return {"label": "weak trend", "borderline": False}
    if value < 25:
        return {"label": "emerging trend", "borderline": value >= 24}
    if value < 30:
        return {"label": "confirmed trend", "borderline": value < 26}
    return {"label": "strong trend", "borderline": False}


def classify_rsi_zone(rsi: float) -> str:
    value = float(rsi or 0)
    if value < 30:
        return "oversold"
    if value < 45:
        return "weak momentum"
    if value < 60:
        return "neutral momentum"
    if value < 70:
        return "bullish momentum"
    return "overbought"


def classify_support_proximity(price: float, support: float) -> str:
    if not support or float(support) <= 0:
        return "support level unavailable"
    distance = abs(float(price) - float(support)) / float(support)
    if distance <= 0.02:
        return "near support"
    if distance <= 0.05:
        return "above support zone"
    return "extended above support"


def classify_resistance_proximity(price: float, resistance: float) -> str:
    if not resistance or float(resistance) <= 0:
        return "resistance level unavailable"
    distance = abs(float(price) - float(resistance)) / float(resistance)
    if distance <= 0.02:
        return "near resistance"
    if distance <= 0.05:
        return "approaching resistance"
    return "well below resistance"


def classify_yield_quality(div_yield: float | None) -> str:
    percent_value = _normalize_yield_percent(div_yield)
    if percent_value is None:
        return "yield unavailable"
    if percent_value < 0.5:
        return "minimal yield"
    if percent_value < 2:
        return "low yield"
    if percent_value < 4:
        return "moderate yield"
    if percent_value < 6:
        return "attractive yield"
    return "high yield"


def classify_entry_quality(current_price: float, entry_price: float) -> str:
    if not entry_price or float(entry_price) <= 0:
        return "entry level unavailable"
    current_value = float(current_price or 0)
    entry_value = float(entry_price)
    if current_value <= entry_value * 1.01:
        return "favorable entry"
    if current_value <= entry_value * 1.03:
        return "acceptable entry"
    if current_value <= entry_value * 1.08:
        return "stretched entry"
    return "poor timing"


def classify_volume_conviction(
    today_volume: float | None,
    avg_volume: float | None,
) -> str:
    if not today_volume or not avg_volume or float(avg_volume) <= 0:
        return "volume confirmation unavailable"
    ratio = float(today_volume) / float(avg_volume)
    if ratio < 0.8:
        return "low-conviction volume"
    if ratio <= 1.2:
        return "normal volume conviction"
    return "strong volume confirmation"


def build_interpretation_labels(
    adx: float = 0,
    rsi: float = 50,
    price: float = 0,
    support: float = 0,
    resistance: float = 0,
    div_yield: float | None = None,
    entry_price: float | None = None,
    volume_today: float | None = None,
    volume_avg: float | None = None,
) -> dict[str, str]:
    _trend_detail = classify_trend_strength(adx)
    return {
        "TrendStrength": _trend_detail["label"],
        "TrendBorderline": _trend_detail["borderline"],
        "RSIZone": classify_rsi_zone(rsi),
        "SupportProximity": (
            classify_support_proximity(price, support)
            if price and support
            else "support level unavailable"
        ),
        "ResistanceProximity": (
            classify_resistance_proximity(price, resistance)
            if price and resistance
            else "resistance level unavailable"
        ),
        "YieldQuality": classify_yield_quality(div_yield),
        "EntryQuality": (
            classify_entry_quality(price, entry_price)
            if price and entry_price
            else "entry level unavailable"
        ),
        "VolumeConviction": classify_volume_conviction(volume_today, volume_avg),
    }


def format_interpretation_block(labels: dict[str, str]) -> str:
    ordered_fields = [
        "TrendStrength",
        "TrendBorderline",
        "RSIZone",
        "SupportProximity",
        "ResistanceProximity",
        "YieldQuality",
        "EntryQuality",
        "VolumeConviction",
    ]
    lines = ["[INTERPRETATION BLOCK - LOCKED]"]
    for field in ordered_fields:
        lines.append(f"{field}: {labels.get(field, 'unavailable')}")
    lines.extend(
        [
            "",
            "RULES:",
            "- You MUST use these exact interpretation labels.",
            "- You MUST NOT reinterpret technical conditions differently.",
            "- You MUST NOT invent alternative labels.",
            "- You MUST NOT state stronger or weaker claims than this block supports.",
        ]
    )
    return "\n".join(lines)
