from __future__ import annotations


def build_timing_phrase(entry_quality: str) -> str:
    phrase_map = {
        "favorable entry": "Timing is improving: price is near the preferred entry zone.",
        "acceptable entry": "Entry conditions are acceptable but not optimal.",
        "stretched entry": "Timing is becoming stretched relative to the preferred entry zone.",
        "poor timing": "Timing remains poor: price is extended above the preferred entry zone.",
        "entry level unavailable": "Entry conditions cannot be assessed from the available data.",
    }
    return phrase_map.get(entry_quality, "Entry conditions cannot be assessed from the available data.")


def build_trend_phrase(
    trend_strength: str,
    rsi_zone: str,
    primary_trend: str = "neutral",
    trend_borderline: bool = False,
) -> str:
    structure_map = {
        "bullish": "The stock is in a bullish structure",
        "bearish": "The primary trend remains bearish",
        "neutral": "The structure remains mixed",
    }
    momentum_map = {
        "oversold": "with oversold momentum conditions.",
        "weak momentum": "while momentum remains weak.",
        "neutral momentum": "with neutral momentum conditions.",
        "bullish momentum": "while momentum is improving.",
        "overbought": "while momentum is overbought.",
    }
    trend_map = {
        "weak trend": "within a weak trend regime",
        "emerging trend": "with an emerging trend structure",
        "confirmed trend": "with confirmed trend strength",
        "strong trend": "with strong trend confirmation",
    }

    # Borderline emerging trend uses composite label
    effective_trend_strength = trend_strength
    if trend_strength == "emerging trend" and trend_borderline:
        trend_map = dict(trend_map)
        trend_map["emerging trend"] = "with a weak-to-emerging trend structure"
        effective_trend_strength = "emerging trend"

    structure = structure_map.get(primary_trend, structure_map["neutral"])
    trend_text = trend_map.get(effective_trend_strength, "with limited trend visibility")
    momentum_text = momentum_map.get(rsi_zone, "with limited momentum visibility.")

    # Macro/technical divergence: bullish macro but weak/emerging technical
    if primary_trend == "bullish" and trend_strength in {"weak trend", "emerging trend"}:
        borderline_note = " (ADX approaching confirmation)" if trend_borderline else ""
        return (
            f"The stock is directionally aligned with the broader market, but ADX-based trend "
            f"confirmation is absent{borderline_note}. Sector tailwinds exist, but stock-specific "
            f"technical conditions have not confirmed follow-through."
        )

    if trend_strength == "weak trend" and rsi_zone == "bullish momentum":
        return f"{structure}, while momentum is improving within a weak trend regime."
    if trend_strength == "confirmed trend" and primary_trend == "bullish":
        return "The stock is in a bullish structure with confirmed trend strength."
    if trend_strength == "weak trend" and rsi_zone in {"neutral momentum", "weak momentum"}:
        return f"{structure} {trend_text}, and momentum remains unconvincing."
    if trend_strength in {"weak trend", "emerging trend"} and rsi_zone == "bullish momentum":
        return "Momentum is positive, but trend confirmation remains insufficient."

    # Bearish primary trend + neutral RSI: momentum is weakening, not outright bearish
    if primary_trend == "bearish" and rsi_zone == "neutral momentum":
        return f"{structure} while momentum is weakening."

    return f"{structure} {trend_text}, {momentum_text}"


def build_yield_phrase(yield_quality: str) -> str:
    phrase_map = {
        "yield unavailable": "Yield contribution cannot be assessed from the available data.",
        "minimal yield": "Yield contribution is minimal.",
        "low yield": "Income contribution remains limited.",
        "moderate yield": "Dividend support is moderate.",
        "attractive yield": "The stock offers an attractive income component.",
        "high yield": "The stock offers a high income component, but sustainability should remain under review.",
    }
    return phrase_map.get(yield_quality, "Yield contribution cannot be assessed from the available data.")


def build_volume_phrase(volume_conviction: str) -> str:
    phrase_map = {
        "volume confirmation unavailable": "Volume confirmation is unavailable.",
        "low-conviction volume": "Volume participation remains low-conviction.",
        "normal volume conviction": "Volume participation is in line with the 90-day average.",
        "strong volume confirmation": "Volume is providing strong confirmation.",
    }
    return phrase_map.get(volume_conviction, "Volume confirmation is unavailable.")


def build_support_phrase(support_proximity: str) -> str:
    phrase_map = {
        "support level unavailable": "Support positioning is unavailable.",
        "near support": "Price is near support.",
        "above support zone": "Price is holding above the support zone.",
        "extended above support": "Price is extended above support.",
    }
    return phrase_map.get(support_proximity, "Support positioning is unavailable.")


def build_resistance_phrase(resistance_proximity: str) -> str:
    phrase_map = {
        "resistance level unavailable": "Resistance positioning is unavailable.",
        "near resistance": "Price is near resistance.",
        "approaching resistance": "Price is approaching resistance.",
        "well below resistance": "Price remains well below resistance.",
    }
    return phrase_map.get(resistance_proximity, "Resistance positioning is unavailable.")


def build_portfolio_role_phrase(yield_quality: str, trend_strength: str) -> str:
    if yield_quality in {"attractive yield", "high yield"} and trend_strength in {"weak trend", "emerging trend"}:
        return "The name is better framed as an income-bearing allocation than a momentum-led trade."
    if yield_quality in {"attractive yield", "high yield"}:
        return "The name can serve as both an income contributor and a strategic core holding."
    if trend_strength in {"confirmed trend", "strong trend"}:
        return "The opportunity is better framed as a tactical growth allocation than an income anchor."
    return "The position is better suited to a measured tactical allocation than an aggressive core weighting."


def build_approved_phrase_map(
    interpretation_labels: dict[str, str],
    primary_trend: str = "neutral",
) -> dict[str, str]:
    trend_strength = interpretation_labels.get("TrendStrength", "")
    trend_borderline = bool(interpretation_labels.get("TrendBorderline", False))
    rsi_zone = interpretation_labels.get("RSIZone", "")
    support_proximity = interpretation_labels.get("SupportProximity", "")
    resistance_proximity = interpretation_labels.get("ResistanceProximity", "")
    yield_quality = interpretation_labels.get("YieldQuality", "")
    entry_quality = interpretation_labels.get("EntryQuality", "")
    volume_conviction = interpretation_labels.get("VolumeConviction", "")

    def _trend() -> str:
        return build_trend_phrase(trend_strength, rsi_zone, primary_trend, trend_borderline)

    return {
        "ExecutiveSummary": f"{_trend()} {build_timing_phrase(entry_quality)}",
        "TechnicalOutlook": " ".join(
            [
                _trend(),
                build_support_phrase(support_proximity),
                build_resistance_phrase(resistance_proximity),
                build_volume_phrase(volume_conviction),
            ]
        ).strip(),
        "WhyNow": f"{build_timing_phrase(entry_quality)} {_trend()}",
        "PortfolioRole": f"{build_yield_phrase(yield_quality)} {build_portfolio_role_phrase(yield_quality, trend_strength)}",
        "Timing": build_timing_phrase(entry_quality),
        "Trend": _trend(),
        "Yield": build_yield_phrase(yield_quality),
        "Volume": build_volume_phrase(volume_conviction),
    }


def format_approved_phrase_block(phrase_map: dict[str, str]) -> str:
    ordered_fields = [
        "ExecutiveSummary",
        "TechnicalOutlook",
        "WhyNow",
        "PortfolioRole",
    ]
    lines = ["[APPROVED PHRASE MAP - LOCKED]"]
    for field in ordered_fields:
        lines.append(f"{field}: {phrase_map.get(field, 'Unavailable.')}")
    lines.extend(
        [
            "",
            "SECTION RULES:",
            "- Executive Summary must use the ExecutiveSummary phrase logic for strength, main risk, and timing posture.",
            "- Technical Outlook must use the TechnicalOutlook phrase logic for trend, RSI, support/resistance, and volume.",
            "- Why Now must use the WhyNow phrase logic for timing and trend confirmation.",
            "- Portfolio Role must use the PortfolioRole phrase logic for yield and tactical versus strategic framing.",
        ]
    )
    return "\n".join(lines)


def build_quick_insight(
    snapshot: dict,
    interpretation_labels: dict[str, str],
    decision: dict | None = None,
) -> str:
    """
    Always returns a non-empty, data-tied phrase. NO generic fallback.

    When *decision* (from build_decision()) is supplied, the phrase is
    enriched with the verdict and its primary constraint so the reader
    understands WHY the stance was taken.
    """
    trend_strength = interpretation_labels.get("TrendStrength", "")
    entry_quality  = interpretation_labels.get("EntryQuality", "")
    yield_quality  = interpretation_labels.get("YieldQuality", "")
    rsi_zone       = interpretation_labels.get("RSIZone", "")

    # ── Decision-aware path (verdict + constraint) ─────────────────────────
    if decision:
        verdict       = str(decision.get("verdict", "HOLD")).upper()
        verdict_type  = decision.get("verdict_type", "Tactical")
        constraints   = decision.get("constraints", [])

        # Map the first constraint to a human-readable note
        constraint_note = ""
        if constraints:
            c = constraints[0]
            if "weak trend" in c and "poor timing" in c:
                constraint_note = "weak ADX and stretched entry conditions"
            elif "RSI overbought" in c:
                constraint_note = "overbought RSI momentum"
            elif "trend_strength" in c and "insufficient" in c:
                constraint_note = "ADX below trend-confirmation threshold"
            elif "upside" in c and "%" in c:
                constraint_note = "insufficient price-to-target upside"
            elif "beta" in c:
                constraint_note = "elevated beta and risk profile"
            elif "Income Hold" in c:
                constraint_note = "attractive yield with weak trend regime"

        if verdict == "HOLD":
            if verdict_type == "Income Hold":
                return (
                    "Income is attractive; HOLD stance appropriate pending "
                    "trend confirmation."
                )
            if constraint_note:
                return (
                    f"Trend confirmation is absent; HOLD stance reflects "
                    f"{constraint_note}."
                )
            # Prevent fallthrough to signal-only path which can imply a different action.
            # Return a neutral HOLD phrase grounded in available signal context.
            _hold_ctx = (
                f"momentum is {rsi_zone}" if rsi_zone and rsi_zone not in ("", "neutral momentum")
                else f"entry quality is {entry_quality}" if entry_quality
                else "conditions are mixed"
            )
            return (
                f"Position maintained — {_hold_ctx}. "
                "Await clearer confirmation before adding or reducing exposure."
            )
        if verdict == "BUY" and trend_strength in ("confirmed trend", "strong trend"):
            return (
                f"Setup supports a BUY: trend is {trend_strength} with "
                f"{entry_quality}."
            )
        # BUY via Rule 8A — strong fundamentals, entry timing is weak
        if verdict == "BUY":
            _timing_reason = constraint_note or rsi_zone or entry_quality or "technical timing"
            return (
                f"Fundamental BUY — strong business quality and upside justify the position. "
                f"Entry timing is suboptimal ({_timing_reason}): wait for a pullback before adding."
            )
        if verdict == "REDUCE":
            return (
                "Elevated risk signals warrant a REDUCE stance; "
                "await improved conditions before re-entering."
            )
        if verdict == "AVOID":
            return (
                "Risk-reward is unfavorable; AVOID positioning until "
                "key technical and fundamental conditions improve."
            )

    # ── Signal-only path (no decision supplied) ────────────────────────────

    # Overbought momentum inside a weak/emerging trend
    if rsi_zone == "overbought" and trend_strength in {"weak trend", "emerging trend"}:
        return (
            "Momentum is overbought within a weak trend regime, "
            "making near-term entry unfavorable."
        )

    # Timing overrides (worst-case first)
    if entry_quality == "poor timing" and trend_strength in {"weak trend", "emerging trend"}:
        return (
            "Timing remains weak: price is extended above the preferred entry zone "
            "while ADX stays below trend-confirmation levels."
        )
    if entry_quality == "poor timing":
        return (
            "Timing remains poor: price is extended above the preferred entry zone "
            "despite established trend strength."
        )

    # Income-first framing
    if yield_quality in {"attractive yield", "high yield"} and trend_strength in {"weak trend", "emerging trend"}:
        return "Income is attractive, but trend confirmation remains weak."

    # Weak/emerging trend — reference RSI
    if trend_strength in {"weak trend", "emerging trend"}:
        if rsi_zone in {"neutral momentum", "weak momentum"}:
            return (
                "Momentum is weakening and trend confirmation is absent; "
                "the setup does not yet support aggressive entry."
            )
        return (
            "Trend confirmation is absent; await ADX strengthening "
            "before adding directional exposure."
        )

    ticker = "Stock"
    if isinstance(snapshot, dict):
        ticker = snapshot.get("ticker") or snapshot.get("symbol") or ticker
    return f"{ticker} is under rule-based review while interpretation signals remain incomplete."
