from __future__ import annotations

import pytest

from core.services.interpretation_engine import (
    build_interpretation_labels,
    classify_entry_quality,
    classify_resistance_proximity,
    classify_rsi_zone,
    classify_support_proximity,
    classify_trend_strength,
    classify_volume_conviction,
    classify_yield_quality,
    format_interpretation_block,
)
from core.services.interpretation_guard import InterpretationGuard
from core.services.phrase_builder import (
    build_approved_phrase_map,
    build_quick_insight,
    build_timing_phrase,
    build_trend_phrase,
    build_yield_phrase,
)
from core.services.report_lint import (
    RenderedReport,
    ReportSection,
    lint_entry_language,
    lint_report,
    lint_support_language,
    lint_trend_language,
    lint_volume_language,
    lint_yield_language,
)
from core.services.report_snapshot import ReportSnapshot


def _base_snapshot_raw() -> dict:
    return {
        "ticker": {"value": "MSFT", "source": "fallback", "timestamp": "2026-04-15T00:00:00"},
        "price": {"value": 395.0, "source": "realtime", "timestamp": "2026-04-15T00:00:00"},
        "entry": {"value": 382.0, "source": "calculated", "timestamp": "2026-04-15T00:00:00"},
        "stop": {"value": 378.0, "source": "calculated", "timestamp": "2026-04-15T00:00:00"},
        "target": {"value": 430.0, "source": "calculated", "timestamp": "2026-04-15T00:00:00"},
        "beta": {"value": 1.1, "source": "cache", "timestamp": "2026-04-15T00:00:00"},
        "pe": {"value": 28.4, "source": "cache", "timestamp": "2026-04-15T00:00:00"},
        "forward_pe": {"value": 26.2, "source": "cache", "timestamp": "2026-04-15T00:00:00"},
        "sma50": {"value": 401.0, "source": "calculated", "timestamp": "2026-04-15T00:00:00"},
        "sma200": {"value": 404.0, "source": "calculated", "timestamp": "2026-04-15T00:00:00"},
        "week52_high": {"value": 468.0, "source": "cache", "timestamp": "2026-04-15T00:00:00"},
        "week52_low": {"value": 344.0, "source": "cache", "timestamp": "2026-04-15T00:00:00"},
        "market_cap": {"value": 3.1e12, "source": "cache", "timestamp": "2026-04-15T00:00:00"},
        "div_yield": {"value": 0.008, "source": "cache", "timestamp": "2026-04-15T00:00:00"},
    }


def _build_snapshot(context: dict | None = None) -> ReportSnapshot:
    snapshot = ReportSnapshot(_base_snapshot_raw())
    final_context = context or {
        "adx": 18.0,
        "rsi": 63.0,
        "price": 395.0,
        "support": 380.0,
        "resistance": 402.0,
        "div_yield": 0.05,
        "entry_price": 382.0,
        "volume_today": 700_000,
        "volume_avg": 1_000_000,
    }
    final_context.setdefault(
        "labels",
        build_interpretation_labels(
            adx=float(final_context.get("adx") or 0),
            rsi=float(final_context.get("rsi") or 50),
            price=float(final_context.get("price") or 0),
            support=float(final_context.get("support") or 0),
            resistance=float(final_context.get("resistance") or 0),
            div_yield=final_context.get("div_yield"),
            entry_price=float(final_context.get("entry_price") or 0) or None,
            volume_today=final_context.get("volume_today"),
            volume_avg=final_context.get("volume_avg"),
        ),
    )
    snapshot.set(
        "_interpretation_context",
        {"value": final_context, "source": "calculated", "timestamp": "2026-04-15T00:00:00"},
    )
    snapshot.freeze()
    return snapshot


def _report(text: str) -> RenderedReport:
    return RenderedReport(
        ticker="MSFT",
        full_text=text,
        sections=[ReportSection("Memo", text)],
        entry=382.0,
        stop=378.0,
        target=430.0,
        observed_prices=[395.0],
    )


@pytest.mark.parametrize(
    ("adx", "expected_label", "expected_borderline"),
    [
        (0,     "weak trend",      False),
        (19.99, "weak trend",      False),
        (20,    "emerging trend",  False),
        (23.99, "emerging trend",  False),
        (24,    "emerging trend",  True),   # borderline — close to confirmed
        (24.9,  "emerging trend",  True),   # ADX 24.9 must NOT block
        (25,    "confirmed trend", True),   # borderline — close to emerging
        (25.9,  "confirmed trend", True),
        (26,    "confirmed trend", False),
        (29.99, "confirmed trend", False),
        (30,    "strong trend",    False),
    ],
)
def test_classify_trend_strength_boundaries(adx, expected_label, expected_borderline):
    result = classify_trend_strength(adx)
    assert result["label"] == expected_label
    assert result["borderline"] == expected_borderline


@pytest.mark.parametrize(
    ("rsi", "expected"),
    [
        (0, "oversold"),
        (29.99, "oversold"),
        (30, "weak momentum"),
        (44.99, "weak momentum"),
        (45, "neutral momentum"),
        (59.99, "neutral momentum"),
        (60, "bullish momentum"),
        (69.99, "bullish momentum"),
        (70, "overbought"),
    ],
)
def test_classify_rsi_zone_boundaries(rsi, expected):
    assert classify_rsi_zone(rsi) == expected


@pytest.mark.parametrize(
    ("price", "support", "expected"),
    [
        (100, 100, "near support"),
        (102, 100, "near support"),
        (104, 100, "above support zone"),
        (110, 100, "extended above support"),
        (100, 0, "support level unavailable"),
    ],
)
def test_classify_support_proximity(price, support, expected):
    assert classify_support_proximity(price, support) == expected


@pytest.mark.parametrize(
    ("price", "resistance", "expected"),
    [
        (100, 101, "near resistance"),
        (100, 104, "approaching resistance"),
        (100, 110, "well below resistance"),
        (100, 0, "resistance level unavailable"),
    ],
)
def test_classify_resistance_proximity(price, resistance, expected):
    assert classify_resistance_proximity(price, resistance) == expected


@pytest.mark.parametrize(
    ("div_yield", "expected"),
    [
        (None, "yield unavailable"),
        (0.001, "minimal yield"),
        (0.005, "low yield"),
        (0.03, "moderate yield"),
        (0.05, "attractive yield"),
        (0.07, "high yield"),
        (5.0, "attractive yield"),
    ],
)
def test_classify_yield_quality_boundaries(div_yield, expected):
    assert classify_yield_quality(div_yield) == expected


@pytest.mark.parametrize(
    ("current_price", "entry_price", "expected"),
    [
        (100, 100, "favorable entry"),
        (101, 100, "favorable entry"),
        (103, 100, "acceptable entry"),
        (108, 100, "stretched entry"),
        (109, 100, "poor timing"),
        (100, 0, "entry level unavailable"),
    ],
)
def test_classify_entry_quality_boundaries(current_price, entry_price, expected):
    assert classify_entry_quality(current_price, entry_price) == expected


@pytest.mark.parametrize(
    ("today_volume", "avg_volume", "expected"),
    [
        (None, 1_000_000, "volume confirmation unavailable"),
        (700_000, 1_000_000, "low-conviction volume"),
        (800_000, 1_000_000, "normal volume conviction"),
        (1_200_000, 1_000_000, "normal volume conviction"),
        (1_300_000, 1_000_000, "strong volume confirmation"),
    ],
)
def test_classify_volume_conviction_boundaries(today_volume, avg_volume, expected):
    assert classify_volume_conviction(today_volume, avg_volume) == expected


def test_build_interpretation_labels_returns_all_required_keys():
    labels = build_interpretation_labels(
        adx=18,
        rsi=63,
        price=395,
        support=380,
        resistance=402,
        div_yield=0.05,
        entry_price=382,
        volume_today=700_000,
        volume_avg=1_000_000,
    )
    assert set(labels) == {
        "TrendStrength",
        "TrendBorderline",
        "RSIZone",
        "SupportProximity",
        "ResistanceProximity",
        "YieldQuality",
        "EntryQuality",
        "VolumeConviction",
    }


def test_format_interpretation_block_contains_locked_header():
    block = format_interpretation_block({"TrendStrength": "weak trend"})
    assert "[INTERPRETATION BLOCK - LOCKED]" in block
    assert "TrendStrength: weak trend" in block


def test_build_timing_phrase_poor_timing():
    assert build_timing_phrase("poor timing") == "Timing remains poor: price is extended above the preferred entry zone."


def test_build_trend_phrase_confirmed_bullish():
    assert build_trend_phrase("confirmed trend", "bullish momentum", "bullish") == "The stock is in a bullish structure with confirmed trend strength."


def test_build_yield_phrase_attractive():
    assert build_yield_phrase("attractive yield") == "The stock offers an attractive income component."


def test_build_approved_phrase_map_includes_section_outputs():
    phrase_map = build_approved_phrase_map(
        {
            "TrendStrength": "weak trend",
            "RSIZone": "bullish momentum",
            "SupportProximity": "extended above support",
            "ResistanceProximity": "near resistance",
            "YieldQuality": "attractive yield",
            "EntryQuality": "poor timing",
            "VolumeConviction": "low-conviction volume",
        },
        primary_trend="bullish",
    )
    assert "ExecutiveSummary" in phrase_map
    assert "PortfolioRole" in phrase_map
    assert phrase_map["Timing"] == "Timing remains poor: price is extended above the preferred entry zone."


def test_interpretation_guard_rewrites_conflicting_trend_sentence():
    guard = InterpretationGuard()
    result = guard.audit_and_sanitize(
        "The stock is in a strong trend and looks extended.",
        {
            "TrendStrength": "weak trend",
            "RSIZone": "bullish momentum",
            "YieldQuality": "low yield",
            "EntryQuality": "poor timing",
            "VolumeConviction": "low-conviction volume",
            "SupportProximity": "extended above support",
        },
    )
    assert result.replacements_made == 1
    assert result.audit_log[0]["event"] == "interpretation_override"
    assert "weak trend regime" in result.text.lower()


def test_interpretation_guard_rewrites_conflicting_yield_sentence():
    guard = InterpretationGuard()
    result = guard.audit_and_sanitize(
        "The stock offers an attractive yield for investors.",
        {
            "TrendStrength": "weak trend",
            "RSIZone": "weak momentum",
            "YieldQuality": "minimal yield",
            "EntryQuality": "acceptable entry",
            "VolumeConviction": "normal volume conviction",
            "SupportProximity": "above support zone",
        },
    )
    assert result.replacements_made == 1
    assert result.audit_log[0]["field"] == "yield_quality"
    assert "minimal" in result.text.lower()


def test_interpretation_guard_leaves_compliant_text_unchanged():
    guard = InterpretationGuard()
    text = "Timing remains poor: price is extended above the preferred entry zone."
    result = guard.audit_and_sanitize(
        text,
        {
            "TrendStrength": "weak trend",
            "RSIZone": "weak momentum",
            "YieldQuality": "low yield",
            "EntryQuality": "poor timing",
            "VolumeConviction": "normal volume conviction",
            "SupportProximity": "extended above support",
        },
    )
    assert result.replacements_made == 0
    assert result.text == text


def test_lint_trend_language_flags_strong_trend_when_adx_is_weak():
    snapshot = _build_snapshot({"adx": 18, "price": 395, "support": 380, "div_yield": 0.05, "entry_price": 382, "volume_today": 700_000, "volume_avg": 1_000_000})
    result = lint_trend_language(_report("The setup shows a strong trend."), snapshot)
    assert result["result"] == "ERROR"


def test_lint_trend_language_borderline_adx_24_downgrades_to_warning():
    # ADX 24 is borderline-emerging; a wrong phrase (confirmed trend) must be WARNING not ERROR
    snapshot = _build_snapshot({"adx": 24, "price": 395, "support": 380, "div_yield": 0.05, "entry_price": 382, "volume_today": 700_000, "volume_avg": 1_000_000})
    result = lint_trend_language(_report("The chart now shows a confirmed trend."), snapshot)
    assert result["result"] == "WARNING"
    assert result.get("event") == "trend_tolerance_override"


def test_lint_trend_language_flags_weak_trend_when_adx_is_confirmed():
    snapshot = _build_snapshot({"adx": 27, "price": 395, "support": 380, "div_yield": 0.05, "entry_price": 382, "volume_today": 700_000, "volume_avg": 1_000_000})
    result = lint_trend_language(_report("The stock is in a weak trend."), snapshot)
    assert result["result"] == "ERROR"


def test_lint_trend_language_passes_when_label_matches():
    snapshot = _build_snapshot({"adx": 31, "price": 395, "support": 380, "div_yield": 0.05, "entry_price": 382, "volume_today": 700_000, "volume_avg": 1_000_000})
    result = lint_trend_language(_report("The stock remains in a strong trend."), snapshot)
    assert result["result"] == "PASS"


def test_lint_trend_language_skips_without_adx():
    snapshot = _build_snapshot({"adx": 0})
    result = lint_trend_language(_report("No trend commentary."), snapshot)
    assert result["result"] == "SKIP"


def test_lint_support_language_flags_near_support_when_distance_is_too_wide():
    snapshot = _build_snapshot({"adx": 18, "price": 110, "support": 100, "div_yield": 0.05, "entry_price": 100, "volume_today": 700_000, "volume_avg": 1_000_000})
    result = lint_support_language(_report("Price is trading near support."), snapshot)
    assert result["result"] == "ERROR"


def test_lint_support_language_flags_extended_above_support_when_near():
    snapshot = _build_snapshot({"adx": 18, "price": 101, "support": 100, "div_yield": 0.05, "entry_price": 100, "volume_today": 700_000, "volume_avg": 1_000_000})
    result = lint_support_language(_report("Price is extended above support."), snapshot)
    assert result["result"] == "ERROR"


def test_lint_support_language_passes_when_label_matches():
    snapshot = _build_snapshot({"adx": 18, "price": 104, "support": 100, "div_yield": 0.05, "entry_price": 100, "volume_today": 700_000, "volume_avg": 1_000_000})
    result = lint_support_language(_report("Price is holding above support zone."), snapshot)
    assert result["result"] == "PASS"


def test_lint_support_language_skips_without_context():
    snapshot = _build_snapshot({"adx": 18, "price": 0, "support": 0})
    result = lint_support_language(_report("Price is near support."), snapshot)
    assert result["result"] == "SKIP"


def test_lint_yield_language_flags_attractive_yield_when_too_low():
    snapshot = _build_snapshot({"adx": 18, "price": 395, "support": 380, "div_yield": 0.02, "entry_price": 382, "volume_today": 700_000, "volume_avg": 1_000_000})
    result = lint_yield_language(_report("The stock offers an attractive yield."), snapshot)
    assert result["result"] == "ERROR"


def test_lint_yield_language_flags_high_yield_when_below_threshold():
    snapshot = _build_snapshot({"adx": 18, "price": 395, "support": 380, "div_yield": 0.04, "entry_price": 382, "volume_today": 700_000, "volume_avg": 1_000_000})
    result = lint_yield_language(_report("The stock offers a high yield."), snapshot)
    assert result["result"] == "ERROR"


def test_lint_yield_language_passes_when_label_matches():
    snapshot = _build_snapshot({"adx": 18, "price": 395, "support": 380, "div_yield": 0.05, "entry_price": 382, "volume_today": 700_000, "volume_avg": 1_000_000})
    result = lint_yield_language(_report("The stock offers an attractive yield."), snapshot)
    assert result["result"] == "PASS"


def test_lint_yield_language_skips_without_yield():
    snapshot = _build_snapshot({"adx": 18, "price": 395, "support": 380, "div_yield": None, "entry_price": 382, "volume_today": 700_000, "volume_avg": 1_000_000})
    result = lint_yield_language(_report("Yield commentary is absent."), snapshot)
    assert result["result"] == "SKIP"


def test_lint_entry_language_flags_favorable_entry_when_price_is_too_high():
    snapshot = _build_snapshot({"adx": 18, "price": 105, "support": 100, "div_yield": 0.05, "entry_price": 100, "volume_today": 700_000, "volume_avg": 1_000_000})
    result = lint_entry_language(_report("This remains a favorable entry."), snapshot)
    assert result["result"] == "ERROR"


def test_lint_entry_language_warns_on_poor_timing_when_price_is_at_entry():
    snapshot = _build_snapshot({"adx": 18, "price": 100, "support": 99, "div_yield": 0.05, "entry_price": 100, "volume_today": 700_000, "volume_avg": 1_000_000})
    result = lint_entry_language(_report("Timing remains poor."), snapshot)
    assert result["result"] == "WARNING"
    assert result["warnings"] == ["POOR_TIMING_UNDERSHOOTS_RULE"]


def test_lint_entry_language_passes_when_poor_timing_is_valid():
    snapshot = _build_snapshot({"adx": 18, "price": 110, "support": 100, "div_yield": 0.05, "entry_price": 100, "volume_today": 700_000, "volume_avg": 1_000_000})
    result = lint_entry_language(_report("Timing remains poor."), snapshot)
    assert result["result"] == "PASS"


def test_lint_entry_language_skips_without_entry_context():
    snapshot = _build_snapshot({"adx": 18, "price": 0, "support": 0, "entry_price": 0})
    result = lint_entry_language(_report("Entry is favorable."), snapshot)
    assert result["result"] == "SKIP"


def test_lint_volume_language_flags_strong_volume_when_ratio_is_too_low():
    snapshot = _build_snapshot({"adx": 18, "price": 395, "support": 380, "div_yield": 0.05, "entry_price": 382, "volume_today": 1_000_000, "volume_avg": 1_000_000})
    result = lint_volume_language(_report("The move has strong volume confirmation."), snapshot)
    assert result["result"] == "ERROR"


def test_lint_volume_language_passes_when_ratio_supports_language():
    snapshot = _build_snapshot({"adx": 18, "price": 395, "support": 380, "div_yield": 0.05, "entry_price": 382, "volume_today": 1_400_000, "volume_avg": 1_000_000})
    result = lint_volume_language(_report("The move has strong volume confirmation."), snapshot)
    assert result["result"] == "PASS"


def test_lint_volume_language_skips_without_volume_context():
    snapshot = _build_snapshot({"adx": 18, "price": 395, "support": 380, "div_yield": 0.05, "entry_price": 382, "volume_today": 0, "volume_avg": 0})
    result = lint_volume_language(_report("The move has strong volume confirmation."), snapshot)
    assert result["result"] == "SKIP"


def test_build_quick_insight_poor_timing_never_empty():
    insight = build_quick_insight(
        {"ticker": "MSFT"},
        {
            "TrendStrength": "weak trend",
            "EntryQuality": "poor timing",
            "YieldQuality": "moderate yield",
        },
    )
    assert insight
    assert "Timing remains weak" in insight


def test_build_quick_insight_income_case_never_empty():
    insight = build_quick_insight(
        {"ticker": "ADNOCGAS.DU"},
        {
            "TrendStrength": "weak trend",
            "EntryQuality": "acceptable entry",
            "YieldQuality": "attractive yield",
        },
    )
    assert insight
    assert "Income is attractive" in insight


def test_build_quick_insight_fallback_never_empty():
    insight = build_quick_insight({"ticker": "MSFT"}, {})
    assert insight
    assert "MSFT" in insight


def test_lint_report_blocks_price_conflict():
    snapshot = _build_snapshot()
    report = RenderedReport(
        ticker="MSFT",
        full_text="Live Price: $395.00. Current price: $401.00.",
        sections=[ReportSection("Memo", "Price moved.")],
        observed_prices=[395.0, 401.0],
    )
    result = lint_report(report, snapshot)
    assert result.safe_to_render is False
    assert any(error.startswith("PRICE_CONFLICT") for error in result.errors)


def test_lint_report_blocks_invalid_positioning():
    snapshot = _build_snapshot()
    report = RenderedReport(
        ticker="MSFT",
        full_text="Positioning Guide present.",
        sections=[ReportSection("Positioning Guide", "Entry 382 / Stop 384 / Target 405")],
        entry=382.0,
        stop=384.0,
        target=405.0,
        observed_prices=[395.0],
    )
    result = lint_report(report, snapshot)
    assert result.safe_to_render is False
    assert any(error.startswith("POSITIONING_INVALID") for error in result.errors)


def test_lint_report_warns_for_empty_section_without_blocking():
    snapshot = _build_snapshot()
    section = ReportSection("News", "")
    report = RenderedReport(
        ticker="MSFT",
        full_text="Core memo only.",
        sections=[section],
        observed_prices=[395.0],
    )
    result = lint_report(report, snapshot)
    assert result.safe_to_render is True
    assert "EMPTY_SECTION: News" in result.warnings
    assert section.suppressed is True


def test_lint_report_warns_for_duplicate_sentences_without_blocking():
    snapshot = _build_snapshot()
    report = RenderedReport(
        ticker="MSFT",
        full_text="Microsoft remains resilient. Microsoft remains resilient.",
        sections=[ReportSection("Memo", "Microsoft remains resilient.")],
        observed_prices=[395.0],
    )
    result = lint_report(report, snapshot)
    assert result.safe_to_render is True
    assert "DUPLICATE_SENTENCES_DETECTED" in result.warnings


def test_lint_report_warns_for_cached_price_label():
    raw = _base_snapshot_raw()
    raw["price"]["source"] = "cache"
    snapshot = ReportSnapshot(raw)
    snapshot.set(
        "_interpretation_context",
        {
            "value": _build_snapshot().get("_interpretation_context"),
            "source": "calculated",
            "timestamp": "2026-04-15T00:00:00",
        },
    )
    snapshot.freeze()
    report = RenderedReport(
        ticker="ADNOCGAS.DU",
        full_text="Cached price in use.",
        sections=[ReportSection("Memo", "ADNOC Gas update.")],
        observed_prices=[395.0],
    )
    result = lint_report(report, snapshot)
    assert result.safe_to_render is True
    assert "GCC_PRICE_FROM_CACHE - label required in UI" in result.warnings
