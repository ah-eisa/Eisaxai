from __future__ import annotations

import pytest

from core.services.pilot_report_json import (
    build_pilot_report_json,
    derive_conviction_level,
    normalize_scenarios,
)


def _sample_report(symbol: str, recommendation: str, score: int, risk_name: str) -> str:
    return f"""## ⚡ Quick View — {symbol}

**{symbol} | {recommendation} | Conviction: Low | EisaX Score: {score}/100**

---
## 📋 Full Report

### 1. Executive Summary
The market still respects the underlying asset quality, but timing remains selective and position discipline matters.

### 4. Key Risks
- **{risk_name} (Severity: High):** A high-severity risk remains active in the current setup.
- **Execution Drift (Severity: Medium-High):** Execution sensitivity remains elevated if momentum weakens.

### 5. Analyst Consensus & Catalysts
Analyst consensus remains constructive.

**Valuation Scenarios (Probability-Weighted):**
| Scenario | Probability | Multiple | Implied Price | vs Current |
|----------|-------------|----------|---------------|------------|
| Bear | 20% | 12.0x | 170 | -15.7% |
| Base | 45% | 17.9x | 210 | +4.1% |
| Bull | 25% | 25.1x | 240 | +19.0% |
| Macro Shock | 10% | -- | 145 | -28.1% |
*Expected Value: +7.0%*
"""


@pytest.mark.parametrize(
    ("symbol", "market", "recommendation", "sector"),
    [
        ("NVDA", "USA", "HOLD", "Technology"),
        ("2222.SR", "SAU", "BUY", "Energy"),
        ("BTC", "CRYPTO", "HOLD", "Crypto Assets"),
    ],
)
def test_build_pilot_report_json_required_fields(symbol: str, market: str, recommendation: str, sector: str):
    report_json = build_pilot_report_json(
        symbol=symbol,
        market=market,
        language="en",
        report_text=_sample_report(symbol, recommendation, 82, "Primary Risk"),
        analysis_data={
            "analytics": {
                "price": 201.68,
                "rsi": 61.4,
                "adx": 24.2,
                "macd": 1.2,
                "macd_signal": 0.8,
                "trend": "Bullish",
                "momentum": "Bullish",
                "sma_50": 190.0,
                "sma_200": 175.0,
            },
            "fundamentals": {
                "company_name": "Sample Asset",
                "sector": sector,
                "market_cap": 4900000000000,
                "volume_today": 32500000,
                "revenue_growth": 73.2,
                "gross_margin": 71.1,
                "roe": 101.5,
                "forward_pe": 17.9,
                "analyst_target": 268.61,
                "week52_high": 240.0,
                "week52_low": 165.0,
                "beta": 2.33 if market != "CRYPTO" else 1.8,
            },
            "trust_layer": {"classification": "SAFE"},
        },
        system_version="v1.0",
        latency_seconds=60,
    )

    assert set(
        [
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
        ]
    ).issubset(report_json.keys())
    assert report_json["headline_view"]["recommendation"] in {"BUY", "HOLD", "REDUCE", "SELL"}
    assert report_json["headline_view"]["conviction_level"] == derive_conviction_level(
        report_json["headline_view"]["conviction_score"]
    )
    assert len(report_json["decision_framework"]["why_this_decision"]) >= 2
    assert len(report_json["risk_map"]) >= 1
    assert report_json["asset"]["market"] == market


def test_normalize_scenarios_computes_probabilities():
    scenarios, validation = normalize_scenarios(
        [
            {"scenario": "Bear", "weight": 20},
            {"scenario": "Base", "weight": 45},
            {"scenario": "Bull", "weight": 25},
            {"scenario": "Macro Shock", "weight": 10},
        ]
    )
    assert [scenario["weight"] for scenario in scenarios] == [20, 45, 25, 10]
    assert validation["normalized_probabilities"] == [0.2, 0.45, 0.25, 0.1]
    assert abs(sum(validation["normalized_probabilities"]) - 1.0) < 0.001


def test_normalize_scenarios_rejects_invalid_weights():
    with pytest.raises(ValueError):
        normalize_scenarios(
            [
                {"scenario": "Bear", "weight": 0},
                {"scenario": "Base", "weight": 10},
            ]
        )


def test_optional_sections_are_omitted_when_unavailable():
    report_json = build_pilot_report_json(
        symbol="NVDA",
        market="USA",
        language="en",
        report_text="""## ⚡ Quick View — NVDA

**NVDA | HOLD | Conviction: Low | EisaX Score: 70/100**

### 1. Executive Summary
Timing remains mixed.

### 4. Key Risks
- **Trend Risk (Severity: Medium):** Trend risk remains visible.
""",
        analysis_data={
            "analytics": {"adx": 18.0, "rsi": 48.0, "trend": "Neutral", "momentum": "Neutral"},
            "fundamentals": {"company_name": "NVIDIA", "sector": "Technology", "beta": 1.9},
            "trust_layer": {"classification": "PARTIAL"},
        },
        system_version="v1.0",
    )

    assert "scenario_analysis" not in report_json
    assert "validation" not in report_json
    assert "market_snapshot" not in report_json


def test_medium_high_risk_is_normalized_to_allowed_enum():
    report_json = build_pilot_report_json(
        symbol="NVDA",
        market="USA",
        language="en",
        report_text=_sample_report("NVDA", "HOLD", 80, "Primary Risk"),
        analysis_data={
            "analytics": {"adx": 21.0, "rsi": 55.0, "trend": "Bullish", "momentum": "Bullish"},
            "fundamentals": {"company_name": "NVIDIA", "sector": "Technology", "beta": 2.0},
            "trust_layer": {"classification": "SAFE"},
        },
        system_version="v1.0",
    )
    severities = {item["risk"]: item["severity"] for item in report_json["risk_map"]}
    assert severities["Execution Drift"] == "high"


def test_conviction_level_mapping():
    assert derive_conviction_level(0) == "low"
    assert derive_conviction_level(39) == "low"
    assert derive_conviction_level(40) == "medium"
    assert derive_conviction_level(69) == "medium"
    assert derive_conviction_level(70) == "high"
