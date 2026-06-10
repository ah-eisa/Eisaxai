import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.services import market_updates as mu


def _sample_moves(regime: str) -> dict:
    templates = {
        "Bullish": {
            "SPY": {"price": 520.0, "d1_pct": 0.8, "d5_pct": 2.6, "range_low": 510.0, "range_high": 521.0, "range_mid": 515.5, "trend": "up"},
            "QQQ": {"price": 445.0, "d1_pct": 1.1, "d5_pct": 3.4, "range_low": 432.0, "range_high": 446.0, "range_mid": 439.0, "trend": "up"},
            "^VIX": {"price": 15.8, "d1_pct": -3.2, "d5_pct": -8.1, "range_low": 15.2, "range_high": 17.6, "range_mid": 16.4, "trend": "down"},
            "GLD": {"price": 219.0, "d1_pct": 0.2, "d5_pct": 0.8, "range_low": 216.5, "range_high": 220.0, "range_mid": 218.25, "trend": "up"},
            "USO": {"price": 80.0, "d1_pct": 0.4, "d5_pct": 1.2, "range_low": 78.0, "range_high": 80.5, "range_mid": 79.25, "trend": "up"},
            "BTC-USD": {"price": 71000.0, "d1_pct": 1.9, "d5_pct": 5.8, "range_low": 67200.0, "range_high": 71500.0, "range_mid": 69350.0, "trend": "up"},
            "^TNX": {"price": 4.20, "d1_pct": -0.6, "d5_pct": -1.1, "range_low": 4.16, "range_high": 4.31, "range_mid": 4.235, "trend": "down"},
            "UUP": {"price": 29.0, "d1_pct": -0.1, "d5_pct": -0.5, "range_low": 28.8, "range_high": 29.2, "range_mid": 29.0, "trend": "flat"},
            "^TASI": {"price": 12400.0, "d1_pct": 0.7, "d5_pct": 1.5, "range_low": 12220.0, "range_high": 12420.0, "range_mid": 12320.0, "trend": "up"},
            "EGX30.CA": {"price": 30100.0, "d1_pct": 0.3, "d5_pct": 0.9, "range_low": 29750.0, "range_high": 30200.0, "range_mid": 29975.0, "trend": "up"},
        },
        "Cautious": {
            "SPY": {"price": 503.0, "d1_pct": 0.1, "d5_pct": 0.4, "range_low": 499.0, "range_high": 506.0, "range_mid": 502.5, "trend": "flat"},
            "QQQ": {"price": 428.0, "d1_pct": -0.2, "d5_pct": 0.7, "range_low": 423.0, "range_high": 431.0, "range_mid": 427.0, "trend": "flat"},
            "^VIX": {"price": 21.5, "d1_pct": 1.8, "d5_pct": 2.1, "range_low": 19.9, "range_high": 22.3, "range_mid": 21.1, "trend": "flat"},
            "GLD": {"price": 224.0, "d1_pct": 0.4, "d5_pct": 1.6, "range_low": 220.5, "range_high": 224.5, "range_mid": 222.5, "trend": "up"},
            "USO": {"price": 77.0, "d1_pct": -0.3, "d5_pct": -0.8, "range_low": 76.2, "range_high": 78.4, "range_mid": 77.3, "trend": "flat"},
            "BTC-USD": {"price": 64200.0, "d1_pct": -0.6, "d5_pct": -1.9, "range_low": 63000.0, "range_high": 65700.0, "range_mid": 64350.0, "trend": "down"},
            "^TNX": {"price": 4.36, "d1_pct": 0.2, "d5_pct": 0.4, "range_low": 4.28, "range_high": 4.42, "range_mid": 4.35, "trend": "flat"},
            "UUP": {"price": 29.4, "d1_pct": 0.2, "d5_pct": 0.7, "range_low": 29.1, "range_high": 29.6, "range_mid": 29.35, "trend": "up"},
            "^TASI": {"price": 12110.0, "d1_pct": -0.2, "d5_pct": 0.1, "range_low": 12000.0, "range_high": 12220.0, "range_mid": 12110.0, "trend": "flat"},
            "EGX30.CA": {"price": 28950.0, "d1_pct": -0.1, "d5_pct": -0.3, "range_low": 28720.0, "range_high": 29100.0, "range_mid": 28910.0, "trend": "flat"},
        },
        "Bearish": {
            "SPY": {"price": 472.0, "d1_pct": -1.4, "d5_pct": -3.7, "range_low": 470.0, "range_high": 486.0, "range_mid": 478.0, "trend": "down"},
            "QQQ": {"price": 398.0, "d1_pct": -2.1, "d5_pct": -4.9, "range_low": 395.0, "range_high": 412.0, "range_mid": 403.5, "trend": "down"},
            "^VIX": {"price": 31.2, "d1_pct": 6.4, "d5_pct": 14.5, "range_low": 26.8, "range_high": 31.8, "range_mid": 29.3, "trend": "up"},
            "GLD": {"price": 229.0, "d1_pct": 0.9, "d5_pct": 2.8, "range_low": 223.0, "range_high": 229.5, "range_mid": 226.25, "trend": "up"},
            "USO": {"price": 73.0, "d1_pct": -1.0, "d5_pct": -3.3, "range_low": 72.8, "range_high": 76.5, "range_mid": 74.65, "trend": "down"},
            "BTC-USD": {"price": 58500.0, "d1_pct": -3.5, "d5_pct": -8.4, "range_low": 58000.0, "range_high": 64000.0, "range_mid": 61000.0, "trend": "down"},
            "^TNX": {"price": 4.71, "d1_pct": 0.8, "d5_pct": 1.7, "range_low": 4.45, "range_high": 4.74, "range_mid": 4.595, "trend": "up"},
            "UUP": {"price": 30.2, "d1_pct": 0.6, "d5_pct": 1.4, "range_low": 29.6, "range_high": 30.3, "range_mid": 29.95, "trend": "up"},
            "^TASI": {"price": 11620.0, "d1_pct": -0.9, "d5_pct": -2.4, "range_low": 11580.0, "range_high": 11960.0, "range_mid": 11770.0, "trend": "down"},
            "EGX30.CA": {"price": 27600.0, "d1_pct": -0.8, "d5_pct": -2.1, "range_low": 27500.0, "range_high": 28150.0, "range_mid": 27825.0, "trend": "down"},
        },
    }
    labels = mu._BENCHMARKS
    return {
        ticker: {"label": labels[ticker], **values}
        for ticker, values in templates[regime].items()
    }


def _mock_openai_daily_json() -> str:
    return json.dumps(
        {
            "regime_confidence": "High",
            "what_matters_now": [
                "Liquidity is still supporting broad risk appetite.",
                "Lower volatility is extending the risk budget.",
                "Leadership remains with quality growth."
            ],
            "key_moves": [
                {"asset": "S&P 500", "move": "+0.8% (1d)", "reason": "Momentum follow-through"},
                {"asset": "Bitcoin", "move": "+1.9% (1d)", "reason": "Risk appetite broadening"},
            ],
            "eisax_view": {"stance": "SELL", "focus": "Wrong", "horizon": "wrong"},
            "why_now": "The setup is clean and directional without macro drag.",
            "what_invalidates": ["Wrong trigger 1", "Wrong trigger 2"],
            "tactical_positioning": "Add selectively to leadership.",
            "next_triggers": ["Earnings", "Rates", "VIX"],
        }
    )


def _mock_openai_weekly_json() -> str:
    return json.dumps(
        {
            "market_summary": "Markets traded with mixed conviction as macro and leadership signals diverged.",
            "positioning": "Keep gross exposure balanced and upgrade only confirmed quality.",
            "asset_allocation_view": {"equities": "Wrong"},
            "regional_view": {"US": "Measured", "GCC": "Watch oil", "Egypt": "Watch dollar"},
            "winners_losers": {"winners": ["Gold +1.0%"], "losers": ["Oil -0.8%"]},
            "highest_conviction_opportunity": "Own gold against policy uncertainty over a tactical horizon.",
            "key_risks": ["Rates stay restrictive", "Oil shock", "Guidance cuts"],
            "what_changes_this_view": ["Wrong allocation trigger"],
            "portfolio_angle": "Keep dry powder and avoid adding to weak cyclicals.",
            "eisax_verdict": "Hold core risk; add only on confirmation.",
        }
    )


def _prepare(monkeypatch, moves: dict, fg: dict, timestamp: str = "2026-04-20T06:30:00+00:00") -> None:
    monkeypatch.setattr(mu, "_collect_market_data", lambda lookback_days=10: moves)
    monkeypatch.setattr(mu, "_get_fear_greed", lambda: fg)
    monkeypatch.setattr(mu, "_get_recent_sentiment_summary", lambda: {})
    monkeypatch.setattr(mu, "_save_update", lambda update_type, data: 1)
    monkeypatch.setattr(mu, "_get_market_data_timestamp", lambda: timestamp)
    monkeypatch.setattr(mu, "_call_openai_text", lambda *args, **kwargs: None)


def test_daily_bullish_stance_is_locked(monkeypatch):
    moves = _sample_moves("Bullish")
    fg = {"score": 62, "rating": "Greed"}
    _prepare(monkeypatch, moves, fg)
    monkeypatch.setattr(mu, "_call_openai", lambda *args, **kwargs: _mock_openai_daily_json())

    update = mu.generate_daily_update()

    assert update["market_regime"] == "Bullish"
    assert update["eisax_view"] == mu.build_eisax_stance(moves, "Bullish", fg)
    assert update["what_invalidates"] == mu.build_invalidation_logic(moves, "Bullish")
    assert {"data_timestamp", "web_version", "linkedin_text", "full_report", "cross_asset_snapshot"} <= set(update)
    assert 90 <= len(update["linkedin_text"].split()) <= 140
    assert "Cross-asset:" in update["web_version"]
    assert "Invalidation:" in update["web_version"]
    assert "EisaX View:" in update["linkedin_text"]
    assert "Risk:" in update["linkedin_text"]
    assert "Positioning:" in update["linkedin_text"]
    assert "Today’s EisaX market pulse" not in update["linkedin_text"]
    assert "broad risk appetite" not in update["linkedin_text"].lower()
    assert "vix decline supports risk-on" not in update["linkedin_text"].lower()


def test_weekly_cautious_allocation_is_locked(monkeypatch):
    moves = _sample_moves("Cautious")
    fg = {"score": 48, "rating": "Neutral"}
    _prepare(monkeypatch, moves, fg)
    monkeypatch.setattr(mu, "_call_openai", lambda *args, **kwargs: _mock_openai_weekly_json())

    update = mu.generate_weekly_update()

    assert update["asset_allocation_view"] == mu._build_asset_allocation_view("Cautious")
    assert update["what_changes_this_view"] == mu.build_invalidation_logic(moves, "Cautious")
    assert {"data_timestamp", "web_version", "linkedin_text", "full_report"} <= set(update)
    assert 90 <= len(update["linkedin_text"].split()) <= 140
    assert "Allocation:" in update["web_version"]
    assert "Invalidation:" in update["web_version"]
    assert "EisaX View:" in update["linkedin_text"]
    assert "Risk:" in update["linkedin_text"]
    assert "Positioning:" in update["linkedin_text"]
    assert "broad risk appetite" not in update["linkedin_text"].lower()


def test_daily_bearish_falls_back_deterministically(monkeypatch):
    moves = _sample_moves("Bearish")
    fg = {"score": 24, "rating": "Fear"}
    _prepare(monkeypatch, moves, fg)
    monkeypatch.setattr(mu, "_call_openai", lambda *args, **kwargs: None)

    update = mu.generate_daily_update()

    assert update["market_regime"] == "Bearish"
    assert update["eisax_view"] == mu.build_eisax_stance(moves, "Bearish", fg)
    assert update["data_timestamp"] == "2026-04-20T06:30:00+00:00"
    assert update["web_version"]
    assert update["full_report"]
    assert "Cross-Asset Snapshot:" in update["full_report"]
    assert "## Market-by-Market View" in update["full_report"]
    assert "Primary trigger:" in update["full_report"]
    assert "Portfolio Translation" in update["full_report"]
    assert "### US Equities" in update["full_report"]
    assert "### GCC" in update["full_report"]


def test_cross_asset_snapshot_uses_market_closed_labels():
    snapshot = mu._build_cross_asset_snapshot({})
    assert snapshot["us_equities"]["price"] == "Market Closed"
    assert snapshot["volatility"]["d1_pct"] == "Market Closed"


def test_weekly_sparse_ai_is_backfilled_consistently(monkeypatch):
    moves = _sample_moves("Bullish")
    fg = {"score": 64, "rating": "Greed"}
    _prepare(monkeypatch, moves, fg)
    monkeypatch.setattr(
        mu,
        "_call_openai",
        lambda *args, **kwargs: json.dumps(
            {
                "market_summary": "Risk appetite held up across equities and crypto.",
                "positioning": "",
                "highest_conviction_opportunity": "",
                "eisax_verdict": "Add selectively to confirmed strength.",
                "regional_view": {"US": "Stay with quality growth."},
                "key_risks": ["Yields reprice higher"],
            }
        ),
    )

    update = mu.generate_weekly_update()

    assert update["asset_allocation_view"] == mu._build_asset_allocation_view("Bullish")
    assert update["regional_view"]["GCC"]
    assert update["regional_view"]["Egypt"]
    assert update["cross_asset_snapshot"]["us_equities"]["label"] == "S&P 500"
    assert len(update["key_risks"]) == 3
    assert update["portfolio_angle"]
    assert update["highest_conviction_opportunity"]
    assert "Allocation View:" in update["full_report"]
    assert "Why Now:" in update["full_report"]
    assert "Markets remain constructive but not crowded." in update["full_report"]
    assert "Primary trigger:" in update["full_report"]
    assert "Portfolio Translation" in update["full_report"]
    assert "## Market-by-Market View" in update["full_report"]
    assert "### Egypt" in update["full_report"]
    assert "### Oil" in update["full_report"]
    assert "Oil weakness argues against reading the equity move as broad global growth." not in update["full_report"]
    assert "Regional View — GCC:" not in update["full_report"]
    assert "Regional View — Egypt:" not in update["full_report"]
    assert 90 <= len(update["linkedin_text"].split()) <= 140
