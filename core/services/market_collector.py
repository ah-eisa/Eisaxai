"""
core/services/market_updates.py
────────────────────────────────
EisaX Market Updates — institutional-grade daily pulse + weekly strategy brief.

Public API (unchanged signatures):
    generate_daily_update()  -> dict
    generate_weekly_update() -> dict
    get_latest_updates()     -> dict
    format_for_linkedin(update_json: dict) -> str

New helpers (internal):
    build_eisax_stance(moves, regime, fg)   -> dict
    build_invalidation_logic(moves, regime) -> list
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path("/home/ubuntu/investwise/data/market_updates.db")
_MARKET_SNAPSHOT_PATH = Path("/home/ubuntu/investwise/data/market_updates_snapshot.json")
_MARKET_CACHE_TTL = timedelta(minutes=15)
_MARKET_CACHE: dict[str, Any] = {
    "lookback_days": None,
    "fetched_at": None,
    "data_timestamp": None,
    "data": None,
}
_LAST_MARKET_DATA_TIMESTAMP: Optional[str] = None

# ── Benchmarks ────────────────────────────────────────────────────────────────

_BENCHMARKS = {
    "SPY":      "S&P 500",
    "QQQ":      "Nasdaq 100",
    "^VIX":     "VIX",
    "GLD":      "Gold",
    "SLV":      "Silver",
    "USO":      "Oil (WTI)",
    "BTC-USD":  "Bitcoin",
    "^TNX":     "10Y Treasury Yield",
    "UUP":      "US Dollar (DXY)",
    "^TASI":    "Saudi Market Composite",
    "^DFMGI":   "UAE Market Composite",
    "EGX30.CA": "Egypt Market Composite",
}

_PIPELINE_REGIONAL_BENCHMARKS = {
    "^TASI": {"market": "ksa", "label": "Saudi Market Composite"},
    "^DFMGI": {"market": "uae", "label": "UAE Market Composite"},
    "EGX30.CA": {"market": "egypt", "label": "Egypt Market Composite"},
}

_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
_DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")


# ── Storage ───────────────────────────────────────────────────────────────────

from core.services.market_db import _init_db, _save_update, _get_latest, _get_cached_market_data, _set_market_cache, _persist_last_good_snapshot, _load_last_good_snapshot, _utc_now_iso

def _weighted_average(values, weights) -> Optional[float]:
    try:
        import pandas as pd
        vals = pd.to_numeric(values, errors="coerce")
        wts = pd.to_numeric(weights, errors="coerce").fillna(0)
        valid = vals.notna()
        vals = vals[valid]
        wts = wts[valid].clip(lower=0)
        if vals.empty:
            return None
        if float(wts.sum()) > 0:
            return float((vals * wts).sum() / wts.sum())
        return float(vals.mean())
    except Exception:
        return None

def _load_pipeline_regional_moves() -> tuple[dict, Optional[str]]:
    try:
        import sys
        import pandas as pd

        root = "/home/ubuntu/investwise"
        if root not in sys.path:
            sys.path.insert(0, root)

        from pipeline import cache, fetcher
    except Exception as exc:
        logger.warning("[market_updates] Regional pipeline unavailable: %s", exc)
        return {}, None

    regional_moves: dict[str, dict] = {}
    timestamps: list[str] = []

    for ticker, cfg in _PIPELINE_REGIONAL_BENCHMARKS.items():
        market_code = cfg["market"]
        try:
            if cache.is_stale(market_code):
                fetcher.fetch_market(market_code)
            df, ts = cache.get_latest(market_code)
            if ts:
                timestamps.append(ts)
            if df is None or df.empty:
                continue

            weights = pd.to_numeric(df.get("market_cap_basic"), errors="coerce")
            closes = pd.to_numeric(df.get("close"), errors="coerce")
            d1 = pd.to_numeric(df.get("change"), errors="coerce")

            d5 = None
            for col in ("Perf.5D", "change|1W", "Perf.W"):
                if col in df.columns:
                    d5 = pd.to_numeric(df.get(col), errors="coerce")
                    break

            price = _weighted_average(closes, weights)
            d1_pct = _weighted_average(d1, weights)
            d5_pct = _weighted_average(d5, weights) if d5 is not None else None

            if d1_pct is None and d5_pct is None:
                continue

            if d5_pct is None:
                d5_pct = d1_pct

            if price is None:
                price = float(closes.dropna().mean()) if closes.notna().any() else None

            trend = "up" if d5_pct > 1 else ("down" if d5_pct < -1 else "flat")
            regional_moves[ticker] = {
                "label": cfg["label"],
                "price": round(price, 4) if isinstance(price, (int, float)) else None,
                "d1_pct": round(d1_pct, 2) if isinstance(d1_pct, (int, float)) else None,
                "d5_pct": round(d5_pct, 2) if isinstance(d5_pct, (int, float)) else None,
                "range_high": round(price, 4) if isinstance(price, (int, float)) else None,
                "range_low": round(price, 4) if isinstance(price, (int, float)) else None,
                "range_mid": round(price, 4) if isinstance(price, (int, float)) else None,
                "trend": trend,
            }
        except Exception as exc:
            logger.warning("[market_updates] Regional composite failed for %s: %s", market_code, exc)

    latest_ts = max(timestamps) if timestamps else None
    return regional_moves, latest_ts

def _collect_market_data(lookback_days: int = 10) -> dict:
    """Fetch price data for all benchmarks. Returns structured market snapshot."""
    import yfinance as yf

    cached = _get_cached_market_data(lookback_days)
    if cached:
        return cached

    tickers = [t for t in _BENCHMARKS.keys() if t not in _PIPELINE_REGIONAL_BENCHMARKS]
    try:
        raw = yf.download(
            tickers,
            period=f"{lookback_days}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    except Exception as exc:
        logger.error("[market_updates] yfinance download failed: %s", exc)
        fallback = _load_last_good_snapshot(lookback_days)
        if fallback:
            logger.warning("[market_updates] Using last good snapshot after yfinance failure")
            return fallback
        return {}

    moves = {}
    data_timestamp = _utc_now_iso()
    try:
        if len(close.index) > 0:
            latest_idx = close.index[-1]
            if hasattr(latest_idx, "to_pydatetime"):
                dt = latest_idx.to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                data_timestamp = dt.astimezone(timezone.utc).isoformat()
    except Exception:
        pass

    for ticker in tickers:
        try:
            series = close[ticker].dropna()
            if len(series) < 2:
                continue
            latest = float(series.iloc[-1])
            prev   = float(series.iloc[-2])
            week   = float(series.iloc[max(-6, -len(series))])
            recent = series.tail(min(5, len(series)))
            range_high = float(recent.max())
            range_low = float(recent.min())
            range_mid = (range_high + range_low) / 2
            d1_pct = (latest - prev) / prev * 100
            d5_pct = (latest - week) / week * 100
            trend = "up" if d5_pct > 1 else ("down" if d5_pct < -1 else "flat")
            moves[ticker] = {
                "label":  _BENCHMARKS[ticker],
                "price":  round(latest, 4),
                "d1_pct": round(d1_pct, 2),
                "d5_pct": round(d5_pct, 2),
                "range_high": round(range_high, 4),
                "range_low": round(range_low, 4),
                "range_mid": round(range_mid, 4),
                "trend": trend,
            }
        except Exception:
            continue

    regional_moves, regional_ts = _load_pipeline_regional_moves()
    if regional_moves:
        moves.update(regional_moves)
        if regional_ts:
            data_timestamp = regional_ts

    if moves:
        _set_market_cache(lookback_days, moves, data_timestamp)
        try:
            _persist_last_good_snapshot(lookback_days, moves, data_timestamp)
        except Exception as exc:
            logger.warning("[market_updates] Failed to persist last good snapshot: %s", exc)
        return moves

    fallback = _load_last_good_snapshot(lookback_days)
    if fallback:
        logger.warning("[market_updates] Empty market data set — using last good snapshot")
        return fallback
    return moves

def _get_fear_greed() -> dict:
    try:
        from core.rapid_data import get_fear_greed
        result = get_fear_greed() or {}
        return result
    except Exception:
        return {"score": 50, "rating": "Neutral"}

def _get_recent_sentiment_summary() -> dict:
    try:
        import sqlite3 as _sq
        db = Path("/home/ubuntu/investwise/data/sentiment.db")
        if not db.exists():
            return {}
        with _sq.connect(db) as con:
            rows = con.execute(
                """SELECT label, COUNT(*) as cnt
                   FROM sentiment_cache
                   WHERE analyzed_at > datetime('now','-24 hours')
                   GROUP BY label""",
            ).fetchall()
        counts = {r[0].lower(): r[1] for r in rows}
        total   = sum(counts.values()) or 1
        bullish = counts.get("bullish", 0)
        bearish = counts.get("bearish", 0)
        neutral = counts.get("neutral", 0)
        return {
            "bullish":   bullish,
            "bearish":   bearish,
            "neutral":   neutral,
            "net_score": round((bullish - bearish) / total * 100, 1),
        }
    except Exception:
        return {}

def _determine_regime(moves: dict) -> str:
    vix  = (moves.get("^VIX") or {}).get("price", 20)
    spy5 = (moves.get("SPY")  or {}).get("d5_pct", 0)
    if vix >= 28 or (vix >= 22 and spy5 < -2):
        return "Bearish"
    if vix >= 20 or abs(spy5) < 1:
        return "Cautious"
    if vix < 18 and spy5 > 1:
        return "Bullish"
    if spy5 > 2:
        return "Bullish"
    return "Cautious"

def _determine_regime_confidence(moves: dict, regime: str) -> str:
    vix  = (moves.get("^VIX")    or {}).get("price", 20)
    spy5 = (moves.get("SPY")     or {}).get("d5_pct", 0)
    btc5 = (moves.get("BTC-USD") or {}).get("d5_pct", 0)
    if regime == "Bearish":
        if vix > 30 and spy5 < -3:
            return "High"
        if vix > 25 or spy5 < -2:
            return "Medium"
        return "Low"
    if regime == "Bullish":
        if vix < 16 and spy5 > 2 and btc5 > 0:
            return "High"
        if vix < 18 or spy5 > 1.5:
            return "Medium"
        return "Low"
    # Cautious
    if abs(spy5) < 0.5 and 19 <= vix <= 24:
        return "Medium"
    return "Low"

def build_eisax_stance(moves: dict, regime: str, fg: dict) -> dict:
    """
    Derive institutional stance, asset allocation direction, and horizon.
    Returns the structured eisax_view dict.
    """
    vix      = (moves.get("^VIX")    or {}).get("price", 20)
    btc5     = (moves.get("BTC-USD") or {}).get("d5_pct", 0)
    gld5     = (moves.get("GLD")     or {}).get("d5_pct", 0)
    fg_score = fg.get("score", 50)

    if regime == "Bearish" or vix > 28:
        return {
            "stance":             "REDUCE RISK",
            "overweight_assets":  ["Gold", "Cash", "Short Duration Bonds"],
            "underweight_assets": ["Equities", "Crypto", "High Beta"],
            "neutral_assets":     ["Commodities", "Defensive Sectors"],
            "focus":              "Capital Preservation",
            "horizon":            "defensive",
        }
    if regime == "Bullish" and fg_score >= 75:
        return {
            "stance":             "HOLD",
            "overweight_assets":  ["Gold", "Defensive Equities"],
            "underweight_assets": ["Speculative Positions", "High Leverage"],
            "neutral_assets":     ["US Equities", "Crypto"],
            "focus":              "Profit-taking on extended names",
            "horizon":            "tactical",
        }
    if regime == "Bullish":
        ow = ["US Equities", "Quality Tech"]
        if btc5 > 3:
            ow.append("Crypto")
        return {
            "stance":             "Tactical BUY",
            "overweight_assets":  ow,
            "underweight_assets": ["Cash", "Long Duration Bonds"],
            "neutral_assets":     ["Gold", "Oil"],
            "focus":              "US Equities / Quality Growth",
            "horizon":            "swing",
        }
    # Cautious
    return {
        "stance":             "HOLD",
        "overweight_assets":  ["Gold"] if gld5 > 0 else ["Cash"],
        "underweight_assets": ["High Beta", "Speculative Crypto"],
        "neutral_assets":     ["US Equities", "Oil"],
        "focus":              "Selective Quality only",
        "horizon":            "tactical",
    }

def _build_asset_allocation_view(regime: str) -> dict:
    if regime == "Bearish":
        return {
            "equities": "Underweight",
            "crypto": "Underweight",
            "metals": "Overweight",
            "commodities": "Neutral",
            "cash": "Overweight",
        }
    if regime == "Bullish":
        return {
            "equities": "Overweight",
            "crypto": "Neutral",
            "metals": "Neutral",
            "commodities": "Neutral",
            "cash": "Underweight",
        }
    return {
        "equities": "Neutral",
        "crypto": "Underweight",
        "metals": "Overweight",
        "commodities": "Neutral",
        "cash": "Neutral",
    }

def build_invalidation_logic(moves: dict, regime: str) -> list:
    """Generate context-aware invalidation triggers using recent range and trend."""
    spy_ctx = moves.get("SPY") or {}
    vix_ctx = moves.get("^VIX") or {}
    tnx_ctx = moves.get("^TNX") or {}

    spy = float(spy_ctx.get("price", 500))
    spy_low = float(spy_ctx.get("range_low", spy * 0.98))
    spy_high = float(spy_ctx.get("range_high", spy * 1.02))
    spy_mid = float(spy_ctx.get("range_mid", (spy_high + spy_low) / 2))
    spy_trend = spy_ctx.get("trend", "flat")

    vix = float(vix_ctx.get("price", 20))
    vix_low = float(vix_ctx.get("range_low", max(vix - 2, 12)))
    vix_high = float(vix_ctx.get("range_high", vix + 3))

    tnx = float(tnx_ctx.get("price", 4.25))
    tnx_low = float(tnx_ctx.get("range_low", max(tnx - 0.20, 0.0)))
    tnx_high = float(tnx_ctx.get("range_high", tnx + 0.20))

    if regime == "Bearish":
        price_level = max(spy_high, spy_mid)
        return [
            f"SPY reclaims ${price_level:.0f} and closes back above the recent upper range — downside regime loses control",
            f"VIX closes below {vix_low:.1f} and breaks its recent floor — hedging demand is normalizing",
            f"10Y yield slips back below {tnx_low:.2f}% — macro pressure eases enough to re-open risk appetite",
        ]
    if regime == "Bullish":
        price_level = spy_low if spy_trend == "up" else min(spy_low, spy_mid)
        return [
            f"SPY loses ${price_level:.0f} and breaks the recent lower range — trend confirmation fails",
            f"VIX closes above {vix_high:.1f} and exceeds the recent stress band — risk premium is repricing higher",
            f"10Y yield breaks above {tnx_high:.2f}% — valuation pressure broadens across equities",
        ]

    return [
        f"SPY resolves outside the ${spy_low:.0f}-${spy_high:.0f} recent range — wait for the break to confirm direction",
        f"VIX closes above {vix_high:.1f} or below {vix_low:.1f} — the volatility regime stops being neutral",
        f"10Y yield moves outside the {tnx_low:.2f}%–{tnx_high:.2f}% recent band — macro conditions reprice materially",
    ]

def _build_cross_asset_snapshot(moves: dict) -> dict:
    """Compute directional cross-asset snapshot — pure data, no AI required."""
    def _closed_or(value: Any) -> Any:
        return "Market Closed" if value is None else value

    def _entry(ticker: str, label: str) -> dict:
        m = moves.get(ticker, {})
        if not m:
            return {
                "label": label,
                "direction": "Market Closed",
                "d1_pct": "Market Closed",
                "d5_pct": "Market Closed",
                "price": "Market Closed",
            }
        d1    = m.get("d1_pct", 0)
        arrow = "↑" if d1 > 0.25 else ("↓" if d1 < -0.25 else "→")
        return {
            "label":     label,
            "direction": _closed_or(arrow),
            "d1_pct":    _closed_or(m.get("d1_pct")),
            "d5_pct":    _closed_or(m.get("d5_pct")),
            "price":     _closed_or(m.get("price")),
        }

    return {
        "us_equities": _entry("SPY",      "S&P 500"),
        "volatility":  _entry("^VIX",     "VIX"),
        "crypto":      _entry("BTC-USD",  "Bitcoin"),
        "metals":      _entry("GLD",      "Gold"),
        "commodities": _entry("USO",      "Oil (WTI)"),
        "rates":       _entry("^TNX",     "10Y Treasury"),
        "gcc":         _entry("^TASI",    "Saudi Market Composite"),
        "egypt":       _entry("EGX30.CA", "Egypt Market Composite"),
    }

def _call_openai(prompt: str, max_tokens: int = 900) -> Optional[str]:
    """OpenAI call with forced JSON output — for structured field generation."""
    if not _OPENAI_KEY:
        return None
    try:
        import requests
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {_OPENAI_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=25,
        )
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("[market_updates] OpenAI call failed: %s", exc)
        return None

def _call_openai_text(prompt: str, max_tokens: int = 1500) -> Optional[str]:
    """OpenAI call for free-form text (no JSON constraint) — for narrative reports."""
    if not _OPENAI_KEY:
        return None
    try:
        import requests
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {_OPENAI_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.25,
                "max_tokens": max_tokens,
            },
            timeout=35,
        )
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("[market_updates] OpenAI text call failed: %s", exc)
        return None

def _call_gemini(prompt: str) -> Optional[str]:
    """DeepSeek (primary) → OpenAI GPT-4.1-nano (fallback)."""
    import httpx as _hx, os as _os
    _payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1200,
        "temperature": 0.5,
    }

    # Primary: DeepSeek
    _ds_key = _DEEPSEEK_KEY or _os.getenv("DEEPSEEK_API_KEY", "")
    if _ds_key:
        try:
            _r = _hx.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {_ds_key}", "Content-Type": "application/json"},
                json={"model": _os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"), **_payload},
                timeout=30.0,
            )
            _r.raise_for_status()
            _text = (_r.json()["choices"][0]["message"]["content"] or "").strip()
            if _text:
                return _text
        except Exception as exc:
            logger.warning("[market_updates] DeepSeek failed: %s — trying OpenAI", exc)

    # Fallback: OpenAI GPT-4.1-nano
    _oai_key = _os.getenv("OPENAI_API_KEY", "")
    _oai_model = _os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4.1-nano")
    if _oai_key:
        try:
            _r = _hx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {_oai_key}", "Content-Type": "application/json"},
                json={"model": _oai_model, **_payload},
                timeout=30.0,
            )
            _r.raise_for_status()
            return (_r.json()["choices"][0]["message"]["content"] or "").strip() or None
        except Exception as exc:
            logger.warning("[market_updates] OpenAI fallback failed: %s", exc)

    return None

def _generate_insight(prompt: str, max_tokens: int = 900) -> Optional[dict]:
    raw = _call_openai(prompt, max_tokens)
    if not raw:
        return None
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception:
        return None

