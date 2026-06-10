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

from core.services.market_report import _clean_text, _clean_text_list, _as_number, _snapshot_brief
from core.services.market_collector import _weighted_average

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


# ── Storage ───────────────────────────────────────────────────────────────────


def _load_pipeline_market_frame(market_code: str):
    try:
        import sys
        root = "/home/ubuntu/investwise"
        if root not in sys.path:
            sys.path.insert(0, root)
        from pipeline import cache, fetcher
        if cache.is_stale(market_code):
            fetcher.fetch_market(market_code)
        return cache.get_latest(market_code)
    except Exception as exc:
        logger.warning("[market_updates] Could not load pipeline frame for %s: %s", market_code, exc)
        return None, None

def _market_top_sector(df, direction: str = "up") -> Optional[str]:
    try:
        import pandas as pd
        if df is None or df.empty or "sector" not in df.columns or "change" not in df.columns:
            return None
        work = df.dropna(subset=["sector"]).copy()
        if work.empty:
            return None
        scores = []
        for sector, grp in work.groupby("sector"):
            avg = _weighted_average(pd.to_numeric(grp.get("change"), errors="coerce"), pd.to_numeric(grp.get("market_cap_basic"), errors="coerce"))
            if avg is not None:
                scores.append((sector, avg))
        if not scores:
            return None
        scores = sorted(scores, key=lambda x: x[1], reverse=(direction == "up"))
        return str(scores[0][0])
    except Exception:
        return None

def _market_top_movers(df, direction: str = "up", limit: int = 2) -> list[str]:
    try:
        import pandas as pd
        if df is None or df.empty or "change" not in df.columns:
            return []
        work = df.copy()
        work["change"] = pd.to_numeric(work.get("change"), errors="coerce")
        work = work.dropna(subset=["change"])
        if work.empty:
            return []
        picked = work.nlargest(limit, "change") if direction == "up" else work.nsmallest(limit, "change")
        out = []
        for _, row in picked.iterrows():
            raw_name = row.get("name") or row.get("ticker") or "Unknown"
            name = str(raw_name).split(":")[-1].strip()
            out.append(f"{name} {float(row['change']):+.1f}%")
        return out
    except Exception:
        return []

def _build_daily_regional_internals() -> dict:
    try:
        import pandas as pd
    except Exception:
        return {}

    config = {
        "ksa": "Saudi Arabia",
        "uae": "UAE",
        "egypt": "Egypt",
    }
    out: dict[str, dict] = {}
    for market_code, label in config.items():
        df, ts = _load_pipeline_market_frame(market_code)
        if df is None or df.empty:
            continue
        work = df.copy()
        work["change"] = pd.to_numeric(work.get("change"), errors="coerce")
        work["market_cap_basic"] = pd.to_numeric(work.get("market_cap_basic"), errors="coerce")
        change = work["change"].dropna()
        adv = int((change > 0).sum())
        dec = int((change < 0).sum())
        flat = int((change == 0).sum())
        weighted = _weighted_average(work.get("change"), work.get("market_cap_basic"))
        out[market_code] = {
            "label": label,
            "timestamp": ts,
            "advancers": adv,
            "decliners": dec,
            "unchanged": flat,
            "weighted_change": round(weighted, 2) if isinstance(weighted, (int, float)) else None,
            "leading_sector": _market_top_sector(work, "up"),
            "lagging_sector": _market_top_sector(work, "down"),
            "top_gainers": _market_top_movers(work, "up", 2),
            "top_losers": _market_top_movers(work, "down", 2),
        }
    return out

def _load_pipeline_snapshot_series(market_code: str, limit: int = 4) -> list[tuple[str, Any]]:
    try:
        import sys
        from pathlib import Path
        import pandas as pd

        root = "/home/ubuntu/investwise"
        if root not in sys.path:
            sys.path.insert(0, root)
        from pipeline import cache, CACHE_DIR

        entries = (cache.get_snapshots(market_code) or [])[-limit:]
        series = []
        for entry in entries:
            filename = entry.get("filename")
            ts = entry.get("timestamp")
            if not filename or not ts:
                continue
            path = Path(CACHE_DIR) / filename
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            series.append((ts, df))
        return series
    except Exception as exc:
        logger.warning("[market_updates] Could not load snapshot series for %s: %s", market_code, exc)
        return []

def _dfm_real_estate_weighted_change(df) -> Optional[float]:
    try:
        import pandas as pd
        if df is None or df.empty:
            return None
        work = df.copy()
        tickers = work.get("ticker", pd.Series("", index=work.index)).fillna("").astype(str).str.lower()
        names = work.get("name", pd.Series("", index=work.index)).fillna("").astype(str).str.lower()
        dfm_mask = tickers.str.startswith("dfm:")
        kw_pattern = "|".join(_DFM_REAL_ESTATE_KEYWORDS)
        re_mask = names.str.contains(kw_pattern, na=False) | tickers.str.contains(kw_pattern, na=False)
        subset = work[dfm_mask & re_mask].copy()
        if subset.empty:
            return None
        return _weighted_average(
            pd.to_numeric(subset.get("change"), errors="coerce"),
            pd.to_numeric(subset.get("market_cap_basic"), errors="coerce"),
        )
    except Exception:
        return None

def _commodities_wti_change(df) -> Optional[float]:
    try:
        import pandas as pd
        if df is None or df.empty:
            return None
        work = df.copy()
        tickers = work.get("ticker", pd.Series("", index=work.index)).fillna("").astype(str)
        subset = work[tickers == "CL=F"]
        if subset.empty:
            subset = work[tickers.str.contains("CL=F|Crude Oil", na=False)]
        if subset.empty:
            return None
        row = subset.iloc[0]
        val = pd.to_numeric(pd.Series([row.get("change")]), errors="coerce").iloc[0]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None

def _compute_gcc_decoupling_signal() -> dict:
    try:
        import pandas as pd
    except Exception:
        return {}

    uae_series = _load_pipeline_snapshot_series("uae", limit=4)
    cmd_series = _load_pipeline_snapshot_series("commodities", limit=4)
    if not uae_series or not cmd_series:
        return {}

    pairs = []
    for (uae_ts, uae_df), (_, cmd_df) in zip(uae_series[-4:], cmd_series[-4:]):
        re_change = _dfm_real_estate_weighted_change(uae_df)
        wti_change = _commodities_wti_change(cmd_df)
        if re_change is None or wti_change is None:
            continue
        pairs.append((uae_ts, float(re_change), float(wti_change)))

    if len(pairs) < 3:
        return {}

    re_series = pd.Series([p[1] for p in pairs], dtype="float64")
    oil_series = pd.Series([p[2] for p in pairs], dtype="float64")
    latest_dfm = round(float(re_series.iloc[-1]), 2)
    latest_wti = round(float(oil_series.iloc[-1]), 2)
    corr = None
    if float(re_series.std()) > 0 and float(oil_series.std()) > 0:
        corr = re_series.corr(oil_series)

    if corr is not None and not pd.isna(corr):
        corr = float(corr)
        score = max(0, min(100, round((1 - abs(corr)) * 100)))
        if abs(corr) < 0.5:
            signal = "Active Decoupling Signal"
        elif abs(corr) < 0.7:
            signal = "Partial Decoupling"
        else:
            signal = "Oil Beta Still Coupled"
        return {
            "method": "correlation",
            "correlation": round(corr, 2),
            "score": score,
            "signal": signal,
            "sample_size": len(pairs),
            "latest_dfm_re_change": latest_dfm,
            "latest_wti_change": latest_wti,
        }

    divergence = abs(latest_dfm - latest_wti)
    sign_mismatch = (latest_dfm * latest_wti) < 0
    score = round(min(100, divergence * 12 + (25 if sign_mismatch else 0)))
    if score >= 60:
        signal = "Active Decoupling Signal"
    elif score >= 35:
        signal = "Partial Decoupling"
    else:
        signal = "Oil Beta Still Coupled"
    return {
        "method": "divergence_score",
        "score": score,
        "signal": signal,
        "sample_size": len(pairs),
        "latest_dfm_re_change": latest_dfm,
        "latest_wti_change": latest_wti,
    }

def _translate_risk_trigger_ar(text: str) -> str:
    raw = _clean_text(text)
    if not raw:
        return ""
    out = raw
    replacements = [
        ("SPY loses ", "هبوط SPY دون "),
        ("SPY resolves outside the ", "خروج SPY خارج "),
        ("and breaks the recent lower range", "وكسر النطاق السفلي الأخير"),
        ("recent range", "النطاق الأخير"),
        ("trend confirmation fails", "يعني أن تأكيد الاتجاه يفشل"),
        ("wait for the break to confirm direction", "يعني انتظار تأكيد الكسر قبل توسيع المخاطرة"),
        ("VIX closes above ", "إغلاق VIX فوق "),
        (" or below ", " أو دون "),
        ("the volatility regime stops being neutral", "يعني أن نظام التذبذب خرج من الحياد"),
        ("risk premium is repricing higher", "يعني أن علاوة المخاطر يعاد تسعيرها صعودًا"),
        ("10Y yield breaks above ", "اختراق عائد 10 سنوات فوق "),
        ("10Y yield moves outside the ", "تحرك عائد 10 سنوات خارج نطاق "),
        ("recent band", "النطاق الأخير"),
        ("macro conditions reprice materially", "يعني إعادة تسعير واضحة للسيولة الكلية"),
        ("valuation pressure broadens across equities", "يعني اتساع ضغط التقييمات على الأسهم"),
    ]
    for old, new in replacements:
        out = out.replace(old, new)
    return out

def _translate_phrase_ar(text: str) -> str:
    raw = _clean_text(text)
    if not raw:
        return ""
    out = raw
    replacements = [
        ("Cash", "النقد"),
        ("Quality Equities", "الأسهم عالية الجودة"),
        ("High Beta", "البيتا المضاربية العالية"),
        ("Speculative Crypto", "الكريبتو المضاربي"),
        ("Selective Quality only", "الجودة الانتقائية فقط"),
        ("Selective Quality", "الجودة الانتقائية"),
        ("Capital Preservation", "حفظ رأس المال"),
        ("Defensive Sectors", "القطاعات الدفاعية"),
        ("Active Decoupling Signal", "إشارة انفصال نشطة"),
        ("Partial Decoupling", "انفصال جزئي"),
        ("Oil Beta Still Coupled", "بيتا النفط ما زالت مرتبطة"),
    ]
    for old, new in replacements:
        out = out.replace(old, new)
    return out

def _translate_catalyst_ar(text: str) -> str:
    raw = _clean_text(text)
    if not raw:
        return ""
    if raw.startswith("VIX >") and raw.endswith("level break"):
        level = raw.replace("VIX >", "").replace("level break", "").strip()
        return f"اختراق VIX فوق مستوى {level}"
    mappings = {
        "Fed communications": "تصريحات الاحتياطي الفيدرالي",
        "CPI / PCE data release": "صدور بيانات CPI / PCE",
        "10Y yield break": "اختراق عائد 10 سنوات",
        "Oil and rate repricing": "إعادة تسعير النفط والعوائد",
        "خطاب الفيدرالي": "خطاب الاحتياطي الفيدرالي",
        "بيانات CPI / PCE": "بيانات CPI / PCE",
        "تحرك النفط": "تحرك النفط",
        "العوائد الأميركية": "العوائد الأميركية",
    }
    return mappings.get(raw, raw)

def _ordered_daily_catalysts(triggers: Any) -> list[str]:
    items = _clean_text_list(triggers, 6)
    ordered: list[str] = []
    seen: set[str] = set()

    def _push(label: str) -> None:
        clean = _clean_text(label)
        if clean and clean not in seen:
            ordered.append(clean)
            seen.add(clean)

    def _contains(*keywords: str) -> bool:
        lowered = [item.lower() for item in items]
        return any(any(keyword.lower() in item for keyword in keywords) for item in lowered)

    if _contains("fed", "federal", "الفيدرالي"):
        _push("Fed communications")
    else:
        _push("Fed communications")

    if _contains("10y", "yield", "rates", "العوائد"):
        _push("10Y yield break")
    else:
        _push("10Y yield break")

    if _contains("cpi", "pce"):
        _push("CPI / PCE data release")
    else:
        _push("CPI / PCE data release")

    for item in items:
        low = item.lower()
        if "fed" in low or "cpi" in low or "pce" in low or "10y" in low or "yield" in low or "rates" in low:
            continue
        _push(item)

    return ordered[:4]

def _spy_range_levels(invalidates: list[str]) -> tuple[float, float]:
    import re as _re

    default_low, default_high = 702.0, 710.0
    text = _clean_text(invalidates[0]) if invalidates else ""
    if not text:
        return default_low, default_high
    match = _re.search(r"\$?(\d+(?:\.\d+)?)\s*-\s*\$?(\d+(?:\.\d+)?)", text)
    if not match:
        return default_low, default_high
    low = float(match.group(1))
    high = float(match.group(2))
    return (low, high) if low <= high else (high, low)

def _format_internal_line_en(internal: dict) -> str:
    if not isinstance(internal, dict) or not internal:
        return ""
    movers = ", ".join((internal.get("top_gainers") or [])[:2]) or "no clear upside leaders"
    laggards = ", ".join((internal.get("top_losers") or [])[:2]) or "no clear downside leaders"
    weighted = internal.get("weighted_change")
    weighted_text = f"{weighted:+.2f}%" if isinstance(weighted, (int, float)) else "N/A"
    return (
        f"{internal.get('label', 'Market')}: breadth {internal.get('advancers', 0)} up / "
        f"{internal.get('decliners', 0)} down / {internal.get('unchanged', 0)} flat, "
        f"cap-weighted move {weighted_text}; leadership {internal.get('leading_sector') or 'N/A'}, "
        f"lagging sector {internal.get('lagging_sector') or 'N/A'}. Movers: {movers}. Laggards: {laggards}."
    )

def _format_internal_line_ar(internal: dict, title_ar: str) -> str:
    if not isinstance(internal, dict) or not internal:
        return ""
    movers = "، ".join((internal.get("top_gainers") or [])[:2]) or "لا توجد قيادات صاعدة واضحة"
    laggards = "، ".join((internal.get("top_losers") or [])[:2]) or "لا توجد ضغوط هابطة واضحة"
    weighted = internal.get("weighted_change")
    weighted_text = f"{weighted:+.2f}%" if isinstance(weighted, (int, float)) else "غير متاح"
    lead = internal.get("leading_sector") or "غير واضح"
    lag = internal.get("lagging_sector") or "غير واضح"
    return (
        f"{title_ar}: اتساع السوق {internal.get('advancers', 0)} صاعد / "
        f"{internal.get('decliners', 0)} هابط / {internal.get('unchanged', 0)} دون تغير، "
        f"والحركة الموزونة بالقيمة السوقية {weighted_text}. القطاع القائد {lead}، والقطاع الأضعف {lag}. "
        f"أبرز الصاعدين: {movers}. أبرز الضاغطين: {laggards}."
    )

def _build_full_report_fallback(update: dict) -> str:
    if "date" in update:
        view = update.get("eisax_view", {}) if isinstance(update.get("eisax_view"), dict) else {}
        lines = [
            f"EisaX Daily Market Pulse — {update.get('market_regime', 'Cautious')}",
            f"Date: {update.get('date', '')}",
            f"Confidence: {update.get('regime_confidence', 'Low')}",
            "",
            "What Matters Now",
        ]
        for item in (update.get("what_matters_now") or [])[:3]:
            lines.append(f"• {item}")
        lines += [
            "",
            f"EisaX View: {view.get('stance', 'HOLD')} | Focus: {view.get('focus', '')} | Horizon: {view.get('horizon', '')}",
            f"Cross-Asset Snapshot: {_snapshot_brief(update.get('cross_asset_snapshot') or {})}",
            f"Why Now: {update.get('why_now', '')}",
            f"Tactical Positioning: {update.get('tactical_positioning', '')}",
            "",
        ]
        lines += _trigger_hierarchy_lines(update.get("what_invalidates") or [], "What Invalidates This View")
        lines += ["", "Portfolio Translation", f"• {_best_expression_line(update)}", f"• {_best_hedge_line(update)}"]
        lines += ["", "What to Watch — Next 24–72h"]
        for item in (update.get("next_triggers") or [])[:3]:
            lines.append(f"• {item}")
        report = "\n".join(line for line in lines if line is not None).strip()
        return _enrich_full_report(report, update)

    regional = update.get("regional_view") or {}
    allocation = update.get("asset_allocation_view") or {}
    lines = [
        f"EisaX Weekly Strategy Brief — {update.get('week_range', '')}",
        f"Market Summary: {update.get('market_summary', '')}",
        "",
        "Why Now:",
        *[f"• {item}" for item in _weekly_why_now_lines(update)],
        f"Allocation View: equities {allocation.get('equities', 'Neutral')}, crypto {allocation.get('crypto', 'Neutral')}, metals {allocation.get('metals', 'Neutral')}, commodities {allocation.get('commodities', 'Neutral')}, cash {allocation.get('cash', 'Neutral')}",
        f"Positioning: {update.get('positioning', '')}",
        f"Highest Conviction Opportunity: {update.get('highest_conviction_opportunity', '')}",
        f"Portfolio Angle: {update.get('portfolio_angle', '')}",
        f"EisaX Verdict: {update.get('eisax_verdict', '')}",
        "",
    ]
    lines += ["Portfolio Translation", f"• {_best_expression_line(update)}", f"• {_best_hedge_line(update)}", ""]
    lines += _trigger_hierarchy_lines(update.get("what_changes_this_view") or [], "What Changes This View")
    report = "\n".join(line for line in lines if line is not None).strip()
    return _enrich_full_report(report, update)

def _build_cio_daily_report_fallback(update: dict) -> str:
    view = update.get("eisax_view", {}) if isinstance(update.get("eisax_view"), dict) else {}
    snapshot = update.get("cross_asset_snapshot") or {}
    regime = update.get("market_regime", "Cautious")
    confidence = update.get("regime_confidence", "Low")
    fg_score = update.get("fear_greed_index", 50)
    decision_type = update.get("decision_type") or _daily_decision_type(view)
    confidence_score = update.get("confidence_score") or _daily_confidence_score(regime, confidence, fg_score)
    mode = update.get("positioning_mode") or _daily_positioning_mode(view)
    invalidates = _clean_text_list(update.get("what_invalidates"), 4)
    triggers = _clean_text_list(update.get("next_triggers"), 4)
    catalysts = _ordered_daily_catalysts(triggers)
    catalysts = _ordered_daily_catalysts(triggers)
    catalysts = _ordered_daily_catalysts(triggers)
    market_state = _daily_market_state(snapshot, regime)

    def _entry(key: str) -> dict:
        return snapshot.get(key) or {}

    def _num(value: Any, pct: bool = False, rates: bool = False, crypto: bool = False) -> str:
        if isinstance(value, str):
            return value
        if not isinstance(value, (int, float)):
            return "Market Closed"
        if crypto:
            return f"{value:,.0f}"
        if rates:
            return f"{value:.2f}%"
        if pct:
            return f"{value:+.2f}%"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.2f}"

    us = _entry("us_equities")
    vix = _entry("volatility")
    rates = _entry("rates")
    oil = _entry("commodities")
    btc = _entry("crypto")
    us_price = _as_number(us.get("price"))
    us_price = _as_number(us.get("price"))
    us_d5 = _as_number(us.get("d5_pct"))
    us_price = _as_number(us.get("price"))
    rate_d1 = _as_number(rates.get("d1_pct"))
    oil_d5 = _as_number(oil.get("d5_pct"))
    btc_d5 = _as_number(btc.get("d5_pct"))
    vix_price = _as_number(vix.get("price"))
    oil_price = _as_number(oil.get("price"))
    rate_level = _num(rates.get("price"), rates=True)

    if regime == "Bearish":
        lead = "Liquidity is tightening faster than equities can absorb, so the regime stays decisively defensive."
        stance_line = "Capital allocation should prioritize protection and dry powder over return maximization."
    elif regime == "Bullish":
        lead = "Risk can still be owned, but only where rates are not re-tightening the discount rate faster than earnings can offset it."
        stance_line = "The book can stay constructive, but only through leadership rather than broad beta."
    else:
        lead = "The market is investable, but only through disciplined sizing and active constraint."
        stance_line = "This is a hold-with-intent setup, not a passive hold and not a launch pad for fresh beta."

    if us_d5 is not None and rate_d1 is not None and us_d5 > 0 and rate_d1 > 0:
        contradiction = "Equities are advancing into tighter rates, which means liquidity is lagging price and upside should be treated as narrower than the index suggests."
    elif us_d5 is not None and rate_d1 is not None and us_d5 < 0 and rate_d1 < 0:
        contradiction = "Rates are offering some relief, but equities are not monetizing it yet, which tells you confidence in growth is still incomplete."
    elif oil_d5 is not None and oil_d5 > 0:
        contradiction = "Firmer oil supports regional cash flow, but it also reopens inflation pressure and limits how far duration-sensitive risk can run."
    else:
        contradiction = "Rates, volatility, and oil still matter more than the latest equity headline."

    equities_line = (
        f"Equities -> SPY is at {_num(us.get('price'))} with {_num(us.get('d1_pct'), pct=True)} on the day and {_num(us.get('d5_pct'), pct=True)} over five sessions; price can hold, but broad beta still needs easier liquidity to deserve more capital."
        if us.get("price") != "Market Closed"
        else "Equities -> US cash equities are closed, so the prior session remains the liquidity anchor."
    )
    rates_line = (
        f"Rates -> 10Y Treasury is at {rate_level} with {_num(rates.get('d1_pct'), pct=True)} on the day; that keeps liquidity, valuation support, and position sizing ahead of headline index strength."
        if rates.get("price") != "Market Closed"
        else "Rates -> Treasury pricing is closed, so yesterday's yield regime still governs today's risk budget."
    )
    oil_line = (
        f"Oil -> WTI proxy sits at {_num(oil.get('price'))} with {_num(oil.get('d1_pct'), pct=True)} on the day; that is supportive for GCC cash generation, but it also keeps the inflation channel alive."
        if oil.get("price") != "Market Closed"
        else "Oil -> Crude is closed, but the last print still matters because regional cash flow and global inflation sensitivity both run through energy."
    )
    vix_line = (
        f"VIX -> Volatility is at {_num(vix.get('price'))}; hedging demand is {'still elevated enough to keep risk constrained' if isinstance(vix_price, (int, float)) and vix_price > 20 else 'contained, but not cheap enough to justify indiscriminate exposure'}."
        if vix.get("price") != "Market Closed"
        else "VIX -> The volatility market is closed, so the last fear premium still frames the risk budget."
    )
    crypto_line = (
        f"Crypto -> Bitcoin at {_num(btc.get('price'), crypto=True)} is {'confirming' if isinstance(btc_d5, (int, float)) and isinstance(us_d5, (int, float)) and btc_d5 * us_d5 >= 0 else 'not confirming'} the broader risk tape; treat it as a liquidity tell, not a primary signal."
        if btc.get("price") != "Market Closed"
        else "Crypto -> Bitcoin remains a secondary liquidity tell while fresh price discovery is unavailable."
    )

    gcc_flow = (
        "Sector flows should stay biased toward GCC energy and banks while oil remains firm."
        if isinstance(oil_price, (int, float)) and oil_price >= 70
        else "Sector flows should stay selective, with GCC defensives and high-quality banks preferred over cyclical chasing."
    )
    egypt_flow = (
        "Egypt still needs hard-currency balance sheets, exporters, and defensives while higher US rates keep financing conditions tight."
        if isinstance(rate_d1, (int, float)) and rate_d1 >= 0
        else "Egypt gets tactical breathing room only if US rates ease without a new oil shock."
    )

    if decision_type == "REDUCE":
        execution_rules = [
            "â€¢ Cut high-beta and speculative exposure before paying up for index protection.",
            "â€¢ Keep liquidity high enough to re-risk only after the SPY and VIX triggers improve together.",
            "â€¢ Hold gold and short duration as the core defensive ballast.",
        ]
    elif decision_type == "BUY_SELECTIVE":
        execution_rules = [
            "â€¢ Add only on pullbacks into support; do not chase extension when rates are still the first variable.",
            "â€¢ Keep new exposure concentrated in quality leadership, not equal-weight beta.",
            "â€¢ Fund additions by trimming laggards and crowded speculative sleeves.",
        ]
    else:
        execution_rules = [
            "â€¢ Maintain core quality exposure, but keep gross risk capped until price and liquidity confirm together.",
            "â€¢ Deploy only on weakness that holds structure, not on emotional upside extension.",
            "â€¢ Recycle capital from speculative beta into quality, defense, and liquid optionality.",
        ]

    tactical_avoid = ", ".join(view.get("underweight_assets") or []) or "speculative beta"
    tactical_focus = view.get("focus") or "selective quality"
    maintain_line = ", ".join(view.get("overweight_assets") or view.get("neutral_assets") or ["Quality Equities"])
    deploy_line = update.get("tactical_positioning") or "Deploy fresh capital only when pullbacks hold and rates stop tightening."
    risk_shift = (
        "A break in any two of the three core triggers forces a regime review and a fresh position-sizing decision."
        if len(invalidates) >= 3
        else "A clear break in price, volatility, or rates is enough to force a new allocation stance."
    )

    lines = [
        "EisaX Daily Market Pulse",
        f"Date: {update.get('date', '')}",
        f"Regime: {regime}",
        f"Confidence: {confidence}",
        f"Decision Type: {decision_type}",
        f"Confidence Score: {confidence_score}",
        "",
        "## Executive Summary",
        f"â€¢ {lead}",
        f"â€¢ {contradiction}",
        f"â€¢ {stance_line}",
        f"â€¢ Stance stays {view.get('stance', 'HOLD')} in {mode.lower()} mode until the risk framework breaks.",
        "",
        "## Cross-Asset Reality",
        f"â€¢ {equities_line}",
        f"â€¢ {rates_line}",
        f"â€¢ {oil_line}",
        f"â€¢ {vix_line}",
        f"â€¢ {crypto_line}",
        "This is not a directional market.",
        f"This is a {market_state} market.",
        "",
        "## Regional Read (GCC + Egypt)",
        f"â€¢ GCC stays anchored to oil at {_num(oil.get('price'))}; firmer crude supports cash flow, fiscal room, and bank liquidity, but it also keeps the inflation channel alive.",
        f"â€¢ US rates at {rate_level} matter more than headline equity strength for regional liquidity: GCC can absorb that pressure better than Egypt, while Egypt remains exposed to imported inflation and funding costs.",
        f"â€¢ {gcc_flow} {egypt_flow}",
        "",
        "## Positioning",
        f"Stance: {view.get('stance', 'HOLD')}",
        f"Mode: {mode}",
        "Execution:",
    ]
    lines.extend(execution_rules)
    lines += [
        "",
        "## Risk Framework",
        f"â€¢ SPY level: {invalidates[0] if len(invalidates) > 0 else 'SPY must hold the current range or the tape loses sponsorship.'}",
        f"â€¢ VIX level: {invalidates[1] if len(invalidates) > 1 else 'VIX needs to stay contained; a renewed volatility spike would tighten the risk budget immediately.'}",
        f"â€¢ 10Y level: {invalidates[2] if len(invalidates) > 2 else 'A higher 10Y yield would reprice liquidity and cap equity upside.'}",
        f"â€¢ Regime shift: {risk_shift}",
        "",
        "## Tactical Playbook",
        f"â€¢ Maintain: keep exposure centered on {maintain_line}.",
        f"â€¢ Deploy: {deploy_line}",
        f"â€¢ Avoid: avoid adding to {tactical_avoid} while rates and volatility remain the gating variables.",
        f"â€¢ Focus: focus on {tactical_focus}, GCC banks/energy when oil confirms, and Egypt only through balance-sheet quality.",
        "",
        "## Catalysts",
    ]
    for item in (triggers or ["Fed communications", "CPI / PCE data release", "Oil and rate repricing"])[:4]:
        lines.append(f"â€¢ {item}")
    lines += [
        "",
        "## Final Line",
        "Capital should follow liquidity, not headlines; until rates and volatility confirm the next leg, the only winning aggression is selective aggression.",
    ]
    return "\n".join(lines).strip()

