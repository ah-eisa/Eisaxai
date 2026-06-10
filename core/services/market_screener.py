"""
core/services/market_route_handler.py
───────────────────────────────────────
Route handlers for STOCK_ANALYSIS, FINANCIAL (CIO/PORTFOLIO_OPTIMIZE),
PORTFOLIO CRUD and GENERAL (Gemini) — extracted from process_message.

Public API
──────────
    handle_stock_analysis(orchestrator, session_id, user_id,
                          message, instruction, user_ctx) -> dict

    handle_financial(orchestrator, session_id, user_id,
                     message, instruction, handler, user_ctx) -> dict | None
        Returns None if it falls through (caller continues to GENERAL).

    handle_portfolio(session_id, user_id, message, reply_saver) -> dict

    handle_general(orchestrator, session_id, user_id,
                   message, instruction, user_ctx) -> dict
"""

from __future__ import annotations

import logging
import os
import re as _re
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Screening intent helpers ──────────────────────────────────────────────────

_LIST_KEYWORDS = [
    "أفضل", "best", "top", "أعلى", "اقترح", "recommend",
    "أسهم توزيعات", "dividend stocks", "dividend yield", "top stocks",
    "أسهم دفاعية", "defensive stocks", "ranking", "قائمة",
    "أعلى عائد", "highest yield", "screen", "فلتر", "screening", "rank",
    "قائمة أسهم", "top performers", "gainers", "losers", "top gainers", "top losers",
]

_PORTFOLIO_KEYWORDS = [
    "محفظة", "portfolio", "optimize", "وزّع", "وزع", "allocate",
    "ابني", "build", "construct", "توزيع الأصول",
]

_SCREENING_SIGNALS = (
    "dividend", "توزيعات", "yield", "عائد", "defensive", "دفاعية",
    "rsi", "oversold", "overbought", "gainers", "losers", "أعلى ارتفاع", "أعلى انخفاض",
)

_MARKET_NAME_MAP = {
    "uae":         "الإمارات (ADX/DFM)",
    "sau":         "السعودية (تداول)",
    "ksa":         "السعودية (تداول)",
    "egypt":       "مصر (EGX)",
    "egy":         "مصر (EGX)",
    "kuwait":      "الكويت",
    "kwt":         "الكويت",
    "qatar":       "قطر",
    "qat":         "قطر",
    "bahrain":     "البحرين",
    "bhr":         "البحرين",
    "morocco":     "المغرب",
    "mar":         "المغرب",
    "tunisia":     "تونس",
    "tun":         "تونس",
    "america":     "الولايات المتحدة (NYSE/NASDAQ)",
    "usa":         "الولايات المتحدة (NYSE/NASDAQ)",
    "crypto":      "Crypto",
    "commodities": "Commodities",
    "global":      "Global",
}

_MARKET_HINTS = {
    "uae": ["uae", "الإمارات", "امارات", "إمارات", "adx", "dfm", "دبي", "أبوظبي", "ابوظبي"],
    "ksa": ["ksa", "السعودية", "سعودية", "سعودي", "تداول", "tadawul", "ارامكو", "aramco"],
    "egypt": ["مصر", "egypt", "egx", "بورصة", "البورصة"],
    "kuwait": ["كويت", "الكويت", "kuwait"],
    "qatar": ["قطر", "qatar"],
}

_MARKET_CURRENCY_MAP = {
    "uae":         "د.إ",
    "sau":         "ر.س",
    "ksa":         "ر.س",
    "egypt":       "ج.م",
    "egy":         "ج.م",
    "kuwait":      "د.ك",
    "kwt":         "د.ك",
    "qatar":       "ر.ق",
    "qat":         "ر.ق",
    "bahrain":     "د.ب",
    "bhr":         "د.ب",
    "morocco":     "MAD",
    "mar":         "MAD",
    "tunisia":     "TND",
    "tun":         "TND",
    "america":     "USD",
    "usa":         "USD",
    "crypto":      "USD",
    "commodities": "USD",
    "global":      "USD",
}


def _detect_market_intent(message: str) -> str:
    """Returns 'screening' or 'portfolio' or 'unknown'."""
    msg_lower = (message or "").lower()
    list_score = sum(1 for kw in _LIST_KEYWORDS if kw.lower() in msg_lower)
    port_score = sum(1 for kw in _PORTFOLIO_KEYWORDS if kw.lower() in msg_lower)
    if list_score > port_score:
        return "screening"
    if port_score > 0:
        return "portfolio"
    return "unknown"


def _detect_market_from_message(message: str) -> str:
    ml = (message or "").lower()
    for market_code, hints in _MARKET_HINTS.items():
        if any(w in ml for w in hints):
            return market_code
    return "uae"


def _coerce_dt(value: Any) -> Any:
    try:
        import datetime as _dt

        if value is None:
            return None
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
        if isinstance(value, _dt.datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=_dt.timezone.utc)
            return value.astimezone(_dt.timezone.utc)
        if isinstance(value, (int, float)):
            if value > 1e12:
                value = value / 1000.0
            return _dt.datetime.fromtimestamp(value, tz=_dt.timezone.utc)
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return None
            txt = txt.replace("Z", "+00:00")
            try:
                parsed = _dt.datetime.fromisoformat(txt)
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=_dt.timezone.utc)
                return parsed.astimezone(_dt.timezone.utc)
            except Exception:
                return None
    except Exception:
        return None
    return None


def _fmt_snapshot_ts(snapshot_ts: Any, stocks: list[dict]) -> str:
    import datetime as _dt

    dt_obj = _coerce_dt(snapshot_ts)
    if dt_obj is None and stocks:
        dt_obj = _coerce_dt(stocks[0].get("_snapshot_ts"))
    if dt_obj is None:
        dt_obj = _dt.datetime.now(_dt.timezone.utc)
    return dt_obj.strftime("%Y-%m-%d %H:%M UTC")


def _resolve_cache_age_minutes(cache_age_min: float | None, snapshot_ts: Any, stocks: list[dict]) -> float:
    import datetime as _dt

    if cache_age_min is not None:
        return max(0.0, float(cache_age_min))
    dt_obj = _coerce_dt(snapshot_ts)
    if dt_obj is None and stocks:
        dt_obj = _coerce_dt(stocks[0].get("_snapshot_ts"))
    if dt_obj is None:
        return 0.0
    now = _dt.datetime.now(_dt.timezone.utc)
    return max(0.0, (now - dt_obj).total_seconds() / 60.0)


def _get_market_stocks_from_cache(market: str) -> tuple[list[dict], float | None, Any]:
    """
    Load latest live market snapshot rows from pipeline cache.
    Returns (rows, age_minutes, snapshot_ts).
    """
    # Map entity-resolution market codes (SAU, EGY, KWT, QAT, CRYPTO, GLOBAL, USA)
    # to pipeline cache keys (ksa, egypt, kuwait, qatar, crypto, commodities, america)
    _ER_TO_PIPELINE: dict[str, str] = {
        "sau":    "ksa",
        "egy":    "egypt",
        "kwt":    "kuwait",
        "qat":    "qatar",
        "bhr":    "bahrain",
        "mar":    "morocco",
        "tun":    "tunisia",
        "usa":    "america",
        "global": "commodities",
    }
    market_code = (market or "uae").lower()
    market_code = _ER_TO_PIPELINE.get(market_code, market_code)
    if market_code not in _MARKET_NAME_MAP:
        market_code = "uae"

    # Primary path: pipeline cache singleton
    try:
        from pipeline import cache as _pipeline_cache
        df, snapshot_ts = _pipeline_cache.get_latest(market_code)
        age = _pipeline_cache.cache_age_minutes(market_code)
        if df is not None and not df.empty:
            rows = df.to_dict(orient="records")
            if snapshot_ts is None and rows:
                snapshot_ts = rows[0].get("_snapshot_ts")
            return rows, age, snapshot_ts
    except Exception as exc:
        logger.debug("[Screening] pipeline cache load failed for %s: %s", market_code, exc)

    # Fallback path: direct allocator snapshot loader
    try:
        from global_allocator import _load_latest_snapshot
        df = _load_latest_snapshot(market_code)
        if df is not None and not df.empty:
            rows = df.to_dict(orient="records")
            snapshot_ts = rows[0].get("_snapshot_ts") if rows else None
            return rows, None, snapshot_ts
    except Exception as exc:
        logger.debug("[Screening] allocator snapshot load failed for %s: %s", market_code, exc)

    return [], None, None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            v = value.strip().replace(",", "").replace("%", "")
            return float(v) if v else default
        return float(value)
    except Exception:
        return default


def _get_row_yield_pct(stock_row: dict) -> float:
    """
    Normalize dividend yield to percent.
    TV cache usually stores percent directly in dividend_yield_recent.
    """
    raw = (
        stock_row.get("div_yield")
        or stock_row.get("dividend_yield")
        or stock_row.get("dividendYield")
        or stock_row.get("dividend_yield_recent")
        or 0
    )
    dy = _to_float(raw, 0.0)
    if 0 < dy <= 1:
        dy *= 100.0
    return max(0.0, dy)


def _get_row_payout_ratio(s: dict) -> float | None:
    """Calculate payout ratio from available data. Returns None if not computable."""
    try:
        price = _to_float(s.get("close") or s.get("price") or s.get("current_price") or 0)
        dy_pct = _get_row_yield_pct(s)
        eps = _to_float(s.get("earnings_per_share_diluted_ttm") or s.get("eps") or 0)
        if price > 0 and dy_pct > 0 and eps > 0:
            dps = (dy_pct / 100) * price
            return round((dps / eps) * 100, 1)
    except Exception:
        pass
    return None


def _sustainability_flag(payout: float | None) -> str:
    """Return emoji sustainability indicator based on payout ratio."""
    if payout is None:
        return "—"
    if payout <= 50:
        return "🟢"
    if payout <= 70:
        return "🟡"
    if payout <= 90:
        return "🟠"
    return "🔴"


def _fmt_num(value: Any, digits: int = 2, fallback: str = "—") -> str:
    num = _to_float(value, default=float("nan"))
    if num != num:  # NaN
        return fallback
    return f"{num:.{digits}f}"


def _get_row_rsi(stock_row: dict) -> float:
    return _to_float(stock_row.get("RSI") or stock_row.get("rsi") or stock_row.get("rsi_14"), 0.0)


def _get_row_change_pct(stock_row: dict) -> float:
    return _to_float(stock_row.get("change") or stock_row.get("change_percent") or stock_row.get("chg"), 0.0)


def _apply_quality_filter(stocks: list[dict], screening_type: str = "dividend") -> list[dict]:
    """Remove anomalous/illiquid stocks before ranking."""
    out: list[dict] = []
    for s in stocks:
        # Skip zero/near-zero volume rows (likely stale, halted, or anomalous).
        vol = _to_float(s.get("volume") or 0)
        if vol < 50_000:
            continue

        # Filter clear RSI outliers that are usually data quality issues.
        rsi = _to_float(s.get("RSI") or s.get("rsi") or 50)
        if rsi >= 99 or rsi <= 5:
            continue

        # Unrealistically high dividend yields are often anomalies/special events.
        if screening_type == "dividend":
            dy = _get_row_yield_pct(s)
            if dy > 25:
                continue

        # Drop micro-caps if market cap exists and is too small for liquid screening.
        mc = _to_float(s.get("market_cap_basic") or 0)
        if mc > 0 and mc < 200_000_000:
            continue

        out.append(s)
    return out


def _div_stability_score(s: dict) -> float:
    """Composite dividend ranking score balancing yield and stability proxies."""
    dy = _get_row_yield_pct(s)
    mc = _to_float(s.get("market_cap_basic") or 0) / 1e9  # billions
    pe = _to_float(s.get("price_earnings_ttm") or 0)
    rsi = _to_float(s.get("RSI") or s.get("rsi") or 50)
    vol = _to_float(s.get("volume") or 0)

    score = dy  # base: yield %

    # Market cap bonus — much stronger weighting
    if mc > 50:
        score += 6    # mega cap (EMAAR, FAB, ENBD)
    elif mc > 10:
        score += 4    # large cap
    elif mc > 2:
        score += 2    # mid cap
    elif mc > 0.5:
        score += 1    # small cap
    # micro cap: no bonus

    # P/E bonus: profitable and reasonably valued
    if 3 < pe < 15:
        score += 1.5
    elif 15 <= pe < 25:
        score += 0.5

    # RSI: healthy range bonus
    if 35 <= rsi <= 65:
        score += 0.5

    # Volume bonus: liquid
    if vol > 5_000_000:
        score += 1.5
    elif vol > 1_000_000:
        score += 1
    elif vol > 200_000:
        score += 0.5

    # Payout ratio — most important for sustainability
    payout = _get_row_payout_ratio(s)
    if payout is not None:
        if payout <= 40:
            score += 4
        elif payout <= 60:
            score += 2
        elif payout <= 80:
            score += 0
        elif payout <= 100:
            score -= 3
        else:
            score -= 6

    return score


_SECTOR_ALIAS_HINTS = {
    "banks": ["bank", "banks", "بنك", "بنوك", "مصرف", "مصارف"],
    "real estate": ["real estate", "realestate", "عقار", "عقاري"],
    "energy": ["energy", "oil", "gas", "طاقة", "نفط", "غاز"],
    "telecommunication": ["telecom", "telecommunication", "communications", "اتصالات"],
    "utilities": ["utilities", "utility", "مرافق"],
    "healthcare": ["health", "healthcare", "medical", "دواء", "صحي", "رعاية صحية"],
    "consumer staples": ["consumer staples", "staples", "سلع أساسية", "اغذية", "أغذية"],
    "industrials": ["industrial", "industrials", "صناعي", "صناعة"],
}

_DEFENSIVE_SECTOR_HINTS = (
    "utilities", "health", "healthcare", "consumer staples", "telecom", "communications",
)


def _detect_screening_type(message: str) -> str:
    ml = (message or "").lower()

    if any(w in ml for w in ["oversold", "تشبع بيعي", "rsi منخفض", "rsi تحت 35"]):
        return "rsi_oversold"
    if any(w in ml for w in ["overbought", "تشبع شرائي", "rsi مرتفع", "rsi فوق 65"]):
        return "rsi_overbought"
    if any(w in ml for w in ["top gainers", "gainers", "أعلى ارتفاع", "الأعلى ارتفاعاً", "الاسهم الصاعدة"]):
        return "top_gainers"
    if any(w in ml for w in ["top losers", "losers", "أعلى انخفاض", "الأكثر انخفاضاً", "الاسهم الهابطة"]):
        return "top_losers"
    if "sector" in ml or "قطاع" in ml:
        return "sector"
    if any(w in ml for w in ["defensive", "دفاعية"]):
        return "defensive"
    if any(w in ml for w in ["dividend", "توزيعات", "yield", "عائد"]):
        return "dividend"
    return "dividend"


def _extract_sector_filter(message: str, stocks: list[dict]) -> str:
    ml = (message or "").lower()
    for sector_key, hints in _SECTOR_ALIAS_HINTS.items():
        if any(h in ml for h in hints):
            return sector_key

    unique_sectors = {
        str(s.get("sector") or "").strip()
        for s in stocks
        if str(s.get("sector") or "").strip()
    }
    for sector_name in unique_sectors:
        if sector_name and sector_name.lower() in ml:
            return sector_name
    return ""


def _row_sector_text(stock_row: dict) -> str:
    return str(stock_row.get("sector") or "—").strip() or "—"


def _screen_rows(
    stocks: list[dict],
    screening_type: str,
    message: str,
    top_n: int = 10,
) -> tuple[list[dict], str, str]:
    filtered_stocks = _apply_quality_filter(stocks, screening_type)

    if screening_type == "rsi_oversold":
        rows = [s for s in filtered_stocks if 0 < _get_row_rsi(s) < 35]
        rows.sort(key=_get_row_rsi)
        return rows[:top_n], "أكثر الأسهم تشبعاً بيعياً (RSI < 35)", "RSI"

    if screening_type == "rsi_overbought":
        rows = [s for s in filtered_stocks if _get_row_rsi(s) > 65]
        rows.sort(key=_get_row_rsi, reverse=True)
        return rows[:top_n], "أكثر الأسهم تشبعاً شرائياً (RSI > 65)", "RSI"

    if screening_type == "top_gainers":
        rows = sorted(filtered_stocks, key=_get_row_change_pct, reverse=True)
        return rows[:top_n], "أعلى الأسهم ارتفاعاً", "التغير %"

    if screening_type == "top_losers":
        rows = sorted(filtered_stocks, key=_get_row_change_pct)
        return rows[:top_n], "أعلى الأسهم انخفاضاً", "التغير %"

    if screening_type == "sector":
        sector_term = _extract_sector_filter(message, filtered_stocks)
        if sector_term:
            rows = [s for s in filtered_stocks if sector_term.lower() in _row_sector_text(s).lower()]
            rows.sort(key=_get_row_yield_pct, reverse=True)
            return rows[:top_n], f"أفضل أسهم قطاع {sector_term}", "القطاع"

    if screening_type == "defensive":
        rows = [
            s for s in filtered_stocks
            if any(h in _row_sector_text(s).lower() for h in _DEFENSIVE_SECTOR_HINTS)
        ]
        if rows:
            rows.sort(key=_get_row_yield_pct, reverse=True)
            return rows[:top_n], "أفضل الأسهم الدفاعية", "Dividend Yield"

    rows = [s for s in filtered_stocks if _get_row_yield_pct(s) > 0]
    if screening_type == "dividend":
        rows.sort(key=_div_stability_score, reverse=True)
    else:
        rows.sort(key=_get_row_yield_pct, reverse=True)
    return rows[:top_n], "أفضل أسهم التوزيعات", "Dividend Yield"


def _fmt_screen_metric(stock_row: dict, metric_label: str) -> str:
    if metric_label == "Dividend Yield":
        return f"{_get_row_yield_pct(stock_row):.2f}%"
    if metric_label == "RSI":
        return _fmt_num(_get_row_rsi(stock_row), 1)
    if metric_label == "التغير %":
        chg = _get_row_change_pct(stock_row)
        return f"{chg:+.2f}%"
    if metric_label == "القطاع":
        return _row_sector_text(stock_row)
    return _fmt_num(stock_row.get(metric_label), 2)


def _build_screening_reply(message: str, market: str | None = None, forced_type: str | None = None) -> str:
    market_code = (market or _detect_market_from_message(message)).lower()
    stocks, cache_age_min, snapshot_ts = _get_market_stocks_from_cache(market_code)
    if not stocks:
        return "⚠️ بيانات السوق غير متاحة حالياً — جاري التحديث\nحاول تاني خلال دقيقتين"

    _num_match = _re.search(r"\b(\d+)\b", message)
    top_n = int(_num_match.group(1)) if _num_match and 3 <= int(_num_match.group(1)) <= 20 else 10

    screening_type = forced_type or _detect_screening_type(message)
    top, title, metric_label = _screen_rows(stocks, screening_type, message, top_n=top_n)
    if not top:
        return "⚠️ لا تتوفر نتائج كافية للفلاتر المطلوبة حالياً"

    market_name = _MARKET_NAME_MAP.get(market_code, market_code.upper())
    currency = _MARKET_CURRENCY_MAP.get(market_code, "")
    age_min = _resolve_cache_age_minutes(cache_age_min, snapshot_ts, stocks)
    ts_text = _fmt_snapshot_ts(snapshot_ts, stocks)

    if screening_type == "dividend":
        rows = [
            "| # | الشركة | الرمز | السعر | Div Yield | Payout | P/E | الاستدامة |",
            "|---|--------|-------|-------|-----------|--------|-----|-----------|",
        ]
    else:
        rows = [
            f"| # | الشركة | الرمز | السعر | {metric_label} | P/E | RSI |",
            "|---|--------|-------|-------|----------|-----|-----|",
        ]
    for i, s in enumerate(top, 1):
        ticker = str(s.get("ticker") or "").strip()
        name = s.get("name") or (ticker.split(":", 1)[-1] if ticker else "N/A")
        price = _to_float(s.get("price") or s.get("current_price") or s.get("close") or 0, 0.0)
        pe = _fmt_num(s.get("price_earnings_ttm") or s.get("pe_ratio") or s.get("pe") or s.get("forwardPE"), 2)
        if screening_type == "dividend":
            dy_pct = _get_row_yield_pct(s)
            payout = _get_row_payout_ratio(s)
            sustain = _sustainability_flag(payout)
            payout_str = f"{payout:.0f}%" if payout is not None else "—"
            rows.append(
                f"| {i} | {name} | {ticker or '—'} | {currency}{price:,.2f} | {dy_pct:.2f}% | {payout_str} | {pe} | {sustain} |"
            )
        else:
            metric_val = _fmt_screen_metric(s, metric_label)
            rsi = _fmt_num(_get_row_rsi(s), 1)
            rows.append(
                f"| {i} | {name} | {ticker or '—'} | {currency}{price:,.2f} | {metric_val} | {pe} | {rsi} |"
            )

    table = "\n".join(rows)
    return (
        f"🏆 {title} — {market_name}\n"
        f"*{len(top)} سهم | آخر تحديث: {age_min:.0f} دقيقة*\n\n"
        f"{table}\n\n"
        f"> المصدر: EisaX Live Cache ({ts_text})"
    )


def _handle_dividend_screening(message: str, market: str | None = None) -> str:
    """Backward-compatible dividend screening adapter."""
    return _build_screening_reply(message, market=market, forced_type="dividend")


async def handle_screening(
    message: str,
    session_id: str,
    user_id: str,
    orchestrator,
    instruction: str = "",
    **kwargs,
) -> dict:
    """Public SCREENER route handler backed by live pipeline cache."""
    screening_input = f"{message or ''} {instruction or ''}".strip()
    reply_text = _build_screening_reply(screening_input)
    try:
        orchestrator.session_mgr.save_message(session_id, user_id, "user", message)
        orchestrator.session_mgr.save_message(session_id, user_id, "assistant", reply_text)
    except Exception as exc:
        logger.warning("[SCREENER] session save failed: %s", exc)
    return {
        "reply": reply_text,
        "session_id": session_id,
        "agent_name": "EisaX Market Screener",
        "model": "SCREENER",
    }

