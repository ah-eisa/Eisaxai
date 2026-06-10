"""
local_market_enricher.py — EisaX Local Market Data Enricher
يجيب داتا الأسواق المحلية ويحقنها في الـ finance agent
يُستخدم في _handle_analytics قبل ما يبعت للـ LLM
"""

import sqlite3
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

from core.config import CORE_DB as _cfg_core_db
DB_PATH = str(_cfg_core_db)


def is_local_ticker(ticker: str) -> bool:
    return any(ticker.upper().endswith(s) for s in [".SR", ".CA", ".AE", ".DU", ".KW", ".QA"])


def get_market(ticker: str) -> Optional[str]:
    t = ticker.upper()
    if t.endswith(".SR"): return "SA"
    if t.endswith(".CA"): return "EG"
    if t.endswith(".AE") or t.endswith(".DU"): return "AE"
    if t.endswith(".KW"): return "KW"
    if t.endswith(".QA"): return "QA"
    return None


def get_local_price(ticker: str) -> Optional[dict]:
    """يجيب آخر سعر من الـ cache المحلي"""
    try:
        import sys; sys.path.insert(0, '/home/ubuntu/investwise'); from core.market_data_engine import get_latest_price
        market = get_market(ticker)
        if not market:
            return None
        return get_latest_price(ticker, market)
    except Exception as e:
        logger.error(f"get_local_price error {ticker}: {e}")
        return None


def get_local_fundamentals(ticker: str) -> dict:
    """
    يجيب الـ fundamentals من SQLite (DFM/ADX data)
    يعمل name matching عشان الـ ticker مش موجود في الجدول
    """
    fund = {}
    try:
        conn = sqlite3.connect(DB_PATH)

        # جيب اسم الشركة من local_tickers أول
        company_name = _get_company_name(ticker)

        if company_name:
            # DFM fundamentals — column avg_vol_3m doesn't exist in the
            # current uae_fundamentals schema; pull div_yield + eps instead
            # so dividend-aware verdict logic gets a regional fallback when
            # yfinance lacks data for ADX/DFM tickers.
            row = conn.execute("""
                SELECT market_cap, pe_ratio, beta, div_yield, revenue, eps
                FROM uae_fundamentals
                WHERE LOWER(name) LIKE LOWER(?) OR LOWER(name) LIKE LOWER(?)
                LIMIT 1
            """, (f"%{company_name}%", f"%{ticker.split('.')[0]}%")).fetchone()

            if row:
                fund["market_cap"]    = row[0]
                fund["pe_ratio"]      = row[1]
                fund["beta"]          = row[2]
                fund["dividend_yield"] = row[3]   # decimal or % per source
                fund["revenue"]       = row[4]
                fund["eps"]           = row[5]

            # ADX signals (uae_signals table is optional — created by the
            # ADX scraper. If the table doesn't exist on this deployment,
            # skip silently rather than logging an error on every analyse).
            try:
                sig = conn.execute("""
                    SELECT signal_daily, signal_weekly, signal_monthly
                    FROM uae_signals
                    WHERE LOWER(name) LIKE LOWER(?) OR LOWER(name) LIKE LOWER(?)
                    LIMIT 1
                """, (f"%{company_name}%", f"%{ticker.split('.')[0]}%")).fetchone()
                if sig:
                    fund["signal_daily"]   = sig[0]
                    fund["signal_weekly"]  = sig[1]
                    fund["signal_monthly"] = sig[2]
            except sqlite3.OperationalError as _sig_err:
                if "no such table" not in str(_sig_err).lower():
                    raise  # surface unexpected DB issues

        conn.close()
    except Exception as e:
        logger.error(f"get_local_fundamentals error {ticker}: {e}")

    return fund


def get_local_historical_context(ticker: str, period: str = "1y") -> dict:
    """يجيب إحصائيات تاريخية من الـ engine"""
    try:
        from core.market_data_engine import prepare_analysis_context
        market = get_market(ticker)
        if not market:
            return {}
        return prepare_analysis_context(ticker, market, period=period)
    except Exception as e:
        logger.error(f"get_local_historical_context error {ticker}: {e}")
        return {}


def _get_company_name(ticker: str) -> Optional[str]:
    """يجيب اسم الشركة من local_tickers.py"""
    try:
        from core.local_tickers import UAE_TICKERS, SAUDI_TICKERS, EGYPT_TICKERS
        all_tickers = {**UAE_TICKERS, **SAUDI_TICKERS, **EGYPT_TICKERS}
        info = all_tickers.get(ticker.upper(), {})
        return info.get("name_en") or info.get("name_ar")
    except Exception as _e:
        return ticker.split(".")[0]


def enrich_local_analysis(ticker: str) -> dict:
    """
    الواجهة الرئيسية — تجيب كل الداتا المحلية لسهم معين
    
    Returns dict يُحقن في prompt الـ LLM
    """
    if not is_local_ticker(ticker):
        return {}

    market = get_market(ticker)
    result = {"ticker": ticker, "market": market, "is_local": True}

    # 0. TradingView cache — authoritative source for GCC/MENA equities.
    # Pulls live price + fundamentals from the 15-min TV snapshot so the
    # LLM narrative sees the same numbers as the peer table and scorecard.
    _tv_row = None
    _tv_snapshot_ts = None
    try:
        from core.data_layer import market_cache_adapter as _mca
        _tv_market_map = {"AE": "uae", "SA": "ksa", "EG": "egypt",
                          "KW": "kuwait", "QA": "qatar"}
        _tv_mkt = _tv_market_map.get(market)
        if _tv_mkt:
            _df = _mca.get_latest_snapshot(_tv_mkt)
            _tv_snapshot_ts = _mca.snapshot_timestamp(_tv_mkt)
            if _df is not None and not _df.empty and "ticker" in _df.columns:
                _bare = ticker.upper().split(".")[0]
                _match = _df[
                    _df["ticker"].astype(str).str.upper().str.endswith(":" + _bare)
                    | (_df["ticker"].astype(str).str.upper() == ticker.upper())
                ]
                if not _match.empty:
                    _tv_row = _match.iloc[0].to_dict()
    except Exception as _tv_e:
        logger.debug(f"[Enricher/TV] {ticker}: {_tv_e}")

    # 1. السعر الحالي — TV authoritative, fallback to Investing.com/yfinance
    if _tv_row:
        try:
            result["price"]      = float(_tv_row.get("close") or 0) or None
            result["change_pct"] = float(_tv_row.get("change") or 0)
            result["currency"]   = "AED" if market == "AE" else ("SAR" if market == "SA" else ("EGP" if market == "EG" else "AED"))
            result["date"]       = (_tv_snapshot_ts or "")[:10]
            result["high"]       = float(_tv_row.get("high") or 0) or None
            result["low"]        = float(_tv_row.get("low") or 0) or None
            result["volume"]     = int(_tv_row.get("volume") or 0) or None
            result["price_source"] = "TradingView (authoritative for GCC)"
            result["snapshot_ts"] = _tv_snapshot_ts
        except Exception:
            _tv_row = None  # fall through to fallback below
    if not _tv_row:
        price_data = get_local_price(ticker)
        if price_data:
            result["price"]      = price_data.get("close")
            result["change_pct"] = price_data.get("change_pct")
            result["currency"]   = price_data.get("currency", "AED")
            result["date"]       = price_data.get("date")
            result["volume"]     = price_data.get("volume")
            result["high"]       = price_data.get("high")
            result["low"]        = price_data.get("low")
            result["price_source"] = "Investing.com/yfinance (fallback)"
        else:
            result["price"]    = None
            result["currency"] = "AED" if market == "AE" else ("SAR" if market == "SA" else "EGP")
            result["price_source"] = "unavailable"

    # 2. Fundamentals — TV first (live snapshot), DB second (cached)
    fund = {}
    if _tv_row:
        if _tv_row.get("market_cap_basic"):
            fund["market_cap"] = float(_tv_row["market_cap_basic"])
        if _tv_row.get("price_earnings_ttm"):
            try:
                _pe = float(_tv_row["price_earnings_ttm"])
                import math as _m
                if not (_m.isnan(_pe) or _m.isinf(_pe)):
                    fund["pe_ratio"] = _pe
            except Exception:
                pass
        if _tv_row.get("beta_1_year"):
            try: fund["beta"] = float(_tv_row["beta_1_year"])
            except Exception: pass
        if _tv_row.get("dividend_yield_recent") is not None:
            try: fund["dividend_yield"] = float(_tv_row["dividend_yield_recent"])
            except Exception: pass
        if _tv_row.get("total_revenue_ttm"):
            try: fund["revenue"] = float(_tv_row["total_revenue_ttm"])
            except Exception: pass
        if _tv_row.get("earnings_per_share_diluted_ttm"):
            try: fund["eps"] = float(_tv_row["earnings_per_share_diluted_ttm"])
            except Exception: pass
    # Fill gaps from DB (signals come from DB-only)
    _db_fund = get_local_fundamentals(ticker)
    for k, v in (_db_fund or {}).items():
        if k not in fund or fund.get(k) is None:
            fund[k] = v
    if fund:
        result["fundamentals"] = fund

    # 3. Historical context
    hist = get_local_historical_context(ticker)
    if hist and "error" not in hist:
        result["historical"] = {
            "total_return_1y": hist.get("returns_stats", {}).get("total_return_pct"),
            "volatility":      hist.get("returns_stats", {}).get("volatility_annual"),
            "sharpe":          hist.get("returns_stats", {}).get("sharpe_approx"),
            "high_52w":        hist.get("price_stats", {}).get("high_52w"),
            "low_52w":         hist.get("price_stats", {}).get("low_52w"),
            "data_points":     hist.get("data_points"),
        }

    return result


def build_local_prompt_injection(ticker: str) -> str:
    """
    يبني نص جاهز للـ inject في prompt الـ LLM
    بيستبدل الداتا اللي مش بتجي من yfinance للأسواق المحلية
    """
    data = enrich_local_analysis(ticker)
    if not data:
        return ""

    market_names = {"SA": "السوق السعودي (تداول)", "EG": "البورصة المصرية", "AE": "السوق الإماراتي (DFM/ADX)"}
    currency_names = {"SAR": "ريال سعودي", "EGP": "جنيه مصري", "AED": "درهم إماراتي"}

    market_label = market_names.get(data.get("market"), data.get("market"))
    currency = data.get("currency", "AED")
    currency_label = currency_names.get(currency, currency)

    lines = [
        f"\n\n## 📊 LOCAL MARKET DATA — {ticker}",
        f"**السوق:** {market_label} | **العملة:** {currency_label}",
    ]

    # السعر
    price = data.get("price")
    if price:
        chg = data.get("change_pct", 0) or 0
        chg_emoji = "📈" if chg > 0 else "📉" if chg < 0 else "➡️"
        lines.append(f"**آخر سعر:** {price:,.2f} {currency} {chg_emoji} ({chg:+.2f}%) — {data.get('date', '')}")
        if data.get("high") and data.get("low"):
            lines.append(f"**نطاق اليوم:** {data['low']:,.2f} — {data['high']:,.2f} {currency}")
        if data.get("volume"):
            lines.append(f"**حجم التداول:** {int(data['volume']):,}")

    # Fundamentals
    fund = data.get("fundamentals", {})
    if fund:
        lines.append("\n**البيانات الأساسية (Fundamentals):**")
        if fund.get("market_cap"):
            mc = fund["market_cap"]
            mc_str = f"{mc/1e9:.1f}B" if mc > 1e9 else f"{mc/1e6:.1f}M"
            lines.append(f"- Market Cap: {mc_str} {currency}")
        if fund.get("pe_ratio"):
            lines.append(f"- P/E Ratio: {fund['pe_ratio']:.1f}x")
        if fund.get("beta"):
            lines.append(f"- Beta: {fund['beta']:.2f}")
        if fund.get("revenue"):
            rev = fund["revenue"]
            rev_str = f"{rev/1e9:.1f}B" if rev > 1e9 else f"{rev/1e6:.1f}M"
            lines.append(f"- Revenue: {rev_str} {currency}")

        # Signals
        if fund.get("signal_daily"):
            lines.append(f"\n**إشارات تقنية (Investing.com):**")
            lines.append(f"- يومي: {fund.get('signal_daily', 'N/A')}")
            lines.append(f"- أسبوعي: {fund.get('signal_weekly', 'N/A')}")
            lines.append(f"- شهري: {fund.get('signal_monthly', 'N/A')}")

    # Historical
    hist = data.get("historical", {})
    if hist:
        lines.append("\n**الأداء التاريخي (آخر سنة):**")
        if hist.get("total_return_1y") is not None:
            ret = hist["total_return_1y"]
            emoji = "📈" if ret > 0 else "📉"
            lines.append(f"- العائد: {ret:+.1f}% {emoji}")
        if hist.get("volatility"):
            lines.append(f"- التذبذب السنوي: {hist['volatility']:.1f}%")
        if hist.get("sharpe"):
            lines.append(f"- Sharpe Ratio: {hist['sharpe']:.2f}")
        if hist.get("high_52w") and hist.get("low_52w"):
            lines.append(f"- نطاق 52 أسبوع: {hist['low_52w']:,.2f} — {hist['high_52w']:,.2f} {currency}")

    _src_tag = data.get("price_source") or "EisaX Local Data Engine"
    _snap_ts = data.get("snapshot_ts") or ""
    if _snap_ts:
        lines.append(f"\n*المصدر: {_src_tag} | snapshot: {_snap_ts}*")
    else:
        lines.append(f"\n*المصدر: {_src_tag}*")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "EMAAR.DU"
    print(build_local_prompt_injection(ticker))