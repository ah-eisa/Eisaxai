from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.getenv("CLAWDBOT_CACHE_DIR", ".cache")) / "prices"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Market Detection ─────────────────────────────────────────────────────────

def _detect_market(ticker: str):
    """يكتشف السوق من الـ ticker"""
    t = ticker.upper()
    if t.endswith(".SR") or t.endswith("^TASI"): return "SA"
    if t.endswith(".CA") or t.endswith("^EGX30"): return "EG"
    if t.endswith(".DU") or t.endswith(".AE"): return "AE"
    return "US"

def _is_local(ticker: str) -> bool:
    return _detect_market(ticker) in ("SA", "EG", "AE")

# ─── Engine Import (lazy) ─────────────────────────────────────────────────────

def _get_engine():
    try:
        from core.market_data_engine import get_stock_data
        return get_stock_data
    except Exception as e:
        logger.warning(f"market_data_engine not available: {e}")
        return None

# ─── Main get_prices ──────────────────────────────────────────────────────────

def get_prices(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    يجيب Close prices للـ tickers.
    - للأسواق المحلية (SA/EG/AE): يستخدم market_data_engine (cache-first)
    - للأسواق الأجنبية (US...): يستخدم yfinance
    """
    tickers = [t.upper().strip() for t in tickers if t and t.strip()]
    if not tickers:
        raise ValueError("tickers list is empty")

    local  = [t for t in tickers if _is_local(t)]
    foreign = [t for t in tickers if not _is_local(t)]

    frames = {}

    # ─── Local Markets ────────────────────────────────────────────────────────
    if local:
        engine = _get_engine()
        if engine:
            for ticker in local:
                market = _detect_market(ticker)
                try:
                    df = engine(ticker, market, period="5y", force_refresh=force_refresh)
                    if df is not None and not df.empty:
                        series = df["Close"].copy()
                        series.index = pd.to_datetime(series.index)
                        if start:
                            series = series[series.index >= pd.to_datetime(start)]
                        if end:
                            series = series[series.index <= pd.to_datetime(end)]
                        frames[ticker] = series
                except Exception as e:
                    logger.error(f"Engine error for {ticker}: {e}")
        else:
            # fallback لـ yfinance لو الـ engine مش موجود
            foreign.extend(local)

    # ─── Foreign Markets (yfinance) ─────────────────────────────────────────
    if foreign:
        try:
            # group_by="ticker" gives (ticker, field) MultiIndex — most robust
            px = yf.download(
                tickers=foreign, start=start, end=end,
                auto_adjust=True, group_by="ticker",
                progress=False, threads=False,
            )
            close = None

            if px is None or px.empty:
                logger.error(f"yfinance returned empty DataFrame for: {foreign}")
            elif isinstance(px.columns, pd.MultiIndex):
                lvl0 = [str(v).upper() for v in px.columns.get_level_values(0)]
                lvl1 = [str(v).upper() for v in px.columns.get_level_values(1)]

                if "CLOSE" in lvl1:
                    # (ticker, field) order — yfinance >= 0.2.38 group_by="ticker"
                    close = px.xs("Close", axis=1, level=1).copy()
                elif "CLOSE" in lvl0:
                    # (field, ticker) order — older yfinance group_by="column"
                    close = px.xs("Close", axis=1, level=0).copy()
                else:
                    # Last resort: pick columns whose second-level label is Close
                    cols = [(t, f) for t, f in px.columns if str(f).lower() == "close"]
                    if cols:
                        close = px[[c for c in px.columns if str(c[1]).lower() == "close"]].copy()
                        close.columns = [str(c[0]).upper().strip() for c in close.columns]
                    else:
                        logger.error(f"Cannot find Close level in MultiIndex columns: {px.columns.tolist()[:8]}")

                if close is not None and not isinstance(close.columns[0], str):
                    close.columns = [str(c).upper().strip() for c in close.columns]
                elif close is not None:
                    close.columns = [str(c).upper().strip() for c in close.columns]

            else:
                # Single ticker → flat columns
                if "Close" in px.columns:
                    close = px[["Close"]].copy()
                    close.columns = [foreign[0].upper()]
                else:
                    close = px.copy()
                    close.columns = [str(c).upper().strip() for c in close.columns]

            if close is not None:
                close = close.dropna(how="all", axis=1)
                for col in close.columns:
                    # Only store columns that map to one of the requested tickers
                    col_up = col.upper()
                    if any(col_up == t.upper() for t in foreign):
                        frames[col_up] = close[col]
                        logger.info(f"[get_prices] loaded {col_up}: {len(close[col])} rows")
                if not frames and not local:
                    logger.error(f"yfinance: no matching columns found. Got: {list(close.columns)}, wanted: {foreign}")
        except Exception as e:
            logger.error(f"yfinance error: {e}", exc_info=True)

    if not frames:
        raise ValueError(f"No price data found for: {tickers}")

    result = pd.DataFrame(frames)
    result = result.sort_index().dropna(how="all").ffill().dropna(how="any")
    result.columns = [str(c).upper().strip() for c in result.columns]
    return result


def to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    prices = prices.sort_index()
    rets = np.log(prices / prices.shift(1)) if log else prices.pct_change()
    return rets.dropna()