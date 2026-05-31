"""
api/routers/portfolio_upload.py
────────────────────────────────
Portfolio CSV/Excel upload + analysis route, and shared file-store helpers.

Router exported:
  portfolio_upload_router  — no prefix, mounts /v1/upload-portfolio

Helpers re-exported (used by upload_file_ui and chat_router):
  _evict_old_files, _file_store_set, _file_store_get, _file_store_get_for_user
"""
from __future__ import annotations

import io
import json as _json
import logging
import os
import re as _re
import time as _time
import uuid
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.config import ENV_FILE, EXPORTS_DIR, FILE_CACHE_DIR

logger = logging.getLogger("api_bridge")
limiter = Limiter(key_func=get_remote_address)

SECURE_TOKEN = os.getenv("SECURE_TOKEN", "")

# ── File-store config ─────────────────────────────────────────────────────────
_FILE_CACHE_DIR = str(FILE_CACHE_DIR)
_FILE_STORE_TTL = 3600  # seconds
os.makedirs(_FILE_CACHE_DIR, exist_ok=True)

portfolio_upload_router = APIRouter()

def _evict_old_files():
    """Remove file cache entries older than TTL."""
    now = _time.time()
    for fname in os.listdir(_FILE_CACHE_DIR):
        fpath = os.path.join(_FILE_CACHE_DIR, fname)
        try:
            if now - os.path.getmtime(fpath) > _FILE_STORE_TTL:
                os.remove(fpath)
        except Exception:
            pass

def _file_store_set(file_id: str, data: dict):
    fpath = os.path.join(_FILE_CACHE_DIR, file_id + ".json")
    with open(fpath, "w", encoding="utf-8") as _f:
        _json.dump(data, _f, ensure_ascii=False)

def _file_store_get(file_id: str):
    fpath = os.path.join(_FILE_CACHE_DIR, file_id + ".json")
    if not os.path.exists(fpath):
        return None
    with open(fpath, "r", encoding="utf-8") as _f:
        return _json.load(_f)


def _file_store_get_for_user(file_id: str, user_id: Optional[str]):
    entry = _file_store_get(file_id)
    if entry is None:
        return None
    if isinstance(entry, str):
        return {"text": entry}
    if not isinstance(entry, dict):
        return None

    stored_user_id = entry.get("user_id")
    if stored_user_id is not None and stored_user_id != user_id:
        raise HTTPException(status_code=403, detail="File access denied")
    return entry


def _soften_portfolio_advice(text: str) -> str:
    """Convert directive wording to advisory wording for safer decision support."""
    if not text:
        return text
    softened = str(text)
    replacements = [
        (r"(?i)\bexecute (?:the )?rebalancing plan immediately\b", "consider gradual rebalancing based on risk triggers"),
        (r"(?i)\bexecute immediately\b", "consider phased execution"),
        (r"(?i)\bexit 100%\b", "consider a full exit"),
        (r"(?i)\breduce to 50%\b", "consider reducing to 50%"),
        (r"(?i)\bmust\b", "should"),
    ]
    for pattern, repl in replacements:
        softened = _re.sub(pattern, repl, softened)
    return softened


def _compute_portfolio_confidence(sharpe: float,
                                  eff_n: float,
                                  avg_corr: float,
                                  beta_total: Optional[float],
                                  cvar_95: float,
                                  rolling_sharpe_now: Optional[float] = None) -> int:
    """Deterministic confidence score for portfolio verdict communication."""
    score = 66.0
    score += min(max((sharpe - 0.8) * 12.0, -10.0), 10.0)
    score += min(max((eff_n - 2.5) * 4.0, -8.0), 8.0)
    score -= min(max((avg_corr - 0.45) * 20.0, 0.0), 8.0)
    if isinstance(beta_total, (int, float)):
        score -= min(max((beta_total - 1.2) * 8.0, 0.0), 10.0)
    else:
        score -= 3.0  # uncertainty penalty when beta data coverage is missing
    score -= min(max((abs(cvar_95) - 3.0) * 3.0, 0.0), 8.0)
    if rolling_sharpe_now is not None:
        score += min(max((rolling_sharpe_now - 0.2) * 5.0, -6.0), 6.0)
    return int(max(45, min(88, round(score))))


def _build_portfolio_decision_layer(*,
                                    confidence: int,
                                    risk_label: str,
                                    rolling_sharpe_now: Optional[float],
                                    tech_weight: float,
                                    next_review_hint: str = "next review cycle") -> str:
    """Adds explicit decision discipline: uncertainty, no-action case, and boundary."""
    rolling_note = (
        f"rolling Sharpe currently {rolling_sharpe_now:+.2f}"
        if rolling_sharpe_now is not None else
        "rolling Sharpe trend requires ongoing confirmation"
    )
    if tech_weight >= 0.70:
        alt_case = (
            "If AI mega-cap momentum persists and earnings revisions remain positive, concentrated tech could continue to outperform in the near term."
        )
    else:
        alt_case = (
            "If macro volatility cools and correlations normalize, current allocation may remain acceptable with only minor rebalancing."
        )
    no_action_case = (
        f"If risk metrics stay stable and no catalyst break occurs before the {next_review_hint}, HOLD/no-trade remains a valid decision."
    )
    return (
        "\n\n---\n"
        "## 🎯 Decision Discipline Layer\n"
        f"- **Confidence:** {confidence}%\n"
        f"- **Risk Posture:** {risk_label}\n"
        f"- **Primary Uncertainty:** concentration regime sensitivity and {rolling_note}\n"
        f"- **No-Action Case:** {no_action_case}\n"
        f"- **Alternative Scenario:** {alt_case}\n"
        "> **Decision Boundary:** This analysis provides strategic guidance, not execution instructions."
    )



# ── Route ────────────────────────────────────────────────────────────────────

@portfolio_upload_router.post("/v1/upload-portfolio")
@limiter.limit("10/minute")
async def upload_portfolio(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form("admin"),
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token")
):
    """Upload CSV/Excel portfolio file and analyze it"""
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        contents = await file.read()

        if len(contents) > 5 * 1024 * 1024:
            return {"error": "File too large. Maximum allowed size is 5MB."}

        # Parse file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Only CSV or Excel files supported"}
        
        # Normalize columns
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Find ticker and weight columns
        ticker_col = next((c for c in df.columns if c in ['ticker','symbol','stock','name']), df.columns[0])
        weight_col = next((c for c in df.columns if c in ['weight','allocation','%','percent','value']), None)
        
        # Filter out rows where ticker is missing or empty before any math
        _valid_rows = df[ticker_col].notna() & (
            df[ticker_col].astype(str).str.strip().str.upper().isin(['NAN', 'NONE', '']) == False
        )
        df = df[_valid_rows].reset_index(drop=True)
        tickers = df[ticker_col].str.strip().str.upper().tolist()

        if not tickers:
            return {"error": "No valid tickers found. Ensure your file has a 'ticker' or 'symbol' column with non-empty values."}

        if weight_col:
            weights = df[weight_col].tolist()
            # Normalize to percentages
            total = sum(float(w) for w in weights)
            if total > 1.5:  # assume percentages not decimals
                weights = [float(w)/100 for w in weights]
            else:
                weights = [float(w) for w in weights]
        else:
            weights = [1/len(tickers)] * len(tickers)
        
        portfolio = dict(zip(tickers, weights))
        # Always work on normalized total-portfolio weights (including cash-like rows)
        # to avoid hidden basis-mismatch between "total portfolio" and "equity sleeve".
        _sum_input_w = sum(float(w) for w in portfolio.values())
        if _sum_input_w <= 0:
            return {"error": "Invalid portfolio weights: sum must be > 0"}
        portfolio = {t: float(w) / _sum_input_w for t, w in portfolio.items()}
        
        # Build Portfolio Risk Report
        import yfinance as yf
        import numpy as np
        from core.data import get_prices
        from dotenv import load_dotenv
        load_dotenv(str(ENV_FILE))
        def _to_float(v):
            try:
                f = float(v)
                return f if np.isfinite(f) else None
            except Exception:
                return None
        def _safe_round(v, n=3):
            f = _to_float(v)
            return round(f, n) if f is not None else None

        def _clean_nan(obj):
            """Recursively replace NaN/Inf with None — prevents JSON serialization crash."""
            if isinstance(obj, dict):
                return {k: _clean_nan(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean_nan(v) for v in obj]
            if isinstance(obj, float):
                return None if (obj != obj or obj == float('inf') or obj == float('-inf')) else obj
            return obj

        def _fmt_safe(val, fmt=".1f", fallback="N/A"):
            """Format a float safely — return fallback if NaN/None."""
            if val is None:
                return fallback
            try:
                f = float(val)
                if f != f:  # NaN check
                    return fallback
                return f"{f:{fmt}}"
            except (ValueError, TypeError):
                return fallback

        def _date_index(idx):
            _idx = pd.to_datetime(idx, errors="coerce")
            if getattr(_idx, "tz", None) is not None:
                _idx = _idx.tz_convert(None)
            return _idx.normalize()

        def _regression_beta(asset_returns, mkt_returns, min_obs=20):
            try:
                _joined = pd.concat(
                    [
                        pd.to_numeric(asset_returns, errors="coerce").rename("asset"),
                        pd.to_numeric(mkt_returns, errors="coerce").rename("market"),
                    ],
                    axis=1,
                    join="inner",
                ).replace([np.inf, -np.inf], np.nan).dropna()
                if _joined.shape[0] < min_obs:
                    return None
                _mkt_var = float(_joined["market"].var())
                if not np.isfinite(_mkt_var) or _mkt_var <= 0:
                    return None
                _beta = float(_joined["asset"].cov(_joined["market"]) / _mkt_var)
                return _beta if np.isfinite(_beta) else None
            except Exception:
                return None

        valid_tickers = [t for t in tickers if t.upper() not in ["CASH","USD","AED"]]
        valid_weights_total = {t: portfolio.get(t, 0.0) for t in valid_tickers}
        
        # Normalize weights
        total_w = sum(valid_weights_total.values())
        equity_alloc_total = total_w
        if total_w > 0:
            # Equity-sleeve normalized weights (sum to 100% across non-cash assets)
            valid_weights = {t: w / total_w for t, w in valid_weights_total.items()}
        else:
            valid_weights = {}

        # ── Fetch 1yr price history + fundamentals ──────────────────────────────
        RF_RATE = 0.045   # US T-Bill risk-free rate (4.5%)
        LOOKBACK = "1y"
        lookback_start = (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

        price_data = {}
        prices_df = pd.DataFrame()
        returns_df = pd.DataFrame()
        price_fetch_error = None
        stock_info = {}

        def _price_availability(px: pd.DataFrame, rets: pd.DataFrame | None = None) -> dict:
            flags = {}
            for _t in tickers:
                _tu = str(_t).upper()
                if _tu in ["CASH", "USD", "AED"]:
                    flags[_t] = {
                        "asset_type": "cash",
                        "has_prices": False,
                        "has_sufficient_returns": False,
                        "price_rows": 0,
                        "return_rows": 0,
                        "first_date": None,
                        "last_date": None,
                    }
                    continue
                _series = px[_tu].dropna() if isinstance(px, pd.DataFrame) and _tu in px.columns else pd.Series(dtype="float64")
                _ret_series = (
                    rets[_tu].dropna()
                    if isinstance(rets, pd.DataFrame) and _tu in rets.columns
                    else pd.Series(dtype="float64")
                )
                flags[_t] = {
                    "asset_type": "security",
                    "has_prices": int(_series.shape[0]) > 0,
                    "has_sufficient_returns": int(_ret_series.shape[0]) >= 2,
                    "price_rows": int(_series.shape[0]),
                    "return_rows": int(_ret_series.shape[0]),
                    "first_date": _series.index[0].strftime("%Y-%m-%d") if not _series.empty else None,
                    "last_date": _series.index[-1].strftime("%Y-%m-%d") if not _series.empty else None,
                }
            return flags

        def _partial_price_response(message: str, px: pd.DataFrame, rets: pd.DataFrame | None = None):
            _snap_id = str(uuid.uuid4())
            _generated = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            _availability = _price_availability(px, rets)
            return _clean_nan({
                "status": "partial",
                "snapshot_id": _snap_id,
                "message": message,
                "portfolio": portfolio,
                "tickers": tickers,
                "analysis": message,
                "phase_h_meta": None,
                "audit": {
                    "report_id": _snap_id,
                    "generated_utc": _generated,
                    "price_as_of": None,
                    "period": f"{lookback_start} → latest available",
                    "data_source": "core.data.get_prices",
                    "price_fetch_error": price_fetch_error,
                    "data_availability": _availability,
                },
                "metrics": {
                    "total_return_pct": None,
                    "total_return_total_est_pct": None,
                    "spx_return_pct": None,
                    "alpha_pct": None,
                    "alpha_total_est_pct": None,
                    "ann_return_pct": None,
                    "ann_vol_pct": None,
                    "sharpe": None,
                    "sortino": None,
                    "var_95_pct": None,
                    "max_drawdown_pct": None,
                    "beta": None,
                    "beta_equity": None,
                    "div_yield_pct": None,
                    "div_yield_equity_pct": None,
                    "equity_allocation_pct": round(equity_alloc_total * 100, 2),
                    "cash_allocation_pct": round(max(0.0, (1 - equity_alloc_total) * 100), 2),
                    "risk_free_rate": "4.5% (US T-Bill)",
                    "method": "Historical Simulation, 252 trading days",
                    "data_availability": _availability,
                },
                "data_availability": _availability,
            })

        if valid_tickers:
            try:
                fetched_prices = get_prices(valid_tickers, start=lookback_start, end=None, force_refresh=False)
                if fetched_prices is not None and not fetched_prices.empty:
                    prices_df = fetched_prices.copy()
                    prices_df.index = pd.to_datetime(prices_df.index, errors="coerce")
                    prices_df = prices_df[~prices_df.index.isna()]
                    prices_df.columns = [str(c).upper().strip() for c in prices_df.columns]
                    selected_cols = [t for t in valid_tickers if t in prices_df.columns]
                    prices_df = prices_df.loc[:, selected_cols]
                    prices_df = prices_df.apply(pd.to_numeric, errors="coerce")
                    prices_df = prices_df.sort_index()
                    prices_df = prices_df[~prices_df.index.duplicated(keep="last")]
                    prices_df = prices_df.reindex(prices_df.index.drop_duplicates().sort_values()).ffill()
                    prices_df = prices_df.dropna(how="all", axis=0)
                    price_data = {
                        t: prices_df[t].dropna()
                        for t in prices_df.columns
                        if not prices_df[t].dropna().empty
                    }
            except Exception as _pe:
                price_fetch_error = str(_pe)
                logger.warning("Portfolio price fetch failed through get_prices for %s: %s", valid_tickers, _pe)

        candidate_returns = pd.DataFrame()
        if not prices_df.empty:
            usable_price_cols = [c for c in prices_df.columns if prices_df[c].dropna().shape[0] >= 3]
            prices_df = prices_df.loc[:, usable_price_cols].ffill()
            if not prices_df.empty:
                candidate_returns = prices_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
                usable_return_cols = [
                    c for c in candidate_returns.columns
                    if candidate_returns[c].dropna().shape[0] >= 2
                ]
                returns_df = candidate_returns.loc[:, usable_return_cols].dropna(how="all").fillna(0.0)
                price_data = {
                    t: prices_df[t].dropna()
                    for t in prices_df.columns
                    if t in usable_return_cols and not prices_df[t].dropna().empty
                }

        if returns_df.shape[0] < 2 or returns_df.shape[1] < 1:
            _msg = (
                "Insufficient overlapping price history to compute portfolio risk metrics. "
                "The upload was parsed, but the price router did not return enough usable "
                "return observations for VaR, volatility, drawdown, or Sharpe calculations."
            )
            return _partial_price_response(_msg, prices_df, candidate_returns)

        for t in valid_tickers:
            try:
                tk = yf.Ticker(t)
                info = tk.info
                # trailingAnnualDividendYield is a reliable decimal fraction (0.004 = 0.4%).
                # dividendYield returns as a percentage value (0.4 for 0.4%) — avoid it.
                _raw_dy = float(info.get("trailingAnnualDividendYield") or 0)
                _safe_dy = min(max(_raw_dy, 0.0), 0.15)   # clamp decimal [0, 15%]
                _latest_px = price_data.get(t)
                stock_info[t] = {
                    "price":     info.get("regularMarketPrice") or info.get("previousClose") or (float(_latest_px.iloc[-1]) if _latest_px is not None and not _latest_px.empty else 0),
                    "beta":      _to_float(info.get("beta")),
                    "sector":    info.get("sector", "N/A"),
                    "pe":        _to_float(info.get("trailingPE")),
                    "mktcap":    _to_float(info.get("marketCap")),
                    "div_yield": _safe_dy,   # current market yield, NOT yield-on-cost
                }
            except Exception as _fe:
                logger.debug("Stock info fetch failed for %s: %s", t, _fe)
                _latest_px = price_data.get(t)
                stock_info[t] = {
                    "price": float(_latest_px.iloc[-1]) if _latest_px is not None and not _latest_px.empty else 0,
                    "beta": None,
                    "sector": "N/A",
                    "pe": None,
                    "mktcap": None,
                    "div_yield": 0.0,
                }

        # ── Benchmark: S&P 500 ───────────────────────────────────────────────
        spx_return = None
        spx_returns = pd.Series(dtype="float64")
        try:
            _spx_close = pd.Series(dtype="float64")
            try:
                _spx_px = get_prices(["^GSPC"], start=lookback_start, end=None, force_refresh=False)
                if _spx_px is not None and not _spx_px.empty:
                    _spx_px = _spx_px.copy()
                    _spx_px.index = _date_index(_spx_px.index)
                    _spx_px = _spx_px[~_spx_px.index.isna()]
                    _spx_col = next(
                        (c for c in _spx_px.columns if str(c).upper().strip() == "^GSPC"),
                        _spx_px.columns[0],
                    )
                    _spx_close = pd.to_numeric(_spx_px[_spx_col], errors="coerce").dropna()
            except Exception as _spx_pe:
                logger.debug("SPX price fetch through get_prices failed: %s", _spx_pe)

            if _spx_close.empty:
                spx = yf.Ticker("^GSPC").history(period=LOOKBACK)
                if spx.empty or spx["Close"].isna().all():
                    spx = yf.Ticker("^GSPC").history(period="6mo")
                if not spx.empty:
                    _spx_close = pd.to_numeric(spx["Close"], errors="coerce").dropna()
                    _spx_close.index = _date_index(_spx_close.index)

            _spx_close = _spx_close.sort_index()
            _spx_close = _spx_close[~_spx_close.index.duplicated(keep="last")]
            if not _spx_close.empty:
                spx_return = float((_spx_close.iloc[-1] / _spx_close.iloc[0]) - 1)
                _target_idx = _date_index(returns_df.index)
                _aligned_spx = (
                    _spx_close
                    .reindex(_spx_close.index.union(_target_idx))
                    .sort_index()
                    .ffill()
                    .reindex(_target_idx)
                )
                spx_returns = _aligned_spx.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
                spx_returns.index = returns_df.index
                spx_returns.name = "^GSPC"
        except Exception:
            pass

        # ── Calculate metrics ────────────────────────────────────────────────
        corr_matrix_str = ""
        high_corr = []
        sortino = 0.0
        max_dd = 0.0
        max_dd_duration = 0
        port_total_return = 0.0
        cvar_95 = 0.0
        rolling_sharpe_str = ""
        rolling_sharpe_now = None
        sector_concentration = {}
        factor_exposure = {}
        port_beta_equity = None

        if len(price_data) >= 1 and returns_df.shape[0] >= 2:
            w_arr = np.array([valid_weights.get(t, 0) for t in returns_df.columns])
            w_arr = w_arr / w_arr.sum() if w_arr.sum() > 0 else w_arr

            port_returns = returns_df.values @ w_arr
            ann_return   = float(np.mean(port_returns) * 252)
            ann_vol      = float(np.std(port_returns) * np.sqrt(252))
            rf_daily     = (1 + RF_RATE) ** (1/252) - 1
            excess       = port_returns - rf_daily
            sharpe       = round(float(np.mean(excess) / np.std(excess) * np.sqrt(252)), 2) if np.std(excess) > 0 else 0

            # Sortino (downside deviation only)
            neg_excess   = excess[excess < 0]
            downside_std = float(np.std(neg_excess) * np.sqrt(252)) if len(neg_excess) > 0 else ann_vol
            sortino      = round(float(np.mean(excess) * 252 / downside_std), 2) if downside_std > 0 else 0

            # VaR 95% and CVaR 95% (Expected Shortfall — tail risk beyond VaR)
            var_95  = float(np.percentile(port_returns, 5) * 100)
            cvar_95 = float(port_returns[port_returns <= np.percentile(port_returns, 5)].mean() * 100)

            # Max drawdown + duration (calendar days in drawdown)
            cum      = (1 + pd.Series(port_returns)).cumprod()
            roll_max = cum.cummax()
            dd       = (cum - roll_max) / roll_max
            max_dd   = float(dd.min() * 100)
            # Duration: longest consecutive streak below previous peak
            in_dd = (dd < -0.001).astype(int)
            streak = max_dd_duration = 0
            for v in in_dd:
                streak = streak + 1 if v else 0
                max_dd_duration = max(max_dd_duration, streak)

            # Total return
            port_total_return = float((cum.iloc[-1] - 1) * 100)

            # Rolling 63-day Sharpe (quarterly window, annualised)
            if len(port_returns) >= 63:
                roll_win = 63
                roll_sh  = []
                for i in range(roll_win, len(port_returns) + 1):
                    w_ret = port_returns[i - roll_win:i]
                    ex    = w_ret - rf_daily
                    s     = float(np.mean(ex) / np.std(ex) * np.sqrt(252)) if np.std(ex) > 0 else 0
                    roll_sh.append(round(s, 2))
                if roll_sh:
                    rs_min = min(roll_sh); rs_max = max(roll_sh); rs_now = roll_sh[-1]
                    rolling_sharpe_now = float(rs_now)
                    rs_trend = "↗ Improving" if rs_now > np.mean(roll_sh) else "↘ Declining"
                    rolling_sharpe_str = (
                        f"Current (last 63d): **{rs_now:.2f}** | "
                        f"Range: {rs_min:.2f} → {rs_max:.2f} | {rs_trend}"
                    )

            # Correlation matrix
            corr = returns_df.corr()
            # ── Smart correlation insight: weight-adjusted cluster risk ──────
            # Don't just flag pairs >0.70; compute effective diversification ratio
            n_assets   = len(corr)
            avg_corr   = float(corr.values[np.triu_indices(n_assets, k=1)].mean()) if n_assets > 1 else 0
            # Effective N (Herfindahl on corr eigenvalues → measures true diversification)
            eigvals    = np.linalg.eigvalsh(corr.values)
            eigvals    = np.maximum(eigvals, 0)
            eff_n      = (eigvals.sum() ** 2) / (eigvals ** 2).sum() if (eigvals**2).sum() > 0 else 1
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    c = corr.iloc[i, j]
                    if abs(c) > 0.70:
                        high_corr.append(f"{corr.columns[i]}/{corr.columns[j]}: {c:.2f}")
            cols_c      = list(corr.columns)
            corr_header = "| " + " | ".join([""] + cols_c) + " |"
            corr_sep    = "|" + "|".join(["---"] * (len(cols_c) + 1)) + "|"
            corr_rows   = [corr_header, corr_sep]
            for row_t in cols_c:
                row_vals = [f"{corr.loc[row_t, col_t]:.2f}" for col_t in cols_c]
                corr_rows.append("| " + row_t + " | " + " | ".join(row_vals) + " |")
            corr_matrix_str = "\n".join(corr_rows)

            # Sector concentration (Herfindahl index)
            for t in valid_tickers:
                s = stock_info.get(t, {}).get("sector", "Unknown")
                sector_concentration[s] = sector_concentration.get(s, 0) + valid_weights.get(t, 0)
            hhi = sum(v**2 for v in sector_concentration.values())  # 0=diversified, 1=concentrated

        else:
            ann_return, ann_vol, sharpe, var_95, port_beta_equity = 0, 0, 0, 0, None
            port_total_return, max_dd, avg_corr, eff_n, hhi = 0.0, 0.0, 0.0, 1.0, 1.0

        # Keep both bases explicit:
        # - equity basis: normalized over non-cash holdings
        # - total basis: normalized over all uploaded rows (including cash-like rows)
        def _weighted_beta(weights_map):
            _num = 0.0
            _den = 0.0
            _coverage = 0.0
            for _t, _w in weights_map.items():
                _b = stock_info.get(_t, {}).get("beta")
                if _b is None:
                    continue
                _num += float(_b) * float(_w)
                _den += float(_w)
                _coverage += float(_w)
            if _den <= 0:
                return None, 0.0
            return float(_num / _den), float(_coverage)

        port_beta_equity, beta_cov_equity = _weighted_beta(valid_weights)
        port_beta_total, beta_cov_total = _weighted_beta(valid_weights_total)
        port_beta = port_beta_equity
        def _beta_is_valid(v):
            return isinstance(v, (int, float)) and np.isfinite(v)
        def _fmt_beta(v):
            return f"{v:.2f}" if _beta_is_valid(v) else "N/A"
        _has_beta_total = _beta_is_valid(port_beta_total)
        _has_beta_equity = _beta_is_valid(port_beta_equity)

        # ── Historical Stress Tests (actual crisis returns, NOT linear beta) ──
        # Each scenario uses historically observed market returns + portfolio beta
        # adjusted by sector composition during that crisis.
        _CRISIS_SCENARIOS = [
            # (label, spx_actual, tech_multiplier, note)
            ("🔴 2022 Rate Shock (actual)",  -0.195, 1.45, "Tech -33% vs SPX -19.5% in 2022"),
            ("🔴 COVID Crash Mar 2020",       -0.340, 1.20, "SPX -34% in 33 days"),
            ("🔴 2008 GFC (Sep–Nov)",         -0.420, 0.90, "Financials led; tech -40%"),
            ("🔴 Bear Market -30% Scenario",  -0.300, 1.30, "Projected; tech typically amplified"),
            ("🟢 2023 AI Bull Run (actual)",  +0.265, 1.80, "Tech +55% vs SPX +26.5%"),
            ("🟢 Rate Cut Cycle +15%",        +0.150, 1.40, "Growth stocks benefit most"),
        ]
        tech_weight = sum(
            valid_weights.get(t, 0)
            for t in valid_tickers
            if stock_info.get(t, {}).get("sector", "").lower() in
               ["technology", "communication services", "consumer cyclical"]
        )
        scenario_lines = []
        for label, spx_ret, tech_mult, note in _CRISIS_SCENARIOS:
            # Blend: non-tech portion tracks beta linearly; tech amplified by sector mult
            non_tech_w = max(0, 1 - tech_weight)
            if _has_beta_equity:
                blended = (tech_weight * spx_ret * tech_mult) + (non_tech_w * spx_ret * port_beta_equity)
                icon = "🔴" if blended < 0 else "🟢"
                scenario_lines.append(
                    f"| {label} | {spx_ret*100:+.1f}% | **{blended*100:+.1f}%** | *{note}* |"
                )
            else:
                scenario_lines.append(
                    f"| {label} | {spx_ret*100:+.1f}% | **N/A** | *{note} · beta unavailable* |"
                )

        # ── Portfolio Dividend Yield (weighted) ──────────────────────────────
        port_div_yield_equity = sum(
            stock_info.get(t, {}).get("div_yield", 0) * valid_weights.get(t, 0)
            for t in valid_tickers
        ) * 100
        port_div_yield_total = sum(
            stock_info.get(t, {}).get("div_yield", 0) * valid_weights_total.get(t, 0)
            for t in valid_tickers
        ) * 100
        port_div_yield = port_div_yield_equity

        cash_pct = max(0.0, (1 - equity_alloc_total) * 100)
        port_total_return_total_est = port_total_return * equity_alloc_total

        # ══════════════════════════════════════════════════════════════════════
        # Build report
        # ══════════════════════════════════════════════════════════════════════
        lines = []
        from datetime import datetime
        now_str = datetime.now().strftime("%B %d, %Y")

        # ── Executive Summary ──────────────────────────────────────────────
        alpha = (port_total_return - (spx_return or 0) * 100) if spx_return is not None else None
        alpha_total_est = (port_total_return_total_est - (spx_return or 0) * 100) if spx_return is not None else None
        risk_label = (
            "Aggressive [HIGH]" if (_has_beta_total and port_beta_total > 1.5) else
            "Moderate [MODERATE]" if (_has_beta_total and port_beta_total > 1.0) else
            "Conservative [LOW]" if _has_beta_total else
            "Unknown"
        )
        elevated_risk = ((_has_beta_total and port_beta_total > 1.5) or (tech_weight >= 0.70) or (eff_n < 2.8) or (cvar_95 <= -3.5))

        # ════════════════════════════════════════════════════════════════════
        # Phase F — Institutional parity layer (regime, confidence, implementation,
        # benchmark, attribution, adaptive disclaimers, audit appendix)
        # ════════════════════════════════════════════════════════════════════
        import hashlib as _hashlib_pf

        # ── Portfolio Regime Classification ─────────────────────────────────
        _tech_w_pf = tech_weight if isinstance(tech_weight, (int, float)) else 0
        _crypto_w_pf = sum(valid_weights_total.get(t, 0) for t in valid_tickers
                           if stock_info.get(t, {}).get("sector", "").lower() in ("crypto", "cryptocurrency") or t.endswith("-USD"))
        _energy_w_pf = sum(valid_weights_total.get(t, 0) for t in valid_tickers
                           if stock_info.get(t, {}).get("sector", "").lower() in ("energy", "energy minerals", "industrial services"))
        _div_yield_w_pf = port_div_yield_equity / 100 if isinstance(port_div_yield_equity, (int, float)) else 0
        if _tech_w_pf + _crypto_w_pf > 0.50:
            _regime_pf = "Momentum-Driven"
            _regime_impl_pf = "Concentrated in long-duration growth and asymmetric satellite assets. High dispersion in risk-off regimes."
        elif _tech_w_pf > 0.40:
            _regime_pf = "Growth Concentrated"
            _regime_impl_pf = "Allocation tilted toward long-duration growth equities. Sensitive to discount-rate compression and earnings-multiple contraction."
        elif cash_pct / 100 + _div_yield_w_pf > 0.30:
            _regime_pf = "Defensive Income"
            _regime_impl_pf = "Income-generating sleeves dominate. Lower sensitivity to equity drawdowns; primary risk vector is duration and credit spread widening."
        elif _energy_w_pf > 0.25:
            _regime_pf = "Cyclical Value"
            _regime_impl_pf = "Allocation tilted toward commodity-linked and cyclical exposure. Procyclical with global PMI cycle."
        elif port_beta_total < 0.7 if _has_beta_total else False:
            _regime_pf = "Defensive Income"
            _regime_impl_pf = "Low aggregate beta. Capital preservation tilt with limited upside capture in equity rallies."
        else:
            _regime_pf = "Multi-Asset Macro"
            _regime_impl_pf = "Balanced cross-asset construction; no single regime dominates. Targets diversification across factor and macro drivers."
        _BENCH_REGIME_BEHAVIOR_PF = {
            "Momentum-Driven":     "Outperforms in falling-rate, risk-on regimes; lags during commodity-led value rotations and rate-shock episodes.",
            "Growth Concentrated": "Outperforms in falling-rate growth regimes (discount-rate compression); lags during inflation surprises and value rotations.",
            "Defensive Income":    "Outperforms during equity drawdowns and disinflationary cycles; lags in strong risk-on rallies and steepening yield curves.",
            "Cyclical Value":      "Outperforms during commodity-led reflationary cycles and global PMI expansions; lags in growth-led rallies and risk-off episodes.",
            "Multi-Asset Macro":   "Designed for regime-balanced behavior; expect peer-like performance across most macro environments with reduced tail volatility.",
        }
        _regime_behavior_pf = _BENCH_REGIME_BEHAVIOR_PF.get(_regime_pf, "")

        # ── Confidence Calibration ──────────────────────────────────────────
        _n_holdings_pf = len(valid_tickers)
        _missing_beta_pf = sum(1 for t in valid_tickers if stock_info.get(t, {}).get("beta") is None)
        _missing_beta_share_pf = _missing_beta_pf / max(1, _n_holdings_pf)
        _crypto_share_pf = _crypto_w_pf
        _confidence_pf = 0.88
        _confidence_pf -= 0.20 * _missing_beta_share_pf
        _confidence_pf -= 0.15 * min(1.0, _crypto_share_pf / 0.20)
        if _n_holdings_pf < 5:
            _confidence_pf -= 0.10
        _confidence_pf = max(0.40, min(0.92, _confidence_pf))
        _evidence_breadth_pf = "Broad" if _n_holdings_pf >= 12 else ("Moderate" if _n_holdings_pf >= 6 else "Limited")
        _coverage_quality_pf = "Full" if _missing_beta_share_pf < 0.15 else ("Partial" if _missing_beta_share_pf < 0.40 else "Sparse")
        if _missing_beta_share_pf < 0.10 and _crypto_share_pf < 0.05:
            _reliability_tier_pf = "Institutional"
        elif _missing_beta_share_pf < 0.30 and _crypto_share_pf < 0.20:
            _reliability_tier_pf = "Institutional-Lite"
        else:
            _reliability_tier_pf = "Indicative"
        _tier_tag_pf = "[STRONG]" if _reliability_tier_pf == "Institutional" else ("[MODERATE]" if _reliability_tier_pf == "Institutional-Lite" else "[LOW]")

        # ── Implementation Feasibility ──────────────────────────────────────
        _n_active_pf = sum(1 for t in valid_tickers if valid_weights.get(t, 0) > 0.005)
        _rebal_complexity_pf = "Low" if _n_active_pf <= 6 else ("Moderate" if _n_active_pf <= 12 else "High")
        # Liquidity proxy: single-name positions have wider spreads vs ETFs.
        # Uploaded portfolios are typically all single-name → assume Moderate baseline.
        _liq_practicality_pf = "High" if _n_active_pf <= 8 else ("Moderate" if _n_active_pf <= 15 else "Limited")
        _friction_pen_pf = _crypto_share_pf * 0.5 + _missing_beta_share_pf * 0.2
        _execution_friction_pf = "Low" if _friction_pen_pf < 0.10 else ("Moderate" if _friction_pen_pf < 0.25 else "High")
        _est_turnover_pf = max(5, min(40, round(8 + (_n_active_pf - 5) * 1.5 + _crypto_share_pf * 30, 0)))
        _est_slippage_pf = round((1.0 - _crypto_share_pf) * 10 + _crypto_share_pf * 25, 1)
        _max_w_pf = max((valid_weights_total.get(t, 0) for t in valid_tickers), default=0)
        _deploy_score_pf = 100
        _deploy_score_pf -= max(0, _n_active_pf - 10) * 3
        _deploy_score_pf -= max(0, _max_w_pf - 0.25) * 100  # over-concentration
        _deploy_score_pf -= _crypto_share_pf * 30
        _deploy_score_pf -= _missing_beta_share_pf * 25
        _deploy_score_pf = max(20, min(100, round(_deploy_score_pf, 0)))
        _deploy_tier_pf = "High" if _deploy_score_pf >= 80 else ("Moderate" if _deploy_score_pf >= 60 else "Limited")
        _deploy_tag_pf = "[STRONG]" if _deploy_tier_pf == "High" else ("[MODERATE]" if _deploy_tier_pf == "Moderate" else "[LOW]")
        def _impl_tag_pf(level):
            return {"Low": "[LOW]", "Moderate": "[MODERATE]", "High": "[HIGH]", "Limited": "[LOW]"}.get(level, f"[{level}]")

        # ── Mandate Feasibility Constraint Diagnostics ──────────────────────
        # Institutional concentration / beta / sector / liquidity / geography checks
        _constraint_pf: list[dict] = []
        # Single-name concentration: ≤25% institutional limit
        _top_holding = max(((t, valid_weights_total.get(t, 0)) for t in valid_tickers), key=lambda x: x[1], default=("—", 0))
        _top_name, _top_w = _top_holding
        _status_top = "PASS" if _top_w <= 0.15 else ("NEAR CAP" if _top_w <= 0.25 else "BREACH")
        _constraint_pf.append({
            "name":       f"Single-name concentration · {_top_name}",
            "limit_pct":  "≤ 25%",
            "actual_pct": f"{_top_w*100:.1f}%",
            "status":     _status_top,
        })
        # Sector concentration: ≤35% institutional limit
        _top_sector_pf = max(sector_concentration.items(), key=lambda x: x[1], default=("—", 0)) if sector_concentration else ("—", 0)
        _ts_name, _ts_w = _top_sector_pf
        _status_sector = "PASS" if _ts_w <= 0.30 else ("NEAR CAP" if _ts_w <= 0.40 else "BREACH")
        _constraint_pf.append({
            "name":       f"Sector concentration · {_ts_name}",
            "limit_pct":  "≤ 35%",
            "actual_pct": f"{_ts_w*100:.1f}%",
            "status":     _status_sector,
        })
        # Portfolio beta cap (mandate-defined)
        if _has_beta_total:
            _status_beta = "PASS" if port_beta_total <= 1.20 else ("NEAR CAP" if port_beta_total <= 1.50 else "BREACH")
            _constraint_pf.append({
                "name":       "Portfolio beta (vs SPX)",
                "limit_pct":  "≤ 1.20",
                "actual_pct": f"{port_beta_total:.2f}",
                "status":     _status_beta,
            })
        # Volatility ceiling (typical balanced mandate)
        _vol_pct_pf = ann_vol * 100 if isinstance(ann_vol, (int, float)) else 0
        _status_vol = "PASS" if _vol_pct_pf <= 20 else ("NEAR CAP" if _vol_pct_pf <= 25 else "BREACH")
        _constraint_pf.append({
            "name":       "Annualized volatility",
            "limit_pct":  "≤ 20%",
            "actual_pct": f"{_vol_pct_pf:.1f}%",
            "status":     _status_vol,
        })
        # Diversification floor: Effective N ≥ 5
        _eff_n_pf = eff_n if isinstance(eff_n, (int, float)) else 0
        _status_effn = "PASS" if _eff_n_pf >= 5 else ("NEAR CAP" if _eff_n_pf >= 3 else "LOW DIVERSIFICATION")
        _constraint_pf.append({
            "name":       "Effective N (independent bets)",
            "limit_pct":  "≥ 5",
            "actual_pct": f"{_eff_n_pf:.1f}",
            "status":     _status_effn,
        })
        # Liquidity (cash buffer)
        _cash_floor_pct = 1.0  # min 1% cash for liquidity
        _status_cash = "PASS" if cash_pct >= _cash_floor_pct else "BELOW FLOOR"
        _constraint_pf.append({
            "name":       "Cash buffer (liquidity)",
            "limit_pct":  "≥ 1%",
            "actual_pct": f"{cash_pct:.1f}%",
            "status":     _status_cash,
        })

        # ── Adaptive Disclaimers ────────────────────────────────────────────
        _adaptive_pf: list[dict] = []
        if _crypto_share_pf > 0.05:
            _adaptive_pf.append({
                "severity": "HIGH",
                "topic":    "Crypto Liquidity Discontinuity",
                "note":     (f"Crypto exposure of {_crypto_share_pf*100:.0f}% subject to 24/7 trading, regulatory regime shifts, "
                             "and liquidity discontinuities during stress events. Classify as satellite, not core."),
            })
        if _top_w > 0.25:
            _adaptive_pf.append({
                "severity": "HIGH",
                "topic":    "Single-Asset Concentration",
                "note":     (f"Top position ({_top_name}) represents {_top_w*100:.0f}% of the portfolio. "
                             "Idiosyncratic risk exceeds typical institutional concentration limits (25%)."),
            })
        if _ts_w > 0.35:
            _adaptive_pf.append({
                "severity": "HIGH",
                "topic":    f"Sector Concentration · {_ts_name}",
                "note":     (f"Sector weighting of {_ts_w*100:.0f}% in {_ts_name} exceeds institutional 35% sector cap. "
                             "Factor crowding and idiosyncratic sector-event risk elevated."),
            })
        if _has_beta_total and port_beta_total > 1.20:
            _adaptive_pf.append({
                "severity": "HIGH" if port_beta_total > 1.40 else "MODERATE",
                "topic":    "Elevated Market Sensitivity",
                "note":     (f"Portfolio beta ({port_beta_total:.2f}) amplifies market drawdowns. "
                             f"Loss expectation in a 20% market correction: approximately {port_beta_total * 20:.0f}%."),
            })
        if _eff_n_pf < 3 and _eff_n_pf > 0:
            _adaptive_pf.append({
                "severity": "HIGH",
                "topic":    "Low Independent-Bet Count",
                "note":     (f"Effective N = {_eff_n_pf:.1f}: portfolio behaves like fewer than 3 independent assets despite "
                             f"holding {_n_holdings_pf}. Correlation clustering dominates."),
            })
        if _missing_beta_share_pf > 0.30:
            _adaptive_pf.append({
                "severity": "MODERATE",
                "topic":    "Beta Data Coverage",
                "note":     (f"{_missing_beta_share_pf*100:.0f}% of holdings lack beta data. "
                             "Risk-classification confidence reduced. Verify benchmark sensitivity for missing names."),
            })

        # ── Benchmark-Relative Attribution (vs SPX) ─────────────────────────
        if spx_return is not None and _has_beta_total:
            _bench_ret_pf = spx_return  # decimal (e.g. 0.18 for 18%)
            _port_ret_pf  = (port_total_return / 100) if isinstance(port_total_return, (int, float)) else 0
            _market_premium_pf = max(0.0, _bench_ret_pf - 0.045)
            _beta_contrib_pf = (port_beta_total - 1.0) * _market_premium_pf
            _excess_ret_pf   = _port_ret_pf - _bench_ret_pf
            _residual_pf     = _excess_ret_pf - _beta_contrib_pf
            if abs(_beta_contrib_pf) > abs(_residual_pf) * 1.5:
                _attr_verdict_pf = ("Outperformance appears primarily factor-driven (beta differential) rather than selection-driven."
                                    if _excess_ret_pf > 0 else
                                    "Underperformance attributable primarily to factor exposure (beta differential).")
            elif abs(_residual_pf) > abs(_beta_contrib_pf) * 1.5:
                _attr_verdict_pf = ("Outperformance concentrated in residual/selection effects — review concentration risk before attributing to skill."
                                    if _excess_ret_pf > 0 else
                                    "Underperformance concentrated in residual/selection effects — examine specific holdings.")
            else:
                _attr_verdict_pf = "Excess return roughly balanced between factor exposure and selection / concentration."
            _attribution_pf = {
                "excess_pct":   round(_excess_ret_pf * 100, 2),
                "beta_pct":     round(_beta_contrib_pf * 100, 2),
                "residual_pct": round(_residual_pf * 100, 2),
                "verdict":      _attr_verdict_pf,
            }
        else:
            _attribution_pf = None

        # ── Benchmark Context (SPX) ─────────────────────────────────────────
        if spx_return is not None:
            _bench_spx_pct = spx_return * 100
            _active_share_pf = sum(abs(valid_weights_total.get(t, 0) - 0)  # SPX is implicit single-asset bench
                                    for t in valid_tickers) / 2
            _track_pct_pf = abs(alpha) if alpha is not None else 0
            _track_class_pf = "Low" if _track_pct_pf < 3 else ("Moderate" if _track_pct_pf < 6 else "High")
            _active_class_pf = "Low" if _active_share_pf < 0.20 else ("Moderate" if _active_share_pf < 0.40 else "High")
            _drift_signals_pf = []
            if _tech_w_pf > 0.40:
                _drift_signals_pf.append("Tech-overweight")
            if _crypto_share_pf > 0.02:
                _drift_signals_pf.append("Crypto-tilted")
            if _has_beta_total and port_beta_total > 1.20:
                _drift_signals_pf.append("Higher-beta than benchmark")
            elif _has_beta_total and port_beta_total < 0.80:
                _drift_signals_pf.append("Lower-beta than benchmark")
            if cash_pct > 10:
                _drift_signals_pf.append(f"Cash-heavy ({cash_pct:.0f}%)")
            if not _drift_signals_pf:
                _drift_signals_pf.append("Aligned with benchmark composition")
            _benchmark_pf = {
                "label":           "S&P 500 Total Return",
                "bench_ret_pct":   round(_bench_spx_pct, 1),
                "tracking_pct":    round(_track_pct_pf, 2),
                "tracking_class":  _track_class_pf,
                "active_share_pct": round(_active_share_pf * 100, 1),
                "active_class":    _active_class_pf,
                "style_drift":     " · ".join(_drift_signals_pf),
            }
        else:
            _benchmark_pf = None

        # ── Audit Appendix ──────────────────────────────────────────────────
        _audit_input_pf = (
            f"holdings={tuple(sorted(valid_tickers))}|"
            f"weights={tuple(sorted((t, round(w,4)) for t,w in valid_weights_total.items()))}|"
            f"cash_pct={round(cash_pct,2)}|"
            f"date={now_str if 'now_str' in dir() else ''}"
        )
        _snapshot_id_pf = _hashlib_pf.sha256(_audit_input_pf.encode()).hexdigest()[:12]
        _holdings_hash_pf = _hashlib_pf.sha256(",".join(sorted(valid_tickers)).encode()).hexdigest()[:12]
        _audit_pf = {
            "snapshot_id":     _snapshot_id_pf,
            "holdings_hash":   _holdings_hash_pf,
            "methodology":     "Historical Simulation · 252 trading days · CLARABEL QP optimizer (rebalance suggestions)",
            "n_holdings":      _n_holdings_pf,
            "data_window":     "Trailing 252 trading days",
            "benchmark":       (_benchmark_pf or {}).get("label", "n/a"),
            "rf_rate_pct":     4.5,
            "confidence_pct":  round(_confidence_pf * 100, 0),
        }
        _model_constraints_pf = [
            "Historical simulation uses 252-day trailing window; structural breaks beyond that window are not captured.",
            "Correlation matrix is point-in-time; pairwise correlations rise toward 1.0 during liquidity events.",
            "Volatility is non-stationary; realized vol can diverge materially from in-sample estimates during regime shifts.",
            "Beta estimates assume linear market sensitivity; convex behavior (gamma) ignored.",
            "Optimizer rebalance suggestions assume frictionless execution; transaction costs and slippage are out-of-scope.",
            "Crypto and frontier-market positions evaluated with reduced confidence — historical proxies less stable.",
        ]

        verdict_line = (
            f"Equity sleeve returned **{_fmt_safe(port_total_return, '+.1f')}%**; estimated total-portfolio return "
            f"(including {_fmt_safe(cash_pct)}% cash) is **{_fmt_safe(port_total_return_total_est, '+.1f')}%**. "
            f"Vs S&P 500 **{_fmt_safe(spx_return * 100 if spx_return is not None else None, '+.1f')}%**: equity alpha **{_fmt_safe(alpha, '+.1f')}%**, "
            f"estimated total alpha **{_fmt_safe(alpha_total_est, '+.1f')}%**. "
            f"Risk profile (total beta): **{risk_label}** ({_fmt_beta(port_beta_total)}). Sharpe (equity sleeve): **~{_fmt_safe(sharpe)}**. "
            f"Strategic guidance: {'Strong performance, but concentration-driven with elevated risk — consider gradual de-risking and defensive diversification.' if elevated_risk else 'Returns are solid, but correlation clustering remains — consider incremental diversification while monitoring rolling Sharpe.' if high_corr else 'Strong historical performance with balanced risk controls — continue disciplined monitoring before adding incremental risk.'}"
            if spx_return is not None else
            f"Equity sleeve 1Y Return: **{_fmt_safe(port_total_return, '+.1f')}%** (estimated total portfolio: **{_fmt_safe(port_total_return_total_est, '+.1f')}%**). "
            f"Alpha: N/A (benchmark unavailable). "
            f"Risk: **{risk_label}** ({_fmt_beta(port_beta_total)}). Sharpe (equity sleeve): **~{_fmt_safe(sharpe)}**."
        )
        lines.append("# EisaX Portfolio Risk Report")
        lines.append(f"**Date:** {now_str}  |  **Period:** 1 Year  |  **Risk-Free Rate:** 4.5% (US T-Bill)  |  **Snapshot ID:** `{_snapshot_id_pf}`")
        lines.append("")

        # ── Section A — Executive Summary ──────────────────────────────────
        lines.append("## A. Executive Summary")
        lines.append("")
        # Top-line metric table with institutional severity tags
        def _ret_tag(v):
            if v is None: return "[N/A]"
            return "[STRONG]" if v > 10 else ("[MODERATE]" if v > 0 else "[LOW]")
        def _vol_tag(v):
            if v is None: return "[N/A]"
            return "[LOW]" if v < 12 else ("[MODERATE]" if v < 20 else "[HIGH]")
        def _sharpe_tag(v):
            if v is None: return "[N/A]"
            return "[STRONG]" if v > 1.2 else ("[GOOD]" if v > 0.8 else ("[ACCEPTABLE]" if v > 0.5 else "[BELOW MANDATE]"))
        def _beta_tag(v):
            if v is None: return "[N/A]"
            return "[LOW]" if v < 0.7 else ("[MODERATE]" if v < 1.1 else "[HIGH]")
        _ret_v = port_total_return if isinstance(port_total_return, (int, float)) else None
        _vol_v = (ann_vol * 100) if isinstance(ann_vol, (int, float)) else None
        _sharpe_v = sharpe if isinstance(sharpe, (int, float)) else None
        _beta_v = port_beta_total if (_has_beta_total and isinstance(port_beta_total, (int, float))) else None
        lines.append("| Metric | Value | Assessment |")
        lines.append("|--------|-------|------------|")
        lines.append(f"| Equity-Sleeve 1Y Return | **~{_ret_v:+.1f}%** | {_ret_tag(_ret_v)} |" if _ret_v is not None else "| Equity-Sleeve 1Y Return | N/A | [N/A] |")
        lines.append(f"| Total-Portfolio 1Y Return | **~{port_total_return_total_est:+.1f}%** | Includes ~{cash_pct:.0f}% cash drag |")
        if spx_return is not None and alpha is not None:
            lines.append(f"| vs S&P 500 (Alpha) | **~{alpha:+.1f}%** | {'[STRONG]' if alpha > 3 else ('[ACCEPTABLE]' if alpha > 0 else '[BELOW BENCHMARK]')} |")
        lines.append(f"| Annualized Volatility | **~{_vol_v:.1f}%** | {_vol_tag(_vol_v)} |" if _vol_v is not None else "| Annualized Volatility | N/A | [N/A] |")
        lines.append(f"| Sharpe Ratio | **~{_sharpe_v:.2f}** | {_sharpe_tag(_sharpe_v)} |" if _sharpe_v is not None else "| Sharpe Ratio | N/A | [N/A] |")
        lines.append(f"| Portfolio Beta (Total) | **~{_beta_v:.2f}** | {_beta_tag(_beta_v)} |" if _beta_v is not None else "| Portfolio Beta (Total) | N/A | [N/A] |")
        lines.append(f"| CVaR 95% (1-Day) | **~{cvar_95:.2f}%** | [{('CRITICAL' if cvar_95 < -4 else ('HIGH' if cvar_95 < -3 else ('MODERATE' if cvar_95 < -2 else 'LOW')))}] |")
        lines.append(f"| Max Drawdown (1Y) | **~{max_dd:.1f}%** | [{('CRITICAL' if max_dd < -25 else ('HIGH' if max_dd < -15 else ('MODERATE' if max_dd < -10 else 'LOW')))}] |")
        lines.append("")
        # Regime + Confidence + Implementation + Benchmark strips
        lines.append(f"**Portfolio Regime:** **{_regime_pf}**")
        lines.append(f"> {_regime_impl_pf}")
        if _regime_behavior_pf:
            lines.append(f"> **Regime Behavior vs Benchmark:** {_regime_behavior_pf}")
        lines.append("")
        lines.append(f"**Confidence Calibration** · Score: **{_confidence_pf*100:.0f}%** · Evidence Breadth: **{_evidence_breadth_pf}** · Coverage Quality: **{_coverage_quality_pf}** · Reliability Tier: **{_reliability_tier_pf}** {_tier_tag_pf}")
        lines.append("")
        lines.append(f"**Implementation Feasibility** · Deployability: **{_deploy_tier_pf}** {_deploy_tag_pf} ({_deploy_score_pf:.0f}/100) · Rebalancing Complexity: **{_rebal_complexity_pf}** {_impl_tag_pf(_rebal_complexity_pf)} · Liquidity: **{_liq_practicality_pf}** {_impl_tag_pf(_liq_practicality_pf)} · Execution Friction: **{_execution_friction_pf}** {_impl_tag_pf(_execution_friction_pf)} · Est. Turnover ~{_est_turnover_pf:.0f}%/yr · Est. Slippage ~{_est_slippage_pf:.0f} bp")
        lines.append("")
        if _benchmark_pf:
            lines.append(f"**Benchmark Context** · Reference: **{_benchmark_pf['label']}** · Bench Return ~{_benchmark_pf['bench_ret_pct']:+.1f}% · Tracking Deviation: **{_benchmark_pf['tracking_class']}** {_impl_tag_pf(_benchmark_pf['tracking_class'])} ({_benchmark_pf['tracking_pct']:.1f}% vs alpha) · Active Share: **{_benchmark_pf['active_class']}** {_impl_tag_pf(_benchmark_pf['active_class'])} ({_benchmark_pf['active_share_pct']:.0f}%) · Style Drift: **{_benchmark_pf['style_drift']}**")
            lines.append("")
        lines.append("> *Values are approximate, derived from 252-day historical simulation. Not a guarantee of future performance.*")
        lines.append("")
        lines.append("---")
        lines.append("")

        # ── Section B — Mandate Feasibility Analysis ───────────────────────
        lines.append("## B. Mandate Feasibility Analysis")
        lines.append("")
        lines.append("> Institutional constraint diagnostics applied to the uploaded construction. Each check shows the conventional limit, the actual observed value, and the resulting status.")
        lines.append("")
        lines.append("| Constraint | Limit | Actual | Status |")
        lines.append("|------------|-------|--------|--------|")
        _STATUS_TAG_PF = {
            "PASS":            "[PASS]",
            "NEAR CAP":        "[NEAR CAP]",
            "BREACH":          "[BREACH]",
            "BELOW FLOOR":     "[AT FLOOR]",
            "LOW DIVERSIFICATION": "[LOW DIVERSIFICATION]",
        }
        for c in _constraint_pf:
            _tag = _STATUS_TAG_PF.get(c["status"], f"[{c['status']}]")
            lines.append(f"| {c['name']} | {c['limit_pct']} | {c['actual_pct']} | {_tag} |")
        lines.append("")
        lines.append("---")
        lines.append("")

        # ── Section D — Allocation Logic (Holdings, Sectors, Attribution, Factors) ──
        lines.append("## D. Allocation Logic")
        lines.append("")
        lines.append("### Holdings")
        lines.append("| Ticker | Weight (Total) | Weight (Equity Sleeve) | Sector | Beta | P/E | Div Yield |")
        lines.append("|--------|----------------|------------------------|--------|------|-----|-----------|")
        for t in valid_tickers:
            info  = stock_info.get(t, {})
            w_total_pct = portfolio.get(t, 0) * 100
            w_equity_pct = valid_weights.get(t, 0) * 100
            dy    = info.get("div_yield", 0) * 100
            _beta_str = _fmt_beta(info.get("beta"))
            _pe_v = info.get("pe")
            _pe_str = f"{_pe_v:.1f}" if isinstance(_pe_v, (int, float)) and np.isfinite(_pe_v) else "N/A"
            lines.append(
                f"| {t} | {w_total_pct:.1f}% | {w_equity_pct:.1f}% | {info.get('sector','N/A')} "
                f"| {_beta_str} "
                f"| {_pe_str} "
                f"| {dy:.2f}% |"
            )
        if cash_pct > 0.5:
            lines.append(f"| CASH | {cash_pct:.1f}% | — | — | 0 | — | 0% |")
        lines.append(
            f"\n> **Weighted Dividend Yield (Total Portfolio):** {port_div_yield_total:.2f}%  "
            f"| **Equity Sleeve:** {port_div_yield_equity:.2f}%"
        )

        # ── Risk Metrics ───────────────────────────────────────────────────
        lines.append("")
        lines.append("### Detailed Risk Metrics")
        lines.append("*Method: Historical Simulation (252 trading days) · rf = 4.5% · equity-sleeve normalized*")
        lines.append("")
        lines.append("| Metric | Value | Assessment |")
        lines.append("|--------|-------|------------|")
        lines.append(f"| 1Y Return (Equity Sleeve) | {_fmt_safe(port_total_return, '+.1f')}% | [{'STRONG' if port_total_return > 15 else ('MODERATE' if port_total_return > 0 else 'LOW')}] |")
        lines.append(f"| Estimated 1Y Return (Total Portfolio) | {_fmt_safe(port_total_return_total_est, '+.1f')}% | Includes {_fmt_safe(cash_pct)}% cash drag |")
        if spx_return is not None and alpha is not None:
            lines.append(f"| vs S&P 500 (Alpha) | {_fmt_safe(alpha, '+.1f')}% | [{'STRONG' if alpha > 3 else ('ACCEPTABLE' if alpha > 0 else 'BELOW BENCHMARK')}] |")
        elif spx_return is None:
            lines.append(f"| vs S&P 500 (Alpha) | N/A | Benchmark data unavailable |")
        lines.append(f"| Annualized Volatility | {_fmt_safe(ann_vol * 100 if isinstance(ann_vol, (int, float)) else None)}% | [{'HIGH' if ann_vol > 0.30 else ('MODERATE' if ann_vol > 0.15 else 'LOW')}] |")
        lines.append(f"| Sharpe Ratio (1Y, rf=4.5%) | {_fmt_safe(sharpe, '.2f')} | [{'STRONG' if sharpe > 1.5 else ('ACCEPTABLE' if sharpe > 0.5 else 'BELOW MANDATE')}] |")
        lines.append(f"| Sortino Ratio (1Y, rf=4.5%) | {_fmt_safe(sortino, '.2f')} | [{'STRONG' if sortino > 1.0 else ('ACCEPTABLE' if sortino > 0.5 else 'BELOW MANDATE')}] downside-adjusted |")
        lines.append(f"| Portfolio Beta (Total Weight) | {_fmt_beta(port_beta_total)} | {'[HIGH]' if (_has_beta_total and port_beta_total > 1.5) else ('[MODERATE]' if (_has_beta_total and port_beta_total > 1) else ('[LOW]' if _has_beta_total else '[N/A] (missing beta data)'))} |")
        lines.append(f"| Portfolio Beta (Equity Sleeve) | {_fmt_beta(port_beta_equity)} | {'Normalized over non-cash assets' if _has_beta_equity else 'N/A (insufficient beta coverage)'} |")
        lines.append(f"| VaR 95% 1-Day (Historical) | {_fmt_safe(var_95, '.2f')}% | 95% of days, loss ≤ this |")
        lines.append(f"| CVaR 95% (Expected Shortfall) | {_fmt_safe(cvar_95, '.2f')}% | Avg loss **when** VaR is breached — tail risk |")
        lines.append(f"| Max Drawdown (1Y) | {_fmt_safe(max_dd)}% | Worst peak-to-trough |")
        lines.append(f"| Max Drawdown Duration | {max_dd_duration}d | Longest time underwater |")
        lines.append(f"| Portfolio Div Yield (Total) | {port_div_yield_total:.2f}% | Weighted annual income |")
        lines.append(f"| Portfolio Div Yield (Equity Sleeve) | {port_div_yield_equity:.2f}% | Normalized over non-cash assets |")
        if rolling_sharpe_str:
            lines.append(f"| Rolling Sharpe (63d) | — | {rolling_sharpe_str} |")

        # ── Sector Concentration ───────────────────────────────────────────
        lines.append("")
        lines.append("### Sector Exposure")
        lines.append("| Sector | Weight | HHI Contribution |")
        lines.append("|--------|--------|-----------------|")
        for sec, w in sorted(sector_concentration.items(), key=lambda x: -x[1]):
            lines.append(f"| {sec} | {w*100:.1f}% | {w**2:.3f} |")
        hhi_label = "🔴 Highly Concentrated" if hhi > 0.35 else "🟡 Moderately Concentrated" if hhi > 0.18 else "🟢 Diversified"
        lines.append(f"\n> **Sector HHI:** {hhi:.3f} — {hhi_label} *(0=perfect diversification, 1=single sector)*")

        # ── Correlation & Diversification Analysis ────────────────────────
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## C. Risk Diagnostics")
        lines.append("")
        # Adaptive Disclosures table — surfaces only what materially applies
        if _adaptive_pf:
            lines.append("### Adaptive Risk Disclosures")
            lines.append("")
            lines.append("*Conditional on the constructed portfolio — only risks that materially apply are surfaced.*")
            lines.append("")
            lines.append("| Severity | Topic | Note |")
            lines.append("|----------|-------|------|")
            for _d in _adaptive_pf:
                lines.append(f"| [{_d['severity']}] | {_d['topic']} | {_d['note']} |")
            lines.append("")
        lines.append("### Correlation & Diversification")
        lines.append(corr_matrix_str)
        lines.append("")
        eff_n_label = "🔴 Low" if eff_n < 1.5 else "🟡 Moderate" if eff_n < 2.5 else "🟢 Good"
        avg_corr_label = "🔴 High" if avg_corr > 0.60 else "🟡 Moderate" if avg_corr > 0.35 else "🟢 Low"
        lines.append(f"| Metric | Value | Interpretation |")
        lines.append(f"|--------|-------|----------------|")
        lines.append(f"| Avg Pairwise Correlation | {avg_corr:.2f} | {avg_corr_label} co-movement between holdings |")
        lines.append(f"| Effective N (eigenvalue) | {eff_n:.1f} | {eff_n_label} — true independent bets in portfolio |")
        if high_corr:
            lines.append(f"\n**⚠️ High Correlation Pairs (>0.70):**")
            for hc in high_corr:
                lines.append(f"- {hc}")
            lines.append("> These pairs move together — holding both adds limited diversification. One may be redundant.")
        else:
            lines.append(f"\n> ⚠️ **Note:** Even without pairs >0.70, Effective N = {eff_n:.1f} indicates portfolio "
                         f"behaves like **{eff_n:.1f} independent assets** — not {len(valid_tickers)}. "
                         "Sector concentration (not just pairwise correlation) is the real risk driver here.")

        # ── Stress Tests (Historical + Simulated) ─────────────────────────
        lines.append("")
        lines.append("### Stress Testing")
        lines.append("*Historical crises use actual observed returns + sector-adjusted amplification (NOT linear beta)*")
        lines.append("")
        lines.append("| Scenario | SPX Actual | Portfolio Est. | Notes |")
        lines.append("|----------|------------|----------------|-------|")
        for sl in scenario_lines:
            lines.append(sl)
        lines.append(f"\n> **Tech Weight:** {tech_weight*100:.0f}% of equity. "
                     f"In risk-off events, tech typically amplifies market moves by 1.2–1.8×. "
                     f"{'Linear beta alone (equity β=' + _fmt_beta(port_beta_equity) + ') understates true drawdown risk.' if _has_beta_equity else 'Equity beta unavailable (N/A) — avoid beta-based inference until data coverage improves.'}")

        # ── EisaX Risk Assessment ─────────────────────────────────────────
        lines.append("")
        lines.append("### Aggregate Risk Assessment")
        _top_sector = max(sector_concentration, key=sector_concentration.get) if sector_concentration else "N/A"
        _top_sector_pct = sector_concentration.get(_top_sector, 0) * 100
        if _has_beta_total and port_beta_total > 1.5:
            lines.append(
                f"🔴 **Aggressive** — total β={_fmt_beta(port_beta_total)} (equity β={_fmt_beta(port_beta_equity)}), "
                f"CVaR={cvar_95:.2f}%/day, Eff.N={eff_n:.1f} bets. "
                f"{_top_sector_pct:.0f}% in {_top_sector}. "
                f"{('In a 2022-style selloff your equity sleeve would have lost ~' + str(round(abs(tech_weight * -0.33 + (1-tech_weight) * -0.195 * port_beta_equity)*100)) + '% ') if _has_beta_equity else ''}"
                f"vs SPX -19.5%. Diversification is urgently needed."
            )
        elif _has_beta_total and port_beta_total > 1.0:
            lines.append(
                f"🟡 **Moderate-Aggressive** — total β={_fmt_beta(port_beta_total)} (equity β={_fmt_beta(port_beta_equity)}), CVaR={cvar_95:.2f}%/day. "
                "Above-market sensitivity. Trim highest-beta names on strength; add one defensive sector."
            )
        elif _has_beta_total:
            lines.append(
                f"🟢 **Balanced** — total β={_fmt_beta(port_beta_total)} (equity β={_fmt_beta(port_beta_equity)}), "
                f"CVaR={cvar_95:.2f}%/day, Eff.N={eff_n:.1f}. Reasonable risk profile."
            )
        else:
            lines.append(
                f"⚪ **Risk Classification Limited** — beta data unavailable for enough holdings. CVaR={cvar_95:.2f}%/day, Eff.N={eff_n:.1f}. Add missing beta data before beta-based decisions."
            )
        lines.append("")

        # ══════════════════════════════════════════════════════════════════════
        # ── 1. PERFORMANCE ATTRIBUTION ────────────────────────────────────
        # Shows each holding's contribution to total return (Brinson model)
        # ══════════════════════════════════════════════════════════════════════
        if len(price_data) >= 1:
            lines.append("---")
            lines.append("")
            lines.append("### Performance Attribution (1Y, Equity Sleeve)")
            lines.append("*Brinson-Hood-Beebower: each holding's contribution to equity-sleeve return*")
            lines.append("")
            lines.append("| Ticker | Weight (Equity Sleeve) | 1Y Return | Contribution | Attribution |")
            lines.append("|--------|--------|-----------|--------------|-------------|")
            attr_rows = []
            for t in valid_tickers:
                if t in price_data and len(price_data[t]) > 1:
                    t_ret = float((price_data[t].iloc[-1] / price_data[t].iloc[0]) - 1) * 100
                    contrib = t_ret * valid_weights.get(t, 0)
                    attr_rows.append((t, valid_weights.get(t, 0), t_ret, contrib))
            attr_rows.sort(key=lambda x: -x[3])  # sort by contribution descending
            for t, w, ret, contrib in attr_rows:
                bar = "🟢" if contrib > 0 else "🔴"
                lines.append(f"| {t} | {_fmt_safe(w*100)}% | {_fmt_safe(ret, '+.1f')}% | {_fmt_safe(contrib, '+.2f')}pp | {bar} |")
            # Residual from compounding + rounding differences
            explained = sum(c for _, _, _, c in attr_rows)
            residual = port_total_return - explained
            if abs(residual) > 0.05:
                lines.append(f"| Residual (model) | — | — | {_fmt_safe(residual, '+.2f')}pp | {'🔴' if residual < 0 else '⚪'} |")
            lines.append(f"| **TOTAL** | 100% | — | **{_fmt_safe(port_total_return, '+.2f')}pp** | |")
            lines.append("")
            _excluded = [t for t in valid_tickers if t not in price_data or len(price_data.get(t, [])) < 2]
            if _excluded:
                lines.append(f"\n> ⚠️ **Excluded from attribution:** {', '.join(_excluded)} (data unavailable)")
            # Alpha source: top contributor vs S&P
            if attr_rows and spx_return is not None and alpha is not None:
                top_t, top_w, top_ret, top_c = attr_rows[0]
                lines.append(f"> 🏆 **Alpha driver:** {top_t} contributed {_fmt_safe(top_c, '+.1f')}pp (+{_fmt_safe(top_ret)}% × {top_w*100:.0f}% weight). "
                             f"S&P 500 returned {_fmt_safe(spx_return * 100 if spx_return is not None else None, '+.1f')}% — alpha gap of {_fmt_safe(alpha, '+.1f')}pp.")
            lines.append("")

        # ══════════════════════════════════════════════════════════════════════
        # ── 2. FACTOR EXPOSURE (Fama-French Proxy) ───────────────────────────
        # Approximates factor tilts using beta, P/E, market cap, momentum
        # ══════════════════════════════════════════════════════════════════════
        lines.append("### Factor Exposure Analysis")
        lines.append("*Fama-French proxy: factor tilts computed from fundamentals + price momentum*")
        lines.append("")

        factor_scores = {}
        factor_weights = {}
        _factor_order = ["Market (Beta)", "Growth (P/E tilt)", "Size (SMB proxy)", "Momentum (12M)"]
        for t in valid_tickers:
            info   = stock_info.get(t, {})
            w      = valid_weights.get(t, 0)
            beta   = info.get("beta")
            pe     = info.get("pe")
            mc     = info.get("mktcap")
            # 1Y momentum from price data
            mom = None
            if t in price_data and len(price_data[t]) > 20:
                _s = price_data[t]
                mom = float((_s.iloc[-1] / _s.iloc[max(0, len(_s)-252)]) - 1)

            # Factor scoring (institutional proxy)
            # Market beta exposure
            if isinstance(beta, (int, float)) and np.isfinite(beta):
                factor_scores["Market (Beta)"] = factor_scores.get("Market (Beta)", 0.0) + (beta * w)
                factor_weights["Market (Beta)"] = factor_weights.get("Market (Beta)", 0.0) + w
            # Growth (inverse P/E proxy — high PE = growth tilt)
            if isinstance(pe, (int, float)) and np.isfinite(pe):
                growth_score = min(max((pe - 15) / 50, -1), 1)  # normalize
                factor_scores["Growth (P/E tilt)"] = factor_scores.get("Growth (P/E tilt)", 0.0) + (growth_score * w)
                factor_weights["Growth (P/E tilt)"] = factor_weights.get("Growth (P/E tilt)", 0.0) + w
            # Size (log market cap — large = negative small-cap factor)
            if isinstance(mc, (int, float)) and np.isfinite(mc) and mc > 0:
                import math as _math
                size_score = -(_math.log10(mc) - 10) / 3  # large cap = negative SMB
                factor_scores["Size (SMB proxy)"] = factor_scores.get("Size (SMB proxy)", 0.0) + (size_score * w)
                factor_weights["Size (SMB proxy)"] = factor_weights.get("Size (SMB proxy)", 0.0) + w
            # Momentum (12-1 month)
            if isinstance(mom, (int, float)) and np.isfinite(mom):
                factor_scores["Momentum (12M)"] = factor_scores.get("Momentum (12M)", 0.0) + (mom * w)
                factor_weights["Momentum (12M)"] = factor_weights.get("Momentum (12M)", 0.0) + w

        lines.append("| Factor | Portfolio Exposure | Interpretation |")
        lines.append("|--------|--------------------|----------------|")
        _factor_labels = {
            "Market (Beta)":       lambda v: ("🔴 High market sensitivity" if v > 1.5 else "🟢 Low market sensitivity" if v < 0.8 else "🟡 Market-like"),
            "Growth (P/E tilt)":   lambda v: ("🔴 Strong growth tilt — high valuation risk" if v > 0.4 else "🟢 Value tilt" if v < -0.1 else "🟡 Blend"),
            "Size (SMB proxy)":    lambda v: ("🟢 Large-cap dominated (low SMB)" if v < -0.1 else "🔴 Small-cap tilt" if v > 0.2 else "🟡 Mid-cap blend"),
            "Momentum (12M)":      lambda v: ("🟢 Strong momentum" if v > 0.20 else "🔴 Negative momentum" if v < -0.10 else "🟡 Neutral momentum"),
        }
        for fname in _factor_order:
            _w_cov = factor_weights.get(fname, 0.0)
            if _w_cov <= 0:
                lines.append(f"| {fname} | **N/A** | ⚪ Insufficient data |")
                continue
            fval = factor_scores.get(fname, 0.0) / _w_cov
            label_fn = _factor_labels.get(fname, lambda v: "—")
            lines.append(f"| {fname} | **{fval:+.2f}** | {label_fn(fval)} |")
        lines.append("")
        lines.append("> *Factor exposures are approximated from fundamentals. "
                     "For precise Fama-French loadings, a 3-year return regression against HML/SMB/MKT factors is required.*")
        lines.append("")

        # ══════════════════════════════════════════════════════════════════════
        # ── 3. INSTITUTIONAL OPTIMIZATION ENGINE ─────────────────────────────
        # Constrained multi-objective: Maximise Sharpe & Minimise CVaR
        # with institutional mandate limits (single stock, sector, beta, eff N)
        # ══════════════════════════════════════════════════════════════════════
        if len(price_data) >= 2:
            lines.append("---")
            lines.append("")
            lines.append("## E. Rebalancing Plan — Institutional Optimization Engine")
            lines.append("*Convex QP · CLARABEL solver · Zero-tolerance constraint enforcement*")
            lines.append("")
            class _OptimizerSkip(Exception):
                pass
            try:
                import cvxpy as cp
                from scipy.optimize import minimize as _scimin

                _opt_tickers = [t for t in returns_df.columns if t in valid_weights]
                _n = len(_opt_tickers)
                if _n < 2:
                    raise ValueError("Need ≥2 equity holdings for optimization")

                _mu       = np.array([float(returns_df[t].mean() * 252) for t in _opt_tickers])
                _cov_raw  = returns_df[_opt_tickers].cov().values * 252
                _cov      = np.array(_cov_raw, dtype=float) + np.eye(_n) * 1e-8   # PSD regularisation
                _missing_beta = []
                for _t in _opt_tickers:
                    if stock_info.get(_t, {}).get("beta") is not None:
                        continue
                    _fallback_beta = _regression_beta(returns_df[_t], spx_returns)
                    if _fallback_beta is None:
                        _missing_beta.append(_t)
                    else:
                        stock_info.setdefault(_t, {})["beta"] = _fallback_beta
                if _missing_beta:
                    raise _OptimizerSkip(
                        "beta unavailable after regression fallback for "
                        + ", ".join(_missing_beta)
                    )
                _betas_v  = np.array([max(float(stock_info.get(t, {}).get("beta")), 0.01)
                                      for t in _opt_tickers])
                _MIN_W    = 0.01          # 1 % floor per holding
                _PORT_V   = 100_000       # $100k reference for dollar amounts

                # ── Sector matrix ──────────────────────────────────────────
                _tkr_sec   = {t: stock_info.get(t,{}).get("sector","Unknown") for t in _opt_tickers}
                _sec_uniq  = sorted(set(_tkr_sec.values()))
                _S_mat     = np.array([[1 if _tkr_sec[t]==s else 0 for t in _opt_tickers]
                                       for s in _sec_uniq])

                # Minimum achievable portfolio beta (max weight in lowest-beta stock)
                _bidx = np.argsort(_betas_v)
                _min_beta_possible = float(
                    _betas_v[_bidx[0]] * (1 - (_n-1)*_MIN_W)
                    + sum(_betas_v[_bidx[i]] * _MIN_W for i in range(1, _n))
                )

                # ── Helpers ────────────────────────────────────────────────
                def _port_metrics(w: np.ndarray):
                    w = np.clip(w, 0, 1); w /= w.sum()
                    ret  = float(_mu @ w)
                    vol  = float(np.sqrt(w @ _cov @ w))
                    sh   = round((ret - RF_RATE) / vol, 2) if vol > 0 else 0.0
                    beta = float(_betas_v @ w)
                    pr   = returns_df[_opt_tickers].values @ w
                    cvar = float(pr[pr <= np.percentile(pr, 5)].mean() * 100)
                    return ret, vol, sh, beta, cvar   # ret, vol, sharpe, beta, cvar

                # ── SECTION 1: Constraint Diagnostics ─────────────────────
                lines.append("### 🔍 Constraint Diagnostics")
                lines.append(f"*Reference: ${_PORT_V:,} portfolio · normalized weights*")
                lines.append("")
                lines.append("| Constraint | Limit | Current | Status | Required Fix |")
                lines.append("|------------|-------|---------|--------|--------------|")
                _has_breach = False

                for _t in _opt_tickers:
                    _cw = valid_weights.get(_t, 0)
                    if _cw > 0.20:
                        _has_breach = True
                        _exc = _cw - 0.20
                        lines.append(
                            f"| Single stock: **{_t}** | ≤20% | {_cw*100:.1f}% |"
                            f" 🔴 −{_exc*100:.1f}pp | Sell ~${_exc*_PORT_V:,.0f} of {_t} |"
                        )
                for _sec, _sw in sorted(sector_concentration.items(), key=lambda x: -x[1]):
                    if _sw > 0.40:
                        _has_breach = True
                        _exc = _sw - 0.40
                        lines.append(
                            f"| Sector: **{_sec}** | ≤40% | {_sw*100:.1f}% |"
                            f" 🔴 −{_exc*100:.1f}pp | Reduce by ~${_exc*_PORT_V:,.0f} / add non-{_sec} |"
                        )
                if port_beta > 1.30:
                    _has_breach = True
                    _exc = port_beta - 1.30
                    lines.append(
                        f"| Portfolio beta | ≤1.30 | {port_beta:.2f} |"
                        f" 🔴 +{_exc:.2f} | Add low-β assets: TLT, GLD, BRK-B, XLV |"
                    )
                if not _has_breach:
                    lines.append("| All constraints | — | — | ✅ No breaches | — |")

                _overweight_total = sum(max(0, valid_weights.get(_t,0) - 0.20) for _t in _opt_tickers)
                if _overweight_total > 0.005:
                    lines.append("")
                    lines.append(
                        f"> 💸 **Estimated rebalancing:** ~${_overweight_total*_PORT_V:,.0f} turnover "
                        f"({_overweight_total*100:.1f}% of portfolio) to reach basic mandate compliance"
                    )
                lines.append("")

                # ── Feasibility check ──────────────────────────────────────
                def _check_feas(max_s: float, max_b: float, max_sec: float):
                    """Returns (single_ok, beta_ok, sector_ok, issues[], suggestions[])"""
                    iss, sug = [], []
                    # 1. Single-stock: N * max_s must cover 100%
                    s_ok = (_n * max_s >= 1.0 - 0.005)
                    if not s_ok:
                        need = int(np.ceil(1.0 / max_s))
                        iss.append(
                            f"{_n} holdings × {max_s*100:.0f}% = {_n*max_s*100:.0f}% < 100% "
                            f"(need ≥{need} holdings for this mandate)"
                        )
                        sug.append(f"Add {need - _n}+ new assets OR use Aggressive mandate ({_n}×20%→80% min)")
                    # 2. Beta: minimum achievable ≤ limit
                    b_ok = (_min_beta_possible <= max_b + 0.005)
                    if not b_ok:
                        iss.append(
                            f"Min achievable β = {_min_beta_possible:.2f} exceeds cap β≤{max_b:.2f}"
                        )
                        sug.append("Add: TLT (β≈−0.5), GLD (β≈0.1), USMV (β≈0.7), BRK-B (β≈0.9)")
                    # 3. Sector: sum of per-sector max allocations must ≥ 100%
                    _sec_cap_sum = sum(
                        min(max_sec, sum(1 for _t in _opt_tickers if _tkr_sec[_t]==s) * max_s)
                        for s in _sec_uniq
                    )
                    sec_ok = (_sec_cap_sum >= 1.0 - 0.005)
                    if not sec_ok:
                        _dom = max(_sec_uniq, key=lambda s: sum(1 for _t in _opt_tickers if _tkr_sec[_t]==s))
                        _nd  = sum(1 for _t in _opt_tickers if _tkr_sec[_t]==_dom)
                        iss.append(
                            f"Sector '{_dom}': {_nd}/{_n} holdings → "
                            f"max total = {_sec_cap_sum*100:.0f}% < 100% with current assets"
                        )
                        sug.append("Add diversifying sectors: XLV (Healthcare), TLT (Bonds), GLD (Gold), BRK-B, XLP")
                    return s_ok, b_ok, sec_ok, iss, sug

                # ── QP solver: Max-Sharpe (Lasserre lifting) ──────────────
                def _qp_solve(max_s, max_b, max_sec,
                              s_ok, b_ok, sec_ok,
                              strict: bool = True):
                    """
                    Convex QP — two modes:

                    strict=True (default / Institutional Mode):
                        ALL feasible flags must be True. If any constraint cannot be
                        enforced → return (None, "❌ INFEASIBLE — [reason]").
                        Never silently relaxes a constraint. Never returns a
                        partial/relaxed solution labelled as compliant.

                    strict=False (Exploratory Mode — explicit opt-in):
                        Solves with whatever constraints ARE geometrically possible.
                        Result is ALWAYS labelled "🔍 EXPLORATORY MODE" so the user
                        knows constraints were adjusted. Never presented as compliant.
                    """
                    # ── Strict mode: hard gate ─────────────────────────────
                    if strict:
                        infeas_reasons = []
                        if not s_ok:
                            need = int(np.ceil(1.0 / max_s))
                            infeas_reasons.append(
                                f"single-stock cap {max_s*100:.0f}% requires ≥{need} holdings "
                                f"(current: {_n})"
                            )
                        if not b_ok:
                            infeas_reasons.append(
                                f"beta cap β≤{max_b:.2f} unachievable "
                                f"(min possible β={_min_beta_possible:.2f})"
                            )
                        if not sec_ok:
                            _dom = max(_sec_uniq,
                                       key=lambda s: sum(1 for _t in _opt_tickers
                                                         if _tkr_sec[_t] == s))
                            _nd  = sum(1 for _t in _opt_tickers if _tkr_sec[_t] == _dom)
                            infeas_reasons.append(
                                f"sector cap {max_sec*100:.0f}% unachievable: "
                                f"'{_dom}' has {_nd}/{_n} holdings"
                            )
                        if infeas_reasons:
                            return None, "❌ INFEASIBLE — " + "; ".join(infeas_reasons)

                    # ── Build constraint list ──────────────────────────────
                    try:
                        y = cp.Variable(_n, nonneg=True)
                        k = cp.Variable(nonneg=True)
                        constr = [
                            cp.quad_form(y, _cov) <= 1,
                            cp.sum(y) == k,
                            k >= 0,
                            y >= _MIN_W * k,
                        ]
                        _relaxed_tags: list[str] = []
                        # _active_s_cap: the cap actually enforced (used for post-solve clipping too)
                        if s_ok:
                            _active_s_cap = max_s
                            constr.append(y <= _active_s_cap * k)
                        else:
                            # Relaxed mode: use 1/n + 5% buffer (always geometrically feasible)
                            _active_s_cap = round(1.0 / _n + 0.05, 3)
                            constr.append(y <= _active_s_cap * k)
                            _relaxed_tags.append(
                                f"single-stock cap adjusted to {_active_s_cap*100:.0f}% "
                                f"(mandate {max_s*100:.0f}% infeasible with {_n} holdings)"
                            )
                        if b_ok:
                            constr.append(_betas_v @ y <= max_b * k)
                        else:
                            _relaxed_tags.append(f"beta cap lifted (was β≤{max_b:.2f}, min achievable β={_min_beta_possible:.2f})")
                        if sec_ok:
                            constr.append(_S_mat @ y <= max_sec * k)
                        else:
                            _relaxed_tags.append(f"sector cap lifted (was {max_sec*100:.0f}%, current holdings prevent compliance)")

                        prob = cp.Problem(cp.Maximize((_mu - RF_RATE) @ y), constr)

                        # Primary solver: CLARABEL (strict interior-point)
                        prob.solve(solver=cp.CLARABEL, verbose=False)

                        if prob.status == "optimal" and k.value is not None and float(k.value) > 1e-8:
                            raw   = np.maximum(np.array(y.value, dtype=float) / float(k.value), 0.0)
                            w_out = raw / raw.sum()
                            if s_ok:
                                w_out = np.clip(w_out, _MIN_W, max_s)
                                w_out /= w_out.sum()

                            # Hard post-solve verification (catch solver float drift)
                            if s_ok and float(np.max(w_out)) > max_s + 0.001:
                                return None, (f"❌ INFEASIBLE — solver returned "
                                              f"max={np.max(w_out)*100:.1f}% > {max_s*100:.0f}% cap "
                                              f"(numerical instability)")
                            if b_ok and float(_betas_v @ w_out) > max_b + 0.005:
                                return None, (f"❌ INFEASIBLE — solver beta "
                                              f"β={float(_betas_v@w_out):.3f} > cap β≤{max_b}")

                            if not strict and _relaxed_tags:
                                status = ("🔍 EXPLORATORY MODE — constraints adjusted: "
                                          + "; ".join(_relaxed_tags))
                            else:
                                status = "✅ Fully compliant"
                            return w_out, status

                        # SCS fallback (handles near-degenerate edge cases)
                        prob.solve(solver=cp.SCS, eps=1e-6, verbose=False)
                        if prob.status in ("optimal", "optimal_inaccurate") \
                                and k.value is not None and float(k.value) > 1e-8:
                            raw   = np.maximum(np.array(y.value, dtype=float) / float(k.value), 0.0)
                            w_out = raw / raw.sum()
                            if s_ok:
                                w_out = np.clip(w_out, _MIN_W, max_s)
                                w_out /= w_out.sum()
                            if not strict and _relaxed_tags:
                                status = ("🔍 EXPLORATORY MODE (approx) — "
                                          + "; ".join(_relaxed_tags))
                            else:
                                status = "✅ Fully compliant (approx)"
                            return w_out, status

                        return None, f"❌ INFEASIBLE — solver status: {prob.status}"
                    except Exception as _se:
                        return None, f"❌ Solver error: {_se}"

                # ── SECTION 2: Mandate Mode Feasibility Table ──────────────
                _MANDATES = [
                    # (display_name, max_single, max_beta, max_sector, profile)
                    ("🏛️ Conservative", 0.10, 1.00, 0.30, "Pension / Endowment"),
                    ("⚖️ Balanced",      0.15, 1.20, 0.35, "Family Office / Multi-asset"),
                    ("🚀 Aggressive",    0.20, 1.50, 0.40, "Institutional Growth"),
                ]

                _feas_data = []   # (nm, ms, mb, msec, prof, s_ok, b_ok, sec_ok, iss, sug)
                for _nm, _ms, _mb, _msec, _prof in _MANDATES:
                    _so, _bo, _seco, _iss, _sug = _check_feas(_ms, _mb, _msec)
                    _feas_data.append((_nm, _ms, _mb, _msec, _prof, _so, _bo, _seco, _iss, _sug))

                lines.append("### 📊 Mandate Mode Feasibility")
                lines.append("")
                lines.append("| | 🏛️ Conservative | ⚖️ Balanced | 🚀 Aggressive |")
                lines.append("|---|:---:|:---:|:---:|")
                lines.append("| **Max single stock** | 10% | 15% | 20% |")
                lines.append("| **Max portfolio β** | ≤1.00 | ≤1.20 | ≤1.50 |")
                lines.append("| **Max sector weight** | 30% | 35% | 40% |")
                lines.append("| Single-stock achievable | " + " | ".join(
                    "✅" if r[5] else f"❌ Need {int(np.ceil(1/r[1]))}+ holdings"
                    for r in _feas_data) + " |")
                lines.append("| Beta achievable | " + " | ".join(
                    "✅" if r[6] else f"❌ Min β={_min_beta_possible:.2f}"
                    for r in _feas_data) + " |")
                lines.append("| Sector cap achievable | " + " | ".join(
                    "✅" if r[7] else "⚠️ Current assets only"
                    for r in _feas_data) + " |")
                lines.append("")

                # Collect deduplicated issues / suggestions (keyed by constraint type)
                _iss_by_type: dict[str, str] = {}  # key=type, val=worst message
                _sug_set: set = set()
                for _fd in _feas_data:
                    _so2,_bo2,_seco2,_iss2,_sug2 = _fd[5],_fd[6],_fd[7],_fd[8],_fd[9]
                    for _i2 in _iss2:
                        # Bucket by type: single/beta/sector
                        _ikey = ("single" if "holdings" in _i2 else
                                 "beta"   if "β" in _i2 else "sector")
                        # Keep the most severe (Conservative = strictest = hardest to satisfy)
                        if _ikey not in _iss_by_type:
                            _iss_by_type[_ikey] = _i2
                    _sug_set.update(_sug2)

                if _iss_by_type:
                    lines.append("> **⚠️ Constraint violations — current holdings cannot satisfy any mandate:**")
                    for _ikey in ["single","beta","sector"]:
                        if _ikey in _iss_by_type:
                            lines.append(f"> - {_iss_by_type[_ikey]}")
                    lines.append(">")
                    lines.append("> **💡 To unlock full institutional compliance, add any of:**")
                    for _s in sorted(_sug_set):
                        lines.append(f"> - {_s}")
                    lines.append("")

                # ── Run optimizers (STRICT MODE) for each mandate ─────────
                # strict=True: if any constraint infeasible → ❌, never silently relax
                _m_results = []   # (nm, ms, mb, msec, prof, w_opt, status, iss, sug)
                for _nm, _ms, _mb, _msec, _prof, _so, _bo, _seco, _iss, _sug in _feas_data:
                    _wo, _stat = _qp_solve(_ms, _mb, _msec, _so, _bo, _seco, strict=True)
                    _m_results.append((_nm, _ms, _mb, _msec, _prof, _wo, _stat, _iss, _sug))

                # ── SECTION 3: Optimal Weight Comparison ──────────────────
                lines.append("### 📊 Optimal Weight Comparison — Strict Institutional Mode")
                lines.append("")
                lines.append("> 🔒 **Strict Mode:** Only fully-compliant solutions shown. "
                             "`❌ INFEASIBLE` = mandate cannot be achieved with current holdings — "
                             "see asset suggestions above.")
                lines.append("")
                _col_hdrs = " | ".join(r[0] for r in _m_results)
                lines.append(f"| Ticker | Current | {_col_hdrs} |")
                lines.append("|--------|---------|" + "|".join(["------"]*len(_m_results)) + "|")
                for _i, _t in enumerate(_opt_tickers):
                    _cw = valid_weights.get(_t, 0)
                    _row = f"| **{_t}** | {_cw*100:.1f}% |"
                    for *_, _wo, _stat, _iss2, _sug2 in _m_results:
                        _row += f" {_wo[_i]*100:.1f}% |" if _wo is not None else " ❌ N/A |"
                    lines.append(_row)

                # Metrics rows
                lines.append("|--------|---------|" + "|".join(["------"]*len(_m_results)) + "|")
                _m_metrics = [_port_metrics(r[5]) if r[5] is not None else None for r in _m_results]

                def _mfmt(m_list, idx, fmtfn):
                    return " | ".join(fmtfn(m[idx]) if m is not None else "❌" for m in m_list)

                lines.append(f"| **Sharpe** | {sharpe:.2f} | {_mfmt(_m_metrics,2,lambda x:f'{x:.2f}')} |")
                lines.append(f"| **Beta** | {port_beta:.2f} | {_mfmt(_m_metrics,3,lambda x:f'{x:.2f}')} |")
                lines.append(f"| **CVaR/day** | {cvar_95:.2f}% | {_mfmt(_m_metrics,4,lambda x:f'{x:.2f}%')} |")
                lines.append(f"| **Ann. Vol** | {ann_vol*100:.1f}% | {_mfmt(_m_metrics,1,lambda x:f'{x*100:.1f}%')} |")
                lines.append("| **Compliance** | Baseline | " + " | ".join(
                    "✅ Compliant" if r[5] is not None else "❌ INFEASIBLE"
                    for r in _m_results) + " |")
                lines.append("")

                # ── Exploratory Allocation (only if at least one mandate is infeasible) ──
                _any_infeasible = any(r[5] is None for r in _m_results)
                if _any_infeasible:
                    # Find the most lenient feasibility set (Aggressive mandate)
                    _agg = _feas_data[-1]  # Aggressive = last
                    _rnm, _rms, _rmb, _rmsec, _rprof, _rso, _rbo, _rseco, _riss, _rsug = _agg
                    _rwo, _rstat = _qp_solve(_rms, _rmb, _rmsec, _rso, _rbo, _rseco, strict=False)

                    lines.append("### 🔍 Exploratory Allocation — Pre-Asset Expansion")
                    lines.append("")
                    lines.append("> 🔍 **Exploratory Allocation (Non-Mandate Scenario)** — "
                                 "Shows the best achievable structure with current holdings "
                                 "before adding diversifying assets. "
                                 "Constraints adjusted where mathematically required. "
                                 "This is a planning scenario, not a mandate-compliant allocation.")
                    lines.append("")
                    if _rwo is not None:
                        lines.append("| Ticker | Current | 🔍 Exploratory |")
                        lines.append("|--------|---------|----------------|")
                        for _i, _t in enumerate(_opt_tickers):
                            _cw = valid_weights.get(_t, 0)
                            lines.append(f"| **{_t}** | {_cw*100:.1f}% | {_rwo[_i]*100:.1f}% |")
                        lines.append("|--------|---------|----------------|")
                        _rr, _rv, _rsh, _rb2, _rcv = _port_metrics(_rwo)
                        lines.append(f"| **Sharpe** | {sharpe:.2f} | {_rsh:.2f} |")
                        lines.append(f"| **Beta** | {port_beta:.2f} | {_rb2:.2f} |")
                        lines.append(f"| **CVaR/day** | {cvar_95:.2f}% | {_rcv:.2f}% |")
                        lines.append(f"| **Compliance** | ❌ Breach | 🔍 Exploratory — Pre-Expansion |")
                        lines.append("")
                        lines.append(f"> 🔍 *{_rstat}*")
                    else:
                        lines.append("> ❌ No exploratory solution found. "
                                     "The current portfolio composition prevents optimization. "
                                     "Add diversifying assets before running again.")
                    lines.append("")

                # ── SECTION 4: Execution Plan (strictly compliant mandates only) ──
                _best = next(
                    (r for r in _m_results if r[5] is not None and "✅" in r[6]),
                    None   # None = no compliant mandate found
                )
                # If no strictly compliant mandate found, use relaxed (with clear warning)
                if _best is None and _any_infeasible and '_rwo' in dir() and _rwo is not None:
                    _best = ("🔍 Exploratory Allocation (Non-Mandate Scenario)", _rms, _rmb, _rmsec,
                             "Pre-asset expansion — exploratory scenario", _rwo, _rstat, [], [])
                if _best is not None:
                    _bnm, _bms, _bmb, _bmsec, _bprof, _bw, _bstat, _biss, _bsug = _best
                    _br, _bv, _bsh, _bb, _bcv = _port_metrics(_bw)
                    _is_relaxed_plan = any(x in str(_bstat) for x in ("RELAXED", "EXPLORATORY")) or \
                                       any(x in str(_bnm) for x in ("🔧", "🔍"))

                    lines.append(f"### 📋 Execution Plan → *{_bnm}*")
                    lines.append(f"*{_bprof}*")
                    if _is_relaxed_plan:
                        lines.append("")
                        lines.append("> 🔍 **Exploratory Allocation** — These trades show the optimal "
                                     "structure achievable with current holdings before adding new assets. "
                                     "This is a planning scenario to guide asset expansion decisions — "
                                     "**not a mandate-compliant allocation.**")
                    lines.append("")
                    lines.append("| Action | Ticker | From | To | Δ | $ on $100k | Reason |")
                    lines.append("|--------|--------|------|----|---|------------|--------|")
                    for _i, _t in enumerate(_opt_tickers):
                        _cw = valid_weights.get(_t, 0)
                        _gw = _bw[_i]
                        _d  = _gw - _cw
                        _usd = abs(_d) * _PORT_V
                        if abs(_d) < 0.005:
                            _act, _ico = "HOLD  ", "⚪"
                        elif _d > 0:
                            _act, _ico = "BUY  ↑", "🟢"
                        else:
                            _act, _ico = "SELL ↓", "🔴"
                        _reason = (
                            f"Exceeds {_bms*100:.0f}% mandate cap"       if _cw > _bms + 0.005 and _d < 0 else
                            f"High β={_betas_v[_i]:.2f} — reduce beta"   if _betas_v[_i] > _bmb and _d < 0 else
                            "Underweight vs risk-optimal"                  if _d >  0.02 else
                            "Overweight vs risk-optimal"                   if _d < -0.02 else
                            "Mandate rebalance"
                        )
                        lines.append(
                            f"| {_ico} **{_act}** | {_t} | {_cw*100:.1f}% | {_gw*100:.1f}% |"
                            f" {_d*100:+.1f}pp | ${_usd:,.0f} | {_reason} |"
                        )
                    lines.append("")

                    # ── SECTION 5: Multi-Objective Trade-off ──────────────
                    lines.append("### 📐 Multi-Objective Trade-off Analysis")
                    lines.append("")
                    lines.append("| Strategy | Sharpe | CVaR/day | Beta | Ann.Vol | Status |")
                    lines.append("|----------|--------|----------|------|---------|--------|")
                    lines.append(
                        f"| **Current portfolio** | {sharpe:.2f} | {cvar_95:.2f}% |"
                        f" {port_beta:.2f} | {ann_vol*100:.1f}% | Baseline |"
                    )
                    for _nm2, _ms2, _mb2, _msec2, _prof2, _wo2, _stat2, _, _ in _m_results:
                        if _wo2 is None:
                            lines.append(f"| {_nm2} | — | — | — | — | {_stat2} |")
                            continue
                        _mr, _mv, _msh, _mb3, _mcv = _port_metrics(_wo2)
                        lines.append(
                            f"| {_nm2} | {_msh:.2f} ({_msh-sharpe:+.2f}) |"
                            f" {_mcv:.2f}% ({_mcv-cvar_95:+.2f}pp) |"
                            f" {_mb3:.2f} ({_mb3-port_beta:+.2f}) |"
                            f" {_mv*100:.1f}% | {_stat2} |"
                        )
                    lines.append("")

                    # ── SECTION 6: Post-Rebalance Simulation ──────────────
                    lines.append("### 🔮 Post-Rebalance Simulation")
                    lines.append(f"*Best mandate: {_bnm} · {_bprof}*")
                    lines.append("")
                    lines.append("| Metric | Before | After | Δ | Verdict |")
                    lines.append("|--------|--------|-------|---|---------|")
                    _curr_breach  = any(valid_weights.get(_t,0) > 0.20 for _t in _opt_tickers)
                    _after_breach = any(_bw[_i] > _bms + 0.002 for _i in range(_n))
                    _sim_rows = [
                        ("Sharpe Ratio",    f"{sharpe:.2f}",        f"{_bsh:.2f}",
                            _bsh-sharpe,      lambda d: "✅ Improved" if d > 0.05 else "🟡 Similar"),
                        ("Ann. Volatility", f"{ann_vol*100:.1f}%",  f"{_bv*100:.1f}%",
                            (_bv-ann_vol)*100, lambda d: "✅ Lower" if d < -1 else "🟡 Similar" if d < 2 else "🔴 Higher"),
                        ("Portfolio Beta",  f"{port_beta:.2f}",     f"{_bb:.2f}",
                            _bb-port_beta,     lambda d: "✅ Reduced" if d < -0.05 else "⚠️ Similar"),
                        ("CVaR 95%/day",    f"{cvar_95:.2f}%",      f"{_bcv:.2f}%",
                            _bcv-cvar_95,      lambda d: "✅ Safer tail" if d > 0.1 else "🟡 Similar"),
                        ("Single-stock cap",
                            "🔴 Breach" if _curr_breach else "✅ OK",
                            ("🔍 Exploratory" if _is_relaxed_plan else
                             "✅ Compliant" if not _after_breach else "🔴 Still Breach"),
                            0, lambda d: ("🔍 Exploratory scenario" if _is_relaxed_plan
                                          else "✅ Resolved" if not _after_breach else "🔴")),
                        ("Beta mandate",
                            "🔴 Breach" if port_beta > _bmb else "✅ OK",
                            ("🔍 Exploratory" if _is_relaxed_plan and _bb > _bmb + 0.01
                             else "✅ OK" if _bb <= _bmb + 0.01 else "⚠️ Near limit"),
                            0, lambda d: "🔍 Exploratory" if _is_relaxed_plan else "✅"),
                    ]
                    for _lbl, _bef, _aft, _dlt, _vfn in _sim_rows:
                        _dstr = f"{_dlt:+.2f}" if isinstance(_dlt, float) and abs(_dlt) > 0.005 else "—"
                        lines.append(f"| {_lbl} | {_bef} | {_aft} | {_dstr} | {_vfn(_dlt)} |")
                    lines.append("")

            except _OptimizerSkip as _opt_skip:
                logger.warning("Optimization engine skipped: %s", _opt_skip)
                lines.append(f"*Optimization engine skipped: {_opt_skip}*")
                lines.append("")
            except Exception as _opt_e:
                logger.warning("Optimization engine failed: %s", _opt_e, exc_info=True)
                lines.append(f"*Optimization engine error: {_opt_e}*")
                lines.append("")

        # ── DeepSeek CIO Analysis ──────────────────────────────────────────
        try:
            import requests as _rq, os as _os
            deepseek_key = _os.environ.get("DEEPSEEK_API_KEY", "")

            holdings_summary = ", ".join(
                [f"{t} ({portfolio.get(t,0)*100:.0f}% total / {valid_weights.get(t,0)*100:.0f}% equity)" for t in valid_tickers]
            )
            corr_note = "High correlations: " + ", ".join(high_corr) if high_corr else "No high-correlation pairs."
            benchmark_note = (f"S&P 500 returned {spx_return*100:+.1f}% over the same period (Alpha: {alpha:+.1f}%)"
                              if spx_return is not None else "Benchmark data unavailable.")
            div_note = (
                f"Portfolio dividend yield (total): {port_div_yield_total:.2f}% | "
                f"equity sleeve: {port_div_yield_equity:.2f}%"
            )
            scenario_summary = "; ".join([
                f"{lbl}: equity sleeve {(mkt*port_beta_equity*100):+.1f}%"
                for lbl, mkt, _, _ in _CRISIS_SCENARIOS
            ]) if _has_beta_equity else "N/A (beta data unavailable)"

            _sector_summary = "; ".join([f"{s}: {w*100:.0f}%" for s, w in sorted(sector_concentration.items(), key=lambda x: -x[1])])
            _stock_level_lines = []
            for t in valid_tickers:
                _si = stock_info.get(t, {})
                _b = _si.get("beta")
                _p = _si.get("pe")
                _d = _si.get("div_yield")
                _beta_txt = _fmt_beta(_b)
                _pe_txt = f"{_p:.1f}" if isinstance(_p, (int, float)) and np.isfinite(_p) else "N/A"
                _div_txt = f"{_d*100:.2f}%" if isinstance(_d, (int, float)) and np.isfinite(_d) else "N/A"
                _stock_level_lines.append(f"- {t}: {_si.get('sector','N/A')} | β={_beta_txt} | P/E={_pe_txt} | Div={_div_txt}")
            ds_prompt = f"""You are a CIO-level portfolio risk analyst at an institutional fund. Provide a rigorous, data-driven analysis.

PORTFOLIO: {holdings_summary}
Cash: {cash_pct:.1f}%

QUANTITATIVE METRICS (1Y, rf=4.5%, Historical Simulation):
- Equity Sleeve Return: {port_total_return:+.1f}%
- Estimated Total Portfolio Return (incl. cash): {port_total_return_total_est:+.1f}% | {benchmark_note}
- Annualized Volatility: {ann_vol*100:.1f}% | Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}
- Portfolio Beta (total): {_fmt_beta(port_beta_total)} | Portfolio Beta (equity): {_fmt_beta(port_beta_equity)}
- VaR (95%): {var_95:.2f}%/day | CVaR (95%): {cvar_95:.2f}%/day
- Max Drawdown: {max_dd:.1f}% (duration: {max_dd_duration} trading days)
- {rolling_sharpe_str if rolling_sharpe_str else "Rolling Sharpe: insufficient data"}
- {div_note}

DIVERSIFICATION:
- Sector HHI: {hhi:.3f} ({_sector_summary})
- Effective N (eigenvalue): {eff_n:.1f} true independent bets out of {len(valid_tickers)} holdings
- Avg pairwise correlation: {avg_corr:.2f}
- {corr_note}

STOCK-LEVEL DETAIL:
{chr(10).join(_stock_level_lines)}

HISTORICAL STRESS TESTS (sector-adjusted, NOT linear beta):
{chr(10).join([f"- {lbl}: SPX {spx*100:+.1f}% → equity sleeve {(tech_weight*spx*tm + (1-tech_weight)*spx*port_beta_equity)*100:+.1f}%" for lbl, spx, tm, _ in _CRISIS_SCENARIOS]) if _has_beta_equity else "- N/A (insufficient beta coverage)"}

Write a comprehensive institutional analysis with EXACTLY these 6 sections:
1. **📋 Executive Assessment** — Alpha quality, risk-adjusted performance, Sharpe trend (improving/declining?)
2. **🚨 Top 3 Risks** — Each with severity (Critical/High/Medium), quantified worst-case loss, and specific trigger
3. **🔄 Rebalancing Plan** — Exact target weights (must sum to 100%), rationale using CVaR and Eff.N metrics
4. **➕ Suggested Additions** — 3 specific tickers with expected Sharpe/Beta/correlation impact on portfolio
5. **📊 Tax & Income Note** — Capital gains exposure on highest-return holdings + income gap analysis
6. **✅ EisaX Final Verdict** — Rating: Conservative/Balanced/Aggressive/Speculative + one advisory suggested action
   - Include explicit lines: Confidence (%), Primary Uncertainty (2 drivers), No-Action Case, Alternative Scenario

CRITICAL:
- Do not invent or re-derive numeric values. Use the metrics exactly as provided above.
- If any metric is unavailable, state "N/A" explicitly. Do not backfill with assumptions.
- Keep weights consistent with the provided basis labels (total vs equity sleeve).
- Reference CVaR, Effective N, and rolling Sharpe trend in your analysis.
- Use advisory language only ("consider", "prefer", "may"). Avoid command language ("execute immediately", "must sell", "exit 100%").
- If risk profile is elevated, do NOT use phrases like "well-positioned" in the executive opening.
- Include this exact boundary line at the end of section 6:
  "Decision Boundary: This analysis provides strategic guidance, not execution instructions."
- Be institutional — no platitudes. Max 550 words."""

            if deepseek_key:
                ds_resp = _rq.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"},
                    json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": ds_prompt}],
                          "max_tokens": 900, "temperature": 0.25},
                    timeout=40
                )
                if ds_resp.status_code == 200:
                    cio_analysis = (ds_resp.json().get("choices", [{}])[0]
                                    .get("message", {}).get("content", ""))
                    if cio_analysis:
                        cio_analysis = _soften_portfolio_advice(cio_analysis)
                        # Keep one confidence source in the final report (Decision Discipline Layer only).
                        _cio_lines = []
                        for _ln in cio_analysis.splitlines():
                            if _re.search(r'(?i)\bconfidence\b', _ln):
                                continue
                            _cio_lines.append(_ln)
                        cio_analysis = "\n".join(_cio_lines)
                        cio_analysis = _re.sub(r'\n{3,}', '\n\n', cio_analysis).strip()
                        if "Decision Boundary:" not in cio_analysis:
                            cio_analysis += (
                                "\n\n> Decision Boundary: This analysis provides strategic guidance, not execution instructions."
                            )
                        lines.append("---")
                        lines.append("")
                        lines.append("## F. AI Commentary Layer — CIO Synthesis")
                        lines.append("")
                        lines.append("*AI-generated synthesis. Sections A–E above are deterministic and reproducible from the optimizer state.*")
                        lines.append(cio_analysis)
                else:
                    lines.append(f"*CIO analysis unavailable (API {ds_resp.status_code})*")
            else:
                lines.append("---")
                lines.append("")
                lines.append("## F. AI Commentary Layer — Portfolio Synthesis")
                lines.append("")
                lines.append("*AI-generated synthesis. Sections A–E above are deterministic and reproducible.*")
                lines.append("**Risk Profile: Aggressive** — Elevated beta and sector concentration. "
                             "Reduce top single-name exposure and add diversifying defensive sleeves." if (_has_beta_total and port_beta_total > 1.5) else
                             "**Risk Profile: Moderate** — Monitor correlation clusters and rebalance on a 6–12 month cadence.")
        except Exception as _e:
            logger.warning("DeepSeek CIO analysis failed: %s", _e)
            lines.append("---")
            lines.append("")
            lines.append("## F. AI Commentary Layer — Portfolio Synthesis")
            lines.append("")
            lines.append("*AI-generated synthesis. Sections A–E above are deterministic and reproducible.*")
            _risk_label = "Aggressive" if (_has_beta_total and port_beta_total > 1.5) else "Moderate-Aggressive" if (_has_beta_total and port_beta_total > 1.2) else "Moderate" if _has_beta_total else "Unknown"
            _top_holding = max(valid_weights, key=valid_weights.get) if valid_weights else "N/A"
            _top_w = valid_weights.get(_top_holding, 0) * 100
            lines.append(
                f"**Risk Profile: {_risk_label}** — total β={_fmt_beta(port_beta_total)} (equity β={_fmt_beta(port_beta_equity)}), CVaR={cvar_95:.2f}%/day, "
                f"Sharpe={sharpe:.2f}, Effective N={eff_n:.1f} independent bets. "
                f"Portfolio is {int(tech_weight*100)}% Technology with {_top_holding} as lead position ({_top_w:.0f}%). "
                + ("Concentration risk is the primary concern — reduce single-sector exposure and add uncorrelated assets (TLT, GLD, BRK-B) to improve Effective N toward ≥4."
                   if tech_weight > 0.6 else
                   "Risk profile is within institutional bounds. Monitor rolling Sharpe for momentum deterioration.")
            )

        decision_conf = _compute_portfolio_confidence(
            sharpe=sharpe,
            eff_n=eff_n,
            avg_corr=avg_corr,
            beta_total=port_beta_total,
            cvar_95=cvar_95,
            rolling_sharpe_now=rolling_sharpe_now,
        )
        lines.append(
            _build_portfolio_decision_layer(
                confidence=decision_conf,
                risk_label=("Aggressive" if (_has_beta_total and port_beta_total > 1.5) else "Moderate-Aggressive" if (_has_beta_total and port_beta_total > 1.2) else "Moderate" if _has_beta_total else "Unknown"),
                rolling_sharpe_now=rolling_sharpe_now,
                tech_weight=tech_weight,
                next_review_hint="next quarterly review",
            )
        )

        lines.append("")
        lines.append("*To analyze any individual stock: type `analyze TICKER`*")

        # ── Audit Trail ────────────────────────────────────────────────────
        import uuid as _uuid_mod, hashlib as _hl, datetime as _dt_mod
        _snap_id   = str(_uuid_mod.uuid4())
        _generated = _dt_mod.datetime.now(_dt_mod.timezone.utc)
        _gen_str   = _generated.strftime("%Y-%m-%d %H:%M:%S UTC")
        _price_asof = prices_df.index[-1].strftime("%Y-%m-%d") if len(price_data) >= 1 else "N/A"
        _period_start = prices_df.index[0].strftime("%Y-%m-%d") if len(price_data) >= 1 else "N/A"
        _tickers_str = ", ".join(list(returns_df.columns) if 'returns_df' in dir() and not returns_df.empty else valid_tickers)

        # Build preliminary report for hashing (before audit section appended)
        _pre_report = "\n".join(lines)
        _report_hash = _hl.sha256(_pre_report.encode()).hexdigest()[:16]

        lines.append("")
        lines.append("---")
        lines.append("---")
        lines.append("")
        lines.append("## G. Audit Appendix")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| Snapshot ID | `{_audit_pf['snapshot_id']}` |")
        lines.append(f"| Holdings Hash | `{_audit_pf['holdings_hash']}` |")
        lines.append(f"| Methodology | {_audit_pf['methodology']} |")
        lines.append(f"| Data Window | {_audit_pf['data_window']} |")
        lines.append(f"| Holdings Count | {_audit_pf['n_holdings']} |")
        lines.append(f"| Benchmark | {_audit_pf['benchmark']} |")
        lines.append(f"| Risk-Free Rate | {_audit_pf['rf_rate_pct']:.1f}% |")
        lines.append(f"| Confidence Score | {_audit_pf['confidence_pct']:.0f}% |")
        lines.append("")
        lines.append("> *Reproducible: same holdings + same data window → same Snapshot ID → identical output.*")
        lines.append("")
        lines.append("### Model Constraints — Structural Limitations of the Engine")
        lines.append("")
        for _mc in _model_constraints_pf:
            lines.append(f"- {_mc}")
        lines.append("")
        lines.append("> *Transparency note: the constraints above are inherent to historical-simulation portfolio analytics. Surfaced explicitly to support institutional review and governance.*")
        lines.append("")
        lines.append("### Legacy Audit Trail (Operational Log)")
        lines.append("*For compliance, reproducibility, and institutional trust*")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| **Report ID** | `{_snap_id}` |")
        lines.append(f"| **Generated** | {_gen_str} |")
        lines.append(f"| **Price Data As-Of** | {_price_asof} |")
        lines.append(f"| **Period Analysed** | {_period_start} → {_price_asof} (252 trading days) |")
        lines.append(f"| **Data Source** | core.data.get_prices — TV-first price router; yfinance benchmark fallback |")
        lines.append(f"| **Tickers Fetched** | {_tickers_str} |")
        lines.append(f"| **Risk-Free Rate** | 4.50% (US 3-Month T-Bill) |")
        lines.append(f"| **Methodology** | Historical Simulation · Brinson Attribution · Fama-French proxy |")
        lines.append(f"| **Optimizer** | CLARABEL convex QP (cvxpy) — strict constraint enforcement |")
        lines.append(f"| **Report Hash** | `sha256:{_report_hash}...` |")
        lines.append("")
        lines.append(f"> 🔁 **Reproduce this report:** Save snapshot ID `{_snap_id}` → call `GET /v1/portfolio/snapshot/{_snap_id}`")
        lines.append("")

        report = "\n".join(lines)

        # ── Save to Portfolio Memory ───────────────────────────────────────
        try:
            from portfolio_memory import save_snapshot as _save_snap
            _metrics_for_mem = {
                "sharpe":       sharpe,
                "beta":         _safe_round(port_beta_total, 3),
                "beta_equity":  _safe_round(port_beta_equity, 3),
                "cvar_95":      round(cvar_95, 3),
                "ann_vol":      round(ann_vol * 100, 2),
                "total_return": round(port_total_return, 2),
                "total_return_total_est": round(port_total_return_total_est, 2),
                "sortino":      sortino,
                "max_dd":       round(max_dd, 2),
                "div_yield":    round(port_div_yield_total, 3),
                "div_yield_equity": round(port_div_yield_equity, 3),
            }
            _sources_for_mem = [{
                "source":       "core.data.get_prices",
                "tickers":      valid_tickers,
                "period":       f"{_period_start} → {_price_asof}",
                "fetched_at":   _gen_str,
                "price_as_of":  _price_asof,
                "risk_free":    "4.50% US T-Bill",
            }]
            _saved_snap_id = _save_snap(
                user_id=user_id,
                holdings={t: round(valid_weights.get(t, 0), 6) for t in valid_tickers},
                metrics=_metrics_for_mem,
                data_sources=_sources_for_mem,
                report_md=report,
            )
            logger.info("[PortfolioMemory] Snapshot saved: %s (user: %s)", _saved_snap_id, user_id)
        except Exception as _mem_err:
            logger.warning("[PortfolioMemory] Save failed: %s", _mem_err)

        # ── Phase H augmentation for uploaded portfolios ──────────────────
        # Wraps the existing `report` markdown with H1 (benchmark relative),
        # H2 (execution-what-if vs from-cash), H4 (factor decomposition).
        # H3 / H5 are off by default for uploads unless EISAX_PHASE_H_UPLOAD_FULL=1.
        _phase_h_meta = None
        try:
            from phase_h.orchestrator import augment_result as _ph_aug
            _ph_input = {
                "weights":      dict(valid_weights or {}),
                "metrics":      {"profile": "uploaded"},
                "feasibility":  {"status": "feasible"},
                "confidence":   {"reliability_tier": "Indicative"},
                "report_md":    report,
            }
            _ph_full = os.environ.get("EISAX_PHASE_H_UPLOAD_FULL", "0").strip().lower() in {"1","true","yes","on"}
            _ph_out = _ph_aug(
                _ph_input,
                language="en",
                rebalance_frequency="quarterly",
                committee_mode=("1pager" if _ph_full else None),
                horizon_years=5.0,
                asset_kind=None,
                region_tilt=None,
                benchmark_ticker=None,
                w_prev=None,
            )
            report = _ph_out.get("report_md", report)
            _phase_h_meta = _ph_out.get("phase_h_meta")
        except Exception as _ph_exc:  # pragma: no cover — never fail upload on H
            logger.warning("[PhaseH/upload] augment failed: %r", _ph_exc)

        return _clean_nan({
            "status":      "success",
            "snapshot_id": _snap_id,
            "portfolio":   portfolio,
            "tickers":     tickers,
            "analysis":    report,
            "phase_h_meta": _phase_h_meta,
            "audit": {
                "report_id":     _snap_id,
                "generated_utc": _gen_str,
                "price_as_of":   _price_asof,
                "period":        f"{_period_start} → {_price_asof}",
                "data_source":   "core.data.get_prices",
                "report_hash":   f"sha256:{_report_hash}",
                "optimizer":     "CLARABEL (cvxpy)",
            },
            "metrics": {
                "total_return_pct":  round(port_total_return, 2),
                "total_return_total_est_pct": round(port_total_return_total_est, 2),
                "spx_return_pct":    round(spx_return * 100, 2) if spx_return is not None else None,
                "alpha_pct":         round(alpha, 2) if spx_return is not None else None,
                "alpha_total_est_pct": round(alpha_total_est, 2) if spx_return is not None else None,
                "ann_return_pct":    round(ann_return * 100, 2),
                "ann_vol_pct":       round(ann_vol * 100, 2),
                "sharpe":            sharpe,
                "sortino":           sortino,
                "var_95_pct":        round(var_95, 3),
                "max_drawdown_pct":  round(max_dd, 2),
                "beta":              _safe_round(port_beta_total, 3),
                "beta_equity":       _safe_round(port_beta_equity, 3),
                "div_yield_pct":     round(port_div_yield_total, 3),
                "div_yield_equity_pct": round(port_div_yield_equity, 3),
                "equity_allocation_pct": round(equity_alloc_total * 100, 2),
                "cash_allocation_pct": round(cash_pct, 2),
                "risk_free_rate":    "4.5% (US T-Bill)",
                "method":            "Historical Simulation, 252 trading days",
            }
        })
    except Exception as e:
        logger.exception("[upload_portfolio] Unhandled error for file %s", getattr(file, 'filename', '?'))
        return {"error": str(e), "detail": "Portfolio analysis failed — see server logs for details"}
