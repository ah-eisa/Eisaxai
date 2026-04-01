import numpy; import yfinance
import os
import logging
import time as _time
from core.config import (
    APP_DB, STATIC_DIR, EXPORTS_DIR, FILE_CACHE_DIR,
    CHAT_HTML, BACKEND_LOG, ENV_FILE,
)
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import io
import jwt as _jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from core.tts_service import TTSService
from core.orchestrator import MultiAgentOrchestrator
from core.news_aggregator import get_news as _get_aggregated_news
from core.export_engine import export as export_engine
from contextlib import asynccontextmanager
# learning_engine runs as a separate service (eisax-learning.service)

# ── Logging with rotation (max 10MB per file, keep 3 backups) ──────────────
_log_handler = RotatingFileHandler(
    str(BACKEND_LOG),
    maxBytes=10 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_log_handler, logging.StreamHandler()])
logger = logging.getLogger("api_bridge")

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app):
    yield

app = FastAPI(title="InvestWise & EisaX AI Gateway", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

static_dir = str(STATIC_DIR)
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ── Include EisaX News Engine router ─────────────────────────────────────
# IMPORTANT: use include_router (not app.mount) so /v1/news and /v1/chat coexist.
# app.mount("/v1", sub_app) hijacks ALL /v1/* routes including /v1/chat.
try:
    import sys as _sys
    _sys.path.insert(0, "/home/ubuntu/eisax-news")
    from db import init_db as _news_init_db
    from news_api import news_router as _news_router   # APIRouter, not FastAPI app
    from engine import start_scheduler as _start_news_scheduler
    _news_init_db()
    app.include_router(_news_router, prefix="/v1")     # → /v1/news, /v1/news/latest …
    _start_news_scheduler()
    import logging as _lg
    _lg.getLogger(__name__).info("[NewsEngine] Router included at /v1/news — scheduler started")
except Exception as _ne:
    import logging as _lg
    _lg.getLogger(__name__).warning("[NewsEngine] Failed to include router: %s", _ne)

orchestrator = MultiAgentOrchestrator(db_path=str(APP_DB))
tts_service = TTSService()

SECURE_TOKEN = os.getenv("SECURE_TOKEN", "")
# Disk-based file store (shared across all workers)
import json as _json
_FILE_CACHE_DIR = str(FILE_CACHE_DIR)
_FILE_STORE_TTL = 3600  # seconds
os.makedirs(_FILE_CACHE_DIR, exist_ok=True)

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


class MessagePayload(BaseModel):
    message: str = Field(..., max_length=16000)
    user_id: Optional[str] = "admin"
    session_id: Optional[str] = None
    files: Optional[list] = []
    settings: Optional[dict] = None

@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse(str(CHAT_HTML))

@app.get("/v1/chart-data")
async def chart_data(ticker: str = "NVDA", access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    import yfinance as yf
    from datetime import datetime, timedelta
    df = None

    # ── Try yfinance first ────────────────────────────────────────────────────
    try:
        end = datetime.now()
        start = end - timedelta(days=65)
        _df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if not _df.empty:
            df = _df
    except Exception:
        pass

    # ── Fallback: investing.com for UAE/local market tickers ─────────────────
    if df is None or df.empty:
        try:
            from core.market_data_engine import UAE_INVESTING, _fetch_investing
            info = UAE_INVESTING.get(ticker)
            if info:
                from datetime import datetime, timedelta
                start_str = (datetime.now() - timedelta(days=75)).strftime("%Y-%m-%d")
                _df = await run_in_threadpool(_fetch_investing, ticker, info, start_str)
                if _df is not None and not _df.empty:
                    df = _df
        except Exception:
            pass

    if df is None or df.empty:
        return {"error": "No data"}

    import math
    tail = df.tail(60)
    close_col = "Close" if "Close" in tail.columns else tail.columns[0]
    dates_raw  = list(tail.index)
    prices_raw = [float(v) for v in tail[close_col].values]

    # Strip rows where price is NaN/Inf (non-trading days, halted stocks)
    # These cause "Out of range float values are not JSON compliant" errors.
    dates  = [d.strftime("%b %d") for d, p in zip(dates_raw, prices_raw)
              if not (math.isnan(p) or math.isinf(p))]
    prices = [round(p, 2) for p in prices_raw
              if not (math.isnan(p) or math.isinf(p))]

    if not prices:
        return {"error": "No valid price data"}
    return {"dates": dates, "prices": prices, "ticker": ticker}

@app.post("/v1/upload-portfolio")
@limiter.limit("10/minute")
async def upload_portfolio(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form("admin"),
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token")
):
    """Upload CSV/Excel portfolio file and analyze it"""
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
        
        tickers = df[ticker_col].str.upper().tolist()
        
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
        
        # Build Portfolio Risk Report
        import yfinance as yf
        import numpy as np
        from dotenv import load_dotenv
        load_dotenv(str(ENV_FILE))

        valid_tickers = [t for t in tickers if t.upper() not in ["CASH","USD","AED"]]
        valid_weights = {t: portfolio[t] for t in valid_tickers}
        
        # Normalize weights
        total_w = sum(valid_weights.values())
        if total_w > 0:
            valid_weights = {t: w/total_w for t,w in valid_weights.items()}

        # ── Fetch 1yr price history + fundamentals ──────────────────────────────
        RF_RATE = 0.045   # US T-Bill risk-free rate (4.5%)
        LOOKBACK = "1y"

        price_data = {}
        stock_info = {}
        for t in valid_tickers:
            try:
                tk = yf.Ticker(t)
                hist = tk.history(period=LOOKBACK)
                if not hist.empty:
                    price_data[t] = hist["Close"]
                info = tk.info
                # trailingAnnualDividendYield is a reliable decimal fraction (0.004 = 0.4%).
                # dividendYield returns as a percentage value (0.4 for 0.4%) — avoid it.
                _raw_dy = float(info.get("trailingAnnualDividendYield") or 0)
                _safe_dy = min(max(_raw_dy, 0.0), 0.15)   # clamp decimal [0, 15%]
                stock_info[t] = {
                    "price":     info.get("regularMarketPrice") or info.get("previousClose", 0),
                    "beta":      info.get("beta", 1.0),
                    "sector":    info.get("sector", "N/A"),
                    "pe":        info.get("trailingPE", 0),
                    "mktcap":    info.get("marketCap", 0),
                    "div_yield": _safe_dy,   # current market yield, NOT yield-on-cost
                }
            except Exception as _fe:
                logger.debug("Stock data fetch failed for %s: %s", t, _fe)

        # ── Benchmark: S&P 500 ───────────────────────────────────────────────
        spx_return = None
        try:
            spx = yf.Ticker("^GSPC").history(period=LOOKBACK)
            if not spx.empty:
                spx_return = float((spx["Close"].iloc[-1] / spx["Close"].iloc[0]) - 1)
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
        sector_concentration = {}
        factor_exposure = {}

        if len(price_data) >= 2:
            prices_df = pd.DataFrame(price_data).dropna()
            returns_df = prices_df.pct_change().dropna()

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

            port_beta = float(sum(stock_info.get(t,{}).get("beta",1) * valid_weights.get(t,0) for t in valid_tickers))

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
            ann_return, ann_vol, sharpe, var_95, port_beta = 0, 0, 0, 0, 1.0
            port_total_return, max_dd, avg_corr, eff_n, hhi = 0.0, 0.0, 0.0, 1.0, 1.0

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
            blended    = (tech_weight * spx_ret * tech_mult) + (non_tech_w * spx_ret * port_beta)
            icon = "🔴" if blended < 0 else "🟢"
            scenario_lines.append(
                f"| {label} | {spx_ret*100:+.1f}% | **{blended*100:+.1f}%** | *{note}* |"
            )

        # ── Portfolio Dividend Yield (weighted) ──────────────────────────────
        port_div_yield = sum(
            stock_info.get(t, {}).get("div_yield", 0) * valid_weights.get(t, 0)
            for t in valid_tickers
        ) * 100

        cash_pct = (1 - sum(portfolio[t] for t in valid_tickers)) * 100

        # ══════════════════════════════════════════════════════════════════════
        # Build report
        # ══════════════════════════════════════════════════════════════════════
        lines = []
        from datetime import datetime
        now_str = datetime.now().strftime("%B %d, %Y")

        # ── Executive Summary ──────────────────────────────────────────────
        alpha = (port_total_return - (spx_return or 0)*100)
        risk_label = "Aggressive 🔴" if port_beta > 1.5 else "Moderate 🟡" if port_beta > 1.0 else "Conservative 🟢"
        verdict_line = (
            f"Portfolio returned **{port_total_return:+.1f}%** vs S&P 500 **{spx_return*100:+.1f}%** "
            f"(Alpha: **{alpha:+.1f}%**). Risk profile: **{risk_label}**. "
            f"Sharpe: **{sharpe:.2f}** (rf=4.5%, 1Y Historical). "
            f"Immediate action: {'Reduce tech concentration & add defensive assets.' if port_beta > 1.5 else 'Monitor correlation clusters.' if high_corr else 'Portfolio is well-positioned.'}"
            if spx_return is not None else
            f"Portfolio 1Y Return: **{port_total_return:+.1f}%**. Risk: **{risk_label}**. Sharpe: **{sharpe:.2f}**."
        )
        lines.append("# 📊 EisaX Portfolio Risk Report")
        lines.append(f"**Date:** {now_str}  |  **Period:** 1 Year  |  **Risk-Free Rate:** 4.5% (US T-Bill)")
        lines.append("")
        lines.append("## 🎯 Executive Summary")
        lines.append(f"> {verdict_line}")
        lines.append("")

        # ── Holdings ───────────────────────────────────────────────────────
        lines.append("## 📋 Holdings")
        lines.append("| Ticker | Weight | Sector | Beta | P/E | Div Yield |")
        lines.append("|--------|--------|--------|------|-----|-----------|")
        for t in valid_tickers:
            info  = stock_info.get(t, {})
            w_pct = portfolio.get(t, 0) * 100
            dy    = info.get("div_yield", 0) * 100
            lines.append(
                f"| {t} | {w_pct:.1f}% | {info.get('sector','N/A')} "
                f"| {info.get('beta',0) or '—':.2f} "
                f"| {info.get('pe',0) or '—':.1f} "
                f"| {dy:.2f}% |"
            )
        if cash_pct > 0.5:
            lines.append(f"| CASH | {cash_pct:.1f}% | — | 0 | — | 0% |")
        lines.append(f"\n> **Weighted Portfolio Dividend Yield:** {port_div_yield:.2f}%")

        # ── Risk Metrics ───────────────────────────────────────────────────
        lines.append("")
        lines.append("## 📈 Risk Metrics")
        lines.append("*Method: Historical Simulation (252 trading days) · rf = 4.5%*")
        lines.append("")
        lines.append("| Metric | Value | Assessment |")
        lines.append("|--------|-------|------------|")
        lines.append(f"| 1Y Total Return | {port_total_return:+.1f}% | {'🟢 Strong' if port_total_return > 15 else '🟡 Moderate' if port_total_return > 0 else '🔴 Negative'} |")
        if spx_return is not None:
            alpha_icon = "🟢" if alpha > 0 else "🔴"
            lines.append(f"| vs S&P 500 (Alpha) | {alpha:+.1f}% | {alpha_icon} {'Outperforming' if alpha > 0 else 'Underperforming'} benchmark |")
        lines.append(f"| Annualized Volatility | {ann_vol*100:.1f}% | {'🔴 High' if ann_vol > 0.30 else '🟡 Moderate' if ann_vol > 0.15 else '🟢 Low'} |")
        lines.append(f"| Sharpe Ratio (1Y, rf=4.5%) | {sharpe:.2f} | {'🟢 Excellent' if sharpe > 1.5 else '🟡 Acceptable' if sharpe > 0.5 else '🔴 Poor'} |")
        lines.append(f"| Sortino Ratio (1Y, rf=4.5%) | {sortino:.2f} | {'🟢 Good' if sortino > 1.0 else '🟡 Acceptable' if sortino > 0.5 else '🔴 Poor'} downside-adjusted |")
        lines.append(f"| Portfolio Beta (weighted) | {port_beta:.2f} | {'🔴 High Risk' if port_beta > 1.5 else '🟡 Moderate' if port_beta > 1 else '🟢 Defensive'} |")
        lines.append(f"| VaR 95% 1-Day (Historical) | {var_95:.2f}% | 95% of days, loss ≤ this |")
        lines.append(f"| CVaR 95% (Expected Shortfall) | {cvar_95:.2f}% | Avg loss **when** VaR is breached — tail risk |")
        lines.append(f"| Max Drawdown (1Y) | {max_dd:.1f}% | Worst peak-to-trough |")
        lines.append(f"| Max Drawdown Duration | {max_dd_duration}d | Longest time underwater |")
        lines.append(f"| Portfolio Div Yield | {port_div_yield:.2f}% | Weighted annual income |")
        if rolling_sharpe_str:
            lines.append(f"| Rolling Sharpe (63d) | — | {rolling_sharpe_str} |")

        # ── Sector Concentration ───────────────────────────────────────────
        lines.append("")
        lines.append("## 🏭 Sector Exposure")
        lines.append("| Sector | Weight | HHI Contribution |")
        lines.append("|--------|--------|-----------------|")
        for sec, w in sorted(sector_concentration.items(), key=lambda x: -x[1]):
            lines.append(f"| {sec} | {w*100:.1f}% | {w**2:.3f} |")
        hhi_label = "🔴 Highly Concentrated" if hhi > 0.35 else "🟡 Moderately Concentrated" if hhi > 0.18 else "🟢 Diversified"
        lines.append(f"\n> **Sector HHI:** {hhi:.3f} — {hhi_label} *(0=perfect diversification, 1=single sector)*")

        # ── Correlation & Diversification Analysis ────────────────────────
        lines.append("")
        lines.append("## 🔗 Correlation & Diversification")
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
        lines.append("## 🧪 Stress Testing")
        lines.append("*Historical crises use actual observed returns + sector-adjusted amplification (NOT linear beta)*")
        lines.append("")
        lines.append("| Scenario | SPX Actual | Portfolio Est. | Notes |")
        lines.append("|----------|------------|----------------|-------|")
        for sl in scenario_lines:
            lines.append(sl)
        lines.append(f"\n> **Tech Weight:** {tech_weight*100:.0f}% of equity. "
                     f"In risk-off events, tech typically amplifies market moves by 1.2–1.8×. "
                     f"Linear beta alone (β={port_beta:.2f}) understates true drawdown risk.")

        # ── EisaX Risk Assessment ─────────────────────────────────────────
        lines.append("")
        lines.append("## 💡 EisaX Risk Assessment")
        _top_sector = max(sector_concentration, key=sector_concentration.get) if sector_concentration else "N/A"
        _top_sector_pct = sector_concentration.get(_top_sector, 0) * 100
        if port_beta > 1.5:
            lines.append(
                f"🔴 **Aggressive** — β={port_beta:.2f}, CVaR={cvar_95:.2f}%/day, Eff.N={eff_n:.1f} bets. "
                f"{_top_sector_pct:.0f}% in {_top_sector}. "
                f"In a 2022-style selloff your portfolio would have lost ~{abs(tech_weight * -0.33 + (1-tech_weight) * -0.195 * port_beta)*100:.0f}% "
                f"vs SPX -19.5%. Diversification is urgently needed."
            )
        elif port_beta > 1.0:
            lines.append(
                f"🟡 **Moderate-Aggressive** — β={port_beta:.2f}, CVaR={cvar_95:.2f}%/day. "
                "Above-market sensitivity. Trim highest-beta names on strength; add one defensive sector."
            )
        else:
            lines.append(f"🟢 **Balanced** — β={port_beta:.2f}, CVaR={cvar_95:.2f}%/day, Eff.N={eff_n:.1f}. Reasonable risk profile.")
        lines.append("")

        # ══════════════════════════════════════════════════════════════════════
        # ── 1. PERFORMANCE ATTRIBUTION ────────────────────────────────────
        # Shows each holding's contribution to total return (Brinson model)
        # ══════════════════════════════════════════════════════════════════════
        if len(price_data) >= 1:
            lines.append("## 📐 Performance Attribution (1Y)")
            lines.append("*Brinson-Hood-Beebower: each holding's contribution to total portfolio return*")
            lines.append("")
            lines.append("| Ticker | Weight | 1Y Return | Contribution | Attribution |")
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
                lines.append(f"| {t} | {w*100:.1f}% | {ret:+.1f}% | {contrib:+.2f}pp | {bar} |")
            # Unexplained (cash drag + rounding)
            explained = sum(c for _, _, _, c in attr_rows)
            cash_drag = port_total_return - explained
            if abs(cash_drag) > 0.05:
                lines.append(f"| CASH/Other | — | — | {cash_drag:+.2f}pp | {'🔴' if cash_drag < 0 else '⚪'} |")
            lines.append(f"| **TOTAL** | 100% | — | **{port_total_return:+.2f}pp** | |")
            lines.append("")
            # Alpha source: top contributor vs S&P
            if attr_rows and spx_return is not None:
                top_t, top_w, top_ret, top_c = attr_rows[0]
                lines.append(f"> 🏆 **Alpha driver:** {top_t} contributed {top_c:+.1f}pp (+{top_ret:.1f}% × {top_w*100:.0f}% weight). "
                             f"S&P 500 returned {spx_return*100:+.1f}% — alpha gap of {alpha:+.1f}pp.")
            lines.append("")

        # ══════════════════════════════════════════════════════════════════════
        # ── 2. FACTOR EXPOSURE (Fama-French Proxy) ───────────────────────────
        # Approximates factor tilts using beta, P/E, market cap, momentum
        # ══════════════════════════════════════════════════════════════════════
        lines.append("## 🧬 Factor Exposure Analysis")
        lines.append("*Fama-French proxy: factor tilts computed from fundamentals + price momentum*")
        lines.append("")

        factor_scores = {}
        for t in valid_tickers:
            info   = stock_info.get(t, {})
            w      = valid_weights.get(t, 0)
            beta   = info.get("beta", 1.0) or 1.0
            pe     = info.get("pe", 20) or 20
            mc     = info.get("mktcap", 1e10) or 1e10
            # 1Y momentum from price data
            mom = 0.0
            if t in price_data and len(price_data[t]) > 20:
                _s = price_data[t]
                mom = float((_s.iloc[-1] / _s.iloc[max(0, len(_s)-252)]) - 1)

            # Factor scoring (institutional proxy)
            # Market beta exposure
            factor_scores.setdefault("Market (Beta)", 0)
            factor_scores["Market (Beta)"] += beta * w
            # Growth (inverse P/E proxy — high PE = growth tilt)
            growth_score = min(max((pe - 15) / 50, -1), 1)  # normalize
            factor_scores.setdefault("Growth (P/E tilt)", 0)
            factor_scores["Growth (P/E tilt)"] += growth_score * w
            # Size (log market cap — large = negative small-cap factor)
            import math as _math
            size_score = -(_math.log10(mc) - 10) / 3  # large cap = negative SMB
            factor_scores.setdefault("Size (SMB proxy)", 0)
            factor_scores["Size (SMB proxy)"] += size_score * w
            # Momentum (12-1 month)
            factor_scores.setdefault("Momentum (12M)", 0)
            factor_scores["Momentum (12M)"] += mom * w

        lines.append("| Factor | Portfolio Exposure | Interpretation |")
        lines.append("|--------|--------------------|----------------|")
        _factor_labels = {
            "Market (Beta)":       lambda v: ("🔴 High market sensitivity" if v > 1.5 else "🟢 Low market sensitivity" if v < 0.8 else "🟡 Market-like"),
            "Growth (P/E tilt)":   lambda v: ("🔴 Strong growth tilt — high valuation risk" if v > 0.4 else "🟢 Value tilt" if v < -0.1 else "🟡 Blend"),
            "Size (SMB proxy)":    lambda v: ("🟢 Large-cap dominated (low SMB)" if v < -0.1 else "🔴 Small-cap tilt" if v > 0.2 else "🟡 Mid-cap blend"),
            "Momentum (12M)":      lambda v: ("🟢 Strong momentum" if v > 0.20 else "🔴 Negative momentum" if v < -0.10 else "🟡 Neutral momentum"),
        }
        for fname, fval in factor_scores.items():
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
            lines.append("## ⚙️ Institutional Optimization Engine")
            lines.append("*Convex QP · CLARABEL solver · Zero-tolerance constraint enforcement*")
            lines.append("")
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
                _betas_v  = np.array([max(stock_info.get(t,{}).get("beta",1.0) or 1.0, 0.01)
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

            except Exception as _opt_e:
                logger.warning("Optimization engine failed: %s", _opt_e, exc_info=True)
                lines.append(f"*Optimization engine error: {_opt_e}*")
                lines.append("")

        # ── DeepSeek CIO Analysis ──────────────────────────────────────────
        try:
            import requests as _rq, os as _os
            deepseek_key = _os.environ.get("DEEPSEEK_API_KEY", "")

            holdings_summary = ", ".join([f"{t} ({portfolio.get(t,0)*100:.0f}%)" for t in valid_tickers])
            corr_note = "High correlations: " + ", ".join(high_corr) if high_corr else "No high-correlation pairs."
            benchmark_note = (f"S&P 500 returned {spx_return*100:+.1f}% over the same period (Alpha: {alpha:+.1f}%)"
                              if spx_return is not None else "Benchmark data unavailable.")
            div_note = f"Portfolio weighted dividend yield: {port_div_yield:.2f}%"
            scenario_summary = "; ".join([
                f"{lbl}: portfolio {mkt*port_beta*100:+.1f}%"
                for lbl, mkt, _, _ in _CRISIS_SCENARIOS
            ])

            _sector_summary = "; ".join([f"{s}: {w*100:.0f}%" for s, w in sorted(sector_concentration.items(), key=lambda x: -x[1])])
            ds_prompt = f"""You are a CIO-level portfolio risk analyst at an institutional fund. Provide a rigorous, data-driven analysis.

PORTFOLIO: {holdings_summary}
Cash: {cash_pct:.1f}%

QUANTITATIVE METRICS (1Y, rf=4.5%, Historical Simulation):
- Total Return: {port_total_return:+.1f}% | {benchmark_note}
- Annualized Volatility: {ann_vol*100:.1f}% | Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}
- Portfolio Beta: {port_beta:.2f} | VaR (95%): {var_95:.2f}%/day | CVaR (95%): {cvar_95:.2f}%/day
- Max Drawdown: {max_dd:.1f}% (duration: {max_dd_duration} trading days)
- {rolling_sharpe_str if rolling_sharpe_str else "Rolling Sharpe: insufficient data"}
- {div_note}

DIVERSIFICATION:
- Sector HHI: {hhi:.3f} ({_sector_summary})
- Effective N (eigenvalue): {eff_n:.1f} true independent bets out of {len(valid_tickers)} holdings
- Avg pairwise correlation: {avg_corr:.2f}
- {corr_note}

STOCK-LEVEL DETAIL:
{chr(10).join([f"- {t}: {stock_info.get(t,{}).get('sector','N/A')} | β={stock_info.get(t,{}).get('beta',1):.2f} | P/E={stock_info.get(t,{}).get('pe',0) or 'N/A'} | Div={stock_info.get(t,{}).get('div_yield',0)*100:.2f}%" for t in valid_tickers])}

HISTORICAL STRESS TESTS (sector-adjusted, NOT linear beta):
{chr(10).join([f"- {lbl}: SPX {spx*100:+.1f}% → portfolio {(tech_weight*spx*tm + (1-tech_weight)*spx*port_beta)*100:+.1f}%" for lbl, spx, tm, _ in _CRISIS_SCENARIOS])}

Write a comprehensive institutional analysis with EXACTLY these 6 sections:
1. **📋 Executive Assessment** — Alpha quality, risk-adjusted performance, Sharpe trend (improving/declining?)
2. **🚨 Top 3 Risks** — Each with severity (Critical/High/Medium), quantified worst-case loss, and specific trigger
3. **🔄 Rebalancing Plan** — Exact target weights (must sum to 100%), rationale using CVaR and Eff.N metrics
4. **➕ Suggested Additions** — 3 specific tickers with expected Sharpe/Beta/correlation impact on portfolio
5. **📊 Tax & Income Note** — Capital gains exposure on highest-return holdings + income gap analysis
6. **✅ EisaX Final Verdict** — Rating: Conservative/Balanced/Aggressive/Speculative + one precise action

CRITICAL: Reference CVaR, Effective N, and rolling Sharpe trend in your analysis. Be institutional — no platitudes. Max 550 words."""

            if deepseek_key:
                ds_resp = _rq.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"},
                    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": ds_prompt}],
                          "max_tokens": 900, "temperature": 0.25},
                    timeout=40
                )
                if ds_resp.status_code == 200:
                    cio_analysis = (ds_resp.json().get("choices", [{}])[0]
                                    .get("message", {}).get("content", ""))
                    if cio_analysis:
                        lines.append("## 🧠 CIO Deep Analysis (AI-Powered)")
                        lines.append(cio_analysis)
                else:
                    lines.append(f"*CIO analysis unavailable (API {ds_resp.status_code})*")
            else:
                lines.append("## 🧠 Portfolio Assessment")
                lines.append("**Risk Level: Aggressive** — High-beta tech concentration. "
                              "Reduce TSLA, add VIG/XLV/JPM for balance." if port_beta > 1.5 else
                              "**Risk Level: Moderate** — Monitor correlation clusters and rebalance quarterly.")
        except Exception as _e:
            logger.warning("DeepSeek CIO analysis failed: %s", _e)
            lines.append("## 🧠 Portfolio Assessment")
            _risk_label = "Aggressive" if port_beta > 1.5 else "Moderate-Aggressive" if port_beta > 1.2 else "Moderate"
            _top_holding = max(valid_weights, key=valid_weights.get) if valid_weights else "N/A"
            _top_w = valid_weights.get(_top_holding, 0) * 100
            lines.append(
                f"**Risk Profile: {_risk_label}** — β={port_beta:.2f}, CVaR={cvar_95:.2f}%/day, "
                f"Sharpe={sharpe:.2f}, Effective N={eff_n:.1f} independent bets. "
                f"Portfolio is {int(tech_weight*100)}% Technology with {_top_holding} as lead position ({_top_w:.0f}%). "
                + ("Concentration risk is the primary concern — reduce single-sector exposure and add uncorrelated assets (TLT, GLD, BRK-B) to improve Effective N toward ≥4."
                   if tech_weight > 0.6 else
                   "Risk profile is within institutional bounds. Monitor rolling Sharpe for momentum deterioration.")
            )

        lines.append("")
        lines.append("*To analyze any individual stock: type `analyze TICKER`*")

        # ── Audit Trail ────────────────────────────────────────────────────
        import uuid as _uuid_mod, hashlib as _hl, datetime as _dt_mod
        _snap_id   = str(_uuid_mod.uuid4())
        _generated = _dt_mod.datetime.now(_dt_mod.timezone.utc)
        _gen_str   = _generated.strftime("%Y-%m-%d %H:%M:%S UTC")
        _price_asof = prices_df.index[-1].strftime("%Y-%m-%d") if len(price_data) >= 2 else "N/A"
        _period_start = prices_df.index[0].strftime("%Y-%m-%d") if len(price_data) >= 2 else "N/A"
        _tickers_str = ", ".join(_opt_tickers if 'price_data' in dir() and len(price_data) >= 2 else valid_tickers)

        # Build preliminary report for hashing (before audit section appended)
        _pre_report = "\n".join(lines)
        _report_hash = _hl.sha256(_pre_report.encode()).hexdigest()[:16]

        lines.append("")
        lines.append("---")
        lines.append("## 📋 Audit Trail")
        lines.append("*For compliance, reproducibility, and institutional trust*")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| **Report ID** | `{_snap_id}` |")
        lines.append(f"| **Generated** | {_gen_str} |")
        lines.append(f"| **Price Data As-Of** | {_price_asof} |")
        lines.append(f"| **Period Analysed** | {_period_start} → {_price_asof} (252 trading days) |")
        lines.append(f"| **Data Source** | Yahoo Finance (yfinance) — 15-min delayed |")
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
                "beta":         round(port_beta, 3),
                "cvar_95":      round(cvar_95, 3),
                "ann_vol":      round(ann_vol * 100, 2),
                "total_return": round(port_total_return, 2),
                "sortino":      sortino,
                "max_dd":       round(max_dd, 2),
                "div_yield":    round(port_div_yield, 3),
            }
            _sources_for_mem = [{
                "source":       "Yahoo Finance (yfinance)",
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

        return {
            "status":      "success",
            "snapshot_id": _snap_id,
            "portfolio":   portfolio,
            "tickers":     tickers,
            "analysis":    report,
            "audit": {
                "report_id":     _snap_id,
                "generated_utc": _gen_str,
                "price_as_of":   _price_asof,
                "period":        f"{_period_start} → {_price_asof}",
                "data_source":   "Yahoo Finance (yfinance)",
                "report_hash":   f"sha256:{_report_hash}",
                "optimizer":     "CLARABEL (cvxpy)",
            },
            "metrics": {
                "total_return_pct":  round(port_total_return, 2),
                "spx_return_pct":    round(spx_return * 100, 2) if spx_return is not None else None,
                "alpha_pct":         round(alpha, 2) if spx_return is not None else None,
                "ann_return_pct":    round(ann_return * 100, 2),
                "ann_vol_pct":       round(ann_vol * 100, 2),
                "sharpe":            sharpe,
                "sortino":           sortino,
                "var_95_pct":        round(var_95, 3),
                "max_drawdown_pct":  round(max_dd, 2),
                "beta":              round(port_beta, 3),
                "div_yield_pct":     round(port_div_yield, 3),
                "risk_free_rate":    "4.5% (US T-Bill)",
                "method":            "Historical Simulation, 252 trading days",
            }
        }
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO MEMORY API
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/v1/portfolio/history/{user_id}")
async def portfolio_history(
    user_id: str,
    limit: int = 20,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    """
    Return portfolio snapshot history for a user.
    Shows how allocation + metrics evolved over time.
    """
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        from portfolio_memory import get_user_snapshots, get_performance_history
        snapshots = get_user_snapshots(user_id, limit=limit)
        history   = get_performance_history(user_id, limit=limit)
        return {
            "user_id":    user_id,
            "count":      len(snapshots),
            "snapshots":  snapshots,
            "performance_history": history,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/v1/portfolio/snapshot/{snapshot_id}")
async def get_portfolio_snapshot(
    snapshot_id: str,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    """
    Retrieve a specific portfolio snapshot by ID — full report + audit data.
    Enables report reproducibility.
    """
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        from portfolio_memory import get_snapshot
        snap = get_snapshot(snapshot_id)
        if not snap:
            raise HTTPException(status_code=404, detail=f"Snapshot {snapshot_id} not found")
        return snap
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


@app.get("/v1/portfolio/compare")
async def compare_portfolio_snapshots(
    snap_a: str,
    snap_b: str,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    """
    Compare two portfolio snapshots: allocation drift + metric changes.
    Use to track how a portfolio evolved between rebalances.
    """
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        from portfolio_memory import compare_snapshots
        diff = compare_snapshots(snap_a, snap_b)
        if not diff:
            raise HTTPException(status_code=404, detail="One or both snapshot IDs not found")

        # Build human-readable markdown summary
        md_lines = [
            f"## 📊 Portfolio Comparison",
            f"**{diff['date_a']}** → **{diff['date_b']}**",
            "",
            "### Allocation Changes",
            "| Ticker | Before | After | Δ | Direction |",
            "|--------|--------|-------|---|-----------|",
        ]
        for t, d in sorted(diff["allocation_diff"].items()):
            direction = "🟢 Added" if d["before"] == 0 else "🔴 Removed" if d["after"] == 0 else ("🔼 Increased" if d["delta"] > 0 else "🔽 Decreased" if d["delta"] < 0 else "⚪ Unchanged")
            md_lines.append(f"| **{t}** | {d['before']:.1f}% | {d['after']:.1f}% | {d['delta']:+.1f}pp | {direction} |")

        md_lines += ["", "### Metric Changes", "| Metric | Before | After | Δ | Better? |", "|--------|--------|-------|---|---------|"]
        for key, label, better_if in [
            ("sharpe",       "Sharpe Ratio",   "higher"),
            ("beta",         "Portfolio Beta", "lower"),
            ("cvar_95",      "CVaR 95%/day",   "higher"),
            ("ann_vol",      "Ann. Volatility","lower"),
            ("total_return", "1Y Total Return","higher"),
        ]:
            d = diff["metric_diff"].get(key)
            if d is None:
                continue
            improved = (d["delta"] > 0) == (better_if == "higher")
            icon = "✅" if improved else "🔴" if d["delta"] != 0 else "⚪"
            md_lines.append(f"| {label} | {d['before']:.2f} | {d['after']:.2f} | {d['delta']:+.2f} | {icon} |")

        diff["summary_md"] = "\n".join(md_lines)
        return diff
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


@app.post("/v1/global-allocate")
async def global_allocate(
    request: Request,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    """
    🌍 Global Allocation Engine — cross-market QP optimization.
    Allocates across US / GCC / Egypt / Crypto / Gold / Bonds.

    Body (JSON):
    {
        "profile":          "conservative" | "balanced" | "growth" | "aggressive",
        "region_include":   ["US","GCC","Gold"],      // optional: only these
        "region_exclude":   ["Crypto"],               // optional: exclude these
        "custom_caps":      {"Crypto": 0.10},         // optional: override caps
        "port_value_usd":   100000,                   // optional: $100k default
        "rf_rate":          0.045                     // optional: risk-free rate
    }
    """
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        body = await request.json()
    except Exception:
        body = {}

    profile          = body.get("profile", "balanced")
    region_include   = body.get("region_include")
    region_exclude   = body.get("region_exclude")
    custom_caps      = body.get("custom_caps")
    port_value_usd   = float(body.get("port_value_usd", 100_000))
    rf_rate          = float(body.get("rf_rate", 0.045))

    try:
        from global_allocator import allocate
        result = allocate(
            profile=profile,
            region_include=region_include,
            region_exclude=region_exclude,
            custom_caps=custom_caps,
            rf_rate=rf_rate,
            port_value_usd=port_value_usd,
        )
        return result
    except Exception as e:
        logger.error("Global allocator error: %s", e, exc_info=True)
        return {"error": str(e)}


@app.get("/v1/global-allocate/profiles")
async def global_allocate_profiles(
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    """List available risk profiles and regions for the Global Allocation Engine."""
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        from global_allocator import _PROFILES, _UNIVERSE
        regions = sorted(set(a.region for a in _UNIVERSE))
        return {
            "profiles": {
                k: {"label": v["label"], "description": v["description"],
                    "max_beta": v["max_beta"], "max_vol": v["max_vol"]}
                for k, v in _PROFILES.items()
            },
            "regions":  regions,
            "assets":   [{"name": a.name, "region": a.region, "proxy": a.proxy,
                          "description": a.description} for a in _UNIVERSE],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/v1/portfolio/performance/{user_id}")
async def portfolio_performance_chart(
    user_id: str,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    """
    Return time-series of key metrics across all snapshots for chart rendering.
    Frontend can plot Sharpe / Beta / CVaR evolution over time.
    """
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        from portfolio_memory import get_performance_history
        history = get_performance_history(user_id, limit=50)
        return {
            "user_id": user_id,
            "data_points": len(history),
            "series": {
                "dates":        [h["date"] for h in history],
                "sharpe":       [h["sharpe"] for h in history],
                "beta":         [h["beta"] for h in history],
                "cvar_95":      [h["cvar_95"] for h in history],
                "total_return": [h["total_return"] for h in history],
                "ann_vol":      [h["ann_vol"] for h in history],
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/upload")
@limiter.limit("10/minute")
async def upload_file_ui(request: Request, file: UploadFile = File(...), access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    """Receive file from chat UI, extract text via Gemini Vision or file_processor."""
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    import uuid as _uuid, base64 as _b64
    from core.file_processor import process_file
    raw = await file.read()
    b64 = _b64.b64encode(raw).decode()
    result = process_file(file.filename, b64)
    file_id = str(_uuid.uuid4())
    _evict_old_files()
    _file_store_set(file_id, {
        "id": file_id,
        "filename": file.filename,
        "text": result.get("text", ""),
        "error": result.get("error"),
        "_ts": _time.time(),
    })
    return {"status": "received", "file_id": file_id, "filename": file.filename}

@app.get("/health")
async def health(access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    import psutil, time
    uptime = time.time() - psutil.boot_time()
    mem = psutil.virtual_memory()
    return {
        "status": "online",
        "agent": "EisaX General AI",
        "uptime_hours": round(uptime / 3600, 1),
        "memory_used_pct": round(mem.percent, 1),
        "cpu_pct": round(psutil.cpu_percent(interval=0.5), 1),
    }

from fastapi.concurrency import run_in_threadpool
import pandas as pd
import io

def _coerce_chat_payload(raw: dict) -> MessagePayload:
    """Accept legacy chat payloads and normalize to MessagePayload."""
    data = dict(raw or {})
    if "message" not in data:
        legacy_text = data.get("text") or data.get("query") or data.get("prompt")
        if isinstance(legacy_text, str):
            data["message"] = legacy_text
    if not data.get("user_id"):
        data["user_id"] = "admin"
    return MessagePayload(**data)

@app.post("/v1/chat")
@limiter.limit("30/minute")
async def unified_chat(
    payload: MessagePayload,
    request: Request,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token")
):
    """نقطة الدخول الرئيسية للمحادثة - مع الحماية"""

    # Accept both X-API-Key and access-token headers (frontend uses access-token)
    _token = access_token or access_token_alt
    if _token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")

    client_ip = request.headers.get("X-Real-IP") or request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or str(request.client.host)
    user_agent = request.headers.get("User-Agent", "")

    # Block check
    if orchestrator.session_mgr.is_user_blocked(payload.user_id):
        raise HTTPException(status_code=403, detail="Your account has been suspended. Please contact support.")

    # Rate limit check
    if orchestrator.session_mgr.is_user_rate_limited(payload.user_id):
        raise HTTPException(status_code=429, detail="Daily message limit reached. Please try again tomorrow.")

    # IP block check
    if orchestrator.session_mgr.is_ip_blocked(client_ip):
        raise HTTPException(status_code=403, detail="Access denied from this network.")

    session_id = payload.session_id or orchestrator.session_mgr.get_or_create_session(
        payload.user_id, ip=client_ip, user_agent=user_agent
    )
    orchestrator.session_mgr.get_or_create_session(
        payload.user_id, session_id=session_id, ip=client_ip, user_agent=user_agent
    )

    # Admin message injection — deliver queued messages before processing user input
    pending = orchestrator.session_mgr.get_pending_admin_messages(payload.user_id)
    if pending:
        orchestrator.session_mgr.mark_admin_messages_delivered(payload.user_id)
        combined = "\n\n".join(f"📢 {m['content']}" for m in pending)
        orchestrator.session_mgr.save_message(session_id, payload.user_id, "assistant", combined)
        return {
            "reply": combined,
            "session_id": session_id,
            "agent": "Admin",
            "model": None,
            "download_url": None,
            "format": None,
        }

    message = payload.message

    # Inject file content from /upload store via active_file_id
    active_file_id = None
    if payload.settings and isinstance(payload.settings, dict):
        active_file_id = payload.settings.get("active_file_id")
    stored_file = _file_store_get(active_file_id) if active_file_id else None
    if stored_file and stored_file.get("text"):
        file_text = stored_file["text"]
        fname = stored_file.get("filename", "file")
        message = ("[FILE ANALYSIS]" + chr(10)
                   + "File content (" + fname + "):" + chr(10) + chr(10)
                   + file_text[:8000] + chr(10) + chr(10)
                   + "User question: " + message)

    # Process uploaded files
    if payload.files:
        try:
            from core.file_processor import process_file
            extracted_parts = []
            for f in payload.files:
                filename = f.get("filename") or f.get("name", "file")
                b64data = f.get("data", "")
                if not b64data:
                    continue
                res = process_file(filename, b64data)
                if res.get("text"):
                    part = "[File: " + filename + "]" + chr(10) + res["text"][:8000]
                    extracted_parts.append(part)
            if extracted_parts:
                file_block = (chr(10) + chr(10)).join(extracted_parts)
                message = ("[FILE ANALYSIS]" + chr(10) + "File content below:" + chr(10) + chr(10)
                           + file_block + chr(10) + chr(10)
                           + "User question: " + message)
        except Exception as e:
            pass

    result = await orchestrator.process_message(
        user_id=payload.user_id,
        message=message,
        session_id=session_id
    )
    
    return {
        "reply": result.get("reply") or result.get("response") or "",
        "session_id": session_id,
        "agent": result.get("agent_name", "EisaX"),
        "model": result.get("model"),
        "download_url": result.get("download_url"),
        "format": result.get("format")
    }

# Backward-compatible aliases used by older UI pages (/chat and /api/chat).
@app.post("/chat")
@app.post("/api/chat")
@limiter.limit("30/minute")
async def unified_chat_legacy(
    request: Request,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    try:
        raw = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid JSON body.")
    try:
        payload = _coerce_chat_payload(raw)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Request body must include 'message' (or legacy 'text').",
        )
    return await unified_chat(
        payload=payload,
        request=request,
        access_token=access_token,
        access_token_alt=access_token_alt,
    )

# ── SSE Streaming Chat Endpoint ───────────────────────────────────────────────
@app.post("/v1/chat/stream")
@limiter.limit("30/minute")
async def unified_chat_stream(
    payload: MessagePayload,
    request: Request,
    access_token: str = Header(None, alias="X-API-Key"),
    access_token_alt: str = Header(None, alias="access-token"),
):
    """
    Server-Sent Events streaming chat endpoint.
    Returns Content-Type: text/event-stream.

    Each SSE message is a JSON-encoded event:
      data: {"type": "status", "text": "..."}   ← progress / loader text
      data: {"type": "token",  "text": "..."}   ← LLM content chunk
      data: {"type": "done",   "session_id": "...", "agent": "...", "model": "..."}
      data: {"type": "error",  "text": "..."}
      data: [DONE]                               ← stream closed
    """
    _token = access_token or access_token_alt
    if _token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if orchestrator.session_mgr.is_user_blocked(payload.user_id):
        raise HTTPException(status_code=403, detail="Your account has been suspended.")
    if orchestrator.session_mgr.is_user_rate_limited(payload.user_id):
        raise HTTPException(status_code=429, detail="Daily message limit reached.")

    client_ip = (
        request.headers.get("X-Real-IP")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or str(request.client.host)
    )
    session_id = payload.session_id or orchestrator.session_mgr.get_or_create_session(
        payload.user_id, ip=client_ip
    )

    async def _generate():
        try:
            # stream_process_message already yields fully-formatted SSE lines
            async for sse_line in orchestrator.stream_process_message(
                user_id=payload.user_id,
                message=payload.message,
                session_id=session_id,
            ):
                yield sse_line
        except Exception as e:
            yield f'data: {_json.dumps({"type":"error","text":str(e)}, ensure_ascii=False)}\n\n'
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


# --- TTS Endpoint ---

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

@app.post("/v1/tts")
async def text_to_speech(request: TTSRequest, access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        audio_bytes = tts_service.generate_speech(request.text, request.language)
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Admin Endpoints ---

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
if not ADMIN_TOKEN:
    logger.warning("[STARTUP] ADMIN_TOKEN is not set — admin endpoints will be disabled")

def _check_admin(token: str):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin access is not configured")
    if not orchestrator.session_mgr.verify_admin_password(token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Forbidden")

@app.get("/admin/sessions")
async def admin_sessions(access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    from collections import defaultdict
    sessions = orchestrator.session_mgr.get_all_sessions_admin()
    grouped = defaultdict(list)
    for s in sessions:
        grouped[s["user_id"]].append(s)
    result = []
    for uid, user_sessions in grouped.items():
        last = max((s["last_active"] or "") for s in user_sessions)
        is_blocked = any(s.get("blocked") for s in user_sessions)
        profile = orchestrator.session_mgr.get_user_profile(uid)
        daily_count = orchestrator.session_mgr.get_user_daily_count(uid)
        result.append({
            "user_id": uid,
            "session_count": len(user_sessions),
            "total_messages": sum(s["msg_count"] for s in user_sessions),
            "last_active": last,
            "ip": user_sessions[0].get("ip", "—"),
            "user_agent": user_sessions[0].get("user_agent", "—"),
            "blocked": is_blocked,
            "sessions": user_sessions,
            "daily_limit": profile.get("daily_limit", 0),
            "daily_count": daily_count,
            "note": profile.get("note", ""),
            "tier": profile.get("tier", "basic"),
        })
    result.sort(key=lambda x: x["last_active"] or "", reverse=True)
    return result

@app.get("/admin/session/{session_id}")
async def admin_session_detail(session_id: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    return orchestrator.session_mgr.get_chat_history(session_id)

@app.get("/admin/stats")
async def admin_stats(access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    return orchestrator.session_mgr.get_admin_stats()

@app.post("/admin/user/{user_id}/block")
async def block_user(user_id: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    orchestrator.session_mgr.set_user_blocked(user_id, True)
    orchestrator.session_mgr.log_admin_action("block_user", user_id)
    return {"status": "blocked", "user_id": user_id}

@app.post("/admin/user/{user_id}/unblock")
async def unblock_user(user_id: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    orchestrator.session_mgr.set_user_blocked(user_id, False)
    orchestrator.session_mgr.log_admin_action("unblock_user", user_id)
    return {"status": "unblocked", "user_id": user_id}

@app.post("/admin/user/{user_id}/message")
async def send_admin_message(user_id: str, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    content = body.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    orchestrator.session_mgr.queue_admin_message(user_id, content)
    orchestrator.session_mgr.log_admin_action("message_user", user_id, content[:80])
    return {"status": "queued", "user_id": user_id}

@app.get("/admin/messages")
async def get_admin_messages(access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    return orchestrator.session_mgr.get_admin_message_history()

@app.post("/admin/settings/password")
async def change_admin_password(body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    new_password = body.get("new_password", "").strip()
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    orchestrator.session_mgr.change_admin_password(new_password)
    return {"status": "password updated"}

@app.post("/admin/user/{user_id}/limit")
async def set_user_limit(user_id: str, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    daily_limit = int(body.get("daily_limit", 0))
    if daily_limit < 0:
        raise HTTPException(status_code=400, detail="daily_limit must be >= 0")
    orchestrator.session_mgr.set_user_profile(user_id, daily_limit=daily_limit)
    orchestrator.session_mgr.log_admin_action("set_limit", user_id, str(daily_limit))
    return {"status": "ok", "user_id": user_id, "daily_limit": daily_limit}

@app.post("/admin/user/{user_id}/note")
async def set_user_note(user_id: str, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    note = body.get("note", "")
    orchestrator.session_mgr.set_user_profile(user_id, note=note)
    orchestrator.session_mgr.log_admin_action("set_note", user_id, note[:60] if note else "cleared")
    return {"status": "ok", "user_id": user_id}

@app.post("/admin/user/{user_id}/tier")
async def set_user_tier(user_id: str, body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    tier = body.get("tier", "basic")
    if tier not in ("basic", "pro", "vip"):
        raise HTTPException(status_code=400, detail="tier must be basic, pro, or vip")
    orchestrator.session_mgr.set_user_profile(user_id, tier=tier)
    orchestrator.session_mgr.log_admin_action("set_tier", user_id, tier)
    return {"status": "ok", "user_id": user_id, "tier": tier}

@app.post("/admin/broadcast")
async def broadcast_message(body: dict, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    content = body.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    count = orchestrator.session_mgr.broadcast_admin_message(content)
    orchestrator.session_mgr.log_admin_action("broadcast", f"{count} users", content[:80])
    return {"status": "broadcast", "recipients": count}

@app.delete("/admin/user/{user_id}/sessions")
async def delete_user_sessions(user_id: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    count = orchestrator.session_mgr.delete_user_sessions(user_id)
    orchestrator.session_mgr.log_admin_action("delete_sessions", user_id, f"{count} sessions deleted")
    return {"status": "deleted", "user_id": user_id, "sessions_deleted": count}

@app.post("/admin/ip/{ip}/block")
async def block_ip_endpoint(ip: str, body: dict = {}, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    reason = (body or {}).get("reason", "")
    orchestrator.session_mgr.block_ip(ip, reason)
    orchestrator.session_mgr.log_admin_action("block_ip", ip, reason or "no reason")
    return {"status": "blocked", "ip": ip}

@app.post("/admin/ip/{ip}/unblock")
async def unblock_ip_endpoint(ip: str, access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    orchestrator.session_mgr.unblock_ip(ip)
    orchestrator.session_mgr.log_admin_action("unblock_ip", ip)
    return {"status": "unblocked", "ip": ip}

@app.get("/admin/blocked-ips")
async def get_blocked_ips(access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    return orchestrator.session_mgr.get_blocked_ips()

@app.get("/admin/audit-log")
async def get_audit_log(access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    return orchestrator.session_mgr.get_audit_log()

@app.get("/admin/notifications")
async def get_notifications(since: str = "", access_token: str = Header(None, alias="X-Admin-Key")):
    _check_admin(access_token)
    if not since:
        from datetime import datetime, timedelta
        since = (datetime.utcnow() - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    return orchestrator.session_mgr.get_new_activity(since)

@app.get("/admin/export/users")
async def export_users(access_token: str = Header(None, alias="X-Admin-Key")):
    from fastapi.responses import StreamingResponse as SR
    import csv
    _check_admin(access_token)
    from collections import defaultdict
    sessions = orchestrator.session_mgr.get_all_sessions_admin()
    grouped = defaultdict(list)
    for s in sessions:
        grouped[s["user_id"]].append(s)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["User ID", "Sessions", "Total Messages", "Last Active", "IP", "Tier", "Daily Limit", "Blocked"])
    for uid, user_sessions in grouped.items():
        last = max((s["last_active"] or "") for s in user_sessions)
        is_blocked = any(s.get("blocked") for s in user_sessions)
        profile = orchestrator.session_mgr.get_user_profile(uid)
        writer.writerow([
            uid, len(user_sessions),
            sum(s["msg_count"] for s in user_sessions),
            last, user_sessions[0].get("ip", ""),
            profile.get("tier", "basic"),
            profile.get("daily_limit", 0),
            "Yes" if is_blocked else "No"
        ])
    output.seek(0)
    return SR(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=eisax_users_export.csv"}
    )

# --- New History Endpoints ---

@app.get("/api/history")
async def get_history(user_id: Optional[str] = "admin", access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return orchestrator.session_mgr.get_user_sessions(user_id)

@app.get("/api/history/{session_id}")
async def get_session_history(session_id: str, access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return orchestrator.session_mgr.get_chat_history(session_id)

@app.delete("/api/history/{session_id}")
async def delete_session(session_id: str, access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    if (access_token or access_token_alt) != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    orchestrator.session_mgr.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}

@app.post("/v1/export")
async def export_chat(request: Request, access_token: str = Header(None, alias="X-API-Key"), access_token_alt: str = Header(None, alias="access-token")):
    import re, shutil
    try:
        body = await request.json()
    except Exception as _e:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    fmt = body.get("format", "pdf")
    messages = body.get("messages", [])
    title = body.get("title", "EisaX Report")
    smart = [m for m in messages if m.get("role") == "assistant" 
             and len(m.get("content","")) > 200
             and not any(x in m.get("content","") for x in ["Hello!", "Hi!", "How can I help", "مرحباً", "أهلاً"])]
    if not smart:
        smart = messages
    
    # === GLM FORMATTING LAYER ===
    try:
        from core.glm_client import GLMClient
        glm = GLMClient()
        
        # Combine all messages
        combined = "\n\n---\n\n".join([
            m.get("content", "") for m in smart if m.get("content")
        ])
        
        # Let GLM clean and format
        logger.debug("Calling GLM with %d chars", len(combined))
        formatted = glm.prepare_for_export(combined, fmt)
        logger.debug("GLM result: success=%s", formatted.get('success'))

        if formatted.get("success"):
            smart = [{"role": "assistant", "content": formatted["content"]}]
            logger.info("GLM formatted export for %s — new length: %d", fmt, len(formatted['content']))
        else:
            logger.warning("GLM formatting failed: %s", formatted.get('error'))
    except Exception as e:
        logger.error("GLM export prep error: %s", e, exc_info=True)
    
    # Clean emojis for PDF compatibility
    emoji_map = {
        "📊": ">>", "📈": "^", "📉": "v", "🔴": "(SELL)",
        "🟢": "(BUY)", "🎯": "(TARGET)", "📰": ">>", "🔍": ">>",
        "✅": "OK", "➕": "+", "⚠️": "(!)", "💡": ">>",
        "🧠": ">>", "👋": "", "📄": "", "💰": "$",
        "–": "-", "→": "->", "—": "-", "–": "-",
        "—": "-", "’": "'", "“": '"', "”": '"',
        "?": "-"
    }
    
    def clean_content(text):
        for emoji, replacement in emoji_map.items():
            text = text.replace(emoji, replacement)
        return text
    
    smart = [{"role": m["role"], "content": clean_content(m.get("content",""))} for m in smart]
    
    for msg in smart:
        c = msg.get("content","")
        m = re.search(r"EisaX Intelligence Report: ([A-Z]+)", c)
        if m:
            title = f"EisaX Report - {m.group(1)}"
            break
        elif "Portfolio Risk Report" in c:
            title = "EisaX Portfolio Risk Report"
            break
    export_dir = str(EXPORTS_DIR)
    os.makedirs(export_dir, exist_ok=True)
    try:
        # CIO engines for exports
        if fmt in ("pdf", "pdf_ar"):
            from core.cio_pdf import generate_cio_pdf
            import time, re as re2

            _lang = "ar" if fmt == "pdf_ar" else "en"
            _suffix = "_AR" if _lang == "ar" else ""
            filename = "EisaX" + _suffix + "_" + time.strftime("%Y%m%d_%H%M%S") + ".pdf"
            out_path = str(EXPORTS_DIR / filename)

            ticker_m = re2.search(r"EisaX (?:Report|Intelligence Report)[:\s-]+([A-Z]{1,5})", title or "")
            ticker = ticker_m.group(1) if ticker_m else ""

            combined = "\n\n".join(m.get("content", "") for m in smart)

            generate_cio_pdf(combined, out_path, ticker=ticker, title=title, lang=_lang)
            result = {"success": True, "filename": filename}

        elif fmt in ("docx", "word"):
            from core.cio_docx import generate_cio_docx
            import time, re as re2

            filename = "EisaX_" + time.strftime("%Y%m%d_%H%M%S") + ".docx"
            out_path = str(EXPORTS_DIR / filename)

            ticker_m = re2.search(r"EisaX (?:Report|Intelligence Report)[:\s-]+([A-Z]{1,5})", title or "")
            ticker = ticker_m.group(1) if ticker_m else ""

            combined = "\n\n".join(m.get("content", "") for m in smart)

            generate_cio_docx(combined, out_path, ticker=ticker, title=title)
            result = {"success": True, "filename": filename}
        else:
            result = export_engine(fmt, smart, title)
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error","Export failed"))
        filename = os.path.basename(result.get("filename",""))
        src = result.get("filename","")
        dst = os.path.join(export_dir, filename)
        if src and os.path.exists(src) and src != dst:
            shutil.copy2(src, dst)
        return {"success": True, "filename": filename, "download_url": f"/v1/download/{filename}", "title": title, "format": fmt}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/download/{filename}")
async def download_file(filename: str):
    """Download exported file — public endpoint so browser links work directly."""
    import re as _re
    from fastapi.responses import FileResponse

    # Only allow safe filenames: letters, digits, underscores, hyphens, dots
    if not _re.fullmatch(r"[\w\-]+\.(pdf|docx|xlsx|pptx|csv)", filename, _re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Invalid filename")

    export_dir = "/home/ubuntu/investwise/static/exports"
    file_path = os.path.join(export_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=filename)

@app.get("/v1/brain/status")
async def brain_status(access_token: str = Header(None, alias="X-API-Key")):
    if access_token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    from learning_engine import get_engine
    return get_engine().status()

@app.get("/v1/brain/wisdom")
async def brain_wisdom(access_token: str = Header(None, alias="X-API-Key")):
    if access_token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    from learning_engine import get_engine
    engine = get_engine()
    conn = engine._get_conn()
    stocks = conn.execute("SELECT COUNT(*) FROM stock_knowledge").fetchone()[0]
    preds = conn.execute(
        "SELECT COUNT(*), ROUND(AVG(was_correct)*100,1) FROM predictions WHERE evaluated=1"
    ).fetchone()
    lessons = conn.execute(
        "SELECT lesson, category, confidence, date FROM learning_log ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    conn.close()
    return {
        "stocks_known": stocks,
        "predictions_evaluated": preds[0],
        "overall_accuracy_pct": preds[1],
        "lessons": [dict(r) for r in lessons],
        "engine_stats": engine._stats
    }
if __name__ == "__main__":
    uvicorn.run("api_bridge_v2:app", host="0.0.0.0", port=8000, workers=2)

# ── HTML → PDF Export ──
class HtmlExportPayload(BaseModel):
    html: str
    filename: str = ""
    access_token: str = ""

@app.post("/v1/export/html")
async def export_html_to_pdf(
    payload: HtmlExportPayload,
    access_token: str = Header(None, alias="X-API-Key")
):
    token = access_token or payload.access_token
    if token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        import time
        from core.playwright_pdf import html_to_pdf, inject_print_css
        fname = payload.filename or f"EisaX_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
        if not fname.endswith('.pdf'):
            fname += '.pdf'
        filepath = str(EXPORTS_DIR / fname)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        html_to_pdf(inject_print_css(payload.html), filepath)
        os.chmod(filepath, 0o644)
        return {"url": f"/v1/download/{fname}", "download_url": f"/v1/download/{fname}", "filename": fname}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/dashboard/{ticker}")
async def dashboard(ticker: str, access_token: str = Header(None, alias="X-API-Key")):
    """Return all dashboard data for a ticker in one call — no LLM, runs concurrently."""
    if access_token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    import asyncio, math
    from core.market_data import get_realtime_quote, get_full_stock_profile
    from core.data import get_prices
    from core.analytics import generate_technical_summary, run_stress_test
    from core.realtime_data import deepcrawl_stock, deepcrawl_news
    from core.rapid_data import get_market_pulse, get_cashflow, get_events_calendar

    ticker = ticker.upper().strip()
    loop = asyncio.get_event_loop()

    # ── Detect Saudi Tadawul ticker ──────────────────────────────────────────
    is_saudi  = ticker.endswith(".SR")
    tadawul_id = ticker.replace(".SR", "") if is_saudi else None

    # ── fetch ALL sources in parallel ──────────────────────────────────────────
    # Group 1: per-ticker data (quote, profile, prices, DeepCrawl, cash flow, events)
    # Group 2: global market data (Fear&Greed, Forex calendar, CNBC news)
    # Group 3 (Saudi only): Tadawul live quote + history
    from core.rapid_data import get_tadawul_quote, get_tadawul_history, _fetch_tadawul_candles
    try:
        if is_saudi:
            # For Saudi tickers: fetch Tadawul candles FIRST (shared cache for quote+history)
            # then derive quote and history from same candles without 2 separate HTTP calls
            (quote, profile, prices_df, dc_data, dc_news,
             cashflow_data, events_data, market_pulse,
             _raw_candles) = await asyncio.gather(
                loop.run_in_executor(None, get_realtime_quote, ticker),
                loop.run_in_executor(None, get_full_stock_profile, ticker),
                loop.run_in_executor(None, get_prices, [ticker]),
                loop.run_in_executor(None, deepcrawl_stock, ticker),
                loop.run_in_executor(None, deepcrawl_news, ticker, 5),
                loop.run_in_executor(None, get_cashflow, ticker),
                loop.run_in_executor(None, get_events_calendar, ticker),
                loop.run_in_executor(None, get_market_pulse),
                loop.run_in_executor(None, _fetch_tadawul_candles, tadawul_id),
            )
            # Build quote + history from same candles (no extra HTTP call)
            tadawul_quote = get_tadawul_quote(tadawul_id)     # reads from shared cache (instant)
            tadawul_hist  = list(reversed(_raw_candles)) if _raw_candles else get_tadawul_history(tadawul_id)
        else:
            (quote, profile, prices_df, dc_data, dc_news,
             cashflow_data, events_data, market_pulse) = await asyncio.gather(
                loop.run_in_executor(None, get_realtime_quote, ticker),
                loop.run_in_executor(None, get_full_stock_profile, ticker),
                loop.run_in_executor(None, get_prices, [ticker]),
                loop.run_in_executor(None, deepcrawl_stock, ticker),
                loop.run_in_executor(None, deepcrawl_news, ticker, 5),
                loop.run_in_executor(None, get_cashflow, ticker),
                loop.run_in_executor(None, get_events_calendar, ticker),
                loop.run_in_executor(None, get_market_pulse),
            )
            tadawul_quote = {}
            tadawul_hist  = []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {e}")

    # ── Override quote with Tadawul live data (more accurate for .SR tickers) ──
    if is_saudi and tadawul_quote.get("price"):
        tq = tadawul_quote
        quote["price"]      = tq.get("price",      quote.get("price"))
        quote["open"]       = tq.get("open",        quote.get("open"))
        quote["high"]       = tq.get("high",        quote.get("high"))
        quote["low"]        = tq.get("low",         quote.get("low"))
        quote["volume"]     = tq.get("volume",      quote.get("volume"))
        quote["change"]     = tq.get("change",      quote.get("change"))
        quote["change_pct"] = tq.get("change_pct",  quote.get("change_pct"))
        quote["source"]     = "Tadawul RapidAPI (live)"

    # ── technicals + stress (instant, local) ──
    try:
        close_series = prices_df[ticker] if ticker in prices_df.columns else prices_df.iloc[:, 0]
        tech   = generate_technical_summary(ticker, close_series)
        beta   = float((profile.get("fundamentals") or {}).get("beta") or 1.0)
        stress = run_stress_test(close_series, beta=beta)
    except Exception as e:
        tech   = {}
        stress = {"scenarios": {}, "annual_vol": 0}

    # ── sanitise NaN/Inf so JSON serialises cleanly ──
    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    def _clean_dict(d):
        return {k: _clean(v) for k, v in (d or {}).items()}

    def _safe_float(v):
        try:
            return float(v) if v not in (None, "", "-") else None
        except (TypeError, ValueError):
            return None

    # ── merge DeepCrawl technicals into tech dict (RSI, SMA, performance) ──
    dc = dc_data or {}
    dc_technicals = {
        "rsi":          _safe_float(dc.get("rsi")),
        "sma50":        _safe_float(dc.get("sma50")),
        "sma200":       _safe_float(dc.get("sma200")),
        "short_float":  _safe_float(dc.get("short_float")),
        "avg_volume":   dc.get("avg_volume"),
        "perf_week":    _safe_float(dc.get("perf_week")),
        "perf_month":   _safe_float(dc.get("perf_month")),
        "perf_ytd":     _safe_float(dc.get("perf_ytd")),
    }
    # Merge into local tech dict — DeepCrawl fills gaps
    for k, v in dc_technicals.items():
        if v is not None:
            tech[k] = v

    # ── DeepCrawl fundamentals enrichment ──
    dc_fundamentals = {
        # Analyst consensus
        "analyst_rating":      dc.get("analyst_rating"),
        "analyst_buy":         dc.get("analyst_buy"),
        "analyst_hold":        dc.get("analyst_hold"),
        "analyst_sell":        dc.get("analyst_sell"),
        # Price targets (from forecast page)
        "price_target":        dc.get("price_target"),
        "price_target_mean":   dc.get("price_target_mean"),
        "price_target_low":    dc.get("price_target_low"),
        "price_target_high":   dc.get("price_target_high"),
        "price_target_median": dc.get("price_target_median"),
        # Valuation
        "forward_pe":          _safe_float(dc.get("forward_pe")),
        "earnings_date":       dc.get("earnings_date"),
        "week_52_range":       dc.get("week_52_range"),
        # Ownership
        "inst_own":            _safe_float(dc.get("inst_own")),
        "insider_own":         _safe_float(dc.get("insider_own")),
        # Financial ratios (from SA ratios page fallback)
        "debt_equity":         _safe_float(dc.get("debt_equity")),
        "roe":                 _safe_float(dc.get("roe")),
        "roa":                 _safe_float(dc.get("roa")),
        "profit_margin":       _safe_float(dc.get("profit_margin")),
        "gross_margin":        dc.get("gross_margin"),
        "net_margin":          dc.get("net_margin_annual"),
        "free_cash_flow":      dc.get("free_cash_flow"),
    }

    # ── DeepCrawl historical financials (revenue + EPS by year) ──
    dc_financials = {
        "revenue_history": dc.get("revenue_history") or {},
        "eps_history":     dc.get("eps_history")     or {},
    }

    # ── Merge existing fundamentals with DeepCrawl (DeepCrawl fills gaps only) ──
    base_fundamentals = _clean_dict(profile.get("fundamentals", {}))
    for k, v in dc_fundamentals.items():
        if v is not None and k not in base_fundamentals:
            base_fundamentals[k] = v

    # ── Enrich fundamentals with Events Calendar data ──────────────────────────
    ev = events_data or {}
    events_fields = {
        "earnings_date":  ev.get("earnings_date"),
        "ex_div_date":    ev.get("ex_div_date"),
        "div_date":       ev.get("div_date"),
        "eps_est_avg":    ev.get("eps_est_avg"),
        "eps_est_high":   ev.get("eps_est_high"),
        "eps_est_low":    ev.get("eps_est_low"),
        "rev_est_avg":    ev.get("rev_est_avg"),
    }
    for k, v in events_fields.items():
        if v is not None and not base_fundamentals.get(k):
            base_fundamentals[k] = v

    # ── Combine news: DeepCrawl stock news + CNBC global news ─────────────────
    mp = market_pulse or {}
    cnbc_news = _get_aggregated_news(ticker=ticker, limit=5)
    combined_news = (dc_news or []) + cnbc_news

    # ── Build final financials with cash flow ──────────────────────────────────
    cf = cashflow_data or {}
    dc_financials["cash_flow"] = {
        "quarters":     cf.get("quarters", []),
        "operating_cf": cf.get("operating_cf", []),
        "free_cf":      cf.get("free_cf", []),
        "capex":        cf.get("capex", []),
        "unit":         cf.get("unit", "B USD"),
        "source":       cf.get("source", ""),
    } if cf else {}

    return {
        "ticker":       ticker,
        "quote":        _clean_dict(quote),
        "fundamentals": base_fundamentals,
        "technicals":   _clean_dict(tech),
        "financials":   dc_financials,
        "stress":       {k: _clean_dict(v) for k, v in stress.get("scenarios", {}).items()},
        "annual_vol":   stress.get("annual_vol", 0),
        "news":         combined_news,
        # ── Market-wide data ───────────────────────────────────────────────────
        "fear_greed":    mp.get("fear_greed") or {},
        "econ_calendar": mp.get("calendar")   or [],
        "dc_source":     dc.get("source", ""),
        # ── Saudi Tadawul (only populated for .SR tickers) ────────────────────
        "is_saudi":        is_saudi,
        "tadawul_intraday": tadawul_hist,   # list of {date,open,high,low,close,volume} 1-min candles
    }


class TranslatePayload(BaseModel):
    text: str
    access_token: str = ""

@app.post("/v1/translate-ar")
async def translate_to_arabic(payload: TranslatePayload, access_token: str = Header(None, alias="X-API-Key")):
    """Translate an English investment report to Arabic. Primary: DeepSeek. Fallback: GLM."""
    token = access_token or payload.access_token
    if token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")

    system_prompt = (
        "أنت محلل مالي محترف. مهمتك ترجمة تقرير استثماري كامل من الإنجليزية إلى العربية الفصحى.\n"
        "القواعد الصارمة:\n"
        "1. ترجم كل النص كاملاً بدون حذف أي قسم أو معلومة\n"
        "2. احتفظ بتنسيق Markdown كما هو: ##، ###، **bold**، | tables |، - lists، > blockquote\n"
        "3. لا تترجم: أسماء الشركات، رموز البورصة (AAPL، BTC)، الأرقام، العملات، النسب المئوية\n"
        "4. الجداول (tables): حافظ على | الفاصل | وترجم محتوى الخلايا فقط\n"
        "5. اكتب بأسلوب مؤسسي احترافي مناسب لتقارير المحللين الماليين\n"
        "6. أخرج النص المترجم فقط — بدون أي تعليق أو مقدمة"
    )
    # Chunks arrive pre-split from client (max 6000 chars each) — accept up to 8000 chars
    text_in = payload.text[:8000]
    user_msg = f"ترجم هذا النص كاملاً مع الحفاظ على تنسيق Markdown:\n\n{text_in}"

    import httpx, os

    # ── Primary: DeepSeek ────────────────────────────────────────────────────
    ds_key = os.getenv("DEEPSEEK_API_KEY", "")
    if ds_key:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                ds_resp = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_msg}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 4000
                    }
                )
            if ds_resp.status_code == 200:
                ar_text = ds_resp.json()["choices"][0]["message"]["content"]
                logger.info("translate-ar: DeepSeek OK (%d chars)", len(ar_text))
                return {"success": True, "text": ar_text}
            else:
                logger.warning("translate-ar DeepSeek failed %s: %s", ds_resp.status_code, ds_resp.text[:150])
        except Exception as _de:
            logger.warning("translate-ar DeepSeek error: %s", _de)

    # ── Fallback: GLM ────────────────────────────────────────────────────────
    try:
        from core.glm_client import GLMClient, GLM_API_URL, GLM_MODEL
        glm = GLMClient()
        async with httpx.AsyncClient(timeout=110) as client:
            glm_resp = await client.post(
                GLM_API_URL,
                headers=glm.headers,
                json={
                    "model": GLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_msg}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 8000
                }
            )
        if glm_resp.status_code == 200:
            ar_text = glm_resp.json()["choices"][0]["message"]["content"]
            logger.info("translate-ar: GLM fallback OK (%d chars)", len(ar_text))
            return {"success": True, "text": ar_text}
        else:
            logger.warning("translate-ar GLM failed: %s", glm_resp.text[:200])
    except Exception as _ge:
        logger.error("translate-ar GLM error: %s", _ge)

    return {"success": False, "text": payload.text, "error": "All translation services unavailable"}


@app.post("/v1/export/html-pdf")
async def export_html_pdf(payload: HtmlExportPayload, access_token: str = Header(None, alias="X-API-Key")):
    token = access_token or payload.access_token
    if token != SECURE_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        import time
        from core.playwright_pdf import html_to_pdf, inject_print_css
        fname = payload.filename or f"EisaX_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
        if not fname.endswith('.pdf'):
            fname += '.pdf'
        filepath = str(EXPORTS_DIR / fname)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        html_to_pdf(inject_print_css(payload.html), filepath)
        os.chmod(filepath, 0o644)
        return {"url": f"/v1/download/{fname}", "download_url": f"/v1/download/{fname}", "filename": fname}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  B2B AUTH  — /auth/*  and  /admin/*
# ══════════════════════════════════════════════════════════════════════════════
from core.auth    import hash_password, verify_password, create_token, decode_token, generate_temp_password
from core.user_db import (init_users_table, create_user, get_user_by_email,
                           get_user_by_id, list_users, update_user, delete_user, record_login)

# Initialise users table on startup (idempotent)
init_users_table()

_bearer = HTTPBearer(auto_error=False)


def _require_jwt(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    """FastAPI dependency — validates Bearer JWT and returns payload."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_token(credentials.credentials)
    except _jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except _jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload


def _require_admin(payload: dict = Depends(_require_jwt)) -> dict:
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
    return payload


# ── Pydantic models ───────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email:    str
    password: str

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

class CreateUserRequest(BaseModel):
    email:    str
    name:     str
    role:     str = "user"   # "user" | "admin"

class UpdateUserRequest(BaseModel):
    name:      Optional[str] = None
    role:      Optional[str] = None
    is_active: Optional[int] = None


# ── Auth endpoints ─────────────────────────────────────────────────────────────
@app.post("/auth/login")
async def auth_login(body: LoginRequest):
    user = get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account disabled")
    record_login(user["id"])
    token = create_token(
        user["id"], user["email"], user["role"],
        must_change=bool(user["must_change_pw"])
    )
    return {
        "token":       token,
        "must_change": bool(user["must_change_pw"]),
        "name":        user["name"],
        "role":        user["role"],
    }


@app.post("/auth/change-password")
async def auth_change_password(body: ChangePasswordRequest, payload: dict = Depends(_require_jwt)):
    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(body.old_password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Wrong current password")
    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")
    update_user(user["id"], password_hash=hash_password(body.new_password), must_change_pw=0)
    token = create_token(user["id"], user["email"], user["role"], must_change=False)
    return {"token": token, "message": "Password changed"}


@app.get("/auth/me")
async def auth_me(payload: dict = Depends(_require_jwt)):
    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id":    user["id"],
        "email": user["email"],
        "name":  user["name"],
        "role":  user["role"],
    }


# ── Admin endpoints ────────────────────────────────────────────────────────────
@app.post("/admin/users")
async def admin_create_user(body: CreateUserRequest, _: dict = Depends(_require_admin)):
    if get_user_by_email(body.email):
        raise HTTPException(status_code=409, detail="Email already exists")
    temp_pw = generate_temp_password()
    uid = create_user(
        email=body.email,
        name=body.name,
        password_hash=hash_password(temp_pw),
        role=body.role,
        must_change_pw=True,
    )
    return {"id": uid, "email": body.email, "name": body.name, "temp_password": temp_pw}


@app.get("/admin/users")
async def admin_list_users(_: dict = Depends(_require_admin)):
    return list_users()


@app.patch("/admin/users/{user_id}")
async def admin_update_user(user_id: int, body: UpdateUserRequest, _: dict = Depends(_require_admin)):
    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    if not update_user(user_id, **changes):
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


@app.delete("/admin/users/{user_id}")
async def admin_delete_user(user_id: int, _: dict = Depends(_require_admin)):
    if not delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


@app.post("/admin/users/{user_id}/reset-password")
async def admin_reset_password(user_id: int, _: dict = Depends(_require_admin)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    temp_pw = generate_temp_password()
    update_user(user_id, password_hash=hash_password(temp_pw), must_change_pw=1)
    return {"temp_password": temp_pw}
