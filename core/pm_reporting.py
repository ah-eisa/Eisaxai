from __future__ import annotations
import logging
import json
import re
import time
from typing import Any, List, Optional, Dict
import pandas as pd
import config
from core.llm import get_client
from core.data import get_prices, to_returns
from core.policy import apply_policy
from core.metrics import perf_metrics
from state import SYSTEM_PROMPTS
logger = logging.getLogger(__name__)

from core.pm_helpers import _kv, get_param, parse_float, parse_int, _fmt_pct, _fmt_float, render_weights, detect_risk_pref, recommend_etfs, method_from_risk
from core.pm_tickers import _normalize_tickers, has_placeholder_tickers, get_ticker_name, _tv_to_yfinance, get_top_regional_tickers, smart_expand_tickers, _RISK_LABELS, _RISK_EMOJIS

def compute_risk_score(perf: dict) -> dict:
    """
    Returns a 1–10 risk score based on portfolio volatility and max drawdown.
    Also computes parametric VaR 95% over a 1-month horizon.

    Score bands:
      1-2  Very Conservative   🔵
      3-4  Conservative        🟢
      5-6  Moderate / Balanced 🟡
      7-8  Growth              🟠
      9-10 Aggressive          🔴
    """
    import math
    vol = abs(perf.get("volatility", 0.15))
    dd  = abs(perf.get("max_drawdown", vol * 3))   # estimate if not yet available

    # Volatility component (1-10)
    if   vol < 0.08: vol_s = 1
    elif vol < 0.12: vol_s = 2
    elif vol < 0.15: vol_s = 3
    elif vol < 0.18: vol_s = 4
    elif vol < 0.22: vol_s = 5
    elif vol < 0.26: vol_s = 6
    elif vol < 0.30: vol_s = 7
    elif vol < 0.35: vol_s = 8
    elif vol < 0.40: vol_s = 9
    else:             vol_s = 10

    # Drawdown component (1-10)
    if   dd < 0.10: dd_s = 1
    elif dd < 0.15: dd_s = 2
    elif dd < 0.20: dd_s = 3
    elif dd < 0.25: dd_s = 4
    elif dd < 0.30: dd_s = 5
    elif dd < 0.40: dd_s = 6
    elif dd < 0.50: dd_s = 7
    elif dd < 0.60: dd_s = 8
    elif dd < 0.70: dd_s = 9
    else:            dd_s = 10

    score = int(round(max(1, min(10, vol_s * 0.5 + dd_s * 0.5))))

    # Parametric VaR 95% — 1-month horizon
    # VaR = Z(95%) × σ_annual / √12
    var_monthly = 1.645 * vol / math.sqrt(12)

    return {
        "score":        score,
        "label":        _RISK_LABELS[score],
        "emoji":        _RISK_EMOJIS[score],
        "var_monthly":  var_monthly,
        "volatility":   vol,
        "max_drawdown": dd,
    }

# ============================================================
# OUTPUT RENDERING
# ============================================================
def render_optimize_reply(w_raw: dict[str, float], perf: dict | None) -> str:
    lines = ["=== Optimized Weights ===", "", render_weights(w_raw)]
    if perf:
        rs = compute_risk_score(perf)
        lines += [
            "",
            "",
            "─────────────────────────────",
            "📊 Optimizer (ex-ante):",
            "",
            f"  Expected Return  : {_fmt_pct(perf['expected_return'])}",
            f"  Volatility       : {_fmt_pct(perf['volatility'])}",
            f"  Sharpe Ratio     : {_fmt_float(perf['sharpe'])}",
            "",
            "🛡️ Risk Profile:",
            "",
            f"  Risk Score       : {rs['score']}/10  {rs['emoji']} {rs['label']}",
            f"  VaR 95% (1-mo)   : -{_fmt_pct(rs['var_monthly'])}",
        ]
        if perf.get("max_drawdown") is not None:
            lines.append(f"  Max Drawdown     : {_fmt_pct(perf['max_drawdown'])}")
        dd_ok = perf.get("drawdown_satisfied")
        if dd_ok is not None:
            flag = "✅ Satisfied" if dd_ok else "⚠️  NOT satisfied"
            limit = perf.get("max_drawdown_constraint", 0)
            lines.append(f"  Drawdown Limit   : {flag} (limit: {_fmt_pct(-abs(limit))})")
        if perf.get("hedge_assets_added"):
            lines.append(f"  Defensive Assets : {', '.join(perf['hedge_assets_added'])} (auto-added)")
        lines.append("─────────────────────────────")
    return "\n".join(lines)

def render_report(weights: dict[str, float], perf: dict, metrics: dict) -> str:
    lines: list[str] = []
    lines.append("=== Portfolio Report ===")
    lines.append("")
    lines.append("Weights:")
    lines.append(render_weights(weights))
    lines.append("")
    lines.append("Optimizer (ex-ante):")
    lines.append(f"- Expected Return: {_fmt_pct(perf['expected_return'])}")
    lines.append(f"- Volatility: {_fmt_pct(perf['volatility'])}")
    lines.append(f"- Sharpe: {_fmt_float(perf['sharpe'])}")
    lines.append("")
    lines.append("Backtest Metrics (ex-post):")
    lines.append(f"- CAGR: {_fmt_pct(metrics['cagr'])}")
    lines.append(f"- Vol: {_fmt_pct(metrics['vol'])}")
    lines.append(f"- Sharpe: {_fmt_float(metrics['sharpe'])}")
    if "sortino" in metrics:
        lines.append(f"- Sortino: {_fmt_float(metrics.get('sortino', 0.0))}")
    if "max_drawdown" in metrics:
        lines.append(f"- Max Drawdown: {_fmt_pct(metrics.get('max_drawdown', 0.0))}")
    if "calmar" in metrics:
        lines.append(f"- Calmar: {_fmt_float(metrics.get('calmar', 0.0))}")
    return "\n".join(lines)

def build_portfolio_report_body(mem: dict[str, Any]) -> str:
    lines = []

    # ── Guard: reject portfolios with negative expected return ────────────────
    performance_pre = mem.get("performance", {})
    exp_ret_pre = performance_pre.get("expected_return", None)
    rf_rate_pre = 0.045  # 4.5% risk-free rate
    if exp_ret_pre is not None and float(exp_ret_pre) < rf_rate_pre:
        return (
            f"⚠️ **المحفظة المقترحة غير مقبولة**\n\n"
            f"العائد المتوقع **{exp_ret_pre*100:.1f}%** أقل من معدل الخطر الصفري ({rf_rate_pre*100:.1f}%). "
            f"ده معناه إنك بتاخد مخاطرة بدون عائد مناسب.\n\n"
            f"**الحل:** أضف أسواق US أو Gold أو Bonds للمحفظة، أو اختار profile مختلف."
        )

    # --- Executive Summary (MAX 5 bullets) ---
    lines.append("# Executive Summary\n")
    tickers = mem.get("tickers", [])
    weights = mem.get("weights", {}) or mem.get("weights_raw", {})
    performance = mem.get("performance", {})
    metrics = mem.get("metrics", {})
    method = mem.get("method", "max_sharpe")
    
    # Dominant asset
    if weights:
        sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_asset, top_weight = sorted_w[0] if sorted_w else ("N/A", 0)
    else:
        top_asset, top_weight = "N/A", 0
    
    # Risk level inference
    vol = performance.get("volatility", 0) or metrics.get("vol", 0)
    sharpe = performance.get("sharpe", 0) or metrics.get("sharpe", 0)
    rs = compute_risk_score(performance)

    lines.append(f"- **Portfolio Size:** {len(tickers)} assets")
    lines.append(f"- **Dominant Holding:** {top_asset} ({top_weight*100:.1f}%)")
    lines.append(f"- **Risk Score:** {rs['score']}/10 {rs['emoji']} {rs['label']}")
    lines.append(f"- **Risk-Adjusted Return:** Sharpe Ratio {sharpe:.2f}")
    lines.append(f"- **Optimization Method:** {method.replace('_', ' ').title()}")
    
    # --- Portfolio Overview ---
    lines.append("\n\n# Portfolio Overview\n")
    lines.append(f"**Assets:** {', '.join(tickers) if tickers else 'No assets defined'}")
    lines.append(f"**Analysis Period:** {mem.get('start', 'N/A')} to {mem.get('end', 'Present')}")
    lines.append(f"**Rebalancing Strategy:** Based on {method.replace('_', ' ').title()} optimization")
    
    # --- Allocation Table ---
    if weights:
        lines.append("\n### Allocation Weights\n")
        lines.append("| Asset | Weight |")
        lines.append("|-------|--------|")
        for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {ticker} | {weight*100:.1f}% |")
    
    # --- Key Metrics ---
    lines.append("\n\n# Key Metrics\n")
    lines.append("### Expected Performance (Ex-Ante)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Expected Return | {performance.get('expected_return', 0)*100:.2f}% |")
    lines.append(f"| Expected Volatility | {performance.get('volatility', 0)*100:.2f}% |")
    lines.append(f"| Sharpe Ratio | {performance.get('sharpe', 0):.2f} |")
    
    if metrics:
        lines.append("\n### Backtest Results (Ex-Post)\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        if "cagr" in metrics:
            lines.append(f"| CAGR | {metrics['cagr']*100:.2f}% |")
        if "vol" in metrics:
            lines.append(f"| Realized Volatility | {metrics['vol']*100:.2f}% |")
        if "sharpe" in metrics:
            lines.append(f"| Historical Sharpe | {metrics['sharpe']:.2f} |")
        if "sortino" in metrics:
            lines.append(f"| Sortino Ratio | {metrics['sortino']:.2f} |")
        if "max_drawdown" in metrics:
            lines.append(f"| Max Drawdown | {metrics['max_drawdown']*100:.2f}% |")
        if "calmar" in metrics:
            lines.append(f"| Calmar Ratio | {metrics['calmar']:.2f} |")
    
    # --- Risk Analysis ---
    lines.append("\n\n# Risk Analysis\n")

    # Risk Score Card
    lines.append("### Risk Score\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Overall Risk Score | **{rs['score']}/10** {rs['emoji']} {rs['label']} |")
    lines.append(f"| Annualised Volatility | {rs['volatility']*100:.1f}% |")
    lines.append(f"| VaR 95% (1-month) | -{rs['var_monthly']*100:.1f}% |")
    dd_src = performance.get("max_drawdown") or (metrics.get("max_drawdown") if metrics else None)
    if dd_src is not None:
        lines.append(f"| Max Drawdown | {abs(dd_src)*100:.1f}% |")
    dd_ok = performance.get("drawdown_satisfied")
    if dd_ok is not None:
        flag = "✅ Satisfied" if dd_ok else "⚠️ NOT satisfied"
        limit = performance.get("max_drawdown_constraint", 0)
        lines.append(f"| Drawdown Constraint | {flag} (limit {abs(limit)*100:.0f}%) |")
    if performance.get("hedge_assets_added"):
        lines.append(f"| Defensive Assets Added | {', '.join(performance['hedge_assets_added'])} |")

    # Concentration Risk
    lines.append("\n### Concentration Risk")
    if weights:
        top_3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_total = sum(w for _, w in top_3)
        lines.append(f"- Top 3 holdings represent **{top_3_total*100:.1f}%** of portfolio")
        if top_3_total > 0.6:
            lines.append("- ⚠️ High concentration risk — consider wider diversification")
        else:
            lines.append("- ✅ Well-diversified — no dangerous concentration")

    # Volatility Assessment
    lines.append("\n### Volatility Assessment")
    if vol > 0.25:
        lines.append("- Portfolio exhibits **high volatility** — suitable for aggressive investors")
    elif vol > 0.15:
        lines.append("- Portfolio exhibits **moderate volatility** — balanced risk profile")
    else:
        lines.append("- Portfolio exhibits **low volatility** — suitable for conservative investors")
    lines.append(f"- **VaR interpretation:** In a typical month, losses are expected to stay within **{rs['var_monthly']*100:.1f}%** at 95% confidence")

    if metrics.get("max_drawdown"):
        mdd = abs(metrics["max_drawdown"])
        lines.append(f"- Historical maximum drawdown: **{mdd*100:.1f}%**")
        if mdd > 0.20:
            lines.append("- ⚠️ Significant drawdown risk — ensure adequate risk tolerance")
    
    # --- Assumptions & Limitations ---
    lines.append("\n\n# Assumptions & Data Limitations\n")
    lines.append("- Historical data sourced from Yahoo Finance")
    lines.append("- Past performance does not guarantee future results")
    lines.append("- Optimization assumes normally distributed returns")
    lines.append(f"- Risk-free rate assumed: {mem.get('rf', 0)*100:.2f}%")
    lines.append(f"- Weight constraints: Min {mem.get('min_w', 0)*100:.1f}%, Max {mem.get('max_w', 0.20)*100:.1f}%")
    
    # --- Actionable Next Steps ---
    lines.append("\n\n# Actionable Next Steps\n")
    lines.append("1. **Rebalance quarterly** to maintain target allocations")
    if metrics.get("max_drawdown") and abs(metrics["max_drawdown"]) > 0.15:
        lines.append(f"2. **Set stop-loss** at {abs(metrics['max_drawdown'])*100:.0f}% drawdown trigger")
    else:
        lines.append("2. **Monitor drawdown** - rebalance if exceeds 15%")
    if top_weight > 0.30:
        lines.append(f"3. **A concentration above 30% in** {top_asset} **increases single-asset risk**")
    else:
        lines.append("3. **Maintain diversification** - no single asset exceeds 30%")
    
    # --- Disclaimer ---
    lines.append("\n\n# Disclaimer\n")
    lines.append("*This analysis is for informational purposes only and does not constitute investment advice. ")
    lines.append("Past performance does not guarantee future results. ")
    lines.append("Please consult with a qualified financial advisor before making investment decisions.*")
    
    return "\n".join(lines)

def generate_executive_report_llm(
    *,
    model: str,
    temperature: float,
    mem: dict[str, Any],
    base_report_md: str,
) -> str:
    """
    Uses LLM to produce an executive, client-ready markdown report.
    """
    # ── Hard Rejection Gate ────────────────────────────────────────────────────
    _performance = mem.get("performance") or {}
    _exp_ret = float(_performance.get("expected_return", 0) or 0)
    _sharpe  = float(_performance.get("sharpe", 0) or 0)
    _rf      = 0.045
    if _performance and (_exp_ret < _rf or _sharpe < 0):
        _weights = mem.get("weights") or mem.get("weights_raw") or {}
        _top3    = sorted(_weights.items(), key=lambda x: -x[1])[:3] if _weights else []
        _conc    = sum(w for _, w in _top3)
        _fixes   = []
        if _exp_ret < _rf:
            _fixes.append("أضف أسهم US أو Gold أو Bonds لرفع العائد المتوقع فوق معدل الخطر الصفري (4.5%)")
        if _sharpe < 0:
            _fixes.append("العائد المتوقع أقل من معدل الخطر الصفري — المحفظة لا تُعوّض المستثمر عن المخاطرة")
        if _conc > 0.55:
            _top3_str = ", ".join(f"{a} ({w*100:.0f}%)" for a, w in _top3)
            _fixes.append(f"الأصول العليا ({_top3_str}) تمثل {_conc*100:.0f}% — حدّ كل أصل بـ 15-20%")
        if len(_weights) < 4:
            _fixes.append("المحفظة تحتوي على أقل من 4 أصول — أضف أسواقاً إضافية لتحقيق التنويع")
        _fixes_md = "\n".join(f"- {f}" for f in _fixes) if _fixes else "- راجع مكونات المحفظة وأضف أصولاً ذات عائد إيجابي"
        return (
            "# ❌ Portfolio Rejected — Strategy Invalid\n\n"
            "**لا يمكن تنفيذ هذه المحفظة — المعايير الأساسية غير مستوفاة.**\n\n"
            "| المؤشر | القيمة | المطلوب |\n"
            "|--------|--------|---------|\n"
            f"| العائد المتوقع | **{_exp_ret*100:.2f}%** | > 4.5% |\n"
            f"| Sharpe Ratio | **{_sharpe:.2f}** | > 0 |\n\n"
            "## السبب\n\n"
            "المحفظة المقترحة **تخسر قيمتها** أو لا تُعوّض عن مخاطرها. "
            "تنفيذها سيضر بالمستثمر بدلاً من مساعدته.\n\n"
            "## الإصلاحات المقترحة\n\n"
            f"{_fixes_md}\n\n"
            "## جرّب بدلاً من ذلك\n\n"
            "> ابني محفظة **balanced** باستخدام **US + GCC + Gold** لضمان عائد إيجابي ومتوازن.\n"
        )
    # ── End Rejection Gate ─────────────────────────────────────────────────────

    # ── Placeholder Ticker Gate ────────────────────────────────────────────────
    _weights_exec = mem.get("weights") or mem.get("weights_raw") or {}
    _placeholders_exec = has_placeholder_tickers(_weights_exec)
    if _placeholders_exec:
        return (
            "# ⛔ Report Blocked — Unverified Assets Detected\n\n"
            f"**Placeholder tickers detected:** `{'`, `'.join(_placeholders_exec)}`\n\n"
            "EisaX does not produce client-facing reports containing unidentified securities.\n\n"
            "**Fix:** Ask EisaX to **rebuild the portfolio** — "
            "the optimizer will select verified assets from the live market library.\n\n"
            "> **Rule:** Every asset in a report must be verified by ticker, name, and market.\n"
        )
    # ── End Placeholder Gate ───────────────────────────────────────────────────

    try:
        client = get_client()

        # Prepare structured data payload for LLM
        # Inject live market data
        live_data = {}
        try:
            from core.market_data import get_macro_context
            from core.realtime_data import get_crypto_price
            from datetime import datetime as _dt
            macro = get_macro_context()
            btc = get_crypto_price("bitcoin")
            eth = get_crypto_price("ethereum")
            live_data = {
                "report_date": _dt.now().strftime("%B %d, %Y"),
                "macro": {
                    "fed_funds": macro.get("fed_funds", {}).get("value", "N/A"),
                    "treasury_10y": macro.get("treasury_10y", {}).get("value", "N/A"),
                    "inflation_yoy": macro.get("inflation", {}).get("value", "N/A"),
                    "unemployment": macro.get("unemployment", {}).get("value", "N/A"),
                    "gdp_growth": macro.get("gdp_growth", {}).get("value", "N/A"),
                },
                "crypto": {
                    "btc_price": btc.get("price", "N/A"),
                    "btc_change_24h": btc.get("change_24h", "N/A"),
                    "eth_price": eth.get("price", "N/A"),
                    "eth_change_24h": eth.get("change_24h", "N/A"),
                }
            }
            logger.debug(f"[Portfolio] Live data injected: BTC=${btc.get('price',0):,.0f}, Fed={macro.get('fed_funds',{}).get('value','N/A')}%")
        except Exception as e:
            logger.error(f"[Portfolio] Live data failed: {e}")
        payload = {
            "tickers": mem.get("tickers", []),
            "start": mem.get("start"),
            "end": mem.get("end"),
            "method": mem.get("method"),
            "weights": mem.get("weights") or mem.get("weights_raw"),
            "performance": mem.get("performance"),
            "metrics": mem.get("metrics"),
            "rf": mem.get("rf"),
            "min_w": mem.get("min_w"),
            "max_w": mem.get("max_w"),
            "min_assets": mem.get("min_assets"),
            "live_market_data": live_data,
        }

        response = client.create_completion(
            model=model,
            temperature=temperature,
            max_tokens=6000,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPTS.get("investment_report", ""),
                },
                {
                    "role": "user",
                    "content": (
                        f"Today's date: {live_data.get('report_date', 'February 2026')}\n\n"
                        f"LIVE MARKET DATA (USE THESE EXACT NUMBERS):\n"
                        f"- BTC Price: ${live_data.get('crypto', {}).get('btc_price', 0):,.0f} ({live_data.get('crypto', {}).get('btc_change_24h', 0):+.2f}% 24h)\n"
                        f"- ETH Price: ${live_data.get('crypto', {}).get('eth_price', 0):,.0f}\n"
                        f"- Fed Funds Rate: {live_data.get('macro', {}).get('fed_funds', 'N/A')}%\n"
                        f"- 10Y Treasury: {live_data.get('macro', {}).get('treasury_10y', 'N/A')}%\n"
                        f"- CPI YoY: {live_data.get('macro', {}).get('inflation_yoy', 'N/A')}%\n"
                        f"- GDP Growth: {live_data.get('macro', {}).get('gdp_growth', 'N/A')}%\n"
                        f"- Unemployment: {live_data.get('macro', {}).get('unemployment', 'N/A')}%\n\n"
                        "Create an EXECUTIVE investment portfolio report.\n"
                        "You MUST reference the live market data above in your analysis.\n"
                        "CRITICAL RULE: Never write 'Assumed', '?', or 'Unknown' next to any ticker. "
                        "Use ONLY the verified names in the ASSET IDENTITY section below.\n"
                        "Return Markdown ONLY.\n\n"
                        "VERIFIED ASSET IDENTITY (use these exact names — no assumptions):\n"
                        + "\n".join(
                            f"  {t}: {get_ticker_name(t)}"
                            for t in (payload.get("tickers") or list((payload.get("weights") or {}).keys()))
                        ) + "\n\n"
                        "PORTFOLIO DATA (JSON):\n"
                        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
                        "BASE DRAFT (for reference):\n"
                        f"{base_report_md}"
                    ),
                },
            ],
        )
        out = (response.choices[0].message.content or "").strip()
        if out:
            return out
        return base_report_md
        return base_report_md
    except Exception as e:
        logger.error(f"[ExecReport] LLM failed: {e}")
        return base_report_md

def _compute_extras(weights: dict, prices: pd.DataFrame, perf: dict) -> dict:
    """Compute Correlation Matrix, Stress Test, and Benchmark Comparison from price data."""
    import numpy as np
    extras: dict = {}
    tickers = [t for t in weights if t in prices.columns]
    if not tickers:
        return extras

    rets = prices[tickers].pct_change().dropna()

    # ── 1. Correlation Matrix ────────────────────────────────────────────────
    try:
        corr = rets.corr().round(2)
        header = "| Asset | " + " | ".join(tickers) + " |"
        sep    = "|-------|" + "|".join(["-------"] * len(tickers)) + "|"
        rows   = [f"| {t} | " + " | ".join(str(corr.loc[t, c]) for c in tickers) + " |" for t in tickers]
        extras["correlation_matrix"] = "\n".join([header, sep] + rows)
    except Exception as _e:
        logger.warning(f"[Extras] Correlation Matrix failed: {_e}")

    # ── 2. Stress Test (portfolio beta vs SPY) ───────────────────────────────
    try:
        spy_df = get_prices(["SPY"], start="2020-01-01", end=None, force_refresh=False)
        spy_rets = spy_df["SPY"].pct_change().dropna() if "SPY" in spy_df.columns else None
        port_rets = (rets * pd.Series({t: weights[t] for t in tickers})).sum(axis=1)
        if spy_rets is not None:
            common = port_rets.index.intersection(spy_rets.index)
            if len(common) > 10:
                p = port_rets.loc[common].values
                s = spy_rets.loc[common].values
                beta = float(np.cov(p, s)[0, 1] / np.var(s)) if np.var(s) != 0 else 1.0
            else:
                beta = 1.0
        else:
            beta = 1.0
        beta = round(max(0.1, min(beta, 3.0)), 2)
        scenarios = [
            ("S&P -10% (Correction)",  -0.10 * beta),
            ("S&P -30% (Bear Market)", -0.30 * beta),
            ("S&P -50% (Crisis)",      -0.50 * beta),
            ("Fed +2% Rate Shock",     -0.12 * beta),
            ("Bull Market +20%",       +0.20 * beta),
        ]
        lines = [f"- **{name}**: ≈ {val*100:+.1f}%" for name, val in scenarios]
        extras["stress_test"] = f"Portfolio Beta vs SPY: **{beta:.2f}**\n\n" + "\n".join(lines)
        extras["beta"] = beta
    except Exception as _e:
        logger.warning(f"[Extras] Stress Test failed: {_e}")

    # ── 3. Benchmark Comparison (60/40 SPY/AGG) ──────────────────────────────
    try:
        bench_df = get_prices(["SPY", "AGG"], start="2020-01-01", end=None, force_refresh=False)
        if "SPY" in bench_df.columns and "AGG" in bench_df.columns:
            b_rets = bench_df["SPY"].pct_change() * 0.60 + bench_df["AGG"].pct_change() * 0.40
            b_rets = b_rets.dropna()
            b_ann  = (1 + b_rets.mean()) ** 252 - 1
            b_vol  = b_rets.std() * (252 ** 0.5)
            b_sh   = (b_ann - 0.05) / b_vol if b_vol > 0 else 0.0
            pr = perf.get("expected_return", 0) * 100
            pv = perf.get("volatility", 0) * 100
            ps = perf.get("sharpe", 0)
            extras["benchmark"] = (
                f"| Metric | This Portfolio | 60/40 Benchmark |\n"
                f"|--------|---------------|------------------|\n"
                f"| Expected Annual Return | {pr:.1f}% | {b_ann*100:.1f}% |\n"
                f"| Annual Volatility | {pv:.1f}% | {b_vol*100:.1f}% |\n"
                f"| Sharpe Ratio | {ps:.2f} | {b_sh:.2f} |"
            )
    except Exception as _e:
        logger.warning(f"[Extras] Benchmark failed: {_e}")

    return extras


def generate_strategy_guide_llm(
    *,
    model: str = config.DEFAULT_MODEL,
    temperature: float = 0.7,
    risk_profile: str,
    tickers: list[str],
    weights: dict[str, float],
    performance: dict,
    target_return: float | None = None,
    max_drawdown: float | None = None,
) -> str:
    """
    Generates a rich, educational 'Blueprint' strategy guide instead of a raw table.
    """
    # ── Hard Rejection Gate ────────────────────────────────────────────────────
    # A portfolio with negative expected return or negative Sharpe must NEVER
    # receive an implementation plan. Output a structured rejection instead.
    _exp_ret  = float(performance.get("expected_return", 0) or 0)
    _vol      = float(performance.get("volatility", 0) or 0)
    _sharpe   = float(performance.get("sharpe", 0) or 0)
    _rf       = 0.045
    if _exp_ret < _rf or _sharpe < 0:
        _top3 = sorted(weights.items(), key=lambda x: -x[1])[:3]
        _conc = sum(w for _, w in _top3)
        _fixes = []
        if _exp_ret < _rf:
            _fixes.append("أضف أسهم US أو Gold أو Bonds لرفع العائد المتوقع فوق معدل الخطر الصفري (4.5%)")
        if _conc > 0.55:
            _fixes.append(f"الأصول العليا ({', '.join(t for t,_ in _top3)}) تمثل {_conc*100:.0f}% — حدّ كل أصل بـ 15-20%")
        if len(tickers) < 4:
            _fixes.append("زيد عدد الأصول للتنويع — المحفظة الحالية متركزة جداً")
        _fixes.append("أعد تشغيل الأوبتيمايزر مع شرط: العائد المتوقع > معدل الخطر الصفري")
        _fix_str = "\n".join(f"- {f}" for f in _fixes)
        from datetime import datetime as _dt2
        return (
            f"# ❌ Portfolio Rejected — Strategy Invalid\n"
            f"**Date:** {_dt2.now().strftime('%B %d, %Y')} | "
            f"**Risk Profile:** {risk_profile.upper()}\n\n"
            f"---\n\n"
            f"## ⚠️ Why This Portfolio Was Rejected\n\n"
            f"| Metric | Value | Required |\n"
            f"|--------|-------|----------|\n"
            f"| Expected Annual Return | **{_exp_ret*100:.2f}%** | > {_rf*100:.1f}% (risk-free rate) |\n"
            f"| Annual Volatility | {_vol*100:.2f}% | — |\n"
            f"| Sharpe Ratio | **{_sharpe:.2f}** | > 0 |\n"
            f"| Max Drawdown Limit | {(max_drawdown or 1)*100:.0f}% | must be respected |\n\n"
            f"> لو المحفظة بتخسر سنوياً وعندها Sharpe سالب، يبقى:\n"
            f"> - بتاخد مخاطرة بدون عائد مناسب\n"
            f"> - مفيش مبرر لتنفيذ هذه المحفظة في وضعها الحالي\n\n"
            f"---\n\n"
            f"## 🔧 المطلوب لتصحيح المحفظة\n\n"
            f"{_fix_str}\n\n"
            f"---\n\n"
            f"## 💡 بديل مقترح\n\n"
            f"اطلب: **\"ابني محفظة aggressive باستخدام US + GCC + Gold\"**\n\n"
            f"الأوبتيمايزر سيختار أفضل الأسهم من المكتبة الحية وسيضمن:\n"
            f"- عائد متوقع موجب\n"
            f"- Sharpe > 0\n"
            f"- Max Drawdown محكوم\n"
        )
    # ── End Rejection Gate ─────────────────────────────────────────────────────

    # ── Placeholder Ticker Gate ────────────────────────────────────────────────
    # A report with NEEDED, AS, or any placeholder MUST be blocked.
    # This is a hard institutional rule: no assumption in client-facing security identity.
    _placeholders = has_placeholder_tickers(weights)
    if _placeholders:
        return (
            "# ⛔ Report Blocked — Unverified Assets Detected\n\n"
            f"**The following assets in this portfolio could not be verified:** `{'`, `'.join(_placeholders)}`\n\n"
            "EisaX does not generate client-facing reports containing placeholder or unidentified securities. "
            "This is a hard institutional rule: **no assumption in client-facing security identity.**\n\n"
            "## How to Fix\n\n"
            "- Remove or replace unrecognized tickers before requesting a report\n"
            "- Ask EisaX to rebuild the portfolio: **\"Build me a GCC portfolio\"** — "
            "the optimizer will select verified, scored assets from the live market library\n\n"
            "> **Golden Rule:** Every asset in a client report must be verified by name, "
            "ticker, and market — no exceptions.\n"
        )
    # ── End Placeholder Gate ───────────────────────────────────────────────────

    try:
        client = get_client()

        # Format weights for the prompt
        w_str = render_weights(weights)

        # Construct the prompt
        from datetime import datetime as _dt
        _today_str = _dt.now().strftime("%B %d, %Y")
        _cash_cap = {"high": 15, "aggressive": 15, "medium": 20, "low": 30, "conservative": 30}.get(risk_profile.lower(), 20)
        system_content = (
            f"You are EisaX AI, an elite institutional portfolio strategist. Today is {_today_str}. "
            "You produce comprehensive, data-driven portfolio reports in Arabic or English based on the user's language. "
            "Your reports are detailed, analytical, and professional — like a CIO memo to a high-net-worth client. "
            "Always use the EXACT weights, tickers, and numbers provided. Never invent numbers. "
            f"IMPORTANT: For a {risk_profile} risk profile, never recommend holding more than {_cash_cap}% in cash. "
            "High cash destroys returns for growth-oriented investors."
        )
        
        constraints_str = ""
        if target_return:
            constraints_str += f"- Target Annual Return: {target_return*100:.0f}%\n"
        if max_drawdown:
            constraints_str += f"- Max Drawdown Limit: {max_drawdown*100:.0f}%\n"
        if performance.get("max_drawdown"):
            constraints_str += f"- Estimated Historical Drawdown: {performance['max_drawdown']*100:.1f}%\n"
            constraints_str += f"- Drawdown Constraint Satisfied: {performance.get('drawdown_satisfied', 'N/A')}\n"

        # ── Extra computed data (correlation, stress, benchmark) ─────────────
        _corr_md  = performance.get("correlation_matrix", "")
        _stress_md = performance.get("stress_test", "")
        _bench_md  = performance.get("benchmark", "")
        _extras_section = ""
        if _corr_md:
            _extras_section += f"\n## Correlation Matrix (pre-computed)\n{_corr_md}\n"
        if _stress_md:
            _extras_section += f"\n## Stress Test Data (pre-computed)\n{_stress_md}\n"
        if _bench_md:
            _extras_section += f"\n## Benchmark vs 60/40 (pre-computed)\n{_bench_md}\n"

        # ── Build verified ticker identity table ─────────────────────────────
        _verified_names = {t: get_ticker_name(t) for t in (tickers or list(weights.keys()))}
        _id_table = "\n".join(
            f"  - {t}: **{name}**" + (" ✓ verified" if name != t else " ⚠ unknown — describe as 'unlisted security'")
            for t, name in _verified_names.items()
        )
        # ────────────────────────────────────────────────────────────────────

        user_content = (
            f"Generate a full institutional portfolio report. Use these EXACT numbers — do not change them.\n\n"
            f"## Verified Asset Identity (USE THESE EXACT NAMES — NEVER write 'Assumed' or '?')\n"
            f"{_id_table}\n\n"
            f"## Portfolio Data\n"
            f"- Risk Profile: {risk_profile.upper()}\n"
            f"- Date: {_today_str}\n"
            f"- Assets: {', '.join(tickers)}\n"
            f"- Optimized Weights:\n{w_str}\n"
            f"- Expected Annual Return: {_fmt_pct(performance.get('expected_return', 0))}\n"
            f"- Annual Volatility: {_fmt_pct(performance.get('volatility', 0))}\n"
            f"- Sharpe Ratio: {_fmt_float(performance.get('sharpe', 0))}\n"
            + (f"- Client Constraints:\n{constraints_str}" if constraints_str else "")
            + _extras_section
            + f"\n\n"
            f"## Required Report Structure (follow exactly):\n\n"
            f"### 1. Portfolio Overview\n"
            f"Write 2-3 paragraphs: What is this portfolio? What is the investment philosophy behind it? "
            f"Who is it suitable for? What are the key characteristics of a {risk_profile} strategy?\n\n"
            f"### 2. Risk Profile Analysis\n"
            f"Explain the risk level in detail: expected volatility, drawdown scenarios, "
            f"what market conditions could hurt this portfolio, and what protects it. "
            f"Use the actual numbers provided.\n\n"
            f"### 3. Investment Thesis — Why This Portfolio?\n"
            f"Answer these questions directly:\n"
            f"- Why were THESE specific assets chosen (not alternatives)?\n"
            f"- What macro/fundamental thesis connects them?\n"
            f"- Are there stronger alternatives the investor should know about?\n"
            f"- What would make you replace any of these assets?\n"
            f"Be honest — if an asset seems weak, say so and suggest a superior alternative.\n\n"
            f"### 4. Asset Selection Rationale\n"
            f"For EACH asset in the portfolio, explain:\n"
            f"- What it is (brief description)\n"
            f"- Why it was selected for this specific portfolio\n"
            f"- What role it plays (Core / Growth / Hedge / Diversifier)\n"
            f"- Its weight and the reasoning behind that allocation percentage\n\n"
            f"### 5. Portfolio Allocation Table\n"
            f"Present a clean markdown table with: Asset | Weight | Role | Expected Contribution\n\n"
            f"### 6. Performance Metrics\n"
            f"Show all metrics in a table: Expected Return | Volatility | Sharpe Ratio | Max Drawdown\n"
            f"Then explain what each metric means for the client in plain language.\n\n"
            f"### 7. Implementation Plan\n"
            f"Step-by-step guide: How to buy these assets, suggested order of purchase, "
            f"rebalancing frequency, and monitoring approach. "
            f"IMPORTANT: For a {risk_profile} portfolio, cash positions should NOT exceed {_cash_cap}% — "
            f"deploy capital into assets, not idle cash.\n\n"
            f"### 8. Benchmark Comparison\n"
            + (f"Use the pre-computed benchmark table above. Compare this portfolio vs the 60/40 (SPY/AGG) benchmark "
               f"across return, volatility, and Sharpe. Explain whether the extra risk is justified.\n\n"
               if _bench_md else
               f"Compare this portfolio to a 60/40 (SPY/AGG) benchmark — estimate relative return and risk.\n\n")
            + f"### 9. Correlation & Diversification Analysis\n"
            + (f"Use the pre-computed correlation matrix above. Identify highly correlated pairs (>0.7) "
               f"and explain how diversification reduces overall portfolio risk. "
               f"Note any concentration risk.\n\n"
               if _corr_md else
               f"Discuss the diversification benefits between the selected assets and any correlation risks.\n\n")
            + f"### 10. Stress Test Scenarios\n"
            + (f"Use the pre-computed stress test data above. Show the impact of each scenario in a table "
               f"(Scenario | Market Move | Portfolio Impact). Explain how the portfolio would behave in each crisis.\n\n"
               if _stress_md else
               f"Estimate portfolio behavior under major market shocks (2008 crisis, COVID crash, rate shock).\n\n")
            + f"### 11. Risk Warning\n"
            f"Professional disclaimer about past performance, market risks, and recommendation to consult a financial advisor.\n\n"
            f"Write in a professional but accessible tone. Use markdown formatting. Be thorough and specific."
        )

        response = client.create_completion(
            model=model,
            temperature=temperature,
            max_tokens=6000,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content or "Error generating guide."
        
    except Exception as e:
        logger.error(f"[StrategyGuide] LLM failed: {e}")
        # ── Direct DeepSeek fallback (bypass LLMClient) ──────────────────────
        try:
            import requests as _req, os as _os
            _key = _os.getenv("DEEPSEEK_API_KEY", "")
            if _key:
                _today = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
                _assets = ", ".join(f"{t}({w*100:.0f}%)" for t, w in weights.items())
                _perf_s = f"Return {performance.get('expected_return',0)*100:.1f}%, Vol {performance.get('volatility',0)*100:.1f}%, Sharpe {performance.get('sharpe',0):.2f}"
                _fb_prompt = f"""You are EisaX, an elite institutional portfolio strategist. Today is {_today}.
Portfolio: {_assets}
Performance: {_perf_s}
Risk Profile: {risk_profile}

Write a concise CIO-grade strategy guide (markdown). Include:
1. Portfolio Overview (2 paragraphs)
2. Asset Rationale (bullet per asset)
3. Allocation Table (markdown table)
4. Key Risks & Opportunities
5. Professional Disclaimer

Be specific, use numbers, institutional tone."""
                _r = _req.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {_key}", "Content-Type": "application/json"},
                    json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": _fb_prompt}],
                          "max_tokens": 2000, "temperature": 0.6},
                    timeout=120
                )
                _data = _r.json()
                _out = _data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if _out:
                    logger.info("[StrategyGuide] Direct DeepSeek fallback succeeded")
                    return _out
        except Exception as _fb_e:
            logger.warning(f"[StrategyGuide] Direct fallback also failed: {_fb_e}")
        return render_optimize_reply(weights, performance) + "\n\n*(Guide generation failed — showing optimizer summary above)*"

# ============================================================
# CORE ACTIONS
