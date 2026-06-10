"""
Dividend Income Projection — EisaX.
Forecasts annual income, yield-on-cost, payout sustainability, and dividend calendar.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

log = logging.getLogger("eisax.dividend_engine")


def _safe_float(val) -> Optional[float]:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def fetch_dividend_data(ticker: str) -> dict:
    """
    Fetch dividend metadata for a single ticker from yfinance.

    Returns:
        {
            "ticker": str,
            "div_yield_pct": float | None,
            "div_rate_annual": float | None,
            "payout_ratio": float | None,
            "fcf_coverage": float | None,
            "5y_growth_rate": float | None,
            "ex_div_date": str | None,
            "last_div_amount": float | None,
            "frequency": str,  # quarterly/semi/annual/irregular
            "history": pd.Series,  # historical dividends (last 5y)
            "error": str | None,
        }
    """
    try:
        import yfinance as yf
        from datetime import datetime, timedelta

        t = yf.Ticker(ticker)
        info = t.info or {}

        div_yield = info.get("dividendYield")
        # yfinance returns yield as decimal (0.025) sometimes, percent (2.5) other times
        if div_yield is not None and div_yield > 1:
            div_yield_pct = div_yield
        elif div_yield is not None:
            div_yield_pct = div_yield * 100
        else:
            div_yield_pct = None

        rate = _safe_float(info.get("dividendRate"))
        payout = _safe_float(info.get("payoutRatio"))
        if payout is not None and payout < 1.5:
            payout_pct = payout * 100
        else:
            payout_pct = payout

        # FCF coverage
        fcf       = _safe_float(info.get("freeCashflow"))
        div_paid  = rate * (_safe_float(info.get("sharesOutstanding")) or 0) if rate else None
        fcf_cov   = (fcf / div_paid) if (fcf and div_paid and div_paid > 0) else None

        # Dividend history
        try:
            div_hist = t.dividends
            cutoff = pd.Timestamp.now(tz=div_hist.index.tz if hasattr(div_hist.index, 'tz') and div_hist.index.tz else None) - pd.Timedelta(days=365 * 5)
            recent_divs = div_hist[div_hist.index >= cutoff] if not div_hist.empty else pd.Series(dtype=float)
        except Exception:
            recent_divs = pd.Series(dtype=float)

        # 5y growth rate
        growth_5y = None
        if not recent_divs.empty and len(recent_divs) >= 4:
            try:
                yearly = recent_divs.groupby(recent_divs.index.year).sum()
                if len(yearly) >= 2:
                    first, last = float(yearly.iloc[0]), float(yearly.iloc[-1])
                    years = len(yearly) - 1
                    if first > 0 and years > 0:
                        growth_5y = ((last / first) ** (1 / years) - 1) * 100
            except Exception:
                pass

        # Frequency detection
        frequency = "irregular"
        if not recent_divs.empty:
            try:
                last_yr_divs = recent_divs[recent_divs.index >= (pd.Timestamp.now(tz=recent_divs.index.tz if hasattr(recent_divs.index, 'tz') and recent_divs.index.tz else None) - pd.Timedelta(days=365))]
                n_per_year = len(last_yr_divs)
                if n_per_year >= 4:
                    frequency = "quarterly"
                elif n_per_year == 2:
                    frequency = "semi-annual"
                elif n_per_year == 1:
                    frequency = "annual"
            except Exception:
                pass

        # Ex-div date
        ex_div = info.get("exDividendDate")
        ex_div_str = None
        if ex_div:
            try:
                if isinstance(ex_div, (int, float)):
                    ex_div_str = datetime.fromtimestamp(ex_div).strftime("%Y-%m-%d")
                else:
                    ex_div_str = str(ex_div)
            except Exception:
                ex_div_str = None

        last_div = float(recent_divs.iloc[-1]) if not recent_divs.empty else None

        return {
            "ticker":          ticker,
            "div_yield_pct":   round(div_yield_pct, 2) if div_yield_pct is not None else None,
            "div_rate_annual": round(rate, 4) if rate else None,
            "payout_ratio":    round(payout_pct, 1) if payout_pct is not None else None,
            "fcf_coverage":    round(fcf_cov, 2) if fcf_cov else None,
            "5y_growth_rate":  round(growth_5y, 2) if growth_5y is not None else None,
            "ex_div_date":     ex_div_str,
            "last_div_amount": round(last_div, 4) if last_div else None,
            "frequency":       frequency,
            "history":         recent_divs,
            "error":           None,
        }
    except Exception as e:
        log.warning(f"[dividend_engine] fetch failed for {ticker}: {e}")
        return {
            "ticker":          ticker,
            "div_yield_pct":   None,
            "div_rate_annual": None,
            "payout_ratio":    None,
            "fcf_coverage":    None,
            "5y_growth_rate":  None,
            "ex_div_date":     None,
            "last_div_amount": None,
            "frequency":       "unknown",
            "history":         pd.Series(dtype=float),
            "error":           str(e),
        }


def project_portfolio_income(
    positions_df: pd.DataFrame,
    annual_contribution: float = 0.0,
    growth_assumption_pct: float = 0.0,
) -> dict:
    """
    Project annual dividend income from current portfolio holdings.

    Args:
        positions_df: DataFrame from Portfolio.summary()["positions"].
                      Required columns: ticker, qty, value, cost_basis (optional), price (optional).
        annual_contribution: Optional additional yearly investment.
        growth_assumption_pct: Optional dividend growth rate to apply forward.

    Returns:
        {
            "total_annual_income": float,
            "monthly_average_income": float,
            "portfolio_yield_pct": float,
            "yield_on_cost_pct": float | None,
            "weighted_payout_ratio": float | None,
            "weighted_growth_rate": float | None,
            "positions": pd.DataFrame,
            "monthly_calendar": pd.DataFrame,
            "projection_5y": pd.DataFrame,
            "sustainability_score": int,    # 0-100
            "warnings": list[str],
        }
    """
    warnings: list[str] = []
    rows = []
    monthly_income_buckets: dict[int, float] = {m: 0.0 for m in range(1, 13)}

    total_income = 0.0
    total_value  = 0.0
    total_cost   = 0.0
    weighted_payout = 0.0
    weighted_growth = 0.0
    growth_weight = 0.0

    for _, pos in positions_df.iterrows():
        ticker = str(pos.get("ticker", ""))
        qty    = float(pos.get("qty", 0) or 0)
        value  = float(pos.get("value", 0) or 0)
        cost   = float(pos.get("cost_basis", 0) or 0) * qty if pos.get("cost_basis") else 0
        price  = float(pos.get("price", 0) or 0)

        if not ticker:
            continue

        div_data = fetch_dividend_data(ticker)
        rate     = div_data.get("div_rate_annual")
        yld      = div_data.get("div_yield_pct")
        payout   = div_data.get("payout_ratio")
        growth   = div_data.get("5y_growth_rate")
        freq     = div_data.get("frequency", "unknown")

        # Compute annual income
        if rate:
            ann_income = rate * qty
        elif yld and value:
            ann_income = (yld / 100) * value
        else:
            ann_income = 0.0
            if value > 0:
                warnings.append(f"{ticker}: no dividend data")

        # Yield-on-cost
        yoc = (ann_income / cost * 100) if cost > 0 else None

        total_income += ann_income
        total_value  += value
        total_cost   += cost

        if payout is not None and value > 0:
            weighted_payout += (payout * value)
        if growth is not None and ann_income > 0:
            weighted_growth += growth * ann_income
            growth_weight   += ann_income

        # Distribute to monthly buckets based on frequency
        if ann_income > 0:
            if freq == "quarterly":
                for m in [3, 6, 9, 12]:
                    monthly_income_buckets[m] += ann_income / 4
            elif freq == "semi-annual":
                for m in [6, 12]:
                    monthly_income_buckets[m] += ann_income / 2
            elif freq == "annual":
                ex_month = 12
                if div_data.get("ex_div_date"):
                    try:
                        ex_month = int(div_data["ex_div_date"][5:7])
                    except Exception:
                        pass
                monthly_income_buckets[ex_month] += ann_income
            else:
                # spread evenly
                for m in range(1, 13):
                    monthly_income_buckets[m] += ann_income / 12

        rows.append({
            "Ticker":             ticker,
            "Qty":                qty,
            "Annual Income":      round(ann_income, 2),
            "Yield %":            yld if yld is not None else "—",
            "Yield on Cost %":    round(yoc, 2) if yoc is not None else "—",
            "Payout %":           payout if payout is not None else "—",
            "5y Growth %":        growth if growth is not None else "—",
            "Frequency":          freq,
            "Last Ex-Div":        div_data.get("ex_div_date") or "—",
        })

    pos_df_out = pd.DataFrame(rows).sort_values("Annual Income", ascending=False) if rows else pd.DataFrame()

    portfolio_yield = (total_income / total_value * 100) if total_value > 0 else 0.0
    yoc_total       = (total_income / total_cost * 100) if total_cost > 0 else None
    avg_payout      = (weighted_payout / total_value) if total_value > 0 and weighted_payout > 0 else None
    avg_growth      = (weighted_growth / growth_weight) if growth_weight > 0 else None

    # Monthly calendar
    cal_rows = [
        {"Month": pd.Timestamp(2025, m, 1).strftime("%b"),
         "Expected Income": round(monthly_income_buckets[m], 2)}
        for m in range(1, 13)
    ]
    monthly_cal_df = pd.DataFrame(cal_rows)

    # 5-year projection
    proj_rows = []
    proj_inc  = total_income
    proj_val  = total_value
    g_rate    = (avg_growth or growth_assumption_pct) / 100
    for yr in range(1, 6):
        proj_inc *= (1 + g_rate)
        # Add contributions (assumed at portfolio yield)
        proj_val += annual_contribution
        proj_inc += annual_contribution * (portfolio_yield / 100)
        proj_rows.append({
            "Year":           yr,
            "Annual Income":  round(proj_inc, 0),
            "Monthly Avg":    round(proj_inc / 12, 0),
            "Cumulative":     round(sum(r["Annual Income"] for r in proj_rows) + proj_inc, 0),
        })
    proj_df = pd.DataFrame(proj_rows)

    # Sustainability score (0-100): payout (lower better), FCF coverage (higher better), growth (higher better)
    sus_score = 50
    if avg_payout is not None:
        if avg_payout < 40:
            sus_score += 25
        elif avg_payout < 60:
            sus_score += 15
        elif avg_payout < 80:
            sus_score += 5
        else:
            sus_score -= 15
    if avg_growth is not None:
        if avg_growth > 8:
            sus_score += 15
        elif avg_growth > 4:
            sus_score += 8
        elif avg_growth < 0:
            sus_score -= 10
    sus_score = max(0, min(100, sus_score))

    return {
        "total_annual_income":   round(total_income, 2),
        "monthly_average_income": round(total_income / 12, 2),
        "portfolio_yield_pct":   round(portfolio_yield, 2),
        "yield_on_cost_pct":     round(yoc_total, 2) if yoc_total is not None else None,
        "weighted_payout_ratio": round(avg_payout, 1) if avg_payout is not None else None,
        "weighted_growth_rate":  round(avg_growth, 2) if avg_growth is not None else None,
        "positions":             pos_df_out,
        "monthly_calendar":      monthly_cal_df,
        "projection_5y":         proj_df,
        "sustainability_score":  sus_score,
        "warnings":              warnings,
    }
