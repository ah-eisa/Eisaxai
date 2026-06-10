"""
Shariah Compliance Screener — EisaX.
Implements AAOIFI screening rules for individual stocks and portfolios.

Reference: AAOIFI Shariah Standard No. 21 + DJ Islamic Market Index methodology.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

log = logging.getLogger("eisax.shariah_screener")

# ── AAOIFI thresholds ──────────────────────────────────────────────────────────
DEBT_TO_MCAP_LIMIT     = 0.33   # Interest-bearing debt / 24m avg market cap
CASH_TO_MCAP_LIMIT     = 0.33   # Cash + interest-bearing securities / market cap
RECEIVABLES_LIMIT      = 0.49   # Accounts receivable / total assets
HARAM_REVENUE_LIMIT    = 0.05   # Non-permissible income / total revenue

# ── Prohibited business activities ─────────────────────────────────────────────
HARAM_SECTORS: set[str] = {
    "Banks",                         # conventional banking
    "Insurance",                     # conventional insurance
    "Consumer Finance",
    "Capital Markets",
    "Mortgage Real Estate",
    "Tobacco",
    "Beverages - Wineries & Distilleries",
    "Beverages - Brewers",
    "Casinos & Gaming",
    "Gambling",
    "Entertainment - Adult",
    "Defense",                       # weapons manufacturing
    "Aerospace & Defense",
}

HARAM_KEYWORDS: list[str] = [
    "alcohol", "wine", "beer", "brewery", "distillery",
    "tobacco", "cigarette",
    "casino", "gambling", "lottery", "betting",
    "pork", "swine",
    "adult", "pornography",
    "weapons", "arms manufacturer",
    "conventional bank", "interest-based lending",
    "interest_based",
]


def _safe_float(val) -> Optional[float]:
    """Safely convert to float, return None on failure."""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def screen_ticker(ticker: str, fundamentals: Optional[dict] = None) -> dict:
    """
    Screen a single ticker for Shariah compliance.

    Args:
        ticker: Stock ticker symbol.
        fundamentals: Optional pre-fetched dict with keys:
            - market_cap, total_debt, cash_and_equivalents,
              short_term_investments, accounts_receivable, total_assets,
              sector, industry, description, total_revenue, interest_income

    Returns:
        {
            "ticker": str,
            "compliant": bool,
            "verdict": "halal" | "haram" | "mixed" | "unknown",
            "reasons": list[str],          # why it failed (empty if halal)
            "warnings": list[str],          # data gaps or borderline
            "ratios": {
                "debt_ratio": float | None,
                "cash_ratio": float | None,
                "receivables_ratio": float | None,
                "haram_revenue_ratio": float | None,
            },
            "business_check": "pass" | "fail" | "unknown",
            "financial_check": "pass" | "fail" | "unknown",
        }
    """
    reasons: list[str] = []
    warnings: list[str] = []

    # ── Fetch fundamentals if not provided ────────────────────────────────────
    if fundamentals is None:
        fundamentals = _fetch_fundamentals(ticker)

    sector      = (fundamentals.get("sector") or "").strip()
    industry    = (fundamentals.get("industry") or "").strip()
    description = (fundamentals.get("description") or "").lower()

    # ── 1. Business activity screen ────────────────────────────────────────────
    business_check = "pass"
    if industry in HARAM_SECTORS or sector in HARAM_SECTORS:
        business_check = "fail"
        reasons.append(f"Prohibited business: {industry or sector}")

    for kw in HARAM_KEYWORDS:
        if kw in description and business_check == "pass":
            business_check = "fail"
            reasons.append(f"Description contains prohibited activity keyword: '{kw}'")
            break

    # ── 2. Financial ratio screens ─────────────────────────────────────────────
    mcap     = _safe_float(fundamentals.get("market_cap"))
    debt     = _safe_float(fundamentals.get("total_debt"))
    cash     = _safe_float(fundamentals.get("cash_and_equivalents")) or 0.0
    sti      = _safe_float(fundamentals.get("short_term_investments")) or 0.0
    ar       = _safe_float(fundamentals.get("accounts_receivable"))
    total_assets = _safe_float(fundamentals.get("total_assets"))
    revenue  = _safe_float(fundamentals.get("total_revenue"))
    int_inc  = _safe_float(fundamentals.get("interest_income"))

    ratios = {
        "debt_ratio":          None,
        "cash_ratio":          None,
        "receivables_ratio":   None,
        "haram_revenue_ratio": None,
    }
    financial_check = "pass"

    if mcap and mcap > 0:
        if debt is not None:
            ratios["debt_ratio"] = round(debt / mcap, 4)
            if ratios["debt_ratio"] > DEBT_TO_MCAP_LIMIT:
                financial_check = "fail"
                reasons.append(
                    f"Debt/MCap = {ratios['debt_ratio']:.1%} > {DEBT_TO_MCAP_LIMIT:.0%}"
                )
        else:
            warnings.append("Total debt unavailable — debt ratio not screened")

        cash_total = cash + sti
        if cash_total > 0:
            ratios["cash_ratio"] = round(cash_total / mcap, 4)
            if ratios["cash_ratio"] > CASH_TO_MCAP_LIMIT:
                financial_check = "fail"
                reasons.append(
                    f"Cash+STI/MCap = {ratios['cash_ratio']:.1%} > {CASH_TO_MCAP_LIMIT:.0%}"
                )
    else:
        warnings.append("Market cap unavailable — financial ratios not screened")
        financial_check = "unknown"

    if total_assets and total_assets > 0 and ar is not None:
        ratios["receivables_ratio"] = round(ar / total_assets, 4)
        if ratios["receivables_ratio"] > RECEIVABLES_LIMIT:
            financial_check = "fail"
            reasons.append(
                f"Receivables/Assets = {ratios['receivables_ratio']:.1%} > {RECEIVABLES_LIMIT:.0%}"
            )

    if revenue and revenue > 0 and int_inc is not None:
        ratios["haram_revenue_ratio"] = round(int_inc / revenue, 4)
        if ratios["haram_revenue_ratio"] > HARAM_REVENUE_LIMIT:
            financial_check = "fail"
            reasons.append(
                f"Interest income/Revenue = {ratios['haram_revenue_ratio']:.1%} > {HARAM_REVENUE_LIMIT:.0%}"
            )

    # ── Final verdict ──────────────────────────────────────────────────────────
    if business_check == "fail":
        verdict = "haram"
        compliant = False
    elif financial_check == "fail":
        verdict = "haram"
        compliant = False
    elif business_check == "unknown" or financial_check == "unknown":
        verdict = "unknown"
        compliant = False
    elif warnings:
        verdict = "mixed"
        compliant = True
    else:
        verdict = "halal"
        compliant = True

    return {
        "ticker":          ticker,
        "compliant":       compliant,
        "verdict":         verdict,
        "reasons":         reasons,
        "warnings":        warnings,
        "ratios":          ratios,
        "business_check":  business_check,
        "financial_check": financial_check,
        "sector":          sector,
        "industry":        industry,
    }


def _fetch_fundamentals(ticker: str) -> dict:
    """Fetch ticker fundamentals from yfinance — gracefully degrades on failure."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}

        # Balance sheet items
        debt = info.get("totalDebt")
        cash = info.get("totalCash")
        sti  = info.get("shortTermInvestments")

        # Try balance sheet if info missing
        if debt is None or cash is None:
            try:
                bs = t.balance_sheet
                if not bs.empty:
                    latest = bs.iloc[:, 0]
                    if debt is None:
                        debt = latest.get("Total Debt") or latest.get("Long Term Debt")
                    if cash is None:
                        cash = latest.get("Cash And Cash Equivalents")
            except Exception:
                pass

        try:
            fin = t.financials
            revenue = info.get("totalRevenue")
            if revenue is None and not fin.empty:
                latest = fin.iloc[:, 0]
                revenue = latest.get("Total Revenue")
            int_inc = info.get("interestIncome") or 0
        except Exception:
            revenue = info.get("totalRevenue")
            int_inc = 0

        return {
            "market_cap":              info.get("marketCap"),
            "total_debt":              debt,
            "cash_and_equivalents":    cash,
            "short_term_investments":  sti,
            "accounts_receivable":     info.get("netReceivables"),
            "total_assets":            info.get("totalAssets"),
            "total_revenue":           revenue,
            "interest_income":         int_inc,
            "sector":                  info.get("sector", ""),
            "industry":                info.get("industry", ""),
            "description":             info.get("longBusinessSummary", ""),
        }
    except Exception as e:
        log.warning(f"[shariah_screener] yf fetch failed for {ticker}: {e}")
        return {}


def screen_portfolio(positions_df: pd.DataFrame) -> dict:
    """
    Run Shariah screen on all positions in a portfolio.

    Args:
        positions_df: DataFrame from Portfolio.summary()["positions"].
                      Required columns: ticker, value.

    Returns:
        {
            "results": pd.DataFrame,
            "compliance_rate_pct": float,    # weighted by value
            "halal_count": int,
            "haram_count": int,
            "unknown_count": int,
            "total_halal_value": float,
            "total_haram_value": float,
            "purification_estimate": float,  # value × max(haram_revenue_ratio, 0) per holding
            "summary": str,
        }
    """
    rows = []
    total_val = float(positions_df["value"].sum()) if "value" in positions_df.columns else 0.0
    halal_val = 0.0
    haram_val = 0.0
    unknown_val = 0.0
    purification = 0.0
    halal_n = haram_n = unknown_n = 0

    for _, pos in positions_df.iterrows():
        ticker = str(pos.get("ticker", ""))
        value  = float(pos.get("value", 0) or 0)
        if not ticker:
            continue

        result = screen_ticker(ticker)
        verdict = result["verdict"]

        if verdict == "halal" or verdict == "mixed":
            halal_val += value
            halal_n += 1
        elif verdict == "haram":
            haram_val += value
            haram_n += 1
            # Estimate purification amount (donate haram revenue portion)
            hrr = result["ratios"].get("haram_revenue_ratio") or 0
            purification += value * max(hrr, 0)
        else:
            unknown_val += value
            unknown_n += 1

        emoji = {"halal": "✅", "haram": "❌", "mixed": "⚠️", "unknown": "❓"}.get(verdict, "❓")
        rows.append({
            "Ticker":   ticker,
            "Verdict":  f"{emoji} {verdict.upper()}",
            "Value":    round(value, 0),
            "Sector":   result["sector"] or "—",
            "Debt %":   f"{result['ratios']['debt_ratio']*100:.1f}" if result["ratios"]["debt_ratio"] is not None else "—",
            "Cash %":   f"{result['ratios']['cash_ratio']*100:.1f}" if result["ratios"]["cash_ratio"] is not None else "—",
            "Issues":   "; ".join(result["reasons"]) if result["reasons"] else ("OK" if verdict == "halal" else "—"),
        })

    compliance_pct = (halal_val / total_val * 100) if total_val > 0 else 0.0

    if compliance_pct >= 95:
        summary = "✅ Portfolio is Shariah-compliant"
    elif compliance_pct >= 70:
        summary = "⚠️ Portfolio is mostly compliant — review flagged holdings"
    else:
        summary = "❌ Portfolio has significant non-compliant holdings"

    return {
        "results":               pd.DataFrame(rows),
        "compliance_rate_pct":   round(compliance_pct, 1),
        "halal_count":           halal_n,
        "haram_count":           haram_n,
        "unknown_count":         unknown_n,
        "total_halal_value":     round(halal_val, 0),
        "total_haram_value":     round(haram_val, 0),
        "total_unknown_value":   round(unknown_val, 0),
        "purification_estimate": round(purification, 0),
        "summary":               summary,
    }
