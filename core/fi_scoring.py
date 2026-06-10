"""fi_scoring.py -- EisaX Fixed Income: scoring engine and prompt formatter."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from core.fi_routing import _SUKUK_STRUCTURES, _COUNTRY_RATINGS

logger = logging.getLogger(__name__)

def compute_fi_score(data: dict, investor_currency: str = "USD") -> dict:
    """
    Compute EisaX Fixed Income Score (0-100).

    Factors:
      1. Yield Attractiveness  — 0-25
      2. Credit Quality        — 0-30
      3. Liquidity             — 0-15
      4. Duration Risk         — 0-15
      5. Sharia / FX           — 0-15  (Sharia compliance if Sukuk, else FX risk)

    Components with no underlying data are flagged N/A and excluded from the
    denominator, then the achieved score is rescaled to /100.  This prevents a
    missing data point from unfairly pulling down an otherwise good instrument.

    Returns dict with individual scores + na_flags + total + verdict.
    """
    _MAX = {
        "yield_attractiveness": 25,
        "credit_quality":       30,
        "liquidity":            15,
        "duration_risk":        15,
        "sharia_or_fx":         15,
    }
    scores:   dict[str, int] = {}
    na_flags: set[str]       = set()   # components excluded from scoring

    # ── 1. Yield Attractiveness (0-25) ────────────────────────────────────────
    # Use YTM if available (more accurate), else fall back to coupon
    benchmarks = data.get("benchmarks", {})
    coupon = data.get("coupon")
    reference_yield = data.get("ytm_pct") or coupon   # prefer YTM over coupon

    if reference_yield is None:
        # No yield data at all → cannot score this factor
        na_flags.add("yield_attractiveness")
    else:
        if benchmarks:
            bench_yield = None
            for _lbl, val in benchmarks.items():
                if isinstance(val, (int, float)):
                    bench_yield = val
                    break
            if bench_yield is not None:
                spread_bps = (reference_yield - bench_yield) * 100
                if spread_bps <= 0:
                    scores["yield_attractiveness"] = 5
                elif spread_bps <= 100:
                    scores["yield_attractiveness"] = 12
                elif spread_bps <= 200:
                    scores["yield_attractiveness"] = 18
                elif spread_bps <= 350:
                    scores["yield_attractiveness"] = 22
                else:
                    scores["yield_attractiveness"] = min(20, int(22 - (spread_bps - 350) / 50))
            else:
                # Benchmark lookup failed — score on absolute yield only
                if reference_yield >= 6.0:
                    scores["yield_attractiveness"] = 18
                elif reference_yield >= 4.0:
                    scores["yield_attractiveness"] = 14
                elif reference_yield >= 2.0:
                    scores["yield_attractiveness"] = 8
                else:
                    scores["yield_attractiveness"] = 4
        else:
            # No benchmarks available — score on absolute yield
            if reference_yield >= 6.0:
                scores["yield_attractiveness"] = 18
            elif reference_yield >= 4.0:
                scores["yield_attractiveness"] = 14
            elif reference_yield >= 2.0:
                scores["yield_attractiveness"] = 8
            else:
                scores["yield_attractiveness"] = 4

    # ── 2. Credit Quality (0-30) ──────────────────────────────────────────────
    # credit_score is pre-computed from rating/CDS; None means no credit data
    raw_credit = data.get("credit_score")
    if raw_credit is None:
        na_flags.add("credit_quality")
    else:
        scores["credit_quality"] = int(raw_credit)

    # ── 3. Liquidity (0-15) ───────────────────────────────────────────────────
    exchange = (data.get("exchange") or "").upper()
    major_exchanges = {"LSE", "NYSE", "NASDAQ", "DIFX", "DIFX/NASDAQ DUBAI", "EURONEXT", "SGX", "TSE"}
    if any(ex in exchange for ex in major_exchanges):
        scores["liquidity"] = 12
    elif exchange and exchange not in ("OTC", "GREY", "PINK", ""):
        scores["liquidity"] = 9
    elif exchange == "OTC":
        sec_type = (data.get("security_type") or "").lower()
        if "govt" in sec_type or "sovereign" in sec_type:
            scores["liquidity"] = 10
        else:
            scores["liquidity"] = 6
    else:
        scores["liquidity"] = 5

    # ── 4. Duration Risk (0-15) ───────────────────────────────────────────────
    ytm_yrs = data.get("years_to_maturity")
    if ytm_yrs is None:
        # Unknown maturity for a Bond ETF (perpetual) → not applicable
        if "etf" in (data.get("security_type") or "").lower():
            na_flags.add("duration_risk")
        else:
            scores["duration_risk"] = 7   # unknown bond — penalise slightly
    elif ytm_yrs < 0:
        scores["duration_risk"] = 0   # matured / past maturity
    elif ytm_yrs <= 1:
        scores["duration_risk"] = 15
    elif ytm_yrs <= 2:
        scores["duration_risk"] = 13
    elif ytm_yrs <= 3:
        scores["duration_risk"] = 11
    elif ytm_yrs <= 5:
        scores["duration_risk"] = 9
    elif ytm_yrs <= 10:
        scores["duration_risk"] = 6
    else:
        scores["duration_risk"] = 3

    # ── 5a. Sharia Compliance (0-15) — for Sukuk ─────────────────────────────
    if data.get("is_sukuk"):
        sukuk_structure = data.get("sukuk_structure")
        scores["sharia_or_fx"] = 15 if sukuk_structure else 11

    # ── 5b. FX Risk (0-15) — for conventional bonds / non-Sukuk ─────────────
    else:
        currency = (data.get("currency") or "USD").upper()
        if currency == investor_currency.upper():
            scores["sharia_or_fx"] = 15
        elif currency in ("AED", "SAR", "QAR", "BHD"):
            scores["sharia_or_fx"] = 13
        elif currency in ("USD", "EUR", "GBP", "JPY", "CHF"):
            scores["sharia_or_fx"] = 11
        elif currency in ("KWD", "OMR"):
            scores["sharia_or_fx"] = 10
        else:
            scores["sharia_or_fx"] = 5

    # ── Total — rescale to /100 using only scored (non-N/A) components ────────
    available_max   = sum(v for k, v in _MAX.items() if k not in na_flags)
    available_score = sum(scores.get(k, 0) for k in _MAX if k not in na_flags)

    if available_max > 0:
        total = round((available_score / available_max) * 100)
    else:
        total = 0
    total = min(100, max(0, total))

    # ── Verdict ───────────────────────────────────────────────────────────────
    if total >= 80:
        verdict, label = "STRONG BUY", "🟢"
    elif total >= 65:
        verdict, label = "BUY", "🟢"
    elif total >= 50:
        verdict, label = "HOLD", "🟡"
    elif total >= 35:
        verdict, label = "UNDERWEIGHT", "🟠"
    else:
        verdict, label = "AVOID", "🔴"

    return {
        "scores":    scores,
        "na_flags":  sorted(na_flags),   # components excluded (no data)
        "available_max": available_max,  # denominator before rescaling
        "total":     total,
        "verdict":   verdict,
        "verdict_label": label,
    }


# ── Prompt Formatter ──────────────────────────────────────────────────────────

def format_fi_for_prompt(data: dict, score: dict) -> str:
    """
    Format instrument data + score as a rich block for LLM prompt injection.
    Includes: YTM, CDS spread, rating with date, precise maturity.
    """
    isin          = data.get("isin", "N/A")
    name          = data.get("name") or "Unknown Instrument"
    issuer        = data.get("issuer") or "Unknown Issuer"
    security_type = data.get("security_type") or ("Sukuk" if data.get("is_sukuk") else "Fixed Income Instrument")
    exchange      = data.get("exchange") or "OTC / Unlisted"
    currency      = data.get("currency", "USD")
    coupon        = data.get("coupon")
    maturity      = data.get("maturity") or "Unknown"
    ytm_years     = data.get("years_to_maturity")
    country_code  = data.get("country_code", "")
    country_rating  = data.get("country_rating", "--")
    rating_date     = data.get("rating_date")
    rating_outlook  = data.get("rating_outlook")
    rating_agency   = data.get("rating_agency")
    cds_bps         = data.get("cds_spread_bps")
    cds_def_prob    = data.get("cds_default_prob_5y")   # WGB-provided, more accurate than formula
    market_price    = data.get("market_price")
    ytm_pct         = data.get("ytm_pct")
    ytm_source      = data.get("ytm_source")
    is_sukuk        = data.get("is_sukuk", False)
    sukuk_structure = data.get("sukuk_structure")
    benchmarks      = data.get("benchmarks", {})
    fx_rate         = data.get("fx_rate")
    fetched_at      = data.get("fetched_at", "")

    lines = [
        "[FIXED INCOME — LIVE INSTRUMENT DATA]",
        f"Fetched: {fetched_at}  |  Source: {data.get('source', 'OpenFIGI')}",
        "",
        "INSTRUMENT OVERVIEW:",
        f"  ISIN:          {isin}",
        f"  Name:          {name}",
        f"  Issuer:        {issuer}",
        f"  Type:          {security_type}" + (" ✦ SUKUK (Islamic)" if is_sukuk else ""),
        f"  Exchange:      {exchange}",
        f"  Currency:      {currency}",
    ]

    # Coupon
    if coupon is not None:
        lines.append(f"  Coupon:        {coupon:.3f}% p.a. (stated/face rate)")
    else:
        lines.append(f"  Coupon:        N/A (not available)")

    # YTM — key new field
    if ytm_pct is not None:
        ytm_src_note = f"  [{ytm_source}]" if ytm_source else ""
        lines.append(f"  YTM (live):    {ytm_pct:.3f}%{ytm_src_note}")
        if coupon and ytm_pct > coupon:
            discount_pct = ((ytm_pct - coupon) / coupon) * 100
            lines.append(f"  ↳ YTM > Coupon by {ytm_pct - coupon:.2f}pp — trading at DISCOUNT (bond price < par)")
        elif coupon and ytm_pct < coupon:
            lines.append(f"  ↳ YTM < Coupon — trading at PREMIUM (bond price > par)")
    elif market_price is not None and coupon is not None:
        lines.append(f"  Market Price:  {market_price:.2f} (% of par)")
        lines.append(f"  YTM:           Not directly available — see Yield Analysis section")
    else:
        lines.append(f"  YTM:           Not available (no market price found)")

    # Maturity — with precise date
    if maturity and maturity != "Unknown":
        # Flag month-only estimates
        if maturity.endswith("-01") and "month-only" in str(data.get("maturity_precision", "")):
            mat_display = maturity[:7] + " (month estimated)"
        else:
            mat_display = maturity
        dur_str = (f"  ({ytm_years:.1f} years remaining)"
                   if ytm_years is not None and ytm_years > 0
                   else "  ⚠️ MATURED / NEAR MATURITY")
        lines.append(f"  Maturity:      {mat_display}{dur_str}")
    else:
        lines.append(f"  Maturity:      Not available — user should confirm from term sheet")

    # Credit rating with date, agency, and staleness warning
    if country_rating and country_rating != "--":
        rating_detail = country_rating
        if rating_agency:
            rating_detail = f"{rating_agency} {rating_detail}"
        if rating_date:
            rating_detail += f"  (last action: {rating_date})"
            # Flag ratings older than 2 years as potentially stale
            try:
                from datetime import datetime as _dt
                _action_year = int(str(rating_date)[:4])
                _age_years = _dt.now().year - _action_year
                if _age_years >= 2:
                    rating_detail += f"  ⚠️ {_age_years}yr old — verify current rating"
            except Exception:
                pass
        else:
            rating_detail += "  ⚠️ date unknown — verify with current source (Moody's/S&P/Fitch)"
        if rating_outlook:
            rating_detail += f"  | Outlook: {rating_outlook}"
        lines.append(f"  Rating:        {rating_detail}")

    # CDS spread — key new field for credit risk
    if cds_bps is not None:
        # Contextualise the CDS level
        if cds_bps < 30:
            cds_label = "VERY LOW (near risk-free)"
        elif cds_bps < 100:
            cds_label = "LOW (investment grade)"
        elif cds_bps < 200:
            cds_label = "MODERATE"
        elif cds_bps < 500:
            cds_label = "HIGH (sub-investment grade)"
        elif cds_bps < 1000:
            cds_label = "VERY HIGH (distressed)"
        else:
            cds_label = "EXTREME (near-default territory)"
        lines.append(f"  5Y CDS:        {cds_bps:.1f} bps — {cds_label}")
        # Use WGB-provided default probability if available; else calculate correctly
        if cds_def_prob is not None:
            lines.append(f"  ↳ Implied 5Y default prob: ~{float(cds_def_prob):.1f}% (source: worldgovernmentbonds.com)")
        else:
            # Correct formula: PD_annual = CDS_spread / (1 - recovery_rate)
            annual_default_prob  = (cds_bps / 10000) / (1 - 0.4)   # 40% recovery
            five_yr_default_prob = 1 - (1 - annual_default_prob) ** 5
            lines.append(f"  ↳ Implied 5Y default prob: ~{five_yr_default_prob*100:.1f}% (recovery 40%)")

    # Sukuk structure
    if is_sukuk:
        lines.append("")
        lines.append("SUKUK STRUCTURE:")
        if sukuk_structure:
            lines.append(f"  Structure:     {sukuk_structure}")
        else:
            lines.append(f"  Structure:     Trust Certificates (specific structure not identified from ISIN)")
        lines.append(f"  Sharia Status: Compliant (Sukuk designation)")
        lines.append(f"  Note: Periodic distributions replace conventional coupon payments")

    # Benchmark yields with spread
    if benchmarks:
        lines.append("")
        lines.append("BENCHMARK YIELDS (for spread analysis):")
        # Use YTM if available, else coupon for spread calculation
        reference_yield = ytm_pct or coupon
        for label, val in benchmarks.items():
            if isinstance(val, (int, float)):
                spread_str = ""
                if reference_yield is not None:
                    spread_bps = (reference_yield - val) * 100
                    ytm_note = " (YTM-based)" if ytm_pct else " (coupon-based)"
                    spread_str = f"  →  Spread: {spread_bps:+.0f}bps{ytm_note}"
                lines.append(f"  {label:<22}: {val:.2f}%{spread_str}")
            else:
                lines.append(f"  {label}")

    # FX context
    if fx_rate:
        lines.append("")
        lines.append("FX CONTEXT:")
        if currency in ("AED", "SAR", "QAR", "BHD", "OMR"):
            lines.append(f"  {currency}/USD: PEGGED ({fx_rate:.4f}) — negligible FX risk for USD investors")
        elif currency == "USD":
            lines.append(f"  USD-denominated — no FX conversion needed for USD investors")
        elif currency == "PKR":
            lines.append(f"  PKR/USD: ~{fx_rate:.2f} (floating, significant volatility) — MAJOR FX RISK for foreign investors")
        else:
            lines.append(f"  USD/{currency}: {fx_rate:.4f} (floating — FX risk applies)")

    # Score block — hard-coded table so the LLM copies it verbatim
    # DO NOT let the LLM recalculate or add a "weighted" column.
    if score:
        s        = score.get("scores", {})
        na_flags = set(score.get("na_flags", []))
        total    = score.get("total", 0)
        avail_max = score.get("available_max", 100)
        fifth_label = "Sharia Compliance" if is_sukuk else "FX Risk (inverted)"

        lines.append("")
        lines.append(f"EISAX FIXED INCOME SCORE: {total}/100  {score.get('verdict_label','')} {score.get('verdict','')}")
        lines.append("─" * 52)
        lines.append(f"  {'Factor':<24} {'Max':>5}  {'Score':>7}")
        lines.append("  " + "─" * 40)

        for key, lbl, mx in [
            ("yield_attractiveness", "Yield Attractiveness", 25),
            ("credit_quality",       "Credit Quality",       30),
            ("liquidity",            "Liquidity",            15),
            ("duration_risk",        "Duration Risk",        15),
            ("sharia_or_fx",         fifth_label,            15),
        ]:
            if key in na_flags:
                val_str = "  N/A"
                note    = "  ← no data, excluded from scoring"
            else:
                val_str = f"{s.get(key, 0):5d}"
                note    = ""
            lines.append(f"  {lbl:<24} {mx:>5}  {val_str}{note}")

        lines.append("  " + "─" * 40)
        if na_flags:
            lines.append(f"  {'TOTAL (rescaled)':<24} {avail_max:>5} → {total:>5}/100")
            lines.append(f"  ⚠ {len(na_flags)} factor(s) had no data and were excluded.")
            lines.append(f"    Score was rescaled from /{avail_max} to /100 to avoid penalising missing data.")
        else:
            lines.append(f"  {'TOTAL':<24} {'100':>5}  {total:>7}")
        lines.append("─" * 52)

        # CDS note if used in credit scoring
        if cds_bps is not None:
            lines.append(f"  ↳ Credit score CDS-adjusted (live market signal: {cds_bps:.0f}bps)")

        lines.append("")
        lines.append("⚠ IMPORTANT FOR REPORT: Copy the score table above EXACTLY as shown.")
        lines.append("  DO NOT add a 'weighted score' column. DO NOT recalculate the total.")
        lines.append("  The total shown IS the final score — no further arithmetic needed.")

    return "\n".join(lines)
