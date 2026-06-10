"""
etf_intelligence.py — EisaX ETF Classification & Analysis Engine
==================================================================

Detects ETF type from yfinance metadata, builds ETF-specific:
  • data_block  (for DeepSeek prompt)
  • scenarios   (type-appropriate stress tests)
  • scorecard   (different factors per ETF type)
  • prompt_hint (section instructions for DeepSeek)

ETF Types recognized:
  commodity_gold    – GLD, IAU, SGOL
  commodity_oil     – USO, BNO, UCO
  commodity_silver  – SLV, SIVR
  commodity_other   – DBA, WEAT, CORN, SOYB, CANE, PDBC
  bond_treasury     – TLT, IEF, SHY, BIL, GOVT, SGOV
  bond_corporate    – LQD, HYG, JNK, VCIT, FALN
  bond_tips         – TIP, SCHP, VTIP
  equity_index_us   – SPY, QQQ, IVV, VTI, DIA, VOO, IWM
  equity_index_intl – EFA, EEM, VWO, IEMG, ACWI
  equity_sector     – XLK, XLF, XLE, XLV, SMH, SOXX, XLY
  leveraged         – TQQQ, SOXL, SPXL, SQQQ, SOXS
  reit_etf          – VNQ, SCHH, IYR, XLRE
  dividend          – SCHD, VYM, DVY, JEPI, JEPQ, HDV
  thematic          – ARKK, ARKG, BOTZ, ROBO, AIQ, DRIV
"""

import logging, re
logger = logging.getLogger(__name__)

_EQUITY_ONLY_SUFFIXES = (".CA",)

# ── Known ETF overrides (when yfinance category is ambiguous) ─────────────────
_KNOWN_ETF_TYPES: dict[str, str] = {
    # Commodity - Gold
    "GC=F":"commodity_gold","GLD":"commodity_gold","IAU":"commodity_gold","SGOL":"commodity_gold","BAR":"commodity_gold",
    "GLDM":"commodity_gold","AAAU":"commodity_gold","PHYS":"commodity_gold",
    # Commodity - Oil
    "CL=F":"commodity_oil","BZ=F":"commodity_oil","USO":"commodity_oil","BNO":"commodity_oil","UCO":"commodity_oil","SCO":"commodity_oil","DBO":"commodity_oil",
    # Commodity - Silver
    "SI=F":"commodity_silver","SLV":"commodity_silver","SIVR":"commodity_silver","PSLV":"commodity_silver",
    # Commodity - Platinum
    "PL=F":"commodity_platinum","PPLT":"commodity_platinum",
    # Commodity - Palladium
    "PA=F":"commodity_palladium","PALL":"commodity_palladium",
    # Commodity - Copper
    "HG=F":"commodity_copper","CPER":"commodity_copper",
    # Commodity - Other
    "DBA":"commodity_other","PDBC":"commodity_other","DJP":"commodity_other",
    "WEAT":"commodity_other","CORN":"commodity_other","SOYB":"commodity_other",
    "CANE":"commodity_other","WOOD":"commodity_other",
    # Bond - Treasury
    "TLT":"bond_treasury","IEF":"bond_treasury","SHY":"bond_treasury","BIL":"bond_treasury",
    "GOVT":"bond_treasury","SGOV":"bond_treasury","VGLT":"bond_treasury","VGIT":"bond_treasury",
    "TBT":"bond_treasury","TBF":"bond_treasury","TMF":"bond_treasury",
    # Bond - Corporate
    "LQD":"bond_corporate","HYG":"bond_corporate","JNK":"bond_corporate",
    "VCIT":"bond_corporate","IGIB":"bond_corporate","FALN":"bond_corporate","ANGL":"bond_corporate",
    # Bond - TIPS
    "TIP":"bond_tips","SCHP":"bond_tips","VTIP":"bond_tips","STIP":"bond_tips",
    # Equity Index - US
    "SPY":"equity_index_us","IVV":"equity_index_us","VOO":"equity_index_us","VTI":"equity_index_us",
    "QQQ":"equity_index_us","QQQM":"equity_index_us","DIA":"equity_index_us","IWM":"equity_index_us",
    "RSP":"equity_index_us","SCHX":"equity_index_us","ITOT":"equity_index_us",
    # Equity Index - International
    "EFA":"equity_index_intl","EEM":"equity_index_intl","VWO":"equity_index_intl",
    "IEMG":"equity_index_intl","ACWI":"equity_index_intl","VEA":"equity_index_intl",
    "MCHI":"equity_index_intl","EWZ":"equity_index_intl","EWJ":"equity_index_intl",
    # Sector ETFs
    "XLK":"equity_sector","XLF":"equity_sector","XLE":"equity_sector","XLV":"equity_sector",
    "XLY":"equity_sector","XLI":"equity_sector","XLB":"equity_sector","XLP":"equity_sector",
    "XLU":"equity_sector","XLRE":"equity_sector","SMH":"equity_sector","SOXX":"equity_sector",
    "IBB":"equity_sector","KRE":"equity_sector","KBE":"equity_sector","HACK":"equity_sector",
    "AIQ":"equity_sector","SKYY":"equity_sector","FINX":"equity_sector",
    # Leveraged
    "TQQQ":"leveraged","SOXL":"leveraged","SPXL":"leveraged","FNGU":"leveraged","NAIL":"leveraged",
    "SQQQ":"leveraged","SOXS":"leveraged","SPXS":"leveraged","FNGD":"leveraged","UVXY":"leveraged",
    "LABD":"leveraged","LABU":"leveraged","NUGT":"leveraged","DUST":"leveraged",
    # REIT ETF
    "VNQ":"reit_etf","SCHH":"reit_etf","IYR":"reit_etf","RWR":"reit_etf","USRT":"reit_etf",
    # Dividend
    "SCHD":"dividend","VYM":"dividend","DVY":"dividend","HDV":"dividend",
    "JEPI":"dividend","JEPQ":"dividend","DIVO":"dividend","NUSI":"dividend","QYLD":"dividend",
    # Thematic
    "ARKK":"thematic","ARKG":"thematic","ARKQ":"thematic","ARKW":"thematic","ARKF":"thematic",
    "BOTZ":"thematic","ROBO":"thematic","DRIV":"thematic","ESGU":"thematic","ICLN":"thematic",
    "CLOU":"thematic","WCLD":"thematic","METV":"thematic","BETZ":"thematic","POTX":"thematic",
}

# ── Category keyword → ETF type mapping ──────────────────────────────────────
_CATEGORY_MAP = {
    "commodities":           "commodity_other",
    "long government":       "bond_treasury",
    "intermediate government":"bond_treasury",
    "short government":      "bond_treasury",
    "ultrashort bond":       "bond_treasury",
    "long-term bond":        "bond_treasury",
    "corporate bond":        "bond_corporate",
    "high yield bond":       "bond_corporate",
    "inflation-protected":   "bond_tips",
    "trading--leveraged":    "leveraged",
    "leveraged":             "leveraged",
    "bear market":           "leveraged",
    "inverse":               "leveraged",
    "real estate":           "reit_etf",
    "large blend":           "equity_index_us",
    "large growth":          "equity_index_us",
    "large value":           "dividend",
    "diversified emerging":  "equity_index_intl",
    "foreign large":         "equity_index_intl",
    "technology":            "equity_sector",
    "financial":             "equity_sector",
    "health":                "equity_sector",
    "energy limited":        "equity_sector",
    "mid-cap":               "thematic",
    "small-cap":             "equity_index_us",
}

# ── Human-readable labels ─────────────────────────────────────────────────────
ETF_LABELS = {
    "commodity_gold":      "Gold Futures",
    "commodity_oil":       "Crude Oil Futures",
    "commodity_silver":    "Silver Futures",
    "commodity_platinum":  "Platinum Futures",
    "commodity_palladium": "Palladium Futures",
    "commodity_copper":    "Copper Futures",
    "commodity_other":     "Commodity Futures",
    "bond_treasury":     "Treasury Bond ETF",
    "bond_corporate":    "Corporate Bond ETF",
    "bond_tips":         "Inflation-Protected Bond ETF (TIPS)",
    "equity_index_us":   "US Equity Index ETF",
    "equity_index_intl": "International Equity ETF",
    "equity_sector":     "Sector Equity ETF",
    "leveraged":         "Leveraged/Inverse ETF",
    "reit_etf":          "Real Estate (REIT) ETF",
    "dividend":          "Dividend Income ETF",
    "thematic":          "Thematic / Active ETF",
}


# ─── 1. ETF Detection ─────────────────────────────────────────────────────────
def detect_etf(ticker: str, fund_data: dict | None = None) -> dict | None:
    """
    Returns ETF metadata dict if ticker is an ETF, else None.
    {
      "is_etf": True,
      "etf_type": "commodity_gold",
      "etf_label": "Gold Commodity ETF",
      "long_name": "SPDR Gold Shares",
      "fund_family": "State Street",
      "category": "Commodities Focused",
      "expense_ratio": 0.40,
      "aum": 58e9,
      "nav": 413.04,
      "yield": 0.0,
    }
    """
    ticker_upper = str(ticker or "").upper()
    if ticker_upper.endswith(_EQUITY_ONLY_SUFFIXES):
        return None

    base = ticker.upper().split(".")[0]

    # 1. Check known list first (fast, no API)
    known_type = _KNOWN_ETF_TYPES.get(base)

    # 2. Check yfinance quoteType
    yf_info: dict = fund_data or {}
    if not yf_info.get("quoteType"):
        try:
            import yfinance as yf, warnings
            warnings.filterwarnings("ignore")
            yf_info = yf.Ticker(ticker).info or {}
        except Exception:
            pass

    quote_type = str(yf_info.get("quoteType", "")).upper()
    if quote_type not in ("ETF", "MUTUALFUND") and known_type is None:
        return None   # Not an ETF

    # Determine ETF type
    category  = str(yf_info.get("category") or "").lower()
    etf_type  = known_type  # hardcoded first
    if not etf_type:
        for kw, tp in _CATEGORY_MAP.items():
            if kw in category:
                etf_type = tp
                break
        # Gold/Silver/Oil from name
        long_name = str(yf_info.get("longName", "") or "").lower()
        if not etf_type:
            if "gold" in long_name:                 etf_type = "commodity_gold"
            elif "silver" in long_name:             etf_type = "commodity_silver"
            elif "platinum" in long_name:           etf_type = "commodity_platinum"
            elif "palladium" in long_name:          etf_type = "commodity_palladium"
            elif "copper" in long_name:             etf_type = "commodity_copper"
            elif "oil" in long_name or "crude" in long_name: etf_type = "commodity_oil"
            elif "treasury" in long_name or "bond" in long_name: etf_type = "bond_treasury"
            elif "dividend" in long_name:           etf_type = "dividend"
            elif "leveraged" in long_name or "3x" in long_name or "2x" in long_name: etf_type = "leveraged"
            elif "reit" in long_name or "real estate" in long_name: etf_type = "reit_etf"
            else:                                   etf_type = "equity_index_us"

    # Pull extra fields
    expense_raw = yf_info.get("annualReportExpenseRatio") or yf_info.get("totalExpenseRatio") or 0
    expense_ratio = round(float(expense_raw) * 100, 3) if expense_raw and float(expense_raw) < 1 else float(expense_raw or 0)

    return {
        "is_etf":       True,
        "etf_type":     etf_type,
        "etf_label":    ETF_LABELS.get(etf_type, "ETF"),
        "long_name":    yf_info.get("longName", ticker),
        "fund_family":  yf_info.get("fundFamily", ""),
        "category":     yf_info.get("category", ""),
        "expense_ratio": expense_ratio,
        "aum":          yf_info.get("totalAssets") or 0,
        "nav":          yf_info.get("navPrice") or yf_info.get("regularMarketPrice") or 0,
        "yield":        yf_info.get("yield") or yf_info.get("trailingAnnualDividendYield") or 0,
        "holdings_count": yf_info.get("holdings_count") or yf_info.get("totalHoldings") or 0,
    }


# ─── 2. ETF-Specific Data Block ───────────────────────────────────────────────
def build_etf_data_block(
    etf_meta: dict,
    ticker: str,
    real_price: float,
    change_pct: float,
    summary: dict,       # technicals
    fg_data: dict,       # fear & greed
    macro: dict | None = None,
    var_95: float = 0,
    max_dd: float = 0,
) -> str:
    """Build ETF-specific data block for DeepSeek prompt (replaces standard data_block)."""
    t = etf_meta["etf_type"]
    label = etf_meta["etf_label"]
    name  = etf_meta["long_name"]
    fam   = etf_meta.get("fund_family","")
    exp   = etf_meta.get("expense_ratio", 0)
    aum   = etf_meta.get("aum", 0)
    yld   = etf_meta.get("yield", 0) or 0
    nav   = etf_meta.get("nav", real_price)

    def _B(v):
        if not v: return "N/A"
        v = float(v)
        if v >= 1e12: return f"${v/1e12:.2f}T"
        if v >= 1e9:  return f"${v/1e9:.1f}B"
        if v >= 1e6:  return f"${v/1e6:.0f}M"
        return f"${v:,.0f}"

    # Macro context for this ETF type
    _macro_relevant = _get_etf_macro(t, macro or {})

    lines = [
        f"TICKER: {ticker}",
        f"INSTRUMENT TYPE: {label} ⚠️ THIS IS AN ETF — NOT a stock. Traditional stock analysis metrics (EPS, P/E, ROE, Revenue) do NOT apply.",
        f"FUND NAME: {name}",
        f"FUND FAMILY: {fam}",
        f"LIVE PRICE / NAV: ${real_price:.2f} ({change_pct:+.2f}%)",
        f"AUM: {_B(aum)} (larger AUM = better liquidity)",
        f"EXPENSE RATIO: {exp:.2f}% per year (cost drag on returns)",
        f"DISTRIBUTION YIELD: {yld*100:.2f}% {'(income-focused fund)' if yld > 0.01 else '(minimal / not income-focused)'}",
        "",
        "TECHNICALS:",
        f"- Price vs SMA50:  {((real_price - summary.get('sma_50',real_price)) / summary.get('sma_50',real_price) * 100) if summary.get('sma_50') else 0:+.1f}%",
        f"- Price vs SMA200: {((real_price - summary.get('sma_200',real_price)) / summary.get('sma_200',real_price) * 100) if summary.get('sma_200') else 0:+.1f}%",
        f"- RSI: {summary.get('rsi', 0):.1f} ({'Oversold' if summary.get('rsi',50)<30 else 'Overbought' if summary.get('rsi',50)>70 else 'Neutral'})",
        f"- ADX: {summary.get('adx', 0):.1f} ({'Strong trend' if summary.get('adx',0)>25 else 'Weak trend'})",
        f"- MACD: {summary.get('macd',0):.2f} vs Signal {summary.get('macd_signal',0):.2f}",
        f"- SMA50: ${summary.get('sma_50',0):,.2f} | SMA200: ${summary.get('sma_200',0):,.2f}",
        "",
        "RISK METRICS:",
        f"- Daily VaR (95%): {var_95*100:.2f}%",
        f"- Max Historical Drawdown: {max_dd*100:.2f}%",
        "",
        f"MARKET SENTIMENT (Fear & Greed): {fg_data.get('score','N/A')}/100 — {fg_data.get('rating','N/A')}",
        "",
    ]

    # Type-specific context block
    lines += _etf_type_context(t, ticker, real_price, macro or {})

    lines += [
        "",
        "⛔ ANALYSIS RULES FOR THIS ETF:",
        "- Do NOT calculate or mention EPS, Revenue, Net Margin, ROE, ROIC, Debt/Equity",
        "- Do NOT mention 'analyst consensus' or 'price target' — ETFs have no analysts",
        "- Do NOT use SMA200 as a price target",
        "- DO discuss: the fund's objective, what it tracks, key macro drivers, expense ratio impact",
        "- DO use the scenario table from the SCENARIO ANALYSIS section below",
    ]

    return "\n".join(lines)


def _get_etf_macro(etf_type: str, macro: dict) -> dict:
    """Return macro prices most relevant to this ETF type."""
    relevance = {
        "commodity_gold":    ["us10y", "dxy", "gold", "vix"],
        "commodity_oil":     ["oil_brent", "oil_wti", "dxy", "vix"],
        "commodity_silver":  ["gold", "silver", "dxy", "us10y"],
        "commodity_other":   ["wheat", "corn", "sugar", "coffee", "dxy"],
        "bond_treasury":     ["us10y", "dxy", "vix", "sp500"],
        "bond_corporate":    ["us10y", "vix", "sp500", "dxy"],
        "bond_tips":         ["us10y", "dxy", "gold"],
        "equity_index_us":   ["sp500", "vix", "us10y", "dxy"],
        "equity_index_intl": ["em_etf", "dxy", "vix", "sp500"],
        "equity_sector":     ["sp500", "vix", "us10y"],
        "leveraged":         ["vix", "sp500", "us10y"],
        "reit_etf":          ["us10y", "dxy", "vix"],
        "dividend":          ["us10y", "sp500", "vix"],
        "thematic":          ["sp500", "vix", "us10y"],
    }
    keys = relevance.get(etf_type, ["sp500", "vix", "us10y", "dxy"])
    return {k: macro.get(k, {}) for k in keys if k in macro}


def _etf_type_context(etf_type: str, ticker: str, price: float, macro: dict) -> list[str]:
    """Returns type-specific context lines for the data block."""
    base = ticker.split(".")[0].upper()
    lines = []

    if etf_type == "commodity_gold":
        # Real yield = 10Y - CPI
        us10y = (macro.get("us10y") or {}).get("price", 4.3)
        gold_p = (macro.get("gold") or {}).get("price", 0)
        gold_chg = (macro.get("gold") or {}).get("chg_pct", 0)
        dxy   = (macro.get("dxy") or {}).get("chg_pct", 0)
        _spot_line = (f"- Spot Gold Price: ${gold_p:,.2f}/oz ({gold_chg:+.1f}% today)"
                      if gold_p else f"- Futures Price (GC=F): ${price:,.2f} (spot unavailable)")
        lines += [
            "GOLD ETF SPECIFIC ANALYSIS:",
            _spot_line,
            f"- US 10Y Yield: {us10y:.2f}% (KEY HEADWIND when rising — gold pays no yield)",
            f"- DXY Today: {dxy:+.1f}% (USD strength is bearish for gold — inverse relationship)",
            f"- Implied Real Yield Pressure: {'HIGH — headwind for gold' if us10y > 2.5 else 'MODERATE' if us10y > 1.5 else 'LOW — tailwind for gold'}",
            "- Central Bank Demand: Major buying from China, India, EM central banks in 2024-2025",
            "- Fund Mechanics: GLD holds physical gold bars in HSBC London vault. 1 share ≈ 0.0928 troy oz",
            "- Key Drivers: Real yields (inverse), USD (inverse), geopolitical risk (direct), inflation (direct)",
        ]

    elif etf_type == "commodity_oil":
        oil = (macro.get("oil_brent") or {}).get("price", 80)
        oil_chg = (macro.get("oil_brent") or {}).get("chg_pct", 0)
        lines += [
            "OIL ETF SPECIFIC ANALYSIS:",
            f"- Brent Crude: ${oil:.2f}/bbl ({oil_chg:+.1f}% today)",
            f"- Key Risk: CONTANGO DECAY — when futures curve is in contango, rolling futures costs ~1-3%/month",
            "- USO/BNO do NOT hold physical oil — they hold futures contracts and roll monthly",
            "- Long-term holding drag can be severe (USO lost 80%+ during 2020 contango crisis)",
            "- Better for: short-term tactical trades only. For long-term: prefer oil majors (XOM, CVX)",
            "- Key Drivers: OPEC+ decisions, US shale output, China demand, geopolitical premium",
        ]

    elif etf_type in ("bond_treasury", "bond_corporate", "bond_tips"):
        us10y     = (macro.get("us10y") or {}).get("price", 4.3)
        us10y_chg = (macro.get("us10y") or {}).get("chg_pct", 0)
        is_hy     = base in ("HYG", "JNK", "FALN", "ANGL", "HYLB", "USHY")
        is_fallen = base in ("FALN", "ANGL")
        is_ig     = base in ("LQD", "VCIT", "IGIB", "IGSB", "FLOT")

        lines += [
            f"BOND ETF ANALYSIS — {ETF_LABELS.get(etf_type, 'Bond ETF')} ({base})",
            f"⚠ This is a BOND ETF (diversified fund), NOT an individual bond.",
            f"  Apply ETF methodology: focus on duration, expense ratio, index, peer comparison.",
            f"  Do NOT apply seniority/covenant analysis — those apply to individual bonds only.",
            f"",
            f"RATE & DURATION CONTEXT:",
            f"- US 10Y Yield: {us10y:.2f}% ({us10y_chg:+.1f}% today)",
            f"- Rate Sensitivity: {'HIGH DURATION ~17yr — 1% rate rise ≈ -17% NAV' if base=='TLT' else 'MEDIUM DURATION ~7yr — 1% rate rise ≈ -7% NAV' if base in ('IEF','GOVT') else 'LOW DURATION — minimal rate sensitivity' if base in ('SHY','BIL','SGOV') else 'MEDIUM effective duration — check fund factsheet for exact figure'}",
            f"- Credit Risk: {'HIGH YIELD — default rates spike in recession; 2008 HY spreads hit 2000bps' if is_hy else 'INVESTMENT GRADE — minimal default risk; spread widening moderate' if is_ig or etf_type in ('bond_treasury','bond_tips') else 'INVESTMENT GRADE CORPORATE — moderate credit risk'}",
        ]

        # ── HY ETF peer comparison block ──────────────────────────────────────
        if is_hy:
            lines += [
                f"",
                f"PEER COMPARISON — HIGH YIELD BOND ETF UNIVERSE:",
                f"  {'Ticker':<8} {'Index':<30} {'~Duration':<12} {'Approx Yield':<14} {'Expense'}",
                f"  {'──────':<8} {'──────':<30} {'─────────':<12} {'────────────':<14} {'───────'}",
                f"  {'HYG':<8} {'iBoxx $ Liquid HY':<30} {'~8yr':<12} {'~7-8%':<14} {'0.49%'}",
                f"  {'JNK':<8} {'Bloomberg US HY':<30} {'~7yr':<12} {'~7-8%':<14} {'0.40%'}",
                f"  {'FALN':<8} {'iBoxx Fallen Angels':<30} {'~8yr':<12} {'~6-7%':<14} {'0.25%'}",
                f"  {'ANGL':<8} {'ICE US Fallen Angel':<30} {'~7yr':<12} {'~6-7%':<14} {'0.35%'}",
                f"  {'HYLB':<8} {'Xtrackers US HY':<30} {'~6yr':<12} {'~7%':<14} {'0.15%'}",
                f"  Note: {base} is one of these — discuss its positioning vs peers above.",
                f"  Key differentiators: index methodology, quality tilt, expense ratio, AUM size",
                f"  HYG/JNK = broadest; FALN/ANGL = higher quality (recently downgraded from IG)",
            ]
        elif is_ig:
            lines += [
                f"",
                f"PEER COMPARISON — INVESTMENT GRADE CORPORATE BOND ETF UNIVERSE:",
                f"  {'Ticker':<8} {'Index':<30} {'~Duration':<12} {'Approx Yield':<14} {'Expense'}",
                f"  {'──────':<8} {'──────':<30} {'─────────':<12} {'────────────':<14} {'───────'}",
                f"  {'LQD':<8} {'iBoxx $ IG Corporate':<30} {'~9yr':<12} {'~5-6%':<14} {'0.14%'}",
                f"  {'VCIT':<8} {'Bloomberg IG Corp':<30} {'~7yr':<12} {'~5-6%':<14} {'0.04%'}",
                f"  {'IGIB':<8} {'iBoxx Int. IG':<30} {'~6yr':<12} {'~5%':<14} {'0.06%'}",
                f"  {'IGSB':<8} {'iBoxx Short-Term IG':<30} {'~3yr':<12} {'~5%':<14} {'0.06%'}",
                f"  Note: {base} is one of these — discuss its positioning, duration, and cost vs peers.",
            ]

        lines += [
            f"",
            f"KEY DRIVERS FOR BOND ETFs: Fed policy path, inflation trajectory, credit spreads,",
            f"  default cycle, economic growth — NOT individual issuer financials.",
        ]

    elif etf_type == "equity_index_us":
        sp5  = (macro.get("sp500") or {}).get("price", 5500)
        sp5c = (macro.get("sp500") or {}).get("chg_pct", 0)
        vix  = (macro.get("vix") or {}).get("price", 18)
        lines += [
            f"US EQUITY INDEX ETF — MARKET CONTEXT:",
            f"- S&P 500: {sp5:,.0f} pts ({sp5c:+.1f}% today)",
            f"- VIX: {vix:.1f} ({'ELEVATED FEAR' if vix>25 else 'NORMAL' if vix>15 else 'COMPLACENT'})",
            f"- This fund tracks a broad market index — analysis should focus on macro and valuation of the INDEX (not individual stocks)",
            f"- Key Drivers: Corporate earnings growth, Fed policy, economic cycle, valuations (S&P P/E)",
        ]

    elif etf_type == "equity_index_intl":
        em   = (macro.get("em_etf") or {}).get("chg_pct", 0)
        dxy  = (macro.get("dxy") or {}).get("price", 103)
        lines += [
            "INTERNATIONAL/EM EQUITY ETF — MARKET CONTEXT:",
            f"- EM ETF (EEM benchmark): {em:+.1f}% today",
            f"- DXY: {dxy:.1f} ({'STRONG USD — headwind for EM returns' if dxy>105 else 'MODERATE USD' if dxy>100 else 'WEAK USD — tailwind for EM'})",
            "- Currency Risk: Returns affected by USD vs local currency moves",
            "- Key Drivers: USD direction, China growth, commodity prices, EM central bank policy",
        ]

    elif etf_type == "equity_sector":
        sp5c = (macro.get("sp500") or {}).get("chg_pct", 0)
        lines += [
            f"SECTOR ETF — CONCENTRATION ANALYSIS:",
            f"- S&P 500 today: {sp5c:+.1f}%",
            f"- Sector ETFs are CONCENTRATED bets — higher risk/reward than broad index",
            f"- Top 5 holdings often represent 30-50% of fund weight",
            "- Key Drivers: Sector-specific catalysts, relative rotation, earnings cycle",
        ]

    elif etf_type == "leveraged":
        vix = (macro.get("vix") or {}).get("price", 18)
        lines += [
            "⚠️ LEVERAGED/INVERSE ETF — CRITICAL WARNINGS:",
            f"- VIX: {vix:.1f} — {'HIGH VOLATILITY = SEVERE DECAY RISK' if vix>25 else 'Moderate volatility environment'}",
            "- VOLATILITY DECAY: In sideways/choppy markets, leveraged ETFs LOSE money even if underlying is flat",
            "- Example: If underlying -10% then +11%, leveraged 3x: -30% then +33% → net: -9.9% (not flat!)",
            "- Holding Period: Designed for SINGLE-DAY use only. NOT suitable for buy-and-hold",
            "- Max Hold Period (recommended): 1-5 trading days in strong trending market",
            "- NEVER hold during earnings, Fed meetings, or high-VIX environments",
        ]

    elif etf_type == "reit_etf":
        us10y = (macro.get("us10y") or {}).get("price", 4.3)
        us10y_chg = (macro.get("us10y") or {}).get("chg_pct", 0)
        lines += [
            "REIT ETF — REAL ESTATE ANALYSIS:",
            f"- US 10Y Yield: {us10y:.2f}% ({us10y_chg:+.1f}%) — PRIMARY driver of REIT valuations",
            f"- Rate Environment: {'HEADWIND — high rates raise cap rates and compress REIT valuations' if us10y > 4.5 else 'NEUTRAL' if us10y > 3.5 else 'TAILWIND — falling rates boost REIT NAV'}",
            "- Key Drivers: Interest rates, occupancy rates, rental growth, commercial real estate cycle",
            "- Provides: Real estate exposure without direct property ownership + dividend income",
        ]

    elif etf_type == "dividend":
        us10y = (macro.get("us10y") or {}).get("price", 4.3)
        lines += [
            "DIVIDEND INCOME ETF — YIELD ANALYSIS:",
            f"- US 10Y Yield: {us10y:.2f}% — competition for income investors",
            f"- Yield Attractiveness: {'Bonds competitive — may reduce dividend ETF appeal' if us10y > 4.5 else 'Dividend ETF yield likely competitive vs bonds'}",
            "- Key Drivers: Dividend sustainability, payout ratios, sector composition, rate environment",
            "- Strategy: Focus on dividend growth and yield vs 10Y bond alternative",
        ]

    elif etf_type == "thematic":
        sp5c = (macro.get("sp500") or {}).get("chg_pct", 0)
        vix  = (macro.get("vix") or {}).get("price", 18)
        lines += [
            "THEMATIC/ACTIVE ETF — THEME ANALYSIS:",
            f"- Broad market: {sp5c:+.1f}% | VIX: {vix:.1f}",
            "- Thematic ETFs carry HIGH CONCENTRATION RISK in specific trends",
            "- Performance is BINARY — theme either accelerates or collapses",
            "- Key Drivers: Theme narrative, sector flows, innovation cycle, risk appetite",
            "- ARKK-style active ETFs also carry manager risk and high turnover costs",
        ]

    return lines


# ─── 3. ETF-Specific Scenarios ────────────────────────────────────────────────
def build_etf_scenarios(etf_type: str, price: float, macro: dict | None = None) -> str:
    """Returns markdown scenario table appropriate for this ETF type."""
    macro = macro or {}

    scenarios: list[tuple] = []  # (name, impact_pct, rationale, hedge)

    if etf_type == "commodity_gold":
        us10y = (macro.get("us10y") or {}).get("price", 4.3)
        scenarios = [
            ("Fed Rate Cut Cycle Begins",       +22.0, "Real yields fall → gold re-rates higher; DXY weakens",        "Hold or add"),
            ("Geopolitical Crisis (Middle East)",+18.0, "Safe-haven demand surge; central bank emergency buying",       "Hold — this is the hedge"),
            ("Fed Resumes Hikes (+1.5%)",        -18.0, f"Real yields spike to ~{us10y+1.5:.1f}% → significant outflows from gold", "Reduce; buy short-dated T-bills"),
            ("USD Surges (DXY +8%)",             -12.0, "Gold priced in USD; strong dollar crushes commodity prices",   "USD-hedged gold or energy stocks"),
            ("Deflation / Recession",            -8.0,  "Risk-off initially positive, but deflation raises real yields → mixed",    "Long Treasuries + small gold allocation"),
        ]

    elif etf_type == "commodity_oil":
        oil = (macro.get("oil_brent") or {}).get("price", 80)
        scenarios = [
            ("OPEC+ Surprise Cut (2Mb/d)",      +25.0, "Supply shock → immediate price spike; futures curve steepens",  "Oil majors (XOM, CVX) preferred over USO"),
            ("Iran/Hormuz Conflict",             +35.0, "20% of global oil through Hormuz; tanker insurance premium explodes", "Physical oil stocks, not futures ETF"),
            ("China Demand Collapse",            -25.0, "China = 15Mb/d demand; economic slowdown crushes oil",           "Short energy, long airlines"),
            ("US Shale Surge (+3Mb/d)",          -18.0, "Breakeven $45-55/bbl; US output record breaks OPEC floor",      "Exit; contango decay will compound losses"),
            ("Global Recession",                 -30.0, "Demand destruction; IEA demand growth estimate cut to zero",     "Cash, bonds, defensive equities"),
        ]

    elif etf_type in ("bond_treasury", "bond_tips"):
        dur_label = "17yr" if "TLT" in str(macro) else "7yr"
        scenarios = [
            ("Fed Cuts Rates 3x (-0.75%)",       +12.0, "Bond prices rise inversely to yields; duration amplifies gains",   "Hold; add on yield spikes"),
            ("Recession / Flight to Safety",      +18.0, "Risk-off: treasuries rally as equities sell off sharply",          "Ideal safe-haven allocation"),
            ("Inflation Re-Acceleration (+1%)",   -15.0, "Stagflation scenario; Fed forced to stay high → bond prices fall", "Switch to TIPS or short-duration"),
            ("Fed Hikes Resumption (+1%)",        -17.0, f"~{dur_label} duration → ~-{int(dur_label[:-2])}% price impact per 1% yield rise", "Cash, floating rate bonds (FLOT)"),
            ("Credit Downgrade (US AAA loss)",    -5.0,  "Confidence shock; spreads widen; short-term disruption",           "Gold, international bonds"),
        ]

    elif etf_type == "bond_corporate":
        scenarios = [
            ("Economic Soft Landing",            +8.0,  "Spreads compress; investment grade bonds rally with equities",     "Hold; add on weakness"),
            ("Recession + Credit Crunch",        -20.0, "Default rates spike; HY bonds can lose 25-35% in severe credit events", "Switch to treasuries or cash"),
            ("Fed Rate Cut Cycle",               +10.0, "Lower rates reduce refinancing risk; spreads tighten",              "Hold or overweight"),
            ("Inflation Surge",                  -12.0, "Nominal bonds lose real value; fixed coupon eroded by inflation",   "TIPS or floating rate bonds"),
        ]

    elif etf_type == "equity_index_us":
        sp5 = (macro.get("sp500") or {}).get("price", 5500)
        vix  = (macro.get("vix") or {}).get("price", 18)
        scenarios = [
            ("Bull Case: AI-Driven Earnings Boom",+20.0, "S&P EPS growth accelerates; multiple expansion continues",         "Hold; momentum still strong"),
            ("Soft Landing: Steady Growth",       +10.0, "Fed cuts gradually; economy avoids recession; earnings stable",    "Core long-term position"),
            ("Valuation Correction (-15%)",       -15.0, "S&P P/E too high vs historical; rotation from growth to value",    "Defensive value, dividend ETFs"),
            ("Recession + Earnings Decline",      -28.0, "EPS falls 20-25%; historical bear market average = -35%",          "Cash, bonds, gold, short volatility"),
            (f"VIX Spike to 45+ (2020/2022 repeat)", -35.0, "Panic selling; margin calls; liquidity freeze",                 "Cash, TLT, GLD"),
        ]

    elif etf_type == "equity_index_intl":
        dxy = (macro.get("dxy") or {}).get("price", 103)
        scenarios = [
            ("China Stimulus + USD Weakens",     +20.0, "EM rally when USD falls; China re-opening premium",                "Hold or overweight EM"),
            ("USD Stays Strong (DXY>110)",        -18.0, "Currency loss compounds stock losses for USD investors",           "Hedge currency or shift to US stocks"),
            ("EM Debt Crisis",                   -30.0, "Rising USD + high rates = EM sovereign defaults; capital flight",  "US Treasuries, DXY long"),
            ("Global Trade War Escalation",      -22.0, "Export-dependent EMs crushed; supply chains disrupt",              "Domestic US large-cap"),
        ]

    elif etf_type == "equity_sector":
        scenarios = [
            ("Sector Outperforms (Bull)",        +25.0, "Sector-specific tailwinds (AI spending, rate cuts, demand cycle)",  "Hold or add"),
            ("Market-Wide Correction",           -20.0, "Concentrated sector ETF amplifies broad market drawdown",           "Diversify; reduce concentration"),
            ("Sector Rotation Out",              -25.0, "Fund flows exit the sector; crowded trade unwinds",                "Rotate to defensive sectors"),
            ("Regulatory Headwind",              -30.0, "Government intervention (antitrust, rate regulation, tariffs)",      "Reduce to benchmark weight"),
        ]

    elif etf_type == "leveraged":
        vix = (macro.get("vix") or {}).get("price", 18)
        scenarios = [
            ("Strong Trending Market (ADX>30)",  +60.0, "Leverage works perfectly in unidirectional moves",                  "Ride trend; tight stop"),
            (f"Choppy Market (VIX {vix:.0f})",  -35.0, "Volatility decay destroys value daily; 3x amplifies random noise",  "CLOSE POSITION immediately"),
            ("Market Crash (-20%)",              -55.0, "3x leverage: -60% for bullish ETF; near-total loss",                "Emergency stop at -15% from entry"),
            ("Overnight Gap (earnings/Fed)",     -40.0, "Cannot exit; leveraged losses amplified by overnight gap",          "Never hold through events"),
        ]

    elif etf_type == "reit_etf":
        us10y = (macro.get("us10y") or {}).get("price", 4.3)
        scenarios = [
            ("Rate Cut Cycle (-2% over 2yr)",   +30.0, "Cap rate compression → REIT NAV expansion + dividend re-rating",    "Add at first cut signal"),
            ("Commercial RE Recovery",          +15.0, "Occupancy recovery post-COVID; rent growth exceeds inflation",       "Core position"),
            (f"Rates Stay High ({us10y:.1f}%+)", -15.0, "Cap rate headwind; refinancing at higher rates squeezes FFO",       "Wait for yield curve inversion"),
            ("Office Market Structural Decline",-25.0, "Work-from-home permanent shift; office REIT impairments",            "Focus on industrial/residential REITs"),
            ("Credit Crunch / Recession",       -35.0, "REIT dividend cuts; forced asset sales at distressed prices",        "Treasuries + cash"),
        ]

    elif etf_type == "dividend":
        us10y = (macro.get("us10y") or {}).get("price", 4.3)
        scenarios = [
            ("Rate Cuts + Dividend Growth",     +15.0, "Lower bond competition; dividend growers attract income seekers",    "Core income position"),
            ("Recession / Dividend Cuts",       -20.0, "Companies cut dividends to preserve cash; ETF yield falls",         "High-quality dividend growth (SCHD > DVY)"),
            (f"Bond Yields Rise Above 5.5%",    -12.0, "T-bills and CDs become more attractive than dividend stocks",       "Shift to short-term treasuries"),
            ("Value Rotation (from growth)",    +12.0, "Dividend stocks re-rate as growth premium compressed",              "Hold; add on weakness"),
        ]

    elif etf_type == "thematic":
        vix = (macro.get("vix") or {}).get("price", 18)
        scenarios = [
            ("Theme Acceleration (AI/Robotics)", +45.0, "Narrative momentum; fund inflows accelerate; top holdings surge",   "Ride momentum; trail stop"),
            ("Risk-Off / De-Rating",            -40.0, "Speculative themes sell off first and hardest in risk-off",          "Exit immediately on VIX>25"),
            ("Theme Fails (Regulation/Reality)",-60.0, "Structural disillusionment; redemptions force selling at any price", "Position sizing: max 3% of portfolio"),
            (f"Broad Market Correction (-20%)", -35.0, "High-beta thematic ETFs lose 1.5-2x the market in drawdowns",        "Hedge with inverse ETF (10% of position)"),
        ]

    else:  # commodity_other or unknown
        scenarios = [
            ("Supply Shock (drought/conflict)",  +25.0, "Physical shortage drives spot prices higher",                       "Hold as inflation hedge"),
            ("Demand Collapse (recession)",      -20.0, "Industrial/food demand falls with GDP",                             "Exit; shift to defensives"),
            ("USD Surge",                        -15.0, "Commodities priced in USD; strong dollar suppresses prices",        "USD-denominated assets"),
            ("Tech/Substitute Disruption",       -30.0, "New supply source or substitute material eliminates shortage premium", "Reduce to minimal allocation"),
        ]

    if not scenarios:
        return ""

    lines = ["SCENARIO ANALYSIS (ETF-SPECIFIC — NOT beta-adjusted stock scenarios):"]
    lines.append("| Scenario | Est. Price Impact | Rationale | Suggested Action |")
    lines.append("|----------|-------------------|-----------|-----------------|")
    for name, pct, reason, hedge in scenarios:
        est_price = price * (1 + pct/100)
        direction = "+" if pct > 0 else ""
        lines.append(f"| {name} | {direction}{pct:.1f}% → ${est_price:.2f} | {reason} | {hedge} |")

    return "\n".join(lines)


# ─── 4. ETF Scorecard ─────────────────────────────────────────────────────────
def calculate_etf_score(etf_meta: dict, summary: dict, fg_data: dict,
                        var_95: float = 0, macro: dict | None = None) -> dict:
    """
    ETF-specific scoring. Returns dict matching scorecard.py output format.
    Factors and weights differ by ETF type.
    """
    macro = macro or {}
    t = etf_meta["etf_type"]
    factors: dict[str, tuple[int, int]] = {}   # factor: (earned, max)

    rsi = summary.get("rsi", 50) or 50
    adx = summary.get("adx", 0) or 0
    sma50  = summary.get("sma_50", 0) or 0
    sma200 = summary.get("sma_200", 0) or 0
    price  = summary.get("price", 0) or 0
    fg     = fg_data.get("score", 50) or 50
    exp    = etf_meta.get("expense_ratio", 0) or 0
    aum    = etf_meta.get("aum", 0) or 0
    yld    = (etf_meta.get("yield") or 0) * 100  # as %
    macd   = summary.get("macd", 0) or 0
    macd_s = summary.get("macd_signal", 0) or 0

    # ── Technical Momentum (universal, 30 pts for ETFs) ─────────────────────
    t_score = 0
    if sma200 and price:
        pct_vs200 = (price - sma200) / sma200 * 100
        if pct_vs200 > 5:   t_score += 8
        elif pct_vs200 > 0: t_score += 6
        elif pct_vs200 > -5:t_score += 3
        else:               t_score += 0
    if 40 <= rsi <= 60:     t_score += 8
    elif 30 < rsi < 40 or 60 < rsi < 70: t_score += 5
    elif rsi <= 30:         t_score += 4   # oversold = contrarian positive
    elif rsi >= 70:         t_score += 2   # overbought = caution
    if adx > 25:            t_score += 6
    elif adx > 18:          t_score += 4
    else:                   t_score += 1
    if macd > macd_s:       t_score += 4
    else:                   t_score += 1
    if t == "leveraged" and adx < 20:
        t_score = min(t_score, 5)   # leveraged ETF in no-trend = punish hard
    factors["Technical Momentum"] = (min(t_score, 30), 30)

    # ── Risk / Volatility (20 pts) ───────────────────────────────────────────
    r_score = 15   # start neutral
    if var_95:
        if abs(var_95) > 0.04:  r_score -= 8
        elif abs(var_95) > 0.02:r_score -= 3
    if t == "leveraged":  r_score = max(r_score - 8, 2)
    if t in ("commodity_oil", "thematic"): r_score = max(r_score - 4, 4)
    if aum > 10e9:        r_score = min(r_score + 3, 20)   # large AUM = better liquidity
    factors["Risk / Volatility"] = (max(0, min(r_score, 20)), 20)

    # ── Macro Alignment (20 pts — different per type) ────────────────────────
    m_score = 10  # neutral default
    us10y   = (macro.get("us10y") or {}).get("price", 4.3)
    dxy_chg = (macro.get("dxy") or {}).get("chg_pct", 0)
    vix     = (macro.get("vix") or {}).get("price", 18)
    oil_chg = (macro.get("oil_brent") or {}).get("chg_pct", 0)
    gold_chg = (macro.get("gold") or {}).get("chg_pct", 0)

    if t == "commodity_gold":
        # Good: low real yields, weak dollar, high fear
        if us10y < 2.0:     m_score += 6
        elif us10y < 3.5:   m_score += 2
        else:               m_score -= 4
        if dxy_chg < -0.5:  m_score += 4
        elif dxy_chg > 0.5: m_score -= 3
        if vix > 25:        m_score += 3   # fear = gold demand

    elif t == "commodity_oil":
        if oil_chg > 1:     m_score += 6
        elif oil_chg > 0:   m_score += 2
        else:               m_score -= 4

    elif t in ("bond_treasury", "bond_tips"):
        # Good: falling rates, recession fear, high VIX
        if us10y < 3.5:     m_score += 6
        elif us10y > 5.0:   m_score -= 4
        if vix > 25:        m_score += 4

    elif t == "equity_index_us":
        # Good: moderate rates, low VIX, positive momentum
        sp5_chg = (macro.get("sp500") or {}).get("chg_pct", 0)
        if sp5_chg > 0.5:   m_score += 5
        elif sp5_chg < -1:  m_score -= 4
        if 15 <= vix <= 20: m_score += 3
        elif vix > 30:      m_score -= 5

    elif t in ("reit_etf", "dividend"):
        if us10y > 5.0:     m_score -= 6
        elif us10y < 3.5:   m_score += 5

    elif t == "leveraged":
        if vix > 25:        m_score = 2   # leveraged ETF in high VIX = terrible
        elif adx > 25:      m_score += 6  # trending market = good for leveraged

    factors["Macro Alignment"] = (max(0, min(m_score, 20)), 20)

    # ── Cost Efficiency (15 pts) ─────────────────────────────────────────────
    c_score = 10
    if exp <= 0.10:     c_score = 15   # ultra-cheap (Vanguard/iShares core)
    elif exp <= 0.20:   c_score = 13
    elif exp <= 0.40:   c_score = 10
    elif exp <= 0.75:   c_score = 7
    elif exp <= 1.5:    c_score = 4
    else:               c_score = 1    # >1.5% = high cost drag
    factors["Cost Efficiency"] = (c_score, 15)

    # ── Liquidity / AUM (15 pts) ─────────────────────────────────────────────
    # Commodity futures (GC=F, CL=F, SI=F) have no AUM — they are among the most
    # liquid markets in the world. Score them at full marks; AUM metric doesn't apply.
    _commodity_types = ("commodity_gold", "commodity_silver", "commodity_oil",
                        "commodity_platinum", "commodity_palladium", "commodity_copper",
                        "commodity_other")
    if t in _commodity_types and aum == 0:
        l_score = 15   # extremely deep market, 24h global trading
    elif aum >= 50e9:   l_score = 15
    elif aum >= 10e9:   l_score = 12
    elif aum >= 1e9:    l_score = 8
    elif aum >= 100e6:  l_score = 5
    else:               l_score = 2
    factors["Liquidity / AUM"] = (l_score, 15)

    # Total
    total = sum(v[0] for v in factors.values())
    max_t = sum(v[1] for v in factors.values())
    score = round(total / max_t * 100) if max_t else 50

    # Hard caps
    if t == "leveraged" and adx < 20:  score = min(score, 45)
    if t == "commodity_oil":           score = min(score, 72)  # contango risk cap

    # Verdict
    if score >= 70:
        verdict, emo = "BUY", "🟢"
    elif score >= 55:
        verdict, emo = "HOLD", "🟡"
    elif score >= 40:
        verdict, emo = "REDUCE", "🟠"
    else:
        verdict, emo = "AVOID", "🔴"

    # Leveraged always REDUCE unless very strong trend
    if t == "leveraged" and adx < 25:
        verdict, emo = "REDUCE", "🟠"

    return {
        "score":   score,
        "verdict": verdict,
        "emoji":   emo,
        "factors": factors,
        "etf_type": t,
        "etf_label": ETF_LABELS.get(t,"ETF"),
    }


# ─── 5. ETF Scorecard Markdown ────────────────────────────────────────────────
def build_etf_scorecard_md(
    ticker: str,
    etf_meta: dict,
    price: float,
    score_result: dict,
    summary: dict,
    resolved_ticker: str = "",
) -> str:
    """Render ETF scorecard markdown (replaces _build_scorecard_md for ETFs)."""
    score   = score_result["score"]
    verdict = score_result["verdict"]
    emo     = score_result["emoji"]
    t       = etf_meta["etf_type"]
    label   = etf_meta["etf_label"]
    raw_name = etf_meta["long_name"]
    # For commodity futures (GC=F, CL=F, SI=F) the long_name equals the ticker symbol
    # which looks bad in the scorecard. Use a human-readable name instead.
    _commodity_name_map = {
        "commodity_gold":      "Gold Futures",
        "commodity_silver":    "Silver Futures",
        "commodity_oil":       "Crude Oil Futures",
        "commodity_platinum":  "Platinum Futures",
        "commodity_palladium": "Palladium Futures",
        "commodity_copper":    "Copper Futures",
        "commodity_other":     "Commodity Futures",
    }
    _rt = (resolved_ticker or ticker).upper()
    _is_futures = _rt.endswith("=F") or _rt in ("GC=F", "SI=F", "CL=F", "NG=F", "PL=F", "PA=F", "HG=F")
    name = _commodity_name_map.get(t, raw_name) if (_is_futures or not raw_name or raw_name == _rt) else raw_name
    factors = score_result["factors"]
    exp     = etf_meta.get("expense_ratio", 0)
    aum_raw = etf_meta.get("aum", 0) or 0
    aum_str = f"${aum_raw/1e9:.1f}B" if aum_raw >= 1e9 else f"${aum_raw/1e6:.0f}M" if aum_raw >= 1e6 else "N/A"

    # Conviction label
    if score >= 75:  conviction = "High"
    elif score >= 60: conviction = "Medium"
    else:            conviction = "Low"

    # Special warnings
    warnings = []
    if t == "leveraged":
        warnings.append("⛔ **LEVERAGED ETF**: Designed for intraday use only. Volatility decay destroys value in choppy markets. Not suitable for buy-and-hold.")
    if t == "commodity_oil":
        warnings.append("⚠️ **CONTANGO RISK**: This oil ETF rolls futures monthly. In contango markets, the roll cost can erode 20-40% per year.")
    if t == "thematic" and score < 55:
        warnings.append("⚠️ **THEMATIC RISK**: High-concentration thematic funds can lose 50%+ when the narrative breaks.")

    # ETF-specific advice
    hold_advice = {
        "commodity_gold":    "Best held as 5-10% portfolio hedge. Add on real yield drops or geopolitical escalation.",
        "commodity_oil":     "Short-term tactical only (days to weeks). Avoid long-term due to contango decay.",
        "bond_treasury":     "Core defensive allocation. Increase in recession/rate-cut environments.",
        "bond_corporate":    "Income allocation. Monitor credit spreads and default rates closely.",
        "bond_tips":         "Inflation hedge. Outperforms when real yields fall or CPI surprises higher.",
        "equity_index_us":   "Core long-term holding. Dollar-cost average during drawdowns.",
        "equity_index_intl": "Diversification allocation 10-20% of equities. Reduces US concentration.",
        "equity_sector":     "Tactical overweight. Size max 10-15% of equity portfolio.",
        "leveraged":         "SPECULATIVE ONLY. Max 2-3% of portfolio. Set hard stops.",
        "reit_etf":          "Income + real estate exposure. Best when rates peak or fall.",
        "dividend":          "Income core position. Reinvest dividends for compounding.",
        "thematic":          "Satellite position max 3-5%. High risk/reward. Strict position sizing.",
        "commodity_other":   "Commodity diversifier or inflation hedge. Small allocation (2-5%).",
    }.get(t, "Use as part of a diversified portfolio.")

    # Progress bars
    def _bar(earned, max_pts, width=10):
        filled = round((earned / max_pts) * width) if max_pts else 0
        color  = "🟢" if earned/max_pts >= 0.7 else "🟡" if earned/max_pts >= 0.5 else "🔴"
        return f"{color} `{'█'*filled}{'░'*(width-filled)}`"

    # Canonical evidence label (replaces Low/Medium/High conviction)
    try:
        from core.services.decision_policy import canonical_evidence as _ce
        _etf_evidence = _ce(conviction)
    except Exception:
        _etf_evidence = conviction

    md = f"""
---
## 🎯 EisaX ETF Score Card
**{ticker}** — *{name}* | **{verdict} {emo}** | Evidence: **{_etf_evidence}** | EisaX Score: **{score}/100**

> 📋 **Fund Type**: {label} | **AUM**: {aum_str} | **Expense Ratio**: {exp:.2f}%/yr

"""
    if warnings:
        for w in warnings:
            md += f"> {w}\n"
        md += "\n"

    md += "**Factor Analysis** *(ETF-specific weights)*:\n"
    md += "| Factor | Score | Bar |\n|--------|-------|-----|\n"
    for fname, (earned, maxp) in factors.items():
        pct = round(earned / maxp * 100) if maxp else 0
        md += f"| {fname} | {pct}% | {_bar(earned, maxp)} |\n"

    md += f"\n**Overall: `{'█' * round(score/10)}{'░' * (10 - round(score/10))}` {score}/100**\n"
    md += f"\n> **EisaX Strategy Note**: {hold_advice}\n"
    md += "\n> *EisaX ETF Score | Abu Dhabi*\n"

    return md


# ─── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for test_ticker in ["GLD", "TLT", "SPY", "TQQQ", "HYG", "SCHD", "EEM", "XLK"]:
        meta = detect_etf(test_ticker)
        if meta:
            print(f"\n{test_ticker}: {meta['etf_type']} — {meta['etf_label']}")
            print(f"  Name: {meta['long_name']} | AUM: ${meta['aum']/1e9:.1f}B | ER: {meta['expense_ratio']:.2f}%")
        else:
            print(f"{test_ticker}: NOT detected as ETF")
