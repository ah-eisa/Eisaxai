"""
CONCEPTS_LIBRARY — Verified Investment Frameworks for EisaX CIO Memos.

Each asset type gets a tailored set of frameworks.
DeepSeek receives ONLY the relevant subset — not the full library.
"""

CONCEPTS_LIBRARY: dict[str, list[dict]] = {

    # ── US Equity ──────────────────────────────────────────────────────────
    "us_equity": [
        {
            "title": "Forward P/E Valuation",
            "formula": "Price / Forward EPS",
            "thresholds": (
                "Cheap: <15x | Fair: 15-22x | Stretched: 22-30x | Expensive: >30x\n"
                "  → Context-dependent: growth stocks (>20% EPS CAGR) warrant 25-35x; "
                "mature/cyclicals warrant 12-18x."
            ),
            "application": (
                "Always anchor the Section 1 valuation sentence to the forward P/E "
                "vs sector median. State the premium/discount explicitly."
            ),
        },
        {
            "title": "EV/EBITDA — Capital-Structure-Neutral Valuation",
            "formula": "Enterprise Value / EBITDA (LTM or NTM)",
            "thresholds": (
                "Low: <8x | Fair: 8-14x | Rich: >14x\n"
                "  → Preferred for capital-intensive or highly-levered companies "
                "where P/E distorts (airlines, telecoms, industrials)."
            ),
            "application": (
                "Use when net debt > 1× EBITDA or when D&A is material. "
                "Cite alongside P/E in Section 2."
            ),
        },
        {
            "title": "Gross Margin Quality",
            "formula": "(Revenue − COGS) / Revenue",
            "thresholds": (
                "SaaS/Software: >70% = strong | Semiconductor: >50% = strong | "
                "Retail: >30% = strong | Industrials: >25% = strong"
            ),
            "application": (
                "Flag if gross margin is contracting YoY by >200 bps — "
                "signals pricing pressure or input cost creep. Mention in Section 2."
            ),
        },
        {
            "title": "Momentum — RSI & MACD",
            "formula": (
                "RSI = 100 − [100 / (1 + RS)] where RS = avg gain / avg loss (14-period)\n"
                "MACD = EMA(12) − EMA(26); Signal = EMA(9) of MACD"
            ),
            "thresholds": (
                "RSI: Oversold <30 | Neutral 30-70 | Overbought >70\n"
                "MACD: Bullish = MACD > Signal | Bearish = MACD < Signal"
            ),
            "application": (
                "Use MACD/RSI for directional bias in Section 3. "
                "⚠️ Do NOT use as standalone BUY/SELL — confirm with ADX trend strength."
            ),
        },
        {
            "title": "Trend Strength — ADX",
            "formula": "ADX = smoothed average of |+DI − −DI| / (+DI + −DI)",
            "thresholds": (
                "Weak/no trend: <20 | Developing trend: 20-25 | "
                "Strong trend: 25-40 | Very strong: >40"
            ),
            "application": (
                "ADX <25 → momentum signals (RSI/MACD) have lower conviction — "
                "price may be range-bound. State this explicitly in Section 3."
            ),
        },
        {
            "title": "Peer Comparison — Relative Valuation",
            "formula": "Premium/Discount = (Subject P/E − Peer P/E) / Peer P/E × 100",
            "thresholds": (
                ">20% premium requires justification (superior growth, moat, margin).\n"
                ">20% discount may signal value OR structural decline — investigate."
            ),
            "application": (
                "Section 6: state the % premium or discount vs each peer. "
                "Justify premium with a specific differentiator (growth rate, margin, moat)."
            ),
        },
    ],

    # ── GCC / MENA Equity ─────────────────────────────────────────────────
    "gcc_equity": [
        {
            "title": "Forward P/E Valuation",
            "formula": "Price / Forward EPS",
            "thresholds": (
                "GCC context — Cheap: <10x | Fair: 10-16x | Stretched: >16x\n"
                "  → GCC multiples are structurally lower than US peers due to "
                "lower liquidity premium and higher political risk discount."
            ),
            "application": (
                "State P/E vs GCC sector median (not S&P 500 median). "
                "Dividend yield is often more relevant than P/E for GCC investors."
            ),
        },
        {
            "title": "Dividend Yield Context (GCC)",
            "formula": "Annual DPS / Current Price",
            "thresholds": (
                "GCC investor benchmark: >4% = attractive | 2-4% = acceptable | "
                "<2% = below hurdle for income-seeking GCC portfolios.\n"
                "  → Compare to UAE risk-free rate (currently ~4.5-5% on T-bills/CBUAE)."
            ),
            "application": (
                "For GCC stocks, lead Section 2 with yield vs risk-free rate spread. "
                "A stock yielding less than T-bills needs a strong capital-gain case."
            ),
        },
        {
            "title": "Oil Sensitivity",
            "formula": (
                "Implied Beta to Brent: ΔStock% / ΔBrent% (rolling 60-day)\n"
                "Direct: E&P, refiners, petrochemicals | "
                "Indirect: banks (credit cycle), real estate (fiscal spending)"
            ),
            "thresholds": (
                "High sensitivity: β_oil > 0.6 | Moderate: 0.3-0.6 | Low: <0.3\n"
                "  → Aramco (2222.SR) β_oil ≈ 0.85 as reference."
            ),
            "application": (
                "State oil sensitivity in Section 4 risks if Brent is a driver. "
                "For UAE/KSA banks: note that oil >$75 = fiscal surplus = credit expansion."
            ),
        },
        {
            "title": "Momentum — RSI & MACD",
            "formula": (
                "RSI = 100 − [100 / (1 + RS)] where RS = avg gain / avg loss (14-period)\n"
                "MACD = EMA(12) − EMA(26); Signal = EMA(9) of MACD"
            ),
            "thresholds": (
                "RSI: Oversold <30 | Neutral 30-70 | Overbought >70\n"
                "MACD: Bullish = MACD > Signal | Bearish = MACD < Signal"
            ),
            "application": (
                "GCC markets are less liquid → RSI extremes (<25 or >75) are more "
                "meaningful as reversal signals than in US markets. Note this in Section 3."
            ),
        },
        {
            "title": "Peer Comparison — GCC Regional",
            "formula": "Premium/Discount = (Subject P/E − Peer P/E) / Peer P/E × 100",
            "thresholds": (
                "Cross-exchange peers valid for GCC investors (UAE/SA/KW/QA treated as "
                "single home region). Compare on P/E, yield, and ROE — not revenue growth alone."
            ),
            "application": (
                "Section 6: name the closest GCC peer in the same sub-sector. "
                "For Islamic-compliant stocks, note Shariah screening status."
            ),
        },
    ],

    # ── ETF ───────────────────────────────────────────────────────────────
    "etf": [
        {
            "title": "Expense Ratio Drag",
            "formula": "Annual return cost = NAV × Expense Ratio",
            "thresholds": (
                "Passive equity ETF: <0.10% = excellent | 0.10-0.50% = acceptable | "
                ">0.50% = expensive unless active/niche.\n"
                "Bond ETF: <0.15% = excellent | Active ETF: <0.75% = acceptable."
            ),
            "application": (
                "Always cite the expense ratio in Section 2 and quantify the annual "
                "drag on a $10,000 position. This is a permanent headwind vs the index."
            ),
        },
        {
            "title": "Tracking Error",
            "formula": "Std Dev of (ETF daily return − Index daily return) annualised",
            "thresholds": (
                "Passive: <0.20% = tight | 0.20-0.50% = acceptable | "
                ">0.50% = meaningful drift from benchmark."
            ),
            "application": (
                "If tracking error data is available, cite in Section 2. "
                "High tracking error in a passive ETF is a quality red flag."
            ),
        },
        {
            "title": "Roll Return (Commodity ETFs)",
            "formula": (
                "Roll return = Spot return − Futures return\n"
                "Contango (futures > spot) → negative roll → structural headwind.\n"
                "Backwardation (futures < spot) → positive roll → structural tailwind."
            ),
            "thresholds": (
                "Contango cost for crude ETFs historically: −5% to −15% per year.\n"
                "Gold (GLD/IAU) avoids roll as it holds physical — note this advantage."
            ),
            "application": (
                "For commodity ETFs: explain roll return impact in Section 2. "
                "Distinguish physical-backed (GLD, IAU, SGOL) vs futures-based (USO, UNG)."
            ),
        },
        {
            "title": "AUM & Liquidity",
            "formula": "Bid-ask spread proxy: (Ask − Bid) / Mid × 100",
            "thresholds": (
                "AUM >$1B = liquid | $100M-$1B = adequate | <$100M = liquidity risk.\n"
                "Spread: <0.05% = excellent | 0.05-0.20% = normal | >0.20% = expensive to trade."
            ),
            "application": (
                "Flag AUM <$500M as a closure/delisting risk. "
                "For GCC investors trading USD ETFs: note that wide spreads are amplified "
                "by USD/AED conversion friction."
            ),
        },
    ],

    # ── Commodity / Futures ───────────────────────────────────────────────
    "commodity": [
        {
            "title": "Convenience Yield",
            "formula": (
                "Convenience Yield = Risk-free rate + Storage cost − (F − S) / S\n"
                "where F = futures price, S = spot price"
            ),
            "thresholds": (
                "High convenience yield → market values immediate physical possession "
                "(supply disruption, scarcity). Low → plentiful supply."
            ),
            "application": (
                "For crude, copper, wheat: state whether spot-futures spread implies "
                "contango or backwardation and what it signals about near-term supply."
            ),
        },
        {
            "title": "Macro Linkage",
            "formula": (
                "Gold: negative real yield correlation (r ≈ −0.80 historically)\n"
                "Crude: global PMI + USD index (inverse)\n"
                "Copper: China PMI proxy ('Dr. Copper' leading indicator)"
            ),
            "thresholds": (
                "Real yield <0% → Gold structurally supported.\n"
                "DXY +1% → Gold −0.5% to −1% historically.\n"
                "China PMI <50 → Copper demand headwind."
            ),
            "application": (
                "Section 2: lead with the primary macro driver for this commodity. "
                "State the current level of the driver (real yield / DXY / China PMI) "
                "and its directional implication."
            ),
        },
        {
            "title": "Momentum — RSI & MACD",
            "formula": (
                "RSI = 100 − [100 / (1 + RS)] | MACD = EMA(12) − EMA(26)"
            ),
            "thresholds": (
                "Commodities: RSI >70 in uptrend = continuation signal (not reversal) "
                "if accompanied by fundamental supply squeeze.\n"
                "RSI <30 in downtrend = capitulation → watch for reversal catalyst."
            ),
            "application": (
                "For commodities, always pair momentum with a fundamental catalyst. "
                "Pure technical momentum in commodities without fundamental backing "
                "is less reliable than in equities."
            ),
        },
    ],

    # ── Crypto ────────────────────────────────────────────────────────────
    "crypto": [
        {
            "title": "Volatility Profile",
            "formula": (
                "Annualised Vol = Daily Vol × √252\n"
                "BTC historical vol: 60-80% | ETH: 70-90% | Altcoins: 90-150%+"
            ),
            "thresholds": (
                "Position sizing rule of thumb:\n"
                "  Max allocation % = (Max acceptable portfolio drawdown) / (Asset volatility × √holding period)\n"
                "  Example: 5% max drawdown, 80% BTC vol, 1-year hold → max 6.25% allocation."
            ),
            "application": (
                "Section 4: state annualised vol and derive the max prudent allocation "
                "for a 10% drawdown tolerance portfolio. Do NOT recommend >10% in any single crypto."
            ),
        },
        {
            "title": "BTC Correlation & Portfolio Impact",
            "formula": (
                "Portfolio Vol² = w_crypto² × σ_crypto² + w_portfolio² × σ_port² "
                "+ 2 × w_crypto × w_portfolio × ρ × σ_crypto × σ_port"
            ),
            "thresholds": (
                "BTC-S&P500 correlation: typically 0.2-0.5 (rises to 0.7+ in risk-off events).\n"
                "Diversification benefit exists only when ρ < 0.7 — in crises, correlations converge."
            ),
            "application": (
                "Warn that BTC's diversification benefit disappears in market stress. "
                "Cite current 90-day rolling correlation to SPY in Section 3."
            ),
        },
        {
            "title": "On-Chain Momentum Proxies",
            "formula": (
                "MVRV Ratio = Market Cap / Realised Cap\n"
                "  → >3.5 = historically overvalued | <1.0 = historically undervalued\n"
                "NVT Ratio = Network Value / Daily Transaction Volume (30-day MA)\n"
                "  → High NVT = low on-chain utility vs price (overvalued signal)"
            ),
            "thresholds": (
                "MVRV >3.5 → elevated distribution risk.\n"
                "MVRV 1.0-2.0 → fair value zone.\n"
                "MVRV <1.0 → historically strong accumulation zone."
            ),
            "application": (
                "If on-chain data is available, cite MVRV in Section 5 as a valuation "
                "sanity check alongside price-based momentum."
            ),
        },
    ],

    # ── Portfolio ─────────────────────────────────────────────────────────
    "portfolio": [
        {
            "title": "Sharpe Ratio",
            "formula": "(Rp − Rf) / σp",
            "thresholds": (
                "Rf = current US risk-free rate (use 4.5% if not specified)\n"
                "Strong: >1.0 | Acceptable: 0.5-1.0 | Weak: <0.5\n"
                "  → Sharpe <0.5 is acceptable ONLY in aggressive mandates where "
                "absolute return is prioritised over risk-adjusted efficiency (CFA L3)."
            ),
            "application": (
                "Always compute Sharpe using the provided portfolio return and vol. "
                "State: 'Sharpe of X falls in the [weak/acceptable/strong] range — "
                "[acceptable/not acceptable] for [mandate type] mandates.'"
            ),
        },
        {
            "title": "Maximum Drawdown",
            "formula": "MDD = (Trough Value − Peak Value) / Peak Value",
            "thresholds": (
                "Conservative mandate: MDD limit typically 10-15%\n"
                "Balanced mandate: 15-25%\n"
                "Aggressive mandate: 25-40%\n"
                "  ⚠️ Stress Warning: Correlation convergence in market crashes means "
                "realised drawdown is typically 20-40% WORSE than model estimate."
            ),
            "application": (
                "Compare model MDD estimate to client mandate limit. "
                "If within 5% of limit, flag as 'approaching mandate boundary'. "
                "Add stress-test caveat if correlations assumed <0.5."
            ),
        },
        {
            "title": "Effective N (Diversification)",
            "formula": "Effective N = 1 / Σ(wᵢ²)   [Herfindahl-based]",
            "thresholds": (
                "<3 = dangerously concentrated (single-stock risk dominates)\n"
                "3-6 = acceptable for tactical/high-conviction portfolios\n"
                ">6 = well-diversified\n"
                ">10 = institutional-grade diversification"
            ),
            "application": (
                "Compute from the provided position weights. "
                "If Effective N <3, MANDATORY warning: 'Portfolio is dangerously concentrated — "
                "a single position drawdown of >30% could breach mandate limits.'"
            ),
        },
        {
            "title": "Conviction Score & Position Sizing",
            "formula": (
                "Kelly fraction = (p × b − q) / b\n"
                "where p = win probability, b = win/loss ratio, q = 1 − p\n"
                "  → Use half-Kelly in practice (full Kelly is too volatile for most mandates)."
            ),
            "thresholds": (
                "High conviction (score 80-100): up to 8-10% single position\n"
                "Medium conviction (60-79): 4-6%\n"
                "Low conviction (<60): 1-3%\n"
                "  → Never exceed 10% in a single position regardless of conviction."
            ),
            "application": (
                "Cross-check top positions against their conviction scores. "
                "Overweight + low conviction = sizing mismatch → flag for rebalancing."
            ),
        },
    ],
}


def get_concepts(asset_type: str) -> list[dict]:
    """Return the concept list for a given asset type key."""
    return CONCEPTS_LIBRARY.get(asset_type, [])


def format_concepts_block(asset_type: str) -> str:
    """
    Build the VERIFIED INVESTMENT FRAMEWORKS block injected into the DeepSeek prompt.
    Returns empty string if no concepts found for asset_type.
    """
    concepts = get_concepts(asset_type)
    if not concepts:
        return ""

    lines = [
        "─" * 60,
        "VERIFIED INVESTMENT FRAMEWORKS — APPLY EXACTLY AS STATED:",
        "These definitions are pre-verified. Do NOT redefine or contradict them.",
        "─" * 60,
    ]
    for c in concepts:
        lines.append(f"\n▸ {c['title']}")
        lines.append(f"  Formula : {c['formula']}")
        lines.append(f"  Guidance: {c['thresholds']}")
        lines.append(f"  → Apply : {c['application']}")
    lines.append("─" * 60)
    return "\n".join(lines)
