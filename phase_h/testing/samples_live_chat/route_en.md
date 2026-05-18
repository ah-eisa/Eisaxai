# EisaX Global Portfolio — Balanced Multi-Asset Mandate
**Date:** May 18, 2026  |  **Capital:** $100,000  |  **Horizon:** 5 years  |  **Markets:** US + GCC + Gold

---

## A. Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Expected Return (annual) | **~13.5%** | [STRONG] |
| Expected Volatility | **~12.3%** | [MODERATE] |
| Sharpe Ratio | **~0.73** | [ACCEPTABLE] |
| Beta (vs MSCI World) | **~0.70** | [LOW] |
| Projected Value in 5 years | **$188,107** | Expected gain **$88,107** |

**Portfolio Regime:** **Multi-Asset Macro**
> Balanced cross-asset construction; no single regime dominates. Targets diversification across factor and macro drivers.
> **Regime Behavior vs Benchmark:** Designed for regime-balanced behavior; expect peer-like performance across most macro environments with reduced tail volatility.

**Confidence Calibration** · Score: **85%** · Evidence Breadth: **Broad** · Coverage Quality: **Full** · Reliability Tier: **Institutional** [STRONG]

**Implementation Feasibility** · Deployability: **High** [STRONG] (97/100) · Rebalancing Complexity: **Moderate** [MODERATE] · Liquidity: **High** [HIGH] · Execution Friction: **Low** [LOW] · Est. Turnover ~14%/yr · Est. Slippage ~3 bp

**Benchmark Context** · Reference: **60/40 Balanced (Global Equity / Bonds, Gold overlay)** · Bench Return ~11.5% · Tracking Deviation: **Moderate** [MODERATE] (4.0% TE) · Active Share: **High** [HIGH] (45%) · Style Drift: **US-underweight · Hedge-overweight**

> *Values are approximate, derived from long-run historical assumptions. Not a guarantee of future performance.*

---

## B. Mandate Feasibility Analysis

> Constraints enforced during optimization, with active value and status. Feasible solution requires all constraints in PASS / NEAR CAP / AUTO-RELAXED.

| Constraint | Limit | Actual | Status |
|------------|-------|--------|--------|
| Region cap · US | 51.7% | 48.5% | [PASS] |
| Region cap · GCC | 31.7% | 31.5% | [AT CAP] |
| Region cap · Gold | 21.7% | 20.0% | [NEAR CAP] |
| Beta cap (vs MSCI World) | 1.00 | 0.70 | [PASS] |
| Volatility cap (annualized) | 18.0% | 12.3% | [PASS] |
| Minimum bonds + cash floor | 15.0% | 0.00 | [AUTO-RELAXED (bonds/cash region not included)] |
| Holdings count | ≥ 5 | 9.0% | [PASS] |

---

## C. Risk Diagnostics

### Adaptive Risk Disclosures

*Conditional on the constructed portfolio — only risks that materially apply are surfaced.*

| Severity | Topic | Note |
|----------|-------|------|
| [MODERATE] | GCC Cyclical Concentration | GCC weighting of 32% creates oil-price and geopolitical concentration. Co-moves with commodity cycle and USD direction (peg-driven). |
| [MODERATE] | Cross-Currency Exposure | 32% of portfolio in non-USD-denominated holdings. SAR/AED pegged to USD; EGP floats — currency translation risk applies during reporting. |

### Structural Asset-Class Risk Factors

| Asset Class | Risk Factor |
|-------------|-------------|
| GCC / KSA | Oil price sensitivity · Geopolitical risk · Currency peg risk |
| Gold | No yield · Storage cost · USD-sensitive |

> *Currency risk applies to non-USD holdings (SAR pegged, AED pegged, EGP floating).*

### Drawdown Modeling Note

> Modeled max drawdown is an estimate, not a guarantee. Crisis episodes (2008, 2020, 2022) frequently exceeded model predictions by 30–50%. Stress-test against −30%, −40%, and −50% scenarios before committing capital.

### Benchmark Relative Diagnostics

*Benchmark: S&P 500 (SPY) · Reliability: Institutional-Lite · Window: 3m*

| Metric | Value | Tag |
| --- | --- | --- |
| Active Return | -0.04% | (moderate) |
| Tracking Error | 0.66% | (low) |
| Information Ratio | -0.06 | (low) |
| Rolling Alpha (12m) | +0.31% | — |
| Rolling Beta (12m) | 0.43 | — |
| Relative Drawdown | -0.20% | (low) |
| Upside Capture | 0.41 | (low) |
| Downside Capture | -0.06 | (low) |
| Relative Volatility | 0.69 | — |
| Active Share | 20.00% | (low) |
| Style Drift | material | — |

**Excess Return Decomposition**

| Component | Contribution (pp) |
| --- | --- |
| Allocation Effect | +0.27 |
| Selection Effect | -0.38 |
| Factor Effect | -0.35 |
| Concentration Effect | +0.42 |

**Benchmark-Relative Regime Behavior**

- Environments where portfolio likely outperforms: insufficient regime history
- Environments where portfolio likely lags: insufficient regime history

Benchmark-relative dispersion is contained; active outcomes are likely to be driven more by incremental allocation differences than by structural benchmark departure.

### Factor Risk Decomposition

*Model: Carhart · R²: 0.00 · Stability: 0.00 · Reliability: Indicative · Window: 0m*

| Factor | Loading | t-stat | Contribution (ret) | Contribution (vol) |
| --- | --- | --- | --- | --- |
| MKT | 0.00 | 0.00 | +0.00% | 0.00% |
| SMB | 0.00 | 0.00 | +0.00% | 0.00% |
| HML | 0.00 | 0.00 | +0.00% | 0.00% |
| MOM | 0.00 | 0.00 | +0.00% | 0.00% |

*Loadings represent rolling 36-month exposures; t-statistics are Newey-West adjusted where available.*

## D. Allocation Logic

**Mandate:** Balanced Multi-Asset Mandate · Moderate growth · Multi-asset · Diversified core allocation

### Regional Allocation

| Region | Weight | ~$ on $100k | Asset Classes |
|--------|--------|-------------|---------------|
| **US** | 48.5% | $48,500 | US Large Cap Tech, US Mid-Cap Equity, GS (GS), VLO (VLO) |
| **GCC** | 31.5% | $31,500 | 2222 (2222), 4190 (4190), FERTIGLB (FERTIGLB), ORDS (ORDS) |
| **Gold** | 20.0% | $20,000 | Gold |

### Optimal Asset Weights

| Asset Class | Region | Weight | ~$ on $100k | Proxy | Role | Construction Rationale |
|-------------|--------|--------|-------------|-------|------|------------------------|
| **US Large Cap Tech** | US | 25.0% | $25,000 | `QQQ` | Strategic Core | Long-duration growth sleeve · captures secular AI/tech earnings |
| **US Mid-Cap Equity** | US | 8.0% | $8,000 | `MDY` | Tactical Allocation | US equity core · liquid global benchmark proxy |
| **Gold** | Gold | 20.0% | $20,000 | `GLD` | Macro Hedge | Macro hedge · equity-duration compression, USD-weakening regimes |
| **2222 (2222)** | GCC | 8.5% | $8,500 | `2222` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **4190 (4190)** | GCC | 8.5% | $8,500 | `4190` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **FERTIGLB (FERTIGLB)** | GCC | 6.0% | $6,000 | `FERTIGLB` | Satellite / Diversifier | Regional exposure · GCC growth premium, low correlation to US equities |
| **ORDS (ORDS)** | GCC | 8.5% | $8,500 | `ORDS` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **GS (GS)** | US | 7.5% | $7,500 | `GS` | Tactical Allocation | US equity core · liquid global benchmark proxy |
| **VLO (VLO)** | US | 8.0% | $8,000 | `VLO` | Tactical Allocation | Energy cyclicality · inflation hedge and macro pro-cyclical exposure |

### Diversification Benefit

> **Diversification Ratio:** 1.50x — portfolio vol (12.3%) is 34% lower than weighted average of individual vols (18.4%)
> **vs Equal Weight:** Optimized vol 12.3% vs equal-weight 12.2%

### Benchmark-Relative Attribution

*Decomposition of excess return vs profile-mapped benchmark — distinguishes factor-driven from selection-driven performance.*

| Component | Value | Interpretation |
|-----------|-------|----------------|
| Excess Return vs Benchmark | +1.94% | Total active return |
| Beta Contribution | -0.68% | Factor-driven (market sensitivity differential) |
| Residual Contribution | +2.61% | Selection / concentration effects |

> *Excess return concentrated in residual/selection effects — review concentration risk before attributing to skill.*

> *Implementation note: Institutional rounding applied (5%/2.5%/1% tiered grid). 7 sub-2.5% positions consolidated. Sharpe drift: -0.00.*

---

## E. Rebalancing Plan

> Targeted reduction candidates with quantified impact on portfolio beta, volatility, and turnover.

| Proposed Action | Beta Before→After | Vol Before→After | Concentration Δ | Turnover | Difficulty |
|-----------------|-------------------|------------------|-----------------|----------|------------|
| Reduce US Large Cap Tech (QQQ) 25.0% → 15.0% | 0.696 → 0.622 (-0.074) | 12.26% → 11.60% (-0.66pp) | −10.0pp | 10.0pp | [HIGH] |
| Reduce Gold (GLD) 20.0% → 12.0% | 0.696 → 0.770 (+0.075) | 12.26% → 13.04% (+0.78pp) | −8.0pp | 8.0pp | [MODERATE] |

> *Impact estimates assume pro-rata redistribution of the trimmed weight across remaining holdings.*

### Implementation Steps

1. Open a brokerage account for target markets: US + GCC + Gold
2. Allocate $100,000 per the asset weights in Section D
3. Rebalance every 6–12 months
4. Avoid liquidating into short-term drawdowns — the 5-year horizon is designed to absorb interim market cycles


---

### Execution Efficiency Diagnostics

| Metric | Value | Tag |
| --- | --- | --- |
| Turnover | 100.0% | (high) |
| Implementation Shortfall | 15.8 bp | (moderate) |
| Market Impact | 2.7 bp | (low) |
| Estimated Slippage | 13.8 bp | (moderate) |
| Execution Complexity | high | — |
| Liquidity Stress | moderate | — |
| Rebalance Frequency | quarterly | — |

*Turnover penalty: not applied (linear λ=0.0010, quadratic λ=0.0005)*

*Tax-aware execution requires account-level lot data; placeholder pending integration with broker tax-lot feed.*

*GCC liquidity note: GCC legs use a Sun-Thu trading calendar; execution windows should account for the four-day local trading week.*

## F. AI Commentary Layer — CIO Synthesis

*AI-generated synthesis. Sections A–E above are deterministic and reproducible from the optimizer state.*

>


---

## H. Forward Scenario Distribution

*Horizon: 5.0y · Inflation: 2.0% · Seed: 42 · Paths: 12000*

| Scenario | Probability | Terminal P10 | Terminal P50 | Terminal P90 | Max Drawdown (P50) | Recovery (months, P50) |
| --- | --- | --- | --- | --- | --- | --- |
| soft landing | 30.0% | -13% | +35% | +109% | -20% | 5.0 |
| recession | 20.0% | -44% | +7% | +87% | -34% | 0.0 |
| stagflation | 10.0% | -36% | +23% | +128% | -33% | 0.0 |
| ai productivity boom | 10.0% | -15% | +45% | +145% | -24% | 4.0 |
| energy shock | 15.0% | -34% | +20% | +119% | -32% | 0.0 |
| liquidity crisis | 15.0% | -56% | +6% | +139% | -45% | 0.0 |

**Aggregate (probability-weighted)**

| Metric | Value |
| --- | --- |
| Expected terminal value range (real) | $0.65 - $2.17 (P10-P90) |
| Probability of loss (real) | 32% |
| Probability of target (>=4% real ann.) | 51% |
| Worst-decile terminal | $0.65 |
| Expected drawdown range | -29% to -54% |
| Recovery duration (median) | 1 months |

*Distributional framing only - outcomes reflect modelled assumptions; not a forecast.*

## G. Audit Appendix

| Field | Value |
|-------|-------|
| Snapshot ID | `408586d9f203` |
| Universe Hash | `d7abbfd1bb0e` |
| Solver | CLARABEL (cvxpy QP) |
| Solver Status | optimal |
| Assets (Universe) | 16 |
| Assets (Selected) | 9 |
| Max Beta | 1.0 |
| Max Volatility | 18.0% |
| Min Bonds + Cash | 15.0% |
| Max Drawdown (Requested) | Unconstrained% |
| Risk Aversion | 4.0 |
| Risk-Free Rate | 4.5% |
| Custom Caps | — |

> *Reproducible: same inputs → same Snapshot ID → identical output. Zero silent corrections.*

### Model Constraints — Structural Limitations of the Engine

- Historical simulation uses 252-day trailing window; structural breaks beyond that window are not captured.
- Correlation matrix is point-in-time; pairwise correlations rise toward 1.0 during liquidity events.
- Volatility is non-stationary; realized vol can diverge materially from in-sample estimates during regime shifts.
- Live-stock prices are cached at 15-minute intervals; intra-window movements not reflected.
- Beta estimates assume linear market sensitivity; convex behavior (gamma) ignored.
- Optimizer assumes frictionless rebalancing; transaction costs, slippage, and tax drag are out-of-scope.

> *Transparency note: the constraints above are inherent to historical-simulation portfolio engineering. Surfaced explicitly to support institutional review and governance.*

**Phase H Reproducibility**

| Metric | Value |
| --- | --- |
| Phase H version | 0.1.0 |
| Seed | 42 |
| Engines ran | benchmark_relative, execution_diag, factor_decomp, forward_scenario |
| Flags | enabled=on, benchmark=on, tc_optimizer=on, forward_sim=on, factor_model=on, committee=on, tone_guard=on, deterministic_seed=42 |
| benchmark_relative hash | 9b2a735e2b684abf |
| execution_diag hash | d232ab7cfedf1efc |
| factor_decomp hash | 3b1a39befb811ff5 |
| forward_scenario hash | b2fa5c4f3b274b61 |
| benchmark_relative version | 0.2.0 |
| execution_diag version | 0.2.0 |
| factor_decomp version | 0.2.0 |
| forward_scenario version | 0.2.0 |
| phase_h version | 0.1.0 |
