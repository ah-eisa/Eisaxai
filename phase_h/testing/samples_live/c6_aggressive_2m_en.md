# EisaX Global Portfolio — Aggressive Growth Mandate
**Date:** May 18, 2026  |  **Capital:** $2.0M  |  **Horizon:** 12 years  |  **Markets:** Global

---

## A. Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Expected Return (annual) | **~16.6%** | [STRONG] |
| Expected Volatility | **~16.7%** | [MODERATE] |
| Sharpe Ratio | **~0.72** | [ACCEPTABLE] |
| Beta (vs MSCI World) | **~0.76** | [MODERATE] |
| Projected Value in 12 years | **$12,578,373** | Expected gain **$10,578,373** |

**Portfolio Regime:** **Cyclical Value**
> Allocation tilted toward commodity-linked and emerging-market cyclical exposure. Procyclical with global PMI cycle.
> **Regime Behavior vs Benchmark:** Outperforms during commodity-led reflationary cycles and global PMI expansions; lags in growth-led rallies and risk-off episodes.

**Confidence Calibration** · Score: **76%** · Evidence Breadth: **Broad** · Coverage Quality: **Full** · Reliability Tier: **Institutional-Lite** [MODERATE]

**Implementation Feasibility** · Deployability: **High** [STRONG] (88/100) · Rebalancing Complexity: **Moderate** [MODERATE] · Liquidity: **High** [HIGH] · Execution Friction: **Low** [LOW] · Est. Turnover ~18%/yr · Est. Slippage ~8 bp

**Benchmark Context** · Reference: **MSCI World Equity Proxy** · Bench Return ~11.8% · Tracking Deviation: **High** [HIGH] (9.9% TE) · Active Share: **High** [HIGH] (63%) · Style Drift: **US-underweight · Crypto-tilted**

> *Values are approximate, derived from long-run historical assumptions. Not a guarantee of future performance.*

---

## B. Mandate Feasibility Analysis

> Constraints enforced during optimization, with active value and status. Feasible solution requires all constraints in PASS / NEAR CAP / AUTO-RELAXED.

| Constraint | Limit | Actual | Status |
|------------|-------|--------|--------|
| Region cap · US | 70.0% | 41.5% | [PASS] |
| Region cap · GCC | 40.0% | 40.0% | [AT CAP] |
| Region cap · Egypt | 20.0% | 8.5% | [PASS] |
| Region cap · Crypto | 10.0% | 10.0% | [AT CAP] |
| Beta cap (vs MSCI World) | 1.80 | 0.76 | [PASS] |
| Volatility cap (annualized) | 40.0% | 16.7% | [PASS] |
| Holdings count | ≥ 5 | 10.0% | [PASS] |

---

## C. Risk Diagnostics

### Adaptive Risk Disclosures

*Conditional on the constructed portfolio — only risks that materially apply are surfaced.*

| Severity | Topic | Note |
|----------|-------|------|
| [HIGH] | Crypto Liquidity Discontinuity | Crypto exposure of 10% subject to 24/7 trading, regulatory regime shifts, and liquidity discontinuities during stress events. Classify as satellite, not core. |
| [MODERATE] | GCC Cyclical Concentration | GCC weighting of 40% creates oil-price and geopolitical concentration. Co-moves with commodity cycle and USD direction (peg-driven). |
| [MODERATE] | Cross-Currency Exposure | 49% of portfolio in non-USD-denominated holdings. SAR/AED pegged to USD; EGP floats — currency translation risk applies during reporting. |

### Structural Asset-Class Risk Factors

| Asset Class | Risk Factor |
|-------------|-------------|
| GCC / KSA | Oil price sensitivity · Geopolitical risk · Currency peg risk |
| Egypt | Currency devaluation risk · Political risk · High inflation |

> *Currency risk applies to non-USD holdings (SAR pegged, AED pegged, EGP floating).*

### Correlation Cluster Risk

| Severity | Cluster | Combined Weight | Note |
|----------|---------|-----------------|------|
| [HIGH] | GCC + Commodities | 40.0% | Oil/commodity cycle co-movement |

### Crypto Analytical Framework

> Crypto positions (10% of portfolio) are evaluated using a separate analytical lens:
> *network activity · ETF flows · realized volatility · liquidity regime · cycle positioning.* Equity valuation multiples and earnings-quality metrics are not applicable.

### Drawdown Modeling Note

> Modeled max drawdown is an estimate, not a guarantee. Crisis episodes (2008, 2020, 2022) frequently exceeded model predictions by 30–50%. Stress-test against −30%, −40%, and −50% scenarios before committing capital.

### Benchmark Relative Diagnostics

*Benchmark: S&P 500 (SPY) · Reliability: Indicative · Window: 3m*

| Metric | Value | Tag |
| --- | --- | --- |
| Active Return | -0.93% | (moderate) |
| Tracking Error | 0.67% | (low) |
| Information Ratio | -1.40 | (low) |
| Rolling Alpha (12m) | +0.20% | — |
| Rolling Beta (12m) | 0.03 | — |
| Relative Drawdown | -0.30% | (low) |
| Upside Capture | 0.08 | (low) |
| Downside Capture | -0.89 | (low) |
| Relative Volatility | 0.08 | — |
| Active Share | 10.00% | (low) |
| Style Drift | mild | — |

**Excess Return Decomposition**

| Component | Contribution (pp) |
| --- | --- |
| Allocation Effect | -0.12 |
| Selection Effect | -0.91 |
| Factor Effect | -1.13 |
| Concentration Effect | +1.22 |

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

**Mandate:** Aggressive Growth Mandate · Maximum return · Elevated risk tolerance · Concentrated growth tilt

### Regional Allocation

| Region | Weight | ~$ on $100k | Asset Classes |
|--------|--------|-------------|---------------|
| **US** | 41.5% | $830,000 | US Large Cap Tech, GS (GS), B (B) |
| **GCC** | 40.0% | $800,000 | 2222 (2222), 4190 (4190), 5110 (5110), FERTIGLB (FERTIGLB), ORDS (ORDS) |
| **Crypto** | 10.0% | $200,000 | Bitcoin |
| **Egypt** | 8.5% | $170,000 | EGAL (EGAL) |

### Optimal Asset Weights

| Asset Class | Region | Weight | ~$ on $100k | Proxy | Role | Construction Rationale |
|-------------|--------|--------|-------------|-------|------|------------------------|
| **US Large Cap Tech** | US | 25.0% | $500,000 | `QQQ` | Strategic Core | Long-duration growth sleeve · captures secular AI/tech earnings |
| **Bitcoin** | Crypto | 10.0% | $200,000 | `BTC-USD` | Opportunistic Satellite | Asymmetric satellite · high-volatility return contributor (not a hedge) |
| **2222 (2222)** | GCC | 8.5% | $170,000 | `2222` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **4190 (4190)** | GCC | 8.5% | $170,000 | `4190` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **5110 (5110)** | GCC | 6.0% | $120,000 | `5110` | Satellite / Diversifier | Regional exposure · GCC growth premium, low correlation to US equities |
| **FERTIGLB (FERTIGLB)** | GCC | 8.5% | $170,000 | `FERTIGLB` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **ORDS (ORDS)** | GCC | 8.5% | $170,000 | `ORDS` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **EGAL (EGAL)** | Egypt | 8.5% | $170,000 | `EGAL` | Tactical Allocation | Frontier-market tilt · structural reform exposure (higher idiosyncratic risk) |
| **GS (GS)** | US | 8.0% | $160,000 | `GS` | Tactical Allocation | US equity core · liquid global benchmark proxy |
| **B (B)** | US | 8.5% | $170,000 | `B` | Tactical Allocation | Energy cyclicality · inflation hedge and macro pro-cyclical exposure |

### Diversification Benefit

> **Diversification Ratio:** 1.38x — portfolio vol (16.7%) is 28% lower than weighted average of individual vols (23.1%)
> **vs Equal Weight:** Optimized vol 16.7% vs equal-weight 12.3%

### Benchmark-Relative Attribution

*Decomposition of excess return vs profile-mapped benchmark — distinguishes factor-driven from selection-driven performance.*

| Component | Value | Interpretation |
|-----------|-------|----------------|
| Excess Return vs Benchmark | +4.71% | Total active return |
| Beta Contribution | -0.55% | Factor-driven (market sensitivity differential) |
| Residual Contribution | +5.26% | Selection / concentration effects |

> *Excess return concentrated in residual/selection effects — review concentration risk before attributing to skill.*

> *Implementation note: Institutional rounding applied (5%/2.5%/1% tiered grid). 21 sub-2.5% positions consolidated. Sharpe drift: +0.00.*

---

## E. Rebalancing Plan

> Targeted reduction candidates with quantified impact on portfolio beta, volatility, and turnover.

| Proposed Action | Beta Before→After | Vol Before→After | Concentration Δ | Turnover | Difficulty |
|-----------------|-------------------|------------------|-----------------|----------|------------|
| Reduce US Large Cap Tech (QQQ) 25.0% → 15.0% | 0.759 → 0.693 (-0.066) | 16.70% → 16.86% (+0.16pp) | −10.0pp | 10.0pp | [HIGH] |

> *Impact estimates assume pro-rata redistribution of the trimmed weight across remaining holdings.*

### Implementation Steps

1. Open a brokerage account for target markets: Global
2. Allocate $2.0M per the asset weights in Section D
3. Rebalance every 6–12 months
4. Avoid liquidating into short-term drawdowns — the 12-year horizon is designed to absorb interim market cycles


---


---

### Execution Efficiency Diagnostics

| Metric | Value | Tag |
| --- | --- | --- |
| Turnover | 100.0% | (high) |
| Implementation Shortfall | 22.6 bp | (moderate) |
| Market Impact | 3.7 bp | (low) |
| Estimated Slippage | 20.6 bp | (moderate) |
| Execution Complexity | high | — |
| Liquidity Stress | moderate | — |
| Rebalance Frequency | quarterly | — |

*Turnover penalty: not applied (linear λ=0.0010, quadratic λ=0.0005)*

*Tax-aware execution requires account-level lot data; placeholder pending integration with broker tax-lot feed.*

*GCC liquidity note: GCC legs use a Sun-Thu trading calendar; execution windows should account for the four-day local trading week.*

## H. Forward Scenario Distribution

*Horizon: 5.0y · Inflation: 2.0% · Seed: 42 · Paths: 12000*

| Scenario | Probability | Terminal P10 | Terminal P50 | Terminal P90 | Max Drawdown (P50) | Recovery (months, P50) |
| --- | --- | --- | --- | --- | --- | --- |
| soft landing | 30.0% | -13% | +46% | +140% | -25% | 4.0 |
| recession | 20.0% | -48% | +8% | +130% | -40% | 0.0 |
| stagflation | 10.0% | -47% | +20% | +168% | -40% | 0.0 |
| ai productivity boom | 10.0% | -20% | +56% | +191% | -29% | 4.0 |
| energy shock | 15.0% | -47% | +15% | +141% | -39% | 0.0 |
| liquidity crisis | 15.0% | -64% | +5% | +189% | -53% | 0.0 |

**Aggregate (probability-weighted)**

| Metric | Value |
| --- | --- |
| Expected terminal value range (real) | $0.57 - $2.49 (P10-P90) |
| Probability of loss (real) | 33% |
| Probability of target (>=4% real ann.) | 54% |
| Worst-decile terminal | $0.57 |
| Expected drawdown range | -35% to -61% |
| Recovery duration (median) | 0 months |

*Distributional framing only - outcomes reflect modelled assumptions; not a forecast.*

## G. Audit Appendix

| Field | Value |
|-------|-------|
| Snapshot ID | `027354835fb1` |
| Universe Hash | `36d2e7a59a86` |
| Solver | CLARABEL (cvxpy QP) |
| Solver Status | optimal |
| Assets (Universe) | 31 |
| Assets (Selected) | 10 |
| Max Beta | 1.8 |
| Max Volatility | 40.0% |
| Min Bonds + Cash | 0.0% |
| Max Drawdown (Requested) | Unconstrained% |
| Risk Aversion | 0.6 |
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
| benchmark_relative hash | 7d6b381cbf05101f |
| execution_diag hash | 3d7c8cd481004d44 |
| factor_decomp hash | 9c409491c5c67ce5 |
| forward_scenario hash | 0bacc7a3e8ce6b58 |
| benchmark_relative version | 0.2.0 |
| execution_diag version | 0.2.0 |
| factor_decomp version | 0.2.0 |
| forward_scenario version | 0.2.0 |
| phase_h version | 0.1.0 |
