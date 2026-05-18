# EisaX Global Portfolio — Balanced Multi-Asset Mandate
**Date:** May 18, 2026  |  **Capital:** $300,000  |  **Horizon:** 6 years  |  **Markets:** GCC + Cash + Bonds

---

## A. Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Expected Return (annual) | **~9.5%** | [MODERATE] |
| Expected Volatility | **~8.3%** | [LOW] |
| Sharpe Ratio | **~0.60** | [ACCEPTABLE] |
| Beta (vs MSCI World) | **~0.20** | [LOW] |
| Projected Value in 6 years | **$517,137** | Expected gain **$217,137** |

**Portfolio Regime:** **Defensive Income**
> Income-generating sleeves dominate. Lower sensitivity to equity drawdowns; primary risk vector is duration and credit spread widening.
> **Regime Behavior vs Benchmark:** Outperforms during equity drawdowns and disinflationary cycles; lags in strong risk-on rallies and steepening yield curves.

**Confidence Calibration** · Score: **85%** · Evidence Breadth: **Broad** · Coverage Quality: **Full** · Reliability Tier: **Institutional** [STRONG]

**Implementation Feasibility** · Deployability: **High** [STRONG] (100/100) · Rebalancing Complexity: **Moderate** [MODERATE] · Liquidity: **High** [HIGH] · Execution Friction: **Low** [LOW] · Est. Turnover ~12%/yr · Est. Slippage ~3 bp

**Benchmark Context** · Reference: **60/40 Balanced (Global Equity / Bonds, Gold overlay)** · Bench Return ~7.4% · Tracking Deviation: **Moderate** [MODERATE] (3.6% TE) · Active Share: **High** [HIGH] (44%) · Style Drift: **Duration-underweight**

> *Values are approximate, derived from long-run historical assumptions. Not a guarantee of future performance.*

---

## B. Mandate Feasibility Analysis

> Constraints enforced during optimization, with active value and status. Feasible solution requires all constraints in PASS / NEAR CAP / AUTO-RELAXED.

| Constraint | Limit | Actual | Status |
|------------|-------|--------|--------|
| Region cap · GCC | 40.0% | 40.0% | [AT CAP] |
| Region cap · Bonds | 45.0% | 40.0% | [PASS] |
| Region cap · Cash | 20.0% | 20.0% | [AT CAP] |
| Beta cap (vs MSCI World) | 1.00 | 0.20 | [PASS] |
| Volatility cap (annualized) | 18.0% | 8.3% | [PASS] |
| Minimum bonds + cash floor | 15.0% | 60.0% | [PASS] |
| Holdings count | ≥ 5 | 8.0% | [PASS] |

---

## C. Risk Diagnostics

### Adaptive Risk Disclosures

*Conditional on the constructed portfolio — only risks that materially apply are surfaced.*

| Severity | Topic | Note |
|----------|-------|------|
| [MODERATE] | GCC Cyclical Concentration | GCC weighting of 40% creates oil-price and geopolitical concentration. Co-moves with commodity cycle and USD direction (peg-driven). |
| [MODERATE] | Cross-Currency Exposure | 40% of portfolio in non-USD-denominated holdings. SAR/AED pegged to USD; EGP floats — currency translation risk applies during reporting. |

### Structural Asset-Class Risk Factors

| Asset Class | Risk Factor |
|-------------|-------------|
| GCC / KSA | Oil price sensitivity · Geopolitical risk · Currency peg risk |
| EM Bonds | Default risk · FX risk · Liquidity risk |

> *Currency risk applies to non-USD holdings (SAR pegged, AED pegged, EGP floating).*

### Correlation Cluster Risk

| Severity | Cluster | Combined Weight | Note |
|----------|---------|-----------------|------|
| [HIGH] | GCC + Commodities | 40.0% | Oil/commodity cycle co-movement |

### Drawdown Modeling Note

> Modeled max drawdown is an estimate, not a guarantee. Crisis episodes (2008, 2020, 2022) frequently exceeded model predictions by 30–50%. Stress-test against −30%, −40%, and −50% scenarios before committing capital.

### Benchmark Relative Diagnostics

*Benchmark: MSCI World (URTH) · Reliability: Indicative · Window: 3m*

| Metric | Value | Tag |
| --- | --- | --- |
| Active Return | +0.18% | (moderate) |
| Tracking Error | 0.06% | (low) |
| Information Ratio | 3.06 | (elevated) |
| Rolling Alpha (12m) | 0.00% | — |
| Rolling Beta (12m) | 1.00 | — |
| Relative Drawdown | 0.00% | (low) |
| Upside Capture | 0.00 | (low) |
| Downside Capture | 0.00 | (low) |
| Relative Volatility | 0.00 | — |
| Active Share | 100.00% | (elevated) |
| Style Drift | severe | — |

**Excess Return Decomposition**

| Component | Contribution (pp) |
| --- | --- |
| Allocation Effect | +0.92 |
| Selection Effect | 0.00 |
| Factor Effect | 0.00 |
| Concentration Effect | -0.75 |

**Benchmark-Relative Regime Behavior**

- Environments where portfolio likely outperforms: insufficient regime history
- Environments where portfolio likely lags: insufficient regime history

Tracking error is elevated relative to benchmark composition; active share above 60% indicates material structural deviation that may amplify factor-driven dispersion in stress regimes.

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
| **Bonds** | 40.0% | $120,000 | US Treasuries (LT), EM Bonds |
| **GCC** | 40.0% | $120,000 | 2222 (2222), 4190 (4190), 5110 (5110), FERTIGLB (FERTIGLB), ORDS (ORDS) |
| **Cash** | 20.0% | $60,000 | Cash / T-Bills |

### Optimal Asset Weights

| Asset Class | Region | Weight | ~$ on $100k | Proxy | Role | Construction Rationale |
|-------------|--------|--------|-------------|-------|------|------------------------|
| **US Treasuries (LT)** | Bonds | 15.0% | $45,000 | `TLT` | Income / Diversification | Long-duration UST · negative correlation to equity in deflationary shocks |
| **EM Bonds** | Bonds | 25.0% | $75,000 | `EMB` | Income / Diversification | EM credit · spread carry with FX/default risk overlay |
| **Cash / T-Bills** | Cash | 20.0% | $60,000 | `BIL` | Income / Diversification | Dry powder · liquidity buffer and risk-free yield anchor |
| **2222 (2222)** | GCC | 8.5% | $25,500 | `2222` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **4190 (4190)** | GCC | 8.5% | $25,500 | `4190` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **5110 (5110)** | GCC | 6.0% | $18,000 | `5110` | Satellite / Diversifier | Regional exposure · GCC growth premium, low correlation to US equities |
| **FERTIGLB (FERTIGLB)** | GCC | 8.5% | $25,500 | `FERTIGLB` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **ORDS (ORDS)** | GCC | 8.5% | $25,500 | `ORDS` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |

### Diversification Benefit

> **Diversification Ratio:** 1.92x — portfolio vol (8.3%) is 48% lower than weighted average of individual vols (15.9%)
> **vs Equal Weight:** Optimized vol 8.3% vs equal-weight 11.7%

### Benchmark-Relative Attribution

*Decomposition of excess return vs profile-mapped benchmark — distinguishes factor-driven from selection-driven performance.*

| Component | Value | Interpretation |
|-----------|-------|----------------|
| Excess Return vs Benchmark | +2.11% | Total active return |
| Beta Contribution | +0.32% | Factor-driven (market sensitivity differential) |
| Residual Contribution | +1.78% | Selection / concentration effects |

> *Excess return concentrated in residual/selection effects — review concentration risk before attributing to skill.*

> *Implementation note: Institutional rounding applied (5%/2.5%/1% tiered grid). 3 sub-2.5% positions consolidated. Sharpe drift: +0.00.*

---

## E. Rebalancing Plan

> Targeted reduction candidates with quantified impact on portfolio beta, volatility, and turnover.

| Proposed Action | Beta Before→After | Vol Before→After | Concentration Δ | Turnover | Difficulty |
|-----------------|-------------------|------------------|-----------------|----------|------------|
| Reduce EM Bonds (EMB) 25.0% → 15.0% | 0.197 → 0.196 (-0.000) | 8.28% → 8.65% (+0.37pp) | −10.0pp | 10.0pp | [HIGH] |
| Reduce US Treasuries (LT) (TLT) 15.0% → 9.0% | 0.197 → 0.232 (+0.035) | 8.28% → 8.44% (+0.16pp) | −6.0pp | 6.0pp | [MODERATE] |

> *Impact estimates assume pro-rata redistribution of the trimmed weight across remaining holdings.*

### Implementation Steps

1. Open a brokerage account for target markets: GCC + Cash + Bonds
2. Allocate $300,000 per the asset weights in Section D
3. Rebalance every 6–12 months
4. Avoid liquidating into short-term drawdowns — the 6-year horizon is designed to absorb interim market cycles


---


---

### Execution Efficiency Diagnostics

| Metric | Value | Tag |
| --- | --- | --- |
| Turnover | 100.0% | (high) |
| Implementation Shortfall | 18.4 bp | (moderate) |
| Market Impact | 1.8 bp | (low) |
| Estimated Slippage | 16.4 bp | (moderate) |
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
| soft landing | 30.0% | -6% | +19% | +50% | -10% | 5.0 |
| recession | 20.0% | -21% | +11% | +57% | -18% | 0.0 |
| stagflation | 10.0% | -26% | +7% | +53% | -20% | 0.0 |
| ai productivity boom | 10.0% | -10% | +20% | +61% | -13% | 5.0 |
| energy shock | 15.0% | -23% | +10% | +53% | -18% | 0.0 |
| liquidity crisis | 15.0% | -37% | +3% | +74% | -29% | 0.0 |

**Aggregate (probability-weighted)**

| Metric | Value |
| --- | --- |
| Expected terminal value range (real) | $0.79 - $1.55 (P10-P90) |
| Probability of loss (real) | 31% |
| Probability of target (>=4% real ann.) | 39% |
| Worst-decile terminal | $0.79 |
| Expected drawdown range | -16% to -33% |
| Recovery duration (median) | 3 months |

*Distributional framing only - outcomes reflect modelled assumptions; not a forecast.*

## G. Audit Appendix

| Field | Value |
|-------|-------|
| Snapshot ID | `ff02c16d069c` |
| Universe Hash | `a46c506fc1d7` |
| Solver | CLARABEL (cvxpy QP) |
| Solver Status | optimal |
| Assets (Universe) | 11 |
| Assets (Selected) | 8 |
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
| benchmark_relative hash | 87d22c0514669ed0 |
| execution_diag hash | 9f4eff8280551ad6 |
| factor_decomp hash | f73ab3c24eea5482 |
| forward_scenario hash | d9126b3600231556 |
| benchmark_relative version | 0.2.0 |
| execution_diag version | 0.2.0 |
| factor_decomp version | 0.2.0 |
| forward_scenario version | 0.2.0 |
| phase_h version | 0.1.0 |
