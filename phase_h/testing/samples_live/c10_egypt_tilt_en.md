# EisaX Global Portfolio — Long-Horizon Growth Mandate
**Date:** May 18, 2026  |  **Capital:** $150,000  |  **Horizon:** 10 years  |  **Markets:** Egypt + GCC + Cash + Bonds

---

## A. Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Expected Return (annual) | **~11.6%** | [STRONG] |
| Expected Volatility | **~12.2%** | [MODERATE] |
| Sharpe Ratio | **~0.58** | [ACCEPTABLE] |
| Beta (vs MSCI World) | **~0.35** | [LOW] |
| Projected Value in 10 years | **$447,494** | Expected gain **$297,494** |

**Portfolio Regime:** **Cyclical Value**
> Allocation tilted toward commodity-linked and emerging-market cyclical exposure. Procyclical with global PMI cycle.
> **Regime Behavior vs Benchmark:** Outperforms during commodity-led reflationary cycles and global PMI expansions; lags in growth-led rallies and risk-off episodes.

**Confidence Calibration** · Score: **70%** · Evidence Breadth: **Broad** · Coverage Quality: **Full** · Reliability Tier: **Indicative** [LOW]

**Implementation Feasibility** · Deployability: **High** [STRONG] (85/100) · Rebalancing Complexity: **Moderate** [MODERATE] · Liquidity: **High** [HIGH] · Execution Friction: **Low** [LOW] · Est. Turnover ~16%/yr · Est. Slippage ~11 bp

**Benchmark Context** · Reference: **80/20 Growth (Global Equity / Bonds)** · Bench Return ~9.3% · Tracking Deviation: **High** [HIGH] (7.1% TE) · Active Share: **High** [HIGH] (44%) · Style Drift: **Duration-underweight**

> *Values are approximate, derived from long-run historical assumptions. Not a guarantee of future performance.*

---

## B. Mandate Feasibility Analysis

> Constraints enforced during optimization, with active value and status. Feasible solution requires all constraints in PASS / NEAR CAP / AUTO-RELAXED.

| Constraint | Limit | Actual | Status |
|------------|-------|--------|--------|
| Region cap · GCC | 40.0% | 40.0% | [AT CAP] |
| Region cap · Egypt | 30.0% | 30.0% | [AT CAP] |
| Region cap · Bonds | 25.0% | 25.0% | [AT CAP] |
| Region cap · Cash | 10.0% | 5.0% | [PASS] |
| Beta cap (vs MSCI World) | 1.30 | 0.35 | [PASS] |
| Volatility cap (annualized) | 25.0% | 12.2% | [PASS] |
| Minimum bonds + cash floor | 5.0% | 30.0% | [PASS] |
| Holdings count | ≥ 5 | 10.0% | [PASS] |

---

## C. Risk Diagnostics

### Adaptive Risk Disclosures

*Conditional on the constructed portfolio — only risks that materially apply are surfaced.*

| Severity | Topic | Note |
|----------|-------|------|
| [MODERATE] | GCC Cyclical Concentration | GCC weighting of 40% creates oil-price and geopolitical concentration. Co-moves with commodity cycle and USD direction (peg-driven). |
| [MODERATE] | Cross-Currency Exposure | 70% of portfolio in non-USD-denominated holdings. SAR/AED pegged to USD; EGP floats — currency translation risk applies during reporting. |

### Structural Asset-Class Risk Factors

| Asset Class | Risk Factor |
|-------------|-------------|
| GCC / KSA | Oil price sensitivity · Geopolitical risk · Currency peg risk |
| Egypt | Currency devaluation risk · Political risk · High inflation |
| EM Bonds | Default risk · FX risk · Liquidity risk |

> *Currency risk applies to non-USD holdings (SAR pegged, AED pegged, EGP floating).*

### Correlation Cluster Risk

| Severity | Cluster | Combined Weight | Note |
|----------|---------|-----------------|------|
| [HIGH] | GCC + Commodities | 40.0% | Oil/commodity cycle co-movement |

### Drawdown Modeling Note

> Modeled max drawdown is an estimate, not a guarantee. Crisis episodes (2008, 2020, 2022) frequently exceeded model predictions by 30–50%. Stress-test against −30%, −40%, and −50% scenarios before committing capital.

### Benchmark Relative Diagnostics

*Benchmark: S&P 500 (SPY) · Reliability: Indicative · Window: 3m*

| Metric | Value | Tag |
| --- | --- | --- |
| Active Return | -1.16% | (moderate) |
| Tracking Error | 0.68% | (low) |
| Information Ratio | -1.69 | (low) |
| Rolling Alpha (12m) | +0.00% | — |
| Rolling Beta (12m) | -0.00 | — |
| Relative Drawdown | -0.32% | (low) |
| Upside Capture | -0.00 | (low) |
| Downside Capture | -0.02 | (low) |
| Relative Volatility | 0.00 | — |
| Active Share | 43.00% | (moderate) |
| Style Drift | severe | — |

**Excess Return Decomposition**

| Component | Contribution (pp) |
| --- | --- |
| Allocation Effect | -0.21 |
| Selection Effect | 0.00 |
| Factor Effect | -1.16 |
| Concentration Effect | +0.22 |

**Benchmark-Relative Regime Behavior**

- Environments where portfolio likely outperforms: insufficient regime history
- Environments where portfolio likely lags: insufficient regime history

Benchmark-relative risk is moderate; portfolio outcomes should remain sensitive to allocation differences and beta dispersion versus the selected benchmark.

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

**Mandate:** Long-Horizon Growth Mandate · High return target · Diversified global growth · Long-horizon equity tilt

### Regional Allocation

| Region | Weight | ~$ on $100k | Asset Classes |
|--------|--------|-------------|---------------|
| **GCC** | 40.0% | $60,000 | 2222 (2222), 4190 (4190), 5110 (5110), FERTIGLB (FERTIGLB), ORDS (ORDS) |
| **Egypt** | 30.0% | $45,000 | Egypt Equities, TMGH (TMGH), EGAL (EGAL) |
| **Bonds** | 25.0% | $37,500 | EM Bonds |
| **Cash** | 5.0% | $7,500 | Cash / T-Bills |

### Optimal Asset Weights

| Asset Class | Region | Weight | ~$ on $100k | Proxy | Role | Construction Rationale |
|-------------|--------|--------|-------------|-------|------|------------------------|
| **Egypt Equities** | Egypt | 13.0% | $19,500 | `EFID.CA` | Tactical Allocation | Frontier-market tilt · structural reform exposure (higher idiosyncratic risk) |
| **EM Bonds** | Bonds | 25.0% | $37,500 | `EMB` | Income / Diversification | EM credit · spread carry with FX/default risk overlay |
| **Cash / T-Bills** | Cash | 5.0% | $7,500 | `BIL` | Income / Diversification | Dry powder · liquidity buffer and risk-free yield anchor |
| **2222 (2222)** | GCC | 8.5% | $12,750 | `2222` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **4190 (4190)** | GCC | 8.5% | $12,750 | `4190` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **5110 (5110)** | GCC | 6.0% | $9,000 | `5110` | Satellite / Diversifier | Regional exposure · GCC growth premium, low correlation to US equities |
| **FERTIGLB (FERTIGLB)** | GCC | 8.5% | $12,750 | `FERTIGLB` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **ORDS (ORDS)** | GCC | 8.5% | $12,750 | `ORDS` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **TMGH (TMGH)** | Egypt | 8.5% | $12,750 | `TMGH` | Tactical Allocation | Frontier-market tilt · structural reform exposure (higher idiosyncratic risk) |
| **EGAL (EGAL)** | Egypt | 8.5% | $12,750 | `EGAL` | Tactical Allocation | Frontier-market tilt · structural reform exposure (higher idiosyncratic risk) |

### Diversification Benefit

> **Diversification Ratio:** 1.57x — portfolio vol (12.2%) is 36% lower than weighted average of individual vols (19.0%)
> **vs Equal Weight:** Optimized vol 12.2% vs equal-weight 12.3%

### Benchmark-Relative Attribution

*Decomposition of excess return vs profile-mapped benchmark — distinguishes factor-driven from selection-driven performance.*

| Component | Value | Interpretation |
|-----------|-------|----------------|
| Excess Return vs Benchmark | +2.27% | Total active return |
| Beta Contribution | +0.61% | Factor-driven (market sensitivity differential) |
| Residual Contribution | +1.66% | Selection / concentration effects |

> *Excess return concentrated in residual/selection effects — review concentration risk before attributing to skill.*

> *Implementation note: Institutional rounding applied (5%/2.5%/1% tiered grid). 5 sub-2.5% positions consolidated. Sharpe drift: -0.00.*

---

## E. Rebalancing Plan

> Targeted reduction candidates with quantified impact on portfolio beta, volatility, and turnover.

| Proposed Action | Beta Before→After | Vol Before→After | Concentration Δ | Turnover | Difficulty |
|-----------------|-------------------|------------------|-----------------|----------|------------|
| Reduce EM Bonds (EMB) 25.0% → 15.0% | 0.347 → 0.366 (+0.020) | 12.16% → 12.97% (+0.81pp) | −10.0pp | 10.0pp | [HIGH] |

> *Impact estimates assume pro-rata redistribution of the trimmed weight across remaining holdings.*

### Implementation Steps

1. Open a brokerage account for target markets: Egypt + GCC + Cash + Bonds
2. Allocate $150,000 per the asset weights in Section D
3. Rebalance every 6–12 months
4. Avoid liquidating into short-term drawdowns — the 10-year horizon is designed to absorb interim market cycles


---


---

### Execution Efficiency Diagnostics

| Metric | Value | Tag |
| --- | --- | --- |
| Turnover | 100.0% | (high) |
| Implementation Shortfall | 30.5 bp | (moderate) |
| Market Impact | 2.6 bp | (low) |
| Estimated Slippage | 28.5 bp | (moderate) |
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
| soft landing | 30.0% | -4% | +27% | +69% | -14% | 5.0 |
| recession | 20.0% | -30% | +7% | +67% | -25% | 0.0 |
| stagflation | 10.0% | -31% | +11% | +79% | -25% | 0.0 |
| ai productivity boom | 10.0% | -11% | +32% | +89% | -17% | 5.0 |
| energy shock | 15.0% | -31% | +10% | +70% | -24% | 0.0 |
| liquidity crisis | 15.0% | -44% | +7% | +98% | -35% | 0.0 |

**Aggregate (probability-weighted)**

| Metric | Value |
| --- | --- |
| Expected terminal value range (real) | $0.74 - $1.74 (P10-P90) |
| Probability of loss (real) | 31% |
| Probability of target (>=4% real ann.) | 47% |
| Worst-decile terminal | $0.74 |
| Expected drawdown range | -21% to -41% |
| Recovery duration (median) | 2 months |

*Distributional framing only - outcomes reflect modelled assumptions; not a forecast.*

## G. Audit Appendix

| Field | Value |
|-------|-------|
| Snapshot ID | `5df481097c42` |
| Universe Hash | `3f6f72bd7edc` |
| Solver | CLARABEL (cvxpy QP) |
| Solver Status | optimal |
| Assets (Universe) | 15 |
| Assets (Selected) | 10 |
| Max Beta | 1.3 |
| Max Volatility | 25.0% |
| Min Bonds + Cash | 5.0% |
| Max Drawdown (Requested) | Unconstrained% |
| Risk Aversion | 1.5 |
| Risk-Free Rate | 4.5% |
| Custom Caps | Egypt ≤ 30.0% |

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
| benchmark_relative hash | 0f5b3f613b6e4a92 |
| execution_diag hash | bdc5fcbb3a6d0985 |
| factor_decomp hash | 9b995998be320e2f |
| forward_scenario hash | 51b8c702bcbf66f7 |
| benchmark_relative version | 0.2.0 |
| execution_diag version | 0.2.0 |
| factor_decomp version | 0.2.0 |
| forward_scenario version | 0.2.0 |
| phase_h version | 0.1.0 |
