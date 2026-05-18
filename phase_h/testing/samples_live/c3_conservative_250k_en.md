# EisaX Global Portfolio — Capital Preservation Mandate
**Date:** May 18, 2026  |  **Capital:** $250,000  |  **Horizon:** 10 years  |  **Markets:** Global

---

## A. Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Expected Return (annual) | **~11.0%** | [STRONG] |
| Expected Volatility | **~8.2%** | [LOW] |
| Sharpe Ratio | **~0.80** | [ACCEPTABLE] |
| Beta (vs MSCI World) | **~0.42** | [LOW] |
| Projected Value in 10 years | **$711,135** | Expected gain **$461,135** |

**Portfolio Regime:** **Multi-Asset Macro**
> Balanced cross-asset construction; no single regime dominates. Targets diversification across factor and macro drivers.
> **Regime Behavior vs Benchmark:** Designed for regime-balanced behavior; expect peer-like performance across most macro environments with reduced tail volatility.

**Confidence Calibration** · Score: **82%** · Evidence Breadth: **Broad** · Coverage Quality: **Full** · Reliability Tier: **Institutional-Lite** [MODERATE]

**Implementation Feasibility** · Deployability: **High** [STRONG] (90/100) · Rebalancing Complexity: **High** [HIGH] · Liquidity: **High** [HIGH] · Execution Friction: **Low** [LOW] · Est. Turnover ~17%/yr · Est. Slippage ~4 bp

**Benchmark Context** · Reference: **30/50/20 Conservative (US Equity / Bonds / Gold)** · Bench Return ~7.9% · Tracking Deviation: **Moderate** [MODERATE] (5.8% TE) · Active Share: **High** [HIGH] (70%) · Style Drift: **Duration-underweight**

> *Values are approximate, derived from long-run historical assumptions. Not a guarantee of future performance.*

---

## B. Mandate Feasibility Analysis

> Constraints enforced during optimization, with active value and status. Feasible solution requires all constraints in PASS / NEAR CAP / AUTO-RELAXED.

| Constraint | Limit | Actual | Status |
|------------|-------|--------|--------|
| Region cap · US | 50.0% | 20.0% | [PASS] |
| Region cap · GCC | 25.0% | 25.0% | [AT CAP] |
| Region cap · Egypt | 5.0% | 5.0% | [AT CAP] |
| Region cap · Gold | 20.0% | 12.5% | [PASS] |
| Region cap · Bonds | 50.0% | 10.5% | [PASS] |
| Region cap · Cash | 20.0% | 20.0% | [AT CAP] |
| Region cap · Diversification | 7.0% | 7.0% | [AT CAP] |
| Beta cap (vs MSCI World) | 0.70 | 0.42 | [PASS] |
| Volatility cap (annualized) | 12.0% | 8.2% | [PASS] |
| Minimum bonds + cash floor | 30.0% | 30.5% | [AT FLOOR] |
| Holdings count | ≥ 5 | 11.0% | [PASS] |

---

## C. Risk Diagnostics

### Adaptive Risk Disclosures

*Conditional on the constructed portfolio — only risks that materially apply are surfaced.*

| Severity | Topic | Note |
|----------|-------|------|
| [MODERATE] | Cross-Currency Exposure | 30% of portfolio in non-USD-denominated holdings. SAR/AED pegged to USD; EGP floats — currency translation risk applies during reporting. |

### Structural Asset-Class Risk Factors

| Asset Class | Risk Factor |
|-------------|-------------|
| GCC / KSA | Oil price sensitivity · Geopolitical risk · Currency peg risk |
| Egypt | Currency devaluation risk · Political risk · High inflation |
| EM Bonds | Default risk · FX risk · Liquidity risk |
| Gold | No yield · Storage cost · USD-sensitive |

> *Currency risk applies to non-USD holdings (SAR pegged, AED pegged, EGP floating).*

### Drawdown Modeling Note

> Modeled max drawdown is an estimate, not a guarantee. Crisis episodes (2008, 2020, 2022) frequently exceeded model predictions by 30–50%. Stress-test against −30%, −40%, and −50% scenarios before committing capital.

### Benchmark Relative Diagnostics

*Benchmark: S&P 500 (SPY) · Reliability: Indicative · Window: 3m*

| Metric | Value | Tag |
| --- | --- | --- |
| Active Return | -0.47% | (moderate) |
| Tracking Error | 0.61% | (low) |
| Information Ratio | -0.76 | (low) |
| Rolling Alpha (12m) | +0.54% | — |
| Rolling Beta (12m) | 0.13 | — |
| Relative Drawdown | -0.23% | (low) |
| Upside Capture | 0.28 | (low) |
| Downside Capture | -2.41 | (low) |
| Relative Volatility | 0.25 | — |
| Active Share | 43.00% | (moderate) |
| Style Drift | severe | — |

**Excess Return Decomposition**

| Component | Contribution (pp) |
| --- | --- |
| Allocation Effect | +0.05 |
| Selection Effect | -0.77 |
| Factor Effect | -1.01 |
| Concentration Effect | +1.26 |

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

**Mandate:** Capital Preservation Mandate · Capital preservation · Low volatility · Long-duration anchor allocation

### Regional Allocation

| Region | Weight | ~$ on $100k | Asset Classes |
|--------|--------|-------------|---------------|
| **GCC** | 25.0% | $62,500 | 2222 (2222), 4190 (4190), ORDS (ORDS) |
| **Cash** | 20.0% | $50,000 | Cash / T-Bills |
| **US** | 20.0% | $50,000 | US Large Cap Tech, B (B) |
| **Gold** | 12.5% | $31,250 | Gold |
| **Bonds** | 10.5% | $26,250 | US Treasuries (LT), EM Bonds |
| **Diversification** | 7.0% | $17,500 | US Healthcare |
| **Egypt** | 5.0% | $12,500 | EGAL (EGAL) |

### Optimal Asset Weights

| Asset Class | Region | Weight | ~$ on $100k | Proxy | Role | Construction Rationale |
|-------------|--------|--------|-------------|-------|------|------------------------|
| **US Large Cap Tech** | US | 17.0% | $42,500 | `QQQ` | Tactical Allocation | Long-duration growth sleeve · captures secular AI/tech earnings |
| **Gold** | Gold | 12.5% | $31,250 | `GLD` | Macro Hedge | Macro hedge · equity-duration compression, USD-weakening regimes |
| **US Treasuries (LT)** | Bonds | 4.0% | $10,000 | `TLT` | Income / Diversification | Long-duration UST · negative correlation to equity in deflationary shocks |
| **EM Bonds** | Bonds | 6.5% | $16,250 | `EMB` | Income / Diversification | EM credit · spread carry with FX/default risk overlay |
| **Cash / T-Bills** | Cash | 20.0% | $50,000 | `BIL` | Income / Diversification | Dry powder · liquidity buffer and risk-free yield anchor |
| **US Healthcare** | Diversification | 7.0% | $17,500 | `XLV` | Income / Diversification | Short-duration anchor · capital preservation with low rate risk |
| **2222 (2222)** | GCC | 8.5% | $21,250 | `2222` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **4190 (4190)** | GCC | 8.5% | $21,250 | `4190` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **ORDS (ORDS)** | GCC | 8.0% | $20,000 | `ORDS` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **EGAL (EGAL)** | Egypt | 5.0% | $12,500 | `EGAL` | Satellite / Diversifier | Frontier-market tilt · structural reform exposure (higher idiosyncratic risk) |
| **B (B)** | US | 3.0% | $7,500 | `B` | Satellite / Diversifier | Energy cyclicality · inflation hedge and macro pro-cyclical exposure |

### Diversification Benefit

> **Diversification Ratio:** 2.33x — portfolio vol (8.2%) is 57% lower than weighted average of individual vols (19.0%)
> **vs Equal Weight:** Optimized vol 8.2% vs equal-weight 10.4%

### Benchmark-Relative Attribution

*Decomposition of excess return vs profile-mapped benchmark — distinguishes factor-driven from selection-driven performance.*

| Component | Value | Interpretation |
|-----------|-------|----------------|
| Excess Return vs Benchmark | +3.08% | Total active return |
| Beta Contribution | +0.53% | Factor-driven (market sensitivity differential) |
| Residual Contribution | +2.55% | Selection / concentration effects |

> *Excess return concentrated in residual/selection effects — review concentration risk before attributing to skill.*

> *Implementation note: Institutional rounding applied (5%/2.5%/1% tiered grid). 18 sub-2.5% positions consolidated. Sharpe drift: -0.00.*

---

## E. Rebalancing Plan

> Targeted reduction candidates with quantified impact on portfolio beta, volatility, and turnover.

| Proposed Action | Beta Before→After | Vol Before→After | Concentration Δ | Turnover | Difficulty |
|-----------------|-------------------|------------------|-----------------|----------|------------|
| Reduce US Large Cap Tech (QQQ) 17.0% → 10.2% | 0.425 → 0.357 (-0.068) | 8.16% → 7.69% (-0.47pp) | −6.8pp | 6.8pp | [MODERATE] |

> *Impact estimates assume pro-rata redistribution of the trimmed weight across remaining holdings.*

### Implementation Steps

1. Open a brokerage account for target markets: Global
2. Allocate $250,000 per the asset weights in Section D
3. Rebalance every 6–12 months
4. Avoid liquidating into short-term drawdowns — the 10-year horizon is designed to absorb interim market cycles


---


---

### Execution Efficiency Diagnostics

| Metric | Value | Tag |
| --- | --- | --- |
| Turnover | 100.0% | (high) |
| Implementation Shortfall | 15.9 bp | (moderate) |
| Market Impact | 2.1 bp | (low) |
| Estimated Slippage | 13.9 bp | (moderate) |
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
| soft landing | 30.0% | -8% | +27% | +76% | -14% | 5.0 |
| recession | 20.0% | -30% | +12% | +71% | -24% | 0.0 |
| stagflation | 10.0% | -27% | +17% | +83% | -25% | 0.0 |
| ai productivity boom | 10.0% | -9% | +34% | +96% | -17% | 5.0 |
| energy shock | 15.0% | -25% | +18% | +81% | -23% | 2.0 |
| liquidity crisis | 15.0% | -46% | +6% | +100% | -36% | 0.0 |

**Aggregate (probability-weighted)**

| Metric | Value |
| --- | --- |
| Expected terminal value range (real) | $0.74 - $1.81 (P10-P90) |
| Probability of loss (real) | 30% |
| Probability of target (>=4% real ann.) | 48% |
| Worst-decile terminal | $0.74 |
| Expected drawdown range | -21% to -41% |
| Recovery duration (median) | 3 months |

*Distributional framing only - outcomes reflect modelled assumptions; not a forecast.*

## G. Audit Appendix

| Field | Value |
|-------|-------|
| Snapshot ID | `8ef12d903689` |
| Universe Hash | `4b93bd0df0ea` |
| Solver | CLARABEL (cvxpy QP) |
| Solver Status | optimal |
| Assets (Universe) | 29 |
| Assets (Selected) | 11 |
| Max Beta | 0.7 |
| Max Volatility | 12.0% |
| Min Bonds + Cash | 30.0% |
| Max Drawdown (Requested) | Unconstrained% |
| Risk Aversion | 10.0 |
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
| benchmark_relative hash | 783af749b32d07c4 |
| execution_diag hash | f358d26de7ddad11 |
| factor_decomp hash | c2b9e7108234205c |
| forward_scenario hash | d517218acd2700d1 |
| benchmark_relative version | 0.2.0 |
| execution_diag version | 0.2.0 |
| factor_decomp version | 0.2.0 |
| forward_scenario version | 0.2.0 |
| phase_h version | 0.1.0 |
