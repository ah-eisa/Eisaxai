# EisaX Global Portfolio — Long-Horizon Growth Mandate
**Date:** May 18, 2026  |  **Capital:** $1.0M  |  **Horizon:** 8 years  |  **Markets:** Global

---

## A. Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Expected Return (annual) | **~16.2%** | [STRONG] |
| Expected Volatility | **~16.5%** | [MODERATE] |
| Sharpe Ratio | **~0.71** | [ACCEPTABLE] |
| Beta (vs MSCI World) | **~0.74** | [MODERATE] |
| Projected Value in 8 years | **$3,319,334** | Expected gain **$2,319,334** |

**Portfolio Regime:** **Multi-Asset Macro**
> Balanced cross-asset construction; no single regime dominates. Targets diversification across factor and macro drivers.
> **Regime Behavior vs Benchmark:** Designed for regime-balanced behavior; expect peer-like performance across most macro environments with reduced tail volatility.

**Confidence Calibration** · Score: **76%** · Evidence Breadth: **Broad** · Coverage Quality: **Full** · Reliability Tier: **Institutional-Lite** [MODERATE]

**Implementation Feasibility** · Deployability: **High** [STRONG] (88/100) · Rebalancing Complexity: **Moderate** [MODERATE] · Liquidity: **High** [HIGH] · Execution Friction: **Low** [LOW] · Est. Turnover ~18%/yr · Est. Slippage ~8 bp

**Benchmark Context** · Reference: **80/20 Growth (Global Equity / Bonds)** · Bench Return ~11.2% · Tracking Deviation: **High** [HIGH] (9.5% TE) · Active Share: **High** [HIGH] (61%) · Style Drift: **US-underweight · Crypto-tilted**

> *Values are approximate, derived from long-run historical assumptions. Not a guarantee of future performance.*

---

## B. Mandate Feasibility Analysis

> Constraints enforced during optimization, with active value and status. Feasible solution requires all constraints in PASS / NEAR CAP / AUTO-RELAXED.

| Constraint | Limit | Actual | Status |
|------------|-------|--------|--------|
| Region cap · US | 60.0% | 42.0% | [PASS] |
| Region cap · GCC | 35.0% | 34.0% | [NEAR CAP] |
| Region cap · Egypt | 15.0% | 8.5% | [PASS] |
| Region cap · Crypto | 10.0% | 10.0% | [AT CAP] |
| Region cap · Bonds | 20.0% | 5.5% | [PASS] |
| Beta cap (vs MSCI World) | 1.30 | 0.74 | [PASS] |
| Volatility cap (annualized) | 25.0% | 16.5% | [PASS] |
| Minimum bonds + cash floor | 5.0% | 5.5% | [AT FLOOR] |
| Holdings count | ≥ 5 | 10.0% | [PASS] |

---

## C. Risk Diagnostics

### Adaptive Risk Disclosures

*Conditional on the constructed portfolio — only risks that materially apply are surfaced.*

| Severity | Topic | Note |
|----------|-------|------|
| [HIGH] | Crypto Liquidity Discontinuity | Crypto exposure of 10% subject to 24/7 trading, regulatory regime shifts, and liquidity discontinuities during stress events. Classify as satellite, not core. |
| [MODERATE] | GCC Cyclical Concentration | GCC weighting of 34% creates oil-price and geopolitical concentration. Co-moves with commodity cycle and USD direction (peg-driven). |
| [MODERATE] | Cross-Currency Exposure | 43% of portfolio in non-USD-denominated holdings. SAR/AED pegged to USD; EGP floats — currency translation risk applies during reporting. |

### Structural Asset-Class Risk Factors

| Asset Class | Risk Factor |
|-------------|-------------|
| GCC / KSA | Oil price sensitivity · Geopolitical risk · Currency peg risk |
| Egypt | Currency devaluation risk · Political risk · High inflation |
| EM Bonds | Default risk · FX risk · Liquidity risk |

> *Currency risk applies to non-USD holdings (SAR pegged, AED pegged, EGP floating).*

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
| Active Share | 15.50% | (low) |
| Style Drift | material | — |

**Excess Return Decomposition**

| Component | Contribution (pp) |
| --- | --- |
| Allocation Effect | -0.12 |
| Selection Effect | -0.89 |
| Factor Effect | -1.13 |
| Concentration Effect | +1.20 |

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

**Mandate:** Long-Horizon Growth Mandate · High return target · Diversified global growth · Long-horizon equity tilt

### Regional Allocation

| Region | Weight | ~$ on $100k | Asset Classes |
|--------|--------|-------------|---------------|
| **US** | 42.0% | $420,000 | US Large Cap Tech, GS (GS), B (B) |
| **GCC** | 34.0% | $340,000 | 2222 (2222), 4190 (4190), FERTIGLB (FERTIGLB), ORDS (ORDS) |
| **Crypto** | 10.0% | $100,000 | Bitcoin |
| **Egypt** | 8.5% | $85,000 | EGAL (EGAL) |
| **Bonds** | 5.5% | $55,000 | EM Bonds |

### Optimal Asset Weights

| Asset Class | Region | Weight | ~$ on $100k | Proxy | Role | Construction Rationale |
|-------------|--------|--------|-------------|-------|------|------------------------|
| **US Large Cap Tech** | US | 25.0% | $250,000 | `QQQ` | Strategic Core | Long-duration growth sleeve · captures secular AI/tech earnings |
| **Bitcoin** | Crypto | 10.0% | $100,000 | `BTC-USD` | Opportunistic Satellite | Asymmetric satellite · high-volatility return contributor (not a hedge) |
| **EM Bonds** | Bonds | 5.5% | $55,000 | `EMB` | Income / Diversification | EM credit · spread carry with FX/default risk overlay |
| **2222 (2222)** | GCC | 8.5% | $85,000 | `2222` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **4190 (4190)** | GCC | 8.5% | $85,000 | `4190` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **FERTIGLB (FERTIGLB)** | GCC | 8.5% | $85,000 | `FERTIGLB` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **ORDS (ORDS)** | GCC | 8.5% | $85,000 | `ORDS` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **EGAL (EGAL)** | Egypt | 8.5% | $85,000 | `EGAL` | Tactical Allocation | Frontier-market tilt · structural reform exposure (higher idiosyncratic risk) |
| **GS (GS)** | US | 8.5% | $85,000 | `GS` | Tactical Allocation | US equity core · liquid global benchmark proxy |
| **B (B)** | US | 8.5% | $85,000 | `B` | Tactical Allocation | Energy cyclicality · inflation hedge and macro pro-cyclical exposure |

### Diversification Benefit

> **Diversification Ratio:** 1.40x — portfolio vol (16.5%) is 29% lower than weighted average of individual vols (23.1%)
> **vs Equal Weight:** Optimized vol 16.5% vs equal-weight 12.3%

### Benchmark-Relative Attribution

*Decomposition of excess return vs profile-mapped benchmark — distinguishes factor-driven from selection-driven performance.*

| Component | Value | Interpretation |
|-----------|-------|----------------|
| Excess Return vs Benchmark | +5.01% | Total active return |
| Beta Contribution | +0.12% | Factor-driven (market sensitivity differential) |
| Residual Contribution | +4.90% | Selection / concentration effects |

> *Excess return concentrated in residual/selection effects — review concentration risk before attributing to skill.*

> *Implementation note: Institutional rounding applied (5%/2.5%/1% tiered grid). 21 sub-2.5% positions consolidated. Sharpe drift: -0.00.*

---

## E. Rebalancing Plan

> Targeted reduction candidates with quantified impact on portfolio beta, volatility, and turnover.

| Proposed Action | Beta Before→After | Vol Before→After | Concentration Δ | Turnover | Difficulty |
|-----------------|-------------------|------------------|-----------------|----------|------------|
| Reduce US Large Cap Tech (QQQ) 25.0% → 15.0% | 0.745 → 0.677 (-0.067) | 16.46% → 16.54% (+0.08pp) | −10.0pp | 10.0pp | [HIGH] |

> *Impact estimates assume pro-rata redistribution of the trimmed weight across remaining holdings.*

### Implementation Steps

1. Open a brokerage account for target markets: Global
2. Allocate $1.0M per the asset weights in Section D
3. Rebalance every 6–12 months
4. Avoid liquidating into short-term drawdowns — the 8-year horizon is designed to absorb interim market cycles


---


---

### Execution Efficiency Diagnostics

| Metric | Value | Tag |
| --- | --- | --- |
| Turnover | 100.0% | (high) |
| Implementation Shortfall | 21.2 bp | (moderate) |
| Market Impact | 3.7 bp | (low) |
| Estimated Slippage | 19.2 bp | (moderate) |
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
| soft landing | 30.0% | -12% | +44% | +133% | -24% | 4.0 |
| recession | 20.0% | -47% | +9% | +127% | -39% | 0.0 |
| stagflation | 10.0% | -46% | +19% | +161% | -40% | 0.0 |
| ai productivity boom | 10.0% | -18% | +55% | +184% | -28% | 4.0 |
| energy shock | 15.0% | -46% | +15% | +140% | -38% | 0.0 |
| liquidity crisis | 15.0% | -63% | +5% | +183% | -52% | 0.0 |

**Aggregate (probability-weighted)**

| Metric | Value |
| --- | --- |
| Expected terminal value range (real) | $0.58 - $2.45 (P10-P90) |
| Probability of loss (real) | 33% |
| Probability of target (>=4% real ann.) | 54% |
| Worst-decile terminal | $0.58 |
| Expected drawdown range | -34% to -61% |
| Recovery duration (median) | 0 months |

*Distributional framing only - outcomes reflect modelled assumptions; not a forecast.*

## G. Audit Appendix

| Field | Value |
|-------|-------|
| Snapshot ID | `cb32951a8bc2` |
| Universe Hash | `36d2e7a59a86` |
| Solver | CLARABEL (cvxpy QP) |
| Solver Status | optimal |
| Assets (Universe) | 31 |
| Assets (Selected) | 10 |
| Max Beta | 1.3 |
| Max Volatility | 25.0% |
| Min Bonds + Cash | 5.0% |
| Max Drawdown (Requested) | Unconstrained% |
| Risk Aversion | 1.5 |
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
| benchmark_relative hash | 968407ff4b70acce |
| execution_diag hash | da78de3a786ada93 |
| factor_decomp hash | 589884420dfffeae |
| forward_scenario hash | 42ab101e4f69a77c |
| benchmark_relative version | 0.2.0 |
| execution_diag version | 0.2.0 |
| factor_decomp version | 0.2.0 |
| forward_scenario version | 0.2.0 |
| phase_h version | 0.1.0 |
