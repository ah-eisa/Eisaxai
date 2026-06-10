# EisaX Global Portfolio — Balanced Multi-Asset Mandate
**Date:** May 18, 2026  |  **Capital:** $100,000  |  **Horizon:** 5 years  |  **Markets:** Global

---

## A. Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Expected Return (annual) | **~14.1%** | [STRONG] |
| Expected Volatility | **~12.7%** | [MODERATE] |
| Sharpe Ratio | **~0.76** | [ACCEPTABLE] |
| Beta (vs MSCI World) | **~0.66** | [LOW] |
| Projected Value in 5 years | **$193,303** | Expected gain **$93,303** |

**Portfolio Regime:** **Multi-Asset Macro**
> Balanced cross-asset construction; no single regime dominates. Targets diversification across factor and macro drivers.
> **Regime Behavior vs Benchmark:** Designed for regime-balanced behavior; expect peer-like performance across most macro environments with reduced tail volatility.

**Confidence Calibration** · Score: **78%** · Evidence Breadth: **Broad** · Coverage Quality: **Full** · Reliability Tier: **Institutional-Lite** [MODERATE]

**Implementation Feasibility** · Deployability: **Moderate** [MODERATE] (71/100) · Rebalancing Complexity: **High** [HIGH] · Liquidity: **High** [HIGH] · Execution Friction: **Low** [LOW] · Est. Turnover ~21%/yr · Est. Slippage ~6 bp

**Benchmark Context** · Reference: **60/40 Balanced (Global Equity / Bonds, Gold overlay)** · Bench Return ~10.0% · Tracking Deviation: **High** [HIGH] (7.2% TE) · Active Share: **High** [HIGH] (75%) · Style Drift: **US-underweight · Duration-underweight · Crypto-tilted**

> *Values are approximate, derived from long-run historical assumptions. Not a guarantee of future performance.*

---

## B. Mandate Feasibility Analysis

> Constraints enforced during optimization, with active value and status. Feasible solution requires all constraints in PASS / NEAR CAP / AUTO-RELAXED.

| Constraint | Limit | Actual | Status |
|------------|-------|--------|--------|
| Region cap · US | 50.0% | 34.5% | [PASS] |
| Region cap · GCC | 30.0% | 30.0% | [AT CAP] |
| Region cap · Egypt | 10.0% | 8.5% | [NEAR CAP] |
| Region cap · Crypto | 10.0% | 5.0% | [PASS] |
| Region cap · Bonds | 35.0% | 5.0% | [PASS] |
| Region cap · Cash | 10.0% | 10.0% | [AT CAP] |
| Region cap · Diversification | 7.0% | 7.0% | [AT CAP] |
| Beta cap (vs MSCI World) | 1.00 | 0.66 | [PASS] |
| Volatility cap (annualized) | 18.0% | 12.7% | [PASS] |
| Minimum bonds + cash floor | 15.0% | 15.0% | [AT FLOOR] |
| Holdings count | ≥ 5 | 13.0% | [PASS] |

---

## C. Risk Diagnostics

### Adaptive Risk Disclosures

*Conditional on the constructed portfolio — only risks that materially apply are surfaced.*

| Severity | Topic | Note |
|----------|-------|------|
| [MODERATE] | Cross-Currency Exposure | 38% of portfolio in non-USD-denominated holdings. SAR/AED pegged to USD; EGP floats — currency translation risk applies during reporting. |

### Structural Asset-Class Risk Factors

| Asset Class | Risk Factor |
|-------------|-------------|
| GCC / KSA | Oil price sensitivity · Geopolitical risk · Currency peg risk |
| Egypt | Currency devaluation risk · Political risk · High inflation |
| EM Bonds | Default risk · FX risk · Liquidity risk |

> *Currency risk applies to non-USD holdings (SAR pegged, AED pegged, EGP floating).*

### Crypto Analytical Framework

> Crypto positions (5% of portfolio) are evaluated using a separate analytical lens:
> *network activity · ETF flows · realized volatility · liquidity regime · cycle positioning.* Equity valuation multiples and earnings-quality metrics are not applicable.

### Drawdown Modeling Note

> Modeled max drawdown is an estimate, not a guarantee. Crisis episodes (2008, 2020, 2022) frequently exceeded model predictions by 30–50%. Stress-test against −30%, −40%, and −50% scenarios before committing capital.

### Benchmark Relative Diagnostics

*Benchmark: S&P 500 (SPY) · Reliability: Indicative · Window: 3m*

| Metric | Value | Tag |
| --- | --- | --- |
| Active Return | 0.00% | (moderate) |
| Tracking Error | 0.00% | (low) |
| Information Ratio | 0.00 | (low) |
| Rolling Alpha (12m) | 0.00% | — |
| Rolling Beta (12m) | 1.00 | — |
| Relative Drawdown | 0.00% | (low) |
| Upside Capture | 0.00 | (low) |
| Downside Capture | 0.00 | (low) |
| Relative Volatility | 0.00 | — |
| Active Share | 19.98% | (low) |
| Style Drift | material | — |

**Excess Return Decomposition**

| Component | Contribution (pp) |
| --- | --- |
| Allocation Effect | 0.00 |
| Selection Effect | 0.00 |
| Factor Effect | 0.00 |
| Concentration Effect | 0.00 |

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
| **US** | 34.5% | $34,500 | US Large Cap Tech, VLO (VLO), MO (MO) |
| **GCC** | 30.0% | $30,000 | 2222 (2222), 4190 (4190), FERTIGLB (FERTIGLB), AGILITY (AGILITY) |
| **Cash** | 10.0% | $10,000 | Cash / T-Bills |
| **Egypt** | 8.5% | $8,500 | ETEL (ETEL) |
| **Diversification** | 7.0% | $7,000 | US Healthcare |
| **Bonds** | 5.0% | $5,000 | US Treasuries (LT), EM Bonds |
| **Crypto** | 5.0% | $5,000 | Bitcoin |

### Optimal Asset Weights

| Asset Class | Region | Weight | ~$ on $100k | Proxy | Role | Construction Rationale |
|-------------|--------|--------|-------------|-------|------|------------------------|
| **US Large Cap Tech** | US | 25.0% | $24,988 | `QQQ` | Strategic Core | Long-duration growth sleeve · captures secular AI/tech earnings |
| **Bitcoin** | Crypto | 5.0% | $4,986 | `BTC-USD` | Opportunistic Satellite | Asymmetric satellite · high-volatility return contributor (not a hedge) |
| **US Treasuries (LT)** | Bonds | 1.8% | $1,752 | `TLT` | Income / Diversification | Long-duration UST · negative correlation to equity in deflationary shocks |
| **EM Bonds** | Bonds | 3.2% | $3,248 | `EMB` | Income / Diversification | EM credit · spread carry with FX/default risk overlay |
| **Cash / T-Bills** | Cash | 10.0% | $10,000 | `BIL` | Income / Diversification | Dry powder · liquidity buffer and risk-free yield anchor |
| **US Healthcare** | Diversification | 7.0% | $7,000 | `XLV` | Income / Diversification | Short-duration anchor · capital preservation with low rate risk |
| **2222 (2222)** | GCC | 8.5% | $8,500 | `2222` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **4190 (4190)** | GCC | 8.5% | $8,500 | `4190` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **FERTIGLB (FERTIGLB)** | GCC | 8.5% | $8,500 | `FERTIGLB` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **AGILITY (AGILITY)** | GCC | 4.5% | $4,500 | `AGILITY` | Satellite / Diversifier | Regional exposure · GCC growth premium, low correlation to US equities |
| **ETEL (ETEL)** | Egypt | 8.5% | $8,466 | `ETEL` | Tactical Allocation | Frontier-market tilt · structural reform exposure (higher idiosyncratic risk) |
| **VLO (VLO)** | US | 1.1% | $1,060 | `VLO` | Satellite / Diversifier | Energy cyclicality · inflation hedge and macro pro-cyclical exposure |
| **MO (MO)** | US | 8.5% | $8,500 | `MO` | Tactical Allocation | US equity core · liquid global benchmark proxy |

### Diversification Benefit

> **Diversification Ratio:** 1.82x — portfolio vol (12.7%) is 45% lower than weighted average of individual vols (23.1%)
> **vs Equal Weight:** Optimized vol 12.7% vs equal-weight 12.3%

### Benchmark-Relative Attribution

*Decomposition of excess return vs profile-mapped benchmark — distinguishes factor-driven from selection-driven performance.*

| Component | Value | Interpretation |
|-----------|-------|----------------|
| Excess Return vs Benchmark | +4.06% | Total active return |
| Beta Contribution | +0.67% | Factor-driven (market sensitivity differential) |
| Residual Contribution | +3.39% | Selection / concentration effects |

> *Excess return concentrated in residual/selection effects — review concentration risk before attributing to skill.*

> *Implementation note: Rounded portfolio failed institutional check (cap breach); raw optimizer weights preserved.*

---

## E. Rebalancing Plan

> Targeted reduction candidates with quantified impact on portfolio beta, volatility, and turnover.

| Proposed Action | Beta Before→After | Vol Before→After | Concentration Δ | Turnover | Difficulty |
|-----------------|-------------------|------------------|-----------------|----------|------------|
| Reduce US Large Cap Tech (QQQ) 25.0% → 15.0% | 0.661 → 0.583 (-0.078) | 12.68% → 12.20% (-0.48pp) | −10.0pp | 10.0pp | [MODERATE] |

> *Impact estimates assume pro-rata redistribution of the trimmed weight across remaining holdings.*

### Implementation Steps

1. Open a brokerage account for target markets: Global
2. Allocate $100,000 per the asset weights in Section D
3. Rebalance every 6–12 months
4. Avoid liquidating into short-term drawdowns — the 5-year horizon is designed to absorb interim market cycles


---


---

### Execution Efficiency Diagnostics

| Metric | Value | Tag |
| --- | --- | --- |
| Turnover | 100.0% | (high) |
| Implementation Shortfall | 19.3 bp | (moderate) |
| Market Impact | 3.0 bp | (low) |
| Estimated Slippage | 17.3 bp | (moderate) |
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
| soft landing | 30.0% | -11% | +38% | +111% | -20% | 5.0 |
| recession | 20.0% | -42% | +9% | +95% | -32% | 0.0 |
| stagflation | 10.0% | -37% | +19% | +125% | -32% | 0.0 |
| ai productivity boom | 10.0% | -14% | +48% | +149% | -23% | 4.0 |
| energy shock | 15.0% | -37% | +14% | +104% | -32% | 0.0 |
| liquidity crisis | 15.0% | -53% | +10% | +146% | -43% | 0.0 |

**Aggregate (probability-weighted)**

| Metric | Value |
| --- | --- |
| Expected terminal value range (real) | $0.67 - $2.18 (P10-P90) |
| Probability of loss (real) | 32% |
| Probability of target (>=4% real ann.) | 53% |
| Worst-decile terminal | $0.67 |
| Expected drawdown range | -28% to -51% |
| Recovery duration (median) | 0 months |

*Distributional framing only - outcomes reflect modelled assumptions; not a forecast.*

## G. Audit Appendix

| Field | Value |
|-------|-------|
| Snapshot ID | `32433cfd0622` |
| Universe Hash | `7f2f3a5be95f` |
| Solver | CLARABEL (cvxpy QP) |
| Solver Status | optimal |
| Assets (Universe) | 31 |
| Assets (Selected) | 13 |
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
| benchmark_relative hash | 5137ab2f5b722388 |
| execution_diag hash | d15595531f8dfe04 |
| factor_decomp hash | dce0949c82c972af |
| forward_scenario hash | b22e32a33e6ad457 |
| benchmark_relative version | 0.2.0 |
| execution_diag version | 0.2.0 |
| factor_decomp version | 0.2.0 |
| forward_scenario version | 0.2.0 |
| phase_h version | 0.1.0 |
