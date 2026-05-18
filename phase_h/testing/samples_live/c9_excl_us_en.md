# EisaX Global Portfolio — Long-Horizon Growth Mandate
**Date:** May 18, 2026  |  **Capital:** $200,000  |  **Horizon:** 8 years  |  **Markets:** Global

---

## A. Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Expected Return (annual) | **~13.7%** | [STRONG] |
| Expected Volatility | **~13.9%** | [MODERATE] |
| Sharpe Ratio | **~0.66** | [ACCEPTABLE] |
| Beta (vs MSCI World) | **~0.37** | [LOW] |
| Projected Value in 8 years | **$558,616** | Expected gain **$358,616** |

**Portfolio Regime:** **Cyclical Value**
> Allocation tilted toward commodity-linked and emerging-market cyclical exposure. Procyclical with global PMI cycle.
> **Regime Behavior vs Benchmark:** Outperforms during commodity-led reflationary cycles and global PMI expansions; lags in growth-led rallies and risk-off episodes.

**Confidence Calibration** · Score: **73%** · Evidence Breadth: **Broad** · Coverage Quality: **Full** · Reliability Tier: **Indicative** [LOW]

**Implementation Feasibility** · Deployability: **Moderate** [MODERATE] (65/100) · Rebalancing Complexity: **High** [HIGH] · Liquidity: **High** [HIGH] · Execution Friction: **Low** [LOW] · Est. Turnover ~24%/yr · Est. Slippage ~9 bp

**Benchmark Context** · Reference: **80/20 Growth (Global Equity / Bonds)** · Bench Return ~9.0% · Tracking Deviation: **High** [HIGH] (9.7% TE) · Active Share: **High** [HIGH] (58%) · Style Drift: **Duration-underweight · Crypto-tilted**

> *Values are approximate, derived from long-run historical assumptions. Not a guarantee of future performance.*

---

## B. Mandate Feasibility Analysis

> Constraints enforced during optimization, with active value and status. Feasible solution requires all constraints in PASS / NEAR CAP / AUTO-RELAXED.

| Constraint | Limit | Actual | Status |
|------------|-------|--------|--------|
| Region cap · GCC | 35.0% | 35.0% | [AT CAP] |
| Region cap · Egypt | 15.0% | 15.0% | [AT CAP] |
| Region cap · Crypto | 10.0% | 10.0% | [AT CAP] |
| Region cap · Gold | 15.0% | 15.0% | [AT CAP] |
| Region cap · Bonds | 20.0% | 5.3% | [PASS] |
| Region cap · Commodities | 15.0% | 12.7% | [NEAR CAP] |
| Region cap · Diversification | 7.0% | 7.0% | [AT CAP] |
| Beta cap (vs MSCI World) | 1.30 | 0.37 | [PASS] |
| Volatility cap (annualized) | 25.0% | 13.9% | [PASS] |
| Minimum bonds + cash floor | 5.0% | 5.3% | [AT FLOOR] |
| Holdings count | ≥ 5 | 14.0% | [PASS] |

---

## C. Risk Diagnostics

### Adaptive Risk Disclosures

*Conditional on the constructed portfolio — only risks that materially apply are surfaced.*

| Severity | Topic | Note |
|----------|-------|------|
| [HIGH] | Crypto Liquidity Discontinuity | Crypto exposure of 10% subject to 24/7 trading, regulatory regime shifts, and liquidity discontinuities during stress events. Classify as satellite, not core. |
| [MODERATE] | GCC Cyclical Concentration | GCC weighting of 35% creates oil-price and geopolitical concentration. Co-moves with commodity cycle and USD direction (peg-driven). |
| [MODERATE] | Cross-Currency Exposure | 50% of portfolio in non-USD-denominated holdings. SAR/AED pegged to USD; EGP floats — currency translation risk applies during reporting. |

### Structural Asset-Class Risk Factors

| Asset Class | Risk Factor |
|-------------|-------------|
| GCC / KSA | Oil price sensitivity · Geopolitical risk · Currency peg risk |
| Egypt | Currency devaluation risk · Political risk · High inflation |
| EM Bonds | Default risk · FX risk · Liquidity risk |
| Gold | No yield · Storage cost · USD-sensitive |

> *Currency risk applies to non-USD holdings (SAR pegged, AED pegged, EGP floating).*

### Correlation Cluster Risk

| Severity | Cluster | Combined Weight | Note |
|----------|---------|-----------------|------|
| [HIGH] | GCC + Commodities | 47.7% | Oil/commodity cycle co-movement |

### Crypto Analytical Framework

> Crypto positions (10% of portfolio) are evaluated using a separate analytical lens:
> *network activity · ETF flows · realized volatility · liquidity regime · cycle positioning.* Equity valuation multiples and earnings-quality metrics are not applicable.

### Drawdown Modeling Note

> Modeled max drawdown is an estimate, not a guarantee. Crisis episodes (2008, 2020, 2022) frequently exceeded model predictions by 30–50%. Stress-test against −30%, −40%, and −50% scenarios before committing capital.

### Benchmark Relative Diagnostics

*Benchmark: S&P 500 (SPY) · Reliability: Indicative · Window: 3m*

| Metric | Value | Tag |
| --- | --- | --- |
| Active Return | -1.39% | (moderate) |
| Tracking Error | 0.57% | (low) |
| Information Ratio | -2.43 | (low) |
| Rolling Alpha (12m) | -0.44% | — |
| Rolling Beta (12m) | 0.18 | — |
| Relative Drawdown | -0.35% | (low) |
| Upside Capture | 0.06 | (low) |
| Downside Capture | 2.24 | (elevated) |
| Relative Volatility | 0.25 | — |
| Active Share | 43.00% | (moderate) |
| Style Drift | severe | — |

**Excess Return Decomposition**

| Component | Contribution (pp) |
| --- | --- |
| Allocation Effect | -0.84 |
| Selection Effect | -1.04 |
| Factor Effect | -0.95 |
| Concentration Effect | +1.44 |

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
| **GCC** | 35.0% | $70,000 | 2222 (2222), 4190 (4190), 5110 (5110), FERTIGLB (FERTIGLB), ORDS (ORDS) |
| **Egypt** | 15.0% | $30,000 | TMGH (TMGH), EGAL (EGAL) |
| **Gold** | 15.0% | $30,000 | Gold |
| **Commodities** | 12.7% | $25,400 | Crude Oil, Silver, Copper |
| **Crypto** | 10.0% | $20,000 | Bitcoin |
| **Diversification** | 7.0% | $14,000 | US Healthcare |
| **Bonds** | 5.3% | $10,600 | EM Bonds |

### Optimal Asset Weights

| Asset Class | Region | Weight | ~$ on $100k | Proxy | Role | Construction Rationale |
|-------------|--------|--------|-------------|-------|------|------------------------|
| **Bitcoin** | Crypto | 10.0% | $20,000 | `BTC-USD` | Opportunistic Satellite | Asymmetric satellite · high-volatility return contributor (not a hedge) |
| **Gold** | Gold | 15.0% | $30,000 | `GLD` | Macro Hedge | Macro hedge · equity-duration compression, USD-weakening regimes |
| **EM Bonds** | Bonds | 5.3% | $10,570 | `EMB` | Income / Diversification | EM credit · spread carry with FX/default risk overlay |
| **US Healthcare** | Diversification | 7.0% | $14,000 | `XLV` | Income / Diversification | Short-duration anchor · capital preservation with low rate risk |
| **Crude Oil** | Commodities | 1.9% | $3,814 | `USO` | Real-Asset / Inflation Sleeve | Real-asset exposure · inflation pass-through and macro cyclicality |
| **Silver** | Commodities | 0.8% | $1,616 | `SLV` | Real-Asset / Inflation Sleeve | Real-asset exposure · inflation pass-through and macro cyclicality |
| **Copper** | Commodities | 10.0% | $20,000 | `CPER` | Real-Asset / Inflation Sleeve | Real-asset exposure · inflation pass-through and macro cyclicality |
| **2222 (2222)** | GCC | 8.5% | $17,000 | `2222` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **4190 (4190)** | GCC | 8.5% | $17,000 | `4190` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **5110 (5110)** | GCC | 1.0% | $2,000 | `5110` | Satellite / Diversifier | Regional exposure · GCC growth premium, low correlation to US equities |
| **FERTIGLB (FERTIGLB)** | GCC | 8.5% | $17,000 | `FERTIGLB` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **ORDS (ORDS)** | GCC | 8.5% | $17,000 | `ORDS` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **TMGH (TMGH)** | Egypt | 6.5% | $13,000 | `TMGH` | Satellite / Diversifier | Frontier-market tilt · structural reform exposure (higher idiosyncratic risk) |
| **EGAL (EGAL)** | Egypt | 8.5% | $17,000 | `EGAL` | Tactical Allocation | Frontier-market tilt · structural reform exposure (higher idiosyncratic risk) |

### Diversification Benefit

> **Diversification Ratio:** 1.74x — portfolio vol (13.9%) is 43% lower than weighted average of individual vols (24.3%)
> **vs Equal Weight:** Optimized vol 13.9% vs equal-weight 13.0%

### Benchmark-Relative Attribution

*Decomposition of excess return vs profile-mapped benchmark — distinguishes factor-driven from selection-driven performance.*

| Component | Value | Interpretation |
|-----------|-------|----------------|
| Excess Return vs Benchmark | +4.68% | Total active return |
| Beta Contribution | +0.84% | Factor-driven (market sensitivity differential) |
| Residual Contribution | +3.84% | Selection / concentration effects |

> *Excess return concentrated in residual/selection effects — review concentration risk before attributing to skill.*

> *Implementation note: Rounded portfolio failed institutional check (cap breach); raw optimizer weights preserved.*

---

## E. Rebalancing Plan

> No concentrated positions above 15% — no immediate rebalancing action required.


### Implementation Steps

1. Open a brokerage account for target markets: Global
2. Allocate $200,000 per the asset weights in Section D
3. Rebalance every 6–12 months
4. Avoid liquidating into short-term drawdowns — the 8-year horizon is designed to absorb interim market cycles


---


---

### Execution Efficiency Diagnostics

| Metric | Value | Tag |
| --- | --- | --- |
| Turnover | 100.0% | (high) |
| Implementation Shortfall | 24.4 bp | (moderate) |
| Market Impact | 3.6 bp | (low) |
| Estimated Slippage | 22.4 bp | (moderate) |
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
| soft landing | 30.0% | -18% | +34% | +116% | -23% | 4.0 |
| recession | 20.0% | -50% | +8% | +119% | -39% | 0.0 |
| stagflation | 10.0% | -47% | +16% | +148% | -40% | 0.0 |
| ai productivity boom | 10.0% | -31% | +29% | +138% | -30% | 2.0 |
| energy shock | 15.0% | -42% | +24% | +155% | -37% | 0.0 |
| liquidity crisis | 15.0% | -68% | -5% | +165% | -55% | 0.0 |

**Aggregate (probability-weighted)**

| Metric | Value |
| --- | --- |
| Expected terminal value range (real) | $0.56 - $2.38 (P10-P90) |
| Probability of loss (real) | 36% |
| Probability of target (>=4% real ann.) | 50% |
| Worst-decile terminal | $0.56 |
| Expected drawdown range | -35% to -62% |
| Recovery duration (median) | 0 months |

*Distributional framing only - outcomes reflect modelled assumptions; not a forecast.*

## G. Audit Appendix

| Field | Value |
|-------|-------|
| Snapshot ID | `daddfe017ccf` |
| Universe Hash | `71750f926b11` |
| Solver | CLARABEL (cvxpy QP) |
| Solver Status | optimal |
| Assets (Universe) | 24 |
| Assets (Selected) | 14 |
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
| benchmark_relative hash | 81b1a4f9243473b9 |
| execution_diag hash | b79495e85683df8b |
| factor_decomp hash | 92abed9fd9231b3a |
| forward_scenario hash | 4363a2381cb61c6d |
| benchmark_relative version | 0.2.0 |
| execution_diag version | 0.2.0 |
| factor_decomp version | 0.2.0 |
| forward_scenario version | 0.2.0 |
| phase_h version | 0.1.0 |
