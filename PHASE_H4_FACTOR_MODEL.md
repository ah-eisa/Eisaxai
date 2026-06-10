# PHASE H4 — True Factor Model Engine — Implementation Spec

**File to modify:** `/home/ubuntu/investwise/phase_h/factor_model.py` (replace stub bodies)
**Optional new file:** `/home/ubuntu/investwise/phase_h/factor_data.py` for loading FF/Carhart panels.
**Do NOT modify:** files outside `phase_h/`.

Read `PHASE_H_SPEC.md` first.

---

## 1. Factor data

### Supported models (new)

Expand `SUPPORTED_MODELS = ("FF3", "Carhart", "FF5", "FF5_QMJ", "LowVol")` to include:
- **FF3** — Mkt-RF, SMB, HML
- **Carhart** — FF3 + MOM (default)
- **FF5** — Mkt-RF, SMB, HML, RMW, CMA
- **FF5_QMJ** — FF5 + AQR Quality-Minus-Junk overlay
- **LowVol** — single-factor low-volatility (BAB or quintile spread)

Configurable via `EISAX_FACTOR_MODEL` env / `FeatureRegistry.get("factor_model")`.

### Bayesian shrinkage for sparse GCC data (mandatory per Phase H priorities)

GCC equities (KSA / UAE / GCC / Egypt) often have <24 months of clean factor
panel due to weekend/holiday calendar mismatch with FF data. When fewer than
24 observations are available for a GCC ticker:

1. Compute OLS β̂ on the available window.
2. Apply James-Stein-style shrinkage toward a region-level prior β̄:
   ```
   β_shrunk = w * β̂ + (1 - w) * β̄
   w = n_obs / (n_obs + tau)
   ```
   - `tau = 18` (effective prior weight in months — tuned for GCC sparsity)
   - `β̄` per region (institutional priors):
     - GCC equity: MKT=0.55, SMB=0.18, HML=0.20, MOM=0.05, RMW=0.10, CMA=0.05
     - Egypt:       MKT=0.40, SMB=0.25, HML=0.30, MOM=0.05, RMW=0.05, CMA=0.05
3. Mark the asset with `shrinkage_applied: True` in the per-asset diagnostic and
   raise the engine-level `reliability_tier` to `"Institutional-Lite"` minimum.

### Factor data source: Ken French Data Library. Cache locally:
- Path: `/home/ubuntu/investwise/market_cache/factor_panels/`
- Files: `ff3_monthly.csv`, `carhart_mom_monthly.csv`, `ff5_monthly.csv`
- Refresh: weekly via existing scheduler if available; otherwise on-demand fetch with 24h cache TTL.

If the cache is empty AND network fetch fails, return a degenerate `FactorDecomp` with `reliability_tier="Indicative"` and `notes=["factor data unavailable — using zero loadings"]`. Never raise.

URL pattern for first-time bootstrap (can be hard-coded):
- FF3: `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip`
- MOM: `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip`
- FF5: `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip`

If network is unavailable in the sandbox, generate a SYNTHETIC factor panel from the existing parquet snapshots (use SPY as MKT-RF proxy, regress fundamentals for SMB/HML/MOM as rough approximations) and label `reliability_tier="Indicative"`.

---

## 2. Rolling regression engine

For each `model in {"FF3","Carhart","FF5"}`:
- Default model: `"Carhart"` (4-factor).
- Rolling window: 36 months.
- Min observations: 24 months; else `Indicative` tier.

Compute portfolio monthly returns by combining the per-asset returns from the parquet cache with the input `weights` (rebalance monthly).

OLS regression: `r_p,t − rf_t = α + sum_k β_k * F_k,t + ε_t`

Outputs:
- `loadings`: most recent window's β_k
- `t_stats`: HAC-robust (Newey-West) t-stats with `lag=3`. If statsmodels not available, use plain OLS t-stats.
- `r_squared`: most recent window
- `rolling_stability`: 1 − coefficient of variation of β over all available rolling windows (averaged across factors). Clip to [0, 1].

### Contribution decomposition

- `contribution_return[k] = β_k * F_k_annualised_excess`
- `contribution_vol[k] = β_k * σ(F_k_monthly) * sqrt(12)` (annualised factor-component vol)
- `contribution_drawdown[k]`: max drawdown of `β_k * F_k,t` cumulative series

Sum across factors gives factor-explained portion of return/vol/drawdown.

### Warnings

Generate `warnings: list[str]` entries when:
- Any |β| > 1.8 → `"Hidden leverage proxy: <factor> loading {value} exceeds 1.8"`
- Any factor t-stat |t| < 1.0 AND |β| > 0.3 → `"Unstable factor exposure: <factor> β={value:.2f}, t={t:.2f}"`
- Sum of |β_SMB| + |β_HML| > 2.0 → `"Material style concentration detected"`
- `rolling_stability < 0.3` → `"Style drift detected — factor loadings shifted materially within lookback window"`
- More than 3 factors with |β| > 0.5 → `"Multi-factor crowding — diversification of factor exposure is limited"`

### reliability_tier

- `Institutional` if window == 36, R² ≥ 0.7, rolling_stability ≥ 0.6
- `Institutional-Lite` if window ≥ 24, R² ≥ 0.5
- `Indicative` otherwise

---

## 3. `render_factor_decomposition_md(payload, language)`

```
### Factor Risk Decomposition

*Model: Carhart 4-factor · R²: 0.78 · Stability: 0.81 · Reliability: Institutional · Window: 36m*

| Factor | Loading | t-stat | Contribution (ret) | Contribution (vol) |
|---|---|---|---|---|
| MKT | 0.94 | 12.3 | +8.10% | 15.20% |
| SMB | 0.12 | 1.4  | +0.20% | 1.10% |
| HML | -0.08 | -0.9 | -0.15% | 0.90% |
| MOM | 0.21 | 2.7 | +0.55% | 1.40% |

**Warnings**

- (only emitted when any warning hit)

*Loadings represent rolling 36-month exposures; t-statistics are Newey-West adjusted.*
```

Bilingual via LABELS. Add entries:
- `factor_decomposition_diagnostics`
- `loading`, `t_stat`, `contribution`, `factor` (already present — extend)
- `stability`, `r_squared_label`, `window_label`
- `factor_warnings_title`
- `factor_loadings_footnote`

---

## 4. Tests — `phase_h/tests/test_factor_model.py`

1. `test_synthetic_pure_market_loading` — synthetic returns = SPY*0.95 + noise → MKT β ~0.95, others ~0.
2. `test_carhart_returns_4_factors` and FF3/FF5 return 3/5.
3. `test_warning_emitted_for_high_beta` (set β=2.0).
4. `test_render_bilingual_complete`.
5. `test_no_forbidden_phrases`.
6. `test_short_history_demotes_tier` (12 obs).
7. `test_no_data_returns_degenerate_not_raise`.

---

## 5. Verification

```bash
cd /home/ubuntu/investwise && \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -m phase_h.tests.test_skeleton && \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -m phase_h.tests.test_factor_model && \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -c "
from global_allocator import allocate
from phase_h.orchestrator import augment_result
r = allocate(profile='balanced')
r = augment_result(r, language='en')
fac = r.get('factor_decomp') or {}
print('model:', fac.get('model'))
print('loadings:', list((fac.get('loadings') or {}).keys()))
print('Factor section:', 'Factor Risk Decomposition' in r['report_md'])
"
```

Bump `ENGINE_VERSION = "0.2.0"`.

---

## 6. NOT to do

- Do NOT introduce `statsmodels` as a hard dependency. Use it if already installed; fall back to numpy OLS.
- Do NOT modify files outside `phase_h/`.
- Do NOT emit factor-loading commentary using retail wording.
