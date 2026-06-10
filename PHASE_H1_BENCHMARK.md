# PHASE H1 — Native Benchmark Analytics Engine — Implementation Spec

**File to modify:** `/home/ubuntu/investwise/phase_h/benchmarks.py`
**Optional new file:** `/home/ubuntu/investwise/phase_h/benchmark_data.py` for data loading
**Do NOT modify:** any file outside `phase_h/` for H1.

Read `PHASE_H_SPEC.md` first for non-negotiable rules (tone discipline, bilingual EN/AR, schema preservation, anchor markers).

The current stub at `phase_h/benchmarks.py` defines the public API. Keep the same function signatures; replace the stub bodies with the full implementation below.

---

## 1. Benchmark catalog (already present — extend if needed)

Keep the catalog dict. Add fallback proxies if a primary ticker has no parquet snapshot:

| Ticker  | Label                       | Region | Kind         | Fallback |
|---------|-----------------------------|--------|--------------|----------|
| SPY     | S&P 500 (SPY)               | US     | equity       | VOO, ^GSPC |
| URTH    | MSCI World (URTH)           | DM     | equity       | ACWI, VT |
| AOR     | 60/40 Balanced (AOR)        | GLB    | balanced     | synthetic 60% URTH + 40% AGG |
| ^TASI   | Tadawul (^TASI)             | KSA    | equity       | KSA (iShares MSCI Saudi) |
| BTC-USD | Bitcoin (BTC-USD)           | GLB    | crypto       | BTC, IBIT |
| ^BCOM   | Bloomberg Commodity (BCOM)  | GLB    | commodities  | DBC, GSG |

Implement `pick_benchmark(region_tilt, asset_kind)` policy:
1. asset_kind explicit wins (`crypto` → BTC-USD; `commodities` → ^BCOM; `balanced` → AOR).
2. region_tilt mapping: `US` → SPY, `KSA`/`GCC` (if KSA-heavy) → ^TASI, `DM`/`World`/`GLB` → URTH.
3. Mixed portfolios: pick the benchmark whose region matches the **largest single-region weight** in the portfolio. If no region dominates (>40%), default to URTH.
4. If chosen ticker has no return data, walk the fallback chain. If everything fails, return URTH and add a note `"benchmark data unavailable — defaulted to URTH"`.

---

## 2. Data loading

Reuse the existing parquet cache at `/home/ubuntu/investwise/market_cache/` (snapshots refreshed every 15min). Grep `global_allocator.py` for `_load_latest_snapshot` to understand the existing pattern — DO NOT duplicate; import or adapt.

If a benchmark series is not in the cache, attempt fallback chain. If still missing, return the degenerate payload with `reliability_tier = "Indicative"` and a `notes` entry — never raise.

For weights mapping, the portfolio is a region-bucket allocation (USA/GCC/EGY/BTC/ETH/GLD/TLT/EMB/Cash/OIL/SLV/COPR). To compute returns, map each bucket to its proxy ticker (same mapping `global_allocator.py` already uses at the top of the file — `AssetClass(... ticker ...)`).

---

## 3. Compute every metric in the BenchmarkRelative TypedDict

For a 36-month rolling window (or longest available, ≥12 months min — otherwise tier = "Indicative"):

- **active_return_pct**: `(annualised portfolio return − annualised benchmark return) * 100`
- **tracking_error_pct**: `std(monthly active returns) * sqrt(12) * 100`
- **information_ratio**: `active_return / tracking_error` (annualised, dimensionless)
- **rolling_alpha_12m_pct**: alpha from CAPM regression over trailing 12m, annualised, in %
- **rolling_beta_12m**: beta from same regression
- **relative_drawdown_pct**: max drawdown of the cumulative active return series, in %
- **upside_capture**: avg portfolio return in months where benchmark > 0, divided by avg benchmark return in those months
- **downside_capture**: same logic for benchmark < 0 months
- **relative_volatility**: portfolio_vol / benchmark_vol
- **active_share_pct**: `0.5 * sum(|w_portfolio − w_benchmark_implied|) * 100`. Use the profile-mapped benchmark composition already computed in `global_allocator.py` (`_bench_w` near line 866) — reuse it.
- **style_drift**: classify as `"aligned"`, `"mild"`, `"material"`, `"severe"` based on Euclidean distance vs benchmark region tilts (`<5%`, `5-15%`, `15-30%`, `>30%`).

### Excess decomposition (4 components, all in %; must sum approximately to active_return_pct)

- **allocation effect**: `sum_i (w_p,i − w_b,i) * (r_b,i − r_b_total)` over regions
- **selection effect**: `sum_i w_b,i * (r_p,i − r_b,i)` over regions
- **factor effect**: portion of active return attributable to FF3 market beta deviation (approximate when factor engine output not yet available — use `(beta − 1) * benchmark_excess_return`)
- **concentration effect**: residual after the three above. Sign-matters; record as the unattributed remainder.

### Regime behavior

Classify environments using the existing macro regime tags from `core/market_regimes.py`. Build two lists:
- `outperform_envs`: regimes where rolling 6m active return median > 0 (e.g. `["risk-on equity rally", "USD weakness"]`)
- `lag_envs`: regimes where rolling 6m active return median < 0

If insufficient history, leave both lists empty and add a note.

### reliability_tier

- `Institutional` if ≥36 months of overlap, R² ≥ 0.5 on the rolling alpha regression
- `Institutional-Lite` if 18–35 months OR R² 0.3–0.5
- `Indicative` otherwise

---

## 4. `render_benchmark_relative_md(payload, language)`

Replace the stub. Output structure (EN; AR is identical with translated labels via `report_helpers.L(...)`):

```
### Benchmark Relative Diagnostics

*Benchmark: <label> · Reliability: <tier> · Window: <N>m*

| Metric | Value | Tag |
|---|---|---|
| Active Return | +X.XX% | (moderate) |
| Tracking Error | X.XX% | (low/moderate/elevated) |
| Information Ratio | 0.XX | (low/moderate/elevated) |
| Rolling Alpha (12m) | +X.XX% | — |
| Rolling Beta (12m) | 0.XX | — |
| Relative Drawdown | -X.XX% | (low/moderate/elevated) |
| Upside Capture | 0.XX | (low/moderate/elevated) |
| Downside Capture | 0.XX | (low/moderate/elevated) |
| Relative Volatility | 0.XX | — |
| Active Share | XX% | (low/moderate/elevated) |
| Style Drift | aligned/mild/material/severe | — |

**Excess Return Decomposition**

| Component | Contribution (pp) |
|---|---|
| Allocation Effect | +X.XX |
| Selection Effect | +X.XX |
| Factor Effect | +X.XX |
| Concentration Effect | +X.XX |

**Benchmark-Relative Regime Behavior**

- Environments where portfolio likely outperforms: <list, comma-separated, or "insufficient regime history">
- Environments where portfolio likely lags: <list, or "insufficient regime history">

<one institutional commentary line, e.g. "Tracking error is elevated relative to benchmark composition; active share above 60% indicates material structural deviation that may amplify factor-driven dispersion in stress regimes.">
```

Severity-tag thresholds (use `severity_tag()` from `report_helpers`):
- Tracking Error: `<3%` low, `3-6%` moderate, `>6%` elevated
- Information Ratio: `<0.2` low, `0.2-0.5` moderate, `>0.5` elevated (sign matters; negative IR → low)
- Active Share: `<30%` low, `30-60%` moderate, `>60%` elevated
- Upside/Downside Capture: out-of-range bands `<0.8 / 0.8-1.2 / >1.2`
- Relative Drawdown: `<5%` low, `5-15%` moderate, `>15%` elevated

All numeric formatting must use `fmt_pct` / `fmt_num` from `report_helpers`. All Arabic strings must come from the `LABELS` dictionary — extend the dictionary if you need a new term, never inline AR strings.

---

## 5. Tone & forbidden phrases

Final markdown must pass through `phase_h.tone_guard.scrub_text` automatically (orchestrator already does this). Do NOT use forbidden phrases anywhere in commentary lines — see PHASE_H_SPEC.md section "Non-negotiable preservation rules" item 2.

---

## 6. Tests to add

Create `/home/ubuntu/investwise/phase_h/tests/test_benchmarks.py` with:

1. `test_pick_benchmark_policy` — crypto → BTC-USD; KSA-heavy → ^TASI; US-heavy → SPY; mixed/no-dominant → URTH.
2. `test_compute_with_synthetic_panel` — construct a 60-month synthetic returns panel (portfolio = 0.6 SPY + 0.4 TLT, benchmark = SPY), assert:
   - `tracking_error_pct > 0`
   - `active_return_pct ≈ 0.4 * (TLT_ann − SPY_ann)` within 1pp
   - `excess_decomp` sums approximately to `active_return_pct` (within 0.5pp)
   - `reliability_tier == "Institutional"`
3. `test_short_history_demotes_tier` — feed 6 months, expect `Indicative`.
4. `test_render_includes_all_metric_labels_en_and_ar` — render twice, assert every required label appears in each language.
5. `test_no_forbidden_phrases_after_scrub` — render then run `tone_guard.audit_block`, assert zero hits.
6. `test_feasibility_failure_skipped_by_orchestrator` — call `augment_result` with `feasibility={"status":"infeasible"}`, assert no benchmark_relative key added.

Run via:
```
cd /home/ubuntu/investwise && /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -m phase_h.tests.test_benchmarks
```

Tests should print `OK: <module> (N/N)` on success and return exit 0. If you use pytest assertions, also support a `__main__` block that runs all tests sequentially and prints a single OK/FAIL line.

---

## 7. Engine version + reliability

When you complete, bump `ENGINE_VERSION = "0.2.0"` in `benchmarks.py`. The orchestrator records this in the audit appendix automatically.

---

## 8. What NOT to do

- Do NOT modify `global_allocator.py`, `portfolio_builder.py`, or `staging.py` in H1.
- Do NOT modify `phase_h/orchestrator.py` unless you need to pass new kwargs (avoid if possible).
- Do NOT introduce new dependencies — use `numpy`, `pandas`, existing parquet reader, `scipy` (already vendored).
- Do NOT touch the A–G markdown structure.
- Do NOT use emojis or any of the forbidden retail phrases.
- Do NOT raise on missing data — degrade gracefully with `reliability_tier="Indicative"` and a `notes` entry.

---

## 9. Final verification you must run before declaring done

```bash
cd /home/ubuntu/investwise && \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -m phase_h.tests.test_skeleton && \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -m phase_h.tests.test_benchmarks && \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -c "
from global_allocator import allocate
from phase_h.orchestrator import augment_result
r = allocate(profile='balanced')
r = augment_result(r, language='en')
print('keys:', sorted(r.keys()))
print('benchmark_relative tier:', r.get('benchmark_relative',{}).get('reliability_tier'))
print('Section presence:', 'Benchmark Relative Diagnostics' in r['report_md'])
print('Audit last:', r['report_md'].rfind('## G.') > r['report_md'].rfind('### Benchmark Relative'))
"
```

All three commands must print success lines. If anything fails, fix and re-run.
