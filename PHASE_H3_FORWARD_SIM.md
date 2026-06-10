# PHASE H3 — Multi-Period Forward Simulation Engine — Implementation Spec

**File to modify:** `/home/ubuntu/investwise/phase_h/forward_sim.py` (replace stub bodies)
**Do NOT modify:** files outside `phase_h/` for H3.

Read `PHASE_H_SPEC.md` first.

---

## 1. Monte Carlo engine

`run_forward_simulation()` must produce a `ForwardScenario` payload that captures forward-looking distribution under multiple regimes. Use NumPy + SciPy only.

### Distribution choice (per asset)
- **Non-normal returns:** Student's t with `df = 6` (fat tails). Falls back to Normal if SciPy unavailable.
- **Stochastic volatility:** simple Heston-lite — `sigma_t+1 = clip(sigma_t * exp(kappa * (1 - sigma_t/sigma_long) + eta * z), 0.4*sigma_long, 3.0*sigma_long)` with `kappa=0.05`, `eta=0.10`.
- **Correlation instability:** for each simulated path, perturb the input correlation matrix `corr` by `corr_t = (1-w) * corr + w * corr_stress` where `corr_stress` is the all-pairs-0.85 risk-off matrix and `w ~ Beta(2, 18)` per timestep (mostly low, occasionally high).
- **Fat-tail jumps:** Poisson jumps with `lambda_jump=0.02` (per month per asset), jump magnitude `N(-0.05, 0.03)` for equities, `N(-0.10, 0.05)` for crypto/commodities. Skip jumps for cash and short bonds.

### Path generation
- Time step: monthly (`steps_per_year = 12`)
- Horizon: `horizon_years` (default 5y)
- Paths per scenario: `paths_per_scenario` (default 2000)
- Total paths: `paths_per_scenario * len(SCENARIO_NAMES)`

### Scenario tree (6 named regimes; per-scenario adjustments)

| name | prob (default) | μ adj | σ adj | corr_stress weight |
|---|---|---|---|---|
| soft_landing       | 0.30 | +0.5%/yr equities, +0.0% bonds | 0.95x | 0.10 |
| recession          | 0.20 | −4%/yr equities, +2%/yr bonds | 1.20x | 0.50 |
| stagflation        | 0.10 | −2%/yr equities, −2%/yr bonds, +5%/yr commodities | 1.30x | 0.40 |
| ai_productivity_boom | 0.10 | +6%/yr equities (concentrated in tech-heavy proxies) | 1.10x | 0.25 |
| energy_shock       | 0.15 | −3%/yr equities, +8%/yr commodities | 1.25x | 0.35 |
| liquidity_crisis   | 0.15 | −2%/yr equities, −1%/yr bonds (correlations collapse to 1) | 1.50x | 0.80 |

Probabilities must sum to 1.0. Allow env override via `EISAX_PHASE_H_SCENARIO_PRIORS` (JSON-encoded dict).

### Recurring contributions / withdrawals
- `contributions_per_year` added at start of each year (split evenly across months)
- `withdrawal_per_year` subtracted at start of each year
- Both expressed in same units as `port_value_usd` (default 0)

### Inflation
- `inflation_assumption_pct` annual real-rate discount. Report inflation-adjusted terminal values as well as nominal. Default 2.0%.

---

## 2. Per-scenario outcome computation

For each scenario, after running its paths:
- `terminal_p10`, `terminal_p50`, `terminal_p90`: %-gain percentiles of terminal value vs starting value (real, inflation-adjusted)
- `max_dd_p50_pct`: median maximum drawdown across paths (in %)
- `recovery_months_p50`: median months from trough back to prior peak (NaN → 0 if never recovers in horizon)
- `prob_loss`: fraction of paths with terminal real value < starting value
- `prob_target`: fraction of paths with terminal real return ≥ (4% × horizon_years) annualised — i.e., beating cash+inflation. Make 4% configurable via `EISAX_PHASE_H_TARGET_REAL_RETURN=0.04`.

For the **aggregate** (probability-weighted across all scenarios), compute the same fields on the pooled sample using `np.random.choice` with scenario probabilities as path weights.

---

## 3. `render_forward_scenario_md(payload, language)`

Stub already produces a usable shape; expand to include:

```
## H. Forward Scenario Distribution

*Horizon: 5y · Inflation: 2.0% · Seed: 42 · Paths: 12000*

| Scenario | Probability | Terminal P10 | Terminal P50 | Terminal P90 | Max Drawdown (P50) | Recovery (months, P50) |
|---|---|---|---|---|---|---|
| soft landing | 30.0% | +6% | +28% | +52% | -8% | 5.0 |
| recession    | 20.0% | -22% | -7% | +11% | -25% | 22.0 |
| ...

**Aggregate (probability-weighted)**

| Metric | Value |
|---|---|
| Expected terminal value range (real) | $1.10 — $1.62 (P10–P90) |
| Probability of loss (real) | 18% |
| Probability of target (≥4% real ann.) | 47% |
| Worst-decile terminal | $0.78 |
| Expected drawdown range | -10% to -28% |
| Recovery duration (median) | 7 months |

*Distributional framing only — outcomes reflect modelled assumptions; not a forecast.*
```

All bilingual via LABELS. Translate the aggregate-block labels: add new entries in `report_helpers.LABELS` for:
- `aggregate_block_title`
- `expected_terminal_range`
- `prob_loss`
- `prob_target`
- `worst_decile`
- `expected_dd_range`
- `recovery_duration_median`
- `distributional_disclaimer`

---

## 4. Probabilistic framing — non-negotiable

Every sentence in the rendered markdown must be probabilistic. Forbidden patterns (Codex must check before rendering):
- "will" → use "is likely to" / "modelled to"
- "expect a return of X%" → use "median modelled return: X%"
- Any guarantee-style verb. Use modal verbs only.

Add a `_probabilistic_lint(text)` helper inside `forward_sim.py` that scrubs leftover deterministic phrasing.

---

## 5. Tests — `phase_h/tests/test_forward_sim.py`

1. `test_seed_determinism` — run twice with same seed → identical aggregate stats.
2. `test_scenario_probabilities_sum_to_one`.
3. `test_recession_terminal_below_soft_landing` (statistical, with seed).
4. `test_contributions_increase_p50` — same params, contributions $10k/yr > $0/yr.
5. `test_render_bilingual_complete`.
6. `test_no_deterministic_language` — render, run `_probabilistic_lint`, expect zero hits.
7. `test_no_forbidden_phrases` (tone_guard).
8. `test_seed_propagated_to_audit` — payload.seed equals env-default 42.

---

## 6. Performance

Target: full simulation (12k paths × 60 months × ~15 assets) under 2 seconds on the staging box. Use vectorised NumPy — no Python loops over paths. If the assets×months×paths array gets too large for memory, chunk in groups of 500 paths.

---

## 7. Verification

```bash
cd /home/ubuntu/investwise && \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -m phase_h.tests.test_skeleton && \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -m phase_h.tests.test_forward_sim && \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -c "
from global_allocator import allocate
from phase_h.orchestrator import augment_result
import time
r = allocate(profile='balanced')
t0 = time.time()
r = augment_result(r, language='en')
print(f'augment time: {time.time()-t0:.2f}s')
print('forward keys:', sorted((r.get('forward_scenario') or {}).keys()))
print('section in report:', '## H.' in r['report_md'])
print('audit-G stays last:', r['report_md'].rfind('## G.') > r['report_md'].rfind('## H.'))
"
```

Bump `ENGINE_VERSION = "0.2.0"`.
