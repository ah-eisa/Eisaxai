# PHASE H2 — Transaction-Cost-Aware Optimizer — Implementation Spec

**Files to modify:**
- `/home/ubuntu/investwise/phase_h/tc_optimizer.py` (replace stub bodies)
- `/home/ubuntu/investwise/global_allocator.py` — surgical edits in `allocate()` only, gated by `PHASE_H_TC_OPTIMIZER`. Touch nothing outside the optimizer block. Preserve CLARABEL solve path, feasibility checks, rounding, constraint validation.

Read `PHASE_H_SPEC.md` first.

---

## 1. CVXPY add-on (`build_turnover_terms`)

Replace the stub with the real builder. Signature stays the same.

Returns a tuple of 3 CVXPY expressions:
1. **linear turnover penalty**: `linear_lambda * cp.norm1(w - w_prev)`
2. **quadratic penalty**: `quadratic_lambda * cp.sum_squares(w - w_prev)` — discourages large block trades
3. **persistence penalty**: `persistence_lambda * cp.norm1(w - w_prev)` over a smoothed window (when w_prev is itself an average of last K weights). For first implementation, behave same as linear when no smoothed window is available.

All three return `cp.Constant(0.0)` if `w_prev is None` or the flag is off — orchestrator and allocate() must remain runnable when there is no prior weight history.

Default lambdas (tunable via env):
- `EISAX_TC_LINEAR_LAMBDA=0.0010` (10 bp per unit-turnover)
- `EISAX_TC_QUADRATIC_LAMBDA=0.0005`
- `EISAX_TC_PERSISTENCE_LAMBDA=0.0002`

Read these via `os.environ.get(name, default)` cast to float.

---

## 2. `allocate()` integration in `global_allocator.py`

Locate the CVXPY problem build inside `allocate()` (around line 600–800). Add:

```python
from phase_h.tc_optimizer import build_turnover_terms
from phase_h.feature_flags import PHASE_H_TC_OPTIMIZER

# (immediately after the existing objective is built)
if PHASE_H_TC_OPTIMIZER:
    w_prev_vec = None  # plug in from caller-provided w_prev when available
    if "w_prev" in locals() and w_prev is not None:
        w_prev_vec = np.array([w_prev.get(a.ticker, 0.0) for a in assets])
    if w_prev_vec is not None:
        lin_l = float(os.environ.get("EISAX_TC_LINEAR_LAMBDA", "0.0010"))
        quad_l = float(os.environ.get("EISAX_TC_QUADRATIC_LAMBDA", "0.0005"))
        pers_l = float(os.environ.get("EISAX_TC_PERSISTENCE_LAMBDA", "0.0002"))
        lin, quad, pers = build_turnover_terms(
            cp, w, cp.Constant(w_prev_vec),
            linear_lambda=lin_l, quadratic_lambda=quad_l, persistence_lambda=pers_l,
        )
        objective = objective + lin + quad + pers
```

Add a new optional parameter to `allocate()` signature:
- `w_prev: Optional[dict[str, float]] = None` — previous weights for turnover penalty
- `rebalance_frequency: str = "quarterly"` — passed through to execution diagnostics

Preserve all existing parameters and defaults. The new params default to None / "quarterly" so existing callers are unaffected.

If solver INFEASIBLE after adding turnover terms, fall back to solving without them and add a note `"turnover penalty relaxed due to infeasibility"` to a new `execution_diag.notes` list.

---

## 3. `estimate_execution(weights, w_prev, asset_meta, ...)`

Replace the stub with a real model.

### Turnover
`turnover_pct = 0.5 * sum_i |w_i − w_prev_i| * 100`  (one-way, in %)
If `w_prev is None`, treat as a from-cash rebuild: `turnover_pct = sum_i |w_i| * 100`.

### Slippage model (per-asset, then aggregated)

For each asset `i`:
- `volatility_i` — daily vol (estimated from parquet or derived from `asset_meta["vol"]`)
- `adv_participation_i` — assume target $ traded / $ ADV. Default 5% participation, configurable via `EISAX_TC_ADV_PARTICIPATION=0.05`.
- `spread_bp_i` — bid-ask spread in bp. Defaults by asset kind:
  - large-cap equity: 5 bp
  - small-cap equity: 25 bp
  - emerging market equity: 30 bp
  - bonds (TLT/AGG/EMB): 8 bp
  - gold/silver: 6 bp
  - crypto: 15 bp
  - commodities (futures-ETF): 12 bp

**Slippage per asset (bp) — square-root market impact (Almgren-Chriss-lite):**
```
slippage_bp_i = spread_bp_i / 2
              + IMPACT_COEF * sigma_daily_pct_i * sqrt(participation_i)
```
where `IMPACT_COEF` default = 10.

### Region-aware liquidity multipliers (new per Phase H priorities)

Apply a **region/asset-class multiplier** to `slippage_bp_i` before aggregation:

| region/kind | multiplier | rationale |
|---|---|---|
| US equity (SPY, QQQ, MDY, VIG, XLV, XLU)     | 1.0   | deep liquidity, tight spreads |
| US bonds (TLT, SHY, BIL)                     | 1.0   | dealer-quoted, electronic |
| Gold / Silver / large commodity ETFs         | 1.2   | wider stress spreads |
| GCC equity (KSA, UAE, GCC)                   | **1.8** | thinner books, settlement frictions, lower turnover days |
| Egypt equity (EFID.CA)                       | **2.4** | currency-translation gap, sparse continuous quotes |
| Crypto (BTC, ETH)                            | **piecewise** — see below |
| EM Bonds (EMB)                               | 1.4   | OTC fragmentation |

**Crypto liquidity discontinuity** (mandatory): crypto slippage is NOT continuous in size — large notional steps through the depth book non-linearly. Compute as:
```
crypto_extra_bp = 0  if notional <= 250k
                  20  if 250k < notional <= 2M
                  60  if 2M < notional <= 10M
                  150 if notional > 10M
slippage_bp_crypto = base_slippage + crypto_extra_bp
```
Add `liquidity_discontinuity_triggered: bool` field to `ExecutionDiagnostics` when notional crosses a step.

### GCC trading-day adjustment

GCC markets (KSA/UAE/GCC) trade Sun–Thu (not Mon–Fri). When `rebalance_frequency` is mapped to days, GCC-only legs use 4-day calendar week. Add `gcc_calendar_note: str` to payload only when ≥10% of weights are GCC.

**Aggregate slippage_bp:** `sum_i (delta_w_i * slippage_bp_i)` where `delta_w_i = |w_i − w_prev_i|`.

**market_impact_bp:** `sum_i (delta_w_i * 10 * sigma_daily_i * sqrt(adv_participation_i))`

**implementation_shortfall_bp:** `slippage_bp + 2.0` (constant 2bp commission floor).

### Liquidity stress

Classify per-asset participation:
- `<2%` ADV → `"low"`
- `2-10%` → `"moderate"`
- `10-25%` → `"elevated"`
- `>25%` → `"high"`

Report the **worst** tier across all rebalanced assets as `liquidity_stress`.

### Complexity tier

Function of turnover + #names changed:
- `turnover < 5%` and `<= 3 names changed` → `"low"`
- `turnover < 15%` and `<= 8 names changed` → `"moderate"`
- `turnover < 35%` → `"elevated"`
- else → `"high"`

### Tax note

Fixed string (bilingual):
- EN: `"Tax-aware execution requires account-level lot data; placeholder pending integration with broker tax-lot feed."`
- AR: `"التنفيذ الواعي بالضرائب يتطلب بيانات الحصص على مستوى الحساب؛ مكان مخصص بانتظار التكامل مع تغذية الحصص الضريبية من الوسيط."`

### Rebalance frequency

Pass-through string in `{"monthly","quarterly","semiannual","annual"}`. If invalid input, coerce to `"quarterly"` and add a note.

### Flags set on payload

- `turnover_penalty_applied: True` if linear lambda > 0 and w_prev was used
- `quadratic_penalty_applied: True` if quadratic lambda > 0
- `persistence_preference_pct: 100 * (1 - turnover/max_observed_turnover_in_lookback)` — proxy 0 when no lookback

---

## 4. `render_execution_md(payload, language)`

```
### Execution Efficiency Diagnostics

| Metric | Value | Tag |
|---|---|---|
| Turnover | XX.X% | (low/moderate/elevated/high) |
| Implementation Shortfall | XX.X bp | (low/moderate/elevated/high) |
| Market Impact | XX.X bp | (low/moderate/elevated) |
| Estimated Slippage | XX.X bp | (low/moderate/elevated) |
| Execution Complexity | <tier> | — |
| Liquidity Stress | <tier> | — |
| Rebalance Frequency | quarterly | — |

*Turnover penalty: applied (linear λ=0.0010, quadratic λ=0.0005)*

*<tax note line>*
```

Severity-tag thresholds:
- Turnover: `<10%` low, `10-25%` moderate, `25-50%` elevated, `>50%` high
- Implementation Shortfall: `<15bp` low, `15-40bp` moderate, `40-100bp` elevated, `>100bp` high
- Market Impact: `<5bp` low, `5-25bp` moderate, `>25bp` elevated
- Slippage: `<10bp` low, `10-30bp` moderate, `>30bp` elevated

All bilingual via `LABELS` — add labels for any new strings.

---

## 5. Tests — `phase_h/tests/test_tc_optimizer.py`

1. `test_turnover_zero_when_no_prior` — `w_prev=None` → turnover_pct == 100% (from-cash) AND linear/quad penalty terms return Constant(0).
2. `test_turnover_matches_l1` — w={SPY:0.5,TLT:0.5}, w_prev={SPY:0.6,TLT:0.4} → turnover == 10%.
3. `test_slippage_increases_with_participation` — same turnover, but EISAX_TC_ADV_PARTICIPATION env doubled → slippage_bp roughly sqrt(2)x higher.
4. `test_render_bilingual_complete` — both languages render full row set.
5. `test_no_forbidden_phrases` — render then `tone_guard.audit_block`, expect 0.
6. `test_infeasibility_fallback` — synthetic forcing problem with absurd turnover lambda; allocate() still returns feasible weights with a `"turnover penalty relaxed"` note in execution_diag.

---

## 6. What NOT to do

- Do NOT touch CLARABEL parameters, constraint validation, or rounding logic.
- Do NOT break the existing `allocate()` signature for any caller — only ADD optional kwargs.
- Do NOT add new heavy dependencies.

---

## 7. Verification before declaring done

```bash
cd /home/ubuntu/investwise && \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -m phase_h.tests.test_skeleton && \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -m phase_h.tests.test_tc_optimizer && \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -c "
from global_allocator import allocate
from phase_h.orchestrator import augment_result
r1 = allocate(profile='balanced')
r1 = augment_result(r1, language='en', w_prev=None)
print('no-prior turnover:', r1['execution_diag']['turnover_pct'])
r2 = allocate(profile='balanced', w_prev={'SPY':0.4,'TLT':0.3,'GLD':0.3})
r2 = augment_result(r2, language='en', w_prev={'SPY':0.4,'TLT':0.3,'GLD':0.3})
print('with prior turnover:', r2['execution_diag']['turnover_pct'])
print('Section in report:', 'Execution Efficiency Diagnostics' in r2['report_md'])
"
```

Bump `ENGINE_VERSION = "0.2.0"` in `tc_optimizer.py`.
