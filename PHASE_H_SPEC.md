# Phase H — Institutional Portfolio Intelligence Expansion

**Status:** in-progress
**Owner:** Claude (architecture + audit) + Codex (implementation)
**Scope:** Upgrade EisaX from institutional-grade reporting into a benchmark-aware, execution-aware, forward-simulation portfolio intelligence system, while preserving all Phase A–G behavior, tone, accordion compatibility, bilingual EN/AR support, and audit reproducibility.

---

## Non-negotiable preservation rules

The following MUST remain intact after every Phase H change. Any regression here is a blocker:

1. **A–G section hierarchy** in `portfolio_builder.py` and `global_allocator.py`. Phase H additions appear as labeled **subsections** under the appropriate parent (B/C/D/E), or as a new `## H.` block placed BEFORE Section G (Audit Appendix) — never above Section A, never replacing existing markdown.
2. **Tone discipline** — BlackRock / MSCI Barra / Morningstar Direct / Bridgewater. No retail wording, no emojis in analytical sections, no hype. Forbidden phrases: "good setup", "strong timing", "AI momentum", "top risk", "return enhancer", "moonshot", "massive upside", "high conviction trade".
3. **Adaptive disclosures, confidence calibration, reliability tiers, regime classification, feasibility diagnostics, institutional rounding, severity tags, audit hash** — all carry forward unchanged.
4. **Bilingual EN/AR** — every new markdown helper accepts `language: str = "en"` and renders Arabic counterpart when `language == "ar"`. Section headings, table headers, body text all bilingual.
5. **Frontend accordion** — section markers (`## A. ...`, `## B. ...`, etc.) keep current format so `app.js` accordion parser still groups correctly. New H subsections under existing parents use `### ` or `#### ` headings.
6. **Backward compatibility** — `allocate()` callers and report consumers must not break. New keys in the result dict are additive. Feature flags default to ON for analytics output but allow OFF for emergency rollback.
7. **Deterministic + reproducible** — same inputs → same outputs (modulo Monte Carlo, where seed is recorded in audit appendix).

---

## Architecture

All Phase H code lives in a new package `/home/ubuntu/investwise/phase_h/` to keep the existing monoliths clean.

```
phase_h/
  __init__.py                # public API
  feature_flags.py           # PHASE_H_* env toggles
  schemas.py                 # TypedDicts / dataclasses for all outputs
  report_helpers.py          # shared bilingual markdown helpers, severity tags
  benchmarks.py              # H1 — Native Benchmark Analytics Engine
  tc_optimizer.py            # H2 — Transaction-Cost-Aware Optimizer (CVXPY add-ons)
  forward_sim.py             # H3 — Multi-Period Forward Simulation Engine
  factor_model.py            # H4 — True Factor Model Engine (FF3/Carhart/FF5)
  committee.py               # H5 — Investment Committee Mode
  tone_guard.py              # forbidden-phrase scrubber, severity normalizer
  audit_extensions.py        # extend audit appendix with H-engine reproducibility
```

### Integration points

- **`global_allocator.allocate()`** (global_allocator.py:382)
  Returns dict already containing `weights`, `metrics`, `report_md`, `feasibility`, `benchmark`. Phase H adds (additively):
  - `benchmark_relative`  ← H1
  - `execution_diag`      ← H2
  - `forward_scenario`    ← H3
  - `factor_decomp`       ← H4
  - `committee_brief`     ← H5
  - `phase_h_meta`        ← reproducibility hashes, flag states, engine versions

- **`portfolio_builder._run_allocator()`** (portfolio_builder.py:284)
  Inject H subsections into existing A–G markdown by calling `phase_h.report_helpers.augment_report(report_md, result, language)`.

- **`portfolio_upload.upload_portfolio()`** (api/routers/portfolio_upload.py:160)
  Apply the same `augment_report()` to the uploaded-portfolio markdown path. Uploaded portfolios bypass the optimizer, so H2 only contributes a *what-if* execution estimate for the user's existing weights.

- **`institutional_stock_wrapper`** (core/services/institutional_stock_wrapper.py)
  Only the **single-asset-relevant** subset of Phase H applies: benchmark-relative (vs SPY / MSCI World / TASI / BTC depending on asset class), rolling beta, and a stripped factor decomposition. No optimizer, no full forward sim.

- **`staging.py`** (`_staging_extract_*` family + `_staging_shape_result`)
  Add extraction patterns for the new section IDs so the staging UI surfaces them. Phase H section markers must be discoverable by these regexes.

---

## Feature flags (env-controlled)

| Flag | Default | Effect |
|---|---|---|
| `EISAX_PHASE_H_ENABLED`           | `1` | Master switch. `0` = bypass all H code, return Phase G unchanged. |
| `EISAX_PHASE_H_BENCHMARK`         | `1` | H1 Native Benchmark Analytics |
| `EISAX_PHASE_H_TC_OPTIMIZER`      | `1` | H2 Transaction-cost terms in objective |
| `EISAX_PHASE_H_FORWARD_SIM`       | `1` | H3 Forward simulation block |
| `EISAX_PHASE_H_FACTOR_MODEL`      | `1` | H4 Factor decomposition |
| `EISAX_PHASE_H_COMMITTEE`         | `1` | H5 Committee brief block |
| `EISAX_PHASE_H_TONE_GUARD`        | `1` | Final tone-guard scrubber pass |
| `EISAX_PHASE_H_DETERMINISTIC_SEED`| `42`| Monte Carlo seed |

Flag-off path always returns the prior Phase G output verbatim.

---

## Report placement (final hierarchy)

```
A. Executive Summary                       (unchanged; H5 may add 1-line committee headline)
B. Mandate Feasibility                     (unchanged)
C. Risk Diagnostics
   └─ ### Benchmark Relative Diagnostics   (H1)
   └─ ### Factor Risk Decomposition        (H4)
D. Allocation Logic
   └─ ### Benchmark-Relative Attribution   (existing — preserve)
E. Rebalancing Plan
   └─ ### Execution Efficiency Diagnostics (H2)
F. AI Commentary Layer                     (unchanged)
H. Forward Scenario Distribution           (H3 — new top-level, placed between F and G)
I. Investment Committee Brief              (H5 — optional, gated by request mode)
G. Audit Appendix                          (extended with H reproducibility — H-engine hashes, MC seed, flag states)
```

Note: Section G stays last. Section H sits before G so audit is still the final block.

---

## Shared schemas (`phase_h/schemas.py`)

All engines return TypedDicts so downstream consumers get static-checkable shape:

```python
class BenchmarkRelative(TypedDict):
    benchmark_ticker: str
    benchmark_label: str
    active_return_pct: float
    tracking_error_pct: float
    information_ratio: float
    rolling_alpha_12m_pct: float
    rolling_beta_12m: float
    relative_drawdown_pct: float
    upside_capture: float
    downside_capture: float
    relative_volatility: float
    active_share_pct: float
    style_drift: str
    excess_decomp: dict   # {allocation, selection, factor, concentration}
    regime_behavior: dict # {outperform_envs: [..], lag_envs: [..]}
    reliability_tier: str

class ExecutionDiagnostics(TypedDict):
    turnover_pct: float
    implementation_shortfall_bp: float
    market_impact_bp: float
    slippage_bp: float
    complexity_tier: str
    liquidity_stress: str
    tax_note: str
    rebalance_frequency: str

class ForwardScenario(TypedDict):
    horizon_years: float
    contributions_per_year: float
    withdrawal_per_year: float
    inflation_assumption_pct: float
    scenarios: dict       # name -> {prob, terminal_p10/p50/p90, max_dd_p50, recovery_months_p50}
    aggregate: dict       # weighted distribution
    seed: int

class FactorDecomp(TypedDict):
    model: str            # "FF3" | "Carhart" | "FF5"
    loadings: dict        # factor -> beta
    t_stats: dict
    contribution_return: dict
    contribution_vol: dict
    contribution_drawdown: dict
    r_squared: float
    rolling_stability: float
    warnings: list[str]
    reliability_tier: str

class CommitteeBrief(TypedDict):
    mode: str             # "1pager" | "cio_memo" | "executive_memo" | "defend" | "bear" | "stress"
    headline: str
    key_decision: str
    key_risks: list[str]
    positioning: str
    implementation_notes: str
    mandate_summary: str
    top_vulnerabilities: list[str]
    challenge_scenarios: list[str]
    exhibits: list[dict]  # {number, title, payload_ref}
```

---

## Engine specs — see sibling files

- `PHASE_H1_BENCHMARK.md`
- `PHASE_H2_TC_OPTIMIZER.md`
- `PHASE_H3_FORWARD_SIM.md`
- `PHASE_H4_FACTOR_MODEL.md`
- `PHASE_H5_COMMITTEE.md`

Each is a complete, self-contained spec for Codex to implement.

---

## Validation checklist (run before each sub-phase merge)

1. `python -c "from phase_h import *"` imports clean.
2. `python -c "from global_allocator import allocate; r = allocate(profile='balanced'); print(list(r))"` succeeds and includes new keys.
3. EN report renders A–G + H subsections in correct order, accordion markers intact.
4. AR report renders, all helpers bilingual.
5. Infeasible case still returns feasibility-failure markdown without crashing H code.
6. Uploaded-portfolio path renders H1/H4 (no optimizer dependence).
7. Staging API `/staging-api/analyze` returns 200 with new section markers extractable.
8. Tone-guard scrubber sees zero forbidden phrases in final markdown.
9. No emoji regressions in analytical sections.
10. No duplicate disclaimers; no broken markdown tables (run `markdownlint` or equivalent on the output).
11. Audit appendix lists Phase H engine versions, flag states, seed.

---

## Codex delegation protocol (used by Claude)

For each sub-phase:
1. Write the sub-phase spec file (`PHASE_H{n}_*.md`).
2. Invoke Codex non-interactively:
   ```bash
   codex exec --dangerously-bypass-approvals-and-sandbox \
     -C /home/ubuntu/investwise \
     -s danger-full-access \
     "Read PHASE_H_SPEC.md and PHASE_H{n}_*.md. Implement exactly as specified. Do not modify Phase A–G code outside the documented integration points. Run the validation steps. Print a diff summary at the end."
   ```
3. Claude audits Codex's diff against this spec (read changed files, run validation steps).
4. If audit passes, mark sub-phase complete and move to next.
5. If audit fails, re-invoke Codex with corrections.
