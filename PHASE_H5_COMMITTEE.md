# PHASE H5 — Investment Committee Mode — Implementation Spec

**File to modify:** `/home/ubuntu/investwise/phase_h/committee.py` (replace stub bodies)
**Optional new file:** `/home/ubuntu/investwise/phase_h/committee_export.py` for PDF/print helpers.
**Do NOT modify:** files outside `phase_h/` except (a) optionally `core/services/cio_pdf.py` (already exists) to wire export, and (b) `api_bridge_v2.py` to add a `committee_mode` query parameter to `/staging-api/analyze` (guarded by a try/except — additive).

Read `PHASE_H_SPEC.md` first.

---

## 1. Modes

`SUPPORTED_MODES = ("1pager", "cio_memo", "executive_memo", "defend", "bear", "stress", "hostile", "challenge_macro", "challenge_concentration", "challenge_liquidity", "challenge_valuation", "challenge_geopolitical")`

### Hostile committee simulation (new — high priority per Phase H priorities)

The `hostile` mode generates a structured adversarial review with:
- 5–8 **objections** drawn from challenge categories (macro / concentration / liquidity / valuation / geopolitical / factor crowding)
- a **counter-argument** for each objection sourced from the existing benchmark/factor/forward payloads
- a **thesis fragility score** in [0, 100] = `100 * (#unaddressed_objections / total)`, with the unaddressed list explicit
- a final **CIO defensibility verdict**: `"defensible"` / `"requires-justification"` / `"weak-thesis"` based on fragility

### Challenge sub-modes

Each `challenge_*` mode focuses the brief on a single objection category and skips the rest. Useful for targeted committee preparation. They share the hostile-mode renderer but with a filtered objection set.

### Objection generator pattern

`build_objections(result, category) -> list[Objection]` where `Objection` has:
```python
{
    "category": "concentration" | "liquidity" | "macro" | "valuation" | "geopolitical" | "factor",
    "claim": str,            # institutional-tone claim
    "evidence_ref": str,     # which engine payload key supports the claim
    "severity": "low" | "moderate" | "elevated",
    "counter": str | None,   # rebuttal sourced from same payload; None = unaddressed
}
```

Severity thresholds (examples):
- concentration: any single ticker > 15%
- liquidity: execution_diag.liquidity_stress in {"elevated","high"}
- macro: forward_scenario.scenarios["recession"].terminal_p10 < -20%
- valuation: factor_decomp.loadings["HML"] < -0.4 (growth-tilted)
- geopolitical: GCC weight > 25% AND committee_mode in {"hostile","challenge_geopolitical"}
- factor: any factor in factor_decomp.warnings

Each mode produces a `CommitteeBrief` payload with the same schema, but different content focus:

| Mode | Focus |
|---|---|
| `1pager` | Headline + 3 risks + 1 positioning line + 3 vulnerabilities. Fits a single printed page. |
| `cio_memo` | Expanded: full positioning + implementation + benchmark-relative summary + scenario tree headline outcomes. |
| `executive_memo` | 1-page narrative form (paragraphs, not tables) for distribution to non-investment executives. |
| `defend` | Argument FOR the current allocation: cites benchmark-relative tracking, factor diversification, scenario robustness. |
| `bear` | Challenge case: cites top vulnerabilities, downside scenarios, factor crowding warnings, liquidity stress. |
| `stress` | Worst-case Composite: liquidity_crisis + recession scenarios from H3, factor concentration warnings from H4, drawdown bands. |

---

## 2. Brief construction

`build_committee_brief(result, mode, language)` must read from `result` only — never re-run engines. Pull from:
- `result["weights"]` — for headline + concentration
- `result["metrics"]` — for return/risk/Sharpe
- `result["feasibility"]` — for mandate summary
- `result["confidence"]` — reliability tier
- `result["benchmark_relative"]` — for positioning + tracking
- `result["execution_diag"]` — for implementation notes
- `result["forward_scenario"]` — for challenge_scenarios list
- `result["factor_decomp"]` — for top_vulnerabilities

### Field rules

- `headline`: one sentence, institutional tone. Format:
  `"Profile {profile}; {n} positions across {regions} regions. Reliability: {tier}. Benchmark-relative: {tracking_class}, IR {ir}."`
- `key_decision`: one sentence describing the implicit recommendation. Examples (must use modal verbs):
  - "Maintain current allocation; tracking error and factor exposures remain within mandate."
  - "Rebalance with reduced equity beta; downside capture above 1.1 indicates asymmetric stress sensitivity."
- `key_risks` (3 items max for 1pager, 5 for cio_memo): pulled from `factor_decomp.warnings + benchmark_relative.notes` and the worst-2 scenarios from forward_scenario.
- `positioning`: one sentence summarising benchmark-relative behavior.
- `implementation_notes`: one sentence summarising turnover/complexity/liquidity tier.
- `mandate_summary`: one sentence on feasibility status + binding constraints (read from `result["feasibility"]`).
- `top_vulnerabilities`: top-3 highest-|β| factors from FactorDecomp, paired with their warnings if present.
- `challenge_scenarios`: top-3 scenarios from forward_scenario sorted by `terminal_p10` ascending (worst tail first).
- `exhibits`: numbered list (1-indexed). Each entry references a payload section by name. For 1pager mode emit at most 2 exhibits. cio_memo emits 4. stress emits 3.

### Tone constraints

- All sentences use modal verbs only (`may`, `likely to`, `modelled to`).
- Forbidden phrases (PHASE_H_SPEC) must not appear — verified by `tone_guard.audit_block` in tests.
- No emojis.
- No motivational language.

---

## 3. `render_committee_brief_md(payload, language)`

Layout (EN; AR mirrors via LABELS):

```
## I. Investment Committee Brief

*Mode: 1pager · Distribution: Investment Committee · Page break recommended below.*

**Headline.** <headline>

| Field | Detail |
|---|---|
| Key Decision | <key_decision> |
| Positioning | <positioning> |
| Implementation | <implementation_notes> |
| Mandate | <mandate_summary> |

**Key Risks**
1. ...
2. ...
3. ...

**Top Vulnerabilities**
- ...

**Challenge Scenarios**
- ...

**Exhibits**
- Exhibit 1. <title> — payload reference: <ref>
- Exhibit 2. <title> — payload reference: <ref>
```

For `executive_memo`, replace the table with 3 short paragraphs (Headline, Positioning, Implementation+Risk).

For `defend` and `bear`, prepend a single-line stance marker:
- defend: `**Stance: Defend.** The allocation remains compliant with mandate and benchmark-relative thresholds.`
- bear: `**Stance: Challenge.** Material vulnerabilities surface under stress regimes.`

PDF/print friendliness:
- Use `\n\n---\n\n` between major blocks so markdown→PDF picks up page-break-friendly rule.
- Add a comment marker `<!-- pagebreak -->` between Brief and Audit Appendix so existing print stylesheet inserts a CSS break.

---

## 4. API + invocation surface

Two activation paths:

a) **Env-driven (already wired):** `EISAX_COMMITTEE_MODE=1pager` causes orchestrator to call `build_committee_brief` and inject Section I.

b) **Per-request (additive):** Add an optional `committee_mode` query/form parameter to `/staging-api/analyze` in `api_bridge_v2.py`. If present, the staging router passes it through to the underlying portfolio_builder. Use a try/except guard so a missing parameter never breaks existing callers.

c) **Programmatic:** `augment_result(result, committee_mode="cio_memo", language="en")` already supports the kwarg.

---

## 5. Tests — `phase_h/tests/test_committee.py`

1. `test_all_supported_modes_build_without_error` — call each mode on a synthetic result.
2. `test_1pager_within_size_budget` — rendered markdown < 4000 characters.
3. `test_defend_vs_bear_distinct_content` — same input, different stance prefixes and different risk emphasis.
4. `test_no_forbidden_phrases` (tone_guard).
5. `test_no_deterministic_language` — assert no sentence contains "will" or "guarantees" or "always".
6. `test_bilingual_complete`.
7. `test_section_appears_before_G` — orchestrator places `## I.` before `## G.`.

---

## 6. NOT to do

- Do NOT depend on a PDF renderer in this engine; emit clean markdown only. PDF export is a separate concern.
- Do NOT add new top-level Phase A–G keys; the brief is a SUMMARY of existing data.
- Do NOT use forbidden retail phrases (PHASE_H_SPEC item 2 of preservation rules).

---

## 7. Verification

```bash
cd /home/ubuntu/investwise && \
  EISAX_COMMITTEE_MODE=cio_memo \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -m phase_h.tests.test_skeleton && \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -m phase_h.tests.test_committee && \
  /home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python -c "
from global_allocator import allocate
from phase_h.orchestrator import augment_result
r = allocate(profile='balanced')
r = augment_result(r, language='en', committee_mode='1pager')
md = r['report_md']
print('Section present:', '## I.' in md)
print('G last:', md.rfind('## G.') > md.rfind('## I.'))
print('Brief mode:', r['committee_brief']['mode'])
"
```

Bump `ENGINE_VERSION = "0.2.0"`.
