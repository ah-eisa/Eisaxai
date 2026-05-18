# EisaX Metadata Taxonomy v1.0

**Status:** authoritative. Every Phase H / Phase I engine that consumes
metadata from `core.data_layer.gcc_metadata` MUST interpret fields
strictly according to this document. Any new field requires a taxonomy
version bump.

**Date:** 2026-05-18  ·  **Maintainers:** EisaX Data Layer team
**Spec dependencies:** `gcc_ingestion_spec.md` (production rules) +
`PHASE_I_SPEC.md` (consumer contracts).

---

## 1. Why this document exists

The institutional moat depends on the difference between:

- **Observed**     — sourced from a published document we can re-fetch.
- **Derived**      — computed from one or more observed inputs by a
                     transparent rule.
- **Estimated**    — produced by an internal heuristic; methodology
                     string must be machine-readable.
- **Missing**      — no authoritative source available.

Confusing these tiers turns the platform into "a generic AI assistant
making finance claims." This taxonomy fixes the boundaries so that
mistake is impossible by construction.

---

## 2. Field-level vocabulary

### 2.1 `value`
The actual datum. Must be `None` when `data_quality == "missing"`. For
boolean fields, `False` is a positive assertion (e.g. "this issuer is
*not* Shariah-compliant"); use `None` when no source supports either
answer.

### 2.2 `as_of_date`
ISO `YYYY-MM-DD`. Represents when the **source** was issued (e.g. the
date the annual report was published), not when the record was
captured. Engines using time-decay weights MUST use this field, not
"now".

### 2.3 `source_type` (strict enum)

| Value | Meaning | Examples |
|-------|---------|----------|
| `issuer` | The fact appears in a document published by the issuer | IPO prospectus, annual report, IR page disclosure |
| `exchange` | Published by the trading venue | Listing record, sector assignment, suspension notice |
| `regulator` | Published by the local financial regulator | SAMA / CMA / SCA / CBE / FRA circular, strategic designation |
| `derived` | Computed from one or more upstream observed sources | MSCI/FTSE inclusion derived from constituent list, Shariah flag derived from index membership |
| `fallback` | Last-resort public reference where stronger sources are unavailable | Wikipedia / Reuters cross-reference (used sparingly, flagged in audit) |
| `missing` | No authoritative source available; field is unset | — |

**Rule:** an engine MUST NOT silently upgrade a `fallback` value to
`issuer`. Provenance never gains tier through transit.

### 2.4 `confidence`
`float ∈ [0.0, 1.0]`. Calibrated. The reviewer must defend a confidence
≥ 0.9 with two independent reads of the source document. ≤ 0.5 implies
"the source is ambiguous or partial — consumer should treat as
indicative only."

### 2.5 `data_quality` (strict enum)

| Value | Meaning | Render policy |
|-------|---------|---------------|
| `verified` | Directly cited from `issuer / exchange / regulator` | Always displayed |
| `derived` | Transparent rule applied to verified inputs | Displayed with "derived" badge |
| `estimated` | Internal heuristic produced this | Displayed only in expanded view; methodology mandatory |
| `missing` | No source | Suppressed in public sections, visible in audit only |

### 2.6 `methodology`
Human-readable derivation. For `derived` and `estimated` values this
must be specific enough that a reviewer can reproduce the value from
the same inputs. For `verified` values it cites the document section
(e.g. `aramco_ipo_prospectus_2019_ownership_section`).

### 2.7 `fallback_used`
`True` iff the value is a placeholder rather than a fact. Mutually
exclusive with `data_quality in {"verified", "derived", "estimated"}`.
Tests assert this invariant.

---

## 3. Provenance tier mapping

The tier is a computed property — engines and the report layer use it
to gate visibility:

| Tier | source_type × data_quality | Render policy | Confidence floor |
|------|----------------------------|---------------|------------------|
| **1** | `issuer / exchange / regulator` × `verified` | Always displayed in any section | ≥ 0.80 |
| **2** | `derived` × `verified / derived` | Displayed with "derived" badge in expanded views | ≥ 0.70 |
| **3** | `derived` × `estimated` | Displayed only in audit / expanded view; methodology mandatory | ≥ 0.50 |
| **4** | `missing` × `missing` | Suppressed in public sections; counted in audit | — |

The `provenance_tier(field_dict)` helper in `core.data_layer.gcc_metadata`
implements this mapping. Phase I's Section J renders the tier histogram.

---

## 4. Per-field semantics

This subsection fixes the **interpretation boundary** for every field
in `SCHEMA_FIELDS`. New fields require a taxonomy version bump.

### 4.1 `ticker`
Exchange-prefixed canonical form `EXCHANGE:SYMBOL`. The exchange
prefix is one of the allowed enum values (§5). Plain symbols (no
prefix) are accepted by `get_gcc_metadata()` and bare-matched.

### 4.2 `country`
ISO-2 uppercase **or** common-form short name when ISO-2 is ambiguous
(e.g. `"KSA"` instead of `"SA"` because internal reports use `"KSA"`).
Always `verified` and Tier 1 — derived from the exchange listing
country, not from the issuer's HQ country.

### 4.3 `exchange`
Allowed enum (§5). Always `verified` and Tier 1.

### 4.4 `sector`
GICS-11 canonical name (§6). Always `verified` and Tier 1; sector
reassignments by the exchange propagate via the next ingestion run.
Vendor-tagged sectors are normalised through
`core.data_layer.sectors.sector_classification` before storage.

### 4.5 `parent_company`
The controlling shareholder as named in the most recent **issuer**
disclosure (annual report / IPO prospectus). Set only when sovereign
ownership table or issuer document explicitly identifies the parent.
Confidence ≥ 0.85 requires citing a specific section of the document.

### 4.6 `government_ownership_pct`
Numeric percentage from issuer disclosure. **Never estimated**.
Acceptable `data_quality`: `verified` or `missing`. Reviewer must
update the value within 90 days of any IR-published change.

### 4.7 `strategic_asset_flag`
Boolean. `True` iff the entity is **either** (a) explicitly designated
strategic / national-champion by the regulator (e.g. CMA strategic
asset designation), **or** (b) the sovereign owns the controlling
block AND the asset class is regulated-utility / strategic-energy.
The two conditions are recorded in `_sovereign_ownership.py` via the
`strategic_designated` field.

### 4.8 `dividend_stability_score`
Numeric `0–100`. **Derived only** from a transparent rule over ≥ 5
years of payout history. Until the dividend ingestion job lands,
this field is `_missing()` for every ticker. **No estimation in v1.**

### 4.9 `domestic_vs_export_split`
Dict `{"domestic": float, "export": float}` from issuer segment
reporting. Both values in `[0, 1]` and sum to 1.0 ± 0.01.

### 4.10 `sovereign_sensitivity`
Categorical `"high" / "medium" / "low" / null`. **Derived only**.
Methodology must cite the inputs (e.g. ownership %, sector,
historical drawdown correlation with sovereign CDS).

### 4.11 `oil_beta_dependency`
Dict `{"bucket": str, "approx": float | null}`. Bucket is
`"very_high" / "high" / "medium" / "low" / "very_low" / null`.
`approx` populated only when a regression vs Brent over ≥ 36 months
of monthly returns is documented in the manifest.

### 4.12 `shariah_compliant_flag`
Boolean **or** `null`. `True` iff the ticker appears in one of the
canonical Shariah indices listed in `_shariah_index.py`. Always Tier 2
`derived` (not Tier 1) because individual instrument-level compliance
may diverge from index-level screening. **Never inferred from sector
alone.**

### 4.13 `inclusion_indices`
Tuple of index names from a published constituents list. Always `derived`
from a snapshotted index file (manifest). Tier 2.

### 4.14 `free_float_pct`
Numeric percentage. `verified` from issuer disclosure or `derived` from
exchange-published free-float band. Estimation is **not** allowed in v1.

### 4.15 `notes`
Free-form analyst note. Not provenance-tracked; analyst text only. Tone
must follow `phase_h.tone_guard.scrub_text` rules.

---

## 5. Allowed enum: `exchange`

```text
TADAWUL   — Saudi Stock Exchange (Saudi Arabia)
ADX       — Abu Dhabi Securities Exchange (UAE)
DFM       — Dubai Financial Market (UAE)
QSE       — Qatar Stock Exchange (Qatar)
KSE       — Boursa Kuwait (Kuwait)
BHB       — Bahrain Bourse (Bahrain)
EGX       — Egyptian Exchange (Egypt)
MAR       — Casablanca Stock Exchange (Morocco)     [reserved]
TUN       — Tunis Stock Exchange (Tunisia)          [reserved]
```

Exchanges marked `[reserved]` have no seed entries yet but are part of
the canonical vocabulary. Engines must accept them without raising.

---

## 6. Allowed enum: `sector` (GICS-11)

```text
Energy
Materials
Industrials
Consumer Discretionary
Consumer Staples
Health Care
Financials
Information Technology
Communication Services
Utilities
Real Estate
```

**Strict policy:** no entries outside this set are accepted in v1.
Conglomerates that don't fit any single GICS sector are recorded as
`Industrials` with a note documenting the business model. A future
taxonomy version may add a `business_model` Tier-2 derived field; do
NOT add it in v1.

---

## 7. Derived-vs-observed rules

| Field | Allowed `data_quality` | Source rule |
|-------|------------------------|-------------|
| `country / exchange / sector` | verified | Exchange listing record |
| `parent_company` | verified | Issuer disclosure only |
| `government_ownership_pct` | verified / missing | Issuer disclosure only; never derived |
| `strategic_asset_flag` | verified / missing | Regulator designation OR sovereign-table `strategic_designated=True` |
| `dividend_stability_score` | derived / missing | Computed over ≥5y payout history |
| `domestic_vs_export_split` | verified / missing | Issuer segment reporting |
| `sovereign_sensitivity` | derived / missing | Documented rule over verified inputs |
| `oil_beta_dependency` | derived / missing | Regression vs Brent, ≥36 monthly obs |
| `shariah_compliant_flag` | derived / missing | Membership in a canonical Shariah index |
| `inclusion_indices` | derived / missing | Snapshotted constituents list |
| `free_float_pct` | verified / derived / missing | Issuer disclosure or exchange-published band |

**Hard rule:** any field that ends up `data_quality="estimated"` in
this version requires a unanimous reviewer override AND a follow-up
v1.x taxonomy update. There are zero `estimated` claims in v1.0.

---

## 8. Consumer-side contract

Phase I engines (Section J, Context Graph, Risk Envelope) MUST:

1. **Refuse to render** a claim sourced from a Tier-4 field in public sections.
2. **Display the tier badge** next to every Tier-2 or Tier-3 claim.
3. **Read `as_of_date`** when applying time decay — never the wall clock.
4. **Honor `confidence`** when aggregating scores (weighted by confidence).
5. **Propagate `fallback_used`** in derived outputs so the audit trail tracks every fallback origin.

Failure to honor any of the above is a regression and trips the seed
coverage test or a future Section-J render assertion.

---

## 9. Versioning rules

- v1.x patch bumps: clarifications, additional enum members in a way
  that is backwards-compatible.
- v2.0: any change to existing field semantics, removal of an enum,
  or restriction tightening (e.g. raising confidence floors).
- Every taxonomy bump triggers a manual audit of every seed entry's
  affected fields.

---

## 10. Negative space (anti-scope)

Things this taxonomy intentionally does NOT include:

- Real-time price / volume / market-cap fields — those live in the
  market_cache adapter, not in the metadata layer.
- Analyst rating / buy-sell-hold signals — never sourced, never invented.
- News headlines or social sentiment — out of scope; would break the
  governance model.
- Forward forecasts of any field — engines compute forecasts, the
  metadata layer never asserts them.

If a future requirement seems to need any of the above, the right
response is to add a new engine that consumes the metadata layer, not
to extend the metadata layer with non-verifiable fields.

---

*This taxonomy is the contract between the data layer and every
engine that depends on it. Treat it like an API surface — break it
deliberately, never accidentally.*
