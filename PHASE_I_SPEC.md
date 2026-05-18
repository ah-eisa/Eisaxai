# Phase I — Institutional Context Graph & Risk Spec

**Status:** SPEC ONLY. No implementation begins until:
1. Phase H + Data Layer are live in staging (✅ done 2026-05-18)
2. GCC metadata moat reaches Tier-1 coverage threshold (≥60 curated names with provenance)
3. Ingestion pipeline produces auditable enriched outputs

This document fixes the architectural boundaries, contracts, and risk-model
interfaces of Phase I. Implementation specs (`PHASE_I1..*.md`) follow only
after the moat foundation is locked.

---

## 1. Scope

Phase I extends EisaX from "portfolio reporting" to **institutional context graph + forward risk**:

| Engine | Purpose | Depends on |
|--------|---------|-----------|
| **I1 — Context Graph**       | A typed knowledge graph linking issuers ↔ regulators ↔ sovereigns ↔ instruments ↔ index memberships. Source: `core.data_layer.gcc_metadata` + ingestion outputs. | Data Layer |
| **I2 — Real-Time Risk Engine** | VaR/CVaR/ES/stress correlations/regime detection at portfolio + position level, refreshed on the 15-min cache cadence. | Data Layer + I1 + Phase H4 (factor model) |
| **I3 — Governance Audit**    | Per-decision audit trail: every Section A–I claim gets a provenance signature + reproducibility hash. | Phase H envelope contract + I1 |
| **I4 — Cross-Border Sensitivities** | Sovereign sensitivity, oil-beta exposure, currency-peg risk, regional shariah-compliance constraints. | I1 |
| **I5 — Committee Workflow Layer** | Multi-party decision states (proposed → defended → ratified → archived), tied to committee mode (H5) and audit (I3). | H5 + I3 |

Phase I is **non-quant first**. Risk math (I2) is the smallest engine. The
moat is the graph (I1) + audit (I3) + workflow (I5).

---

## 2. Non-negotiable rules (inherited from Phase H + extended)

1. **No direct cache reads.** All I-engines read through `core.data_layer.*`.
2. **No invented data.** Every datum carries the full provenance contract.
3. **Deterministic envelopes.** Outputs use `core.data_layer.utils.versioning.embed_version` (never `make_envelope` directly).
4. **Provenance tier visible in every output.** Section G + Section J (new) display Tier-1/2/3/4 coverage stats.
5. **Bilingual EN/AR.** Every label routed through `phase_h.report_helpers.LABELS`.
6. **Tone discipline.** No retail vocabulary. Modal verbs in committee/forward sections.
7. **Read-only against `market_cache/`.** Same constraint as Phase H.
8. **Feature-flag gated.** Every engine has its own `EISAX_PHASE_I_*` flag with category `phase_i`.

---

## 3. Package layout (planned)

```
/home/ubuntu/investwise/phase_i/
  __init__.py                   # PHASE_I_VERSION = "0.1.0"
  feature_flags.py              # EISAX_PHASE_I_* env toggles
  schemas.py                    # GraphNode, GraphEdge, RiskEnvelope, GovernanceTrail, …
  report_helpers.py             # bilingual labels (extends Phase H LABELS)
  orchestrator.py               # augment_result + inject_phase_i_sections
  context_graph.py              # I1 — graph build + queries
  real_time_risk.py             # I2 — VaR/CVaR/ES + stress correlation
  governance_audit.py           # I3 — provenance hashing + section claim trail
  cross_border.py               # I4 — sovereign sensitivity matrix
  committee_workflow.py         # I5 — decision-state machine
  tests/                        # 1 test module per engine + a graph regression suite
```

**Hard rule:** every I-engine MUST consume Data Layer outputs via the
canonical accessors below. No engine may import private helpers from
`phase_h.*` except the published envelope/contract surface.

---

## 4. Contracts

### 4.1 Graph node / edge contract (I1)

Stored under `phase_i.schemas.GraphNode` / `GraphEdge`. Both carry the
provenance contract from `core.data_layer.gcc_metadata.MetadataField`:

```python
class GraphNode(TypedDict):
    id: str                     # canonical ticker / entity id
    kind: str                   # "issuer" | "regulator" | "sovereign" | "instrument" | "index"
    country: str                # ISO-2 (uppercase)
    attributes: Dict[str, MetadataField]
    provenance_tier: int        # 1 best ... 4 missing


class GraphEdge(TypedDict):
    from_id: str
    to_id: str
    relation: str               # "listed_on" | "regulated_by" | "owns" | "guarantees" | …
    weight: Optional[float]     # e.g. ownership %
    attributes: Dict[str, MetadataField]
```

Graph queries are pure — they never mutate the graph and never trigger
network I/O. Bulk graph rebuilds happen in ingestion (out of band).

### 4.2 Real-time risk envelope (I2)

```python
class RiskEnvelope(TypedDict):
    horizon_days: int
    var_pct: float              # 95% historical VaR
    cvar_pct: float             # 95% conditional VaR / expected shortfall
    stress_scenarios: List[StressScenario]   # named regimes (e.g. "oil_shock")
    factor_exposures: Dict[str, float]       # from Phase H4
    regime_tag: str             # "soft_landing" | "recession" | … (from H3 scenarios)
    snapshot_ts: str            # source data-layer snapshot timestamp
```

I2 outputs always go through `embed_version(snapshot_ts=...)` so each
risk envelope is deterministic w.r.t. the underlying snapshot.

### 4.3 Governance trail (I3)

Every claim rendered into Section A–I gets a `ClaimSignature`:

```python
class ClaimSignature(TypedDict):
    claim_id: str               # stable hash of (section, claim_text)
    section: str                # "A" | "B" | ... | "I"
    sources: List[str]          # data_layer / engine envelope ids
    provenance_tier: int        # min tier across all sources
    reproducibility_hash: str   # sha256 of (inputs + engine version)
```

The audit appendix (Section G) is extended into Section J (new) which
lists, for every section claim, its tier + reproducibility hash. This is
the institutional "show your work" surface.

### 4.4 Cross-border sensitivity matrix (I4)

```python
class SovereignSensitivity(TypedDict):
    ticker: str
    sovereign: str              # ISO-2 country code
    sensitivity_score: float    # 0..1
    drivers: List[str]          # "fx_peg_break" | "oil_price" | "fiscal_stance" | …
    methodology: str
    confidence: float
```

All values populated from `gcc_metadata` provenance fields. I4 does NOT
invent sensitivity scores — it derives them from the verified Tier-1/2
inputs and labels every output as `data_quality="derived"`.

### 4.5 Committee workflow state (I5)

```python
class CommitteeState(TypedDict):
    decision_id: str
    state: str                  # "proposed" | "defended" | "ratified" | "archived" | "rejected"
    transitions: List[Transition]
    artifacts: List[str]        # snapshot ids + claim signatures
    committee_mode: str         # from EISAX_COMMITTEE_MODE
    chairperson: Optional[str]
```

State transitions are append-only — the workflow never deletes history.

---

## 5. Section ordering (post-Phase-I)

```
A → B → C
  → [Benchmark Relative (H1)]
  → [Factor Risk Decomposition (H4)]
  → [Context Graph Summary (I1)]                  ← new
→ D → E
  → [Execution Efficiency (H2)]
  → [Real-Time Risk Envelope (I2)]                ← new
→ F
→ ## H. Forward Scenario Distribution
→ ## I. Investment Committee Brief
  → [Workflow State (I5)]                         ← new
→ ## J. Governance Trail                          ← new top-level
→ ## G. Audit Appendix (extended with Phase I reproducibility)
```

The audit appendix (G) stays the **last** top-level section.

---

## 6. Determinism + provenance gating

- Every Phase I engine output MUST pass `is_deterministic_envelope`.
- Engines refuse to render Section J entries when source provenance tier == 4.
- A new flag `EISAX_PHASE_I_REQUIRE_TIER_1` (default False, opt-in for CIO mode) hard-fails the render when any displayed claim has tier > 1.

---

## 7. Out of scope (do NOT build in Phase I)

- Order management / execution routing — Phase H2 stops at slippage estimation.
- Trade settlement infrastructure.
- Anything Bloomberg-clone-shaped (live tick streams, terminal UI, news headlines).
- Direct data scraping inside engines — all enrichment lives in `core/data_layer/ingestion/`.

---

## 8. Implementation gating

| Gate | Required artefact | Owner |
|------|-------------------|-------|
| G1 — Moat threshold | ≥60 curated GCC entries with ≥3 Tier-1 fields each | Data Layer team |
| G2 — Ingestion job  | `gcc_ingestion_spec.md` + skeleton + ≥1 sample run with provenance audit | Data Layer team |
| G3 — Risk math kernel | `phase_i/real_time_risk.py` numerical layer reviewed by Claude | Phase I author |
| G4 — Bilingual labels | `LABELS` extended with all Section J / I-engine strings | Phase I author |
| G5 — Regression baseline | Updated golden snapshots + 7+ regression cases in `phase_h.testing.runner` | QA |

Engines proceed in order: **I1 → I3 → I4 → I5 → I2**. Risk math last so
the graph/audit/workflow can lock the provenance surface first.

---

## 9. Versioning + roll-out

- `phase_i/__init__.py::PHASE_I_VERSION` starts at `"0.1.0"`; each engine
  bumps `ENGINE_VERSION` independently.
- Master flag `EISAX_PHASE_I_ENABLED` defaults **OFF** until G1–G5 pass.
- Staging-only flag `EISAX_PHASE_I_STAGING_ONLY` gates render in production
  even when the master flag is on, until a green CIO review.

---

*This spec is intentionally short. The hard work is the moat (GCC
metadata + ingestion), not the engine boilerplate. Implementation specs
will land per-engine after gates G1 + G2 are met.*
