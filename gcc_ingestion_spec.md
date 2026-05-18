# GCC Metadata Ingestion Spec

**Goal:** populate `core.data_layer.gcc_metadata.GCC_METADATA` with a
provenance-audited dataset of GCC + Egypt listed equities. **Coverage <
quality** — 60 high-quality names beats 600 noisy ones.

---

## 1. Non-negotiable rules

1. **No invented values.** Ever. Missing = `_missing()`, period.
2. **Every datum carries the full provenance contract** (`value`,
   `as_of_date`, `source_type`, `confidence`, `data_quality`,
   `methodology`, `fallback_used`).
3. **Read-only against `market_cache/`.** Ingestion writes only into
   `core/data_layer/ingestion/sources/*.json` and into the curated
   registry (`core/data_layer/seed/*.py`).
4. **Deterministic outputs.** Same source state ⇒ same registry hash.
5. **Manual curation > scraping.** Tier-1 entries are reviewed by hand.
6. **No PII, no proprietary data, no credentialed APIs without audit.**

---

## 2. Source taxonomy → `source_type` enum

| `source_type` | Examples | Reliability | Used for |
|---|---|---|---|
| `issuer` | IPO prospectus, annual report, official IR page | Very high | Government ownership, parent, dividend history, free float |
| `exchange` | Tadawul / DFM / ADX / QSE / EGX listing tables, sector assignments | Very high | country, exchange, sector, listing date, trading currency |
| `regulator` | SAMA / CMA / SCA / CBE / FRA publications | Very high | strategic-asset designations, sovereign sensitivity |
| `derived` | MSCI EM / FTSE EM / S&P GCC constituents, Refinitiv schemas | High | inclusion_indices, free_float estimates |
| `fallback` | Cross-referenced public data (Wikipedia / Reuters), low confidence | Low | Last-resort context; flagged in audit |
| `missing` | No authoritative source available | — | Default for unverified fields |

Ingestion code **must** use one of these strings — anything else raises
`ValueError` via `MetadataField.__post_init__`.

---

## 3. Provenance tier mapping

| Tier | source_type × data_quality | Render policy |
|------|----------------------------|---------------|
| 1 | `issuer/exchange/regulator` × `verified` | Always displayed |
| 2 | `derived` × `verified|derived` | Displayed with "derived" badge |
| 3 | `derived` × `estimated` | Displayed only in expanded view |
| 4 | `missing` × `missing` | Suppressed from public sections; visible in audit only |

---

## 4. Coverage targets (seed phase)

Total = **60 curated names**.

| Market | Target count | Priority sectors |
|--------|--------------|------------------|
| KSA (Tadawul) | 25 | Energy, Financials, Materials, Telecom, Consumer Staples |
| UAE (ADX + DFM) | 25 | Financials, Real Estate, Energy, Utilities, Conglomerates |
| Kuwait + Qatar | 10 (5+5) | Financials, Materials, Telecom |

Each name MUST hit ≥3 Tier-1 fields out of (country, exchange, sector,
parent_company, government_ownership_pct, listing_date).

---

## 5. Priority fields (per ticker)

Ordered by institutional value:

1. **country / exchange / sector** — trivial, exchange-sourced, Tier 1
2. **parent_company** — IR / prospectus, Tier 1 where disclosed
3. **government_ownership_pct** — issuer disclosure required; never estimated
4. **strategic_asset_flag** — regulator/issuer signal only
5. **sovereign_sensitivity** — derived from #2 + #3 + sector
6. **oil_beta_dependency** — derived from sector + Brent regression (later)
7. **shariah_compliant_flag** — from a published shariah-screening source (e.g. AAOIFI compliant indices)
8. **dividend_stability_score** — derived once we have ≥5 years of payout data; **not in seed**
9. **free_float_pct** — issuer disclosure; estimated where derived from index data
10. **inclusion_indices** — MSCI / FTSE / S&P direct verification

Fields 8 and 9 stay `_missing()` in the seed dataset by design.

---

## 6. Seed dataset layout

```
core/data_layer/seed/
  __init__.py
  ksa.py                 # 25 entries
  uae.py                 # 25 entries
  kuwait.py              # 5 entries
  qatar.py               # 5 entries
  _shariah_index.py      # canonical Shariah-compliant ticker set
  _msci_em_gcc.py        # MSCI EM GCC constituent list
```

`gcc_metadata.py` merges seed modules at import time:

```python
from .seed import ksa, uae, kuwait, qatar
GCC_METADATA: Dict[str, Dict[str, Any]] = {}
for mod in (ksa, uae, kuwait, qatar):
    GCC_METADATA.update(mod.ENTRIES)
```

---

## 7. Ingestion pipeline

```
core/data_layer/ingestion/
  __init__.py
  base.py                # IngestionRecord, IngestionRun, contract validators
  sources/
    issuer_filings.py    # Per-issuer IR page parsers (manual, audited)
    exchange_lists.py    # Tadawul / DFM / ADX / QSE / EGX listing tables
    msci_ftse.py         # Index constituents loader (offline JSON snapshots)
    aaoifi_shariah.py    # Shariah screening list
  runners/
    seed_audit.py        # Validates the seed registry vs source manifests
    enrichment.py        # Builds derived fields (sovereign_sensitivity, oil_beta)
    report_audit.py      # Renders an ingestion provenance report
```

### Run modes
- `dry_run` (default): no writes, prints diff
- `audit`: emits `ingestion_audit_YYYYMMDD_HHMM.json`
- `apply`: writes back into `core/data_layer/seed/*.py` (only after human review)

---

## 8. Source manifests

Every source has an `IngestionRecord`:

```python
@dataclass(frozen=True)
class IngestionRecord:
    ticker: str
    field: str
    value: Any
    source_type: str            # enum
    source_url: str             # canonical reference
    document_id: str            # filing id / page id
    captured_at: str            # ISO date the source was captured
    captured_by: str            # human reviewer or "automated:<job>"
    confidence: float
    methodology: str
```

Manifests are stored as JSON snapshots under
`core/data_layer/ingestion/snapshots/<source>/<ticker>.json` so any
future re-run is reproducible.

---

## 9. Reviewer responsibilities

A human reviewer must sign off every Tier-1 entry. The audit log records:

- Reviewer initials / email
- Date of review
- Source URL
- One-line note on disagreements

This is the **moat**. Without it, this is just another data product.

---

## 10. CI / regression hooks

- New test: `core/data_layer/tests/test_seed_coverage.py`
  - Asserts coverage thresholds (≥25 KSA, ≥25 UAE, ≥5 KW, ≥5 QA).
  - Asserts every entry has ≥3 Tier-1 fields.
  - Asserts no `data_quality="estimated"` claim without an explicit `methodology` string.
- New test: `core/data_layer/tests/test_ingestion_audit.py`
  - Runs the seed_audit pipeline in dry-run; fails if any field changed
    without a matching `IngestionRecord` snapshot.

---

## 11. Out of scope

- Live web scraping inside engines (only the manifest loader can read snapshots).
- Real-time corporate-actions feeds.
- M&A / private-market data.

---

## 12. Definition of done (DoD)

The ingestion spec is "done" when:

1. 60 curated entries in `core/data_layer/seed/` with reviewer signatures
2. `test_seed_coverage` and `test_ingestion_audit` are green
3. `provenance_summary(GCC_METADATA[ticker])` shows `tier_1 ≥ 3` for every
   curated ticker
4. A sample report renders Section I (committee mode) and shows
   provenance tier counts in the audit appendix

Phase I implementation is **gated** on this DoD.
