"""
core.data_layer.gcc_metadata — provenance-aware GCC + Egypt equity metadata.

CRITICAL DESIGN RULE: this module does **not** invent values. Every field
is a structured `MetadataField` carrying the institutional provenance
contract:

    value          : the actual datum (None when missing)
    as_of_date     : ISO date the source publication was issued
    source_type    : "issuer" | "exchange" | "regulator" | "derived" |
                     "fallback" | "missing"
    confidence     : 0.0–1.0 calibrated confidence in `value`
    data_quality   : "verified" | "derived" | "estimated" | "missing"
    methodology    : short string describing how the datum was obtained
    fallback_used  : True when this entry is a placeholder, not a fact

A field with no authoritative source is always emitted as
`MetadataField(value=None, data_quality="missing", source_type="missing",
fallback_used=True, ...)` so engines can route around it and the audit
appendix can flag every fallback.

Provenance tiers (informational — used by `provenance_tier(field)`):
    Tier 1 = Exchange / issuer / regulator
    Tier 2 = MSCI / FTSE / Refinitiv-derived
    Tier 3 = Internal derived estimates
    Tier 4 = Missing / inferred
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from phase_h.registry import FeatureRegistry

from . import _flags  # noqa: F401


# ──────────────────────────────────────────────────────────────────────
# Per-field provenance container
# ──────────────────────────────────────────────────────────────────────

# Strict enum vocabularies.
SOURCE_TYPES: Tuple[str, ...] = (
    "issuer", "exchange", "regulator", "derived", "fallback", "missing",
)
DATA_QUALITY_LEVELS: Tuple[str, ...] = (
    "verified", "derived", "estimated", "missing",
)


@dataclass(frozen=True)
class MetadataField:
    """Provenance-aware wrapper around a single metadata value."""
    value: Any = None
    as_of_date: Optional[str] = None
    source_type: str = "missing"
    confidence: float = 0.0
    data_quality: str = "missing"
    methodology: str = ""
    fallback_used: bool = True

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(
                f"invalid source_type {self.source_type!r} — must be one of {SOURCE_TYPES}"
            )
        if self.data_quality not in DATA_QUALITY_LEVELS:
            raise ValueError(
                f"invalid data_quality {self.data_quality!r} — must be one of {DATA_QUALITY_LEVELS}"
            )
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def provenance_tier(field_dict: Mapping[str, Any]) -> int:
    """Map a stored field to its provenance tier (1 best, 4 missing)."""
    st = str(field_dict.get("source_type", "missing"))
    dq = str(field_dict.get("data_quality", "missing"))
    if st in {"issuer", "exchange", "regulator"} and dq == "verified":
        return 1
    if st == "derived" and dq in {"verified", "derived"}:
        return 2
    if dq == "estimated":
        return 3
    return 4


# Convenience constructors.
def _verified(
    value: Any, *,
    source_type: str,
    confidence: float,
    methodology: str,
    as_of_date: str,
) -> MetadataField:
    """Tier-1 record — direct from issuer / exchange / regulator filing."""
    return MetadataField(
        value=value,
        as_of_date=as_of_date,
        source_type=source_type,
        confidence=confidence,
        data_quality="verified",
        methodology=methodology,
        fallback_used=False,
    )


def _derived(
    value: Any, *,
    confidence: float,
    methodology: str,
    as_of_date: str,
) -> MetadataField:
    """Tier-2 record — derived from secondary index data (MSCI / FTSE / Refinitiv)."""
    return MetadataField(
        value=value,
        as_of_date=as_of_date,
        source_type="derived",
        confidence=confidence,
        data_quality="derived",
        methodology=methodology,
        fallback_used=False,
    )


def _estimated(
    value: Any, *,
    confidence: float,
    methodology: str,
    as_of_date: str,
) -> MetadataField:
    """Tier-3 record — internal estimate; transparent about the heuristic used."""
    return MetadataField(
        value=value,
        as_of_date=as_of_date,
        source_type="derived",
        confidence=confidence,
        data_quality="estimated",
        methodology=methodology,
        fallback_used=False,
    )


def _missing() -> MetadataField:
    """Tier-4 record — no authoritative source available."""
    return MetadataField(
        value=None,
        as_of_date=None,
        source_type="missing",
        confidence=0.0,
        data_quality="missing",
        methodology="no_authoritative_source",
        fallback_used=True,
    )


# Backwards-compat alias for any helpers that still call _unverified()
_unverified = _missing


# Schema — every entry MUST carry exactly these keys (no more, no less).
SCHEMA_FIELDS: Tuple[str, ...] = (
    "ticker",
    "country",
    "exchange",
    "sector",
    "parent_company",
    "government_ownership_pct",
    "strategic_asset_flag",
    "dividend_stability_score",
    "domestic_vs_export_split",
    "sovereign_sensitivity",
    "oil_beta_dependency",
    "shariah_compliant_flag",
    "inclusion_indices",
    "free_float_pct",
    "notes",
)


def _entry(
    *,
    ticker: str,
    country: MetadataField,
    exchange: MetadataField,
    sector: MetadataField,
    parent_company: MetadataField = _unverified(),
    government_ownership_pct: MetadataField = _unverified(),
    strategic_asset_flag: MetadataField = _unverified(),
    dividend_stability_score: MetadataField = _unverified(),
    domestic_vs_export_split: MetadataField = _unverified(),
    sovereign_sensitivity: MetadataField = _unverified(),
    oil_beta_dependency: MetadataField = _unverified(),
    shariah_compliant_flag: MetadataField = _unverified(),
    inclusion_indices: MetadataField = _unverified(),
    free_float_pct: MetadataField = _unverified(),
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "country": country.as_dict(),
        "exchange": exchange.as_dict(),
        "sector": sector.as_dict(),
        "parent_company": parent_company.as_dict(),
        "government_ownership_pct": government_ownership_pct.as_dict(),
        "strategic_asset_flag": strategic_asset_flag.as_dict(),
        "dividend_stability_score": dividend_stability_score.as_dict(),
        "domestic_vs_export_split": domestic_vs_export_split.as_dict(),
        "sovereign_sensitivity": sovereign_sensitivity.as_dict(),
        "oil_beta_dependency": oil_beta_dependency.as_dict(),
        "shariah_compliant_flag": shariah_compliant_flag.as_dict(),
        "inclusion_indices": inclusion_indices.as_dict(),
        "free_float_pct": free_float_pct.as_dict(),
        "notes": notes,
    }


# ──────────────────────────────────────────────────────────────────────
# Curated entries
#
# Each entry populates ONLY fields that are publicly verifiable. All
# quantitative scoring fields (dividend_stability_score, oil_beta_dependency,
# domestic_vs_export_split, free_float_pct) are left as `_unverified()`
# until an authoritative ingest job replaces them.
# ──────────────────────────────────────────────────────────────────────

_AS_OF = "2026-05-17"

# Convenience builders — map old call-sites onto the strict enum vocabulary.
# `_iss` / `_exc` / `_reg` produce Tier-1 verified records; `_der` produces
# Tier-2 derived records; `_est` produces Tier-3 estimates. Engines should
# branch on `data_quality` and `source_type`, never on the helper name.

def _iss(value: Any, *, confidence: float, methodology: str) -> MetadataField:
    """Tier-1 — issuer disclosure (IPO prospectus, annual report, IR filing)."""
    return _verified(value, source_type="issuer",
                     confidence=confidence, methodology=methodology,
                     as_of_date=_AS_OF)


def _exc(value: Any, *, confidence: float, methodology: str) -> MetadataField:
    """Tier-1 — exchange-published fact (listing record, sector assignment)."""
    return _verified(value, source_type="exchange",
                     confidence=confidence, methodology=methodology,
                     as_of_date=_AS_OF)


def _reg(value: Any, *, confidence: float, methodology: str) -> MetadataField:
    """Tier-1 — regulator-published fact (SAMA / CMA / CBE / SCA)."""
    return _verified(value, source_type="regulator",
                     confidence=confidence, methodology=methodology,
                     as_of_date=_AS_OF)


def _der(value: Any, *, confidence: float, methodology: str) -> MetadataField:
    """Tier-2 — derived from index/data-vendor reference (MSCI/FTSE/Refinitiv)."""
    return _derived(value, confidence=confidence, methodology=methodology,
                    as_of_date=_AS_OF)


def _est(value: Any, *, confidence: float, methodology: str) -> MetadataField:
    """Tier-3 — internal estimate; methodology string must be explicit."""
    return _estimated(value, confidence=confidence, methodology=methodology,
                      as_of_date=_AS_OF)


# Legacy aliases — old _pd / _ck / _idx still appear in inline entries below
# until the full seed migration. They are intentionally distinct functions so
# the source_type stays faithful to the underlying provenance.
def _pd(value: Any, *, confidence: float, methodology: str) -> MetadataField:
    return _iss(value, confidence=confidence, methodology=methodology)


def _ck(value: Any, *, confidence: float, methodology: str) -> MetadataField:
    return _exc(value, confidence=confidence, methodology=methodology)


def _idx(value: Any, *, confidence: float, methodology: str) -> MetadataField:
    return _der(value, confidence=confidence, methodology=methodology)


# ──────────────────────────────────────────────────────────────────────
# Reference-table derivations
# ──────────────────────────────────────────────────────────────────────
#
# These helpers convert the lookup tables under
# `core/data_layer/seed/_shariah_index.py` and
# `core/data_layer/seed/_sovereign_ownership.py` into MetadataField
# instances. Lookups that miss the table return `_missing()` so the
# audit trail captures the gap rather than inventing a default.

def shariah_field_for(ticker: str) -> MetadataField:
    """Tier-2 derived Shariah-compliance field for a known ticker."""
    from .seed._shariah_index import shariah_provenance  # local — avoid cycle
    hit = shariah_provenance(ticker)
    if hit is None:
        return _missing()
    index_name, doc_id, conf = hit
    return _derived(
        value=True,
        confidence=conf,
        methodology=f"derived_from:{index_name}:{doc_id}",
        as_of_date=_AS_OF,
    )


def sovereign_parent_field_for(ticker: str) -> MetadataField:
    """Tier-1 issuer-disclosed parent_company derived from the sovereign table."""
    from .seed._sovereign_ownership import sovereign_record
    rec = sovereign_record(ticker)
    if rec is None:
        return _missing()
    return _iss(rec.parent, confidence=rec.confidence, methodology=rec.document_id)


def strategic_asset_field_for(ticker: str) -> MetadataField:
    """Tier-1 strategic-asset flag — True only when sovereign table designates it."""
    from .seed._sovereign_ownership import sovereign_record
    rec = sovereign_record(ticker)
    if rec is None:
        return _missing()
    return _iss(
        value=bool(rec.strategic_designated),
        confidence=rec.confidence,
        methodology=rec.document_id,
    )


GCC_METADATA: Dict[str, Dict[str, Any]] = {

    # ── KSA / Tadawul ────────────────────────────────────────────────
    "TADAWUL:2222": _entry(
        ticker="TADAWUL:2222",
        country=_ck("KSA", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("Tadawul", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Energy", confidence=1.0, methodology="exchange_sector_assignment"),
        parent_company=_pd("Government of Saudi Arabia / PIF",
                           confidence=0.95, methodology="aramco_ipo_prospectus_2019"),
        notes="Saudi Aramco — quantitative fields require an authoritative ingest "
              "(IR data feed). Provenance fields populated above are from public "
              "IPO prospectus and exchange listing only.",
    ),
    "TADAWUL:1180": _entry(
        ticker="TADAWUL:1180",
        country=_ck("KSA", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("Tadawul", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Financials", confidence=1.0, methodology="exchange_sector_assignment"),
        notes="Saudi National Bank — verified facts: country/exchange/sector only.",
    ),
    "TADAWUL:2010": _entry(
        ticker="TADAWUL:2010",
        country=_ck("KSA", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("Tadawul", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Materials", confidence=1.0, methodology="exchange_sector_assignment"),
        notes="SABIC — verified facts: country/exchange/sector only.",
    ),
    "TADAWUL:7010": _entry(
        ticker="TADAWUL:7010",
        country=_ck("KSA", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("Tadawul", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Communication Services", confidence=1.0, methodology="exchange_sector_assignment"),
        notes="STC — verified facts: country/exchange/sector only.",
    ),
    "TADAWUL:2280": _entry(
        ticker="TADAWUL:2280",
        country=_ck("KSA", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("Tadawul", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Consumer Staples", confidence=1.0, methodology="exchange_sector_assignment"),
        notes="Almarai — verified facts: country/exchange/sector only.",
    ),

    # ── UAE / DFM + ADX ─────────────────────────────────────────────
    "ADX:IHC": _entry(
        ticker="ADX:IHC",
        country=_ck("UAE", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("ADX", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Diversified", confidence=0.9, methodology="exchange_sector_assignment"),
        notes="International Holding Company — verified facts: country/exchange/sector only.",
    ),
    "ADX:FAB": _entry(
        ticker="ADX:FAB",
        country=_ck("UAE", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("ADX", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Financials", confidence=1.0, methodology="exchange_sector_assignment"),
        notes="First Abu Dhabi Bank — verified facts: country/exchange/sector only.",
    ),
    "DFM:EMAAR": _entry(
        ticker="DFM:EMAAR",
        country=_ck("UAE", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("DFM", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Real Estate", confidence=1.0, methodology="exchange_sector_assignment"),
        notes="Emaar Properties — verified facts: country/exchange/sector only.",
    ),
    "ADX:ADNOCGAS": _entry(
        ticker="ADX:ADNOCGAS",
        country=_ck("UAE", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("ADX", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Energy", confidence=1.0, methodology="exchange_sector_assignment"),
        parent_company=_pd("ADNOC Group",
                           confidence=0.95, methodology="adnoc_gas_ipo_prospectus_2023"),
        notes="ADNOC Gas — verified facts: country/exchange/sector/parent.",
    ),

    # ── Qatar / QSE ─────────────────────────────────────────────────
    "QSE:QNBK": _entry(
        ticker="QSE:QNBK",
        country=_ck("Qatar", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("QSE", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Financials", confidence=1.0, methodology="exchange_sector_assignment"),
        notes="Qatar National Bank — verified facts: country/exchange/sector only.",
    ),
    "QSE:IQCD": _entry(
        ticker="QSE:IQCD",
        country=_ck("Qatar", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("QSE", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Materials", confidence=1.0, methodology="exchange_sector_assignment"),
        notes="Industries Qatar — verified facts: country/exchange/sector only.",
    ),

    # ── Egypt / EGX ─────────────────────────────────────────────────
    "EGX:COMI": _entry(
        ticker="EGX:COMI",
        country=_ck("Egypt", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("EGX", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Financials", confidence=1.0, methodology="exchange_sector_assignment"),
        notes="Commercial International Bank — verified facts: country/exchange/sector only.",
    ),
    "EGX:TMGH": _entry(
        ticker="EGX:TMGH",
        country=_ck("Egypt", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("EGX", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Real Estate", confidence=1.0, methodology="exchange_sector_assignment"),
        notes="Talaat Moustafa Group — verified facts: country/exchange/sector only.",
    ),
    "EGX:FWRY": _entry(
        ticker="EGX:FWRY",
        country=_ck("Egypt", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("EGX", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Information Technology", confidence=0.9, methodology="exchange_sector_assignment"),
        notes="Fawry — verified facts: country/exchange/sector only.",
    ),

    # ── Kuwait / KSE ────────────────────────────────────────────────
    "KSE:NBK": _entry(
        ticker="KSE:NBK",
        country=_ck("Kuwait", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("KSE", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Financials", confidence=1.0, methodology="exchange_sector_assignment"),
        notes="National Bank of Kuwait — verified facts: country/exchange/sector only.",
    ),

    # ── Bahrain / BHB ───────────────────────────────────────────────
    "BHB:ALBH": _entry(
        ticker="BHB:ALBH",
        country=_ck("Bahrain", confidence=1.0, methodology="exchange_listing"),
        exchange=_ck("BHB", confidence=1.0, methodology="exchange_listing"),
        sector=_ck("Materials", confidence=1.0, methodology="exchange_sector_assignment"),
        notes="Aluminium Bahrain — verified facts: country/exchange/sector only.",
    ),
}


# ──────────────────────────────────────────────────────────────────────
# Seed merge — pull in per-market modules under core/data_layer/seed/*
# The seed modules import their `_entry`/`_iss`/`_exc`/`_missing` helpers
# from this module — which is safe because those helpers are defined
# above this import block.
# ──────────────────────────────────────────────────────────────────────

try:
    from .seed import build_registry as _build_seed_registry  # type: ignore[import-not-found]
    _seed_entries = _build_seed_registry()
    # Seed entries take precedence — they carry the curated provenance.
    GCC_METADATA.update(_seed_entries)
except Exception as _seed_exc:  # pragma: no cover — defensive
    import logging as _logging
    _logging.getLogger("data_layer.gcc_metadata").warning(
        "seed merge skipped: %r", _seed_exc,
    )


# ──────────────────────────────────────────────────────────────────────
# Default fallback entry — never invents values
# ──────────────────────────────────────────────────────────────────────

def _default_unknown_entry(ticker: str) -> Dict[str, Any]:
    base = _missing().as_dict()
    return {
        "ticker": ticker,
        "country": dict(base),
        "exchange": dict(base),
        "sector": dict(base),
        "parent_company": dict(base),
        "government_ownership_pct": dict(base),
        "strategic_asset_flag": dict(base),
        "dividend_stability_score": dict(base),
        "domestic_vs_export_split": dict(base),
        "sovereign_sensitivity": dict(base),
        "oil_beta_dependency": dict(base),
        "shariah_compliant_flag": dict(base),
        "inclusion_indices": dict(base),
        "free_float_pct": dict(base),
        "notes": "ticker not in curated registry — every field is missing",
    }


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def get_gcc_metadata(ticker: str) -> Dict[str, Any]:
    """
    Return a provenance-aware metadata record for `ticker`.

    Output is ALWAYS shape-stable (every SCHEMA_FIELDS key is present).
    Missing tickers return the default unknown entry with `source="fallback"`
    so engines never branch on `None`.
    """
    if not FeatureRegistry.is_enabled("data_layer_gcc_metadata"):
        out = _default_unknown_entry(ticker or "")
        out["source"] = "feature_disabled"
        return out
    if not ticker:
        out = _default_unknown_entry("")
        out["source"] = "empty_input"
        return out
    key = ticker.strip().upper()
    if key in GCC_METADATA:
        out = dict(GCC_METADATA[key])
        out["source"] = "curated"
        return out
    # Bare-symbol match (e.g. "2222" → "TADAWUL:2222")
    if ":" not in key:
        for full_key, payload in GCC_METADATA.items():
            if full_key.split(":")[-1] == key:
                out = dict(payload)
                out["source"] = "curated_via_bare_match"
                return out
    out = _default_unknown_entry(key)
    out["source"] = "fallback"
    return out


def list_gcc_tickers(country: Optional[str] = None) -> List[str]:
    """List all curated GCC tickers; optional country filter."""
    if country is None:
        return list(GCC_METADATA.keys())
    needle = country.strip().lower()
    return [
        tk for tk, meta in GCC_METADATA.items()
        if (meta.get("country", {}).get("value") or "").lower() == needle
    ]


def validate_entry(entry: Mapping[str, Any]) -> List[str]:
    """Return a list of missing schema fields for a registry entry."""
    return [f for f in SCHEMA_FIELDS if f not in entry]


def provenance_summary(entry: Mapping[str, Any]) -> Dict[str, int]:
    """
    Count provenance buckets across an entry's fields. Used by the audit
    appendix so reports can show 'verified=4 / derived=2 / estimated=1 /
    missing=8' headers. Counts cover the strict enum vocabulary plus the
    fallback flag and the four provenance tiers.
    """
    counts: Dict[str, int] = {
        "verified": 0, "derived": 0, "estimated": 0, "missing": 0,
        "fallback_used": 0,
        "tier_1": 0, "tier_2": 0, "tier_3": 0, "tier_4": 0,
    }
    for k in SCHEMA_FIELDS:
        v = entry.get(k)
        if not isinstance(v, Mapping):
            continue
        dq = str(v.get("data_quality", "missing"))
        if dq in counts:
            counts[dq] += 1
        if v.get("fallback_used"):
            counts["fallback_used"] += 1
        tier = provenance_tier(v)
        counts[f"tier_{tier}"] += 1
    return counts


__all__ = [
    "MetadataField",
    "SCHEMA_FIELDS",
    "SOURCE_TYPES",
    "DATA_QUALITY_LEVELS",
    "GCC_METADATA",
    "get_gcc_metadata",
    "list_gcc_tickers",
    "validate_entry",
    "provenance_summary",
    "provenance_tier",
]
