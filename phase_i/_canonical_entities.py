"""
phase_i._canonical_entities — curated catalog of sovereign / regulator /
index nodes.

These are the only allowed targets for `owned_by`, `regulated_by`,
`included_in`, and `shariah_compliant_per` edges. Adding a new entity
requires a manual entry here — there is no auto-discovery.

Each entry carries a `source_document_id` that points to the
constitutional / regulatory document establishing the entity, so the
audit trail can be re-walked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CanonicalEntity:
    id: str
    kind: str                  # "sovereign" | "regulator" | "index"
    label: str
    country: str
    source_document_id: str    # founding / mandate document
    confidence: float = 1.0


# ── Sovereigns ──────────────────────────────────────────────────────
SOVEREIGNS: Dict[str, CanonicalEntity] = {
    "SOV:KSA":     CanonicalEntity("SOV:KSA",     "sovereign",
                                   "Kingdom of Saudi Arabia",
                                   "KSA", "saudi_basic_law_1992"),
    "SOV:UAE":     CanonicalEntity("SOV:UAE",     "sovereign",
                                   "United Arab Emirates",
                                   "UAE", "uae_constitution_1971"),
    "SOV:QATAR":   CanonicalEntity("SOV:QATAR",   "sovereign",
                                   "State of Qatar",
                                   "Qatar", "qatar_permanent_constitution_2004"),
    "SOV:KUWAIT":  CanonicalEntity("SOV:KUWAIT",  "sovereign",
                                   "State of Kuwait",
                                   "Kuwait", "kuwait_constitution_1962"),
    "SOV:BAHRAIN": CanonicalEntity("SOV:BAHRAIN", "sovereign",
                                   "Kingdom of Bahrain",
                                   "Bahrain", "bahrain_constitution_2002"),
    "SOV:EGYPT":   CanonicalEntity("SOV:EGYPT",   "sovereign",
                                   "Arab Republic of Egypt",
                                   "Egypt", "egypt_constitution_2014"),
}


# ── Regulators ──────────────────────────────────────────────────────
REGULATORS: Dict[str, CanonicalEntity] = {
    "REG:CMA-KSA":  CanonicalEntity("REG:CMA-KSA",  "regulator",
                                    "Capital Market Authority (Saudi Arabia)",
                                    "KSA", "ksa_capital_market_law_2003"),
    "REG:SAMA":     CanonicalEntity("REG:SAMA",     "regulator",
                                    "Saudi Central Bank",
                                    "KSA", "ksa_sama_law_2020"),
    "REG:SCA-UAE":  CanonicalEntity("REG:SCA-UAE",  "regulator",
                                    "Securities and Commodities Authority (UAE)",
                                    "UAE", "uae_federal_law_4_of_2000"),
    "REG:CBUAE":    CanonicalEntity("REG:CBUAE",    "regulator",
                                    "Central Bank of the UAE",
                                    "UAE", "uae_federal_law_14_2018"),
    "REG:QFMA":     CanonicalEntity("REG:QFMA",     "regulator",
                                    "Qatar Financial Markets Authority",
                                    "Qatar", "qatar_law_8_of_2012"),
    "REG:QCB":      CanonicalEntity("REG:QCB",      "regulator",
                                    "Qatar Central Bank",
                                    "Qatar", "qatar_law_13_of_2012"),
    "REG:CMA-KW":   CanonicalEntity("REG:CMA-KW",   "regulator",
                                    "Capital Markets Authority (Kuwait)",
                                    "Kuwait", "kuwait_law_7_of_2010"),
    "REG:CBK":      CanonicalEntity("REG:CBK",      "regulator",
                                    "Central Bank of Kuwait",
                                    "Kuwait", "kuwait_law_32_of_1968"),
    "REG:CBB":      CanonicalEntity("REG:CBB",      "regulator",
                                    "Central Bank of Bahrain",
                                    "Bahrain", "bahrain_decree_64_of_2006"),
    "REG:FRA-EG":   CanonicalEntity("REG:FRA-EG",   "regulator",
                                    "Financial Regulatory Authority (Egypt)",
                                    "Egypt", "egypt_law_10_of_2009"),
    "REG:CBE":      CanonicalEntity("REG:CBE",      "regulator",
                                    "Central Bank of Egypt",
                                    "Egypt", "egypt_law_194_of_2020"),
}


# ── Indices ─────────────────────────────────────────────────────────
INDICES: Dict[str, CanonicalEntity] = {
    "IDX:SP-KSA-SHARIAH": CanonicalEntity(
        "IDX:SP-KSA-SHARIAH", "index",
        "S&P Saudi Arabia Shariah Index", "KSA",
        "spdji_saudi_shariah_methodology"),
    "IDX:SP-GCC-SHARIAH": CanonicalEntity(
        "IDX:SP-GCC-SHARIAH", "index",
        "S&P GCC Shariah Index", "Regional",
        "spdji_gcc_shariah_methodology"),
    "IDX:MSCI-EM":        CanonicalEntity(
        "IDX:MSCI-EM",        "index",
        "MSCI Emerging Markets Index", "Global",
        "msci_em_methodology"),
    "IDX:FTSE-EM":        CanonicalEntity(
        "IDX:FTSE-EM",        "index",
        "FTSE Emerging Markets Index", "Global",
        "ftse_em_methodology"),
    "IDX:SP-GCC":         CanonicalEntity(
        "IDX:SP-GCC",         "index",
        "S&P GCC Composite Index", "Regional",
        "spdji_gcc_composite_methodology"),
}


# Exchange-to-regulator mapping — every issuer listed on the exchange
# is structurally regulated by the listing-authority regulator AND, for
# banking entities, by the central bank. For v1 we encode the listing
# regulator only; bank-specific edges require a manifest record.
EXCHANGE_TO_LISTING_REGULATOR: Dict[str, str] = {
    "Tadawul": "REG:CMA-KSA",
    "ADX":     "REG:SCA-UAE",
    "DFM":     "REG:SCA-UAE",
    "QSE":     "REG:QFMA",
    "KSE":     "REG:CMA-KW",
    "BHB":     "REG:CBB",
    "EGX":     "REG:FRA-EG",
}


# Display-name → canonical IDX id, used when wiring Shariah edges from
# `core.data_layer.seed._shariah_index.SHARIAH_REFERENCE` (which stores
# the display name rather than the canonical id).
SHARIAH_INDEX_NAME_TO_ID: Dict[str, str] = {
    "S&P Saudi Arabia Shariah Index": "IDX:SP-KSA-SHARIAH",
    "S&P GCC Shariah Index":          "IDX:SP-GCC-SHARIAH",
}


def all_canonical() -> Dict[str, CanonicalEntity]:
    """Merge sovereigns + regulators + indices into one dict."""
    out: Dict[str, CanonicalEntity] = {}
    for src in (SOVEREIGNS, REGULATORS, INDICES):
        out.update(src)
    return out


__all__ = [
    "CanonicalEntity",
    "SOVEREIGNS",
    "REGULATORS",
    "INDICES",
    "EXCHANGE_TO_LISTING_REGULATOR",
    "SHARIAH_INDEX_NAME_TO_ID",
    "all_canonical",
]
