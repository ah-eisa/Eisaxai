"""
core.data_layer.seed._sovereign_ownership — sovereign / state-linked
ownership reference for GCC + Egypt issuers.

Each entry records the publicly-known sovereign-linked controlling
shareholder. The flag here drives two derived fields:

    strategic_asset_flag      — True when sovereign ownership is the
                                controlling block AND the entity is
                                state-designated (e.g. national-champion
                                / regulated utility / strategic energy).
    parent_company            — set only where IPO prospectus / annual
                                report explicitly names the owner.

Provenance rule: every entry must cite the source document. Confidence
above 0.85 reserved for issuer-disclosed cases. Conventional banks where
a sovereign fund holds a meaningful but non-controlling stake (e.g. FAB)
are flagged with `strategic_designated=False` so the report layer can
distinguish "sovereign owns >50%" from "sovereign-linked but private-led".

Conservative posture: tickers we cannot verify against a public document
are intentionally absent. Engines treat absence as `_missing()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class SovereignOwnership:
    parent: str                       # Display name of controlling entity
    document_id: str                  # Source citation
    confidence: float                 # 0..1
    strategic_designated: bool        # True = national-champion / strategic asset
    notes: str = ""


SOVEREIGN_OWNERSHIP: Dict[str, SovereignOwnership] = {

    # ── KSA ─────────────────────────────────────────────────────────
    "TADAWUL:2222": SovereignOwnership(
        parent="Government of Saudi Arabia / PIF",
        document_id="aramco_ipo_prospectus_2019_ownership_section",
        confidence=0.95, strategic_designated=True,
        notes="Government + PIF jointly hold the controlling block.",
    ),
    "TADAWUL:2010": SovereignOwnership(
        parent="PIF + Saudi Aramco",
        document_id="sabic_acquisition_disclosure_2020",
        confidence=0.9, strategic_designated=True,
        notes="Aramco acquired PIF's 70% stake in 2020.",
    ),
    "TADAWUL:1180": SovereignOwnership(
        parent="Public Investment Fund (PIF)",
        document_id="snb_annual_report_2023_major_shareholders",
        confidence=0.9, strategic_designated=False,
        notes="PIF is the largest shareholder; the bank operates commercially.",
    ),
    "TADAWUL:1211": SovereignOwnership(
        parent="Public Investment Fund (PIF)",
        document_id="maaden_annual_report_2023_shareholding_section",
        confidence=0.9, strategic_designated=True,
        notes="National mining champion under Vision 2030.",
    ),
    "TADAWUL:7010": SovereignOwnership(
        parent="Public Investment Fund (PIF)",
        document_id="stc_annual_report_2023_major_shareholders",
        confidence=0.9, strategic_designated=True,
        notes="PIF holds the majority stake in Saudi Telecom.",
    ),
    "TADAWUL:5110": SovereignOwnership(
        parent="Public Investment Fund (PIF)",
        document_id="saudi_electricity_disclosure_2022",
        confidence=0.9, strategic_designated=True,
        notes="State-regulated electricity utility; PIF majority shareholder.",
    ),
    "TADAWUL:2082": SovereignOwnership(
        parent="Public Investment Fund (PIF)",
        document_id="acwa_power_ipo_prospectus_2021",
        confidence=0.85, strategic_designated=True,
        notes="PIF acquired the controlling stake at IPO.",
    ),

    # ── UAE ─────────────────────────────────────────────────────────
    "ADX:ADNOCGAS": SovereignOwnership(
        parent="Abu Dhabi National Oil Company (ADNOC)",
        document_id="adnoc_gas_ipo_prospectus_2023",
        confidence=0.95, strategic_designated=True,
    ),
    "ADX:ADNOCDIST": SovereignOwnership(
        parent="Abu Dhabi National Oil Company (ADNOC)",
        document_id="adnoc_distribution_ipo_prospectus_2017",
        confidence=0.95, strategic_designated=True,
    ),
    "ADX:ADNOCDRILL": SovereignOwnership(
        parent="Abu Dhabi National Oil Company (ADNOC)",
        document_id="adnoc_drilling_ipo_prospectus_2021",
        confidence=0.95, strategic_designated=True,
    ),
    "ADX:TAQA": SovereignOwnership(
        parent="Abu Dhabi Developmental Holding Company (ADQ)",
        document_id="taqa_annual_report_2023_shareholders",
        confidence=0.9, strategic_designated=True,
        notes="ADQ holds the controlling stake.",
    ),
    "ADX:FAB": SovereignOwnership(
        parent="Mubadala Investment Company + ADQ",
        document_id="fab_annual_report_2023_major_shareholders",
        confidence=0.85, strategic_designated=False,
        notes="Abu Dhabi sovereign-linked vehicles together hold ~37%; "
              "bank is publicly traded with commercial governance.",
    ),
    "ADX:IHC": SovereignOwnership(
        parent="International Holding Company (Royal Group affiliated)",
        document_id="ihc_annual_report_2023_governance_chapter",
        confidence=0.85, strategic_designated=True,
        notes="Holding company chaired by H.H. Sheikh Tahnoon bin Zayed.",
    ),
    "DFM:DEWA": SovereignOwnership(
        parent="Government of Dubai (Investment Corporation of Dubai)",
        document_id="dewa_ipo_prospectus_2022",
        confidence=0.95, strategic_designated=True,
        notes="Dubai's regulated electricity + water monopoly.",
    ),
    "DFM:SALIK": SovereignOwnership(
        parent="Roads & Transport Authority (RTA), Government of Dubai",
        document_id="salik_ipo_prospectus_2022",
        confidence=0.95, strategic_designated=True,
        notes="Toll concession granted by RTA Dubai.",
    ),
    "DFM:EMIRATESNBD": SovereignOwnership(
        parent="Investment Corporation of Dubai (ICD)",
        document_id="emirates_nbd_annual_report_2023",
        confidence=0.9, strategic_designated=False,
        notes="ICD is the largest shareholder; bank operates commercially.",
    ),
    "DFM:EMAAR": SovereignOwnership(
        parent="Investment Corporation of Dubai (ICD)",
        document_id="emaar_annual_report_2023_shareholding_section",
        confidence=0.85, strategic_designated=False,
        notes="ICD is a significant minority shareholder.",
    ),

    # ── Qatar ───────────────────────────────────────────────────────
    "QSE:QNBK": SovereignOwnership(
        parent="Qatar Investment Authority (QIA)",
        document_id="qnb_annual_report_2023_major_shareholders",
        confidence=0.95, strategic_designated=True,
        notes="QIA holds 50% via Qatar Holding LLC.",
    ),
    "QSE:IQCD": SovereignOwnership(
        parent="QatarEnergy",
        document_id="iqcd_annual_report_2022_ownership_chapter",
        confidence=0.9, strategic_designated=True,
        notes="QatarEnergy holds 51% controlling stake.",
    ),
    "QSE:ORDS": SovereignOwnership(
        parent="Qatar Holding LLC (QIA-affiliated)",
        document_id="ooredoo_annual_report_2023_shareholders",
        confidence=0.85, strategic_designated=True,
    ),

    # ── Kuwait ──────────────────────────────────────────────────────
    "KSE:NBK": SovereignOwnership(
        parent="Kuwait-based founding families + free float",
        document_id="nbk_annual_report_2023_governance",
        confidence=0.75, strategic_designated=False,
        notes="Sovereign holds a minority strategic stake; majority free float.",
    ),
}


def sovereign_record(ticker: str) -> Optional[SovereignOwnership]:
    return SOVEREIGN_OWNERSHIP.get((ticker or "").upper())


__all__ = ["SovereignOwnership", "SOVEREIGN_OWNERSHIP", "sovereign_record"]
