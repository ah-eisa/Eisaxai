"""
core.data_layer.seed.ksa — curated Tadawul (Saudi) metadata seed.

Coverage target: 25 names.
Tier-1 fields populated (country / exchange / sector) from Tadawul listings.
Parent companies populated from public IPO prospectuses where on record.
All quantitative fields remain `_missing()` until a reviewer signs off.

Reviewer log: seed prepared 2026-05-18 by EisaX Data Layer team.
Pending Tier-1 sign-off on government_ownership_pct, parent_company,
strategic_asset_flag (per gcc_ingestion_spec.md §9).
"""

from __future__ import annotations

from typing import Any, Dict

from ..gcc_metadata import (
    _entry, _exc, _iss, _missing,
    shariah_field_for, sovereign_parent_field_for, strategic_asset_field_for,
)


def _ksa(*, ticker: str, sector: str, parent: str = "", parent_conf: float = 0.0,
         parent_method: str = "", notes: str = "") -> Dict[str, Any]:
    """
    Compact builder for KSA entries. Reference tables take precedence:
      - parent_company is sourced from sovereign_ownership reference when known;
        falls back to inline parent/parent_conf/parent_method if explicitly passed.
      - strategic_asset_flag is derived from the sovereign reference.
      - shariah_compliant_flag is derived from the canonical Shariah index reference.
    """
    parent_field = sovereign_parent_field_for(ticker)
    if parent_field.value is None and parent and parent_conf > 0:
        parent_field = _iss(parent, confidence=parent_conf, methodology=parent_method)
    return _entry(
        ticker=ticker,
        country=_exc("KSA", confidence=1.0, methodology="tadawul_listing"),
        exchange=_exc("Tadawul", confidence=1.0, methodology="tadawul_listing"),
        sector=_exc(sector, confidence=1.0, methodology="tadawul_sector_assignment"),
        parent_company=parent_field,
        strategic_asset_flag=strategic_asset_field_for(ticker),
        shariah_compliant_flag=shariah_field_for(ticker),
        notes=notes,
    )


ENTRIES: Dict[str, Dict[str, Any]] = {

    # ── Energy ──────────────────────────────────────────────────────
    "TADAWUL:2222": _ksa(ticker="TADAWUL:2222", sector="Energy",
        parent="Government of Saudi Arabia / PIF", parent_conf=0.95,
        parent_method="aramco_ipo_prospectus_2019",
        notes="Saudi Aramco — sovereign-controlled supermajor."),
    "TADAWUL:2380": _ksa(ticker="TADAWUL:2380", sector="Energy",
        notes="Rabigh Refining & Petrochemical."),
    "TADAWUL:2381": _ksa(ticker="TADAWUL:2381", sector="Energy",
        notes="Arabian Drilling Company."),

    # ── Materials ───────────────────────────────────────────────────
    "TADAWUL:2010": _ksa(ticker="TADAWUL:2010", sector="Materials",
        parent="PIF + Saudi Aramco", parent_conf=0.9,
        parent_method="sabic_ownership_disclosure_2020",
        notes="SABIC — petrochemical national champion."),
    "TADAWUL:1211": _ksa(ticker="TADAWUL:1211", sector="Materials",
        notes="Saudi Arabian Mining Co (Ma'aden)."),
    "TADAWUL:3030": _ksa(ticker="TADAWUL:3030", sector="Materials",
        notes="Saudi Cement Co."),
    "TADAWUL:2350": _ksa(ticker="TADAWUL:2350", sector="Materials",
        notes="Saudi Kayan Petrochemical."),

    # ── Financials ──────────────────────────────────────────────────
    "TADAWUL:1180": _ksa(ticker="TADAWUL:1180", sector="Financials",
        notes="Saudi National Bank — largest KSA lender."),
    "TADAWUL:1120": _ksa(ticker="TADAWUL:1120", sector="Financials",
        notes="Al Rajhi Bank — largest Islamic bank globally by assets."),
    "TADAWUL:1010": _ksa(ticker="TADAWUL:1010", sector="Financials",
        notes="Riyad Bank."),
    "TADAWUL:1140": _ksa(ticker="TADAWUL:1140", sector="Financials",
        notes="Bank Albilad."),
    "TADAWUL:1150": _ksa(ticker="TADAWUL:1150", sector="Financials",
        notes="Alinma Bank."),
    "TADAWUL:8210": _ksa(ticker="TADAWUL:8210", sector="Financials",
        notes="Bupa Arabia (Health Insurance)."),

    # ── Telecom / Communication Services ───────────────────────────
    "TADAWUL:7010": _ksa(ticker="TADAWUL:7010", sector="Communication Services",
        notes="STC — Saudi Telecom incumbent."),
    "TADAWUL:7020": _ksa(ticker="TADAWUL:7020", sector="Communication Services",
        notes="Etihad Etisalat (Mobily)."),
    "TADAWUL:7030": _ksa(ticker="TADAWUL:7030", sector="Communication Services",
        notes="Zain KSA."),

    # ── Consumer Staples ────────────────────────────────────────────
    "TADAWUL:2280": _ksa(ticker="TADAWUL:2280", sector="Consumer Staples",
        notes="Almarai — staples leader, defensive cash flows."),
    "TADAWUL:6010": _ksa(ticker="TADAWUL:6010", sector="Consumer Staples",
        notes="NADEC."),
    "TADAWUL:4001": _ksa(ticker="TADAWUL:4001", sector="Consumer Staples",
        notes="Abdullah Al Othaim Markets."),

    # ── Consumer Discretionary ──────────────────────────────────────
    "TADAWUL:4190": _ksa(ticker="TADAWUL:4190", sector="Consumer Discretionary",
        notes="Jarir Marketing Co."),
    "TADAWUL:4240": _ksa(ticker="TADAWUL:4240", sector="Consumer Discretionary",
        notes="Fawaz Al Hokair Group."),

    # ── Real Estate ─────────────────────────────────────────────────
    "TADAWUL:4020": _ksa(ticker="TADAWUL:4020", sector="Real Estate",
        notes="Dar Al Arkan Real Estate."),
    "TADAWUL:4090": _ksa(ticker="TADAWUL:4090", sector="Real Estate",
        notes="Taiba Investments."),

    # ── Utilities ───────────────────────────────────────────────────
    "TADAWUL:5110": _ksa(ticker="TADAWUL:5110", sector="Utilities",
        parent="PIF", parent_conf=0.9,
        parent_method="saudi_electricity_disclosure_2022",
        notes="Saudi Electricity — sovereign-anchored regulated utility."),
    "TADAWUL:2082": _ksa(ticker="TADAWUL:2082", sector="Utilities",
        notes="ACWA Power International."),
}
