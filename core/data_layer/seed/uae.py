"""
core.data_layer.seed.uae — curated ADX + DFM (UAE) metadata seed.

Coverage target: 25 names (mix of ADX + DFM).
Tier-1 fields populated (country / exchange / sector) from ADX / DFM listings.
Parent companies populated from public IPO prospectuses where on record.
All quantitative fields remain `_missing()` until reviewer sign-off.

Reviewer log: seed prepared 2026-05-18.
"""

from __future__ import annotations

from typing import Any, Dict

from ..gcc_metadata import (
    _entry, _exc, _iss, _missing,
    shariah_field_for, sovereign_parent_field_for, strategic_asset_field_for,
)


def _uae(*, ticker: str, exchange: str, sector: str,
         parent: str = "", parent_conf: float = 0.0, parent_method: str = "",
         notes: str = "") -> Dict[str, Any]:
    parent_field = sovereign_parent_field_for(ticker)
    if parent_field.value is None and parent and parent_conf > 0:
        parent_field = _iss(parent, confidence=parent_conf, methodology=parent_method)
    return _entry(
        ticker=ticker,
        country=_exc("UAE", confidence=1.0, methodology=f"{exchange.lower()}_listing"),
        exchange=_exc(exchange, confidence=1.0, methodology=f"{exchange.lower()}_listing"),
        sector=_exc(sector, confidence=0.95, methodology=f"{exchange.lower()}_sector_assignment"),
        parent_company=parent_field,
        strategic_asset_flag=strategic_asset_field_for(ticker),
        shariah_compliant_flag=shariah_field_for(ticker),
        notes=notes,
    )


ENTRIES: Dict[str, Dict[str, Any]] = {

    # ── ADX — Financials ────────────────────────────────────────────
    "ADX:FAB":   _uae(ticker="ADX:FAB",   exchange="ADX", sector="Financials",
        parent="Mubadala / ADQ", parent_conf=0.9,
        parent_method="fab_annual_report_2023",
        notes="First Abu Dhabi Bank — largest UAE lender."),
    "ADX:ADIB":  _uae(ticker="ADX:ADIB",  exchange="ADX", sector="Financials",
        notes="Abu Dhabi Islamic Bank."),
    "ADX:ADCB":  _uae(ticker="ADX:ADCB",  exchange="ADX", sector="Financials",
        notes="Abu Dhabi Commercial Bank."),

    # ── ADX — Energy / Utilities ────────────────────────────────────
    "ADX:ADNOCGAS": _uae(ticker="ADX:ADNOCGAS", exchange="ADX", sector="Energy",
        parent="ADNOC Group", parent_conf=0.95,
        parent_method="adnoc_gas_ipo_prospectus_2023",
        notes="ADNOC Gas — long-term LNG/feedstock off-take."),
    "ADX:ADNOCDIST": _uae(ticker="ADX:ADNOCDIST", exchange="ADX", sector="Energy",
        parent="ADNOC Group", parent_conf=0.95,
        parent_method="adnoc_distribution_ipo_prospectus_2017",
        notes="ADNOC Distribution — retail fuel concessionaire."),
    "ADX:ADNOCDRILL": _uae(ticker="ADX:ADNOCDRILL", exchange="ADX", sector="Energy",
        parent="ADNOC Group", parent_conf=0.95,
        parent_method="adnoc_drilling_ipo_prospectus_2021",
        notes="ADNOC Drilling."),
    "ADX:TAQA":  _uae(ticker="ADX:TAQA",  exchange="ADX", sector="Utilities",
        notes="Abu Dhabi National Energy Company."),
    "ADX:PUREHEALTH": _uae(ticker="ADX:PUREHEALTH", exchange="ADX", sector="Health Care",
        notes="PureHealth — regional integrated health platform."),

    # ── ADX — Diversified / Industrials ────────────────────────────
    "ADX:IHC":   _uae(ticker="ADX:IHC",   exchange="ADX", sector="Industrials",
        notes="International Holding Company — Abu Dhabi sovereign-linked diversified holding (classified Industrials under GICS-11; conglomerate-of-conglomerates business model)."),
    "ADX:MULTIPLY": _uae(ticker="ADX:MULTIPLY", exchange="ADX", sector="Industrials",
        notes="Multiply Group (conglomerate; classified Industrials under GICS-11)."),
    "ADX:ALPHADHABI": _uae(ticker="ADX:ALPHADHABI", exchange="ADX", sector="Industrials",
        notes="Alpha Dhabi Holding (conglomerate; classified Industrials under GICS-11)."),
    "ADX:Q":     _uae(ticker="ADX:Q",     exchange="ADX", sector="Industrials",
        notes="Q Holding (industrial conglomerate)."),

    # ── ADX — Real Estate ───────────────────────────────────────────
    "ADX:ALDAR": _uae(ticker="ADX:ALDAR", exchange="ADX", sector="Real Estate",
        notes="Aldar Properties — Abu Dhabi RE developer."),

    # ── ADX — Telecom ───────────────────────────────────────────────
    "ADX:EAND":  _uae(ticker="ADX:EAND",  exchange="ADX", sector="Communication Services",
        notes="e& (formerly Etisalat) — telecom incumbent."),

    # ── DFM — Real Estate / Construction ────────────────────────────
    "DFM:EMAAR":  _uae(ticker="DFM:EMAAR",  exchange="DFM", sector="Real Estate",
        notes="Emaar Properties — Dubai RE bellwether."),
    "DFM:EMAARDEV": _uae(ticker="DFM:EMAARDEV", exchange="DFM", sector="Real Estate",
        notes="Emaar Development."),
    "DFM:DAMAC":  _uae(ticker="DFM:DAMAC",  exchange="DFM", sector="Real Estate",
        notes="DAMAC Properties (delisted 2022 — listed for historical reference)."),
    "DFM:UPP":    _uae(ticker="DFM:UPP",    exchange="DFM", sector="Real Estate",
        notes="Union Properties."),

    # ── DFM — Financials ────────────────────────────────────────────
    "DFM:EMIRATESNBD": _uae(ticker="DFM:EMIRATESNBD", exchange="DFM", sector="Financials",
        notes="Emirates NBD — Dubai's largest bank."),
    "DFM:DIB":    _uae(ticker="DFM:DIB",    exchange="DFM", sector="Financials",
        notes="Dubai Islamic Bank."),
    "DFM:MASHREQ": _uae(ticker="DFM:MASHREQ", exchange="DFM", sector="Financials",
        notes="Mashreqbank."),
    "DFM:CBD":    _uae(ticker="DFM:CBD",    exchange="DFM", sector="Financials",
        notes="Commercial Bank of Dubai."),

    # ── DFM — Logistics / Transport ─────────────────────────────────
    "DFM:DUBAIINV": _uae(ticker="DFM:DUBAIINV", exchange="DFM", sector="Industrials",
        notes="Dubai Investments (conglomerate; classified Industrials under GICS-11)."),
    "DFM:SALIK":  _uae(ticker="DFM:SALIK",  exchange="DFM", sector="Industrials",
        parent="Roads & Transport Authority (RTA), Dubai", parent_conf=0.9,
        parent_method="salik_ipo_prospectus_2022",
        notes="Salik — Dubai's road toll concession."),
    "DFM:DEWA":   _uae(ticker="DFM:DEWA",   exchange="DFM", sector="Utilities",
        parent="Government of Dubai (Investment Corporation of Dubai)",
        parent_conf=0.9, parent_method="dewa_ipo_prospectus_2022",
        notes="DEWA — Dubai Electricity & Water Authority, sovereign-controlled utility."),
}
