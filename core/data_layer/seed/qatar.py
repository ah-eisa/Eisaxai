"""
core.data_layer.seed.qatar — curated Qatar Stock Exchange (QSE) metadata seed.

Coverage target: 5 names (priority: Financials, Materials, Telecom).
Tier-1 fields populated from QSE listings.
Reviewer log: seed prepared 2026-05-18.
"""

from __future__ import annotations

from typing import Any, Dict

from ..gcc_metadata import (
    _entry, _exc, _iss, _missing,
    shariah_field_for, sovereign_parent_field_for, strategic_asset_field_for,
)


def _qse(*, ticker: str, sector: str, parent: str = "", parent_conf: float = 0.0,
         parent_method: str = "", notes: str = "") -> Dict[str, Any]:
    parent_field = sovereign_parent_field_for(ticker)
    if parent_field.value is None and parent and parent_conf > 0:
        parent_field = _iss(parent, confidence=parent_conf, methodology=parent_method)
    return _entry(
        ticker=ticker,
        country=_exc("Qatar", confidence=1.0, methodology="qse_listing"),
        exchange=_exc("QSE", confidence=1.0, methodology="qse_listing"),
        sector=_exc(sector, confidence=0.95, methodology="qse_sector_assignment"),
        parent_company=parent_field,
        strategic_asset_flag=strategic_asset_field_for(ticker),
        shariah_compliant_flag=shariah_field_for(ticker),
        notes=notes,
    )


ENTRIES: Dict[str, Dict[str, Any]] = {
    "QSE:QNBK":  _qse(ticker="QSE:QNBK",  sector="Financials",
        parent="Qatar Investment Authority", parent_conf=0.95,
        parent_method="qnb_annual_report_2023",
        notes="QNB Group — largest MENA bank by assets."),
    "QSE:QIBK":  _qse(ticker="QSE:QIBK",  sector="Financials",
        notes="Qatar Islamic Bank."),
    "QSE:IQCD":  _qse(ticker="QSE:IQCD",  sector="Materials",
        parent="QatarEnergy", parent_conf=0.9,
        parent_method="iqcd_annual_report_2022",
        notes="Industries Qatar — petrochemical + steel + fertiliser."),
    "QSE:ORDS":  _qse(ticker="QSE:ORDS",  sector="Communication Services",
        notes="Ooredoo Group — pan-MENA telecom operator."),
    "QSE:QFLS":  _qse(ticker="QSE:QFLS",  sector="Consumer Staples",
        notes="Qatar Fuel (Woqod)."),
}
