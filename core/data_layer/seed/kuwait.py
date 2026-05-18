"""
core.data_layer.seed.kuwait — curated Boursa Kuwait metadata seed.

Coverage target: 5 names (priority: Financials, Telecom, Materials).
Tier-1 fields populated from Boursa Kuwait listings.
Reviewer log: seed prepared 2026-05-18.
"""

from __future__ import annotations

from typing import Any, Dict

from ..gcc_metadata import (
    _entry, _exc, _missing,
    shariah_field_for, sovereign_parent_field_for, strategic_asset_field_for,
)


def _kse(*, ticker: str, sector: str, notes: str = "") -> Dict[str, Any]:
    return _entry(
        ticker=ticker,
        country=_exc("Kuwait", confidence=1.0, methodology="boursa_kuwait_listing"),
        exchange=_exc("KSE", confidence=1.0, methodology="boursa_kuwait_listing"),
        sector=_exc(sector, confidence=0.95, methodology="boursa_kuwait_sector_assignment"),
        parent_company=sovereign_parent_field_for(ticker),
        strategic_asset_flag=strategic_asset_field_for(ticker),
        shariah_compliant_flag=shariah_field_for(ticker),
        notes=notes,
    )


ENTRIES: Dict[str, Dict[str, Any]] = {
    "KSE:NBK":   _kse(ticker="KSE:NBK",   sector="Financials",
        notes="National Bank of Kuwait — largest commercial bank."),
    "KSE:KFH":   _kse(ticker="KSE:KFH",   sector="Financials",
        notes="Kuwait Finance House — largest Islamic bank in Kuwait."),
    "KSE:ZAIN":  _kse(ticker="KSE:ZAIN",  sector="Communication Services",
        notes="Zain Group — pan-MENA telecom operator."),
    "KSE:AGLTY": _kse(ticker="KSE:AGLTY", sector="Industrials",
        notes="Agility Public Warehousing (now relisted)."),
    "KSE:MEZZAN": _kse(ticker="KSE:MEZZAN", sector="Consumer Staples",
        notes="Mezzan Holding — regional food + pharma distributor."),
}
