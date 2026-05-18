"""
core.data_layer.trading_calendars — GCC Sun–Thu + US Mon–Fri + holidays.

A deliberately small calendar module. Heavy financial-grade calendars
(half-days, partial sessions, regional Eid-al-Fitr ranges) belong in a
dedicated dependency; here we model only the practical primitives that
H2 (TC optimizer) and H3 (forward sim) need to align horizons:

    - weekly cycle (Sun–Thu vs Mon–Fri)
    - region-tagged static holidays (Eid, National Day, US federals)
    - next / previous trading day stepping
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Optional, Set, Tuple

from phase_h.registry import FeatureRegistry

from . import _flags  # noqa: F401

# Weekly cycle: 0 = Monday, 6 = Sunday (Python's weekday() convention).
_GCC_TRADING_WEEKDAYS = {6, 0, 1, 2, 3}   # Sun, Mon, Tue, Wed, Thu
_US_TRADING_WEEKDAYS = {0, 1, 2, 3, 4}    # Mon–Fri

_CALENDARS: Dict[str, Dict[str, object]] = {
    "GCC": {"weekdays": _GCC_TRADING_WEEKDAYS, "note": "Sun-Thu session"},
    "KSA": {"weekdays": _GCC_TRADING_WEEKDAYS, "note": "Tadawul Sun-Thu"},
    "UAE": {"weekdays": _GCC_TRADING_WEEKDAYS, "note": "ADX/DFM Mon-Fri (2022+)" },
    "QAT": {"weekdays": _GCC_TRADING_WEEKDAYS, "note": "QSE Sun-Thu"},
    "EGY": {"weekdays": _GCC_TRADING_WEEKDAYS, "note": "EGX Sun-Thu"},
    "MAR": {"weekdays": _US_TRADING_WEEKDAYS,  "note": "CSE Mon-Fri"},
    "TUN": {"weekdays": _US_TRADING_WEEKDAYS,  "note": "BVMT Mon-Fri"},
    "US":  {"weekdays": _US_TRADING_WEEKDAYS,  "note": "NYSE Mon-Fri"},
}

# UAE switched to Mon-Fri in 2022 — encode that exception explicitly.
_CALENDARS["UAE"] = {"weekdays": _US_TRADING_WEEKDAYS, "note": "ADX/DFM Mon-Fri (post-2022)"}

# Region → static observed holidays (ISO YYYY-MM-DD). Intentionally minimal.
_STATIC_HOLIDAYS: Dict[str, Set[str]] = {
    "GCC": {"2026-01-01", "2026-04-15", "2026-04-16", "2026-09-23", "2026-12-02"},
    "KSA": {"2026-02-22", "2026-09-23"},
    "UAE": {"2026-12-02", "2026-12-03"},
    "EGY": {"2026-01-07", "2026-04-25", "2026-07-23", "2026-10-06"},
    "US":  {"2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25",
            "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"},
    "QAT": {"2026-12-18"},
}


def market_calendar(region: str) -> Dict[str, object]:
    """Return the calendar definition for a region; defaults to US."""
    if not FeatureRegistry.is_enabled("data_layer_calendars"):
        return {"weekdays": _US_TRADING_WEEKDAYS, "note": "calendars disabled"}
    key = (region or "").upper()
    if key in _CALENDARS:
        return _CALENDARS[key]
    if key in {"BAH", "BAHRAIN"}:
        return _CALENDARS["GCC"]
    if key in {"OMA", "OMAN"}:
        return _CALENDARS["GCC"]
    if key in {"KWT", "KUWAIT"}:
        return _CALENDARS["GCC"]
    return _CALENDARS["US"]


def _holiday_set(region: str) -> Set[str]:
    return _STATIC_HOLIDAYS.get(region.upper(), set())


def is_trading_day(date: _dt.date, region: str = "US") -> bool:
    """True iff `date` is a trading day for `region`."""
    cal = market_calendar(region)
    weekdays: Set[int] = cal["weekdays"]  # type: ignore[assignment]
    if date.weekday() not in weekdays:
        return False
    if date.isoformat() in _holiday_set(region):
        return False
    return True


def next_trading_day(date: _dt.date, region: str = "US") -> _dt.date:
    d = date + _dt.timedelta(days=1)
    for _ in range(60):  # safety bound
        if is_trading_day(d, region):
            return d
        d += _dt.timedelta(days=1)
    return d


def previous_trading_day(date: _dt.date, region: str = "US") -> _dt.date:
    d = date - _dt.timedelta(days=1)
    for _ in range(60):
        if is_trading_day(d, region):
            return d
        d -= _dt.timedelta(days=1)
    return d


def trading_days_between(start: _dt.date, end: _dt.date, region: str = "US") -> int:
    """Inclusive trading-day count between two dates."""
    if end < start:
        return 0
    n = 0
    d = start
    while d <= end:
        if is_trading_day(d, region):
            n += 1
        d += _dt.timedelta(days=1)
    return n


__all__ = [
    "market_calendar",
    "is_trading_day",
    "next_trading_day",
    "previous_trading_day",
    "trading_days_between",
]
