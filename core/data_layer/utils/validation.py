"""
Validation primitives for the data layer.

Heavy numerical checks live in phase_h.numerics; this module only handles
DataFrame-shape and per-record sanity that the layer needs locally.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover — pandas always present in venv
    pd = None  # type: ignore


def require_columns(df: "pd.DataFrame", columns: Iterable[str], *, context: str = "") -> None:
    """Raise ValueError if any required column is missing."""
    if pd is None or df is None:
        raise ValueError(f"{context}: dataframe unavailable")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing columns {missing}")


def require_non_empty(df: "pd.DataFrame", *, context: str = "") -> None:
    """Raise ValueError if df is None or empty."""
    if pd is None or df is None or len(df) == 0:
        raise ValueError(f"{context}: dataframe empty or unavailable")


def coerce_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    """Best-effort float coercion that swallows None / NaN / 'None'."""
    if v is None:
        return default
    try:
        out = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def coerce_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    """Best-effort int coercion."""
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default


def is_finite(v: Any) -> bool:
    f = coerce_float(v)
    return f is not None


__all__ = [
    "require_columns",
    "require_non_empty",
    "coerce_float",
    "coerce_int",
    "is_finite",
]
