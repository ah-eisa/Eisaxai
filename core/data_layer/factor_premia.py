"""
core.data_layer.factor_premia — read-only access to FF3 / FF5 / Carhart panels.

Wraps the existing /home/ubuntu/investwise/market_cache/factor_panels/*.csv
files. phase_h.factor_model owns the regression logic; this module only
loads + caches the underlying panels.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from phase_h.cache import memoize
from phase_h.registry import FeatureRegistry

from . import _flags  # noqa: F401
from .base import DEFAULT_SNAPSHOT_TTL_SECONDS
from .market_cache_adapter import FACTOR_PANEL_DIR  # constants live in one place

logger = logging.getLogger("data_layer.factor_premia")

# Canonical filename mapping.
_PANELS = {
    "ff3":     "ff3_monthly.csv",
    "ff5":     "ff5_monthly.csv",
    "carhart": "carhart_mom_monthly.csv",
}


def list_factor_models() -> List[str]:
    return list(_PANELS.keys())


@memoize("data_layer.factor_panel", ttl_seconds=DEFAULT_SNAPSHOT_TTL_SECONDS)
def get_factor_panel(*, model: str) -> Optional["pd.DataFrame"]:
    """
    Return the factor panel for `model`. Memoised by phase_h.cache.

    Models:
        - "ff3"     → MKT, SMB, HML, RF
        - "ff5"     → MKT, SMB, HML, RMW, CMA, RF
        - "carhart" → momentum factor (MOM)
    """
    if not FeatureRegistry.is_enabled("data_layer_factor_panels") or pd is None:
        return None
    fname = _PANELS.get(model.lower())
    if fname is None:
        return None
    path = os.path.join(FACTOR_PANEL_DIR, fname)
    if not os.path.exists(path):
        logger.warning("data_layer: factor panel %s not present at %s", model, path)
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception as exc:  # pragma: no cover
        logger.warning("data_layer: failed to read factor panel %s (%s)", path, exc)
        return None
    return df


def panel_metadata(model: str) -> Dict[str, Any]:
    """Lightweight panel metadata for the audit appendix."""
    df = get_factor_panel(model=model)
    if df is None:
        return {"model": model, "rows": 0, "columns": [], "first_date": None, "last_date": None}
    return {
        "model": model,
        "rows": int(len(df)),
        "columns": [c for c in df.columns if c != "date"],
        "first_date": str(df["date"].iloc[0].date()) if len(df) and "date" in df.columns else None,
        "last_date":  str(df["date"].iloc[-1].date()) if len(df) and "date" in df.columns else None,
    }


# ──────────────────────────────────────────────────────────────────────
# Writer helpers — used by engines that refresh Ken French panels.
# Engines must NOT construct cache paths themselves; they call these.
# ──────────────────────────────────────────────────────────────────────

def factor_panel_path(filename: str) -> str:
    """Absolute path for a factor panel file inside the canonical cache."""
    os.makedirs(FACTOR_PANEL_DIR, exist_ok=True)
    return os.path.join(FACTOR_PANEL_DIR, filename)


def factor_panel_age_seconds(filename: str) -> Optional[float]:
    """Seconds since the file was last modified, or None when absent."""
    import time as _t
    path = factor_panel_path(filename)
    if not os.path.exists(path):
        return None
    try:
        return _t.time() - os.path.getmtime(path)
    except OSError:
        return None


def write_factor_panel(filename: str, df: "pd.DataFrame") -> None:
    """Persist a refreshed factor panel into the canonical cache."""
    if pd is None or df is None or df.empty:
        return
    path = factor_panel_path(filename)
    df.to_csv(path)


__all__ = [
    "list_factor_models",
    "get_factor_panel",
    "panel_metadata",
    "factor_panel_path",
    "factor_panel_age_seconds",
    "write_factor_panel",
]
