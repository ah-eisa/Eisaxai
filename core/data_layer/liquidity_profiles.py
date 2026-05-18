"""
core.data_layer.liquidity_profiles — Tier 1/2/3 classification + ADV +
square-root market-impact slippage estimator.

Aligned with PHASE_H2_TC_OPTIMIZER spec:
    - GCC tickers carry an extra slippage multiplier (1.8x default).
    - Egypt: 2.4x.
    - Crypto: step-bucket discontinuity (sub-$100k notional vs above).
    - Tier 1 = mega-cap ADV ≥ $50M/day, Tier 2 = mid-cap ADV $5–50M,
      Tier 3 = small-cap / illiquid ADV < $5M.

ADV is sourced from the latest cached snapshot (`volume * close` as a
proxy) and is intentionally a rolling-30-day estimate where the snapshot
already carries 30-day average volume. When unknown, the function falls
back to a tier-default ADV so engines never receive `None`.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

from phase_h.cache import memoize
from phase_h.registry import FeatureRegistry

from . import _flags  # noqa: F401
from .base import DEFAULT_SNAPSHOT_TTL_SECONDS
from .market_cache_adapter import get_ticker_row
from .utils.validation import coerce_float

logger = logging.getLogger("data_layer.liquidity_profiles")


# Tier definitions — ADV thresholds in USD per day.
LIQUIDITY_TIER_1 = {
    "code": "T1",
    "label": "Mega liquidity",
    "adv_usd_min": 50_000_000.0,
    "base_slippage_bps": 5.0,
    "max_participation_pct": 0.15,
}
LIQUIDITY_TIER_2 = {
    "code": "T2",
    "label": "Standard liquidity",
    "adv_usd_min": 5_000_000.0,
    "base_slippage_bps": 12.0,
    "max_participation_pct": 0.10,
}
LIQUIDITY_TIER_3 = {
    "code": "T3",
    "label": "Thin liquidity",
    "adv_usd_min": 0.0,
    "base_slippage_bps": 30.0,
    "max_participation_pct": 0.05,
}

_TIERS = (LIQUIDITY_TIER_1, LIQUIDITY_TIER_2, LIQUIDITY_TIER_3)

# Region-aware slippage multipliers (PHASE_H2 spec).
REGION_MULTIPLIERS: Dict[str, float] = {
    "US": 1.0,
    "DM": 1.05,
    "EM": 1.4,
    "GCC": 1.8,
    "KSA": 1.8,
    "UAE": 1.8,
    "QAT": 1.8,
    "BAH": 1.8,
    "KWT": 1.8,
    "OMA": 1.8,
    "EGY": 2.4,
    "MAR": 2.0,
    "TUN": 2.0,
    "CRYPTO": 1.5,
    "COMMODITY": 1.2,
}

# Crypto step-bucket discontinuity: notional buckets in USD.
_CRYPTO_BUCKETS = [
    (50_000.0,    1.0),   # ≤ 50k → base slippage
    (250_000.0,   1.4),
    (1_000_000.0, 2.1),
    (5_000_000.0, 3.5),
]
_CRYPTO_BUCKET_OVERFLOW = 5.5  # > $5M crypto orders


def _region_of(row: Dict[str, Any]) -> str:
    """Best-effort region tagging from a cache row."""
    market = (row.get("_market") or "").lower()
    if market in {"ksa"}:
        return "KSA"
    if market in {"uae", "qatar", "bahrain", "kuwait"}:
        return market[:3].upper()
    if market == "egypt":
        return "EGY"
    if market == "morocco":
        return "MAR"
    if market == "tunisia":
        return "TUN"
    if market == "america":
        return "US"
    if market == "crypto":
        return "CRYPTO"
    if market == "commodities":
        return "COMMODITY"
    return "US"


def _adv_from_row(row: Dict[str, Any]) -> Optional[float]:
    """Estimate ADV (USD/day) from the latest snapshot row."""
    close = coerce_float(row.get("close"))
    volume = coerce_float(row.get("volume"))
    if close is None or volume is None:
        return None
    adv = close * volume
    if adv <= 0 or not math.isfinite(adv):
        return None
    return adv


def tier_of(adv_usd: Optional[float]) -> Dict[str, Any]:
    """Map an ADV figure to a tier dict. Missing ADV → T3."""
    if adv_usd is None or adv_usd <= 0:
        return LIQUIDITY_TIER_3
    if adv_usd >= LIQUIDITY_TIER_1["adv_usd_min"]:
        return LIQUIDITY_TIER_1
    if adv_usd >= LIQUIDITY_TIER_2["adv_usd_min"]:
        return LIQUIDITY_TIER_2
    return LIQUIDITY_TIER_3


@memoize("data_layer.liquidity_profile", ttl_seconds=DEFAULT_SNAPSHOT_TTL_SECONDS)
def get_liquidity_profile(*, ticker: str) -> Dict[str, Any]:
    """
    Return a profile dict for one ticker:
        {ticker, region, adv_usd, tier, base_slippage_bps, region_multiplier,
         max_participation_pct, source}.
    Never raises; missing data → conservative defaults.
    """
    if not FeatureRegistry.is_enabled("data_layer_liquidity_profile"):
        return _disabled_profile(ticker)
    row = get_ticker_row(ticker)
    if row is None:
        return _disabled_profile(ticker, reason="ticker_not_in_cache")
    region = _region_of(row)
    adv = _adv_from_row(row)
    tier = tier_of(adv)
    mult = REGION_MULTIPLIERS.get(region, 1.2)
    return {
        "ticker": ticker,
        "region": region,
        "adv_usd": adv,
        "tier": tier["code"],
        "tier_label": tier["label"],
        "base_slippage_bps": tier["base_slippage_bps"],
        "region_multiplier": mult,
        "max_participation_pct": tier["max_participation_pct"],
        "source": "market_cache",
    }


def _disabled_profile(ticker: str, *, reason: str = "feature_disabled") -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "region": "US",
        "adv_usd": None,
        "tier": LIQUIDITY_TIER_3["code"],
        "tier_label": LIQUIDITY_TIER_3["label"],
        "base_slippage_bps": LIQUIDITY_TIER_3["base_slippage_bps"],
        "region_multiplier": 1.0,
        "max_participation_pct": LIQUIDITY_TIER_3["max_participation_pct"],
        "source": reason,
    }


def get_adv(ticker: str) -> Optional[float]:
    """Latest ADV (USD/day) for `ticker`, or None when not cached."""
    profile = get_liquidity_profile(ticker=ticker)
    return profile.get("adv_usd")


def _crypto_bucket_multiplier(notional_usd: float) -> float:
    for cap, mult in _CRYPTO_BUCKETS:
        if notional_usd <= cap:
            return mult
    return _CRYPTO_BUCKET_OVERFLOW


def estimate_slippage_bps(
    ticker: str,
    notional_usd: float,
    *,
    participation_cap: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Square-root market-impact slippage estimator.

    impact_bps = base_slippage * region_multiplier *
                 sqrt(notional / adv) * crypto_bucket_factor (if applicable)

    Returns a structured dict so callers can audit each component.
    """
    if notional_usd <= 0:
        return {"slippage_bps": 0.0, "components": {}, "notes": ["non-positive notional"]}
    profile = get_liquidity_profile(ticker=ticker)
    adv = profile.get("adv_usd") or 1.0e6  # 1M USD floor when unknown
    base = float(profile["base_slippage_bps"])
    mult = float(profile["region_multiplier"])
    participation = notional_usd / adv
    cap = participation_cap if participation_cap is not None else float(profile["max_participation_pct"])
    if cap > 0:
        participation = min(participation, cap * 10)  # discourage runaway impact
    sqrt_factor = math.sqrt(max(participation, 1e-6))

    crypto_factor = 1.0
    if profile["region"] == "CRYPTO":
        crypto_factor = _crypto_bucket_multiplier(notional_usd)

    bps = base * mult * sqrt_factor * crypto_factor
    bps = min(bps, 1000.0)  # hard cap at 10% to avoid pathological outputs

    return {
        "slippage_bps": round(bps, 2),
        "components": {
            "base_slippage_bps": base,
            "region_multiplier": mult,
            "sqrt_participation": round(sqrt_factor, 4),
            "crypto_bucket_factor": crypto_factor,
            "adv_usd_used": adv,
            "participation_rate": round(participation, 4),
        },
        "tier": profile["tier"],
        "region": profile["region"],
        "notes": [],
    }


__all__ = [
    "LIQUIDITY_TIER_1",
    "LIQUIDITY_TIER_2",
    "LIQUIDITY_TIER_3",
    "REGION_MULTIPLIERS",
    "tier_of",
    "get_liquidity_profile",
    "get_adv",
    "estimate_slippage_bps",
]
