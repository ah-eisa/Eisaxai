"""
global_allocator.py — EisaX Global Allocation Engine
Cross-market QP optimization: US + GCC + Egypt + Crypto + Gold + Bonds + Commodities
Returns optimal multi-asset, multi-geography portfolio allocation.
"""
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("global_allocator")

# ── Phase H — orchestrator (additive, flag-gated; import is best-effort)
try:
    from phase_h.orchestrator import augment_result as _phase_h_augment
    _PHASE_H_AVAILABLE = True
except Exception as _ph_exc:  # pragma: no cover — never break allocate() on import
    _phase_h_augment = None
    _PHASE_H_AVAILABLE = False
    logger.warning("phase_h orchestrator unavailable: %r", _ph_exc)

# ── Institutional Data Layer (sole sanctioned read path for market_cache) ─────
from core.data_layer import market_cache_adapter as _mca

# ── Sector → Long-run Annual Return Assumptions ────────────────────────────────
# Based on 10Y+ historical data. NOT derived from recent short-term performance.
_SECTOR_MU = {
    "Finance":              0.090,
    "Energy Minerals":      0.095,
    "Real Estate":          0.085,
    "Process Industries":   0.090,
    "Electronic Technology":0.115,
    "Technology Services":  0.110,
    "Utilities":            0.072,
    "Consumer Non-Durables":0.085,
    "Consumer Durables":    0.088,
    "Commercial Services":  0.090,
    "Health Technology":    0.100,
    "Retail Trade":         0.090,
    "Communications":       0.088,
    "Transportation":       0.085,
    "Producer Manufacturing":0.088,
}
_SECTOR_MU_DEFAULT = 0.090

# ── Market → Base Annual Volatility ───────────────────────────────────────────
_MARKET_VOL = {
    "ksa":        0.175,
    "uae":        0.195,
    "egypt":      0.275,
    "america":    0.220,
    "kuwait":     0.185,
    "qatar":      0.185,
    "commodities":0.250,
}

# ── Market → Beta vs MSCI World ───────────────────────────────────────────────
_MARKET_BETA = {
    "ksa": 0.50, "uae": 0.45, "egypt": 0.35,
    "america": 1.00, "kuwait": 0.42, "qatar": 0.43,
}


def _load_latest_snapshot(market_code: str):
    """Latest snapshot for a market — routed through core.data_layer."""
    try:
        return _mca.get_latest_snapshot(market_code)
    except Exception as e:
        # The Phase H committee regression case reloads phase_h.registry,
        # which wipes data_layer_* flags. Re-register defensively and retry.
        try:
            from core.data_layer import _flags as _dl_flags
            _dl_flags.register()
            return _mca.get_latest_snapshot(market_code)
        except Exception as e2:
            logger.warning(f"[Allocator] Cache load failed for {market_code}: {e2}")
            return None


def _estimate_mu(row, market_code: str) -> float:
    """
    Estimate annualized expected return from FUNDAMENTALS — not recent price history.
    This prevents negative-return portfolios caused by short-term drawdowns.
    """
    sector   = str(row.get("sector") or "")
    base_mu  = _SECTOR_MU.get(sector, _SECTOR_MU_DEFAULT)

    # Technical trend adjustment (±2%)
    close  = float(row.get("close")  or 0)
    sma50  = float(row.get("SMA50")  or 0)
    sma200 = float(row.get("SMA200") or 0)
    if close > 0 and sma50 > 0 and sma200 > 0:
        if close > sma50 > sma200:
            base_mu += 0.020   # confirmed uptrend
        elif close < sma50 < sma200:
            base_mu -= 0.015   # confirmed downtrend

    # Dividend yield adds to total return
    div = float(row.get("dividend_yield_recent") or 0)
    if 0 < div < 20:           # TradingView returns % (e.g. 5.0 = 5%)
        base_mu += min(div / 100.0, 0.08)

    # Valuation adjustment
    pe = float(row.get("price_earnings_ttm") or 0)
    if 5 < pe < 12:
        base_mu += 0.010       # value premium
    elif pe > 40:
        base_mu -= 0.010       # growth premium already priced in

    return round(max(0.04, min(base_mu, 0.28)), 4)


def _estimate_vol(row, market_code: str) -> float:
    """Estimate annualized volatility from market baseline + RSI."""
    base_vol = _MARKET_VOL.get(market_code, 0.200)
    rsi = row.get("RSI")
    if rsi is not None:
        try:
            if float(rsi) < 30 or float(rsi) > 70:
                base_vol *= 1.15   # extremes = more volatile
        except (ValueError, TypeError):
            pass
    return round(min(base_vol, 0.50), 4)


def _select_top_stocks(market_code: str, region_tag: str, n: int = 4) -> list:
    """
    Load live market snapshot and select top N stocks by composite quality score.
    Returns a list of AssetClass objects with fundamentally-derived return/vol estimates.
    """
    df = _load_latest_snapshot(market_code)
    if df is None or df.empty:
        return []

    try:
        import pandas as pd

        def _score(row) -> float:
            s = 0.0
            # RSI health: prefer 40-65 (momentum without being overbought)
            rsi = row.get("RSI")
            if rsi is not None:
                try:
                    r = float(rsi)
                    s += 1.0 if 40 <= r <= 65 else 0.4 if 30 <= r <= 75 else 0.0
                except (ValueError, TypeError):
                    pass
            # Positive daily momentum
            chg = float(row.get("change") or 0)
            s += 0.5 if chg > 0 else 0.0
            # MACD bullish crossover
            macd = row.get("MACD.macd")
            sig  = row.get("MACD.signal")
            if macd is not None and sig is not None:
                try:
                    s += 0.5 if float(macd) > float(sig) else 0.0
                except (ValueError, TypeError):
                    pass
            # Uptrend (price > SMA50 > SMA200)
            cl   = float(row.get("close")  or 0)
            s50  = float(row.get("SMA50")  or 0)
            s200 = float(row.get("SMA200") or 0)
            if cl > 0 and s50 > 0 and s200 > 0:
                s += 1.0 if cl > s50 > s200 else 0.0
            # Reasonable valuation
            pe = float(row.get("price_earnings_ttm") or 0)
            s += 1.0 if 5 < pe < 20 else 0.5 if pe == 0 else 0.0
            # Large market cap (prefer liquid stocks)
            mc = float(row.get("market_cap_basic") or 0)
            s += 1.0 if mc > 5e9 else 0.5 if mc > 1e9 else 0.0
            return s

        df = df.copy()
        df["_score"] = df.apply(_score, axis=1)
        # Filter: must have a price and positive market cap
        df = df[(df["close"] > 0) & (df["market_cap_basic"].fillna(0) > 0)]
        top = df.nlargest(n, "_score")

        results = []
        for _, row in top.iterrows():
            ticker = str(row.get("ticker") or row.get("name") or "")
            name   = str(row.get("name")   or ticker)
            mu     = _estimate_mu(row, market_code)
            vol    = _estimate_vol(row, market_code)
            beta   = _MARKET_BETA.get(market_code, 0.50)
            results.append(AssetClass(
                name        = f"{name} ({ticker.split(':')[-1]})",
                region      = region_tag,
                proxy       = ticker.split(":")[-1] if ":" in ticker else ticker,
                mu_annual   = mu,
                vol_annual  = vol,
                beta_world  = beta,
                min_w       = 0.0,
                max_w       = 0.085,       # RULE: single position max 8.5%
                currency    = {"ksa":"SAR","uae":"AED","egypt":"EGP"}.get(market_code,"USD"),
                description = str(row.get("sector") or ""),
            ))
        return results
    except Exception as e:
        logger.warning(f"[Allocator] Stock selection failed for {market_code}: {e}")
        return []

# ── Asset Universe ─────────────────────────────────────────────────────────────
# Each asset class has: representative ETF/proxy, expected return, volatility,
# beta vs MSCI World, and region/type tags.
# Returns and vols are long-run assumptions (can be overridden with live data).

@dataclass
class AssetClass:
    name:        str
    region:      str           # "US" | "GCC" | "Egypt" | "Crypto" | "Gold" | "Bonds" | "Cash"
    proxy:       str           # ticker / symbol for live data fetching
    mu_annual:   float         # expected annual return (decimal)
    vol_annual:  float         # expected annual volatility (decimal)
    beta_world:  float         # beta vs MSCI World
    min_w:       float = 0.0   # minimum weight
    max_w:       float = 0.40  # maximum weight per asset class
    currency:    str   = "USD"
    description: str   = ""

# ── Strategic Asset Universe ───────────────────────────────────────────────────
_UNIVERSE: list[AssetClass] = [
    # US Equities
    AssetClass("US Large Cap Tech",   "US",     "QQQ",   0.155, 0.220, 1.25, 0.0, 0.25, "USD", "NASDAQ-100 Tech"),
    AssetClass("US S&P 500 Broad",    "US",     "SPY",   0.105, 0.155, 1.00, 0.0, 0.35, "USD", "S&P 500 Core"),
    AssetClass("US Mid-Cap Equity",   "US",     "MDY",   0.115, 0.170, 1.10, 0.0, 0.15, "USD", "S&P MidCap 400"),
    AssetClass("US Dividend/Value",   "US",     "VIG",   0.090, 0.130, 0.80, 0.0, 0.25, "USD", "Dividend Growth"),
    # GCC / Arab Markets
    AssetClass("Saudi Equities ETF",  "GCC",    "KSA",   0.095, 0.185, 0.55, 0.0, 0.25, "USD", "iShares MSCI Saudi"),
    AssetClass("UAE Real Estate",     "GCC",    "EMAAR.DU", 0.085, 0.200, 0.50, 0.0, 0.20, "AED", "Emaar / Dubai RE"),
    AssetClass("GCC Banks/Financials","GCC",    "QETF.QA", 0.085, 0.170, 0.45, 0.0, 0.20, "USD", "Qatar/GCC Financials"),
    # Egypt
    AssetClass("Egypt Equities",      "Egypt",  "EFID.CA", 0.100, 0.280, 0.35, 0.0, 0.20, "EGP", "Egyptian large cap"),
    # Crypto
    AssetClass("Bitcoin",             "Crypto", "BTC-USD", 0.300, 0.750, 0.60, 0.0, 0.10, "USD", "Opportunistic satellite — high-volatility, non-hedge exposure"),
    AssetClass("Ethereum",            "Crypto", "ETH-USD", 0.250, 0.900, 0.65, 0.0, 0.07, "USD", "Opportunistic satellite — high-volatility, non-hedge exposure"),
    # Gold
    AssetClass("Gold",                "Gold",   "GLD",   0.075, 0.145, -0.05, 0.0, 0.30, "USD", "Inflation hedge"),
    # Bonds
    AssetClass("US Treasuries (LT)",  "Bonds",  "TLT",   0.045, 0.155, -0.30, 0.0, 0.40, "USD", "Long-duration UST"),
    AssetClass("EM Bonds",            "Bonds",  "EMB",   0.065, 0.120, 0.20, 0.0, 0.25, "USD", "Emerging Market Bonds"),
    # Cash
    AssetClass("Cash / T-Bills",      "Cash",        "BIL",   0.045, 0.005, 0.00, 0.0, 1.00, "USD", "3-Month US T-Bill"),
    # Diversification Sleeve (low-correlation, 5-7% allocation target)
    AssetClass("US Healthcare",       "Diversification", "XLV",   0.095, 0.140, 0.70, 0.0, 0.07, "USD", "Defensive health — low market correlation"),
    AssetClass("US Utilities",        "Diversification", "XLU",   0.075, 0.130, 0.50, 0.0, 0.07, "USD", "Yield-defensive, rate-sensitive"),
    AssetClass("Short-Duration Bonds","Diversification", "SHY",   0.048, 0.030, 0.05, 0.0, 0.07, "USD", "Capital preservation + low duration risk"),
    # Commodities
    AssetClass("Crude Oil",           "Commodities", "USO",   0.080, 0.330, 0.25, 0.0, 0.15, "USD", "WTI Oil price exposure"),
    AssetClass("Silver",              "Commodities", "SLV",   0.075, 0.270, -0.02,0.0, 0.12, "USD", "Industrial + precious metal"),
    AssetClass("Copper",              "Commodities", "CPER",  0.085, 0.280, 0.30, 0.0, 0.10, "USD", "Global growth indicator"),
]

# Long-run correlation matrix (estimated from 10Y data).
# Order matches _UNIVERSE above exactly (20 assets).
#
# Block layout:
#   [ 0..16 ]  17 original assets (US equity 4 / GCC 3 / Egypt 1 / Crypto 2 /
#              Gold / Bonds (TLT, EMB) / Cash / Commodities (OIL, SLV, COPR))
#   [ 17..19 ] Diversification block: XLV (Healthcare), XLU (Utilities),
#              SHY (Short-Duration UST).
#
# Diversification calibration (institutional, 10Y panels):
#   XLV: defensive-equity. High-beta to broad US equity (~0.75 SPY) but lower
#        rate-sensitivity. Modest correlation to GCC, near-zero to crypto.
#   XLU: dividend-defensive. Material correlation to long-duration bonds
#        (rate-sensitive utility yields), strong to dividend equity (VIG).
#   SHY: near-cash. Very low correlation to risk assets, partial co-movement
#        with TLT (same direction, shorter duration).
_CORR = np.array([
    # QQQ   SPY   MDY   VIG   KSA   UAE   GCC   EGY   BTC   ETH   GLD   TLT   EMB  Cash   OIL   SLV  COPR   XLV   XLU   SHY
    [ 1.00, 0.92, 0.85, 0.75, 0.35, 0.30, 0.30, 0.25, 0.35, 0.33, 0.02,-0.25, 0.30, 0.00, 0.10, 0.05, 0.30,  0.65, 0.40,-0.05],  # QQQ
    [ 0.92, 1.00, 0.90, 0.85, 0.38, 0.32, 0.32, 0.27, 0.30, 0.28, 0.05,-0.28, 0.35, 0.00, 0.12, 0.06, 0.32,  0.75, 0.50,-0.05],  # SPY
    [ 0.85, 0.90, 1.00, 0.80, 0.35, 0.30, 0.30, 0.25, 0.32, 0.30, 0.04,-0.26, 0.33, 0.00, 0.10, 0.05, 0.30,  0.65, 0.45,-0.05],  # MDY
    [ 0.75, 0.85, 0.80, 1.00, 0.30, 0.27, 0.28, 0.22, 0.22, 0.20, 0.08,-0.20, 0.30, 0.00, 0.08, 0.05, 0.25,  0.72, 0.55, 0.00],  # VIG
    [ 0.35, 0.38, 0.35, 0.30, 1.00, 0.60, 0.65, 0.35, 0.18, 0.16, 0.10,-0.10, 0.40, 0.00, 0.55, 0.12, 0.35,  0.25, 0.18,-0.02],  # KSA
    [ 0.30, 0.32, 0.30, 0.27, 0.60, 1.00, 0.58, 0.40, 0.15, 0.13, 0.08,-0.08, 0.35, 0.00, 0.40, 0.10, 0.30,  0.22, 0.15,-0.02],  # UAE
    [ 0.30, 0.32, 0.30, 0.28, 0.65, 0.58, 1.00, 0.38, 0.16, 0.14, 0.08,-0.08, 0.40, 0.00, 0.45, 0.10, 0.32,  0.25, 0.18,-0.02],  # GCC
    [ 0.25, 0.27, 0.25, 0.22, 0.35, 0.40, 0.38, 1.00, 0.12, 0.10, 0.05,-0.05, 0.35, 0.00, 0.20, 0.08, 0.18,  0.18, 0.12,-0.02],  # Egypt
    [ 0.35, 0.30, 0.32, 0.22, 0.18, 0.15, 0.16, 0.12, 1.00, 0.82, 0.10,-0.10, 0.15, 0.00, 0.08, 0.10, 0.10,  0.20, 0.10,-0.05],  # BTC
    [ 0.33, 0.28, 0.30, 0.20, 0.16, 0.13, 0.14, 0.10, 0.82, 1.00, 0.08,-0.08, 0.13, 0.00, 0.06, 0.08, 0.08,  0.18, 0.10,-0.05],  # ETH
    [ 0.02, 0.05, 0.04, 0.08, 0.10, 0.08, 0.08, 0.05, 0.10, 0.08, 1.00, 0.20, 0.10, 0.00, 0.25, 0.70, 0.30,  0.10, 0.20, 0.10],  # GLD
    [-0.25,-0.28,-0.26,-0.20,-0.10,-0.08,-0.08,-0.05,-0.10,-0.08, 0.20, 1.00,-0.05, 0.00,-0.15, 0.10,-0.10, -0.05, 0.45, 0.45],  # TLT
    [ 0.30, 0.35, 0.33, 0.30, 0.40, 0.35, 0.40, 0.35, 0.15, 0.13, 0.10,-0.05, 1.00, 0.00, 0.20, 0.10, 0.25,  0.25, 0.30, 0.10],  # EMB
    [ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.85],  # Cash
    [ 0.10, 0.12, 0.10, 0.08, 0.55, 0.40, 0.45, 0.20, 0.08, 0.06, 0.25,-0.15, 0.20, 0.00, 1.00, 0.35, 0.55,  0.05, 0.15,-0.05],  # OIL
    [ 0.05, 0.06, 0.05, 0.05, 0.12, 0.10, 0.10, 0.08, 0.10, 0.08, 0.70, 0.10, 0.10, 0.00, 0.35, 1.00, 0.45,  0.10, 0.15, 0.00],  # SLV
    [ 0.30, 0.32, 0.30, 0.25, 0.35, 0.30, 0.32, 0.18, 0.10, 0.08, 0.30,-0.10, 0.25, 0.00, 0.55, 0.45, 1.00,  0.18, 0.20,-0.02],  # COPR
    [ 0.65, 0.75, 0.65, 0.72, 0.25, 0.22, 0.25, 0.18, 0.20, 0.18, 0.10,-0.05, 0.25, 0.00, 0.05, 0.10, 0.18,  1.00, 0.55, 0.00],  # XLV (Healthcare)
    [ 0.40, 0.50, 0.45, 0.55, 0.18, 0.15, 0.18, 0.12, 0.10, 0.10, 0.20, 0.45, 0.30, 0.00, 0.15, 0.15, 0.20,  0.55, 1.00, 0.15],  # XLU (Utilities)
    [-0.05,-0.05,-0.05, 0.00,-0.02,-0.02,-0.02,-0.02,-0.05,-0.05, 0.10, 0.45, 0.10, 0.85,-0.05, 0.00,-0.02,  0.00, 0.15, 1.00],  # SHY (Short-Duration UST)
])

# ── Numerical safety: validate at module import so misalignments crash on boot,
# not later via IndexError inside an optimizer call. Uses phase_h.numerics if
# available; falls back to inline asserts when phase_h hasn't been deployed.
#
# We hard-assert ONLY structural invariants (shape, symmetry). PSD is
# soft-validated and the existing `_build_cov_matrix` performs the
# eigenvalue clip downstream (preserves the original tolerance behavior
# of the 17x17 matrix which was also marginally near-PSD by design).
try:
    from phase_h.numerics import (
        assert_universe_synchronised as _ph_assert_universe,
        validate_psd as _ph_validate_psd,
    )
    _ph_assert_universe(universe_size=len(_UNIVERSE), cov=_CORR,
                        asset_names=[a.name for a in _UNIVERSE])
    _ph_psd_res, _ = _ph_validate_psd(_CORR, label="_CORR", fix_if_violation=False)
    if not _ph_psd_res.ok:
        _ph_bad = [f for f in _ph_psd_res.findings if f.severity == "fail"]
        logger.warning(
            "_CORR is not strictly PSD (will be eigenvalue-clipped in _build_cov_matrix): %s",
            "; ".join(f"{f.check}={f.detail}" for f in _ph_bad),
        )
except ImportError:
    pass
assert _CORR.shape == (len(_UNIVERSE), len(_UNIVERSE)), \
    f"_CORR shape {_CORR.shape} ≠ universe size {len(_UNIVERSE)}"
assert np.allclose(_CORR, _CORR.T, atol=1e-9), "_CORR not symmetric"

# ── Risk Profile Presets ───────────────────────────────────────────────────────
_PROFILES = {
    "conservative": {
        "label":        "Capital Preservation Mandate",
        "description":  "Capital preservation · Low volatility · Long-duration anchor allocation",
        "max_beta":     0.70,
        "max_vol":      0.12,
        "region_caps":  {"US": 0.50, "GCC": 0.25, "Egypt": 0.05, "Crypto": 0.00, "Gold": 0.20, "Bonds": 0.50, "Cash": 0.20, "Commodities": 0.05, "Diversification": 0.07},
        "min_bonds_cash": 0.30,
        "risk_aversion": 10.0,
    },
    "balanced": {
        "label":        "Balanced Multi-Asset Mandate",
        "description":  "Moderate growth · Multi-asset · Diversified core allocation",
        "max_beta":     1.00,
        "max_vol":      0.18,
        "region_caps":  {"US": 0.50, "GCC": 0.30, "Egypt": 0.10, "Crypto": 0.10, "Gold": 0.20, "Bonds": 0.35, "Cash": 0.10, "Commodities": 0.10, "Diversification": 0.07},
        "min_bonds_cash": 0.15,
        "risk_aversion": 4.0,
    },
    "growth": {
        "label":        "Long-Horizon Growth Mandate",
        "description":  "High return target · Diversified global growth · Long-horizon equity tilt",
        "max_beta":     1.30,
        "max_vol":      0.25,
        "region_caps":  {"US": 0.60, "GCC": 0.35, "Egypt": 0.15, "Crypto": 0.10, "Gold": 0.15, "Bonds": 0.20, "Cash": 0.05, "Commodities": 0.15, "Diversification": 0.07},
        "min_bonds_cash": 0.05,
        "risk_aversion": 1.5,
    },
    "aggressive": {
        "label":        "Aggressive Growth Mandate",
        "description":  "Maximum return · Elevated risk tolerance · Concentrated growth tilt",
        "max_beta":     1.80,
        "max_vol":      0.40,
        "region_caps":  {"US": 0.70, "GCC": 0.40, "Egypt": 0.20, "Crypto": 0.10, "Gold": 0.20, "Bonds": 0.15, "Cash": 0.05, "Commodities": 0.20, "Diversification": 0.05},
        "min_bonds_cash": 0.00,
        "risk_aversion": 0.6,
    },
}


def _build_cov_matrix(assets: list[AssetClass]) -> np.ndarray:
    """
    Build covariance matrix supporting both static universe assets and
    dynamically-added live stocks. Static pairs use the pre-calibrated _CORR;
    dynamic pairs use region-based cross-correlation estimates.
    """
    n = len(assets)
    vols = np.array([a.vol_annual for a in assets])

    # Build a map: asset name → index in _UNIVERSE (None if dynamic/live stock)
    _static_idx = {}
    for i, u in enumerate(_UNIVERSE):
        _static_idx[u.name] = i

    # Cross-region correlation defaults
    _REGION_CROSS = {
        ("US", "GCC"):         0.35,
        ("US", "Egypt"):       0.27,
        ("US", "Gold"):        0.03,
        ("US", "Commodities"): 0.12,
        ("US", "Bonds"):      -0.25,
        ("US", "Crypto"):      0.32,
        ("GCC", "Egypt"):      0.38,
        ("GCC", "Gold"):       0.10,
        ("GCC", "Commodities"):0.50,
        ("GCC", "Bonds"):     -0.08,
        ("GCC", "Crypto"):     0.17,
        ("Egypt", "Gold"):     0.05,
        ("Egypt", "Commodities"):0.20,
        ("Egypt", "Bonds"):   -0.05,
        ("Gold", "Commodities"):0.30,
        ("Gold", "Bonds"):     0.20,
        ("Commodities", "Bonds"):-0.12,
        ("US", "Diversification"):     0.25,
        ("GCC", "Diversification"):    0.15,
        ("Crypto", "Diversification"): 0.05,
        ("Gold", "Diversification"):   0.10,
        ("Bonds", "Diversification"):  0.30,
        ("Commodities", "Diversification"): 0.10,
        ("Egypt", "Diversification"):  0.12,
    }

    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = assets[i], assets[j]
            si = _static_idx.get(ai.name)
            sj = _static_idx.get(aj.name)
            if si is not None and sj is not None:
                # Both in static universe — use calibrated _CORR (full 20x20)
                c = float(_CORR[si, sj])
            elif ai.region == aj.region:
                # Same region (e.g. two KSA live stocks)
                c = 0.65
            else:
                key = tuple(sorted([ai.region, aj.region]))
                c = _REGION_CROSS.get(key, 0.20)
            corr[i, j] = corr[j, i] = c

    cov = np.outer(vols, vols) * corr
    # Ensure strict PSD via eigenvalue clipping (handles dynamic asset additions)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-7)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    cov = (cov + cov.T) / 2   # symmetry guard
    return cov


def allocate(
    profile:          str = "balanced",
    region_include:   Optional[list[str]] = None,   # e.g. ["US","GCC","Gold"]
    region_exclude:   Optional[list[str]] = None,   # e.g. ["Crypto"]
    custom_caps:      Optional[dict] = None,         # {region: max_weight}
    language:         str = "en",                    # "en" | "ar" — drives error message language
    rf_rate:          float = 0.045,
    port_value_usd:   float = 100_000,
    max_drawdown:     float = 1.0,                   # e.g. 0.25 = 25% max drawdown
    # ── Phase H additive params (default-None preserves prior callers) ──
    w_prev:               Optional[dict] = None,
    rebalance_frequency:  str = "quarterly",
    committee_mode:       Optional[str] = None,
    horizon_years:        float = 5.0,
    asset_kind:           Optional[str] = None,
    region_tilt:          Optional[str] = None,
    benchmark_ticker:     Optional[str] = None,
) -> dict:
    """
    Main entry point. Returns full global allocation recommendation.

    Parameters
    ----------
    profile : "conservative" | "balanced" | "growth" | "aggressive"
    region_include : only include these regions (None = all)
    region_exclude : exclude these regions (None = none excluded)
    custom_caps : override default region caps
    rf_rate : risk-free rate (default 4.5%)
    port_value_usd : portfolio size for dollar amounts

    Returns
    -------
    dict with keys: weights, metrics, report_md, feasibility
    """
    try:
        import cvxpy as cp
    except ImportError:
        return {"error": "cvxpy not installed"}

    prof = _PROFILES.get(profile, _PROFILES["balanced"])
    region_caps = dict(prof["region_caps"])
    if custom_caps:
        region_caps.update(custom_caps)

    # ── Feasibility guard: ensure included regions' caps sum ≥ 100% ────────
    # When the user pins tight caps (e.g. "crypto and metals 5% each") on top
    # of a region_include filter, the per-region maxes can sum below 1.0,
    # making sum(w)=1 infeasible. Auto-expand the largest un-custom-capped
    # region so the universe can still fill the portfolio.
    if region_include:
        _custom_keys = set((custom_caps or {}).keys())
        _included_cap_sum = sum(region_caps.get(r, 0.40) for r in region_include)
        if _included_cap_sum < 1.05:  # need 5% headroom for solver
            _gap = 1.05 - _included_cap_sum
            _expandable = [r for r in region_include if r not in _custom_keys]
            if _expandable:
                # Prefer expanding higher-default-cap regions (US/GCC) first
                _expandable.sort(key=lambda r: region_caps.get(r, 0.40), reverse=True)
                _slice = _gap / len(_expandable)
                for r in _expandable:
                    region_caps[r] = min(1.0, region_caps.get(r, 0.40) + _slice)

    # ── Filter static universe ─────────────────────────────────────────────
    assets = []
    for a in _UNIVERSE:
        if region_include and a.region not in region_include:
            continue
        if region_exclude and a.region in region_exclude:
            continue
        a_cap = region_caps.get(a.region, 0.40)
        if a_cap == 0:
            continue
        ac = AssetClass(
            name=a.name, region=a.region, proxy=a.proxy,
            mu_annual=a.mu_annual, vol_annual=a.vol_annual, beta_world=a.beta_world,
            min_w=a.min_w, max_w=min(a.max_w, a_cap),
            currency=a.currency, description=a.description,
        )
        assets.append(ac)

    # ── Inject live stocks from market cache ───────────────────────────────
    # Maps: requested region tag → (market_code, n_stocks)
    _LIVE_MARKET_MAP = {
        "GCC":   [("ksa", 3), ("uae", 2)],
        "Egypt": [("egypt", 3)],
        "US":    [("america", 3)],
    }
    _live_names = set()
    for region_tag, markets in _LIVE_MARKET_MAP.items():
        # Only inject if this region is requested and not excluded
        if region_include and region_tag not in region_include:
            continue
        if region_exclude and region_tag in region_exclude:
            continue
        a_cap = region_caps.get(region_tag, 0.40)
        if a_cap == 0:
            continue
        for mkt_code, n in markets:
            live_stocks = _select_top_stocks(mkt_code, region_tag, n=n)
            for ls in live_stocks:
                if ls.name not in _live_names:
                    ls.max_w = min(ls.max_w, a_cap)
                    assets.append(ls)
                    _live_names.add(ls.name)

    if len(assets) < 2:
        return {"error": "Too few asset classes after filtering. Include more regions."}

    n   = len(assets)
    mu  = np.array([a.mu_annual for a in assets])
    cov = _build_cov_matrix(assets)
    betas = np.array([a.beta_world for a in assets])

    # Region grouping matrix
    regions = sorted(set(a.region for a in assets))
    R_mat = np.array([[1 if a.region == r else 0 for a in assets] for r in regions])
    r_caps = np.array([region_caps.get(r, 0.40) for r in regions])

    # ── QP: Mean-variance utility maximization ────────────────────────────
    risk_aversion = prof.get("risk_aversion", 4.0)
    w = cp.Variable(n, nonneg=True)

    constr = [cp.sum(w) == 1]
    # Min weight per asset (only if included)
    for i, a in enumerate(assets):
        if a.min_w > 0:
            constr.append(w[i] >= a.min_w)
    # Max weight per asset
    for i, a in enumerate(assets):
        constr.append(w[i] <= a.max_w)
    # Region caps
    constr.append(R_mat @ w <= r_caps)
    # Thematic cap: Crypto ≤ 10% (enforced via region cap above)
    # Semiconductor: handled post-hoc with flagging (no explicit ETF to filter by)
    # Beta cap
    constr.append(betas @ w <= prof["max_beta"])
    # Volatility cap in portfolio weight space
    constr.append(cp.quad_form(w, cov) <= prof["max_vol"]**2)
    # Min bonds + cash (stability floor)
    # Auto-relaxed when bonds/cash region absent from filtered universe — otherwise
    # any user request that excludes bonds (e.g. "USA + KSA stocks only") becomes infeasible.
    if prof["min_bonds_cash"] > 0:
        bonds_cash_idx = [i for i, a in enumerate(assets) if a.region in ("Bonds","Cash")]
        if bonds_cash_idx:
            constr.append(cp.sum(w[bonds_cash_idx]) >= prof["min_bonds_cash"])

    objective_base = mu @ w - (risk_aversion / 2.0) * cp.quad_form(w, cov)
    objective_expr = objective_base
    _tc_optimizer_notes: list[str] = []
    _tc_terms_applied = False

    try:
        from phase_h.registry import FeatureRegistry
        if FeatureRegistry.is_enabled("phase_h_tc_optimizer"):
            from phase_h.tc_optimizer import build_turnover_terms

            w_prev_vec = None
            if w_prev is not None:
                _prev_vals = []
                for a in assets:
                    _prev_vals.append(float(w_prev.get(a.name, w_prev.get(a.proxy, 0.0)) or 0.0))
                w_prev_vec = np.array(_prev_vals, dtype=float)
                if w_prev_vec.sum() > 1.5:
                    w_prev_vec = w_prev_vec / 100.0

            if w_prev_vec is not None:
                lin_l = float(os.environ.get("EISAX_TC_LINEAR_LAMBDA", "0.0010"))
                quad_l = float(os.environ.get("EISAX_TC_QUADRATIC_LAMBDA", "0.0005"))
                pers_l = float(os.environ.get("EISAX_TC_PERSISTENCE_LAMBDA", "0.0002"))
                if not all(np.isfinite(x) for x in (lin_l, quad_l, pers_l)):
                    _tc_optimizer_notes.append("turnover penalty relaxed due to infeasibility")
                else:
                    lin, quad, pers = build_turnover_terms(
                        cp, w, cp.Constant(w_prev_vec),
                        linear_lambda=lin_l, quadratic_lambda=quad_l, persistence_lambda=pers_l,
                    )
                    objective_expr = objective_expr - lin - quad - pers
                    _tc_terms_applied = True
    except Exception as _tc_exc:
        logger.warning("Phase H2 turnover objective skipped: %r", _tc_exc)
        _tc_optimizer_notes.append("turnover penalty relaxed due to infeasibility")

    prob = cp.Problem(
        cp.Maximize(objective_expr),
        constr,
    )

    # Try CLARABEL first, fall back to SCS
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status != "optimal" or w.value is None:
            prob.solve(solver=cp.SCS, eps=1e-5, verbose=False)
    except Exception as _tc_solve_exc:
        if not _tc_terms_applied:
            raise
        logger.warning("Phase H2 turnover solve failed; retrying base objective: %r", _tc_solve_exc)
        _tc_optimizer_notes.append("turnover penalty relaxed due to infeasibility")
        prob = cp.Problem(cp.Maximize(objective_base), constr)
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status != "optimal" or w.value is None:
            prob.solve(solver=cp.SCS, eps=1e-5, verbose=False)

    if _tc_terms_applied and (prob.status not in ("optimal", "optimal_inaccurate") or w.value is None):
        _tc_optimizer_notes.append("turnover penalty relaxed due to infeasibility")
        prob = cp.Problem(cp.Maximize(objective_base), constr)
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status != "optimal" or w.value is None:
            prob.solve(solver=cp.SCS, eps=1e-5, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        if language == "ar":
            _msg = (f"لم يتم العثور على تخصيص ممكن مع قيود الـ {profile}. "
                    "جرّب profile أقل تشدداً أو أضف المزيد من المناطق.")
        else:
            _msg = (f"No feasible allocation under {profile} constraints. "
                    "Try a less restrictive profile or include more regions.")
        return {
            "error": _msg,
            "profile": profile,
            "solver_status": prob.status,
        }

    w_opt = np.maximum(np.array(w.value, dtype=float), 0.0)
    w_opt /= w_opt.sum()

    # Clip to per-asset max (absorb float drift ≤ 0.001)
    for i, a in enumerate(assets):
        w_opt[i] = min(w_opt[i], a.max_w + 0.001)
    w_opt /= w_opt.sum()

    # ── Compute portfolio metrics ──────────────────────────────────────────
    port_ret  = float(mu @ w_opt)
    port_vol  = float(np.sqrt(w_opt @ cov @ w_opt))
    port_sharpe = round((port_ret - rf_rate) / port_vol, 2) if port_vol > 0 else 0
    port_beta   = float(betas @ w_opt)

    # ── Guard: reject portfolios with negative expected return ─────────────
    if port_ret < rf_rate:
        if language == "ar":
            _msg = (
                f"الأوبتيمايزر أعطى عائداً متوقعاً سالباً ({port_ret*100:.1f}%) — "
                f"وده أقل من معدل الخطر الصفري ({rf_rate*100:.1f}%). "
                f"جرّب: (1) أضف US أو Gold للتنويع، أو (2) غيّر الـ profile لـ 'growth'، "
                f"أو (3) أضف Bonds كـ anchor للمحفظة."
            )
        else:
            _msg = (
                f"Optimizer returned a portfolio with sub-risk-free expected return ({port_ret*100:.1f}%) — "
                f"below the risk-free rate ({rf_rate*100:.1f}%). "
                f"Try: (1) add US or Gold for diversification, (2) switch profile to 'growth', "
                f"or (3) add Bonds as a portfolio anchor."
            )
        return {
            "error": _msg,
            "profile": profile,
            "computed_return": round(port_ret * 100, 2),
        }

    # ── Max Drawdown approximation guard ──────────────────────────────────
    # Approximate: Max Drawdown ≈ 2× annual vol (empirical rule for equity portfolios)
    if max_drawdown < 1.0:
        approx_mdd = port_vol * 2.0
        if approx_mdd > max_drawdown:
            if language == "ar":
                _msg = (
                    f"المحفظة المقترحة لها تقلب سنوي {port_vol*100:.1f}% → "
                    f"أقصى خسارة متوقعة ~{approx_mdd*100:.0f}% وده أكبر من الحد المطلوب {max_drawdown*100:.0f}%. "
                    f"جرّب: (1) أضف Bonds أو Gold للتهدئة، أو (2) ارفع حد الـ max-drawdown إلى ~{int(round(approx_mdd*100))}%."
                )
            else:
                _msg = (
                    f"The optimal portfolio has annual volatility of {port_vol*100:.1f}% → "
                    f"estimated max drawdown ~{approx_mdd*100:.0f}%, which exceeds your {max_drawdown*100:.0f}% limit. "
                    f"Try: (1) add Bonds or increase Gold for stability, (2) raise your max-drawdown limit to ~{int(round(approx_mdd*100))}%, "
                    f"or (3) use a more conservative profile."
                )
            return {
                "error": _msg,
                "profile": profile,
                "approx_max_drawdown": round(approx_mdd * 100, 1),
                "requested_max_drawdown": round(max_drawdown * 100, 1),
            }

    # ──────────────────────────────────────────────────────────────────────
    # Phase E — Institutional implementation layer
    # Step 1: simplicity filter — drop sub-threshold positions (<2.5%)
    # Step 2: tiered institutional rounding (5% / 2.5% / 1% grid)
    # Step 3: constraint preservation — verify rounded weights still respect caps
    # ──────────────────────────────────────────────────────────────────────
    _raw_w = w_opt.copy()

    def _simplicity_filter(weights: np.ndarray, threshold: float = 0.025) -> tuple[np.ndarray, int]:
        out = weights.copy()
        drop_mask = (out > 0) & (out < threshold)
        n_dropped = int(drop_mask.sum())
        if n_dropped > 0:
            kept_total = out[~drop_mask].sum()
            if kept_total > 0:
                out[~drop_mask] = out[~drop_mask] * (1.0 / kept_total)
            out[drop_mask] = 0
        return out, n_dropped

    def _institutional_round(weights: np.ndarray) -> np.ndarray:
        # Tiered grid by initial weight; round DOWN by default to never exceed
        # per-asset cap, then redistribute the rounding-down deficit.
        tiered = np.zeros_like(weights)
        for i, w in enumerate(weights):
            if w <= 0:
                tiered[i] = 0
                continue
            if w >= 0.15:
                grid = 0.05
            elif w >= 0.05:
                grid = 0.025
            else:
                grid = 0.01
            # Round to nearest grid unit
            candidate = round(w / grid) * grid
            # Clamp at per-asset max (preserves region caps when assets fully
            # absorb the cap; treats per-asset max as the binding upper bound)
            asset_max = assets[i].max_w
            if candidate > asset_max:
                candidate = (int(asset_max / grid)) * grid
            tiered[i] = candidate
        # Distribute residual (1.0 - sum) to under-allocated positions by
        # largest-remainder method, while respecting caps.
        residual = 1.0 - tiered.sum()
        # Use 0.5% basis units for residual distribution
        unit = 0.005
        steps = int(round(residual / unit))
        if steps != 0:
            # Sort by (raw weight - tiered weight) descending → asset with
            # largest unfilled portion gets the next unit
            sign = 1 if steps > 0 else -1
            for _ in range(abs(steps)):
                gaps = []
                for i in range(len(tiered)):
                    if tiered[i] <= 0:
                        continue
                    asset_max = assets[i].max_w
                    if sign > 0 and tiered[i] + unit <= asset_max + 1e-9:
                        gaps.append((weights[i] - tiered[i], i))
                    elif sign < 0 and tiered[i] - unit > 0:
                        gaps.append((tiered[i] - weights[i], i))
                if not gaps:
                    break
                gaps.sort(reverse=True)
                idx = gaps[0][1]
                tiered[idx] += sign * unit
        # Final clamp
        for i in range(len(tiered)):
            if tiered[i] > assets[i].max_w + 1e-9:
                tiered[i] = assets[i].max_w
        # Renormalize tiny drift
        s = tiered.sum()
        if 0.98 < s < 1.02 and abs(s - 1.0) > 0.001:
            # Scale gently then snap to 0.5%
            tiered = tiered * (1.0 / s)
            tiered = np.round(tiered * 200) / 200
        return np.maximum(tiered, 0)

    def _validate_caps(weights: np.ndarray, region_cap_map: dict, hard_caps_only: bool = True) -> list[str]:
        breaches: list[str] = []
        # Region caps
        for r in set(a.region for a in assets):
            used = sum(weights[i] for i, a in enumerate(assets) if a.region == r)
            cap = region_cap_map.get(r, 0.40)
            if used > cap + 0.005:  # 0.5% rounding tolerance
                breaches.append(f"{r} {used*100:.1f}% > cap {cap*100:.0f}%")
        # Per-asset max
        for i, a in enumerate(assets):
            if weights[i] > a.max_w + 0.005:
                breaches.append(f"{a.name} {weights[i]*100:.1f}% > asset cap {a.max_w*100:.0f}%")
        return breaches

    # Apply simplicity filter, then institutional rounding
    _w_filtered, _n_dropped = _simplicity_filter(_raw_w, threshold=0.025)
    _w_rounded = _institutional_round(_w_filtered)
    _round_breaches = _validate_caps(_w_rounded, region_caps)

    # Sharpe improvement check — accept rounding only if marginal improvement loss is small.
    # Compute baseline (raw) and rounded portfolio Sharpes
    _baseline_sharpe = (float(mu @ _raw_w) - rf_rate) / max(float(np.sqrt(_raw_w @ cov @ _raw_w)), 1e-9)
    _rounded_sharpe  = (float(mu @ _w_rounded) - rf_rate) / max(float(np.sqrt(_w_rounded @ cov @ _w_rounded)), 1e-9)
    _sharpe_delta = _rounded_sharpe - _baseline_sharpe

    if _round_breaches or _sharpe_delta < -0.20:
        # Rounding caused constraint breach OR significant Sharpe loss — fall back to raw
        _use_rounded = False
        w_opt = _raw_w
        _institutional_note = (
            f"Rounded portfolio failed institutional check ({'cap breach' if _round_breaches else 'Sharpe loss > 0.20'}); "
            "raw optimizer weights preserved."
        )
    else:
        _use_rounded = True
        w_opt = _w_rounded
        # Recompute aggregate metrics on rounded weights
        port_ret    = float(mu @ w_opt)
        port_vol    = float(np.sqrt(w_opt @ cov @ w_opt))
        port_sharpe = round((port_ret - rf_rate) / port_vol, 2) if port_vol > 0 else 0
        port_beta   = float(betas @ w_opt)
        _institutional_note = (
            f"Institutional rounding applied (5%/2.5%/1% tiered grid). "
            + (f"{_n_dropped} sub-2.5% positions consolidated. " if _n_dropped else "")
            + f"Sharpe drift: {_sharpe_delta:+.2f}."
        )

    # Region allocations (computed AFTER rounding)
    region_alloc = {}
    for r in regions:
        ridx = [i for i, a in enumerate(assets) if a.region == r]
        region_alloc[r] = round(float(sum(w_opt[i] for i in ridx)) * 100, 1)

    # Asset weights dict
    asset_weights = {
        assets[i].name: round(float(w_opt[i]) * 100, 1)
        for i in range(n) if w_opt[i] > 0.005
    }
    asset_usd = {
        assets[i].name: round(float(w_opt[i]) * port_value_usd, 0)
        for i in range(n) if w_opt[i] > 0.005
    }

    # ──────────────────────────────────────────────────────────────────────
    # Phase E — Asset role classification
    # ──────────────────────────────────────────────────────────────────────
    def _classify_asset_role(a, w: float) -> str:
        r = a.region
        if r in ("Bonds", "Cash", "Diversification"):
            return "Income / Diversification"
        if r == "Gold":
            return "Macro Hedge"
        if r == "Crypto":
            return "Opportunistic Satellite"
        if r == "Commodities":
            return "Real-Asset / Inflation Sleeve"
        # Equity assets — split by weight tier
        if w >= 0.20:
            return "Strategic Core"
        if w >= 0.075:
            return "Tactical Allocation"
        return "Satellite / Diversifier"

    asset_roles = {
        assets[i].name: _classify_asset_role(assets[i], float(w_opt[i]))
        for i in range(n) if w_opt[i] > 0.005
    }

    # ──────────────────────────────────────────────────────────────────────
    # Phase E — Implementation Feasibility Scoring
    # ──────────────────────────────────────────────────────────────────────
    _n_active = sum(1 for i in range(n) if w_opt[i] > 0.005)
    _live_pos_share = sum(w_opt[i] for i, a in enumerate(assets) if getattr(a, "source_tag", "") == "live")
    _crypto_pos_share = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "Crypto")
    _frontier_share   = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "Egypt")

    # Rebalancing complexity — fewer holdings = simpler
    if _n_active <= 6:
        _rebal_complexity = "Low"
    elif _n_active <= 10:
        _rebal_complexity = "Moderate"
    else:
        _rebal_complexity = "High"

    # Liquidity practicality — based on individual stock vs ETF exposure
    if _live_pos_share < 0.10:
        _liq_practicality = "High"
    elif _live_pos_share < 0.30:
        _liq_practicality = "Moderate"
    else:
        _liq_practicality = "Limited"

    # Execution friction — crypto + frontier introduce wider spreads
    _friction_pen = _crypto_pos_share * 0.5 + _frontier_share * 0.3 + _live_pos_share * 0.2
    if _friction_pen < 0.10:
        _execution_friction = "Low"
    elif _friction_pen < 0.25:
        _execution_friction = "Moderate"
    else:
        _execution_friction = "High"

    # Estimated annual turnover (heuristic — rebalance + drift recapture)
    _est_turnover_pct = 8 + (_n_active - 5) * 1.5 + _crypto_pos_share * 30
    _est_turnover_pct = max(5, min(40, round(_est_turnover_pct, 0)))

    # Estimated slippage on a 1× turnover roll (institutional ETF universe)
    # ETF: ~3 bp. Live single-name: ~10 bp. Crypto: ~25 bp. Frontier: ~30 bp.
    _est_slippage_bp = (
        (1.0 - _live_pos_share - _crypto_pos_share - _frontier_share) * 3
        + _live_pos_share * 10
        + _crypto_pos_share * 25
        + _frontier_share * 30
    )
    _est_slippage_bp = round(_est_slippage_bp, 1)

    # Deployability composite score (0–100)
    _deploy_score = 100
    _deploy_score -= max(0, _n_active - 8) * 3                   # over-fragmentation
    _deploy_score -= _live_pos_share * 25                        # live single-name penalty
    _deploy_score -= _crypto_pos_share * 30                      # crypto liquidity
    _deploy_score -= _frontier_share * 30                        # frontier markets
    _deploy_score -= len(_round_breaches) * 10                   # constraint friction
    _deploy_score = max(20, min(100, round(_deploy_score, 0)))
    if _deploy_score >= 80:
        _deploy_tier = "High"
    elif _deploy_score >= 60:
        _deploy_tier = "Moderate"
    else:
        _deploy_tier = "Limited"

    implementation = {
        "rebalancing_complexity":  _rebal_complexity,
        "liquidity_practicality":  _liq_practicality,
        "execution_friction":      _execution_friction,
        "est_turnover_pct":        _est_turnover_pct,
        "est_slippage_bp":         _est_slippage_bp,
        "deployability_score":     int(_deploy_score),
        "deployability_tier":      _deploy_tier,
        "n_active_positions":      _n_active,
        "institutional_note":      _institutional_note,
        "rounding_applied":        _use_rounded,
    }

    # ──────────────────────────────────────────────────────────────────────
    # Phase E — Benchmark synthesis + tracking diagnostics
    # ──────────────────────────────────────────────────────────────────────
    # Profile-mapped benchmark structure (equity/bonds/hedge mix)
    _BENCH_TARGETS = {
        "conservative": [("US", 0.30), ("Bonds", 0.50), ("Gold", 0.20)],
        "balanced":     [("US", 0.50), ("Bonds", 0.30), ("Gold", 0.10), ("GCC", 0.10)],
        "growth":       [("US", 0.65), ("Bonds", 0.15), ("Gold", 0.05), ("GCC", 0.15)],
        "aggressive":   [("US", 0.75), ("Bonds", 0.05), ("Gold", 0.05), ("GCC", 0.15)],
    }
    _bench_name_map = {
        "conservative": "30/50/20 Conservative (US Equity / Bonds / Gold)",
        "balanced":     "60/40 Balanced (Global Equity / Bonds, Gold overlay)",
        "growth":       "80/20 Growth (Global Equity / Bonds)",
        "aggressive":   "MSCI World Equity Proxy",
    }
    _bench_targets = _BENCH_TARGETS.get(profile, _BENCH_TARGETS["balanced"])
    _bench_label = _bench_name_map.get(profile, "Profile-mapped Benchmark")

    # Build aligned benchmark weights: distribute each region's target evenly across
    # that region's available assets in our universe.
    w_bench = np.zeros(n)
    for region_target, w_target in _bench_targets:
        region_idx = [i for i, a in enumerate(assets) if a.region == region_target]
        if region_idx:
            per_asset = w_target / len(region_idx)
            for i in region_idx:
                w_bench[i] = per_asset
    # Renormalize (in case some target regions absent in current universe)
    _bench_sum = w_bench.sum()
    if _bench_sum > 0:
        w_bench = w_bench / _bench_sum

    # Benchmark metrics
    _bench_ret  = float(mu @ w_bench) if _bench_sum > 0 else 0
    _bench_vol  = float(np.sqrt(w_bench @ cov @ w_bench)) if _bench_sum > 0 else 0
    _bench_beta = float(betas @ w_bench) if _bench_sum > 0 else 0
    _bench_sharpe = round((_bench_ret - rf_rate) / _bench_vol, 2) if _bench_vol > 0 else 0

    # Tracking error: vol of (port − bench)
    _active_diff = w_opt - w_bench
    _tracking_error = float(np.sqrt(_active_diff @ cov @ _active_diff)) if _bench_sum > 0 else 0
    # Active share: sum |port - bench| / 2
    _active_share = float(np.sum(np.abs(_active_diff)) / 2) if _bench_sum > 0 else 0

    # Style drift — compare portfolio vs benchmark region tilts
    _drift_signals = []
    _port_us  = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "US")
    _bench_us = sum(w_bench[i] for i, a in enumerate(assets) if a.region == "US")
    if _port_us - _bench_us > 0.10:
        _drift_signals.append("US-overweight")
    elif _bench_us - _port_us > 0.10:
        _drift_signals.append("US-underweight")
    _port_bonds  = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "Bonds")
    _bench_bonds = sum(w_bench[i] for i, a in enumerate(assets) if a.region == "Bonds")
    if _port_bonds - _bench_bonds > 0.10:
        _drift_signals.append("Duration-overweight")
    elif _bench_bonds - _port_bonds > 0.10:
        _drift_signals.append("Duration-underweight")
    if sum(w_opt[i] for i, a in enumerate(assets) if a.region == "Gold") - sum(w_bench[i] for i, a in enumerate(assets) if a.region == "Gold") > 0.05:
        _drift_signals.append("Hedge-overweight")
    if sum(w_opt[i] for i, a in enumerate(assets) if a.region == "Crypto") - sum(w_bench[i] for i, a in enumerate(assets) if a.region == "Crypto") > 0.02:
        _drift_signals.append("Crypto-tilted")
    if not _drift_signals:
        _drift_signals.append("Aligned with benchmark composition")

    # Tracking deviation classification
    if _tracking_error < 0.03:
        _track_class = "Low"
    elif _tracking_error < 0.06:
        _track_class = "Moderate"
    else:
        _track_class = "High"
    # Active share classification
    if _active_share < 0.20:
        _active_class = "Low"
    elif _active_share < 0.40:
        _active_class = "Moderate"
    else:
        _active_class = "High"

    benchmark = {
        "label":             _bench_label,
        "expected_ret_pct":  round(_bench_ret * 100, 1),
        "expected_vol_pct":  round(_bench_vol * 100, 1),
        "beta_world":        round(_bench_beta, 2),
        "sharpe":            _bench_sharpe,
        "tracking_error_pct":round(_tracking_error * 100, 2),
        "tracking_class":    _track_class,
        "active_share_pct":  round(_active_share * 100, 1),
        "active_class":      _active_class,
        "style_drift":       " · ".join(_drift_signals),
    }

    # ──────────────────────────────────────────────────────────────────────
    # Phase E — Benchmark-relative attribution
    # Decompose portfolio excess return vs benchmark into beta and alpha-like
    # components. This is heuristic; flags whether outperformance is factor-
    # driven (beta differential) vs. selection/concentration-driven.
    # ──────────────────────────────────────────────────────────────────────
    _excess_ret = port_ret - _bench_ret
    _beta_diff  = port_beta - _bench_beta
    # Beta-attribution: assume market premium = bench_ret - rf_rate
    _market_premium = max(0.0, _bench_ret - rf_rate)
    _beta_contribution = _beta_diff * _market_premium
    _residual_contribution = _excess_ret - _beta_contribution

    if abs(_beta_contribution) > abs(_residual_contribution) * 1.5:
        _attribution_verdict = (
            "Outperformance appears primarily factor-driven (beta differential) rather than selection-driven."
            if _excess_ret > 0 else
            "Underperformance attributable primarily to factor exposure (beta differential)."
        )
    elif abs(_residual_contribution) > abs(_beta_contribution) * 1.5:
        _attribution_verdict = (
            "Excess return concentrated in residual/selection effects — review concentration risk before attributing to skill."
            if _excess_ret > 0 else
            "Underperformance concentrated in residual/selection effects — examine specific holdings."
        )
    else:
        _attribution_verdict = "Excess return roughly balanced between factor exposure and selection / concentration."

    attribution = {
        "excess_return_pct":          round(_excess_ret * 100, 2),
        "beta_contribution_pct":      round(_beta_contribution * 100, 2),
        "residual_contribution_pct":  round(_residual_contribution * 100, 2),
        "interpretation":             _attribution_verdict,
    }

    # ──────────────────────────────────────────────────────────────────────
    # B. Mandate Feasibility Diagnostics — checked constraints + status
    # ──────────────────────────────────────────────────────────────────────
    constraint_diagnostics: list[dict] = []

    # Region cap utilization
    for r, cap in region_caps.items():
        used = region_alloc.get(r, 0) / 100.0
        if used > 0.001:
            margin = cap - used
            if margin < 0.005:
                status = "AT CAP"
            elif margin < 0.03:
                status = "NEAR CAP"
            else:
                status = "PASS"
            constraint_diagnostics.append({
                "name": f"Region cap · {r}",
                "limit_pct": round(cap * 100, 1),
                "actual_pct": round(used * 100, 1),
                "status": status,
            })

    # Beta cap
    _beta_margin = prof["max_beta"] - port_beta
    constraint_diagnostics.append({
        "name": "Beta cap (vs MSCI World)",
        "limit_pct": prof["max_beta"],
        "actual_pct": round(port_beta, 3),
        "status": "AT CAP" if _beta_margin < 0.05 else ("NEAR CAP" if _beta_margin < 0.15 else "PASS"),
    })

    # Vol cap
    _vol_margin = prof["max_vol"] - port_vol
    constraint_diagnostics.append({
        "name": "Volatility cap (annualized)",
        "limit_pct": round(prof["max_vol"] * 100, 1),
        "actual_pct": round(port_vol * 100, 2),
        "status": "AT CAP" if _vol_margin < 0.01 else ("NEAR CAP" if _vol_margin < 0.03 else "PASS"),
    })

    # Bonds/cash floor
    _bc_used = sum(w_opt[i] for i, a in enumerate(assets) if a.region in ("Bonds", "Cash"))
    if prof["min_bonds_cash"] > 0:
        if any(a.region in ("Bonds", "Cash") for a in assets):
            _bc_status = "PASS" if (_bc_used - prof["min_bonds_cash"]) > 0.01 else "AT FLOOR"
            constraint_diagnostics.append({
                "name": "Minimum bonds + cash floor",
                "limit_pct": round(prof["min_bonds_cash"] * 100, 1),
                "actual_pct": round(_bc_used * 100, 1),
                "status": _bc_status,
            })
        else:
            constraint_diagnostics.append({
                "name": "Minimum bonds + cash floor",
                "limit_pct": round(prof["min_bonds_cash"] * 100, 1),
                "actual_pct": 0.0,
                "status": "AUTO-RELAXED (bonds/cash region not included)",
            })

    # Max drawdown estimate
    if max_drawdown < 1.0:
        _approx_mdd = port_vol * 2.0
        constraint_diagnostics.append({
            "name": "Max drawdown limit (modeled)",
            "limit_pct": round(max_drawdown * 100, 1),
            "actual_pct": round(_approx_mdd * 100, 1),
            "status": "PASS" if _approx_mdd <= max_drawdown else "BREACH",
        })

    # Number of holdings (diversification floor)
    _n_holdings = sum(1 for i in range(n) if w_opt[i] > 0.005)
    constraint_diagnostics.append({
        "name": "Holdings count",
        "limit_pct": "≥ 5",
        "actual_pct": _n_holdings,
        "status": "PASS" if _n_holdings >= 5 else "LOW DIVERSIFICATION",
    })

    # ──────────────────────────────────────────────────────────────────────
    # E. Quantified Rebalance Suggestions — top concentrations + impact
    # ──────────────────────────────────────────────────────────────────────
    rebalance_suggestions: list[dict] = []
    _top_concentrated = sorted(
        [(i, w_opt[i]) for i in range(n) if w_opt[i] > 0.15 and assets[i].region != "Cash"],
        key=lambda x: -x[1],
    )[:3]
    for _i, _w in _top_concentrated:
        _a = assets[_i]
        _reduction = _w * 0.4  # propose cutting by 40%
        _new_w = _w - _reduction
        _other_idx = [_j for _j in range(n) if _j != _i and w_opt[_j] > 0]
        _other_total = sum(w_opt[_j] for _j in _other_idx)
        _new_w_arr = w_opt.copy()
        _new_w_arr[_i] = _new_w
        if _other_total > 0:
            _scale = (1.0 - _new_w) / _other_total
            for _j in _other_idx:
                _new_w_arr[_j] *= _scale
        _new_beta = float(betas @ _new_w_arr)
        _new_vol = float(np.sqrt(_new_w_arr @ cov @ _new_w_arr))
        # Concentration delta: drop in this asset's weight
        _conc_delta_pp = (_w - _new_w) * 100
        _diff = "LOW" if _reduction < 0.05 else ("MODERATE" if _reduction < 0.10 else "HIGH")
        rebalance_suggestions.append({
            "asset_name":    _a.name,
            "proxy":         _a.proxy,
            "region":        _a.region,
            "weight_before_pct": round(_w * 100, 1),
            "weight_after_pct":  round(_new_w * 100, 1),
            "beta_before":       round(port_beta, 3),
            "beta_after":        round(_new_beta, 3),
            "beta_delta":        round(_new_beta - port_beta, 3),
            "vol_before_pct":    round(port_vol * 100, 2),
            "vol_after_pct":     round(_new_vol * 100, 2),
            "vol_delta_pp":      round((_new_vol - port_vol) * 100, 2),
            "concentration_delta_pp": round(_conc_delta_pp, 1),
            "turnover_pp":       round(_reduction * 100, 1),
            "implementation_difficulty": _diff,
        })

    # ──────────────────────────────────────────────────────────────────────
    # G. Audit Appendix — reproducibility hash + constraint values
    # ──────────────────────────────────────────────────────────────────────
    import hashlib as _hashlib
    _audit_input = (
        f"prof={profile}|"
        f"inc={tuple(sorted(region_include or []))}|"
        f"exc={tuple(sorted(region_exclude or []))}|"
        f"caps={tuple(sorted((custom_caps or {}).items()))}|"
        f"port={port_value_usd}|"
        f"mdd={max_drawdown}|"
        f"rf={rf_rate}|"
        f"univ={tuple(sorted(a.name for a in assets))}"
    )
    _snapshot_id = _hashlib.sha256(_audit_input.encode()).hexdigest()[:12]
    _universe_hash = _hashlib.sha256(",".join(sorted(a.name for a in assets)).encode()).hexdigest()[:12]

    # ──────────────────────────────────────────────────────────────────────
    # Phase D — Confidence calibration
    # Reflects evidence breadth, data coverage, and reliability of inputs.
    # ──────────────────────────────────────────────────────────────────────
    _live_asset_count = sum(1 for a in assets if getattr(a, "source_tag", "") == "live")
    _live_share = _live_asset_count / max(1, n)
    _illiquid_share = sum(w_opt[i] for i, a in enumerate(assets) if a.region in ("Egypt", "Crypto") or _live_asset_count > 0 and getattr(a, "source_tag", "") == "live")
    _base_confidence = 0.85
    _base_confidence -= 0.20 * _live_share            # live stocks have shallower data history
    _base_confidence -= 0.15 * min(1.0, _illiquid_share / 0.30)  # illiquid exposure penalty
    if prob.status == "optimal_inaccurate":
        _base_confidence -= 0.05
    _base_confidence = max(0.40, min(0.92, _base_confidence))
    _evidence_breadth = "Broad" if _n_holdings >= 8 else ("Moderate" if _n_holdings >= 5 else "Limited")
    _coverage_quality = "Full" if _live_share < 0.15 else ("Partial" if _live_share < 0.40 else "Sparse")
    if _live_share < 0.10 and _illiquid_share < 0.05:
        _reliability_tier = "Institutional"
    elif _live_share < 0.30 and _illiquid_share < 0.20:
        _reliability_tier = "Institutional-Lite"
    else:
        _reliability_tier = "Indicative"
    confidence = {
        "score_pct":        round(_base_confidence * 100, 0),
        "evidence_breadth": _evidence_breadth,
        "coverage_quality": _coverage_quality,
        "reliability_tier": _reliability_tier,
        "live_asset_share_pct": round(_live_share * 100, 1),
    }

    # ──────────────────────────────────────────────────────────────────────
    # Phase D — Portfolio Regime Classification
    # Identifies dominant style of the constructed portfolio.
    # ──────────────────────────────────────────────────────────────────────
    _bonds_w  = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "Bonds")
    _cash_w   = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "Cash")
    _div_w    = sum(w_opt[i] for i, a in enumerate(assets) if "dividend" in (a.description or "").lower() or "yield" in (a.description or "").lower())
    _us_tech_w_d = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "US" and any(x in (a.description or "").lower() for x in ("tech", "nasdaq", "semiconductor", "ai")))
    _crypto_w_d  = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "Crypto")
    _gold_w   = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "Gold")
    _comm_w_d = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "Commodities")
    _lt_bond_w = sum(w_opt[i] for i, a in enumerate(assets) if a.proxy == "TLT")
    # Choose regime by dominant signature
    if _us_tech_w_d + _crypto_w_d > 0.40:
        _regime = "Momentum-Driven"
        _regime_note = "Concentrated in high-multiple growth and asymmetric satellite assets. High dispersion in risk-off regimes."
    elif _us_tech_w_d > 0.30:
        _regime = "Growth Concentrated"
        _regime_note = "Allocation tilted toward long-duration growth equities. Sensitive to discount-rate compression."
    elif _bonds_w + _cash_w + _div_w > 0.45:
        _regime = "Defensive Income"
        _regime_note = "Income-generating sleeves dominate. Lower sensitivity to equity drawdowns; primary risk vector is duration and credit spread widening."
    elif _gold_w + _comm_w_d > 0.30:
        _regime = "Inflation-Sensitive"
        _regime_note = "Real-asset weighting elevated. Performs in stagflationary or USD-weakening regimes; lags during disinflation."
    elif _lt_bond_w > 0.20:
        _regime = "Duration-Exposed"
        _regime_note = "Long-duration bond exposure dominates fixed-income sleeve. High sensitivity to real rates and curve steepening."
    elif _comm_w_d + sum(w_opt[i] for i, a in enumerate(assets) if a.region == "GCC") > 0.40:
        _regime = "Cyclical Value"
        _regime_note = "Allocation tilted toward commodity-linked and emerging-market cyclical exposure. Procyclical with global PMI cycle."
    else:
        _regime = "Multi-Asset Macro"
        _regime_note = "Balanced cross-asset construction; no single regime dominates. Targets diversification across factor and macro drivers."
    # Benchmark-aware regime behavior — which macro environments favor vs. fade the construction
    _BENCH_REGIME_BEHAVIOR = {
        "Momentum-Driven":     "Outperforms in falling-rate, risk-on regimes; lags during commodity-led value rotations and rate-shock episodes.",
        "Growth Concentrated": "Outperforms in falling-rate growth regimes (discount-rate compression); lags during inflation surprises and value rotations.",
        "Defensive Income":    "Outperforms during equity drawdowns and disinflationary cycles; lags in strong risk-on rallies and steepening yield curves.",
        "Inflation-Sensitive": "Outperforms in stagflationary and USD-weakening regimes; lags during disinflation and strong USD episodes.",
        "Duration-Exposed":    "Outperforms in falling-real-rate regimes; vulnerable to inflation surprises and curve steepening.",
        "Cyclical Value":      "Outperforms during commodity-led reflationary cycles and global PMI expansions; lags in growth-led rallies and risk-off episodes.",
        "Multi-Asset Macro":   "Designed for regime-balanced behavior; expect peer-like performance across most macro environments with reduced tail volatility.",
    }
    regime = {
        "classification":   _regime,
        "implication":      _regime_note,
        "regime_behavior":  _BENCH_REGIME_BEHAVIOR.get(_regime, ""),
    }

    # ──────────────────────────────────────────────────────────────────────
    # Phase D — Adaptive Risk Disclosures
    # Conditional on portfolio characteristics, not boilerplate.
    # ──────────────────────────────────────────────────────────────────────
    adaptive_disclaimers: list[dict] = []
    if _crypto_w_d > 0.05:
        adaptive_disclaimers.append({
            "severity": "HIGH",
            "topic":    "Crypto Liquidity Discontinuity",
            "note":     (f"Crypto exposure of {_crypto_w_d*100:.0f}% subject to 24/7 trading, regulatory regime shifts, "
                         "and liquidity discontinuities during stress events. Classify as satellite, not core."),
        })
    _gcc_w_d = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "GCC")
    if _gcc_w_d > 0.30:
        adaptive_disclaimers.append({
            "severity": "MODERATE",
            "topic":    "GCC Cyclical Concentration",
            "note":     (f"GCC weighting of {_gcc_w_d*100:.0f}% creates oil-price and geopolitical concentration. "
                         "Co-moves with commodity cycle and USD direction (peg-driven)."),
        })
    # Single-name concentration
    _max_single_w = float(max(w_opt)) if len(w_opt) else 0.0
    if _max_single_w > 0.25:
        _max_idx = int(np.argmax(w_opt))
        _max_name = assets[_max_idx].name
        adaptive_disclaimers.append({
            "severity": "HIGH",
            "topic":    "Single-Asset Concentration",
            "note":     (f"Top position ({_max_name}) represents {_max_single_w*100:.0f}% of the portfolio. "
                         "Idiosyncratic risk exceeds typical institutional concentration limits."),
        })
    # FX exposure
    _fx_w = sum(w_opt[i] for i, a in enumerate(assets) if a.region in ("GCC", "Egypt"))
    if _fx_w > 0.10:
        adaptive_disclaimers.append({
            "severity": "MODERATE",
            "topic":    "Cross-Currency Exposure",
            "note":     (f"{_fx_w*100:.0f}% of portfolio in non-USD-denominated holdings. SAR/AED pegged to USD; "
                         "EGP floats — currency translation risk applies during reporting."),
        })
    # High beta
    if port_beta > 1.10:
        adaptive_disclaimers.append({
            "severity": "HIGH" if port_beta > 1.30 else "MODERATE",
            "topic":    "Elevated Market Sensitivity",
            "note":     (f"Portfolio beta ({port_beta:.2f}) above 1.0 — amplifies market drawdowns. "
                         "Loss expectation in a 20% market correction: approximately {:.0f}%.".format(port_beta * 20)),
        })

    # ──────────────────────────────────────────────────────────────────────
    # Phase D — Model Constraints (structural limitations of the engine)
    # ──────────────────────────────────────────────────────────────────────
    model_constraints = [
        "Historical simulation uses 252-day trailing window; structural breaks beyond that window are not captured.",
        "Correlation matrix is point-in-time; pairwise correlations rise toward 1.0 during liquidity events.",
        "Volatility is non-stationary; realized vol can diverge materially from in-sample estimates during regime shifts.",
        "Live-stock prices are cached at 15-minute intervals; intra-window movements not reflected.",
        "Beta estimates assume linear market sensitivity; convex behavior (gamma) ignored.",
        "Optimizer assumes frictionless rebalancing; transaction costs, slippage, and tax drag are out-of-scope.",
    ]

    audit = {
        "snapshot_id":       _snapshot_id,
        "solver_primary":    "CLARABEL (cvxpy QP)",
        "solver_status":     prob.status,
        "n_assets_universe": n,
        "n_assets_selected": _n_holdings,
        "universe_hash":     _universe_hash,
        "constraint_values": {
            "max_beta":              prof["max_beta"],
            "max_vol_pct":           round(prof["max_vol"] * 100, 1),
            "min_bonds_cash_pct":    round(prof["min_bonds_cash"] * 100, 1),
            "max_drawdown_pct":      (round(max_drawdown * 100, 1) if max_drawdown < 1.0 else None),
            "risk_aversion":         prof["risk_aversion"],
            "rf_rate_pct":           round(rf_rate * 100, 2),
            "custom_caps":           dict((k, round(v*100, 1)) for k, v in (custom_caps or {}).items()),
        },
        "model_constraints":   model_constraints,
    }

    # ──────────────────────────────────────────────────────────────────────
    # Build markdown report  —  Sections C (Risk Diagnostics) + D (Allocation Logic)
    # ──────────────────────────────────────────────────────────────────────
    # Helper: text severity tag
    def _tag(value: float, thresholds: tuple, labels: tuple) -> str:
        for t, lbl in zip(thresholds, labels):
            if value < t:
                return f"[{lbl}]"
        return f"[{labels[-1]}]"

    # Pre-compute risk diagnostics content (Section C)
    _risk_lines: list[str] = []

    # Risk Disclosures table (only for asset classes present)
    _present_regions = {a.region for i, a in enumerate(assets) if w_opt[i] > 0.001}
    _has_emb = any(a.proxy == "EMB" and w_opt[i] > 0.001 for i, a in enumerate(assets))
    _has_non_usd = any(a.region in ("GCC", "Egypt") for i, a in enumerate(assets) if w_opt[i] > 0.001)

    # Adaptive risk disclosures — driven by actual portfolio characteristics
    if adaptive_disclaimers:
        _risk_lines += [
            "### Adaptive Risk Disclosures",
            "",
            "*Conditional on the constructed portfolio — only risks that materially apply are surfaced.*",
            "",
            "| Severity | Topic | Note |",
            "|----------|-------|------|",
        ]
        for _d in adaptive_disclaimers:
            _risk_lines.append(f"| [{_d['severity']}] | {_d['topic']} | {_d['note']} |")
        _risk_lines.append("")

    # Asset-class structural risks (only for classes actually held)
    _disclosure_rows: list[str] = []
    if "GCC" in _present_regions:
        _disclosure_rows.append("| GCC / KSA | Oil price sensitivity · Geopolitical risk · Currency peg risk |")
    if "Egypt" in _present_regions:
        _disclosure_rows.append("| Egypt | Currency devaluation risk · Political risk · High inflation |")
    if _has_emb:
        _disclosure_rows.append("| EM Bonds | Default risk · FX risk · Liquidity risk |")
    if "Gold" in _present_regions:
        _disclosure_rows.append("| Gold | No yield · Storage cost · USD-sensitive |")
    # Crypto, single-name, beta, FX are already covered by adaptive disclaimers
    # so we drop them from this table to avoid duplication.

    if _disclosure_rows:
        _risk_lines += [
            "### Structural Asset-Class Risk Factors",
            "",
            "| Asset Class | Risk Factor |",
            "|-------------|-------------|",
            *_disclosure_rows,
            "",
        ]
        if _has_non_usd:
            _risk_lines.append("> *Currency risk applies to non-USD holdings (SAR pegged, AED pegged, EGP floating).*")
            _risk_lines.append("")

    # Correlation cluster detection with severity tags
    _us_tech_w = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "US" and any(x in a.description.lower() for x in ("tech", "nasdaq", "semiconductor", "ai")))
    _crypto_w  = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "Crypto")
    _gcc_w     = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "GCC")
    _comm_w    = sum(w_opt[i] for i, a in enumerate(assets) if a.region == "Commodities")
    _cluster_rows: list[str] = []
    if _us_tech_w + _crypto_w > 0.35:
        _sev = "CRITICAL" if (_us_tech_w + _crypto_w) > 0.50 else "HIGH"
        _cluster_rows.append(f"| [{_sev}] | US Tech + Crypto | {(_us_tech_w+_crypto_w)*100:.1f}% | High correlation in risk-off episodes |")
    if _gcc_w + _comm_w > 0.35:
        _sev = "CRITICAL" if (_gcc_w + _comm_w) > 0.50 else "HIGH"
        _cluster_rows.append(f"| [{_sev}] | GCC + Commodities | {(_gcc_w+_comm_w)*100:.1f}% | Oil/commodity cycle co-movement |")

    if _cluster_rows:
        _risk_lines += [
            "### Correlation Cluster Risk",
            "",
            "| Severity | Cluster | Combined Weight | Note |",
            "|----------|---------|-----------------|------|",
            *_cluster_rows,
            "",
        ]

    # Crypto framework note (only if held, kept compact — taxonomic, no severity)
    if _crypto_w > 0:
        _risk_lines += [
            "### Crypto Analytical Framework",
            "",
            f"> Crypto positions ({_crypto_w*100:.0f}% of portfolio) are evaluated using a separate analytical lens:",
            "> *network activity · ETF flows · realized volatility · liquidity regime · cycle positioning.* "
            "Equity valuation multiples and earnings-quality metrics are not applicable.",
            "",
        ]

    # Semiconductor thematic cap (deterministic mandate enforcement, kept as audit note)
    _semi_w = sum(w_opt[i] for i, a in enumerate(assets) if any(x in (a.description or '').lower() for x in ("semiconductor", "chip", "nvidia", "amd", "intel")))
    if _semi_w > 0.20:
        _semi_sev = "CRITICAL" if _semi_w > 0.30 else "HIGH"
        _risk_lines += [
            f"### Thematic Concentration · Semiconductors · [{_semi_sev}]",
            "",
            f"> Semiconductor exposure of {_semi_w*100:.0f}% exceeds the 20% thematic cap. "
            "Single-theme risk above institutional concentration limit.",
            "",
        ]

    # Drawdown reality note (kept once, here)
    _risk_lines += [
        "### Drawdown Modeling Note",
        "",
        "> Modeled max drawdown is an estimate, not a guarantee. Crisis episodes (2008, 2020, 2022) "
        "frequently exceeded model predictions by 30–50%. Stress-test against −30%, −40%, and −50% scenarios "
        "before committing capital.",
        "",
    ]

    # Build report
    lines = [
        "## C. Risk Diagnostics",
        "",
        *_risk_lines,
        "## D. Allocation Logic",
        "",
        f"**Mandate:** {prof['label']} · {prof['description']}",
        "",
        "### Regional Allocation",
        "",
        "| Region | Weight | ~$ on $100k | Asset Classes |",
        "|--------|--------|-------------|---------------|",
    ]
    for r in sorted(region_alloc, key=lambda x: -region_alloc[x]):
        if region_alloc[r] < 0.1:
            continue
        r_assets = [assets[i].name for i in range(n) if assets[i].region == r and w_opt[i] > 0.005]
        r_usd    = region_alloc[r] / 100 * port_value_usd
        lines.append(f"| **{r}** | {region_alloc[r]:.1f}% | ${r_usd:,.0f} | {', '.join(r_assets) or '—'} |")

    # Per-asset rationale — explains WHY the asset sits in the construction
    def _asset_rationale(a) -> str:
        r = a.region
        desc = (a.description or "").lower()
        if r == "US":
            if "tech" in desc or "nasdaq" in desc:
                return "Long-duration growth sleeve · captures secular AI/tech earnings"
            if "mid-cap" in desc:
                return "Domestic mid-cap exposure · improves factor diversification beyond mega-cap"
            if "dividend" in desc or "value" in desc:
                return "Quality/value tilt · cash-flow stability and lower beta anchor"
            if "energy" in desc:
                return "Energy cyclicality · inflation hedge and macro pro-cyclical exposure"
            return "US equity core · liquid global benchmark proxy"
        if r == "GCC":
            return "Regional exposure · GCC growth premium, low correlation to US equities"
        if r == "Egypt":
            return "Frontier-market tilt · structural reform exposure (higher idiosyncratic risk)"
        if r == "Gold":
            return "Macro hedge · equity-duration compression, USD-weakening regimes"
        if r == "Bonds":
            if a.proxy == "TLT":
                return "Long-duration UST · negative correlation to equity in deflationary shocks"
            if a.proxy == "EMB":
                return "EM credit · spread carry with FX/default risk overlay"
            return "Fixed-income sleeve · duration and credit diversification"
        if r == "Cash":
            return "Dry powder · liquidity buffer and risk-free yield anchor"
        if r == "Crypto":
            return "Asymmetric satellite · high-volatility return contributor (not a hedge)"
        if r == "Commodities":
            return "Real-asset exposure · inflation pass-through and macro cyclicality"
        if r == "Diversification":
            return "Short-duration anchor · capital preservation with low rate risk"
        return "Diversifying exposure · added for factor / region balance"

    lines += [
        "",
        "### Optimal Asset Weights",
        "",
        "| Asset Class | Region | Weight | ~$ on $100k | Proxy | Role | Construction Rationale |",
        "|-------------|--------|--------|-------------|-------|------|------------------------|",
    ]
    for i in range(n):
        if w_opt[i] < 0.005:
            continue
        a   = assets[i]
        usd = w_opt[i] * port_value_usd
        _role = _classify_asset_role(a, float(w_opt[i]))
        lines.append(
            f"| **{a.name}** | {a.region} | {w_opt[i]*100:.1f}% | ${usd:,.0f} | `{a.proxy}` | {_role} | {_asset_rationale(a)} |"
        )

    # Diversification benefit summary (informational, stays in Section D)
    equal_w   = np.ones(n) / n
    eq_vol    = float(np.sqrt(equal_w @ cov @ equal_w))
    avg_vol   = float(np.mean([a.vol_annual for a in assets]))
    div_ratio = port_vol / avg_vol if avg_vol > 0 else 1
    lines += [
        "",
        "### Diversification Benefit",
        "",
        f"> **Diversification Ratio:** {1/div_ratio:.2f}x — portfolio vol ({port_vol*100:.1f}%) is "
        f"{(1-div_ratio)*100:.0f}% lower than weighted average of individual vols ({avg_vol*100:.1f}%)",
        f"> **vs Equal Weight:** Optimized vol {port_vol*100:.1f}% vs equal-weight {eq_vol*100:.1f}%",
        "",
    ]

    # Phase E — Benchmark-Relative Attribution (subsection of Section D)
    lines += [
        "### Benchmark-Relative Attribution",
        "",
        "*Decomposition of excess return vs profile-mapped benchmark — distinguishes factor-driven from selection-driven performance.*",
        "",
        "| Component | Value | Interpretation |",
        "|-----------|-------|----------------|",
        f"| Excess Return vs Benchmark | {attribution['excess_return_pct']:+.2f}% | Total active return |",
        f"| Beta Contribution | {attribution['beta_contribution_pct']:+.2f}% | Factor-driven (market sensitivity differential) |",
        f"| Residual Contribution | {attribution['residual_contribution_pct']:+.2f}% | Selection / concentration effects |",
        "",
        f"> *{attribution['interpretation']}*",
        "",
    ]

    # Phase E — institutional rounding note (always shown for transparency)
    lines += [
        f"> *Implementation note: {implementation.get('institutional_note', '')}*",
        "",
    ]

    report_md = "\n".join(lines)

    result = {
        "profile":        profile,
        "profile_label":  prof["label"],
        "region_alloc":   region_alloc,
        "asset_weights":  asset_weights,
        "asset_usd":      asset_usd,
        "weights":        asset_weights,   # canonical alias for Phase H engines
        "metrics": {
            "expected_return_pct": round(port_ret * 100, 2),
            "expected_vol_pct":    round(port_vol * 100, 2),
            "sharpe":              port_sharpe,
            "beta_world":          round(port_beta, 3),
            "rf_rate":             rf_rate,
            "profile":             profile,
        },
        "solver_status": prob.status,
        "report_md":     report_md,
        "feasibility":   "All constraints satisfied" if prob.status == "optimal" else "Approximate solution",
        "constraint_diagnostics": constraint_diagnostics,
        "rebalance_suggestions":  rebalance_suggestions,
        "audit":                  audit,
        "regime":                 regime,
        "confidence":             confidence,
        "adaptive_disclaimers":   adaptive_disclaimers,
        "asset_roles":            asset_roles,
        "implementation":         implementation,
        "benchmark":              benchmark,
        "attribution":            attribution,
        "asset_meta": {
            assets[i].name: {
                "proxy": assets[i].proxy,
                "region": assets[i].region,
                "vol": assets[i].vol_annual,
                "currency": assets[i].currency,
                "description": assets[i].description,
            }
            for i in range(n)
        },
    }

    # ── Phase H compute (no report injection here — report_md from
    # allocate() is partial; the assembled full A-G is built downstream
    # by portfolio_builder._run_allocator. We attach typed payloads only.)
    if _PHASE_H_AVAILABLE and _phase_h_augment is not None:
        try:
            result = _phase_h_augment(
                result,
                language=language,
                rebalance_frequency=rebalance_frequency,
                committee_mode=committee_mode,
                horizon_years=horizon_years,
                asset_kind=asset_kind,
                region_tilt=region_tilt,
                benchmark_ticker=benchmark_ticker,
                w_prev=w_prev,
                inject_into_report=False,
            )
            if _tc_optimizer_notes and isinstance(result.get("execution_diag"), dict):
                _diag = result["execution_diag"]
                _payload = _diag.get("payload") if isinstance(_diag.get("payload"), dict) else _diag
                for _target in (_diag, _payload):
                    _notes = list(_target.get("notes", []) or [])
                    for _note in _tc_optimizer_notes:
                        if _note not in _notes:
                            _notes.append(_note)
                    _target["notes"] = _notes
        except Exception as _ph_exc:
            logger.warning("phase_h augment failed (returning pre-H result): %r", _ph_exc)
            result.setdefault("_phase_h_errors", []).append(repr(_ph_exc))

    return result
