"""
Macro elasticity constants for EisaX.
Shared by macro_simulator, scenario_builder, and market_regimes.

Elasticity definition:
  sector_return_change% = elasticity * macro_variable_delta
  where delta = (user_input - baseline)

Oil elasticity: per $10/bbl change
USD elasticity: per 5 DXY points change
All others: per 1% change in the variable
"""

from __future__ import annotations

# ── Sector elasticities ────────────────────────────────────────────────────────
# Outer key: sector name (matches portfolio.py / yfinance taxonomy)
# Inner keys: gdp_growth, inflation, fed_rate, oil_brent, usd_index

SECTOR_ELASTICITIES: dict[str, dict[str, float]] = {
    "Energy Minerals": {
        "gdp_growth":  0.80,
        "inflation":   0.50,   # commodity hedge
        "fed_rate":   -0.40,
        "oil_brent":   0.15,   # per $10
        "usd_index":  -0.12,   # per 5 pts — commodities priced in USD
    },
    "Energy": {
        "gdp_growth":  0.70,
        "inflation":   0.40,
        "fed_rate":   -0.30,
        "oil_brent":   0.12,
        "usd_index":  -0.10,
    },
    "Finance": {
        "gdp_growth":  1.00,
        "inflation":  -0.50,   # margin compression from inflation
        "fed_rate":    0.30,   # NIM benefit for banks at moderate rates
        "oil_brent":  -0.03,
        "usd_index":   0.04,
    },
    "Real Estate": {
        "gdp_growth":  0.60,
        "inflation":   0.30,   # hard asset inflation hedge
        "fed_rate":   -0.80,   # most rate-sensitive sector
        "oil_brent":  -0.02,
        "usd_index":  -0.02,
    },
    "Technology Services": {
        "gdp_growth":  1.20,   # high-growth amplifier
        "inflation":  -0.60,   # PE compression from high inflation
        "fed_rate":   -0.70,   # long-duration equities
        "oil_brent":  -0.04,
        "usd_index":  -0.06,
    },
    "Consumer Cyclical": {
        "gdp_growth":  1.10,
        "inflation":  -0.40,
        "fed_rate":   -0.40,
        "oil_brent":  -0.05,
        "usd_index":  -0.03,
    },
    "Consumer Discretionary": {
        "gdp_growth":  1.10,
        "inflation":  -0.40,
        "fed_rate":   -0.40,
        "oil_brent":  -0.05,
        "usd_index":  -0.03,
    },
    "Consumer Defensive": {
        "gdp_growth":  0.20,
        "inflation":   0.10,
        "fed_rate":   -0.20,
        "oil_brent":  -0.03,
        "usd_index":  -0.02,
    },
    "Consumer Non-Cyclical": {
        "gdp_growth":  0.20,
        "inflation":   0.10,
        "fed_rate":   -0.20,
        "oil_brent":  -0.03,
        "usd_index":  -0.02,
    },
    "Healthcare": {
        "gdp_growth":  0.30,
        "inflation":  -0.20,
        "fed_rate":   -0.20,
        "oil_brent":  -0.01,
        "usd_index":  -0.03,
    },
    "Industrials": {
        "gdp_growth":  0.90,
        "inflation":  -0.10,
        "fed_rate":   -0.30,
        "oil_brent":  -0.04,
        "usd_index":  -0.04,
    },
    "Materials": {
        "gdp_growth":  0.80,
        "inflation":   0.30,
        "fed_rate":   -0.30,
        "oil_brent":   0.05,
        "usd_index":  -0.08,
    },
    "Basic Materials": {
        "gdp_growth":  0.80,
        "inflation":   0.30,
        "fed_rate":   -0.30,
        "oil_brent":   0.06,
        "usd_index":  -0.08,
    },
    "Utilities": {
        "gdp_growth":  0.10,
        "inflation":  -0.30,
        "fed_rate":   -0.60,   # high debt, rate-sensitive
        "oil_brent":  -0.02,
        "usd_index":   0.01,
    },
    "Communication Services": {
        "gdp_growth":  0.60,
        "inflation":  -0.30,
        "fed_rate":   -0.30,
        "oil_brent":  -0.02,
        "usd_index":  -0.04,
    },
    "Unknown": {
        "gdp_growth":  0.50,
        "inflation":  -0.10,
        "fed_rate":   -0.30,
        "oil_brent":  -0.02,
        "usd_index":  -0.03,
    },
}

# ── Baseline macro values (current conditions, May 2026) ─────────────────────
MACRO_VAR_DEFAULTS: dict[str, float] = {
    "gdp_growth":  2.3,    # % annualized global GDP growth
    "inflation":   3.0,    # % YoY CPI
    "fed_rate":    4.50,   # % fed funds rate
    "oil_brent":   75.0,   # USD per barrel
    "usd_index":  102.0,   # DXY index
}

# ── Slider UI ranges for dashboard ───────────────────────────────────────────
MACRO_VAR_RANGES: dict[str, tuple[float, float, float]] = {
    # variable: (min, max, step)
    "gdp_growth":  (-3.0,  6.0,  0.1),
    "inflation":   ( 0.0, 12.0,  0.1),
    "fed_rate":    ( 0.0,  9.0,  0.25),
    "oil_brent":   (30.0, 150.0, 5.0),
    "usd_index":   (85.0, 120.0, 1.0),
}

MACRO_VAR_LABELS: dict[str, tuple[str, str]] = {
    # variable: (Arabic label, English label)
    "gdp_growth":  ("نمو الناتج المحلي %", "GDP Growth %"),
    "inflation":   ("التضخم % (CPI)",       "Inflation % (CPI)"),
    "fed_rate":    ("سعر الفائدة الفيدرالي %", "Fed Rate %"),
    "oil_brent":   ("سعر النفط $/برميل",    "Brent Oil $/bbl"),
    "usd_index":   ("مؤشر الدولار (DXY)",   "USD Index (DXY)"),
}

# ── Market regime macro profiles ──────────────────────────────────────────────
# Characteristic macro environment for each regime
REGIME_MACRO_PROFILES: dict[str, dict[str, float]] = {
    "bull": {
        "gdp_growth":  4.0,
        "inflation":   2.0,
        "fed_rate":    2.5,
        "oil_brent":   85.0,
        "usd_index":   96.0,
    },
    "bear": {
        "gdp_growth": -1.5,
        "inflation":   4.5,
        "fed_rate":    6.0,
        "oil_brent":   55.0,
        "usd_index":  110.0,
    },
    "sideways": {
        "gdp_growth":  2.0,
        "inflation":   3.0,
        "fed_rate":    4.5,
        "oil_brent":   75.0,
        "usd_index":  102.0,
    },
    "stagflation": {
        "gdp_growth":  0.3,
        "inflation":   8.0,
        "fed_rate":    7.0,
        "oil_brent":  120.0,
        "usd_index":  105.0,
    },
}

REGIME_LABELS: dict[str, tuple[str, str]] = {
    "bull":        ("صعودي 🚀",      "Bull 🚀"),
    "bear":        ("هبوطي 🐻",      "Bear 🐻"),
    "sideways":    ("جانبي ➡️",     "Sideways ➡️"),
    "stagflation": ("ركود تضخمي 🔥", "Stagflation 🔥"),
}
