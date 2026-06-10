from __future__ import annotations
import logging
import json
import re
import time
from typing import Any, List, Optional, Dict
import pandas as pd
import config
from core.llm import get_client
from core.data import get_prices, to_returns
from core.data import get_prices, to_returns
from core.policy import apply_policy
from core.metrics import perf_metrics
from state import SYSTEM_PROMPTS
logger = logging.getLogger(__name__)

# ============================================================
# PARSING HELPERS
# ============================================================
def _kv(text: str, key: str) -> str | None:
    m = re.search(rf"{key}\s*=\s*([^\s]+)", text, flags=re.IGNORECASE)
    return m.group(1) if m else None

def get_param(mem: dict[str, Any], msg: str, name: str, default: Any) -> Any:
    v = _kv(msg, name)
    if v is None:
        return mem.get(name, default)
    return v

def parse_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def parse_int(x: Any, default: int) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)

def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def _fmt_float(x: float) -> str:
    return f"{x:.4f}"

ETF_NAMES = {
    "QQQ": "US Tech Stocks",
    "SPY": "S&P 500",
    "IVV": "S&P 500",
    "VOO": "S&P 500",
    "VTI": "Total US Market",
    "EEM": "Emerging Markets",
    "VEA": "Developed International",
    "GLD": "Gold",
    "IAU": "Gold",
    "SLV": "Silver",
    "BND": "Total Bond Market",
    "AGG": "Total Bond Market",
    "TLT": "20+ Year Treasuries",
    "IEF": "7-10 Year Treasuries",
    "SHY": "1-3 Year Treasuries",
    "IBIT": "Bitcoin ETF",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "USMV": "Low Volatility Stocks",
    "SCHD": "High Dividend Stocks",
    "XLE": "Energy Sector",
    "XLF": "Financial Sector",
    "XLK": "Tech Sector",
    "XLV": "Healthcare Sector",
    "VNQ": "Real Estate (REITs)"
}

def render_weights(weights: dict[str, float]) -> str:
    w_sorted = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    lines = []
    for k, v in w_sorted:
        name = ETF_NAMES.get(k, k)
        label = f"{k} ({name})" if name != k else k
        lines.append(f"{label}: {_fmt_pct(v)}")
    return "\n".join(lines)

# ============================================================
# LOGIC
# ============================================================
def detect_risk_pref(text: str) -> str | None:
    low = (text or "").lower()
    # English
    if any(k in low for k in ["low risk", "conservative", "safer", "less risk", "lower risk", "min volatility", "min vol"]):
        return "low"
    if any(k in low for k in ["high risk", "aggressive", "more risk", "higher risk"]):
        return "high"
    # Arabic — low risk
    if any(k in low for k in ["مخاطرة قليلة", "اقل مخاطرة", "محافظ", "آمن", "اقل تذبذب",
                               "قلل المخاطرة", "منخفض المخاطره", "مخاطره منخفضه", "امان"]):
        return "low"
    # Arabic — high risk (مخاطره/مخاطرة alone + aggressive signals)
    if any(k in low for k in ["مخاطرة عالية", "اعلى مخاطرة", "هجومي", "زوّد المخاطرة",
                               "مخاطره عاليه", "مخاطره كبيره", "مخاطرة كبيرة",
                               "اعلى مكسب", "اقصى ربح", "اقصى عائد", "اقصى مكسب",
                               "مخاطره", "مخاطرة"]):
        return "high"
    # English bonus signals
    if any(k in low for k in ["maximum return", "max profit", "max gain", "max return"]):
        return "high"
    return None

def recommend_etfs(risk: str | None) -> list[str]:
    """
    Returns a list of ETFs representing broad asset classes based on risk profile.
    """
    if risk == "high":
        # Tech (QQQ), S&P500 (SPY), Emerging (EEM), Gold (GLD), Bitcoin (IBIT)
        return ["QQQ", "SPY", "EEM", "GLD", "IBIT"]
    elif risk == "low":
        # Bonds (BND), Dividend (SCHD), Gold (GLD), Low Vol (USMV)
        return ["BND", "SCHD", "GLD", "USMV"]
    else:
        # Balanced: VTI (Total US), BND (Bonds), VEA (Dev Markets), GLD (Gold)
        return ["VTI", "BND", "VEA", "GLD"]

def method_from_risk(risk: str | None) -> str | None:
    if risk == "low":
        return "min_vol"
    if risk == "high":
        return "max_sharpe"
    return None

# ── Ticker normalization map ──────────────────────────────────────────────────
# Maps user-typed or LLM-generated fake/invalid tickers → real yfinance tickers.
# "UAE" is not a valid yfinance ticker; "SAUDI"/"ARABIA" are completely fake.
_TICKER_MAP: dict[str, str | None] = {
    # GCC region — map to real ETFs
    "UAE":      "GULF",     # WisdomTree Middle East Dividend Fund
    "SAUDI":    "KSA",      # iShares MSCI Saudi Arabia ETF
    "ARABIA":   "KSA",
    "GCC":      "KSA",
    "QATAR":    "QAT",      # iShares MSCI Qatar ETF
    "KUWAIT":   "KSA",      # no dedicated ETF — use KSA as proxy
    "BAHRAIN":  "KSA",
    "OMAN":     "KSA",
    "EGYPT":    "EGPT",     # VanEck Egypt Index ETF
    # Common typos / bad LLM outputs
    "ARAMCO":   "2222.SR",
    "SABIC":    "2010.SR",
    "STC":      "7010.SR",
    "ZAINKSA":  "7030.SR",
    "EMAAR":    "EMAAR.DU",
    "ADNOC":    "ADNOCDIST.AE",
    "FAB":      "FAB.AE",
    "ADCB":     "ADCB.AE",
}

