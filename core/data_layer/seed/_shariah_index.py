"""
core.data_layer.seed._shariah_index — canonical Shariah-compliance reference.

Source policy (gcc_ingestion_spec.md §2): every ticker listed here MUST
appear in at least one of these published Shariah indices:

    - S&P Saudi Arabia Shariah Index
    - MSCI Saudi Arabia Domestic Shariah Index
    - FTSE Saudi Shariah Index
    - S&P GCC Shariah Index
    - Dow Jones Islamic Market Index (regional sleeves)
    - Tadawul-published "Sharia-compliant" classification (where applicable)

Banks here are explicitly the Islamic-banking institutions whose
business models are Shariah-compliant by charter. Conventional banks
(SNB, FAB, QNB, NBK …) are **deliberately not flagged** even when
individual instruments comply, because business-line classification is
the institutional-grade signal.

Every entry carries a source citation so the audit trail can be
reconstructed. Reviewer-driven additions go through the same pattern.

`SHARIAH_REFERENCE` keys are exchange-prefixed canonical tickers
(EXCHANGE:SYMBOL). The provenance produced from this table is always
Tier-2 (`source_type="derived"`, `data_quality="derived"`).
"""

from __future__ import annotations

from typing import Dict, Tuple


# ticker → (display_index_name, source_url_id, confidence)
SHARIAH_REFERENCE: Dict[str, Tuple[str, str, float]] = {

    # ── KSA — Islamic banks (Shariah-compliant by charter) ──────────
    "TADAWUL:1120": ("S&P Saudi Arabia Shariah Index", "spdji_saudi_shariah_2024_constituents", 0.95),  # Al Rajhi
    "TADAWUL:1140": ("S&P Saudi Arabia Shariah Index", "spdji_saudi_shariah_2024_constituents", 0.90),  # Bank Albilad
    "TADAWUL:1150": ("S&P Saudi Arabia Shariah Index", "spdji_saudi_shariah_2024_constituents", 0.90),  # Alinma Bank

    # ── KSA — Industrials / Energy / Materials commonly screened in ──
    "TADAWUL:2222": ("S&P Saudi Arabia Shariah Index", "spdji_saudi_shariah_2024_constituents", 0.85),  # Aramco
    "TADAWUL:2010": ("S&P Saudi Arabia Shariah Index", "spdji_saudi_shariah_2024_constituents", 0.85),  # SABIC
    "TADAWUL:1211": ("S&P Saudi Arabia Shariah Index", "spdji_saudi_shariah_2024_constituents", 0.85),  # Ma'aden
    "TADAWUL:2280": ("S&P Saudi Arabia Shariah Index", "spdji_saudi_shariah_2024_constituents", 0.85),  # Almarai
    "TADAWUL:7010": ("S&P Saudi Arabia Shariah Index", "spdji_saudi_shariah_2024_constituents", 0.85),  # STC
    "TADAWUL:7020": ("S&P Saudi Arabia Shariah Index", "spdji_saudi_shariah_2024_constituents", 0.85),  # Mobily
    "TADAWUL:5110": ("S&P Saudi Arabia Shariah Index", "spdji_saudi_shariah_2024_constituents", 0.85),  # Saudi Electricity
    "TADAWUL:2082": ("S&P Saudi Arabia Shariah Index", "spdji_saudi_shariah_2024_constituents", 0.85),  # ACWA Power
    "TADAWUL:4190": ("S&P Saudi Arabia Shariah Index", "spdji_saudi_shariah_2024_constituents", 0.80),  # Jarir

    # ── UAE — Islamic banks ──────────────────────────────────────────
    "ADX:ADIB":   ("S&P GCC Shariah Index", "spdji_gcc_shariah_2024_constituents", 0.95),  # Abu Dhabi Islamic Bank
    "DFM:DIB":    ("S&P GCC Shariah Index", "spdji_gcc_shariah_2024_constituents", 0.95),  # Dubai Islamic Bank

    # ── UAE — Energy / Utilities commonly screened in ───────────────
    "ADX:ADNOCGAS":   ("S&P GCC Shariah Index", "spdji_gcc_shariah_2024_constituents", 0.85),
    "ADX:ADNOCDIST":  ("S&P GCC Shariah Index", "spdji_gcc_shariah_2024_constituents", 0.85),
    "ADX:ADNOCDRILL": ("S&P GCC Shariah Index", "spdji_gcc_shariah_2024_constituents", 0.85),
    "ADX:TAQA":   ("S&P GCC Shariah Index", "spdji_gcc_shariah_2024_constituents", 0.80),
    "DFM:DEWA":   ("S&P GCC Shariah Index", "spdji_gcc_shariah_2024_constituents", 0.80),
    "DFM:EMAAR":  ("S&P GCC Shariah Index", "spdji_gcc_shariah_2024_constituents", 0.80),

    # ── Qatar — Islamic banks ────────────────────────────────────────
    "QSE:QIBK":   ("S&P GCC Shariah Index", "spdji_gcc_shariah_2024_constituents", 0.95),  # Qatar Islamic Bank

    # ── Qatar — Materials commonly screened in ──────────────────────
    "QSE:IQCD":   ("S&P GCC Shariah Index", "spdji_gcc_shariah_2024_constituents", 0.85),

    # ── Kuwait — Islamic banks ───────────────────────────────────────
    "KSE:KFH":    ("S&P GCC Shariah Index", "spdji_gcc_shariah_2024_constituents", 0.95),  # Kuwait Finance House
}


def is_shariah_listed(ticker: str) -> bool:
    """Membership predicate used by tests + report layer."""
    return (ticker or "").upper() in SHARIAH_REFERENCE


def shariah_provenance(ticker: str) -> Tuple[str, str, float] | None:
    """(display_index, document_id, confidence) when known, else None."""
    return SHARIAH_REFERENCE.get((ticker or "").upper())


__all__ = ["SHARIAH_REFERENCE", "is_shariah_listed", "shariah_provenance"]
