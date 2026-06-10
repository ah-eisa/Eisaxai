"""
fixed_income.py — EisaX Sukuk & Bond Analysis Engine
=====================================================
Analyzes fixed-income instruments by ISIN using:

Data sources (priority order):
  1. OpenFIGI API  — instrument metadata (free, no key for basic requests)
     POST https://api.openfigi.com/v3/mapping
  2. FMP API       — coupon/maturity details (if FMP_API_KEY is set)
  3. FRED API      — US Treasury & benchmark sovereign yields
  4. open.er-api   — FX rates (no auth)
  5. worldgovernmentbonds.com — sovereign yield context (existing engine)

Report sections:
  - Instrument Overview
  - Yield Analysis vs benchmarks
  - Credit Risk Assessment
  - Sukuk Structure (Ijara/Murabaha/Mudarabah/Wakala)
  - FX Risk
  - Liquidity Assessment
  - EisaX Fixed Income Score (0-100)

Usage (from finance.py):
    from core.fixed_income import is_fixed_income_query, extract_isin, get_instrument_data, compute_fi_score, format_fi_for_prompt
"""

from __future__ import annotations

import os
import re
import time
import logging
import requests
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── In-process TTL cache (1h — instrument metadata is stable) ─────────────────
_cache: Dict = {}
_CACHE_TTL = 3600

# ── ISIN Regex ────────────────────────────────────────────────────────────────
# Format: 2-letter country code + 9 alphanumeric NSIN + 1 check digit = 12 chars
ISIN_RE = re.compile(r'\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b')

# ISIN country prefixes we handle (GCC, MENA, international)
VALID_ISIN_PREFIXES = {
    "XS",   # Euroclear / Clearstream (international Eurobonds/Sukuk)
    "XD",   # ANNA DSB-registered OTC
    "US",   # United States
    "GB",   # United Kingdom
    "AE",   # UAE
    "SA",   # Saudi Arabia
    "KW",   # Kuwait
    "QA",   # Qatar
    "BH",   # Bahrain
    "OM",   # Oman
    "EG",   # Egypt
    "JO",   # Jordan
    "TR",   # Turkey
    "MY",   # Malaysia (major Sukuk market)
    "ID",   # Indonesia
    "DE",   # Germany
    "FR",   # France
    "NL",   # Netherlands
    "CH",   # Switzerland
    "AU",   # Australia
    "JP",   # Japan
    "SG",   # Singapore
}

# ── Country ratings (sovereign, approximate) ──────────────────────────────────
# Used for credit scoring when no explicit rating is available
_COUNTRY_RATINGS: Dict[str, Tuple[str, int]] = {
    # (rating_label, credit_score 0-30)
    "US":  ("AAA",  30),  "GB":  ("AA",   28),  "DE":  ("AAA",  30),
    "FR":  ("AA",   28),  "NL":  ("AAA",  30),  "AU":  ("AAA",  30),
    "JP":  ("A+",   25),  "SG":  ("AAA",  30),  "CH":  ("AAA",  30),
    "AE":  ("AA",   28),  "SA":  ("A1",   25),  "QA":  ("AA-",  27),
    "KW":  ("AA-",  27),  "BH":  ("B+",    8),  "OM":  ("BB",   12),
    "MY":  ("A3",   22),  "ID":  ("BBB",  18),  "TR":  ("B",     9),
    "EG":  ("B",     9),  "JO":  ("BB-",  11),  "MA":  ("BB+",  13),
    "PK":  ("Caa3",  3),  "LK":  ("SD",    1),  "NG":  ("B-",    7),
    "ZA":  ("BB-",  11),  "NG":  ("B-",    7),  "KE":  ("B",     8),
    "NG":  ("B-",    7),  "GH":  ("CC",    2),  "CI":  ("BB-",  11),
    "XS":  ("--",   15),  # International — unknown, assume IG mid
}

# Worldgovernmentbonds.com country data: cc → (url_page_slug, wgb_symbol_id)
# symbol_id confirmed by fetching the country page; "" = fetch dynamically
_WGB_COUNTRY_DATA: Dict[str, Tuple[str, str]] = {
    "US": ("united-states",        ""),
    "GB": ("united-kingdom",       ""),
    "DE": ("germany",              ""),
    "FR": ("france",               ""),
    "JP": ("japan",                ""),
    "AU": ("australia",            ""),
    "SG": ("singapore",            ""),
    "AE": ("united-arab-emirates", ""),
    "SA": ("saudi-arabia",         ""),
    "QA": ("qatar",                ""),
    "KW": ("kuwait",               ""),
    "BH": ("bahrain",              ""),
    "OM": ("oman",                 ""),
    "EG": ("egypt",                "30"),
    "TR": ("turkey",               "13"),
    "PK": ("pakistan",             "48"),
    "MY": ("malaysia",             ""),
    "ID": ("indonesia",            "39"),
    "JO": ("jordan",               ""),
    "MA": ("morocco",              ""),
    "ZA": ("south-africa",         ""),
    "NG": ("nigeria",              ""),
    "GH": ("ghana",                ""),
    "KE": ("kenya",                ""),
}
# Legacy mapping for backward compat
_WGB_COUNTRY_SLUGS: Dict[str, str] = {cc: v[0] for cc, v in _WGB_COUNTRY_DATA.items()}

# ── Sukuk structure keywords ───────────────────────────────────────────────────
_SUKUK_STRUCTURES = {
    "IJARA":    "Ijara (Lease-based) — asset-backed lease payments",
    "IJARAH":   "Ijara (Lease-based) — asset-backed lease payments",
    "MURABAHA": "Murabaha (Cost-plus sale) — deferred payment structure",
    "MUDARABAH":"Mudarabah (Profit-sharing) — equity-style risk sharing",
    "MUSHARAKA":"Musharaka (Partnership) — joint venture structure",
    "WAKALA":   "Wakala (Agency-based) — fund invested on agency basis",
    "HYBRID":   "Hybrid Multi-tranche Sukuk",
    "TRUST CERT": "Trust Certificates (generic Sukuk wrapper)",
}

# ── HTTP headers for scraping ──────────────────────────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# ── ISIN detection helpers ────────────────────────────────────────────────────

def extract_isin(text: str) -> Optional[str]:
    """Extract first valid ISIN from text. Returns None if not found."""
    if not text:
        return None
    for m in ISIN_RE.finditer(text.upper()):
        isin = m.group(1)
        # Require recognisable 2-letter prefix
        if len(isin) == 12:
            return isin
    return None


# ── Keyword sets ──────────────────────────────────────────────────────────────
FIXED_INCOME_KEYWORDS_EN = [
    "sukuk", "bond", "isin", "fixed income", "coupon", "maturity",
    "yield to maturity", "ytm", "duration", "credit rating", "spread",
    "trust certificate", "eurobond", "t-bill", "treasury bill",
    "sovereign bond", "corporate bond", "convertible bond",
    "sukuks", "islamic bond", "fixed-income",
]

FIXED_INCOME_KEYWORDS_AR = [
    "صكوك", "صك", "سندات", "سند", "دخل ثابت", "عائد ثابت",
    "كوبون", "استحقاق", "سندات إسلامية", "تورق", "إجارة",
    "مرابحة", "مضاربة", "وكالة", "شهادات استثمار",
    "سندات حكومية", "سندات شركات",
]

# Patterns that might indicate an ISIN even without the user labeling it
_ISIN_CONTEXT_WORDS = ["isin", "bond", "sukuk", "instrument", "securities", "note"]


def is_fixed_income_query(message: str) -> bool:
    """Return True if the message is asking about bonds/sukuk or contains an ISIN."""
    if not message:
        return False
    low = message.lower()

    # Direct ISIN pattern match
    if extract_isin(message):
        return True

    # English keywords
    for kw in FIXED_INCOME_KEYWORDS_EN:
        if kw in low:
            return True

    # Arabic keywords
    for kw in FIXED_INCOME_KEYWORDS_AR:
        if kw in message:
            return True

    return False


# ── Source 1: OpenFIGI ─────────────────────────────────────────────────────────

def _validate_isin(isin: str) -> bool:
    """
    Validate ISIN check digit using the modified Luhn algorithm.
    Returns True if valid, False otherwise.
    """
    if not isin or len(isin) != 12:
        return False
    isin = isin.upper()
    if not re.match(r'^[A-Z]{2}[A-Z0-9]{10}$', isin):
        return False
    # Convert letters to digits (A=10 ... Z=35)
    digits_str = ""
    for ch in isin:
        if ch.isdigit():
            digits_str += ch
        else:
            digits_str += str(ord(ch) - 55)  # A→10, B→11, ..., Z→35
    # Luhn check
    total = 0
    n = len(digits_str)
    for i, ch in enumerate(reversed(digits_str)):
        d = int(ch)
        if i % 2 == 1:   # double every second from right (0-indexed)
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0

def _infer_country_code(text: str) -> Optional[str]:
    """
    Infer ISO 2-letter country code from free text (issuer name, Serper snippets).
    Essential for XS-prefix international bonds where ISIN doesn't encode country.
    Returns 2-letter code or None.
    """
    text_lower = text.lower()
    # Ordered by specificity (longer/more specific patterns first)
    _COUNTRY_HINTS = [
        # Gulf / MENA
        ("AE", ["united arab emirates", "uae", "abu dhabi", "dubai", "adnoc",
                "emaar", "dewa", "mubadala", "aldar", "fab ", "emirates nbd",
                "إمارات", "دبي", "أبوظبي"]),
        ("SA", ["saudi arabia", "saudi", "kingdom of saudi", "aramco", "sabic",
                "ncb", "al rajhi", "riyad bank", "sama", "samba",
                "السعودية", "أرامكو", "الراجحي"]),
        ("QA", ["qatar", "doha", "qnb", "qatarenergy", "rasgas", "qatargas",
                "قطر", "الدوحة"]),
        ("KW", ["kuwait", "knpc", "kfh", "nbk", "boubyan",
                "الكويت"]),
        ("BH", ["bahrain", "batelco", "nbob",
                "البحرين"]),
        ("OM", ["oman", "muscat", "oq", "bank muscat",
                "عُمان", "مسقط"]),
        ("PK", ["pakistan", "government of pakistan", "islamic republic of pakistan",
                "باكستان"]),
        ("EG", ["egypt", "egyptian", "nbe", "cib egypt", "telecom egypt",
                "مصر", "مصرية"]),
        ("TR", ["turkey", "türkiye", "republic of turkey", "garanti", "akbank",
                "تركيا"]),
        ("MY", ["malaysia", "maybank", "cimb", "petronas",
                "ماليزيا"]),
        ("ID", ["indonesia", "republic of indonesia", "pertamina", "bank mandiri",
                "إندونيسيا"]),
        ("JO", ["jordan", "hashemite kingdom", "arab bank jordan",
                "الأردن"]),
        # Developed
        ("US", ["united states", "treasury", "u.s. government", "federal reserve"]),
        ("GB", ["united kingdom", "uk gilt", "his majesty's treasury", "bank of england"]),
        ("DE", ["germany", "federal republic of germany", "deutschland", "bund"]),
    ]

    for cc, hints in _COUNTRY_HINTS:
        if any(h in text_lower for h in hints):
            return cc
    return None


def detect_sukuk_query_language(message: str) -> str:
    """Return 'ar' if the message is primarily Arabic, else 'en'."""
    arabic_chars = sum(1 for c in message if "\u0600" <= c <= "\u06FF")
    return "ar" if arabic_chars > len(message) * 0.3 else "en"
