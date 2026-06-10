"""
sector_overrides.py — Ticker-level sector classification overrides.

TradingView's `sector` field is occasionally wrong for Gulf tickers — for
example DFM/ADX real-estate developers (EMAAR, EMAARDEV, ALDAR, DAMAC,
UPP, DEYAAR, ...) are tagged "Finance" instead of "Real Estate". When
the report routes off the wrong sector, the regional peer table fills
with banks, the risk taxonomy quotes oil-cycle boilerplate, and the
narrative loses its mandate fit.

This module owns two things:

1. SECTOR_OVERRIDES — bare-symbol → corrected `(sector_en, industry_en)`
   used by the post-processing cleanup layer in `api/routers/staging.py`.

2. REAL_ESTATE_PEER_UNIVERSE — Gulf real-estate ticker universe used to
   build a curated peer table when the upstream sector lookup returned
   banks. Symbols are TV exchange-prefixed (e.g. "DFM:EMAAR").

No engine code changes are required — these are read by the cleanup
layer when the subject ticker matches an override.
"""

from __future__ import annotations

from typing import Optional, Tuple


# Bare symbol (no exchange prefix, no .DU/.AE suffix) → (sector, industry)
SECTOR_OVERRIDES: dict[str, Tuple[str, str]] = {
    # UAE real-estate developers
    "EMAAR":         ("Real Estate", "Real Estate Development"),
    "EMAARDEV":      ("Real Estate", "Real Estate Development"),
    "ALDAR":         ("Real Estate", "Real Estate Development"),
    "DAMAC":         ("Real Estate", "Real Estate Development"),
    "DEYAAR":        ("Real Estate", "Real Estate Development"),
    "UPP":           ("Real Estate", "Real Estate Development"),
    "RAKPROPERTIES": ("Real Estate", "Real Estate Development"),
    "TECOM":         ("Real Estate", "Real Estate Operations"),
    "AMLAK":         ("Real Estate", "Real Estate Finance"),
    # Saudi real-estate developers (Tadawul real-estate codes 42xx)
    "4020":          ("Real Estate", "Real Estate Development"),
    "4090":          ("Real Estate", "Real Estate Development"),
    "4220":          ("Real Estate", "Real Estate Development"),
    "4230":          ("Real Estate", "Real Estate Development"),
    "4250":          ("Real Estate", "Real Estate Development"),
    "4280":          ("Real Estate", "Real Estate Development"),
    "4300":          ("Real Estate", "Real Estate Development"),
    "4310":          ("Real Estate", "Real Estate Development"),
    # Qatar real-estate developers
    "BRES":          ("Real Estate", "Real Estate Development"),
    "UDCD":          ("Real Estate", "Real Estate Development"),
    "MRDS":          ("Real Estate", "Real Estate Development"),
    "ERES":          ("Real Estate", "Real Estate Development"),
}


# Curated Gulf real-estate peer universe — used when subject ticker is
# real estate and we need to replace the bank-laden TV peer table.
# Each entry: (market_code_for_TV_cache, bare_symbol).
REAL_ESTATE_PEER_UNIVERSE: list[Tuple[str, str]] = [
    # UAE
    ("uae",   "EMAAR"),
    ("uae",   "EMAARDEV"),
    ("uae",   "ALDAR"),
    ("uae",   "DEYAAR"),
    ("uae",   "UPP"),
    # Saudi
    ("ksa",   "4020"),
    ("ksa",   "4090"),
    ("ksa",   "4220"),
    ("ksa",   "4280"),
    ("ksa",   "4300"),
    ("ksa",   "4310"),
    # Qatar
    ("qatar", "BRES"),
    ("qatar", "UDCD"),
    ("qatar", "ERES"),
]


def get_corrected_sector(ticker_or_bare: str) -> Optional[Tuple[str, str]]:
    """Return (sector, industry) override for the ticker, or None if none.

    Accepts either bare symbol ("EMAAR"), suffixed ("EMAAR.DU", "4020.SR"),
    or TV exchange-prefixed ("DFM:EMAAR").
    """
    if not ticker_or_bare:
        return None
    t = str(ticker_or_bare).upper().strip()
    # Strip exchange prefix
    if ":" in t:
        t = t.split(":", 1)[1]
    # Strip suffix
    t = t.split(".")[0]
    return SECTOR_OVERRIDES.get(t)


# Real-estate-specific risk taxonomy — used to replace any oil/commodity
# boilerplate the LLM may insert into the risk section when a real-estate
# ticker is misclassified upstream as Finance / Energy.
REAL_ESTATE_RISK_TAXONOMY_EN = """\
**Real-Estate Risk Profile**

- **Interest-Rate & Funding Cost Sensitivity** (Severity: High) — Higher rates compress mortgage demand, raise developer financing costs, and pressure valuation multiples on yield-sensitive assets.
- **Dubai Real-Estate Cycle / Supply-Demand Balance** (Severity: Medium-High) — Off-plan launches and new inventory waves can shift price momentum sharply; absorption risk grows when handover pipelines exceed market take-up.
- **Project Execution & Off-Plan Delivery Risk** (Severity: Medium) — Construction delays, cost inflation, and contractor reliability affect milestone-based revenue recognition.
- **Leverage / Refinancing Profile** (Severity: Medium) — Developer debt maturity ladder and access to revolving facilities determine resilience through cycle troughs.
- **Demand Concentration & Buyer Mix** (Severity: Medium) — Reliance on specific buyer pools (foreign retail, regional HNW, institutional) creates demand-side fragility if any cohort retreats.
- **Geopolitical Risk Premium** (Severity: Medium) — Regional tensions can shift risk-on / risk-off sentiment in Dubai property; impact is indirect via capital flows and tourism.
"""

REAL_ESTATE_RISK_TAXONOMY_AR = """\
**ملف مخاطر العقار**

- **حساسية أسعار الفائدة وتكلفة التمويل** (الشدة: عالية) — ارتفاع الفائدة يضغط الطلب على الرهن، يرفع تكلفة تمويل المطوّر، ويقلّص مضاعفات التقييم على الأصول الحساسة للعائد.
- **دورة عقار دبي / توازن العرض والطلب** (الشدة: عالية-متوسطة) — موجات الإطلاقات والمعروض الجديد يحوّل زخم الأسعار بحدّة؛ مخاطر الامتصاص ترتفع لو خط التسليمات تجاوز الاستيعاب.
- **مخاطر التنفيذ والتسليم off-plan** (الشدة: متوسطة) — تأخّر البناء، تضخّم التكلفة، وثقة المقاول تؤثّر على الاعتراف بالإيراد المرتبط بمراحل التنفيذ.
- **ملف الرفع المالي وإعادة التمويل** (الشدة: متوسطة) — سلّم استحقاقات دين المطوّر ووصوله للتسهيلات يحدّد مرونته في القاع الدوري.
- **تركّز الطلب ومزيج المشترين** (الشدة: متوسطة) — الاعتماد على فئات شراء محددة (تجزئة أجنبية، أثرياء إقليميون، مؤسسات) يخلق هشاشة لو انسحبت إحدى الفئات.
- **علاوة المخاطر الجيوسياسية** (الشدة: متوسطة) — التوترات الإقليمية تحوّل معنويات المخاطرة في عقار دبي عبر تدفقات رأس المال والسياحة.
"""


__all__ = [
    "SECTOR_OVERRIDES",
    "REAL_ESTATE_PEER_UNIVERSE",
    "get_corrected_sector",
    "REAL_ESTATE_RISK_TAXONOMY_EN",
    "REAL_ESTATE_RISK_TAXONOMY_AR",
]
