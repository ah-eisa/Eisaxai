"""
FactSheet — Single Source of Truth (SSOT) for report facts.

Gathers authoritative values from the highest-trust sources:
    • Price / SMA50 / SMA200 / RSI / PE / Market Cap → TradingView pipeline cache
    • Sector classification                          → sector_overrides + TV
    • Verdict / Score / Action / Confidence          → DecisionState
    • Fair Value                                     → analytics_enricher
    • LLM body                                       → NEVER authoritative (diagnostics only)

Guardrails (per architecture review):
    1. No LLM body as authority. DecisionState/TV cache only.
    2. No currency fallback to "$" for MENA tickers — blocks instead.
    3. TV is authoritative for GCC + EGX + MENA (not just GCC).
    4. Sector subtypes (bank, real_estate_developer, energy_producer, ...)
       drive peer/risk/news routing.
    5. Validation rules raise blocking_errors that downstream MUST honor.

Consumed by:
    • core/services/report_reconciler.py — uses FactSheet to swap any
      disagreeing number in the LLM body with the SSOT value.
    • api/routers/staging.py — calls reconciler in the response shaper.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger("eisax.fact_sheet")


# ── Enums ─────────────────────────────────────────────────────────────────
class SectorSubtype(str, Enum):
    REAL_ESTATE_DEVELOPER = "real_estate_developer"
    REAL_ESTATE_OPERATIONS = "real_estate_operations"
    REAL_ESTATE_FINANCE = "real_estate_finance"
    BANK = "bank"
    INSURANCE = "insurance"
    ENERGY_PRODUCER = "energy_producer"
    ENERGY_INTEGRATED = "energy_integrated"
    GAS_LNG = "gas_lng"
    PETROCHEMICAL = "petrochemical"
    TECHNOLOGY = "technology"
    INDUSTRIAL = "industrial"
    CONSUMER = "consumer"
    UTILITIES = "utilities"
    HEALTHCARE = "healthcare"
    BASIC_MATERIALS = "basic_materials"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    UNKNOWN = "unknown"


# ── Constants / Maps ──────────────────────────────────────────────────────
_CURRENCY_BY_SUFFIX = {
    ".AE":  ("د.إ", "AED"),
    ".DU":  ("د.إ", "AED"),
    ".SR":  ("﷼",   "SAR"),
    ".CA":  ("ج.م", "EGP"),
    ".CAI": ("ج.م", "EGP"),
    ".KW":  ("KD",  "KWD"),
    ".QA":  ("ر.ق", "QAR"),
    ".BH":  ("BD",  "BHD"),
    ".OM":  ("OMR", "OMR"),
    ".MA":  ("DH",  "MAD"),
    ".TN":  ("TND", "TND"),
    "=F":   ("$",   "USD"),
    "-USD": ("$",   "USD"),    # Crypto pairs (BTC-USD, ETH-USD…)
}

# Extended: TV authoritative for ALL MENA + GCC + US, not just GCC
_MARKET_BY_SUFFIX = {
    ".AE": "uae", ".DU": "uae",
    ".SR": "ksa",
    ".CA": "egypt", ".CAI": "egypt",
    ".KW": "kuwait", ".QA": "qatar", ".BH": "bahrain",
    ".OM": "oman", ".MA": "morocco", ".TN": "tunisia",
    "=F": "commodities",
    "-USD": "crypto",
}

_GCC_MARKETS = {"uae", "ksa", "qatar", "kuwait", "bahrain", "oman"}
_MENA_MARKETS = _GCC_MARKETS | {"egypt", "morocco", "tunisia"}

# Bare crypto symbols — entity_resolution strips the "-USD" pair suffix
# (turning BTC-USD → BTC), so by the time FactSheet sees the ticker, the
# market-by-suffix lookup misses. Treat these as crypto when no other
# market suffix matches.
_CRYPTO_BARE_SYMBOLS = {
    "BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "AVAX", "DOT",
    "MATIC", "LINK", "BNB", "LTC", "BCH", "TRX", "ATOM", "NEAR",
    "ARB", "OP", "APT", "SUI",
}

_SECTOR_KEYWORD_SUBTYPE: list[tuple[str, SectorSubtype]] = [
    # Order matters — more specific first
    (r"real\s+estate.*develop",     SectorSubtype.REAL_ESTATE_DEVELOPER),
    (r"real\s+estate.*operation",   SectorSubtype.REAL_ESTATE_OPERATIONS),
    (r"real\s+estate.*finance",     SectorSubtype.REAL_ESTATE_FINANCE),
    (r"real\s+estate|reit",         SectorSubtype.REAL_ESTATE_DEVELOPER),
    (r"\bgas\b|\bLNG\b|liquefied",  SectorSubtype.GAS_LNG),
    (r"petrochemical",              SectorSubtype.PETROCHEMICAL),
    (r"integrated.*oil|integrated.*gas|oil.*gas.*integrated",
                                    SectorSubtype.ENERGY_INTEGRATED),
    (r"energy|petroleum|hydrocarb|crude|oil(?:\s|$)",
                                    SectorSubtype.ENERGY_PRODUCER),
    (r"commercial\s+bank|retail\s+bank|investment\s+bank|\bbank\b|banking",
                                    SectorSubtype.BANK),
    (r"insurance|reinsurance|takaful",
                                    SectorSubtype.INSURANCE),
    (r"technology|software|internet|semicond",
                                    SectorSubtype.TECHNOLOGY),
    (r"industrial|industrials|manufactur",
                                    SectorSubtype.INDUSTRIAL),
    (r"consumer",                   SectorSubtype.CONSUMER),
    (r"utilit",                     SectorSubtype.UTILITIES),
    (r"healthcare|pharma|biotech",  SectorSubtype.HEALTHCARE),
    (r"materials|mining|chemicals", SectorSubtype.BASIC_MATERIALS),
    # Commodity / crypto detectors (sector text from TV cache or Yahoo)
    (r"commodit|precious\s+metal|bullion|\bgold\b|\bsilver\b|grain|crude\s+oil\s+futures",
                                    SectorSubtype.COMMODITY),
    (r"crypto|cryptocurrenc|digital\s+asset|bitcoin|ethereum|blockchain",
                                    SectorSubtype.CRYPTO),
]

# Hard-coded subtype overrides for tickers TV/engine misclassify
_TICKER_SUBTYPE_OVERRIDE: dict[str, SectorSubtype] = {
    # Egyptian banks (EGX often tagged "Finance" / "Banks")
    "COMI": SectorSubtype.BANK,
    "HRHO": SectorSubtype.BANK,
    "CIEB": SectorSubtype.BANK,
    "FAITA": SectorSubtype.BANK,
    "ADIB": SectorSubtype.BANK,
    "QNBE": SectorSubtype.BANK,
    "CIB":  SectorSubtype.BANK,
    # UAE/Saudi banks
    "EMIRATESNBD": SectorSubtype.BANK,
    "FAB":  SectorSubtype.BANK,
    "ADCB": SectorSubtype.BANK,
    "DIB":  SectorSubtype.BANK,
    "ENBD": SectorSubtype.BANK,
    "1010": SectorSubtype.BANK,  # Riyad Bank
    "1020": SectorSubtype.BANK,  # Bank Aljazira
    "1050": SectorSubtype.BANK,  # BSF
    "1120": SectorSubtype.BANK,  # Al Rajhi
    "1180": SectorSubtype.BANK,  # SNB
    # Qatar banks
    "QNBK": SectorSubtype.BANK,
}

_RE_SUBTYPES = {
    SectorSubtype.REAL_ESTATE_DEVELOPER,
    SectorSubtype.REAL_ESTATE_OPERATIONS,
    SectorSubtype.REAL_ESTATE_FINANCE,
}
_ENERGY_SUBTYPES = {
    SectorSubtype.ENERGY_PRODUCER,
    SectorSubtype.ENERGY_INTEGRATED,
    SectorSubtype.GAS_LNG,
    SectorSubtype.PETROCHEMICAL,
}
_FINANCIAL_SUBTYPES = {SectorSubtype.BANK, SectorSubtype.INSURANCE}

# News-filter profiles per subtype: required = at least one must match;
# excluded = none may match. Used to drop unrelated wire items.
_NEWS_PROFILES: dict[SectorSubtype, dict[str, list[str]]] = {
    SectorSubtype.GAS_LNG: {
        "required": [
            r"LNG", r"crude", r"Brent", r"OPEC", r"hydrocarbon",
            r"pipeline", r"Hormuz",
            r"gas\s+(?:price|market|production|export|supply|contract|demand)",
            r"oil\s+(?:price|market|disruption|sanctions?|spike|drop|fall|export|production)",
            r"energy\s+(?:price|market|disruption|sanctions?|sector|policy)",
        ],
        "excluded": [],
    },
    SectorSubtype.ENERGY_PRODUCER: {
        "required": [
            r"crude", r"Brent", r"OPEC", r"hydrocarbon", r"refinery",
            r"oil\s+(?:price|market|disruption|sanctions?|spike|drop|fall|export|production)",
            r"energy\s+(?:price|market|disruption|sanctions?|sector|policy)",
        ],
        "excluded": [],
    },
    SectorSubtype.PETROCHEMICAL: {
        "required": [
            r"petrochemical", r"chemical", r"polymer", r"plastic",
            r"crude", r"naphtha", r"feedstock", r"olefin", r"ethylene",
        ],
        "excluded": [],
    },
    SectorSubtype.REAL_ESTATE_DEVELOPER: {
        "required": [
            r"real\s+estate", r"property", r"housing", r"developer",
            r"residential", r"commercial\s+property", r"mortgage",
            r"off[- ]?plan", r"land\s+(?:sales?|deal)", r"construction",
            r"infrastructure",
        ],
        "excluded": [
            r"\boil\b", r"\bgas\b", r"\bcrude\b", r"Brent", r"OPEC",
            r"\bLNG\b", r"petrochemical", r"refinery",
        ],
    },
    SectorSubtype.BANK: {
        "required": [
            r"\bbank", r"banking", r"interest\s+rate", r"central\s+bank",
            r"\bcredit\b", r"\bloan", r"deposit", r"\bNIM\b", r"net\s+interest",
            r"CBE|SAMA|CBUAE|CBK|QCB|CBO",  # MENA central banks
            r"basel", r"capital\s+adequacy",
        ],
        "excluded": [
            r"\boil\b", r"\bgas\b", r"\bcrude\b", r"Brent", r"OPEC", r"\bLNG\b",
        ],
    },
    SectorSubtype.INSURANCE: {
        "required": [
            r"insurance", r"reinsur", r"takaful", r"premium\s+(?:growth|written)",
            r"underwriting", r"claim", r"actuarial",
        ],
        "excluded": [],
    },
}


# ── Dataclasses ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Conflict:
    field_name: str
    primary_value: Any
    secondary_value: Any
    chosen: Any
    chosen_from: str
    note: str = ""


@dataclass
class FactSheet:
    # ── Identity ──
    ticker: str
    bare_symbol: str
    tv_symbol: str | None = None
    market: str = "unknown"
    exchange: str = ""
    name: str = ""
    sector: str = "Unknown"
    industry: str | None = None
    sector_original: str | None = None
    sector_overridden: bool = False
    sector_subtype: SectorSubtype = SectorSubtype.UNKNOWN
    currency_symbol: str | None = None   # None if unknown — never default to "$"
    currency_code: str | None = None

    # ── Price + Technical (TV authoritative) ──
    price: float | None = None
    change_pct: float | None = None
    volume: int | None = None
    market_cap: float | None = None
    pe_ttm: float | None = None
    eps_ttm: float | None = None
    dividend_yield: float | None = None
    sma50: float | None = None
    sma200: float | None = None
    rsi: float | None = None
    macd_value: float | None = None
    macd_signal: float | None = None
    snapshot_ts: str | None = None

    # ── Computed technicals ──
    price_vs_sma200_pct: float | None = None
    price_vs_sma50_pct: float | None = None
    sma50_vs_sma200_pct: float | None = None
    rsi_zone: str | None = None
    macd_direction: str | None = None

    # ── Decision (DecisionState — NEVER LLM body) ──
    verdict: str | None = None
    verdict_type: str | None = None
    action: str | None = None
    evidence: str | None = None
    confidence: str | None = None
    timing: str | None = None

    # ── Scores ──
    eisax_score: int | None = None
    blended_score: int | None = None
    fundamental_quality_score: int | None = None
    risk_score: int | None = None
    overall_risk_label: str | None = None

    # ── Valuation ──
    fair_value: float | None = None
    fair_value_method: str | None = None
    fair_value_label: str | None = None
    fair_value_multiple: float | None = None
    fair_value_eps_used: float | None = None
    upside_pct: float | None = None
    bear_target: float | None = None
    base_target: float | None = None
    bull_target: float | None = None

    # ── Sector flags ──
    is_real_estate: bool = False
    is_energy: bool = False
    is_financial: bool = False
    is_egyptian: bool = False
    is_gcc: bool = False
    is_mena: bool = False

    # ── News config ──
    news_required_keywords: list[str] = field(default_factory=list)
    news_excluded_keywords: list[str] = field(default_factory=list)

    # ── Provenance ──
    source_tv_cache: bool = False
    source_decision_engine: bool = False
    missing_fields: list[str] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    snapshot_age_seconds: int | None = None

    # ── Errors / blockers ──
    blocking_errors: list[str] = field(default_factory=list)
    warning_flags: list[str] = field(default_factory=list)


# ── Builder ───────────────────────────────────────────────────────────────
def build_fact_sheet(
    ticker: str,
    live_payload: dict | None = None,
    tv_cache_row: dict | None = None,
    decision_state: dict | None = None,
) -> FactSheet:
    """
    Build a FactSheet for `ticker`. Pulls TV cache automatically if
    tv_cache_row is not supplied. Pulls DecisionState from live_payload
    if not supplied.

    Args:
        ticker:         Normalized ticker (e.g. "ADNOCGAS.AE", "COMI.CA").
        live_payload:   The orchestrator/handler response dict containing
                        `data.fundamentals`, `data.technical`, `data.decision`,
                        and optionally `report_json.report_meta`.
        tv_cache_row:   Optional pre-fetched TV cache row (pandas Series or
                        dict). If None, fetched via market_cache_adapter.
        decision_state: Optional explicit DecisionState dict; defaults to
                        `live_payload.data.decision`.

    Returns:
        FactSheet (immutable struct of authoritative values + provenance).
    """
    live_payload = live_payload or {}
    data = (live_payload.get("data") or {}) if isinstance(live_payload, dict) else {}
    fundamentals = data.get("fundamentals") or {}
    technical = data.get("technical") or {}
    decision_state = decision_state or data.get("decision") or {}
    report_meta = (live_payload.get("report_json") or {}).get("report_meta") or {}

    # ── Identity ────────────────────────────────────────────────────────
    tkr = (ticker or "").upper().strip()
    bare = tkr.split(":")[-1].split(".")[0]

    market = "america"
    for suffix, mk in _MARKET_BY_SUFFIX.items():
        if tkr.endswith(suffix):
            market = mk
            break

    # Bare crypto symbol (entity resolution strips "-USD" pair) → reclassify
    if market == "america" and bare in _CRYPTO_BARE_SYMBOLS:
        market = "crypto"

    # Currency — NO "$" fallback for MENA
    cur_sym, cur_code = None, None
    for suffix, (sym, code) in _CURRENCY_BY_SUFFIX.items():
        if tkr.endswith(suffix):
            cur_sym, cur_code = sym, code
            break
    if cur_sym is None and market in ("america", "crypto"):
        cur_sym, cur_code = "$", "USD"

    fs = FactSheet(
        ticker=tkr,
        bare_symbol=bare,
        market=market,
        currency_symbol=cur_sym,
        currency_code=cur_code,
    )

    # ── TV cache row (auto-fetch if not supplied) ───────────────────────
    if tv_cache_row is None:
        tv_cache_row = _fetch_tv_row(tkr, market)

    # ── Sector classification ───────────────────────────────────────────
    try:
        from core.sector_overrides import get_corrected_sector
        override = get_corrected_sector(bare)
    except Exception:
        override = None

    raw_sector = ""
    if tv_cache_row is not None:
        raw_sector = str(_row_get(tv_cache_row, "sector") or "").strip()
    if not raw_sector:
        raw_sector = str(fundamentals.get("sector") or "").strip()

    if override:
        fs.sector, fs.industry = override
        fs.sector_overridden = True
        fs.sector_original = raw_sector or None
    else:
        fs.sector = raw_sector or "Unknown"
        fs.industry = (
            (_row_get(tv_cache_row, "industry") if tv_cache_row is not None else None)
            or fundamentals.get("industry")
        )

    fs.sector_subtype = _detect_subtype(fs.sector, fs.industry, bare)

    # Market-based subtype overrides for crypto / commodities — TV cache
    # may not have a sector text for these so keyword matcher misses them.
    if fs.sector_subtype == SectorSubtype.UNKNOWN:
        if market == "crypto":
            fs.sector_subtype = SectorSubtype.CRYPTO
            if fs.sector in ("", "Unknown"):
                fs.sector = "Cryptocurrency"
        elif market == "commodities":
            fs.sector_subtype = SectorSubtype.COMMODITY
            if fs.sector in ("", "Unknown"):
                fs.sector = "Commodities"

    fs.is_real_estate = fs.sector_subtype in _RE_SUBTYPES
    fs.is_energy = fs.sector_subtype in _ENERGY_SUBTYPES
    fs.is_financial = fs.sector_subtype in _FINANCIAL_SUBTYPES
    fs.is_egyptian = market == "egypt"
    fs.is_gcc = market in _GCC_MARKETS
    fs.is_mena = market in _MENA_MARKETS

    # ── Price & Technical (TV authoritative; fallback to engine) ────────
    if tv_cache_row is not None:
        fs.source_tv_cache = True
        fs.tv_symbol = str(_row_get(tv_cache_row, "ticker") or "")
        fs.name = str(_row_get(tv_cache_row, "name") or "")
        fs.price          = _to_float(_row_get(tv_cache_row, "close"))
        fs.change_pct     = _to_float(_row_get(tv_cache_row, "change"))
        fs.volume         = _to_int(_row_get(tv_cache_row, "volume"))
        fs.market_cap     = _to_float(_row_get(tv_cache_row, "market_cap_basic"))
        fs.pe_ttm         = _to_float(_row_get(tv_cache_row, "price_earnings_ttm"))
        fs.eps_ttm        = _to_float(_row_get(tv_cache_row, "earnings_per_share_diluted_ttm"))
        fs.dividend_yield = _to_float(_row_get(tv_cache_row, "dividend_yield_recent"))
        fs.sma50          = _to_float(_row_get(tv_cache_row, "SMA50"))
        fs.sma200         = _to_float(_row_get(tv_cache_row, "SMA200"))
        fs.rsi            = _to_float(_row_get(tv_cache_row, "RSI"))
        fs.macd_value     = _to_float(_row_get(tv_cache_row, "MACD.macd"))
        fs.macd_signal    = _to_float(_row_get(tv_cache_row, "MACD.signal"))
        fs.snapshot_ts    = str(_row_get(tv_cache_row, "_snapshot_ts") or "")

    # Fallback to engine technical (still NOT LLM body)
    if fs.price is None:
        fs.price = _to_float(technical.get("close") or fundamentals.get("price"))
    if fs.sma50 is None:
        fs.sma50 = _to_float(technical.get("sma50"))
    if fs.sma200 is None:
        fs.sma200 = _to_float(technical.get("sma200"))
    if fs.rsi is None:
        fs.rsi = _to_float(technical.get("rsi"))
    if fs.pe_ttm is None:
        fs.pe_ttm = _to_float(fundamentals.get("pe") or fundamentals.get("pe_ttm"))
    if fs.eps_ttm is None:
        fs.eps_ttm = _to_float(fundamentals.get("eps") or fundamentals.get("eps_ttm"))

    # yfinance fast_info fallback for crypto + commodities (TV cache often empty)
    if fs.price is None and market in ("crypto", "commodities"):
        # For crypto, entity_resolution may have stripped "-USD" — re-add it
        # so yfinance recognises the pair (yf needs BTC-USD, not bare BTC)
        yf_ticker = (
            f"{tkr}-USD" if market == "crypto" and not tkr.endswith("-USD") else tkr
        )
        px, prev, chg = _yfinance_price(yf_ticker)
        if px is not None:
            fs.price = px
            if fs.change_pct is None and chg is not None:
                fs.change_pct = chg
            if not fs.snapshot_ts:
                fs.snapshot_ts = "yfinance:fast_info"

    # ── Computed technicals ────────────────────────────────────────────
    if fs.price and fs.sma200:
        fs.price_vs_sma200_pct = (fs.price - fs.sma200) / fs.sma200 * 100
    if fs.price and fs.sma50:
        fs.price_vs_sma50_pct = (fs.price - fs.sma50) / fs.sma50 * 100
    if fs.sma50 and fs.sma200:
        fs.sma50_vs_sma200_pct = (fs.sma50 - fs.sma200) / fs.sma200 * 100
    if fs.rsi is not None:
        if fs.rsi < 30:
            fs.rsi_zone = "Oversold"
        elif fs.rsi >= 70:
            fs.rsi_zone = "Overbought"
        elif fs.rsi >= 65:
            fs.rsi_zone = "Near Overbought"
        else:
            fs.rsi_zone = "Neutral"
    if fs.macd_value is not None and fs.macd_signal is not None:
        if fs.macd_value > fs.macd_signal:
            fs.macd_direction = "Bullish"
        elif fs.macd_value < fs.macd_signal:
            fs.macd_direction = "Bearish"
        else:
            fs.macd_direction = "Neutral"

    # ── Decision (DecisionState ONLY) ──────────────────────────────────
    # Prefer report_meta (built by pilot_report_parsers from DecisionState)
    # over the top-level live_payload verdict (which goes through extra
    # legacy transformations and is the source of Buy/Hold contradictions).
    if report_meta or decision_state:
        fs.source_decision_engine = True
        ds = report_meta or decision_state
        fs.verdict      = _normalize_verdict(ds.get("verdict"))
        fs.verdict_type = ds.get("verdict_type") or ds.get("fundamental_verdict")
        fs.action       = ds.get("action") or ds.get("entry_timing")
        fs.evidence     = ds.get("evidence") or ds.get("evidence_label")
        fs.confidence   = ds.get("confidence") or ds.get("confidence_label")
        fs.timing       = ds.get("timing") or ds.get("entry_timing")
        fs.eisax_score  = _to_int(ds.get("eisax_score") or ds.get("score"))
        fs.blended_score = _to_int(ds.get("blended_score"))
        fs.risk_score   = _to_int(ds.get("risk_score"))
        fs.overall_risk_label = ds.get("overall_risk_label") or ds.get("risk_label")

    fs.fundamental_quality_score = _to_int(
        report_meta.get("fundamental_quality_score")
        or fundamentals.get("quality_score")
    )

    # ── Fair Value ─────────────────────────────────────────────────────
    fv = (
        fundamentals.get("fair_value")
        or fundamentals.get("fv_estimate")
        or report_meta.get("fair_value")
        or report_meta.get("target_price")
    )
    fs.fair_value = _to_float(fv)
    fs.fair_value_multiple = _to_float(
        fundamentals.get("fair_value_multiple") or fundamentals.get("valuation_pe")
    )
    fs.fair_value_eps_used = _to_float(
        fundamentals.get("fair_value_eps_used") or fundamentals.get("forward_eps")
    )
    fs.fair_value_label = fundamentals.get("fair_value_label")
    fs.fair_value_method = "EisaX proprietary (Forward EPS × peer P/E × growth)"
    if fs.fair_value and fs.price and fs.price > 0:
        fs.upside_pct = (fs.fair_value - fs.price) / fs.price * 100

    fs.bear_target = _to_float(
        report_meta.get("bear_target") or fundamentals.get("bear_target")
    )
    fs.base_target = _to_float(
        report_meta.get("base_target") or fundamentals.get("base_target")
    )
    fs.bull_target = _to_float(
        report_meta.get("bull_target") or fundamentals.get("bull_target")
    )

    # ── News keywords ───────────────────────────────────────────────────
    profile = _NEWS_PROFILES.get(fs.sector_subtype)
    if profile:
        # Always include the ticker root + parent company root as first
        # required keyword (so e.g. "ADNOC Gas" passes for ADNOCGAS).
        req = []
        if fs.bare_symbol:
            req.append(re.escape(fs.bare_symbol))
        req.extend(profile["required"])
        fs.news_required_keywords = req
        fs.news_excluded_keywords = list(profile["excluded"])

    # ── Snapshot age ────────────────────────────────────────────────────
    if fs.snapshot_ts:
        try:
            ts = datetime.fromisoformat(fs.snapshot_ts.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            fs.snapshot_age_seconds = int((datetime.now(timezone.utc) - ts).total_seconds())
        except Exception:
            pass

    # ── Validation rules ───────────────────────────────────────────────
    _validate(fs)
    logger.info(
        "[FactSheet] %s subtype=%s sector=%s price=%s sma200=%s verdict=%s score=%s "
        "fv=%s tv=%s ds=%s blockers=%d warnings=%d",
        fs.ticker, fs.sector_subtype.value, fs.sector, fs.price, fs.sma200,
        fs.verdict, fs.eisax_score, fs.fair_value,
        fs.source_tv_cache, fs.source_decision_engine,
        len(fs.blocking_errors), len(fs.warning_flags),
    )
    return fs


# ── Helpers ───────────────────────────────────────────────────────────────
def _fetch_tv_row(ticker: str, market: str):
    """Auto-fetch TV cache row for ticker. Returns None on miss."""
    try:
        from core.data_layer import market_cache_adapter as mca
        df = mca.get_latest_snapshot(market)
        if df is None or df.empty:
            return None
        bare = ticker.split(":")[-1].split(".")[0]
        col = df["ticker"].astype(str).str.upper()
        matches = df[
            col.str.endswith(":" + bare)
            | (col == ticker)
            | (col == bare)
        ]
        if matches.empty:
            return None
        return matches.iloc[0]
    except Exception as e:
        logger.debug("[FactSheet] TV cache lookup failed for %s: %s", ticker, e)
        return None


def _row_get(row, key):
    """Get a key from a pandas Series or dict, returning None if missing."""
    if row is None:
        return None
    try:
        if hasattr(row, "get"):
            return row.get(key)
        return row[key]
    except (KeyError, IndexError, AttributeError):
        return None


def _detect_subtype(sector: str, industry: str | None, bare: str) -> SectorSubtype:
    if bare in _TICKER_SUBTYPE_OVERRIDE:
        return _TICKER_SUBTYPE_OVERRIDE[bare]
    blob = f"{sector or ''} {industry or ''}".lower()
    for pat, sub in _SECTOR_KEYWORD_SUBTYPE:
        if re.search(pat, blob, re.IGNORECASE):
            return sub
    return SectorSubtype.UNKNOWN


def _yfinance_price(ticker: str) -> tuple[float | None, float | None, float | None]:
    """
    Last-resort price source for tickers TV cache doesn't carry
    (BTC-USD, ETH-USD, GC=F when TV is cold, etc.).

    Returns (price, prev_close, change_pct). Any failure → all None.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        fi = t.fast_info
        price = float(getattr(fi, "last_price", None) or 0) or None
        prev  = float(getattr(fi, "previous_close", None) or 0) or None
        change_pct = ((price - prev) / prev * 100.0) if (price and prev) else None
        return price, prev, change_pct
    except Exception as exc:
        logger.warning("[FactSheet] yfinance fallback failed for %s: %s", ticker, exc)
        return None, None, None


def _normalize_verdict(raw):
    if not raw:
        return None
    rv = str(raw).strip().title()
    if rv in ("Buy", "Hold", "Reduce", "Sell"):
        return rv
    return None


def _to_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        if f != f or f == float("inf") or f == -float("inf"):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _to_int(v):
    f = _to_float(v)
    if f is None:
        return None
    try:
        return int(round(f))
    except (TypeError, ValueError, OverflowError):
        return None


def _validate(fs: FactSheet) -> None:
    """Apply validation rules; populate blocking_errors and warning_flags."""
    # Rule 1: price sanity
    if fs.price is None:
        fs.blocking_errors.append("price_missing")
    elif not (0.001 < fs.price < 1_000_000):
        fs.blocking_errors.append(f"price_out_of_range:{fs.price}")
    # Rule 2: SMA200 sanity
    if fs.price and fs.sma200:
        ratio = abs(fs.price - fs.sma200) / fs.sma200
        if ratio > 0.5:
            fs.warning_flags.append(f"sma200_far_from_price:ratio={ratio:.1%}")
    # Rule 3: verdict canonical
    if fs.verdict and fs.verdict not in ("Buy", "Hold", "Reduce", "Sell"):
        fs.blocking_errors.append(f"verdict_invalid:{fs.verdict}")
    if fs.verdict is None:
        fs.warning_flags.append("verdict_missing_from_decision_state")
    # Rule 4: score range
    if fs.eisax_score is not None and not (0 <= fs.eisax_score <= 100):
        fs.blocking_errors.append(f"eisax_score_out_of_range:{fs.eisax_score}")
    if fs.fundamental_quality_score is not None and not (0 <= fs.fundamental_quality_score <= 100):
        fs.blocking_errors.append(
            f"fundamental_quality_score_out_of_range:{fs.fundamental_quality_score}"
        )
    # Rule 5: currency required for MENA tickers — NO $ fallback
    if fs.is_mena and not fs.currency_symbol:
        fs.blocking_errors.append(f"currency_unknown_for_mena:{fs.ticker}")
    # Rule 6: Buy with low upside contradiction
    if fs.verdict == "Buy" and fs.upside_pct is not None and fs.upside_pct < 5:
        fs.warning_flags.append(f"buy_with_low_upside:{fs.upside_pct:.1f}%")
    # Rule 7: stale snapshot (>2h)
    if fs.snapshot_age_seconds is not None and fs.snapshot_age_seconds > 7200:
        fs.warning_flags.append(f"snapshot_stale:{fs.snapshot_age_seconds}s")


# ── Phase D — Pre-grounding ─────────────────────────────────────────────────
# Forward-facing theme guardrails per sector subtype. Mirrors the implicit
# rules the reconciler's sector_scrub enforces post-hoc, but expressed so the
# LLM can respect them BEFORE generating. Subtypes not listed default to empty
# allowed/banned lists (no guardrail line emitted).
_THEME_GUARDRAILS = {
    SectorSubtype.ENERGY_PRODUCER:      {"allowed": ["crude", "Brent", "OPEC", "refining", "hydrocarbon margins"],
                                          "banned":  ["bank/credit themes", "real-estate", "consumer-discretionary"]},
    SectorSubtype.ENERGY_INTEGRATED:    {"allowed": ["crude", "Brent", "OPEC", "downstream margins", "refining"],
                                          "banned":  ["bank/credit themes", "real-estate"]},
    SectorSubtype.GAS_LNG:              {"allowed": ["LNG spot", "gas demand", "pipeline capacity", "Hormuz transit"],
                                          "banned":  ["bank/credit themes", "real-estate", "consumer-discretionary"]},
    SectorSubtype.PETROCHEMICAL:        {"allowed": ["feedstock cost", "petchem spreads", "ethylene/propylene"],
                                          "banned":  ["bank/credit themes", "real-estate"]},
    SectorSubtype.REAL_ESTATE_DEVELOPER:{"allowed": ["off-plan sales", "land acquisition", "construction backlog",
                                                      "mortgage rates", "rental yields"],
                                          "banned":  ["oil price", "crude/Brent", "OPEC", "refinery margins"]},
    SectorSubtype.REAL_ESTATE_OPERATIONS:{"allowed": ["occupancy", "rental yields", "lease renewals", "mortgage rates"],
                                          "banned":  ["oil price", "crude/Brent", "OPEC"]},
    SectorSubtype.BANK:                 {"allowed": ["NIM", "loan growth", "deposits", "central bank rate",
                                                      "capital adequacy / Basel ratios", "cost of risk"],
                                          "banned":  ["oil price (except as country macro context)",
                                                      "crude/Brent direct exposure", "OPEC quotas"]},
    SectorSubtype.INSURANCE:            {"allowed": ["premiums", "claims", "underwriting", "combined ratio"],
                                          "banned":  ["crude/Brent direct exposure"]},
    SectorSubtype.CRYPTO:               {"allowed": ["volatility", "halving cycle", "stablecoin flows", "on-chain activity"],
                                          "banned":  ["P/E", "EPS", "dividends", "central bank policy specific to fiat issuers"]},
    SectorSubtype.COMMODITY:            {"allowed": ["spot", "futures curve", "carry", "storage"],
                                          "banned":  ["P/E", "EPS", "dividends"]},
}


def _fmt_money(symbol: str | None, value) -> str:
    """Render a price with the FactSheet currency symbol, or N/A."""
    if value is None:
        return "N/A"
    try:
        return f"{symbol or ''}{float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def render_pregrounding_block(fs: "FactSheet") -> str:
    """
    Render the SSOT FactSheet as a 'GROUND TRUTH' system-prompt block for the
    LLM, so the body grounds on authoritative values before generation.

    Pure function — no side-effects, no I/O. Idempotent. Safe to call before
    every LLM request. Returns "" if the FactSheet has blocking errors (caller
    should not pre-ground a report that the FactSheet itself rejects).

    Special cases (per Phase D design §3.3):
      - price is None        → omit the LIVE TECHNICAL FACTS section.
      - sma200 is None       → emit "SMA200: not available — use SMA50 instead".
      - verdict is None      → omit the VERDICT section.
    """
    if fs.blocking_errors:
        return ""

    sym = fs.currency_symbol
    code = fs.currency_code or ""
    subtype = fs.sector_subtype.value if fs.sector_subtype else "unknown"
    sector = fs.sector or "Unknown"

    bar = "═" * 75
    lines = [
        bar,
        "GROUND TRUTH FOR THIS REPORT (Single Source of Truth — DO NOT CONTRADICT)",
        bar,
        "",
        f"Ticker          : {fs.ticker}",
        f"Bare symbol     : {fs.bare_symbol}",
        f"Market          : {fs.market}        Sector subtype: {subtype} ({sector})",
    ]
    if sym:
        if sym == "$":
            lines.append(f"Currency        : {sym} ({code})  — write all prices in USD")
        else:
            lines.append(f"Currency        : {sym} ({code})  — write all prices in this symbol; never use $ (except USD/Brent oil refs)")
    if fs.snapshot_age_seconds is not None and fs.snapshot_age_seconds >= 0:
        mins = int(fs.snapshot_age_seconds // 60)
        ts = f"  (timestamp {fs.snapshot_ts})" if fs.snapshot_ts else ""
        lines.append(f"Snapshot age    : {mins} minutes{ts}")

    # LIVE TECHNICAL FACTS — only if price is known
    if fs.price is not None:
        lines += ["", "LIVE TECHNICAL FACTS  (TV cache — do not invent alternative numbers)"]
        lines.append(f"- Price          : {_fmt_money(sym, fs.price)}")
        if fs.sma50 is not None:
            vs50 = f"   (price vs SMA50 = {fs.price_vs_sma50_pct:+.1f}%)" if fs.price_vs_sma50_pct is not None else ""
            lines.append(f"- SMA50          : {_fmt_money(sym, fs.sma50)}{vs50}")
        if fs.sma200 is not None:
            vs200 = f"   (price vs SMA200 = {fs.price_vs_sma200_pct:+.1f}%)" if fs.price_vs_sma200_pct is not None else ""
            lines.append(f"- SMA200         : {_fmt_money(sym, fs.sma200)}{vs200}")
        else:
            lines.append("- SMA200         : not available — use SMA50 instead; do NOT invent a 200-day average")
        if fs.rsi is not None:
            lines.append(f"- RSI            : {fs.rsi:.0f}")

    # VERDICT — only if known
    if fs.verdict:
        lines += ["", "VERDICT  (DecisionState authoritative — match this in the report header)"]
        lines.append(f"- Verdict        : {fs.verdict}")
        if fs.action:
            lines.append(f"- Action         : {fs.action}")
        if fs.overall_risk_label:
            lines.append(f"- Risk           : {fs.overall_risk_label}")
        if fs.confidence:
            lines.append(f"- Confidence     : {fs.confidence}")
        if fs.eisax_score is not None:
            fq = f"  (Fundamental quality {fs.fundamental_quality_score}/100)" if fs.fundamental_quality_score is not None else ""
            lines.append(f"- Score          : {fs.eisax_score}/100{fq}")

    # SECTOR GUARDRAILS — only if defined for this subtype
    guard = _THEME_GUARDRAILS.get(fs.sector_subtype)
    if guard and (guard.get("allowed") or guard.get("banned")):
        lines += ["", f"SECTOR GUARDRAILS  ({subtype})"]
        if guard.get("allowed"):
            lines.append(f"- ALLOWED themes : {', '.join(guard['allowed'])}")
        if guard.get("banned"):
            lines.append(f"- BANNED themes  : {', '.join(guard['banned'])}")

    # WRITE RULES
    lines += ["", "WRITE RULES"]
    n = 1
    if fs.verdict:
        lines.append(f"{n}. The \"Verdict\" line in the report header MUST equal: **{fs.verdict}**."); n += 1
    if sym:
        if sym == "$":
            lines.append(f"{n}. Every price you mention MUST be in USD ($)."); n += 1
        else:
            lines.append(f"{n}. Every price you mention MUST be in {sym}. Never write $ except for USD/Brent oil refs."); n += 1
    if fs.sma200 is not None:
        lines.append(f"{n}. SMA200 must equal {_fmt_money(sym, fs.sma200)} — do not round or substitute another feed."); n += 1
    else:
        lines.append(f"{n}. Do NOT invent SMA200 if it is unavailable — say it is not available and use SMA50."); n += 1
    if guard and guard.get("banned"):
        lines.append(f"{n}. Stay in the {subtype} thesis. Do not include {guard['banned'][0]} language."); n += 1

    lines += ["", bar, "END OF GROUND TRUTH BLOCK", bar]
    return "\n".join(lines)


__all__ = [
    "FactSheet",
    "SectorSubtype",
    "Conflict",
    "build_fact_sheet",
    "render_pregrounding_block",
]
