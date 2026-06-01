from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import logging
from pathlib import Path
import re
from typing import Callable, Iterable, Optional


logger = logging.getLogger(__name__)

UNIVERSE_SOURCE = (
    "core.local_tickers.MARKET_DB + core.tools.ticker_resolver global dictionaries "
    "(US stocks, crypto, ETFs, indices, commodities, forex)"
)

_ANALYSIS_PREFIX_PATTERNS: tuple[str, ...] = (
    r"^\s*(?:please\s+)?(?:analyze|analysis of|full analysis of|quick analysis of|brief analysis of)\s+",
    r"^\s*(?:حلل|حللي|حللى|تحليل|تحليل سهم|حلل سهم|حلل شركة)\s+",
)

_NOISE_WORDS: frozenset[str] = frozenset(
    {
        "stock",
        "stocks",
        "share",
        "shares",
        "equity",
        "company",
        "corp",
        "corporation",
        "inc",
        "limited",
        "plc",
        "co",
        "the",
        "سهم",
        "اسهم",
        "شركة",
    }
)

_CORPORATE_SUFFIXES: frozenset[str] = frozenset(
    {
        "pjsc",
        "plc",
        "llc",
        "ltd",
        "limited",
        "co",
        "company",
        "holding",
        "holdings",
    }
)

_TOKEN_STOPWORDS: frozenset[str] = frozenset(
    {
        "bank",
        "group",
        "company",
        "holding",
        "holdings",
        "national",
        "international",
        "investment",
        "investments",
        "properties",
        "real",
        "estate",
        "pjsc",
        "plc",
    }
)

_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_ARABIC_NORMALIZATION_MAP: tuple[tuple[str, str], ...] = (
    ("أ", "ا"),
    ("إ", "ا"),
    ("آ", "ا"),
    ("ة", "ه"),
    ("ى", "ي"),
)

_TICKER_WITH_SUFFIX_RE = re.compile(r"^[A-Z0-9]{1,10}\.[A-Z]{2,4}$")
_CRYPTO_PAIR_RE = re.compile(r"^([A-Z]{2,10})-USD$")
_BARE_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9-]{0,5}$")

_SUFFIX_METADATA: dict[str, tuple[str, str]] = {
    "US": ("USA", "USD"),
    "SR": ("SAU", "SAR"),
    "AE": ("UAE", "AED"),
    "DU": ("UAE", "AED"),
    "KW": ("KWT", "KWD"),
    "QA": ("QAT", "QAR"),
    "CA": ("EGY", "EGP"),
    "BH": ("BHR", "BHD"),
    "MA": ("MAR", "MAD"),
    "TN": ("TUN", "TND"),
}

_MARKET_CODE_MAP: dict[str, str] = {
    "saudi": "SAU",
    "ksa": "SAU",
    "uae": "UAE",
    "egypt": "EGY",
    "kuwait": "KWT",
    "qatar": "QAT",
    "bahrain": "BHR",
    "morocco": "MAR",
    "tunisia": "TUN",
    "america": "USA",
    "crypto": "CRYPTO",
    "commodities": "GLOBAL",
}

_UNIVERSE_NAME_PREFERENCES: dict[str, str] = {
    # UAE
    "adib": "ADIB.AE",
    "emaar": "EMAAR.DU",
    "اعمار": "EMAAR.DU",
    # Saudi
    "saudi aramco": "2222.SR",
    "aramco": "2222.SR",
    "al rajhi": "1180.SR",
    "al rajhi bank": "1180.SR",
    "sabic": "2010.SR",
    # Kuwait — prefer KWT over BHR when ambiguous
    "kfh": "KFH.KW",
    "kuwait finance house": "KFH.KW",
    "nbk": "NBK.KW",
    "national bank of kuwait": "NBK.KW",
    # Qatar
    "qnb": "QNBK.QA",
    "qatar national bank": "QNBK.QA",
    "ooredoo": "ORDS.QA",
    "ooredoo qatar": "ORDS.QA",
    # Egypt
    "cib": "COMI.CA",
    "commercial international bank": "COMI.CA",
    "commercial int bank": "COMI.CA",
    "comi": "COMI.CA",
    "tmgh": "TMGH.CA",
    "talaat mostafa": "TMGH.CA",
    "talaat moustafa": "TMGH.CA",
    "talaat mostafa group": "TMGH.CA",
    "palm hills": "PHDC.CA",
    "phdc": "PHDC.CA",
    "ccap": "CCAP.CA",
    "emfd": "EMFD.CA",
    "zmid": "ZMID.CA",
    "efg hermes": "HRHO.CA",
    # Commodities
    "gold": "GC=F",
    "silver": "SI=F",
    "crude oil": "CL=F",
    "brent": "BZ=F",
    "natural gas": "NG=F",
    "copper": "HG=F",
    "platinum": "PL=F",
    # Arabic names
    "ارامكو": "2222.SR",
    "أرامكو": "2222.SR",
    "الراجحي": "1180.SR",
    "بيتكوين": "BTC",
    "ذهب": "GC=F",
    "فضة": "SI=F",
}


@dataclass(frozen=True)
class CanonicalInstrument:
    symbol: str
    market: str
    asset_type: str
    currency: str
    name: Optional[str] = None
    local_name: Optional[str] = None
    exchange: Optional[str] = None
    alternate_names: tuple[str, ...] = ()
    cache_backed: Optional[bool] = None
    source_tag: Optional[str] = None

    def to_candidate(self) -> dict[str, str]:
        payload = {
            "symbol": self.symbol,
            "market": self.market,
            "asset_type": self.asset_type,
            "currency": self.currency,
        }
        if self.name:
            payload["name"] = self.name
        if self.local_name:
            payload["local_name"] = self.local_name
        if self.exchange:
            payload["exchange"] = self.exchange
        return payload


@dataclass(frozen=True)
class EntityResolution:
    query_raw: str
    normalized_query: str
    resolution_status: str
    symbol: Optional[str] = None
    market: Optional[str] = None
    asset_type: Optional[str] = None
    currency: Optional[str] = None
    resolution_source: Optional[str] = None
    confidence: Optional[str] = None
    name: Optional[str] = None
    local_name: Optional[str] = None
    exchange: Optional[str] = None
    universe_source: Optional[str] = None
    candidates: tuple[dict[str, str], ...] = ()

    @property
    def is_resolved(self) -> bool:
        return self.resolution_status == "resolved" and bool(self.symbol)

    @property
    def analysis_instruction(self) -> str:
        if not self.is_resolved:
            raise RuntimeError("analysis_instruction is only available for resolved entities")
        return f"analyze {self.symbol}"

    def to_dict(self) -> dict:
        payload = {
            "query_raw": self.query_raw,
            "normalized_query": self.normalized_query,
            "resolution_status": self.resolution_status,
        }
        if self.symbol:
            payload.update(
                {
                    "symbol": self.symbol,
                    "market": self.market,
                    "asset_type": self.asset_type,
                    "currency": self.currency,
                    "resolution_source": self.resolution_source,
                    "confidence": self.confidence,
                }
            )
            if self.name:
                payload["name"] = self.name
            if self.local_name:
                payload["local_name"] = self.local_name
            if self.exchange:
                payload["exchange"] = self.exchange
            if self.universe_source:
                payload["universe_source"] = self.universe_source
        if self.candidates:
            payload["candidates"] = list(self.candidates)
        return payload


@dataclass(frozen=True)
class InstrumentUniverseIndex:
    instruments: tuple[CanonicalInstrument, ...]
    by_symbol: dict[str, CanonicalInstrument]
    exact_name_index: dict[str, tuple[CanonicalInstrument, ...]]
    token_index: dict[str, tuple[CanonicalInstrument, ...]]
    fuzzy_names: tuple[str, ...]


_CANONICAL_INSTRUMENTS: dict[str, CanonicalInstrument] = {
    "NVDA": CanonicalInstrument("NVDA", "USA", "equity", "USD", name="NVIDIA"),
    "MSFT": CanonicalInstrument("MSFT", "USA", "equity", "USD", name="Microsoft"),
    "AAPL": CanonicalInstrument("AAPL", "USA", "equity", "USD", name="Apple"),
    "TSLA": CanonicalInstrument("TSLA", "USA", "equity", "USD", name="Tesla"),
    "AMD": CanonicalInstrument("AMD", "USA", "equity", "USD", name="AMD"),
    "PLTR": CanonicalInstrument("PLTR", "USA", "equity", "USD", name="Palantir"),
    "BTC": CanonicalInstrument("BTC", "CRYPTO", "crypto", "USD", name="Bitcoin"),
    "ETH": CanonicalInstrument("ETH", "CRYPTO", "crypto", "USD", name="Ethereum"),
    "SOL": CanonicalInstrument("SOL", "CRYPTO", "crypto", "USD", name="Solana"),
    "XRP": CanonicalInstrument("XRP", "CRYPTO", "crypto", "USD", name="XRP"),
    "BNB": CanonicalInstrument("BNB", "CRYPTO", "crypto", "USD", name="BNB"),
    "ADA": CanonicalInstrument("ADA", "CRYPTO", "crypto", "USD", name="Cardano"),
    "DOGE": CanonicalInstrument("DOGE", "CRYPTO", "crypto", "USD", name="Dogecoin"),
}

_ALIAS_MAP: dict[str, str] = {
    "nvda": "NVDA",
    "nvidia": "NVDA",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "apple": "AAPL",
    "aapl": "AAPL",
    "tesla": "TSLA",
    "tsla": "TSLA",
    "amd": "AMD",
    "palantir": "PLTR",
    "pltr": "PLTR",
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "xrp": "XRP",
    "binance coin": "BNB",
    "bnb": "BNB",
    "cardano": "ADA",
    "ada": "ADA",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "saudi aramco": "2222.SR",
    "aramco": "2222.SR",
    "adnoc gas": "ADNOCGAS.AE",
    "adnocgas": "ADNOCGAS.AE",
    "adnoc distribution": "ADNOCDIST.AE",
    "adnoc dist": "ADNOCDIST.AE",
    "adnoc drill": "ADNOCDRILL.AE",
    "adnocdrill": "ADNOCDRILL.AE",
    "emaar": "EMAAR.DU",
    "adib": "ADIB.AE",
    "qnb": "QNBK.QA",
    "nbk": "NBK.KW",
    "cib": "COMI.CA",
    "fab": "FAB.AE",
    "eand": "EAND.AE",
    "aldar": "ALDAR.AE",
    # Saudi — common name forms
    "al rajhi": "1180.SR",
    "al rajhi bank": "1180.SR",
    "alrajhi": "1180.SR",
    "sabic": "2010.SR",
    # Kuwait
    "kfh": "KFH.KW",
    "kuwait finance house": "KFH.KW",
    # Qatar
    "qatar national bank": "QNBK.QA",
    "qnbk": "QNBK.QA",
    "ooredoo": "ORDS.QA",
    "ooredoo qatar": "ORDS.QA",
    # Egypt
    "talaat mostafa": "TMGH.CA",
    "efg hermes": "HRHO.CA",
    "commercial international bank": "COMI.CA",
    # Commodities
    "gold": "GC=F",
    "xau": "GC=F",
    "xauusd": "GC=F",
    "silver": "SI=F",
    "xag": "SI=F",
    "xagusd": "SI=F",
    "crude oil": "CL=F",
    "wti": "CL=F",
    "brent crude": "BZ=F",
    "natural gas": "NG=F",
    "copper": "HG=F",
    "platinum": "PL=F",
    # Arabic
    "ارامكو": "2222.SR",
    "أرامكو": "2222.SR",
    "الراجحي": "1180.SR",
    "ادنوك غاز": "ADNOCGAS.AE",
    "ادنوك للغاز": "ADNOCGAS.AE",
    "ادنوك للتوزيع": "ADNOCDIST.AE",
    "ادنوك للحفر": "ADNOCDRILL.AE",
    "اعمار": "EMAAR.DU",
    "البنك التجاري الدولي": "COMI.CA",
    "بيتكوين": "BTC",
    "البيتكوين": "BTC",
    "اثيريوم": "ETH",
    "ايثيريوم": "ETH",
    "الايثيريوم": "ETH",
    "ايثريوم": "ETH",
    "إيثريوم": "ETH",
    "ذهب": "GC=F",
    "فضة": "SI=F",
    # US mega-cap Arabic aliases (common transliterations)
    "نيفيديا": "NVDA",
    "انفيديا": "NVDA",
    "نفيديا": "NVDA",
    "مايكروسوفت": "MSFT",
    "ميكروسوفت": "MSFT",
    "ابل": "AAPL",
    "آبل": "AAPL",
    "أبل": "AAPL",
    "تسلا": "TSLA",
}

_AMBIGUOUS_ALIASES: dict[str, tuple[str, ...]] = {
    "adnoc": ("ADNOCGAS.AE", "ADNOCDIST.AE", "ADNOCDRILL.AE"),
    "ادنوك": ("ADNOCGAS.AE", "ADNOCDIST.AE", "ADNOCDRILL.AE"),
}

_BLOCK_GENERIC_US_TICKERS: frozenset[str] = frozenset(
    {
        "ADNOC",
        "ARAMCO",
        "EMAAR",
        "ADIB",
        "EAND",
        "TAQA",
        "ALDAR",
        "QNB",
        "NBK",
        "CIB",
    }
)


def normalize_analysis_query(raw_query: str) -> str:
    text = (raw_query or "").strip()
    for pattern in _ANALYSIS_PREFIX_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = text.strip(" \t\r\n-:,.")

    cleaned_tokens: list[str] = []
    for token in text.split():
        stripped = token.strip("()[]{}.,;:!?")
        if not stripped:
            continue
        if normalize_lookup_key(stripped) in _NOISE_WORDS:
            continue
        cleaned_tokens.append(stripped)
    normalized = " ".join(cleaned_tokens).strip()
    return normalized or text


def normalize_lookup_key(text: str) -> str:
    normalized = (text or "").casefold().replace("&", " and ")
    normalized = normalized.replace("ـ", "")
    normalized = _ARABIC_DIACRITICS_RE.sub("", normalized)
    for source, target in _ARABIC_NORMALIZATION_MAP:
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[^0-9a-z\u0600-\u06FF]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_instrument_name(name: str) -> str:
    tokens = normalize_lookup_key(name).split()
    if len(tokens) > 1:
        tokens = [token for token in tokens if token not in _CORPORATE_SUFFIXES]
    return " ".join(tokens).strip()


def _force_asset_type_for_symbol(symbol: str, asset_type: Optional[str]) -> Optional[str]:
    if str(symbol or "").upper().endswith(".CA"):
        return "equity"
    return asset_type


def _with_forced_asset_type(instrument: CanonicalInstrument) -> CanonicalInstrument:
    asset_type = _force_asset_type_for_symbol(instrument.symbol, instrument.asset_type)
    if asset_type == instrument.asset_type:
        return instrument
    return CanonicalInstrument(
        symbol=instrument.symbol,
        market=instrument.market,
        asset_type=asset_type or instrument.asset_type,
        currency=instrument.currency,
        name=instrument.name,
        local_name=instrument.local_name,
        exchange=instrument.exchange,
        alternate_names=instrument.alternate_names,
        cache_backed=instrument.cache_backed,
        source_tag=instrument.source_tag,
    )


def is_exact_ticker(text: str) -> Optional[CanonicalInstrument]:
    candidate = (text or "").strip().upper()
    if not candidate:
        return None

    if _TICKER_WITH_SUFFIX_RE.fullmatch(candidate):
        if candidate in _universe_symbol_index():
            return _with_forced_asset_type(_universe_symbol_index()[candidate])
        suffix = candidate.rsplit(".", 1)[1]
        metadata = _SUFFIX_METADATA.get(suffix)
        if metadata:
            market, currency = metadata
            return _with_forced_asset_type(
                CanonicalInstrument(candidate, market, "equity", currency, name=candidate)
            )

    pair_match = _CRYPTO_PAIR_RE.fullmatch(candidate)
    if pair_match:
        crypto_symbol = pair_match.group(1)
        if crypto_symbol in _CANONICAL_INSTRUMENTS and _CANONICAL_INSTRUMENTS[crypto_symbol].market == "CRYPTO":
            return _CANONICAL_INSTRUMENTS[crypto_symbol]

    if candidate in _CANONICAL_INSTRUMENTS:
        return _CANONICAL_INSTRUMENTS[candidate]

    universe_match = _universe_symbol_index().get(candidate)
    if universe_match:
        return _with_forced_asset_type(universe_match)

    return None


@lru_cache(maxsize=1)
def load_instrument_universe() -> tuple[CanonicalInstrument, ...]:
    loaded_sources: list[tuple[str, tuple[CanonicalInstrument, ...]]] = []
    for source_name, loader in load_global_universe_sources():
        try:
            loaded_sources.append((source_name, tuple(loader())))
        except Exception as exc:
            logger.warning("[entity-resolution] universe source %s failed: %s", source_name, exc)

    if not loaded_sources:
        logger.warning("[entity-resolution] no universe sources loaded successfully")
        return ()

    universe = merge_universe_sources(loaded_sources)
    logger.info(
        "[entity-resolution] loaded %d universe instruments from %s",
        len(universe),
        ", ".join(source_name for source_name, _ in loaded_sources),
    )
    return universe


def _load_pipeline_parquet_universe() -> tuple[CanonicalInstrument, ...]:
    """
    Load every live parquet snapshot from the 15-min pipeline cache as
    CanonicalInstruments.  Covers Arab equities, US equities, crypto, and
    commodities (Gold, Silver, Crude, etc.).  Listed first in
    load_global_universe_sources() so cache_backed instruments take priority.
    """
    _PM: dict[str, tuple[str, str, str]] = {
        "uae":         ("UAE",    "equity",    "AED"),
        "ksa":         ("SAU",    "equity",    "SAR"),
        "egypt":       ("EGY",    "equity",    "EGP"),
        "kuwait":      ("KWT",    "equity",    "KWD"),
        "qatar":       ("QAT",    "equity",    "QAR"),
        "bahrain":     ("BHR",    "equity",    "BHD"),
        "morocco":     ("MAR",    "equity",    "MAD"),
        "tunisia":     ("TUN",    "equity",    "TND"),
        "america":     ("USA",    "equity",    "USD"),
        "crypto":      ("CRYPTO", "crypto",    "USD"),
        "commodities": ("GLOBAL", "commodity", "USD"),
    }
    _EXCHANGE_SUFFIX: dict[str, str] = {
        "EGX": "CA", "TADAWUL": "SR", "ADX": "AE", "DFM": "DU",
        "KSE": "KW", "QSE": "QA", "BSE": "BH", "CSE": "BH",
        "BAHRAIN": "BH", "BHB": "BH",
        "BVC": "MA", "BRVM": "TN",
        "CSEMA": "MA", "BVMT": "TN",
    }
    instruments: list[CanonicalInstrument] = []
    try:
        from pipeline import cache as _pipeline_cache
    except Exception:
        return ()

    for pipeline_key, (market_code, asset_type, currency) in _PM.items():
        try:
            df, _ts = _pipeline_cache.get_latest(pipeline_key)
            if df is None or df.empty:
                continue
        except Exception:
            continue

        for row in df.to_dict(orient="records"):
            raw_ticker = str(row.get("ticker") or "").strip()
            if not raw_ticker:
                continue

            if ":" in raw_ticker:
                exch, base = raw_ticker.split(":", 1)
                suffix = _EXCHANGE_SUFFIX.get(exch.upper())
                if suffix:
                    symbol = f"{base}.{suffix}"
                elif asset_type == "crypto":
                    symbol = base.replace("USDT", "").replace("USD", "") or base
                else:
                    symbol = base
            else:
                symbol = raw_ticker

            symbol = symbol.strip().upper()
            if not symbol:
                continue

            name = str(row.get("name") or "").strip() or symbol
            alt_names: list[str] = []
            if raw_ticker != symbol:
                alt_names.append(raw_ticker)
            if ":" in raw_ticker:
                alt_names.append(raw_ticker.split(":", 1)[1])

            instruments.append(
                CanonicalInstrument(
                    symbol=symbol,
                    market=market_code,
                    asset_type=_force_asset_type_for_symbol(symbol, asset_type),
                    currency=currency,
                    name=name,
                    alternate_names=tuple(alt_names),
                    cache_backed=True,
                    source_tag="pipeline.parquet_cache",
                )
            )

    return tuple(instruments)


def invalidate_instrument_index_cache() -> None:
    """Force rebuild of universe index (call after parquet refresh)."""
    build_instrument_index.cache_clear()
    load_instrument_universe.cache_clear()


def load_global_universe_sources() -> tuple[tuple[str, Callable[[], tuple[CanonicalInstrument, ...]]], ...]:
    return (
        ("pipeline_parquet_cache", _load_pipeline_parquet_universe),
        ("local_market_universe", _load_local_market_universe),
        ("global_tool_universe", _load_tool_resolver_global_universe),
    )


def merge_universe_sources(
    loaded_sources: Iterable[tuple[str, tuple[CanonicalInstrument, ...]]],
) -> tuple[CanonicalInstrument, ...]:
    merged: dict[str, CanonicalInstrument] = {}

    for _, instruments in loaded_sources:
        for instrument in instruments:
            existing = merged.get(instrument.symbol)
            if existing is None:
                merged[instrument.symbol] = instrument
                continue
            merged[instrument.symbol] = _merge_instrument(existing, instrument)

    return tuple(merged.values())


@lru_cache(maxsize=1)
def build_instrument_index() -> InstrumentUniverseIndex:
    universe = load_instrument_universe()
    exact_name_index: dict[str, list[CanonicalInstrument]] = {}
    token_index: dict[str, list[CanonicalInstrument]] = {}

    for instrument in universe:
        for raw_name in _iter_instrument_names(instrument):
            normalized_name = normalize_instrument_name(raw_name)
            if not normalized_name:
                continue
            exact_name_index.setdefault(normalized_name, []).append(instrument)
            for token in normalized_name.split():
                if len(token) < 4 or token in _TOKEN_STOPWORDS:
                    continue
                token_index.setdefault(token, []).append(instrument)

    return InstrumentUniverseIndex(
        instruments=universe,
        by_symbol={instrument.symbol: instrument for instrument in universe},
        exact_name_index={key: tuple(_dedupe_instruments(value)) for key, value in exact_name_index.items()},
        token_index={key: tuple(_dedupe_instruments(value)) for key, value in token_index.items()},
        fuzzy_names=tuple(exact_name_index.keys()),
    )


def lookup_from_universe(text: str) -> EntityResolution | None:
    index = build_instrument_index()
    if not index.instruments:
        return None

    raw_lookup_key = normalize_lookup_key(text)
    normalized_query = normalize_instrument_name(text)
    if not normalized_query:
        return None
    exact_or_normalized = "universe_normalized" if raw_lookup_key != normalized_query else "universe_exact"

    ambiguous_symbols = _AMBIGUOUS_ALIASES.get(normalized_query) or _AMBIGUOUS_ALIASES.get(raw_lookup_key)
    if ambiguous_symbols:
        return EntityResolution(
            query_raw=text,
            normalized_query=normalized_query,
            resolution_status="ambiguous",
            candidates=tuple(_candidate_for_symbol(symbol) for symbol in ambiguous_symbols),
        )

    exact_matches = index.exact_name_index.get(normalized_query)
    if exact_matches:
        return _resolve_or_ambiguous(
            raw_query=text,
            normalized_query=normalized_query,
            instruments=exact_matches,
            resolution_source=exact_or_normalized,
        )

    if " " not in normalized_query:
        token_matches = index.token_index.get(normalized_query)
        if token_matches:
            return _resolve_or_ambiguous(
                raw_query=text,
                normalized_query=normalized_query,
                instruments=token_matches,
                resolution_source=exact_or_normalized,
            )

    fuzzy_result = _resolve_with_universe_fuzzy_match(text, normalized_query, index)
    if fuzzy_result:
        return fuzzy_result
    return None


def lookup_alias(text: str) -> EntityResolution | None:
    key = normalize_lookup_key(text)
    if not key:
        return None

    ambiguous = _AMBIGUOUS_ALIASES.get(key)
    if ambiguous:
        return EntityResolution(
            query_raw=text,
            normalized_query=text,
            resolution_status="ambiguous",
            candidates=tuple(_candidate_for_symbol(symbol) for symbol in ambiguous),
        )

    symbol = _ALIAS_MAP.get(key)
    if symbol:
        instrument = is_exact_ticker(symbol)
        if instrument:
            return _resolved(text, text, instrument, "alias_map")
        return None

    compact_key = key.replace(" ", "")
    symbol = _ALIAS_MAP.get(compact_key)
    if symbol:
        instrument = is_exact_ticker(symbol)
        if instrument:
            return _resolved(text, text, instrument, "alias_map")
        return None

    return None


def resolve_with_fuzzy_match(text: str) -> EntityResolution | None:
    key = normalize_lookup_key(text)
    if len(key) < 4:
        return None

    best_key: Optional[str] = None
    best_score = 0.0
    second_score = 0.0

    for candidate in _ALIAS_MAP:
        score = SequenceMatcher(None, key, candidate).ratio()
        if score > best_score:
            second_score = best_score
            best_score = score
            best_key = candidate
        elif score > second_score:
            second_score = score

    if not best_key or best_score < 0.94 or (best_score - second_score) < 0.05:
        return None

    symbol = _ALIAS_MAP[best_key]
    instrument = is_exact_ticker(symbol)
    if not instrument:
        return None
    return _resolved(text, text, instrument, "fallback")


def resolve_asset_entity(query_text: str) -> EntityResolution:
    normalized_query = normalize_analysis_query(query_text)
    if not normalized_query:
        return EntityResolution(
            query_raw=query_text,
            normalized_query=normalized_query,
            resolution_status="unresolved",
        )

    exact = is_exact_ticker(normalized_query)
    if exact:
        return _resolved(query_text, normalized_query, exact, "exact_ticker")

    universe_match = lookup_from_universe(normalized_query)
    if universe_match:
        return _clone_resolution(query_text, normalized_query, universe_match)

    alias = lookup_alias(normalized_query)
    if alias:
        return _clone_resolution(query_text, normalized_query, alias)

    upper = normalized_query.upper()
    if " " not in normalized_query and _BARE_TICKER_RE.fullmatch(upper) and upper not in _BLOCK_GENERIC_US_TICKERS:
        return _resolved(
            query_text,
            normalized_query,
            CanonicalInstrument(upper, "USA", "equity", "USD", name=upper),
            "fallback",
        )

    fuzzy = resolve_with_fuzzy_match(normalized_query)
    if fuzzy:
        return _clone_resolution(query_text, normalized_query, fuzzy)

    return EntityResolution(
        query_raw=query_text,
        normalized_query=normalized_query,
        resolution_status="unresolved",
    )


def _load_local_market_universe() -> tuple[CanonicalInstrument, ...]:
    from core.config import HISTORICAL_DATA_DIR
    from core.local_tickers import MARKET_DB

    cache_backed_symbols = _load_cache_backed_symbols(HISTORICAL_DATA_DIR)
    instruments: list[CanonicalInstrument] = []

    for market_key, market_entries in MARKET_DB.items():
        if market_key not in _MARKET_CODE_MAP:
            continue
        market_code = _MARKET_CODE_MAP[market_key]
        for symbol, info in market_entries.items():
            if not isinstance(info, dict):
                continue
            aliases = []
            aliases.extend(info.get("aliases_en") or [])
            aliases.extend(info.get("aliases_ar") or [])
            name = (info.get("name_en") or symbol).strip()
            local_name = (info.get("name_ar") or "").strip() or None
            instruments.append(
                CanonicalInstrument(
                    symbol=symbol.strip().upper(),
                    market=market_code,
                    asset_type=_force_asset_type_for_symbol(symbol, "equity"),
                    currency=(info.get("currency") or _default_currency_for_market(market_code)),
                    name=name,
                    local_name=local_name,
                    exchange=info.get("exchange"),
                    alternate_names=tuple(str(alias).strip() for alias in aliases if str(alias).strip()),
                    cache_backed=(symbol.strip().upper() in cache_backed_symbols) if cache_backed_symbols else None,
                    source_tag="core.local_tickers.MARKET_DB",
                )
            )

    return tuple(instruments)


def _load_tool_resolver_global_universe() -> tuple[CanonicalInstrument, ...]:
    from core.tools import ticker_resolver as tool_resolver

    source_maps = (
        ("us_stocks", getattr(tool_resolver, "_US_STOCKS"), "USA", "equity", "USD"),
        ("crypto", getattr(tool_resolver, "_CRYPTO"), "CRYPTO", "crypto", "USD"),
        ("etfs", getattr(tool_resolver, "_ETFS"), "USA", "etf", "USD"),
        ("indices", getattr(tool_resolver, "_INDICES"), "GLOBAL", "index", "USD"),
        ("commodities", getattr(tool_resolver, "_COMMODITIES"), "GLOBAL", "commodity", "USD"),
        ("forex", getattr(tool_resolver, "_FOREX"), "FX", "forex", "USD"),
    )

    grouped: dict[str, dict] = {}
    for source_name, mapping, market, asset_type, currency in source_maps:
        for raw_name, raw_symbol in mapping.items():
            if not raw_name or not raw_symbol:
                continue

            symbol = str(raw_symbol).strip().upper()
            normalized_name = str(raw_name).strip()
            if not normalized_name:
                continue

            if source_name == "crypto":
                if symbol.endswith("-USD"):
                    symbol = symbol.split("-", 1)[0]
                elif symbol in {"IBIT"}:
                    # Keep ETF-like crypto wrappers in the ETF universe only.
                    continue

            entry = grouped.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "market": market,
                    "asset_type": _force_asset_type_for_symbol(symbol, asset_type),
                    "currency": currency,
                    "names": [],
                    "source_tag": "core.tools.ticker_resolver",
                },
            )
            entry["market"] = _prefer_market(entry["market"], market)
            entry["asset_type"] = _force_asset_type_for_symbol(
                symbol,
                _prefer_asset_type(entry["asset_type"], asset_type),
            )
            entry["currency"] = currency or entry["currency"]
            entry["names"].append(normalized_name)

    instruments: list[CanonicalInstrument] = []
    for entry in grouped.values():
        names = _stable_unique(entry["names"])
        display_name = _choose_display_name(names, entry["symbol"])
        alternate_names = tuple(name for name in names if name != display_name)
        instruments.append(
            CanonicalInstrument(
                symbol=entry["symbol"],
                market=entry["market"],
                asset_type=_force_asset_type_for_symbol(entry["symbol"], entry["asset_type"]),
                currency=entry["currency"],
                name=display_name,
                alternate_names=alternate_names,
                source_tag=entry["source_tag"],
            )
        )

    return tuple(instruments)


def _load_cache_backed_symbols(historical_root: Path) -> set[str]:
    symbols: set[str] = set()
    if not historical_root.exists():
        return symbols
    for path in historical_root.rglob("*.parquet"):
        stem = path.stem.upper()
        if "_" not in stem:
            continue
        base, suffix = stem.rsplit("_", 1)
        if suffix and len(suffix) <= 4:
            symbols.add(f"{base}.{suffix}")
    return symbols


def _iter_instrument_names(instrument: CanonicalInstrument) -> tuple[str, ...]:
    values = [instrument.name, instrument.local_name]
    values.extend(instrument.alternate_names)
    values.append(instrument.symbol)
    if instrument.name:
        values.append(f"{instrument.name} {instrument.symbol}")
    cleaned: list[str] = []
    for value in values:
        if not value:
            continue
        stripped = str(value).strip()
        if stripped:
            cleaned.append(stripped)
    return tuple(dict.fromkeys(cleaned))


def _resolve_with_universe_fuzzy_match(
    raw_query: str,
    normalized_query: str,
    index: InstrumentUniverseIndex,
) -> EntityResolution | None:
    if len(normalized_query) < 5:
        return None

    best_name: Optional[str] = None
    best_score = 0.0
    second_score = 0.0

    for candidate_name in index.fuzzy_names:
        score = SequenceMatcher(None, normalized_query, candidate_name).ratio()
        if score > best_score:
            second_score = best_score
            best_score = score
            best_name = candidate_name
        elif score > second_score:
            second_score = score

    if not best_name or best_score < 0.95 or (best_score - second_score) < 0.05:
        return None

    matches = index.exact_name_index.get(best_name, ())
    if not matches:
        return None
    return _resolve_or_ambiguous(
        raw_query=raw_query,
        normalized_query=normalized_query,
        instruments=matches,
        resolution_source="universe_fuzzy",
    )


def _resolve_or_ambiguous(
    *,
    raw_query: str,
    normalized_query: str,
    instruments: tuple[CanonicalInstrument, ...],
    resolution_source: str,
) -> EntityResolution:
    unique_instruments = tuple(_dedupe_instruments(instruments))
    preferred_symbol = _UNIVERSE_NAME_PREFERENCES.get(normalized_query)
    if preferred_symbol:
        preferred_matches = [instrument for instrument in unique_instruments if instrument.symbol == preferred_symbol]
        if len(preferred_matches) == 1:
            return _resolved(raw_query, normalized_query, preferred_matches[0], resolution_source)
    if len(unique_instruments) == 1:
        return _resolved(raw_query, normalized_query, unique_instruments[0], resolution_source)
    return EntityResolution(
        query_raw=raw_query,
        normalized_query=normalized_query,
        resolution_status="ambiguous",
        candidates=tuple(instrument.to_candidate() for instrument in unique_instruments),
    )


def _candidate_for_symbol(symbol: str) -> dict[str, str]:
    instrument = is_exact_ticker(symbol)
    if instrument:
        return instrument.to_candidate()
    return {"symbol": symbol}


def _resolved(
    raw_query: str,
    normalized_query: str,
    instrument: CanonicalInstrument,
    resolution_source: str,
) -> EntityResolution:
    return EntityResolution(
        query_raw=raw_query,
        normalized_query=normalized_query,
        resolution_status="resolved",
        symbol=instrument.symbol,
        market=instrument.market,
        asset_type=_force_asset_type_for_symbol(instrument.symbol, instrument.asset_type),
        currency=instrument.currency,
        resolution_source=resolution_source,
        confidence="high",
        name=instrument.name,
        local_name=instrument.local_name,
        exchange=instrument.exchange,
        universe_source=(instrument.source_tag or UNIVERSE_SOURCE) if resolution_source.startswith("universe") else None,
    )


def _clone_resolution(
    raw_query: str,
    normalized_query: str,
    resolution: EntityResolution,
) -> EntityResolution:
    return EntityResolution(
        query_raw=raw_query,
        normalized_query=normalized_query,
        resolution_status=resolution.resolution_status,
        symbol=resolution.symbol,
        market=resolution.market,
        asset_type=_force_asset_type_for_symbol(resolution.symbol or "", resolution.asset_type),
        currency=resolution.currency,
        resolution_source=resolution.resolution_source,
        confidence=resolution.confidence,
        name=resolution.name,
        local_name=resolution.local_name,
        exchange=resolution.exchange,
        universe_source=resolution.universe_source,
        candidates=resolution.candidates,
    )


def _dedupe_instruments(instruments: tuple[CanonicalInstrument, ...] | list[CanonicalInstrument]) -> list[CanonicalInstrument]:
    seen: set[str] = set()
    deduped: list[CanonicalInstrument] = []
    for instrument in instruments:
        if instrument.symbol in seen:
            continue
        seen.add(instrument.symbol)
        deduped.append(instrument)
    return deduped


def _merge_instrument(primary: CanonicalInstrument, secondary: CanonicalInstrument) -> CanonicalInstrument:
    alternate_names = _stable_unique(
        [
            *primary.alternate_names,
            *(secondary.alternate_names),
            *( [secondary.name] if secondary.name and secondary.name != primary.name else [] ),
        ]
    )
    return CanonicalInstrument(
        symbol=primary.symbol,
        market=primary.market or secondary.market,
        asset_type=_force_asset_type_for_symbol(primary.symbol, primary.asset_type or secondary.asset_type),
        currency=primary.currency or secondary.currency,
        name=primary.name or secondary.name,
        local_name=primary.local_name or secondary.local_name,
        exchange=primary.exchange or secondary.exchange,
        alternate_names=tuple(name for name in alternate_names if name and name != (primary.name or secondary.name)),
        cache_backed=primary.cache_backed if primary.cache_backed is not None else secondary.cache_backed,
        source_tag=primary.source_tag or secondary.source_tag,
    )


def _choose_display_name(names: list[str], fallback_symbol: str) -> str:
    if not names:
        return fallback_symbol

    english_candidates = [
        name for name in names
        if re.search(r"[A-Za-z]", name) and not name.isupper() and len(name) > 2
    ]
    if english_candidates:
        best = max(english_candidates, key=lambda value: (len(value.split()), len(value)))
        return best.title()

    alpha_candidates = [name for name in names if re.search(r"[A-Za-z]", name)]
    if alpha_candidates:
        return max(alpha_candidates, key=len).title()

    return names[0]


def _prefer_market(current: str, incoming: str) -> str:
    if current == "GLOBAL" and incoming != "GLOBAL":
        return incoming
    return current


def _prefer_asset_type(current: str, incoming: str) -> str:
    if current == "index" and incoming in {"etf", "equity"}:
        return incoming
    return current


def _stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def _default_currency_for_market(market_code: str) -> str:
    return {
        "SAU": "SAR",
        "UAE": "AED",
        "EGY": "EGP",
        "KWT": "KWD",
        "QAT": "QAR",
    }.get(market_code, "USD")


def _universe_symbol_index() -> dict[str, CanonicalInstrument]:
    return build_instrument_index().by_symbol
