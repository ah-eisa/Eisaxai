"""
core/ticker_index.py — Fast in-memory ticker lookup index.

Loads the latest parquet file per market from market_cache/ once (lazy,
thread-safe singleton).  Exposes quick_scan(query) for pre-routing asset
queries before expensive LLM-based entity resolution.

Coverage: ~1 900 tickers across UAE, KSA, Egypt, Kuwait, Qatar, Bahrain,
Morocco, Tunisia, US (top-500), commodities.

Ambiguity policy
----------------
* Exact suffix match  (SEEF.BH, TMGH.CA)    → high confidence, match_type="suffixed"
* Exchange-prefixed   (EGX:TMGH, KSE:KFH)   → high confidence, match_type="exchange_prefixed"
* Bare, unique market (NVDA, AAPL, EMAAR)    → high confidence, match_type="bare_unique"
* Bare, multi-market  (KFH, GFH, SALAM)     → NOT returned by quick_scan;
                                               falls through to entity resolution.
"""
from __future__ import annotations

import glob
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "market_cache")

# ── Exchange → (dot-suffix, market_code, currency) ──────────────────────────
_EXCHANGE_MAP: dict[str, tuple[str, str, str]] = {
    "TADAWUL": (".SR", "SA",  "SAR"),
    "DFM":     (".AE", "AE",  "AED"),
    "ADX":     (".AE", "AE",  "AED"),
    "EGX":     (".CA", "EG",  "EGP"),
    "KSE":     (".KW", "KW",  "KWD"),
    "QSE":     (".QA", "QA",  "QAR"),
    "BAHRAIN": (".BH", "BH",  "BHD"),
    "CSEMA":   (".MA", "MA",  "MAD"),
    "BVMT":    (".TN", "TN",  "TND"),
    "NASDAQ":  ("",    "US",  "USD"),
    "NYSE":    ("",    "US",  "USD"),
    "CBOE":    ("",    "US",  "USD"),
    "AMEX":    ("",    "US",  "USD"),
    "ARCA":    ("",    "US",  "USD"),
    "OTC":     ("",    "US",  "USD"),
    "BATS":    ("",    "US",  "USD"),
    "NYSE MKT":("",   "US",  "USD"),
}

# Market file prefix → asset_type override
_MARKET_ASSET_TYPE: dict[str, str] = {
    "commodities": "commodity",
    "crypto":      "crypto",
}

# Tokens that should never be treated as tickers
_STOPWORDS: frozenset[str] = frozenset({
    "AND", "OR", "IN", "OF", "THE", "FOR", "TO", "AT", "BY", "ON",
    "IS", "IT", "BE", "DO", "GO", "IF", "US", "UK", "EU", "ME",
    "MY", "AN", "AM", "AS", "UP", "SO", "NO", "HI", "OK", "YES",
    "BUY", "SELL", "HOLD", "RISK", "HIGH", "LOW", "RATE",
    "FUND", "BOND", "GOLD", "OIL", "USD", "AED", "SAR", "EGP",
    "KWD", "QAR", "BHD", "MAD", "TND",
    # Common English query words that should not be treated as tickers
    "ANALYZE", "ANALYSIS", "STOCK", "PRICE", "MARKET", "SHOW",
    "TECHNICAL", "FUNDAMENTAL", "SECTOR", "STRONG", "LATEST",
    "WHAT", "HOW", "WHEN", "CURRENT", "PLEASE", "GIVE", "REPORT",
})


@dataclass(frozen=True)
class TickerMatch:
    symbol: str         # canonical form: EMAAR.AE / NVDA / GC=F
    exchange: str       # DFM / NASDAQ / ""
    market: str         # AE / US / commodity / crypto
    asset_type: str     # equity / commodity / crypto
    currency: str       # AED / USD / …
    name: str           # display name
    match_type: str     # "suffixed" | "exchange_prefixed" | "bare_unique" | "bare_ambiguous"
    candidates: tuple  = field(default_factory=tuple)  # populated for bare_ambiguous


# ── Singleton index ──────────────────────────────────────────────────────────

# Main index: lookup key → TickerMatch
_INDEX: dict[str, TickerMatch] = {}
# Ambiguous bare symbols that must NOT auto-resolve
_BARE_AMBIGUOUS: frozenset[str] = frozenset()
_INDEX_READY = False
_INDEX_LOCK = threading.Lock()


def _latest_parquet_files() -> list[str]:
    """Return the single latest parquet file per market."""
    markets: dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(_CACHE_DIR, "*.parquet"))):
        parts = os.path.basename(f).rsplit("_", 2)
        market_key = parts[0]
        markets[market_key] = f  # sorted ascending → last = latest timestamp
    return list(markets.values())


def _build_index() -> tuple[dict[str, TickerMatch], frozenset[str]]:
    try:
        import pandas as pd
    except ImportError:
        logger.warning("[ticker_index] pandas not available — index disabled")
        return {}, frozenset()

    # ── Pass 1: collect all (bare_upper → set of non-crypto market_keys) ────
    bare_to_markets: dict[str, set[str]] = {}
    raw_rows: list[tuple[str, str, str, str, str, str, str]] = []
    # fields: raw_ticker, bare, exchange_part, suffix, market_code, currency, asset_type

    files = _latest_parquet_files()
    for fpath in files:
        market_key = os.path.basename(fpath).rsplit("_", 2)[0]
        asset_type = _MARKET_ASSET_TYPE.get(market_key, "equity")
        try:
            df = pd.read_parquet(fpath, columns=["ticker", "name"])
        except Exception as exc:
            logger.warning("[ticker_index] cannot read %s: %s", fpath, exc)
            continue

        for _, row in df.iterrows():
            raw_ticker = str(row.get("ticker") or "").strip()
            raw_name   = str(row.get("name")   or "").strip()
            if not raw_ticker:
                continue

            # Skip noisy crypto pairs
            if asset_type == "crypto" and (len(raw_ticker) > 20 or "." in raw_ticker):
                continue

            if ":" in raw_ticker:
                exchange_part, bare = raw_ticker.split(":", 1)
                exchange_part = exchange_part.strip().upper()
                bare = bare.strip()
            else:
                exchange_part = ""
                bare = raw_ticker

            suffix, market_code, currency = _EXCHANGE_MAP.get(
                exchange_part, ("", "OTHER", "USD")
            )

            if asset_type == "commodity":
                market_code = "commodity"
                currency = "USD"
            elif asset_type == "crypto":
                market_code = "crypto"
                currency = "USD"

            raw_rows.append((raw_ticker, bare, exchange_part, suffix, market_code, currency, asset_type, raw_name))

            # Track bare→markets only for non-crypto (crypto duplicates are noise)
            if asset_type not in ("crypto",):
                bare_up = bare.upper()
                bare_to_markets.setdefault(bare_up, set()).add(market_key)

    # ── Determine ambiguous bare symbols ────────────────────────────────────
    ambiguous_bare: frozenset[str] = frozenset(
        b for b, mkts in bare_to_markets.items() if len(mkts) > 1
    )
    logger.info("[ticker_index] ambiguous bare symbols: %d", len(ambiguous_bare))

    # ── Pass 2: build index ──────────────────────────────────────────────────
    index: dict[str, TickerMatch] = {}

    for (raw_ticker, bare, exchange_part, suffix, market_code, currency, asset_type, raw_name) in raw_rows:
        if asset_type == "commodity":
            symbol = bare
        elif asset_type == "crypto":
            symbol = bare
        else:
            symbol = (bare + suffix) if suffix else bare

        bare_up     = bare.upper()
        sym_up      = symbol.upper()
        full_key_up = raw_ticker.upper()

        is_bare_ambiguous = (bare_up in ambiguous_bare) and asset_type not in ("commodity", "crypto")

        # ── Suffixed key (EMAAR.AE, TMGH.CA, 1120.SR) — always high confidence
        if sym_up != bare_up:  # only if there's a suffix
            index[sym_up] = TickerMatch(
                symbol=symbol,
                exchange=exchange_part,
                market=market_code,
                asset_type=asset_type,
                currency=currency,
                name=raw_name or bare,
                match_type="suffixed",
            )

        # ── Exchange-prefixed key (DFM:EMAAR, BVMT:TJARI) — always high confidence
        index[full_key_up] = TickerMatch(
            symbol=symbol,
            exchange=exchange_part,
            market=market_code,
            asset_type=asset_type,
            currency=currency,
            name=raw_name or bare,
            match_type="exchange_prefixed",
        )

        # ── Bare key — only register as "bare_unique" if unambiguous
        if not is_bare_ambiguous:
            index[bare_up] = TickerMatch(
                symbol=symbol,
                exchange=exchange_part,
                market=market_code,
                asset_type=asset_type,
                currency=currency,
                name=raw_name or bare,
                match_type="bare_unique",
            )
        else:
            # Register as bare_ambiguous so lookup() can signal ambiguity,
            # but quick_scan() will skip it.
            index[bare_up] = TickerMatch(
                symbol=symbol,
                exchange=exchange_part,
                market=market_code,
                asset_type=asset_type,
                currency=currency,
                name=raw_name or bare,
                match_type="bare_ambiguous",
            )

    logger.info(
        "[ticker_index] built %d lookup keys (%d ambiguous bare excluded from auto-route)",
        len(index), len(ambiguous_bare),
    )
    return index, ambiguous_bare


def _ensure_index() -> None:
    global _INDEX, _BARE_AMBIGUOUS, _INDEX_READY
    if _INDEX_READY:
        return
    with _INDEX_LOCK:
        if _INDEX_READY:
            return
        _INDEX, _BARE_AMBIGUOUS = _build_index()
        _INDEX_READY = True


# ── Token extraction ─────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(
    r'(?:'
    r'[A-Z][A-Z0-9]{1,15}(?:\.[A-Z]{2})?(?:=F)?'  # equity / commodity (up to 16 chars for long MENA names)
    r'|\d{4}'                                        # 4-digit KSA codes
    r')'
)


def lookup(token: str) -> Optional[TickerMatch]:
    """
    O(1) lookup for a single uppercase token. Returns TickerMatch or None.
    match_type on result indicates confidence level:
      "suffixed" / "exchange_prefixed" / "bare_unique" → safe to auto-route
      "bare_ambiguous" → present in index but should NOT auto-route
    """
    _ensure_index()
    return _INDEX.get(token.upper())


def quick_scan(query: str) -> Optional[TickerMatch]:
    """
    Scan *query* for any token matching a known ticker.

    Resolution priority (first match wins):
      1. Suffixed tokens (SEEF.BH, TMGH.CA, 1120.SR)
      2. Exchange-prefixed tokens (EGX:TMGH — not extracted by regex but
         passed as selected_symbol by caller when present)
      3. Bare unique tokens (NVDA, AAPL, EMAAR)
      4. Bare ambiguous tokens → skipped (returns None → falls to resolver)

    Returns None if no unambiguous match found.
    """
    _ensure_index()
    if not query or not _INDEX:
        return None

    upper_query = query.upper()

    # Check for EXCHANGE:SYMBOL form first (highest priority)
    _exch_m = re.search(r'\b([A-Z]{2,12}:[A-Z0-9]{1,15})\b', upper_query)
    if _exch_m:
        hit = _INDEX.get(_exch_m.group(1))
        if hit and hit.match_type == "exchange_prefixed":
            logger.debug("[ticker_index] exchange-prefixed match: %s → %s", _exch_m.group(1), hit.symbol)
            return hit

    tokens = _TOKEN_RE.findall(upper_query)
    # Deduplicate, filter stopwords, sort: suffixed tokens first, then longest
    def _sort_key(t: str) -> tuple:
        has_dot = '.' in t
        return (not has_dot, -len(t))  # suffixed first, then longest

    candidates = sorted(
        {t for t in tokens if t not in _STOPWORDS},
        key=_sort_key,
    )

    for tok in candidates:
        match = _INDEX.get(tok)
        if match is None:
            continue
        if match.match_type == "bare_ambiguous":
            # Ambiguous: do not auto-route with high confidence
            logger.debug(
                "[ticker_index] bare_ambiguous '%s' — skipping auto-route, "
                "falling through to entity resolution",
                tok,
            )
            continue
        logger.debug("[ticker_index] '%s' → %s (%s, %s)", tok, match.symbol, match.market, match.match_type)
        return match

    return None


def index_size() -> int:
    """Return number of lookup keys in the index (for diagnostics)."""
    _ensure_index()
    return len(_INDEX)


def reload() -> None:
    """Force a full index rebuild (e.g. after market_cache refresh)."""
    global _INDEX, _BARE_AMBIGUOUS, _INDEX_READY
    with _INDEX_LOCK:
        _INDEX_READY = False
        _INDEX, _BARE_AMBIGUOUS = _build_index()
        _INDEX_READY = True
