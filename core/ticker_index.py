"""
core/ticker_index.py — Fast in-memory ticker lookup index.

Loads the latest parquet file per market from market_cache/ once (lazy,
thread-safe singleton).  Exposes quick_scan(query) for pre-routing asset
queries before expensive LLM-based entity resolution.

Coverage: ~1 900 tickers across UAE, KSA, Egypt, Kuwait, Qatar, Bahrain,
Morocco, Tunisia, US (top-500), commodities.
"""
from __future__ import annotations

import glob
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "market_cache")

# ── Exchange → (dot-suffix, market_code, currency) ──────────────────────────
_EXCHANGE_MAP: dict[str, tuple[str, str, str]] = {
    "TADAWUL": (".SR", "SA",        "SAR"),
    "DFM":     (".AE", "AE",        "AED"),
    "ADX":     (".AE", "AE",        "AED"),
    "EGX":     (".CA", "EG",        "EGP"),
    "KSE":     (".KW", "KW",        "KWD"),
    "QSE":     (".QA", "QA",        "QAR"),
    "BAHRAIN": (".BH", "BH",        "BHD"),
    "CSEMA":   (".MA", "MA",        "MAD"),
    "BVMT":    (".TN", "TN",        "TND"),
    "NASDAQ":  ("",    "US",        "USD"),
    "NYSE":    ("",    "US",        "USD"),
    "CBOE":    ("",    "US",        "USD"),
    "AMEX":    ("",    "US",        "USD"),
    "ARCA":    ("",    "US",        "USD"),
    "OTC":     ("",    "US",        "USD"),
    "BATS":    ("",    "US",        "USD"),
    "NYSE MKT":("",   "US",        "USD"),
}

# Market file prefix → asset_type override
_MARKET_ASSET_TYPE: dict[str, str] = {
    "commodities": "commodity",
    "crypto":      "crypto",
}

# Tokens that are common English words / should not be treated as tickers
_STOPWORDS: frozenset[str] = frozenset({
    "AND", "OR", "IN", "OF", "THE", "FOR", "TO", "AT", "BY", "ON",
    "IS", "IT", "BE", "DO", "GO", "IF", "US", "UK", "EU", "ME",
    "MY", "AN", "AM", "AS", "UP", "SO", "NO", "HI", "OK", "YES",
    "BUY", "SELL", "HOLD", "RISK", "HIGH", "LOW", "RATE",
    "FUND", "BOND", "GOLD", "OIL", "USD", "AED", "SAR", "EGP",
    "KWD", "QAR", "BHD", "MAD", "TND",
})


@dataclass(frozen=True)
class TickerMatch:
    symbol: str       # canonical form used by finance.py: EMAAR.AE / NVDA / GC=F
    exchange: str     # DFM / NASDAQ / ""
    market: str       # AE / US / commodity / crypto
    asset_type: str   # equity / commodity / crypto
    currency: str     # AED / USD / …
    name: str         # display name


# ── Singleton index ──────────────────────────────────────────────────────────

_INDEX: dict[str, TickerMatch] = {}
_INDEX_READY = False
_INDEX_LOCK = threading.Lock()


def _latest_parquet_files() -> list[str]:
    """Return the single latest parquet file per market."""
    markets: dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(_CACHE_DIR, "*.parquet"))):
        # filename: <market>_<YYYYMMDD>_<HHMM>.parquet
        parts = os.path.basename(f).rsplit("_", 2)
        market_key = parts[0]
        markets[market_key] = f  # sorted → last entry = latest timestamp
    return list(markets.values())


def _build_index() -> dict[str, TickerMatch]:
    try:
        import pandas as pd
    except ImportError:
        logger.warning("[ticker_index] pandas not available — index disabled")
        return {}

    index: dict[str, TickerMatch] = {}
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

            # Skip noisy crypto pairs (long exchange names, long symbols)
            if asset_type == "crypto" and (len(raw_ticker) > 20 or "." in raw_ticker):
                continue

            # Parse EXCHANGE:SYMBOL
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
                symbol = bare
                market_code = "commodity"
                currency = "USD"
            elif asset_type == "crypto":
                symbol = bare
                market_code = "crypto"
                currency = "USD"
            else:
                symbol = (bare + suffix) if suffix else bare

            match = TickerMatch(
                symbol=symbol,
                exchange=exchange_part,
                market=market_code,
                asset_type=asset_type,
                currency=currency,
                name=raw_name or bare,
            )

            bare_up      = bare.upper()
            sym_up       = symbol.upper()
            full_key_up  = raw_ticker.upper()

            # Register under: bare symbol, suffixed symbol, EXCHANGE:SYMBOL
            index[bare_up]     = match
            index[sym_up]      = match
            index[full_key_up] = match

    logger.info(
        "[ticker_index] index built — %d lookup keys from %d market files",
        len(index), len(files),
    )
    return index


def _ensure_index() -> None:
    global _INDEX, _INDEX_READY
    if _INDEX_READY:
        return
    with _INDEX_LOCK:
        if _INDEX_READY:
            return
        _INDEX = _build_index()
        _INDEX_READY = True


# ── Token extraction ─────────────────────────────────────────────────────────
# Matches:
#   equity tickers   — 2-8 uppercase letters/digits starting with a letter  (NVDA, EMAAR, KFH)
#   KSA 4-digit codes— exactly 4 digits                                     (1120, 2222)
#   suffixed tickers — above + .XX suffix                                   (EMAAR.AE, TJARI.TN)
#   commodity codes  — letter-starts + =F                                   (GC=F, CL=F)
_TOKEN_RE = re.compile(
    r'\b(?:'
    r'[A-Z][A-Z0-9]{1,7}(?:\.[A-Z]{2})?(?:=F)?'   # equity / commodity
    r'|\d{4}'                                        # 4-digit KSA codes
    r')\b'
)


def lookup(token: str) -> Optional[TickerMatch]:
    """O(1) lookup for a single uppercase token. Returns None if not found."""
    _ensure_index()
    return _INDEX.get(token.upper())


def quick_scan(query: str) -> Optional[TickerMatch]:
    """
    Scan *query* for any token matching a known ticker.

    Returns the first (longest) match found, or None.
    Skips common English stopwords to reduce false positives.
    """
    _ensure_index()
    if not query or not _INDEX:
        return None

    upper_query = query.upper()
    tokens = _TOKEN_RE.findall(upper_query)

    # Deduplicate, filter stopwords, sort longest-first (more specific wins)
    candidates = sorted(
        {t for t in tokens if t not in _STOPWORDS},
        key=len,
        reverse=True,
    )

    for tok in candidates:
        match = _INDEX.get(tok)
        if match:
            logger.debug("[ticker_index] '%s' matched → %s (%s)", tok, match.symbol, match.market)
            return match

    return None


def index_size() -> int:
    """Return number of lookup keys in the index (for diagnostics)."""
    _ensure_index()
    return len(_INDEX)


def reload() -> None:
    """Force a full index rebuild (call after market_cache refresh)."""
    global _INDEX, _INDEX_READY
    with _INDEX_LOCK:
        _INDEX_READY = False
        _INDEX = _build_index()
        _INDEX_READY = True
