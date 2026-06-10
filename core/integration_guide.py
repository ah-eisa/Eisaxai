"""
EisaX Ticker Resolver
======================
Resolves user input (Arabic/English) to Yahoo Finance ticker symbols.
Handles fuzzy matching, partial names, and common aliases.

Usage:
    from core.ticker_resolver import TickerResolver
    
    resolver = TickerResolver()
    results = resolver.resolve("أرامكو")
    # → [{"ticker": "2222.SR", "name_ar": "أرامكو", "name_en": "Saudi Aramco", ...}]
"""

import re
import unicodedata
from typing import Optional
from core.local_tickers import MARKET_DB, MARKET_INDICES, get_all_tickers_flat


class TickerResolver:
    """Resolves Arabic/English company names to Yahoo Finance tickers."""

    def __init__(self):
        self._build_lookup_index()

    def _build_lookup_index(self):
        """Build inverted index: normalized_alias → ticker"""
        self._alias_map = {}       # normalized string → ticker
        self._ticker_info = {}     # ticker → full info

        all_tickers = get_all_tickers_flat()

        # Also include market indices
        for ticker, info in MARKET_INDICES.items():
            all_tickers[ticker] = {**info, "market": "indices"}

        for ticker, info in all_tickers.items():
            self._ticker_info[ticker] = info

            # Index all names and aliases
            names_to_index = []

            # Primary names
            if "name_en" in info:
                names_to_index.append(info["name_en"])
            if "name_ar" in info:
                names_to_index.append(info["name_ar"])

            # Aliases
            names_to_index.extend(info.get("aliases_ar", []))
            names_to_index.extend(info.get("aliases_en", []))

            # Also index the raw ticker itself
            names_to_index.append(ticker)
            # And ticker without suffix
            base_ticker = ticker.split(".")[0]
            names_to_index.append(base_ticker)

            for name in names_to_index:
                normalized = self._normalize(name)
                if normalized:
                    self._alias_map[normalized] = ticker

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Normalize text for matching:
        - Lowercase
        - Remove diacritics (tashkeel)
        - Normalize Arabic characters (alef variants, taa marbuta, etc.)
        - Strip extra whitespace
        """
        if not text:
            return ""

        text = text.lower().strip()

        # Remove Arabic diacritics (tashkeel)
        arabic_diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')
        text = arabic_diacritics.sub('', text)

        # Normalize Arabic characters
        # Alef variants → ا
        text = re.sub(r'[إأآٱ]', 'ا', text)
        # Taa marbuta → ه
        text = text.replace('ة', 'ه')
        # Alef maksura → ي
        text = text.replace('ى', 'ي')
        # Waw with hamza
        text = text.replace('ؤ', 'و')
        # Ya with hamza
        text = text.replace('ئ', 'ي')

        # Remove non-alphanumeric (keep Arabic + English letters + digits)
        text = re.sub(r'[^\w\u0600-\u06FF]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def resolve(self, query: str, market: Optional[str] = None) -> list:
        """
        Resolve user query to matching tickers.
        
        Args:
            query: User input (Arabic or English)
            market: Optional filter ('saudi', 'egypt', 'uae')
            
        Returns:
            List of dicts with ticker info, sorted by relevance
        """
        normalized_query = self._normalize(query)
        if not normalized_query:
            return []

        results = []

        # 1. Exact match
        if normalized_query in self._alias_map:
            ticker = self._alias_map[normalized_query]
            info = self._ticker_info[ticker]
            if not market or info.get("market") == market:
                results.append({"ticker": ticker, **info, "match_type": "exact"})

        # 2. Check if query is already a valid ticker format
        query_upper = query.upper().strip()
        for suffix in [".SR", ".CA", ".AE", ".DU"]:
            if query_upper.endswith(suffix) and query_upper in self._ticker_info:
                if not any(r["ticker"] == query_upper for r in results):
                    info = self._ticker_info[query_upper]
                    if not market or info.get("market") == market:
                        results.append({"ticker": query_upper, **info, "match_type": "exact"})

        # 3. Partial / contains match
        if not results:
            for alias, ticker in self._alias_map.items():
                info = self._ticker_info[ticker]
                if market and info.get("market") != market:
                    continue

                # Check if query is contained in alias or vice versa
                if normalized_query in alias or alias in normalized_query:
                    if not any(r["ticker"] == ticker for r in results):
                        results.append({"ticker": ticker, **info, "match_type": "partial"})

        # 4. Word-level match (for multi-word queries)
        if not results:
            query_words = set(normalized_query.split())
            for alias, ticker in self._alias_map.items():
                info = self._ticker_info[ticker]
                if market and info.get("market") != market:
                    continue

                alias_words = set(alias.split())
                # If any significant word matches
                common = query_words & alias_words
                if common and len(common) >= 1:
                    # Filter out very short common words
                    significant = [w for w in common if len(w) > 2]
                    if significant and not any(r["ticker"] == ticker for r in results):
                        results.append({"ticker": ticker, **info, "match_type": "word"})

        return results[:10]  # Limit to top 10

    def resolve_single(self, query: str, market: Optional[str] = None) -> Optional[str]:
        """
        Resolve to a single best-match ticker string.
        Returns None if no match.
        """
        results = self.resolve(query, market)
        if results:
            return results[0]["ticker"]
        return None

    def resolve_multiple(self, queries: list, market: Optional[str] = None) -> dict:
        """
        Resolve multiple queries at once.
        Returns: {query: ticker_or_None}
        """
        return {q: self.resolve_single(q, market) for q in queries}

    def is_local_ticker(self, text: str) -> bool:
        """Check if text matches any local market ticker or name."""
        return bool(self.resolve(text))

    def get_market_for_ticker(self, ticker: str) -> Optional[str]:
        """Returns the market name for a given ticker."""
        info = self._ticker_info.get(ticker)
        return info.get("market") if info else None

    def get_ticker_info(self, ticker: str) -> Optional[dict]:
        """Returns full info for a ticker."""
        return self._ticker_info.get(ticker)

    def search_by_sector(self, sector_query: str, market: Optional[str] = None) -> list:
        """Search tickers by sector name (Arabic or English)."""
        normalized = self._normalize(sector_query)
        results = []
        for ticker, info in self._ticker_info.items():
            if market and info.get("market") != market:
                continue
            sector_en = self._normalize(info.get("sector", ""))
            sector_ar = self._normalize(info.get("sector_ar", ""))
            if normalized in sector_en or normalized in sector_ar:
                results.append({"ticker": ticker, **info})
        return results

    def list_markets(self) -> dict:
        """List available markets and their ticker counts."""
        return {
            market: len(tickers)
            for market, tickers in MARKET_DB.items()
        }


# ═══════════════════════════════════════════════════════════════
#  PATTERN DETECTION — for integration with intent_classifier.py
# ═══════════════════════════════════════════════════════════════

# Regex patterns for local market tickers
LOCAL_TICKER_PATTERNS = [
    # Saudi: 4-digit number + .SR
    r'\b\d{4}\.SR\b',
    # Egypt: 2-5 uppercase letters + .CA
    r'\b[A-Z]{2,6}\.CA\b',
    # UAE ADX: uppercase letters + .AE
    r'\b[A-Z]{2,12}\.AE\b',
    # UAE DFM: uppercase letters + .DU
    r'\b[A-Z]{2,12}\.DU\b',
]

COMBINED_LOCAL_PATTERN = re.compile(
    '|'.join(LOCAL_TICKER_PATTERNS),
    re.IGNORECASE
)


def extract_local_tickers_from_text(text: str) -> list:
    """
    Extract local market ticker patterns from text.
    Returns list of ticker strings found.
    """
    matches = COMBINED_LOCAL_PATTERN.findall(text.upper())
    return list(set(matches))


# Common Arabic keywords that indicate market/stock queries
ARABIC_STOCK_KEYWORDS = [
    "سهم", "أسهم", "اسهم",
    "سعر", "أسعار",
    "تحليل", "حلل",
    "بورصة", "سوق",
    "تداول", "تاسي",
    "محفظة",
    "شراء", "بيع",
    "ارتفاع", "انخفاض", "هبوط",
    "أرباح", "توزيعات",
    "مكرر", "ربحية",
    "قطاع",
    "مؤشر",
]

ARABIC_STOCK_KEYWORDS_PATTERN = re.compile(
    '|'.join(ARABIC_STOCK_KEYWORDS)
)


def has_arabic_stock_context(text: str) -> bool:
    """Check if text contains Arabic stock/market keywords."""
    return bool(ARABIC_STOCK_KEYWORDS_PATTERN.search(text))


# ═══════════════════════════════════════════════════════════════
#  QUICK TEST
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    resolver = TickerResolver()

    test_queries = [
        "أرامكو",
        "الراجحي",
        "aramco",
        "sabic",
        "2222.SR",
        "سي آي بي",
        "CIB",
        "طلعت مصطفى",
        "إعمار",
        "اتصالات",
        "تاسي",
        "بنك الرياض",
        "المراعي",
        "هيرميس",
        "fab",
        "stc",
    ]

    print("=" * 60)
    print("EisaX Ticker Resolver — Test Results")
    print("=" * 60)

    for q in test_queries:
        results = resolver.resolve(q)
        if results:
            top = results[0]
            print(f"  '{q}' → {top['ticker']} ({top.get('name_en', '?')}) [{top['match_type']}]")
        else:
            print(f"  '{q}' → ❌ No match")

    print(f"\n📊 Markets: {resolver.list_markets()}")
