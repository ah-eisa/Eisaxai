"""
EisaX Forex Module — H-4
─────────────────────────
Fetches major FX pairs with emphasis on Arab-relevant currencies.
Uses yfinance as primary, exchangerate-api.com as fallback.

Usage
─────
    from core.forex import ForexFetcher
    data = ForexFetcher().fetch()   # list[dict]
"""

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Pair definitions ──────────────────────────────────────────────────────────
# (yfinance_symbol, display_name, base, quote, category)
FOREX_PAIRS = [
    # Arab-relevant (most important for target market)
    ("USDAED=X",  "USD/AED",  "USD", "AED", "arab"),
    ("USDSAR=X",  "USD/SAR",  "USD", "SAR", "arab"),
    ("USDEGP=X",  "USD/EGP",  "USD", "EGP", "arab"),
    ("USDKWD=X",  "USD/KWD",  "USD", "KWD", "arab"),
    ("USDQAR=X",  "USD/QAR",  "USD", "QAR", "arab"),
    ("USDBHD=X",  "USD/BHD",  "USD", "BHD", "arab"),
    # Global majors
    ("EURUSD=X",  "EUR/USD",  "EUR", "USD", "major"),
    ("GBPUSD=X",  "GBP/USD",  "GBP", "USD", "major"),
    ("USDJPY=X",  "USD/JPY",  "USD", "JPY", "major"),
    ("USDCHF=X",  "USD/CHF",  "USD", "CHF", "major"),
    ("AUDUSD=X",  "AUD/USD",  "AUD", "USD", "major"),
    ("USDCAD=X",  "USD/CAD",  "USD", "CAD", "major"),
    # Emerging / commodity currencies
    ("USDTRY=X",  "USD/TRY",  "USD", "TRY", "em"),
    ("USDZAR=X",  "USD/ZAR",  "USD", "ZAR", "em"),
    ("USDINR=X",  "USD/INR",  "USD", "INR", "em"),
    ("USDPKR=X",  "USD/PKR",  "USD", "PKR", "em"),
]

# Fallback rates (used when all APIs fail) — approximate mid rates
_FALLBACK_RATES: dict[str, float] = {
    "USDAED=X": 3.6725, "USDSAR=X": 3.7500, "USDEGP=X": 50.90,
    "USDKWD=X": 0.3070, "USDQAR=X": 3.6400, "USDBHD=X": 0.3770,
    "EURUSD=X":  1.0850, "GBPUSD=X": 1.2700, "USDJPY=X": 149.50,
    "USDCHF=X":  0.9050, "AUDUSD=X": 0.6500, "USDCAD=X": 1.3600,
    "USDTRY=X": 32.00,  "USDZAR=X": 18.50,  "USDINR=X": 83.50,
    "USDPKR=X": 278.0,
}


class ForexFetcher:
    """Fetch live FX rates with Redis caching + yfinance backend."""

    CACHE_TTL = 300   # 5 minutes

    def __init__(self) -> None:
        self._cache_key = "forex:rates"

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _from_cache(self):
        try:
            from core.redis_store import cache_get
            import json
            raw = cache_get(self._cache_key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def _to_cache(self, data) -> None:
        try:
            from core.redis_store import cache_set
            import json
            cache_set(self._cache_key, json.dumps(data), ttl_seconds=self.CACHE_TTL)
        except Exception:
            pass

    # ── yfinance fetch ────────────────────────────────────────────────────────

    def _fetch_yfinance(self, symbol: str) -> tuple[float | None, float | None]:
        """Returns (price, prev_close) or (None, None) on failure."""
        try:
            import yfinance as yf
            info = yf.Ticker(symbol).fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
            prev  = getattr(info, "previous_close", None)
            if price and price > 0:
                return float(price), float(prev) if prev else None
        except Exception as exc:
            logger.debug("[forex] yfinance %s: %s", symbol, exc)
        return None, None

    # ── exchangerate-api fallback ─────────────────────────────────────────────

    def _fetch_exchangerate_api(self, base: str, quote: str) -> float | None:
        """Free tier: 1500 requests/month. Used as last-resort fallback."""
        api_key = os.getenv("EXCHANGERATE_API_KEY", "")
        if not api_key:
            return None
        try:
            import urllib.request, json as _json
            url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base}/{quote}"
            with urllib.request.urlopen(url, timeout=5) as r:
                data = _json.loads(r.read())
                if data.get("result") == "success":
                    return float(data["conversion_rate"])
        except Exception as exc:
            logger.debug("[forex] exchangerate-api %s/%s: %s", base, quote, exc)
        return None

    # ── Main fetch ────────────────────────────────────────────────────────────

    def fetch(self, use_cache: bool = True) -> list[dict]:
        """
        Fetch all FX pairs.

        Returns
        -------
        list of {
          symbol, name, base, quote, category,
          price, prev_close, change_pct,
          source, timestamp
        }
        """
        if use_cache:
            cached = self._from_cache()
            if cached:
                return cached

        results = []
        now = datetime.now(timezone.utc).isoformat()

        for symbol, name, base, quote, category in FOREX_PAIRS:
            price, prev = self._fetch_yfinance(symbol)

            # Fallback: exchangerate-api (only for USD pairs)
            if price is None and base == "USD":
                rate = self._fetch_exchangerate_api(base, quote)
                if rate:
                    price = rate
                    prev  = None
                    source = "exchangerate-api"
                else:
                    source = "fallback"
                    price = _FALLBACK_RATES.get(symbol)
                    prev  = None
            else:
                source = "yfinance" if price else "fallback"
                if price is None:
                    price = _FALLBACK_RATES.get(symbol)
                    prev  = None

            change_pct = None
            if price and prev and prev > 0:
                change_pct = round((price - prev) / prev * 100, 4)

            results.append({
                "symbol":     symbol,
                "name":       name,
                "base":       base,
                "quote":      quote,
                "category":   category,
                "price":      round(float(price), 6) if price else None,
                "prev_close": round(float(prev), 6) if prev else None,
                "change_pct": change_pct,
                "source":     source,
                "timestamp":  now,
            })

        self._to_cache(results)
        return results

    def get_pair(self, symbol: str, use_cache: bool = True) -> dict | None:
        """Fetch a single pair by symbol (e.g. 'USDAED=X')."""
        all_pairs = self.fetch(use_cache=use_cache)
        for p in all_pairs:
            if p["symbol"] == symbol:
                return p
        return None

    def get_usd_to(self, currency: str, use_cache: bool = True) -> float | None:
        """Convenience: get USD → currency rate. e.g. get_usd_to('AED') → 3.6725"""
        symbol = f"USD{currency}=X"
        pair = self.get_pair(symbol, use_cache=use_cache)
        return pair["price"] if pair else None
