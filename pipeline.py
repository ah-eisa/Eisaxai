"""
EisaX Market Data Pipeline  v1.0
=================================
CacheManager  ← parquet snapshots (rolling 4 per market)
MarketFetcher ← tradingview-screener + yfinance
Scheduler     ← every 15 min, background thread

Usage:
    # run as standalone service
    python pipeline.py

    # import in app
    from pipeline import scheduler, cache, fetcher
    scheduler.start(fetch_now=False)   # start background scheduler
    df, ts = cache.get_latest("uae")   # read cache instantly
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import schedule
import yfinance as yf
from tradingview_screener import Query

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CACHE_DIR  = BASE_DIR / "market_cache"
INDEX_FILE = CACHE_DIR / "index.json"
LOG_FILE   = BASE_DIR / "pipeline.log"

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_SNAPSHOTS       = 4   # rolling window per market
FETCH_INTERVAL_MIN  = 15
STALE_THRESHOLD_MIN = 30  # fetch first if cache older than this

# ── Logging ────────────────────────────────────────────────────────────────────
log = logging.getLogger("eisax.pipeline")
if not log.handlers:
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)

# ── Market Definitions ─────────────────────────────────────────────────────────
TV_MARKETS = {
    "uae":      {"name": "UAE",            "limit": 500},
    "ksa":      {"name": "Saudi Arabia",   "limit": 500},
    "egypt":    {"name": "Egypt",          "limit": 500},
    "kuwait":   {"name": "Kuwait",         "limit": 500},
    "qatar":    {"name": "Qatar",          "limit": 500},
    "bahrain":  {"name": "Bahrain",        "limit": 500},
    "morocco":  {"name": "Morocco",        "limit": 500},
    "tunisia":  {"name": "Tunisia",        "limit": 500},
    "america":  {"name": "USA Top 500",    "limit": 500},
    "crypto":   {"name": "Crypto Top 200", "limit": 200},
}

COMMODITIES = {
    "GC=F": "Gold",
    "CL=F": "Crude Oil (WTI)",
    "BZ=F": "Brent Crude",
    "SI=F": "Silver",
    "NG=F": "Natural Gas",
    "HG=F": "Copper",
    "PL=F": "Platinum",
}

# Fields to pull from TradingView for every equity/crypto market
TV_FIELDS = [
    "name", "close", "change", "volume", "market_cap_basic",
    "price_earnings_ttm", "dividend_yield_recent",
    "earnings_per_share_diluted_ttm", "sector",
    "RSI", "MACD.macd", "MACD.signal",
    "Stoch.K", "Stoch.D", "CCI20", "AO",
    "SMA50", "SMA200",
]

CRYPTO_FIELDS = [
    "name", "close", "change", "volume", "market_cap_basic",
    "RSI", "MACD.macd", "MACD.signal", "SMA50", "SMA200",
]


# ══════════════════════════════════════════════════════════════════════════════
# CacheManager
# ══════════════════════════════════════════════════════════════════════════════

class CacheManager:
    """
    Manages parquet snapshots in market_cache/.
    Keeps the last MAX_SNAPSHOTS per market, auto-deletes older ones.
    Thread-safe via a simple file lock.
    """

    def __init__(self):
        CACHE_DIR.mkdir(exist_ok=True)
        if not INDEX_FILE.exists():
            INDEX_FILE.write_text(json.dumps({}))
        self._lock = threading.Lock()

    # ── Index helpers ──────────────────────────────────────────────────────────

    def _load_index(self) -> dict:
        try:
            return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_index(self, index: dict):
        INDEX_FILE.write_text(
            json.dumps(index, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8"
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def save_snapshot(self, market: str, df: pd.DataFrame) -> str:
        """
        Save DataFrame as parquet. Prune oldest if > MAX_SNAPSHOTS.
        Returns filename.
        """
        ts_str = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{market}_{ts_str}.parquet"
        filepath = CACHE_DIR / filename

        # stamp metadata columns
        df = df.copy()
        df["_market"]      = market
        df["_snapshot_ts"] = datetime.now().isoformat()

        df.to_parquet(filepath, index=False)

        with self._lock:
            index = self._load_index()
            if market not in index:
                index[market] = []

            index[market].append({
                "filename":  filename,
                "timestamp": datetime.now().isoformat(),
                "count":     len(df),
            })

            # rolling window: delete oldest
            if len(index[market]) > MAX_SNAPSHOTS:
                old_entries = index[market][:-MAX_SNAPSHOTS]
                for entry in old_entries:
                    old_path = CACHE_DIR / entry["filename"]
                    if old_path.exists():
                        old_path.unlink()
                        log.info(f"🗑️  Pruned old snapshot: {entry['filename']}")
                index[market] = index[market][-MAX_SNAPSHOTS:]

            self._save_index(index)

        log.info(f"💾 Saved [{market}] → {filename} ({len(df)} rows)")
        return filename

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_latest(self, market: str) -> tuple:
        """
        Returns (DataFrame, iso_timestamp_str) or (None, None) if no cache.
        """
        index = self._load_index()
        entries = index.get(market, [])
        if not entries:
            return None, None

        latest = entries[-1]
        filepath = CACHE_DIR / latest["filename"]
        if not filepath.exists():
            log.warning(f"⚠️  Index points to missing file: {latest['filename']}")
            return None, None

        try:
            df = pd.read_parquet(filepath)
            return df, latest["timestamp"]
        except Exception as e:
            log.error(f"❌ Failed to read parquet {filepath}: {e}")
            return None, None

    def get_snapshots(self, market: str) -> list:
        """Returns list of snapshot metadata dicts for a market."""
        return self._load_index().get(market, [])

    # ── Age helpers ────────────────────────────────────────────────────────────

    def cache_age_minutes(self, market: str) -> float | None:
        """Age of latest snapshot in minutes. None if no cache."""
        _, ts = self.get_latest(market)
        if ts is None:
            return None
        dt = datetime.fromisoformat(ts)
        return (datetime.now() - dt).total_seconds() / 60

    def is_stale(self, market: str) -> bool:
        """True if cache is missing or older than STALE_THRESHOLD_MIN."""
        age = self.cache_age_minutes(market)
        return age is None or age > STALE_THRESHOLD_MIN

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Returns a status dict for all cached markets."""
        index = self._load_index()
        result = {}
        for market, entries in index.items():
            if not entries:
                continue
            latest = entries[-1]
            age = self.cache_age_minutes(market)
            result[market] = {
                "snapshots":   len(entries),
                "latest":      latest["timestamp"],
                "count":       latest["count"],
                "age_minutes": round(age, 1) if age is not None else None,
                "stale":       self.is_stale(market),
            }
        return result


# ══════════════════════════════════════════════════════════════════════════════
# MarketFetcher
# ══════════════════════════════════════════════════════════════════════════════

class MarketFetcher:
    """
    Fetches market data from TradingView Screener (equities + crypto)
    and yfinance (commodities). Saves results via CacheManager.
    """

    def __init__(self, cache: CacheManager):
        self.cache = cache

    # ── TradingView ────────────────────────────────────────────────────────────

    def _fetch_tv(self, market_code: str) -> pd.DataFrame | None:
        cfg = TV_MARKETS[market_code]
        try:
            log.info(f"📡 Fetching [{cfg['name']}] from TradingView...")
            fields = CRYPTO_FIELDS if market_code == "crypto" else TV_FIELDS
            _, df = (
                Query()
                .set_markets(market_code)
                .select(*fields)
                .limit(cfg["limit"])
                .get_scanner_data()
            )
            if market_code == "crypto":
                equity_only = [
                    "price_earnings_ttm",
                    "dividend_yield_recent",
                    "earnings_per_share_diluted_ttm",
                ]
                for col in equity_only:
                    if col not in df.columns:
                        df[col] = None
            log.info(f"   ✅ {len(df)} rows")
            return df
        except Exception as e:
            log.error(f"   ❌ TradingView [{market_code}]: {e}")
            return None

    # ── Commodities (yfinance) ─────────────────────────────────────────────────

    def _fetch_commodities(self) -> pd.DataFrame | None:
        log.info("📡 Fetching [Commodities] from yfinance...")
        rows = []
        for ticker, commodity_name in COMMODITIES.items():
            try:
                info  = yf.Ticker(ticker).fast_info
                price = info.last_price or 0
                prev  = info.previous_close or price
                change = round((price - prev) / prev * 100, 2) if prev else 0.0
                rows.append({
                    "ticker":            ticker,
                    "name":              commodity_name,
                    "close":             round(price, 4),
                    "change":            change,
                    "volume":            getattr(info, "three_month_average_volume", None),
                    "market_cap_basic":  None,
                    "sector":            "Commodities",
                    # technicals not available via fast_info
                    "RSI": None, "MACD.macd": None, "MACD.signal": None,
                    "Stoch.K": None, "Stoch.D": None, "CCI20": None, "AO": None,
                    "SMA50": None, "SMA200": None,
                    "price_earnings_ttm": None,
                    "dividend_yield_recent": None,
                    "earnings_per_share_diluted_ttm": None,
                })
            except Exception as e:
                log.warning(f"   ⚠️  {ticker}: {e}")

        if rows:
            df = pd.DataFrame(rows)
            log.info(f"   ✅ {len(df)} commodities")
            return df

        log.error("   ❌ Commodities: no data returned")
        return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch_market(self, market_code: str) -> bool:
        """
        Fetch one market and cache it.
        Returns True on success.
        """
        if market_code == "commodities":
            df = self._fetch_commodities()
        elif market_code in TV_MARKETS:
            df = self._fetch_tv(market_code)
        else:
            log.error(f"Unknown market: {market_code}")
            return False

        if df is not None and len(df) > 0:
            self.cache.save_snapshot(market_code, df)
            return True

        return False

    def fetch_all(self) -> dict:
        """
        Fetch all markets sequentially.
        Returns {market_code: True/False}.
        """
        all_markets = list(TV_MARKETS.keys()) + ["commodities"]
        results = {}

        log.info("=" * 60)
        log.info(f"🌍 Starting full fetch — {len(all_markets)} markets")
        log.info("=" * 60)

        for market in all_markets:
            results[market] = self.fetch_market(market)
            time.sleep(1.5)  # polite delay between requests

        ok  = sum(results.values())
        bad = len(results) - ok
        log.info("=" * 60)
        log.info(f"🏁 Fetch complete: ✅ {ok} succeeded  ❌ {bad} failed")
        log.info("=" * 60)

        return results

    def fetch_if_stale(self, market_code: str) -> bool:
        """
        Fetch only if cache is missing or older than STALE_THRESHOLD_MIN.
        Returns True if fresh data is now available.
        """
        if self.cache.is_stale(market_code):
            log.info(f"⚡ Cache stale for [{market_code}] — fetching now...")
            return self.fetch_market(market_code)
        return True  # already fresh


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler
# ══════════════════════════════════════════════════════════════════════════════

class PipelineScheduler:
    """
    Runs MarketFetcher.fetch_all() every FETCH_INTERVAL_MIN minutes
    in a daemon background thread.

    Usage:
        scheduler.start()          # fetch immediately + schedule
        scheduler.start(fetch_now=False)  # schedule only
        scheduler.stop()
        scheduler.status()
    """

    def __init__(self, fetcher: MarketFetcher, cache: CacheManager):
        self.fetcher   = fetcher
        self.cache     = cache
        self._thread   = None
        self._running  = False
        self._last_run = None
        self._last_result = None

    def _run_job(self):
        self._last_run    = datetime.now()
        self._last_result = self.fetcher.fetch_all()

    def start(self, fetch_now: bool = True):
        """Start the background scheduler thread."""
        if self._running:
            log.warning("Scheduler already running.")
            return

        if fetch_now:
            log.info("🚀 Initial fetch before starting scheduler...")
            self._run_job()

        schedule.every(FETCH_INTERVAL_MIN).minutes.do(self._run_job)

        self._running = True
        self._thread  = threading.Thread(target=self._loop, name="EisaXPipeline", daemon=True)
        self._thread.start()
        log.info(f"⏱️  Scheduler started — every {FETCH_INTERVAL_MIN} minutes")

    def _loop(self):
        while self._running:
            schedule.run_pending()
            time.sleep(30)

    def stop(self):
        self._running = False
        schedule.clear()
        log.info("🛑 Scheduler stopped")

    def status(self) -> dict:
        return {
            "running":     self._running,
            "last_run":    self._last_run.isoformat() if self._last_run else None,
            "last_result": self._last_result,
            "cache":       self.cache.status(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Module-level singletons  (import these in your app)
# ══════════════════════════════════════════════════════════════════════════════

cache     = CacheManager()
fetcher   = MarketFetcher(cache)
scheduler = PipelineScheduler(fetcher, cache)


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry-point  (python pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EisaX Market Data Pipeline")
    parser.add_argument("--fetch-only",  action="store_true",
                        help="Run one full fetch then exit")
    parser.add_argument("--market",      type=str, default=None,
                        help="Fetch a single market (e.g. uae, ksa, crypto)")
    parser.add_argument("--status",      action="store_true",
                        help="Print cache status and exit")
    args = parser.parse_args()

    if args.status:
        import pprint
        pprint.pprint(cache.status())

    elif args.fetch_only:
        if args.market:
            fetcher.fetch_market(args.market)
        else:
            fetcher.fetch_all()

    else:
        # run as persistent service
        log.info("🛡️  EisaX Pipeline — running as service")
        scheduler.start(fetch_now=True)

        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            scheduler.stop()
            log.info("👋 Pipeline shut down cleanly")
