"""
EisaX Analysis Cache
TTL: 60 min للأسهم، 15 min للـ crypto
Shared via SQLite عشان الـ 4 workers يشاركوه
"""
import sqlite3
import time
import hashlib
import logging
from core.config import BASE_DIR

logger = logging.getLogger(__name__)

DB = str(BASE_DIR / "analysis_cache.db")
TTL_STOCK = 3600   # 60 دقيقة
TTL_CRYPTO = 900   # 15 دقيقة

_CRYPTO_PREFIXES = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX"]


def _init():
    with sqlite3.connect(DB) as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                cache_key  TEXT PRIMARY KEY,
                ticker     TEXT,
                reply      TEXT,
                model      TEXT,
                fetched_at REAL
            )
            """
        )


_init()


def _is_crypto(ticker: str) -> bool:
    return any(ticker.upper().startswith(x) for x in _CRYPTO_PREFIXES)


def _ttl(ticker: str) -> int:
    return TTL_CRYPTO if _is_crypto(ticker) else TTL_STOCK


def make_key(ticker: str, mode: str = "full") -> str:
    return hashlib.md5(f"{ticker.upper()}:{mode}".encode()).hexdigest()


def get(ticker: str, mode: str = "full") -> dict | None:
    try:
        key = make_key(ticker, mode)
        ttl = _ttl(ticker)
        with sqlite3.connect(DB) as c:
            row = c.execute(
                "SELECT reply, model, fetched_at FROM analyses WHERE cache_key=?",
                (key,),
            ).fetchone()
            if row and (time.time() - row[2]) < ttl:
                age = int(time.time() - row[2])
                logger.info(f"[AnalysisCache] HIT {ticker} (age {age}s)")
                return {"reply": row[0], "model": row[1], "from_cache": True, "cache_age": age}
    except Exception as e:
        logger.warning(f"[AnalysisCache] get failed: {e}")
    return None


def set(ticker: str, reply: str, model: str = "deepseek", mode: str = "full"):
    try:
        key = make_key(ticker, mode)
        with sqlite3.connect(DB) as c:
            c.execute(
                """
                INSERT OR REPLACE INTO analyses (cache_key, ticker, reply, model, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, ticker.upper(), reply, model, time.time()),
            )
        logger.info(f"[AnalysisCache] SET {ticker} ({len(reply)} chars)")
    except Exception as e:
        logger.warning(f"[AnalysisCache] set failed: {e}")


def invalidate(ticker: str, mode: str = "full"):
    try:
        key = make_key(ticker, mode)
        with sqlite3.connect(DB) as c:
            c.execute("DELETE FROM analyses WHERE cache_key=?", (key,))
        logger.info(f"[AnalysisCache] INVALIDATED {ticker} ({mode})")
    except Exception as e:
        logger.warning(f"[AnalysisCache] invalidate failed: {e}")


def cleanup():
    """Delete expired entries"""
    try:
        with sqlite3.connect(DB) as c:
            now = time.time()
            c.execute("DELETE FROM analyses WHERE fetched_at < ?", (now - TTL_STOCK,))
        logger.info("[AnalysisCache] Cleanup done")
    except Exception as e:
        logger.warning(f"[AnalysisCache] cleanup failed: {e}")
