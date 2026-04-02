"""
News filter result cache — 30 min TTL
Avoids re-filtering same articles on repeated ticker queries.
"""
import sqlite3
import time
import json
import hashlib
import logging
from core.config import BASE_DIR

logger = logging.getLogger(__name__)
DB = str(BASE_DIR / "news_filter_cache.db")
TTL = 1800  # 30 minutes


def _init():
    with sqlite3.connect(DB) as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key        TEXT PRIMARY KEY,
                result     TEXT,
                fetched_at REAL
            )
            """
        )


_init()


def _make_key(ticker: str, bucket: str, headlines: list) -> str:
    raw = f"{ticker.upper()}:{bucket}:" + "|".join(sorted(headlines))
    return hashlib.md5(raw.encode()).hexdigest()


def get(ticker: str, bucket: str, headlines: list):
    try:
        key = _make_key(ticker, bucket, headlines)
        with sqlite3.connect(DB) as c:
            row = c.execute("SELECT result, fetched_at FROM cache WHERE key=?", (key,)).fetchone()
            if row and (time.time() - row[1]) < TTL:
                logger.info(f"[NewsFilterCache] HIT {ticker}/{bucket}")
                return json.loads(row[0])
    except Exception as e:
        logger.debug(f"[NewsFilterCache] get failed: {e}")
    return None


def set(ticker: str, bucket: str, headlines: list, result: list):
    try:
        key = _make_key(ticker, bucket, headlines)
        with sqlite3.connect(DB) as c:
            c.execute(
                "INSERT OR REPLACE INTO cache (key, result, fetched_at) VALUES (?,?,?)",
                (key, json.dumps(result), time.time()),
            )
        logger.info(f"[NewsFilterCache] SET {ticker}/{bucket} ({len(result)} articles)")
    except Exception as e:
        logger.debug(f"[NewsFilterCache] set failed: {e}")
