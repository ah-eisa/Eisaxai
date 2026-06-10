"""
EisaX Price Cache
TTL: 5 min للأسهم العادية، 1 min للـ crypto
Shared via SQLite عشان الـ 4 workers يشاركوه
"""
import sqlite3, time, logging, os
logger = logging.getLogger(__name__)

from core.config import PRICE_CACHE_DB as _cfg_pc_db
DB = str(_cfg_pc_db)
TTL_STOCK  = 300   # 5 دقايق
TTL_CRYPTO = 60    # دقيقة

def _init():
    with sqlite3.connect(DB) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT PRIMARY KEY,
                price  REAL,
                currency TEXT,
                fetched_at REAL
            )
        """)
_init()

def get(ticker: str) -> float | None:
    try:
        is_crypto = any(ticker.upper().startswith(x) for x in ["BTC","ETH","BNB","SOL","XRP","DOGE"])
        ttl = TTL_CRYPTO if is_crypto else TTL_STOCK
        with sqlite3.connect(DB) as c:
            row = c.execute("SELECT price, fetched_at FROM prices WHERE ticker=?", (ticker.upper(),)).fetchone()
            if row and (time.time() - row[1]) < ttl:
                logger.debug(f"[PriceCache] HIT {ticker} → {row[0]}")
                return float(row[0])
    except Exception as e:
        logger.warning(f"[PriceCache] get failed: {e}")
    return None

def set(ticker: str, price: float, currency: str = "USD"):
    try:
        with sqlite3.connect(DB) as c:
            c.execute("""
                INSERT OR REPLACE INTO prices (ticker, price, currency, fetched_at)
                VALUES (?, ?, ?, ?)
            """, (ticker.upper(), price, currency, time.time()))
        logger.debug(f"[PriceCache] SET {ticker} → {price}")
    except Exception as e:
        logger.warning(f"[PriceCache] set failed: {e}")

def invalidate(ticker: str):
    try:
        with sqlite3.connect(DB) as c:
            c.execute("DELETE FROM prices WHERE ticker=?", (ticker.upper(),))
    except Exception:
        pass
