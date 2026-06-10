"""
EisaX Database Abstraction Layer
Provides thread-safe connection pooling and context managers for SQLite.
Replaces scattered sqlite3.connect() calls with a centralized pool.
"""
import sqlite3
import logging
import threading
from queue import Queue, Empty
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

from core.config import APP_DB as _cfg_app_db, CORE_DB as _cfg_core_db
_DEFAULT_DB = str(_cfg_app_db)
_BRAIN_DB   = str(_cfg_core_db)


class ConnectionPool:
    """
    Thread-safe SQLite connection pool using queue.Queue.

    - Pre-configures WAL mode, busy timeout, and synchronous=NORMAL.
    - Context manager auto-commits on success, rolls back on exception.
    - Connections are reused across requests (returned to pool on exit).
    """

    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created = 0

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new connection with optimized pragmas."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _get_raw(self) -> sqlite3.Connection:
        """Get a connection from pool or create new one (up to pool_size)."""
        # Try to get from pool first (non-blocking)
        try:
            conn = self._pool.get_nowait()
            # Verify connection is still valid
            try:
                conn.execute("SELECT 1")
                return conn
            except Exception:
                # Connection went stale, create new one
                with self._lock:
                    self._created -= 1
        except Empty:
            pass

        # Create new connection if under limit
        with self._lock:
            if self._created < self.pool_size:
                self._created += 1
                return self._create_connection()

        # Pool exhausted, wait for one to be returned (blocking, 30s timeout)
        try:
            conn = self._pool.get(timeout=30)
            try:
                conn.execute("SELECT 1")
                return conn
            except Exception:
                with self._lock:
                    self._created -= 1
                return self._create_connection()
        except Empty:
            # Last resort: create even if over limit
            logger.warning("[db] Connection pool exhausted (%d), creating overflow connection", self.pool_size)
            return self._create_connection()

    def _return(self, conn: sqlite3.Connection):
        """Return a connection to the pool."""
        try:
            self._pool.put_nowait(conn)
        except Exception:
            # Pool full, close the connection
            try:
                conn.close()
            except Exception:
                pass

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Usage:
            with pool.get_connection() as conn:
                conn.execute("INSERT INTO ...")
                # auto-commits on success, auto-rollbacks on exception

        The connection is returned to the pool afterwards.
        """
        conn = self._get_raw()
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self._return(conn)

    @contextmanager
    def get_cursor(self):
        """
        Convenience: yields (conn, cursor) pair.

        Usage:
            with pool.get_cursor() as (conn, c):
                c.execute("SELECT ...")
                rows = c.fetchall()
        """
        with self.get_connection() as conn:
            yield conn, conn.cursor()

    def close_all(self):
        """Close all pooled connections (for shutdown)."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                pass


# ── Singleton Pools ──────────────────────────────────────────────────────────
# Main database (sessions, chat_history, admin, user_memory, stock_memory)
db = ConnectionPool(_DEFAULT_DB, pool_size=5)

# Brain database (user_profiles with watchlists, sectors, risk profiles)
brain_db = ConnectionPool(_BRAIN_DB, pool_size=3)
