"""
Pytest configuration and shared fixtures for EisaX tests.
"""
import sys
import os
import sqlite3
import tempfile
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite database for testing."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.close()
    return db_path


@pytest.fixture
def test_session_manager(tmp_db, monkeypatch):
    """SessionManager connected to a temporary DB with proper isolation."""
    # Patch the db module's pool to use our temp DB
    from core.db import ConnectionPool
    test_pool = ConnectionPool(tmp_db, pool_size=2)

    import core.db
    monkeypatch.setattr(core.db, "db", test_pool)

    # Reload session_manager module to pick up the patched db
    import importlib
    if 'core.session_manager' in sys.modules:
        del sys.modules['core.session_manager']

    from core.session_manager import SessionManager
    mgr = SessionManager(db_path=tmp_db)

    # Clean up any stale data before test
    def cleanup_db():
        try:
            with test_pool.get_cursor() as (conn, c):
                # Delete all data from test tables
                c.execute("DELETE FROM admin_messages WHERE id > 0")
                c.execute("DELETE FROM chat_history")
                c.execute("DELETE FROM user_profiles")
                c.execute("DELETE FROM blocked_ips")
                c.execute("DELETE FROM sessions")
                c.execute("DELETE FROM admin_audit_log")
                conn.commit()
        except Exception as e:
            pass  # Tables might not exist yet

    cleanup_db()
    yield mgr

    # Clean up after test as well
    cleanup_db()
    test_pool.close_all()


@pytest.fixture
def test_db_pool(tmp_db):
    """A ConnectionPool instance pointed at a temp database."""
    from core.db import ConnectionPool
    pool = ConnectionPool(tmp_db, pool_size=3)
    yield pool
    pool.close_all()
