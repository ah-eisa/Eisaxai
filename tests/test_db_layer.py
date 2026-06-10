"""
Tests for the Database Abstraction Layer (core/db.py).
"""
import threading
import pytest


class TestConnectionPool:
    """Test the ConnectionPool class."""

    def test_get_connection_returns_working_connection(self, test_db_pool):
        """Connection from pool can execute queries."""
        with test_db_pool.get_connection() as conn:
            result = conn.execute("SELECT 1").fetchone()
            assert result[0] == 1

    def test_context_manager_auto_commits(self, test_db_pool):
        """Successful block auto-commits changes."""
        with test_db_pool.get_connection() as conn:
            conn.execute("CREATE TABLE test_commit (id INTEGER PRIMARY KEY, val TEXT)")
            conn.execute("INSERT INTO test_commit (val) VALUES ('hello')")

        # Verify data persisted
        with test_db_pool.get_connection() as conn:
            row = conn.execute("SELECT val FROM test_commit").fetchone()
            assert row[0] == "hello"

    def test_context_manager_auto_rollbacks_on_error(self, test_db_pool):
        """Failed block auto-rolls back changes."""
        with test_db_pool.get_connection() as conn:
            conn.execute("CREATE TABLE test_rollback (id INTEGER PRIMARY KEY, val TEXT)")

        with pytest.raises(ValueError):
            with test_db_pool.get_connection() as conn:
                conn.execute("INSERT INTO test_rollback (val) VALUES ('should_not_persist')")
                raise ValueError("test error")

        # Verify data was rolled back
        with test_db_pool.get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM test_rollback").fetchone()
            assert row[0] == 0

    def test_get_cursor_helper(self, test_db_pool):
        """get_cursor() yields (conn, cursor) tuple."""
        with test_db_pool.get_cursor() as (conn, c):
            c.execute("CREATE TABLE test_cursor (id INTEGER)")
            c.execute("INSERT INTO test_cursor VALUES (42)")

        with test_db_pool.get_cursor() as (conn, c):
            c.execute("SELECT id FROM test_cursor")
            assert c.fetchone()[0] == 42

    def test_concurrent_access(self, test_db_pool):
        """Multiple threads can use the pool safely."""
        with test_db_pool.get_connection() as conn:
            conn.execute("CREATE TABLE test_concurrent (id INTEGER, thread_id INTEGER)")

        errors = []

        def worker(thread_id):
            try:
                for i in range(10):
                    with test_db_pool.get_cursor() as (conn, c):
                        c.execute(
                            "INSERT INTO test_concurrent (id, thread_id) VALUES (?, ?)",
                            (i, thread_id)
                        )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent access errors: {errors}"

        with test_db_pool.get_cursor() as (conn, c):
            c.execute("SELECT COUNT(*) FROM test_concurrent")
            count = c.fetchone()[0]
            assert count == 30  # 3 threads × 10 inserts

    def test_connection_reuse(self, test_db_pool):
        """Connections are returned to pool and reused."""
        # Get and return a connection
        with test_db_pool.get_connection() as conn1:
            id1 = id(conn1)

        # Next connection should be the same object (reused from pool)
        with test_db_pool.get_connection() as conn2:
            id2 = id(conn2)

        assert id1 == id2, "Connection should be reused from pool"

    def test_close_all(self, test_db_pool):
        """close_all() empties the pool."""
        # Pre-warm the pool
        with test_db_pool.get_connection() as conn:
            conn.execute("SELECT 1")

        test_db_pool.close_all()
        assert test_db_pool._pool.empty()
