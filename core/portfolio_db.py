"""
portfolio_db.py — SQLite persistence for saved portfolio holdings (Phase J).
Stores raw holdings text per user so they can reload past portfolios.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DB = Path("/home/ubuntu/investwise/data/portfolios.db")
_DB.parent.mkdir(parents=True, exist_ok=True)


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(str(_DB), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_portfolio_table():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS saved_portfolios (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                name         TEXT    NOT NULL,
                holdings     TEXT    NOT NULL,
                created_at   TEXT    NOT NULL,
                updated_at   TEXT    NOT NULL,
                UNIQUE(user_id, name)
            )
        """)
        con.commit()


def save_portfolio(user_id: int, name: str, holdings: str) -> bool:
    """Insert or replace a named portfolio for the user."""
    init_portfolio_table()
    now = datetime.now(timezone.utc).isoformat()
    try:
        with _conn() as con:
            con.execute("""
                INSERT INTO saved_portfolios (user_id, name, holdings, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, name) DO UPDATE SET
                    holdings   = excluded.holdings,
                    updated_at = excluded.updated_at
            """, (user_id, name.strip(), holdings.strip(), now, now))
            con.commit()
        return True
    except Exception:
        return False


def load_portfolios(user_id: int) -> list[dict]:
    """Return all saved portfolios for user, newest first."""
    init_portfolio_table()
    with _conn() as con:
        rows = con.execute(
            "SELECT id, name, holdings, created_at, updated_at "
            "FROM saved_portfolios WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_portfolio(user_id: int, name: str) -> bool:
    """Delete a named portfolio for the user."""
    init_portfolio_table()
    with _conn() as con:
        cur = con.execute(
            "DELETE FROM saved_portfolios WHERE user_id = ? AND name = ?",
            (user_id, name.strip())
        )
        con.commit()
    return cur.rowcount > 0
