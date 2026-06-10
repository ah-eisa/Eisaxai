"""
user_db.py — Users table CRUD for EisaX B2B Auth
Uses the existing investwise.db SQLite database.
"""
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional

DB_PATH = "/home/ubuntu/investwise/investwise.db"
logger = logging.getLogger("user_db")


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_users_table():
    """Create the users table if it doesn't exist (idempotent). Migrates existing table."""
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                email           TEXT    NOT NULL UNIQUE,
                name            TEXT    NOT NULL,
                password_hash   TEXT    NOT NULL,
                role            TEXT    NOT NULL DEFAULT 'user',
                is_active       INTEGER NOT NULL DEFAULT 1,
                must_change_pw  INTEGER NOT NULL DEFAULT 1,
                created_at      TEXT    NOT NULL,
                last_login      TEXT,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until    TEXT
            )
        """)
        # Migrate existing table — add columns if absent
        existing = {row[1] for row in con.execute("PRAGMA table_info(users)").fetchall()}
        if "failed_attempts" not in existing:
            con.execute("ALTER TABLE users ADD COLUMN failed_attempts INTEGER NOT NULL DEFAULT 0")
        if "locked_until" not in existing:
            con.execute("ALTER TABLE users ADD COLUMN locked_until TEXT")
        con.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        con.commit()
    logger.info("users table ready")


def create_user(email: str, name: str, password_hash: str,
                role: str = "user", must_change_pw: bool = True) -> int:
    """Returns new user id."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO users (email, name, password_hash, role, is_active, must_change_pw, created_at)
               VALUES (?, ?, ?, ?, 1, ?, ?)""",
            (email.lower().strip(), name.strip(), password_hash,
             role, 1 if must_change_pw else 0, now)
        )
        con.commit()
        return cur.lastrowid


def get_user_by_email(email: str) -> Optional[dict]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def list_users() -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id, email, name, role, is_active, must_change_pw, created_at, last_login "
            "FROM users ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def increment_failed_attempts(user_id: int, max_attempts: int = 5, lockout_minutes: int = 15):
    """Increment failed login counter. Lock account if max_attempts reached."""
    from datetime import timezone
    with _conn() as con:
        con.execute(
            "UPDATE users SET failed_attempts = failed_attempts + 1 WHERE id = ?", (user_id,)
        )
        row = con.execute("SELECT failed_attempts FROM users WHERE id = ?", (user_id,)).fetchone()
        if row and row[0] >= max_attempts:
            locked_until = (datetime.now(timezone.utc) + timedelta(minutes=lockout_minutes)).isoformat()
            con.execute("UPDATE users SET locked_until = ? WHERE id = ?", (locked_until, user_id))
        con.commit()


def reset_failed_attempts(user_id: int):
    """Clear failed counter and lockout on successful login."""
    with _conn() as con:
        con.execute(
            "UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = ?", (user_id,)
        )
        con.commit()


def update_user(user_id: int, **fields) -> bool:
    """Update any subset of: name, role, is_active, must_change_pw, password_hash, last_login, failed_attempts, locked_until."""
    allowed = {"name", "role", "is_active", "must_change_pw", "password_hash", "last_login",
               "failed_attempts", "locked_until"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    parts = ", ".join(f"{k} = ?" for k in updates)
    vals  = list(updates.values()) + [user_id]
    with _conn() as con:
        con.execute(f"UPDATE users SET {parts} WHERE id = ?", vals)
        con.commit()
    return True


def delete_user(user_id: int) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM users WHERE id = ?", (user_id,))
        con.commit()
    return cur.rowcount > 0


def record_login(user_id: int):
    now = datetime.now(timezone.utc).isoformat()
    update_user(user_id, last_login=now)
