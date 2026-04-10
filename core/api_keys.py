import os, sqlite3, secrets, hashlib, logging
from pathlib import Path
from datetime import datetime, timezone

_DB = Path('/home/ubuntu/investwise/data/api_keys.db')
_DB.parent.mkdir(parents=True, exist_ok=True)


def _conn():
    c = sqlite3.connect(str(_DB)); c.row_factory = sqlite3.Row
    c.execute('''CREATE TABLE IF NOT EXISTS api_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key_hash TEXT NOT NULL UNIQUE,
        key_prefix TEXT NOT NULL,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL DEFAULT "Default",
        tier TEXT NOT NULL DEFAULT "basic",
        daily_limit INTEGER NOT NULL DEFAULT 0,
        active INTEGER NOT NULL DEFAULT 1,
        last_used TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )''')
    c.commit(); return c


def generate_key(user_id, name='Default', tier='basic', daily_limit=0) -> str:
    raw = 'eixa_' + secrets.token_urlsafe(32)
    h = hashlib.sha256(raw.encode()).hexdigest()
    prefix = raw[:12]
    with _conn() as c:
        c.execute('INSERT INTO api_keys(key_hash,key_prefix,user_id,name,tier,daily_limit) VALUES(?,?,?,?,?,?)',
                  (h, prefix, user_id, name, tier, daily_limit))
    return raw


def validate_key(raw_key: str) -> dict | None:
    if not raw_key.startswith('eixa_'): return None
    h = hashlib.sha256(raw_key.encode()).hexdigest()
    c = _conn()
    row = c.execute('SELECT * FROM api_keys WHERE key_hash=? AND active=1', (h,)).fetchone()
    c.close()
    if not row: return None
    # update last_used
    with _conn() as conn:
        conn.execute('UPDATE api_keys SET last_used=? WHERE key_hash=?',
                     (datetime.now(timezone.utc).isoformat(), h))
    return dict(row)


def list_user_keys(user_id) -> list:
    c = _conn()
    rows = c.execute('SELECT id,key_prefix,name,tier,daily_limit,active,last_used,created_at FROM api_keys WHERE user_id=? ORDER BY created_at DESC', (user_id,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def revoke_key(key_id: int, user_id: str):
    with _conn() as c:
        c.execute('UPDATE api_keys SET active=0 WHERE id=? AND user_id=?', (key_id, user_id))
