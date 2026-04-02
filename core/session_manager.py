import logging
import hashlib
import bcrypt
from typing import Optional, List, Dict
from datetime import datetime

from core.db import db

logger = logging.getLogger(__name__)

from core.config import APP_DB as _cfg_app_db

class SessionManager:
    def __init__(self, db_path: str = str(_cfg_app_db)):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with db.get_cursor() as (conn, c):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")

            # sessions table
            c.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip TEXT,
                    user_agent TEXT,
                    blocked INTEGER DEFAULT 0
                )
            ''')
            existing = [row[1] for row in c.execute('PRAGMA table_info(sessions)').fetchall()]
            for col, defn in [
                ('ip',         'TEXT'),
                ('user_agent', 'TEXT'),
                ('blocked',    'INTEGER DEFAULT 0'),
                ('state',      'TEXT'),
            ]:
                if col not in existing:
                    c.execute(f'ALTER TABLE sessions ADD COLUMN {col} {defn}')

            # chat_history table
            c.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT,
                    user_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # admin_messages
            c.execute('''
                CREATE TABLE IF NOT EXISTS admin_messages (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    delivered INTEGER DEFAULT 0
                )
            ''')

            # admin_settings
            c.execute('''
                CREATE TABLE IF NOT EXISTS admin_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # user_profiles
            c.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY
                )
            ''')
            existing_up = [row[1] for row in c.execute('PRAGMA table_info(user_profiles)').fetchall()]
            for col, defn in [('daily_limit','INTEGER DEFAULT 0'), ('admin_note','TEXT DEFAULT ""'), ('tier','TEXT DEFAULT "basic"')]:
                if col not in existing_up:
                    c.execute(f'ALTER TABLE user_profiles ADD COLUMN {col} {defn}')

            # admin_audit_log
            c.execute('''
                CREATE TABLE IF NOT EXISTS admin_audit_log (
                    id INTEGER PRIMARY KEY,
                    action TEXT NOT NULL,
                    target TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # blocked_ips
            c.execute('''
                CREATE TABLE IF NOT EXISTS blocked_ips (
                    ip TEXT PRIMARY KEY,
                    reason TEXT DEFAULT '',
                    blocked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Performance indexes
            c.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_user_ts ON chat_history(user_id, timestamp)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")

    # ── DB Maintenance ───────────────────────────────────────────────────────

    def cleanup_old_sessions(self, days: int = 90) -> int:
        """Delete sessions and chat history older than `days` days. Returns rows deleted."""
        with db.get_cursor() as (conn, c):
            cutoff = f"datetime('now', '-{days} days')"
            old_sessions = c.execute(
                f"SELECT session_id FROM sessions WHERE created_at < {cutoff}"
            ).fetchall()
            session_ids = [r[0] for r in old_sessions]
            deleted = 0
            if session_ids:
                placeholders = ",".join("?" * len(session_ids))
                c.execute(f"DELETE FROM chat_history WHERE session_id IN ({placeholders})", session_ids)
                deleted = c.rowcount
                c.execute(f"DELETE FROM sessions WHERE session_id IN ({placeholders})", session_ids)
            c.execute("DELETE FROM admin_audit_log WHERE timestamp < datetime('now', '-180 days')")
        # VACUUM must run outside a transaction
        with db.get_connection() as conn:
            conn.execute("VACUUM")
        logger.info("[DB Cleanup] Removed %d old sessions (>%d days), vacuumed DB", len(session_ids), days)
        return deleted

    # ── Sessions ────────────────────────────────────────────────────────────

    def get_or_create_session(self, user_id: str, session_id: Optional[str] = None,
                               ip: Optional[str] = None, user_agent: Optional[str] = None) -> str:
        if not session_id:
            session_id = f"session_{int(datetime.now().timestamp())}"
        with db.get_cursor() as (conn, c):
            c.execute(
                "INSERT OR IGNORE INTO sessions (session_id, user_id, ip, user_agent) VALUES (?, ?, ?, ?)",
                (session_id, user_id, ip, user_agent)
            )
        return session_id

    def save_message(self, session_id: str, user_id: str, role: str, content: str):
        with db.get_cursor() as (conn, c):
            c.execute(
                "INSERT INTO chat_history (session_id, user_id, role, content) VALUES (?, ?, ?, ?)",
                (session_id, user_id, role, content)
            )

    def get_chat_history(self, session_id: str) -> List[Dict]:
        with db.get_cursor() as (conn, c):
            c.execute(
                "SELECT role, content, timestamp FROM chat_history WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,)
            )
            rows = c.fetchall()
        return [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in rows]

    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Return all sessions for a given user, newest first."""
        with db.get_cursor() as (conn, c):
            c.execute(
                "SELECT session_id, created_at, ip, user_agent, blocked "
                "FROM sessions WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)
            )
            rows = c.fetchall()
        return [
            {"session_id": r[0], "created_at": r[1], "ip": r[2],
             "user_agent": r[3], "blocked": bool(r[4])}
            for r in rows
        ]

    def delete_session(self, session_id: str):
        with db.get_cursor() as (conn, c):
            c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            c.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))

    # ── Block / Unblock ─────────────────────────────────────────────────────

    def is_user_blocked(self, user_id: str) -> bool:
        with db.get_cursor() as (conn, c):
            c.execute("SELECT MAX(blocked) FROM sessions WHERE user_id = ?", (user_id,))
            row = c.fetchone()
        return bool(row and row[0])

    def set_user_blocked(self, user_id: str, blocked: bool):
        with db.get_cursor() as (conn, c):
            c.execute("UPDATE sessions SET blocked = ? WHERE user_id = ?", (1 if blocked else 0, user_id))

    # ── Admin Messages ──────────────────────────────────────────────────────

    def queue_admin_message(self, user_id: str, content: str):
        with db.get_cursor() as (conn, c):
            c.execute("INSERT INTO admin_messages (user_id, content) VALUES (?, ?)", (user_id, content))

    def get_pending_admin_messages(self, user_id: str) -> List[Dict]:
        with db.get_cursor() as (conn, c):
            c.execute(
                "SELECT id, content FROM admin_messages WHERE user_id = ? AND delivered = 0 ORDER BY sent_at ASC",
                (user_id,)
            )
            rows = c.fetchall()
        return [{"id": r[0], "content": r[1]} for r in rows]

    def mark_admin_messages_delivered(self, user_id: str):
        with db.get_cursor() as (conn, c):
            c.execute(
                "UPDATE admin_messages SET delivered = 1 WHERE user_id = ? AND delivered = 0",
                (user_id,)
            )

    def get_admin_message_history(self) -> List[Dict]:
        with db.get_cursor() as (conn, c):
            c.execute("SELECT id, user_id, content, sent_at, delivered FROM admin_messages ORDER BY sent_at DESC LIMIT 100")
            rows = c.fetchall()
        return [{"id": r[0], "user_id": r[1], "content": r[2], "sent_at": r[3], "delivered": bool(r[4])} for r in rows]

    # ── Admin Settings / Password ────────────────────────────────────────────

    def get_admin_setting(self, key: str) -> Optional[str]:
        with db.get_cursor() as (conn, c):
            c.execute("SELECT value FROM admin_settings WHERE key = ?", (key,))
            row = c.fetchone()
        return row[0] if row else None

    def set_admin_setting(self, key: str, value: str):
        with db.get_cursor() as (conn, c):
            c.execute(
                "INSERT OR REPLACE INTO admin_settings (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (key, value)
            )

    def verify_admin_password(self, provided: str, fallback_token: str) -> bool:
        stored_hash = self.get_admin_setting("admin_password_hash")
        if stored_hash:
            if stored_hash.startswith("$2b$") or stored_hash.startswith("$2a$"):
                try:
                    return bcrypt.checkpw(provided.encode(), stored_hash.encode())
                except Exception:
                    return False
            else:
                if hashlib.sha256(provided.encode()).hexdigest() == stored_hash:
                    self.change_admin_password(provided)
                    return True
                return False
        return provided == fallback_token

    def change_admin_password(self, new_password: str):
        hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        self.set_admin_setting("admin_password_hash", hashed)

    # ── Admin Full Sessions ─────────────────────────────────────────────────

    def get_all_sessions_admin(self) -> List[Dict]:
        with db.get_cursor() as (conn, c):
            c.execute('''
                SELECT
                    s.user_id,
                    s.session_id,
                    s.created_at,
                    s.ip,
                    s.user_agent,
                    s.blocked,
                    COUNT(h.id)          AS msg_count,
                    MAX(h.timestamp)     AS last_active,
                    SUM(CASE WHEN h.role='user' THEN 1 ELSE 0 END) AS user_msgs
                FROM sessions s
                LEFT JOIN chat_history h ON s.session_id = h.session_id
                GROUP BY s.session_id
                ORDER BY last_active DESC
            ''')
            rows = c.fetchall()
        return [
            {
                "user_id":    row[0],
                "session_id": row[1],
                "created_at": row[2],
                "ip":         row[3] or "—",
                "user_agent": row[4] or "—",
                "blocked":    bool(row[5]),
                "msg_count":  row[6],
                "last_active":row[7],
                "user_msgs":  row[8],
            }
            for row in rows
        ]

    # ── User Profiles (tier, limit, note) ──────────────────────────────────

    def get_user_profile(self, user_id: str) -> Dict:
        with db.get_cursor() as (conn, c):
            c.execute("SELECT daily_limit, admin_note, tier FROM user_profiles WHERE user_id = ?", (user_id,))
            row = c.fetchone()
        if row:
            return {"daily_limit": row[0] or 0, "note": row[1] or "", "tier": row[2] or "basic"}
        return {"daily_limit": 0, "note": "", "tier": "basic"}

    def set_user_profile(self, user_id: str, daily_limit: int = None, note: str = None, tier: str = None):
        with db.get_cursor() as (conn, c):
            c.execute("INSERT OR IGNORE INTO user_profiles (user_id) VALUES (?)", (user_id,))
            if daily_limit is not None:
                c.execute("UPDATE user_profiles SET daily_limit = ? WHERE user_id = ?", (daily_limit, user_id))
            if note is not None:
                c.execute("UPDATE user_profiles SET admin_note = ? WHERE user_id = ?", (note, user_id))
            if tier is not None:
                c.execute("UPDATE user_profiles SET tier = ? WHERE user_id = ?", (tier, user_id))

    def get_user_daily_count(self, user_id: str) -> int:
        with db.get_cursor() as (conn, c):
            c.execute(
                "SELECT COUNT(*) FROM chat_history WHERE user_id = ? AND role = 'user' AND timestamp >= datetime('now', 'start of day')",
                (user_id,)
            )
            count = c.fetchone()[0]
        return count

    def is_user_rate_limited(self, user_id: str) -> bool:
        profile = self.get_user_profile(user_id)
        limit = profile.get("daily_limit", 0)
        if not limit or limit <= 0:
            return False
        return self.get_user_daily_count(user_id) >= limit

    def broadcast_admin_message(self, content: str) -> int:
        with db.get_cursor() as (conn, c):
            c.execute("SELECT DISTINCT user_id FROM sessions")
            users = [row[0] for row in c.fetchall()]
            for user_id in users:
                c.execute("INSERT INTO admin_messages (user_id, content) VALUES (?, ?)", (user_id, content))
        return len(users)

    def get_admin_stats(self) -> Dict:
        with db.get_cursor() as (conn, c):
            row = c.execute("""
                SELECT
                    COUNT(DISTINCT user_id),
                    COUNT(*),
                    SUM(CASE WHEN blocked = 1 THEN 1 ELSE 0 END)
                FROM sessions
            """).fetchone()
            total_users, total_sessions, blocked_users = row[0], row[1], (row[2] or 0)

            row2 = c.execute("""
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN timestamp >= datetime('now','-1 day') THEN 1 ELSE 0 END),
                    SUM(CASE WHEN timestamp >= datetime('now','-7 days') THEN 1 ELSE 0 END),
                    COUNT(DISTINCT CASE WHEN timestamp >= datetime('now','-5 minutes') THEN user_id END)
                FROM chat_history
            """).fetchone()
            total_messages = row2[0] or 0
            messages_today = row2[1] or 0
            messages_week = row2[2] or 0
            active_now = row2[3] or 0

            c.execute("""
                SELECT strftime('%H', timestamp) as hr, COUNT(*) as cnt
                FROM chat_history
                WHERE timestamp >= datetime('now','-1 day')
                GROUP BY hr ORDER BY hr
            """)
            hourly = {row[0]: row[1] for row in c.fetchall()}

        return {
            "total_users": total_users,
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "messages_today": messages_today,
            "messages_week": messages_week,
            "active_now": active_now,
            "blocked_users": blocked_users,
            "hourly": hourly,
        }

    # ── Audit Log ────────────────────────────────────────────────────────────

    def log_admin_action(self, action: str, target: str = None, details: str = None):
        with db.get_cursor() as (conn, c):
            c.execute(
                "INSERT INTO admin_audit_log (action, target, details) VALUES (?, ?, ?)",
                (action, target, details)
            )

    def get_audit_log(self, limit: int = 200) -> List[Dict]:
        with db.get_cursor() as (conn, c):
            c.execute(
                "SELECT id, action, target, details, timestamp FROM admin_audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            rows = c.fetchall()
        return [{"id": r[0], "action": r[1], "target": r[2], "details": r[3], "timestamp": r[4]} for r in rows]

    # ── Delete User Sessions ─────────────────────────────────────────────────

    def delete_user_sessions(self, user_id: str) -> int:
        with db.get_cursor() as (conn, c):
            c.execute("SELECT session_id FROM sessions WHERE user_id = ?", (user_id,))
            session_ids = [row[0] for row in c.fetchall()]
            if session_ids:
                placeholders = ','.join(['?'] * len(session_ids))
                c.execute(f"DELETE FROM chat_history WHERE session_id IN ({placeholders})", session_ids)
                c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return len(session_ids)

    # ── IP Blocking ──────────────────────────────────────────────────────────

    def block_ip(self, ip: str, reason: str = ''):
        with db.get_cursor() as (conn, c):
            c.execute(
                "INSERT OR REPLACE INTO blocked_ips (ip, reason, blocked_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (ip, reason or '')
            )

    def unblock_ip(self, ip: str):
        with db.get_cursor() as (conn, c):
            c.execute("DELETE FROM blocked_ips WHERE ip = ?", (ip,))

    def is_ip_blocked(self, ip: str) -> bool:
        if not ip or ip in ('—', '', 'unknown'):
            return False
        with db.get_cursor() as (conn, c):
            c.execute("SELECT 1 FROM blocked_ips WHERE ip = ?", (ip,))
            result = c.fetchone() is not None
        return result

    def get_blocked_ips(self) -> List[Dict]:
        with db.get_cursor() as (conn, c):
            c.execute("SELECT ip, reason, blocked_at FROM blocked_ips ORDER BY blocked_at DESC")
            rows = c.fetchall()
        return [{"ip": r[0], "reason": r[1] or '', "blocked_at": r[2]} for r in rows]

    # ── New Activity (for notifications) ─────────────────────────────────────

    def get_new_activity(self, since_ts: str) -> Dict:
        with db.get_cursor() as (conn, c):
            try:
                c.execute(
                    "SELECT COUNT(DISTINCT user_id) FROM sessions WHERE created_at > ?", (since_ts,)
                )
                new_users = c.fetchone()[0]
                c.execute(
                    "SELECT COUNT(*) FROM chat_history WHERE timestamp > ? AND role = 'user'", (since_ts,)
                )
                new_messages = c.fetchone()[0]
            except Exception:
                new_users = 0
                new_messages = 0
        return {"new_users": new_users, "new_messages": new_messages}

    # ── Admin Chat Session State ──────────────────────────────────────────────

    def get_session_state(self, session_id: str) -> dict:
        """Load the admin session state dict (JSON) for a given session."""
        import json
        with db.get_cursor() as (conn, c):
            c.execute("SELECT state FROM sessions WHERE session_id = ?", (session_id,))
            row = c.fetchone()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except Exception:
                return {}
        return {}

    def save_session_state(self, session_id: str, state: dict):
        """Persist the admin session state dict (JSON) for a given session."""
        import json
        with db.get_cursor() as (conn, c):
            c.execute(
                "UPDATE sessions SET state = ? WHERE session_id = ?",
                (json.dumps(state), session_id)
            )

    def cleanup_old_sessions(self, days_to_keep: int = 30) -> dict:
        """
        Delete sessions and their chat history older than `days_to_keep` days.
        Also caps chat_history to last 200 messages per session (keeps recent context).
        Returns dict with counts of deleted rows.
        """
        import logging
        _log = logging.getLogger(__name__)
        deleted_sessions = 0
        deleted_messages = 0
        trimmed_sessions = 0
        try:
            with db.get_cursor() as (conn, c):
                # 1. Delete sessions older than N days (cascades to chat_history via session_id)
                c.execute(
                    "DELETE FROM sessions WHERE created_at < datetime('now', ?)",
                    (f"-{days_to_keep} days",)
                )
                deleted_sessions = c.rowcount

                # 2. Delete orphaned chat_history rows (no parent session)
                c.execute(
                    "DELETE FROM chat_history WHERE session_id NOT IN (SELECT session_id FROM sessions)"
                )
                deleted_messages = c.rowcount

                # 3. For each remaining session, keep only the last 200 messages
                c.execute("SELECT DISTINCT session_id FROM chat_history")
                session_ids = [row[0] for row in c.fetchall()]
                for sid in session_ids:
                    c.execute(
                        "SELECT COUNT(*) FROM chat_history WHERE session_id = ?", (sid,)
                    )
                    count = c.fetchone()[0]
                    if count > 200:
                        # Delete oldest messages beyond 200
                        c.execute(
                            """DELETE FROM chat_history WHERE session_id = ? AND id NOT IN (
                                SELECT id FROM chat_history WHERE session_id = ?
                                ORDER BY timestamp DESC LIMIT 200
                            )""",
                            (sid, sid)
                        )
                        trimmed_sessions += 1

                conn.commit()

            _log.info(
                "[Cleanup] deleted %d sessions, %d orphan messages, trimmed %d sessions",
                deleted_sessions, deleted_messages, trimmed_sessions
            )
            return {
                "deleted_sessions": deleted_sessions,
                "deleted_messages": deleted_messages,
                "trimmed_sessions": trimmed_sessions,
            }
        except Exception as exc:
            _log.error("[Cleanup] Failed: %s", exc)
            return {"error": str(exc)}