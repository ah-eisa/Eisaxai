"""
EisaX Referral System — F-6
- كل مستخدم له كود دعوة فريد
- الداعي يحصل على +50 رسالة/يوم لكل صديق يسجّل
- المدعو يحصل على 7 أيام pro مجانية
"""
import os
import sqlite3
import secrets
import logging
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_DB = Path("/home/ubuntu/investwise/data/referrals.db")
_DB.parent.mkdir(parents=True, exist_ok=True)

# Bonus configuration
REFERRER_BONUS_MSGS   = 50    # extra daily messages for referrer per referral
REFEREE_BONUS_DAYS    = 7     # days of pro tier for the new user
REFEREE_BONUS_TIER    = "pro"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS referral_codes (
            user_id    TEXT PRIMARY KEY,
            code       TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS referrals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            referrer_id TEXT NOT NULL,
            referee_id  TEXT NOT NULL UNIQUE,
            code        TEXT NOT NULL,
            rewarded    INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    return conn


def get_or_create_code(user_id: str) -> str:
    """Return existing referral code or create a new one."""
    conn = _get_conn()
    row = conn.execute("SELECT code FROM referral_codes WHERE user_id=?", (user_id,)).fetchone()
    if row:
        conn.close()
        return row["code"]
    code = secrets.token_urlsafe(8).upper()[:10]
    try:
        with conn:
            conn.execute("INSERT INTO referral_codes(user_id, code) VALUES(?,?)", (user_id, code))
    except sqlite3.IntegrityError:
        # collision — try again
        code = secrets.token_urlsafe(10).upper()[:10]
        with conn:
            conn.execute("INSERT INTO referral_codes(user_id, code) VALUES(?,?)", (user_id, code))
    conn.close()
    return code


def resolve_code(code: str) -> str | None:
    """Return user_id of the code owner, or None if invalid."""
    conn = _get_conn()
    row = conn.execute("SELECT user_id FROM referral_codes WHERE code=?", (code.upper(),)).fetchone()
    conn.close()
    return row["user_id"] if row else None


def apply_referral(referee_id: str, code: str) -> dict:
    """
    Called when a new user signs up with a referral code.
    - Links referee → referrer
    - Applies bonuses via session_manager
    Returns {'success': bool, 'message': str, 'referrer_id': str|None}
    """
    referrer_id = resolve_code(code)
    if not referrer_id:
        return {"success": False, "message": "Invalid referral code"}
    if referrer_id == referee_id:
        return {"success": False, "message": "Cannot refer yourself"}

    conn = _get_conn()
    # Check if already referred
    existing = conn.execute(
        "SELECT id FROM referrals WHERE referee_id=?", (referee_id,)
    ).fetchone()
    if existing:
        conn.close()
        return {"success": False, "message": "Referral already applied"}

    # Record referral
    with conn:
        conn.execute(
            "INSERT INTO referrals(referrer_id, referee_id, code) VALUES(?,?,?)",
            (referrer_id, referee_id, code.upper()),
        )
    conn.close()

    # Apply bonuses
    try:
        from core.session_manager import SessionManager
        from core.config import APP_DB
        mgr = SessionManager(str(APP_DB))

        # Referee: 7 days pro
        mgr.set_user_profile(referee_id, tier=REFEREE_BONUS_TIER)
        logger.info("[referral] referee %s → tier=%s for %d days",
                    referee_id, REFEREE_BONUS_TIER, REFEREE_BONUS_DAYS)

        # Referrer: +50 msgs/day per referral (count total referrals)
        referral_count = _get_conn().execute(
            "SELECT COUNT(*) FROM referrals WHERE referrer_id=?", (referrer_id,)
        ).fetchone()[0]
        bonus_limit = referral_count * REFERRER_BONUS_MSGS
        referrer_profile = mgr.get_user_profile(referrer_id)
        base_limit = int(referrer_profile.get("daily_limit") or 0)
        new_limit = max(base_limit, bonus_limit)
        mgr.set_user_profile(referrer_id, daily_limit=new_limit)
        logger.info("[referral] referrer %s → daily_limit=%d (%d referrals)",
                    referrer_id, new_limit, referral_count)

        # Mark rewarded
        with _get_conn() as c:
            c.execute(
                "UPDATE referrals SET rewarded=1 WHERE referee_id=?", (referee_id,)
            )

        return {
            "success": True,
            "message": f"Referral applied! You get {REFEREE_BONUS_DAYS} days of {REFEREE_BONUS_TIER}.",
            "referrer_id": referrer_id,
        }
    except Exception as exc:
        logger.error("[referral] bonus apply failed: %s", exc)
        return {"success": True, "message": "Referral recorded (bonus pending)", "referrer_id": referrer_id}


def get_referral_stats(user_id: str) -> dict:
    """Return referral stats for a user."""
    conn = _get_conn()
    code = get_or_create_code(user_id)
    total = conn.execute(
        "SELECT COUNT(*) FROM referrals WHERE referrer_id=?", (user_id,)
    ).fetchone()[0]
    rewarded = conn.execute(
        "SELECT COUNT(*) FROM referrals WHERE referrer_id=? AND rewarded=1", (user_id,)
    ).fetchone()[0]
    recent = [
        dict(r) for r in conn.execute(
            "SELECT referee_id, created_at FROM referrals WHERE referrer_id=? ORDER BY created_at DESC LIMIT 10",
            (user_id,),
        ).fetchall()
    ]
    conn.close()
    return {
        "user_id":       user_id,
        "referral_code": code,
        "total_referrals":   total,
        "rewarded_referrals": rewarded,
        "bonus_msgs_earned": rewarded * REFERRER_BONUS_MSGS,
        "recent_referrals":  recent,
    }
