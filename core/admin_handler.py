"""
EisaX Admin Handler
-------------------
Gives Ahmed (owner) the ability to manage the system directly from chat:
- Read any server file or log
- Propose + apply code changes (with backup + confirmation)
- Append rules to the Playbook live
- Lock / unlock admin session

Drop this file at: /home/ubuntu/investwise/core/admin_handler.py
"""

import os
import random
import secrets
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Config from .env ─────────────────────────────────────────────────────────
ADMIN_PASSPHRASE      = os.getenv("ADMIN_PASSPHRASE", "")
ADMIN_USER_ID         = os.getenv("ADMIN_USER_ID", "ahmed")
from core.config import PLAYBOOK_PATH as _cfg_pb, BACKUPS_DIR as _cfg_bk, MODIFICATIONS_LOG as _cfg_log
PLAYBOOK_PATH         = os.getenv("PLAYBOOK_PATH", str(_cfg_pb))
BACKUP_DIR            = os.getenv("BACKUP_DIR",    str(_cfg_bk))
LOG_PATH              = os.getenv("LOG_PATH",      str(_cfg_log))
ADMIN_TIMEOUT_MINUTES = 30


# ── Session State ─────────────────────────────────────────────────────────────

def unlock_admin(session: dict, message: str, user_id: str) -> bool:
    """
    Check passphrase + user identity.
    If both match → mark session as admin and return True.
    """
    if not ADMIN_PASSPHRASE:
        return False
    if secrets.compare_digest(
        message.strip().encode("utf-8"), ADMIN_PASSPHRASE.encode("utf-8")
    ):
        session["is_admin"] = True
        session["admin_unlocked_at"] = datetime.now().isoformat()
        session["pending_modification"] = None
        logger.info("[Admin] Session unlocked for user: %s", user_id)
        return True
    return False


def is_admin_active(session: dict) -> bool:
    """Return True if admin session is active and not expired (30 min timeout)."""
    if not session.get("is_admin"):
        return False
    unlocked_at_str = session.get("admin_unlocked_at")
    if not unlocked_at_str:
        return False
    try:
        unlocked_at = datetime.fromisoformat(unlocked_at_str)
        if datetime.now() - unlocked_at > timedelta(minutes=ADMIN_TIMEOUT_MINUTES):
            session["is_admin"] = False
            logger.info("[Admin] Session expired (timeout)")
            return False
    except Exception:
        session["is_admin"] = False
        return False
    return True


def lock_admin(session: dict):
    """Manually lock the admin session."""
    session["is_admin"] = False
    session["pending_modification"] = None
    logger.info("[Admin] Session locked manually")


# ── File Operations ───────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    """Read any UTF-8 file on the server and return its content."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info("[Admin] Read: %s", path)
        return content
    except FileNotFoundError:
        return f"ERROR: File not found — {path}"
    except Exception as e:
        return f"ERROR: Could not read {path} — {e}"


def read_logs(lines: int = 50) -> str:
    """Return the last N lines from the modification log."""
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:]) if all_lines else "(log is empty)"
    except FileNotFoundError:
        return "No modification log yet."
    except Exception as e:
        return f"ERROR reading log: {e}"


def backup_file(path: str) -> str:
    """
    Create a timestamped backup copy before any write.
    Returns the backup path.
    Raises RuntimeError if backup fails (caller should not proceed with write).
    """
    try:
        Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
        filename    = Path(path).name
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{BACKUP_DIR}{filename}.backup_{timestamp}"
        shutil.copy(path, backup_path)
        logger.info("[Admin] Backup created: %s", backup_path)
        return backup_path
    except Exception as e:
        raise RuntimeError(f"Backup failed for {path}: {e}")


def write_file(path: str, content: str, reason: str = "") -> dict:
    """
    Write content to a file — ALWAYS backs up first.
    Returns {"success": True, "backup_path": ...} or {"success": False, "error": ...}
    """
    try:
        backup_path = backup_file(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        _log_modification(path, reason, backup_path)
        logger.info("[Admin] File written: %s", path)
        return {"success": True, "backup_path": backup_path}
    except Exception as e:
        logger.error("[Admin] Write failed for %s: %s", path, e)
        return {"success": False, "error": str(e)}


def append_playbook(new_rule: str, reason: str = "") -> dict:
    """
    Append a new rule row to the Playbook update log section (Section 11).
    Safe — backs up before writing.
    """
    try:
        today   = datetime.now().strftime("%B %d, %Y")
        new_row = f"| {today} | {new_rule} | {reason} |"

        with open(PLAYBOOK_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # Find the update log section and insert before the end marker
        marker = "*Add new rules here whenever the agent makes a mistake you want to prevent.*"
        if marker in content:
            updated = content.replace(marker, f"{marker}\n{new_row}")
        else:
            # Fallback: find the table end (last | row) and append after it
            lines   = content.splitlines()
            last_row_idx = max(
                (i for i, l in enumerate(lines) if l.strip().startswith("|")),
                default=len(lines) - 1
            )
            lines.insert(last_row_idx + 1, new_row)
            updated = "\n".join(lines)

        backup_path = backup_file(PLAYBOOK_PATH)
        with open(PLAYBOOK_PATH, "w", encoding="utf-8") as f:
            f.write(updated)

        _log_modification(PLAYBOOK_PATH, f"Rule appended: {new_rule}", backup_path)
        return {"success": True, "rule": new_row}
    except Exception as e:
        logger.error("[Admin] append_playbook failed: %s", e)
        return {"success": False, "error": str(e)}


def _log_modification(path: str, reason: str, backup_path: str):
    """Append one line to the modification audit log."""
    try:
        Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] FILE: {path} | REASON: {reason} | BACKUP: {backup_path}\n"
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        logger.warning("[Admin] Audit log write failed: %s", e)


# ── Pending Modification Flow ─────────────────────────────────────────────────
# BUG-04 FIX: use a one-time random token per proposal instead of bare "CONFIRM".
# The token is stored in session and must appear in the user's reply exactly as:
#   CONFIRM <token>   (e.g. "CONFIRM 48392")
# This prevents accidental or injected confirmations.

def store_pending_modification(session: dict, proposal: str) -> str:
    """
    Store a Gemini-proposed code change and generate a one-time 5-digit token.
    Returns the token so the orchestrator can display it to the admin.
    """
    token = str(random.randint(10000, 99999))
    session["pending_modification"] = proposal
    session["confirm_token"] = token
    return token


def get_pending_modification(session: dict) -> str | None:
    """Return pending modification string if any, else None."""
    return session.get("pending_modification")


def clear_pending_modification(session: dict):
    """Clear after the user confirms or cancels."""
    session["pending_modification"] = None
    session.pop("confirm_token", None)


def is_confirmation(message: str, session: dict) -> bool:
    """
    True only if the message contains the session-specific token:
      CONFIRM <token>   e.g. "CONFIRM 48392"
    Falls back to Arabic equivalents for UX convenience (still token-gated).
    """
    token = session.get("confirm_token", "")
    if not token:
        return False
    msg_up = message.upper().strip()
    # Primary check: exact token match
    if f"CONFIRM {token}" in msg_up:
        return True
    # Arabic convenience aliases — still require token in same message
    arabic_confirms = ["تمام", "وافق", "نعم طبقه", "اطبق"]
    return any(c in message for c in arabic_confirms) and token in message


def is_rejection(message: str) -> bool:
    rejections = ["لا", "cancel", "no", "discard", "الغي", "مش تمام", "CANCEL"]
    return any(r in message.lower() for r in rejections)