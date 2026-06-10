"""
core/services/admin_orchestrator.py
─────────────────────────────────────
Admin-mode message handling extracted from process_message.

Public API
──────────
    handle_admin_mode(orchestrator, session_id, user_id, message) -> dict | None
        If the message is handled by admin logic, returns a response dict.
        Returns None if admin mode is not active and the message is not an
        unlock attempt — caller should continue normal routing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Avoid circular imports — orchestrator imports this module
    pass


async def handle_admin_mode(
    orchestrator: Any,
    session_id:   str,
    user_id:      str,
    message:      str,
) -> dict | None:
    """
    Handle all admin-mode interactions.

    Returns
    ───────
    dict  — a ready-to-return response (caller should return it immediately)
    None  — admin mode is not active / not triggered; continue normal routing
    """
    from core.admin_handler import (
        unlock_admin,
        is_admin_active,
        lock_admin,
        get_pending_modification,
        is_confirmation,
        is_rejection,
        clear_pending_modification,
        store_pending_modification,
        read_file,
        read_logs,
        append_playbook,
    )
    from core.prompt_manager import ADMIN_SYSTEM_PROMPT

    sess_mgr    = orchestrator.session_mgr
    _adm_session = sess_mgr.get_session_state(session_id) or {}

    # ── Unlock attempt ─────────────────────────────────────────────────────────
    if unlock_admin(_adm_session, message, user_id):
        sess_mgr.save_session_state(session_id, _adm_session)
        return {
            "reply": (
                "🔓 **Admin Mode Active** — expires in 30 minutes.\n\n"
                "**الأوامر المتاحة:**\n"
                "- `read file <path>` — اقرأ أي ملف\n"
                "- `show last 50 logs` — سجل التعديلات\n"
                "- `add rule: <القاعدة>` — أضف لـ Playbook فوراً\n"
                "- صف أي مشكلة وسأقترح الحل للموافقة\n"
                "- `lock admin` — أقفل الجلسة"
            ),
            "session_id": session_id,
            "agent_name": "EisaX Admin",
        }

    # ── Admin session active ───────────────────────────────────────────────────
    if not is_admin_active(_adm_session):
        return None  # not in admin mode — continue normal routing

    # ── Pending confirmation ───────────────────────────────────────────────────
    _pending = get_pending_modification(_adm_session)
    if _pending:
        if is_confirmation(message, _adm_session):
            _result = await orchestrator._apply_pending_modification(_pending)
            clear_pending_modification(_adm_session)
            sess_mgr.save_session_state(session_id, _adm_session)
            return {"reply": _result, "session_id": session_id, "agent_name": "EisaX Admin"}
        elif is_rejection(message):
            clear_pending_modification(_adm_session)
            sess_mgr.save_session_state(session_id, _adm_session)
            return {"reply": "❌ تم الإلغاء. لم يُعدَّل شيء.", "session_id": session_id}
        else:
            _tok = _adm_session.get("confirm_token", "?")
            return {
                "reply": (
                    f"⏳ Pending proposal. Type **`CONFIRM {_tok}`** to apply "
                    "or **`CANCEL`** to discard."
                ),
                "session_id": session_id,
            }

    # ── Lock command ───────────────────────────────────────────────────────────
    if "lock admin" in message.lower():
        lock_admin(_adm_session)
        sess_mgr.save_session_state(session_id, _adm_session)
        return {"reply": "🔒 Admin session locked.", "session_id": session_id}

    # ── Read file ──────────────────────────────────────────────────────────────
    if message.lower().startswith("read file"):
        path    = message.split("read file", 1)[-1].strip()
        content = read_file(path)
        return {
            "reply":      f"📄 **{path}**\n\n```\n{content[:4000]}\n```",
            "session_id": session_id,
        }

    # ── Show logs ──────────────────────────────────────────────────────────────
    if "show" in message.lower() and "log" in message.lower():
        logs = read_logs(50)
        return {
            "reply":      f"📋 **Last 50 modifications:**\n\n```\n{logs}\n```",
            "session_id": session_id,
        }

    # ── Add playbook rule ──────────────────────────────────────────────────────
    if message.lower().startswith("add rule:"):
        rule = message.split("add rule:", 1)[-1].strip()
        res  = append_playbook(rule, reason="Added by admin via chat")
        if res["success"]:
            return {"reply": f"✅ Rule added:\n`{res['rule']}`", "session_id": session_id}
        else:
            return {"reply": f"❌ Failed: {res['error']}", "session_id": session_id}

    # ── Anything else → DeepSeek in Admin Mode ────────────────────────────────
    try:
        prompt    = f"{ADMIN_SYSTEM_PROMPT}\n\nAdmin request: {message}"
        adm_reply = orchestrator._gemini_generate(prompt, label="ADMIN")
        if "PROPOSED CODE" in adm_reply or "PROPOSED:" in adm_reply:
            tok = store_pending_modification(_adm_session, adm_reply)
            sess_mgr.save_session_state(session_id, _adm_session)
            adm_reply += (
                f"\n\n---\n🔐 **To apply, reply:** `CONFIRM {tok}` | To cancel: `CANCEL`"
            )
        return {"reply": adm_reply, "session_id": session_id, "agent_name": "EisaX Admin"}
    except Exception as exc:
        return {"reply": f"❌ Admin error: {exc}", "session_id": session_id}
