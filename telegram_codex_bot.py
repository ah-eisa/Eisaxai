#!/usr/bin/env python3
"""
Telegram bot that routes messages to Codex CLI.
Supports per-chat conversation memory and /clear to reset.
"""
import asyncio
import logging
import os
import time
from typing import List

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

MAX_TELEGRAM_MSG  = 4096
CODEX_TIMEOUT_SEC = 120
DEDUP_TTL_SEC     = 600
MAX_HISTORY       = 20   # max exchanges kept per chat

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ALLOW_ALL = os.getenv("TELEGRAM_ALLOW_ALL", "0").strip() == "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("telegram_codex_bot")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

_seen_messages: dict[str, float] = {}

# ── Per-chat conversation history ─────────────────────────────────────────────
# { chat_id: [ {"role": "user"|"assistant", "content": str}, ... ] }
_history: dict[int, list[dict]] = {}

def get_history(chat_id: int) -> list[dict]:
    return _history.get(chat_id, [])

def add_to_history(chat_id: int, role: str, content: str) -> None:
    if chat_id not in _history:
        _history[chat_id] = []
    _history[chat_id].append({"role": role, "content": content})
    # Trim to last MAX_HISTORY messages
    if len(_history[chat_id]) > MAX_HISTORY:
        _history[chat_id] = _history[chat_id][-MAX_HISTORY:]

def clear_history(chat_id: int) -> None:
    _history[chat_id] = []
    logger.info("History cleared for chat_id=%s", chat_id)

def build_prompt(chat_id: int, new_message: str) -> str:
    """Build full prompt with conversation history prepended."""
    history = get_history(chat_id)
    if not history:
        return CODEX_INSTRUCTIONS + "\nTask: " + new_message

    lines = ["=== Conversation so far ==="]
    for msg in history:
        prefix = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{prefix}: {msg['content']}")
    lines.append("=== New message ===")
    lines.append(f"User: {new_message}")
    lines.append("\nRespond to the latest User message, keeping full context of the conversation above.")

    return CODEX_INSTRUCTIONS + "\n\n" + "\n".join(lines)


# ── Auth ──────────────────────────────────────────────────────────────────────

def _parse_id_set(raw: str) -> set[int]:
    values: set[int] = set()
    if not raw:
        return values
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.add(int(token))
        except ValueError:
            logger.warning("Invalid numeric id: %s", token)
    return values


_allowed_chats_env = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "").strip()
_single_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
if not _allowed_chats_env and _single_chat:
    _allowed_chats_env = _single_chat

ALLOWED_CHAT_IDS = _parse_id_set(_allowed_chats_env)
ALLOWED_USER_IDS = _parse_id_set(os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").strip())


def is_authorized(update: Update) -> bool:
    if ALLOW_ALL:
        return True
    chat_id = update.effective_chat.id if update.effective_chat else None
    user_id = update.effective_user.id if update.effective_user else None
    if not ALLOWED_CHAT_IDS and not ALLOWED_USER_IDS:
        return False
    chat_ok = True if not ALLOWED_CHAT_IDS else (chat_id in ALLOWED_CHAT_IDS)
    user_ok = True if not ALLOWED_USER_IDS else (user_id in ALLOWED_USER_IDS)
    return chat_ok and user_ok


def _auth_fail_text(update: Update) -> str:
    chat_id = update.effective_chat.id if update.effective_chat else "unknown"
    user_id = update.effective_user.id if update.effective_user else "unknown"
    return f"⛔ Unauthorized.\nchat_id={chat_id}\nuser_id={user_id}"


# ── Message helpers ───────────────────────────────────────────────────────────

def split_message(text: str, limit: int = MAX_TELEGRAM_MSG) -> List[str]:
    if not text:
        return ["(no output)"]
    chunks: List[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    chunks.append(remaining)
    return chunks


# ── Codex runner ──────────────────────────────────────────────────────────────

CODEX_INSTRUCTIONS = """You are EisaX Server Assistant running on the EisaX production server.

SERVER:
- Main project: /home/ubuntu/investwise (FastAPI + gunicorn port 8000)
- Frontend: /home/ubuntu/eisax-ui (nginx static files)
- Nginx config: /etc/nginx/sites-enabled/eisax
- Python venv: /home/ubuntu/investwise/venv/bin/python3
- Gunicorn master PID 149096 (reload: kill -HUP 149096)
- Key files: api_bridge_v2.py, .env, core/

You have FULL access — read/write files, run shell commands, sudo available.
Actually DO the task. Don't explain without executing.
You have memory of this conversation — use the history above to understand context.
"""


async def run_codex_query(full_prompt: str) -> str:
    cmd = [
        "codex", "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C", "/home/ubuntu/investwise",
        full_prompt,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=CODEX_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return "⏱ Timeout: Codex exceeded 120 seconds."

        out = (stdout or b"").decode("utf-8", errors="replace").strip()
        err = (stderr or b"").decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            details = err or out or "Unknown error"
            return f"❌ Codex failed (exit {proc.returncode}):\n{details}"

        return out or err or "✅ Done — no output."
    except FileNotFoundError:
        return "❌ Codex CLI not found in PATH."
    except Exception as exc:
        return f"❌ Unexpected error: {exc}"


# ── Command handlers ──────────────────────────────────────────────────────────

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if not is_authorized(update):
        await update.message.reply_text(_auth_fail_text(update))
        return
    await update.message.reply_text(
        "⚡ *Codex bot is online.*\n\n"
        "أنا بتذكر الشات كله — كل رسالة بتكمّل على اللي قبلها.\n\n"
        "• /clear — امسح الذاكرة وابدأ من الأول\n"
        "• /whoami — chat\\_id + user\\_id",
        parse_mode="Markdown"
    )


async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if not is_authorized(update):
        await update.message.reply_text(_auth_fail_text(update))
        return
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    await update.message.reply_text("🗑 تم مسح الذاكرة — الشات بدأ من الأول.")


async def whoami_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    chat_id  = update.effective_chat.id if update.effective_chat else "unknown"
    user_id  = update.effective_user.id if update.effective_user else "unknown"
    username = update.effective_user.username if update.effective_user else "unknown"
    msgs     = len(_history.get(chat_id, []))
    await update.message.reply_text(
        f"chat_id={chat_id}\nuser_id={user_id}\nusername=@{username}\nhistory={msgs} messages"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    if not is_authorized(update):
        logger.warning("Unauthorized: chat=%s user=%s",
                       getattr(update.effective_chat, "id", None),
                       getattr(update.effective_user, "id", None))
        await update.message.reply_text(_auth_fail_text(update))
        return

    chat_id   = update.effective_chat.id if update.effective_chat else 0
    msg_id    = update.message.message_id
    dedup_key = f"{chat_id}:{msg_id}"
    now       = time.time()

    for k in [k for k, ts in _seen_messages.items() if now - ts > DEDUP_TTL_SEC]:
        _seen_messages.pop(k, None)
    if dedup_key in _seen_messages:
        return
    _seen_messages[dedup_key] = now

    user_message = update.message.text.strip()
    if not user_message:
        await update.message.reply_text("Please send a non-empty message.")
        return

    logger.info("user=%s hist=%d msg=%s",
                getattr(update.effective_user, "id", "?"),
                len(get_history(chat_id)), user_message[:80])

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    thinking_msg = await update.message.reply_text("⏳ شغّال…")

    full_prompt = build_prompt(chat_id, user_message)
    result = await run_codex_query(full_prompt)

    # Save to history
    add_to_history(chat_id, "user", user_message)
    add_to_history(chat_id, "assistant", result[:500])  # store trimmed reply

    try:
        await thinking_msg.delete()
    except Exception:
        pass

    for part in split_message(result):
        await update.message.reply_text(part)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not BOT_TOKEN or BOT_TOKEN == "BOT_TOKEN":
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN environment variable.")

    if ALLOW_ALL:
        logger.warning("TELEGRAM_ALLOW_ALL=1 — auth bypassed.")
    else:
        if not ALLOWED_CHAT_IDS and not ALLOWED_USER_IDS:
            raise RuntimeError("No allowlists configured.")
        logger.info("Authorization active. allowed_chats=%s allowed_users=%s",
                    sorted(ALLOWED_CHAT_IDS), sorted(ALLOWED_USER_IDS))

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",  start_cmd))
    app.add_handler(CommandHandler("clear",  clear_cmd))
    app.add_handler(CommandHandler("whoami", whoami_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
