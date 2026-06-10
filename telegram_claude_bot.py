#!/usr/bin/env python3
"""
Telegram bot that routes messages to Claude Code (claude -p).
Supports per-chat conversation memory and /clear to reset.
"""
import asyncio
import logging
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import List

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

MAX_TELEGRAM_MSG   = 4096
CLAUDE_TIMEOUT_SEC = 300
DEDUP_TTL_SEC      = 600
MAX_HISTORY        = 40   # max messages kept per chat (user+assistant pairs)

CLAUDE_BIN = "/home/ubuntu/.local/bin/claude"
WORK_DIR   = "/home/ubuntu"
SESSION_DB_PATH = Path("/home/ubuntu/investwise/data/claude_telegram_bot.db")

# Claude stores session files here (key = working-dir path with / → -)
_CLAUDE_SESSIONS_DIR = Path.home() / ".claude" / "projects" / "-home-ubuntu"

def _session_file_exists(session_id: str) -> bool:
    """True if Claude already has a session file for this ID on disk."""
    return (_CLAUDE_SESSIONS_DIR / f"{session_id}.jsonl").exists()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN_CLAUDE", "").strip()
ALLOW_ALL = os.getenv("TELEGRAM_ALLOW_ALL", "0").strip() == "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("telegram_claude_bot")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

_seen_messages: dict[str, float] = {}

# ── Per-chat session IDs (persisted to disk) ─────────────────────────────────
_chat_locks: dict[int, asyncio.Lock] = {}


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[int, str] = {}
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    chat_id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get_or_create(self, chat_id: int) -> str:
        with self._lock:
            cached = self._cache.get(chat_id)
            if cached:
                return cached

            with self._connect() as conn:
                row = conn.execute(
                    "SELECT session_id FROM chat_sessions WHERE chat_id = ?",
                    (chat_id,),
                ).fetchone()
                if row and row["session_id"]:
                    session_id = str(row["session_id"])
                else:
                    session_id = str(uuid.uuid4())
                    conn.execute(
                        """
                        INSERT INTO chat_sessions(chat_id, session_id, updated_at)
                        VALUES(?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(chat_id) DO UPDATE SET
                            session_id = excluded.session_id,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (chat_id, session_id),
                    )
                    conn.commit()

            self._cache[chat_id] = session_id
            return session_id

    def set(self, chat_id: int, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions(chat_id, session_id, updated_at)
                VALUES(?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(chat_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (chat_id, session_id),
            )
            conn.commit()
            self._cache[chat_id] = session_id

    def clear(self, chat_id: int) -> str:
        session_id = str(uuid.uuid4())
        self.set(chat_id, session_id)
        return session_id


session_store = SessionStore(SESSION_DB_PATH)


def get_session(chat_id: int) -> str:
    return session_store.get_or_create(chat_id)

def get_chat_lock(chat_id: int) -> asyncio.Lock:
    lock = _chat_locks.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        _chat_locks[chat_id] = lock
    return lock

def clear_session(chat_id: int) -> None:
    session_id = session_store.clear(chat_id)
    logger.info("Session cleared for chat_id=%s → new session %s", chat_id, session_id)


# ── Auth ──────────────────────────────────────────────────────────────────────

def _parse_id_set(raw: str) -> set[int]:
    values: set[int] = set()
    for part in (raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.add(int(token))
        except ValueError:
            logger.warning("Invalid numeric id: %s", token)
    return values


_allowed_chats_env = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "").strip()
_single_chat       = os.getenv("TELEGRAM_CHAT_ID", "").strip()
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


def _auth_fail(update: Update) -> str:
    chat_id = update.effective_chat.id if update.effective_chat else "?"
    user_id = update.effective_user.id if update.effective_user else "?"
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


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are EisaX Server Assistant — a powerful AI agent with full control over the EisaX production server.

SERVER ENVIRONMENT:
- OS: Ubuntu, user: ubuntu, home: /home/ubuntu
- Main project: /home/ubuntu/investwise (FastAPI + gunicorn, port 8000)
- Frontend: /home/ubuntu/eisax-ui (static HTML/JS served by nginx)
- Webmail PWA: /home/ubuntu/eisax-webmail (built to dist/, served at /webmail/)
- Nginx config: /etc/nginx/sites-enabled/eisax
- Gunicorn master PID: 149096 (reload with: kill -HUP 149096)
- Python venv: /home/ubuntu/investwise/venv/bin/python3
- Key files: api_bridge_v2.py, .env, core/, telegram_claude_bot.py

RUNNING SERVICES:
- gunicorn (port 8000) — EisaX AI API
- nginx — reverse proxy + static files
- telegram-codex-bot.service — Codex Telegram bot
- telegram-claude-bot.service — this bot (you)

DATABASES:
- /home/ubuntu/investwise/investwise.db — users, sessions
- /home/ubuntu/investwise/analysis_cache.db — analysis cache

YOUR CAPABILITIES:
- Read/write any file on the server
- Run any bash command (sudo available)
- Edit nginx config and reload: sudo nginx -s reload
- Restart services: sudo systemctl restart <service>
- Deploy code changes immediately

INSTRUCTIONS:
- The user talks to you in Arabic or English — understand their intent and DO the task
- Don't just explain — actually execute, fix, create, or change whatever they ask
- After doing the task, confirm what you did in simple Arabic or English
- If something fails, debug and fix it automatically
- Be proactive: if you notice a related issue while doing a task, fix it too
- You have memory of this conversation — use it to understand context from previous messages
"""


# ── Claude runner ─────────────────────────────────────────────────────────────

async def run_claude(user_message: str, session_id: str, chat_id: int) -> str:
    """
    Run Claude with session continuity.
    - Uses --resume  if the session file already exists on disk (survives restarts).
    - Uses --session-id for brand-new sessions.
    - If both fail → generates a fresh session UUID and retries once.
    """
    async def _invoke(sid: str, resume: bool) -> tuple[int, str, str]:
        cmd = [
            CLAUDE_BIN, "-p", user_message,
            "--dangerously-skip-permissions",
            "--output-format", "text",
            "--append-system-prompt", SYSTEM_PROMPT,
            "--add-dir", "/home/ubuntu",
            "--add-dir", "/etc/nginx",
            "--add-dir", "/etc/systemd",
            "--add-dir", "/var/log",
        ]
        if resume:
            cmd.extend(["--resume", sid])
        else:
            cmd.extend(["--session-id", sid])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=WORK_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=CLAUDE_TIMEOUT_SEC
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return 124, "", "⏱ Timeout: Claude took more than 5 minutes."

        out = (stdout or b"").decode("utf-8", errors="replace").strip()
        err = (stderr or b"").decode("utf-8", errors="replace").strip()
        return proc.returncode, out, err

    def _is_in_use(out: str, err: str) -> bool:
        combined = f"{out or ''}\n{err or ''}".lower()
        return "session id" in combined and "already in use" in combined

    try:
        # ── Attempt 1: disk-based resume detection (survives restarts) ──────
        resume = _session_file_exists(session_id)
        logger.info("chat=%s session=%s resume=%s", chat_id, session_id, resume)
        rc, out, err = await _invoke(session_id, resume)

        # ── Attempt 2: if "already in use" flip resume flag and retry ───────
        if rc != 0 and _is_in_use(out, err):
            logger.info("Session %s in use — retrying with resume=True", session_id)
            rc, out, err = await _invoke(session_id, True)

        # ── Attempt 3: session is truly stuck → fresh UUID ───────────────────
        if rc != 0 and _is_in_use(out, err):
            new_sid = str(uuid.uuid4())
            logger.warning(
                "Session %s stuck — generating fresh session %s for chat %s",
                session_id, new_sid, chat_id,
            )
            session_store.set(chat_id, new_sid)
            rc, out, err = await _invoke(new_sid, False)

        if rc == 124 and err.startswith("⏱ Timeout:"):
            logger.error("chat=%s session=%s TIMEOUT after %ss", chat_id, session_id, CLAUDE_TIMEOUT_SEC)
            return err

        if rc != 0:
            details = err or out or "Unknown error"
            logger.error("chat=%s session=%s Claude exit=%s stderr=%s", chat_id, session_id, rc, details[:500])
            return f"❌ Claude failed (exit {rc}):\n{details[:2000]}"

        logger.info("chat=%s session=%s Claude OK len=%s", chat_id, session_id, len(out))
        return out or err or "✅ Done — no text output."

    except FileNotFoundError:
        logger.error("chat=%s Claude binary not found at %s", chat_id, CLAUDE_BIN)
        return f"❌ Claude binary not found at: {CLAUDE_BIN}"
    except Exception as exc:
        logger.error("chat=%s session=%s unexpected error: %s", chat_id, session_id, exc, exc_info=True)
        return f"❌ Unexpected error: {exc}"


# ── Command handlers ──────────────────────────────────────────────────────────

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if not is_authorized(update):
        await update.message.reply_text(_auth_fail(update))
        return
    await update.message.reply_text(
        "🤖 *Claude Code bot is online.*\n\n"
        "أنا بتذكر الشات كله — كل رسالة بتبعتها بتكمّل على اللي قبلها.\n\n"
        "• /clear — امسح الذاكرة وابدأ من الأول\n"
        "• /whoami — chat\\_id + user\\_id\n\n"
        "ابعت أي تاسك وأنا هنفذه على السيرفر مباشرة.",
        parse_mode="Markdown"
    )


async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if not is_authorized(update):
        await update.message.reply_text(_auth_fail(update))
        return
    chat_id = update.effective_chat.id
    clear_session(chat_id)
    await update.message.reply_text("🗑 تم مسح الذاكرة — الشات بدأ من الأول.")


async def whoami_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    chat_id  = update.effective_chat.id if update.effective_chat else "?"
    user_id  = update.effective_user.id if update.effective_user else "?"
    username = update.effective_user.username if update.effective_user else "?"
    session  = get_session(chat_id) if isinstance(chat_id, int) else "none"
    await update.message.reply_text(
        f"chat_id={chat_id}\nuser_id={user_id}\nusername=@{username}\nsession={session}"
    )


async def clone_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate speech using the cloned voice. Usage: /clone النص هنا"""
    if not update.message:
        return
    if not is_authorized(update):
        await update.message.reply_text(_auth_fail(update))
        return

    text = " ".join(context.args) if context.args else ""
    if not text:
        await update.message.reply_text("الاستخدام: /clone النص الذي تريد سماعه")
        return

    chat_id = update.effective_chat.id if update.effective_chat else 0
    sample  = Path("/home/ubuntu/investwise/data/voice_samples/sample_933252341.ogg")
    if not sample.exists():
        await update.message.reply_text("❌ لا يوجد نموذج صوتي محفوظ. ابعت رسالة صوتية أولاً.")
        return

    thinking = await update.message.reply_text("🎙 جاري توليد الصوت…")
    try:
        import asyncio, tempfile
        out_wav = Path(tempfile.mktemp(suffix=".wav"))

        def _run_clone():
            import subprocess as _sp
            venv_py = "/home/ubuntu/investwise/venv_cpu_20260409_123021/bin/python3"
            script = (
                "import os,torch,numpy as np,scipy.io.wavfile as wav; "
                "os.environ['COQUI_TOS_AGREED']='1'; "
                "from TTS.api import TTS; "
                "tts=TTS('tts_models/multilingual/multi-dataset/xtts_v2',progress_bar=False); "
                "m=tts.synthesizer.tts_model; m.eval(); "
                "g,s=m.get_conditioning_latents(audio_path=['/tmp/sample_clean.wav'],gpt_cond_len=6,max_ref_length=10); "
                f"out=m.inference(text={text!r},language='ar',gpt_cond_latent=g,speaker_embedding=s,"
                "temperature=0.65,repetition_penalty=7.0,top_k=50,top_p=0.80,speed=1.25); "
                "a=out['wav']; a=a.cpu().numpy() if hasattr(a,'cpu') else a; "
                f"wav.write({str(out_wav)!r},24000,a.astype(np.float32))"
            )
            r = _sp.run([venv_py, "-c", script], capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-500:])

        await asyncio.get_event_loop().run_in_executor(None, _run_clone)
        await thinking.delete()
        with open(out_wav, "rb") as f:
            await update.message.reply_voice(voice=f)
        out_wav.unlink(missing_ok=True)
    except Exception as exc:
        await thinking.delete()
        await update.message.reply_text(f"❌ خطأ: {exc}")


VOICE_SAMPLES_DIR = Path("/home/ubuntu/investwise/data/voice_samples")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download and save voice/audio messages for cloning."""
    if not update.message:
        return
    if not is_authorized(update):
        await update.message.reply_text(_auth_fail(update))
        return

    voice = update.message.voice or update.message.audio
    if not voice:
        return

    chat_id = update.effective_chat.id if update.effective_chat else 0
    file_id = voice.file_id
    duration = getattr(voice, "duration", 0)
    mime = getattr(voice, "mime_type", "audio/ogg")
    ext = "ogg" if "ogg" in mime else "mp3" if "mp3" in mime else "oga"

    VOICE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VOICE_SAMPLES_DIR / f"sample_{chat_id}.{ext}"

    try:
        tg_file = await context.bot.get_file(file_id)
        await tg_file.download_to_drive(str(out_path))
        logger.info("voice saved: file_id=%s path=%s duration=%ss", file_id, out_path, duration)
        await update.message.reply_text(
            f"✅ تم حفظ الصوت ({duration}s) في:\n`{out_path}`\n\n"
            "جاهز للاستخدام في voice cloning. أرسل /clone لبدء العملية.",
            parse_mode="Markdown"
        )
    except Exception as exc:
        logger.error("voice download failed: %s", exc)
        await update.message.reply_text(f"❌ فشل تحميل الصوت: {exc}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    if not is_authorized(update):
        logger.warning("Unauthorized: chat=%s user=%s",
                       getattr(update.effective_chat, "id", None),
                       getattr(update.effective_user, "id", None))
        await update.message.reply_text(_auth_fail(update))
        return

    chat_id = update.effective_chat.id if update.effective_chat else 0
    msg_id  = update.message.message_id
    dedup_key = f"{chat_id}:{msg_id}"
    now = time.time()

    for k in [k for k, ts in _seen_messages.items() if now - ts > DEDUP_TTL_SEC]:
        _seen_messages.pop(k, None)
    if dedup_key in _seen_messages:
        return
    _seen_messages[dedup_key] = now

    user_message = update.message.text.strip()
    if not user_message:
        await update.message.reply_text("Please send a non-empty message.")
        return

    async with get_chat_lock(chat_id):
        session_id = get_session(chat_id)
        logger.info("user=%s session=%s msg=%s",
                    getattr(update.effective_user, "id", "?"), session_id, user_message[:80])

        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        thinking_msg = await update.message.reply_text("⏳ شغّال…")

        result = await run_claude(user_message, session_id, chat_id)

        try:
            await thinking_msg.delete()
        except Exception:
            pass

        for part in split_message(result):
            await update.message.reply_text(part)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN_CLAUDE in environment.")

    if ALLOW_ALL:
        logger.warning("TELEGRAM_ALLOW_ALL=1 — auth bypassed.")
    else:
        if not ALLOWED_CHAT_IDS and not ALLOWED_USER_IDS:
            raise RuntimeError("No allowlists configured.")
        logger.info("Auth active. allowed_chats=%s allowed_users=%s",
                    sorted(ALLOWED_CHAT_IDS), sorted(ALLOWED_USER_IDS))

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",  start_cmd))
    app.add_handler(CommandHandler("clear",  clear_cmd))
    app.add_handler(CommandHandler("whoami", whoami_cmd))
    app.add_handler(CommandHandler("clone",  clone_cmd))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
