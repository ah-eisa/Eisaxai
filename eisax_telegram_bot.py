#!/usr/bin/env python3
import asyncio
import base64
import contextlib
import io
import logging
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import httpx
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.helpers import escape_markdown

EISAX_API_URL = "http://localhost:8000"
EISAX_TOKEN = os.getenv("SECURE_TOKEN", "")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN_EISAX", "").strip()
MAX_TELEGRAM_MSG = 4096
LOCAL_THROTTLE_SECONDS = 2.0
HTTP_TIMEOUT = httpx.Timeout(120.0, connect=15.0, read=120.0, write=120.0)
SESSION_DB_PATH = Path("/home/ubuntu/investwise/data/eisax_telegram_bot.db")
_MDV2_SPECIALS = set("_*[]()~`>#+-=|{}.!\\")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eisax_telegram_bot")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)


def _parse_id_set(raw: str) -> set[int]:
    values: set[int] = set()
    for part in (raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.add(int(token))
        except ValueError:
            logger.warning("Ignoring invalid TELEGRAM_ALLOWED_CHATS entry: %s", token)
    return values


ALLOWED_CHAT_IDS = _parse_id_set(os.getenv("TELEGRAM_ALLOWED_CHATS", "").strip())


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
last_message_at: dict[int, float] = {}
seen_updates: dict[str, float] = {}


def is_authorized(chat_id: int) -> bool:
    return not ALLOWED_CHAT_IDS or chat_id in ALLOWED_CHAT_IDS


def mdv2(text: str) -> str:
    return escape_markdown(text or "", version=2)


def split_for_markdown_v2(text: str, limit: int = MAX_TELEGRAM_MSG) -> list[str]:
    if not text:
        return [""]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    last_break = -1

    for ch in text:
        escaped_len = 2 if ch in _MDV2_SPECIALS else 1
        if current and current_len + escaped_len > limit:
            if last_break >= 0:
                chunk = "".join(current[:last_break]).rstrip()
                remainder = "".join(current[last_break:]).lstrip()
            else:
                chunk = "".join(current)
                remainder = ""

            if chunk:
                chunks.append(chunk)

            current = list(remainder)
            current_len = sum(2 if c in _MDV2_SPECIALS else 1 for c in current)
            last_break = max((idx for idx, c in enumerate(current) if c in {" ", "\n"}), default=-1)

            if current and current_len + escaped_len > limit:
                chunks.append("".join(current))
                current = []
                current_len = 0
                last_break = -1

        current.append(ch)
        current_len += escaped_len
        if ch in {" ", "\n"}:
            last_break = len(current)

    if current:
        chunks.append("".join(current))

    return chunks


async def send_markdown_reply(message, text: str) -> None:
    body = text or "(no response)"
    for chunk in split_for_markdown_v2(body):
        safe = mdv2(chunk)
        if not safe:
            safe = mdv2("(empty)")
        await message.reply_text(
            safe,
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=True,
        )


def _api_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if EISAX_TOKEN:
        headers["X-API-Key"] = EISAX_TOKEN
    return headers


async def typing_loop(bot, chat_id: int, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            return
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            continue


def should_throttle(chat_id: int) -> bool:
    now = time.monotonic()
    previous = last_message_at.get(chat_id, 0.0)
    if now - previous < LOCAL_THROTTLE_SECONDS:
        return True
    last_message_at[chat_id] = now
    return False


def mark_seen(update: Update) -> bool:
    if not update.effective_chat or not update.effective_message:
        return False

    now = time.monotonic()
    for key, seen_at in list(seen_updates.items()):
        if now - seen_at > 600:
            seen_updates.pop(key, None)

    dedup_key = f"{update.effective_chat.id}:{update.effective_message.message_id}"
    if dedup_key in seen_updates:
        return True

    seen_updates[dedup_key] = now
    return False


def format_health_summary(payload: dict) -> str:
    status = str(payload.get("status", "unknown")).upper()
    summary = str(payload.get("summary") or "No summary returned.")
    version = str(payload.get("version") or "unknown")
    uptime = payload.get("uptime_seconds")
    worker_uptime = payload.get("worker_uptime_seconds")
    services = payload.get("services") or {}

    lines = [
        f"EisaX status: {status}",
        f"Summary: {summary}",
        f"Version: {version}",
    ]

    if isinstance(uptime, (int, float)):
        lines.append(f"API uptime: {uptime / 3600:.1f} hours")
    if isinstance(worker_uptime, (int, float)):
        lines.append(f"Worker uptime: {worker_uptime / 3600:.1f} hours")

    if isinstance(services, dict) and services:
        ok_count = sum(1 for item in services.values() if isinstance(item, dict) and item.get("status") == "ok")
        lines.append(f"Healthy services: {ok_count}/{len(services)}")
        for name, detail in services.items():
            if not isinstance(detail, dict):
                continue
            svc_status = str(detail.get("status", "unknown")).upper()
            latency = detail.get("latency_ms")
            info = str(detail.get("detail") or "").strip()
            latency_text = f" ({latency:.0f} ms)" if isinstance(latency, (int, float)) else ""
            if info:
                lines.append(f"- {name}: {svc_status}{latency_text} - {info}")
            else:
                lines.append(f"- {name}: {svc_status}{latency_text}")

    return "\n".join(lines)


def format_alerts(alerts: list[dict]) -> str:
    if not alerts:
        return "No price alerts are active for this chat."

    lines = [f"Active alerts for this chat: {len(alerts)}"]
    for alert in alerts[:25]:
        ticker = str(alert.get("ticker", "?")).upper()
        condition = str(alert.get("condition", "?")).lower()
        threshold = alert.get("threshold")
        status = "active" if int(alert.get("active", 1) or 0) else "triggered"
        created_at = str(alert.get("created_at") or "unknown")
        lines.append(
            f"- #{alert.get('id', '?')}: {ticker} {condition} {threshold} [{status}] created {created_at}"
        )

    if len(alerts) > 25:
        lines.append(f"... plus {len(alerts) - 25} more")

    return "\n".join(lines)


async def send_api_error(message, exc: Exception | None = None, detail: str | None = None) -> None:
    if detail:
        text = f"EisaX API error: {detail}"
    elif exc:
        text = f"EisaX is temporarily unavailable. Please try again in a moment.\nDetails: {exc}"
    else:
        text = "EisaX is temporarily unavailable. Please try again in a moment."
    await send_markdown_reply(message, text)


async def post_chat_message(
    client: httpx.AsyncClient,
    chat_id: int,
    message_text: str,
    session_id: str,
    *,
    active_file_id: str | None = None,
    inline_files: list[dict] | None = None,
) -> dict:
    payload: dict = {
        "message": message_text,
        "user_id": str(chat_id),
        "session_id": session_id,
    }
    if active_file_id:
        payload["settings"] = {"active_file_id": active_file_id}
    if inline_files:
        payload["files"] = inline_files

    response = await client.post(
        f"{EISAX_API_URL}/v1/chat",
        json=payload,
        headers=_api_headers(),
    )
    response.raise_for_status()
    return response.json()


async def upload_file_to_api(
    client: httpx.AsyncClient,
    filename: str,
    content: bytes,
    mime_type: str,
) -> str:
    b64data = base64.b64encode(content).decode("ascii")

    try:
        response = await client.post(
            f"{EISAX_API_URL}/upload",
            json={"filename": filename, "data": b64data},
            headers=_api_headers(),
        )
        if response.is_success:
            payload = response.json()
            file_id = payload.get("file_id")
            if file_id:
                return str(file_id)
    except httpx.RequestError:
        raise
    except Exception:
        logger.exception("Compatibility JSON upload attempt failed")

    files = {
        "file": (filename, io.BytesIO(content), mime_type or "application/octet-stream"),
    }
    response = await client.post(
        f"{EISAX_API_URL}/upload",
        files=files,
        headers=_api_headers(),
    )
    response.raise_for_status()
    payload = response.json()
    file_id = payload.get("file_id")
    if not file_id:
        raise RuntimeError("Upload completed but no file_id was returned.")
    return str(file_id)


async def handle_authorization(update: Update) -> bool:
    if not update.effective_chat or not update.effective_message:
        return False

    chat_id = update.effective_chat.id
    if is_authorized(chat_id):
        return True

    await send_markdown_reply(
        update.effective_message,
        f"Unauthorized chat.\nchat_id: {chat_id}",
    )
    return False


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message or not update.effective_chat:
        return
    if not await handle_authorization(update):
        return

    session_store.get_or_create(update.effective_chat.id)
    text = (
        "Welcome to EisaX.\n"
        "I can route your messages to the /v1/chat API, keep a persistent chat session, "
        "check system health, list price alerts, create alerts, and analyze uploaded files.\n\n"
        "Commands:\n"
        "/help\n"
        "/clear\n"
        "/status\n"
        "/alerts\n"
        "/alert TICKER above PRICE\n"
        "/alert TICKER below PRICE"
    )
    await send_markdown_reply(update.effective_message, text)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    if not await handle_authorization(update):
        return

    text = (
        "EisaX capabilities:\n"
        "- Ask any market or portfolio question and I will send it to /v1/chat.\n"
        "- /status checks /v1/health.\n"
        "- /alerts lists active alerts for this Telegram chat.\n"
        "- /alert TICKER above PRICE creates an upside alert.\n"
        "- /alert TICKER below PRICE creates a downside alert.\n"
        "- Send a document or photo and I will upload it before asking EisaX to analyze it.\n"
        "- /clear resets this chat to a new persistent session."
    )
    await send_markdown_reply(update.effective_message, text)


async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message or not update.effective_chat:
        return
    if not await handle_authorization(update):
        return

    session_id = session_store.clear(update.effective_chat.id)
    await send_markdown_reply(
        update.effective_message,
        f"Conversation history cleared. New session_id: {session_id}",
    )


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    if not await handle_authorization(update):
        return

    client: httpx.AsyncClient = context.application.bot_data["http_client"]
    try:
        response = await client.get(f"{EISAX_API_URL}/v1/health", headers=_api_headers())
        payload = response.json()
        if response.is_success:
            await send_markdown_reply(update.effective_message, format_health_summary(payload))
            return
        detail = payload.get("detail") if isinstance(payload, dict) else response.text
        await send_api_error(update.effective_message, detail=str(detail))
    except httpx.RequestError as exc:
        await send_api_error(update.effective_message, exc=exc)
    except Exception as exc:
        await send_api_error(update.effective_message, exc=exc)


async def alerts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message or not update.effective_chat:
        return
    if not await handle_authorization(update):
        return

    client: httpx.AsyncClient = context.application.bot_data["http_client"]
    chat_id = update.effective_chat.id
    try:
        response = await client.get(
            f"{EISAX_API_URL}/v1/alerts",
            params={"user_id": str(chat_id)},
            headers=_api_headers(),
        )
        payload = response.json()
        if response.is_success and isinstance(payload, list):
            await send_markdown_reply(update.effective_message, format_alerts(payload))
            return
        detail = payload.get("detail") if isinstance(payload, dict) else response.text
        await send_api_error(update.effective_message, detail=str(detail))
    except httpx.RequestError as exc:
        await send_api_error(update.effective_message, exc=exc)
    except Exception as exc:
        await send_api_error(update.effective_message, exc=exc)


async def alert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message or not update.effective_chat:
        return
    if not await handle_authorization(update):
        return

    if len(context.args) != 3:
        await send_markdown_reply(
            update.effective_message,
            "Usage: /alert TICKER above PRICE or /alert TICKER below PRICE",
        )
        return

    ticker = context.args[0].upper().strip()
    condition = context.args[1].lower().strip()
    threshold_raw = context.args[2].strip()
    if condition not in {"above", "below"}:
        await send_markdown_reply(
            update.effective_message,
            "Condition must be either above or below.",
        )
        return

    try:
        threshold = float(threshold_raw)
    except ValueError:
        await send_markdown_reply(update.effective_message, "Price must be numeric.")
        return

    client: httpx.AsyncClient = context.application.bot_data["http_client"]
    try:
        response = await client.post(
            f"{EISAX_API_URL}/v1/alerts",
            json={
                "user_id": str(update.effective_chat.id),
                "ticker": ticker,
                "condition": condition,
                "threshold": threshold,
            },
            headers=_api_headers(),
        )
        payload = response.json()
        if response.is_success:
            await send_markdown_reply(
                update.effective_message,
                (
                    f"Alert created.\n"
                    f"Ticker: {ticker}\n"
                    f"Condition: {condition}\n"
                    f"Threshold: {threshold}\n"
                    f"Alert ID: {payload.get('alert_id', 'unknown')}"
                ),
            )
            return
        detail = payload.get("detail") if isinstance(payload, dict) else response.text
        await send_api_error(update.effective_message, detail=str(detail))
    except httpx.RequestError as exc:
        await send_api_error(update.effective_message, exc=exc)
    except Exception as exc:
        await send_api_error(update.effective_message, exc=exc)


async def process_chat_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    message_text: str,
    *,
    active_file_id: str | None = None,
    inline_files: list[dict] | None = None,
    manage_typing: bool = True,
) -> None:
    if not update.effective_message or not update.effective_chat:
        return

    chat_id = update.effective_chat.id
    client: httpx.AsyncClient = context.application.bot_data["http_client"]
    session_id = session_store.get_or_create(chat_id)
    stop_event = asyncio.Event()
    typing_task = None
    if manage_typing:
        typing_task = asyncio.create_task(typing_loop(context.bot, chat_id, stop_event))

    try:
        result = await post_chat_message(
            client,
            chat_id,
            message_text,
            session_id,
            active_file_id=active_file_id,
            inline_files=inline_files,
        )
        new_session_id = result.get("session_id")
        if isinstance(new_session_id, str) and new_session_id:
            session_store.set(chat_id, new_session_id)

        reply_text = str(result.get("reply") or result.get("response") or "(empty response)")
        await send_markdown_reply(update.effective_message, reply_text)
    except httpx.HTTPStatusError as exc:
        detail = None
        try:
            payload = exc.response.json()
            if isinstance(payload, dict):
                detail = payload.get("detail") or payload.get("error")
        except Exception:
            detail = exc.response.text
        await send_api_error(update.effective_message, detail=detail or str(exc))
    except httpx.RequestError as exc:
        await send_api_error(update.effective_message, exc=exc)
    except Exception as exc:
        logger.exception("Chat request failed")
        await send_api_error(update.effective_message, exc=exc)
    finally:
        stop_event.set()
        if typing_task:
            typing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message or not update.effective_message.text or not update.effective_chat:
        return
    if mark_seen(update):
        return
    if not await handle_authorization(update):
        return

    chat_id = update.effective_chat.id
    if should_throttle(chat_id):
        await send_markdown_reply(
            update.effective_message,
            "Please wait about 2 seconds before sending another message.",
        )
        return

    message_text = update.effective_message.text.strip()
    if not message_text:
        await send_markdown_reply(update.effective_message, "Please send a non-empty message.")
        return

    await process_chat_request(update, context, message_text)


async def handle_attachment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message or not update.effective_chat:
        return
    if mark_seen(update):
        return
    if not await handle_authorization(update):
        return

    chat_id = update.effective_chat.id
    if should_throttle(chat_id):
        await send_markdown_reply(
            update.effective_message,
            "Please wait about 2 seconds before sending another message.",
        )
        return

    message = update.effective_message
    tg_file = None
    filename = "upload.bin"
    mime_type = "application/octet-stream"

    if message.document:
        tg_file = await message.document.get_file()
        filename = message.document.file_name or f"document_{message.document.file_unique_id}"
        mime_type = message.document.mime_type or mime_type
    elif message.photo:
        photo = message.photo[-1]
        tg_file = await photo.get_file()
        filename = f"photo_{photo.file_unique_id}.jpg"
        mime_type = "image/jpeg"
    else:
        await send_markdown_reply(message, "Only documents and photos are supported.")
        return

    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(typing_loop(context.bot, chat_id, stop_event))
    client: httpx.AsyncClient = context.application.bot_data["http_client"]

    try:
        file_bytes = bytes(await tg_file.download_as_bytearray())
        file_id = await upload_file_to_api(client, filename, file_bytes, mime_type)
        prompt = (message.caption or "").strip() or f"Please analyze the uploaded file: {filename}"
        await process_chat_request(
            update,
            context,
            prompt,
            active_file_id=file_id,
            manage_typing=False,
        )
    except httpx.HTTPStatusError as exc:
        detail = None
        try:
            payload = exc.response.json()
            if isinstance(payload, dict):
                detail = payload.get("detail") or payload.get("error")
        except Exception:
            detail = exc.response.text
        await send_api_error(message, detail=detail or str(exc))
    except httpx.RequestError as exc:
        await send_api_error(message, exc=exc)
    except Exception as exc:
        logger.exception("Attachment handling failed")
        await send_api_error(message, exc=exc)
    finally:
        stop_event.set()
        typing_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await typing_task


async def post_init(application: Application) -> None:
    application.bot_data["http_client"] = httpx.AsyncClient(
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
    )
    logger.info("EisaX Telegram bot initialized")


async def post_shutdown(application: Application) -> None:
    client = application.bot_data.get("http_client")
    if client:
        await client.aclose()


def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN_EISAX")

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("clear", clear_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CommandHandler("alerts", alerts_cmd))
    application.add_handler(CommandHandler("alert", alert_cmd))
    application.add_handler(
        MessageHandler(filters.Document.ALL | filters.PHOTO, handle_attachment)
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )

    logger.info(
        "Starting EisaX Telegram bot; allowed chats=%s",
        "ALL" if not ALLOWED_CHAT_IDS else ",".join(str(chat_id) for chat_id in sorted(ALLOWED_CHAT_IDS)),
    )
    application.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
