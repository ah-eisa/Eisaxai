#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/ubuntu/investwise"
LOG_FILE="$BASE_DIR/codex_bot.log"
PYTHON_BIN="$BASE_DIR/venv/bin/python3"
BOT_SCRIPT="$BASE_DIR/telegram_codex_bot.py"
ENV_FILE="$BASE_DIR/.env"

cd "$BASE_DIR"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
  echo "ERROR: TELEGRAM_BOT_TOKEN is not set."
  exit 1
fi

if [[ "${TELEGRAM_ALLOW_ALL:-0}" != "1" && -z "${TELEGRAM_ALLOWED_CHAT_IDS:-${TELEGRAM_CHAT_ID:-}}" && -z "${TELEGRAM_ALLOWED_USER_IDS:-}" ]]; then
  echo "ERROR: No Telegram allowlist configured. Set TELEGRAM_CHAT_ID or TELEGRAM_ALLOWED_CHAT_IDS/TELEGRAM_ALLOWED_USER_IDS."
  exit 1
fi

if pgrep -f "telegram_codex_bot.py" >/dev/null 2>&1; then
  echo "Bot already running."
  exit 0
fi

nohup "$PYTHON_BIN" "$BOT_SCRIPT" >> "$LOG_FILE" 2>&1 &
echo "Bot started (PID: $!). Logs: $LOG_FILE"
