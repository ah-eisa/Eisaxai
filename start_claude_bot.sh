#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/ubuntu/investwise"
LOG_FILE="$BASE_DIR/claude_bot.log"
PYTHON_BIN="$BASE_DIR/venv/bin/python3"
BOT_SCRIPT="$BASE_DIR/telegram_claude_bot.py"
ENV_FILE="$BASE_DIR/.env"

cd "$BASE_DIR"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

if [[ -z "${TELEGRAM_BOT_TOKEN_CLAUDE:-}" ]]; then
  echo "ERROR: TELEGRAM_BOT_TOKEN_CLAUDE is not set."
  exit 1
fi

if pgrep -f "telegram_claude_bot.py" >/dev/null 2>&1; then
  echo "Claude bot already running."
  exit 0
fi

nohup "$PYTHON_BIN" "$BOT_SCRIPT" >> "$LOG_FILE" 2>&1 &
echo "Claude bot started (PID: $!). Logs: $LOG_FILE"
