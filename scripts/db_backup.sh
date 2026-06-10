#!/bin/bash
# Daily SQLite backup — keeps last 7 days

DB_SOURCE="/home/ubuntu/investwise/core/investwise.db"
BACKUP_DIR="/home/ubuntu/backups/db"
DATE=$(date +%Y-%m-%d)
BACKUP_FILE="$BACKUP_DIR/investwise_$DATE.db"
LOG="/home/ubuntu/investwise/backend.log"

mkdir -p "$BACKUP_DIR"

# Backup using SQLite safe copy
if sqlite3 "$DB_SOURCE" ".backup '$BACKUP_FILE'"; then
    SIZE=$(du -sh "$BACKUP_FILE" | cut -f1)
    echo "$(date '+%Y-%m-%d %H:%M:%S') INFO [backup] SUCCESS: $BACKUP_FILE ($SIZE)" >> "$LOG"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR [backup] FAILED: could not backup database" >> "$LOG"
    # Alert via Telegram
    source /home/ubuntu/investwise/.env
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="${TELEGRAM_CHAT_ID}" \
        -d text="🔴 EisaX DB Backup FAILED on $(date '+%Y-%m-%d %H:%M')" > /dev/null
    exit 1
fi

# Keep only last 7 days
find "$BACKUP_DIR" -name "investwise_*.db" -mtime +7 -delete

echo "$(date '+%Y-%m-%d %H:%M:%S') INFO [backup] Old backups cleaned. Current backups:" >> "$LOG"
ls -lh "$BACKUP_DIR/" >> "$LOG"
