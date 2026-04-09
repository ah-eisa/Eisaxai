#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# EisaX — Daily SQLite WAL Backup
# Backs up all production SQLite databases, keeps last 7 copies per DB,
# logs to /var/log/eisax-backup.log.
#
# Usage: run manually or via systemd eisax-backup.service
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BACKUP_DIR="/home/ubuntu/backups"
LOG="/var/log/eisax-backup.log"
TIMESTAMP=$(date '+%Y%m%d_%H%M')
DATE_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"

# ── Helper: log to file (and stdout for journald capture) ──────────────────
log() {
    local level="$1"; shift
    local msg="$*"
    echo "$DATE_HUMAN [$level] $msg" | tee -a "$LOG"
}

log "INFO" "=== EisaX DB backup started (timestamp: $TIMESTAMP) ==="

# ── Databases to back up ───────────────────────────────────────────────────
# Format: "label:source_path:required(yes/no)"
declare -a DBS=(
    "app:/home/ubuntu/investwise/investwise.db:yes"
    "core:/home/ubuntu/investwise/core/investwise.db:yes"
    "price_cache:/home/ubuntu/investwise/price_cache.db:yes"
    "sessions:/home/ubuntu/investwise/data/sessions.db:no"
    "vector_memory:/home/ubuntu/investwise/data/vector_memory.db:no"
    "analysis_cache:/home/ubuntu/investwise/analysis_cache.db:no"
)

FAILED=0
SUCCESS=0

for entry in "${DBS[@]}"; do
    IFS=':' read -r label src required <<< "$entry"

    # Skip optional DBs that don't exist
    if [[ ! -f "$src" ]]; then
        if [[ "$required" == "yes" ]]; then
            log "ERROR" "Required DB not found: $src"
            FAILED=$((FAILED + 1))
        else
            log "INFO"  "Optional DB not present, skipping: $src"
        fi
        continue
    fi

    dest="$BACKUP_DIR/${label}_${TIMESTAMP}.db"

    # Force WAL checkpoint so WAL frames are flushed into the main DB file
    log "INFO" "Checkpointing WAL for $src ..."
    if ! sqlite3 "$src" "PRAGMA wal_checkpoint(TRUNCATE);" > /dev/null 2>&1; then
        log "WARN" "WAL checkpoint failed for $src (DB may not use WAL — continuing)"
    fi

    # SQLite online backup API (safe for live databases)
    log "INFO" "Backing up $src -> $dest"
    if sqlite3 "$src" ".backup '$dest'"; then
        SIZE=$(du -sh "$dest" | cut -f1)
        log "INFO" "SUCCESS: $label backup written ($SIZE) -> $dest"
        SUCCESS=$((SUCCESS + 1))
    else
        log "ERROR" "FAILED to back up $src"
        FAILED=$((FAILED + 1))
        # Alert via Telegram if credentials are available
        if [[ -f /home/ubuntu/investwise/.env ]]; then
            # shellcheck disable=SC1091
            source /home/ubuntu/investwise/.env 2>/dev/null || true
            if [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]]; then
                curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
                    -d chat_id="${TELEGRAM_CHAT_ID}" \
                    -d text="EisaX DB Backup FAILED: $label on $(date '+%Y-%m-%d %H:%M')" \
                    > /dev/null || true
            fi
        fi
        continue
    fi

    # ── Prune: keep only the 7 most recent backups for this label ──────────
    # List backups for this label sorted newest-first, delete beyond 7
    mapfile -t old_backups < <(
        ls -t "$BACKUP_DIR/${label}_"*.db 2>/dev/null | tail -n +8
    )
    for old in "${old_backups[@]}"; do
        rm -f "$old"
        log "INFO" "Pruned old backup: $old"
    done
done

log "INFO" "=== Backup complete — $SUCCESS succeeded, $FAILED failed ==="

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
