#!/bin/bash
#
# Staging workers=2 soak monitor.
# Run every 15 minutes via cron for the 24h validation window.
# Appends one JSON line per tick to LOG.
#
# Tracks:
#   - service active (yes/no)
#   - master + worker PIDs and RSS
#   - APScheduler ownership (lock file PID)
#   - "Scheduler started" count since boot — must be 1
#   - new "Traceback", 5xx, and "429" lines since last tick
#   - parquet cache write count + last-modified delta
#   - last "EisaX News Collection" run timestamp (sanity that scheduler is firing)
#
# Output: /home/ubuntu/investwise/logs/staging_soak.jsonl
# Alerts: /home/ubuntu/investwise/logs/staging_soak_alerts.log

set -u

LOG=/home/ubuntu/investwise/logs/staging_soak.jsonl
ALERT=/home/ubuntu/investwise/logs/staging_soak_alerts.log
STATE=/home/ubuntu/investwise/logs/staging_soak_state.json
SVC=eisax-gunicorn-staging
ERR_LOG=/home/ubuntu/investwise/logs/gunicorn_staging_test_error.log
CACHE_DIR=/home/ubuntu/investwise/market_cache
LOCK_FILE=/tmp/eisax-staging-scheduler.lock

# Establish previous offset (for incremental log reads).
# On first run (no STATE), seed at CURRENT size so historical noise doesn't
# fire false alerts. STATE then advances forward each tick.
CUR_SIZE=$(stat -c %s "$ERR_LOG" 2>/dev/null || echo 0)
if [ -f "$STATE" ]; then
    PREV_OFFSET=$(jq -r '.err_log_offset // 0' "$STATE" 2>/dev/null || echo 0)
else
    PREV_OFFSET=$CUR_SIZE   # first tick — skip historical content
fi
# If file shrank (logrotate), reset offset
if [ "$CUR_SIZE" -lt "$PREV_OFFSET" ]; then PREV_OFFSET=0; fi

# Service state
ACTIVE=$(systemctl is-active "$SVC" 2>/dev/null || echo unknown)
SVC_SINCE=$(systemctl show "$SVC" -p ActiveEnterTimestamp --value 2>/dev/null)

# Process state
MAIN_PID=$(systemctl show "$SVC" -p MainPID --value 2>/dev/null)
# Workers = children of master gunicorn
WORKER_PIDS=$(pgrep -P "$MAIN_PID" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
WORKER_RSS_KB=$(if [ -n "$WORKER_PIDS" ]; then \
  for p in $(echo "$WORKER_PIDS" | tr ',' ' '); do \
    ps -o rss= -p "$p" 2>/dev/null | tr -d ' '; \
  done | paste -sd, - ; \
fi)
WORKER_COUNT=$(echo "$WORKER_PIDS" | awk -F',' '{print NF}')

# Scheduler lock state
LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null | tr -d '\n' | head -c 16)
LOCK_ALIVE=$(if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then echo yes; else echo no; fi)

# Scheduler-started count since service ActiveEnter (sanity: must be 1)
SCHED_STARTED=$(journalctl -u "$SVC" --since "$SVC_SINCE" 2>/dev/null | grep -c "Scheduler started")
SCHED_SKIPPED=$(journalctl -u "$SVC" --since "$SVC_SINCE" 2>/dev/null | grep -c "scheduler skipped (worker is not owner)")
NEWS_FIRES_24H=$(journalctl -u "$SVC" --since "24 hours ago" 2>/dev/null | grep -c "EisaX News Collection")

# New error-log lines (between previous and current offset)
NEW_LINES=$(if [ "$CUR_SIZE" -gt "$PREV_OFFSET" ]; then \
  tail -c +$((PREV_OFFSET + 1)) "$ERR_LOG" 2>/dev/null; \
fi)
NEW_TB=$(echo "$NEW_LINES" | grep -c "Traceback")
NEW_429=$(echo "$NEW_LINES" | grep -ciE "(429|rate.?limit)")
NEW_5XX=$(echo "$NEW_LINES" | grep -ciE "(5[0-9]{2} (Internal|Bad|Service)|status=5[0-9]{2})")
NEW_FUTEX=$(echo "$NEW_LINES" | grep -ciE "(futex_do_wait|fork|deadlock)")

# Cache state
CACHE_FILES=$(find "$CACHE_DIR" -name '*.parquet' -type f 2>/dev/null | wc -l)
CACHE_RECENT=$(find "$CACHE_DIR" -name '*.parquet' -mmin -16 2>/dev/null | wc -l)

# Timestamp
TS=$(date -Is)

# Emit JSON row
ROW=$(printf '{"ts":"%s","active":"%s","master_pid":%s,"worker_pids":"%s","worker_count":%s,"worker_rss_kb":"%s","sched_lock_pid":"%s","sched_lock_alive":"%s","sched_started_n":%s,"sched_skipped_n":%s,"news_fires_24h":%s,"new_tb":%s,"new_429":%s,"new_5xx":%s,"new_futex":%s,"cache_files":%s,"cache_recent":%s,"err_log_offset":%s}\n' \
  "$TS" "$ACTIVE" "${MAIN_PID:-0}" "${WORKER_PIDS:-}" "${WORKER_COUNT:-0}" "${WORKER_RSS_KB:-}" \
  "${LOCK_PID:-none}" "$LOCK_ALIVE" "$SCHED_STARTED" "$SCHED_SKIPPED" "$NEWS_FIRES_24H" \
  "$NEW_TB" "$NEW_429" "$NEW_5XX" "$NEW_FUTEX" "$CACHE_FILES" "$CACHE_RECENT" "$CUR_SIZE")
echo "$ROW" >> "$LOG"

# Save state
echo "{\"err_log_offset\": $CUR_SIZE}" > "$STATE"

# Alert on critical signals
ALERT_LINE=""
if [ "$ACTIVE" != "active" ];          then ALERT_LINE="$ALERT_LINE  service_inactive=$ACTIVE"; fi
if [ "${WORKER_COUNT:-0}" -lt 2 ];     then ALERT_LINE="$ALERT_LINE  worker_count=${WORKER_COUNT:-0}<2"; fi
if [ "$SCHED_STARTED" -gt 1 ];         then ALERT_LINE="$ALERT_LINE  sched_started=$SCHED_STARTED>1"; fi
if [ "$LOCK_ALIVE" = "no" ] && [ "$ACTIVE" = "active" ]; then
  ALERT_LINE="$ALERT_LINE  sched_lock_orphan"
fi
if [ "$NEW_TB" -gt 0 ];                then ALERT_LINE="$ALERT_LINE  tracebacks=$NEW_TB"; fi
if [ "$NEW_5XX" -gt 0 ];               then ALERT_LINE="$ALERT_LINE  5xx=$NEW_5XX"; fi
if [ "$NEW_FUTEX" -gt 0 ];             then ALERT_LINE="$ALERT_LINE  futex_signals=$NEW_FUTEX"; fi
if [ "$NEW_429" -gt 20 ];              then ALERT_LINE="$ALERT_LINE  high_429=$NEW_429"; fi

if [ -n "$ALERT_LINE" ]; then
  echo "[$TS] ALERT$ALERT_LINE" >> "$ALERT"
fi
