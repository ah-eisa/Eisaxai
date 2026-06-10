#!/bin/bash
#
# Production Phase 2 soak monitor.
# Read-only. Mirrors staging_soak_monitor.sh but targets eisax-gunicorn (port 8000).
# Run every 15 minutes via cron for the 4h validation window.
# Appends one JSON line per tick to LOG.
#
# Tracks:
#   - service active (yes/no)
#   - master + worker PIDs and RSS
#   - APScheduler ownership (lock file PID) — staging-validated dedup pattern
#   - "Scheduler started" count since boot — must be 1
#   - new "Traceback", 5xx, and futex lines since last tick
#   - last "EisaX News Collection" run timestamp
#
# Output: /home/ubuntu/investwise/logs/prod_soak.jsonl
# Alerts: /home/ubuntu/investwise/logs/prod_soak_alerts.log

set -u

LOG=/home/ubuntu/investwise/logs/prod_soak.jsonl
ALERT=/home/ubuntu/investwise/logs/prod_soak_alerts.log
STATE=/home/ubuntu/investwise/logs/prod_soak_state.json
SVC=eisax-gunicorn
ERR_LOG=/home/ubuntu/investwise/logs/gunicorn_test_error.log
LOCK_FILE=/tmp/eisax-production-scheduler.lock

CUR_SIZE=$(stat -c %s "$ERR_LOG" 2>/dev/null || echo 0)
if [ -f "$STATE" ]; then
    PREV_OFFSET=$(jq -r '.err_log_offset // 0' "$STATE" 2>/dev/null || echo 0)
else
    PREV_OFFSET=$CUR_SIZE   # first tick — skip historical content
fi
if [ "$CUR_SIZE" -lt "$PREV_OFFSET" ]; then PREV_OFFSET=0; fi

# Service state
ACTIVE=$(systemctl is-active "$SVC" 2>/dev/null || echo unknown)
SVC_SINCE=$(systemctl show "$SVC" -p ActiveEnterTimestamp --value 2>/dev/null)

# Process state
MAIN_PID=$(systemctl show "$SVC" -p MainPID --value 2>/dev/null)
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

# Scheduler-started count since service ActiveEnter (must be 1)
SCHED_STARTED=$(journalctl -u "$SVC" --since "$SVC_SINCE" 2>/dev/null | grep -c "Scheduler started")
SCHED_SKIPPED=$(journalctl -u "$SVC" --since "$SVC_SINCE" 2>/dev/null | grep -c "scheduler skipped (worker is not owner)")
# News fires lines in last hour (rough — each fire emits Running + executed = 2 lines)
NEWS_LINES_1H=$(journalctl -u "$SVC" --since "1 hour ago" 2>/dev/null | grep -c "EisaX News Collection")

# New error-log lines (between previous and current offset)
NEW_LINES=$(if [ "$CUR_SIZE" -gt "$PREV_OFFSET" ]; then \
  tail -c +$((PREV_OFFSET + 1)) "$ERR_LOG" 2>/dev/null; \
fi)
NEW_TB=$(echo "$NEW_LINES" | grep -c "Traceback")
NEW_429=$(echo "$NEW_LINES" | grep -ciE "returned 429|rate.?limit")
NEW_5XX=$(echo "$NEW_LINES" | grep -ciE "status=5[0-9]{2}|5[0-9]{2} (Internal|Bad|Service)")
NEW_FUTEX=$(echo "$NEW_LINES" | grep -ciE "futex_do_wait|deadlock|SIGSEGV|SIGKILL")

# Health endpoint check (cheap, read-only)
TOKEN=$(grep -E "^SECURE_TOKEN=" /home/ubuntu/investwise/.env | cut -d= -f2- | tr -d '"' | tr -d "'")
HEALTH=$(curl -s --max-time 5 -H "X-API-Key: $TOKEN" http://127.0.0.1:8000/health 2>/dev/null \
         | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "unreachable")

TS=$(date -Is)

ROW=$(printf '{"ts":"%s","active":"%s","health":"%s","master_pid":%s,"worker_pids":"%s","worker_count":%s,"worker_rss_kb":"%s","sched_lock_pid":"%s","sched_lock_alive":"%s","sched_started_n":%s,"sched_skipped_n":%s,"news_lines_1h":%s,"new_tb":%s,"new_429":%s,"new_5xx":%s,"new_futex":%s,"err_log_offset":%s}\n' \
  "$TS" "$ACTIVE" "$HEALTH" "${MAIN_PID:-0}" "${WORKER_PIDS:-}" "${WORKER_COUNT:-0}" "${WORKER_RSS_KB:-}" \
  "${LOCK_PID:-none}" "$LOCK_ALIVE" "$SCHED_STARTED" "$SCHED_SKIPPED" "$NEWS_LINES_1H" \
  "$NEW_TB" "$NEW_429" "$NEW_5XX" "$NEW_FUTEX" "$CUR_SIZE")
echo "$ROW" >> "$LOG"

echo "{\"err_log_offset\": $CUR_SIZE}" > "$STATE"

# Alert on critical signals
ALERT_LINE=""
if [ "$ACTIVE" != "active" ];               then ALERT_LINE="$ALERT_LINE  service_inactive=$ACTIVE"; fi
if [ "${WORKER_COUNT:-0}" -lt 2 ];          then ALERT_LINE="$ALERT_LINE  worker_count=${WORKER_COUNT:-0}<2"; fi
if [ "$SCHED_STARTED" -gt 1 ];              then ALERT_LINE="$ALERT_LINE  sched_started=$SCHED_STARTED>1"; fi
if [ "$LOCK_ALIVE" = "no" ] && [ "$ACTIVE" = "active" ]; then
  ALERT_LINE="$ALERT_LINE  sched_lock_orphan"
fi
if [ "$NEW_TB" -gt 0 ];                     then ALERT_LINE="$ALERT_LINE  tracebacks=$NEW_TB"; fi
if [ "$NEW_5XX" -gt 0 ];                    then ALERT_LINE="$ALERT_LINE  5xx=$NEW_5XX"; fi
if [ "$NEW_FUTEX" -gt 0 ];                  then ALERT_LINE="$ALERT_LINE  futex_signals=$NEW_FUTEX"; fi
if [ "$HEALTH" != "online" ];               then ALERT_LINE="$ALERT_LINE  health=$HEALTH"; fi

if [ -n "$ALERT_LINE" ]; then
  echo "[$TS] PROD_ALERT$ALERT_LINE" >> "$ALERT"
fi
