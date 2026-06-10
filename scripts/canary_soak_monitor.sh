#!/bin/bash
#
# Phase D pre-grounding CANARY soak monitor (production, port 8000).
# Read-only. Tracks pre-grounding-specific signals + standard health.
# Run every 15 min via cron during the canary soak window.
# One JSON line per tick to LOG; alert line on critical signals.
#
# Tracks (deltas since last tick, via journal cursor by timestamp):
#   - service active + /health
#   - master + worker PIDs and RSS (leak watch)
#   - scheduler started/skipped (must be 1/1)
#   - NEW [PreGround] injections   (canary path exercised)
#   - NEW [LLMUsage] lines + whether any pregrounding=1 seen
#   - NEW [SSOT] reconciler correction totals
#   - NEW Traceback / 5xx / futex
#
# Output: logs/canary_soak.jsonl   Alerts: logs/canary_soak_alerts.log

set -u
LOG=/home/ubuntu/investwise/logs/canary_soak.jsonl
ALERT=/home/ubuntu/investwise/logs/canary_soak_alerts.log
STATE=/home/ubuntu/investwise/logs/canary_soak_state.json
SVC=eisax-gunicorn
LOCK=/tmp/eisax-production-scheduler.lock
ERR=/home/ubuntu/investwise/logs/gunicorn_test_error.log

# journal window: since last tick ts (default: 20 min ago on first run)
if [ -f "$STATE" ]; then
  PREV_TS=$(jq -r '.ts // empty' "$STATE" 2>/dev/null)
fi
[ -z "${PREV_TS:-}" ] && PREV_TS="$(date -d '20 min ago' '+%Y-%m-%d %H:%M:%S')"

ACTIVE=$(systemctl is-active "$SVC" 2>/dev/null || echo unknown)
MAIN_PID=$(systemctl show "$SVC" -p MainPID --value 2>/dev/null)
WORKER_PIDS=$(pgrep -P "$MAIN_PID" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
WORKER_COUNT=$(echo "$WORKER_PIDS" | awk -F',' '{print (length($0)? NF:0)}')
WORKER_RSS_KB=$(if [ -n "$WORKER_PIDS" ]; then for p in $(echo "$WORKER_PIDS"|tr ',' ' '); do ps -o rss= -p "$p" 2>/dev/null|tr -d ' '; done|paste -sd, -; fi)

LOCK_PID=$(cat "$LOCK" 2>/dev/null | tr -d '\n' | head -c 16)
SVC_SINCE=$(systemctl show "$SVC" -p ActiveEnterTimestamp --value 2>/dev/null)
SCHED_STARTED=$(journalctl -u "$SVC" --since "$SVC_SINCE" 2>/dev/null | grep -c "Scheduler started")
SCHED_SKIPPED=$(journalctl -u "$SVC" --since "$SVC_SINCE" 2>/dev/null | grep -c "scheduler skipped (worker is not owner)")

# Deltas since previous tick
J=$(journalctl -u "$SVC" --since "$PREV_TS" 2>/dev/null)
NEW_PREGROUND=$(echo "$J" | grep -c "\[PreGround\].*injected")
NEW_LLMUSAGE=$(echo "$J" | grep -c "\[LLMUsage\]")
NEW_PG1=$(echo "$J" | grep "\[LLMUsage\]" | grep -c "pregrounding=1")
NEW_CORR=$(echo "$J" | grep -oE "\[SSOT\].* reconciler corrections=[0-9]+" | grep -oE "corrections=[0-9]+" | grep -oE "[0-9]+" | awk '{s+=$1} END{print s+0}')
NEW_TB=$(echo "$J" | grep -c "Traceback")
NEW_FUTEX=$(echo "$J" | grep -ciE "futex_do_wait|deadlock|SIGSEGV|SIGKILL")
NEW_PG_ERR=$(echo "$J" | grep -c "\[PreGround\].*skipped —")

TOKEN=$(grep -E "^SECURE_TOKEN=" /home/ubuntu/investwise/.env | cut -d= -f2- | tr -d '"' | tr -d "'")
HEALTH=$(curl -s --max-time 5 -H "X-API-Key: $TOKEN" http://127.0.0.1:8000/health 2>/dev/null \
         | python3 -c "import sys,json;print(json.load(sys.stdin).get('status','?'))" 2>/dev/null || echo unreachable)

# 5xx in access log since prev (rough: count recent 5xx lines)
NEW_5XX=$(tail -n 400 /home/ubuntu/investwise/logs/gunicorn_test_access.log 2>/dev/null | grep -cE 'HTTP/1.1" 5[0-9][0-9]')

TS=$(date '+%Y-%m-%d %H:%M:%S')
TSI=$(date -Is)
ROW=$(printf '{"ts":"%s","active":"%s","health":"%s","master_pid":%s,"worker_pids":"%s","worker_count":%s,"worker_rss_kb":"%s","sched_lock_pid":"%s","sched_started_n":%s,"sched_skipped_n":%s,"new_preground":%s,"new_llmusage":%s,"new_pg1":%s,"new_corrections":%s,"new_pg_err":%s,"new_tb":%s,"new_5xx_recent":%s,"new_futex":%s}\n' \
  "$TSI" "$ACTIVE" "$HEALTH" "${MAIN_PID:-0}" "${WORKER_PIDS:-}" "${WORKER_COUNT:-0}" "${WORKER_RSS_KB:-}" \
  "${LOCK_PID:-none}" "$SCHED_STARTED" "$SCHED_SKIPPED" "$NEW_PREGROUND" "$NEW_LLMUSAGE" "$NEW_PG1" \
  "$NEW_CORR" "$NEW_PG_ERR" "$NEW_TB" "$NEW_5XX" "$NEW_FUTEX")
echo "$ROW" >> "$LOG"
echo "{\"ts\": \"$TS\"}" > "$STATE"

AL=""
[ "$ACTIVE" != "active" ] && AL="$AL active=$ACTIVE"
[ "${WORKER_COUNT:-0}" -lt 2 ] && AL="$AL workers=${WORKER_COUNT:-0}<2"
[ "$SCHED_STARTED" -gt 1 ] && AL="$AL sched_started=$SCHED_STARTED>1"
[ "$NEW_TB" -gt 0 ] && AL="$AL tracebacks=$NEW_TB"
[ "$NEW_FUTEX" -gt 0 ] && AL="$AL futex=$NEW_FUTEX"
[ "$NEW_PG_ERR" -gt 0 ] && AL="$AL preground_errors=$NEW_PG_ERR"
[ "$HEALTH" != "online" ] && AL="$AL health=$HEALTH"
[ -n "$AL" ] && echo "[$TSI] CANARY_ALERT$AL" >> "$ALERT"
