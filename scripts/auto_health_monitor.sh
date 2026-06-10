#!/usr/bin/env bash
# EisaX automated health monitor — installed 2026-06-10 (MON-1 phase).
# tick  mode: every 15 min via cron — counts error signals in the last 15 min,
#             appends one line to logs/auto_health.log (ALERT-prefixed on breach).
# daily mode: 23:55 via cron — aggregates today's ticks into auto_health_daily.log.
set -u
LOGDIR=/home/ubuntu/investwise/logs
LOG=$LOGDIR/auto_health.log
DAILY=$LOGDIR/auto_health_daily.log
mkdir -p "$LOGDIR"
MODE=${1:-tick}
TS=$(date "+%Y-%m-%d %H:%M:%S")

if [ "$MODE" = "tick" ]; then
    PROD=$(systemctl is-active eisax-gunicorn 2>/dev/null)
    STAG=$(systemctl is-active eisax-gunicorn-staging 2>/dev/null)
    J() { sudo journalctl -u "$1" --since "15 min ago" 2>/dev/null; }
    PJ=$(J eisax-gunicorn); SJ=$(J eisax-gunicorn-staging)
    TB=$(printf '%s\n%s' "$PJ" "$SJ" | grep -ci "traceback" || true)
    E5=$(printf '%s\n%s' "$PJ" "$SJ" | grep -cE " 50[0-9] " || true)
    R429=$(printf '%s' "$PJ" | grep -c "429 Too Many Requests" || true)
    BRK=$(printf '%s' "$PJ" | grep -c "circuit breaker tripped" || true)
    FUTEX=$(printf '%s' "$PJ" | grep -ci "futex" || true)
    DISK=$(df / --output=pcent | tail -1 | tr -dc '0-9')

    ALERTS=""
    [ "$PROD" != "active" ] && ALERTS="$ALERTS prod=$PROD"
    [ "$STAG" != "active" ] && ALERTS="$ALERTS staging=$STAG"
    [ "${TB:-0}" -gt 0 ] && ALERTS="$ALERTS tb=$TB"
    [ "${E5:-0}" -gt 0 ] && ALERTS="$ALERTS 5xx=$E5"
    [ "${R429:-0}" -gt 10 ] && ALERTS="$ALERTS 429=$R429"
    [ "${FUTEX:-0}" -gt 0 ] && ALERTS="$ALERTS futex=$FUTEX"
    [ "${DISK:-0}" -gt 85 ] && ALERTS="$ALERTS disk=${DISK}%"

    STATUS=OK
    [ -n "$ALERTS" ] && STATUS="ALERT:$ALERTS"
    echo "$TS | prod=$PROD staging=$STAG tb=$TB 5xx=$E5 429=$R429 breaker=$BRK futex=$FUTEX disk=${DISK}% | $STATUS" >> "$LOG"

elif [ "$MODE" = "daily" ]; then
    TODAY=$(date +%Y-%m-%d)
    LINES=$(grep -c "^$TODAY" "$LOG" 2>/dev/null); LINES=${LINES:-0}
    ALERTS=$(grep "^$TODAY" "$LOG" 2>/dev/null | grep -c "ALERT"); ALERTS=${ALERTS:-0}
    {
        echo "═══ $TODAY — ticks=$LINES alerts=$ALERTS ═══"
        if [ "$ALERTS" -gt 0 ]; then
            grep "^$TODAY" "$LOG" | grep "ALERT" | tail -20
        else
            echo "all clear"
        fi
    } >> "$DAILY"
    # Rotate tick log if over 5 MB (keep one previous generation)
    if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG")" -gt 5242880 ]; then
        mv "$LOG" "$LOG.1"
    fi
fi
