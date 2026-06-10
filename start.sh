#!/bin/bash
# EisaX Start Script — uses systemd (not screen) for reliable process management

echo "----------------------------------------------------------------"
echo "Starting EisaX Agent..."
echo "Timestamp: $(date)"
echo "----------------------------------------------------------------"

# ── Use systemd if available (preferred — handles auto-restart, no duplicates) ──
if systemctl is-active --quiet eisax.service 2>/dev/null || systemctl list-unit-files eisax.service &>/dev/null; then
    echo "Using systemd to restart EisaX..."
    sudo systemctl restart eisax.service
    sleep 3
    if systemctl is-active --quiet eisax.service; then
        echo "✅ EisaX Agent is running (systemd managed)."
        echo ""
        echo "Useful commands:"
        echo "  Status:  sudo systemctl status eisax.service"
        echo "  Logs:    sudo journalctl -u eisax.service -f"
        echo "  Stop:    sudo systemctl stop eisax.service"
    else
        echo "❌ systemd restart failed. Checking logs..."
        sudo journalctl -u eisax.service --no-pager -n 20
        exit 1
    fi
else
    # ── Fallback: screen session (if systemd unavailable) ──
    APP_DIR=~/investwise
    API_SCRIPT="api_bridge_v2.py"
    PYTHON="$APP_DIR/venv/bin/python"
    PORT=8000

    echo "systemd not available — using screen fallback..."

    # Kill ALL old processes cleanly
    echo "Stopping existing server..."
    pkill -f "$API_SCRIPT" 2>/dev/null
    sleep 1

    # Force kill anything still on port 8000
    PIDS=$(ss -tlnp sport = :$PORT 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u)
    [ -n "$PIDS" ] && echo "Force-killing PIDs on port $PORT: $PIDS" && echo "$PIDS" | xargs kill -9 2>/dev/null

    # Kill old screen sessions
    screen -ls 2>/dev/null | grep "eisax-api" | awk -F. '{print $1}' | tr -d ' \t' | xargs -I{} screen -S {} -X quit 2>/dev/null
    sleep 2

    echo "Starting API Bridge ($API_SCRIPT)..."
    screen -dmS eisax-api bash -c "cd $APP_DIR && $PYTHON $API_SCRIPT; exec bash"
    sleep 3

    if pgrep -f "$API_SCRIPT" > /dev/null; then
        echo "✅ EisaX Agent is running (screen session)."
        echo "View logs: tail -f $APP_DIR/backend.log"
    else
        echo "❌ Failed to start. Check backend.log."
        exit 1
    fi
fi

echo "----------------------------------------------------------------"
