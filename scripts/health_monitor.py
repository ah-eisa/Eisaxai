#!/usr/bin/env python3
"""
EisaX Health Monitor v3
- polls /v1/health every 60s when degraded/down, 5 min when ok
- AUTO-RESTART gunicorn on any DOWN (not just zombie loop)
- AUTO-RESTART nginx if it goes down
- alerts via Telegram with ROOT CAUSE details
- sends daily summary at 8 AM UTC
"""
import os
import time
import subprocess
import requests
import logging
from collections import deque
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv("/home/ubuntu/investwise/.env")

BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID")
HEALTH_URL  = "http://localhost:8000/v1/health"
TOKEN       = os.getenv("SECURE_TOKEN")

INTERVAL_OK   = 300   # 5 minutes when all good
INTERVAL_BAD  = 60    # 1 minute when down/degraded

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

last_status        = None
degraded_count     = 0
down_count         = 0
last_alert_time    = {}
last_daily_report  = None
last_heal_time     = 0   # prevent heal spam

ALERT_COOLDOWN             = 1800  # 30 min between same-status alerts
DEGRADED_THRESHOLD         = 3     # 3 checks = 3 min
PORT_ZOMBIE_PROC_THRESHOLD = 7
PORT_ZOMBIE_RESTART_THRESHOLD = 50
RESTART_LOOKBACK_SECONDS   = 3600
HEAL_COOLDOWN              = 120   # min 2 min between auto-heals

gunicorn_restart_samples = deque()


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────────────────────────
def send_telegram(msg: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        logging.error(f"Telegram send failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
def get_diagnostics() -> dict:
    info = {}

    try:
        result = subprocess.run(
            ["lsof", "-ti:8000"], capture_output=True, text=True, timeout=5
        )
        pids = [p for p in result.stdout.strip().split("\n") if p]
        info["port_8000_procs"] = len(pids)
        info["port_8000_pids"]  = pids[:5]
    except Exception:
        info["port_8000_procs"] = "?"

    restarts = get_gunicorn_restart_total()
    info["gunicorn_restarts"] = restarts if restarts is not None else "?"
    if restarts is not None:
        info["gunicorn_restarts_last_hour"] = get_gunicorn_restart_count_last_hour(restarts)
    else:
        info["gunicorn_restarts_last_hour"] = "?"

    try:
        result = subprocess.run(
            ["systemctl", "is-active", "eisax-gunicorn"],
            capture_output=True, text=True, timeout=5,
        )
        info["gunicorn_state"] = result.stdout.strip()
    except Exception:
        info["gunicorn_state"] = "?"

    try:
        result = subprocess.run(
            ["systemctl", "is-active", "nginx"],
            capture_output=True, text=True, timeout=5,
        )
        info["nginx_state"] = result.stdout.strip()
    except Exception:
        info["nginx_state"] = "?"

    try:
        with open("/proc/meminfo") as f:
            lines = {l.split(":")[0]: l.split(":")[1].strip() for l in f.readlines()}
        total  = int(lines.get("MemTotal",    "0 kB").split()[0]) // 1024
        avail  = int(lines.get("MemAvailable","0 kB").split()[0]) // 1024
        used   = total - avail
        pct    = round(used / total * 100) if total else 0
        info["memory"] = f"{used}MB / {total}MB ({pct}%)"
    except Exception:
        info["memory"] = "?"

    try:
        result = subprocess.run(
            ["df", "-h", "/"], capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            parts = lines[1].split()
            info["disk"] = f"{parts[2]} used / {parts[1]} total ({parts[4]})"
    except Exception:
        info["disk"] = "?"

    return info


def get_gunicorn_restart_total() -> int | None:
    try:
        result = subprocess.run(
            ["systemctl", "show", "eisax-gunicorn", "--property=NRestarts"],
            capture_output=True, text=True, timeout=5,
        )
        return int(result.stdout.strip().replace("NRestarts=", "") or "0")
    except Exception:
        return None


def get_gunicorn_restart_count_last_hour(total_restarts: int, now: float | None = None) -> int:
    now = time.time() if now is None else now

    if gunicorn_restart_samples and total_restarts < gunicorn_restart_samples[-1][1]:
        gunicorn_restart_samples.clear()

    if not gunicorn_restart_samples or gunicorn_restart_samples[-1][1] != total_restarts:
        gunicorn_restart_samples.append((now, total_restarts))

    cutoff = now - RESTART_LOOKBACK_SECONDS
    while len(gunicorn_restart_samples) > 1 and gunicorn_restart_samples[1][0] <= cutoff:
        gunicorn_restart_samples.popleft()

    baseline_total = gunicorn_restart_samples[0][1] if gunicorn_restart_samples else total_restarts
    return max(total_restarts - baseline_total, 0)


def is_zombie_loop(port_8000_procs: int, restart_count_last_hour: int) -> bool:
    return (
        port_8000_procs > PORT_ZOMBIE_PROC_THRESHOLD
        or restart_count_last_hour > PORT_ZOMBIE_RESTART_THRESHOLD
    )


def can_heal() -> bool:
    """Rate-limit: don't heal more than once per HEAL_COOLDOWN seconds."""
    return (time.time() - last_heal_time) > HEAL_COOLDOWN


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-HEAL: zombie port loop (force-kill + restart)
# ─────────────────────────────────────────────────────────────────────────────
def auto_heal_port_zombie() -> bool:
    global last_heal_time
    try:
        result = subprocess.run(
            ["lsof", "-ti:8000"], capture_output=True, text=True, timeout=5
        )
        pids = [p for p in result.stdout.strip().split("\n") if p]
        restarts = get_gunicorn_restart_total()
        restart_count_last_hour = (
            get_gunicorn_restart_count_last_hour(restarts) if restarts is not None else 0
        )

        if is_zombie_loop(len(pids), restart_count_last_hour):
            logging.warning(
                "[AUTO-HEAL] Zombie loop: %s procs, %s restarts/hr. Healing...",
                len(pids), restart_count_last_hour,
            )
            subprocess.run(["sudo", "fuser", "-k", "8000/tcp"], timeout=10)
            time.sleep(3)
            subprocess.run(["sudo", "systemctl", "restart", "eisax-gunicorn"], timeout=30)
            last_heal_time = time.time()
            return True
    except Exception as e:
        logging.error(f"[AUTO-HEAL zombie] Failed: {e}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-HEAL: simple DOWN — restart gunicorn directly
# ─────────────────────────────────────────────────────────────────────────────
def auto_heal_gunicorn() -> bool:
    """Restart eisax-gunicorn when it's simply down (no zombie needed)."""
    global last_heal_time
    if not can_heal():
        logging.info("[AUTO-HEAL] Cooldown active, skipping restart.")
        return False
    try:
        logging.warning("[AUTO-HEAL] Gunicorn DOWN — restarting eisax-gunicorn...")
        subprocess.run(
            ["sudo", "systemctl", "restart", "eisax-gunicorn"],
            timeout=40, check=True
        )
        last_heal_time = time.time()
        logging.info("[AUTO-HEAL] eisax-gunicorn restart command sent.")
        return True
    except Exception as e:
        logging.error(f"[AUTO-HEAL gunicorn] Failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-HEAL: nginx down
# ─────────────────────────────────────────────────────────────────────────────
def check_and_heal_nginx():
    """Restart nginx if it's not active."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "nginx"],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip() != "active":
            logging.warning("[AUTO-HEAL] nginx is %s — restarting...", result.stdout.strip())
            subprocess.run(["sudo", "systemctl", "restart", "nginx"], timeout=20)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            send_telegram(
                f"🔧 <b>nginx AUTO-RESTARTED</b>\n"
                f"Time: {now_str}\n"
                f"nginx was down — restarted automatically."
            )
    except Exception as e:
        logging.error(f"[AUTO-HEAL nginx] Failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_health() -> tuple[str, dict]:
    details = {}
    try:
        r = requests.get(HEALTH_URL, headers={"access-token": TOKEN}, timeout=30)
        data     = r.json()
        services = data.get("services", {})
        details["services"] = services

        db_ok       = services.get("database", {}).get("status") == "ok"
        deepseek_ok = services.get("deepseek", {}).get("status") == "ok"

        failed = []
        if not db_ok:       failed.append("Database ❌")
        if not deepseek_ok: failed.append("DeepSeek ❌")

        details["failed"] = failed
        details["optional_failed"] = []
        details["latency_ms"] = round(r.elapsed.total_seconds() * 1000)

        if not db_ok:
            return "down", details
        if not deepseek_ok:
            return "degraded", details

        return "ok", details

    except requests.exceptions.ConnectionError:
        details["error"] = "Connection refused — gunicorn not responding"
        return "down", details
    except requests.exceptions.Timeout:
        details["error"] = "Health endpoint timed out (>30s)"
        return "down", details
    except Exception as e:
        details["error"] = str(e)
        return "down", details


def should_alert(status: str) -> bool:
    return (time.time() - last_alert_time.get(status, 0)) > ALERT_COOLDOWN


def format_diag(diag: dict) -> str:
    lines = []
    procs = diag.get("port_8000_procs", "?")
    restarts = diag.get("gunicorn_restarts", "?")
    restarts_last_hour = diag.get("gunicorn_restarts_last_hour", "?")
    zombie_warning = ""
    if isinstance(procs, int) and isinstance(restarts_last_hour, int):
        zombie_warning = "  ⚠️ zombie loop!" if is_zombie_loop(procs, restarts_last_hour) else ""
    gunicorn_state = diag.get("gunicorn_state", "?")
    nginx_state    = diag.get("nginx_state", "?")
    lines.append(f"⚙️ Port 8000 procs: <b>{procs}</b>{zombie_warning}")
    lines.append(f"🔁 Gunicorn restarts: <b>{restarts}</b> total / <b>{restarts_last_hour}</b> in last hour")
    lines.append(f"🟢 gunicorn service: <b>{gunicorn_state}</b>  |  nginx: <b>{nginx_state}</b>")
    lines.append(f"💾 Memory: {diag.get('memory', '?')}")
    lines.append(f"💿 Disk: {diag.get('disk', '?')}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# DAILY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def maybe_send_daily_summary():
    global last_daily_report
    now_utc = datetime.now(timezone.utc)
    today   = now_utc.date()
    if last_daily_report == today:
        return
    if now_utc.hour == 8 and now_utc.minute < 10:
        diag = get_diagnostics()
        send_telegram(
            f"📊 <b>EisaX Daily Summary</b>\n"
            f"Date: {today}\n\n"
            f"{format_diag(diag)}\n\n"
            f"Monitor v3: running ✅"
        )
        last_daily_report = today


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global last_status, degraded_count, down_count

    logging.info("EisaX Health Monitor v3 started — waiting 30s for services to stabilize...")
    time.sleep(30)   # grace period: let gunicorn finish starting before first check
    logging.info("EisaX Health Monitor v3 — first check starting now")

    while True:
        status, details = check_health()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        # ── Always check nginx ────────────────────────────────────────────────
        check_and_heal_nginx()

        # ── DOWN handling ────────────────────────────────────────────────────
        if status == "down":
            degraded_count = 0
            down_count    += 1

            # Step 1: try zombie heal first
            healed = auto_heal_port_zombie()

            # Step 2: if not a zombie, just restart gunicorn directly
            if not healed:
                healed = auto_heal_gunicorn()

            if healed:
                time.sleep(15)   # give service time to come up
                # Quick re-check
                recheck_status, recheck_details = check_health()
                if recheck_status == "ok":
                    send_telegram(
                        f"✅ <b>EisaX AUTO-RECOVERED</b>\n"
                        f"Time: {now_str}\n"
                        f"Service was DOWN — auto-restarted and is now <b>OK</b>."
                    )
                    last_status    = "ok"
                    down_count     = 0
                    last_alert_time["down"]      = 0  # reset cooldown
                    last_alert_time["recovered"] = time.time()
                    interval = INTERVAL_OK
                    maybe_send_daily_summary()
                    time.sleep(interval)
                    continue
                else:
                    # Still down after restart — alert
                    if should_alert("down"):
                        diag   = get_diagnostics()
                        failed = ", ".join(details.get("failed", [])) or details.get("error", "Unknown")
                        send_telegram(
                            f"🔴 <b>EisaX DOWN</b> (auto-restart attempted, still failing)\n"
                            f"Time: {now_str}\n\n"
                            f"❌ Failed: <b>{failed}</b>\n"
                            f"📡 Latency: {details.get('latency_ms', '—')}ms\n\n"
                            f"{format_diag(diag)}"
                        )
                        last_alert_time["down"] = time.time()
            else:
                # Heal was rate-limited — just alert
                if should_alert("down"):
                    diag   = get_diagnostics()
                    failed = ", ".join(details.get("failed", [])) or details.get("error", "Unknown")
                    send_telegram(
                        f"🔴 <b>EisaX DOWN</b>\n"
                        f"Time: {now_str}\n\n"
                        f"❌ Failed: <b>{failed}</b>\n"
                        f"📡 Latency: {details.get('latency_ms', '—')}ms\n\n"
                        f"{format_diag(diag)}"
                    )
                    last_alert_time["down"] = time.time()

        # ── DEGRADED ──────────────────────────────────────────────────────────
        elif status == "degraded":
            down_count     = 0
            degraded_count += 1
            if degraded_count >= DEGRADED_THRESHOLD and should_alert("degraded"):
                diag   = get_diagnostics()
                failed = ", ".join(details.get("failed", [])) or "Unknown"
                send_telegram(
                    f"🟡 <b>EisaX DEGRADED</b>\n"
                    f"Time: {now_str} ({degraded_count} consecutive checks)\n\n"
                    f"⚠️ Degraded: <b>{failed}</b>\n"
                    f"📡 Latency: {details.get('latency_ms', '—')}ms\n\n"
                    f"{format_diag(diag)}"
                )
                last_alert_time["degraded"] = time.time()

        # ── OK ────────────────────────────────────────────────────────────────
        elif status == "ok":
            prev_bad = last_status in ("down", "degraded") and (down_count > 0 or degraded_count >= DEGRADED_THRESHOLD)
            degraded_count = 0
            down_count     = 0
            if prev_bad and should_alert("recovered"):
                send_telegram(
                    f"✅ <b>EisaX RECOVERED</b>\n"
                    f"Time: {now_str}\n"
                    f"All systems operational. Latency: {details.get('latency_ms', '—')}ms"
                )
                last_alert_time["recovered"] = time.time()

        last_status = status
        logging.info(
            "Health: %s | down: %s | degraded: %s | error: %s",
            status, down_count, degraded_count,
            details.get("failed") or details.get("error", ""),
        )

        maybe_send_daily_summary()

        # Dynamic interval: faster when something is wrong
        interval = INTERVAL_BAD if status in ("down", "degraded") else INTERVAL_OK
        time.sleep(interval)


if __name__ == "__main__":
    main()
