"""
data_scheduler.py — EisaX Daily Data Scheduler
يضبط الـ cron job تلقائياً لتحديث الداتا بعد إغلاق كل سوق

أوقات الإغلاق (UTC+3):
  - السعودي (Tadawul): يغلق 3:00 PM → نحدث 3:30 PM
  - المصري (EGX):      يغلق 2:30 PM → نحدث 3:00 PM
  - الإماراتي (ADX):   يغلق 3:00 PM → نحدث 3:30 PM
  
نشغل update واحد الساعة 6:00 PM يوميًا يغطي الثلاثة
"""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = "/home/ubuntu/investwise"
PYTHON_BIN = "/home/ubuntu/investwise/venv/bin/python"
FETCHER_SCRIPT = f"{PROJECT_DIR}/core/data_fetcher.py"
LOG_FILE = "/home/ubuntu/investwise/logs/data_fetcher.log"


# ─── Setup Cron ──────────────────────────────────────────────────────────────

CRON_JOB = f"0 15 * * 0-4 {PYTHON_BIN} {FETCHER_SCRIPT} daily >> {LOG_FILE} 2>&1"
# 15:00 UTC = 18:00 UAE/SA (UTC+3), من الأحد للخميس (0-4)


def install_cron():
    """يضيف الـ cron job تلقائياً"""
    # اقرأ الـ crontab الحالي
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    current = result.stdout if result.returncode == 0 else ""

    if CRON_JOB in current:
        print("✅ Cron job already installed.")
        return

    # أضف الجديد
    new_crontab = current.rstrip() + f"\n{CRON_JOB}\n"
    proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True, capture_output=True)

    if proc.returncode == 0:
        print(f"✅ Cron job installed:\n   {CRON_JOB}")
    else:
        print(f"❌ Failed to install cron: {proc.stderr}")


def remove_cron():
    """يحذف الـ cron job"""
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0:
        print("No crontab found.")
        return

    lines = [l for l in result.stdout.splitlines() if CRON_JOB not in l]
    new_crontab = "\n".join(lines) + "\n"
    subprocess.run(["crontab", "-"], input=new_crontab, text=True)
    print("✅ Cron job removed.")


def show_cron():
    """يعرض الـ cron jobs الحالية"""
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode == 0:
        print("Current crontab:")
        print(result.stdout)
    else:
        print("No crontab installed.")


# ─── Systemd Timer (بديل أقوى من cron) ──────────────────────────────────────

SYSTEMD_SERVICE = """[Unit]
Description=EisaX Daily Market Data Update
After=network.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/investwise
ExecStart={python} {script} daily
StandardOutput=append:/home/ubuntu/investwise/logs/data_fetcher.log
StandardError=append:/home/ubuntu/investwise/logs/data_fetcher.log
Environment=PATH=/home/ubuntu/investwise/venv/bin:/usr/bin:/bin
""".format(python=PYTHON_BIN, script=FETCHER_SCRIPT)

SYSTEMD_TIMER = """[Unit]
Description=EisaX Daily Market Data Timer
Requires=eisax-data.service

[Timer]
# كل يوم الساعة 6:00 مساءً (توقيت السيرفر UTC+3)
OnCalendar=Sun-Thu 15:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
"""


def install_systemd():
    """يثبت systemd timer (أفضل من cron)"""
    service_path = Path("/etc/systemd/system/eisax-data.service")
    timer_path = Path("/etc/systemd/system/eisax-data.timer")

    # تأكد من وجود مجلد الـ logs
    Path("/home/ubuntu/investwise/logs").mkdir(parents=True, exist_ok=True)

    service_path.write_text(SYSTEMD_SERVICE)
    timer_path.write_text(SYSTEMD_TIMER)

    cmds = [
        ["systemctl", "daemon-reload"],
        ["systemctl", "enable", "eisax-data.timer"],
        ["systemctl", "start", "eisax-data.timer"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, check=True)

    print("✅ Systemd timer installed!")
    print("   Run: systemctl status eisax-data.timer")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "install"

    if cmd == "install":
        install_cron()
    elif cmd == "remove":
        remove_cron()
    elif cmd == "show":
        show_cron()
    elif cmd == "systemd":
        install_systemd()
    else:
        print("Usage: python data_scheduler.py [install|remove|show|systemd]")