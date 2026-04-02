#!/usr/bin/env python3
"""
EisaX Health Monitor - polls /v1/health every 5 min, alerts via Telegram.
"""
import os
import time
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("/home/ubuntu/investwise/.env")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HEALTH_URL = "http://localhost:8000/v1/health"
TOKEN = os.getenv("SECURE_TOKEN")
INTERVAL = 300  # 5 minutes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

last_status = None
degraded_count = 0
last_alert_time = {}   # {status: timestamp}
ALERT_COOLDOWN = 1800  # 30 minutes in seconds
DEGRADED_THRESHOLD = 5 # consecutive checks before alerting


def send_telegram(msg: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        logging.error(f"Telegram send failed: {e}")


def check_health() -> str:
    try:
        r = requests.get(HEALTH_URL, headers={"access-token": TOKEN}, timeout=30)
        data = r.json()
        services = data.get("services", {})

        # Database down = always down
        if services.get("database", {}).get("status") == "down":
            return "down"

        # DeepSeek or Kimi down = degraded
        deepseek_ok = services.get("deepseek", {}).get("status") == "ok"
        kimi_ok = services.get("kimi", {}).get("status") == "ok"

        if not deepseek_ok and not kimi_ok:
            return "down"       # all primary LLMs down
        if not deepseek_ok or not kimi_ok:
            return "degraded"   # one of them is down

        # Gemini down alone = ignore (free tier quota, non-critical)
        return "ok"
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return "down"


def should_alert(status: str) -> bool:
    now = time.time()
    last = last_alert_time.get(status, 0)
    return (now - last) > ALERT_COOLDOWN


def main():
    global last_status, degraded_count

    logging.info("EisaX Health Monitor started - silent mode (no Telegram on startup)")

    while True:
        status = check_health()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        if status == "down":
            degraded_count = 0
            if should_alert("down"):
                send_telegram(
                    f"🔴 <b>EisaX DOWN</b>\n"
                    f"Time: {now_str}\n"
                    f"All services unreachable. Immediate action required."
                )
                last_alert_time["down"] = time.time()

        elif status == "degraded":
            degraded_count += 1
            if degraded_count >= DEGRADED_THRESHOLD and should_alert("degraded"):
                send_telegram(
                    f"🟡 <b>EisaX DEGRADED</b>\n"
                    f"Time: {now_str}\n"
                    f"Sustained degradation ({DEGRADED_THRESHOLD} checks). System still operational."
                )
                last_alert_time["degraded"] = time.time()

        elif status == "ok":
            degraded_count = 0
            if last_status in ("down",) and should_alert("recovered"):
                send_telegram(
                    f"✅ <b>EisaX RECOVERED</b>\n"
                    f"Time: {now_str}\n"
                    f"All systems operational."
                )
                last_alert_time["recovered"] = time.time()

        last_status = status
        logging.info(f"Health: {status} | degraded_count: {degraded_count}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
