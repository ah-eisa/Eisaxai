#!/usr/bin/env python3
"""
EisaX Market Updates — Scheduled Auto-Generator
────────────────────────────────────────────────
Triggered daily by systemd timer (eisax-updates-daily.timer).

Behaviour:
  - Every day at 07:30 UTC: generate daily update
  - Every Friday at 07:30 UTC: also generate weekly update
  - Logs to /home/ubuntu/investwise/updates_generator.log
"""
import os
import sys
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
ROOT = "/home/ubuntu/investwise"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))


def run_post_generation_log(daily, weekly=None) -> None:
    """Log a clean summary after generation."""
    ev     = daily.get("eisax_view", {})
    stance = ev.get("stance") if isinstance(ev, dict) else ev
    logger.info(
        "Post-generation summary | date=%s | regime=%s | confidence=%s | stance=%s | f&g=%s | weekly=%s",
        daily.get("date"),
        daily.get("market_regime"),
        daily.get("regime_confidence"),
        stance,
        daily.get("fear_greed_index"),
        "yes" if weekly is not None else "no",
    )


def main() -> None:
    from core.services.market_updates import (
        generate_daily_update,
        generate_weekly_update,
    )

    now = datetime.now(timezone.utc)
    logger.info("=== EisaX Market Updates Generator — %s UTC ===", now.strftime("%Y-%m-%d %H:%M"))

    # ── Daily ──────────────────────────────────────────────────────────────────
    daily = None
    try:
        daily = generate_daily_update()
        ev    = daily.get("eisax_view", {})
        stance = ev.get("stance") if isinstance(ev, dict) else ev
        logger.info(
            "Daily saved ✓ | regime=%s | confidence=%s | stance=%s | f&g=%s",
            daily.get("market_regime"),
            daily.get("regime_confidence"),
            stance,
            daily.get("fear_greed_index"),
        )
    except Exception as exc:
        logger.error("Daily generation FAILED: %s", exc, exc_info=True)

    # ── Weekly (Fridays only) ──────────────────────────────────────────────────
    weekly = None
    if now.weekday() == 4:  # 4 = Friday
        logger.info("Friday — generating weekly strategy brief...")
        try:
            weekly = generate_weekly_update()
            logger.info(
                "Weekly saved ✓ | verdict=%s",
                weekly.get("eisax_verdict", "")[:60],
            )
        except Exception as exc:
            logger.error("Weekly generation FAILED: %s", exc, exc_info=True)
    else:
        logger.info("Not Friday — weekly generation skipped.")

    if daily is not None:
        run_post_generation_log(daily, weekly)
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
