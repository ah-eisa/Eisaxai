#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path

import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.price_alerts import check_alerts, get_active_alerts, mark_triggered, send_telegram_alert

ENV_PATH = PROJECT_ROOT / ".env"
LOG_PATH = Path("/var/log/eisax-alerts.log")


def _configure_logging() -> logging.Logger:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handlers = [logging.StreamHandler()]

    try:
        file_handler = logging.FileHandler(LOG_PATH)
        file_handler.setFormatter(formatter)
        handlers.insert(0, file_handler)
    except Exception as exc:
        logging.basicConfig(level=logging.INFO, handlers=handlers, format="%(asctime)s [%(levelname)s] %(message)s")
        logger = logging.getLogger("eisax.alert_monitor")
        logger.warning("Unable to open %s: %s", LOG_PATH, exc)
        return logger

    logging.basicConfig(level=logging.INFO, handlers=handlers, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("eisax.alert_monitor")


logger = _configure_logging()


def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _fetch_prices(tickers: list[str]) -> dict[str, float]:
    prices: dict[str, float] = {}
    bundle = yf.Tickers(" ".join(tickers))

    for ticker in tickers:
        ticker_obj = bundle.tickers.get(ticker, yf.Ticker(ticker))
        price = None

        try:
            fast_info = ticker_obj.fast_info
            for key in (
                "lastPrice",
                "last_price",
                "regularMarketPrice",
                "regular_market_price",
                "previousClose",
                "previous_close",
            ):
                try:
                    value = fast_info.get(key)
                except Exception:
                    value = None
                if value is not None:
                    price = float(value)
                    break
        except Exception as exc:
            logger.warning("fast_info lookup failed for %s: %s", ticker, exc)

        if price is None:
            try:
                history = ticker_obj.history(period="1d", interval="1m", auto_adjust=False)
                if not history.empty:
                    close_series = history["Close"].dropna()
                    if not close_series.empty:
                        price = float(close_series.iloc[-1])
            except Exception as exc:
                logger.warning("history lookup failed for %s: %s", ticker, exc)

        if price is not None:
            prices[ticker] = price
        else:
            logger.warning("No price found for %s", ticker)

    return prices


def main() -> int:
    _load_env(ENV_PATH)
    alerts = get_active_alerts()
    if not alerts:
        logger.info("No active alerts to check")
        return 0

    tickers = sorted({str(alert["ticker"]).upper() for alert in alerts if alert.get("ticker")})
    logger.info("Checking %d active alerts across %d tickers", len(alerts), len(tickers))

    prices = _fetch_prices(tickers)
    logger.info("Fetched prices for %d/%d tickers", len(prices), len(tickers))

    triggered = check_alerts(prices)
    if not triggered:
        logger.info("No alerts triggered")
        return 0

    for alert in triggered:
        current_price = float(alert["current_price"])
        send_telegram_alert(alert, current_price)
        mark_triggered(int(alert["id"]))
        logger.info(
            "Triggered alert %s for %s (%s %.4f, current %.4f)",
            alert["id"],
            alert["ticker"],
            alert["condition"],
            float(alert["threshold"]),
            current_price,
        )

    logger.info("Processed %d triggered alerts", len(triggered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
