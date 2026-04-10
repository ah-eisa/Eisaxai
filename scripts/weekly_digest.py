#!/usr/bin/env python3
"""
EisaX Weekly Portfolio Digest
Runs every Monday at 09:00 UTC via systemd timer.
Sends a Telegram summary of each active user's portfolio performance.
"""
import os, sys, json, logging, sqlite3, requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────
sys.path.insert(0, "/home/ubuntu/investwise")
load_dotenv("/home/ubuntu/investwise/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [weekly_digest] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/var/log/eisax-weekly-digest.log"),
    ],
)
logger = logging.getLogger("weekly_digest")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
APP_DB    = os.getenv("APP_DB", "/home/ubuntu/investwise/investwise.db")

# Override with config if available
try:
    from core.config import APP_DB as _cfg_db
    APP_DB = str(_cfg_db)
except Exception:
    pass


def _send_telegram(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram not configured — skipping notification")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)
        return False


def _get_live_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch current prices via yfinance for a list of tickers."""
    prices = {}
    if not tickers:
        return prices
    try:
        import yfinance as yf
        data = yf.download(tickers, period="1d", auto_adjust=True, progress=False)
        if hasattr(data, "columns") and "Close" in data:
            close = data["Close"].iloc[-1]
            for t in tickers:
                try:
                    prices[t] = float(close[t]) if t in close else 0.0
                except Exception:
                    prices[t] = 0.0
        else:
            for t in tickers:
                try:
                    prices[t] = float(yf.Ticker(t).fast_info.last_price or 0)
                except Exception:
                    prices[t] = 0.0
    except Exception as exc:
        logger.error("yfinance fetch failed: %s", exc)
    return prices


def _compute_pnl(holdings: dict, current_prices: dict, ref_value: float = 100_000) -> dict:
    """
    holdings: {ticker: weight (0-1)}
    current_prices: {ticker: price}
    Returns {pnl_pct, pnl_usd, by_ticker: [{ticker, weight, price}]}
    """
    # We can't compute real P&L without purchase prices, so we show
    # the 1-week return of the weighted portfolio instead.
    try:
        import yfinance as yf
        one_week_ago = (datetime.now(timezone.utc) - timedelta(days=8)).strftime("%Y-%m-%d")
        today        = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tickers = list(holdings.keys())
        hist = yf.download(tickers, start=one_week_ago, end=today,
                           auto_adjust=True, progress=False)
        close = hist["Close"] if "Close" in hist else hist
        weekly_return = 0.0
        by_ticker = []
        for t, w in holdings.items():
            try:
                col = close[t] if hasattr(close, "__getitem__") else close
                prices_series = col.dropna()
                if len(prices_series) >= 2:
                    ret = (prices_series.iloc[-1] / prices_series.iloc[0] - 1)
                    weekly_return += ret * w
                    by_ticker.append({
                        "ticker": t,
                        "weight_pct": round(w * 100, 1),
                        "week_ret_pct": round(ret * 100, 2),
                        "current_price": round(current_prices.get(t, 0), 2),
                    })
            except Exception:
                pass
        pnl_usd = ref_value * weekly_return
        return {
            "weekly_return_pct": round(weekly_return * 100, 2),
            "pnl_usd": round(pnl_usd, 2),
            "by_ticker": sorted(by_ticker, key=lambda x: x["week_ret_pct"], reverse=True),
        }
    except Exception as exc:
        logger.error("P&L compute failed: %s", exc)
        return {"weekly_return_pct": 0.0, "pnl_usd": 0.0, "by_ticker": []}


def _format_digest(user_id: str, snapshot: dict, pnl: dict) -> str:
    holdings = json.loads(snapshot["holdings"])
    metrics  = json.loads(snapshot["metrics"])

    top_ticker = pnl["by_ticker"][0] if pnl["by_ticker"] else {}
    worst_ticker = pnl["by_ticker"][-1] if pnl["by_ticker"] else {}

    now_str  = datetime.now(timezone.utc).strftime("%d %b %Y")
    ret_icon = "📈" if pnl["weekly_return_pct"] >= 0 else "📉"
    pnl_sign = "+" if pnl["pnl_usd"] >= 0 else ""

    lines = [
        f"📊 *EisaX Weekly Portfolio Digest*",
        f"User: `{user_id[:16]}` | {now_str}",
        f"",
        f"{ret_icon} *Weekly Return:* `{pnl['weekly_return_pct']:+.2f}%`",
        f"💵 *Est. P&L (on $100k):* `{pnl_sign}${pnl['pnl_usd']:,.0f}`",
        f"",
        f"📐 *Risk Metrics:*",
        f"  • Sharpe: `{metrics.get('sharpe', 'N/A')}`",
        f"  • Beta: `{metrics.get('beta', 'N/A')}`",
        f"  • CVaR 95%: `{metrics.get('cvar_95', 'N/A')}`",
        f"  • Ann. Vol: `{metrics.get('ann_vol', 'N/A')}`",
        f"",
    ]

    if top_ticker:
        lines.append(f"🏆 *Top Mover:* {top_ticker['ticker']} `{top_ticker['week_ret_pct']:+.2f}%`")
    if worst_ticker and worst_ticker != top_ticker:
        lines.append(f"⚠️ *Laggard:* {worst_ticker['ticker']} `{worst_ticker['week_ret_pct']:+.2f}%`")

    lines += [
        f"",
        f"🗂 *Holdings ({len(holdings)} positions):*",
    ]
    for item in pnl["by_ticker"][:5]:
        icon = "🟢" if item["week_ret_pct"] >= 0 else "🔴"
        lines.append(f"  {icon} {item['ticker']} {item['weight_pct']}% → `{item['week_ret_pct']:+.2f}%`")
    if len(holdings) > 5:
        lines.append(f"  … and {len(holdings) - 5} more positions")

    lines += [
        f"",
        f"_EisaX AI · For informational purposes only_",
    ]
    return "\n".join(lines)


def run_weekly_digest():
    logger.info("Starting weekly portfolio digest")
    conn = sqlite3.connect(APP_DB)
    conn.row_factory = sqlite3.Row

    # Get most recent snapshot per user (last 30 days)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    rows = conn.execute("""
        SELECT s.*
        FROM portfolio_snapshots s
        INNER JOIN (
            SELECT user_id, MAX(timestamp) AS max_ts
            FROM portfolio_snapshots
            WHERE timestamp >= ?
            GROUP BY user_id
        ) latest ON s.user_id = latest.user_id AND s.timestamp = latest.max_ts
        WHERE s.user_id != 'anonymous'
        ORDER BY s.timestamp DESC
    """, (cutoff,)).fetchall()
    conn.close()

    if not rows:
        logger.info("No active portfolio snapshots found — nothing to digest")
        _send_telegram("📊 *EisaX Weekly Digest*\n\nNo active portfolio snapshots this week.")
        return

    logger.info("Processing digest for %d users", len(rows))
    sent = 0

    for row in rows:
        try:
            holdings = json.loads(row["holdings"])
            tickers  = list(holdings.keys())
            current_prices = _get_live_prices(tickers)
            pnl = _compute_pnl(holdings, current_prices)
            msg = _format_digest(row["user_id"], dict(row), pnl)
            if _send_telegram(msg):
                sent += 1
                logger.info("Digest sent for user %s", row["user_id"][:16])
        except Exception as exc:
            logger.error("Failed to process digest for user %s: %s", row["user_id"], exc)

    # Admin summary
    _send_telegram(
        f"✅ *Weekly Digest Complete*\n"
        f"Sent {sent}/{len(rows)} digests\n"
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    logger.info("Weekly digest complete — sent %d/%d", sent, len(rows))


if __name__ == "__main__":
    run_weekly_digest()
