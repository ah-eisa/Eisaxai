import os, sqlite3, logging, requests
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
_DB_PATH = Path('/home/ubuntu/investwise/data/price_alerts.db')
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _get_conn():
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute('''CREATE TABLE IF NOT EXISTS user_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        ticker TEXT NOT NULL,
        condition TEXT NOT NULL CHECK(condition IN ("above","below","change_pct")),
        threshold REAL NOT NULL,
        last_triggered TEXT,
        active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    return conn

def add_alert(user_id, ticker, condition, threshold) -> int:
    with _get_conn() as c:
        cur = c.execute('INSERT INTO user_alerts(user_id,ticker,condition,threshold) VALUES(?,?,?,?)',
                        (user_id, ticker.upper(), condition, float(threshold)))
        return cur.lastrowid

def get_active_alerts() -> list:
    c = _get_conn()
    rows = c.execute('SELECT * FROM user_alerts WHERE active=1').fetchall()
    c.close()
    return [dict(r) for r in rows]

def mark_triggered(alert_id: int):
    with _get_conn() as c:
        c.execute('UPDATE user_alerts SET last_triggered=?, active=0 WHERE id=?',
                  (datetime.now(timezone.utc).isoformat(), alert_id))

def delete_alert(alert_id: int, user_id: str):
    with _get_conn() as c:
        c.execute('DELETE FROM user_alerts WHERE id=? AND user_id=?', (alert_id, user_id))

def get_user_alerts(user_id: str) -> list:
    c = _get_conn()
    rows = c.execute('SELECT * FROM user_alerts WHERE user_id=? ORDER BY created_at DESC', (user_id,)).fetchall()
    c.close()
    return [dict(r) for r in rows]

def check_alerts(prices: dict) -> list:
    triggered = []
    for alert in get_active_alerts():
        t = alert['ticker']
        price = prices.get(t)
        if price is None: continue
        cond = alert['condition']
        thresh = alert['threshold']
        if (cond == 'above' and price >= thresh) or (cond == 'below' and price <= thresh):
            triggered.append({**alert, 'current_price': price})
    return triggered

def send_telegram_alert(alert: dict, current_price: float):
    bot = os.getenv('TELEGRAM_BOT_TOKEN','')
    cid = os.getenv('TELEGRAM_CHAT_ID','')
    if not bot or not cid:
        logger.warning('Telegram not configured for price alerts')
        return
    direction = '📈 ABOVE' if alert['condition'] == 'above' else '📉 BELOW'
    text = (f'🚨 *Price Alert Triggered!*\n'
            f'{alert["ticker"]} is {direction} {alert["threshold"]}\n'
            f'Current price: *{current_price:.4f}*\n'
            f'Alert ID: {alert["id"]}')
    try:
        requests.post(f'https://api.telegram.org/bot{bot}/sendMessage',
                      json={'chat_id': cid, 'text': text, 'parse_mode': 'Markdown'}, timeout=8)
    except Exception as e:
        logger.error('Telegram alert failed: %s', e)
