"""
Alpaca Portfolio Integration for EisaX
يجيب المحفظة الحقيقية من Alpaca
"""
import os
from dotenv import load_dotenv
load_dotenv("/home/ubuntu/investwise/.env")

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER      = os.getenv("ALPACA_PAPER", "True").lower() == "true"
BASE_URL   = "https://paper-api.alpaca.markets" if PAPER else "https://api.alpaca.markets"

def _headers():
    return {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}

def get_account():
    import requests
    r = requests.get(f"{BASE_URL}/v2/account", headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()

def get_positions():
    import requests
    r = requests.get(f"{BASE_URL}/v2/positions", headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()

def get_portfolio_summary():
    account   = get_account()
    positions = get_positions()

    total_value   = float(account.get("portfolio_value", 0))
    cash          = float(account.get("cash", 0))
    buying_power  = float(account.get("buying_power", 0))

    pos_list = []
    total_pnl = 0

    for p in positions:
        pnl     = float(p.get("unrealized_pl", 0))
        pnl_pct = float(p.get("unrealized_plpc", 0)) * 100
        total_pnl += pnl
        pos_list.append({
            "ticker":         p["symbol"],
            "shares":         float(p["qty"]),
            "current_price":  float(p["current_price"]),
            "avg_entry":      float(p["avg_entry_price"]),
            "market_value":   float(p["market_value"]),
            "cost_basis":     float(p["cost_basis"]),
            "pnl":            pnl,
            "pnl_pct":        round(pnl_pct, 2),
            "side":           p.get("side", "long"),
        })

    return {
        "success":      True,
        "source":       "Alpaca " + ("Paper" if PAPER else "Live"),
        "account": {
            "portfolio_value": total_value,
            "cash":            cash,
            "buying_power":    buying_power,
            "total_pnl":       round(total_pnl, 2),
        },
        "positions": pos_list,
    }

if __name__ == "__main__":
    import json
    print(json.dumps(get_portfolio_summary(), indent=2))
