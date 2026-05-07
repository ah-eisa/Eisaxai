# Auto-extracted mixin — do not edit directly; source of truth is git history.
from __future__ import annotations
from typing import Any, Dict, Optional
import logging
import re
import state
from datetime import datetime
from core.intent_classifier import IntentClassifier
from core.broker import BrokerClient
logger = logging.getLogger(__name__)


class TradeMixin:
    def _handle_forecast(self, sid: str, mem: Dict[str, Any], msg: str) -> Dict[str, Any]:
        # ... (imports) ...
        import core.analytics as ca
        import numpy as np
        from core.data import get_prices
        
        tickers = IntentClassifier.extract_tickers(msg)
        if not tickers and not state.get_artifact(sid):
            tickers = mem.get("tickers", [])
            
        if not tickers and not state.get_artifact(sid):
            return {"type": "chat.reply", "reply": "Please specify a ticker to forecast."}
            
        target = tickers[0]
        try:
            prices = get_prices([target], start="2020-01-01", end=None)
            if prices.empty:
                return {"type": "error", "reply": f"Could not fetch data for {target}."}
                
            series = prices[target]
            sim_days = 252 * 1 # 1 Year default
            if "5 year" in msg.lower(): sim_days = 252 * 5
            if "10 year" in msg.lower(): sim_days = 252 * 10
            
            paths = ca.calculate_monte_carlo(series, days=sim_days)
            stats = ca.get_simulation_stats(paths)
            
            current_price = series.iloc[-1]
            p50_ret = (stats['p50'] / current_price) - 1
            
            reply = (
                f"# Monte Carlo Forecast: {target}\n\n"
                f"**Horizon:** {sim_days/252:.1f} Years\n"
                f"**Simulations:** 1,000 Paths\n"
                f"**Current Price:** ${current_price:.2f}\n\n"
                f"## Projected Outcomes\n"
                f"- **Bear Case (P10):** ${stats['p10']:.2f}\n"
                f"- **Base Case (P50):** ${stats['p50']:.2f} ({p50_ret*100:+.1f}%)\n"
                f"- **Bull Case (P90):** ${stats['p90']:.2f}\n\n"
                f"**Analysis:** Based on historical volatility of {series.pct_change().std()*np.sqrt(252)*100:.1f}%. "
                f"The range of outcomes indicates the inherent uncertainty in long-term projections."
            )
            
            # SAVE ARTIFACT
            state.set_artifact(sid, {
                "type": "forecast",
                "content": reply,
                "source": "self_generated",
                "exportable": True,
                "timestamp": datetime.now()
            })
            
            return {
                "type": "chat.reply",
                "reply": reply,
                "data": {"agent": "finance", "forecast": stats}
            }
            
        except Exception as e:
            return {"type": "error", "reply": f"Forecast failed for {target}: {e}"}

    def _handle_trade(self, sid: str, mem: Dict[str, Any], msg: str) -> Dict[str, Any]:
        """
        Executes paper trades via the BrokerClient.
        """
        from core.broker import BrokerClient
        
        # 1. Initialize Broker
        broker = BrokerClient()
        if not broker.is_active():
             return {"type": "error", "reply": "Broker connection failed. Please check ALPACA_API_KEY and ALPACA_SECRET_KEY."}

        # 2. Parse Intent
        tickers = IntentClassifier.extract_tickers(msg)
        if not tickers and not state.get_artifact(sid):
            return {"type": "chat.reply", "reply": "Please specify a ticker to trade (e.g., 'Buy 10 AAPL')."}
        
        symbol = tickers[0]
        side = "buy" if "buy" in msg.lower() else "sell" if "sell" in msg.lower() else None
        
        if not side:
             return {"type": "chat.reply", "reply": "Please specify 'buy' or 'sell'."}
             
        # Simple quantity parser: look for first number
        import re
        # Find integers or floats
        qty_match = re.search(r'\b\d+\b', msg)
        qty = float(qty_match.group(0)) if qty_match else 1.0
        
        # 3. Execute
        try:
            order = broker.submit_order(symbol, qty, side, "market", "day")
            
            if "error" in order:
                return {"type": "error", "reply": f"Trade rejected: {order['error']}"}
            
            reply = (
                f"# Trade Submitted: {side.upper()} {symbol}\n"
                f"**Qty:** {order['qty']}\n"
                f"**Status:** {order['status'].upper()}\n"
                f"**Order ID:** `{order['id']}`\n"
            )
            return {
                "type": "chat.reply", 
                "reply": reply,
                "data": {"agent": "finance", "trade_id": order['id']}
            }
            
        except Exception as e:
             return {"type": "error", "reply": f"Trade execution failed: {e}"}

    def _handle_greeks(self, sid: str, msg: str) -> Dict[str, Any]:
        """Calculates Option Greeks using Black-Scholes."""
        import core.analytics as ca
        import re
        
        # 1. Parameter Extraction (Defaults)
        S = 100.0; K = 100.0; T = 0.25; r = 0.05; sigma = 0.20
        option_type = "call" if "call" in msg.lower() else "put"
        
        # Try to find specific values via regex
        spot_match = re.search(r"(?:spot|price|current)\s*(?:is|at|=)?\s*\$?(\d+\.?\d*)", msg, re.I)
        if spot_match: S = float(spot_match.group(1))
        
        strike_match = re.search(r"(?:strike)\s*(?:is|at|=)?\s*\$?(\d+\.?\d*)", msg, re.I)
        if strike_match: K = float(strike_match.group(1))
        
        iv_match = re.search(r"(?:iv|vol|volatility)\s*(?:is|at|=)?\s*(\d+\.?\d*)", msg, re.I)
        if iv_match: 
            val = float(iv_match.group(1))
            sigma = val / 100.0 if val > 1.0 else val
            
        rate_match = re.search(r"(?:rate|rf)\s*(?:is|at|=)?\s*(\d+\.?\d*)", msg, re.I)
        if rate_match:
            val = float(rate_match.group(1))
            r = val / 100.0 if val > 1.0 else val
            
        months_match = re.search(r"(\d+)\s*month", msg, re.I)
        if months_match: T = float(months_match.group(1)) / 12.0
        
        # 2. Calculate
        try:
            res = ca.calculate_black_scholes(S, K, T, r, sigma, option_type)
            
            reply = (
                f"# Strategic Greeks Analysis: {option_type.upper()}\n\n"
                f"**Engine:** Black-Scholes-Merton Model\n\n"
                f"### Input Parameters\n"
                f"- **Spot:** ${S:.2f}\n"
                f"- **Strike:** ${K:.2f} ({((K/S)-1)*100:+.1f}% from spot)\n"
                f"- **Volatility (IV):** {sigma*100:.1f}%\n"
                f"- **Time to Expiry:** {T*12:.1f} months\n"
                f"- **Risk-free Rate:** {r*100:.2f}%\n\n"
                f"### Derived Greeks\n"
                f"| Metric | Value | Interpretation |\n"
                f"|---|---|---|\n"
                f"| **Delta** | {res['delta']:.4f} | Probabilistic exposure to price move |\n"
                f"| **Theta** | {res['theta']:.4f} | Daily time decay (value loss) |\n"
                f"| **Theory Price** | ${res['price']:.2f} | Fair value projection |\n\n"
                f"**EISAX Operational Note:** Theta decay accelerates sharply in the final 30 days. Plan your entries accordingly."
            )
            
            # SAVE ARTIFACT
            state.set_artifact(sid, {
                "type": "greeks",
                "content": reply,
                "source": "self_generated",
                "exportable": True,
                "timestamp": datetime.now()
            })
                
            return {
                "type": "chat.reply", 
                "reply": reply, 
                "data": {"agent": "finance", "greeks": res}
            }
        except Exception as e:
            # Re-raise so the try-except in think() can log it and fallback
            raise ValueError(f"Greeks calculation failed: {e}")


