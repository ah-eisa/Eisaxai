from typing import Optional, Dict, Any
import os
import logging
from dotenv import load_dotenv
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST
except ImportError:
    tradeapi = None
    REST = None

logger = logging.getLogger(__name__)

class BrokerClient:
    """
    Client for interacting with brokerage APIs (Alpaca).
    """
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.api = None
        
        if self.api_key and self.secret_key and REST:
            try:
                self.api = REST(self.api_key, self.secret_key, self.base_url, api_version='v2')
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca API: {e}")
        elif not REST:
            logger.warning("alpaca-trade-api not installed.")
        else:
            logger.warning("Alpaca API keys not found. Brokerage features disabled.")

    def is_active(self) -> bool:
        return self.api is not None

    def get_account(self) -> Dict[str, Any]:
        """Returns account equity and buying power."""
        if not self.is_active():
            return {"error": "Broker not connected"}
        
        try:
            acct = self.api.get_account()
            return {
                "equity": float(acct.equity),
                "buying_power": float(acct.buying_power),
                "cash": float(acct.cash),
                "status": acct.status
            }
        except Exception as e:
            logger.error(f"Account fetch failed: {e}")
            return {"error": str(e)}

    def submit_order(self, symbol: str, qty: float, side: str, order_type: str = "market", time_in_force: str = "day") -> Dict[str, Any]:
        """Submits an order to the broker."""
        if not self.is_active():
            return {"error": "Broker not connected"}
        
        try:
            # Basic validation
            if side.lower() not in ["buy", "sell"]:
                return {"error": "Invalid side. Use 'buy' or 'sell'."}
                
            order = self.api.submit_order(
                symbol=symbol.upper(),
                qty=qty,
                side=side.lower(),
                type=order_type.lower(),
                time_in_force=time_in_force.lower()
            )
            return {
                "id": order.id,
                "symbol": order.symbol,
                "qty": float(order.qty) if order.qty else 0.0, 
                "side": order.side,
                "status": order.status
            }
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return {"error": str(e)}

    def get_positions(self) -> list:
        """Returns current open positions."""
        if not self.is_active():
            return []
            
        try:
            positions = self.api.list_positions()
            return [{
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc)
            } for p in positions]
        except Exception as e:
            logger.error(f"Positions fetch failed: {e}")
            return []
