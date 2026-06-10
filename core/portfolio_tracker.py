"""
Portfolio Tracking System for EisaX
Allows users to track their investments in real-time
"""
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
logger = logging.getLogger(__name__)

from core.config import PORTFOLIOS_DIR as _cfg_ptf_dir

class PortfolioTracker:
    def __init__(self, data_dir: str = str(_cfg_ptf_dir)):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def _get_portfolio_path(self, user_id: str) -> str:
        """Get the file path for a user's portfolio"""
        return os.path.join(self.data_dir, f"{user_id}.json")
    
    def get_portfolio(self, user_id: str) -> Dict:
        """Load user's portfolio from disk"""
        path = self._get_portfolio_path(user_id)
        if not os.path.exists(path):
            return {"positions": [], "cash": 0.0, "created_at": datetime.now().isoformat()}
        with open(path, 'r') as f:
            data = json.load(f)
        # backwards compat: inject cash key if missing
        data.setdefault("cash", 0.0)
        return data

    def set_cash(self, user_id: str, amount: float) -> bool:
        """Set the cash amount for a user's portfolio."""
        portfolio = self.get_portfolio(user_id)
        portfolio["cash"] = max(0.0, float(amount))
        return self.save_portfolio(user_id, portfolio)
    
    def save_portfolio(self, user_id: str, portfolio: Dict) -> bool:
        """Save user's portfolio to disk"""
        try:
            path = self._get_portfolio_path(user_id)
            with open(path, 'w') as f:
                json.dump(portfolio, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"[Portfolio] Save failed: {e}")
            return False
    
    def add_position(self, user_id: str, ticker: str, shares: float,
                     purchase_price: float, date: Optional[str] = None,
                     market: Optional[str] = None) -> Dict:
        """Add a new position to portfolio"""
        portfolio = self.get_portfolio(user_id)

        position = {
            "ticker": ticker.upper(),
            "market": (market or "uae").lower(),
            "shares": shares,
            "purchase_price": purchase_price,
            "purchase_date": date or datetime.now().strftime("%Y-%m-%d"),
            "added_at": datetime.now().isoformat()
        }

        # Check if ticker already exists - add to existing position
        existing = next((p for p in portfolio["positions"] if p["ticker"] == ticker.upper()), None)
        if existing:
            # Calculate weighted average price
            total_shares = existing["shares"] + shares
            avg_price = ((existing["shares"] * existing["purchase_price"]) +
                        (shares * purchase_price)) / total_shares
            existing["shares"] = total_shares
            existing["purchase_price"] = avg_price
            existing["last_updated"] = datetime.now().isoformat()
        else:
            portfolio["positions"].append(position)
        
        self.save_portfolio(user_id, portfolio)
        return {"success": True, "position": position, "action": "updated" if existing else "added"}
    
    def remove_position(self, user_id: str, ticker: str, shares: Optional[float] = None) -> Dict:
        """Remove a position or reduce shares"""
        portfolio = self.get_portfolio(user_id)
        ticker = ticker.upper()
        
        position = next((p for p in portfolio["positions"] if p["ticker"] == ticker), None)
        if not position:
            return {"success": False, "error": f"{ticker} not found in portfolio"}
        
        if shares is None or shares >= position["shares"]:
            # Remove entire position
            portfolio["positions"] = [p for p in portfolio["positions"] if p["ticker"] != ticker]
            action = "removed"
        else:
            # Reduce shares
            position["shares"] -= shares
            position["last_updated"] = datetime.now().isoformat()
            action = "reduced"
        
        self.save_portfolio(user_id, portfolio)
        return {"success": True, "ticker": ticker, "action": action}
    
    def get_position_value(self, ticker: str, shares: float, purchase_price: float) -> Dict:
        """Calculate current value and P&L for a position"""
        try:
            # Get live price
            from core.market_data import get_full_stock_profile
            profile = get_full_stock_profile(ticker)
            current_price = profile.get("quote", {}).get("price")
            
            if not current_price:
                return {"error": f"Could not fetch price for {ticker}"}
            
            cost_basis = shares * purchase_price
            current_value = shares * current_price
            pnl = current_value - cost_basis
            pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
            
            return {
                "ticker": ticker,
                "shares": shares,
                "purchase_price": purchase_price,
                "current_price": current_price,
                "cost_basis": cost_basis,
                "current_value": current_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_portfolio_summary(self, user_id: str) -> Dict:
        """Get complete portfolio with live values"""
        portfolio = self.get_portfolio(user_id)
        
        if not portfolio["positions"]:
            return {"success": False, "message": "Portfolio is empty"}
        
        positions_data = []
        total_cost = 0
        total_value = 0
        
        for pos in portfolio["positions"]:
            value = self.get_position_value(
                pos["ticker"], 
                pos["shares"], 
                pos["purchase_price"]
            )
            
            if "error" not in value:
                positions_data.append(value)
                total_cost += value["cost_basis"]
                total_value += value["current_value"]
        
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        return {
            "success": True,
            "positions": positions_data,
            "summary": {
                "total_cost": total_cost,
                "total_value": total_value,
                "total_pnl": total_pnl,
                "total_pnl_pct": total_pnl_pct,
                "position_count": len(positions_data)
            }
        }
