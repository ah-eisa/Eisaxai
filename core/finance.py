import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from core.data import get_prices  # نتأكد إن الدالة دي موجودة في core/data.py

logger = logging.getLogger(__name__)

class PortfolioCore:
    def __init__(self):
        self.annual_trading_days = 252

    def get_portfolio_data(self, tickers: List[str], period: str = "1y") -> Tuple[Optional[List[float]], Optional[List[List[float]]], List[str]]:
        """
        يجلب البيانات ويحسب العوائد ومصفوفة التباين
        """
        try:
            df = get_prices(tickers)
            if df.empty or len(df.columns) < len(tickers):
                return None, None, []

            # 1. حساب العوائد اليومية
            returns = df.pct_change().dropna()
            
            # 2. متوسط العوائد (Annualized Mean Returns)
            mean_returns = (returns.mean() * self.annual_trading_days).tolist()
            
            # 3. مصفوفة التباين (Annualized Covariance Matrix)
            cov_matrix = (returns.cov() * self.annual_trading_days).values.tolist()
            
            return mean_returns, cov_matrix, list(df.columns)
            
        except Exception as e:
            logger.error(f"Error preparing portfolio data: {e}")
            return None, None, []

    def calculate_position_sizing(self, capital: float, risk_per_trade: float, var_95: float) -> Dict:
        """
        يحسب حجم الصفقة المقترح بناءً على الـ Value at Risk
        """
        # إذا كان الـ VaR هو 4%، يعني المخاطرة لكل سهم عالية
        suggested_allocation = (capital * risk_per_trade) / abs(var_95)
        return {
            "suggested_capital": round(suggested_allocation, 2),
            "risk_exposure": risk_per_trade * 100
        }

def detect_risk_pref(message: str) -> str:
    """تحليل نص المستخدم لمعرفة مستوى المخاطرة"""
    msg = message.lower()
    if any(word in msg for word in ["aggressive", "high risk", "مخاطرة عالية", "نمو"]):
        return "high"
    if any(word in msg for word in ["conservative", "low risk", "آمن", "محافظ"]):
        return "low"
    return "medium"
def calculate_trade_execution(self, total_capital: float, weights: Dict[str, float], prices: Dict[str, float]) -> List[Dict]:
        """
        تحويل الأوزان لعدد أسهم فعلي بناءً على رأس المال المتاح
        """
        orders = []
        for ticker, weight in weights.items():
            price = prices.get(ticker)
            if price and price > 0:
                allocation = total_capital * weight
                shares = int(allocation / price)
                orders.append({
                    "ticker": ticker,
                    "shares": shares,
                    "notional": round(shares * price, 2),
                    "weight_actual": round((shares * price / total_capital) * 100, 2)
                })
        return orders
