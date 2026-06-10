import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# MOCK MISSING DEPS BEFORE IMPORTS
sys.modules["pypfopt"] = MagicMock()
sys.modules["pypfopt.expected_returns"] = MagicMock()
sys.modules["pypfopt.risk_models"] = MagicMock()
sys.modules["pypfopt.efficient_frontier"] = MagicMock()
sys.modules["empyrical"] = MagicMock()
sys.modules["cvxpy"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()
sys.modules["yfinance"] = MagicMock() # Mock yfinance too to be safe

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import core.analytics as ca
# Ensure core.data is loaded so patch can find it
import core.data 
from core.agents.finance import FinancialAgent

print("Running Verified verify_analytics.py v2")

class TestAnalytics(unittest.TestCase):
    
    def test_sma(self):
        data = pd.Series([10, 20, 30, 40, 50])
        sma = ca.calculate_sma(data, window=3)
        # SMA of [30, 40, 50] is 40
        self.assertEqual(sma.iloc[-1], 40.0)
        print("SUCCESS: SMA Calculation.")

    def test_max_drawdown(self):
        # 100 -> 120 (peak) -> 60 (trough) -> 90
        # Drawdown = (60 - 120) / 120 = -0.5 (-50%)
        data = pd.Series([100, 110, 120, 100, 60, 90])
        mdd = ca.calculate_max_drawdown(data)
        self.assertAlmostEqual(mdd, -0.5)
        print("SUCCESS: Max Drawdown Calculation.")

    @patch('core.data.get_prices')
    def test_agent_analytics_route(self, mock_get_prices):
        """Verify FinancialAgent routes 'analyze AAPL' to analytics."""
        
        # Mock Price Data
        dates = pd.date_range(start='2023-01-01', periods=100)
        prices = pd.DataFrame({'AAPL': np.linspace(100, 150, 100)}, index=dates)
        mock_get_prices.return_value = prices

        agent = FinancialAgent()
        
        # Test Input
        response = agent.think("Analyze AAPL for me", {"memory": {}})
        
        reply = response.get("reply", "")
        self.assertIn("CIO Memorandum: AAPL", reply)
        self.assertIn("SMA 200", reply)
        self.assertIn("Value at Risk", reply)
        print("SUCCESS: Agent routed 'Analyze AAPL' correctly.")

if __name__ == "__main__":
    unittest.main()
