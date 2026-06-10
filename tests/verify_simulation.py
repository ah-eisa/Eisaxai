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
sys.modules["yfinance"] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import core.analytics as ca
# Ensure core.data is loaded so patch can find it
import core.data 
from core.agents.finance import FinancialAgent

print("Running Monte Carlo Verification")

class TestSimulation(unittest.TestCase):
    
    def test_monte_carlo_stats(self):
        # Create a synthetic price series with known drift/vol
        # Start 100, 1% daily return (huge drift for test), 0 vol
        prices = pd.Series([100 * (1.01)**i for i in range(100)])
        
        # Simulate 10 days
        paths = ca.calculate_monte_carlo(prices, days=10, simulations=100)
        stats = ca.get_simulation_stats(paths)
        
        # Expected price after 10 days ~= 100 * (1.01)^100 * (1.01)^10
        # Wait, the MC uses the *last* price as start.
        # Last price index 99 is 100 * 1.01^99
        start_price = prices.iloc[-1]
        expected_end = start_price * (1.01 ** 10)
        
        # With 0 vol in input (it's a perfect curve), returns std should be near 0
        # So MC should produce paths very close to the expected drift
        
        print(f"Start: {start_price:.2f}, Mean End: {stats['mean']:.2f}, Expected: {expected_end:.2f}")
        self.assertAlmostEqual(stats['mean'], expected_end, delta=expected_end*0.05) # Allow 5% variance for numerical noise
        print("SUCCESS: Monte Carlo Drift Logic.")

    @patch('core.data.get_prices')
    def test_agent_forecast_route(self, mock_get_prices):
        """Verify FinancialAgent routes 'forecast AAPL' to simulation."""
        
        # Mock Price Data
        dates = pd.date_range(start='2020-01-01', periods=252)
        prices = pd.DataFrame({'AAPL': np.linspace(100, 150, 252)}, index=dates)
        mock_get_prices.return_value = prices

        agent = FinancialAgent()
        
        # Test Input
        response = agent.think("Forecast AAPL results", {"memory": {}})
        
        reply = response.get("reply", "")
        self.assertIn("Monte Carlo Forecast: AAPL", reply)
        self.assertIn("Projected Outcomes", reply)
        self.assertIn("1,000 Paths", reply)
        print("SUCCESS: Agent routed 'Forecast AAPL' correctly.")

if __name__ == "__main__":
    unittest.main()
