import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# MOCK ALPACA
mock_alpaca = MagicMock()
sys.modules["alpaca_trade_api"] = mock_alpaca
sys.modules["alpaca_trade_api.rest"] = MagicMock()

# Mock other deps 
sys.modules["pypfopt"] = MagicMock()
sys.modules["empyrical"] = MagicMock()
sys.modules["cvxpy"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()
sys.modules["yfinance"] = MagicMock()

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.broker import BrokerClient
from core.agents.finance import FinancialAgent

print("Running Broker Verification")

class TestBroker(unittest.TestCase):
    
    def test_broker_client_mock(self):
        """Verify BrokerClient wraps Alpaca correctly."""
        # Setup env vars for this test
        with patch.dict(os.environ, {"ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test"}):
            # Mock REST class
            with patch("core.broker.REST") as MockREST:
                mock_api = MockREST.return_value
                client = BrokerClient()
                
                # Test submit_order
                client.submit_order("AAPL", 10, "buy")
                mock_api.submit_order.assert_called_with(
                    symbol="AAPL", qty=10, side="buy", type="market", time_in_force="day"
                )
                print("SUCCESS: BrokerClient.submit_order called Alpaca API.")

                print("SUCCESS: BrokerClient.submit_order called Alpaca API.")

    @patch("core.broker.BrokerClient")
    def test_agent_trade_route(self, MockBrokerClient):
        """Verify FinancialAgent routes 'Buy 10 AAPL' to BrokerClient."""
        
        # Setup Mock Broker instance
        mock_broker = MockBrokerClient.return_value
        mock_broker.is_active.return_value = True
        mock_broker.submit_order.return_value = {
            "id": "12345", "symbol": "AAPL", "qty": 10, "side": "buy", "status": "new"
        }
        
        agent = FinancialAgent()
        
        # Test Input: "Buy 10 AAPL"
        # We need IntentClassifier to interpret this as "trade_execution"
        # The agent uses IntentClassifier.detect_primary_intent internally.
        
        response = agent.think("Buy 10 AAPL", {"memory": {}})
        
        reply = response.get("reply", "")
        self.assertIn("Trade Submitted: BUY AAPL", reply)
        self.assertIn("Qty:** 10", reply)
        self.assertIn("12345", reply)
        
        mock_broker.submit_order.assert_called_once()
        args = mock_broker.submit_order.call_args
        # args[0] is positional args: symbol, qty, side, type, time_in_force
        print(f"Agent called broker with: {args}")
        self.assertEqual(args[0][0], "AAPL")
        self.assertEqual(args[0][1], 10.0)
        self.assertEqual(args[0][2], "buy")
        
        print("SUCCESS: Agent routed 'Buy 10 AAPL' to Broker.")

if __name__ == "__main__":
    unittest.main()
