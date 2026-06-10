import unittest
from core.intent_classifier import IntentClassifier
from core.agents.finance import FinancialAgent
import state

class TestDispatchStability(unittest.TestCase):
    def setUp(self):
        self.agent = FinancialAgent()
        self.context = {
            "session_id": "test_session",
            "memory": {},
            "history": []
        }

    def test_aggressive_ticker_prevention(self):
        """Test that general words do not trigger ticker extraction or optimization"""
        msg = "Tell me as a friend, without looking at any charts, what was the strike price we talked about?"
        tickers = IntentClassifier.extract_tickers(msg)
        
        # None of these should be tickers now
        blacklisted = ["TELL", "FRIEND", "STRIKE", "CHARTS", "TALKED"]
        for b in blacklisted:
            self.assertNotIn(b, tickers)
            
        intent = IntentClassifier.detect_primary_intent(msg)
        self.assertNotEqual(intent, "portfolio_optimize")
        print("SUCCESS: General conversation no longer triggers false ticker extraction.")

    def test_greeks_handler(self):
        """Test that Delta/Theta requests are handled via the new Greeks engine"""
        msg = "Calculate the Delta and Theta for QQQ $360 Put options expiring in 3 months. Assume spot is $400 and IV is 20%."
        
        # FinancialAgent should detect 'delta' or 'theta' and route to _handle_greeks
        response = self.agent.think(msg, self.context)
        
        self.assertEqual(response.get("type"), "chat.reply")
        self.assertIn("Greeks Analysis", response.get("reply"))
        self.assertIn("Delta", response.get("reply"))
        self.assertIn("Theta", response.get("reply"))
        print("SUCCESS: Greeks requests are handled locally without crashing.")

    def test_handler_error_isolation(self):
        """Test that if a handler fails, it falls back to chat rather than crashing"""
        # We'll simulate a failure by passing something that triggers a handler but then fails inside
        # Actually, let's just mock a failure or use a known one.
        # If we trigger 'optimize' but provide an invalid state, it should fallback.
        
        msg = "optimize THESE_TICKERS_DONT_EXIST"
        # This will trigger _handle_optimize, which calls get_prices, which fails.
        # Because of our try/except in think(), it should fallback to default chat.
        
        response = self.agent.think(msg, self.context)
        
        # It should still be a chat.reply (from the fallback LLM block)
        self.assertEqual(response.get("type"), "chat.reply")
        # And it shouldn't contain the system error message prefix
        self.assertNotIn("Agent Dispatch Error", response.get("reply"))
        print("SUCCESS: Handler failures now fall back to default chat gracefully.")

if __name__ == "__main__":
    unittest.main()
