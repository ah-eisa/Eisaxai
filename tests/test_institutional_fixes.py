import unittest
from core.intent_classifier import IntentClassifier
from core.decision_engine import DecisionEngine
from core.response_builder import ResponseBuilder
from core.router import Router

# Mock orchestrator
class MockOrchestrator:
    def think(self, text, settings=None, history=None):
        lower_text = text.lower().strip()
        if "greeting" in lower_text or lower_text in ["hi", "hello"]:
            return {"type": "chat.reply", "reply": "Hi. Do you want a portfolio analysis or a market report?", "data": None}
        elif "portfolio" in lower_text or "analyze" in lower_text:
            return {"type": "chat.reply", "reply": "Portfolio data required.\nSend holdings as weights (%) or values—your choice?", "data": None}
        elif "report" in lower_text or "pdf" in lower_text:
            return {"type": "chat.reply", "reply": "Please specify time window: Today, Week, or Month?", "data": None}
        elif lower_text in ["today", "week", "month"]:
            # Handle time window follow-up
            return {"type": "chat.reply", "reply": f"Generating market report for: {lower_text.capitalize()}", "data": {"time_window": lower_text.capitalize()}}
        return {"type": "chat.reply", "reply": f"Echo: {text}", "data": None}

class TestInstitutionalFixes(unittest.TestCase):

    def setUp(self):
        self.router = Router(orchestrator=MockOrchestrator())
        self.classifier = IntentClassifier()
        self.decision = DecisionEngine()
        self.builder = ResponseBuilder()

    def test_router_hard_enforcement(self):
        # Test the _normalize_reply private method
        raw_text = """
EXECUTIVE SUMMARY

This is a valid line.

## Main Analysis
- Bullet 1
- Bullet 2

Investment Recommendation: Buy
Investment Strategy Report: Private
        """
        clean = self.router._normalize_reply(raw_text)
        self.assertNotIn("EXECUTIVE SUMMARY", clean)
        self.assertNotIn("Main Analysis", clean)
        self.assertNotIn("Investment Strategy Report", clean)
        self.assertIn("This is a valid line.", clean)
        self.assertIn("- Bullet 1", clean)
        
        # Test max 9 lines
        long_text = "\n".join([f"Line {i}" for i in range(20)])
        clean_long = self.router._normalize_reply(long_text)
        self.assertLessEqual(len(clean_long.split('\n')), 9)

    def test_report_arabic_intent(self):
        # Arabic: الأسواق (markets), تقرير (report)
        text = "pdf تقرير الأسواق"
        intent_str = self.classifier.detect_primary_intent(text)
        # Report export intent
        self.assertEqual(intent_str, "report_export")

    def test_report_does_not_ask_for_holdings(self):
        text = "market pdf report"
        intent_str = self.classifier.detect_primary_intent(text)
        # Report export intent
        self.assertEqual(intent_str, "report_export")
        

    def test_greeting_shortcut(self):
        res = self.router.handle_request("sess-greet", "hi")
        self.assertEqual(res["reply"], "Hi. Do you want a portfolio analysis or a market report?")

    def test_report_follow_up_flow(self):
        sid = "sess-flow-1"
        # 1. Ask for report
        res1 = self.router.handle_request(sid, "give me market report in pdf")
        self.assertIn("Today, Week, or Month?", res1["reply"])
        
        # 2. Provide follow-up answer
        res2 = self.router.handle_request(sid, "week")
        self.assertIn("Generating market report for: Week", res2["reply"])
        self.assertEqual(res2["data"]["time_window"], "Week")
        
    def test_portfolio_missing_data_intercept(self):
        # This tests the Router intercept
        text = "analyze my portfolio please"
        res = self.router.handle_request("session_test", text, meta={})
        
        # Should be the manual short reply
        self.assertIn("Portfolio data required", res["reply"])
        self.assertIn("Send holdings as weights", res["reply"])
        self.assertLessEqual(len(res["reply"].split('\n')), 9)
        # Should have exactly ONE question
        self.assertEqual(res["reply"].count("?"), 1, f"Expected 1 question mark, got {res['reply'].count('?')} in: {res['reply']}")

if __name__ == "__main__":
    unittest.main()
