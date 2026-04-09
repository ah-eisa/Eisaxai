import unittest
from core.intent_classifier import IntentClassifier
from core.session_manager import SessionManager
from core.decision_engine import DecisionEngine
from core.router import Router

# Mock orchestrator
class MockOrchestrator:
    def think(self, text, settings=None, history=None):
        return {"type": "chat.reply", "reply": f"Echo: {text}", "data": None}

    def handle_message(self, session_id, user_text, meta):
        return {"type": "chat.reply", "reply": f"Echo: {user_text}", "data": None}

class TestInstitutional(unittest.TestCase):
    
    def setUp(self):
        self.classifier = IntentClassifier()
        self.session = SessionManager()
        self.decision = DecisionEngine()
        self.router = Router(orchestrator=MockOrchestrator())

    # --- Intent Classification Tests ---
    def test_intent_portfolio(self):
        res = self.classifier.detect_primary_intent("optimize my portfolio")
        self.assertEqual(res, "portfolio_optimize")

    def test_intent_report(self):
        res = self.classifier.detect_primary_intent("generate pdf report")
        self.assertEqual(res, "report_export")

    def test_intent_policy(self):
        res = self.classifier.classify_intent_hybrid("what are the compliance policies?")
        self.assertEqual(res, "general")

    def test_intent_chat_fallback(self):
        res = self.classifier.detect_primary_intent("hello there")
        self.assertIsNone(res)
        
    # --- Session Tests ---
    def test_session_state_update(self):
        sid = "test-session-1"
        # First create a session
        self.session.get_or_create_session(sid, "test-user")
        # Then update state
        self.session.update_state(sid, {"mode": "investment"})
        state = self.session.get_state(sid)
        self.assertEqual(state["mode"], "investment")
        
    # --- Decision Tests ---
    def test_decision_approved(self):
        intent = {"intent": "CHAT", "confidence": 0.9}
        decision = self.decision.evaluate(intent, {})
        self.assertEqual(decision["decision"], "APPROVED")
        
    def test_decision_rejected(self):
        intent = {"intent": "PORTFOLIO", "confidence": 0.9}
        context = {"last_text": "invest in bitconnect ponzi scheme"}
        decision = self.decision.evaluate(intent, context)
        self.assertEqual(decision["decision"], "REJECTED")

    # --- Router Integration ---
    def test_router_flow(self):
        res = self.router.handle_request("sess-1", "hello world")
        self.assertEqual(res["type"], "chat.reply")
        self.assertIn("Echo", res["reply"])

if __name__ == "__main__":
    unittest.main()
