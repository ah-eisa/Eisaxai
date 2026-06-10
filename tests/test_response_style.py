import unittest
from core.response_builder import ResponseBuilder
from core.router import Router
from core.style_card import StyleCard
from core.decision_engine import DecisionEngine

class TestResponseStyle(unittest.TestCase):
    
    def setUp(self):
        self.builder = ResponseBuilder()

    def test_rejection_format(self):
        decision = {
            "decision": "REJECTED",
            "decision_id": "test",
            "risk_flags": ["policy_violation"],
            "rationale": ["Sensitive content"]
        }
        res = self.builder.build_response({}, decision)
        self.assertIn("Request declined", res["reply"])
        self.assertIn("Recommendation:", res["reply"])
        
    def test_needs_info_format(self):
        decision = {
            "decision": "NEEDS_INFO",
            "decision_id": "test",
            "required_inputs": ["portfolio_amount"],
            "rationale": ["Amount missing"]
        }
        res = self.builder.build_response({}, decision)
        self.assertIn("Amount missing", res["reply"])
        self.assertIn("Please confirm:", res["reply"])
        # Verify it asks only ONE question (heuristic)
        self.assertEqual(res["reply"].count("?"), 1)

    def test_approved_fallback_truncation(self):
        decision = {"decision": "APPROVED", "decision_id": "ok"}
        long_text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\nLine 9\nLine 10\nLine 11"
        res = self.builder.build_response({}, decision, raw_reply=long_text)
        
        # Should be truncated to <= 9 lines
        line_count = len(res["reply"].split('\n'))
        self.assertLessEqual(line_count, 9)

    def test_header_removal(self):
        decision = {"decision": "APPROVED", "decision_id": "ok"}
        text_with_headers = "EXECUTIVE SUMMARY\n\nThis is a summary.\n\n## Main Analysis\n- Point 1\n- Point 2"
        res = self.builder.build_response({}, decision, raw_reply=text_with_headers)
        
        # Should NOT contain headers
        print(f"DEBUG REPLY: {res['reply']}")
        self.assertNotIn("EXECUTIVE SUMMARY", res["reply"])
        self.assertNotIn("## Main Analysis", res["reply"])
        self.assertIn("- Point 1", res["reply"])

    def test_guaranteed_refusal(self):
        de = DecisionEngine()
        ctx = {"last_text": "I need guaranteed 50% monthly returns"}
        intent = {"intent": "PORTFOLIO", "confidence": 0.9}
        decision = de.evaluate(intent, ctx)
        
        self.assertEqual(decision["decision"], "REJECTED")
        self.assertIn("Guaranteed high returns", decision["rationale"][0])
        
        # Verify Refusal Builder length
        res = self.builder.build_response(intent, decision)
        line_count = len(res["reply"].strip().split('\n'))
        self.assertLessEqual(line_count, 9)

if __name__ == "__main__":
    unittest.main()
