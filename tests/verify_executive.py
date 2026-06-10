import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# MOCK HEAVY MODULES
sys.modules["core.agents.finance"] = MagicMock()
sys.modules["core.portfolio_manager"] = MagicMock()
sys.modules["yfinance"] = MagicMock()

from core.agents.general import GeneralAgent
import state

class TestExecutivePersona(unittest.TestCase):
    
    def test_system_prompt_alignment(self):
        """Verify the system prompt contains key Executive Operator keywords."""
        prompt = state.SYSTEM_PROMPTS["assistant"]
        
        # Key checkpoints from the user request
        keywords = [
            "Executive Operator",
            "Strategic Reviewer",
            "Capital preservation is the baseline",
            "Identify the real decision",
            "Analyze downside first",
            "Analyze upside second",
            "Sound impressed",
            "helpdesk assistant",
            "ACTION VS CLARIFICATION RULE",
            "act immediately"
        ]
        
        for kw in keywords:
            with self.subTest(keyword=kw):
                self.assertIn(kw.lower(), prompt.lower())
        
        print("SUCCESS: System prompt is aligned with Executive Operator Spec.")

    @patch('core.agents.general.get_client')
    def test_action_bias_flow(self, mock_get_client):
        """Verify that 'export this' triggers execution logic directly."""
        # Setup Mock Client
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        
        # Setup Mock Response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Exporting to PDF now..."))]
        mock_client_instance.create_completion.return_value = mock_response

        agent = GeneralAgent()
        
        # Mock Context with some history so "this" has context
        context = {
            "history": [{"role": "assistant", "content": "Here is your portfolio analysis."}],
            "memory": {}
        }

        # Call think with an execution request
        # In a real scenario, this might trigger the Orchestrator/Router to call a specific handler,
        # but here we check if GeneralAgent proceeds to LLM without 'clarification' logic in code.
        result = agent.think("export this to pdf", context)
        
        self.assertEqual(result["type"], "chat.reply")
        self.assertIn("Exporting", result["reply"])
        
        # Check LLM call
        mock_client_instance.create_completion.assert_called_once()
        print("SUCCESS: Agent proceeds to execution on 'export this'.")

if __name__ == "__main__":
    unittest.main()
