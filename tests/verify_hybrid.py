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
from core.intent_classifier import IntentClassifier
import state

class TestHybridPersona(unittest.TestCase):
    
    @patch('core.agents.general.get_client')
    def test_assistant_mode_default(self, mock_get_client):
        """Verify 'hi' triggers Assistant Persona."""
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        
        # Mock Response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_client_instance.create_completion.return_value = mock_response

        agent = GeneralAgent()
        agent.think("hi", {"history": [], "memory": {}})
        
        # Check messages sent to LLM
        args, kwargs = mock_client_instance.create_completion.call_args
        messages = kwargs.get('messages', [])
        system_prompt = next((m['content'] for m in messages if m['role'] == 'system'), "")
        
        self.assertIn("Personal Assistant", system_prompt)
        self.assertNotIn("Chief Investment Officer", system_prompt)
        print("SUCCESS: 'hi' triggered Assistant Mode.")

    @patch('core.agents.general.get_client')
    @patch('core.llm.LLMClient.create_completion') # If classify_intent_hybrid calls LLM
    def test_cio_mode_trigger(self, mock_classify_llm, mock_get_client):
        """Verify 'crypto' triggers CIO Persona."""
        # Mocking get_client for the agent
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        
        # Mocking the classification LLM response if it falls through to LLM
        mock_classify_resp = MagicMock()
        mock_classify_resp.choices = [MagicMock(message=MagicMock(content="INVESTMENT"))]
        mock_classify_llm.return_value = mock_classify_resp

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Downside risk for crypto..."))]
        mock_client_instance.create_completion.return_value = mock_response

        agent = GeneralAgent()
        agent.think("what is the risk of crypto?", {"history": [], "memory": {}})
        
        # Check messages sent to LLM
        args, kwargs = mock_client_instance.create_completion.call_args
        messages = kwargs.get('messages', [])
        system_prompt = next((m['content'] for m in messages if m['role'] == 'system'), "")
        
        self.assertIn("Chief Investment Officer", system_prompt)
        self.assertIn("Downside must be assessed before upside", system_prompt)
        print("SUCCESS: 'crypto' triggered CIO Mode.")

    @patch('core.agents.general.get_client')
    def test_execution_in_assistant_mode(self, mock_get_client):
        """Verify 'export' stays in Assistant Mode."""
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Exporting..."))]
        mock_client_instance.create_completion.return_value = mock_response

        agent = GeneralAgent()
        agent.think("export this to pdf", {"history": [], "memory": {}})
        
        args, kwargs = mock_client_instance.create_completion.call_args
        messages = kwargs.get('messages', [])
        system_prompt = next((m['content'] for m in messages if m['role'] == 'system'), "")
        
        self.assertIn("Assistant", system_prompt)
        print("SUCCESS: 'export' triggered Assistant Mode.")

if __name__ == "__main__":
    unittest.main()
