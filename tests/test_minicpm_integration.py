import sys
import os
import unittest
import pytest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# core.minicpm_client requires the optional `ollama` package, which is
# not installed on the staging/prod server. Skip the suite gracefully
# so the rest of pytest collection still works.
try:
    from core.conversation_router import ConversationRouter, RequestType
    from core.minicpm_client import MiniCPMClient
except ImportError as _minicpm_err:
    pytest.skip(
        f"MiniCPM/ollama dependency unavailable: {_minicpm_err}",
        allow_module_level=True,
    )

class TestMiniCPMIntegration(unittest.TestCase):

    def setUp(self):
        self.router = ConversationRouter()
        self.client = MiniCPMClient()

    @patch('core.minicpm_client.ollama.chat')
    def test_minicpm_client_chat(self, mock_chat):
        mock_chat.return_value = {'message': {'content': 'Hello human!'}}
        response = self.client.chat("Hello")
        self.assertEqual(response, 'Hello human!')
        mock_chat.assert_called_once()

    @patch('core.minicpm_client.ollama.chat')
    def test_minicpm_classification(self, mock_chat):
        # Mocking classification response
        mock_chat.return_value = {'message': {'content': 'analysis'}}
        
        intent = self.client.classify_intent("optimize my portfolio")
        self.assertEqual(intent, 'analysis')

    def test_router_classification_heuristic(self):
        # Test easy heuristics
        self.assertEqual(
            self.router.classify_request("Optimize my portfolio"),
            RequestType.FINANCIAL_ACTION
        )
        self.assertEqual(
            self.router.classify_request("Write a python script"),
            RequestType.CODING
        )

    @patch('core.minicpm_client.ollama.chat')
    def test_router_classification_fallback(self, mock_chat):
        # Mock MiniCPM classification for ambiguous query
        mock_chat.return_value = {'message': {'content': 'chat'}}
        
        req_type = self.router.classify_request("What is the meaning of life?")
        self.assertEqual(req_type, RequestType.CASUAL)

    def test_should_use_minicpm(self):
        self.assertTrue(self.router.should_use_minicpm(RequestType.CASUAL))
        self.assertTrue(self.router.should_use_minicpm(RequestType.FINANCIAL_QUERY))
        self.assertFalse(self.router.should_use_minicpm(RequestType.FINANCIAL_ACTION))
        self.assertFalse(self.router.should_use_minicpm(RequestType.CODING))

if __name__ == '__main__':
    unittest.main()
