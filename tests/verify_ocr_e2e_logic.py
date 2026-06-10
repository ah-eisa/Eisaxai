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
sys.modules["pandas"] = MagicMock() # Mock pandas just in case

import state
from core.orchestrator import MultiAgentOrchestrator

class TestOCRE2E(unittest.TestCase):
    
    @patch('core.agents.general.get_client')
    def test_orchestrator_passes_file_context(self, mock_get_client):
        """Verify Orchestrator passes state.uploaded_files to Agent."""
        
        # Setup Mock Client
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        # Mock Response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="I see the image."))]
        mock_client_instance.create_completion.return_value = mock_response

        # Setup State
        fake_file_id = "img_test_001"
        fake_image_data = "base64data"
        state.uploaded_files = [{
            "id": fake_file_id,
            "filename": "test.png",
            "content_type": "image/png",
            "image_data": fake_image_data
        }]
        state.active_file_id = fake_file_id
        
        # Initialize Orchestrator
        orchestrator = MultiAgentOrchestrator()
        
        # Call Think
        print("Calling orchestrator.think()...")
        orchestrator.think("Describe this image", settings={"active_file_id": fake_file_id})
        
        # Verify GeneralAgent.think was called with context containing the file
        # Since we can't easily spy on the internal agent instance without patching,
        # we check the LLM client call which proves the data made it through.
        
        mock_client_instance.create_completion.assert_called_once()
        call_args = mock_client_instance.create_completion.call_args
        messages = call_args.kwargs['messages']
        
        last_msg = messages[-1]
        print(f"Last message content type: {type(last_msg['content'])}")
        
        self.assertIsInstance(last_msg['content'], list)
        image_part = next((p for p in last_msg['content'] if p['type'] == 'image_url'), None)
        
        self.assertIsNotNone(image_part)
        expected_url = f"data:image/png;base64,{fake_image_data}"
        self.assertEqual(image_part['image_url']['url'], expected_url)
        print("SUCCESS: Orchestrator passed file context correctly.")

if __name__ == "__main__":
    unittest.main()
