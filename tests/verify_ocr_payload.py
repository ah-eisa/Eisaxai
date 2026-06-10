import sys
import os
print("DEBUG: Imports started")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import MagicMock, patch
print("DEBUG: Unittest imported")

# MOCK HEAVY MODULES BEFORE IMPORTING AGENT
sys.modules["core.agents.finance"] = MagicMock()
sys.modules["core.portfolio_manager"] = MagicMock()
sys.modules["yfinance"] = MagicMock()
print("DEBUG: Heavy modules mocked")

# Now import target
try:
    print("DEBUG: Importing GeneralAgent...")
    from core.agents.general import GeneralAgent
    print("DEBUG: GeneralAgent imported")
except Exception as e:
    print(f"DEBUG: Error importing GeneralAgent: {e}")

try:
    print("DEBUG: Importing state...")
    import state
    print("DEBUG: state imported")
except Exception as e:
    print(f"DEBUG: Error importing state: {e}")

# Mock State
state.SYSTEM_PROMPTS = {"assistant": "You are a helpful assistant."}

class TestOCRIntegration(unittest.TestCase):
    
    @patch('core.agents.general.get_client')
    def test_multimodal_payload_construction(self, mock_get_client):
        print("Starting test_multimodal_payload_construction...")
        
        # Setup Mock Client
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        
        # Setup Mock Response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="I see the image."))]
        mock_client_instance.create_completion.return_value = mock_response

        # Initialize Agent
        print("Initializing GeneralAgent...")
        agent = GeneralAgent()
        # Mock the internal logic just in case
        agent._finance_logic = MagicMock()

        # Mock Image Data
        fake_image_content = b"fake_image_bytes"
        import base64
        b64_data = base64.b64encode(fake_image_content).decode("utf-8")
        
        # Mock Context with Active Image File
        context = {
            "active_file_id": "img123",
            "files": [
                {
                    "id": "img123", 
                    "filename": "test.png", 
                    "content_type": "image/png", 
                    "image_data": b64_data
                }
            ],
            "history": []
        }

        # Call think
        print("Calling agent.think()...")
        agent.think("Describe this image", context)

        # Verify calls
        mock_client_instance.create_completion.assert_called_once()
        call_args = mock_client_instance.create_completion.call_args
        messages = call_args.kwargs['messages']
        
        # Check last message (User)
        last_msg = messages[-1]
        self.assertEqual(last_msg['role'], 'user')
        self.assertIsInstance(last_msg['content'], list)
        
        # content should have text and image
        text_part = next((p for p in last_msg['content'] if p['type'] == 'text'), None)
        image_part = next((p for p in last_msg['content'] if p['type'] == 'image_url'), None)
        
        self.assertIsNotNone(text_part)
        self.assertEqual(text_part['text'], "Describe this image")
        
        self.assertIsNotNone(image_part)
        # Expected data URI format
        expected_url = f"data:image/png;base64,{b64_data}"
        self.assertEqual(image_part['image_url']['url'], expected_url)
        
        print("SUCCESS: Multimodal payload constructed correctly.")

if __name__ == "__main__":
    unittest.main()
