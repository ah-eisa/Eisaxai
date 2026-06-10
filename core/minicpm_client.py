import ollama
from typing import Optional

class MiniCPMClient:
    """
    Client for interacting with local MiniCPM-V 2.6 model via Ollama.
    Handles casual chat, intent classification, and vision tasks.
    """
    def __init__(self, model_name: str = "openbmb/minicpm-v2.6"):
        self.model = model_name

    def chat(self, message: str) -> str:
        """
        Handles casual conversation.
        """
        try:
            system_prompt = (
                "You are Eisa, a warm and intelligent AI assistant. "
                "You help with anything — questions, analysis, creative tasks, conversation. "
                "Be natural and conversational. Talk like a smart, thoughtful friend. "
                "Vary your tone to match the user's energy. "
                "Do NOT identify as ChatGPT, OpenAI, or any other model. "
                "If asked who you are, say you are Eisa."
            )
            
            response = ollama.chat(model=self.model, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': message}
            ])
            return response['message']['content']
        except Exception as e:
            return f"⚠️ Error communicating with MiniCPM: {e}"

    def analyze_image(self, prompt: str, image_path: str) -> str:
        """
        Analyzes an image using MiniCPM-V.
        """
        try:
            response = ollama.chat(model=self.model, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }
            ])
            return response['message']['content']
        except Exception as e:
            return f"⚠️ Error analyzing image with MiniCPM: {e}"

    def classify_intent(self, message: str) -> str:
        """
        Classifies user intent into 'chat', 'coding', or 'analysis'.
        """
        try:
            prompt = (
                f"Classify the following user message into ONE of these categories: "
                f"'chat' (casual, greetings, general questions), "
                f"'coding' (writing code, snippets), "
                f"'analysis' (complex financial analysis, portfolio optimization, reports). "
                f"Return ONLY the category name in lowercase.\n\n"
                f"Message: {message}"
            )
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt}
            ])
            intent = response['message']['content'].strip().lower()
            # Basic validation
            if "analysis" in intent: return "analysis"
            if "coding" in intent: return "coding"
            return "chat"
        except Exception as e:
            print(f"MiniCPM Classification Error: {e}")
            return "chat"  # Default to simple chat if detailed classification fails

# Singleton instance
minicpm_client = MiniCPMClient()
