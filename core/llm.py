import logging
import requests
import config
logger = logging.getLogger(__name__)

class LLMClient:
    """
    Lightweight HTTP client for Chat Completions (OpenAI/DeepSeek/Generic).
    """
    def __init__(self, api_key: str = None, base_url: str = None):
        # Determine provider based on config or defaults
        # If MODEL_NAME implies OpenAI (gpt-*), prefer OpenAI Settings
        default_model = config.DEFAULT_MODEL.lower()
        is_openai = "gpt" in default_model or "o1" in default_model
        
        if is_openai:
            self.api_key = api_key or config.OPENAI_API_KEY
            self.base_url = (base_url or "https://api.openai.com").rstrip("/")
            if not self.api_key:
                 # Fallback: maybe they put OpenAI key in DEEPSEEK var?
                 self.api_key = config.DEEPSEEK_API_KEY
        else:
            self.api_key = api_key or config.DEEPSEEK_API_KEY
            self.base_url = (base_url or config.DEEPSEEK_BASE_URL).rstrip("/")

        if not self.api_key:
            # If still missing, check if we can fallback to the other key
            if config.OPENAI_API_KEY:
                self.api_key = config.OPENAI_API_KEY
                self.base_url = "https://api.openai.com"
            else:
                # We can't proceed without a key
                raise ValueError(f"API Key is missing. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY in .env")

    def create_completion(self, model: str, messages: list[dict], temperature: float = 0.7, max_tokens: int = 6000):
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=180)
            
            if resp.status_code != 200:
                error_msg = f"API Error ({resp.status_code}): {resp.text}"
                logger.error(f"[ERROR] {error_msg}")
                raise Exception(error_msg)
            
            data = resp.json()
            
            # AttributeDict wrapper for OpenAI-like access (response.choices[0].message.content)
            class AttrDict(dict):
                def __getattr__(self, name):
                    if name in self: return self[name]
                    raise AttributeError(name)

            return AttrDict({
                "choices": [
                    AttrDict({
                        "message": AttrDict(data["choices"][0]["message"])
                    })
                ]
            })
            
        except Exception as e:
            logger.error(f"[LLM CLIENT ERROR] {str(e)}")
            raise e

# Singleton instance for lazy loading
_client = None

def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
