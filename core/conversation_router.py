from typing import Optional
from enum import Enum
import re
from core.minicpm_client import minicpm_client

class RequestType(Enum):
    CASUAL = "casual"
    FINANCIAL_QUERY = "financial_query"
    FINANCIAL_ACTION = "financial_action"
    CODING = "coding"
    VISION = "vision"

class ConversationRouter:
    """
    Intelligent router to dispatch requests between:
    1. Local MiniCPM (Fast/Free) -> Casual chat, simple queries, vision.
    2. Cloud DeepSeek/Engine (Powerful/Costly) -> Complex analysis, portfolio actions, report generation.
    """

    @staticmethod
    def classify_request(message: str, image_path: Optional[str] = None) -> RequestType:
        """
        Determines the request type based on image presence, keywords, and MiniCPM classification.
        """
        if image_path:
            return RequestType.VISION

        msg = message.lower().strip()

        # heuristic 1: Strong keywords for Financial Action (Engine required)
        action_keywords = [
            "optimize", "portfolio", "rebalance", "backtest", "forecast", "report", "allocation",
            "risk analysis", "stress test", "generate", "create strategy"
        ]
        if any(w in msg for w in action_keywords):
            return RequestType.FINANCIAL_ACTION

        # heuristic 2: Coding (Engine/Tools required)
        coding_keywords = ["python", "script", "code", "debug", "function", "class"]
        if any(w in msg for w in coding_keywords):
            return RequestType.CODING

        # heuristic 3: Use MiniCPM for rapid semantic classification
        # Since heuristics didn't catch obvious heavy tasks, let's confirm intent.
        intent = minicpm_client.classify_intent(message)
        
        if intent == "analysis":
            return RequestType.FINANCIAL_ACTION
        elif intent == "coding":
            return RequestType.CODING
        
        # Default fallback
        return RequestType.CASUAL

    @staticmethod
    def should_use_minicpm(request_type: RequestType) -> bool:
        """
        Returns True if the request should be handled by MiniCPM locally.
        """
        return request_type in [RequestType.CASUAL, RequestType.VISION, RequestType.FINANCIAL_QUERY]

# Singleton instance
router = ConversationRouter()
