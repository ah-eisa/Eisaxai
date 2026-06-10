from __future__ import annotations
from typing import Optional, Dict, Any
from core.intent_classifier import IntentClassifier
from core.logger import StructuredLogger

class SmartRouter:
    """
    Refined routing logic for EISAX.
    Enforces a "Single Speaker" architecture and "Hard-Gated" investment analysis.
    """
    
    def __init__(self, agent_callable):
        self.agent_callable = agent_callable
        self.classifier = IntentClassifier()
        self.logger = StructuredLogger("smart_router")

    def route(self, session_id: str, text: str, settings: Optional[Dict[str, Any]] = None, history: Optional[list] = None) -> Dict[str, Any]:
        """
        Main entry point for the layered routing.
        """
        settings = settings or {}
        text_low = text.lower().strip()
        
        # 2. Detect Intent
        irp = self.classifier.classify(text)
        intent = irp.get("intent")
        confidence = irp.get("confidence", 0)
        
        # 3. Gating Logic
        is_investment = intent in ["PORTFOLIO", "RISK", "STRATEGY"]
        
        # 1. Check for explicit confirmation strings
        is_direct_confirm = text_low == "proceed with comprehensive analysis"
        
        # If in pending state or direct confirm
        if is_direct_confirm or (settings.get("pending_analysis") and any(k in text_low for k in ["yes", "run", "شغل", "نفذ", "ابدأ", "proceed", "ok"])):
            self.logger.info("Investment analysis confirmed by user", session_id=session_id)
            # Route to investment brain
            res = self.agent_callable(text, {**settings, "mode": "investment", "skip_confirmation": True})
            res["layer_used"] = "investment_brain"
            return res

        # 4. DEFAULT: Casual Chat
        # If it seems like investment but not confirmed yet
        if is_investment and not settings.get("analysis_confirmed") and not settings.get("skip_confirmation"):
             self.logger.info("Investment intent detected, asking for confirmation", session_id=session_id, intent=intent)
             return {
                 "type": "chat.reply",
                 "layer_used": "base_llm",
                 "reply": "I've detected a request for investment analysis. Would you like me to run the institutional portfolio tools for you? (Yes/No)",
                 "data": {
                     "suggest_analysis": True,
                     "intent": intent,
                     "confidence": confidence
                 }
             }

        # 5. Normal Chat Fallback
        res = self.agent_callable(text, {**settings, "mode": "assistant"})
        res["layer_used"] = "base_llm"
        return res
