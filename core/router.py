from __future__ import annotations
from typing import Dict, Any, Optional

from core.logger import StructuredLogger
from core.session_manager import SessionManager
from core.intent_classifier import IntentClassifier
from core.orchestrator import _orchestrator

class Router:
    """
    Simplified Central dispatch for Multi-Agent System.
    Delegates routing and execution logic to MultiAgentOrchestrator.
    """
    
    def __init__(self, 
                 agent_callable=None,
                 orchestrator: Any = None,
                 session_manager: Optional[SessionManager] = None):
        self.logger = StructuredLogger("router")
        self.session_manager = session_manager or SessionManager()
        self.orchestrator = orchestrator or _orchestrator
        self.classifier = IntentClassifier()
    
    def handle_request(self, 
                       session_id: str, 
                       text: str, 
                       meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point for request handling.
        """
        # 1. Log Request Entry
        meta = meta or {}
        req_id = meta.get("request_id")
        self.logger.info("Request received", session_id=session_id, request_id=req_id, text_snippet=text[:50])
        
        # 2. Get/Validate Session State
        try:
            state = self.session_manager.get_state(session_id)
        except Exception as e:
             self.logger.error("Session fetch failed", error=str(e))
             state = {}  # Fallback
             
        # 3. Classify Intent (Optional, mostly for logging or meta-tagging now)
        # Use static method from IntentClassifier
        raw_intent = IntentClassifier.detect_primary_intent(text) or "general"
        intent_result = {"intent": raw_intent.upper(), "confidence": 0.8}
        
        self.logger.info("Intent classified", intent=intent_result["intent"], confidence=intent_result["confidence"])

        # 4. Route to Multi-Agent Orchestrator
        # We no longer gate based on "Institutional" vs "General" here. 
        # The Orchestrator handles agent dispatch.
        
        meta["router_intent"] = intent_result
        self.session_manager.update_state(session_id, {"last_intent": intent_result["intent"]})

        try:
            response = self.orchestrator.think(
                text,
                meta.get("settings"),
                meta.get("history")
            )
            
            # 5. Log Final Outcome
            self.logger.info("Request processed successfully", response_type=response.get("type"))
            
            # Post-process reply to remove internal debug markers if any
            if "reply" in response:
                response["reply"] = self._normalize_reply(response["reply"])
                
            return response

        except Exception as e:
            self.logger.error("Execution pipeline failed", error=str(e))
            return {
                "type": "error", 
                "reply": "An internal error occurred processing your request.",
                "data": {"error": str(e)}
            }
            
    def _normalize_reply(self, text: str) -> str:
        """
        Normalize reply by:
        1. Removing internal debug markers
        2. Removing institutional headers (EXECUTIVE SUMMARY, etc.)
        3. Removing investment strategy headers
        4. Limiting to 9 lines maximum
        """
        if not text: return ""

        lines = text.split('\n')

        # Headers to remove entirely
        headers_to_remove = {
            'EXECUTIVE SUMMARY',
            'Main Analysis',
            'Investment Strategy Report',
            'Investment Recommendation',
        }

        # Filter out debug lines and headers
        cleaned = []
        for line in lines:
            # Skip internal debug lines
            if "internal_command" in line.lower():
                continue
            # Skip header lines (exact match, case-insensitive)
            if line.strip() in headers_to_remove or any(h in line for h in headers_to_remove):
                continue
            # Skip markdown header lines that are headers
            if line.strip().startswith('##') and any(h.lower() in line.lower() for h in headers_to_remove):
                continue
            cleaned.append(line)

        # Join and limit to 9 lines
        result = "\n".join(cleaned).strip()
        result_lines = result.split('\n')

        if len(result_lines) > 9:
            result_lines = result_lines[:9]

        return "\n".join(result_lines)
