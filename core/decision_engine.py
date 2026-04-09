from __future__ import annotations
import uuid
import datetime
from datetime import timezone
from typing import Dict, Any, List

class DecisionEngine:
    """
    Institutional decision logic to validate and refine intents.
    Enforces policy compliance and audit trail generation.
    """
    
    SYSTEM_CONFIG = {
        "policy_version": "v1.0.0",
        "model_version": "deepseek-v3.1",
        "prompt_version": "inst-p1"
    }
    
    def evaluate(self, 
                 intent_obj: Dict[str, Any], 
                 session_context: Dict[str, Any], 
                 policy_rules: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate user intent against institutional rules.
        """
        decision_id = str(uuid.uuid4())
        intent = intent_obj.get("intent", "CHAT")
        confidence = intent_obj.get("confidence", 0.0)
        target_text = str(session_context.get("last_text", "")).lower()
        
        # Default assume pass
        result = {
            "decision_id": decision_id,
            "decision": "APPROVED",
            "confidence": confidence,
            "rationale": ["Intent is valid and within normal thresholds."],
            "required_inputs": [],
            "risk_flags": [],
            "audit": {
                **self.SYSTEM_CONFIG,
                "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
                "intent_classified": intent
            }
        }

        # --- Policy Enforcement Logic ---
        
        # 1. REPORT Intent Specific Logic
        if intent == "REPORT":
            # If requesting market report, ensure we have time window
            is_market = "market" in target_text or "الأسواق" in target_text
            if is_market and not any(w in target_text for w in ["today", "week", "month"]):
                 # Default to 'Today' implicitly if missing, no need to block unless strict
                 # BUT user requested "Ask ONE question only: time window"
                 result["decision"] = "NEEDS_INFO"
                 result["rationale"] = ["Time window required for market report."]
                 result["required_inputs"].append("time_window")
                 return result

        # 2. Reject low confidence on critical intents
        if intent == "PORTFOLIO" and confidence < 0.6:
            result["decision"] = "NEEDS_INFO"
            result["rationale"] = ["User intent is ambiguous regarding portfolio action."]
            result["required_inputs"].append("confirm_action")
        
        # Streamline Portfolio Input Request
        if intent == "PORTFOLIO" and "analyze" in target_text and not session_context.get("portfolio_data"):
             # If user asks to analyze portfolio but we have no data, ask for it in ONE specific way
             # Check if we already have data? Mock check session_context.get("holdings")
             # For now, if decision is to ask for holdings, ensure specifically:
             if "holdings" not in target_text and "weights" not in target_text:
                 result["decision"] = "NEEDS_INFO"
                 result["rationale"] = ["Portfolio data required for analysis."]
                 result["required_inputs"].append("holdings_weights_or_values")
                 return result
            
        # 3. Check explicitly banned actions
        if "bitconnect" in target_text or "ponzi" in target_text:
            result["decision"] = "REJECTED"
            result["risk_flags"].append("fraud_risk")
            result["rationale"] = ["Entity flagged as high-risk/fraudulent."]
            return result
            
        if "guaranteed" in target_text and ("50%" in target_text or "monthly" in target_text):
             result["decision"] = "REJECTED"
             result["risk_flags"].append("compliance_violation")
             result["rationale"] = ["Guaranteed high returns are flagged as unrealistic/scam."]
             return result

        if "all in" in target_text or "100%" in target_text:
             result["risk_flags"].append("concentration_risk")
             result["rationale"].append("Request implies excessive concentration (>30%).")
             result["decision"] = "NEEDS_INFO"
             result["required_inputs"].append("risk_acknowledgement")

        # 4. Compliance checks for specific modes
        mode = session_context.get("mode", "assistant")
        if mode == "code_review" and intent == "PORTFOLIO":
             result["decision"] = "REJECTED"
             result["rationale"] = ["Portfolio actions not permitted in Code Review mode."]
             return result

        return result
