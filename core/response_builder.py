from __future__ import annotations
from typing import Dict, Any, List, Optional
import uuid
import datetime

from core.style_card import StyleCard

class ResponseBuilder:
    """
    Constructs the final user-facing reply.
    Inputs: intent, session_state, decision_constraints, style_card.
    Output: short, smart human-like reply.
    """
    
    def __init__(self):
        self.style = StyleCard()

    def build_response(self, 
                       intent_result: Dict[str, Any],
                       decision: Dict[str, Any], 
                       raw_reply: Optional[str] = None,
                       tool_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build the final response dictionary.
        Uses StyleCard to format or generate text based on decision.
        """
        
        # 1. Handle REJECTED Decisions
        if decision["decision"] == "REJECTED":
            # Correctly handle rejected case using raw strings if necessary but preferably formatted
            failed_checks = decision.get("risk_flags", []) or decision.get("rationale", ["policy restriction"])
            text = self._format_rejection(failed_checks)
            return self._wrap(text, decision)

        # 2. Handle NEEDS_INFO Decisions
        if decision["decision"] == "NEEDS_INFO":
            missing_info = decision.get("required_inputs", ["confirmation"])
            rationale = decision.get("rationale", ["Needs clarification"])
            text = self._format_needs_info(missing_info, rationale)
            return self._wrap(text, decision)
            
        # 3. Handle APPROVED / Executed Results
        final_text = raw_reply or "Request processed."
        
        # Apply strict formatting and truncation
        final_text = self._enforce_strict_style(final_text)

        return self._wrap(final_text, decision, tool_outputs)
        
    def _enforce_strict_style(self, text: str) -> str:
        """
        Strict cleanup of response formatting:
        1. Remove institutional headers (EXECUTIVE SUMMARY, etc.)
        2. Remove markdown analysis headers
        3. Remove investment strategy markers
        4. Filter empty lines intelligently
        5. Truncate to maximum 9 lines
        """
        lines = text.split('\n')

        # Headers/sections to completely skip
        headers_to_skip = {
            'EXECUTIVE SUMMARY',
            'Main Analysis',
            'Investment Strategy Report',
            'Investment Recommendation',
        }

        filtered = []
        for line in lines:
            # Skip lines that contain institutional headers
            if any(h in line for h in headers_to_skip):
                continue
            # Skip markdown analysis headers
            if '##' in line and any(h.lower() in line.lower() for h in headers_to_skip):
                continue
            # Keep normal content
            filtered.append(line)

        # Remove excessive empty lines while preserving structure
        result_lines = []
        prev_empty = False
        for line in filtered:
            is_empty = not line.strip()
            if is_empty:
                if not prev_empty:
                    result_lines.append(line)
                prev_empty = True
            else:
                result_lines.append(line)
                prev_empty = False

        # Truncate to 9 lines maximum
        if len(result_lines) > 9:
            result_lines = result_lines[:9]

        result = "\n".join(result_lines).strip()
        return result

    def _format_rejection(self, reasons: List[str]) -> str:
        """Construct short rejection based on internal flags."""
        base = "Request declined due to policy constraints."
        bullets = "\n".join([f"- {r}" for r in reasons[:3]]) # Max 3 reasons
        return f"{base}\n{bullets}\n\nRecommendation: Check compliance guidelines."

    def _format_needs_info(self, fields: List[str], rationale: List[str]) -> str:
        """Construct short clarification request."""
        # Use specific rationale as base
        base = rationale[0] if rationale else "Clarification required."
        
        # Ask ONE specific question
        # For REPORT intent, default to asking about time window if generic
        field = fields[0] if fields else "details"
        if field == "time_window":
             question = "Please specify time window: Today, Week, or Month?"
        elif field == "holdings_weights_or_values":
             question = "Send holdings as weights (%) or values—your choice."
        else:
             question = f"Please confirm: {field}?"
             
        return f"{base}\n\n{question}"

    def _wrap(self, text: str, decision: Dict[str, Any], extra_data: Any = None) -> Dict[str, Any]:
        """Standard response envelope."""
        return {
            "type": "chat.reply",
            "reply": text,
            "data": {
                "decision": decision,
                "tool_outputs": extra_data,
                "confidence": decision.get("confidence", 1.0)
            }
        }
