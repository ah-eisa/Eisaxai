import logging
from typing import Any, Dict, Optional
import os
import re
from datetime import datetime
import config
from core.llm import get_client
from core.agents.base import BaseAgent
import state
logger = logging.getLogger(__name__)

class GeneralAgent(BaseAgent):
    """
    The Single Unified Agent "Ahmed".
    Handles EVERYTHING: Chat, Coding, and Investment Analysis.
    """
    def __init__(self):
        super().__init__(name="Ahmed")
        self.client_factory = get_client
        from core.agents.finance import FinancialAgent
        self._finance_logic = FinancialAgent() 

    @staticmethod
    def _normalize_history(history: list) -> list:
        """
        Normalize history from frontend format to LLM API format.
        Frontend sends: {role: 'bot'/'user', text: '...'}
        LLM needs:      {role: 'assistant'/'user', content: '...'}
        """
        normalized = []
        for turn in history:
            role = turn.get("role", "user")
            # Map 'bot' to 'assistant' (LLM API standard)
            if role == "bot":
                role = "assistant"
            # Try 'content' first, fallback to 'text' (frontend uses 'text')
            content = turn.get("content") or turn.get("text", "")
            if content and isinstance(content, str) and content.strip():
                normalized.append({"role": role, "content": content})
        return normalized

    @staticmethod
    def _detect_investment_context(history: list) -> bool:
        """
        Check if the recent conversation is in an investment/CIO context.
        If the last AI response contained financial analysis content,
        follow-up messages should stay in CIO mode even without finance keywords.
        """
        # Look at the last 4 messages for investment context
        recent = history[-4:] if len(history) >= 4 else history
        investment_signals = [
            "portfolio", "allocation", "drawdown", "capital", "time horizon",
            "liquidity", "holdings", "risk", "aggressive", "conservative",
            "returns", "sharpe", "investment", "jurisdiction", "tax",
            "downside", "upside", "suitability", "asset class"
        ]
        for turn in recent:
            content = (turn.get("content") or turn.get("text", "")).lower()
            role = turn.get("role", "")
            # If the AI (bot/assistant) recently discussed investment topics
            if role in ("bot", "assistant"):
                matches = sum(1 for sig in investment_signals if sig in content)
                if matches >= 3:  # Strong signal: 3+ investment terms in one response
                    return True
        return False

    def think(self, 
              message: str, 
              context: Dict[str, Any], 
              settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        
        s = settings or {}
        from core.intent_classifier import IntentClassifier
        from core.ticker_resolver import has_arabic_stock_context
        mem = context.get("memory", {})
        sid = context.get("session_id", "default")

        # Normalize history FIRST (fix bot→assistant, text→content)
        raw_history = context.get("history", [])
        history = self._normalize_history(raw_history)

        # ── Single intent classification pass ──────────────────────────────
        # detect_primary_intent is rule-based (fast, no LLM).
        # classify_intent_hybrid calls it internally then does the same Arabic checks
        # we do below — so calling both was running the same logic twice.
        intent = IntentClassifier.detect_primary_intent(message, mem)

        # Map fine-grained intent → broad CIO vs general mode
        _CIO_INTENTS = {
            "portfolio_optimize", "portfolio_report", "portfolio_metrics",
            "portfolio_edit", "irr_calc", "var_calc", "investment_query",
        }
        if intent in _CIO_INTENTS:
            intent_type = "investment"
        elif intent in ("report_export",):
            intent_type = "general"
        elif IntentClassifier.has_arabic_financial_intent(message) or has_arabic_stock_context(message):
            intent_type = "investment"
        else:
            intent_type = "general"

        # STICKINESS: If intent is 'general' but conversation context is investment,
        # keep CIO mode for follow-up answers (e.g., "5 years, 3M$, no, 25%...")
        if intent_type == "general" and self._detect_investment_context(raw_history):
            intent_type = "investment"

        # Hard financial intents that REQUIRE the FinancialAgent engine
        financial_intents = {
            "portfolio_optimize", "portfolio_report", "report_export",
            "analyze", "technical_analysis", "risk_analysis",
            "forecast", "simulate", "project",
            "trade_execution",
        }
        if intent in financial_intents or (intent and any(k in intent for k in ("analyze", "forecast", "trade"))):
            return self._finance_logic.think(message, context, settings)

        model = s.get("model") or os.getenv("MODEL_NAME", config.DEFAULT_MODEL)
        temperature = s.get("temperature", 0.7)
        max_tokens = s.get("max_tokens", 6000)

        # Select System Prompt based on Intent
        if intent_type == "investment":
            system_prompt = state.SYSTEM_PROMPTS.get("cio", "")
        else:
            system_prompt = state.SYSTEM_PROMPTS.get("assistant", "")

        # Standard Chat — prepare messages with normalized history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Inject Memory Context
        if mem.get("tickers"):
            portfolio_context = (
                f"\n[ACTIVE PORTFOLIO MEMORY]\n"
                f"Tickers: {', '.join(mem['tickers'])}\n"
                f"Risk: {mem.get('risk', 'Unknown')}\n"
                f"Weights: {mem.get('weights', 'N/A')}\n"
            )
            messages[0]["content"] += portfolio_context

        # Inject file content if an active file has extracted text
        active_file_id = context.get("active_file_id")
        files = context.get("files", [])
        active_file = next((f for f in files if f.get("id") == active_file_id), None)

        if active_file and active_file.get("text") and not active_file["text"].startswith("[IMAGE:"):
            file_text = active_file["text"]
            max_chars = s.get("max_file_chars", 12000)
            if len(file_text) > max_chars:
                file_text = file_text[:max_chars] + f"\n\n... [truncated — {len(active_file['text'])} total chars]"
            
            file_context = (
                f"\n[ATTACHED FILE: {active_file.get('filename', 'unknown')}]\n"
                f"The user has uploaded this file. Use its content to answer their questions.\n"
                f"---\n{file_text}\n---\n"
            )
            messages.append({"role": "system", "content": file_context})

        # Add normalized conversation history (last 10 turns)
        for turn in history[-10:]: 
            messages.append({"role": turn["role"], "content": turn["content"]})
            
        # Add Current User Message (Check for Image Payload)
        image_file = next((f for f in files if f.get("id") == active_file_id and f.get("image_data")), None)

        if image_file:
            user_content = [
                {"type": "text", "text": message},
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:{image_file['content_type']};base64,{image_file['image_data']}"
                    }
                }
            ]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": message})

        try:
            client = self.client_factory()
            response = client.create_completion(
                model=model,
                temperature=temperature,
                messages=messages,
            )
            reply_content = response.choices[0].message.content
            
            # --- Artifact Binding ---
            finance_indicators = ["|", "1.", "- ", "الفئة", "التوزيع", "Allocation", "Portfolio", "Asset", "Expected Return", "Sharpe Ratio", "Guide", "Strategy"]
            is_structured = any(x in reply_content for x in finance_indicators)
            is_long = len(reply_content) > 150
            
            if intent_type == "investment" or is_structured:
                if is_long or is_structured:
                    title_match = re.search(r"^(?:#{1,3})\s*(.*)$", reply_content, re.MULTILINE)
                    ext_title = title_match.group(1).strip() if title_match else f"EISAX Investment Report {datetime.now().strftime('%Y-%m-%d')}"
                    
                    logger.debug(f"[Ahmed] BINDING ARTIFACT: {ext_title} ({len(reply_content)} chars)")
                    state.set_artifact(sid, {
                        "type": "investment_strategy",
                        "title": ext_title,
                        "content": reply_content,
                        "source": "general_agent_chat",
                        "exportable": True,
                        "timestamp": datetime.now()
                    })
            
            return {
                "type": "chat.reply",
                "reply": reply_content
            }
            
        except Exception as e:
            return {"type": "error", "reply": f"Ahmed (Agent) Error: {e}"}
