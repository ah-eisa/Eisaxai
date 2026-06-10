"""
core/tool_agent.py
───────────────────
EisaX ToolAgent — a real agentic loop using DeepSeek function calling.

Unlike the old monolithic approach (fetch everything → one giant prompt),
the ToolAgent lets DeepSeek DECIDE what data to fetch:

  User → Agent thinks → calls tool(s) → gets data → generates answer

This produces more accurate, focused, and faster analyses.

Usage
─────
    from core.tool_agent import ToolAgent

    agent = ToolAgent(user_id="user_123")
    reply = await agent.run("Analyze EMAAR.DU")

    # Streaming
    async for chunk in agent.stream("حلل داماك"):
        print(chunk["text"], end="", flush=True)
"""

import os
import json
import logging
import asyncio
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

# Max tool-call iterations per request (prevents infinite loops)
MAX_ITERATIONS = 6

# DeepSeek system prompt for the tool agent
_SYSTEM_PROMPT = """You are EisaX — an institutional-grade AI investment intelligence system built by Ahmed Eisa.
You are a senior investment advisor with 20+ years of experience. You are calm, direct, and data-driven.

You have access to real-time financial tools. Use them strategically:
- For stock analysis: ALWAYS call get_stock_price first, then get_fundamentals, then get_news
- For technical analysis: call get_technical_analysis
- For portfolio: call get_user_profile first, then calculate_portfolio_metrics
- For general questions: answer directly without tools

RESPONSE RULES:
- Arabic message → reply in Arabic
- English message → reply in English
- Be direct and numbers-first
- Never guess prices — use tool data only
- Never say "I cannot" — use tools to get what you need
- Format with markdown (tables, bold key numbers)
- End stock analysis with a clear BUY / HOLD / SELL / REDUCE verdict"""


class ToolAgent:
    """
    Agentic loop: DeepSeek LLM + tool calling.

    The agent runs up to MAX_ITERATIONS rounds of:
      1. Send messages (including previous tool results) to DeepSeek
      2. If DeepSeek returns a tool call → execute it, append result, repeat
      3. If DeepSeek returns a final text → return it
    """

    def __init__(self, user_id: str = ""):
        self.user_id   = user_id
        self.ds_key    = os.getenv("DEEPSEEK_API_KEY", "")
        self._messages: list[dict] = []

    # ── Public: non-streaming ─────────────────────────────────────────────────

    async def run(self, message: str, *, max_tokens: int = 4500) -> str:
        """Run the full agent loop and return the final reply text."""
        full = []
        async for chunk in self.stream(message, max_tokens=max_tokens):
            if chunk.get("type") == "token":
                full.append(chunk["text"])
        return "".join(full)

    # ── Public: streaming ─────────────────────────────────────────────────────

    async def stream(
        self,
        message: str,
        *,
        max_tokens: int = 4500,
    ) -> AsyncGenerator[dict, None]:
        """
        Async generator — yields event dicts:
          {"type": "status",     "text": "..."}   — tool being called
          {"type": "tool_result","tool": "...", "result": {...}}
          {"type": "token",      "text": "..."}   — LLM content
          {"type": "done"}
        """
        from core.agent_tools import TOOLS, execute_tool

        if not self.ds_key:
            yield {"type": "error", "text": "DEEPSEEK_API_KEY not set"}
            return

        # Build initial messages
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": message},
        ]

        import httpx

        for iteration in range(MAX_ITERATIONS):
            # ── Call DeepSeek (non-streaming first pass to get tool calls) ───
            payload = {
                "model":      "deepseek-v4-flash",
                "messages":   messages,
                "tools":      TOOLS,
                "tool_choice":"auto",
                "max_tokens": max_tokens,
                "temperature":0.3,
            }

            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.ds_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )

                if resp.status_code != 200:
                    yield {"type": "error", "text": f"DeepSeek HTTP {resp.status_code}"}
                    return

                data      = resp.json()
                choice    = data["choices"][0]
                finish    = choice.get("finish_reason", "")
                assistant = choice["message"]

                # Append assistant message to history
                messages.append(assistant)

                # ── Final text answer ─────────────────────────────────────────
                if finish == "stop" or not assistant.get("tool_calls"):
                    content = (assistant.get("content") or "").strip()
                    if content:
                        # Stream in chunks for smooth UX
                        chunk_size = 32
                        for i in range(0, len(content), chunk_size):
                            yield {"type": "token", "text": content[i:i+chunk_size]}
                            await asyncio.sleep(0)
                    yield {"type": "done"}
                    return

                # ── Tool calls ────────────────────────────────────────────────
                tool_calls = assistant.get("tool_calls", [])
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"].get("arguments", "{}"))
                    except Exception:
                        fn_args = {}

                    yield {
                        "type": "status",
                        "text": _tool_status_msg(fn_name, fn_args),
                    }
                    await asyncio.sleep(0)

                    # Execute synchronously in executor (tools are blocking I/O)
                    loop   = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda _n=fn_name, _a=fn_args: execute_tool(_n, _a, user_id=self.user_id),
                    )

                    yield {"type": "tool_result", "tool": fn_name, "result": result}

                    # Append tool result to messages
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc["id"],
                        "content":      json.dumps(result, ensure_ascii=False),
                    })

            except Exception as e:
                logger.error("[ToolAgent] iteration %d error: %s", iteration, e)
                yield {"type": "error", "text": str(e)}
                return

        # Exceeded max iterations
        yield {"type": "error", "text": "Agent exceeded maximum iterations"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tool_status_msg(tool_name: str, args: dict) -> str:
    """Human-readable status message shown while a tool runs."""
    ticker = args.get("ticker", "")
    query  = args.get("query", "")
    msgs = {
        "get_stock_price":            f"📡 جلب سعر {ticker}...",
        "get_fundamentals":           f"📊 تحليل أساسيات {ticker}...",
        "get_news":                   f"📰 جلب أخبار {query or ticker}...",
        "get_user_profile":           "👤 قراءة ملف المستثمر...",
        "calculate_portfolio_metrics": "🧮 حساب مقاييس المحفظة...",
        "get_technical_analysis":     f"📈 تحليل تقني {ticker}...",
    }
    return msgs.get(tool_name, f"⚙️ {tool_name}...")
