"""
EisaX Agent Loop
────────────────
A real agentic loop: Plan → Tool Call → Observe → Respond.

The agent (DeepSeek) decides which tools to call, calls them via the
TOOLS_REGISTRY, incorporates the results, then produces a final answer.

Max iterations: 4 (prevents infinite loops)
Model: deepseek-chat with function calling

Usage:
    from core.agent_loop import run_agent
    result = await run_agent(user_id, message, user_ctx)
    # result = {"reply": "...", "tools_used": [...], "iterations": N}
"""
import os
import json
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)

MAX_ITER       = 4      # max tool-call rounds
MAX_TOKENS_OUT = 4000   # max response tokens
DEEPSEEK_URL   = "https://api.deepseek.com/v1/chat/completions"


async def run_agent(
    message:   str,
    user_ctx:  dict = None,
    user_id:   str  = "anonymous",
    session_id:str  = None,
) -> dict:
    """
    Main agent loop entry point.

    Returns:
        {
          "reply":       str,
          "tools_used":  [{"tool": str, "args": dict, "result": dict}],
          "iterations":  int,
          "model":       str,
        }
    """
    from core.tools import TOOLS_REGISTRY, TOOLS_SCHEMA
    from core.memory_manager import format_ctx_for_prompt

    # Load .env if running outside uvicorn context
    try:
        from dotenv import load_dotenv as _lde
        _lde()
    except Exception:
        pass

    ds_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not ds_key:
        return {"reply": "DeepSeek API key not configured.", "tools_used": [], "iterations": 0}

    # ── Build system prompt with user context ─────────────────────────────────
    from datetime import datetime as _dt
    today = _dt.now().strftime("%B %d, %Y")

    user_block = ""
    if user_ctx:
        user_block = format_ctx_for_prompt(user_ctx) or ""

    system = (
        f"You are EisaX — an institutional-grade AI investment intelligence system "
        f"built by Ahmed Eisa. Today: {today}.\n\n"
        "You have access to real-time financial tools. Use them when needed:\n"
        "- Always call get_price before quoting any live price\n"
        "- Always call get_fundamentals for valuation questions\n"
        "- Call search_news for recent events affecting a stock\n"
        "- Call screen_market when asked for stock recommendations or screening\n"
        "- Call calculate_portfolio for portfolio analysis requests\n\n"
        "You NEVER invent prices or fabricate data. If a tool fails, say so.\n"
        "Reply in the same language as the user (Arabic or English).\n"
        "Be direct, professional, numbers-first.\n"
        f"{user_block}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": message},
    ]

    tools_used = []
    iterations = 0

    import httpx
    import asyncio as _aio

    async def _call_deepseek(client: httpx.AsyncClient, payload: dict, retries: int = 2) -> dict:
        """Call DeepSeek with automatic retry on transient errors."""
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = await client.post(
                    DEEPSEEK_URL,
                    headers={"Authorization": f"Bearer {ds_key}",
                             "Content-Type": "application/json"},
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_err = e
                if attempt < retries:
                    await _aio.sleep(1.5 * (attempt + 1))  # 1.5s, 3s back-off
                    logger.warning("[AgentLoop] Retry %d/%d after transient error: %s",
                                   attempt + 1, retries, e)
            except Exception as e:
                last_err = e
                break  # Non-transient error — don't retry
        raise last_err or RuntimeError("DeepSeek unreachable")

    async with httpx.AsyncClient(timeout=60) as client:
        for _ in range(MAX_ITER):
            iterations += 1
            # ── Call DeepSeek with tools ──────────────────────────────────────
            payload = {
                "model":       "deepseek-chat",
                "messages":    messages,
                "tools":       TOOLS_SCHEMA,
                "tool_choice": "auto",
                "max_tokens":  MAX_TOKENS_OUT,
                "temperature": 0,
            }
            try:
                data = await _call_deepseek(client, payload)
            except Exception as e:
                logger.error("[AgentLoop] DeepSeek failed after retries: %s", e)
                return {
                    "reply": "عذراً، خدمة التحليل غير متاحة حالياً. حاول مرة أخرى.",
                    "tools_used": tools_used,
                    "iterations": iterations,
                    "model": "deepseek-chat",
                }

            choice  = data["choices"][0]
            msg     = choice["message"]
            finish  = choice.get("finish_reason", "")

            # ── No tool calls → final answer ─────────────────────────────────
            if finish == "stop" or not msg.get("tool_calls"):
                final_reply = (msg.get("content") or "").strip()
                logger.info("[AgentLoop] Done in %d iteration(s), %d tools used",
                            iterations, len(tools_used))
                return {
                    "reply":       final_reply,
                    "tools_used":  tools_used,
                    "iterations":  iterations,
                    "model":       "deepseek-chat (agent)",
                }

            # ── Process tool calls ────────────────────────────────────────────
            messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg["tool_calls"]})

            for tc in msg["tool_calls"]:
                tool_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                # Execute tool
                tool_fn = TOOLS_REGISTRY.get(tool_name)
                if tool_fn:
                    try:
                        # Run sync tools in executor to avoid blocking
                        loop = asyncio.get_running_loop()
                        import functools
                        tool_result = await loop.run_in_executor(
                            None, functools.partial(tool_fn, **args)
                        )
                    except Exception as te:
                        tool_result = {"error": str(te)}
                        logger.warning("[AgentLoop] Tool %s failed: %s", tool_name, te)
                else:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}

                logger.info("[AgentLoop] Tool: %s(%s) → %s",
                            tool_name, list(args.keys()),
                            "OK" if not tool_result.get("error") else tool_result["error"])

                tools_used.append({
                    "tool":   tool_name,
                    "args":   args,
                    "result": tool_result,
                })

                # Feed result back to conversation
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      json.dumps(tool_result, ensure_ascii=False),
                })

    # Exceeded max iterations
    logger.warning("[AgentLoop] Max iterations (%d) reached", MAX_ITER)
    return {
        "reply":      "⚠️ تعذّر إتمام التحليل في الوقت المحدد — حاول مرة أخرى.",
        "tools_used": tools_used,
        "iterations": iterations,
        "model":      "deepseek-chat (agent)",
    }
