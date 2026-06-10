"""
core/streaming.py
─────────────────
Streaming token generators for DeepSeek.

Usage
─────
    from core.streaming import stream_deepseek

    async for chunk in stream_deepseek(messages, system_prompt):
        yield chunk   # {"type": "token"|"status"|"done"|"error", "text": "..."}
"""

import os
import json
import logging
import httpx

logger = logging.getLogger(__name__)

# ── SSE event helpers ─────────────────────────────────────────────────────────

def _evt(type_: str, text: str = "", **extra) -> dict:
    return {"type": type_, "text": text, **extra}

def status(text: str) -> dict:
    """Emit a status/progress event (shown as loader text in UI)."""
    return _evt("status", text)

def token(text: str) -> dict:
    """Emit a content token (appended to message bubble)."""
    return _evt("token", text)

def done(**meta) -> dict:
    """Emit a done event with optional metadata."""
    return _evt("done", **meta)

def error(text: str) -> dict:
    """Emit an error event."""
    return _evt("error", text)


# ── DeepSeek Streaming ────────────────────────────────────────────────────────

async def stream_deepseek(
    messages: list,
    *,
    model: str = "deepseek-v4-flash",
    max_tokens: int = 4500,
    temperature: float = 0.3,
    timeout: int = 90,
):
    """
    Async generator — streams DeepSeek tokens one by one.
    Yields dicts: {"type": "token", "text": "..."} for each chunk.
    Yields {"type": "done"} when complete.
    Yields {"type": "error", "text": "..."} on failure.
    """
    ds_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not ds_key:
        yield error("DEEPSEEK_API_KEY not set")
        return

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {ds_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield error(f"DeepSeek HTTP {resp.status_code}: {body[:200].decode()}")
                    return

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        yield done()
                        return
                    try:
                        chunk = json.loads(raw)
                        delta = chunk["choices"][0].get("delta", {})
                        text = delta.get("content") or ""
                        if text:
                            yield token(text)
                    except Exception:
                        continue

    except httpx.TimeoutException:
        yield error("DeepSeek streaming timed out")
    except Exception as e:
        logger.error("[Streaming] DeepSeek error: %s", e)
        yield error(str(e))


