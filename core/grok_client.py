"""
EisaX — Grok xAI Client
========================
يجلب X/Twitter sentiment و key themes لأي ticker عبر Grok-4.1-fast.

المصدر:  xAI Responses API  (POST /v1/responses)
الموديل: grok-4-1-fast-non-reasoning  ($0.20/M input · $0.50/M output)
الأداة:  x_search  (يحل محل live search القديم المهمل)

الاستخدام:
    from core.grok_client import get_x_sentiment
    data = get_x_sentiment("MSFT", "Microsoft Corporation", "Technology")
    # → {"sentiment": "Bullish", "score": 0.6, "themes": [...], "top_posts": [...]}
"""

import os
import json
import logging
import time
import requests
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"
XAI_MODEL         = "grok-4-1-fast-non-reasoning"

# In-process cache — 60 min TTL  (consistent within a session; X posts don't change often)
_CACHE: dict = {}
_CACHE_TTL  = 3600


def get_x_sentiment(ticker: str,
                    asset_name: str = "",
                    sector: str = "") -> dict:
    """
    Fetch real-time X/Twitter sentiment for a ticker via Grok x_search tool.

    Returns:
        {
          "ticker":     str,
          "sentiment":  "Bullish" | "Bearish" | "Neutral" | "Mixed",
          "score":      float,   # -1.0 → +1.0
          "themes":     [str, ...],
          "top_posts":  [{"source", "text", "date", "impact", "likes"}, ...],
          "breaking":   str | None,
          "x_summary":  str,
          "source":     "grok-live",
          "cached":     bool,
        }

    On any failure returns an empty-safe dict so the report continues.
    """
    ticker = ticker.upper().strip()

    # ── Cache ────────────────────────────────────────────────────────────────
    hit = _CACHE.get(ticker)
    if hit and time.time() - hit["_ts"] < _CACHE_TTL:
        r = dict(hit["data"])
        r["cached"] = True
        return r

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.getenv("XAI_API_KEY", "")
    if not api_key:
        logger.warning("[Grok] XAI_API_KEY not set — skipping X sentiment")
        return _empty(ticker, "no_api_key")

    # ── Date range: last 48 hours ─────────────────────────────────────────────
    _now      = datetime.now(timezone.utc)
    _from_dt  = (_now - timedelta(hours=48)).strftime("%Y-%m-%d")
    _to_dt    = _now.strftime("%Y-%m-%d")

    # ── Prompt ────────────────────────────────────────────────────────────────
    _name_ctx   = f" ({asset_name})" if asset_name else ""
    _sector_ctx = f" — {sector} sector" if sector else ""

    prompt = (
        f"Search X/Twitter for the latest posts about {ticker}{_name_ctx}{_sector_ctx} "
        f"from the last 48 hours. Focus on credible financial accounts: analysts, "
        f"portfolio managers, financial data platforms, and financial media.\n\n"
        f"Return ONLY valid JSON (no markdown, no explanation):\n"
        f'{{\n'
        f'  "ticker": "{ticker}",\n'
        f'  "sentiment": "Bullish|Bearish|Neutral|Mixed",\n'
        f'  "score": <float -1.0 to +1.0>,\n'
        f'  "themes": ["theme1", "theme2", "theme3"],\n'
        f'  "top_posts": [\n'
        f'    {{\n'
        f'      "source": "@handle",\n'
        f'      "text": "<post text max 200 chars>",\n'
        f'      "date": "YYYY-MM-DD",\n'
        f'      "impact": "Positive|Negative|Neutral",\n'
        f'      "likes": <int>\n'
        f'    }}\n'
        f'  ],\n'
        f'  "breaking": "<breaking news or unusual signal, or null>",\n'
        f'  "x_summary": "<1-2 sentence human summary of X sentiment>"\n'
        f'}}\n\n'
        f"IMPORTANT: Only include posts with at least 50 likes — low-engagement posts are not representative. "
        f"Prioritize posts from institutional accounts, financial data platforms, or professional traders with high engagement. "
        f"Include up to 5 top posts sorted by likes descending. If fewer than 2 posts meet the 50-like threshold, include the highest-engagement posts available. "
        f'If no relevant posts found, return {{"sentiment": "Neutral", "score": 0, "themes": [], "top_posts": [], "breaking": null, "x_summary": "No significant X activity found."}}'
    )

    # ── Responses API call with x_search tool ────────────────────────────────
    try:
        resp = requests.post(
            XAI_RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model": XAI_MODEL,
                "input": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_output_tokens": 900,
                "tools": [
                    {
                        "type":      "x_search",
                        "from_date": _from_dt,
                        "to_date":   _to_dt,
                    }
                ],
            },
            timeout=18,
        )

        if resp.status_code != 200:
            logger.warning(
                f"[Grok] API error {resp.status_code} for {ticker}: "
                f"{resp.text[:300]}"
            )
            return _empty(ticker, f"http_{resp.status_code}")

        # ── Parse Responses API output ────────────────────────────────────────
        resp_json = resp.json()
        raw = _extract_text(resp_json)

        if not raw:
            logger.warning(f"[Grok] Empty response for {ticker}")
            return _empty(ticker, "empty_response")

        # Strip markdown fences
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data              = json.loads(raw)
        data["score"]     = max(-1.0, min(1.0, float(data.get("score", 0))))
        data["source"]    = "grok-live"
        data["cached"]    = False
        data["ticker"]    = ticker

        # ── Filter low-engagement posts (< 20 likes) — not representative ────
        _posts = data.get("top_posts", [])
        _filtered = [p for p in _posts if int(p.get("likes", 0)) >= 20]
        # Keep originals if filter removed everything
        data["top_posts"] = _filtered if _filtered else _posts

        # ── Cache & return ────────────────────────────────────────────────────
        _CACHE[ticker] = {"data": dict(data), "_ts": time.time()}
        logger.info(
            f"[Grok] {ticker}: {data['sentiment']} ({data['score']:+.2f}) | "
            f"themes={data.get('themes', [])} | posts={len(data.get('top_posts', []))}"
        )
        return data

    except json.JSONDecodeError as e:
        logger.warning(f"[Grok] JSON parse error for {ticker}: {e} | raw={raw[:200]}")
        return _empty(ticker, "json_parse_error")
    except requests.Timeout:
        logger.warning(f"[Grok] Timeout for {ticker}")
        return _empty(ticker, "timeout")
    except Exception as e:
        logger.warning(f"[Grok] Unexpected error for {ticker}: {e}")
        return _empty(ticker, str(e))


def _extract_text(resp_json: dict) -> str:
    """
    Extract the assistant's text from the xAI Responses API response.
    The Responses API returns output as a list of content items.
    """
    # Format: {"output": [{"type": "message", "content": [{"type": "output_text", "text": "..."}]}]}
    try:
        for item in resp_json.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        return block.get("text", "")
    except Exception:
        pass

    # Fallback: try choices (in case API switches format)
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        pass

    return ""


def _empty(ticker: str, reason: str = "") -> dict:
    """Safe empty result — report continues without X sentiment block."""
    return {
        "ticker":    ticker,
        "sentiment": "",
        "score":     0.0,
        "themes":    [],
        "top_posts": [],
        "breaking":  None,
        "x_summary": "",
        "source":    "grok-unavailable",
        "cached":    False,
        "_reason":   reason,
    }
