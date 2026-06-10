"""
EisaX - DeepSeek V3 Client (Robust Version)
Features:
- High Timeouts (120s) to fix 504 Errors.
- Strict Full-Stack Prompt for complete HTML/CSS/JS.
- JSON Mode for Financial Analysis.
"""

import os
import json
import httpx
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"

class DeepSeekClient:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ DEEPSEEK_API_KEY not set")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # إعدادات اتصال قوية (120 ثانية) لمنع انقطاع الاتصال
        self.timeout = httpx.Timeout(300.0, connect=30.0, read=300.0)
        self.limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)

    # ─────────────────────────────────────────
    # 1. واجهة طلب الكود (Strict Full-Stack Mode)
    # ─────────────────────────────────────────
    async def write_code(self, user_request: str) -> Dict[str, Any]:
        """
        Generates full, self-contained code.
        """
        # برومبت صارم جداً لضمان كود كامل بدون اختصارات
        system_prompt = """You are an Elite Full-Stack Developer.

CRITICAL INSTRUCTIONS:
1. If the user asks for a website/app, return a SINGLE HTML file.
2. Embed ALL CSS inside <style> tags (Make it modern & beautiful).
3. Embed ALL JavaScript inside <script> tags.
4. MUST start with <!DOCTYPE html>.
5. Do NOT use markdown blocks (no ```html). Return raw code only.
6. Do NOT abbreviate. Write the FULL code (no placeholders like '').
7. Ensure the code is production-ready and bug-free.
"""
        return await self.chat(system_prompt, user_request)

    # ─────────────────────────────────────────
    # 2. واجهة التحليل المالي (JSON Mode)
    # ─────────────────────────────────────────
    async def prepare_financial_params(self, user_request: str, market_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts financial parameters into JSON.
        """
        context = f"\nLatest Market Data:\n{market_data}" if market_data else ""
        system_prompt = """You are a quantitative financial analyst.
Extract parameters into JSON format.
OUTPUT FORMAT:
{
  "intent": "analyze|optimize|forecast",
  "tickers": ["AAPL", "BTC-USD"],
  "summary": "brief text"
}"""
        message = f"User request: {user_request}{context}"
        return await self.chat(system_prompt, message, expect_json=True)

    # ─────────────────────────────────────────
    # 3. المحرك الأساسي (The Core Engine)
    # ─────────────────────────────────────────
    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        expect_json: bool = False,
        temperature: float = 0.0,  # 0.0 للدقة القصوى
        max_tokens: int = 8000     # الحد الأقصى للكتابة
    ) -> Dict[str, Any]:
        
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        if expect_json:
            payload["response_format"] = {"type": "json_object"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout, limits=self.limits) as client:
                response = await client.post(
                    DEEPSEEK_API_URL,
                    headers=self.headers,
                    json=payload
                )
                
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                if expect_json:
                    try:
                        return {"success": True, "data": json.loads(content)}
                    except json.JSONDecodeError:
                        return {"success": True, "data": {"error": "JSON parse failed", "raw": content}}

                return {"success": True, "data": content.strip()}

        except httpx.TimeoutException:
            logger.error("❌ DeepSeek timeout (300s limit reached)")
            return {"success": False, "error": "Request timed out"}
        except Exception as e:
            logger.error(f"❌ DeepSeek error: {e}")
            return {"success": False, "error": str(e)}

    async def chat_as_eisax(self, prompt: str, history: list = []):
        try:
            system_instruction = "You are EisaX, a Senior Portfolio Strategist. Analyze the provided portfolio data and provide an institutional-grade mandate."
            result = await self.chat(
                system_prompt=system_instruction,
                user_message=prompt,
                expect_json=False
            )
            if result.get("success"):
                return result.get("data", "")
            return f"Error: {result.get('error')}"
        except Exception as e:
            return f"Strategy Error: {str(e)}"
