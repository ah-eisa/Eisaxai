import requests
"""
EisaX - GLM 4.7 Flash Client
الدور المزدوج:
1. مراجع الكود (Security + Logic فقط - لا يعيد الكتابة)
2. بديل Gemini الكامل في حالة الفشل
"""

import os
import httpx
import json
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

GLM_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_MODEL = "glm-4.7-flash"


class GLMClient:
    def __init__(self):
        self.api_key = os.getenv("GLM_API_KEY", "")
        if not self.api_key:
            logger.warning("⚠️ GLM_API_KEY not set")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    # ─────────────────────────────────────────
    # مسار الكود: GLM يراجع فقط (Security + Logic)
    # ─────────────────────────────────────────
    async def review_code(self, code: str, original_request: str) -> dict:
        """
        يراجع الكود للأمان والمنطق فقط - لا يعيد الكتابة أبداً
        """
        system_prompt = """You are a strict code security and logic reviewer.

YOUR ONLY JOB:
1. Check for SECURITY vulnerabilities (SQL injection, XSS, exposed secrets, etc.)
2. Check for LOGIC errors (infinite loops, wrong conditions, off-by-one, etc.)

STRICT RULES:
- Do NOT rewrite the code
- Do NOT change the coding style
- Do NOT suggest refactoring
- ONLY report actual security or logic problems

OUTPUT FORMAT (JSON only):
{
  "approved": true/false,
  "security_issues": ["issue1", "issue2"] or [],
  "logic_issues": ["issue1", "issue2"] or [],
  "verdict": "APPROVED - Code is safe and logical" or "ISSUES FOUND - [brief summary]"
}"""

        message = f"Original request: {original_request}\n\nCode to review:\n{code}"
        return await self._call(system_prompt, message, expect_json=True)

    # ─────────────────────────────────────────
    # بديل Gemini: GLM يتولى الدور الكامل
    # ─────────────────────────────────────────
    async def chat_as_eisax(self, message: str, history: Optional[List[Dict]] = None) -> dict:
        """
        يعمل كبديل كامل لـ Gemini بنفس شخصية EisaX
        """
        eisax_identity = """You are EisaX, a bilingual AI assistant created by Eng. Ahmed Eisa.

IDENTITY (never change):
- Creator: Eng. Ahmed Eisa (Investment Portfolio Manager & Electronic Engineer, Abu Dhabi, UAE)
- You are NOT created by Google/Anthropic/OpenAI/ZhipuAI

LANGUAGE RULES:
- Reply in the same language the user uses (Arabic or English)
- Be warm, professional, and clear

You are now acting as the main interface since Gemini is temporarily unavailable."""

        messages = [{"role": "system", "content": eisax_identity}]

        if history:
            for msg in history[-16:]:
                role = "user" if msg.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": msg.get("content", "")})

        messages.append({"role": "user", "content": message})

        return await self._call_with_messages(messages)

    async def present_final_result(self, result_type: str, data: str, context: str = "") -> dict:
        """
        GLM يعرض النتيجة النهائية بأسلوب احترافي (بديل Gemini في العرض)
        """
        try:
            system_prompt = f"""You are a Chief Investment Officer preparing this financial analysis for professional export to {format_type.upper()}.

Take the raw analysis below and format it as a polished, institutional-grade investment memorandum suitable for presentation to an investment committee.

Use your expertise to:
- Structure it professionally (proper memo format, clear sections, clean tables)
- Remove any informal elements (emojis, markdown symbols, casual language)
- Ensure it looks like a document from a top-tier investment firm

Preserve ALL data, numbers, and analytical content exactly as provided. Only improve the presentation and professionalism.

Do NOT add analysis or opinions - just format what's given into CIO-quality output."""

            response = requests.post(
                GLM_API_URL,
                headers=self.headers,
                json={
                    "model": GLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"نظف هذا المحتوى للتصدير:\n\n{content}"}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 4000
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                cleaned = result["choices"][0]["message"]["content"]
                return {"success": True, "content": cleaned}
            else:
                return {"success": False, "content": content, "error": "GLM formatting failed"}
                
        except Exception as e:
            logger.warning(f"GLM export prep failed: {e}")
            return {"success": False, "content": content, "error": str(e)}

    def filter_news_relevance(
        self,
        articles: list,
        asset_name: str,
        ticker: str,
        sector: str,
        asset_type: str = "stock",
        min_score: int = 60,
        **kwargs,
    ) -> list:
        """
        Filter news articles by relevance using DeepSeek (primary) → OpenAI → GLM (fallback).
        Pass bucket="direct"|"sector"|"country"|"related" to adjust scoring strictness.

        Sends all headlines in a single batch call (1 API request regardless of list size).
        Returns articles with relevance score >= min_score.
        Fallback chain: DeepSeek → OpenAI → GLM → original articles unchanged.
        """
        if not articles:
            return articles

        # ── Cache check ──────────────────────────────────────────────────
        _bucket = kwargs.get("bucket", "direct")
        _headlines = [a.get("title", "") for a in articles]
        _nfc_set = None
        try:
            from core.news_filter_cache import get as _nfc_get, set as _nfc_set
            _cached_result = _nfc_get(ticker, _bucket, _headlines)
            if _cached_result is not None:
                return _cached_result
        except Exception:
            _nfc_set = None

        headlines = [a.get("title", "") for a in articles]
        headlines_text = "\n".join(f'{i}. "{h}"' for i, h in enumerate(headlines))

        # Build scoring guidance based on news bucket type
        if _bucket == "direct":
            _scoring_guide = (
                "- 80-100: Directly about this company, its products, or executives\n"
                "- 50-79: About a close competitor or direct business partner\n"
                "- 20-49: Same sector but not about this company\n"
                "- 0-19: Completely unrelated (different sector, weather, sports, etc.)"
            )
        elif _bucket == "sector":
            _scoring_guide = (
                "- 80-100: Directly about this sector/industry trends or major players\n"
                "- 60-79: About a major company or theme within this exact sector\n"
                "- 30-59: Tangentially related (adjacent industry, general business)\n"
                "- 0-29: Different sector entirely (e.g. biotech for software, exchange outages for tech companies)\n"
                "NOTE: 'Technology sector' = software, hardware, AI, cloud, semiconductors. "
                "Stock exchange technical glitches, trading platform outages, or financial infrastructure issues are NOT Technology sector news."
            )
        else:  # country / related
            _scoring_guide = (
                "- 80-100: Major market-moving news directly about THIS region's financial markets "
                "(Fed decisions, US indices, regional economic data, sector-wide events)\n"
                "- 60-79: Economic or policy news clearly relevant to investors in THIS specific market\n"
                "- 30-59: News about OTHER countries or regions (even if the source is a major bank)\n"
                "- 0-29: Unrelated (weather, sports, local crime, entertainment, other countries)\n"
                "CRITICAL: News about a DIFFERENT country (e.g. Goldman Sachs downgrades India) "
                "scores 20-30 for a USA bucket — it is NOT relevant to US investors tracking this asset. "
                "Only score high if the news directly impacts THIS region's equity markets."
            )

        prompt = (
            f"Asset: {asset_name} ({ticker}) - {sector} - {asset_type}\n"
            f"News category: {_bucket}\n\n"
            f"Rate each headline's relevance (0-100):\n{_scoring_guide}\n\n"
            f'Reply in JSON array only, no explanation:\n'
            f'[{{"index": 0, "score": 75}}, {{"index": 1, "score": 20}}, ...]\n\n'
            f"Headlines:\n{headlines_text}"
        )

        result = None

        # ── Try 0: DeepSeek (fastest, already in system) ────────────────────
        import os as _os
        ds_key = _os.getenv("DEEPSEEK_API_KEY", "")
        if ds_key:
            try:
                resp = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {ds_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 300,
                    },
                    timeout=15,
                )
                if resp.status_code == 200:
                    result = self._parse_relevance_scores(
                        resp.json()["choices"][0]["message"]["content"],
                        articles, min_score, ticker, "DeepSeek"
                    )
                    if result is not None:
                        try:
                            if _nfc_set is not None:
                                _nfc_set(ticker, _bucket, _headlines, result)
                        except Exception:
                            pass
                        return result
                else:
                    logger.warning(f"[News Filter] DeepSeek returned {resp.status_code} for {ticker}")
            except Exception as e:
                logger.warning(f"[News Filter] DeepSeek failed for {ticker}: {e}")

        # ── Try 1: OpenAI gpt-4o-mini (best English comprehension) ──────────
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            try:
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openai_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 300,
                    },
                    timeout=8,
                )
                if resp.status_code == 200:
                    result = self._parse_relevance_scores(
                        resp.json()["choices"][0]["message"]["content"],
                        articles, min_score, ticker, "OpenAI"
                    )
                    if result is not None:
                        try:
                            if _nfc_set is not None:
                                _nfc_set(ticker, _bucket, _headlines, result)
                        except Exception:
                            pass
                        return result
                else:
                    logger.warning(f"[News Filter] OpenAI returned {resp.status_code} for {ticker}")
            except Exception as e:
                logger.warning(f"[News Filter] OpenAI failed for {ticker}: {e}")

        # ── Try 2: GLM Flash 4.7 (fallback) ─────────────────────────────────
        if self.api_key:
            try:
                resp = requests.post(
                    GLM_API_URL,
                    headers=self.headers,
                    json={
                        "model": GLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 300,
                    },
                    timeout=8,
                )
                if resp.status_code == 200:
                    result = self._parse_relevance_scores(
                        resp.json()["choices"][0]["message"]["content"],
                        articles, min_score, ticker, "GLM"
                    )
                    if result is not None:
                        try:
                            if _nfc_set is not None:
                                _nfc_set(ticker, _bucket, _headlines, result)
                        except Exception:
                            pass
                        return result
                else:
                    logger.warning(f"[News Filter] GLM returned {resp.status_code} for {ticker}")
            except Exception as e:
                logger.warning(f"[News Filter] GLM fallback failed for {ticker}: {e}")

        # ── Cache the successful result ───────────────────────────────────
        try:
            if result is not None:
                from core.news_filter_cache import set as _nfc_set
                _nfc_set(ticker, _bucket, _headlines, result)
        except Exception:
            pass

        # ── All providers failed — return original articles unchanged ────────
        logger.warning(f"[News Filter] All providers failed for {ticker}, using original news")
        return articles

    @staticmethod
    def _parse_relevance_scores(
        raw_content: str,
        articles: list,
        min_score: int,
        ticker: str,
        provider: str,
    ) -> list | None:
        """Parse JSON relevance scores from any LLM provider. Returns filtered list or None on parse failure."""
        try:
            raw = raw_content.strip()
            # Strip markdown code fences if model wrapped the JSON
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            scores_list = json.loads(raw)
            score_map = {int(item["index"]): int(item.get("score", 0)) for item in scores_list}

            filtered = [
                a for i, a in enumerate(articles)
                if score_map.get(i, 100) >= min_score
            ]

            kept, total = len(filtered), len(articles)
            logger.info(f"[News Filter] {provider} → {ticker}: {kept}/{total} passed (min={min_score})")

            # Never return empty — fall back to originals
            return filtered if filtered else articles

        except Exception as e:
            logger.warning(f"[News Filter] {provider} parse failed for {ticker}: {e}")
            return None  # signals caller to try next provider

    def prepare_for_export(self, content: str, format_type: str = "pdf") -> dict:
        """Format content for professional export"""
        try:
            system_prompt = f"""You are a Chief Investment Officer. Format this financial analysis as a polished institutional memorandum for {format_type.upper()} export.

Make it professional: clean structure, no emojis/symbols, proper memo format, institutional tone.
Keep ALL data and numbers exactly as provided - only improve presentation."""

            response = requests.post(
                GLM_API_URL,
                headers=self.headers,
                json={
                    "model": GLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Format this report professionally:\n\n{content}"}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 4000
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                cleaned = result["choices"][0]["message"]["content"]
                return {"success": True, "content": cleaned}
            else:
                return {"success": False, "content": content, "error": "GLM failed"}
                
        except Exception as e:
            return {"success": False, "content": content, "error": str(e)}
