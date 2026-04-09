"""
LLM Fallback Chain
==================
Handles quota exhaustion by intelligently routing through multiple LLM providers.

Priority: Kimi → DeepSeek → Cache/Fallback

This module ensures the system never gets blocked by a single provider's quota.
"""

import os
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

KIMI_API_KEY = os.getenv("MOONSHOT_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Fallback priority order
FALLBACK_CHAIN = ["kimi", "deepseek", "gemini", "cache"]

# Circuit breaker thresholds
CIRCUIT_BREAKER_THRESHOLD = 5  # errors before switching
CIRCUIT_BREAKER_TIMEOUT = 300  # seconds before retry


@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    content: str
    provider: str
    success: bool
    error: Optional[str] = None
    latency_ms: float = 0.0
    cached: bool = False


class CircuitBreaker:
    """Simple circuit breaker for provider health tracking."""

    def __init__(self, threshold=CIRCUIT_BREAKER_THRESHOLD, timeout=CIRCUIT_BREAKER_TIMEOUT):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.open = False

    def record_success(self):
        """Reset on success."""
        self.failures = 0
        self.open = False

    def record_failure(self):
        """Increment failure count."""
        self.failures += 1
        self.last_failure_time = datetime.now()
        if self.failures >= self.threshold:
            self.open = True
            logger.warning(f"Circuit breaker OPEN after {self.failures} failures")

    def is_available(self) -> bool:
        """Check if provider is available."""
        if not self.open:
            return True

        # Check if timeout elapsed
        if self.last_failure_time:
            elapsed = (datetime.now() - self.last_failure_time).total_seconds()
            if elapsed > self.timeout:
                logger.info("Circuit breaker timeout elapsed, attempting recovery")
                self.open = False
                self.failures = 0
                return True

        return False


class KimiClient:
    """Kimi (Moonshot) LLM client with quota awareness."""

    def __init__(self):
        self.api_key = KIMI_API_KEY
        self.circuit_breaker = CircuitBreaker()
        self.base_url = "https://api.moonshot.cn/v1"
        self.model = "kimi-k2.5"

    def is_available(self) -> bool:
        """Check if Kimi is configured and healthy."""
        return bool(self.api_key) and self.circuit_breaker.is_available()

    def generate(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        """Generate response using Kimi API."""
        if not self.is_available():
            return LLMResponse(
                content="",
                provider="kimi",
                success=False,
                error="Circuit breaker open or API key missing",
                cached=False
            )

        try:
            import httpx
            import json

            start = time.time()

            client = httpx.Client(timeout=30.0)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }

            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                self.circuit_breaker.record_success()

                logger.info(f"Kimi response success ({latency:.0f}ms)")
                return LLMResponse(
                    content=content,
                    provider="kimi",
                    success=True,
                    latency_ms=latency,
                    cached=False
                )
            else:
                error = f"HTTP {response.status_code}: {response.text}"
                logger.warning(f"Kimi error: {error}")
                self.circuit_breaker.record_failure()

                return LLMResponse(
                    content="",
                    provider="kimi",
                    success=False,
                    error=error,
                    latency_ms=latency
                )

        except Exception as e:
            error = f"Kimi exception: {str(e)}"
            logger.error(error)
            self.circuit_breaker.record_failure()

            return LLMResponse(
                content="",
                provider="kimi",
                success=False,
                error=error,
                cached=False
            )


class DeepSeekClient:
    """DeepSeek LLM client as secondary fallback."""

    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.circuit_breaker = CircuitBreaker()
        self.base_url = "https://api.deepseek.com/v1"
        self.model = "deepseek-chat"

    def is_available(self) -> bool:
        """Check if DeepSeek is configured and healthy."""
        return bool(self.api_key) and self.circuit_breaker.is_available()

    def generate(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        """Generate response using DeepSeek API."""
        if not self.is_available():
            return LLMResponse(
                content="",
                provider="deepseek",
                success=False,
                error="Circuit breaker open or API key missing",
                cached=False
            )

        try:
            import httpx

            start = time.time()

            client = httpx.Client(timeout=30.0)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }

            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                self.circuit_breaker.record_success()

                logger.info(f"DeepSeek response success ({latency:.0f}ms)")
                return LLMResponse(
                    content=content,
                    provider="deepseek",
                    success=True,
                    latency_ms=latency,
                    cached=False
                )
            else:
                error = f"HTTP {response.status_code}: {response.text}"
                logger.warning(f"DeepSeek error: {error}")
                self.circuit_breaker.record_failure()

                return LLMResponse(
                    content="",
                    provider="deepseek",
                    success=False,
                    error=error,
                    latency_ms=latency
                )

        except Exception as e:
            error = f"DeepSeek exception: {str(e)}"
            logger.error(error)
            self.circuit_breaker.record_failure()

            return LLMResponse(
                content="",
                provider="deepseek",
                success=False,
                error=error,
                cached=False
            )


# ═══════════════════════════════════════════════════════════════════════════
# Response Cache
# ═══════════════════════════════════════════════════════════════════════════

_response_cache: Dict[str, tuple] = {}  # {prompt_hash: (response, timestamp)}
CACHE_TTL = 3600  # 1 hour


def _get_cache_key(prompt: str) -> str:
    """Generate cache key from prompt."""
    import hashlib
    return hashlib.md5(prompt.encode()).hexdigest()


def _get_cached_response(prompt: str) -> Optional[str]:
    """Retrieve cached response if fresh."""
    key = _get_cache_key(prompt)
    if key in _response_cache:
        response, timestamp = _response_cache[key]
        elapsed = (datetime.now() - timestamp).total_seconds()
        if elapsed < CACHE_TTL:
            logger.info(f"Cache HIT for prompt (age: {elapsed:.0f}s)")
            return response
        else:
            del _response_cache[key]
    return None


def _cache_response(prompt: str, response: str):
    """Store response in cache."""
    key = _get_cache_key(prompt)
    _response_cache[key] = (response, datetime.now())
    logger.debug(f"Response cached (key: {key})")


# ═══════════════════════════════════════════════════════════════════════════
# Main Fallback Chain
# ═══════════════════════════════════════════════════════════════════════════

_kimi_client = None
_deepseek_client = None


def get_kimi_client() -> KimiClient:
    """Get or create Kimi client (singleton)."""
    global _kimi_client
    if _kimi_client is None:
        _kimi_client = KimiClient()
    return _kimi_client


def get_deepseek_client() -> DeepSeekClient:
    """Get or create DeepSeek client (singleton)."""
    global _deepseek_client
    if _deepseek_client is None:
        _deepseek_client = DeepSeekClient()
    return _deepseek_client


async def generate_with_fallback(
    prompt: str,
    system: str = "",
    preferred_provider: str = "kimi",
    **kwargs
) -> LLMResponse:
    """
    Generate response with intelligent fallback chain.

    Priority: Kimi → DeepSeek → Cache → Fallback

    Args:
        prompt: User prompt
        system: System prompt
        preferred_provider: Primary provider to try first
        **kwargs: Additional args (temperature, max_tokens, etc)

    Returns:
        LLMResponse with content and provider info
    """

    logger.info(f"Generating response (preferred: {preferred_provider})")

    # 1. Try cache first
    cached = _get_cached_response(prompt)
    if cached:
        return LLMResponse(
            content=cached,
            provider="cache",
            success=True,
            cached=True
        )

    # 2. Try Kimi
    kimi = get_kimi_client()
    if preferred_provider == "kimi" and kimi.is_available():
        result = kimi.generate(prompt, system, **kwargs)
        if result.success:
            _cache_response(prompt, result.content)
            return result
        logger.warning(f"Kimi failed: {result.error}, trying fallback")

    # 3. Try DeepSeek
    deepseek = get_deepseek_client()
    if deepseek.is_available():
        result = deepseek.generate(prompt, system, **kwargs)
        if result.success:
            _cache_response(prompt, result.content)
            return result
        logger.warning(f"DeepSeek failed: {result.error}, trying fallback")

    # 4. Return cache as last resort
    cached = _get_cached_response(prompt)
    if cached:
        return LLMResponse(
            content=cached,
            provider="cache",
            success=True,
            cached=True
        )

    # 5. Return generic fallback
    logger.error("All LLM providers failed, returning fallback response")
    return LLMResponse(
        content="I apologize, but I'm temporarily unable to process your request. Please try again in a moment.",
        provider="fallback",
        success=False,
        error="All providers unavailable",
        cached=False
    )


# ═══════════════════════════════════════════════════════════════════════════
# Synchronous Fallback Chain (safe to call from sync contexts)
# ═══════════════════════════════════════════════════════════════════════════

def generate_with_fallback_sync(
    prompt: str,
    system: str = "",
    preferred_provider: str = "kimi",
    **kwargs,
) -> LLMResponse:
    """
    Synchronous fallback chain: Kimi → DeepSeek → Cache → Static.

    Identical logic to ``generate_with_fallback`` but fully synchronous,
    safe to call from any sync context (e.g. gunicorn worker threads,
    ``_gemini_generate`` inside the orchestrator).

    Args:
        prompt: Prompt / message content.
        system: Optional system prompt.
        preferred_provider: First provider to attempt (default ``"kimi"``).
        **kwargs: Extra args forwarded to client.generate() (temperature, max_tokens, …).

    Returns:
        LLMResponse — always returns, never raises.
    """
    logger.info("generate_with_fallback_sync (preferred: %s)", preferred_provider)

    # 1. Cache first
    cached = _get_cached_response(prompt)
    if cached:
        return LLMResponse(content=cached, provider="cache", success=True, cached=True)

    # 2. Kimi
    kimi = get_kimi_client()
    if kimi.is_available():
        result = kimi.generate(prompt, system, **kwargs)
        if result.success:
            _cache_response(prompt, result.content)
            return result
        logger.warning("Kimi failed: %s — trying DeepSeek", result.error)

    # 3. DeepSeek
    deepseek = get_deepseek_client()
    if deepseek.is_available():
        result = deepseek.generate(prompt, system, **kwargs)
        if result.success:
            _cache_response(prompt, result.content)
            return result
        logger.warning("DeepSeek failed: %s — trying cache", result.error)

    # 4. Cache as last resort
    cached = _get_cached_response(prompt)
    if cached:
        return LLMResponse(content=cached, provider="cache", success=True, cached=True)

    # 5. Static fallback — never raises
    logger.error("All LLM providers exhausted, returning static fallback response")
    return LLMResponse(
        content=(
            "I apologize, but I'm temporarily unable to process your request. "
            "Please try again in a moment."
        ),
        provider="fallback",
        success=False,
        error="All providers unavailable",
        cached=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════════════════════════════════════

def get_llm_health() -> Dict[str, Any]:
    """Get health status of all LLM providers."""
    return {
        "kimi": {
            "available": get_kimi_client().is_available(),
            "failures": get_kimi_client().circuit_breaker.failures,
            "configured": bool(KIMI_API_KEY)
        },
        "deepseek": {
            "available": get_deepseek_client().is_available(),
            "failures": get_deepseek_client().circuit_breaker.failures,
            "configured": bool(DEEPSEEK_API_KEY)
        },
        "cache": {
            "entries": len(_response_cache),
            "ttl_seconds": CACHE_TTL
        }
    }
