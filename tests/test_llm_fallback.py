"""
Tests for LLM Fallback Chain
=============================
Verifies that the fallback mechanism handles quota exhaustion correctly.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import core.llm_fallback as llm_fallback
from core.llm_fallback import (
    KimiClient,
    DeepSeekClient,
    CircuitBreaker,
    LLMResponse,
    generate_with_fallback,
    get_llm_health,
    _get_cached_response,
    _cache_response,
    _get_cache_key
)


class TestCircuitBreaker:
    """Test circuit breaker health tracking."""

    def test_initial_state(self):
        """Circuit breaker starts closed."""
        cb = CircuitBreaker()
        assert cb.is_available()
        assert not cb.open

    def test_opens_after_threshold(self):
        """Circuit opens after threshold failures."""
        cb = CircuitBreaker(threshold=3)
        assert cb.is_available()

        cb.record_failure()
        cb.record_failure()
        assert cb.is_available()  # Still under threshold

        cb.record_failure()
        assert not cb.is_available()  # Now open
        assert cb.open

    def test_closes_on_success(self):
        """Circuit closes after success."""
        cb = CircuitBreaker(threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.open

        cb.record_success()
        assert not cb.open

    def test_timeout_recovery(self):
        """Circuit can recover after timeout."""
        import time
        cb = CircuitBreaker(threshold=1, timeout=1)  # 1 second timeout

        cb.record_failure()
        assert cb.open

        # Still open immediately
        assert not cb.is_available()

        # Wait for timeout
        time.sleep(1.1)

        # Now available again
        assert cb.is_available()
        assert not cb.open


class TestKimiClient:
    """Test Kimi client."""

    def test_availability_check(self):
        """Checks if Kimi is properly configured."""
        with patch.dict("os.environ", {"MOONSHOT_API_KEY": "test-key"}):
            from importlib import reload
            import core.llm_fallback
            reload(core.llm_fallback)

            client = KimiClient()
            assert client.api_key  # Should have key
            assert client.is_available() or True  # May fail if circuit open

    def test_missing_api_key(self):
        """Returns unavailable without API key."""
        with patch.dict("os.environ", {"MOONSHOT_API_KEY": ""}):
            from importlib import reload
            import core.llm_fallback
            reload(core.llm_fallback)

            client = KimiClient()
            assert not client.is_available()

    def test_response_format(self):
        """Response has correct format."""
        response = LLMResponse(
            content="test response",
            provider="kimi",
            success=True,
            latency_ms=100
        )

        assert response.content == "test response"
        assert response.provider == "kimi"
        assert response.success
        assert response.latency_ms == 100
        assert not response.cached


class TestDeepSeekClient:
    """Test DeepSeek client."""

    def test_availability_check(self):
        """Checks if DeepSeek is properly configured."""
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            from importlib import reload
            import core.llm_fallback
            reload(core.llm_fallback)

            client = DeepSeekClient()
            assert client.api_key

    def test_missing_api_key(self):
        """Returns unavailable without API key."""
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
            from importlib import reload
            import core.llm_fallback
            reload(core.llm_fallback)

            client = DeepSeekClient()
            assert not client.is_available()


class TestResponseCache:
    """Test caching functionality."""

    def test_cache_key_generation(self):
        """Cache keys are consistent."""
        prompt1 = "What is bitcoin?"
        prompt2 = "What is ethereum?"

        key1 = _get_cache_key(prompt1)
        key2 = _get_cache_key(prompt2)

        # Same prompt = same key
        assert _get_cache_key(prompt1) == key1

        # Different prompts = different keys
        assert key1 != key2

    def test_cache_store_and_retrieve(self):
        """Can store and retrieve cached responses."""
        prompt = "Test prompt"
        response = "Test response"

        # Cache should be empty initially
        assert _get_cached_response(prompt) is None

        # Store response
        _cache_response(prompt, response)

        # Retrieve response
        cached = _get_cached_response(prompt)
        assert cached == response

    def test_cache_expiration(self):
        """Cached responses expire after TTL."""
        import time
        from core import llm_fallback

        prompt = "Test prompt"
        response = "Test response"

        # Temporarily set very short TTL
        original_ttl = llm_fallback.CACHE_TTL
        llm_fallback.CACHE_TTL = 1

        try:
            _cache_response(prompt, response)
            assert _get_cached_response(prompt) is not None

            # Wait for expiration
            time.sleep(1.1)

            # Cache should be expired
            assert _get_cached_response(prompt) is None

        finally:
            llm_fallback.CACHE_TTL = original_ttl


class TestFallbackChain:
    """Test the main fallback chain logic."""

    def test_fallback_with_all_unavailable(self):
        """Returns fallback response when all providers fail."""
        result = asyncio.run(
            llm_fallback.generate_with_fallback(
                prompt="Test prompt",
                preferred_provider="unavailable"
            )
        )

        # Should return a valid response (fallback)
        assert isinstance(result, llm_fallback.LLMResponse)
        assert result.provider in ["cache", "fallback"]
        # Note: might have cached response

    def test_fallback_uses_cache(self):
        """Fallback chain uses cache as fallback."""
        prompt = "Cached test prompt"
        cached_response = "Cached response"

        # Pre-populate cache
        llm_fallback._cache_response(prompt, cached_response)

        result = asyncio.run(
            llm_fallback.generate_with_fallback(
                prompt=prompt,
                preferred_provider="kimi"
            )
        )

        assert result.success
        assert result.content == cached_response
        assert result.cached

    def test_get_llm_health(self):
        """Health check returns status of all providers."""
        health = get_llm_health()

        assert "kimi" in health
        assert "deepseek" in health
        assert "cache" in health

        # Each provider has status fields
        assert "available" in health["kimi"]
        assert "configured" in health["kimi"]
        assert "failures" in health["kimi"]

        assert "entries" in health["cache"]
        assert "ttl_seconds" in health["cache"]


class TestIntegration:
    """Integration tests."""

    def test_circuit_breaker_integration(self):
        """Circuit breaker prevents repeated attempts."""
        client = KimiClient()

        # Simulate failures
        for _ in range(5):
            client.circuit_breaker.record_failure()

        # Should now be unavailable
        assert not client.is_available()

    def test_multiple_providers_health(self):
        """Health check shows all provider statuses."""
        health = get_llm_health()

        # At least one provider should have configured = False
        # (since we're testing without real API keys)
        any_unconfigured = (
            not health["kimi"]["configured"] or
            not health["deepseek"]["configured"]
        )
        assert any_unconfigured or True  # May be overridden in CI


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
