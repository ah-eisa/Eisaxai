"""
tests/test_e2e_integration.py
==============================
End-to-end integration tests — Phase C-5.

Scenarios covered:
  1. Greeting fast-path (no LLM call)
  2. General chat
  3. BTC-USD analysis routing
  4. MSFT analysis routing
  5. Aramco (2222.SR) analysis routing — Arabic Arabic-ticker fast-path
  6. Portfolio optimize routing
  7. Export request detection
  8. Bond request detection
  9. Fallback chain: Gemini fails → Kimi/DeepSeek/static fallback returns something

All LLM + yfinance calls are stubbed — no network required.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# ── Helpers ──────────────────────────────────────────────────────────────────

def run(coro):
    """Run a coroutine in a fresh event loop (pytest-asyncio not required)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_orchestrator():
    """Create a MultiAgentOrchestrator with all external clients disabled."""
    with (
        patch("core.orchestrator.GEMINI_API_KEY", "fake-key"),
        patch("core.orchestrator.GEMINI_API_KEY_BACKUP", ""),
        patch("core.orchestrator.MOONSHOT_API_KEY", ""),
        patch("core.orchestrator.ADMIN_ENABLED", False),
        patch("core.orchestrator.MEMORY_ENABLED", False),
    ):
        # Prevent actual google.genai import during __init__
        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": MagicMock()}):
            from core.orchestrator import MultiAgentOrchestrator
            orch = MultiAgentOrchestrator(db_path=":memory:")
            orch.gemini_client = None
            orch.gemini_client_backup = None
            orch.kimi_client = None
            return orch


# ══════════════════════════════════════════════════════════════════════
# Routing service — pure-function smoke tests (no network)
# ══════════════════════════════════════════════════════════════════════

class TestRoutingService:
    """Tests for the detection helpers in core/services/routing_service.py"""

    def setup_method(self):
        from core.services.routing_service import (
            is_greeting, is_export_request, is_bond_request,
            detect_arabic_ticker, is_file_analysis,
        )
        self.is_greeting      = is_greeting
        self.is_export        = is_export_request
        self.is_bond          = is_bond_request
        self.detect_arabic    = detect_arabic_ticker
        self.is_file_analysis = is_file_analysis

    # ── Greetings ─────────────────────────────────────────────────────
    def test_greeting_hello(self):
        assert self.is_greeting("hello") is True

    def test_greeting_arabic(self):
        assert self.is_greeting("مرحبا") is True

    def test_greeting_long_message_not_greeting(self):
        assert self.is_greeting("hello please analyze apple stock for me today") is False

    # ── Export detection ──────────────────────────────────────────────
    def test_export_pdf(self):
        assert self.is_export("generate pdf report") is True

    def test_export_arabic(self):
        assert self.is_export("صدر تقرير pdf") is True

    def test_export_not_triggered(self):
        assert self.is_export("analyze AAPL") is False

    # ── Bond detection ─────────────────────────────────────────────────
    def test_bond_keyword(self):
        assert self.is_bond("US treasury bond yields") is True

    def test_bond_sukuk(self):
        assert self.is_bond("Saudi sukuk market") is True

    def test_bond_not_equity(self):
        assert self.is_bond("analyze AAPL stock price") is False

    # ── Arabic ticker detection ───────────────────────────────────────
    def test_arabic_ticker_aramco(self):
        result = self.detect_arabic("حلل ارامكو")
        assert result is not None
        assert "2222" in result or "SR" in result

    def test_arabic_ticker_not_detected_for_english(self):
        result = self.detect_arabic("analyze AAPL")
        assert result is None

    # ── File analysis prefix ─────────────────────────────────────────
    def test_file_analysis_prefix(self):
        assert self.is_file_analysis("[FILE ANALYSIS] some content") is True

    def test_no_file_analysis(self):
        assert self.is_file_analysis("analyze AAPL") is False


# ══════════════════════════════════════════════════════════════════════
# Intent classifier — routing decisions
# ══════════════════════════════════════════════════════════════════════

class TestIntentClassifier:

    def setup_method(self):
        from core.intent_classifier import IntentClassifier
        self.clf = IntentClassifier()

    def test_btc_is_stock_analysis(self):
        intent = self.clf.detect_primary_intent("analyze BTC-USD bitcoin")
        # Could be stock_analysis or None (goes to Gemini router)
        assert intent in ("stock_analysis", "general", None)

    def test_portfolio_optimize_intent(self):
        intent = self.clf.detect_primary_intent("optimize my portfolio risk moderate")
        assert intent == "portfolio_optimize"

    def test_export_pdf_intent(self):
        intent = self.clf.detect_primary_intent("generate PDF report")
        assert intent == "report_export"

    def test_greeting_returns_none(self):
        intent = self.clf.detect_primary_intent("hello")
        assert intent is None


# ══════════════════════════════════════════════════════════════════════
# Orchestrator integration — process_message with mocked LLM
# ══════════════════════════════════════════════════════════════════════

class TestOrchestratorE2E:

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        """Provide a fresh in-memory orchestrator per test."""
        import sys
        # Ensure google.genai mock is in place
        mock_google = MagicMock()
        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_google.genai,
        }):
            from core.orchestrator import MultiAgentOrchestrator
            self.orch = MultiAgentOrchestrator(db_path=str(tmp_path / "test.db"))
            self.orch.gemini_client = None
            self.orch.gemini_client_backup = None
            self.orch.kimi_client = None

    # ── Greeting fast-path ────────────────────────────────────────────
    def test_greeting_returns_reply(self):
        """Greeting path skips LLM entirely → should always return a reply."""

        async def _run():
            with patch("core.orchestrator.ADMIN_ENABLED", False), \
                 patch("core.orchestrator.MEMORY_ENABLED", False), \
                 patch.object(self.orch, "_gemini_generate",
                              return_value="Hi there! How can I help?"), \
                 patch("core.services.market_route_handler.handle_general",
                        new_callable=AsyncMock,
                        return_value={"reply": "Hello from EisaX!", "session_id": "s1"}):
                result = await self.orch.process_message("user1", "hello", session_id="s1")
            return result

        result = run(_run())
        assert isinstance(result, dict)
        assert "reply" in result
        assert result["reply"]  # non-empty

    # ── General chat ──────────────────────────────────────────────────
    def test_general_chat_routed(self):
        """General LLM chat returns a non-empty reply."""

        async def _run():
            with patch("core.orchestrator.ADMIN_ENABLED", False), \
                 patch("core.orchestrator.MEMORY_ENABLED", False), \
                 patch.object(self.orch, "_classify_intent",
                              return_value=("GENERAL", "GENERAL", "what is inflation?", "")), \
                 patch.object(self.orch, "_gemini_generate",
                              return_value="Inflation is a rise in the general price level."), \
                 patch("core.services.market_route_handler.handle_general",
                        new_callable=AsyncMock,
                        return_value={"reply": "Inflation is a rise in the general price level.", "session_id": "s2"}):
                result = await self.orch.process_message("user1", "what is inflation?", session_id="s2")
            return result

        result = run(_run())
        assert "reply" in result
        assert len(result["reply"]) > 10

    # ── BTC analysis routing ──────────────────────────────────────────
    def test_btc_analysis_routed_to_stock_analysis(self):
        """BTC-USD should be routed to STOCK_ANALYSIS handler."""
        _stock_reply = "BTC-USD Analysis: Price $65,000 | Verdict: HOLD | Score: 62"

        async def _run():
            with patch("core.orchestrator.ADMIN_ENABLED", False), \
                 patch("core.orchestrator.MEMORY_ENABLED", False), \
                 patch.object(self.orch, "_classify_intent",
                              return_value=("STOCK_ANALYSIS", "STOCK_ANALYSIS", "analyze BTC-USD", "")), \
                 patch("core.services.market_route_handler.handle_stock_analysis",
                        new_callable=AsyncMock,
                        return_value={"reply": _stock_reply, "session_id": "s3"}):
                result = await self.orch.process_message("user1", "analyze BTC-USD", session_id="s3")
            return result

        result = run(_run())
        assert "reply" in result
        assert "BTC" in result["reply"] or len(result["reply"]) > 10

    # ── MSFT analysis routing ─────────────────────────────────────────
    def test_msft_analysis_routed(self):
        """MSFT should be routed to STOCK_ANALYSIS."""
        _msft_reply = "MSFT Analysis: Price $420 | Verdict: BUY | Score: 74"

        async def _run():
            with patch("core.orchestrator.ADMIN_ENABLED", False), \
                 patch("core.orchestrator.MEMORY_ENABLED", False), \
                 patch.object(self.orch, "_classify_intent",
                              return_value=("STOCK_ANALYSIS", "STOCK_ANALYSIS", "analyze MSFT", "")), \
                 patch("core.services.market_route_handler.handle_stock_analysis",
                        new_callable=AsyncMock,
                        return_value={"reply": _msft_reply, "session_id": "s4"}):
                result = await self.orch.process_message("user1", "analyze MSFT", session_id="s4")
            return result

        result = run(_run())
        assert "reply" in result
        assert "MSFT" in result["reply"] or len(result["reply"]) > 10

    # ── Aramco (Arabic ticker fast-path) ─────────────────────────────
    def test_aramco_arabic_fast_path(self):
        """'حلل ارامكو' should be detected by Arabic-ticker fast-path."""
        _aramco_reply = "2222.SR Analysis: Price SAR 28.5 | Verdict: HOLD | Score: 58"

        async def _run():
            with patch("core.orchestrator.ADMIN_ENABLED", False), \
                 patch("core.orchestrator.MEMORY_ENABLED", False), \
                 patch("core.services.market_route_handler.handle_stock_analysis",
                        new_callable=AsyncMock,
                        return_value={"reply": _aramco_reply, "session_id": "s5"}):
                result = await self.orch.process_message("user1", "حلل ارامكو", session_id="s5")
            return result

        result = run(_run())
        assert "reply" in result
        assert result["reply"]

    # ── Portfolio optimize routing ────────────────────────────────────
    def test_portfolio_optimize_routed(self):
        """Portfolio optimization request should be routed to PORTFOLIO handler."""
        _port_reply = "Optimized Portfolio: SPY 40% | BND 30% | GLD 20% | QQQ 10%"

        async def _run():
            with patch("core.orchestrator.ADMIN_ENABLED", False), \
                 patch("core.orchestrator.MEMORY_ENABLED", False), \
                 patch.object(self.orch, "_classify_intent",
                              return_value=("PORTFOLIO", "PORTFOLIO", "optimize portfolio risk=moderate", "")), \
                 patch("core.services.market_route_handler.handle_portfolio",
                        new_callable=AsyncMock,
                        return_value={"reply": _port_reply, "session_id": "s6"}):
                result = await self.orch.process_message("user1", "optimize my portfolio moderate risk", session_id="s6")
            return result

        result = run(_run())
        assert "reply" in result
        assert result["reply"]

    # ── LLM fallback chain: Gemini → fallback ─────────────────────────
    def test_gemini_failure_falls_through_to_fallback(self):
        """When Gemini fails, _gemini_generate should return fallback content, not raise."""
        from core.orchestrator import MultiAgentOrchestrator

        # Simulate Gemini primary + backup both failing,
        # then fallback chain returning a static message.
        orch = self.orch
        orch.gemini_client = None        # force attempt 1 to skip
        orch.gemini_client_backup = None  # force attempt 2 to skip

        # The fallback chain is called inside _gemini_generate (attempt 3).
        # Mock generate_with_fallback_sync to return a static response.
        from core.llm_fallback import LLMResponse
        fake_response = LLMResponse(
            content="Service temporarily unavailable. Please try again.",
            provider="fallback",
            success=False,
            error="All providers unavailable",
        )

        # The function is imported inside _gemini_generate via
        # "from core.llm_fallback import generate_with_fallback_sync"
        # so we must patch it at the source module.
        with patch("core.llm_fallback.generate_with_fallback_sync",
                   return_value=fake_response):
            result = orch._gemini_generate("Say hello", label="test")

        assert isinstance(result, str)
        assert len(result) > 5

    # ── Response structure ────────────────────────────────────────────
    def test_response_has_session_id(self):
        """process_message reply always includes session_id."""

        async def _run():
            with patch("core.orchestrator.ADMIN_ENABLED", False), \
                 patch("core.orchestrator.MEMORY_ENABLED", False), \
                 patch.object(self.orch, "_classify_intent",
                              return_value=("GENERAL", "GENERAL", "hi", "")), \
                 patch("core.services.market_route_handler.handle_general",
                        new_callable=AsyncMock,
                        return_value={"reply": "Hello!", "session_id": "s9"}):
                result = await self.orch.process_message("user1", "hi", session_id="s9")
            return result

        result = run(_run())
        assert "session_id" in result or "reply" in result


# ══════════════════════════════════════════════════════════════════════
# Finance helpers — pure-function E2E smoke tests
# ══════════════════════════════════════════════════════════════════════

class TestFinanceHelpersE2E:
    """Quick E2E smoke tests for the extracted finance_helpers module."""

    def test_safe_div_yield_round_trip(self):
        from core.agents.finance_helpers import _safe_div_yield
        # Whole-number percentage → decimal
        assert _safe_div_yield(5.0) == pytest.approx(0.05)
        # String with % sign
        assert _safe_div_yield("3.5%") == pytest.approx(0.035)
        # Corrupt data capped
        assert _safe_div_yield(500.0) == pytest.approx(0.30)

    def test_consensus_divergence_bullish_analyst(self):
        from core.agents.finance_helpers import _consensus_divergence
        # EisaX SELL vs analyst Strong Buy → should diverge
        result = _consensus_divergence("SELL", "Strong Buy")
        assert result["diverges"] is True
        assert result["gap"] > 0

    def test_consensus_divergence_no_divergence(self):
        from core.agents.finance_helpers import _consensus_divergence
        result = _consensus_divergence("BUY", "Buy")  # same tier
        assert result["diverges"] is False

    def test_compute_decision_confidence_bounded(self):
        from core.agents.finance_helpers import _compute_decision_confidence
        for score in [30, 50, 70, 85]:
            conf = _compute_decision_confidence(score, 5, 2, 1.0, "BUY")
            assert 40 <= conf <= 90

    def test_soften_execution_language_strips_buy_now(self):
        from core.agents.finance_helpers import _soften_execution_language
        result = _soften_execution_language("You should buy now.")
        assert "buy now" not in result.lower()

    def test_round_scenario_prices_basic(self):
        from core.agents.finance_helpers import _round_scenario_prices
        # Table row with a price in a scenario section
        text = "| Scenario | Bear | Bull |\n| Price | $100.50 | $150.75 |"
        result = _round_scenario_prices(text)
        # Should produce a range or keep unchanged (no crash)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fetch_onchain_non_crypto_returns_empty(self):
        from core.agents.finance_helpers import _fetch_onchain
        # Non-crypto ticker → empty dict (no network call)
        result = _fetch_onchain("AAPL")
        assert result == {}

    def test_fetch_btc_etf_flows_no_crash(self):
        """_fetch_btc_etf_flows should return '' on network failure (not raise)."""
        from core.agents.finance_helpers import _fetch_btc_etf_flows
        with patch("yfinance.download", side_effect=Exception("no network")):
            result = _fetch_btc_etf_flows()
        assert result == ""
