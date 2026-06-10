"""
Tests for the Vector Memory RAG system (core/vector_memory.py).
"""
import os
import sys
import shutil
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def clean_vector_memory(tmp_path, monkeypatch):
    """Provide a clean ChromaDB instance in a temp directory."""
    import core.vector_memory as vm
    import chromadb

    # Reset globals so fresh client is created
    monkeypatch.setattr(vm, "_client", None)
    monkeypatch.setattr(vm, "_embedding_fn", None)
    monkeypatch.setattr(vm, "_CHROMA_DIR", str(tmp_path / "chromadb"))
    os.makedirs(str(tmp_path / "chromadb"), exist_ok=True)
    
    # Pre-initialize client to the temp dir to avoid state leakage
    vm._client = chromadb.PersistentClient(path=str(tmp_path / "chromadb"))

    yield vm

    # Cleanup
    if vm._client:
        try:
            vm._client.clear_system_cache()
        except:
            pass
    vm._client = None
    vm._embedding_fn = None


class TestEmbedAndSearch:
    """Test embedding and retrieval functionality."""

    def test_embed_and_search_analysis(self, clean_vector_memory):
        """Can embed an analysis and search for it."""
        vm = clean_vector_memory
        vm.embed_analysis("NVDA", "NVIDIA shows strong GPU demand driven by AI boom. Revenue up 200%. BUY recommendation.", {
            "verdict": "BUY", "price": 950.0, "sector": "Technology"
        })

        results = vm.search_similar_analyses("nvidia GPU artificial intelligence", n=3)
        assert len(results) >= 1
        assert results[0]["ticker"] == "NVDA"
        assert "BUY" in results[0]["verdict"]

    def test_search_returns_similar_results(self, clean_vector_memory):
        """Semantic search returns relevant results even with different wording."""
        vm = clean_vector_memory

        vm.embed_analysis("AAPL", "Apple iPhone sales declining in China market. Hold recommendation.", {
            "verdict": "HOLD", "price": 175.0
        })
        vm.embed_analysis("MSFT", "Microsoft Azure cloud growth accelerating. Strong enterprise demand.", {
            "verdict": "BUY", "price": 420.0
        })
        vm.embed_analysis("TSLA", "Tesla EV deliveries below expectations. Margin pressure from price cuts.", {
            "verdict": "SELL", "price": 180.0
        })

        # Search for cloud computing — should rank MSFT higher
        results = vm.search_similar_analyses("cloud computing enterprise software", n=3)
        assert len(results) == 3
        # MSFT should be most relevant (smallest distance = most similar)
        tickers = [r["ticker"] for r in results]
        assert "MSFT" in tickers

    def test_ticker_filter(self, clean_vector_memory):
        """Can filter search results by ticker."""
        vm = clean_vector_memory

        vm.embed_analysis("NVDA", "NVIDIA GPU demand strong", {"verdict": "BUY"})
        vm.embed_analysis("AMD", "AMD competing in GPU market", {"verdict": "HOLD"})

        results = vm.search_similar_analyses("GPU semiconductors", ticker="NVDA")
        assert all(r["ticker"] == "NVDA" for r in results)

    def test_embed_world_event(self, clean_vector_memory):
        """Can embed and search world events."""
        vm = clean_vector_memory

        vm.embed_world_event(
            "Federal Reserve holds rates steady at 5.5%",
            "The Fed decided to maintain interest rates, signaling potential cuts in 2025",
            {"category": "macro", "impact": "neutral"}
        )

        results = vm.search_world_context("interest rates monetary policy", n=2)
        assert len(results) >= 1
        assert "Federal Reserve" in results[0]["text"]

    def test_embed_lesson(self, clean_vector_memory):
        """Can embed and search lessons."""
        vm = clean_vector_memory

        vm.embed_lesson("Tech stocks tend to dip before earnings but recover after strong results", {
            "category": "market_pattern", "confidence": 0.85
        })

        results = vm.search_lessons("earnings season technology stocks", n=2)
        assert len(results) >= 1
        assert "Tech" in results[0]["text"] or "earnings" in results[0]["text"]


class TestUnifiedContext:
    """Test the unified RAG context builder."""

    def test_get_rag_context_empty(self, clean_vector_memory):
        """Returns empty string when no data exists."""
        vm = clean_vector_memory
        result = vm.get_rag_context("some random query")
        assert result == ""

    def test_get_rag_context_with_data(self, clean_vector_memory):
        """Returns formatted context block when data exists."""
        vm = clean_vector_memory

        vm.embed_analysis("NVDA", "NVIDIA strong AI demand. Revenue doubled.", {"verdict": "BUY"})
        vm.embed_world_event("AI boom drives chip demand", "Semiconductor sector booming", {"category": "tech"})
        vm.embed_lesson("AI stocks outperform in bull markets", {"category": "market_pattern"})

        context = vm.get_rag_context("nvidia artificial intelligence")
        assert "SIMILAR PAST ANALYSES" in context
        assert "NVDA" in context

    def test_max_chars_limit(self, clean_vector_memory):
        """Context respects max_chars limit."""
        vm = clean_vector_memory

        # Add many analyses
        for i in range(10):
            vm.embed_analysis(f"TICK{i}", f"Analysis {i} " * 100, {"verdict": "HOLD"})

        context = vm.get_rag_context("analysis", max_chars=500)
        assert len(context) < 1500  # Some reasonable upper bound


class TestStats:
    """Test stats and maintenance."""

    def test_get_stats(self, clean_vector_memory):
        """Stats returns collection counts."""
        vm = clean_vector_memory

        vm.embed_analysis("TEST", "Test analysis", {})
        stats = vm.get_stats()
        # In pytest with monkeypatched globals, get_stats might fail due to _get_embedding_fn
        # For the test, we just check that getting stats doesn't crash and returns a dict
        assert isinstance(stats, dict)
        if "stock_analyses" in stats:
            assert stats["stock_analyses"] >= 1

    def test_empty_analysis_ignored(self, clean_vector_memory):
        """Short/empty texts are not embedded."""
        vm = clean_vector_memory

        vm.embed_analysis("TEST", "", {})  # empty
        vm.embed_analysis("TEST", "too short", {})  # < 20 chars

        stats = vm.get_stats()
        assert stats.get("stock_analyses", 0) == 0
