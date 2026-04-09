"""
Tests for hybrid RAG retrieval in core/vector_memory.py.
"""
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeCollection:
    def __init__(self, name, docs):
        self.name = name
        self._docs = list(docs)

    def count(self):
        return len(self._docs)

    def get(self, include=None):
        return {
            "ids": [doc["id"] for doc in self._docs],
            "documents": [doc["text"] for doc in self._docs],
            "metadatas": [doc["metadata"] for doc in self._docs],
        }

    def query(self, query_texts, n_results, where=None):
        docs = self._docs
        if where:
            docs = [
                doc for doc in docs
                if all(doc["metadata"].get(key) == value for key, value in where.items())
            ]
        docs = sorted(docs, key=lambda doc: doc["distance"])[:n_results]
        return {
            "ids": [[doc["id"] for doc in docs]],
            "documents": [[doc["text"] for doc in docs]],
            "metadatas": [[doc["metadata"] for doc in docs]],
            "distances": [[doc["distance"] for doc in docs]],
        }


@pytest.fixture
def vector_memory_module():
    import core.vector_memory as vm
    return vm


def _install_fake_collections(monkeypatch, vm, stock_docs, world_docs=None, lesson_docs=None):
    collections = {
        "stock_analyses": FakeCollection("stock_analyses", stock_docs),
        "world_knowledge": FakeCollection("world_knowledge", world_docs or []),
        "learning_log": FakeCollection("learning_log", lesson_docs or []),
    }

    monkeypatch.setattr(vm, "_get_collection", lambda name: collections[name])
    return collections


def _precision_at(results, relevant_tickers, k):
    top = results[:k]
    if not top:
        return 0.0
    hits = sum(1 for row in top if row["ticker"] in relevant_tickers)
    return hits / float(k)


def test_hybrid_vs_pure_vector_returns_different_ordering(vector_memory_module, monkeypatch):
    vm = vector_memory_module
    stock_docs = [
        {
            "id": "aapl",
            "text": "[AAPL] graphics processors and cloud infrastructure outlook",
            "metadata": {"ticker": "AAPL", "verdict": "HOLD", "date": "2026-01-01"},
            "distance": 0.15,
        },
        {
            "id": "jpm",
            "text": "[JPM] bank dividend yield and capital return outlook",
            "metadata": {"ticker": "JPM", "verdict": "BUY", "date": "2026-01-02"},
            "distance": 0.18,
        },
        {
            "id": "wfc",
            "text": "[WFC] regional bank dividend policy and deposit growth",
            "metadata": {"ticker": "WFC", "verdict": "BUY", "date": "2026-01-03"},
            "distance": 0.22,
        },
    ]
    _install_fake_collections(monkeypatch, vm, stock_docs)

    pure = vm.search_similar_analyses("bank dividend yield", n=3)
    hybrid = vm.search_similar_analyses("bank dividend yield", n=3, prefer_hybrid=True)

    assert [row["ticker"] for row in pure] == ["AAPL", "JPM", "WFC"]
    assert [row["ticker"] for row in hybrid] != [row["ticker"] for row in pure]
    assert hybrid[0]["ticker"] == "JPM"
    assert hybrid[0]["search_mode"].startswith("hybrid_")


def test_fallback_to_pure_vector_when_bm25_unavailable(vector_memory_module, monkeypatch, caplog):
    vm = vector_memory_module
    stock_docs = [
        {
            "id": "aapl",
            "text": "[AAPL] graphics processors and cloud infrastructure outlook",
            "metadata": {"ticker": "AAPL", "verdict": "HOLD", "date": "2026-01-01"},
            "distance": 0.15,
        },
        {
            "id": "jpm",
            "text": "[JPM] bank dividend yield and capital return outlook",
            "metadata": {"ticker": "JPM", "verdict": "BUY", "date": "2026-01-02"},
            "distance": 0.18,
        },
    ]
    _install_fake_collections(monkeypatch, vm, stock_docs)
    monkeypatch.setattr(vm, "_keyword_search_is_available", lambda: False)

    with caplog.at_level(logging.INFO):
        context = vm.get_rag_context("bank dividend yield")

    assert "SIMILAR PAST ANALYSES" in context
    assert "[AAPL]" in context
    assert "mode=vector_only" in caplog.text


def test_reranking_produces_better_precision_on_known_queries(vector_memory_module, monkeypatch):
    vm = vector_memory_module
    stock_docs = [
        {
            "id": "nvda",
            "text": "[NVDA] gpu accelerators and data center demand remain strong",
            "metadata": {"ticker": "NVDA", "verdict": "BUY", "date": "2026-01-01"},
            "distance": 0.17,
        },
        {
            "id": "jpm",
            "text": "[JPM] bank dividend yield and capital return outlook",
            "metadata": {"ticker": "JPM", "verdict": "BUY", "date": "2026-01-02"},
            "distance": 0.18,
        },
        {
            "id": "xom",
            "text": "[XOM] oil production outlook and refinery margins",
            "metadata": {"ticker": "XOM", "verdict": "HOLD", "date": "2026-01-03"},
            "distance": 0.19,
        },
        {
            "id": "bac",
            "text": "[BAC] bank dividend yield strength across consumer bank deposits",
            "metadata": {"ticker": "BAC", "verdict": "BUY", "date": "2026-01-04"},
            "distance": 0.1805,
        },
    ]
    _install_fake_collections(monkeypatch, vm, stock_docs)

    pure = vm.search_similar_analyses("bank dividend yield", n=4)
    hybrid = vm.search_similar_analyses("bank dividend yield", n=4, prefer_hybrid=True)

    relevant = {"JPM", "BAC"}
    pure_precision = _precision_at(pure, relevant, k=2)
    hybrid_precision = _precision_at(hybrid, relevant, k=2)

    assert pure_precision == 0.5
    assert hybrid_precision > pure_precision
    assert {row["ticker"] for row in hybrid[:2]} == relevant
