"""
EisaX RAG Vector Memory
Semantic search over past analyses, world events, and lessons using ChromaDB.

Collections:
  - stock_analyses: Embedded stock analysis text with metadata (ticker, verdict, price, date)
  - world_knowledge: News headlines + summaries with metadata (category, impact, date)
  - learning_log: AI self-learning lessons with metadata (category, confidence, date)

Usage:
    from core.vector_memory import embed_analysis, search_similar_analyses, get_rag_context

    # After every stock analysis:
    embed_analysis("NVDA", analysis_text, {"verdict": "BUY", "price": 950.0, "sector": "Tech"})

    # Before generating a new analysis — retrieve similar past work:
    context = get_rag_context("nvidia earnings outlook GPU demand", ticker="NVDA")
"""
import os
import logging
import hashlib
import math
import re
import time
from collections import Counter
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from core.metrics import track_performance

logger = logging.getLogger(__name__)

# ── ChromaDB Setup ───────────────────────────────────────────────────────────
_CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chromadb")
os.makedirs(_CHROMA_DIR, exist_ok=True)

_client = None
_embedding_fn = None
_BM25_CLASS = None
_BM25_IMPORT_ATTEMPTED = False

# Max documents per collection (auto-prune oldest beyond this)
MAX_COLLECTION_SIZE = 10_000
# Max text length to embed (ChromaDB has limits, and long texts reduce quality)
MAX_EMBED_LENGTH = 2000


def _get_client():
    """Lazy-init ChromaDB persistent client."""
    global _client
    if _client is None:
        try:
            import chromadb
            _client = chromadb.PersistentClient(path=_CHROMA_DIR)
            logger.info("[VectorMemory] ChromaDB initialized at %s", _CHROMA_DIR)
        except Exception as e:
            logger.error("[VectorMemory] ChromaDB init failed: %s", e)
            raise
    return _client


def _get_embedding_fn():
    """Lazy-init sentence-transformers embedding function."""
    global _embedding_fn
    if _embedding_fn is None:
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            _embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("[VectorMemory] Embedding model loaded: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error("[VectorMemory] Embedding model failed: %s", e)
            raise
    return _embedding_fn


def _get_collection(name: str):
    """Get or create a ChromaDB collection with the embedding function."""
    client = _get_client()
    ef = _get_embedding_fn()
    return client.get_or_create_collection(
        name=name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def _make_id(text: str, prefix: str = "") -> str:
    """Generate a deterministic ID from text content."""
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}_{h}" if prefix else h


def _truncate(text: str, max_len: int = MAX_EMBED_LENGTH) -> str:
    """Truncate text to max_len for embedding quality."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _tokenize(text: str) -> List[str]:
    """Tokenize text for keyword search."""
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def _normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to a 0..1 range."""
    if not scores:
        return []

    low = min(scores)
    high = max(scores)
    if math.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in scores]

    spread = high - low
    return [(score - low) / spread for score in scores]


def _metadata_matches(metadata: Optional[Dict], where_filter: Optional[Dict]) -> bool:
    """Apply a simple equality-only metadata filter client-side."""
    if not where_filter:
        return True

    metadata = metadata or {}
    for key, expected in where_filter.items():
        if metadata.get(key) != expected:
            return False
    return True


def _get_keyword_backend():
    """Return BM25 backend when installed."""
    global _BM25_CLASS, _BM25_IMPORT_ATTEMPTED
    if not _BM25_IMPORT_ATTEMPTED:
        _BM25_IMPORT_ATTEMPTED = True
        try:
            from rank_bm25 import BM25Okapi
            _BM25_CLASS = BM25Okapi
        except Exception:
            _BM25_CLASS = None
    return _BM25_CLASS


def _keyword_search_is_available() -> bool:
    """Whether hybrid keyword search can run."""
    return True


def _score_with_tfidf(query_tokens: List[str], tokenized_docs: List[List[str]]) -> List[float]:
    """Simple TF-IDF fallback when rank_bm25 is unavailable."""
    if not query_tokens or not tokenized_docs:
        return [0.0 for _ in tokenized_docs]

    doc_count = len(tokenized_docs)
    doc_freq = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    query_tf = Counter(query_tokens)
    scores = []
    for tokens in tokenized_docs:
        if not tokens:
            scores.append(0.0)
            continue

        tf = Counter(tokens)
        length = float(len(tokens))
        score = 0.0
        for term, query_weight in query_tf.items():
            term_freq = tf.get(term, 0)
            if not term_freq:
                continue
            idf = math.log((1.0 + doc_count) / (1.0 + doc_freq.get(term, 0))) + 1.0
            score += (term_freq / length) * idf * query_weight
        scores.append(score)
    return scores


def _score_keyword_matches(query: str, documents: List[str]) -> Tuple[str, List[float]]:
    """Score documents with BM25 or TF-IDF fallback."""
    query_tokens = _tokenize(query)
    tokenized_docs = [_tokenize(doc) for doc in documents]

    if not query_tokens or not tokenized_docs:
        return "tfidf", [0.0 for _ in documents]

    bm25_cls = _get_keyword_backend()
    if bm25_cls is not None:
        scorer = bm25_cls(tokenized_docs)
        return "bm25", [float(score) for score in scorer.get_scores(query_tokens)]

    return "tfidf", _score_with_tfidf(query_tokens, tokenized_docs)


def _fetch_collection_records(collection, where_filter: Optional[Dict] = None) -> List[Dict]:
    """Fetch collection records for keyword scoring."""
    all_docs = collection.get(include=["documents", "metadatas"])
    records = []

    for i, doc_id in enumerate(all_docs.get("ids", [])):
        metadata = all_docs["metadatas"][i] if all_docs.get("metadatas") else {}
        if not _metadata_matches(metadata, where_filter):
            continue
        records.append({
            "id": doc_id,
            "text": all_docs["documents"][i] if all_docs.get("documents") else "",
            "metadata": metadata or {},
        })

    return records


def _vector_search_results(collection, query: str, n: int = 5, where_filter: Optional[Dict] = None) -> List[Dict]:
    """Run the existing vector search path."""
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n, count),
        where=where_filter,
    )

    output = []
    docs = results.get("documents", [[]])
    for i, doc in enumerate(docs[0] if docs else []):
        metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
        distance = results["distances"][0][i] if results.get("distances") else 0.0
        ids = results.get("ids", [[]])
        output.append({
            "id": ids[0][i] if ids and ids[0] else None,
            "text": doc,
            "metadata": metadata or {},
            "distance": float(distance),
            "vector_score": None,
            "keyword_score": None,
            "hybrid_score": None,
            "search_mode": "vector_only",
        })
    return output


def _hybrid_search_results(collection, query: str, n: int = 5, where_filter: Optional[Dict] = None) -> Tuple[List[Dict], str]:
    """Combine vector and keyword scores, then rerank."""
    if not _keyword_search_is_available():
        return _vector_search_results(collection, query, n=n, where_filter=where_filter), "vector_only"

    records = _fetch_collection_records(collection, where_filter=where_filter)
    if not records:
        return [], "vector_only"

    vector_limit = min(max(n * 3, 10), len(records))
    vector_candidates = _vector_search_results(collection, query, n=vector_limit, where_filter=where_filter)
    if not vector_candidates:
        return [], "vector_only"

    keyword_backend, keyword_scores = _score_keyword_matches(query, [record["text"] for record in records])
    if not keyword_scores:
        return vector_candidates[:n], "vector_only"

    keyword_by_id = {
        record["id"]: float(score)
        for record, score in zip(records, keyword_scores)
    }
    keyword_ranked_ids = [
        record["id"]
        for record, _ in sorted(
            zip(records, keyword_scores),
            key=lambda item: item[1],
            reverse=True,
        )[:vector_limit]
    ]

    vector_by_id = {candidate["id"]: candidate for candidate in vector_candidates}
    records_by_id = {record["id"]: record for record in records}

    candidate_ids = []
    for doc_id in list(vector_by_id.keys()) + keyword_ranked_ids:
        if doc_id not in candidate_ids:
            candidate_ids.append(doc_id)

    candidate_rows = []
    vector_raw_scores = []
    keyword_raw_scores = []
    for doc_id in candidate_ids:
        vector_candidate = vector_by_id.get(doc_id)
        record = records_by_id.get(doc_id)
        if not record:
            continue

        distance = float(vector_candidate["distance"]) if vector_candidate else float("inf")
        vector_raw = 1.0 / (1.0 + max(distance, 0.0)) if math.isfinite(distance) else 0.0
        keyword_raw = float(keyword_by_id.get(doc_id, 0.0))

        candidate_rows.append({
            "id": doc_id,
            "text": record["text"],
            "metadata": record["metadata"],
            "distance": round(distance, 4) if math.isfinite(distance) else None,
            "_vector_raw": vector_raw,
            "_keyword_raw": keyword_raw,
        })
        vector_raw_scores.append(vector_raw)
        keyword_raw_scores.append(keyword_raw)

    vector_norm = _normalize_scores(vector_raw_scores)
    keyword_norm = _normalize_scores(keyword_raw_scores)

    for row, vector_score, keyword_score in zip(candidate_rows, vector_norm, keyword_norm):
        row["vector_score"] = round(vector_score, 4)
        row["keyword_score"] = round(keyword_score, 4)
        row["hybrid_score"] = round((0.6 * vector_score) + (0.4 * keyword_score), 4)
        row["search_mode"] = f"hybrid_{keyword_backend}"
        del row["_vector_raw"]
        del row["_keyword_raw"]

    candidate_rows.sort(
        key=lambda row: (
            row["hybrid_score"],
            row["vector_score"],
            row["keyword_score"],
        ),
        reverse=True,
    )
    return candidate_rows[:n], f"hybrid_{keyword_backend}"


# ── Embed Functions ──────────────────────────────────────────────────────────

def embed_analysis(ticker: str, analysis_text: str, metadata: Optional[Dict] = None):
    """
    Embed a stock analysis into the vector store.

    Args:
        ticker: Stock ticker (e.g., "NVDA")
        analysis_text: Full analysis text
        metadata: Optional dict with keys like verdict, price, sector, date
    """
    if not analysis_text or len(analysis_text.strip()) < 20:
        return

    try:
        collection = _get_collection("stock_analyses")
        ticker = ticker.upper()
        now = datetime.now().isoformat()

        meta = {
            "ticker": ticker,
            "date": now,
            "verdict": (metadata or {}).get("verdict", ""),
            "sector": (metadata or {}).get("sector", ""),
        }

        # Add price as string (ChromaDB metadata doesn't support all types via filter)
        price = (metadata or {}).get("price")
        if price:
            meta["price"] = str(round(float(price), 2))

        doc_id = _make_id(f"{ticker}_{now}", prefix="analysis")
        text = _truncate(f"[{ticker}] {analysis_text}")

        # Ensure collection operations are synced in chromadb 
        collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )
        logger.info("[VectorMemory] Embedded analysis: %s (%d chars)", ticker, len(text))

        # Auto-prune if collection too large
        _prune_collection(collection, MAX_COLLECTION_SIZE)

    except Exception as e:
        logger.warning("[VectorMemory] embed_analysis failed: %s", e)


def embed_world_event(headline: str, summary: str, metadata: Optional[Dict] = None):
    """
    Embed a world news event into the vector store.

    Args:
        headline: News headline
        summary: News summary text
        metadata: Optional dict with keys like category, impact, affected_tickers
    """
    if not headline:
        return

    try:
        collection = _get_collection("world_knowledge")
        now = datetime.now().isoformat()

        meta = {
            "date": now,
            "category": (metadata or {}).get("category", "general"),
            "impact": (metadata or {}).get("impact", "neutral"),
        }

        tickers = (metadata or {}).get("affected_tickers")
        if tickers:
            meta["tickers"] = ",".join(tickers) if isinstance(tickers, list) else str(tickers)

        doc_id = _make_id(headline, prefix="world")
        text = _truncate(f"{headline}\n{summary}")

        collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )
        logger.debug("[VectorMemory] Embedded world event: %s", headline[:60])

    except Exception as e:
        logger.warning("[VectorMemory] embed_world_event failed: %s", e)


def embed_lesson(lesson: str, metadata: Optional[Dict] = None):
    """
    Embed a learning log lesson into the vector store.

    Args:
        lesson: Lesson text
        metadata: Optional dict with keys like category, confidence
    """
    if not lesson:
        return

    try:
        collection = _get_collection("learning_log")
        now = datetime.now().isoformat()

        meta = {
            "date": now,
            "category": (metadata or {}).get("category", "general"),
        }

        confidence = (metadata or {}).get("confidence")
        if confidence is not None:
            meta["confidence"] = str(round(float(confidence), 2))

        doc_id = _make_id(lesson, prefix="lesson")
        text = _truncate(lesson)

        collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )
        logger.debug("[VectorMemory] Embedded lesson: %s", lesson[:60])

    except Exception as e:
        logger.warning("[VectorMemory] embed_lesson failed: %s", e)


# ── Search Functions ─────────────────────────────────────────────────────────

def search_similar_analyses(
    query: str,
    n: int = 5,
    ticker: str = None,
    prefer_hybrid: bool = False,
) -> List[Dict]:
    """
    Search for semantically similar past stock analyses.

    Args:
        query: Search query (natural language)
        n: Number of results to return
        ticker: Optional ticker to filter by

    Returns:
        List of dicts with keys: text, ticker, verdict, price, date, distance
    """
    try:
        collection = _get_collection("stock_analyses")
        where_filter = {"ticker": ticker.upper()} if ticker else None
        if collection.count() == 0:
            return []

        if prefer_hybrid:
            rows, _ = _hybrid_search_results(collection, query, n=n, where_filter=where_filter)
        else:
            rows = _vector_search_results(collection, query, n=n, where_filter=where_filter)

        output = []
        for row in rows:
            meta = row["metadata"]
            output.append({
                "text": row["text"],
                "ticker": meta.get("ticker", ""),
                "verdict": meta.get("verdict", ""),
                "price": meta.get("price", ""),
                "date": meta.get("date", ""),
                "distance": row["distance"],
                "vector_score": row.get("vector_score"),
                "keyword_score": row.get("keyword_score"),
                "hybrid_score": row.get("hybrid_score"),
                "search_mode": row.get("search_mode", "vector_only"),
            })
        return output

    except Exception as e:
        logger.warning("[VectorMemory] search_similar_analyses failed: %s", e)
        return []


def search_world_context(query: str, n: int = 3, prefer_hybrid: bool = False) -> List[Dict]:
    """
    Search for semantically relevant world events.

    Args:
        query: Search query
        n: Number of results

    Returns:
        List of dicts with keys: text, category, impact, date, distance
    """
    try:
        collection = _get_collection("world_knowledge")
        if collection.count() == 0:
            return []
        if prefer_hybrid:
            rows, _ = _hybrid_search_results(collection, query, n=n)
        else:
            rows = _vector_search_results(collection, query, n=n)

        output = []
        for row in rows:
            meta = row["metadata"]
            output.append({
                "text": row["text"],
                "category": meta.get("category", ""),
                "impact": meta.get("impact", ""),
                "date": meta.get("date", ""),
                "distance": row["distance"],
                "vector_score": row.get("vector_score"),
                "keyword_score": row.get("keyword_score"),
                "hybrid_score": row.get("hybrid_score"),
                "search_mode": row.get("search_mode", "vector_only"),
            })
        return output

    except Exception as e:
        logger.warning("[VectorMemory] search_world_context failed: %s", e)
        return []


def search_lessons(query: str, n: int = 3, prefer_hybrid: bool = False) -> List[Dict]:
    """Search for relevant lessons learned by EisaX."""
    try:
        collection = _get_collection("learning_log")
        if collection.count() == 0:
            return []
        if prefer_hybrid:
            rows, _ = _hybrid_search_results(collection, query, n=n)
        else:
            rows = _vector_search_results(collection, query, n=n)

        output = []
        for row in rows:
            meta = row["metadata"]
            output.append({
                "text": row["text"],
                "category": meta.get("category", ""),
                "confidence": meta.get("confidence", ""),
                "date": meta.get("date", ""),
                "vector_score": row.get("vector_score"),
                "keyword_score": row.get("keyword_score"),
                "hybrid_score": row.get("hybrid_score"),
                "search_mode": row.get("search_mode", "vector_only"),
            })
        return output

    except Exception as e:
        logger.warning("[VectorMemory] search_lessons failed: %s", e)
        return []


# ── Unified RAG Context Builder ──────────────────────────────────────────────

@track_performance
def get_rag_context(query: str, ticker: str = None, max_chars: int = 1500) -> str:
    """
    Build a RAG context block for prompt injection.

    Searches all 3 collections and formats results as a text block
    ready to be injected into an LLM prompt.

    Args:
        query: The user's question or analysis request
        ticker: Optional ticker to prioritize
        max_chars: Maximum total context length

    Returns:
        Formatted string with relevant past analyses, news, and lessons.
        Empty string if no relevant results found.
    """
    start = time.perf_counter()
    parts = []
    total_chars = 0
    modes = []

    # 1. Similar past analyses (most valuable)
    analyses = search_similar_analyses(query, n=5, ticker=ticker, prefer_hybrid=True)
    if analyses:
        modes.extend(a.get("search_mode") for a in analyses if a.get("search_mode"))
        # Also search without ticker filter for cross-stock insights
        if ticker:
            cross = search_similar_analyses(query, n=3, prefer_hybrid=True)
            seen_tickers = {a["ticker"] for a in analyses}
            for c in cross:
                if c["ticker"] not in seen_tickers:
                    analyses.append(c)

        ranked_analyses = sorted(
            analyses,
            key=lambda item: (
                item.get("hybrid_score", -1.0),
                item.get("vector_score", -1.0),
                -(item.get("distance") or 9999.0),
            ),
            reverse=True,
        )

        analysis_lines = ["📚 SIMILAR PAST ANALYSES (RAG):"]
        for a in ranked_analyses[:5]:
            line = f"  [{a['ticker']}] {a['verdict']} — {a['text'][:200]}..."
            if a.get("date"):
                line += f" ({a['date'][:10]})"
            analysis_lines.append(line)
            total_chars += len(line)
            if total_chars > max_chars * 0.6:
                break
        parts.append("\n".join(analysis_lines))

    # 2. Relevant world events
    if total_chars < max_chars * 0.8:
        world = search_world_context(query, n=3, prefer_hybrid=True)
        if world:
            modes.extend(w.get("search_mode") for w in world if w.get("search_mode"))
            world_lines = ["🌍 RELEVANT WORLD CONTEXT (RAG):"]
            for w in world:
                line = f"  [{w['category'].upper()}] {w['text'][:150]}"
                world_lines.append(line)
                total_chars += len(line)
                if total_chars > max_chars * 0.9:
                    break
            parts.append("\n".join(world_lines))

    # 3. Relevant lessons (if space)
    if total_chars < max_chars * 0.9:
        lessons = search_lessons(query, n=2, prefer_hybrid=True)
        if lessons:
            modes.extend(l.get("search_mode") for l in lessons if l.get("search_mode"))
            lesson_lines = ["🎓 EISAX LESSONS LEARNED (RAG):"]
            for l in lessons:
                lesson_lines.append(f"  - {l['text'][:100]}")
            parts.append("\n".join(lesson_lines))

    mode_label = "vector_only"
    distinct_modes = sorted(set(modes))
    if len(distinct_modes) == 1:
        mode_label = distinct_modes[0]
    elif len(distinct_modes) > 1:
        mode_label = f"mixed:{','.join(distinct_modes)}"

    latency_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "[VectorMemory] get_rag_context mode=%s latency_ms=%.2f",
        mode_label,
        latency_ms,
    )

    if not parts:
        return ""

    return "\n\n".join(parts)


# ── Maintenance ──────────────────────────────────────────────────────────────

def _prune_collection(collection, max_size: int):
    """Remove oldest documents if collection exceeds max_size."""
    try:
        count = collection.count()
        if count <= max_size:
            return

        overflow = count - max_size
        # Get all IDs sorted by date metadata
        all_docs = collection.get(include=["metadatas"])
        if not all_docs or not all_docs["ids"]:
            return

        # Sort by date and remove oldest
        id_date_pairs = []
        for i, doc_id in enumerate(all_docs["ids"]):
            meta = all_docs["metadatas"][i] if all_docs["metadatas"] else {}
            date = meta.get("date", "")
            id_date_pairs.append((doc_id, date))

        id_date_pairs.sort(key=lambda x: x[1])  # oldest first
        to_remove = [p[0] for p in id_date_pairs[:overflow]]

        if to_remove:
            collection.delete(ids=to_remove)
            logger.info("[VectorMemory] Pruned %d old documents from %s", len(to_remove), collection.name)

    except Exception as e:
        logger.warning("[VectorMemory] prune failed: %s", e)


def get_stats() -> Dict:
    """Return stats about vector memory collections."""
    try:
        client = _get_client()
        collections = client.list_collections()
        stats = {}
        for col in collections:
            col_name = getattr(col, "name", str(col))
            try:
                c = _get_collection(col_name)
                stats[col_name] = c.count()
            except Exception:
                stats[col_name] = getattr(col, "count", lambda: 0)() if callable(getattr(col, "count", None)) else 0
        return stats
    except Exception as e:
        logger.warning("[VectorMemory] stats failed: %s", e)
        return {}


def backfill_from_sqlite():
    """
    One-time migration: embed existing stock_memory and world_knowledge
    from SQLite into ChromaDB for immediate RAG availability.
    """
    from core.db import db, brain_db
    count = 0

    # 1. Backfill stock analyses from stock_memory table
    try:
        with db.get_cursor() as (conn, c):
            c.execute("SELECT ticker, last_verdict, last_price, summary, last_analyzed FROM stock_memory")
            rows = c.fetchall()

        for row in rows:
            ticker, verdict, price, summary, date = row
            if summary:
                embed_analysis(ticker, summary, {
                    "verdict": verdict or "",
                    "price": price or 0,
                    "date": date or "",
                })
                count += 1
    except Exception as e:
        logger.warning("[VectorMemory] stock_memory backfill failed: %s", e)

    # 2. Backfill from brain's stock_knowledge (richer summaries)
    try:
        with brain_db.get_cursor() as (conn, c):
            c.execute("SELECT ticker, last_verdict, last_price, summary, sector, last_updated FROM stock_knowledge")
            rows = c.fetchall()

        for row in rows:
            ticker, verdict, price, summary, sector, date = row
            if summary:
                # Use the last summary segment (most recent)
                latest = summary.split("|||")[-1] if summary else ""
                if latest:
                    embed_analysis(ticker, latest, {
                        "verdict": verdict or "",
                        "price": price or 0,
                        "sector": sector or "",
                        "date": date or "",
                    })
                    count += 1
    except Exception as e:
        logger.warning("[VectorMemory] stock_knowledge backfill failed: %s", e)

    # 3. Backfill world knowledge
    try:
        with brain_db.get_cursor() as (conn, c):
            c.execute("SELECT headline, summary, category, impact, affected_tickers, date FROM world_knowledge ORDER BY date DESC LIMIT 200")
            rows = c.fetchall()

        for row in rows:
            headline, summary, category, impact, tickers, date = row
            if headline:
                import json
                embed_world_event(headline, summary or "", {
                    "category": category or "general",
                    "impact": impact or "neutral",
                    "affected_tickers": json.loads(tickers) if tickers else [],
                    "date": date or "",
                })
                count += 1
    except Exception as e:
        logger.warning("[VectorMemory] world_knowledge backfill failed: %s", e)

    # 4. Backfill learning log
    try:
        with brain_db.get_cursor() as (conn, c):
            c.execute("SELECT lesson, category, confidence, date FROM learning_log ORDER BY date DESC LIMIT 100")
            rows = c.fetchall()

        for row in rows:
            lesson, category, confidence, date = row
            if lesson:
                embed_lesson(lesson, {
                    "category": category or "general",
                    "confidence": confidence or 0.5,
                    "date": date or "",
                })
                count += 1
    except Exception as e:
        logger.warning("[VectorMemory] learning_log backfill failed: %s", e)

    logger.info("[VectorMemory] Backfill complete: %d documents embedded", count)
    return count
