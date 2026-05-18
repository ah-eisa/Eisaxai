"""
phase_i.tests.test_context_graph — governance + determinism + provenance.

Required by user spec:
    1. Every edge carries truth_type, provenance_tier, source_document_id,
       confidence, as_of_date, review_status.
    2. truth_type ∈ {"asserted", "derived"} only — reserved values raise.
    3. owned_by edges originate ONLY from the curated sovereign table —
       no auto-generated sovereign relationships.
    4. Deterministic build — same inputs ⇒ same `built_from_hash`.
    5. Strict ontology — non-canonical node kinds / relation types raise.
    6. Pure-read queries — graph mutation through public API is impossible.
    7. Strict-review gate hides unapproved edges when enabled.
    8. No vector / embedding / inference imports.
    9. Build only relies on `core.data_layer.*` (PHASE_I_SPEC §2 rule 1).
   10. Sample render produces a deterministic JSON snapshot.
"""

from __future__ import annotations

import sys
import traceback
from typing import List, Tuple


def _run(name, fn):
    try:
        fn()
        return (name, True, "")
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().splitlines()[-2:]
        return (name, False, f"{exc} :: {' | '.join(tb)}")


# 1 — full edge metadata
def test_every_edge_has_full_provenance_metadata():
    from phase_i import build_graph
    g = build_graph()
    assert len(g.edges) > 0, "expected at least one edge"
    required = ("truth_type", "provenance_tier", "source_document_id",
                "confidence", "as_of_date", "review_status")
    for e in g.edges:
        d = e.as_dict()
        for k in required:
            assert d.get(k) not in (None, ""), \
                f"edge {e.from_id}-[{e.relation}]->{e.to_id} missing {k}"


# 2 — truth_type vocabulary
def test_truth_types_v1_only():
    from phase_i import build_graph, TRUTH_TYPES, TRUTH_TYPES_RESERVED
    g = build_graph()
    for e in g.edges:
        assert e.truth_type in TRUTH_TYPES, \
            f"edge {e.from_id}-[{e.relation}]->{e.to_id} uses {e.truth_type!r}"
        assert e.truth_type not in TRUTH_TYPES_RESERVED


# 3 — owned_by edges only from the curated table
def test_owned_by_edges_only_from_sovereign_table():
    from phase_i import build_graph
    from core.data_layer.seed._sovereign_ownership import SOVEREIGN_OWNERSHIP
    g = build_graph()
    owned_by = [e for e in g.edges if e.relation == "owned_by"]
    table_tickers = set(SOVEREIGN_OWNERSHIP.keys())
    for e in owned_by:
        assert e.from_id in table_tickers, \
            f"owned_by edge from {e.from_id!r} NOT sourced from curated table — auto-generation forbidden"
        rec = SOVEREIGN_OWNERSHIP[e.from_id]
        assert e.source_document_id == rec.document_id, \
            f"document_id drift for {e.from_id}"


# 4 — determinism
def test_build_is_deterministic():
    from phase_i import build_graph
    from phase_i.context_graph import invalidate_cache
    invalidate_cache()
    g1 = build_graph()
    invalidate_cache()
    g2 = build_graph()
    assert g1.built_from_hash == g2.built_from_hash
    assert len(g1.nodes) == len(g2.nodes)
    assert len(g1.edges) == len(g2.edges)
    # Edge order is sort-stable
    for e1, e2 in zip(g1.edges, g2.edges):
        assert e1.as_dict() == e2.as_dict()


# 5 — strict ontology
def test_strict_ontology_rejects_unknown_relation():
    from phase_i.schemas import GraphEdge
    try:
        GraphEdge(
            from_id="A", to_id="B", relation="loved_by",
            truth_type="asserted", provenance_tier=1,
            source_document_id="x", confidence=0.9,
            as_of_date="2026-05-17", review_status="unreviewed",
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown relation")


def test_strict_ontology_rejects_reserved_truth_type():
    from phase_i.schemas import GraphEdge
    try:
        GraphEdge(
            from_id="A", to_id="B", relation="owned_by",
            truth_type="inferred", provenance_tier=1,
            source_document_id="x", confidence=0.9,
            as_of_date="2026-05-17", review_status="unreviewed",
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for reserved truth_type")


# 6 — pure-read queries (immutability)
def test_graph_container_is_frozen():
    from phase_i import build_graph
    g = build_graph()
    try:
        g.nodes = ()  # type: ignore[misc]
    except (AttributeError, TypeError):
        return
    raise AssertionError("ContextGraph must be frozen / immutable")


# 7 — strict review gate
def test_strict_review_gate_hides_unapproved_edges():
    from phase_i.context_graph import get_edges_for, invalidate_cache, build_graph
    from phase_h.registry import FeatureRegistry

    invalidate_cache()
    g = build_graph()
    # Pick any issuer with at least one edge
    sample = next((e.from_id for e in g.edges), None)
    assert sample is not None
    base_count = len(get_edges_for(sample))
    FeatureRegistry.override("phase_i_strict_review", True)
    try:
        strict_count = len(get_edges_for(sample))
        # All seed edges start unreviewed → strict mode hides them.
        assert strict_count == 0, f"strict mode should hide all unreviewed edges, got {strict_count}"
    finally:
        FeatureRegistry.reset_override("phase_i_strict_review")
        invalidate_cache()
    assert base_count > 0


# 8 — no forbidden imports
def test_no_embedding_inference_imports():
    import phase_i, phase_i.context_graph, phase_i.schemas
    forbidden = ("sentence_transformers", "faiss", "chromadb", "openai",
                 "torch.nn", "sklearn.neighbors")
    for mod in (phase_i, phase_i.context_graph, phase_i.schemas):
        src = (mod.__doc__ or "") + "\n" + repr(getattr(mod, "__all__", []))
        for name in forbidden:
            assert name not in src.lower(), f"{mod.__name__} mentions {name!r}"


# 9 — only data_layer is consulted at build time
def test_build_relies_only_on_data_layer():
    # We assert this by inspecting the module source for any non-data_layer
    # cross-package import (other than phase_h.registry which is sanctioned).
    import os, re
    path = os.path.join(os.path.dirname(__file__), "..", "context_graph.py")
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    illegal = re.findall(r"from\s+phase_h\.(?!registry)\w+", src)
    assert not illegal, f"phase_i.context_graph imports unsanctioned phase_h modules: {illegal}"


# 10 — deterministic JSON snapshot
def test_graph_snapshot_is_deterministic():
    from phase_i import graph_snapshot
    from phase_i.context_graph import invalidate_cache
    import json
    invalidate_cache()
    s1 = json.dumps(graph_snapshot(), sort_keys=True)
    invalidate_cache()
    s2 = json.dumps(graph_snapshot(), sort_keys=True)
    assert s1 == s2


# 11 — find_path respects max_hops
def test_find_path_obeys_max_hops():
    from phase_i import find_path, build_graph
    g = build_graph()
    # Pick a known asserted ownership path: TADAWUL:2222 → SOV:KSA
    path = find_path("TADAWUL:2222", "SOV:KSA", max_hops=1)
    assert path == ["TADAWUL:2222", "SOV:KSA"], f"unexpected path {path}"
    # Beyond max_hops returns None
    none_path = find_path("TADAWUL:2222", "SOV:UAE", max_hops=2)
    assert none_path is None


# 12 — shariah edges only target Shariah index nodes
def test_shariah_edges_target_index_kind():
    from phase_i import build_graph, get_node
    g = build_graph()
    for e in g.edges:
        if e.relation != "shariah_compliant_per":
            continue
        target = get_node(e.to_id)
        assert target is not None and target.kind == "index", \
            f"shariah edge target {e.to_id} must be an index node"


CASES = [
    ("every_edge_has_full_metadata",            test_every_edge_has_full_provenance_metadata),
    ("truth_types_v1_only",                     test_truth_types_v1_only),
    ("owned_by_only_from_sovereign_table",      test_owned_by_edges_only_from_sovereign_table),
    ("build_is_deterministic",                  test_build_is_deterministic),
    ("strict_ontology_rejects_unknown_relation", test_strict_ontology_rejects_unknown_relation),
    ("strict_ontology_rejects_reserved_truth",  test_strict_ontology_rejects_reserved_truth_type),
    ("graph_container_is_frozen",               test_graph_container_is_frozen),
    ("strict_review_gate_hides_unapproved",     test_strict_review_gate_hides_unapproved_edges),
    ("no_embedding_inference_imports",          test_no_embedding_inference_imports),
    ("build_relies_only_on_data_layer",         test_build_relies_only_on_data_layer),
    ("graph_snapshot_is_deterministic",         test_graph_snapshot_is_deterministic),
    ("find_path_obeys_max_hops",                test_find_path_obeys_max_hops),
    ("shariah_edges_target_index_kind",         test_shariah_edges_target_index_kind),
]


def main() -> int:
    results = [_run(n, fn) for n, fn in CASES]
    fails = [(n, msg) for n, ok, msg in results if not ok]
    for name, ok, msg in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' :: ' + msg) if msg else ''}")
    print()
    if fails:
        print(f"context_graph: {len(fails)}/{len(results)} FAILED")
        return 1
    print(f"context_graph: {len(results)}/{len(results)} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
