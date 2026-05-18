"""
phase_i.context_graph — provenance-first context graph (I1).

Build path is deterministic: same `core.data_layer` reference state
produces the same graph hash. Queries are pure reads — they never
mutate the graph, never touch the network, and never trigger inference.

Edge construction rules (v1.0):

    owned_by              ← `_sovereign_ownership.SOVEREIGN_OWNERSHIP`
                            (truth_type = "asserted", review_status =
                             "unreviewed", source_document_id = the
                             table's document_id)

    regulated_by          ← derived from the listing exchange via
                            `EXCHANGE_TO_LISTING_REGULATOR`
                            (truth_type = "derived")

    shariah_compliant_per ← `_shariah_index.SHARIAH_REFERENCE`
                            (truth_type = "derived")

    included_in           ← reserved for v1.1 once `inclusion_indices`
                            field is populated in `gcc_metadata`.

No edge is added beyond these rules. Engines that want to discover new
edges must extend the seed reference tables.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from phase_h.registry import FeatureRegistry

from . import _flags  # noqa: F401 — side-effect: register data_layer flags
from ._canonical_entities import (
    SOVEREIGNS, REGULATORS, INDICES,
    EXCHANGE_TO_LISTING_REGULATOR,
    SHARIAH_INDEX_NAME_TO_ID,
    all_canonical,
)
from .schemas import GraphNode, GraphEdge

logger = logging.getLogger("phase_i.context_graph")

_AS_OF = "2026-05-17"


# ──────────────────────────────────────────────────────────────────────
# Immutable graph container
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContextGraph:
    """Read-only graph snapshot. Constructed by `build_graph`."""
    nodes: Tuple[GraphNode, ...]
    edges: Tuple[GraphEdge, ...]
    built_from_hash: str

    def node_index(self) -> Dict[str, GraphNode]:
        return {n.id: n for n in self.nodes}

    def edge_index(self) -> Dict[str, GraphEdge]:
        return {e.edge_id: e for e in self.edges}


# ──────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────

def _stable_inputs_hash(*inputs: Any) -> str:
    """Deterministic hash of all builder inputs — drives reproducibility."""
    canon = json.dumps(inputs, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _country_to_sovereign(country: str) -> Optional[str]:
    """Map a country label to its canonical sovereign id (strict)."""
    country = (country or "").strip().lower()
    mapping = {
        "ksa":              "SOV:KSA",
        "saudi arabia":     "SOV:KSA",
        "uae":              "SOV:UAE",
        "qatar":            "SOV:QATAR",
        "kuwait":           "SOV:KUWAIT",
        "bahrain":          "SOV:BAHRAIN",
        "egypt":            "SOV:EGYPT",
    }
    return mapping.get(country)


def _entity_nodes() -> List[GraphNode]:
    """Issuer + canonical (sovereign/regulator/index) nodes."""
    from core.data_layer.gcc_metadata import GCC_METADATA  # local — avoid cycle
    nodes: List[GraphNode] = []
    # Issuers
    for ticker, entry in sorted(GCC_METADATA.items()):
        nodes.append(GraphNode(
            id=ticker,
            kind="issuer",
            label=ticker,
            country=(entry.get("country", {}) or {}).get("value"),
            attributes={
                "sector":   (entry.get("sector", {}) or {}).get("value"),
                "exchange": (entry.get("exchange", {}) or {}).get("value"),
            },
        ))
    # Canonical entities
    for cid, ent in sorted(all_canonical().items()):
        nodes.append(GraphNode(
            id=ent.id, kind=ent.kind, label=ent.label,
            country=ent.country,
            attributes={"source_document_id": ent.source_document_id},
        ))
    return nodes


def _owned_by_edges() -> List[GraphEdge]:
    """Sovereign-ownership edges — sourced ONLY from the curated table."""
    from core.data_layer.seed._sovereign_ownership import SOVEREIGN_OWNERSHIP
    from core.data_layer.gcc_metadata import GCC_METADATA
    edges: List[GraphEdge] = []
    for ticker, rec in SOVEREIGN_OWNERSHIP.items():
        entry = GCC_METADATA.get(ticker)
        country = (entry or {}).get("country", {}).get("value")
        sovereign_id = _country_to_sovereign(country or "")
        if sovereign_id is None:
            logger.debug("owned_by skipped — no sovereign mapping for %s", ticker)
            continue
        # provenance tier: 1 when issuer-disclosed (which the table is by design)
        edges.append(GraphEdge(
            from_id=ticker,
            to_id=sovereign_id,
            relation="owned_by",
            truth_type="asserted",
            provenance_tier=1,
            source_document_id=rec.document_id,
            confidence=float(rec.confidence),
            as_of_date=_AS_OF,
            review_status="unreviewed",
            methodology="curated_sovereign_ownership_table",
        ))
    return edges


def _regulated_by_edges() -> List[GraphEdge]:
    """Listing-regulator edges — derived from the issuer's exchange."""
    from core.data_layer.gcc_metadata import GCC_METADATA
    edges: List[GraphEdge] = []
    for ticker, entry in GCC_METADATA.items():
        exchange = (entry.get("exchange", {}) or {}).get("value")
        reg_id = EXCHANGE_TO_LISTING_REGULATOR.get(exchange or "")
        if reg_id is None:
            continue
        edges.append(GraphEdge(
            from_id=ticker,
            to_id=reg_id,
            relation="regulated_by",
            truth_type="derived",
            provenance_tier=2,
            source_document_id=f"exchange_listing:{exchange}",
            confidence=0.95,
            as_of_date=_AS_OF,
            review_status="unreviewed",
            methodology="derived_from:exchange_listing_to_regulator_mapping",
        ))
    return edges


def _shariah_edges() -> List[GraphEdge]:
    """Shariah-index membership edges — sourced from the curated index table."""
    from core.data_layer.seed._shariah_index import SHARIAH_REFERENCE
    edges: List[GraphEdge] = []
    for ticker, (index_name, doc_id, conf) in SHARIAH_REFERENCE.items():
        idx_id = SHARIAH_INDEX_NAME_TO_ID.get(index_name)
        if idx_id is None:
            logger.debug("shariah edge skipped — unknown index name %s", index_name)
            continue
        edges.append(GraphEdge(
            from_id=ticker,
            to_id=idx_id,
            relation="shariah_compliant_per",
            truth_type="derived",
            provenance_tier=2,
            source_document_id=doc_id,
            confidence=float(conf),
            as_of_date=_AS_OF,
            review_status="unreviewed",
            methodology=f"derived_from_index_membership:{index_name}",
        ))
    return edges


def build_graph() -> ContextGraph:
    """
    Build the full context graph deterministically.

    Returns a frozen `ContextGraph` whose `built_from_hash` changes only
    when the underlying reference tables change.
    """
    if not FeatureRegistry.is_enabled("phase_i_enabled"):
        return ContextGraph(nodes=(), edges=(), built_from_hash="phase_i_disabled")

    nodes = tuple(sorted(_entity_nodes(), key=lambda n: (n.kind, n.id)))
    edges_unsorted: List[GraphEdge] = []
    edges_unsorted.extend(_owned_by_edges())
    edges_unsorted.extend(_regulated_by_edges())
    edges_unsorted.extend(_shariah_edges())
    edges = tuple(sorted(edges_unsorted,
                         key=lambda e: (e.from_id, e.relation, e.to_id)))

    inputs = {
        "node_ids":    [n.id for n in nodes],
        "edge_triples": [(e.from_id, e.relation, e.to_id, e.source_document_id)
                         for e in edges],
        "as_of":       _AS_OF,
    }
    return ContextGraph(
        nodes=nodes, edges=edges,
        built_from_hash=_stable_inputs_hash(inputs),
    )


# ──────────────────────────────────────────────────────────────────────
# Cached singleton + query API
# ──────────────────────────────────────────────────────────────────────

_GRAPH_CACHE: Dict[str, ContextGraph] = {}


def _graph() -> ContextGraph:
    """Return the cached default graph, building it on first use."""
    if "default" not in _GRAPH_CACHE:
        _GRAPH_CACHE["default"] = build_graph()
    return _GRAPH_CACHE["default"]


def invalidate_cache() -> None:
    _GRAPH_CACHE.clear()


def _filter_edges(edges: Iterable[GraphEdge]) -> List[GraphEdge]:
    """Apply consumer-side gates: min provenance tier + strict review."""
    min_tier = int(FeatureRegistry.get("phase_i_min_tier"))
    strict = bool(FeatureRegistry.is_enabled("phase_i_strict_review"))
    out: List[GraphEdge] = []
    for e in edges:
        if e.provenance_tier > min_tier:  # tier > limit means weaker provenance
            continue
        if strict and e.review_status != "approved":
            continue
        out.append(e)
    return out


def get_node(node_id: str) -> Optional[GraphNode]:
    return _graph().node_index().get(node_id)


def get_edges_for(node_id: str, *, direction: str = "out") -> List[GraphEdge]:
    """`direction`: 'out' | 'in' | 'both'."""
    g = _graph()
    if direction not in {"out", "in", "both"}:
        raise ValueError(f"direction must be 'out'/'in'/'both', got {direction!r}")
    edges = []
    for e in g.edges:
        if direction in {"out", "both"} and e.from_id == node_id:
            edges.append(e)
        elif direction in {"in", "both"} and e.to_id == node_id:
            edges.append(e)
    return _filter_edges(edges)


def get_neighbors(node_id: str, *, relation: Optional[str] = None,
                  direction: str = "out") -> List[GraphNode]:
    """Neighbour nodes reachable in one hop."""
    g = _graph()
    idx = g.node_index()
    out_ids: List[str] = []
    for e in get_edges_for(node_id, direction=direction):
        if relation and e.relation != relation:
            continue
        nid = e.to_id if e.from_id == node_id else e.from_id
        out_ids.append(nid)
    return [idx[n] for n in out_ids if n in idx]


def find_path(from_id: str, to_id: str, *, max_hops: int = 3) -> Optional[List[str]]:
    """
    Simple breadth-first shortest path. Strict — no semantic expansion.
    Returns a list of node ids (including endpoints) or None when no
    path exists within `max_hops`.
    """
    if from_id == to_id:
        return [from_id]
    g = _graph()
    visited = {from_id}
    queue: List[Tuple[str, List[str]]] = [(from_id, [from_id])]
    while queue:
        current, path = queue.pop(0)
        if len(path) - 1 >= max_hops:
            continue
        for e in _filter_edges(g.edges):
            if e.from_id != current or e.to_id in visited:
                continue
            new_path = path + [e.to_id]
            if e.to_id == to_id:
                return new_path
            visited.add(e.to_id)
            queue.append((e.to_id, new_path))
    return None


def graph_snapshot() -> Dict[str, Any]:
    """
    Return a JSON-friendly snapshot of the current graph for audit.

    Output is deterministic — sorted, no timestamps, includes the
    `built_from_hash` so reviewers can verify reproducibility.
    """
    g = _graph()
    return {
        "phase_i_version": "0.1.0",
        "built_from_hash": g.built_from_hash,
        "node_count":      len(g.nodes),
        "edge_count":      len(g.edges),
        "nodes":           [n.as_dict() for n in g.nodes],
        "edges":           [e.as_dict() for e in g.edges],
    }


def graph_summary() -> Dict[str, Any]:
    """High-level histogram for the audit appendix / Section J prep."""
    g = _graph()
    nodes_by_kind: Dict[str, int] = {}
    edges_by_relation: Dict[str, int] = {}
    edges_by_truth: Dict[str, int] = {}
    edges_by_tier: Dict[int, int] = {}
    edges_by_review: Dict[str, int] = {}
    for n in g.nodes:
        nodes_by_kind[n.kind] = nodes_by_kind.get(n.kind, 0) + 1
    for e in g.edges:
        edges_by_relation[e.relation] = edges_by_relation.get(e.relation, 0) + 1
        edges_by_truth[e.truth_type] = edges_by_truth.get(e.truth_type, 0) + 1
        edges_by_tier[e.provenance_tier] = edges_by_tier.get(e.provenance_tier, 0) + 1
        edges_by_review[e.review_status] = edges_by_review.get(e.review_status, 0) + 1
    return {
        "phase_i_version":    "0.1.0",
        "built_from_hash":    g.built_from_hash,
        "node_count":         len(g.nodes),
        "edge_count":         len(g.edges),
        "nodes_by_kind":      nodes_by_kind,
        "edges_by_relation":  edges_by_relation,
        "edges_by_truth":     edges_by_truth,
        "edges_by_tier":      {str(k): v for k, v in edges_by_tier.items()},
        "edges_by_review":    edges_by_review,
    }


__all__ = [
    "ContextGraph",
    "build_graph",
    "graph_snapshot",
    "graph_summary",
    "get_node",
    "get_neighbors",
    "get_edges_for",
    "find_path",
    "invalidate_cache",
]
