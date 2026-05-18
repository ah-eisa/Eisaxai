"""
phase_i — institutional context graph (I1).

Scope is strict (see PHASE_I_SPEC.md):
    - Entity nodes for issuers / instruments / sovereigns / regulators / indices
    - Relationship edges: owned_by / regulated_by / included_in / shariah_compliant_per
    - Provenance edges only (every edge carries truth_type + provenance_tier +
      source_document_id + confidence + as_of_date + review_status).

What this package intentionally does NOT contain:
    - No embeddings / vector layer.
    - No inference engine.
    - No market prediction logic.
    - No auto-generated sovereign relationships — every owned_by edge
      must originate from `core.data_layer.seed._sovereign_ownership`,
      which is reviewer-curated.
    - No semantic expansion beyond the approved ontology.

Public surface kept narrow so consumers depend on stable names.
"""

from __future__ import annotations

PHASE_I_VERSION = "0.1.0"

from .schemas import (
    GraphNode,
    GraphEdge,
    NODE_KINDS,
    RELATION_TYPES,
    TRUTH_TYPES,
    TRUTH_TYPES_RESERVED,
    REVIEW_STATUSES,
)
from .context_graph import (
    ContextGraph,
    build_graph,
    graph_snapshot,
    get_node,
    get_neighbors,
    get_edges_for,
    find_path,
    graph_summary,
)

__all__ = [
    "PHASE_I_VERSION",
    # schemas
    "GraphNode",
    "GraphEdge",
    "NODE_KINDS",
    "RELATION_TYPES",
    "TRUTH_TYPES",
    "TRUTH_TYPES_RESERVED",
    "REVIEW_STATUSES",
    # graph
    "ContextGraph",
    "build_graph",
    "graph_snapshot",
    "get_node",
    "get_neighbors",
    "get_edges_for",
    "find_path",
    "graph_summary",
]
