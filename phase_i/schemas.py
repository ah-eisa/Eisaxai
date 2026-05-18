"""
phase_i.schemas — strict-enum types for the context graph.

The graph is intentionally narrow:
    - 5 node kinds
    - 4 relation types
    - 2 active truth types (3 more reserved for future use)
    - 4 review statuses

Every constraint is enforced in `__post_init__`. The graph builder
relies on these errors to surface ontology drift early.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Mapping, Optional, Tuple


# ── Ontology vocabulary ─────────────────────────────────────────────
NODE_KINDS: Tuple[str, ...] = (
    "issuer", "instrument", "sovereign", "regulator", "index",
)

RELATION_TYPES: Tuple[str, ...] = (
    "owned_by",                # issuer → sovereign
    "regulated_by",            # issuer → regulator
    "included_in",             # issuer → index
    "shariah_compliant_per",   # issuer → index (Shariah-specific)
)

TRUTH_TYPES: Tuple[str, ...] = (
    "asserted",                # direct factual claim from a primary source
    "derived",                 # produced by a documented rule over asserted inputs
)

# Reserved — engines must accept them as valid values but no edge in
# v1.0 emits them. Adding any of these requires a Phase I version bump.
TRUTH_TYPES_RESERVED: Tuple[str, ...] = (
    "inferred", "disputed", "stale",
)

REVIEW_STATUSES: Tuple[str, ...] = (
    "unreviewed",              # default for auto-built edges
    "pending",                 # picked up by a reviewer, decision outstanding
    "approved",                # reviewer signed off
    "rejected",                # reviewer disowned the edge
)


# ── Nodes ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GraphNode:
    """A single entity in the context graph."""
    id: str
    kind: str
    label: str
    country: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in NODE_KINDS:
            raise ValueError(f"node {self.id}: kind {self.kind!r} not in {NODE_KINDS}")
        if not self.id or not self.label:
            raise ValueError(f"node {self.id!r}: id and label are required")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id":         self.id,
            "kind":       self.kind,
            "label":      self.label,
            "country":    self.country,
            "attributes": dict(self.attributes),
        }


# ── Edges ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GraphEdge:
    """
    A directed relationship with full provenance.

    Required metadata (per user spec):
        truth_type, provenance_tier, source_document_id,
        confidence, as_of_date, review_status.
    """
    from_id: str
    to_id: str
    relation: str
    truth_type: str
    provenance_tier: int                # 1..4
    source_document_id: str
    confidence: float
    as_of_date: str                     # ISO YYYY-MM-DD
    review_status: str
    methodology: str = ""

    def __post_init__(self) -> None:
        if self.relation not in RELATION_TYPES:
            raise ValueError(
                f"edge {self.from_id}-[{self.relation}]->{self.to_id}: "
                f"relation must be one of {RELATION_TYPES}"
            )
        active = TRUTH_TYPES + TRUTH_TYPES_RESERVED
        if self.truth_type not in active:
            raise ValueError(
                f"edge {self.from_id}->{self.to_id}: truth_type "
                f"{self.truth_type!r} not in {active}"
            )
        if self.truth_type in TRUTH_TYPES_RESERVED:
            raise ValueError(
                f"edge {self.from_id}->{self.to_id}: truth_type "
                f"{self.truth_type!r} is reserved for a future phase"
            )
        if self.review_status not in REVIEW_STATUSES:
            raise ValueError(
                f"edge {self.from_id}->{self.to_id}: review_status "
                f"{self.review_status!r} not in {REVIEW_STATUSES}"
            )
        if not (1 <= int(self.provenance_tier) <= 4):
            raise ValueError(
                f"edge {self.from_id}->{self.to_id}: provenance_tier must be 1..4"
            )
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(
                f"edge {self.from_id}->{self.to_id}: confidence must be in [0,1]"
            )
        if not self.source_document_id:
            raise ValueError(
                f"edge {self.from_id}->{self.to_id}: source_document_id is required"
            )
        if not self.as_of_date:
            raise ValueError(
                f"edge {self.from_id}->{self.to_id}: as_of_date is required"
            )

    @property
    def edge_id(self) -> str:
        """Deterministic hash — same triple yields same id."""
        key = f"{self.from_id}|{self.relation}|{self.to_id}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["edge_id"] = self.edge_id
        return d


__all__ = [
    "NODE_KINDS",
    "RELATION_TYPES",
    "TRUTH_TYPES",
    "TRUTH_TYPES_RESERVED",
    "REVIEW_STATUSES",
    "GraphNode",
    "GraphEdge",
]
