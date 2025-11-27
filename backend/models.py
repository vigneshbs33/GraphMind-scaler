"""Pydantic models for GraphMind."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from .config import settings


class NodeCreate(BaseModel):
    """Payload for creating a graph node.

    Example:
        {"content": "Machine Learning overview", "node_type": "concept"}
    """

    id: Optional[str] = Field(
        default=None, description="Optional UUID; generated automatically if omitted."
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Node body text, up to 10k characters.",
    )
    metadata: Dict = Field(default_factory=dict, description="Arbitrary metadata.")
    node_type: Optional[str] = Field(
        default="general", description="Node classification label."
    )

    @field_validator("id", mode="before")
    @classmethod
    def _ensure_uuid(cls, value: Optional[str]) -> str:
        if value is None or value == "":
            return str(uuid4())
        UUID(str(value))  # raises if invalid
        return str(value)


class EdgeCreate(BaseModel):
    """Relationship definition between nodes."""

    source_id: str = Field(..., min_length=1, description="Origin node ID.")
    target_id: str = Field(..., min_length=1, description="Destination node ID.")
    relationship: str = Field(..., min_length=1, description="Relationship label.")
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Edge strength between 0 and 1.",
    )


class FileUploadRequest(BaseModel):
    """Metadata describing an incoming upload batch."""

    file_type: Literal["text", "pdf", "xml", "json", "csv"]
    file_size: int = Field(
        ...,
        gt=0,
        le=settings.MAX_FILE_SIZE,
        description="Declared file size in bytes (max 50MB).",
    )
    metadata: Dict = Field(default_factory=dict)


class SearchQuery(BaseModel):
    """Query payload for vector/graph/hybrid search."""

    query: str = Field(..., min_length=3, description="Search text (>=3 chars).")
    mode: Literal["vector", "graph", "hybrid"] = Field(
        default="hybrid", description="Search strategy."
    )
    top_k: int = Field(
        default=settings.DEFAULT_TOP_K,
        ge=1,
        le=20,
        description="Maximum number of hits (1-20).",
    )
    alpha: float = Field(
        default=settings.HYBRID_ALPHA,
        ge=0.0,
        le=1.0,
        description="Blend factor for hybrid search.",
    )


class SearchResult(BaseModel):
    """Unified representation of ranked results."""

    node_id: str
    content: str
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict = Field(default_factory=dict)
    reasoning: str = Field(default="")


class ComparisonResponse(BaseModel):
    """Evaluation payload comparing search modes."""

    vector_results: List[SearchResult]
    graph_results: List[SearchResult]
    hybrid_results: List[SearchResult]
    winner: Literal["vector", "graph", "hybrid"]
    metrics: Dict[str, float]
    llm_answer: Optional[str] = None


class ComparisonRequest(BaseModel):
    """Request payload for evaluating search strategies."""

    query: str = Field(..., min_length=3)
    ground_truth_ids: List[str] = Field(default_factory=list)
    top_k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=20)
    alpha: float = Field(default=settings.HYBRID_ALPHA, ge=0.0, le=1.0)


class NodeUpdate(BaseModel):
    """Payload for updating a graph node."""

    content: Optional[str] = Field(default=None, max_length=10_000, description="Updated node content.")
    metadata: Optional[Dict] = Field(default=None, description="Updated metadata.")
    node_type: Optional[str] = Field(default=None, description="Updated node type.")


class EdgeInfo(BaseModel):
    """Edge information in node relationships."""

    edge_id: str
    source_id: str
    target_id: str
    relationship: str
    weight: float = Field(..., ge=0.0, le=1.0)


class NodeResponse(BaseModel):
    """Complete node response with relationships."""

    node_id: str
    content: str
    metadata: Dict = Field(default_factory=dict)
    relationships: List[EdgeInfo] = Field(default_factory=list)
    created_at: Optional[datetime] = None


class VectorSearchRequest(BaseModel):
    """Request for vector-only search."""

    query_text: str = Field(..., min_length=3, description="Search query text.")
    top_k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=settings.MAX_TOP_K, description="Number of results.")


class GraphTraversalRequest(BaseModel):
    """Request for graph traversal search."""

    start_id: str = Field(..., description="Starting node ID for traversal.")
    depth: int = Field(default=3, ge=1, le=10, description="Maximum traversal depth.")
    max_nodes: int = Field(default=100, ge=1, le=1000, description="Maximum nodes to return.")


class HybridSearchRequest(BaseModel):
    """Request for hybrid search with explicit weights."""

    query_text: str = Field(..., min_length=3, description="Search query text.")
    vector_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="Weight for vector similarity.")
    graph_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for graph proximity.")
    top_k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=settings.MAX_TOP_K, description="Number of results.")


class TraversalNode(BaseModel):
    """Node in graph traversal result."""

    node_id: str
    content: str
    distance: int = Field(..., ge=0, description="Distance from start node.")
    path: List[str] = Field(default_factory=list, description="Path from start to this node.")
    metadata: Dict = Field(default_factory=dict)


class GraphTraversalResponse(BaseModel):
    """Response from graph traversal search."""

    start_id: str
    depth: int
    nodes: List[TraversalNode]


class HybridSearchResult(BaseModel):
    """Result from hybrid search with score breakdown."""

    node_id: str
    content: str
    score: float = Field(..., ge=0.0, le=1.0)
    vector_score: float = Field(..., ge=0.0, le=1.0)
    graph_score: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict = Field(default_factory=dict)

