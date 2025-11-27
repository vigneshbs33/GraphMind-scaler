"""Search evaluation helpers."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .config import settings
from .llm_processor import LLMProcessor
from .models import ComparisonRequest, ComparisonResponse, SearchResult
from .storage import GraphMindStorage


def _as_search_results(raw: Sequence[Dict]) -> List[SearchResult]:
    return [
        SearchResult(
            node_id=item["node_id"],
            content=item.get("content", ""),
            score=item.get("score", 0.0),
            metadata=item.get("metadata", {}),
            reasoning=item.get("reasoning", ""),
        )
        for item in raw
    ]


def _precision(results: Sequence[Dict], truth: Sequence[str], top_k: int) -> float:
    if not results or not truth or top_k <= 0:
        return 0.0
    subset = [entry["node_id"] for entry in results[:top_k]]
    hits = sum(1 for node_id in subset if node_id in truth)
    return hits / float(top_k)


class SearchEvaluator:
    """Compare retrieval strategies and summarize findings."""

    def __init__(self, storage: GraphMindStorage, llm: LLMProcessor):
        self.storage = storage
        self.llm = llm

    async def compare(self, payload: ComparisonRequest) -> ComparisonResponse:
        top_k = payload.top_k
        alpha = payload.alpha or settings.HYBRID_ALPHA
        vector_raw = self.storage.vector_search(payload.query, top_k)
        graph_raw = self.storage.graph_search(payload.query, top_k)
        hybrid_raw = self.storage.hybrid_search(payload.query, top_k, alpha)

        vector_results = _as_search_results(vector_raw)
        graph_results = _as_search_results(graph_raw)
        hybrid_results = _as_search_results(hybrid_raw)

        truth = payload.ground_truth_ids
        metrics = {
            "vector_precision": _precision(vector_raw, truth, top_k),
            "graph_precision": _precision(graph_raw, truth, top_k),
            "hybrid_precision": _precision(hybrid_raw, truth, top_k),
        }
        winner = max(
            ("vector", metrics["vector_precision"]),
            ("graph", metrics["graph_precision"]),
            ("hybrid", metrics["hybrid_precision"]),
            key=lambda item: item[1],
        )[0]

        # Generate explanation for comparison
        llm_answer = await self.llm.explain_comparison(
            payload.query,
            vector_results,
            graph_results,
            hybrid_results,
            winner,
            metrics
        )

        return ComparisonResponse(
            vector_results=vector_results,
            graph_results=graph_results,
            hybrid_results=hybrid_results,
            winner=winner,
            metrics=metrics,
            llm_answer=llm_answer,
        )

