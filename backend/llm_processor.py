"""Simple LLM orchestration (mock implementation)."""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class LLMProcessor:
    """Lightweight wrapper around an LLM provider (mock by default)."""

    def __init__(self, provider: str = "mock", api_key: str = "") -> None:
        self.provider = provider
        self.api_key = api_key
        logger.info("LLM initialized: %s", provider)

    async def understand_query(self, query: str) -> Dict:
        """Heuristically analyze query intent."""
        query_lower = query.lower()
        if any(token in query_lower for token in ("how", "why", "explain")):
            intent = "exploratory"
            alpha = 0.5
        elif any(token in query_lower for token in ("similar", "like")):
            intent = "semantic"
            alpha = 0.8
        elif any(token in query_lower for token in ("related", "connected", "relationship")):
            intent = "relational"
            alpha = 0.3
        else:
            intent = "general"
            alpha = 0.6

        entities = [word.strip(".,") for word in query.split() if len(word) > 2 and word[0].isupper()]
        result = {
            "intent": intent,
            "entities": entities,
            "relationships": ["related_to"],
            "suggested_alpha": alpha,
            "rephrased_query": query_lower,
        }
        return result

    async def refine_results(self, results: List[Dict], query: str) -> str:
        """Generate a mock natural language answer from ranked results."""
        if not results:
            return "No relevant results found."
        answer = [f"**Answer to: {query}**", "", "Based on the knowledge graph:"]
        for idx, result in enumerate(results[:3], 1):
            snippet = result["content"][:100].rstrip()
            answer.append(f"{idx}. **{result['node_id']}** — {snippet}... (relevance {result['score']:.2f})")
        answer.append("")
        answer.append("These insights blend semantic similarity with explicit graph context.")
        return "\n".join(answer)


def get_llm_processor(provider: str, api_key: str = "") -> LLMProcessor:
    """Factory for future provider-specific implementations."""
    if provider in {"claude", "ollama"}:
        raise NotImplementedError(f"{provider} integration not implemented yet.")
    return LLMProcessor(provider="mock")

