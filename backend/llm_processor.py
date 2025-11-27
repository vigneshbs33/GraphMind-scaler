"""Simple LLM orchestration (Gemini integration)."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Any

import httpx

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
        """Generate a natural language answer from ranked results using Gemini."""
        if not results:
            return "No relevant results found."
        
        if self.provider == "gemini" and self.api_key:
            return await self._gemini_refine(results, query)
        
        # Fallback to mock
        answer = [f"**Answer to: {query}**", "", "Based on the knowledge graph:"]
        for idx, result in enumerate(results[:3], 1):
            snippet = result["content"][:100].rstrip()
            answer.append(f"{idx}. **{result['node_id']}** — {snippet}... (relevance {result['score']:.2f})")
        answer.append("")
        answer.append("These insights blend semantic similarity with explicit graph context.")
        return "\n".join(answer)
    
    async def explain_comparison(
        self,
        query: str,
        vector_results: List[Any],
        graph_results: List[Any],
        hybrid_results: List[Any],
        winner: str,
        metrics: Dict[str, float]
    ) -> str:
        """Explain why a particular search method won the comparison."""
        if self.provider == "gemini" and self.api_key:
            return await self._gemini_explain_comparison(query, vector_results, graph_results, hybrid_results, winner, metrics)
        
        # Fallback
        return f"**Comparison Analysis**\n\nThe {winner} search method performed best with a precision of {metrics.get(f'{winner}_precision', 0):.3f}."
    
    async def _gemini_refine(self, results: List[Dict], query: str) -> str:
        """Use Gemini API to generate structured answer."""
        try:
            # Prepare context from top results
            context_parts = []
            for idx, result in enumerate(results[:5], 1):
                context_parts.append(
                    f"Result {idx} (ID: {result['node_id']}, Score: {result['score']:.2f}):\n"
                    f"{result['content'][:300]}"
                )
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""You are an AI assistant helping users understand search results from a hybrid vector+graph knowledge database. Provide a concise, well-structured answer using markdown formatting.

User Query: {query}

Search Results:
{context}

Instructions:
1. **Be concise** - Maximum 3-4 paragraphs, focus on key insights
2. **Use markdown** - Format with headers (##), **bold** for emphasis, bullet points (-), and code blocks when needed
3. **Synthesize** - Don't just list results; explain what they mean together
4. **Reference IDs** - Mention node IDs (e.g., `node_123`) when relevant
5. **Explain relationships** - If results are connected, explain how
6. **Be specific** - Use concrete details from the results, not generic statements

Format your response as:
## Answer
[Your concise answer here]

## Key Insights
- [Bullet point 1]
- [Bullet point 2]

## Related Nodes
- `node_id_1`: [brief description]
- `node_id_2`: [brief description]

Answer:"""

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    json={
                        "contents": [
                            {
                                "parts": [
                                    {"text": prompt}
                                ]
                            }
                        ]
                    },
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract text from Gemini response
                if "candidates" in data and len(data["candidates"]) > 0:
                    content = data["candidates"][0].get("content", {})
                    parts = content.get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
                
                logger.warning("Unexpected Gemini response format")
                return "Unable to generate answer from LLM."
                
        except httpx.TimeoutException:
            logger.error("Gemini API timeout")
            return "LLM request timed out. Please try again."
        except httpx.HTTPStatusError as e:
            logger.error(f"Gemini API error: {e.response.status_code} - {e.response.text}")
            return f"LLM API error: {e.response.status_code}"
        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            return f"Error generating answer: {str(e)}"
    
    async def _gemini_explain_comparison(
        self,
        query: str,
        vector_results: List[Any],
        graph_results: List[Any],
        hybrid_results: List[Any],
        winner: str,
        metrics: Dict[str, float]
    ) -> str:
        """Use Gemini to explain comparison results."""
        try:
            # Prepare context
            vector_precision = metrics.get("vector_precision", 0)
            graph_precision = metrics.get("graph_precision", 0)
            hybrid_precision = metrics.get("hybrid_precision", 0)
            
            top_vector = vector_results[0] if vector_results else None
            top_graph = graph_results[0] if graph_results else None
            top_hybrid = hybrid_results[0] if hybrid_results else None
            
            vector_info = f"{top_vector.node_id} (score: {top_vector.score:.3f})" if top_vector else "N/A"
            graph_info = f"{top_graph.node_id} (score: {top_graph.score:.3f})" if top_graph else "N/A"
            hybrid_info = f"{top_hybrid.node_id} (score: {top_hybrid.score:.3f})" if top_hybrid else "N/A"
            
            prompt = f"""You are analyzing search method comparisons for a hybrid vector+graph knowledge database. Explain why {winner} search won and provide insights.

User Query: {query}

Performance Metrics:
- Vector Search Precision: {vector_precision:.3f}
- Graph Search Precision: {graph_precision:.3f}
- Hybrid Search Precision: {hybrid_precision:.3f}
- Winner: {winner.upper()}

Top Results:
- Vector: {vector_info}
- Graph: {graph_info}
- Hybrid: {hybrid_info}

Instructions:
1. **Be concise** - Maximum 2-3 paragraphs
2. **Explain the winner** - Why did {winner} perform best?
3. **Compare methods** - What are the strengths/weaknesses of each?
4. **Use markdown** - Format with ## headers, **bold**, and bullet points
5. **Be specific** - Reference the precision scores and query type

Format:
## Why {winner.upper()} Won
[Explanation]

## Method Comparison
- **Vector Search**: [Brief analysis]
- **Graph Search**: [Brief analysis]
- **Hybrid Search**: [Brief analysis]

## Recommendation
[When to use each method]

Answer:"""

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    json={
                        "contents": [
                            {
                                "parts": [
                                    {"text": prompt}
                                ]
                            }
                        ]
                    },
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                
                if "candidates" in data and len(data["candidates"]) > 0:
                    content = data["candidates"][0].get("content", {})
                    parts = content.get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
                
                logger.warning("Unexpected Gemini response format")
                return f"**Comparison Analysis**\n\nThe {winner} search method performed best with a precision of {metrics.get(f'{winner}_precision', 0):.3f}."
                
        except httpx.TimeoutException:
            logger.error("Gemini API timeout")
            return f"**Comparison Analysis**\n\nThe {winner} search method performed best with a precision of {metrics.get(f'{winner}_precision', 0):.3f}."
        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            return f"**Comparison Analysis**\n\nThe {winner} search method performed best with a precision of {metrics.get(f'{winner}_precision', 0):.3f}."


def get_llm_processor(provider: str, api_key: str = "") -> LLMProcessor:
    """Factory for provider-specific implementations."""
    if provider == "gemini":
        return LLMProcessor(provider="gemini", api_key=api_key)
    elif provider in {"claude", "ollama"}:
        raise NotImplementedError(f"{provider} integration not implemented yet.")
    return LLMProcessor(provider="mock", api_key="")

