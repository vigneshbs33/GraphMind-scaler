"""Storage layer combining ChromaDB vectors and NetworkX graph."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import chromadb
import networkx as nx
import numpy as np
from chromadb.config import Settings as ChromaSettings
import sys

import huggingface_hub
from huggingface_hub import constants as hub_constants
import types

snapshot_module = types.ModuleType("huggingface_hub.snapshot_download")
setattr(snapshot_module, "REPO_ID_SEPARATOR", getattr(hub_constants, "REPO_ID_SEPARATOR", "--"))
sys.modules.setdefault("huggingface_hub.snapshot_download", snapshot_module)

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class GraphMindStorage:
    """Coordinates ChromaDB vectors with a NetworkX knowledge graph."""

    def __init__(self, chroma_path: Path, embedding_model: str) -> None:
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="graphmind_nodes",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_model = SentenceTransformer(embedding_model)
        self.graph = nx.DiGraph()
        self.vector_to_graph_map: Dict[str, str] = {}
        self.merge_runs = 0
        
        # Caching layer
        self.query_cache: OrderedDict[str, Tuple[List[Dict], float]] = OrderedDict()  # query -> (results, timestamp)
        self.embedding_cache: Dict[str, Tuple[List[float], float]] = {}  # text -> (embedding, timestamp)
        self.cache_max_size = 100  # Maximum number of cached queries
        self.cache_ttl = 300  # 5 minutes TTL
        
        logger.info(
            "Storage initialized: %s vectors present",
            self.collection.count(),
        )

    def add_node(self, node_id: Optional[str], content: str, metadata: Dict, embedding: Optional[List[float]] = None) -> str:
        """Insert a node across vector+graph stores."""
        from datetime import datetime
        
        node_id = node_id or str(uuid.uuid4())
        if embedding is None:
            embedding = self.embedding_model.encode(content).tolist()
        safe_meta = metadata or {}
        safe_meta["created_at"] = datetime.utcnow().isoformat()
        self.collection.add(
            ids=[node_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[safe_meta],
        )
        self.graph.add_node(node_id, content=content, **safe_meta)
        self.vector_to_graph_map[node_id] = node_id
        logger.debug("Node added: %s", node_id)
        return node_id

    def add_edge(self, source_id: str, target_id: str, relationship: str, weight: float) -> None:
        """Add directional edge if both nodes exist."""
        if source_id not in self.graph or target_id not in self.graph:
            raise ValueError("Both nodes must exist before adding an edge.")
        self.graph.add_edge(
            source_id,
            target_id,
            relationship=relationship,
            weight=weight,
        )
        logger.debug("Edge %s -> %s (%s)", source_id, target_id, relationship)

    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if available and not expired."""
        if text in self.embedding_cache:
            embedding, timestamp = self.embedding_cache[text]
            if time.time() - timestamp < self.cache_ttl:
                return embedding
            else:
                del self.embedding_cache[text]
        return None
    
    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding with timestamp."""
        # Limit cache size
        if len(self.embedding_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = min(self.embedding_cache.keys(), key=lambda k: self.embedding_cache[k][1])
            del self.embedding_cache[oldest_key]
        self.embedding_cache[text] = (embedding, time.time())
    
    def _get_cached_query(self, cache_key: str) -> Optional[List[Dict]]:
        """Get cached query results if available and not expired."""
        if cache_key in self.query_cache:
            results, timestamp = self.query_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                # Move to end (most recently used)
                self.query_cache.move_to_end(cache_key)
                return results
            else:
                del self.query_cache[cache_key]
        return None
    
    def _cache_query(self, cache_key: str, results: List[Dict]) -> None:
        """Cache query results with timestamp."""
        # Limit cache size using LRU eviction
        if len(self.query_cache) >= self.cache_max_size:
            # Remove oldest entry (first in OrderedDict)
            self.query_cache.popitem(last=False)
        self.query_cache[cache_key] = (results, time.time())
    
    def vector_search(self, query: str, top_k: int, query_embedding: Optional[List[float]] = None, metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """Return top_k vector similarities with optional metadata filtering and caching."""
        if not query and not query_embedding:
            return []
        
        # Check cache (only if no metadata filter and using query text)
        if query and not metadata_filter and not query_embedding:
            cache_key = f"vector:{query}:{top_k}"
            cached = self._get_cached_query(cache_key)
            if cached is not None:
                logger.debug("Cache hit for vector search: %s", query[:50])
                return cached
        
        if query_embedding is None:
            # Check embedding cache
            embedding = self._get_cached_embedding(query)
            if embedding is None:
                embedding = self.embedding_model.encode(query).tolist()
                self._cache_embedding(query, embedding)
        else:
            embedding = query_embedding
        
        # Build where clause for metadata filtering
        where = None
        if metadata_filter:
            where = metadata_filter
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.collection.count()) if self.collection.count() > 0 else top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        raw_scores = [1 - d for d in distances]
        if raw_scores:
            min_score = min(raw_scores)
            max_score = max(raw_scores)
            norm = (
                [(s - min_score) / (max_score - min_score) if max_score != min_score else 1.0 for s in raw_scores]
            )
        else:
            norm = []
        payload = []
        for idx, node_id in enumerate(ids):
            payload.append(
                {
                    "node_id": node_id,
                    "content": documents[idx] if idx < len(documents) else "",
                    "score": norm[idx] if idx < len(norm) else 0.0,
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                }
            )
        
        # Cache results if applicable
        if query and not metadata_filter and not query_embedding:
            cache_key = f"vector:{query}:{top_k}"
            self._cache_query(cache_key, payload)
        
        return payload

    def graph_search(self, query: str, top_k: int) -> List[Dict]:
        """Traverse neighbors starting from vector seed with caching."""
        # Check cache
        cache_key = f"graph:{query}:{top_k}"
        cached = self._get_cached_query(cache_key)
        if cached is not None:
            logger.debug("Cache hit for graph search: %s", query[:50])
            return cached
        
        seed_results = self.vector_search(query, top_k=1)
        if not seed_results:
            return []
        seed_id = seed_results[0]["node_id"]
        if seed_id not in self.graph:
            return []
        paths = nx.single_source_shortest_path_length(self.graph, seed_id, cutoff=3)
        scored = [
            (
                node_id,
                1 / (dist + 1),
                self.graph.nodes[node_id].get("content", ""),
                {k: v for k, v in self.graph.nodes[node_id].items() if k != "content"},
            )
            for node_id, dist in paths.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        results = [
            {"node_id": node_id, "content": content, "score": score, "metadata": meta}
            for node_id, score, content, meta in scored[:top_k]
        ]
        
        # Cache results
        self._cache_query(cache_key, results)
        return results

    def hybrid_search(
        self, 
        query: str, 
        top_k: int, 
        alpha: float,
        query_embedding: Optional[List[float]] = None,
        relationship_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """Blend vector and graph scores with optional relationship weighting using parallel execution."""
        import concurrent.futures
        
        # Run vector and graph searches in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(self.vector_search, query, top_k * 2, query_embedding, None)
            graph_future = executor.submit(self.graph_search, query, top_k * 2)
            
            # Wait for both to complete
            vector_results = vector_future.result()
            graph_results = graph_future.result()
        combined: Dict[str, Dict] = {}
        
        # Apply relationship weights to graph scores if provided
        if relationship_weights:
            for item in graph_results:
                node_id = item["node_id"]
                # Get edges for this node to calculate relationship-weighted score
                if node_id in self.graph:
                    incoming = list(self.graph.in_edges(node_id, data=True))
                    outgoing = list(self.graph.out_edges(node_id, data=True))
                    
                    weighted_score = 0.0
                    total_weight = 0.0
                    
                    for src, dst, attrs in incoming + outgoing:
                        rel_type = attrs.get("relationship", "")
                        weight = attrs.get("weight", 1.0)
                        rel_weight = relationship_weights.get(rel_type, 1.0)
                        weighted_score += weight * rel_weight
                        total_weight += rel_weight
                    
                    if total_weight > 0:
                        item["score"] = weighted_score / total_weight
        
        for item in vector_results:
            combined[item["node_id"]] = {
                "vector": item["score"],
                "graph": 0.0,
                "content": item["content"],
                "metadata": item["metadata"],
            }
        for item in graph_results:
            bucket = combined.setdefault(
                item["node_id"],
                {"vector": 0.0, "graph": 0.0, "content": item["content"], "metadata": item["metadata"]},
            )
            bucket["graph"] = item["score"]
        results = []
        for node_id, scores in combined.items():
            final = alpha * scores["vector"] + (1 - alpha) * scores["graph"]
            reason = f"Vector: {scores['vector']:.2f}, Graph: {scores['graph']:.2f}, Combined: {final:.2f}"
            results.append(
                {
                    "node_id": node_id,
                    "content": scores["content"],
                    "score": final,
                    "metadata": scores["metadata"],
                    "reasoning": reason,
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        final_results = results[:top_k]
        
        # Cache hybrid search results
        cache_key = f"hybrid:{query}:{top_k}:{alpha}"
        self._cache_query(cache_key, final_results)
        
        return final_results

    def merge_algorithm(self) -> None:
        """Run PageRank and add semantic similarity edges."""
        if self.graph.number_of_nodes() == 0:
            return
        ranks = nx.pagerank(self.graph)
        nx.set_node_attributes(self.graph, ranks, "centrality")
        payload = self.collection.get(include=["embeddings"])
        embeddings = payload.get("embeddings") or []
        ids = payload.get("ids") or []
        if len(embeddings) < 2:
            return
        matrix = np.array(embeddings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = matrix / np.clip(norms, a_min=1e-10, a_max=None)
        similarities = normalized @ normalized.T
        threshold = 0.7
        for i, src in enumerate(ids):
            for j, dst in enumerate(ids):
                if i >= j:
                    continue
                score = similarities[i, j]
                if score >= threshold and src in self.graph and dst in self.graph:
                    self.graph.add_edge(src, dst, relationship="semantic_similarity", weight=float(score))
        self.merge_runs += 1
        logger.info("Merge run %s completed with %s nodes", self.merge_runs, len(ids))

    def get_node(self, node_id: str) -> Dict:
        """Retrieve node from graph with all connected edges."""
        from .exceptions import NotFoundError
        
        if node_id not in self.graph:
            raise NotFoundError(f"Node {node_id} not found")
        
        node_data = self.graph.nodes[node_id]
        
        # Get embedding from ChromaDB
        embedding = None
        try:
            chroma_result = self.collection.get(ids=[node_id], include=["embeddings"])
            if chroma_result["embeddings"] and len(chroma_result["embeddings"]) > 0:
                embedding = chroma_result["embeddings"][0]
        except Exception as e:
            logger.warning("Could not retrieve embedding for node %s: %s", node_id, str(e))
        
        # Get all connected edges (incoming and outgoing)
        incoming_edges = [
            {
                "edge_id": f"{src}-{node_id}",
                "source_id": src,
                "target_id": node_id,
                "relationship": self.graph.edges[src, node_id].get("relationship", ""),
                "weight": self.graph.edges[src, node_id].get("weight", 1.0),
            }
            for src in self.graph.predecessors(node_id)
        ]
        
        outgoing_edges = [
            {
                "edge_id": f"{node_id}-{dst}",
                "source_id": node_id,
                "target_id": dst,
                "relationship": self.graph.edges[node_id, dst].get("relationship", ""),
                "weight": self.graph.edges[node_id, dst].get("weight", 1.0),
            }
            for dst in self.graph.successors(node_id)
        ]
        
        result = {
            "node_id": node_id,
            "content": node_data.get("content", ""),
            "metadata": {k: v for k, v in node_data.items() if k not in ("content", "embedding")},
            "relationships": incoming_edges + outgoing_edges,
        }
        
        if embedding is not None:
            result["embedding"] = embedding
        
        return result

    def update_node(self, node_id: str, content: Optional[str] = None, metadata: Optional[Dict] = None, node_type: Optional[str] = None, regen_embedding: bool = True) -> Dict:
        """Update node attributes in NetworkX and ChromaDB if content changed."""
        from .exceptions import NotFoundError
        
        if node_id not in self.graph:
            raise NotFoundError(f"Node {node_id} not found")
        
        node_data = self.graph.nodes[node_id]
        content_changed = False
        
        # Update content if provided
        if content is not None and content != node_data.get("content", ""):
            node_data["content"] = content
            content_changed = True
            
            # Regenerate embedding if requested
            if regen_embedding:
                embedding = self.embedding_model.encode(content).tolist()
                self.collection.update(
                    ids=[node_id],
                    embeddings=[embedding],
                    documents=[content],
                )
                logger.debug("Updated node content and embedding: %s", node_id)
            else:
                # Just update document without changing embedding
                self.collection.update(
                    ids=[node_id],
                    documents=[content],
                )
                logger.debug("Updated node content without regenerating embedding: %s", node_id)
        
        # Update node_type if provided
        if node_type is not None:
            node_data["node_type"] = node_type
            logger.debug("Updated node_type: %s", node_id)
        
        # Update metadata if provided
        if metadata is not None:
            node_data.update(metadata)
        
        # Update ChromaDB metadata if content or metadata changed
        if content_changed or metadata is not None or node_type is not None:
            # Prepare metadata dict excluding 'content' for ChromaDB
            chroma_metadata = {k: v for k, v in node_data.items() if k != "content"}
            self.collection.update(
                ids=[node_id],
                metadatas=[chroma_metadata],
            )
            logger.debug("Updated node metadata: %s", node_id)
        
        return self.get_node(node_id)

    def delete_node(self, node_id: str) -> Dict:
        """Remove node from NetworkX graph and ChromaDB."""
        from .exceptions import NotFoundError
        
        if node_id not in self.graph:
            raise NotFoundError(f"Node {node_id} not found")
        
        # Count edges before deletion (outgoing + incoming)
        # Note: self-loops would be counted in both, but that's acceptable
        edge_count = self.graph.degree(node_id)  # Total degree (in + out) for DiGraph
        
        # Remove from graph (automatically removes all connected edges)
        self.graph.remove_node(node_id)
        
        # Remove from ChromaDB
        try:
            self.collection.delete(ids=[node_id])
        except Exception as e:
            logger.warning("Failed to delete from ChromaDB: %s", e)
        
        # Remove from mapping
        self.vector_to_graph_map.pop(node_id, None)
        
        logger.info("Deleted node %s and %d edges", node_id, edge_count)
        
        return {
            "node_id": node_id,
            "status": "deleted",
            "edges_removed": edge_count,
        }

    def get_edge(self, edge_id: str, source_id: Optional[str] = None, target_id: Optional[str] = None) -> Dict:
        """Retrieve edge from graph with all attributes."""
        from .exceptions import NotFoundError, ValidationError
        
        # Parse edge_id or use provided source_id and target_id
        if source_id and target_id:
            src, dst = source_id, target_id
        elif edge_id:
            # Edge ID format: source_id-target_id
            # For UUIDs (36 chars each), the format is: {36 chars}-{36 chars}
            # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (exactly 36 chars)
            # So if edge_id is 73 chars (36 + 1 dash + 36), split at position 36
            if len(edge_id) == 73 and edge_id[36] == '-':
                # Two UUIDs separated by a dash
                src = edge_id[:36]
                dst = edge_id[37:]
            else:
                # For non-UUID cases or if format doesn't match, try to find split point
                # Look for pattern where we have a complete UUID followed by dash and another UUID
                # UUID has 4 dashes at positions: 8, 13, 18, 23
                # So after position 23, we have 12 more chars = position 35, then dash at 36
                # Try splitting at various positions
                dash_positions = [i for i, c in enumerate(edge_id) if c == '-']
                if len(dash_positions) >= 5:  # At least 4 for first UUID + 1 separator
                    # First UUID ends at position 35 (0-indexed), dash at 36
                    # Check if position 36 is a dash and length suggests two UUIDs
                    if len(edge_id) >= 37 and edge_id[36] == '-':
                        src = edge_id[:36]
                        dst = edge_id[37:]
                    else:
                        # Try to find the separator dash (should be after a complete UUID)
                        # A complete UUID is 36 chars, so look for dash at index 36
                        # If not found, fall back to splitting on first dash
                        parts = edge_id.split("-", 1)
                        if len(parts) == 2:
                            src, dst = parts
                        else:
                            raise ValidationError(f"Invalid edge_id format: {edge_id}")
                else:
                    # Not UUIDs or simple format, split on first dash
                    parts = edge_id.split("-", 1)
                    if len(parts) == 2:
                        src, dst = parts
                    else:
                        raise ValidationError(f"Invalid edge_id format: {edge_id}")
        else:
            raise ValidationError("Either edge_id or both source_id and target_id must be provided")
        
        if not self.graph.has_edge(src, dst):
            raise NotFoundError(f"Edge from {src} to {dst} not found")
        
        edge_data = self.graph.edges[src, dst]
        
        return {
            "edge_id": f"{src}-{dst}",
            "source_id": src,
            "target_id": dst,
            "relationship": edge_data.get("relationship", ""),
            "weight": edge_data.get("weight", 1.0),
            "metadata": {k: v for k, v in edge_data.items() if k not in ("relationship", "weight")},
        }

    def graph_traversal(self, start_id: str, depth: int = 3, max_nodes: int = 100, relationship_type: Optional[str] = None) -> List[Dict]:
        """Traverse graph from start node up to specified depth with optional relationship filtering."""
        from .exceptions import NotFoundError
        
        if start_id not in self.graph:
            raise NotFoundError(f"Start node {start_id} not found")
        
        # Create filtered graph if relationship type is specified
        if relationship_type:
            filtered_graph = nx.DiGraph()
            for u, v, data in self.graph.edges(data=True):
                if data.get("relationship") == relationship_type:
                    filtered_graph.add_edge(u, v, **data)
            # Add all nodes
            for node in self.graph.nodes():
                filtered_graph.add_node(node, **self.graph.nodes[node])
            traversal_graph = filtered_graph
        else:
            traversal_graph = self.graph
        
        # Use BFS to find all reachable nodes (handles cycles by visited set)
        visited = set()
        queue = [(start_id, 0, [start_id])]
        paths_dict = {}
        
        while queue and len(visited) < max_nodes:
            node_id, current_depth, path = queue.pop(0)
            
            if node_id in visited or current_depth > depth:
                continue
                
            visited.add(node_id)
            paths_dict[node_id] = (current_depth, path)
            
            if current_depth < depth:
                for neighbor in traversal_graph.successors(node_id):
                    if neighbor not in visited:
                        queue.append((neighbor, current_depth + 1, path + [neighbor]))
        
        # Build traversal results with paths
        results = []
        for node_id, (distance, path) in paths_dict.items():
            if len(results) >= max_nodes:
                break
            
            node_data = self.graph.nodes[node_id]
            
            # Get edge info for path
            edge_info = []
            if len(path) > 1:
                for i in range(len(path) - 1):
                    src, dst = path[i], path[i + 1]
                    if self.graph.has_edge(src, dst):
                        edge_data = self.graph.edges[src, dst]
                        edge_info.append({
                            "edge": edge_data.get("relationship", ""),
                            "weight": edge_data.get("weight", 1.0)
                        })
            
            results.append({
                "node_id": node_id,
                "content": node_data.get("content", ""),
                "distance": distance,
                "path": path,
                "metadata": {k: v for k, v in node_data.items() if k != "content"},
            })
        
        # Sort by distance
        results.sort(key=lambda x: x["distance"])
        
        return results

    def multi_hop_reasoning(self, query: str, max_hops: int = 3) -> List[Dict]:
        """Multi-hop reasoning: find seed nodes via vector search, then traverse graph."""
        from .exceptions import NotFoundError
        
        # Find seed nodes using vector search
        seed_results = self.vector_search(query, top_k=3)
        if not seed_results:
            return []
        
        all_paths = []
        seen_nodes = set()
        
        for seed in seed_results:
            seed_id = seed["node_id"]
            if seed_id not in self.graph:
                continue
            
            # Traverse from seed up to max_hops
            paths = nx.single_source_shortest_path_length(self.graph, seed_id, cutoff=max_hops)
            
            for node_id, distance in paths.items():
                if node_id in seen_nodes:
                    continue
                seen_nodes.add(node_id)
                
                # Get shortest path
                try:
                    path = nx.shortest_path(self.graph, seed_id, node_id)
                except nx.NetworkXNoPath:
                    path = [seed_id, node_id]
                
                node_data = self.graph.nodes[node_id]
                all_paths.append({
                    "node_id": node_id,
                    "content": node_data.get("content", ""),
                    "distance": distance,
                    "seed_node": seed_id,
                    "path": path,
                    "reasoning": f"Found via seed '{seed_id}' through {len(path)-1} hops",
                    "metadata": {k: v for k, v in node_data.items() if k != "content"},
                })
        
        # Sort by distance, then by seed relevance
        all_paths.sort(key=lambda x: (x["distance"], x["node_id"]))
        return all_paths[:50]  # Limit results

    def get_stats(self) -> Dict[str, float]:
        """Return storage statistics."""
        node_count = self.graph.number_of_nodes()
        return {
            "node_count": node_count,
            "edge_count": self.graph.number_of_edges(),
            "vector_count": float(self.collection.count()),
            "merge_runs": float(self.merge_runs),
            "alignment_score": float(len(self.vector_to_graph_map)) / max(node_count, 1),
        }

