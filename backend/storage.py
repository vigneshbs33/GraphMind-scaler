"""Storage layer combining ChromaDB vectors and NetworkX graph."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

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
        logger.info(
            "Storage initialized: %s vectors present",
            self.collection.count(),
        )

    def add_node(self, node_id: Optional[str], content: str, metadata: Dict) -> str:
        """Insert a node across vector+graph stores."""
        node_id = node_id or str(uuid.uuid4())
        embedding = self.embedding_model.encode(content).tolist()
        safe_meta = metadata or {}
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

    def vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Return top_k vector similarities."""
        if not query:
            return []
        embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
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
        return payload

    def graph_search(self, query: str, top_k: int) -> List[Dict]:
        """Traverse neighbors starting from vector seed."""
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
        return [
            {"node_id": node_id, "content": content, "score": score, "metadata": meta}
            for node_id, score, content, meta in scored[:top_k]
        ]

    def hybrid_search(
        self, 
        query: str, 
        top_k: int, 
        alpha: float,
        relationship_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """Blend vector and graph scores with optional relationship weighting."""
        vector_results = self.vector_search(query, top_k * 2)
        graph_results = self.graph_search(query, top_k * 2)
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
        return results[:top_k]

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
        
        return {
            "node_id": node_id,
            "content": node_data.get("content", ""),
            "metadata": {k: v for k, v in node_data.items() if k != "content"},
            "relationships": incoming_edges + outgoing_edges,
        }

    def update_node(self, node_id: str, content: Optional[str] = None, metadata: Optional[Dict] = None, node_type: Optional[str] = None) -> Dict:
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
            
            # Regenerate embedding and update ChromaDB
            embedding = self.embedding_model.encode(content).tolist()
            self.collection.update(
                ids=[node_id],
                embeddings=[embedding],
                documents=[content],
            )
            logger.debug("Updated node content and embedding: %s", node_id)
        
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
        elif "-" in edge_id:
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

    def graph_traversal(self, start_id: str, depth: int = 3, max_nodes: int = 100) -> List[Dict]:
        """Traverse graph from start node up to specified depth."""
        from .exceptions import NotFoundError
        
        if start_id not in self.graph:
            raise NotFoundError(f"Start node {start_id} not found")
        
        # Use BFS to find all reachable nodes
        paths = nx.single_source_shortest_path_length(self.graph, start_id, cutoff=depth)
        
        # Build traversal results with paths
        results = []
        for node_id, distance in paths.items():
            if len(results) >= max_nodes:
                break
            
            # Get shortest path
            try:
                path = nx.shortest_path(self.graph, start_id, node_id)
            except nx.NetworkXNoPath:
                path = [start_id, node_id]
            
            node_data = self.graph.nodes[node_id]
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

