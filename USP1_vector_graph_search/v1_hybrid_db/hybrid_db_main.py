from fastapi import FastAPI, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel # type: ignore
from typing import List, Optional, Dict, Any
import numpy as np # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import networkx as nx # type: ignore
import json
import uuid
from datetime import datetime

# Initialize FastAPI
app = FastAPI(
    title="Vector + Graph Hybrid Database",
    description="A hybrid retrieval system combining vector similarity and graph relationships",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding model (lightweight and fast)
print("Loading embedding model...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded successfully!")

# Initialize graph
graph = nx.DiGraph()

# In-memory storage for embeddings (node_id -> embedding vector)
embeddings_store = {}

# ============================================================================
# DATA MODELS
# ============================================================================

class NodeCreate(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = {}
    embedding: Optional[List[float]] = None

class NodeUpdate(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    regenerate_embedding: bool = False

class NodeResponse(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    relationships: List[Dict[str, Any]]
    created_at: str

class EdgeCreate(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0

class EdgeResponse(BaseModel):
    id: str
    source_id: str
    target_id: str
    relationship_type: str
    weight: float

class VectorSearchRequest(BaseModel):
    query_text: str
    top_k: int = 5

class HybridSearchRequest(BaseModel):
    query_text: str
    top_k: int = 5
    vector_weight: float = 0.5
    graph_weight: float = 0.5
    start_node_id: Optional[str] = None

class SearchResult(BaseModel):
    node_id: str
    text: str
    score: float
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    explanation: Optional[str] = None
    metadata: Dict[str, Any]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding vector for text"""
    return embedding_model.encode(text, convert_to_numpy=True)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return float(dot_product / (norm1 * norm2))

def calculate_graph_proximity(source_id: str, target_id: str) -> float:
    """Calculate graph-based proximity score"""
    try:
        # Check if path exists
        if not nx.has_path(graph, source_id, target_id):
            return 0.0
        
        # Calculate shortest path
        path_length = nx.shortest_path_length(graph, source_id, target_id)
        
        # Convert to similarity score (closer = higher score)
        # Formula: 1 / (1 + distance)
        proximity = 1.0 / (1.0 + path_length)
        
        # Bonus for direct connection
        if graph.has_edge(source_id, target_id):
            edge_data = graph.get_edge_data(source_id, target_id)
            weight = edge_data.get('weight', 1.0)
            proximity *= weight
        
        return proximity
    except:
        return 0.0

def get_node_data(node_id: str) -> Dict[str, Any]:
    """Get complete node data"""
    if node_id not in graph.nodes:
        return None
    
    node_data = graph.nodes[node_id]
    
    # Get relationships
    relationships = []
    for source, target, data in graph.edges(node_id, data=True):
        relationships.append({
            "edge_id": data.get('id'),
            "target_id": target,
            "type": data.get('type'),
            "weight": data.get('weight', 1.0)
        })
    
    return {
        "id": node_id,
        "text": node_data.get('text', ''),
        "metadata": node_data.get('metadata', {}),
        "relationships": relationships,
        "created_at": node_data.get('created_at', '')
    }

# ============================================================================
# NODE CRUD ENDPOINTS
# ============================================================================

@app.post("/nodes", response_model=NodeResponse)
async def create_node(node: NodeCreate):
    """Create a new node with text, metadata, and embedding"""
    node_id = str(uuid.uuid4())
    
    # Generate or use provided embedding
    if node.embedding:
        embedding = np.array(node.embedding)
    else:
        embedding = generate_embedding(node.text)
    
    # Store embedding
    embeddings_store[node_id] = embedding
    
    # Add node to graph
    graph.add_node(
        node_id,
        text=node.text,
        metadata=node.metadata,
        created_at=datetime.now().isoformat()
    )
    
    return get_node_data(node_id)

@app.get("/nodes/{node_id}", response_model=NodeResponse)
async def get_node(node_id: str):
    """Get a node by ID with all its relationships"""
    node_data = get_node_data(node_id)
    if not node_data:
        raise HTTPException(status_code=404, detail="Node not found")
    return node_data

@app.put("/nodes/{node_id}", response_model=NodeResponse)
async def update_node(node_id: str, update: NodeUpdate):
    """Update node metadata or regenerate embeddings"""
    if node_id not in graph.nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node_data = graph.nodes[node_id]
    
    # Update text if provided
    if update.text:
        node_data['text'] = update.text
        if update.regenerate_embedding:
            embeddings_store[node_id] = generate_embedding(update.text)
    
    # Update metadata if provided
    if update.metadata:
        node_data['metadata'].update(update.metadata)
    
    return get_node_data(node_id)

@app.delete("/nodes/{node_id}")
async def delete_node(node_id: str):
    """Delete a node and all its edges"""
    if node_id not in graph.nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Remove from graph
    graph.remove_node(node_id)
    
    # Remove embedding
    if node_id in embeddings_store:
        del embeddings_store[node_id]
    
    return {"message": "Node deleted successfully", "node_id": node_id}

# ============================================================================
# EDGE CRUD ENDPOINTS
# ============================================================================

@app.post("/edges", response_model=EdgeResponse)
async def create_edge(edge: EdgeCreate):
    """Create a relationship between two nodes"""
    # Validate nodes exist
    if edge.source_id not in graph.nodes:
        raise HTTPException(status_code=404, detail=f"Source node {edge.source_id} not found")
    if edge.target_id not in graph.nodes:
        raise HTTPException(status_code=404, detail=f"Target node {edge.target_id} not found")
    
    edge_id = str(uuid.uuid4())
    
    # Add edge to graph
    graph.add_edge(
        edge.source_id,
        edge.target_id,
        id=edge_id,
        type=edge.relationship_type,
        weight=edge.weight
    )
    
    return EdgeResponse(
        id=edge_id,
        source_id=edge.source_id,
        target_id=edge.target_id,
        relationship_type=edge.relationship_type,
        weight=edge.weight
    )

@app.get("/edges/{edge_id}", response_model=EdgeResponse)
async def get_edge(edge_id: str):
    """Get edge details by ID"""
    for source, target, data in graph.edges(data=True):
        if data.get('id') == edge_id:
            return EdgeResponse(
                id=edge_id,
                source_id=source,
                target_id=target,
                relationship_type=data.get('type'),
                weight=data.get('weight', 1.0)
            )
    
    raise HTTPException(status_code=404, detail="Edge not found")

@app.delete("/edges/{edge_id}")
async def delete_edge(edge_id: str):
    """Delete an edge"""
    for source, target, data in graph.edges(data=True):
        if data.get('id') == edge_id:
            graph.remove_edge(source, target)
            return {"message": "Edge deleted successfully", "edge_id": edge_id}
    
    raise HTTPException(status_code=404, detail="Edge not found")

# ============================================================================
# VECTOR SEARCH ENDPOINT
# ============================================================================

@app.post("/search/vector", response_model=List[SearchResult])
async def vector_search(request: VectorSearchRequest):
    """Search nodes by semantic similarity"""
    if not embeddings_store:
        return []
    
    # Generate query embedding
    query_embedding = generate_embedding(request.query_text)
    
    # Calculate similarities
    results = []
    for node_id, node_embedding in embeddings_store.items():
        similarity = cosine_similarity(query_embedding, node_embedding)
        node_data = graph.nodes[node_id]
        
        results.append(SearchResult(
            node_id=node_id,
            text=node_data['text'],
            score=similarity,
            vector_score=similarity,
            metadata=node_data.get('metadata', {})
        ))
    
    # Sort by similarity and return top_k
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:request.top_k]

# ============================================================================
# GRAPH TRAVERSAL ENDPOINT
# ============================================================================

@app.get("/search/graph")
async def graph_traversal(start_id: str, depth: int = 2):
    """Traverse graph from a starting node up to specified depth"""
    if start_id not in graph.nodes:
        raise HTTPException(status_code=404, detail="Start node not found")
    
    # BFS traversal up to depth
    visited = set()
    queue = [(start_id, 0)]
    results = []
    
    while queue:
        current_id, current_depth = queue.pop(0)
        
        if current_id in visited or current_depth > depth:
            continue
        
        visited.add(current_id)
        node_data = get_node_data(current_id)
        results.append({
            **node_data,
            "depth": current_depth
        })
        
        # Add neighbors to queue
        if current_depth < depth:
            for neighbor in graph.successors(current_id):
                if neighbor not in visited:
                    queue.append((neighbor, current_depth + 1))
    
    return {"start_node": start_id, "max_depth": depth, "nodes": results}

# ============================================================================
# HYBRID SEARCH ENDPOINT (THE MAGIC!)
# ============================================================================

@app.post("/search/hybrid", response_model=List[SearchResult])
async def hybrid_search(request: HybridSearchRequest):
    """
    Hybrid search combining vector similarity and graph proximity
    This is the core innovation!
    """
    if not embeddings_store:
        return []
    
    # Generate query embedding
    query_embedding = generate_embedding(request.query_text)
    
    # Step 1: Get vector similarities
    vector_results = {}
    for node_id, node_embedding in embeddings_store.items():
        similarity = cosine_similarity(query_embedding, node_embedding)
        vector_results[node_id] = similarity
    
    # Step 2: Calculate graph scores
    graph_results = {}
    if request.start_node_id and request.start_node_id in graph.nodes:
        # Calculate proximity from start node
        for node_id in graph.nodes:
            proximity = calculate_graph_proximity(request.start_node_id, node_id)
            graph_results[node_id] = proximity
    else:
        # Use PageRank as a general graph importance measure
        pagerank_scores = nx.pagerank(graph, weight='weight') if graph.number_of_nodes() > 0 else {}
        graph_results = {node_id: score for node_id, score in pagerank_scores.items()}
    
    # Step 3: Combine scores (THE HYBRID MAGIC!)
    combined_results = []
    for node_id in graph.nodes:
        vector_score = vector_results.get(node_id, 0.0)
        graph_score = graph_results.get(node_id, 0.0)
        
        # Normalize graph scores to 0-1 range if using proximity
        if graph_results:
            max_graph = max(graph_results.values()) if graph_results.values() else 1.0
            graph_score = graph_score / max_graph if max_graph > 0 else 0.0
        
        # Combined score with weights
        final_score = (request.vector_weight * vector_score) + (request.graph_weight * graph_score)
        
        # Generate explanation
        explanation = f"Vector similarity: {vector_score:.3f} | Graph proximity: {graph_score:.3f}"
        if request.start_node_id and graph_score > 0:
            try:
                path = nx.shortest_path(graph, request.start_node_id, node_id)
                path_types = []
                for i in range(len(path) - 1):
                    edge_data = graph.get_edge_data(path[i], path[i+1])
                    path_types.append(edge_data.get('type', 'connected'))
                explanation += f" | Path: {' → '.join(path_types)} ({len(path)-1} hops)"
            except:
                pass
        
        node_data = graph.nodes[node_id]
        combined_results.append(SearchResult(
            node_id=node_id,
            text=node_data['text'],
            score=final_score,
            vector_score=vector_score,
            graph_score=graph_score,
            explanation=explanation,
            metadata=node_data.get('metadata', {})
        ))
    
    # Sort by combined score and return top_k
    combined_results.sort(key=lambda x: x.score, reverse=True)
    return combined_results[:request.top_k]

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """System health and statistics"""
    return {
        "status": "healthy",
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "embeddings": len(embeddings_store),
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimensions": 384
    }

@app.get("/stats")
async def get_stats():
    """Detailed system statistics"""
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "avg_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
        "is_connected": nx.is_weakly_connected(graph) if graph.number_of_nodes() > 0 else False,
        "embedding_dimensions": 384
    }

@app.post("/reset")
async def reset_database():
    """Clear all data (useful for testing)"""
    global graph, embeddings_store
    graph = nx.DiGraph()
    embeddings_store = {}
    return {"message": "Database reset successfully"}

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Vector + Graph Hybrid Database",
        "version": "1.0.0",
        "description": "A hybrid retrieval system combining vector similarity and graph relationships",
        "docs": "/docs",
        "endpoints": {
            "nodes": "/nodes",
            "edges": "/edges",
            "vector_search": "/search/vector",
            "graph_traversal": "/search/graph",
            "hybrid_search": "/search/hybrid",
            "health": "/health",
            "stats": "/stats"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)