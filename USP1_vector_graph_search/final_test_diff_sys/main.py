from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
import json
import uuid
from datetime import datetime
import asyncio
import httpx
from bs4 import BeautifulSoup
import io
import re
from collections import deque
import PyPDF2
import docx
import csv

# Initialize FastAPI
app = FastAPI(
    title="Vector + Graph Hybrid Database v3.0",
    description="Production-grade hybrid retrieval with web search, file upload, and RAG",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
TEST_MODE = False  # Set to True for test case validation with mock embeddings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma2:2b"

# Initialize embedding model
print("🚀 Loading embedding model...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ Model loaded!")

# Storage
graph = nx.DiGraph()
embeddings_store = {}  # node_id -> np.array
metadata_store = {}    # node_id -> metadata dict

# ============================================================================
# DATA MODELS
# ============================================================================

class NodeCreate(BaseModel):
    id: Optional[str] = None
    text: str
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    embedding: Optional[List[float]] = None

class NodeUpdate(BaseModel):
    text: Optional[str] = None
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    regen_embedding: bool = False

class NodeResponse(BaseModel):
    id: str
    title: Optional[str]
    text: str
    metadata: Dict[str, Any]
    embedding: List[float]
    edges: List[Dict[str, Any]]
    created_at: str

class EdgeCreate(BaseModel):
    source: str
    target: str
    type: str = "related_to"
    weight: float = 1.0

class EdgeResponse(BaseModel):
    edge_id: str
    source: str
    target: str
    type: str
    weight: float

class VectorSearchRequest(BaseModel):
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    top_k: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None

class GraphSearchRequest(BaseModel):
    start_id: str
    depth: int = 2
    type_filter: Optional[str] = None

class HybridSearchRequest(BaseModel):
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    vector_weight: float = 0.6
    graph_weight: float = 0.4
    top_k: int = 5
    anchor_node: Optional[str] = None

class SearchResult(BaseModel):
    id: str
    title: Optional[str]
    text: str
    vector_score: Optional[float]
    graph_score: Optional[float]
    final_score: float
    info: Optional[Dict[str, Any]] = {}

class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5
    auto_ingest: bool = True

class LLMRequest(BaseModel):
    query: str
    context: List[SearchResult]
    model: str = OLLAMA_MODEL

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_embedding(text: str, mock: bool = False) -> np.ndarray:
    """Generate embedding (real or mock for testing)"""
    if mock or TEST_MODE:
        # Mock embeddings for test case validation
        mock_map = {
            "redis caching": [0.88, 0.12, 0.02, 0, 0, 0],
            "cache invalidation": [0.80, 0.15, 0, 0, 0, 0],
            "redisgraph": [0.70, 0.10, 0.60, 0, 0, 0],
            "distributed systems": [0.10, 0.05, 0, 0.90, 0, 0],
            "graph algorithms": [0.05, 0, 0.90, 0.10, 0, 0],
            "redis graph": [0.60, 0.05, 0.50, 0, 0.10, 0],
        }
        
        text_lower = text.lower()
        for key, vec in mock_map.items():
            if key in text_lower:
                return np.array(vec, dtype=np.float32)
        
        # Default mock vector
        return np.array([0.5, 0.5, 0, 0, 0, 0], dtype=np.float32)
    
    return embedding_model.encode(text, convert_to_numpy=True)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    if vec1.shape != vec2.shape:
        # Pad shorter vector with zeros
        max_len = max(len(vec1), len(vec2))
        vec1 = np.pad(vec1, (0, max_len - len(vec1)), 'constant')
        vec2 = np.pad(vec2, (0, max_len - len(vec2)), 'constant')
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))

def calculate_graph_proximity(source_id: str, target_id: str) -> Tuple[float, int]:
    """Calculate graph proximity score and hop distance"""
    if source_id == target_id:
        return 1.0, 0
    
    try:
        # BFS to find shortest path (handles both directed and undirected)
        if nx.has_path(graph.to_undirected(), source_id, target_id):
            path_length = nx.shortest_path_length(graph.to_undirected(), source_id, target_id)
            proximity = 1.0 / (1.0 + path_length)
            return proximity, path_length
        else:
            return 0.0, -1
    except:
        return 0.0, -1

# ============================================================================
# FILE PARSING FUNCTIONS
# ============================================================================

def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF"""
    try:
        pdf_file = io.BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing error: {str(e)}")

def parse_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX"""
    try:
        doc_file = io.BytesIO(file_bytes)
        doc = docx.Document(doc_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DOCX parsing error: {str(e)}")

def parse_txt(file_bytes: bytes) -> str:
    """Extract text from TXT"""
    try:
        return file_bytes.decode('utf-8').strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"TXT parsing error: {str(e)}")

def parse_csv(file_bytes: bytes) -> List[Dict[str, str]]:
    """Parse CSV and return list of rows as dicts"""
    try:
        csv_file = io.StringIO(file_bytes.decode('utf-8'))
        reader = csv.DictReader(csv_file)
        return list(reader)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")

def parse_json(file_bytes: bytes) -> Any:
    """Parse JSON"""
    try:
        return json.loads(file_bytes.decode('utf-8'))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON parsing error: {str(e)}")

# ============================================================================
# WEB SEARCH & SCRAPING (REAL IMPLEMENTATION)
# ============================================================================

async def search_web_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search using DuckDuckGo HTML (free, no API key needed)"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # DuckDuckGo HTML search
            url = "https://html.duckduckgo.com/html/"
            data = {"q": query}
            headers = {"User-Agent": "Mozilla/5.0"}
            
            response = await client.post(url, data=data, headers=headers, follow_redirects=True)
            
            if response.status_code != 200:
                print(f"⚠️ DuckDuckGo search failed: {response.status_code}")
                return []
            
            # Parse results
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for result in soup.find_all('div', class_='result')[:max_results]:
                title_tag = result.find('a', class_='result__a')
                snippet_tag = result.find('a', class_='result__snippet')
                
                if title_tag:
                    title = title_tag.get_text(strip=True)
                    url = title_tag.get('href', '')
                    snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                    
                    if url and title:
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet
                        })
            
            print(f"✅ Found {len(results)} search results")
            return results
            
    except Exception as e:
        print(f"❌ Web search error: {e}")
        return []

async def scrape_webpage(url: str) -> Optional[str]:
    """Scrape content from a webpage"""
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = await client.get(url, headers=headers)
            
            if response.status_code != 200:
                return None
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            # Limit length
            if len(text) > 3000:
                text = text[:3000] + "..."
            
            return text
            
    except Exception as e:
        print(f"❌ Scraping error for {url}: {e}")
        return None

# ============================================================================
# NODE CRUD ENDPOINTS
# ============================================================================

@app.post("/nodes", status_code=201)
async def create_node(node: NodeCreate):
    """Create a new node with text, metadata, and embedding"""
    node_id = node.id if node.id else str(uuid.uuid4())
    
    # Check if node already exists
    if node_id in graph.nodes:
        raise HTTPException(status_code=400, detail=f"Node {node_id} already exists")
    
    # Generate or use provided embedding
    if node.embedding:
        embedding = np.array(node.embedding, dtype=np.float32)
    else:
        embedding = generate_embedding(node.text, mock=TEST_MODE)
    
    # Store embedding
    embeddings_store[node_id] = embedding
    
    # Store metadata
    metadata_store[node_id] = {
        "title": node.title,
        "text": node.text,
        "metadata": node.metadata,
        "created_at": datetime.now().isoformat()
    }
    
    # Add node to graph
    graph.add_node(node_id)
    
    return {
        "status": "created",
        "id": node_id,
        "created_at": metadata_store[node_id]["created_at"]
    }

@app.get("/nodes/{node_id}")
async def get_node(node_id: str):
    """Get node with all its properties and relationships"""
    if node_id not in graph.nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Get node data
    node_data = metadata_store.get(node_id, {})
    embedding = embeddings_store.get(node_id, np.array([]))
    
    # Get edges
    edges = []
    for source, target, data in graph.edges(node_id, data=True):
        edges.append({
            "id": data.get('id', ''),
            "target": target,
            "type": data.get('type', 'related_to'),
            "weight": data.get('weight', 1.0)
        })
    
    # Also get incoming edges
    for source, target, data in graph.in_edges(node_id, data=True):
        edges.append({
            "id": data.get('id', ''),
            "source": source,
            "type": data.get('type', 'related_to') + "_from",
            "weight": data.get('weight', 1.0)
        })
    
    return {
        "id": node_id,
        "title": node_data.get("title"),
        "text": node_data.get("text", ""),
        "metadata": node_data.get("metadata", {}),
        "embedding": embedding.tolist(),
        "edges": edges,
        "created_at": node_data.get("created_at", "")
    }

@app.put("/nodes/{node_id}")
async def update_node(node_id: str, update: NodeUpdate):
    """Update node properties"""
    if node_id not in graph.nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node_data = metadata_store[node_id]
    
    # Update text
    if update.text:
        node_data["text"] = update.text
        if update.regen_embedding:
            new_embedding = generate_embedding(update.text, mock=TEST_MODE)
            embeddings_store[node_id] = new_embedding
    
    # Update title
    if update.title:
        node_data["title"] = update.title
    
    # Update metadata
    if update.metadata:
        node_data["metadata"].update(update.metadata)
    
    return {"status": "updated", "id": node_id}

@app.delete("/nodes/{node_id}", status_code=204)
async def delete_node(node_id: str):
    """Delete node and all associated edges"""
    if node_id not in graph.nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Count edges to be removed
    edge_count = graph.degree(node_id)
    
    # Remove from graph (automatically removes all edges)
    graph.remove_node(node_id)
    
    # Remove from stores
    if node_id in embeddings_store:
        del embeddings_store[node_id]
    if node_id in metadata_store:
        del metadata_store[node_id]
    
    return {"status": "deleted", "id": node_id, "removed_edges_count": edge_count}

# ============================================================================
# EDGE CRUD ENDPOINTS
# ============================================================================

@app.post("/edges", status_code=201)
async def create_edge(edge: EdgeCreate):
    """Create a relationship between two nodes"""
    if edge.source not in graph.nodes:
        raise HTTPException(status_code=404, detail=f"Source node {edge.source} not found")
    if edge.target not in graph.nodes:
        raise HTTPException(status_code=404, detail=f"Target node {edge.target} not found")
    
    edge_id = str(uuid.uuid4())
    
    graph.add_edge(
        edge.source,
        edge.target,
        id=edge_id,
        type=edge.type,
        weight=edge.weight
    )
    
    return {
        "status": "created",
        "edge_id": edge_id,
        "source": edge.source,
        "target": edge.target
    }

@app.get("/edges/{edge_id}")
async def get_edge(edge_id: str):
    """Get edge details"""
    for source, target, data in graph.edges(data=True):
        if data.get('id') == edge_id:
            return {
                "edge_id": edge_id,
                "source": source,
                "target": target,
                "type": data.get('type'),
                "weight": data.get('weight')
            }
    
    raise HTTPException(status_code=404, detail="Edge not found")

@app.delete("/edges/{edge_id}", status_code=204)
async def delete_edge(edge_id: str):
    """Delete an edge"""
    for source, target, data in list(graph.edges(data=True)):
        if data.get('id') == edge_id:
            graph.remove_edge(source, target)
            return {"status": "deleted", "edge_id": edge_id}
    
    raise HTTPException(status_code=404, detail="Edge not found")

# ============================================================================
# VECTOR SEARCH
# ============================================================================

@app.post("/search/vector")
async def vector_search(request: VectorSearchRequest):
    """Vector-only search with cosine similarity"""
    # Get query embedding
    if request.query_embedding:
        query_embedding = np.array(request.query_embedding, dtype=np.float32)
    elif request.query_text:
        query_embedding = generate_embedding(request.query_text, mock=TEST_MODE)
    else:
        raise HTTPException(status_code=400, detail="Provide query_text or query_embedding")
    
    # Calculate similarities
    results = []
    for node_id, node_embedding in embeddings_store.items():
        # Apply metadata filter if provided
        if request.filter_metadata:
            node_meta = metadata_store[node_id].get("metadata", {})
            match = all(node_meta.get(k) == v for k, v in request.filter_metadata.items())
            if not match:
                continue
        
        similarity = cosine_similarity(query_embedding, node_embedding)
        
        node_data = metadata_store[node_id]
        results.append({
            "id": node_id,
            "title": node_data.get("title"),
            "text": node_data.get("text", "")[:200] + "...",
            "vector_score": similarity
        })
    
    # Sort by similarity
    results.sort(key=lambda x: x['vector_score'], reverse=True)
    
    return {
        "query_text": request.query_text,
        "results": results[:request.top_k]
    }

# ============================================================================
# GRAPH TRAVERSAL
# ============================================================================

@app.get("/search/graph")
async def graph_traversal(start_id: str, depth: int = 2, type_filter: Optional[str] = None):
    """BFS graph traversal up to specified depth"""
    if start_id not in graph.nodes:
        raise HTTPException(status_code=404, detail="Start node not found")
    
    # BFS traversal
    visited = {start_id: 0}
    queue = deque([(start_id, 0)])
    results = []
    
    while queue:
        current_id, current_depth = queue.popleft()
        
        if current_depth > depth:
            continue
        
        # Get neighbors (both outgoing and incoming for undirected-like behavior)
        neighbors = []
        
        # Outgoing edges
        for _, target, data in graph.out_edges(current_id, data=True):
            if type_filter is None or data.get('type') == type_filter:
                neighbors.append((target, data))
        
        # Incoming edges (treat as undirected)
        for source, _, data in graph.in_edges(current_id, data=True):
            if type_filter is None or data.get('type') == type_filter:
                neighbors.append((source, data))
        
        for neighbor_id, edge_data in neighbors:
            if neighbor_id not in visited and current_depth < depth:
                visited[neighbor_id] = current_depth + 1
                queue.append((neighbor_id, current_depth + 1))
                
                results.append({
                    "id": neighbor_id,
                    "hop": current_depth + 1,
                    "edge": edge_data.get('type'),
                    "weight": edge_data.get('weight')
                })
    
    return {
        "start_id": start_id,
        "depth": depth,
        "nodes": results
    }

# ============================================================================
# HYBRID SEARCH (THE CORE FEATURE!)
# ============================================================================

@app.post("/search/hybrid")
async def hybrid_search(request: HybridSearchRequest):
    """Hybrid search combining vector similarity and graph proximity"""
    # Get query embedding
    if request.query_embedding:
        query_embedding = np.array(request.query_embedding, dtype=np.float32)
    elif request.query_text:
        query_embedding = generate_embedding(request.query_text, mock=TEST_MODE)
    else:
        raise HTTPException(status_code=400, detail="Provide query_text or query_embedding")
    
    # Calculate vector scores
    vector_scores = {}
    for node_id, node_embedding in embeddings_store.items():
        vector_scores[node_id] = cosine_similarity(query_embedding, node_embedding)
    
    # Determine anchor node for graph proximity
    if request.anchor_node and request.anchor_node in graph.nodes:
        anchor = request.anchor_node
    else:
        # Use highest vector similarity node as anchor
        anchor = max(vector_scores.items(), key=lambda x: x[1])[0] if vector_scores else None
    
    # Calculate graph scores
    graph_scores = {}
    if anchor:
        for node_id in graph.nodes:
            proximity, hop = calculate_graph_proximity(anchor, node_id)
            graph_scores[node_id] = proximity
    else:
        # Fallback: use PageRank
        if graph.number_of_nodes() > 0:
            pagerank = nx.pagerank(graph, weight='weight')
            max_pr = max(pagerank.values()) if pagerank else 1.0
            graph_scores = {nid: score/max_pr for nid, score in pagerank.items()}
        else:
            graph_scores = {nid: 0.0 for nid in graph.nodes}
    
    # Combine scores
    results = []
    for node_id in graph.nodes:
        vector_score = vector_scores.get(node_id, 0.0)
        graph_score = graph_scores.get(node_id, 0.0)
        final_score = (request.vector_weight * vector_score) + (request.graph_weight * graph_score)
        
        node_data = metadata_store.get(node_id, {})
        _, hop = calculate_graph_proximity(anchor, node_id) if anchor else (0, -1)
        
        results.append({
            "id": node_id,
            "title": node_data.get("title"),
            "text": node_data.get("text", "")[:300],
            "vector_score": round(vector_score, 8),
            "graph_score": round(graph_score, 8),
            "final_score": round(final_score, 8),
            "info": {"hop": hop}
        })
    
    # Sort by final score
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    return {
        "query_text": request.query_text,
        "vector_weight": request.vector_weight,
        "graph_weight": request.graph_weight,
        "results": results[:request.top_k]
    }

# ============================================================================
# FILE UPLOAD ENDPOINT
# ============================================================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    create_edges: bool = Form(True),
    similarity_threshold: float = Form(0.6)
):
    """Upload and ingest file (PDF, TXT, DOCX, CSV, JSON)"""
    file_bytes = await file.read()
    filename = file.filename
    file_ext = filename.split('.')[-1].lower()
    
    created_nodes = []
    created_edges = []
    
    try:
        if file_ext == 'pdf':
            text = parse_pdf(file_bytes)
            node_id = await create_node(NodeCreate(
                id=f"doc_{uuid.uuid4().hex[:8]}",
                text=text,
                title=filename,
                metadata={"type": "document", "source": "pdf", "filename": filename}
            ))
            created_nodes.append(node_id["id"])
            
        elif file_ext == 'docx':
            text = parse_docx(file_bytes)
            node_id = await create_node(NodeCreate(
                id=f"doc_{uuid.uuid4().hex[:8]}",
                text=text,
                title=filename,
                metadata={"type": "document", "source": "docx", "filename": filename}
            ))
            created_nodes.append(node_id["id"])
            
        elif file_ext == 'txt':
            text = parse_txt(file_bytes)
            node_id = await create_node(NodeCreate(
                id=f"doc_{uuid.uuid4().hex[:8]}",
                text=text,
                title=filename,
                metadata={"type": "document", "source": "txt", "filename": filename}
            ))
            created_nodes.append(node_id["id"])
            
        elif file_ext == 'csv':
            rows = parse_csv(file_bytes)
            for i, row in enumerate(rows):
                text = " | ".join([f"{k}: {v}" for k, v in row.items()])
                node_id = await create_node(NodeCreate(
                    id=f"csv_{uuid.uuid4().hex[:8]}",
                    text=text,
                    title=f"{filename} - Row {i+1}",
                    metadata={"type": "csv_row", "source": "csv", "filename": filename, "row": i}
                ))
                created_nodes.append(node_id["id"])
                
        elif file_ext == 'json':
            data = parse_json(file_bytes)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    text = json.dumps(item, indent=2)
                    node_id = await create_node(NodeCreate(
                        id=f"json_{uuid.uuid4().hex[:8]}",
                        text=text,
                        title=f"{filename} - Item {i+1}",
                        metadata={"type": "json_item", "source": "json", "filename": filename}
                    ))
                    created_nodes.append(node_id["id"])
            else:
                text = json.dumps(data, indent=2)
                node_id = await create_node(NodeCreate(
                    id=f"json_{uuid.uuid4().hex[:8]}",
                    text=text,
                    title=filename,
                    metadata={"type": "json_doc", "source": "json", "filename": filename}
                ))
                created_nodes.append(node_id["id"])
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        
        # Create similarity edges if requested
        if create_edges and len(created_nodes) > 1:
            for i, node1 in enumerate(created_nodes):
                for node2 in created_nodes[i+1:]:
                    emb1 = embeddings_store[node1]
                    emb2 = embeddings_store[node2]
                    similarity = cosine_similarity(emb1, emb2)
                    
                    if similarity >= similarity_threshold:
                        edge = await create_edge(EdgeCreate(
                            source=node1,
                            target=node2,
                            type="similar_content",
                            weight=similarity
                        ))
                        created_edges.append(edge["edge_id"])
        
        return {
            "status": "success",
            "filename": filename,
            "nodes_created": len(created_nodes),
            "edges_created": len(created_edges),
            "node_ids": created_nodes
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# WEB SEARCH ENDPOINT
# ============================================================================

@app.post("/search/web")
async def web_search(request: WebSearchRequest):
    """Search web and optionally ingest results"""
    # Search
    search_results = await search_web_duckduckgo(request.query, request.max_results)
    
    if not search_results:
        return {"status": "no_results", "message": "No search results found"}
    
    created_nodes = []
    
    if request.auto_ingest:
        # Scrape and ingest each result
        for result in search_results:
            content = await scrape_webpage(result['url'])
            
            if content:
                text = f"{result['title']}\n\n{content}"
                node_response = await create_node(NodeCreate(
                    id=f"web_{uuid.uuid4().hex[:8]}",
                    text=text,
                    title=result['title'],
                    metadata={
                        "type": "web_page",
                        "url": result['url'],
                        "snippet": result['snippet'],
                        "query": request.query
                    }
                ))
                created_nodes.append(node_response["id"])
    
    return {
        "status": "success",
        "query": request.query,
        "results_found": len(search_results),
        "nodes_created": len(created_nodes),
        "node_ids": created_nodes,
        "search_results": search_results
    }

# ============================================================================
# LLM ENHANCEMENT (OPTIONAL)
# ============================================================================

@app.post("/llm/enhance")
async def enhance_with_llm(request: LLMRequest):
    """Enhance search results with LLM (optional feature)"""
    try:
        # Build context
        context_text = "# Search Results:\n\n"
        for i, result in enumerate(request.context, 1):
            context_text += f"## Source {i}: {result.title}\n"
            context_text += f"{result.text[:300]}...\n\n"
        
        # Build prompt
        prompt = f"""Based on these search results, provide a comprehensive answer to the query.

Query: {request.query}

{context_text}

Provide a clear, structured answer with citations [Source N]."""
        
        # Call Ollama
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": request.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "response": result.get('response', '')
                }
            else:
                return {"status": "error", "message": "Ollama not available"}
                
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """System health check"""
    ollama_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_status = "connected" if response.status_code == 200 else "error"
    except:
        ollama_status = "disconnected"
    
    return {
        "status": "healthy",
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "test_mode": TEST_MODE,
        "ollama": ollama_status
    }

@app.get("/graph/export")
async def export_graph():
    """Export graph for visualization"""
    nodes = []
    edges = []
    
    for node_id in graph.nodes:
        node_data = metadata_store.get(node_id, {})
        nodes.append({
            "id": node_id,
            "label": node_data.get("title", node_id)[:30],
            "title": node_data.get("title", ""),
            "type": node_data.get("metadata", {}).get("type", "unknown")
        })
    
    for source, target, data in graph.edges(data=True):
        edges.append({
            "from": source,
            "to": target,
            "label": data.get('type', ''),
            "weight": data.get('weight', 1.0)
        })
    
    return {"nodes": nodes, "edges": edges}

@app.post("/reset")
async def reset_system():
    """Clear all data"""
    global graph, embeddings_store, metadata_store
    graph = nx.DiGraph()
    embeddings_store = {}
    metadata_store = {}
    return {"status": "reset", "message": "All data cleared"}

@app.get("/")
async def root():
    return {
        "name": "Vector + Graph Hybrid Database v3.0",
        "version": "3.0.0",
        "features": [
            "Full CRUD operations",
            "Vector search with filtering",
            "Graph traversal (BFS)",
            "Hybrid search (vector + graph)",
            "File upload (PDF, DOCX, TXT, CSV, JSON)",
            "Real web search & scraping",
            "Optional LLM enhancement",
            "Graph visualization export",
            "Test mode for validation"
        ],
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Vector + Graph Hybrid Database v3.0")
    print(f"📊 Test Mode: {TEST_MODE}")
    print(f"🤖 Ollama: {OLLAMA_BASE_URL}\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)