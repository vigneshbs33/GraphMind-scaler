"""FastAPI application for GraphMind - Production-ready."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request

from .config import settings
from .evaluation import SearchEvaluator
from .exceptions import GraphMindError, ValidationError, NotFoundError, StorageError, LLMError
from .ingestion import IngestionPipeline, parse_metadata
from .llm_processor import get_llm_processor
from .middleware import LoggingMiddleware, SecurityHeadersMiddleware
from .models import (
    ComparisonRequest,
    ComparisonResponse,
    EdgeCreate,
    EdgeInfo,
    GraphTraversalRequest,
    GraphTraversalResponse,
    HybridSearchRequest,
    HybridSearchResult,
    NodeCreate,
    NodeResponse,
    NodeUpdate,
    SearchQuery,
    SearchResult,
    TraversalNode,
    VectorSearchRequest,
)
from .storage import GraphMindStorage

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GraphMind API",
    version="1.0.0",
    description="Hybrid Vector + Graph Database for Efficient AI Retrieval",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Add middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Exception handlers
@app.exception_handler(GraphMindError)
async def graphmind_exception_handler(request: Request, exc: GraphMindError):
    """Handle custom GraphMind exceptions."""
    logger.error("GraphMind error: %s", exc.message, exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
            "type": exc.__class__.__name__,
        },
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.warning("Validation error: %s", exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "details": exc.errors(),
        },
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error("Unexpected error: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": str(exc) if settings.DEBUG else "An unexpected error occurred",
        },
    )


@app.on_event("startup")
async def _startup() -> None:
    """Initialize application on startup."""
    try:
        logger.info("Starting GraphMind API...")
        logger.info("Environment: %s, Debug: %s", settings.ENVIRONMENT, settings.DEBUG)
        
        storage = GraphMindStorage(settings.CHROMA_DIR, settings.EMBEDDING_MODEL)
        logger.info("Storage initialized: %d nodes, %d edges", 
                   storage.graph.number_of_nodes(), 
                   storage.graph.number_of_edges())
        
        # Use Gemini API key if provider is gemini, otherwise use Claude key
        api_key = settings.GEMINI_API_KEY if settings.LLM_PROVIDER == "gemini" else settings.CLAUDE_API_KEY
        llm = get_llm_processor(settings.LLM_PROVIDER, api_key)
        logger.info("LLM processor initialized: %s", settings.LLM_PROVIDER)
        
        app.state.storage = storage
        app.state.ingestion = IngestionPipeline(storage)
        app.state.evaluator = SearchEvaluator(storage, llm)
        app.state.llm = llm
        
        logger.info("GraphMind API started successfully")
    except Exception as e:
        logger.error("Failed to start application: %s", str(e), exc_info=True)
        raise


def get_storage() -> GraphMindStorage:
    return app.state.storage


def get_ingestion() -> IngestionPipeline:
    return app.state.ingestion


def get_evaluator() -> SearchEvaluator:
    return app.state.evaluator


@app.get("/")
async def root():
    """Serve frontend or redirect to docs."""
    frontend_path = settings.BASE_DIR / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health(storage: GraphMindStorage = Depends(get_storage)) -> Dict[str, Any]:
    """Health check endpoint with system status."""
    stats = storage.get_stats()
    return {
        "status": "ok",
        "environment": settings.ENVIRONMENT,
        "storage": {
            "nodes": int(stats.get("node_count", 0)),
            "edges": int(stats.get("edge_count", 0)),
            "vectors": int(stats.get("vector_count", 0)),
        },
    }


@app.get("/stats")
async def stats(storage: GraphMindStorage = Depends(get_storage)) -> Dict[str, float]:
    return storage.get_stats()


@app.post("/nodes", response_model=SearchResult)
async def create_node(
    payload: NodeCreate, 
    storage: GraphMindStorage = Depends(get_storage)
) -> SearchResult:
    """Create a new node in the knowledge graph."""
    try:
        # Validate graph size limits
        current_nodes = storage.graph.number_of_nodes()
        if current_nodes >= settings.MAX_GRAPH_NODES:
            raise ValidationError(
                f"Graph node limit reached ({settings.MAX_GRAPH_NODES})",
                {"current_nodes": current_nodes, "limit": settings.MAX_GRAPH_NODES}
            )
        
        node_id = storage.add_node(payload.id, payload.content, payload.metadata)
        logger.info("Node created: %s", node_id)
        
        return SearchResult(
            node_id=node_id,
            content=payload.content,
            score=1.0,
            metadata=payload.metadata,
            reasoning=payload.node_type or "",
        )
    except ValueError as e:
        raise ValidationError(str(e))
    except Exception as e:
        logger.error("Failed to create node: %s", str(e), exc_info=True)
        raise StorageError(f"Failed to create node: {str(e)}")


@app.get("/nodes/{node_id}", response_model=NodeResponse)
async def get_node(
    node_id: str,
    storage: GraphMindStorage = Depends(get_storage)
) -> NodeResponse:
    """Get a node by ID with all its relationships."""
    try:
        node_data = storage.get_node(node_id)
        
        # Convert relationships to EdgeInfo models
        relationships = [
            EdgeInfo(
                edge_id=rel["edge_id"],
                source_id=rel["source_id"],
                target_id=rel["target_id"],
                relationship=rel["relationship"],
                weight=rel["weight"],
            )
            for rel in node_data["relationships"]
        ]
        
        return NodeResponse(
            node_id=node_data["node_id"],
            content=node_data["content"],
            metadata=node_data["metadata"],
            relationships=relationships,
        )
    except NotFoundError:
        raise
    except Exception as e:
        logger.error("Failed to get node: %s", str(e), exc_info=True)
        raise StorageError(f"Failed to get node: {str(e)}")


@app.put("/nodes/{node_id}", response_model=NodeResponse)
async def update_node(
    node_id: str,
    payload: NodeUpdate,
    storage: GraphMindStorage = Depends(get_storage)
) -> NodeResponse:
    """Update a node's content and/or metadata."""
    try:
        updated_data = storage.update_node(
            node_id,
            content=payload.content,
            metadata=payload.metadata,
            node_type=payload.node_type
        )
        
        # Convert relationships to EdgeInfo models
        relationships = [
            EdgeInfo(
                edge_id=rel["edge_id"],
                source_id=rel["source_id"],
                target_id=rel["target_id"],
                relationship=rel["relationship"],
                weight=rel["weight"],
            )
            for rel in updated_data["relationships"]
        ]
        
        logger.info("Node updated: %s", node_id)
        
        return NodeResponse(
            node_id=updated_data["node_id"],
            content=updated_data["content"],
            metadata=updated_data["metadata"],
            relationships=relationships,
        )
    except NotFoundError:
        raise
    except Exception as e:
        logger.error("Failed to update node: %s", str(e), exc_info=True)
        raise StorageError(f"Failed to update node: {str(e)}")


@app.delete("/nodes/{node_id}")
async def delete_node(
    node_id: str,
    storage: GraphMindStorage = Depends(get_storage)
) -> Dict[str, Any]:
    """Delete a node and all its associated edges."""
    try:
        result = storage.delete_node(node_id)
        logger.info("Node deleted: %s, edges removed: %d", node_id, result["edges_removed"])
        return result
    except NotFoundError:
        raise
    except Exception as e:
        logger.error("Failed to delete node: %s", str(e), exc_info=True)
        raise StorageError(f"Failed to delete node: {str(e)}")


@app.post("/edges")
async def create_edge(
    payload: EdgeCreate, 
    storage: GraphMindStorage = Depends(get_storage)
) -> Dict[str, str]:
    """Create an edge between two nodes."""
    try:
        # Validate graph size limits
        current_edges = storage.graph.number_of_edges()
        if current_edges >= settings.MAX_GRAPH_EDGES:
            raise ValidationError(
                f"Graph edge limit reached ({settings.MAX_GRAPH_EDGES})",
                {"current_edges": current_edges, "limit": settings.MAX_GRAPH_EDGES}
            )
        
        storage.add_edge(payload.source_id, payload.target_id, payload.relationship, payload.weight)
        logger.info("Edge created: %s -> %s", payload.source_id, payload.target_id)
        return {"status": "ok", "source": payload.source_id, "target": payload.target_id}
    except ValueError as e:
        raise ValidationError(str(e))
    except Exception as e:
        logger.error("Failed to create edge: %s", str(e), exc_info=True)
        raise StorageError(f"Failed to create edge: {str(e)}")


@app.get("/edges/{edge_id}")
async def get_edge_by_id(
    edge_id: str,
    storage: GraphMindStorage = Depends(get_storage)
) -> Dict[str, Any]:
    """Get edge details by edge_id."""
    try:
        edge_data = storage.get_edge(edge_id)
        return edge_data
    except NotFoundError:
        raise
    except ValidationError:
        raise
    except Exception as e:
        logger.error("Failed to get edge: %s", str(e), exc_info=True)
        raise StorageError(f"Failed to get edge: {str(e)}")


@app.get("/edges")
async def get_edge_by_nodes(
    source_id: str,
    target_id: str,
    storage: GraphMindStorage = Depends(get_storage)
) -> Dict[str, Any]:
    """Get edge details by source_id and target_id."""
    try:
        edge_data = storage.get_edge("", source_id=source_id, target_id=target_id)
        return edge_data
    except NotFoundError:
        raise
    except ValidationError:
        raise
    except Exception as e:
        logger.error("Failed to get edge: %s", str(e), exc_info=True)
        raise StorageError(f"Failed to get edge: {str(e)}")


@app.get("/graph")
async def graph_snapshot(
    page: int = 1,
    limit: int = 100,
    node_type: Optional[str] = None,
    storage: GraphMindStorage = Depends(get_storage)
) -> Dict[str, Any]:
    """Get full graph snapshot for visualization with pagination and filtering."""
    try:
        all_nodes = [
            {
                "node_id": str(node_id),
                "content": attrs.get("content", ""),
                **{k: v for k, v in attrs.items() if k != "content"}
            }
            for node_id, attrs in storage.graph.nodes(data=True)
        ]
        
        all_edges = [
            {
                "source": str(src),
                "target": str(dst),
                "relationship": attrs.get("relationship", ""),
                "weight": attrs.get("weight", 1.0),
                **{k: v for k, v in attrs.items() if k not in ("relationship", "weight")}
            }
            for src, dst, attrs in storage.graph.edges(data=True)
        ]
        
        # Filter by node_type if provided
        if node_type:
            all_nodes = [n for n in all_nodes if n.get("node_type") == node_type]
            # Filter edges to only include filtered nodes
            node_ids = {n["node_id"] for n in all_nodes}
            all_edges = [e for e in all_edges if e["source"] in node_ids and e["target"] in node_ids]
        
        # Pagination
        total_nodes = len(all_nodes)
        total_edges = len(all_edges)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        
        nodes = all_nodes[start_idx:end_idx]
        edges = all_edges[start_idx:end_idx]
        
        logger.debug("Graph snapshot: %d nodes, %d edges (page %d, limit %d)", len(nodes), len(edges), page, limit)
        return {
            "nodes": nodes,
            "edges": edges,
            "total": total_nodes,
            "page": page,
            "limit": limit,
            "total_pages": (total_nodes + limit - 1) // limit,
        }
    except Exception as e:
        logger.error("Failed to get graph snapshot: %s", str(e), exc_info=True)
        raise StorageError(f"Failed to get graph snapshot: {str(e)}")


@app.post("/search/multi-hop")
async def multi_hop_reasoning(
    query: str,
    max_hops: int = 3,
    storage: GraphMindStorage = Depends(get_storage),
) -> Dict[str, Any]:
    """Multi-hop reasoning query starting from vector search seeds."""
    try:
        if max_hops < 1 or max_hops > 10:
            raise ValidationError("max_hops must be between 1 and 10")
        
        logger.info("Multi-hop reasoning: query=%s, max_hops=%d", query[:50], max_hops)
        
        results = storage.multi_hop_reasoning(query, max_hops)
        
        return {
            "query": query,
            "max_hops": max_hops,
            "results": results,
            "count": len(results),
        }
    except Exception as e:
        logger.error("Multi-hop reasoning failed: %s", str(e), exc_info=True)
        raise StorageError(f"Multi-hop reasoning failed: {str(e)}")


@app.post("/search")
async def search(
    payload: SearchQuery,
    storage: GraphMindStorage = Depends(get_storage),
) -> Dict[str, Any]:
    """Search the knowledge graph using vector, graph, or hybrid mode."""
    try:
        # Validate top_k
        if payload.top_k > settings.MAX_TOP_K:
            raise ValidationError(
                f"top_k exceeds maximum ({settings.MAX_TOP_K})",
                {"requested": payload.top_k, "maximum": settings.MAX_TOP_K}
            )
        
        logger.info("Search: mode=%s, query=%s, top_k=%d", payload.mode, payload.query[:50], payload.top_k)
        
        if payload.mode == "vector":
            raw = storage.vector_search(payload.query, payload.top_k)
        elif payload.mode == "graph":
            raw = storage.graph_search(payload.query, payload.top_k)
        else:
            raw = storage.hybrid_search(payload.query, payload.top_k, payload.alpha)
        
        results = [
            SearchResult(
                node_id=item["node_id"],
                content=item.get("content", ""),
                score=item.get("score", 0.0),
                metadata=item.get("metadata", {}),
                reasoning=item.get("reasoning", ""),
            )
            for item in raw
        ]
        
        logger.info("Search completed: %d results", len(results))
        return {"results": results, "mode": payload.mode, "query": payload.query}
    except Exception as e:
        logger.error("Search failed: %s", str(e), exc_info=True)
        raise StorageError(f"Search failed: {str(e)}")


@app.post("/search/vector")
async def vector_search(
    payload: VectorSearchRequest,
    storage: GraphMindStorage = Depends(get_storage),
) -> Dict[str, Any]:
    """Vector-only search by cosine similarity."""
    try:
        if payload.top_k > settings.MAX_TOP_K:
            raise ValidationError(
                f"top_k exceeds maximum ({settings.MAX_TOP_K})",
                {"requested": payload.top_k, "maximum": settings.MAX_TOP_K}
            )
        
        logger.info("Vector search: query=%s, top_k=%d", payload.query_text[:50], payload.top_k)
        
        raw = storage.vector_search(payload.query_text, payload.top_k)
        
        results = [
            SearchResult(
                node_id=item["node_id"],
                content=item.get("content", ""),
                score=item.get("score", 0.0),
                metadata=item.get("metadata", {}),
                reasoning="",
            )
            for item in raw
        ]
        
        return {"results": results, "query": payload.query_text}
    except Exception as e:
        logger.error("Vector search failed: %s", str(e), exc_info=True)
        raise StorageError(f"Vector search failed: {str(e)}")


@app.get("/search/graph", response_model=GraphTraversalResponse)
async def graph_traversal_search(
    start_id: str,
    depth: int = 3,
    max_nodes: int = 100,
    storage: GraphMindStorage = Depends(get_storage),
) -> GraphTraversalResponse:
    """Graph traversal search from a starting node."""
    try:
        if depth < 1 or depth > 10:
            raise ValidationError("Depth must be between 1 and 10")
        if max_nodes < 1 or max_nodes > 1000:
            raise ValidationError("max_nodes must be between 1 and 1000")
        
        logger.info("Graph traversal: start_id=%s, depth=%d, max_nodes=%d", start_id, depth, max_nodes)
        
        raw = storage.graph_traversal(start_id, depth, max_nodes)
        
        nodes = [
            TraversalNode(
                node_id=item["node_id"],
                content=item["content"],
                distance=item["distance"],
                path=item["path"],
                metadata=item.get("metadata", {}),
            )
            for item in raw
        ]
        
        return GraphTraversalResponse(
            start_id=start_id,
            depth=depth,
            nodes=nodes,
        )
    except NotFoundError:
        raise
    except Exception as e:
        logger.error("Graph traversal failed: %s", str(e), exc_info=True)
        raise StorageError(f"Graph traversal failed: {str(e)}")


@app.post("/search/hybrid")
async def hybrid_search(
    payload: HybridSearchRequest,
    storage: GraphMindStorage = Depends(get_storage),
) -> Dict[str, Any]:
    """Hybrid search combining vector and graph scores."""
    try:
        if payload.top_k > settings.MAX_TOP_K:
            raise ValidationError(
                f"top_k exceeds maximum ({settings.MAX_TOP_K})",
                {"requested": payload.top_k, "maximum": settings.MAX_TOP_K}
            )
        
        # Calculate alpha from weights
        total_weight = payload.vector_weight + payload.graph_weight
        if total_weight == 0:
            alpha = 0.5  # Default to equal weights
        else:
            alpha = payload.vector_weight / total_weight
        
        logger.info(
            "Hybrid search: query=%s, vector_weight=%.2f, graph_weight=%.2f, alpha=%.2f, top_k=%d",
            payload.query_text[:50],
            payload.vector_weight,
            payload.graph_weight,
            alpha,
            payload.top_k,
        )
        
        # Get hybrid results
        hybrid_raw = storage.hybrid_search(payload.query_text, payload.top_k, alpha)
        
        # Get individual scores for breakdown
        vector_raw = storage.vector_search(payload.query_text, payload.top_k * 2)
        graph_raw = storage.graph_search(payload.query_text, payload.top_k * 2)
        
        # Create score maps
        vector_scores = {item["node_id"]: item["score"] for item in vector_raw}
        graph_scores = {item["node_id"]: item["score"] for item in graph_raw}
        
        results = [
            HybridSearchResult(
                node_id=item["node_id"],
                content=item["content"],
                score=item["score"],
                vector_score=vector_scores.get(item["node_id"], 0.0),
                graph_score=graph_scores.get(item["node_id"], 0.0),
                metadata=item.get("metadata", {}),
            )
            for item in hybrid_raw
        ]
        
        return {"results": results, "query": payload.query_text, "alpha": alpha}
    except Exception as e:
        logger.error("Hybrid search failed: %s", str(e), exc_info=True)
        raise StorageError(f"Hybrid search failed: {str(e)}")


@app.post("/compare", response_model=ComparisonResponse)
async def compare(
    payload: ComparisonRequest,
    evaluator: SearchEvaluator = Depends(get_evaluator),
) -> ComparisonResponse:
    """Compare all three search methods side-by-side."""
    try:
        logger.info("Comparison: query=%s, top_k=%d", payload.query[:50], payload.top_k)
        result = await evaluator.compare(payload)
        logger.info("Comparison completed: winner=%s", result.winner)
        return result
    except Exception as e:
        logger.error("Comparison failed: %s", str(e), exc_info=True)
        raise StorageError(f"Comparison failed: {str(e)}")


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    file_type: str = Form(...),
    metadata: str = Form(""),
    ingestion: IngestionPipeline = Depends(get_ingestion),
) -> Dict[str, Any]:
    """Upload and ingest a file into the knowledge graph."""
    try:
        # Validate file type
        if file_type not in ["text", "pdf", "xml", "json", "csv"]:
            raise ValidationError(
                f"Invalid file type: {file_type}",
                {"allowed_types": ["text", "pdf", "xml", "json", "csv"]}
            )
        
        # Validate file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        await file.seek(0)  # Reset file pointer
        
        if file_size > settings.MAX_FILE_SIZE:
            raise ValidationError(
                f"File size exceeds maximum ({settings.MAX_FILE_SIZE / 1024 / 1024:.1f}MB)",
                {"file_size": file_size, "maximum": settings.MAX_FILE_SIZE}
            )
        
        if file_size == 0:
            raise ValidationError("File is empty")
        
        logger.info("Ingesting file: %s, type=%s, size=%d bytes", file.filename, file_type, file_size)
        
        meta = parse_metadata(metadata)
        result = await ingestion.ingest_upload(file, file_type, meta)
        
        if result.get("status") == "error":
            raise StorageError(result.get("error", "Ingestion failed"), result)
        
        logger.info("Ingestion completed: %d nodes, %d edges", 
                   result.get("nodes_created", 0), 
                   result.get("edges_created", 0))
        
        return result
    except ValidationError:
        raise
    except Exception as e:
        logger.error("Ingestion failed: %s", str(e), exc_info=True)
        raise StorageError(f"Ingestion failed: {str(e)}")


@app.get("/files")
async def list_files() -> Dict[str, Any]:
    """List all uploaded files."""
    try:
        import os
        from pathlib import Path
        
        upload_dir = settings.UPLOAD_DIR
        files = []
        
        if upload_dir.exists():
            for file_path in upload_dir.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        "filename": file_path.name,
                        "size": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "uploaded_at": stat.st_mtime,
                        "path": str(file_path)
                    })
        
        # Sort by upload time (newest first)
        files.sort(key=lambda x: x["uploaded_at"], reverse=True)
        
        return {
            "files": files,
            "count": len(files),
            "total_size_mb": round(sum(f["size"] for f in files) / (1024 * 1024), 2)
        }
    except Exception as e:
        logger.error("Failed to list files: %s", str(e), exc_info=True)
        raise StorageError(f"Failed to list files: {str(e)}")


@app.delete("/files/{filename}")
async def delete_file(filename: str) -> Dict[str, Any]:
    """Delete an uploaded file."""
    try:
        from pathlib import Path
        
        file_path = settings.UPLOAD_DIR / filename
        
        if not file_path.exists():
            raise NotFoundError(f"File {filename} not found")
        
        if not file_path.is_file():
            raise ValidationError(f"{filename} is not a file")
        
        # Security check: ensure file is in upload directory
        if not str(file_path).startswith(str(settings.UPLOAD_DIR)):
            raise ValidationError("Invalid file path")
        
        file_path.unlink()
        
        logger.info("File deleted: %s", filename)
        
        return {
            "status": "deleted",
            "filename": filename
        }
    except NotFoundError:
        raise
    except Exception as e:
        logger.error("Failed to delete file: %s", str(e), exc_info=True)
        raise StorageError(f"Failed to delete file: {str(e)}")


@app.delete("/data/clear")
async def clear_all_data(storage: GraphMindStorage = Depends(get_storage)) -> Dict[str, Any]:
    """Clear all data from the system (nodes, edges, vectors, files)."""
    try:
        import shutil
        from pathlib import Path
        
        # Clear graph
        node_count = storage.graph.number_of_nodes()
        edge_count = storage.graph.number_of_edges()
        storage.graph.clear()
        
        # Clear ChromaDB collection
        vector_count = storage.collection.count()
        try:
            # Delete all items in collection
            all_ids = storage.collection.get()["ids"]
            if all_ids:
                storage.collection.delete(ids=all_ids)
        except Exception as e:
            logger.warning("Error clearing ChromaDB collection: %s", e)
            # Try to delete and recreate collection
            try:
                storage.chroma_client.delete_collection(name="graphmind_nodes")
            except:
                pass
            storage.collection = storage.chroma_client.get_or_create_collection(
                name="graphmind_nodes",
                metadata={"hnsw:space": "cosine"},
            )
        
        # Clear vector mapping
        storage.vector_to_graph_map.clear()
        
        # Clear uploaded files
        upload_dir = settings.UPLOAD_DIR
        file_count = 0
        if upload_dir.exists():
            for file_path in upload_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    file_count += 1
        
        logger.info("Cleared all data: %d nodes, %d edges, %d vectors, %d files", 
                   node_count, edge_count, vector_count, file_count)
        
        return {
            "status": "cleared",
            "nodes_deleted": node_count,
            "edges_deleted": edge_count,
            "vectors_deleted": vector_count,
            "files_deleted": file_count
        }
    except Exception as e:
        logger.error("Failed to clear data: %s", str(e), exc_info=True)
        raise StorageError(f"Failed to clear data: {str(e)}")

