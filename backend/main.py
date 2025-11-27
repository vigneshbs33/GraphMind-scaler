"""FastAPI application for GraphMind."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .config import settings
from .evaluation import SearchEvaluator
from .ingestion import IngestionPipeline, parse_metadata
from .llm_processor import get_llm_processor
from .models import (
    ComparisonRequest,
    ComparisonResponse,
    EdgeCreate,
    NodeCreate,
    SearchQuery,
    SearchResult,
)
from .storage import GraphMindStorage

app = FastAPI(title="GraphMind API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.on_event("startup")
async def _startup() -> None:
    storage = GraphMindStorage(settings.CHROMA_DIR, settings.EMBEDDING_MODEL)
    llm = get_llm_processor(settings.LLM_PROVIDER, settings.CLAUDE_API_KEY)
    app.state.storage = storage
    app.state.ingestion = IngestionPipeline(storage)
    app.state.evaluator = SearchEvaluator(storage, llm)
    app.state.llm = llm


def get_storage() -> GraphMindStorage:
    return app.state.storage


def get_ingestion() -> IngestionPipeline:
    return app.state.ingestion


def get_evaluator() -> SearchEvaluator:
    return app.state.evaluator


@app.get("/")
async def root():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
async def stats(storage: GraphMindStorage = Depends(get_storage)) -> Dict[str, float]:
    return storage.get_stats()


@app.post("/nodes", response_model=SearchResult)
async def create_node(payload: NodeCreate, storage: GraphMindStorage = Depends(get_storage)) -> SearchResult:
    node_id = storage.add_node(payload.id, payload.content, payload.metadata)
    return SearchResult(
        node_id=node_id,
        content=payload.content,
        score=1.0,
        metadata=payload.metadata,
        reasoning=payload.node_type or "",
    )


@app.post("/edges")
async def create_edge(payload: EdgeCreate, storage: GraphMindStorage = Depends(get_storage)) -> Dict[str, str]:
    storage.add_edge(payload.source_id, payload.target_id, payload.relationship, payload.weight)
    return {"status": "ok"}


@app.get("/graph")
async def graph_snapshot(storage: GraphMindStorage = Depends(get_storage)) -> Dict[str, Any]:
    nodes = [{"node_id": node_id, **attrs} for node_id, attrs in storage.graph.nodes(data=True)]
    edges = [{"source": src, "target": dst, **attrs} for src, dst, attrs in storage.graph.edges(data=True)]
    return {"nodes": nodes, "edges": edges}


@app.post("/search")
async def search(
    payload: SearchQuery,
    storage: GraphMindStorage = Depends(get_storage),
) -> Dict[str, Any]:
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
    return {"results": results}


@app.post("/compare", response_model=ComparisonResponse)
async def compare(
    payload: ComparisonRequest,
    evaluator: SearchEvaluator = Depends(get_evaluator),
) -> ComparisonResponse:
    return await evaluator.compare(payload)


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    file_type: str = Form(...),
    metadata: str = Form(""),
    ingestion: IngestionPipeline = Depends(get_ingestion),
) -> Dict[str, Any]:
    meta = parse_metadata(metadata)
    return await ingestion.ingest_upload(file, file_type, meta)

