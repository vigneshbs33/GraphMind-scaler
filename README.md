# 🧠 GraphMind - Hybrid Vector + Graph Database

**A hackathon project for efficient AI retrieval combining semantic similarity (vectors) with relational knowledge (graphs)**

## 🎯 Project Overview

GraphMind is a hybrid database system that combines:
- **Vector Search** (ChromaDB) - Semantic similarity using embeddings
- **Graph Search** (NetworkX) - Relational traversal and connections
- **Hybrid Search** - Intelligent combination of both approaches

Built for the Devfolio Problem Statement: "Vector + Graph Native Database for Efficient AI Retrieval"

## ✨ Key Features

- ✅ **Dual Storage**: ChromaDB (vectors) + NetworkX (graph)
- ✅ **Full CRUD Operations**: Create, Read, Update, Delete nodes and edges
- ✅ **3 Search Modes**: Vector-only, Graph-only, Hybrid
- ✅ **Separate Search Endpoints**: `/search/vector`, `/search/graph`, `/search/hybrid`
- ✅ **Graph Traversal**: BFS traversal with depth control
- ✅ **File Ingestion**: Upload and parse text, PDF, XML, JSON, CSV files
- ✅ **Entity Extraction**: Automatic entity detection and graph building
- ✅ **Semantic Edges**: Automatic similarity-based connections
- ✅ **PageRank Centrality**: Identify important nodes
- ✅ **LLM Integration**: Gemini API for natural language answers
- ✅ **Web Interface**: Complete UI with graph visualization
- ✅ **Comparison Tool**: Side-by-side evaluation of search methods
- ✅ **Multi-hop Reasoning**: Find connections through multiple graph hops
- ✅ **Relationship-weighted Search**: Weight graph scores by relationship type
- ✅ **Pagination & Filtering**: Graph endpoint supports pagination and node type filtering

## 🚀 Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Server

```bash
uvicorn backend.main:app --reload
```

### 3. Access Application

- **Web UI**: http://127.0.0.1:8000/
- **API Docs**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

## 📋 Usage

### Upload Files

1. Go to the **Upload** tab
2. Drag & drop or select files (txt, pdf, xml, json, csv)
3. Select file type
4. Files are automatically parsed, chunked, and stored

### Search

1. Go to the **Search** tab
2. Enter your query
3. Select search mode:
   - **Hybrid** (Recommended) - Combines vector + graph
   - **Vector Only** - Semantic similarity
   - **Graph Only** - Relational traversal
4. View results with AI-generated answer

### Compare Methods

1. Go to the **Compare** tab
2. Enter a query
3. See side-by-side comparison of all three methods
4. View precision metrics and winner
5. Get LLM-generated summary

### Visualize Graph

1. Go to the **Graph** tab
2. See interactive visualization of your knowledge graph
3. Nodes represent concepts/entities
4. Edges show relationships

### Manage Nodes

1. Go to the **Nodes** tab
2. Enter a node ID to view details
3. See all relationships (incoming and outgoing)
4. Edit node content or metadata
5. Delete node (removes all connected edges)

### Graph Traversal

1. Go to the **Traversal** tab
2. Enter a start node ID
3. Adjust depth slider (1-10)
4. Set max nodes to return
5. View traversal results with paths
6. See visualization of traversal paths

## 🔧 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check with system status |
| `/stats` | GET | Database statistics |

### Node CRUD

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/nodes` | POST | Create a node with text, metadata, and embedding |
| `/nodes/{id}` | GET | Get node by ID with all relationships |
| `/nodes/{id}` | PUT | Update node content and/or metadata |
| `/nodes/{id}` | DELETE | Delete node and all associated edges |

### Edge CRUD

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/edges` | POST | Create a relationship: {source, target, type, weight} |
| `/edges/{edge_id}` | GET | Get edge details (or use ?source_id=X&target_id=Y) |

### Search Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | Unified search (vector/graph/hybrid mode) |
| `/search/vector` | POST | Vector-only search: {query_text, top_k} |
| `/search/graph` | GET | Graph traversal: ?start_id=X&depth=3&max_nodes=100 |
| `/search/hybrid` | POST | Hybrid search: {query_text, vector_weight, graph_weight, top_k} |
| `/search/multi-hop` | POST | Multi-hop reasoning query |
| `/compare` | POST | Compare all search methods side-by-side |

### Graph & Ingestion

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/graph` | GET | Get full graph snapshot (with pagination & filtering) |
| `/ingest` | POST | Upload and process file |

## 🏗️ Architecture

```
Input File
    ↓
Parser (MultiFormatParser)
    ↓
Chunking + Entity Extraction
    ↓
Parallel Processing
    ├→ ChromaDB (Vector Embeddings)
    └→ NetworkX (Graph Structure)
    ↓
Merge Algorithm
    ├→ PageRank (Centrality)
    └→ Semantic Edges (Similarity > 0.7)
    ↓
Query
    ├→ Vector Search (Cosine Similarity)
    ├→ Graph Search (BFS Traversal)
    └→ Hybrid Search (Weighted Combination)
    ↓
LLM Processing (Gemini)
    ↓
Structured Answer
```

## 🧮 Hybrid Scoring Algorithm

```python
# Normalize scores to [0, 1]
vector_score = normalize(cosine_similarity(query, node))
graph_score = normalize(1 / (distance + 1))

# Weighted combination
final_score = α * vector_score + (1 - α) * graph_score

# Default: α = 0.6 (60% vector, 40% graph)
```

## 📦 Tech Stack

- **Backend**: FastAPI (async, auto-docs)
- **Vector DB**: ChromaDB (embedded, persistent)
- **Graph DB**: NetworkX (pure Python)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini API
- **Frontend**: HTML + Cytoscape.js
- **File Parsing**: PyPDF2, xmltodict, csv

## 🎯 Demo Use Case

**Personal Knowledge Graph for AI Research**

- Nodes: Concepts (ML, NLP), Papers, Techniques, Applications
- Edges: Relationships, citations, applications
- Query: "Healthcare AI"
- Result: Found through both semantic similarity AND graph connections to "Medical Imaging"

## 📊 Stretch Goals Implemented

- ✅ **Multi-hop Reasoning**: `/search/multi-hop` endpoint finds connections through multiple graph hops
- ✅ **Relationship-weighted Search**: Enhanced hybrid search with relationship type weighting
- ✅ **Pagination & Filtering**: Graph endpoint supports pagination (`page`, `limit`) and filtering (`node_type`)
- ✅ **Graph Traversal**: Depth-controlled BFS traversal with path visualization

## 🎯 Devfolio Requirements Compliance

### Required Features ✅

- ✅ Vector storage with cosine similarity search
- ✅ Graph storage with nodes, edges, and metadata
- ✅ Hybrid retrieval merging vector similarity + graph adjacency
- ✅ API endpoints for CRUD operations
- ✅ Vector search endpoint (`POST /search/vector`)
- ✅ Graph traversal endpoint (`GET /search/graph`)
- ✅ Hybrid search endpoint (`POST /search/hybrid`)
- ✅ Simple scoring/ranking mechanism for hybrid results
- ✅ Embeddings pipeline (SentenceTransformers)
- ✅ Local persistence (ChromaDB + NetworkX)

### Stretch Goals ✅

- ✅ Multi-hop reasoning query
- ✅ Relationship-weighted search
- ✅ Pagination and filtering

## 📊 Evaluation Criteria

✅ **Working CRUD + Search** (20/50 pts)
✅ **Hybrid Logic Clarity** (10/50 pts)
✅ **Real-world Demo** (30/100 pts)
✅ **Hybrid Effectiveness Proof** (25/100 pts)

## 🔍 Verification

Run the verification script:

```bash
python verify_project.py
```

This checks:
- All imports
- Configuration
- Storage operations
- File parsing
- Ingestion pipeline
- LLM processor
- API endpoints

## 🛠️ Configuration

Edit `backend/config.py` or create `.env`:

```python
LLM_PROVIDER = "gemini"  # or "mock"
GEMINI_API_KEY = "your-key-here"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HYBRID_ALPHA = 0.6  # Vector weight
```

## 📝 Project Structure

```
graphmind/
├── backend/
│   ├── config.py          # Settings
│   ├── models.py          # Pydantic models
│   ├── storage.py         # ChromaDB + NetworkX
│   ├── parsers.py         # File parsing
│   ├── ingestion.py       # Parallel pipeline
│   ├── llm_processor.py   # Gemini integration
│   ├── evaluation.py      # Comparison logic
│   └── main.py            # FastAPI app
├── frontend/
│   └── index.html         # Web UI
├── data/
│   ├── uploads/           # Uploaded files
│   └── chroma/            # ChromaDB storage
├── requirements.txt
└── README.md
```

## 🚨 Troubleshooting

### Server won't start
- Check port 8000 is available
- Verify virtual environment is activated
- Run `pip install -r requirements.txt`

### Import errors
- Ensure you're in project root
- Activate virtual environment
- Check Python version (3.8+)

### LLM not working
- Verify Gemini API key in `config.py`
- Check internet connection
- Fallback to mock LLM if needed

### Graph not showing
- Upload some files first
- Check browser console for errors
- Verify Cytoscape.js is loading

## 🎉 Features Completed

- [x] Dual storage (ChromaDB + NetworkX)
- [x] Vector search
- [x] Graph traversal search
- [x] Hybrid search algorithm
- [x] Parallel file ingestion
- [x] Entity extraction
- [x] Semantic similarity edges
- [x] PageRank centrality
- [x] Gemini LLM integration
- [x] Web interface
- [x] Graph visualization
- [x] Comparison tool
- [x] CRUD APIs

## 📄 License

MIT License - Hackathon Project

## 🙏 Acknowledgments

- ChromaDB for vector storage
- NetworkX for graph operations
- FastAPI for the web framework
- Google Gemini for LLM capabilities
- Cytoscape.js for graph visualization

---

**Built for Devfolio Hackathon - 12 Hour Challenge**

