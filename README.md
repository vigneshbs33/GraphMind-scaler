# 🧠 GraphMind - Hybrid Vector + Graph Database

**A hackathon project for efficient AI retrieval combining semantic similarity (vectors) with relational knowledge (graphs)**

## 🎯 Project Overview

GraphMind is a hybrid database system that combines:
- **Vector Search** (ChromaDB) - Semantic similarity using embeddings
- **Graph Search** (NetworkX) - Relational traversal and connections
- **Hybrid Search** - Intelligent combination of both approaches

Built for the Devfolio Problem Statement: "Vector + Graph Native Database for Efficient AI Retrieval"

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB RAM minimum (for embedding model)

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd GraphMind-scaler
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements-full.txt

# Or if requirements-full.txt doesn't exist, use:
pip install -r requirements.txt
```

**Note:** The `requirements.txt` file contains minimal dependencies for Streamlit Cloud deployment. For local development with full features, use `requirements-full.txt` which includes all backend dependencies.

### Step 4: Start Backend Server

```bash
uvicorn backend.main:app --reload
```

The API will be available at: `http://localhost:8000`
API documentation at: `http://localhost:8000/docs`

### Step 5: Run Streamlit UI (USP)

**Option 1: Main Streamlit App**
```bash
streamlit run streamlit_app.py
```

**Option 2: USP Interface**
```bash
streamlit run USP1_vector_graph_search/parse_zip/streamlit_ui.py
```

**Note:** The USP interface requires the backend API to be running on port 8000.

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

## 📋 Usage

### Upload Files

1. Start the backend server: `uvicorn backend.main:app --reload`
2. Go to the **Upload** tab in the UI
3. Drag & drop or select files (txt, pdf, xml, json, csv)
4. Select file type
5. Files are automatically parsed, chunked, and stored

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

## 🔧 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check with system status |
| `/stats` | GET | Database statistics |

### Search Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search/vector` | POST | Vector-only search: {query_text, top_k} |
| `/search/graph` | GET | Graph traversal: ?start_id=X&depth=3&max_nodes=100 |
| `/search/hybrid` | POST | Hybrid search: {query_text, vector_weight, graph_weight, top_k} |

### Node & Edge CRUD

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/nodes` | POST | Create a node with text, metadata, and embedding |
| `/nodes/{id}` | GET | Get node by ID with all relationships |
| `/nodes/{id}` | PUT | Update node content and/or metadata |
| `/nodes/{id}` | DELETE | Delete node and all associated edges |
| `/edges` | POST | Create a relationship: {source, target, type, weight} |

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
Query
    ├→ Vector Search (Cosine Similarity)
    ├→ Graph Search (BFS Traversal)
    └→ Hybrid Search (Weighted Combination)
    ↓
LLM Processing (Gemini)
    ↓
Structured Answer
```

## 📦 Tech Stack

- **Backend**: FastAPI (async, auto-docs)
- **Vector DB**: ChromaDB (embedded, persistent)
- **Graph DB**: NetworkX (pure Python)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini API
- **Frontend**: HTML + Cytoscape.js + Streamlit
- **File Parsing**: PyPDF2, xmltodict, csv

## 🚨 Troubleshooting

### Server won't start
- Check port 8000 is available
- Verify virtual environment is activated
- Run `pip install -r requirements.txt`

### Import errors
- Ensure you're in project root
- Activate virtual environment
- Check Python version (3.9+)

### LLM not working
- Verify Gemini API key in `backend/config.py`
- Check internet connection
- Fallback to mock LLM if needed

### USP Interface not connecting
- Make sure backend is running: `uvicorn backend.main:app --reload`
- Check API is accessible at `http://localhost:8000`
- Verify API health: `curl http://localhost:8000/health`

## 📝 Project Structure

```
GraphMind-scaler/
├── backend/
│   ├── config.py          # Settings
│   ├── models.py          # Pydantic models
│   ├── storage.py         # ChromaDB + NetworkX
│   ├── parsers.py         # File parsing
│   ├── ingestion.py       # Parallel pipeline
│   ├── llm_processor.py   # Gemini integration
│   └── main.py            # FastAPI app
├── frontend/
│   └── index.html         # Web UI
├── USP1_vector_graph_search/
│   └── parse_zip/
│       └── streamlit_ui.py # USP Streamlit interface
├── streamlit_app.py       # Main Streamlit app
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## 📄 License

MIT License - Hackathon Project

---

**Built for Devfolio Hackathon - 12 Hour Challenge**
