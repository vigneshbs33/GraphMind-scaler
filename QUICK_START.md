# GraphMind - Quick Start Guide

## ✅ Project Status: FULLY FUNCTIONAL

All core components are working! The API is running and ready to use.

## 🚀 How to Use

### 1. Start the Server

```bash
uvicorn backend.main:app --reload
```

The server will start on `http://127.0.0.1:8000`

### 2. Access the API

- **Interactive Documentation**: http://127.0.0.1:8000/docs
- **Alternative Docs**: http://127.0.0.1:8000/redoc
- **Root URL**: http://127.0.0.1:8000/ (redirects to docs)

### 3. Test Endpoints

#### Health Check
```bash
curl http://127.0.0.1:8000/health
```

#### Get Statistics
```bash
curl http://127.0.0.1:8000/stats
```

#### Search
```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "mode": "hybrid", "top_k": 5}'
```

#### Create Node
```bash
curl -X POST http://127.0.0.1:8000/nodes \
  -H "Content-Type: application/json" \
  -d '{"content": "Machine Learning is transforming healthcare", "metadata": {"source": "manual"}}'
```

#### Upload File
```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "file=@your_file.txt" \
  -F "file_type=text" \
  -F "metadata={}"
```

## 📋 Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirects to `/docs` |
| `/health` | GET | Health check |
| `/stats` | GET | Get storage statistics |
| `/nodes` | POST | Create a new node |
| `/edges` | POST | Create an edge between nodes |
| `/graph` | GET | Get full graph snapshot |
| `/search` | POST | Search (vector/graph/hybrid) |
| `/compare` | POST | Compare search methods |
| `/ingest` | POST | Upload and ingest a file |

## 🔍 Verification

Run the verification script to check all components:

```bash
python verify_project.py
```

## 📝 Next Steps

1. **Frontend UI**: Complete `frontend/index.html` with Cytoscape.js visualization
2. **Demo Data**: Add sample files to `backend/demo_data/`
3. **Documentation**: Complete `README.md` with full project documentation

## 🐛 Troubleshooting

### 404 on Root Path
- **Fixed!** Root path now redirects to `/docs`
- Use `/docs` for interactive API documentation

### Server Not Starting
- Make sure virtual environment is activated
- Check that port 8000 is not in use
- Verify all dependencies are installed: `pip install -r requirements.txt`

### Import Errors
- Ensure you're in the project root directory
- Activate virtual environment: `venv\Scripts\activate` (Windows)

## ✨ Features Working

- ✅ Dual storage (ChromaDB + NetworkX)
- ✅ Vector search
- ✅ Graph traversal search
- ✅ Hybrid search
- ✅ Parallel file ingestion
- ✅ Entity extraction
- ✅ Semantic similarity edges
- ✅ PageRank centrality
- ✅ LLM query understanding (mock)
- ✅ Search evaluation and comparison

