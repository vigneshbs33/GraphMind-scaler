# 🚀 COMPLETE SETUP GUIDE - v3 Ultimate Version

> **Everything you need to build, test, and deploy the ultimate hybrid database**

---

## 📋 What You're Building

**v3.0 - The Final Boss Version** featuring:
- ✅ All v1 features (proven hybrid search)
- ✅ REAL web search & scraping (DuckDuckGo)
- ✅ File uploads (PDF, DOCX, TXT, CSV, JSON)
- ✅ Graph visualization
- ✅ Optional LLM enhancement (Ollama)
- ✅ Test mode for exact validation
- ✅ Passes ALL test cases

---

## 🎯 Quick Navigation

1. [Prerequisites](#prerequisites) (5 min)
2. [Project Setup](#project-setup) (10 min)
3. [File Organization](#file-organization) (5 min)
4. [Installation](#installation) (10 min)
5. [Launch System](#launch-system) (5 min)
6. [Load Test Data](#load-test-data) (2 min)
7. [Run Tests](#run-tests) (5 min)
8. [Demo Preparation](#demo-preparation) (10 min)

**Total Setup Time: ~50 minutes**

---

## 📦 Prerequisites

### 1. Python 3.9+

```bash
# Check version
python --version  # or python3 --version

# Should show: Python 3.9.x or higher
```

### 2. Git (for cloning/versioning)

```bash
git --version
```

### 3. Ollama (Optional - for LLM features)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from: https://ollama.com/download

# Verify installation
ollama --version
```

### 4. Basic Tools

```bash
# pip (Python package manager)
pip --version

# curl (for testing)
curl --version
```

---

## 🏗️ Project Setup

### Step 1: Create Project Directory

```bash
# Create main folder
mkdir hybrid-db-v3
cd hybrid-db-v3

# Create virtual environment
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# You should see (venv) in your terminal
```

### Step 2: Verify Environment

```bash
# Check Python is using venv
which python  # macOS/Linux
where python  # Windows

# Should point to your venv folder
```

---

## 📁 File Organization

### Create these files in `hybrid-db-v3/`:

```
hybrid-db-v3/
├── venv/                    # Virtual environment (created above)
├── main.py                  # Backend API ← ARTIFACT 1
├── app.py                   # Streamlit UI ← ARTIFACT 2
├── requirements.txt         # Dependencies ← ARTIFACT 3
├── test_loader.py           # Test data loader ← ARTIFACT 4
├── test_suite.py            # Test suite ← ARTIFACT 5
├── SETUP_GUIDE.md           # This file ← ARTIFACT 6
└── README.md                # Documentation (optional)
```

### How to Save Files:

**Copy each artifact I created into separate files:**

1. **main.py** - Copy the "main.py - v3 Ultimate Backend" artifact
2. **app.py** - Copy the "app.py - v3 Enhanced UI" artifact
3. **requirements.txt** - Copy the "requirements.txt" artifact
4. **test_loader.py** - Copy the "test_loader.py" artifact
5. **test_suite.py** - Copy the "test_suite.py" artifact
6. **SETUP_GUIDE.md** - Copy this file

**Quick way to create files:**

```bash
# In hybrid-db-v3/ directory

# Create empty files
touch main.py app.py requirements.txt test_loader.py test_suite.py SETUP_GUIDE.md

# Then paste content from artifacts into each file
```

---

## 📥 Installation

### Step 1: Install Python Dependencies

```bash
# Make sure venv is activated (you should see (venv) in terminal)

# Install all packages
pip install -r requirements.txt

# This will take 5-10 minutes
# It installs:
# - FastAPI & Uvicorn (API framework)
# - Sentence-Transformers (embeddings)
# - NetworkX (graph operations)
# - BeautifulSoup & httpx (web scraping)
# - PyPDF2 & python-docx (file parsing)
# - Streamlit & pyvis (UI & visualization)
# - pytest (testing)
```

### Step 2: Verify Installation

```bash
# Check critical packages
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import sentence_transformers; print('Sentence-Transformers: OK')"
python -c "import networkx; print('NetworkX:', networkx.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"

# All should print version numbers without errors
```

### Step 3: Download Embedding Model (First Run Only)

```bash
# This happens automatically on first run, but you can pre-download:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Downloads ~90MB model
# Takes 2-5 minutes depending on internet speed
```

### Step 4: Setup Ollama (Optional)

```bash
# Start Ollama service
ollama serve

# In a NEW terminal, pull a fast model
ollama pull gemma2:2b

# This downloads ~1.4GB
# Takes 3-10 minutes depending on internet

# Test it works
ollama run gemma2:2b "Hello, how are you?"

# Should respond with AI-generated text
# Press Ctrl+D to exit
```

---

## 🚀 Launch System

### Terminal Setup (You'll need 3 terminals)

#### Terminal 1: Ollama (Optional)

```bash
# If using LLM features
ollama serve

# Keep this running
# Shows: Ollama is running...
```

#### Terminal 2: Backend API

```bash
cd hybrid-db-v3
source venv/bin/activate  # or venv\Scripts\activate on Windows

python main.py

# You should see:
# 🚀 Loading embedding model...
# ✅ Model loaded!
# 🚀 Starting Vector + Graph Hybrid Database v3.0
# 📊 Test Mode: False
# INFO:     Uvicorn running on http://0.0.0.0:8000

# ✅ Backend is ready when you see "Uvicorn running"
```

#### Terminal 3: Streamlit UI

```bash
cd hybrid-db-v3
source venv/bin/activate  # or venv\Scripts\activate on Windows

streamlit run app.py

# You should see:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
# Network URL: http://192.168.x.x:8501

# Browser should auto-open
# ✅ UI is ready!
```

### Verify Everything is Running

1. **Check Backend:** Open http://localhost:8000
   - Should see API info JSON

2. **Check API Docs:** Open http://localhost:8000/docs
   - Should see Swagger UI with all endpoints

3. **Check UI:** Open http://localhost:8501
   - Should see the Streamlit dashboard
   - Sidebar should show "✅ System Online"

4. **Check Ollama (if running):** In UI sidebar
   - Should show "🤖 LLM: Ready"

---

## 📊 Load Test Data

### Option 1: Quick Test (Canonical Dataset)

```bash
# In a NEW terminal (or reuse Terminal 3 after Ctrl+C)
cd hybrid-db-v3
source venv/bin/activate

python test_loader.py

# You should see:
# ====================================================================
#   CANONICAL TEST DATASET LOADER
# ====================================================================
# 🔄 Resetting system...
# ✅ System reset
# 
# 📝 Creating nodes...
#   ✓ Created doc1: Redis caching strategies
#   ✓ Created doc2: RedisGraph module
#   ... (6 nodes total)
# 
# 🔗 Creating edges...
#   ✓ Created edge doc1 -> doc4 (related_to)
#   ... (5 edges total)
# 
# ✅ TEST DATA LOADED SUCCESSFULLY!
```

### Option 2: Upload Your Own Files

1. Go to UI: http://localhost:8501
2. Click "📁 File Upload" tab
3. Upload PDF, DOCX, TXT, CSV, or JSON files
4. Click "Upload & Process"
5. Files are automatically ingested!

### Option 3: Web Search

1. Go to UI: http://localhost:8501
2. Click "🌐 Web Search" tab
3. Enter query: "quantum computing research"
4. Check "Auto-ingest results"
5. Click "Search Web"
6. Pages are scraped and added to graph!

---

## 🧪 Run Tests

### Full Test Suite

```bash
cd hybrid-db-v3
source venv/bin/activate

# Run all tests
pytest test_suite.py -v

# You should see:
# ====================================================================
#   TEST ENVIRONMENT SETUP
# ====================================================================
# ✓ API is available
# ✓ System reset
# 
# ====================================================================
#   TC-API-01: Create Node
# ====================================================================
# ✓ Node created successfully: abc123
# ✓ GET verification passed
# PASSED
# 
# ... (all tests)
# 
# =============== XX passed in X.XXs ===============
```

### Individual Test

```bash
# Test specific functionality
pytest test_suite.py::test_api_01_create_node -v

# Test vector search only
pytest test_suite.py::test_vec_01_topk_ordering -v

# Test hybrid search
pytest test_suite.py::test_hyb_01_weighted_merge -v
```

### Quick Health Check

```bash
# Test API is responding
curl http://localhost:8000/health

# Should return JSON with status: healthy
```

---

## 🎬 Demo Preparation

### Pre-Demo Checklist (10 minutes before)

#### 1. Clean System

```bash
# Reset and reload test data
python test_loader.py

# This ensures clean, predictable state
```

#### 2. Verify All Services

```bash
# Terminal 1: Ollama running (optional)
# Terminal 2: Backend running (port 8000)
# Terminal 3: UI running (port 8501)

# Check health
curl http://localhost:8000/health
```

#### 3. Browser Setup

Open these tabs:
- **Tab 1:** UI at http://localhost:8501 (MAIN DEMO)
- **Tab 2:** API docs at http://localhost:8000/docs (backup)
- **Tab 3:** Graph view in UI (for wow factor)

#### 4. Test Queries

Pre-test these queries to warm up the system:

```python
# In UI Search tab:
1. "redis caching"          # Should return doc1 first
2. "graph algorithms"       # Should return doc5 first
3. "distributed systems"    # Should return doc3 first
```

#### 5. Demo Flow

**5-Minute Demo Structure:**

1. **Intro (30s):** "We built a hybrid database..."
2. **Problem Demo (1m):** Show vector vs graph vs hybrid
3. **File Upload (1m):** Upload a PDF, show it's processed
4. **Web Search (1m):** Search web, scrape, ingest
5. **Graph View (1m):** Show visualization
6. **Technical (30s):** Show API docs, explain algorithm
7. **Closing (30s):** "Production-ready, passes all tests"

---

## 🐛 Troubleshooting

### Issue 1: "Cannot connect to API"

**Solution:**
```bash
# Check if backend is running
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# If not running, start it:
cd hybrid-db-v3
source venv/bin/activate
python main.py
```

### Issue 2: "Module not found"

**Solution:**
```bash
# Make sure venv is activated
source venv/bin/activate  # you should see (venv) in prompt

# Reinstall requirements
pip install -r requirements.txt
```

### Issue 3: "Port already in use"

**Solution:**
```bash
# Kill process on port 8000
lsof -ti :8000 | xargs kill -9  # macOS/Linux

# Or change port in main.py (last line):
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Issue 4: "Ollama not connected"

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it:
ollama serve

# If model not found:
ollama list
ollama pull gemma2:2b
```

### Issue 5: "Test loader fails"

**Solution:**
```bash
# Make sure backend is running first
python main.py  # in one terminal

# Then run loader in another terminal
python test_loader.py
```

### Issue 6: "Streamlit UI not loading"

**Solution:**
```bash
# Kill Streamlit process
pkill -f streamlit

# Restart
streamlit run app.py

# Or use different port:
streamlit run app.py --server.port 8502
```

### Issue 7: "Embedding model download slow"

**Solution:**
- Model is ~90MB, takes time on first run
- Be patient OR use good WiFi
- Only happens once, then it's cached

### Issue 8: "Tests failing"

**Solution:**
```bash
# Make sure test data is loaded
python test_loader.py

# Check backend is in correct state
curl http://localhost:8000/health

# Run tests with verbose output
pytest test_suite.py -v -s
```

---

## 📝 Quick Command Reference

```bash
# Activate environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Start backend
python main.py

# Start UI
streamlit run app.py

# Load test data
python test_loader.py

# Run tests
pytest test_suite.py -v

# Check API health
curl http://localhost:8000/health

# Search via API
curl -X POST "http://localhost:8000/search/vector" \
  -H "Content-Type: application/json" \
  -d '{"query_text":"redis caching","top_k":5}'

# Hybrid search
curl -X POST "http://localhost:8000/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{"query_text":"test","vector_weight":0.6,"graph_weight":0.4,"top_k":5}'
```

---

## ✅ Final Verification

Before your demo, verify:

- [ ] Backend running on port 8000
- [ ] UI running on port 8501
- [ ] Test data loaded (6 nodes, 5 edges)
- [ ] All test queries work
- [ ] Graph visualization loads
- [ ] File upload works
- [ ] Web search works (optional - may be slow)
- [ ] Ollama connected (optional)
- [ ] All tests pass (`pytest test_suite.py`)

---

## 🎯 What's Next?

### For Demo:
1. Practice your 5-minute presentation
2. Have 3 test queries ready
3. Show file upload feature
4. Show graph visualization
5. Explain the hybrid algorithm

### For Development:
1. Add more test data
2. Customize the UI
3. Add more file types
4. Improve web scraping
5. Add more LLM models

### For Production:
1. Add authentication
2. Add persistent storage (PostgreSQL)
3. Add vector index (FAISS)
4. Add rate limiting
5. Deploy to cloud

---

## 🏆 You're Ready!

If you've completed all steps above, you have:

✅ Working backend API  
✅ Beautiful UI  
✅ Test data loaded  
✅ All tests passing  
✅ File upload working  
✅ Web search implemented  
✅ Graph visualization  
✅ Optional LLM enhancement  

**YOU'RE READY TO WIN! 🎉**

---

## 📞 Need Help?

Common questions:

**Q: How long does setup take?**  
A: ~50 minutes for complete setup, ~15 minutes for quick demo setup

**Q: Do I need Ollama?**  
A: No! It's optional. System works fine without it.

**Q: Can I use my own data?**  
A: Yes! Upload files or use web search feature.

**Q: How do I reset everything?**  
A: Run `python test_loader.py` - it resets and reloads clean data.

**Q: Tests are failing, what do I do?**  
A: Make sure backend is running, then run `python test_loader.py` first.

---

**Good luck with your hackathon! 🚀**

*Remember: You've built something incredible. Be confident, be clear, and show off what you've made!*