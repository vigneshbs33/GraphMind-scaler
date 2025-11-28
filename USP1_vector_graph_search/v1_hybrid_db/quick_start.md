# ⚡ Quick Start Guide - 10 Minutes to Demo-Ready

> Get your Vector + Graph Hybrid Database running in 10 minutes or less!

---

## 📋 Prerequisites Check

```bash
# Check Python version (need 3.9+)
python --version

# Check pip
pip --version
```

If you don't have Python 3.9+, install it from [python.org](https://python.org)

---

## 🚀 Step-by-Step Setup

### Step 1: Project Setup (2 minutes)

```bash
# Create project directory
mkdir hybrid-db
cd hybrid-db

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Your terminal should now show (venv) prefix
```

### Step 2: Install Dependencies (3 minutes)

Create `requirements.txt`:
```text
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
sentence-transformers==2.2.2
numpy==1.24.3
networkx==3.2.1
python-multipart==0.0.6
torch==2.1.0
streamlit==1.28.0
requests==2.31.0
```

Install everything:
```bash
pip install -r requirements.txt

# This will take 2-3 minutes
# The sentence-transformers model will download on first run
```

### Step 3: Copy Code Files (1 minute)

You should now have these files in your directory:
```
hybrid-db/
├── main.py                     # Backend API
├── app.py                      # Streamlit UI
├── populate_demo_data.py       # Demo dataset
├── test_api.py                 # Testing script
├── requirements.txt            # Dependencies
├── README.md                   # Full documentation
├── DEMO_SCRIPT.md             # Presentation guide
└── QUICKSTART.md              # This file
```

### Step 4: Start the Backend (2 minutes)

```bash
# Start the FastAPI server
python main.py
```

You should see:
```
Loading embedding model...
Model loaded successfully!
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ **Backend is running!** Keep this terminal open.

Test it: Open browser to `http://localhost:8000` - you should see API info.

### Step 5: Load Demo Data (1 minute)

Open a **NEW terminal** (keep the first one running!):

```bash
# Navigate to your project folder
cd hybrid-db

# Activate virtual environment again
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Load demo data
python populate_demo_data.py
```

You should see:
```
🚀 Starting demo data population...
✅ Database reset
📚 Creating Institutions...
   ✓ Created 3 institutions
👤 Creating Authors...
   ✓ Created 5 authors
...
✅ DEMO DATA POPULATION COMPLETE!
```

✅ **Demo data loaded!**

### Step 6: Start the UI (1 minute)

In the same second terminal:

```bash
streamlit run app.py
```

Your browser should automatically open to `http://localhost:8501`

✅ **UI is running!**

---

## ✅ Verification Checklist

You should now have:

- [ ] Terminal 1: Backend running on port 8000
- [ ] Terminal 2: Streamlit UI on port 8501
- [ ] Browser showing the Streamlit interface
- [ ] Sidebar showing: Nodes: 22, Edges: 34

---

## 🎯 Quick Test

In the Streamlit UI:

1. Go to **Search** tab
2. Enter: `"Neural architecture search by Stanford researchers"`
3. Select: **Hybrid (Best!)**
4. Click: **Search**

You should see results with:
- Scores around 0.8-0.9
- Explanations showing graph paths
- Papers by Stanford researchers about NAS

✅ **If you see this, you're ready to demo!**

---

## 🧪 Run Tests (Optional but Recommended)

In terminal 2:

```bash
python test_api.py
```

You should see all tests pass:
```
╔════════════════════════════════════════════════════════════╗
║           🎉 ALL TESTS PASSED! 🎉                          ║
║     Your system is ready for the demo!                     ║
╚════════════════════════════════════════════════════════════╝
```

---

## 🎬 Demo Preparation

### What to Open Before Presenting:

**Browser Tab 1:** `http://localhost:8501` (Streamlit UI)
**Browser Tab 2:** `http://localhost:8000/docs` (API Documentation)

### Have Ready:
- This laptop with everything running
- Demo script (DEMO_SCRIPT.md) on phone or printed
- Backup: `test_api.py` output screenshot

### Practice These 3 Queries:

1. **"Neural architecture search papers by Stanford researchers"**
   - Shows: Topic match + institution connection

2. **"Graph neural networks for knowledge graphs"**  
   - Shows: Topic overlap + citations

3. **"Transformer models for computer vision"**
   - Shows: Cross-domain connections

---

## 🔥 Pro Tips

### Performance:
- First query might be slow (model loading) - run one practice query before demo
- Close other applications to free RAM
- Use Chrome/Firefox for best performance

### If Something Breaks:

**Backend crashes:**
```bash
# Restart it
python main.py
```

**UI crashes:**
```bash
# Restart it
streamlit run app.py
```

**Data corruption:**
```bash
# Reload demo data
python populate_demo_data.py
```

**Everything is broken:**
```bash
# Nuclear option - restart everything
# Stop all terminals (Ctrl+C)
# Restart backend
python main.py

# In new terminal, reload data
python populate_demo_data.py

# Start UI
streamlit run app.py
```

---

## 📊 Understanding Your System

### System Stats:
- **22 Nodes:** 3 institutions, 5 authors, 10 papers, 4 topics
- **34 Edges:** Authorship, affiliations, citations, topics
- **Average Degree:** ~3.1 (well-connected graph)
- **Embedding Dimensions:** 384
- **Query Speed:** ~50-100ms

### Data Schema:
```
Institutions ←affiliated_with← Authors →authored_by→ Papers
    ↓                                                   ↓
  (3)                                                (10)
                                                      ↓
                                              →belongs_to_topic→ Topics
                                                                   (4)
Papers →cites→ Papers
```

---

## 🎯 Sample Queries for Demo

### Easy (High Confidence):
1. "machine learning" - broad, should match many
2. "Stanford researchers" - institution filter
3. "neural networks" - specific topic

### Medium (Show Hybrid Power):
1. "neural architecture search by Stanford" - topic + institution
2. "graph neural networks knowledge graphs" - related topics
3. "transformer models vision" - cross-domain

### Advanced (Wow Factor):
1. "AutoML papers that cite reinforcement learning" - multi-hop
2. "recent deep learning research at MIT" - institution + topic
3. "computer vision by Sarah Chen" - author-specific

---

## 🏆 You're Ready!

If you got here successfully, you have:

✅ Working backend API
✅ Loaded demo dataset  
✅ Running UI interface
✅ Tested the system
✅ 3 demo queries practiced

**Time to win this hackathon! 🚀**

---

## 📞 Quick Commands Reference

```bash
# Start backend
python main.py

# Load data
python populate_demo_data.py

# Start UI
streamlit run app.py

# Run tests
python test_api.py

# Check API health
curl http://localhost:8000/health

# Search via API
curl -X POST "http://localhost:8000/search/hybrid" \
     -H "Content-Type: application/json" \
     -d '{"query_text": "neural networks", "top_k": 5, "vector_weight": 0.5, "graph_weight": 0.5}'
```

---

## ❓ Troubleshooting

### "ModuleNotFoundError: No module named 'fastapi'"
**Solution:** Activate virtual environment and reinstall
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Port 8000 already in use"
**Solution:** Kill the process or change port
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9

# Or change port in main.py (last line)
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### "Cannot connect to API"
**Solution:** Make sure backend is running in terminal 1

### "Model download is slow"
**Solution:** First run downloads ~90MB model. Be patient or use WiFi.

### "Low memory warning"
**Solution:** Close other applications, or reduce batch size in code

---

**Need help? Check the full README.md for detailed documentation!**

🎉 **GOOD LUCK WITH YOUR DEMO!** 🎉