# 🚀 GraphMind Streamlit Deployment - Step by Step

## Quick Deployment Steps

### Step 1: Test Locally First
```bash
cd GraphMind-scaler
pip install -r requirements.txt
streamlit run app.py
```

### Step 2: Push to GitHub
```bash
git add .
git commit -m "Add Streamlit deployment"
git push origin main
```

### Step 3: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file: `app.py`
6. Click "Deploy"

### Step 4: Configure Secrets (Optional)

In Streamlit Cloud → Your App → Settings → Secrets:
```
GEMINI_API_KEY=your-key-here
```

## What Was Created

✅ **app.py** - Main Streamlit application
✅ **requirements.txt** - Updated with Streamlit dependencies
✅ **.streamlit/config.toml** - Streamlit configuration
✅ **README_STREAMLIT.md** - Full deployment guide

## Features Available

- 🔍 **Search**: Vector, Graph, and Hybrid search modes
- 📤 **Upload**: Upload and ingest documents (txt, pdf, xml, json, csv)
- 🕸️ **Graph**: Interactive knowledge graph visualization
- 📊 **Stats**: Database statistics and analytics
- ⚙️ **Settings**: Configuration and data management

## Troubleshooting

If the app doesn't start:
1. Check Streamlit Cloud logs
2. Verify all dependencies in requirements.txt
3. Ensure `app.py` is in the root directory

## Next Steps

1. Test locally: `streamlit run app.py`
2. Push to GitHub
3. Deploy to Streamlit Cloud
4. Share your app URL!

---

**Your app will be available at:** `https://your-app-name.streamlit.app`

