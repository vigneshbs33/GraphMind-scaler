# 🚀 GraphMind Streamlit Deployment Guide

## ✅ Ready to Deploy!

Your `streamlit_app.py` is now the main entry point and ready for deployment.

## 📋 Quick Start

### 1. Test Locally First

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### 2. Deploy to Streamlit Cloud

1. **Push to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Add Streamlit deployment"
   git push origin main
   ```

2. **Go to Streamlit Cloud**: [share.streamlit.io](https://share.streamlit.io)

3. **Sign in** with your GitHub account

4. **Click "New app"**

5. **Configure**:
   - **Repository**: Select your GitHub repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `streamlit_app.py` ⭐
   - **App URL**: Choose a custom subdomain (optional)

6. **Click "Deploy"**

7. **Wait 2-5 minutes** for first deployment

## 🔧 Configuration

### Environment Variables (Optional)

If you need API keys, add them in Streamlit Cloud:

1. Go to your app → Settings → Secrets
2. Add:
   ```
   GEMINI_API_KEY=your-key-here
   CLAUDE_API_KEY=your-claude-key-here
   ```

Or create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key-here"
CLAUDE_API_KEY = "your-claude-key-here"
```

## 📁 Project Structure

```
deploy/
├── streamlit_app.py          # ⭐ Main entry point
├── requirements.txt          # Dependencies
├── .streamlit/
│   └── config.toml          # Streamlit config
├── GraphMind-scaler/
│   ├── backend/             # Backend modules
│   └── data/                # Data storage
└── DEPLOYMENT_GUIDE.md      # This file
```

## ✨ Features

- 🔍 **Search**: Vector, Graph, and Hybrid search with AI answers
- 📤 **Upload**: Upload documents (txt, pdf, xml, json, csv)
- 🕸️ **Graph**: Interactive knowledge graph visualization
- 📊 **Stats**: Database statistics and analytics
- ⚙️ **Settings**: Configuration and data management

## 🐛 Troubleshooting

### App Won't Start

1. **Check logs** in Streamlit Cloud dashboard
2. **Verify** `streamlit_app.py` is in root directory
3. **Check** `requirements.txt` has all dependencies
4. **Ensure** `GraphMind-scaler/backend/` exists

### Import Errors

- Verify `GraphMind-scaler/backend/` directory structure
- Check that all backend modules are present
- Ensure Python 3.9+ is used

### File Upload Issues

- Check file size limits (default: 50MB)
- Verify file types are supported
- Check `GraphMind-scaler/data/uploads/` is writable

### ChromaDB Issues

- Data is stored in `GraphMind-scaler/data/chroma/`
- On Streamlit Cloud, this is ephemeral (resets on redeploy)
- For persistence, consider external storage

## 📝 Notes

- **Data Persistence**: Data in `GraphMind-scaler/data/` is ephemeral on Streamlit Cloud
- **API Keys**: Never commit API keys to GitHub. Use Streamlit secrets.
- **Performance**: Large graphs may take time to visualize
- **First Load**: First deployment takes 2-5 minutes to install dependencies

## 🎉 Success!

Once deployed, your app will be available at:
`https://your-app-name.streamlit.app`

## 📚 Next Steps

1. ✅ Test locally: `streamlit run streamlit_app.py`
2. ✅ Push to GitHub
3. ✅ Deploy to Streamlit Cloud
4. ✅ Share your app URL!

---

**Your app is ready! 🚀**

