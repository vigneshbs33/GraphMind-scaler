# 🚀 Streamlit Deployment Guide for GraphMind

This guide will help you deploy GraphMind to Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub Account** - Your code needs to be in a GitHub repository
2. **Streamlit Cloud Account** - Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)
3. **API Keys** (Optional) - Gemini API key if you want LLM features

## 🎯 Deployment Steps

### Step 1: Prepare Your Repository

1. Make sure your code is pushed to GitHub
2. Ensure `app.py` is in the root directory
3. Ensure `requirements.txt` includes all dependencies

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: [share.streamlit.io](https://share.streamlit.io)
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Fill in the details**:
   - **Repository**: Select your GitHub repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom subdomain (optional)
5. **Click "Deploy"**

### Step 3: Configure Environment Variables (Optional)

If you need to set API keys or other environment variables:

1. In Streamlit Cloud, go to your app settings
2. Click on "Secrets" or "Environment variables"
3. Add your variables:
   ```
   GEMINI_API_KEY=your-api-key-here
   CLAUDE_API_KEY=your-claude-key-here
   ```

Or create a `.streamlit/secrets.toml` file in your repository:
```toml
GEMINI_API_KEY = "your-api-key-here"
CLAUDE_API_KEY = "your-claude-key-here"
```

### Step 4: Wait for Deployment

- Streamlit will automatically install dependencies from `requirements.txt`
- First deployment may take 2-5 minutes
- You'll see logs in the Streamlit Cloud dashboard

## 🔧 Local Testing Before Deployment

Test your app locally first:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
GraphMind-scaler/
├── app.py                 # Streamlit app (main entry point)
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── backend/               # Backend modules
│   ├── config.py
│   ├── storage.py
│   ├── ingestion.py
│   └── ...
├── data/                  # Data storage (created automatically)
│   ├── chroma/           # ChromaDB storage
│   └── uploads/          # Uploaded files
└── README_STREAMLIT.md   # This file
```

## ⚙️ Configuration

### Streamlit Config (`.streamlit/config.toml`)

The config file includes:
- Server settings (port, CORS)
- Theme customization
- Performance settings

### Backend Config (`backend/config.py`)

Key settings you might want to adjust:
- `LLM_PROVIDER`: "gemini", "claude", or "mock"
- `EMBEDDING_MODEL`: Embedding model name
- `MAX_FILE_SIZE`: Maximum upload size (default: 50MB)
- `CHUNK_SIZE`: Text chunking size

## 🐛 Troubleshooting

### App Won't Start

1. **Check logs** in Streamlit Cloud dashboard
2. **Verify dependencies** in `requirements.txt` are correct
3. **Check Python version** - Streamlit Cloud uses Python 3.9+

### Import Errors

- Ensure all backend modules are in the `backend/` directory
- Check that `app.py` has the correct path setup:
  ```python
  sys.path.insert(0, str(Path(__file__).parent))
  ```

### File Upload Issues

- Check file size limits in `backend/config.py`
- Ensure `data/uploads/` directory is writable
- Check file type is supported (txt, pdf, xml, json, csv)

### ChromaDB Issues

- ChromaDB data is stored in `data/chroma/`
- On Streamlit Cloud, this is ephemeral (resets on redeploy)
- For persistent storage, consider using external storage

### LLM Not Working

- Verify API keys are set correctly
- Check internet connectivity
- Try using "mock" provider for testing

## 📊 Features Available in Streamlit App

1. **🔍 Search**: Vector, graph, and hybrid search
2. **📤 Upload**: Upload and ingest documents
3. **🕸️ Graph**: Interactive graph visualization
4. **📊 Stats**: Database statistics and analytics
5. **⚙️ Settings**: Configuration and data management

## 🔄 Updating Your App

1. **Make changes** to your code
2. **Push to GitHub**
3. **Streamlit Cloud automatically redeploys** (or click "Reboot app" in dashboard)

## 📝 Notes

- **Data Persistence**: Data stored in `data/` directory is ephemeral on Streamlit Cloud
- **File Limits**: Streamlit Cloud has resource limits (CPU, memory, storage)
- **API Keys**: Never commit API keys to GitHub. Use Streamlit secrets instead.
- **Performance**: Large graphs may take time to visualize

## 🎉 Success!

Once deployed, you'll have a public URL like:
`https://your-app-name.streamlit.app`

Share this URL to let others use your GraphMind application!

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Cloud Guide](https://docs.streamlit.io/streamlit-community-cloud)
- [GraphMind README](./README.md) - Full project documentation

---

**Happy Deploying! 🚀**

