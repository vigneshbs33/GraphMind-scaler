# Streamlit Cloud Deployment Guide

## ✅ Deployment Ready

The Streamlit app (`streamlit_app.py`) is now configured for deployment on Streamlit Cloud.

## 📋 What's Included

- **Frontend**: `streamlit_app.py` - Complete Streamlit UI
- **Backend**: `backend/` directory - All backend modules
- **Dependencies**: `requirements.txt` - Fixed for Python 3.13 compatibility

## 🚀 Deployment Steps

1. **Push to GitHub**: Ensure your repository is pushed to GitHub
2. **Connect to Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file: `streamlit_app.py`
3. **Deploy**: Streamlit Cloud will automatically:
   - Clone the full repository (including backend)
   - Install dependencies from `requirements.txt`
   - Run the Streamlit app

## 🔍 Backend Detection

The app automatically:
- ✅ Checks if `backend/` directory exists
- ✅ Verifies backend imports work
- ✅ Shows helpful error messages if backend is missing
- ✅ Uses the full repository structure

## 📁 Required Repository Structure

```
GraphMind-scaler/
├── streamlit_app.py      # Main Streamlit app
├── backend/               # Backend modules
│   ├── __init__.py
│   ├── config.py
│   ├── storage.py
│   ├── ingestion.py
│   └── ...
├── requirements.txt       # Dependencies
└── data/                  # Data directory (created automatically)
    └── chroma/            # ChromaDB storage
```

## ⚠️ Important Notes

- **Full Repository Required**: The entire repository (including `backend/`) must be available
- **Streamlit Cloud**: Automatically clones the full repo, so backend will be available
- **Local Testing**: Run `streamlit run streamlit_app.py` from the repository root

## 🐛 Troubleshooting

If you see "Backend not found":
- Ensure the repository contains the `backend/` folder
- Check that all files are committed and pushed to GitHub
- Verify the repository structure matches the requirements above

If you see import errors:
- Check that `requirements.txt` is up to date
- Ensure all dependencies are installed
- Verify Python version compatibility (3.13+)

