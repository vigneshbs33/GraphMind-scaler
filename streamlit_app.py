"""Streamlit app for GraphMind - Simple Frontend with USP Button."""

import streamlit as st
from pathlib import Path
import sys

# Page config
st.set_page_config(
    page_title="GraphMind - Hybrid RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get paths
project_root = Path(__file__).parent
usp_path = project_root / "USP1_vector_graph_search" / "parse_zip" / "streamlit_ui.py"

# Sidebar
with st.sidebar:
    st.title("🧠 GraphMind")
    st.markdown("**Hybrid Vector + Graph Database**")
    st.markdown("---")
    
    # USP Button
    if st.button("🚀 USP", type="primary", use_container_width=True):
        st.session_state.show_usp = True
        st.rerun()
    
    # Upload Button - Show popup
    if st.button("📤 Upload", use_container_width=True):
        st.session_state.show_download_popup = True
        st.rerun()

# Show download popup if upload button was clicked
if st.session_state.get("show_download_popup", False):
    st.error("""
    ⚠️ **Backend Required for Upload**
    
    To use the upload feature, please download the full repository:
    
    1. **Clone the repository:**
       ```bash
       git clone https://github.com/vigneshbs33/GraphMind-scaler
       cd GraphMind-scaler
       ```
    
    2. **Install dependencies:**
       ```bash
       pip install -r requirements.txt
       ```
    
    3. **Start the backend:**
       ```bash
       uvicorn backend.main:app --reload
       ```
    
    4. **Then use the full application with all features!**
    
    See README.md for complete installation instructions.
    """)
    
    if st.button("Got it", type="primary"):
        st.session_state.show_download_popup = False
        st.rerun()

# Show USP interface
if st.session_state.get("show_usp", False):
    if usp_path.exists():
        st.info("""
        **🚀 USP Interface**
        
        To run the full USP interface with backend:
        
        1. **Download the repository** (if you haven't already)
        2. **Install dependencies:** `pip install -r requirements.txt`
        3. **Start backend:** `uvicorn backend.main:app --reload` (in one terminal)
        4. **Run USP:** `streamlit run USP1_vector_graph_search/parse_zip/streamlit_ui.py` (in another terminal)
        
        The USP interface requires the backend API to be running on `http://localhost:8000`
        """)
        
        st.markdown("---")
        st.markdown("### USP Features Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔍 Search Tab:**
            - Vector Search (Semantic Similarity)
            - Graph Traversal (Relationship Reasoning)
            - Hybrid Search (Best of Both!)
            - Adjustable weights
            """)
        
        with col2:
            st.markdown("""
            **📊 Other Tabs:**
            - Comparison: Side-by-side method comparison
            - Statistics: System health metrics
            - Add Data: Create nodes and relationships
            """)
        
        st.code("""
# To run USP locally:
# Terminal 1:
uvicorn backend.main:app --reload

# Terminal 2:
streamlit run USP1_vector_graph_search/parse_zip/streamlit_ui.py
        """, language="bash")
        
    else:
        st.error("❌ USP file not found!")
        st.info(f"Expected path: `{usp_path}`")
else:
    # Default landing page
    st.header("Welcome to GraphMind 🧠")
    st.markdown("**Hybrid Vector + Graph Database**")
    
    st.info("""
    👆 **Click the USP button in the sidebar** to learn about the Vector + Graph Hybrid Database interface.
    
    **Note:** For full functionality including uploads and the USP interface, please download the repository and follow the installation instructions in README.md
    """)
    
    st.markdown("---")
    st.markdown("### Quick Links")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📚 Features:**
        - Vector Search (Semantic Similarity)
        - Graph Traversal (Relationship Reasoning)
        - Hybrid Search (Best of Both!)
        - Interactive UI
        """)
    
    with col2:
        st.markdown("""
        **🚀 Get Started:**
        1. Click **USP** button for info
        2. Download repo for full features
        3. See README.md for setup
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "🧠 GraphMind - Hybrid Vector + Graph Database | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
