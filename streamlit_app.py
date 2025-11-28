"""Streamlit app for GraphMind - Static Frontend Display."""

import streamlit as st
from pathlib import Path

# Page config
st.set_page_config(
    page_title="GraphMind - Hybrid RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get the frontend HTML file path
project_root = Path(__file__).parent
frontend_path = project_root / "frontend" / "index.html"

# Sidebar with USP button
with st.sidebar:
    st.title("🧠 GraphMind")
    st.markdown("**Hybrid Vector + Graph Database**")
    st.markdown("---")
    if st.button("🚀 USP", type="primary", use_container_width=True):
        st.session_state.show_usp = True

# Check if frontend exists
if frontend_path.exists():
    # Read the HTML content
    with open(frontend_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Read CSS and JS
    css_path = project_root / "frontend" / "style.css"
    js_path = project_root / "frontend" / "app.js"
    
    css_content = ""
    if css_path.exists():
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
    
    js_content = ""
    if js_path.exists():
        with open(js_path, 'r', encoding='utf-8') as f:
            js_content = f.read()
    
    # Inject CSS into HTML
    if css_content:
        html_content = html_content.replace(
            '<link rel="stylesheet" href="/static/style.css">',
            f'<style>{css_content}</style>'
        )
    
    # Inject JS into HTML
    if js_content:
        html_content = html_content.replace(
            '<script src="/static/app.js"></script>',
            f'<script>{js_content}</script>'
        )
    
    # Fix any static paths
    html_content = html_content.replace('/static/', '')
    
    # Display the HTML
    st.components.v1.html(html_content, height=900, scrolling=True)
    
else:
    st.error("❌ Frontend files not found!")
    st.info("""
    **Expected structure:**
    - `frontend/index.html`
    - `frontend/style.css`  
    - `frontend/app.js`
    """)
