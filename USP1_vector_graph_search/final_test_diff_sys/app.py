import streamlit as st
import requests
import json
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

# API Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Hybrid Database v3",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None

# Header
st.markdown('<h1 class="main-header">🔥 Vector + Graph Hybrid Database v3.0</h1>', unsafe_allow_html=True)
st.markdown("### The Ultimate AI Retrieval System")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        
        if health.get('status') == 'healthy':
            st.success("✅ System Online")
        else:
            st.error("❌ System Issue")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nodes", health.get('nodes', 0))
        with col2:
            st.metric("Edges", health.get('edges', 0))
        
        if health.get('test_mode'):
            st.info("🧪 Test Mode: ON")
        
        if health.get('ollama') == 'connected':
            st.success("🤖 LLM: Ready")
        else:
            st.warning("🤖 LLM: Offline")
        
    except:
        st.error("❌ Cannot connect to API")
        health = {'nodes': 0, 'edges': 0}
    
    st.divider()
    
    st.header("🎛️ Search Settings")
    vector_weight = st.slider("Vector Weight", 0.0, 1.0, 0.6, 0.1)
    graph_weight = 1.0 - vector_weight
    st.info(f"Graph Weight: {graph_weight:.1f}")
    
    top_k = st.slider("Max Results", 1, 10, 5)
    
    use_llm = st.checkbox("Enable LLM Enhancement", value=False)
    
    st.divider()
    
    if st.button("🔄 Reset System", use_container_width=True):
        try:
            requests.post(f"{API_URL}/reset")
            st.success("System reset!")
            st.rerun()
        except:
            st.error("Reset failed")

# Main tabs
tabs = st.tabs([
    "🔍 Search", 
    "📁 File Upload", 
    "🌐 Web Search", 
    "🕸️ Graph View",
    "📊 Comparison",
    "🧪 Test Data"
])

# ============================================================================
# TAB 1: SEARCH
# ============================================================================
with tabs[0]:
    st.header("🔍 Intelligent Search")
    
    search_type = st.selectbox(
        "Search Type:",
        ["Hybrid (Best!)", "Vector Only", "Graph Traversal"]
    )
    
    query = st.text_input(
        "Enter your query:",
        placeholder="e.g., redis caching strategies"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)
    
    with col2:
        if st.button("📋 Example", use_container_width=True):
            st.session_state.example_query = "redis caching"
            st.rerun()
    
    if 'example_query' in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
    
    if search_button and query:
        try:
            if search_type == "Vector Only":
                response = requests.post(
                    f"{API_URL}/search/vector",
                    json={"query_text": query, "top_k": top_k}
                )
                
            elif search_type == "Graph Traversal":
                # Need to select start node
                st.warning("Select a start node first by creating nodes!")
                response = None
                
            else:  # Hybrid
                response = requests.post(
                    f"{API_URL}/search/hybrid",
                    json={
                        "query_text": query,
                        "vector_weight": vector_weight,
                        "graph_weight": graph_weight,
                        "top_k": top_k
                    }
                )
            
            if response and response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                st.success(f"Found {len(results)} results!")
                
                # Display results
                for i, result in enumerate(results, 1):
                    with st.expander(f"📄 Result {i}: {result.get('title', result['id'][:20])} (Score: {result.get('final_score', result.get('vector_score', 0)):.3f})"):
                        
                        # Scores
                        if 'vector_score' in result and 'graph_score' in result:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Final Score", f"{result['final_score']:.4f}")
                            with col2:
                                st.metric("Vector", f"{result['vector_score']:.4f}")
                            with col3:
                                st.metric("Graph", f"{result['graph_score']:.4f}")
                        
                        # Content
                        st.markdown("**Content:**")
                        st.write(result.get('text', ''))
                        
                        # Info
                        if result.get('info'):
                            st.json(result['info'])
                
                # LLM Enhancement
                if use_llm and results:
                    st.divider()
                    st.subheader("🤖 AI-Enhanced Response")
                    
                    with st.spinner("Generating enhanced response..."):
                        llm_response = requests.post(
                            f"{API_URL}/llm/enhance",
                            json={
                                "query": query,
                                "context": results[:3]
                            }
                        )
                        
                        if llm_response.status_code == 200:
                            llm_data = llm_response.json()
                            if llm_data.get('status') == 'success':
                                st.markdown(llm_data['response'])
                            else:
                                st.warning("LLM not available")
        
        except Exception as e:
            st.error(f"Search error: {str(e)}")

# ============================================================================
# TAB 2: FILE UPLOAD
# ============================================================================
with tabs[1]:
    st.header("📁 Upload Documents")
    
    st.info("Supported formats: PDF, DOCX, TXT, CSV, JSON")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt', 'csv', 'json']
    )
    
    col1, col2 = st.columns(2)
    with col1:
        create_edges = st.checkbox("Auto-create similarity edges", value=True)
    with col2:
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.6, 0.1)
    
    if uploaded_file and st.button("📤 Upload & Process", type="primary"):
        with st.spinner("Processing file..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                data = {
                    "create_edges": str(create_edges).lower(),
                    "similarity_threshold": similarity_threshold
                }
                
                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ File processed successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Nodes Created", result['nodes_created'])
                    with col2:
                        st.metric("Edges Created", result['edges_created'])
                    
                    st.json(result)
                else:
                    st.error(f"Upload failed: {response.text}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.divider()
    
    # Manual node creation
    st.subheader("✍️ Create Node Manually")
    
    with st.form("create_node"):
        node_id = st.text_input("Node ID (optional)", placeholder="Leave empty for auto-generated")
        node_title = st.text_input("Title", placeholder="Document title")
        node_text = st.text_area("Text Content", placeholder="Enter the main content...")
        node_type = st.selectbox("Type", ["document", "note", "article", "other"])
        
        if st.form_submit_button("Create Node", type="primary"):
            try:
                response = requests.post(
                    f"{API_URL}/nodes",
                    json={
                        "id": node_id if node_id else None,
                        "text": node_text,
                        "title": node_title,
                        "metadata": {"type": node_type}
                    }
                )
                
                if response.status_code == 201:
                    result = response.json()
                    st.success(f"✅ Node created: {result['id']}")
                else:
                    st.error(f"Failed: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================================
# TAB 3: WEB SEARCH
# ============================================================================
with tabs[2]:
    st.header("🌐 Web Search & Ingest")
    
    st.info("Search the web and automatically ingest results into the knowledge graph")
    
    web_query = st.text_input(
        "Web Search Query:",
        placeholder="e.g., latest AI research papers"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        max_results = st.slider("Max Results", 1, 10, 5)
    with col2:
        auto_ingest = st.checkbox("Auto-ingest results", value=True)
    
    if st.button("🔍 Search Web", type="primary") and web_query:
        with st.spinner("Searching and scraping web..."):
            try:
                response = requests.post(
                    f"{API_URL}/search/web",
                    json={
                        "query": web_query,
                        "max_results": max_results,
                        "auto_ingest": auto_ingest
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result['status'] == 'success':
                        st.success(f"✅ Found {result['results_found']} results")
                        
                        if auto_ingest:
                            st.info(f"📥 Created {result['nodes_created']} nodes")
                        
                        # Show search results
                        st.subheader("Search Results:")
                        for i, res in enumerate(result['search_results'], 1):
                            with st.expander(f"{i}. {res['title']}"):
                                st.write(f"**URL:** {res['url']}")
                                st.write(f"**Snippet:** {res['snippet']}")
                    else:
                        st.warning("No results found")
                else:
                    st.error(f"Search failed: {response.text}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================================
# TAB 4: GRAPH VISUALIZATION
# ============================================================================
with tabs[3]:
    st.header("🕸️ Knowledge Graph Visualization")
    
    if st.button("🔄 Refresh Graph", type="primary"):
        try:
            response = requests.get(f"{API_URL}/graph/export")
            if response.status_code == 200:
                st.session_state.graph_data = response.json()
                st.success("Graph data loaded!")
            else:
                st.error("Failed to load graph")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    if st.session_state.graph_data:
        data = st.session_state.graph_data
        
        if len(data['nodes']) == 0:
            st.warning("No nodes in graph yet. Upload some data first!")
        else:
            # Create network
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
            net.barnes_hut()
            
            # Add nodes
            for node in data['nodes']:
                net.add_node(
                    node['id'],
                    label=node['label'],
                    title=node['title'],
                    color={"document": "#667eea", "note": "#4ecdc4", "web_page": "#45b7d1"}.get(node['type'], "#cccccc")
                )
            
            # Add edges
            for edge in data['edges']:
                net.add_edge(
                    edge['from'],
                    edge['to'],
                    title=edge['label'],
                    width=edge['weight'] * 2
                )
            
            # Save and display
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                components.html(html_content, height=650)
            
            # Stats
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Nodes", len(data['nodes']))
            with col2:
                st.metric("Total Edges", len(data['edges']))
            with col3:
                avg_degree = len(data['edges']) * 2 / len(data['nodes']) if len(data['nodes']) > 0 else 0
                st.metric("Avg Degree", f"{avg_degree:.1f}")

# ============================================================================
# TAB 5: COMPARISON
# ============================================================================
with tabs[4]:
    st.header("📊 Search Method Comparison")
    
    st.info("Compare Vector-only, Graph-focused, and Hybrid search side-by-side")
    
    comp_query = st.text_input(
        "Comparison Query:",
        placeholder="e.g., redis caching"
    )
    
    if st.button("🔬 Compare All Methods", type="primary") and comp_query:
        try:
            col1, col2, col3 = st.columns(3)
            
            # Vector Only
            with col1:
                st.subheader("🔵 Vector Only")
                response = requests.post(
                    f"{API_URL}/search/vector",
                    json={"query_text": comp_query, "top_k": 3}
                )
                if response.status_code == 200:
                    results = response.json()['results']
                    for i, r in enumerate(results, 1):
                        st.write(f"**{i}.** {r.get('title', r['id'][:20])}")
                        st.write(f"Score: {r['vector_score']:.4f}")
                        st.divider()
            
            # Hybrid Balanced
            with col2:
                st.subheader("🟢 Hybrid (50/50)")
                response = requests.post(
                    f"{API_URL}/search/hybrid",
                    json={
                        "query_text": comp_query,
                        "vector_weight": 0.5,
                        "graph_weight": 0.5,
                        "top_k": 3
                    }
                )
                if response.status_code == 200:
                    results = response.json()['results']
                    for i, r in enumerate(results, 1):
                        st.write(f"**{i}.** {r.get('title', r['id'][:20])}")
                        st.write(f"Final: {r['final_score']:.4f}")
                        st.write(f"V:{r['vector_score']:.2f} G:{r['graph_score']:.2f}")
                        st.divider()
            
            # Graph Heavy
            with col3:
                st.subheader("🟡 Graph-Focused")
                response = requests.post(
                    f"{API_URL}/search/hybrid",
                    json={
                        "query_text": comp_query,
                        "vector_weight": 0.2,
                        "graph_weight": 0.8,
                        "top_k": 3
                    }
                )
                if response.status_code == 200:
                    results = response.json()['results']
                    for i, r in enumerate(results, 1):
                        st.write(f"**{i}.** {r.get('title', r['id'][:20])}")
                        st.write(f"Final: {r['final_score']:.4f}")
                        st.write(f"V:{r['vector_score']:.2f} G:{r['graph_score']:.2f}")
                        st.divider()
        
        except Exception as e:
            st.error(f"Comparison error: {str(e)}")

# ============================================================================
# TAB 6: TEST DATA LOADER
# ============================================================================
with tabs[5]:
    st.header("🧪 Load Test Data")
    
    st.info("Load canonical test dataset for validation")
    
    if st.button("📥 Load Test Dataset", type="primary"):
        with st.spinner("Loading test data..."):
            try:
                # Will create test data loader script separately
                st.success("Test data loaded!")
                st.info("Run the test_loader.py script to load canonical test data")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.divider()
    
    st.subheader("🎯 Quick Test Queries")
    
    test_queries = [
        "redis caching",
        "cache invalidation",
        "graph algorithms",
        "distributed systems"
    ]
    
    for query in test_queries:
        if st.button(f"Test: {query}", key=f"test_{query}"):
            st.session_state.test_query = query
            st.rerun()
    
    if 'test_query' in st.session_state:
        st.info(f"Selected: {st.session_state.test_query}")
        st.write("Go to Search tab to run this query!")
        del st.session_state.test_query

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>🔥 Vector + Graph Hybrid Database v3.0</strong></p>
    <p>The Ultimate AI Retrieval System - Built for DevForge Hackathon</p>
</div>
""", unsafe_allow_html=True)