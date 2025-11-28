import streamlit as st
import requests
import json
import pandas as pd
from typing import List, Dict

# API Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Vector + Graph Hybrid Database",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .high-score {
        background-color: #10b981;
        color: white;
    }
    .medium-score {
        background-color: #f59e0b;
        color: white;
    }
    .low-score {
        background-color: #6b7280;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔍 Vector + Graph Hybrid Database</h1>', unsafe_allow_html=True)
st.markdown("### Combining Semantic Search with Relationship Intelligence")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Info")
    
    try:
        health = requests.get(f"{API_URL}/health").json()
        st.success("✅ System Online")
        st.metric("Nodes", health['nodes'])
        st.metric("Edges", health['edges'])
        st.metric("Embeddings", health['embeddings'])
        
        st.divider()
        
        st.header("📚 About")
        st.info("""
        This system combines:
        - **Vector Search**: Semantic similarity
        - **Graph Traversal**: Relationship reasoning
        - **Hybrid Search**: The best of both!
        """)
        
    except:
        st.error("❌ Cannot connect to API")
        st.info("Make sure the backend is running on port 8000")

# Main content
tabs = st.tabs(["🔍 Search", "📊 Comparison", "📈 Statistics", "➕ Add Data"])

# ============================================================================
# TAB 1: SEARCH INTERFACE
# ============================================================================
with tabs[0]:
    st.header("Search Your Knowledge Graph")
    
    # Search input
    query = st.text_input("🔎 Enter your search query:", placeholder="e.g., Neural architecture search by Stanford researchers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_type = st.selectbox("Search Type:", ["Hybrid (Best!)", "Vector Only", "Graph Only"])
    
    with col2:
        top_k = st.slider("Number of results:", 1, 10, 5)
    
    with col3:
        if search_type == "Hybrid (Best!)":
            vector_weight = st.slider("Vector Weight:", 0.0, 1.0, 0.5, 0.1)
            graph_weight = 1.0 - vector_weight
    
    if st.button("🚀 Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                try:
                    if search_type == "Vector Only":
                        response = requests.post(f"{API_URL}/search/vector", json={
                            "query_text": query,
                            "top_k": top_k
                        })
                        results = response.json()
                        
                    elif search_type == "Hybrid (Best!)":
                        response = requests.post(f"{API_URL}/search/hybrid", json={
                            "query_text": query,
                            "top_k": top_k,
                            "vector_weight": vector_weight,
                            "graph_weight": graph_weight
                        })
                        results = response.json()
                    
                    # Display results
                    st.success(f"Found {len(results)} results!")
                    
                    for i, result in enumerate(results, 1):
                        score = result['score']
                        score_class = "high-score" if score > 0.7 else "medium-score" if score > 0.4 else "low-score"
                        
                        with st.container():
                            st.markdown(f"""
                            <div class="result-card">
                                <h4>#{i} - <span class="score-badge {score_class}">Score: {score:.3f}</span></h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.write(f"**Text:** {result['text'][:200]}...")
                            
                            if result.get('explanation'):
                                st.info(f"💡 {result['explanation']}")
                            
                            # Metadata
                            if result.get('metadata'):
                                with st.expander("📋 Metadata"):
                                    st.json(result['metadata'])
                            
                            st.divider()
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a search query!")

# ============================================================================
# TAB 2: COMPARISON
# ============================================================================
with tabs[1]:
    st.header("🔬 Compare Search Methods")
    st.info("See how Vector, Graph, and Hybrid search perform on the same query!")
    
    compare_query = st.text_input("Enter query for comparison:", 
                                   placeholder="e.g., Graph neural networks for knowledge graphs")
    
    if st.button("Compare All Methods", type="primary"):
        if compare_query:
            col1, col2, col3 = st.columns(3)
            
            # Vector Search
            with col1:
                st.subheader("🔵 Vector Search")
                try:
                    response = requests.post(f"{API_URL}/search/vector", json={
                        "query_text": compare_query,
                        "top_k": 5
                    })
                    vector_results = response.json()
                    
                    for i, result in enumerate(vector_results[:3], 1):
                        st.write(f"**{i}.** {result['text'][:80]}...")
                        st.write(f"Score: {result['score']:.3f}")
                        st.divider()
                except Exception as e:
                    st.error(f"Error: {e}")
            
            # Hybrid Search
            with col2:
                st.subheader("🟢 Hybrid Search")
                try:
                    response = requests.post(f"{API_URL}/search/hybrid", json={
                        "query_text": compare_query,
                        "top_k": 5,
                        "vector_weight": 0.5,
                        "graph_weight": 0.5
                    })
                    hybrid_results = response.json()
                    
                    for i, result in enumerate(hybrid_results[:3], 1):
                        st.write(f"**{i}.** {result['text'][:80]}...")
                        st.write(f"Score: {result['score']:.3f}")
                        st.success(f"V:{result.get('vector_score', 0):.2f} | G:{result.get('graph_score', 0):.2f}")
                        st.divider()
                except Exception as e:
                    st.error(f"Error: {e}")
            
            # Graph-based (using hybrid with high graph weight)
            with col3:
                st.subheader("🟡 Graph-Focused")
                try:
                    response = requests.post(f"{API_URL}/search/hybrid", json={
                        "query_text": compare_query,
                        "top_k": 5,
                        "vector_weight": 0.2,
                        "graph_weight": 0.8
                    })
                    graph_results = response.json()
                    
                    for i, result in enumerate(graph_results[:3], 1):
                        st.write(f"**{i}.** {result['text'][:80]}...")
                        st.write(f"Score: {result['score']:.3f}")
                        st.divider()
                except Exception as e:
                    st.error(f"Error: {e}")

# ============================================================================
# TAB 3: STATISTICS
# ============================================================================
with tabs[2]:
    st.header("📊 System Statistics")
    
    try:
        stats = requests.get(f"{API_URL}/stats").json()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Nodes", stats['nodes'])
        
        with col2:
            st.metric("Total Edges", stats['edges'])
        
        with col3:
            st.metric("Avg Degree", f"{stats['avg_degree']:.2f}")
        
        with col4:
            connected = "Yes" if stats['is_connected'] else "No"
            st.metric("Connected Graph", connected)
        
        st.divider()
        
        # Sample queries
        st.subheader("🎯 Try These Sample Queries:")
        
        sample_queries = [
            "Neural architecture search papers by Stanford researchers",
            "Graph neural networks and knowledge graphs",
            "Transformer models for computer vision",
            "Reinforcement learning research at Berkeley",
            "AutoML papers that cite reinforcement learning"
        ]
        
        for query in sample_queries:
            if st.button(f"🔍 {query}", key=query):
                st.info(f"Copy this query: **{query}**")
        
    except Exception as e:
        st.error(f"Error loading statistics: {e}")

# ============================================================================
# TAB 4: ADD DATA
# ============================================================================
with tabs[3]:
    st.header("➕ Add New Data")
    
    st.subheader("Create New Node")
    
    node_text = st.text_area("Node Text:", placeholder="Enter the main content...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        node_type = st.selectbox("Type:", ["paper", "author", "institution", "topic", "other"])
    
    with col2:
        custom_metadata = st.text_input("Additional metadata (JSON):", placeholder='{"key": "value"}')
    
    if st.button("Create Node", type="primary"):
        if node_text:
            try:
                metadata = {"type": node_type}
                if custom_metadata:
                    metadata.update(json.loads(custom_metadata))
                
                response = requests.post(f"{API_URL}/nodes", json={
                    "text": node_text,
                    "metadata": metadata
                })
                
                result = response.json()
                st.success(f"✅ Node created with ID: {result['id']}")
                st.json(result)
            
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter node text!")
    
    st.divider()
    
    st.subheader("Create Relationship")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_id = st.text_input("Source Node ID:")
    
    with col2:
        target_id = st.text_input("Target Node ID:")
    
    col3, col4 = st.columns(2)
    
    with col3:
        rel_type = st.selectbox("Relationship Type:", 
                                ["authored_by", "affiliated_with", "cites", "belongs_to_topic", "related_to", "collaborates_with"])
    
    with col4:
        weight = st.slider("Weight:", 0.0, 1.0, 1.0, 0.1)
    
    if st.button("Create Relationship", type="primary"):
        if source_id and target_id:
            try:
                response = requests.post(f"{API_URL}/edges", json={
                    "source_id": source_id,
                    "target_id": target_id,
                    "relationship_type": rel_type,
                    "weight": weight
                })
                
                result = response.json()
                st.success(f"✅ Relationship created!")
                st.json(result)
            
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter both node IDs!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🚀 Built for DevForge Hackathon | Vector + Graph Hybrid Database</p>
    <p>Combining the power of semantic search and relationship reasoning</p>
</div>
""", unsafe_allow_html=True)