"""Streamlit app for GraphMind - Hybrid Vector + Graph Database."""

import asyncio
import logging
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.config import settings
from backend.storage import GraphMindStorage
from backend.ingestion import IngestionPipeline
from backend.llm_processor import get_llm_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="GraphMind - Hybrid RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "storage" not in st.session_state:
    try:
        st.session_state.storage = GraphMindStorage(settings.CHROMA_DIR, settings.EMBEDDING_MODEL)
        st.session_state.ingestion = IngestionPipeline(st.session_state.storage)
        api_key = settings.GEMINI_API_KEY if settings.LLM_PROVIDER == "gemini" else settings.CLAUDE_API_KEY
        st.session_state.llm = get_llm_processor(settings.LLM_PROVIDER, api_key)
        st.session_state.initialized = True
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        st.session_state.initialized = False

# Sidebar
with st.sidebar:
    st.title("🧠 GraphMind")
    st.markdown("**Hybrid Vector + Graph Database**")
    
    if st.session_state.get("initialized", False):
        stats = st.session_state.storage.get_stats()
        st.metric("Nodes", int(stats.get("node_count", 0)))
        st.metric("Edges", int(stats.get("edge_count", 0)))
        st.metric("Vectors", int(stats.get("vector_count", 0)))
    
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Select Page",
        ["🔍 Search", "📤 Upload", "🕸️ Graph", "📊 Stats", "⚙️ Settings"],
        label_visibility="collapsed"
    )

# Main content
if not st.session_state.get("initialized", False):
    st.error("❌ Application failed to initialize. Please check the logs.")
    st.stop()

# Search Page
if page == "🔍 Search":
    st.header("🔍 Search Knowledge Graph")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input("Enter your query", placeholder="e.g., What is machine learning?")
        search_mode = st.selectbox("Search Mode", ["hybrid", "vector", "graph"], index=0)
    
    with col2:
        top_k = st.slider("Top K Results", 1, 20, 5)
        if search_mode == "hybrid":
            alpha = st.slider("Vector Weight (α)", 0.0, 1.0, 0.6, 0.1)
    
    if st.button("🔎 Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                try:
                    storage = st.session_state.storage
                    
                    if search_mode == "vector":
                        results = storage.vector_search(query, top_k)
                    elif search_mode == "graph":
                        results = storage.graph_search(query, top_k)
                    else:  # hybrid
                        results = storage.hybrid_search(query, top_k, alpha)
                    
                    if results:
                        st.success(f"Found {len(results)} results")
                        
                        # Display results
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Result {i}: {result.get('node_id', 'N/A')[:50]}... (Score: {result.get('score', 0):.3f})"):
                                st.write("**Content:**")
                                st.write(result.get("content", ""))
                                st.write("**Metadata:**")
                                st.json(result.get("metadata", {}))
                                st.write("**Score:**", result.get("score", 0))
                        
                        # Generate LLM answer
                        if st.session_state.llm:
                            with st.spinner("Generating AI answer..."):
                                try:
                                    formatted_results = [
                                        {
                                            "node_id": r.get("node_id", ""),
                                            "content": r.get("content", ""),
                                            "score": r.get("score", 0.0),
                                            "metadata": r.get("metadata", {})
                                        }
                                        for r in results
                                    ]
                                    answer = asyncio.run(
                                        st.session_state.llm.refine_results(formatted_results, query)
                                    )
                                    st.markdown("### 🤖 AI-Generated Answer")
                                    st.write(answer)
                                except Exception as e:
                                    st.warning(f"Could not generate AI answer: {str(e)}")
                    else:
                        st.info("No results found. Try uploading some documents first.")
                        
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
        else:
            st.warning("Please enter a query")

    st.divider()
    st.subheader("🧠 Node Intelligence")
    node_col1, node_col2 = st.columns([2, 1])
    with node_col1:
        node_lookup_id = st.text_input("Node ID", key="node_lookup_id", placeholder="e.g., doc1")
    with node_col2:
        lookup_button = st.button("📌 View Node Details")
    
    if lookup_button and node_lookup_id:
        try:
            storage = st.session_state.storage
            node_data = storage.get_node(node_lookup_id.strip())
            
            st.markdown(f"### Node `{node_data['node_id']}`")
            st.markdown("**Content**")
            st.info(node_data.get("content", "No content available"))
            
            st.markdown("**Metadata**")
            st.json(node_data.get("metadata", {}))
            
            relationships = node_data.get("relationships", [])
            st.markdown(f"**Relationships ({len(relationships)})**")
            if relationships:
                rel_df = pd.DataFrame(relationships)
                st.dataframe(rel_df, use_container_width=True, hide_index=True)
            else:
                st.write("No relationships found.")
            
            if st.session_state.llm:
                try:
                    with st.spinner("Generating Gemini summary..."):
                        formatted_result = [{
                            "node_id": node_data["node_id"],
                            "content": node_data.get("content", ""),
                            "score": 1.0,
                            "metadata": node_data.get("metadata", {})
                        }]
                        summary_prompt = f"Provide a concise summary for node {node_data['node_id']}."
                        summary = asyncio.run(
                            st.session_state.llm.refine_results(formatted_result, summary_prompt)
                        )
                        st.markdown("#### 🤖 Gemini Summary")
                        st.write(summary)
                except Exception as e:
                    st.warning(f"Could not generate summary: {str(e)}")
        except Exception as e:
            st.error(f"Failed to load node: {str(e)}")
    elif lookup_button:
        st.warning("Please enter a node ID.")

    st.divider()
    st.subheader("⚡ Auto-Relate Nodes")
    auto_col1, auto_col2 = st.columns([2, 1])
    with auto_col1:
        auto_node_id = st.text_input("Node ID to auto-relate", key="auto_node_id", placeholder="e.g., doc1")
    with auto_col2:
        auto_relationship = st.text_input("Relationship label", value="auto_related")
    
    auto_col3, auto_col4, auto_col5 = st.columns(3)
    with auto_col3:
        auto_top_k = st.slider("Top K Similar Nodes", 1, 20, 3)
    with auto_col4:
        auto_min_score = st.slider("Minimum Cosine Similarity", 0.0, 1.0, 0.6, 0.05)
    with auto_col5:
        auto_bidirectional = st.checkbox("Create reverse edges", value=False)
    
    if st.button("⚡ Auto Relate"):
        if not auto_node_id:
            st.warning("Please provide a node ID to auto-relate.")
        else:
            try:
                storage = st.session_state.storage
                result = storage.auto_relate(
                    auto_node_id.strip(),
                    top_k=auto_top_k,
                    min_score=auto_min_score,
                    relationship=auto_relationship or "auto_related",
                    bidirectional=auto_bidirectional,
                )
                created_edges = result.get("created_edges", [])
                if created_edges:
                    st.success(f"Created {len(created_edges)} new relationships for {result['node_id']}.")
                    st.dataframe(pd.DataFrame(created_edges), use_container_width=True, hide_index=True)
                else:
                    st.info(
                        f"No edges created. Try lowering the similarity threshold "
                        f"(current min score: {auto_min_score})."
                    )
            except Exception as e:
                st.error(f"Auto-relate failed: {str(e)}")

# Upload Page
elif page == "📤 Upload":
    st.header("📤 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "pdf", "xml", "json", "csv", "md"]
    )
    
    if uploaded_file:
        file_type = st.selectbox("File Type", ["text", "pdf", "xml", "json", "csv"])
        metadata_text = st.text_area("Metadata (JSON format)", value="{}")
        
        if st.button("📥 Ingest File", type="primary"):
            with st.spinner("Processing file..."):
                try:
                    import json
                    from io import BytesIO
                    
                    # Parse metadata
                    try:
                        metadata = json.loads(metadata_text) if metadata_text else {}
                    except:
                        metadata = {}
                    
                    # Save uploaded file temporarily
                    file_content = uploaded_file.read()
                    file_size = len(file_content)
                    
                    if file_size > settings.MAX_FILE_SIZE:
                        st.error(f"File too large. Max size: {settings.MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
                    else:
                        # Save file to upload directory
                        import tempfile
                        import os
                        
                        # Create a temporary file
                        temp_path = Path(tempfile.gettempdir()) / uploaded_file.name
                        with open(temp_path, "wb") as f:
                            f.write(file_content)
                        
                        try:
                            # Ingest using file path directly
                            result = asyncio.run(
                                st.session_state.ingestion.ingest_file(
                                    temp_path, file_type, metadata
                                )
                            )
                        finally:
                            # Cleanup temp file
                            if temp_path.exists():
                                os.remove(temp_path)
                        
                        if result.get("status") == "success":
                            st.success(f"✅ File ingested successfully!")
                            st.json(result)
                        else:
                            st.error(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}")
                            
                except Exception as e:
                    st.error(f"Ingestion failed: {str(e)}")
                    logger.exception("Upload error")

# Graph Page
elif page == "🕸️ Graph":
    st.header("🕸️ Knowledge Graph Visualization")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        max_nodes = st.slider("Max Nodes to Display", 10, 200, 50)
        node_type_filter = st.text_input("Filter by Node Type (optional)", "")
    
    with col2:
        if st.button("🔄 Refresh Graph"):
            st.rerun()
    
    try:
        storage = st.session_state.storage
        graph = storage.graph
        
        if graph.number_of_nodes() == 0:
            st.info("No nodes in graph. Upload some documents first.")
        else:
            # Get graph data
            nodes_data = []
            edges_data = []
            
            for node_id, attrs in graph.nodes(data=True):
                if node_type_filter and attrs.get("node_type") != node_type_filter:
                    continue
                nodes_data.append({
                    "id": str(node_id),
                    "label": str(node_id)[:30],
                    "content": attrs.get("content", "")[:50],
                    "node_type": attrs.get("node_type", "general")
                })
            
            for src, dst, attrs in graph.edges(data=True):
                if node_type_filter:
                    if graph.nodes[src].get("node_type") != node_type_filter or \
                       graph.nodes[dst].get("node_type") != node_type_filter:
                        continue
                edges_data.append({
                    "source": str(src),
                    "target": str(dst),
                    "relationship": attrs.get("relationship", ""),
                    "weight": attrs.get("weight", 1.0)
                })
            
            # Limit nodes
            if len(nodes_data) > max_nodes:
                st.warning(f"Showing {max_nodes} of {len(nodes_data)} nodes")
                nodes_data = nodes_data[:max_nodes]
                node_ids = {n["id"] for n in nodes_data}
                edges_data = [e for e in edges_data if e["source"] in node_ids and e["target"] in node_ids]
            
            # Create NetworkX graph for visualization
            G = nx.DiGraph()
            for node in nodes_data:
                G.add_node(node["id"], **node)
            for edge in edges_data:
                G.add_edge(edge["source"], edge["target"], **edge)
            
            # Use Plotly for visualization
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_info = G.nodes[node]
                node_text.append(f"{node_info.get('id', node)}<br>{node_info.get('content', '')[:50]}")
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[G.nodes[node].get('id', node)[:10] for node in G.nodes()],
                textposition="middle center",
                hovertext=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    reversescale=True,
                    color=[],
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title="Node Connections",
                        xanchor="left",
                        titleside="right"
                    ),
                    line=dict(width=2)
                )
            )
            
            # Color nodes by degree
            node_adjacencies = []
            for node in G.nodes():
                node_adjacencies.append(len(list(G.neighbors(node))))
            node_trace.marker.color = node_adjacencies
            
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Knowledge Graph',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        text="Interactive graph - hover over nodes for details",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor="left", yanchor="bottom",
                        font=dict(color="#888", size=12)
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display graph stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Nodes", len(nodes_data))
            with col2:
                st.metric("Total Edges", len(edges_data))
            with col3:
                if G.number_of_nodes() > 0:
                    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
                    st.metric("Avg Connections", f"{avg_degree:.1f}")
            
    except Exception as e:
        st.error(f"Error displaying graph: {str(e)}")
        logger.exception("Graph visualization error")

# Stats Page
elif page == "📊 Stats":
    st.header("📊 Database Statistics")
    
    try:
        storage = st.session_state.storage
        stats = storage.get_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", int(stats.get("node_count", 0)))
        with col2:
            st.metric("Edges", int(stats.get("edge_count", 0)))
        with col3:
            st.metric("Vectors", int(stats.get("vector_count", 0)))
        with col4:
            st.metric("Graph Density", f"{stats.get('graph_density', 0):.4f}")
        
        # Additional stats
        st.markdown("### Detailed Statistics")
        stats_df = pd.DataFrame([stats])
        st.dataframe(stats_df.T, use_container_width=True)
        
        # Node type distribution
        if storage.graph.number_of_nodes() > 0:
            node_types = {}
            for node_id, attrs in storage.graph.nodes(data=True):
                node_type = attrs.get("node_type", "general")
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            if node_types:
                st.markdown("### Node Type Distribution")
                types_df = pd.DataFrame(list(node_types.items()), columns=["Node Type", "Count"])
                st.bar_chart(types_df.set_index("Node Type"))
        
    except Exception as e:
        st.error(f"Error loading stats: {str(e)}")

# Settings Page
elif page == "⚙️ Settings":
    st.header("⚙️ Settings")
    
    st.markdown("### Configuration")
    st.json({
        "LLM Provider": settings.LLM_PROVIDER,
        "Embedding Model": settings.EMBEDDING_MODEL,
        "Environment": settings.ENVIRONMENT,
        "Max File Size": f"{settings.MAX_FILE_SIZE / 1024 / 1024:.1f} MB",
        "Chunk Size": settings.CHUNK_SIZE,
        "Default Top K": settings.DEFAULT_TOP_K,
        "Hybrid Alpha": settings.HYBRID_ALPHA
    })
    
    st.markdown("### Data Management")
    
    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.checkbox("I understand this will delete all data"):
            try:
                storage = st.session_state.storage
                node_count = storage.graph.number_of_nodes()
                edge_count = storage.graph.number_of_edges()
                
                storage.graph.clear()
                # Clear ChromaDB
                try:
                    all_ids = storage.collection.get()["ids"]
                    if all_ids:
                        storage.collection.delete(ids=all_ids)
                except:
                    pass
                
                storage.vector_to_graph_map.clear()
                
                st.success(f"Cleared {node_count} nodes, {edge_count} edges")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear data: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "🧠 GraphMind - Hybrid Vector + Graph Database | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)

