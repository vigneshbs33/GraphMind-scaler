// GraphMind Frontend JavaScript
const API_BASE = 'http://localhost:8000';

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    initOnboarding();
    initFileUpload();
});

// Tab switching
function switchTab(tabName) {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    
    document.querySelector(`[data-tab="${tabName}"]`)?.classList.add('active');
    const tabElement = document.getElementById(tabName + '-tab');
    if (tabElement) {
        tabElement.classList.add('active');
    }
    
    // If nodes tab is opened, show helpful message if empty (only once, no recursion)
    if (tabName === 'nodes') {
        const container = document.getElementById('node-details-container');
        if (container) {
            const currentContent = container.innerHTML.trim();
            // Only show default message if truly empty (no content at all)
            if (!currentContent || currentContent === '') {
                container.innerHTML = `
                    <div class="card" style="margin-top: 20px;">
                        <h3>🔍 Node Lookup</h3>
                        <p style="color: var(--text-secondary); margin-top: 15px;">
                            Enter a node ID above to view its details, relationships, and AI-generated summary.
                        </p>
                        <p style="color: var(--text-muted); margin-top: 10px; font-size: 0.9rem;">
                            💡 Tip: Click on any search result to automatically view that node here.
                        </p>
                    </div>
                `;
            }
        }
    }
}

// Landing page functions
function startApp() {
    document.getElementById('landing-page').style.display = 'none';
    document.getElementById('app-container').classList.add('active');
    loadStats();
}

async function clearAllData() {
    if (!confirm('Are you sure you want to clear ALL data? This cannot be undone!')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/data/clear`, { method: 'DELETE' });
        const data = await response.json();
        alert(`Cleared: ${data.nodes_deleted} nodes, ${data.edges_deleted} edges, ${data.vectors_deleted} vectors`);
        loadStats();
    } catch (error) {
        alert('Error clearing data: ' + error.message);
    }
}

// Stats loading
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const stats = await response.json();
        const nodeCount = Math.floor(stats.node_count || 0);
        const edgeCount = Math.floor(stats.edge_count || 0);
        const vectorCount = Math.floor(stats.vector_count || 0);
        
        // Update landing page stats
        const landingNodes = document.getElementById('landing-stat-nodes');
        const landingEdges = document.getElementById('landing-stat-edges');
        const landingVectors = document.getElementById('landing-stat-vectors');
        
        if (landingNodes) landingNodes.textContent = nodeCount;
        if (landingEdges) landingEdges.textContent = edgeCount;
        if (landingVectors) landingVectors.textContent = vectorCount;
        
        // Update app stats
        const statNodes = document.getElementById('stat-nodes');
        const statEdges = document.getElementById('stat-edges');
        const statVectors = document.getElementById('stat-vectors');
        
        if (statNodes) statNodes.textContent = nodeCount;
        if (statEdges) statEdges.textContent = edgeCount;
        if (statVectors) statVectors.textContent = vectorCount;
        
        // Update files count
        try {
            const filesResponse = await fetch(`${API_BASE}/files`);
            const filesData = await filesResponse.json();
            const filesCount = document.getElementById('landing-stat-files');
            if (filesCount) filesCount.textContent = filesData.count || 0;
        } catch (e) {
            console.error('Failed to load files count:', e);
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// File upload
function initFileUpload() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    
    if (!uploadZone || !fileInput) return;
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    });
}

async function uploadFile(file) {
    const statusDiv = document.getElementById('upload-status');
    const processingDiv = document.getElementById('processing-animation');
    const transformDiv = document.getElementById('transformation-visualization');
    const unstructuredPreview = document.getElementById('unstructured-preview');
    
    if (!statusDiv || !processingDiv) return;
    
    statusDiv.innerHTML = '';
    processingDiv.classList.add('active');
    if (transformDiv) transformDiv.style.display = 'none';
    
    // Read and preview file content
    let fileContent = '';
    try {
        fileContent = await file.text();
        if (unstructuredPreview) {
            const truncated = fileContent.substring(0, 2000);
            const isTruncated = fileContent.length > 2000;
            unstructuredPreview.innerHTML = `
                <pre style="white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; word-break: break-word; max-width: 100%; width: 100%; margin: 0; padding: 0; box-sizing: border-box; overflow-x: hidden;">${truncated}${isTruncated ? '\n\n... (truncated)' : ''}</pre>
            `;
        }
    } catch (e) {
        if (unstructuredPreview) {
            unstructuredPreview.innerHTML = '<p style="color: var(--text-secondary);">Unable to preview file content</p>';
        }
    }
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('file_type', document.getElementById('file-type')?.value || 'text');
    formData.append('metadata', '{}');
    
    try {
        const response = await fetch(`${API_BASE}/ingest`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        processingDiv.classList.remove('active');
        
        if (result.status === 'success' || result.status === 'ok') {
            statusDiv.innerHTML = `<div class="message success">
                ✅ Upload successful!<br>
                Nodes created: ${result.nodes_created || 0}<br>
                Edges created: ${result.edges_created || 0}
            </div>`;
            
            // Show transformation visualization
            await showTransformationVisualization(result);
            if (transformDiv) transformDiv.style.display = 'block';
            
            loadStats();
        } else {
            statusDiv.innerHTML = `<div class="message error">❌ Error: ${result.error || result.detail || 'Unknown error'}</div>`;
        }
    } catch (error) {
        processingDiv.classList.remove('active');
        statusDiv.innerHTML = `<div class="message error">❌ Upload failed: ${error.message}</div>`;
    }
}

async function showTransformationVisualization(result) {
    try {
        const graphResponse = await fetch(`${API_BASE}/graph?limit=50`);
        const graphData = await graphResponse.json();
        
        // Extract entities (nodes with entity_type metadata)
        const entities = (graphData.nodes || []).filter(n => n.metadata?.entity_type === 'extracted').slice(0, 20);
        const nodes = (graphData.nodes || []).slice(0, 20);
        const edges = (graphData.edges || []).slice(0, 20);
        
        // Display entities
        const entitiesList = document.getElementById('entities-list');
        if (entitiesList) {
            if (entities.length > 0) {
                entitiesList.innerHTML = entities.map(e => `
                    <div class="data-item">
                        <strong>${e.content || e.node_id}</strong>
                        <div class="content">Type: ${e.metadata?.entity_type || 'extracted'}</div>
                    </div>
                `).join('');
            } else {
                entitiesList.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 20px;">No entities extracted</p>';
            }
        }
        
        // Display nodes
        const nodesList = document.getElementById('nodes-list');
        if (nodesList) {
            if (nodes.length > 0) {
                nodesList.innerHTML = nodes.map(n => `
                    <div class="data-item">
                        <strong>${n.node_id}</strong>
                        <div class="content">${(n.content || '').substring(0, 100)}${(n.content || '').length > 100 ? '...' : ''}</div>
                    </div>
                `).join('');
            } else {
                nodesList.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 20px;">No nodes created</p>';
            }
        }
        
        // Display edges
        const edgesList = document.getElementById('edges-list');
        if (edgesList) {
            if (edges.length > 0) {
                edgesList.innerHTML = edges.map(e => `
                    <div class="data-item">
                        <strong>${e.source} → ${e.target}</strong>
                        <div class="content">Relationship: ${e.relationship || 'related_to'}</div>
                    </div>
                `).join('');
            } else {
                edgesList.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 20px;">No edges created</p>';
            }
        }
        
        // Display embeddings info
        const embeddingsList = document.getElementById('embeddings-list');
        if (embeddingsList) {
            const embeddingCount = result.nodes_created || graphData.nodes?.length || 0;
            embeddingsList.innerHTML = `
                <div class="data-item">
                    <strong>Vector Embeddings Generated</strong>
                    <div class="content">Total: ${embeddingCount} embeddings<br>Dimension: 384<br>Storage: ChromaDB</div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Failed to load transformation data:', error);
    }
}

// Search functions
function switchSearchMode(mode) {
    document.querySelectorAll('#search-tab-vector, #search-tab-graph, #search-tab-hybrid').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.search-mode-content').forEach(c => c.style.display = 'none');
    
    const tabElement = document.getElementById('search-tab-' + mode);
    const contentElement = document.getElementById('search-' + mode);
    
    if (tabElement) tabElement.classList.add('active');
    if (contentElement) contentElement.style.display = 'block';
}

async function performVectorSearch() {
    const query = document.getElementById('vector-query')?.value;
    const topK = parseInt(document.getElementById('vector-top-k')?.value || '5');
    const resultsDiv = document.getElementById('search-results');
    
    if (!query || !query.trim()) {
        alert('Please enter a search query');
        return;
    }
    
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Searching...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/search/vector`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ query_text: query, top_k: topK })
        });
        
        const data = await response.json();
        displaySearchResults(data.results || []);
    } catch (error) {
        resultsDiv.innerHTML = `<div class="message error">❌ Search failed: ${error.message}</div>`;
    }
}

async function performHybridSearch() {
    const query = document.getElementById('hybrid-query')?.value;
    const vectorWeight = parseFloat(document.getElementById('hybrid-vector-weight')?.value || '0.6');
    const graphWeight = parseFloat(document.getElementById('hybrid-graph-weight')?.value || '0.4');
    const topK = parseInt(document.getElementById('hybrid-top-k')?.value || '5');
    const resultsDiv = document.getElementById('search-results');
    const answerDiv = document.getElementById('llm-answer-container');
    
    if (!query || !query.trim()) {
        alert('Please enter a search query');
        return;
    }
    
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Performing hybrid search...</div>';
    if (answerDiv) answerDiv.innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/search/hybrid`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query_text: query,
                vector_weight: vectorWeight,
                graph_weight: graphWeight,
                top_k: topK
            })
        });
        
        const data = await response.json();
        
        if (data.results && data.results.length > 0) {
            displaySearchResults(data.results);
            await getLLMExplanation(query, data.results);
        } else {
            resultsDiv.innerHTML = '<div class="message error">No results found</div>';
        }
    } catch (error) {
        resultsDiv.innerHTML = `<div class="message error">❌ Hybrid search failed: ${error.message}</div>`;
    }
}

async function performGraphTraversal() {
    const startId = document.getElementById('traversal-start-id')?.value;
    const depth = parseInt(document.getElementById('traversal-depth')?.value || '3');
    const maxNodes = parseInt(document.getElementById('traversal-max-nodes')?.value || '100');
    const resultsDiv = document.getElementById('search-results');
    
    if (!startId || !startId.trim()) {
        alert('Please enter a start node ID');
        return;
    }
    
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Traversing graph...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/search/graph?start_id=${encodeURIComponent(startId)}&depth=${depth}&max_nodes=${maxNodes}`);
        const data = await response.json();
        
        if (data.nodes && data.nodes.length > 0) {
            displaySearchResults(data.nodes.map(n => ({
                node_id: n.node_id,
                content: n.content,
                score: 1 / (n.distance + 1),
                metadata: n.metadata || {}
            })));
        } else {
            resultsDiv.innerHTML = '<div class="message error">No nodes found</div>';
        }
    } catch (error) {
        resultsDiv.innerHTML = `<div class="message error">❌ Traversal failed: ${error.message}</div>`;
    }
}

function displaySearchResults(results) {
    const resultsDiv = document.getElementById('search-results');
    if (!resultsDiv) return;
    
    if (!results || results.length === 0) {
        resultsDiv.innerHTML = '<div class="message error">No results found</div>';
        return;
    }
    
    resultsDiv.innerHTML = results.map(result => {
        const score = result.score || result.vector_score || result.graph_score || 0;
        const scoreText = typeof score === 'number' ? score.toFixed(3) : 'N/A';
        
        return `
            <div class="result-card" onclick="viewNodeDetails('${result.node_id}')" style="cursor: pointer;">
                <h3>${result.node_id}</h3>
                <div class="score">Score: ${scoreText}</div>
                <div class="content">${(result.content || '').substring(0, 200)}${(result.content || '').length > 200 ? '...' : ''}</div>
            </div>
        `;
    }).join('');
}

async function getLLMExplanation(query, results) {
    const answerDiv = document.getElementById('llm-answer-container');
    if (!answerDiv) return;
    
    answerDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Generating answer...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/search/hybrid/explain`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: query,
                results: results.map(r => ({
                    node_id: r.node_id,
                    content: r.content,
                    score: r.score || r.vector_score || r.graph_score || 0,
                    metadata: r.metadata || {}
                }))
            })
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        
        if (data.answer) {
            const html = marked.parse(data.answer);
            answerDiv.innerHTML = `
                <div class="card" style="margin-top: 30px;">
                    <h2>AI Answer</h2>
                    <div class="markdown-content">${html}</div>
                </div>
            `;
        }
    } catch (error) {
        answerDiv.innerHTML = `<div class="message error">Failed to generate answer: ${error.message}</div>`;
    }
}

async function viewNodeDetails(nodeId) {
    if (!nodeId) {
        console.error('viewNodeDetails called without nodeId');
        return;
    }
    
    // Switch tab first
    switchTab('nodes');
    
    // Wait for DOM to update
    await new Promise(resolve => setTimeout(resolve, 200));
    
    const nodeInput = document.getElementById('node-lookup-id');
    const container = document.getElementById('node-details-container');
    
    if (!container) {
        console.error('node-details-container not found');
        return;
    }
    
    if (nodeInput) {
        nodeInput.value = nodeId;
    }
    
    // Call lookupNode directly with the nodeId to avoid recursion
    await lookupNodeDirect(nodeId);
}

// Comparison
async function performCompare() {
    const query = document.getElementById('compare-query')?.value;
    const resultsDiv = document.getElementById('comparison-results');
    
    if (!query || !query.trim()) {
        alert('Please enter a search query');
        return;
    }
    
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Comparing all search methods in parallel...</div>';
    
    try {
        // Run all 3 searches in parallel
        const [vectorResponse, graphResponse, hybridResponse] = await Promise.all([
            fetch(`${API_BASE}/search/vector`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ query_text: query, top_k: 5 })
            }),
            fetch(`${API_BASE}/search`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ query: query, mode: 'graph', top_k: 5 })
            }),
            fetch(`${API_BASE}/search/hybrid`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ query_text: query, vector_weight: 0.6, graph_weight: 0.4, top_k: 5 })
            })
        ]);

        const vectorData = await vectorResponse.json();
        const graphData = await graphResponse.json();
        const hybridData = await hybridResponse.json();

        // Get comparison metrics
        const compareResponse = await fetch(`${API_BASE}/compare`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: query,
                top_k: 5,
                alpha: 0.6,
                ground_truth_ids: []
            })
        });

        const compareData = await compareResponse.json();
        
        // Combine results
        const combinedData = {
            ...compareData,
            vector_results: vectorData.results || [],
            graph_results: graphData.results || [],
            hybrid_results: hybridData.results || []
        };
        
        displayComparison(combinedData);
    } catch (error) {
        resultsDiv.innerHTML = `<div class="message error">❌ Comparison failed: ${error.message}</div>`;
    }
}

function displayComparison(data) {
    const resultsDiv = document.getElementById('comparison-results');
    if (!resultsDiv) return;
    
    const vectorResults = data.vector_results || [];
    const graphResults = data.graph_results || [];
    const hybridResults = data.hybrid_results || [];
    
    // Calculate metrics
    const vectorIds = new Set(vectorResults.map(r => r.node_id || r.id));
    const graphIds = new Set(graphResults.map(r => r.node_id || r.id));
    const hybridIds = new Set(hybridResults.map(r => r.node_id || r.id));
    
    // Calculate overlap scores
    const vectorOverlap = data.metrics?.vector_precision || (vectorIds.size > 0 ? (Array.from(vectorIds).filter(id => hybridIds.has(id)).length / vectorIds.size) : 0);
    const graphOverlap = data.metrics?.graph_precision || (graphIds.size > 0 ? (Array.from(graphIds).filter(id => hybridIds.has(id)).length / graphIds.size) : 0);
    const hybridOverlap = data.metrics?.hybrid_precision || 1.0;
    
    // Calculate average scores
    const vectorAvgScore = vectorResults.length > 0 ? (vectorResults.reduce((sum, r) => sum + (r.score || r.vector_score || 0), 0) / vectorResults.length) : 0;
    const graphAvgScore = graphResults.length > 0 ? (graphResults.reduce((sum, r) => sum + (r.score || r.graph_score || 0), 0) / graphResults.length) : 0;
    const hybridAvgScore = hybridResults.length > 0 ? (hybridResults.reduce((sum, r) => sum + (r.score || 0), 0) / hybridResults.length) : 0;
    
    // Determine winner
    const scores = {
        vector: (vectorOverlap * 0.4) + (vectorAvgScore * 0.6),
        graph: (graphOverlap * 0.4) + (graphAvgScore * 0.6),
        hybrid: (hybridOverlap * 0.4) + (hybridAvgScore * 0.6)
    };
    
    const winner = data.winner || Object.keys(scores).reduce((a, b) => scores[a] > scores[b] ? a : b);
    const winnerScore = scores[winner];
    
    resultsDiv.innerHTML = `
        <div class="card" style="margin-bottom: 30px;">
            <h3>📊 Performance Analysis</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;">
                <div style="background: ${winner === 'vector' ? 'rgba(59, 130, 246, 0.1)' : 'var(--bg-glass)'}; border: 1px solid ${winner === 'vector' ? 'rgba(59, 130, 246, 0.3)' : 'var(--border-glass)'}; border-radius: 12px; padding: 20px;">
                    <h4 style="margin-bottom: 15px;">🔍 Vector Search</h4>
                    <div style="font-size: 2rem; font-weight: 700; margin-bottom: 5px;">${(scores.vector * 100).toFixed(1)}%</div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 10px;">Overall Score</div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">
                        <div>Precision: ${(vectorOverlap * 100).toFixed(1)}%</div>
                        <div>Avg Score: ${vectorAvgScore.toFixed(3)}</div>
                        <div>Results: ${vectorResults.length}</div>
                    </div>
                    ${winner === 'vector' ? '<div style="margin-top: 10px; color: var(--accent-blue); font-weight: 600;">🏆 Winner</div>' : ''}
                </div>
                
                <div style="background: ${winner === 'graph' ? 'rgba(59, 130, 246, 0.1)' : 'var(--bg-glass)'}; border: 1px solid ${winner === 'graph' ? 'rgba(59, 130, 246, 0.3)' : 'var(--border-glass)'}; border-radius: 12px; padding: 20px;">
                    <h4 style="margin-bottom: 15px;">🕸️ Graph Search</h4>
                    <div style="font-size: 2rem; font-weight: 700; margin-bottom: 5px;">${(scores.graph * 100).toFixed(1)}%</div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 10px;">Overall Score</div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">
                        <div>Precision: ${(graphOverlap * 100).toFixed(1)}%</div>
                        <div>Avg Score: ${graphAvgScore.toFixed(3)}</div>
                        <div>Results: ${graphResults.length}</div>
                    </div>
                    ${winner === 'graph' ? '<div style="margin-top: 10px; color: var(--accent-blue); font-weight: 600;">🏆 Winner</div>' : ''}
                </div>
                
                <div style="background: ${winner === 'hybrid' ? 'rgba(59, 130, 246, 0.1)' : 'var(--bg-glass)'}; border: 1px solid ${winner === 'hybrid' ? 'rgba(59, 130, 246, 0.3)' : 'var(--border-glass)'}; border-radius: 12px; padding: 20px;">
                    <h4 style="margin-bottom: 15px;">⚡ Hybrid Search</h4>
                    <div style="font-size: 2rem; font-weight: 700; margin-bottom: 5px;">${(scores.hybrid * 100).toFixed(1)}%</div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 10px;">Overall Score</div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">
                        <div>Precision: ${(hybridOverlap * 100).toFixed(1)}%</div>
                        <div>Avg Score: ${hybridAvgScore.toFixed(3)}</div>
                        <div>Results: ${hybridResults.length}</div>
                    </div>
                    ${winner === 'hybrid' ? '<div style="margin-top: 10px; color: var(--accent-blue); font-weight: 600;">🏆 Winner</div>' : ''}
                </div>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: var(--bg-glass); border-radius: 12px;">
                <strong>Scoring Formula:</strong> Overall Score = (Precision × 0.4) + (Average Relevance Score × 0.6)
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px;">
            <div class="card">
                <h3>Vector Results (${vectorResults.length})</h3>
                <div class="results-grid" style="grid-template-columns: 1fr;">
                    ${vectorResults.slice(0, 5).map((r, idx) => {
                        const score = r.score || r.vector_score || 0;
                        return `
                            <div class="result-card" onclick="viewNodeDetails('${r.node_id || r.id}')" style="cursor: pointer;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <h4 style="margin: 0;">#${idx + 1}</h4>
                                    <div class="score">${score.toFixed(3)}</div>
                                </div>
                                <div style="font-size: 0.9rem; color: var(--text-secondary);">${(r.content || '').substring(0, 120)}...</div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
            
            <div class="card">
                <h3>Graph Results (${graphResults.length})</h3>
                <div class="results-grid" style="grid-template-columns: 1fr;">
                    ${graphResults.slice(0, 5).map((r, idx) => {
                        const score = r.score || r.graph_score || 0;
                        return `
                            <div class="result-card" onclick="viewNodeDetails('${r.node_id || r.id}')" style="cursor: pointer;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <h4 style="margin: 0;">#${idx + 1}</h4>
                                    <div class="score">${score.toFixed(3)}</div>
                                </div>
                                <div style="font-size: 0.9rem; color: var(--text-secondary);">${(r.content || '').substring(0, 120)}...</div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
            
            <div class="card">
                <h3>Hybrid Results (${hybridResults.length})</h3>
                <div class="results-grid" style="grid-template-columns: 1fr;">
                    ${hybridResults.slice(0, 5).map((r, idx) => {
                        const score = r.score || 0;
                        return `
                            <div class="result-card" onclick="viewNodeDetails('${r.node_id || r.id}')" style="cursor: pointer;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <h4 style="margin: 0;">#${idx + 1}</h4>
                                    <div class="score">${score.toFixed(3)}</div>
                                </div>
                                <div style="font-size: 0.9rem; color: var(--text-secondary);">${(r.content || '').substring(0, 120)}...</div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        </div>
        
        ${data.llm_answer ? `
            <div class="card">
                <h2>AI Analysis</h2>
                <div class="markdown-content">${marked.parse(data.llm_answer)}</div>
            </div>
        ` : ''}
    `;
}

// Node lookup - called from button click
async function lookupNode() {
    const nodeId = document.getElementById('node-lookup-id')?.value;
    if (!nodeId || !nodeId.trim()) {
        const container = document.getElementById('node-details-container');
        if (container) {
            container.innerHTML = '<div class="message error" style="margin-top: 20px;">Please enter a node ID</div>';
        }
        return;
    }
    await lookupNodeDirect(nodeId.trim());
}

// Direct node lookup - no recursion, called internally
async function lookupNodeDirect(nodeId) {
    const container = document.getElementById('node-details-container');
    
    if (!container) {
        console.error('node-details-container not found');
        return;
    }
    
    // Clear and show loading
    container.innerHTML = '<div class="loading" style="margin-top: 20px;"><div class="spinner"></div>Loading node...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/nodes/${encodeURIComponent(nodeId)}`);
        
        if (!response.ok) {
            if (response.status === 404) {
                throw new Error(`Node "${nodeId}" not found. Please check the node ID.`);
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Fetch LLM summary (non-blocking, show node first)
        let summaryHtml = '';
        const summaryPromise = fetch(`${API_BASE}/nodes/${encodeURIComponent(nodeId)}/summary`)
            .then(summaryResponse => {
                if (summaryResponse.ok) {
                    return summaryResponse.json();
                }
                return null;
            })
            .then(summaryData => {
                if (summaryData && summaryData.summary) {
                    summaryHtml = `
                        <div style="margin: 20px 0; padding: 20px; background: var(--bg-glass); border-radius: 12px; border: 1px solid var(--border-glass);">
                            <h3 style="margin-bottom: 10px;">🤖 AI Summary</h3>
                            <div class="markdown-content">${marked.parse(summaryData.summary)}</div>
                        </div>
                    `;
                    // Update container with summary
                    const summaryContainer = container.querySelector('.node-summary-placeholder');
                    if (summaryContainer) {
                        summaryContainer.outerHTML = summaryHtml;
                    }
                }
            })
            .catch(err => {
                console.error('Failed to load summary:', err);
            });
        
        // Display node immediately
        container.innerHTML = `
            <div class="card" style="margin-top: 20px;">
                <h2>📌 Node: ${data.node_id}</h2>
                <div style="margin: 20px 0;">
                    <h3 style="margin-bottom: 10px;">Content</h3>
                    <div style="color: var(--text-secondary); padding: 15px; background: var(--bg-glass); border-radius: 8px; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; max-width: 100%;">${(data.content || 'No content').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>
                </div>
                <div style="margin: 20px 0;">
                    <h3 style="margin-bottom: 10px;">🔗 Relationships (${data.relationships?.length || 0})</h3>
                    ${data.relationships && data.relationships.length > 0 ? `
                        <div class="results-grid">
                            ${data.relationships.map(rel => `
                                <div class="result-card" onclick="viewNodeDetails('${rel.target_id}')" style="cursor: pointer;">
                                    <div class="score">Weight: ${(rel.weight || 0).toFixed(2)}</div>
                                    <div><strong>${rel.source_id}</strong> → <strong>${rel.target_id}</strong></div>
                                    <div style="color: var(--text-muted); font-size: 0.9em; margin-top: 8px;">Type: ${rel.relationship || 'related_to'}</div>
                                </div>
                            `).join('')}
                        </div>
                    ` : '<p style="color: var(--text-muted); padding: 20px; text-align: center;">No relationships found</p>'}
                </div>
                <div class="node-summary-placeholder"></div>
            </div>
        `;
        
        // Load summary in background
        await summaryPromise;
        if (summaryHtml) {
            const summaryPlaceholder = container.querySelector('.node-summary-placeholder');
            if (summaryPlaceholder) {
                summaryPlaceholder.outerHTML = summaryHtml;
            }
        } else {
            // Remove placeholder if no summary
            const summaryPlaceholder = container.querySelector('.node-summary-placeholder');
            if (summaryPlaceholder) {
                summaryPlaceholder.remove();
            }
        }
    } catch (error) {
        container.innerHTML = `<div class="message error" style="margin-top: 20px;">❌ Failed to load node: ${error.message}</div>`;
    }
}

// CRUD operations
async function createNode() {
    const nodeId = document.getElementById('create-node-id')?.value;
    const content = document.getElementById('create-node-content')?.value;
    const nodeType = document.getElementById('create-node-type')?.value;
    const statusDiv = document.getElementById('crud-status');
    
    if (!content || !content.trim()) {
        alert('Please enter node content');
        return;
    }
    
    if (!statusDiv) return;
    
    statusDiv.innerHTML = '<div class="loading">Creating node...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/nodes`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                id: nodeId || null,
                content: content,
                node_type: nodeType || 'general',
                metadata: {}
            })
        });
        
        const data = await response.json();
        statusDiv.innerHTML = `<div class="message success">✅ Node created: ${data.id}</div>`;
        loadStats();
    } catch (error) {
        statusDiv.innerHTML = `<div class="message error">❌ Failed to create node: ${error.message}</div>`;
    }
}

async function createEdge() {
    const sourceId = document.getElementById('create-edge-source')?.value;
    const targetId = document.getElementById('create-edge-target')?.value;
    const relationship = document.getElementById('create-edge-relationship')?.value;
    const weight = parseFloat(document.getElementById('create-edge-weight')?.value || '1.0');
    const statusDiv = document.getElementById('crud-status');
    
    if (!sourceId || !targetId || !relationship) {
        alert('Please fill in all edge fields');
        return;
    }
    
    if (!statusDiv) return;
    
    statusDiv.innerHTML = '<div class="loading">Creating edge...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/edges`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                source_id: sourceId,
                target_id: targetId,
                relationship: relationship,
                weight: weight
            })
        });
        
        const data = await response.json();
        statusDiv.innerHTML = `<div class="message success">✅ Edge created: ${sourceId} → ${targetId}</div>`;
        loadStats();
    } catch (error) {
        statusDiv.innerHTML = `<div class="message error">❌ Failed to create edge: ${error.message}</div>`;
    }
}

// Graph visualization
async function loadGraph() {
    const container = document.getElementById('graph-container');
    if (!container) return;
    
    container.innerHTML = '<div class="loading"><div class="spinner"></div>Loading graph...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/graph?limit=100`);
        const data = await response.json();
        
        if (!data.nodes || data.nodes.length === 0) {
            container.innerHTML = '<div class="loading">No nodes in graph. Upload some files first!</div>';
            return;
        }
        
        container.innerHTML = '';
        
        const elements = [];
        data.nodes.forEach(node => {
            const nodeId = String(node.node_id || '');
            if (!nodeId) return;
            
            let label = node.content || nodeId;
            if (label.length > 25) {
                label = label.substring(0, 22) + '...';
            }
            
            elements.push({
                data: {
                    id: nodeId,
                    label: label,
                    content: node.content || nodeId
                }
            });
        });
        
        if (data.edges) {
            data.edges.forEach(edge => {
                const source = String(edge.source || '');
                const target = String(edge.target || '');
                if (source && target) {
                    elements.push({
                        data: {
                            id: `${source}-${target}`,
                            source: source,
                            target: target
                        }
                    });
                }
            });
        }
        
        const cy = cytoscape({
            container: container,
            elements: elements,
            style: [
                {
                    selector: 'node',
                    style: {
                        'background-color': 'rgba(255, 255, 255, 0.9)',
                        'label': 'data(label)',
                        'width': '60',
                        'height': '60',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'color': '#000',
                        'font-size': '11px',
                        'text-wrap': 'wrap',
                        'text-max-width': '80'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 2,
                        'line-color': 'rgba(255, 255, 255, 0.4)',
                        'target-arrow-color': 'rgba(255, 255, 255, 0.4)',
                        'target-arrow-shape': 'triangle'
                    }
                }
            ],
            layout: {
                name: 'cose',
                idealEdgeLength: 120,
                fit: true,
                padding: 40,
                nodeDimensionsIncludeLabels: true,
                spacingFactor: 1.5
            }
        });
        
        cy.on('tap', 'node', function(evt) {
            const node = evt.target;
            viewNodeDetails(node.id());
        });
    } catch (error) {
        container.innerHTML = `<div class="message error">Failed to load graph: ${error.message}</div>`;
    }
}

// Traversal
async function performTraversal() {
    const startId = document.getElementById('traversal-start')?.value;
    const depth = parseInt(document.getElementById('traversal-depth-slider')?.value || '3');
    const maxNodes = parseInt(document.getElementById('traversal-max')?.value || '100');
    const resultsDiv = document.getElementById('traversal-results');
    
    if (!startId || !startId.trim()) {
        alert('Please enter a start node ID');
        return;
    }
    
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Traversing graph...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/search/graph?start_id=${encodeURIComponent(startId)}&depth=${depth}&max_nodes=${maxNodes}`);
        const data = await response.json();
        
        if (data.nodes && data.nodes.length > 0) {
            resultsDiv.innerHTML = `
                <h3 style="margin-bottom: 20px;">Traversal Results: ${data.nodes.length} nodes (Depth: ${data.depth})</h3>
                <div class="results-grid">
                    ${data.nodes.map(node => `
                        <div class="result-card" onclick="viewNodeDetails('${node.node_id}')" style="cursor: pointer;">
                            <h3>${node.node_id}</h3>
                            <div class="score">Distance: ${node.distance}</div>
                            <div class="content">${node.content.substring(0, 150)}...</div>
                        </div>
                    `).join('')}
                </div>
            `;
        } else {
            resultsDiv.innerHTML = '<div class="message error">No nodes found</div>';
        }
    } catch (error) {
        resultsDiv.innerHTML = `<div class="message error">❌ Traversal failed: ${error.message}</div>`;
    }
}

// Transformation tabs
function switchTransformationTab(tabName) {
    // Remove active from all tabs
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('#entities-tab, #nodes-tab, #edges-tab, #embeddings-tab').forEach(content => content.classList.remove('active'));
    
    // Add active to clicked button
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => {
        if (btn.textContent.toLowerCase().includes(tabName.toLowerCase())) {
            btn.classList.add('active');
        }
    });
    
    // Show corresponding content
    const tabContent = document.getElementById(tabName + '-tab');
    if (tabContent) {
        tabContent.classList.add('active');
    }
}

// Onboarding
function initOnboarding() {
    // Skip onboarding for now
}

function skipOnboarding() {
    document.getElementById('onboarding-overlay')?.classList.remove('active');
}

function nextOnboardingStep() {
    skipOnboarding();
}

// Autofill functions (placeholders)
function autofillUpload() {
    alert('Autofill demo - upload a file to test');
}

function autofillVectorSearch() {
    const input = document.getElementById('vector-query');
    if (input) input.value = 'machine learning';
}

function autofillHybridSearch() {
    const input = document.getElementById('hybrid-query');
    if (input) input.value = 'artificial intelligence';
}

function autofillGraphTraversal() {
    const input = document.getElementById('traversal-start-id');
    if (input) input.value = 'node_1';
}

function autofillCompare() {
    const input = document.getElementById('compare-query');
    if (input) input.value = 'data science';
}

function autofillCreateNode() {
    document.getElementById('create-node-content').value = 'Sample node content';
}

function autofillCreateEdge() {
    document.getElementById('create-edge-source').value = 'node_1';
    document.getElementById('create-edge-target').value = 'node_2';
    document.getElementById('create-edge-relationship').value = 'related_to';
}

function autofillNodeLookup() {
    const input = document.getElementById('node-lookup-id');
    if (input) input.value = 'node_1';
}

function autofillTraversal() {
    const input = document.getElementById('traversal-start');
    if (input) input.value = 'node_1';
}

