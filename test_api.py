"""
Comprehensive API Test Suite for GraphMind
Tests all specified test cases and reports results.
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List, Tuple
from math import sqrt

BASE_URL = "http://localhost:8000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class TestResult:
    def __init__(self, test_id: str, name: str, priority: str):
        self.test_id = test_id
        self.name = name
        self.priority = priority
        self.passed = False
        self.error = None
        self.details = {}

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}\n")

def print_test(test_id: str, name: str):
    print(f"{Colors.BOLD}[{test_id}]{Colors.RESET} {name}")

def print_pass(msg: str = ""):
    print(f"{Colors.GREEN}[PASS]{Colors.RESET} {msg}")

def print_fail(msg: str = ""):
    print(f"{Colors.RED}[FAIL]{Colors.RESET} {msg}")

def print_info(msg: str):
    print(f"{Colors.YELLOW}[INFO]{Colors.RESET} {msg}")

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sqrt(sum(a * a for a in vec1))
    magnitude2 = sqrt(sum(a * a for a in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def wait_for_server(max_retries=10):
    """Wait for server to be ready."""
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def clear_all_data():
    """Clear all data before testing."""
    try:
        response = requests.delete(f"{BASE_URL}/data/clear", timeout=10)
        return response.status_code in [200, 204]
    except Exception as e:
        print_info(f"Could not clear data: {e}")
        return False

# ============================================================================
# API & CRUD Tests
# ============================================================================

def test_api_01_create_node() -> TestResult:
    """TC-API-01: Create node"""
    result = TestResult("TC-API-01", "Create node", "P0")
    try:
        clear_all_data()
        payload = {
            "content": "Venkat's note on caching",
            "metadata": {"type": "note", "author": "v"}
        }
        response = requests.post(f"{BASE_URL}/nodes", json=payload, timeout=10)
        
        if response.status_code != 201:
            result.error = f"Expected 201, got {response.status_code}: {response.text}"
            return result
        
        data = response.json()
        if "id" not in data:
            result.error = "Response missing 'id' field"
            return result
        
        node_id = data["id"]
        
        # Verify GET returns same content
        get_response = requests.get(f"{BASE_URL}/nodes/{node_id}", timeout=10)
        if get_response.status_code != 200:
            result.error = f"GET failed with {get_response.status_code}"
            return result
        
        get_data = get_response.json()
        if get_data.get("content") != payload["content"]:
            result.error = "Content mismatch"
            return result
        
        if "embedding" not in data and "embedding" not in get_data:
            result.error = "Embedding field missing"
            return result
        
        result.passed = True
        result.details = {"node_id": node_id}
        return result
    except Exception as e:
        result.error = str(e)
        return result

def test_api_02_read_node_with_relationships() -> TestResult:
    """TC-API-02: Read node with relationships"""
    result = TestResult("TC-API-02", "Read node with relationships", "P0")
    try:
        clear_all_data()
        
        # Create node A
        node_a = requests.post(f"{BASE_URL}/nodes", json={
            "content": "Node A content"
        }, timeout=10).json()
        node_a_id = node_a["id"]
        
        # Create node B
        node_b = requests.post(f"{BASE_URL}/nodes", json={
            "content": "Node B content"
        }, timeout=10).json()
        node_b_id = node_b["id"]
        
        # Create edge A->B
        edge_response = requests.post(f"{BASE_URL}/edges", json={
            "source_id": node_a_id,
            "target_id": node_b_id,
            "relationship": "connects_to",
            "weight": 0.8
        }, timeout=10)
        
        if edge_response.status_code not in [200, 201]:
            result.error = f"Edge creation failed: {edge_response.status_code}"
            return result
        
        # GET node A
        get_response = requests.get(f"{BASE_URL}/nodes/{node_a_id}", timeout=10)
        if get_response.status_code != 200:
            result.error = f"GET failed: {get_response.status_code}"
            return result
        
        data = get_response.json()
        relationships = data.get("relationships", [])
        
        if not relationships:
            result.error = "No relationships found"
            return result
        
        rel = relationships[0]
        if "relationship" not in rel or "target_id" not in rel or "weight" not in rel:
            result.error = "Relationship missing required fields"
            return result
        
        if rel["target_id"] != node_b_id:
            result.error = "Wrong target_id in relationship"
            return result
        
        result.passed = True
        result.details = {"relationships_count": len(relationships)}
        return result
    except Exception as e:
        result.error = str(e)
        return result

def test_api_03_update_node_regenerate_embedding() -> TestResult:
    """TC-API-03: Update node & re-generate embedding"""
    result = TestResult("TC-API-03", "Update node & re-generate embedding", "P0")
    try:
        clear_all_data()
        
        # Create node
        node = requests.post(f"{BASE_URL}/nodes", json={
            "content": "Original content"
        }, timeout=10).json()
        node_id = node["id"]
        
        # Wait a bit for embedding to be generated
        time.sleep(2)
        
        # Get original embedding
        original_response = requests.get(f"{BASE_URL}/nodes/{node_id}", timeout=10)
        original_data = original_response.json()
        original_embedding = original_data.get("embedding")
        
        if not original_embedding:
            # Try creating with explicit embedding
            result.error = "Original embedding not found - embeddings may not be generated automatically"
            return result
        
        time.sleep(1)  # Small delay to ensure different timestamp
        
        # Update node with new content and regen_embedding=true
        update_response = requests.put(f"{BASE_URL}/nodes/{node_id}", json={
            "content": "Updated content with different text",
            "regen_embedding": True
        }, timeout=10)
        
        if update_response.status_code != 200:
            result.error = f"Update failed: {update_response.status_code}"
            return result
        
        # Get updated node
        updated_response = requests.get(f"{BASE_URL}/nodes/{node_id}", timeout=10)
        updated_data = updated_response.json()
        
        if updated_data.get("content") != "Updated content with different text":
            result.error = "Content not updated"
            return result
        
        updated_embedding = updated_data.get("embedding")
        if not updated_embedding:
            result.error = "Updated embedding not found"
            return result
        
        # Check cosine similarity < 0.99
        similarity = cosine_similarity(original_embedding, updated_embedding)
        if similarity >= 0.99:
            result.error = f"Embeddings too similar (similarity={similarity:.4f})"
            return result
        
        result.passed = True
        result.details = {"similarity": similarity}
        return result
    except Exception as e:
        result.error = str(e)
        return result

def test_api_04_delete_node_cascading_edges() -> TestResult:
    """TC-API-04: Delete node cascading edges"""
    result = TestResult("TC-API-04", "Delete node cascading edges", "P0")
    try:
        clear_all_data()
        
        # Create nodes
        node_a = requests.post(f"{BASE_URL}/nodes", json={"content": "Node A"}).json()
        node_b = requests.post(f"{BASE_URL}/nodes", json={"content": "Node B"}).json()
        node_a_id = node_a["id"]
        node_b_id = node_b["id"]
        
        # Create edges
        requests.post(f"{BASE_URL}/edges", json={
            "source_id": node_a_id,
            "target_id": node_b_id,
            "relationship": "connects",
            "weight": 0.5
        }, timeout=10)
        
        # Delete node A
        delete_response = requests.delete(f"{BASE_URL}/nodes/{node_a_id}", timeout=10)
        if delete_response.status_code not in [200, 204]:
            result.error = f"Delete failed: {delete_response.status_code}"
            return result
        
        # Verify node is deleted
        get_node_response = requests.get(f"{BASE_URL}/nodes/{node_a_id}", timeout=10)
        if get_node_response.status_code != 404:
            result.error = f"Node still exists: {get_node_response.status_code}"
            return result
        
        # Verify edges are deleted (check via node B's relationships)
        get_b_response = requests.get(f"{BASE_URL}/nodes/{node_b_id}", timeout=10)
        if get_b_response.status_code == 200:
            b_data = get_b_response.json()
            # Node B should have no incoming edges from A
            relationships = b_data.get("relationships", [])
            for rel in relationships:
                if rel.get("source_id") == node_a_id:
                    result.error = "Edge still exists after node deletion"
                    return result
        
        result.passed = True
        return result
    except Exception as e:
        result.error = str(e)
        return result

def test_api_05_relationship_crud() -> TestResult:
    """TC-API-05: Relationship CRUD"""
    result = TestResult("TC-API-05", "Relationship CRUD", "P1")
    try:
        clear_all_data()
        
        # Create nodes
        node_a = requests.post(f"{BASE_URL}/nodes", json={"content": "Node A"}).json()
        node_b = requests.post(f"{BASE_URL}/nodes", json={"content": "Node B"}).json()
        node_a_id = node_a["id"]
        node_b_id = node_b["id"]
        
        # Create edge
        edge_response = requests.post(f"{BASE_URL}/edges", json={
            "source_id": node_a_id,
            "target_id": node_b_id,
            "relationship": "test_rel",
            "weight": 0.5
        }, timeout=10)
        
        if edge_response.status_code not in [200, 201]:
            result.error = f"Edge creation failed: {edge_response.status_code}"
            return result
        
        # Get all edges to find edge_id
        edges_response = requests.get(f"{BASE_URL}/edges", timeout=10)
        if edges_response.status_code != 200:
            result.error = f"Get edges failed: {edges_response.status_code}"
            return result
        
        edges_data = edges_response.json()
        # Handle both old format (list) and new format (dict with "edges" key)
        if isinstance(edges_data, dict) and "edges" in edges_data:
            edges = edges_data["edges"]
        elif isinstance(edges_data, list):
            edges = edges_data
        else:
            result.error = f"Unexpected edges response format: {edges_data}"
            return result
        
        if not edges or len(edges) == 0:
            result.error = "No edges found"
            return result
        
        edge_id = edges[0].get("edge_id")
        if not edge_id:
            result.error = "Edge ID not found"
            return result
        
        # Get specific edge
        get_edge_response = requests.get(f"{BASE_URL}/edges/{edge_id}", timeout=10)
        if get_edge_response.status_code != 200:
            result.error = f"Get edge failed: {get_edge_response.status_code}"
            return result
        
        # Delete edge
        delete_response = requests.delete(f"{BASE_URL}/edges/{edge_id}", timeout=10)
        if delete_response.status_code not in [200, 204]:
            result.error = f"Delete edge failed: {delete_response.status_code}"
            return result
        
        # Verify edge is deleted
        get_deleted_response = requests.get(f"{BASE_URL}/edges/{edge_id}", timeout=10)
        if get_deleted_response.status_code != 404:
            result.error = "Edge still exists after deletion"
            return result
        
        result.passed = True
        return result
    except Exception as e:
        result.error = str(e)
        return result

# ============================================================================
# Vector Search Tests
# ============================================================================

def test_vec_01_top_k_cosine_similarity() -> TestResult:
    """TC-VEC-01: Top-k cosine similarity ordering"""
    result = TestResult("TC-VEC-01", "Top-k cosine similarity ordering", "P0")
    try:
        clear_all_data()
        
        # Create nodes with different content
        # Node A: very similar to query
        node_a = requests.post(f"{BASE_URL}/nodes", json={
            "content": "Machine learning and artificial intelligence"
        }, timeout=10).json()
        
        # Node B: medium similarity
        node_b = requests.post(f"{BASE_URL}/nodes", json={
            "content": "Data science and analytics"
        }, timeout=10).json()
        
        # Node C: low similarity
        node_c = requests.post(f"{BASE_URL}/nodes", json={
            "content": "Cooking recipes and food preparation"
        }, timeout=10).json()
        
        time.sleep(2)  # Wait for embeddings to be generated
        
        # Search for "machine learning"
        search_response = requests.post(f"{BASE_URL}/search/vector", json={
            "query_text": "machine learning",
            "top_k": 3
        }, timeout=10)
        
        if search_response.status_code != 200:
            result.error = f"Search failed: {search_response.status_code}"
            return result
        
        data = search_response.json()
        results = data.get("results", [])
        
        if len(results) < 3:
            result.error = f"Expected 3 results, got {len(results)}"
            return result
        
        # Check ordering (first result should be most similar)
        scores = [r.get("vector_score", 0) for r in results]
        if scores != sorted(scores, reverse=True):
            result.error = "Results not ordered by similarity"
            return result
        
        # Top result should have high similarity
        top_score = scores[0]
        if top_score < 0.3:  # Reasonable threshold
            result.error = f"Top score too low: {top_score}"
            return result
        
        result.passed = True
        result.details = {"top_score": top_score, "scores": scores}
        return result
    except Exception as e:
        result.error = str(e)
        return result

def test_vec_02_top_k_larger_than_dataset() -> TestResult:
    """TC-VEC-02: Top-k with k > dataset size"""
    result = TestResult("TC-VEC-02", "Top-k with k > dataset size", "P1")
    try:
        clear_all_data()
        
        # Create only 2 nodes
        requests.post(f"{BASE_URL}/nodes", json={"content": "Node 1"}).json()
        requests.post(f"{BASE_URL}/nodes", json={"content": "Node 2"}).json()
        
        time.sleep(1)
        
        # Search with k=10 (larger than dataset)
        search_response = requests.post(f"{BASE_URL}/search/vector", json={
            "query_text": "test query",
            "top_k": 10
        }, timeout=10)
        
        if search_response.status_code != 200:
            result.error = f"Search failed: {search_response.status_code}"
            return result
        
        data = search_response.json()
        results = data.get("results", [])
        
        # Should return all items (2) without error
        if len(results) != 2:
            result.error = f"Expected 2 results, got {len(results)}"
            return result
        
        result.passed = True
        result.details = {"result_count": len(results)}
        return result
    except Exception as e:
        result.error = str(e)
        return result

def test_vec_03_filtering_by_metadata() -> TestResult:
    """TC-VEC-03: Filtering by metadata"""
    result = TestResult("TC-VEC-03", "Filtering by metadata", "P1")
    try:
        clear_all_data()
        
        # Create nodes with different metadata
        requests.post(f"{BASE_URL}/nodes", json={
            "content": "Note about caching",
            "metadata": {"type": "note"}
        }, timeout=10).json()
        
        requests.post(f"{BASE_URL}/nodes", json={
            "content": "Article about caching",
            "metadata": {"type": "article"}
        }, timeout=10).json()
        
        requests.post(f"{BASE_URL}/nodes", json={
            "content": "Another note",
            "metadata": {"type": "note"}
        }, timeout=10).json()
        
        time.sleep(1)
        
        # Search with filter
        search_response = requests.post(f"{BASE_URL}/search/vector", json={
            "query_text": "caching",
            "top_k": 10,
            "filter": {"type": "note"}
        }, timeout=10)
        
        if search_response.status_code != 200:
            result.error = f"Search failed: {search_response.status_code}"
            return result
        
        data = search_response.json()
        results = data.get("results", [])
        
        # All results should have type=note
        for r in results:
            if r.get("metadata", {}).get("type") != "note":
                result.error = f"Result has wrong type: {r.get('metadata')}"
                return result
        
        if len(results) != 2:
            result.error = f"Expected 2 results, got {len(results)}"
            return result
        
        result.passed = True
        result.details = {"result_count": len(results)}
        return result
    except Exception as e:
        result.error = str(e)
        return result

# ============================================================================
# Graph Traversal Tests
# ============================================================================

def test_graph_01_bfs_depth_limited() -> TestResult:
    """TC-GRAPH-01: BFS / depth-limited traversal"""
    result = TestResult("TC-GRAPH-01", "BFS / depth-limited traversal", "P0")
    try:
        clear_all_data()
        
        # Build chain A->B->C->D
        node_a = requests.post(f"{BASE_URL}/nodes", json={"content": "Node A"}).json()
        node_b = requests.post(f"{BASE_URL}/nodes", json={"content": "Node B"}).json()
        node_c = requests.post(f"{BASE_URL}/nodes", json={"content": "Node C"}).json()
        node_d = requests.post(f"{BASE_URL}/nodes", json={"content": "Node D"}).json()
        
        node_a_id = node_a["id"]
        node_b_id = node_b["id"]
        node_c_id = node_c["id"]
        node_d_id = node_d["id"]
        
        # Create edges
        requests.post(f"{BASE_URL}/edges", json={
            "source_id": node_a_id, "target_id": node_b_id, "relationship": "connects", "weight": 1.0
        }, timeout=10)
        requests.post(f"{BASE_URL}/edges", json={
            "source_id": node_b_id, "target_id": node_c_id, "relationship": "connects", "weight": 1.0
        }, timeout=10)
        requests.post(f"{BASE_URL}/edges", json={
            "source_id": node_c_id, "target_id": node_d_id, "relationship": "connects", "weight": 1.0
        }, timeout=10)
        
        time.sleep(1)
        
        # Query depth=2 from A
        search_response = requests.get(
            f"{BASE_URL}/search/graph?start_id={node_a_id}&depth=2&max_nodes=100",
            timeout=10
        )
        
        if search_response.status_code != 200:
            result.error = f"Search failed: {search_response.status_code}"
            return result
        
        data = search_response.json()
        nodes = data.get("nodes", [])
        
        # Should return B (depth 1) and C (depth 2), not D
        node_ids = [n.get("node_id") for n in nodes]
        
        if node_b_id not in node_ids:
            result.error = "Node B (depth 1) not found"
            return result
        
        if node_c_id not in node_ids:
            result.error = "Node C (depth 2) not found"
            return result
        
        if node_d_id in node_ids:
            result.error = "Node D (depth 3) should not be included"
            return result
        
        # Check distances
        for n in nodes:
            if n.get("node_id") == node_b_id and n.get("distance") != 1:
                result.error = f"Node B has wrong distance: {n.get('distance')}"
                return result
            if n.get("node_id") == node_c_id and n.get("distance") != 2:
                result.error = f"Node C has wrong distance: {n.get('distance')}"
                return result
        
        result.passed = True
        result.details = {"nodes_found": len(nodes), "node_ids": node_ids}
        return result
    except Exception as e:
        result.error = str(e)
        return result

def test_graph_02_multi_type_relationships() -> TestResult:
    """TC-GRAPH-02: Multi-type relationships"""
    result = TestResult("TC-GRAPH-02", "Multi-type relationships", "P1")
    try:
        clear_all_data()
        
        # Create nodes
        node_a = requests.post(f"{BASE_URL}/nodes", json={"content": "Author A"}).json()
        node_b = requests.post(f"{BASE_URL}/nodes", json={"content": "Book B"}).json()
        node_c = requests.post(f"{BASE_URL}/nodes", json={"content": "Book C"}).json()
        
        node_a_id = node_a["id"]
        node_b_id = node_b["id"]
        node_c_id = node_c["id"]
        
        # Create edges with different types
        requests.post(f"{BASE_URL}/edges", json={
            "source_id": node_a_id, "target_id": node_b_id,
            "relationship": "author_of", "weight": 1.0
        }, timeout=10)
        requests.post(f"{BASE_URL}/edges", json={
            "source_id": node_a_id, "target_id": node_c_id,
            "relationship": "reviewed", "weight": 1.0
        }, timeout=10)
        
        time.sleep(1)
        
        # Traverse with filter type=author_of
        search_response = requests.get(
            f"{BASE_URL}/search/graph?start_id={node_a_id}&depth=1&relationship_type=author_of&max_nodes=100",
            timeout=10
        )
        
        if search_response.status_code != 200:
            result.error = f"Search failed: {search_response.status_code}"
            return result
        
        data = search_response.json()
        nodes = data.get("nodes", [])
        node_ids = [n.get("node_id") for n in nodes]
        
        # Should only return node B (author_of), not C (reviewed)
        if node_b_id not in node_ids:
            result.error = "Node B (author_of) not found"
            return result
        
        if node_c_id in node_ids:
            result.error = "Node C (reviewed) should not be included"
            return result
        
        result.passed = True
        result.details = {"nodes_found": len(nodes)}
        return result
    except Exception as e:
        result.error = str(e)
        return result

def test_graph_03_cycle_handling() -> TestResult:
    """TC-GRAPH-03: Cycle handling"""
    result = TestResult("TC-GRAPH-03", "Cycle handling", "P1")
    try:
        clear_all_data()
        
        # Create cycle A->B->A
        node_a = requests.post(f"{BASE_URL}/nodes", json={"content": "Node A"}).json()
        node_b = requests.post(f"{BASE_URL}/nodes", json={"content": "Node B"}).json()
        
        node_a_id = node_a["id"]
        node_b_id = node_b["id"]
        
        requests.post(f"{BASE_URL}/edges", json={
            "source_id": node_a_id, "target_id": node_b_id, "relationship": "connects", "weight": 1.0
        }, timeout=10)
        requests.post(f"{BASE_URL}/edges", json={
            "source_id": node_b_id, "target_id": node_a_id, "relationship": "connects", "weight": 1.0
        }, timeout=10)
        
        time.sleep(1)
        
        # Traverse from A with depth=5 (should not infinite loop)
        search_response = requests.get(
            f"{BASE_URL}/search/graph?start_id={node_a_id}&depth=5&max_nodes=100",
            timeout=10
        )
        
        if search_response.status_code != 200:
            result.error = f"Search failed: {search_response.status_code}"
            return result
        
        data = search_response.json()
        nodes = data.get("nodes", [])
        
        # Should only visit each node once per depth level
        # At depth 1: B
        # At depth 2: A (via cycle)
        # At depth 3: B (via cycle)
        # etc.
        node_ids = [n.get("node_id") for n in nodes]
        
        # Both nodes should be found, but traversal should terminate
        if node_a_id not in node_ids or node_b_id not in node_ids:
            result.error = "Not all nodes in cycle found"
            return result
        
        # Check that nodes are not duplicated excessively (max 2-3 visits per node)
        node_a_count = sum(1 for n in nodes if n.get("node_id") == node_a_id)
        node_b_count = sum(1 for n in nodes if n.get("node_id") == node_b_id)
        
        # Should have reasonable visit counts (not infinite)
        if node_a_count > 10 or node_b_count > 10:
            result.error = f"Excessive node visits: A={node_a_count}, B={node_b_count}"
            return result
        
        result.passed = True
        result.details = {"total_nodes": len(nodes), "node_a_visits": node_a_count, "node_b_visits": node_b_count}
        return result
    except Exception as e:
        result.error = str(e)
        return result

# ============================================================================
# Hybrid Search Tests
# ============================================================================

def test_hyb_01_weighted_merge_correctness() -> TestResult:
    """TC-HYB-01: Weighted merge correctness"""
    result = TestResult("TC-HYB-01", "Weighted merge correctness", "P0")
    try:
        clear_all_data()
        
        # Create three nodes:
        # V-similar: high vector score but graph distant
        # G-close: low vector score but directly connected to a seed node
        # Neutral: medium both
        
        node_v = requests.post(f"{BASE_URL}/nodes", json={
            "content": "Machine learning and artificial intelligence algorithms"
        }, timeout=10).json()
        
        node_g = requests.post(f"{BASE_URL}/nodes", json={
            "content": "Cooking recipes and food preparation techniques"
        }, timeout=10).json()
        
        node_n = requests.post(f"{BASE_URL}/nodes", json={
            "content": "Data science and machine learning"
        }, timeout=10).json()
        
        node_v_id = node_v["id"]
        node_g_id = node_g["id"]
        node_n_id = node_n["id"]
        
        # Create a seed node that connects to G-close
        # This seed node should be found by vector search for "machine learning"
        seed_node = requests.post(f"{BASE_URL}/nodes", json={
            "content": "machine learning techniques and algorithms"
        }, timeout=10).json()
        seed_node_id = seed_node["id"]
        
        # Connect seed to G-close (graph proximity)
        requests.post(f"{BASE_URL}/edges", json={
            "source_id": seed_node_id, "target_id": node_g_id,
            "relationship": "connects", "weight": 1.0
        }, timeout=10)
        
        time.sleep(3)  # Wait for embeddings
        
        # Hybrid search with vector_weight=0.7, graph_weight=0.3
        search_response = requests.post(f"{BASE_URL}/search/hybrid", json={
            "query_text": "machine learning",
            "vector_weight": 0.7,
            "graph_weight": 0.3,
            "top_k": 5
        }, timeout=10)
        
        if search_response.status_code != 200:
            result.error = f"Search failed: {search_response.status_code}"
            return result
        
        data = search_response.json()
        results = data.get("results", [])
        
        if len(results) < 2:
            result.error = f"Expected at least 2 results, got {len(results)}"
            return result
        
        # Check that results have score breakdown
        for r in results:
            if "vector_score" not in r or "graph_score" not in r or "score" not in r:
                result.error = "Result missing score breakdown"
                return result
        
        # Find V-similar and G-close in results
        v_result = next((r for r in results if r.get("node_id") == node_v_id), None)
        g_result = next((r for r in results if r.get("node_id") == node_g_id), None)
        
        if not v_result:
            result.error = f"V-similar node not found in results. Found: {[r.get('node_id') for r in results]}"
            return result
        
        if not g_result:
            result.error = f"G-close node not found in results. Found: {[r.get('node_id') for r in results]}"
            return result
        
        # V-similar should have higher vector score
        # With vector_weight=0.7, V-similar should rank higher if vector advantage is significant
        v_vector = v_result.get("vector_score", 0)
        g_vector = g_result.get("vector_score", 0)
        v_final = v_result.get("score", 0)
        g_final = g_result.get("score", 0)
        
        # Verify score breakdown is present and reasonable
        if v_vector > 0.3 and g_vector < 0.3:
            # V-similar has much higher vector score, so with vector_weight=0.7 it should rank higher
            if v_final < g_final:
                result.error = f"V-similar should rank above G-close with vector_weight=0.7. V: {v_final:.4f}, G: {g_final:.4f}"
                return result
        
        result.passed = True
        result.details = {
            "v_score": v_final,
            "g_score": g_final,
            "v_vector": v_vector,
            "g_vector": g_vector,
            "v_graph": v_result.get("graph_score", 0),
            "g_graph": g_result.get("graph_score", 0)
        }
        return result
    except Exception as e:
        result.error = str(e)
        return result

def test_hyb_02_tuning_extremes() -> TestResult:
    """TC-HYB-02: Tuning extremes"""
    result = TestResult("TC-HYB-02", "Tuning extremes", "P0")
    try:
        clear_all_data()
        
        # Create nodes
        node_a = requests.post(f"{BASE_URL}/nodes", json={
            "content": "Machine learning algorithms"
        }, timeout=10).json()
        
        node_b = requests.post(f"{BASE_URL}/nodes", json={
            "content": "Cooking recipes"
        }, timeout=10).json()
        
        node_a_id = node_a["id"]
        node_b_id = node_b["id"]
        
        # Create graph connection
        query_node = requests.post(f"{BASE_URL}/nodes", json={
            "content": "test"
        }, timeout=10).json()
        query_node_id = query_node["id"]
        
        requests.post(f"{BASE_URL}/edges", json={
            "source_id": query_node_id, "target_id": node_b_id,
            "relationship": "connects", "weight": 1.0
        }, timeout=10)
        
        time.sleep(2)
        
        # Test 1: vector_weight=1.0, graph_weight=0.0 (should match vector-only)
        hybrid_response = requests.post(f"{BASE_URL}/search/hybrid", json={
            "query_text": "machine learning",
            "vector_weight": 1.0,
            "graph_weight": 0.0,
            "top_k": 2
        }, timeout=10)
        
        if hybrid_response.status_code != 200:
            result.error = f"Hybrid search failed: {hybrid_response.status_code}"
            return result
        
        vector_response = requests.post(f"{BASE_URL}/search/vector", json={
            "query_text": "machine learning",
            "top_k": 2
        }, timeout=10)
        
        if vector_response.status_code != 200:
            result.error = f"Vector search failed: {vector_response.status_code}"
            return result
        
        hybrid_results = hybrid_response.json().get("results", [])
        vector_results = vector_response.json().get("results", [])
        
        # Ordering should match (at least top result)
        if len(hybrid_results) > 0 and len(vector_results) > 0:
            hybrid_top = hybrid_results[0].get("node_id")
            vector_top = vector_results[0].get("id")
            if hybrid_top != vector_top:
                result.error = f"Ordering mismatch: hybrid={hybrid_top}, vector={vector_top}"
                return result
        
        # Test 2: vector_weight=0.0, graph_weight=1.0 (should match graph-only)
        hybrid_response2 = requests.post(f"{BASE_URL}/search/hybrid", json={
            "query_text": "test",
            "vector_weight": 0.0,
            "graph_weight": 1.0,
            "top_k": 2
        }, timeout=10)
        
        if hybrid_response2.status_code != 200:
            result.error = f"Hybrid search 2 failed: {hybrid_response2.status_code}"
            return result
        
        # Graph search from query_node
        graph_response = requests.get(
            f"{BASE_URL}/search/graph?start_id={query_node_id}&depth=1&max_nodes=2",
            timeout=10
        )
        
        if graph_response.status_code != 200:
            result.error = f"Graph search failed: {graph_response.status_code}"
            return result
        
        hybrid_results2 = hybrid_response2.json().get("results", [])
        graph_results = graph_response.json().get("nodes", [])
        
        # Top result should be graph-connected node
        if len(hybrid_results2) > 0 and len(graph_results) > 0:
            hybrid_top2 = hybrid_results2[0].get("node_id")
            graph_top = graph_results[0].get("node_id")
            # Should prioritize graph-connected node
            if hybrid_top2 != node_b_id and graph_top == node_b_id:
                result.error = "Graph weight=1.0 should prioritize graph-connected nodes"
                return result
        
        result.passed = True
        return result
    except Exception as e:
        result.error = str(e)
        return result

# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all test cases and report results."""
    print_header("GraphMind API Test Suite")
    
    # Check server
    print_info("Checking server availability...")
    if not wait_for_server():
        print_fail("Server is not available. Please start the server first.")
        sys.exit(1)
    print_pass("Server is available")
    
    # Clear data
    print_info("Clearing existing data...")
    clear_all_data()
    
    # Define all tests
    tests = [
        # API & CRUD
        test_api_01_create_node,
        test_api_02_read_node_with_relationships,
        test_api_03_update_node_regenerate_embedding,
        test_api_04_delete_node_cascading_edges,
        test_api_05_relationship_crud,
        # Vector Search
        test_vec_01_top_k_cosine_similarity,
        test_vec_02_top_k_larger_than_dataset,
        test_vec_03_filtering_by_metadata,
        # Graph Traversal
        test_graph_01_bfs_depth_limited,
        test_graph_02_multi_type_relationships,
        test_graph_03_cycle_handling,
        # Hybrid Search
        test_hyb_01_weighted_merge_correctness,
        test_hyb_02_tuning_extremes,
    ]
    
    results = []
    passed = 0
    failed = 0
    
    print_header("Running Tests")
    
    for test_func in tests:
        print_test(test_func.__name__, test_func.__doc__ or "")
        try:
            result = test_func()
            results.append(result)
            if result.passed:
                print_pass()
                if result.details:
                    print_info(f"Details: {result.details}")
                passed += 1
            else:
                print_fail(result.error or "Unknown error")
                failed += 1
        except Exception as e:
            print_fail(f"Test exception: {str(e)}")
            failed += 1
        print()
    
    # Summary
    print_header("Test Summary")
    print(f"{Colors.BOLD}Total Tests:{Colors.RESET} {len(tests)}")
    print(f"{Colors.GREEN}Passed:{Colors.RESET} {passed}")
    print(f"{Colors.RED}Failed:{Colors.RESET} {failed}")
    print()
    
    # Detailed results
    print_header("Detailed Results")
    for result in results:
        status = f"{Colors.GREEN}[PASS]{Colors.RESET}" if result.passed else f"{Colors.RED}[FAIL]{Colors.RESET}"
        print(f"{status} [{result.test_id}] {result.name} ({result.priority})")
        if not result.passed and result.error:
            print(f"  Error: {result.error}")
        if result.details:
            print(f"  Details: {result.details}")
    
    # Priority breakdown
    p0_passed = sum(1 for r in results if r.priority == "P0" and r.passed)
    p0_total = sum(1 for r in results if r.priority == "P0")
    p1_passed = sum(1 for r in results if r.priority == "P1" and r.passed)
    p1_total = sum(1 for r in results if r.priority == "P1")
    
    print()
    print_header("Priority Breakdown")
    print(f"P0 (Critical): {p0_passed}/{p0_total} passed")
    print(f"P1 (Important): {p1_passed}/{p1_total} passed")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

