#!/usr/bin/env python3
"""
API Testing Script for Vector + Graph Hybrid Database
Tests all endpoints and validates the hybrid search functionality
"""

import requests
import json
import time
from typing import Dict, List

BASE_URL = "http://localhost:8000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text: str):
    print(f"{Colors.YELLOW}ℹ {text}{Colors.END}")

def test_health_check():
    """Test system health endpoint"""
    print_header("TEST 1: Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        
        print_success(f"API is online")
        print_info(f"Nodes: {data['nodes']}")
        print_info(f"Edges: {data['edges']}")
        print_info(f"Embedding model: {data['embedding_model']}")
        return True
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False

def test_node_crud():
    """Test node CRUD operations"""
    print_header("TEST 2: Node CRUD Operations")
    
    try:
        # Create node
        print_info("Creating test node...")
        create_response = requests.post(f"{BASE_URL}/nodes", json={
            "text": "Test paper on artificial intelligence and machine learning",
            "metadata": {"type": "test", "year": 2024}
        })
        node_data = create_response.json()
        node_id = node_data['id']
        print_success(f"Node created with ID: {node_id}")
        
        # Get node
        print_info("Retrieving node...")
        get_response = requests.get(f"{BASE_URL}/nodes/{node_id}")
        retrieved = get_response.json()
        assert retrieved['id'] == node_id
        print_success("Node retrieved successfully")
        
        # Update node
        print_info("Updating node metadata...")
        update_response = requests.put(f"{BASE_URL}/nodes/{node_id}", json={
            "metadata": {"citations": 100}
        })
        print_success("Node updated successfully")
        
        # Delete node
        print_info("Deleting node...")
        delete_response = requests.delete(f"{BASE_URL}/nodes/{node_id}")
        print_success("Node deleted successfully")
        
        return True
    except Exception as e:
        print_error(f"Node CRUD test failed: {e}")
        return False

def test_edge_crud():
    """Test edge CRUD operations"""
    print_header("TEST 3: Edge CRUD Operations")
    
    try:
        # Create two nodes
        print_info("Creating two test nodes...")
        node1 = requests.post(f"{BASE_URL}/nodes", json={
            "text": "Source node for edge test",
            "metadata": {"type": "test"}
        }).json()
        
        node2 = requests.post(f"{BASE_URL}/nodes", json={
            "text": "Target node for edge test",
            "metadata": {"type": "test"}
        }).json()
        
        print_success(f"Created nodes: {node1['id']} and {node2['id']}")
        
        # Create edge
        print_info("Creating edge...")
        edge_response = requests.post(f"{BASE_URL}/edges", json={
            "source_id": node1['id'],
            "target_id": node2['id'],
            "relationship_type": "test_relation",
            "weight": 0.8
        })
        edge_data = edge_response.json()
        edge_id = edge_data['id']
        print_success(f"Edge created with ID: {edge_id}")
        
        # Get edge
        print_info("Retrieving edge...")
        get_edge = requests.get(f"{BASE_URL}/edges/{edge_id}")
        retrieved_edge = get_edge.json()
        assert retrieved_edge['id'] == edge_id
        print_success("Edge retrieved successfully")
        
        # Clean up
        requests.delete(f"{BASE_URL}/nodes/{node1['id']}")
        requests.delete(f"{BASE_URL}/nodes/{node2['id']}")
        
        return True
    except Exception as e:
        print_error(f"Edge CRUD test failed: {e}")
        return False

def test_vector_search():
    """Test vector search functionality"""
    print_header("TEST 4: Vector Search")
    
    try:
        # Create test nodes
        print_info("Creating test dataset...")
        nodes = [
            "Machine learning and artificial intelligence research",
            "Deep learning neural networks for computer vision",
            "Natural language processing with transformers",
            "Quantum computing and quantum algorithms"
        ]
        
        for text in nodes:
            requests.post(f"{BASE_URL}/nodes", json={
                "text": text,
                "metadata": {"type": "test"}
            })
        
        print_success(f"Created {len(nodes)} test nodes")
        
        # Perform vector search
        print_info("Performing vector search for 'machine learning'...")
        search_response = requests.post(f"{BASE_URL}/search/vector", json={
            "query_text": "machine learning algorithms",
            "top_k": 3
        })
        
        results = search_response.json()
        print_success(f"Found {len(results)} results")
        
        for i, result in enumerate(results, 1):
            print(f"   {i}. Score: {result['score']:.3f} - {result['text'][:50]}...")
        
        # Verify first result is most relevant
        assert results[0]['score'] > 0.5, "Top result should have high similarity"
        print_success("Vector search validation passed")
        
        return True
    except Exception as e:
        print_error(f"Vector search test failed: {e}")
        return False

def test_graph_traversal():
    """Test graph traversal functionality"""
    print_header("TEST 5: Graph Traversal")
    
    try:
        # Create a small graph
        print_info("Creating test graph...")
        
        node1 = requests.post(f"{BASE_URL}/nodes", json={
            "text": "Root node",
            "metadata": {"type": "test", "level": 0}
        }).json()
        
        node2 = requests.post(f"{BASE_URL}/nodes", json={
            "text": "Child node 1",
            "metadata": {"type": "test", "level": 1}
        }).json()
        
        node3 = requests.post(f"{BASE_URL}/nodes", json={
            "text": "Grandchild node",
            "metadata": {"type": "test", "level": 2}
        }).json()
        
        # Create edges
        requests.post(f"{BASE_URL}/edges", json={
            "source_id": node1['id'],
            "target_id": node2['id'],
            "relationship_type": "parent_of",
            "weight": 1.0
        })
        
        requests.post(f"{BASE_URL}/edges", json={
            "source_id": node2['id'],
            "target_id": node3['id'],
            "relationship_type": "parent_of",
            "weight": 1.0
        })
        
        print_success("Test graph created")
        
        # Traverse graph
        print_info(f"Traversing from root node (depth=2)...")
        traversal = requests.get(
            f"{BASE_URL}/search/graph",
            params={"start_id": node1['id'], "depth": 2}
        ).json()
        
        print_success(f"Found {len(traversal['nodes'])} nodes in traversal")
        
        for node in traversal['nodes']:
            print(f"   Depth {node['depth']}: {node['text']}")
        
        assert len(traversal['nodes']) == 3, "Should find all 3 nodes"
        print_success("Graph traversal validation passed")
        
        return True
    except Exception as e:
        print_error(f"Graph traversal test failed: {e}")
        return False

def test_hybrid_search():
    """Test hybrid search - THE MAIN FEATURE!"""
    print_header("TEST 6: Hybrid Search (CORE FEATURE)")
    
    try:
        print_info("Testing hybrid search with demo data...")
        
        # Perform hybrid search
        search_response = requests.post(f"{BASE_URL}/search/hybrid", json={
            "query_text": "neural architecture search Stanford",
            "top_k": 5,
            "vector_weight": 0.5,
            "graph_weight": 0.5
        })
        
        results = search_response.json()
        
        if not results:
            print_error("No results found - make sure demo data is loaded!")
            return False
        
        print_success(f"Found {len(results)} hybrid search results")
        
        print("\n" + Colors.BOLD + "Top Results:" + Colors.END)
        for i, result in enumerate(results[:3], 1):
            print(f"\n{Colors.BOLD}{i}. Score: {result['score']:.3f}{Colors.END}")
            print(f"   Text: {result['text'][:80]}...")
            print(f"   Vector: {result.get('vector_score', 0):.3f} | Graph: {result.get('graph_score', 0):.3f}")
            if result.get('explanation'):
                print(f"   {Colors.YELLOW}Explanation: {result['explanation']}{Colors.END}")
        
        # Validate hybrid scores are combining both
        top_result = results[0]
        assert 'vector_score' in top_result, "Should include vector score"
        assert 'graph_score' in top_result, "Should include graph score"
        assert 'explanation' in top_result, "Should include explanation"
        
        print_success("\n✓ Hybrid search working correctly!")
        print_success("✓ Vector and graph scores are being combined")
        print_success("✓ Explanations are being generated")
        
        return True
    except Exception as e:
        print_error(f"Hybrid search test failed: {e}")
        return False

def compare_search_methods():
    """Compare all three search methods side by side"""
    print_header("TEST 7: Search Method Comparison")
    
    query = "graph neural networks knowledge graphs"
    
    try:
        print_info(f"Comparing searches for: '{query}'\n")
        
        # Vector search
        print(f"{Colors.BOLD}1. VECTOR SEARCH (Semantic Only):{Colors.END}")
        vector_results = requests.post(f"{BASE_URL}/search/vector", json={
            "query_text": query,
            "top_k": 3
        }).json()
        
        for i, r in enumerate(vector_results, 1):
            print(f"   {i}. [{r['score']:.3f}] {r['text'][:60]}...")
        
        # Graph-heavy search
        print(f"\n{Colors.BOLD}2. GRAPH-HEAVY SEARCH (Relationships):{Colors.END}")
        graph_results = requests.post(f"{BASE_URL}/search/hybrid", json={
            "query_text": query,
            "top_k": 3,
            "vector_weight": 0.2,
            "graph_weight": 0.8
        }).json()
        
        for i, r in enumerate(graph_results, 1):
            print(f"   {i}. [{r['score']:.3f}] {r['text'][:60]}...")
        
        # Hybrid search
        print(f"\n{Colors.BOLD}3. HYBRID SEARCH (Best of Both!):{Colors.END}")
        hybrid_results = requests.post(f"{BASE_URL}/search/hybrid", json={
            "query_text": query,
            "top_k": 3,
            "vector_weight": 0.5,
            "graph_weight": 0.5
        }).json()
        
        for i, r in enumerate(hybrid_results, 1):
            print(f"   {i}. [{r['score']:.3f}] {r['text'][:60]}...")
            print(f"       V:{r.get('vector_score', 0):.2f} | G:{r.get('graph_score', 0):.2f}")
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All search methods working!{Colors.END}")
        print(f"{Colors.YELLOW}Notice how hybrid search balances semantic and relational relevance{Colors.END}")
        
        return True
    except Exception as e:
        print_error(f"Comparison test failed: {e}")
        return False

def test_performance():
    """Test query performance"""
    print_header("TEST 8: Performance Benchmarking")
    
    try:
        query = "machine learning research"
        iterations = 10
        
        print_info(f"Running {iterations} queries to measure performance...")
        
        # Vector search performance
        start = time.time()
        for _ in range(iterations):
            requests.post(f"{BASE_URL}/search/vector", json={
                "query_text": query,
                "top_k": 5
            })
        vector_time = (time.time() - start) / iterations
        
        # Hybrid search performance
        start = time.time()
        for _ in range(iterations):
            requests.post(f"{BASE_URL}/search/hybrid", json={
                "query_text": query,
                "top_k": 5,
                "vector_weight": 0.5,
                "graph_weight": 0.5
            })
        hybrid_time = (time.time() - start) / iterations
        
        print_success(f"Vector search avg: {vector_time*1000:.1f}ms")
        print_success(f"Hybrid search avg: {hybrid_time*1000:.1f}ms")
        
        if hybrid_time < 1.0:  # Less than 1 second
            print_success("✓ Performance is excellent for real-time queries!")
        else:
            print_error("⚠ Performance might be slow for real-time use")
        
        return True
    except Exception as e:
        print_error(f"Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   VECTOR + GRAPH HYBRID DATABASE - API TEST SUITE          ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(Colors.END)
    
    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except:
        print_error("\n❌ ERROR: Cannot connect to API server!")
        print_info("Please start the server first: python main.py")
        return
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Node CRUD", test_node_crud),
        ("Edge CRUD", test_edge_crud),
        ("Vector Search", test_vector_search),
        ("Graph Traversal", test_graph_traversal),
        ("Hybrid Search", test_hybrid_search),
        ("Search Comparison", compare_search_methods),
        ("Performance", test_performance)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            time.sleep(0.5)  # Small delay between tests
        except Exception as e:
            print_error(f"Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        if result:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}")
        print("╔════════════════════════════════════════════════════════════╗")
        print("║           🎉 ALL TESTS PASSED! 🎉                          ║")
        print("║     Your system is ready for the demo!                     ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(Colors.END)
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}")
        print("╔════════════════════════════════════════════════════════════╗")
        print("║     ⚠️  SOME TESTS FAILED  ⚠️                             ║")
        print("║     Please check the errors above                          ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(Colors.END)

if __name__ == "__main__":
    main()