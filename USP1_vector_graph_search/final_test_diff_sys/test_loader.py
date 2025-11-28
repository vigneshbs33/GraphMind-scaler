#!/usr/bin/env python3
"""
Test Data Loader - Loads canonical test dataset for validation
Matches the exact test cases from DevForge evaluation criteria
"""

import requests
import json

API_URL = "http://localhost:8000"

# Canonical test documents with mock embeddings
TEST_DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Redis caching strategies",
        "text": """Redis became the default choice for caching mostly because people like avoiding slow databases.
There are the usual headaches: eviction policies like LRU vs LFU, memory pressure, and when
someone forgets to set TTLs and wonders why servers fall over. A funny incident last month:
our checkout service kept missing prices because a stale cache key survived a deploy.""",
        "embedding": [0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
        "metadata": {"type": "article", "tags": ["cache", "redis"], "author": "alice"}
    },
    {
        "id": "doc2",
        "title": "RedisGraph module",
        "text": """The RedisGraph module promises a weird marriage: pretend your cache is also a graph database.
Honestly, it works better than expected. You can store relationships like user -> viewed -> product
and then still query it with cypher-like syntax. Someone even built a PageRank demo over it.""",
        "embedding": [0.70, 0.10, 0.60, 0.00, 0.00, 0.00],
        "metadata": {"type": "article", "tags": ["redis", "graph"]}
    },
    {
        "id": "doc3",
        "title": "Distributed systems",
        "text": """Distributed systems are basically long-distance relationships. Nodes drift apart, messages get lost,
and during network partitions everyone blames everyone else. Leader election decides who gets
boss privileges until the next heartbeat timeout. Caching across a cluster is especially fun because
one stale node ruins the whole party.""",
        "embedding": [0.10, 0.05, 0.00, 0.90, 0.00, 0.00],
        "metadata": {"type": "article", "tags": ["distributed", "systems"]}
    },
    {
        "id": "doc4",
        "title": "Cache invalidation note",
        "text": """A short note on cache invalidation: you think you understand it until your application grows. Patterns
like write-through, write-behind, and cache-aside all behave differently under load. Versioned keys
help, but someone will always ship code that forgets to update them. The universe trends toward chaos.""",
        "embedding": [0.80, 0.15, 0.00, 0.00, 0.00, 0.00],
        "metadata": {"type": "note", "tags": ["cache"]}
    },
    {
        "id": "doc5",
        "title": "Graph algorithms",
        "text": """Graph algorithms show up in real life more than people notice. Social feeds rely on BFS for exploring
connections, recommendations rely on random walks, and PageRank still refuses to die. Even your
team's on-call rotation effectively forms a directed cycle, complete with its own failure modes.""",
        "embedding": [0.05, 0.00, 0.90, 0.10, 0.00, 0.00],
        "metadata": {"type": "article", "tags": ["graph", "algorithms"]}
    },
    {
        "id": "doc6",
        "title": "README: Redis+Graph",
        "text": """README draft: to combine Redis with a graph database, you start by defining nodes for each entity,
like articles, users, or configuration snippets. Then you create edges describing interactions: mentions,
references, imports, or even blame (use sparingly). The magic happens when semantic search embeddings
overlay this structure and suddenly the system feels smarter than it is.""",
        "embedding": [0.60, 0.05, 0.50, 0.00, 0.10, 0.00],
        "metadata": {"type": "readme", "tags": ["redis", "graph", "guide"]}
    }
]

# Canonical edges
TEST_EDGES = [
    {"id": "E1", "source": "doc1", "target": "doc4", "type": "related_to", "weight": 0.8},
    {"id": "E2", "source": "doc2", "target": "doc6", "type": "mentions", "weight": 0.9},
    {"id": "E3", "source": "doc6", "target": "doc1", "type": "references", "weight": 0.6},
    {"id": "E4", "source": "doc3", "target": "doc5", "type": "related_to", "weight": 0.5},
    {"id": "E5", "source": "doc2", "target": "doc5", "type": "example_of", "weight": 0.3},
]

def reset_system():
    """Clear existing data"""
    print("🔄 Resetting system...")
    try:
        response = requests.post(f"{API_URL}/reset")
        if response.status_code == 200:
            print("✅ System reset")
            return True
    except:
        print("❌ Failed to reset system")
        return False

def create_nodes():
    """Create all test nodes"""
    print("\n📝 Creating nodes...")
    created = 0
    
    for doc in TEST_DOCUMENTS:
        try:
            response = requests.post(
                f"{API_URL}/nodes",
                json={
                    "id": doc["id"],
                    "text": doc["text"],
                    "title": doc["title"],
                    "metadata": doc["metadata"],
                    "embedding": doc["embedding"]
                }
            )
            
            if response.status_code == 201:
                print(f"  ✓ Created {doc['id']}: {doc['title']}")
                created += 1
            else:
                print(f"  ✗ Failed to create {doc['id']}: {response.status_code}")
                
        except Exception as e:
            print(f"  ✗ Error creating {doc['id']}: {e}")
    
    print(f"\n✅ Created {created}/{len(TEST_DOCUMENTS)} nodes")
    return created == len(TEST_DOCUMENTS)

def create_edges():
    """Create all test edges"""
    print("\n🔗 Creating edges...")
    created = 0
    
    for edge in TEST_EDGES:
        try:
            response = requests.post(
                f"{API_URL}/edges",
                json={
                    "source": edge["source"],
                    "target": edge["target"],
                    "type": edge["type"],
                    "weight": edge["weight"]
                }
            )
            
            if response.status_code == 201:
                print(f"  ✓ Created edge {edge['source']} -> {edge['target']} ({edge['type']})")
                created += 1
            else:
                print(f"  ✗ Failed to create edge: {response.status_code}")
                
        except Exception as e:
            print(f"  ✗ Error creating edge: {e}")
    
    print(f"\n✅ Created {created}/{len(TEST_EDGES)} edges")
    return created == len(TEST_EDGES)

def verify_data():
    """Verify data loaded correctly"""
    print("\n🔍 Verifying data...")
    
    try:
        # Check health
        response = requests.get(f"{API_URL}/health")
        health = response.json()
        
        print(f"  Nodes: {health['nodes']}")
        print(f"  Edges: {health['edges']}")
        
        if health['nodes'] == len(TEST_DOCUMENTS) and health['edges'] == len(TEST_EDGES):
            print("✅ Data loaded correctly!")
            return True
        else:
            print("⚠️ Data count mismatch")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def run_sample_queries():
    """Run sample queries to demonstrate functionality"""
    print("\n🧪 Running sample queries...")
    
    # Query 1: Vector search
    print("\n1️⃣ Vector Search: 'redis caching'")
    try:
        response = requests.post(
            f"{API_URL}/search/vector",
            json={
                "query_text": "redis caching",
                "query_embedding": [0.88, 0.12, 0.02, 0, 0, 0],
                "top_k": 5
            }
        )
        
        if response.status_code == 200:
            results = response.json()['results']
            print("   Results:")
            for i, r in enumerate(results[:3], 1):
                print(f"   {i}. {r['id']} (score: {r['vector_score']:.6f})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Query 2: Graph traversal
    print("\n2️⃣ Graph Traversal: from doc6, depth=2")
    try:
        response = requests.get(
            f"{API_URL}/search/graph",
            params={"start_id": "doc6", "depth": 2}
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"   Found {len(results['nodes'])} reachable nodes:")
            for node in results['nodes'][:5]:
                print(f"   - {node['id']} (hop: {node['hop']})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Query 3: Hybrid search
    print("\n3️⃣ Hybrid Search: 'redis caching' (vector: 0.6, graph: 0.4)")
    try:
        response = requests.post(
            f"{API_URL}/search/hybrid",
            json={
                "query_text": "redis caching",
                "query_embedding": [0.88, 0.12, 0.02, 0, 0, 0],
                "vector_weight": 0.6,
                "graph_weight": 0.4,
                "top_k": 5
            }
        )
        
        if response.status_code == 200:
            results = response.json()['results']
            print("   Results:")
            for i, r in enumerate(results[:3], 1):
                print(f"   {i}. {r['id']}")
                print(f"      Final: {r['final_score']:.6f} (V: {r['vector_score']:.4f}, G: {r['graph_score']:.4f})")
    except Exception as e:
        print(f"   Error: {e}")

def main():
    """Main execution"""
    print("=" * 70)
    print("  CANONICAL TEST DATASET LOADER")
    print("  DevForge Hackathon - Vector + Graph Hybrid Database")
    print("=" * 70)
    
    # Check API availability
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code != 200:
            print("\n❌ API not available. Start the backend first:")
            print("   python main.py")
            return
    except:
        print("\n❌ Cannot connect to API at", API_URL)
        print("   Make sure the backend is running: python main.py")
        return
    
    # Load data
    if not reset_system():
        print("\n❌ Failed to reset system. Aborting.")
        return
    
    if not create_nodes():
        print("\n⚠️ Some nodes failed to create")
    
    if not create_edges():
        print("\n⚠️ Some edges failed to create")
    
    # Verify
    if not verify_data():
        print("\n⚠️ Data verification failed")
        return
    
    # Run samples
    run_sample_queries()
    
    print("\n" + "=" * 70)
    print("✅ TEST DATA LOADED SUCCESSFULLY!")
    print("=" * 70)
    print("\nYou can now:")
    print("  1. Run the UI: streamlit run app.py")
    print("  2. Run test suite: pytest test_suite.py")
    print("  3. Test queries manually via API docs: http://localhost:8000/docs")
    print("\n")

if __name__ == "__main__":
    main()