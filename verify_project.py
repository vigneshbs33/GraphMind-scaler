"""
Comprehensive verification script for GraphMind project.
Run this to check if all components are working correctly.
"""

import asyncio
import sys
from pathlib import Path

def check_imports():
    """Check if all required modules can be imported."""
    print("=" * 60)
    print("STEP 1: Checking Imports")
    print("=" * 60)
    
    try:
        from backend.config import settings
        print("[OK] Config module imported")
        
        from backend.models import NodeCreate, EdgeCreate, SearchQuery, SearchResult
        print("[OK] Models module imported")
        
        from backend.parsers import MultiFormatParser
        print("[OK] Parsers module imported")
        
        from backend.storage import GraphMindStorage
        print("[OK] Storage module imported")
        
        from backend.llm_processor import get_llm_processor
        print("[OK] LLM processor module imported")
        
        from backend.ingestion import ParallelIngestionPipeline
        print("[OK] Ingestion pipeline module imported")
        
        from backend.evaluation import SearchEvaluator
        print("[OK] Evaluation module imported")
        
        from backend.main import app
        print("[OK] FastAPI app imported")
        
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False

def check_config():
    """Check configuration settings."""
    print("\n" + "=" * 60)
    print("STEP 2: Checking Configuration")
    print("=" * 60)
    
    try:
        from backend.config import settings
        
        print(f"[OK] LLM Provider: {settings.LLM_PROVIDER}")
        print(f"[OK] Embedding Model: {settings.EMBEDDING_MODEL}")
        print(f"[OK] Data Directory: {settings.DATA_DIR}")
        print(f"[OK] Upload Directory exists: {settings.UPLOAD_DIR.exists()}")
        print(f"[OK] Chroma Directory exists: {settings.CHROMA_DIR.exists()}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Config error: {e}")
        return False

def check_storage():
    """Test storage operations."""
    print("\n" + "=" * 60)
    print("STEP 3: Testing Storage Layer")
    print("=" * 60)
    
    try:
        from backend.storage import GraphMindStorage
        from backend.config import settings
        
        storage = GraphMindStorage(settings.CHROMA_DIR, settings.EMBEDDING_MODEL)
        print("[OK] Storage initialized")
        
        # Test add_node
        node_id = storage.add_node("test_node_1", "Test content for verification", {"test": True})
        print(f"[OK] Node added: {node_id}")
        
        # Test add_edge (need two nodes)
        storage.add_node("test_node_2", "Another test node", {"test": True})
        storage.add_edge("test_node_1", "test_node_2", "related_to", 0.8)
        print("[OK] Edge added")
        
        # Test searches
        vector_results = storage.vector_search("test", 2)
        print(f"[OK] Vector search: {len(vector_results)} results")
        
        graph_results = storage.graph_search("test", 2)
        print(f"[OK] Graph search: {len(graph_results)} results")
        
        hybrid_results = storage.hybrid_search("test", 2, 0.6)
        print(f"[OK] Hybrid search: {len(hybrid_results)} results")
        
        # Test stats
        stats = storage.get_stats()
        print(f"[OK] Stats retrieved: {stats['node_count']} nodes, {stats['edge_count']} edges")
        
        return True
    except Exception as e:
        print(f"[FAIL] Storage error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_parsers():
    """Test file parsers."""
    print("\n" + "=" * 60)
    print("STEP 4: Testing File Parsers")
    print("=" * 60)
    
    try:
        from backend.parsers import MultiFormatParser
        from pathlib import Path
        
        # Create test file
        test_file = Path("verify_test.txt")
        test_file.write_text("Machine Learning is used in Healthcare. Deep Learning improves diagnosis.")
        
        # Test parsing
        result = MultiFormatParser.parse_file(test_file, "text")
        if result.get("error"):
            print(f"[FAIL] Parse error: {result['error']}")
            return False
        print("[OK] File parsed successfully")
        
        # Test entity extraction
        entities = MultiFormatParser.extract_entities(result["content"])
        print(f"[OK] Entities extracted: {entities}")
        
        # Test chunking
        chunks = MultiFormatParser.chunk_text(result["content"], 30)
        print(f"[OK] Text chunked: {len(chunks)} chunks")
        
        # Cleanup
        test_file.unlink()
        
        return True
    except Exception as e:
        print(f"[FAIL] Parser error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def check_ingestion():
    """Test ingestion pipeline."""
    print("\n" + "=" * 60)
    print("STEP 5: Testing Ingestion Pipeline")
    print("=" * 60)
    
    try:
        from backend.ingestion import ParallelIngestionPipeline
        from backend.storage import GraphMindStorage
        from backend.config import settings
        from pathlib import Path
        
        storage = GraphMindStorage(settings.CHROMA_DIR, settings.EMBEDDING_MODEL)
        pipeline = ParallelIngestionPipeline(storage)
        
        # Create test file
        test_file = Path("verify_ingestion.txt")
        test_file.write_text("Artificial Intelligence transforms modern Healthcare. Machine Learning algorithms help doctors make better decisions.")
        
        # Test ingestion
        result = await pipeline.ingest_file(test_file, "text", {"source": "verification"})
        
        if result.get("status") == "error":
            print(f"[FAIL] Ingestion error: {result.get('error')}")
            return False
        
        print(f"[OK] Ingestion successful:")
        print(f"   - Nodes created: {result.get('nodes_created', 0)}")
        print(f"   - Edges created: {result.get('edges_created', 0)}")
        print(f"   - Processing time: {result.get('processing_time', 0)}s")
        
        # Cleanup
        test_file.unlink()
        
        return True
    except Exception as e:
        print(f"[FAIL] Ingestion error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def check_llm():
    """Test LLM processor."""
    print("\n" + "=" * 60)
    print("STEP 6: Testing LLM Processor")
    print("=" * 60)
    
    try:
        from backend.llm_processor import get_llm_processor
        
        llm = get_llm_processor("mock")
        
        # Test query understanding
        analysis = await llm.understand_query("How is Machine Learning used in Healthcare?")
        print(f"[OK] Query analysis: {analysis.get('intent')}")
        
        # Test result refinement
        results = [
            {"node_id": "test1", "content": "Test content", "score": 0.9, "metadata": {}}
        ]
        answer = await llm.refine_results(results, "test query")
        print(f"[OK] Result refinement: {len(answer)} chars")
        
        return True
    except Exception as e:
        print(f"[FAIL] LLM error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_files():
    """Check if required files exist."""
    print("\n" + "=" * 60)
    print("STEP 7: Checking Required Files")
    print("=" * 60)
    
    required_files = [
        "backend/config.py",
        "backend/models.py",
        "backend/parsers.py",
        "backend/storage.py",
        "backend/llm_processor.py",
        "backend/ingestion.py",
        "backend/evaluation.py",
        "backend/main.py",
        "requirements.txt",
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"[OK] {file_path}")
        else:
            print(f"[FAIL] {file_path} - MISSING")
            all_exist = False
    
    # Check optional files
    optional_files = {
        "frontend/index.html": "Frontend UI",
        "README.md": "Documentation",
        ".gitignore": "Git ignore file",
    }
    
    print("\nOptional files:")
    for file_path, desc in optional_files.items():
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            if size > 0:
                print(f"[OK] {file_path} ({desc})")
            else:
                print(f"[WARN] {file_path} ({desc}) - EMPTY")
        else:
            print(f"[MISSING] {file_path} ({desc}) - MISSING")
    
    return all_exist

def check_api_endpoints():
    """Check if API endpoints are defined."""
    print("\n" + "=" * 60)
    print("STEP 8: Checking API Endpoints")
    print("=" * 60)
    
    try:
        from backend.main import app
        
        routes = [route.path for route in app.routes]
        expected_routes = [
            "/health",
            "/stats",
            "/nodes",
            "/edges",
            "/graph",
            "/search",
            "/compare",
            "/ingest",
        ]
        
        all_present = True
        for route in expected_routes:
            if route in routes:
                print(f"[OK] {route}")
            else:
                print(f"[FAIL] {route} - MISSING")
                all_present = False
        
        return all_present
    except Exception as e:
        print(f"[FAIL] API check error: {e}")
        return False

async def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("GraphMind Project Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", check_imports()))
    results.append(("Configuration", check_config()))
    results.append(("Storage", check_storage()))
    results.append(("Parsers", check_parsers()))
    results.append(("Ingestion", await check_ingestion()))
    results.append(("LLM Processor", await check_llm()))
    results.append(("Files", check_files()))
    results.append(("API Endpoints", check_api_endpoints()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n[SUCCESS] All core components are working!")
        print("\nNext steps:")
        print("1. Start the API: uvicorn backend.main:app --reload")
        print("2. Open http://localhost:8000/docs for API documentation")
        print("3. Complete frontend/index.html for UI")
        print("4. Add demo data loader")
        return 0
    else:
        print("\n[WARNING] Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

