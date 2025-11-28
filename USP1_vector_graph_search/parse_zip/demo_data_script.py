import requests
import json
from typing import Dict, List

# API base URL
BASE_URL = "http://localhost:8000"

def create_node(text: str, metadata: dict) -> str:
    """Create a node and return its ID"""
    response = requests.post(f"{BASE_URL}/nodes", json={
        "text": text,
        "metadata": metadata
    })
    return response.json()["id"]

def create_edge(source_id: str, target_id: str, relationship_type: str, weight: float = 1.0):
    """Create an edge between two nodes"""
    requests.post(f"{BASE_URL}/edges", json={
        "source_id": source_id,
        "target_id": target_id,
        "relationship_type": relationship_type,
        "weight": weight
    })

def populate_research_paper_dataset():
    """
    Create a realistic research paper knowledge graph
    This demonstrates the power of hybrid search!
    """
    print("🚀 Starting demo data population...")
    print("=" * 60)
    
    # Reset database
    requests.post(f"{BASE_URL}/reset")
    print("✅ Database reset")
    
    node_ids = {}
    
    # ========================================================================
    # CREATE INSTITUTIONS
    # ========================================================================
    print("\n📚 Creating Institutions...")
    
    node_ids["stanford"] = create_node(
        "Stanford University",
        {"type": "institution", "location": "California", "ranking": 1}
    )
    
    node_ids["mit"] = create_node(
        "Massachusetts Institute of Technology",
        {"type": "institution", "location": "Massachusetts", "ranking": 2}
    )
    
    node_ids["berkeley"] = create_node(
        "UC Berkeley",
        {"type": "institution", "location": "California", "ranking": 3}
    )
    
    print(f"   ✓ Created 3 institutions")
    
    # ========================================================================
    # CREATE AUTHORS
    # ========================================================================
    print("\n👤 Creating Authors...")
    
    node_ids["andrew"] = create_node(
        "Dr. Andrew Martinez - Expert in Neural Architecture Search and AutoML",
        {"type": "author", "field": "AutoML", "h_index": 45}
    )
    
    node_ids["sarah"] = create_node(
        "Prof. Sarah Chen - Leading researcher in Deep Learning and Computer Vision",
        {"type": "author", "field": "Computer Vision", "h_index": 52}
    )
    
    node_ids["james"] = create_node(
        "Dr. James Kumar - Specialist in Natural Language Processing and Transformers",
        {"type": "author", "field": "NLP", "h_index": 38}
    )
    
    node_ids["emily"] = create_node(
        "Prof. Emily Wang - Pioneer in Reinforcement Learning and Robotics",
        {"type": "author", "field": "Reinforcement Learning", "h_index": 41}
    )
    
    node_ids["david"] = create_node(
        "Dr. David Patel - Expert in Graph Neural Networks and Knowledge Graphs",
        {"type": "author", "field": "Graph ML", "h_index": 35}
    )
    
    print(f"   ✓ Created 5 authors")
    
    # ========================================================================
    # CREATE AUTHOR-INSTITUTION RELATIONSHIPS
    # ========================================================================
    print("\n🔗 Linking Authors to Institutions...")
    
    create_edge(node_ids["andrew"], node_ids["stanford"], "affiliated_with", weight=1.0)
    create_edge(node_ids["sarah"], node_ids["mit"], "affiliated_with", weight=1.0)
    create_edge(node_ids["james"], node_ids["stanford"], "affiliated_with", weight=1.0)
    create_edge(node_ids["emily"], node_ids["berkeley"], "affiliated_with", weight=1.0)
    create_edge(node_ids["david"], node_ids["mit"], "affiliated_with", weight=1.0)
    
    print(f"   ✓ Created 5 affiliations")
    
    # ========================================================================
    # CREATE RESEARCH PAPERS
    # ========================================================================
    print("\n📄 Creating Research Papers...")
    
    # Stanford - AutoML Papers
    node_ids["paper1"] = create_node(
        "Efficient Neural Architecture Search using Reinforcement Learning. "
        "This paper introduces a novel approach to automatically design neural network architectures "
        "using policy gradient methods, achieving state-of-the-art results on ImageNet with 50% less computation.",
        {"type": "paper", "year": 2023, "citations": 156, "venue": "NeurIPS"}
    )
    
    node_ids["paper2"] = create_node(
        "AutoML-Zero: Evolving Machine Learning Algorithms from Scratch. "
        "We demonstrate that machine learning algorithms can be discovered automatically through "
        "evolutionary search, starting from basic mathematical operations.",
        {"type": "paper", "year": 2023, "citations": 203, "venue": "ICML"}
    )
    
    # MIT - Computer Vision Papers
    node_ids["paper3"] = create_node(
        "Vision Transformers for Dense Prediction Tasks. "
        "This work adapts the transformer architecture for pixel-level prediction tasks like "
        "semantic segmentation and depth estimation, outperforming CNNs on multiple benchmarks.",
        {"type": "paper", "year": 2024, "citations": 89, "venue": "CVPR"}
    )
    
    node_ids["paper4"] = create_node(
        "Self-Supervised Learning for Medical Image Analysis. "
        "We propose a contrastive learning framework specifically designed for medical imaging, "
        "reducing the need for expensive labeled data in healthcare applications.",
        {"type": "paper", "year": 2024, "citations": 67, "venue": "MICCAI"}
    )
    
    # Stanford - NLP Papers
    node_ids["paper5"] = create_node(
        "Scaling Language Models: Methods and Analysis. "
        "This paper investigates scaling laws for large language models and proposes efficient "
        "training techniques that reduce computational costs by 40% without sacrificing performance.",
        {"type": "paper", "year": 2023, "citations": 421, "venue": "ACL"}
    )
    
    node_ids["paper6"] = create_node(
        "Context-Aware Prompt Engineering for Few-Shot Learning. "
        "We introduce a systematic approach to prompt design that improves few-shot learning "
        "performance across diverse NLP tasks using in-context learning.",
        {"type": "paper", "year": 2024, "citations": 134, "venue": "EMNLP"}
    )
    
    # Berkeley - Reinforcement Learning Papers
    node_ids["paper7"] = create_node(
        "Sample-Efficient Reinforcement Learning for Robotics. "
        "This work presents a model-based RL algorithm that learns robotic manipulation tasks "
        "from fewer than 100 real-world trials, bridging the sim-to-real gap.",
        {"type": "paper", "year": 2024, "citations": 78, "venue": "RSS"}
    )
    
    node_ids["paper8"] = create_node(
        "Multi-Agent Reinforcement Learning with Communication. "
        "We develop a framework for emergent communication between RL agents, enabling cooperative "
        "behavior in complex multi-agent environments without explicit coordination protocols.",
        {"type": "paper", "year": 2023, "citations": 112, "venue": "ICLR"}
    )
    
    # MIT - Graph ML Papers
    node_ids["paper9"] = create_node(
        "Graph Neural Networks for Knowledge Graph Reasoning. "
        "This paper introduces a novel GNN architecture for multi-hop reasoning over knowledge graphs, "
        "achieving human-level performance on complex question answering tasks.",
        {"type": "paper", "year": 2024, "citations": 95, "venue": "KDD"}
    )
    
    node_ids["paper10"] = create_node(
        "Hybrid Vector-Graph Representations for Information Retrieval. "
        "We propose combining vector embeddings with graph structures for retrieval systems, "
        "demonstrating significant improvements in relevance and ranking quality.",
        {"type": "paper", "year": 2024, "citations": 43, "venue": "SIGIR"}
    )
    
    print(f"   ✓ Created 10 research papers")
    
    # ========================================================================
    # CREATE AUTHORSHIP RELATIONSHIPS
    # ========================================================================
    print("\n✍️ Linking Papers to Authors...")
    
    # Paper 1 & 2 by Andrew (Stanford - AutoML)
    create_edge(node_ids["paper1"], node_ids["andrew"], "authored_by", weight=1.0)
    create_edge(node_ids["paper2"], node_ids["andrew"], "authored_by", weight=1.0)
    
    # Paper 3 & 4 by Sarah (MIT - Vision)
    create_edge(node_ids["paper3"], node_ids["sarah"], "authored_by", weight=1.0)
    create_edge(node_ids["paper4"], node_ids["sarah"], "authored_by", weight=1.0)
    
    # Paper 5 & 6 by James (Stanford - NLP)
    create_edge(node_ids["paper5"], node_ids["james"], "authored_by", weight=1.0)
    create_edge(node_ids["paper6"], node_ids["james"], "authored_by", weight=1.0)
    
    # Paper 7 & 8 by Emily (Berkeley - RL)
    create_edge(node_ids["paper7"], node_ids["emily"], "authored_by", weight=1.0)
    create_edge(node_ids["paper8"], node_ids["emily"], "authored_by", weight=1.0)
    
    # Paper 9 & 10 by David (MIT - Graph ML)
    create_edge(node_ids["paper9"], node_ids["david"], "authored_by", weight=1.0)
    create_edge(node_ids["paper10"], node_ids["david"], "authored_by", weight=1.0)
    
    print(f"   ✓ Created 10 authorship links")
    
    # ========================================================================
    # CREATE CITATION RELATIONSHIPS
    # ========================================================================
    print("\n📚 Creating Citation Network...")
    
    # Paper 10 (Hybrid systems) cites relevant papers
    create_edge(node_ids["paper10"], node_ids["paper9"], "cites", weight=0.8)  # GNN paper
    create_edge(node_ids["paper10"], node_ids["paper3"], "cites", weight=0.6)  # Vision transformers
    
    # AutoML papers cite each other
    create_edge(node_ids["paper2"], node_ids["paper1"], "cites", weight=0.9)
    
    # NLP scaling paper cites AutoML
    create_edge(node_ids["paper5"], node_ids["paper1"], "cites", weight=0.7)
    
    # RL papers cite each other
    create_edge(node_ids["paper8"], node_ids["paper7"], "cites", weight=0.8)
    
    # Vision paper cites NLP (transformers)
    create_edge(node_ids["paper3"], node_ids["paper5"], "cites", weight=0.5)
    
    print(f"   ✓ Created 6 citation links")
    
    # ========================================================================
    # CREATE TOPIC/CONCEPT NODES
    # ========================================================================
    print("\n🏷️ Creating Topics and Concepts...")
    
    node_ids["automl"] = create_node(
        "Automated Machine Learning (AutoML)",
        {"type": "topic", "area": "ML Systems"}
    )
    
    node_ids["transformers"] = create_node(
        "Transformer Neural Networks",
        {"type": "topic", "area": "Deep Learning"}
    )
    
    node_ids["graph_learning"] = create_node(
        "Graph Machine Learning",
        {"type": "topic", "area": "Structured Data"}
    )
    
    node_ids["rl"] = create_node(
        "Reinforcement Learning",
        {"type": "topic", "area": "Decision Making"}
    )
    
    print(f"   ✓ Created 4 topics")
    
    # ========================================================================
    # LINK PAPERS TO TOPICS
    # ========================================================================
    print("\n🔗 Linking Papers to Topics...")
    
    # AutoML papers
    create_edge(node_ids["paper1"], node_ids["automl"], "belongs_to_topic", weight=1.0)
    create_edge(node_ids["paper2"], node_ids["automl"], "belongs_to_topic", weight=1.0)
    
    # Transformer papers
    create_edge(node_ids["paper3"], node_ids["transformers"], "belongs_to_topic", weight=1.0)
    create_edge(node_ids["paper5"], node_ids["transformers"], "belongs_to_topic", weight=0.8)
    create_edge(node_ids["paper6"], node_ids["transformers"], "belongs_to_topic", weight=0.9)
    
    # Graph ML papers
    create_edge(node_ids["paper9"], node_ids["graph_learning"], "belongs_to_topic", weight=1.0)
    create_edge(node_ids["paper10"], node_ids["graph_learning"], "belongs_to_topic", weight=1.0)
    
    # RL papers
    create_edge(node_ids["paper7"], node_ids["rl"], "belongs_to_topic", weight=1.0)
    create_edge(node_ids["paper8"], node_ids["rl"], "belongs_to_topic", weight=1.0)
    create_edge(node_ids["paper1"], node_ids["rl"], "belongs_to_topic", weight=0.6)  # Uses RL for NAS
    
    print(f"   ✓ Created 10 topic links")
    
    # ========================================================================
    # CREATE COLLABORATION RELATIONSHIPS
    # ========================================================================
    print("\n🤝 Creating Collaborations...")
    
    create_edge(node_ids["andrew"], node_ids["james"], "collaborates_with", weight=0.7)
    create_edge(node_ids["sarah"], node_ids["david"], "collaborates_with", weight=0.8)
    create_edge(node_ids["emily"], node_ids["andrew"], "collaborates_with", weight=0.5)
    
    print(f"   ✓ Created 3 collaborations")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 60)
    print("✅ DEMO DATA POPULATION COMPLETE!")
    print("=" * 60)
    
    # Get stats
    stats = requests.get(f"{BASE_URL}/stats").json()
    print(f"\n📊 Database Statistics:")
    print(f"   • Nodes: {stats['nodes']}")
    print(f"   • Edges: {stats['edges']}")
    print(f"   • Average Degree: {stats['avg_degree']:.2f}")
    print(f"   • Connected: {stats['is_connected']}")
    
    print("\n🎯 Sample Queries to Try:")
    print("   1. 'Neural architecture search papers by Stanford researchers'")
    print("   2. 'Graph neural networks and knowledge graphs'")
    print("   3. 'Transformer models for computer vision'")
    print("   4. 'Reinforcement learning research at Berkeley'")
    print("   5. 'AutoML papers that cite each other'")
    
    print("\n🚀 Ready for demo! Your hybrid database is loaded.\n")
    
    return node_ids

if __name__ == "__main__":
    try:
        node_ids = populate_research_paper_dataset()
        
        # Save node IDs for reference
        with open("node_ids.json", "w") as f:
            json.dump(node_ids, f, indent=2)
        print("💾 Node IDs saved to 'node_ids.json'")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the API server.")
        print("   Make sure the server is running: python main.py")
        print("   Then run this script again.\n")