#!/usr/bin/env python3
"""
Example script demonstrating how to use the Faiss vector store skill.

This example shows:
1. How to configure the Faiss vector store skill
2. How to create and store document embeddings
3. How to perform similarity search
4. Different index types (flat, IVF, HNSW) and their use cases

To integrate Faiss into the main system, you need to:
1. Add FAISS = "faiss" to the AvailableSkillName enum in factory.py
2. Add the import: from docs2vecs.subcommands.indexer.skills.faiss_vector_store_skill import FaissVectorStoreSkill
3. Add to the SKILLS_REGISTRY in factory.py under SkillType.VECTOR_STORE
"""

import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add the src directory to the path to import docs2vecs modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docs2vecs.subcommands.indexer.skills.faiss_vector_store_skill import FaissVectorStoreSkill
from docs2vecs.subcommands.indexer.config.config import Config
from docs2vecs.subcommands.indexer.document.document import Document


def create_sample_documents():
    """Create sample documents with embeddings for testing."""
    documents = []
    
    # Sample document contents and their embeddings (using random embeddings for demo)
    sample_data = [
        ("Machine learning is a subset of artificial intelligence.", np.random.rand(768).tolist()),
        ("Deep learning uses neural networks with multiple layers.", np.random.rand(768).tolist()),
        ("Python is a popular programming language for data science.", np.random.rand(768).tolist()),
        ("Vector databases are used for similarity search.", np.random.rand(768).tolist()),
        ("FAISS is a library for efficient similarity search and clustering.", np.random.rand(768).tolist()),
    ]
    
    for i, (content, embedding) in enumerate(sample_data):
        doc = Document(
            page_content=content,
            metadata={
                "source": f"sample_doc_{i}.txt",
                "chunk_id": f"chunk_{i}",
                "document_id": f"doc_{i}",
                "topic": "AI/ML" if i < 2 else "Programming" if i == 2 else "Databases"
            }
        )
        doc.embedding = embedding
        documents.append(doc)
    
    return documents


def demonstrate_faiss_flat_index():
    """Demonstrate Faiss with flat (exact search) index."""
    print("=== Faiss Flat Index Example ===")
    
    # Create temporary directory for this example
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Configuration for flat index (exact search)
        config = {
            "index_path": str(Path(temp_dir) / "faiss_flat.index"),
            "metadata_path": str(Path(temp_dir) / "faiss_flat_metadata.json"),
            "index_type": "flat",
            "dimension": 768
        }
        
        # Create global config (minimal for this example)
        global_config = Config({})
        
        # Initialize Faiss skill
        faiss_skill = FaissVectorStoreSkill(config, global_config)
        
        # Create sample documents
        documents = create_sample_documents()
        
        # Store documents in Faiss
        print(f"Storing {len(documents)} documents...")
        result = faiss_skill.run(documents)
        
        # Show index statistics
        stats = faiss_skill.get_index_stats()
        print(f"Index stats: {stats}")
        
        # Perform similarity search
        query_embedding = np.random.rand(768).tolist()
        search_results = faiss_skill.search(query_embedding, k=3)
        
        print(f"\nSearch results for random query:")
        for i, result in enumerate(search_results):
            print(f"  {i+1}. Score: {result['score']:.4f}")
            print(f"     Content: {result['document']['content'][:100]}...")
            print(f"     Source: {result['document']['source']}")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def demonstrate_faiss_ivf_index():
    """Demonstrate Faiss with IVF (approximate search) index."""
    print("\n=== Faiss IVF Index Example ===")
    
    # Create temporary directory for this example
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Configuration for IVF index (approximate search)
        config = {
            "index_path": str(Path(temp_dir) / "faiss_ivf.index"),
            "metadata_path": str(Path(temp_dir) / "faiss_ivf_metadata.json"),
            "index_type": "ivf",
            "dimension": 768,
            "nlist": 10  # Number of clusters (smaller for demo with few vectors)
        }
        
        # Create global config
        global_config = Config({})
        
        # Initialize Faiss skill
        faiss_skill = FaissVectorStoreSkill(config, global_config)
        
        # Create more documents for IVF training
        documents = create_sample_documents()
        # Add more documents to have enough for training
        for i in range(15):  # Add 15 more documents
            doc = Document(
                page_content=f"Additional document {i} for training purposes.",
                metadata={
                    "source": f"training_doc_{i}.txt",
                    "chunk_id": f"train_chunk_{i}",
                }
            )
            doc.embedding = np.random.rand(768).tolist()
            documents.append(doc)
        
        # Store documents in Faiss
        print(f"Storing {len(documents)} documents...")
        faiss_skill.run(documents)
        
        # Show index statistics
        stats = faiss_skill.get_index_stats()
        print(f"Index stats: {stats}")
        
        # Perform similarity search
        query_embedding = np.random.rand(768).tolist()
        search_results = faiss_skill.search(query_embedding, k=3)
        
        print(f"\nSearch results:")
        for i, result in enumerate(search_results):
            print(f"  {i+1}. Score: {result['score']:.4f}")
            print(f"     Content: {result['document']['content'][:100]}...")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def demonstrate_faiss_hnsw_index():
    """Demonstrate Faiss with HNSW index."""
    print("\n=== Faiss HNSW Index Example ===")
    
    # Create temporary directory for this example
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Configuration for HNSW index
        config = {
            "index_path": str(Path(temp_dir) / "faiss_hnsw.index"),
            "metadata_path": str(Path(temp_dir) / "faiss_hnsw_metadata.json"),
            "index_type": "hnsw",
            "dimension": 768,
            "m": 16,  # Number of connections per layer
            "ef_construction": 200  # Size of dynamic candidate list
        }
        
        # Create global config
        global_config = Config({})
        
        # Initialize Faiss skill
        faiss_skill = FaissVectorStoreSkill(config, global_config)
        
        # Create sample documents
        documents = create_sample_documents()
        
        # Store documents in Faiss
        print(f"Storing {len(documents)} documents...")
        faiss_skill.run(documents)
        
        # Show index statistics
        stats = faiss_skill.get_index_stats()
        print(f"Index stats: {stats}")
        
        # Perform similarity search
        query_embedding = np.random.rand(768).tolist()
        search_results = faiss_skill.search(query_embedding, k=3)
        
        print(f"\nSearch results:")
        for i, result in enumerate(search_results):
            print(f"  {i+1}. Score: {result['score']:.4f}")
            print(f"     Content: {result['document']['content'][:100]}...")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def integration_instructions():
    """Print instructions for integrating Faiss into the main system."""
    print("\n" + "="*60)
    print("INTEGRATION INSTRUCTIONS")
    print("="*60)
    print("""
To integrate the Faiss vector store skill into the main docs2vecs system:

1. Add to factory.py in AvailableSkillName enum:
   FAISS = "faiss"

2. Add import to factory.py:
   from docs2vecs.subcommands.indexer.skills.faiss_vector_store_skill import FaissVectorStoreSkill

3. Add to SKILLS_REGISTRY in factory.py under SkillType.VECTOR_STORE:
   AvailableSkillName.FAISS: FaissVectorStoreSkill,

4. Configuration example for config files:
   {
     "name": "faiss",
     "type": "vector-store", 
     "config": {
       "index_path": "path/to/faiss.index",
       "metadata_path": "path/to/metadata.json",
       "index_type": "flat",  // or "ivf", "hnsw"
       "dimension": 768,
       "nlist": 100,  // for IVF index
       "m": 64,       // for HNSW index
       "ef_construction": 200  // for HNSW index
     }
   }

5. Index Types:
   - "flat": Exact search, best accuracy, slower for large datasets
   - "ivf": Approximate search, good balance of speed and accuracy
   - "hnsw": Fast approximate search, good for real-time applications

6. Performance considerations:
   - Flat: Use for < 100K vectors or when exact results needed
   - IVF: Use for 100K - 10M vectors, requires training
   - HNSW: Use for fast queries on any size dataset
""")


def main():
    """Run all Faiss demonstrations."""
    print("Faiss Vector Store Skill Demonstration")
    print("="*50)
    
    try:
        demonstrate_faiss_flat_index()
        demonstrate_faiss_ivf_index() 
        demonstrate_faiss_hnsw_index()
        integration_instructions()
        
    except Exception as e:
        print(f"Error running demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
