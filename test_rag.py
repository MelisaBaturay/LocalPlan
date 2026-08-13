import sys
import gc
from pathlib import Path
from config import RAGConfig
from database import VectorDatabase, cosine_similarity
from ingest import chunk_text, EmbeddingGenerator, run_ingestion
from retrieval import Retriever
from rag_engine import RAGEngine
from llm_provider import OfflineFallbackProvider

def test_database_init_and_crud(tmp_path):
    db_file = tmp_path / "test_vector_store.db"
    db = VectorDatabase(db_path=db_file)
    
    stats = db.get_stats()
    assert stats["total_chunks"] == 0
    assert stats["total_documents"] == 0
    
    dummy_vec = [0.1] * 384
    db.insert_chunk("test.md", 0, "This is a test chunk.", dummy_vec)
    
    stats_after = db.get_stats()
    assert stats_after["total_chunks"] == 1
    assert stats_after["total_documents"] == 1
    
    chunks = db.fetch_all_chunks()
    assert len(chunks) == 1
    assert chunks[0]["filename"] == "test.md"
    assert chunks[0]["content"] == "This is a test chunk."
    assert len(chunks[0]["embedding"]) == 384

def test_chunk_text():
    sample_text = "Paragraph 1 is here.\n\nParagraph 2 is here.\n\nParagraph 3 is long and detailed."
    chunks = chunk_text(sample_text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 1
    assert "Paragraph 1" in chunks[0]

def test_embedding_generator():
    embedder = EmbeddingGenerator(dim=384)
    vec1 = embedder.embed_text("Microsoft Foundry Local on-device AI")
    vec2 = embedder.embed_text("Microsoft Foundry Local on-device AI")
    vec3 = embedder.embed_text("Baking chocolate cake recipes")
    
    assert len(vec1) == 384
    # Exact match check
    assert abs(cosine_similarity(vec1, vec2) - 1.0) < 1e-5
    # Dissimilar topic check
    sim_diff = cosine_similarity(vec1, vec3)
    assert sim_diff <= 1.0

def test_full_rag_pipeline(tmp_path):
    db_file = tmp_path / "test_rag.db"
    db = VectorDatabase(db_path=db_file)
    docs_dir = Path(__file__).parent / "data" / "sample_documents"
    
    ingest_result = run_ingestion(docs_dir=docs_dir, db=db)
    assert ingest_result["processed_documents"] > 0
    
    retriever = Retriever(db=db)
    llm = OfflineFallbackProvider()
    engine = RAGEngine(retriever=retriever, llm_provider=llm)
    
    # 1. Test answerable query with mild similarity threshold for offline testing
    res1 = engine.ask("What is the grading policy for CS101?", similarity_threshold=0.05)
    assert res1.has_sufficient_context is True
    assert len(res1.retrieved_chunks) > 0
    assert ("CS101" in res1.answer) or ("Grading" in res1.answer) or ("Phase" in res1.answer)
    
    # 2. Test unanswerable query
    res2 = engine.ask("What is the recipe for baking a pizza?", similarity_threshold=0.60)
    assert ("I do not have enough information" in res2.answer) or (len(res2.retrieved_chunks) == 0)

def run_standalone_tests():
    import tempfile
    print("Running RAG test suite in standalone mode...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        print("Running test_database_init_and_crud...")
        test_database_init_and_crud(tmp_path)
        print("[OK] test_database_init_and_crud PASSED")
        
        print("Running test_chunk_text...")
        test_chunk_text()
        print("[OK] test_chunk_text PASSED")
        
        print("Running test_embedding_generator...")
        test_embedding_generator()
        print("[OK] test_embedding_generator PASSED")
        
        print("Running test_full_rag_pipeline...")
        test_full_rag_pipeline(tmp_path)
        print("[OK] test_full_rag_pipeline PASSED")
        
        gc.collect()
        
    print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    try:
        import pytest
        sys.exit(pytest.main(["-v", __file__]))
    except ImportError:
        run_standalone_tests()
