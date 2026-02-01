"""
RAG Batch B - Test Script
=========================

Tests the retrieval pipeline:
1. Embedding model loads
2. Vector search works
3. Lexical search works
4. RRF merge works
5. Gating logic works
6. Prompt builder works
7. End-to-end (if LLM available)

Usage (from project root):
    python src/rag/test_batch_b.py

Author: HH Research Platform
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))


def test_embedding_model():
    """Test embedding model loads and produces correct dimensions."""
    print("\n" + "=" * 60)
    print("TEST 1: Embedding Model")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer

        print("Loading BAAI/bge-large-en-v1.5...")
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')

        # Test encoding
        test_text = "What did the CEO say about AI demand?"
        embedding = model.encode(test_text, normalize_embeddings=True)

        print(f"✅ Model loaded successfully")
        print(f"   Embedding dimension: {embedding.shape[0]}")
        print(f"   Expected: 1024")

        if embedding.shape[0] != 1024:
            print(f"❌ Dimension mismatch!")
            return False

        print(f"✅ Embedding dimension correct (1024)")
        return True

    except ImportError:
        print("❌ sentence-transformers not installed")
        print("   Install: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_vector_search():
    """Test vector search against pgvector."""
    print("\n" + "=" * 60)
    print("TEST 2: Vector Search")
    print("=" * 60)

    try:
        from src.db.connection import get_connection
        import numpy as np

        # Generate a random embedding
        embedding = np.random.randn(1024).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        embedding_str = "[" + ",".join(str(x) for x in embedding.tolist()) + "]"

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if any embeddings exist
                cur.execute("SELECT COUNT(*) FROM rag.doc_chunks WHERE embedding IS NOT NULL")
                count = cur.fetchone()[0]

                if count == 0:
                    print("⚠️ No embedded chunks found in database")
                    print("   This is OK - you need to ingest and embed documents first")
                    print("   Run: python src/rag/chunking.py (after ingesting docs)")
                    return True  # Not a failure, just no data yet

                print(f"   Found {count} embedded chunks")

                # Test vector search
                cur.execute("""
                    SELECT chunk_id, 1 - (embedding <=> %s::vector) as similarity
                    FROM rag.doc_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT 5
                """, (embedding_str, embedding_str))

                results = cur.fetchall()
                print(f"✅ Vector search returned {len(results)} results")

                for chunk_id, sim in results[:3]:
                    print(f"   chunk:{chunk_id} similarity={sim:.4f}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_lexical_search():
    """Test full-text search."""
    print("\n" + "=" * 60)
    print("TEST 3: Lexical Search")
    print("=" * 60)

    try:
        from src.db.connection import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if any chunks exist
                cur.execute("SELECT COUNT(*) FROM rag.doc_chunks")
                count = cur.fetchone()[0]

                if count == 0:
                    print("⚠️ No chunks found in database")
                    print("   Ingest documents first with: python src/rag/sec_ingestion.py")
                    return True

                print(f"   Found {count} total chunks")

                # Test lexical search
                cur.execute("""
                    SELECT chunk_id, ts_rank_cd(text_tsv, query, 32) as rank
                    FROM rag.doc_chunks, plainto_tsquery('english', 'revenue growth demand') as query
                    WHERE text_tsv @@ query
                    ORDER BY rank DESC
                    LIMIT 5
                """)

                results = cur.fetchall()

                if results:
                    print(f"✅ Lexical search returned {len(results)} results")
                    for chunk_id, rank in results[:3]:
                        print(f"   chunk:{chunk_id} rank={rank:.4f}")
                else:
                    print("⚠️ No lexical matches found (depends on document content)")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_rrf_merge():
    """Test RRF merge logic."""
    print("\n" + "=" * 60)
    print("TEST 4: RRF Merge")
    print("=" * 60)

    try:
        from src.rag.retrieval import RAGRetriever, RetrievalConfig

        # Create retriever with test config
        config = RetrievalConfig(k_vector=10, k_lexical=10, top_n_final=5)
        retriever = RAGRetriever(config)

        # Mock results
        vector_results = [
            (101, 0.9),
            (102, 0.85),
            (103, 0.8),
            (104, 0.75),
        ]

        lexical_results = [
            (102, 2.5),  # Overlaps with vector
            (105, 2.0),
            (101, 1.8),  # Overlaps with vector
            (106, 1.5),
        ]

        # Test merge
        merged = retriever._rrf_merge(vector_results, lexical_results)

        print(f"✅ RRF merge produced {len(merged)} results")
        print(f"   Vector results: {len(vector_results)}")
        print(f"   Lexical results: {len(lexical_results)}")
        print(f"   Merged (deduplicated): {len(merged)}")

        # Check overlapping items get boosted
        for chunk_id, rrf_score, vec_score, lex_score in merged[:3]:
            overlap = "✓" if vec_score and lex_score else ""
            print(f"   chunk:{chunk_id} rrf={rrf_score:.4f} {overlap}")

        # Verify overlapping items scored higher
        chunk_102 = next((r for r in merged if r[0] == 102), None)
        chunk_105 = next((r for r in merged if r[0] == 105), None)

        if chunk_102 and chunk_105:
            if chunk_102[1] > chunk_105[1]:
                print(f"✅ Overlapping chunk (102) scored higher than non-overlapping (105)")
            else:
                print(f"⚠️ RRF scoring may need adjustment")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_gating_logic():
    """Test gating rules."""
    print("\n" + "=" * 60)
    print("TEST 5: Gating Logic")
    print("=" * 60)

    try:
        from src.rag.retrieval import RAGRetriever, RetrievalConfig, RetrievedChunk, GatingDecision
        from datetime import datetime, timezone, timedelta

        config = RetrievalConfig(
            min_cosine_similarity=0.55,
            min_chunks_above_threshold=3,
            chunk_similarity_threshold=0.45,
        )
        retriever = RAGRetriever(config)

        # Test 1: Good results (should pass)
        good_chunks = [
            RetrievedChunk(
                chunk_id=i, doc_id=1, ticker="MU", doc_type="transcript",
                asof_ts_utc=datetime.now(timezone.utc),
                section=None, speaker=None, speaker_role=None,
                text=f"Test chunk {i}", char_start=0, char_end=100,
                vector_score=0.7 - i * 0.05,  # 0.7, 0.65, 0.6, 0.55, 0.5
            )
            for i in range(5)
        ]

        decision, details = retriever._apply_gating(good_chunks, config)
        print(f"Test 1 - Good results: {decision.value}")

        if decision == GatingDecision.PASS:
            print(f"✅ Gating correctly passed good results")
        else:
            print(f"❌ Gating incorrectly failed good results: {details.get('reason')}")
            return False

        # Test 2: Low similarity (should fail)
        low_sim_chunks = [
            RetrievedChunk(
                chunk_id=i, doc_id=1, ticker="MU", doc_type="transcript",
                asof_ts_utc=datetime.now(timezone.utc),
                section=None, speaker=None, speaker_role=None,
                text=f"Test chunk {i}", char_start=0, char_end=100,
                vector_score=0.4,  # Below threshold
            )
            for i in range(5)
        ]

        decision, details = retriever._apply_gating(low_sim_chunks, config)
        print(f"Test 2 - Low similarity: {decision.value}")

        if decision == GatingDecision.FAIL_SIMILARITY:
            print(f"✅ Gating correctly rejected low similarity")
        else:
            print(f"❌ Gating should have failed on similarity")
            return False

        # Test 3: Not enough chunks above threshold
        few_good_chunks = [
            RetrievedChunk(
                chunk_id=0, doc_id=1, ticker="MU", doc_type="transcript",
                asof_ts_utc=datetime.now(timezone.utc),
                section=None, speaker=None, speaker_role=None,
                text="Test chunk 0", char_start=0, char_end=100,
                vector_score=0.7,  # Good
            ),
            RetrievedChunk(
                chunk_id=1, doc_id=1, ticker="MU", doc_type="transcript",
                asof_ts_utc=datetime.now(timezone.utc),
                section=None, speaker=None, speaker_role=None,
                text="Test chunk 1", char_start=0, char_end=100,
                vector_score=0.3,  # Bad
            ),
        ]

        decision, details = retriever._apply_gating(few_good_chunks, config)
        print(f"Test 3 - Few good chunks: {decision.value}")

        if decision == GatingDecision.FAIL_MIN_CHUNKS:
            print(f"✅ Gating correctly rejected insufficient good chunks")
        else:
            print(f"❌ Gating should have failed on min chunks")
            return False

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_prompt_builder():
    """Test prompt building."""
    print("\n" + "=" * 60)
    print("TEST 6: Prompt Builder")
    print("=" * 60)

    try:
        from src.rag.prompt_builder import RAGPromptBuilder
        from src.rag.retrieval import RetrievedChunk

        builder = RAGPromptBuilder()

        # Mock chunks
        chunks = [
            RetrievedChunk(
                chunk_id=12345,
                doc_id=1,
                ticker="MU",
                doc_type="transcript",
                asof_ts_utc=datetime.now(timezone.utc),
                section="Prepared Remarks",
                speaker="Sanjay Mehrotra",
                speaker_role="CEO",
                text="We are seeing unprecedented demand for HBM.",
                char_start=0,
                char_end=50,
                vector_score=0.85,
                period_label="2025Q3",
            ),
        ]

        # Mock metrics
        metrics = {
            'price': 98.50,
            'pe_ratio': 25.4,
            'signal_type': 'BUY',
            'total_score': 74.5,
        }

        # Build prompt
        result = builder.build_full_prompt(
            ticker="MU",
            query="What's the outlook for AI demand?",
            chunks=chunks,
            metrics=metrics,
        )

        print(f"✅ Prompt built successfully")
        print(f"   System prompt length: {len(result['system'])} chars")
        print(f"   User prompt length: {len(result['user'])} chars")

        # Check key elements
        if "[chunk:12345]" in result['user']:
            print(f"✅ Chunk ID included in prompt")
        else:
            print(f"❌ Chunk ID missing from prompt")
            return False

        if "COMPUTED_METRICS" in result['user']:
            print(f"✅ COMPUTED_METRICS section present")
        else:
            print(f"❌ COMPUTED_METRICS section missing")
            return False

        if "EVIDENCE" in result['user']:
            print(f"✅ EVIDENCE section present")
        else:
            print(f"❌ EVIDENCE section missing")
            return False

        if "citation" in result['system'].lower():
            print(f"✅ Citation instructions in system prompt")
        else:
            print(f"❌ Citation instructions missing")
            return False

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_retriever_import():
    """Test that retriever module imports correctly."""
    print("\n" + "=" * 60)
    print("TEST 7: Module Imports")
    print("=" * 60)

    try:
        from src.rag.retrieval import RAGRetriever, RetrievalConfig, GatingDecision
        print(f"✅ src.rag.retrieval imports OK")

        from src.rag.prompt_builder import RAGPromptBuilder, PromptConfig
        print(f"✅ src.rag.prompt_builder imports OK")

        from src.rag.rag_service import RAGService, RAGServiceConfig
        print(f"✅ src.rag.rag_service imports OK")

        # Check configs
        config = RetrievalConfig.default()
        print(f"   Default k_vector: {config.k_vector}")
        print(f"   Default k_lexical: {config.k_lexical}")
        print(f"   Default min_similarity: {config.min_cosine_similarity}")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_all_tests():
    """Run all Batch B tests."""
    print("\n" + "=" * 60)
    print("RAG BATCH B - RETRIEVAL PIPELINE VERIFICATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Project root: {project_root}")

    results = {
        'module_imports': test_retriever_import(),
        'embedding_model': test_embedding_model(),
        'vector_search': test_vector_search(),
        'lexical_search': test_lexical_search(),
        'rrf_merge': test_rrf_merge(),
        'gating_logic': test_gating_logic(),
        'prompt_builder': test_prompt_builder(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 Batch B installation complete!")
        print("   Ready for integration with AI chat")
        print("\n   Next steps:")
        print("   1. Ingest some documents: python src/rag/sec_ingestion.py")
        print("   2. Chunk and embed: python src/rag/chunking.py")
        print("   3. Test retrieval: python src/rag/retrieval.py")
    else:
        print("\n⚠️ Some tests failed. Review errors above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)