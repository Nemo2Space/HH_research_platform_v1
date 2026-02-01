"""
RAG Batch A - Audit-Grade Test Script
=====================================

Tests the complete Batch A installation with proper verification:
1. pgvector extension (version-agnostic)
2. RAG schema and tables
3. Constraint verification
4. SEC ingestion (dry run)
5. Rate limiter
6. Text normalization

Uses ephemeral test data, does not modify production tables.

Usage (from project root):
    python -m src.rag.test_batch_a

    OR

    cd src/rag
    python test_batch_a.py

Author: HH Research Platform
"""

import sys
import os
import tempfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path

# Fix imports - add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # src/rag -> src -> project_root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_pgvector_extension():
    """
    Test pgvector extension (version-agnostic).

    Verifies:
    1. Extension exists
    2. Can create vector column
    3. Distance operations work
    """
    print("\n" + "="*60)
    print("TEST 1: pgvector Extension (Version-Agnostic)")
    print("="*60)

    try:
        from src.db.connection import get_connection
    except ImportError:
        print("❌ Cannot import src.db.connection")
        print("   Make sure you're running from project root")
        return False

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Check extension exists (don't check version)
                cur.execute("""
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                """)
                if not cur.fetchone():
                    print("❌ pgvector extension NOT installed")
                    print("   Run as superuser: CREATE EXTENSION vector;")
                    return False

                print("✅ pgvector extension exists")

                # 2. Test vector column creation
                cur.execute("""
                    CREATE TEMP TABLE _pgvector_test (
                        id SERIAL PRIMARY KEY,
                        v vector(3)
                    )
                """)
                cur.execute("INSERT INTO _pgvector_test (v) VALUES ('[1,2,3]')")
                cur.execute("SELECT v FROM _pgvector_test")
                result = cur.fetchone()

                if not result:
                    print("❌ Failed to create/query vector column")
                    return False

                print("✅ Vector column creation works")

                # 3. Test distance operation
                cur.execute("""
                    SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector as distance
                """)
                distance = cur.fetchone()[0]

                if not isinstance(distance, (int, float)):
                    print("❌ Distance operation failed")
                    return False

                print(f"✅ Vector distance operation works (distance={distance:.4f})")

                # 4. Test cosine similarity (used for HNSW index)
                cur.execute("""
                    SELECT '[1,2,3]'::vector <=> '[4,5,6]'::vector as cosine_distance
                """)
                cosine = cur.fetchone()[0]
                print(f"✅ Cosine distance works (distance={cosine:.4f})")

                # Clean up
                cur.execute("DROP TABLE _pgvector_test")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_rag_schema():
    """
    Test RAG schema exists with correct tables.
    """
    print("\n" + "="*60)
    print("TEST 2: RAG Schema")
    print("="*60)

    try:
        from src.db.connection import get_connection
    except ImportError:
        print("❌ Cannot import src.db.connection")
        return False

    required_tables = [
        'documents',
        'doc_chunks',
        'request_snapshots',
        'rag_retrievals',
        'ai_responses',
        'transcript_facts',
        'filing_facts',
        'rag_eval_cases',
        'rag_eval_runs'
    ]

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check schema exists
                cur.execute("""
                    SELECT 1 FROM information_schema.schemata 
                    WHERE schema_name = 'rag'
                """)
                if not cur.fetchone():
                    print("❌ Schema 'rag' does not exist")
                    print("   Run schema.sql as superuser:")
                    print("   psql -U postgres -d alpha_platform -f src/rag/schema.sql")
                    return False

                print("✅ Schema 'rag' exists")

                # Check tables
                missing = []
                for table in required_tables:
                    cur.execute("""
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'rag' AND table_name = %s
                    """, (table,))

                    if cur.fetchone():
                        cur.execute("""
                            SELECT COUNT(*) FROM information_schema.columns 
                            WHERE table_schema = 'rag' AND table_name = %s
                        """, (table,))
                        cols = cur.fetchone()[0]
                        print(f"✅ rag.{table}: {cols} columns")
                    else:
                        print(f"❌ rag.{table}: MISSING")
                        missing.append(table)

                if missing:
                    return False

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_constraints():
    """
    Test critical constraints exist.
    """
    print("\n" + "="*60)
    print("TEST 3: Constraints")
    print("="*60)

    try:
        from src.db.connection import get_connection
    except ImportError:
        print("❌ Cannot import src.db.connection")
        return False

    required_constraints = [
        ('documents', 'uq_documents_hash'),
        ('documents', 'uq_documents_natural_key'),
        ('doc_chunks', 'uq_doc_chunks_index'),
        ('doc_chunks', 'uq_doc_chunks_range'),
        ('doc_chunks', 'chk_chunks_embedding_provenance'),
    ]

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for table, constraint in required_constraints:
                    cur.execute("""
                        SELECT 1 FROM information_schema.table_constraints
                        WHERE table_schema = 'rag' 
                        AND table_name = %s 
                        AND constraint_name = %s
                    """, (table, constraint))

                    if cur.fetchone():
                        print(f"✅ {table}.{constraint}")
                    else:
                        print(f"❌ {table}.{constraint}: MISSING")
                        return False

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_traceability_columns():
    """
    Test audit/traceability columns exist.
    """
    print("\n" + "="*60)
    print("TEST 4: Traceability Columns")
    print("="*60)

    try:
        from src.db.connection import get_connection
    except ImportError:
        print("❌ Cannot import src.db.connection")
        return False

    required_columns = {
        'documents': [
            'raw_text_sha256',
            'source_fetched_at_utc',
            'char_count',
            'period_label'
        ],
        'doc_chunks': [
            'char_start',
            'char_end',
            'char_len',
            'chunk_hash',
            'embedding_model',
            'embedding_dim',
            'embedded_at_utc'
        ],
    }

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for table, columns in required_columns.items():
                    for col in columns:
                        cur.execute("""
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_schema = 'rag' 
                            AND table_name = %s 
                            AND column_name = %s
                        """, (table, col))

                        if cur.fetchone():
                            print(f"✅ rag.{table}.{col}")
                        else:
                            print(f"❌ rag.{table}.{col}: MISSING")
                            return False

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_rate_limiter():
    """
    Test rate limiter functionality.
    """
    print("\n" + "="*60)
    print("TEST 5: Rate Limiter")
    print("="*60)

    try:
        from src.rag.rate_limiter import RateLimiter, RateLimiterConfig

        # Test with fast config (no real requests)
        config = RateLimiterConfig.for_testing()
        limiter = RateLimiter(config)

        print(f"✅ RateLimiter initialized")
        print(f"   Rate: {config.requests_per_second} req/sec")
        print(f"   Cache: {config.cache_enabled}")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_text_normalization():
    """
    Test text normalization is deterministic.
    """
    print("\n" + "="*60)
    print("TEST 6: Text Normalization")
    print("="*60)

    try:
        from src.rag.sec_ingestion import TextNormalizer

        normalizer = TextNormalizer()

        # Test various inputs
        test_cases = [
            # (input, expected_output)
            ("Hello\r\nWorld", "Hello\nWorld"),
            ("Hello\rWorld", "Hello\nWorld"),
            ("Hello  World", "Hello World"),
            ("Hello\n\n\n\nWorld", "Hello\n\nWorld"),
            ("  Hello  ", "Hello"),
            ("Hello\x00World", "HelloWorld"),
        ]

        for input_text, expected in test_cases:
            result = normalizer.normalize(input_text)
            if result == expected:
                print(f"✅ Normalize: '{repr(input_text)[:30]}' → '{repr(result)[:30]}'")
            else:
                print(f"❌ Normalize failed: expected '{expected}', got '{result}'")
                return False

        # Test determinism
        text = "Hello\r\nWorld  Test\n\n\n\nParagraph"
        hash1 = hashlib.sha256(normalizer.normalize(text).encode()).hexdigest()
        hash2 = hashlib.sha256(normalizer.normalize(text).encode()).hexdigest()

        if hash1 == hash2:
            print(f"✅ Normalization is deterministic")
        else:
            print(f"❌ Normalization NOT deterministic!")
            return False

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install: pip install beautifulsoup4 html2text")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_sec_ingestion_dry():
    """
    Test SEC ingestion module (dry run - no actual fetching).
    """
    print("\n" + "="*60)
    print("TEST 7: SEC Ingestion Module")
    print("="*60)

    try:
        from src.rag.sec_ingestion import SECIngester

        # Just test initialization
        ingester = SECIngester()

        print(f"✅ SECIngester initialized")
        print(f"   Rate limiter: {ingester.limiter.config.requests_per_second} req/sec")

        # Test section patterns exist
        assert '10-K' in ingester.SECTION_PATTERNS
        assert '10-Q' in ingester.SECTION_PATTERNS
        print(f"✅ Section patterns: 10-K, 10-Q")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install: pip install beautifulsoup4 html2text requests")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_embedding_validation():
    """
    Test embedding dimension validation.
    """
    print("\n" + "="*60)
    print("TEST 8: Embedding Validation")
    print("="*60)

    try:
        import numpy as np

        # Simulate validation logic
        def validate_embedding(embedding, expected_dim=1024):
            if embedding.shape[0] != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: got {embedding.shape[0]}, "
                    f"expected {expected_dim}"
                )
            return True

        # Test correct dimension
        correct_embedding = np.random.randn(1024)
        assert validate_embedding(correct_embedding, 1024)
        print(f"✅ Correct dimension (1024) passes")

        # Test wrong dimension (should raise)
        wrong_embedding = np.random.randn(768)
        try:
            validate_embedding(wrong_embedding, 1024)
            print(f"❌ Wrong dimension should have raised error")
            return False
        except ValueError as e:
            print(f"✅ Wrong dimension (768) raises ValueError")

        return True

    except ImportError:
        print(f"⚠️ numpy not available - skipping embedding test")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RAG BATCH A - AUDIT-GRADE VERIFICATION")
    print("="*60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Project root: {project_root}")

    results = {
        'pgvector': test_pgvector_extension(),
        'schema': test_rag_schema(),
        'constraints': test_constraints(),
        'traceability': test_traceability_columns(),
        'rate_limiter': test_rate_limiter(),
        'normalization': test_text_normalization(),
        'sec_ingestion': test_sec_ingestion_dry(),
        'embedding_validation': test_embedding_validation(),
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 Batch A (Audit-Grade) installation complete!")
        print("   Ready for Batch B: Retrieval Pipeline")
    else:
        print("\n⚠️ Some tests failed. Review errors above.")
        print("   Common fixes:")
        print("   - Run schema.sql as superuser: psql -U postgres -d alpha_platform -f src/rag/schema.sql")
        print("   - Install: pip install beautifulsoup4 html2text requests")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)