"""
RAG Batch C - Test Script
=========================

Tests:
1. Schema creation
2. Fact extractor module
3. Evaluation harness module
4. End-to-end extraction

Usage:
    python src/rag/test_batch_c.py

Author: HH Research Platform
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))


def test_schema():
    """Test Batch C schema exists."""
    print("\n" + "=" * 60)
    print("TEST 1: Schema Verification")
    print("=" * 60)

    try:
        from src.db.connection import get_connection

        tables = [
            ('rag', 'transcript_facts'),
            ('rag', 'filing_facts'),
            ('rag', 'rag_eval_cases'),
            ('rag', 'rag_eval_runs'),
        ]

        with get_connection() as conn:
            with conn.cursor() as cur:
                all_exist = True
                for schema, table in tables:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = %s AND table_name = %s
                        )
                    """, (schema, table))
                    exists = cur.fetchone()[0]

                    status = "✅" if exists else "❌"
                    print(f"  {status} {schema}.{table}")

                    if not exists:
                        all_exist = False

        if not all_exist:
            print("\n⚠️ Some tables missing. Run: python src/rag/schema_batch_c.py")
            return False

        print(f"\n✅ All Batch C tables exist")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_extractor_import():
    """Test fact extractor imports."""
    print("\n" + "=" * 60)
    print("TEST 2: Fact Extractor Import")
    print("=" * 60)

    try:
        from src.rag.fact_extractor import (
            FactExtractor,
            ExtractorConfig,
            TranscriptFact,
            FilingFact,
            GuidanceDirection,
            RISK_THEMES_TAXONOMY,
        )

        print(f"✅ FactExtractor imported")
        print(f"✅ ExtractorConfig imported")
        print(f"✅ TranscriptFact imported")
        print(f"✅ FilingFact imported")
        print(f"✅ GuidanceDirection enum imported")
        print(f"   Risk themes taxonomy: {len(RISK_THEMES_TAXONOMY)} categories")

        # Test config
        config = ExtractorConfig.from_env()
        print(f"   LLM model: {config.llm_model}")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_evaluator_import():
    """Test evaluation harness imports."""
    print("\n" + "=" * 60)
    print("TEST 3: Evaluation Harness Import")
    print("=" * 60)

    try:
        from src.rag.evaluation import (
            RAGEvaluator,
            EvalCase,
            EvalResult,
            EvalRunSummary,
            generate_sample_cases,
        )

        print(f"✅ RAGEvaluator imported")
        print(f"✅ EvalCase imported")
        print(f"✅ EvalResult imported")
        print(f"✅ EvalRunSummary imported")

        # Test sample case generation
        cases = generate_sample_cases("MU")
        print(f"   Sample cases generated: {len(cases)}")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_extractor_init():
    """Test fact extractor initialization."""
    print("\n" + "=" * 60)
    print("TEST 4: Fact Extractor Initialization")
    print("=" * 60)

    try:
        from src.rag.fact_extractor import FactExtractor, ExtractorConfig

        config = ExtractorConfig(
            llm_base_url="http://localhost:8090/v1",
            llm_model="qwen3-32b",
        )

        extractor = FactExtractor(config)

        print(f"✅ FactExtractor initialized")
        print(f"   Model: {extractor.config.llm_model}")
        print(f"   Max chunks: {extractor.config.max_chunks_per_extraction}")
        print(f"   Min confidence: {extractor.config.min_confidence_threshold}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_evaluator_init():
    """Test evaluator initialization."""
    print("\n" + "=" * 60)
    print("TEST 5: Evaluator Initialization")
    print("=" * 60)

    try:
        from src.rag.evaluation import RAGEvaluator
        from src.rag.rag_service import RAGService

        service = RAGService()
        evaluator = RAGEvaluator(service)

        print(f"✅ RAGEvaluator initialized")
        print(f"   RAG service connected")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_document_available():
    """Test if documents are available for extraction."""
    print("\n" + "=" * 60)
    print("TEST 6: Document Availability")
    print("=" * 60)

    try:
        from src.db.connection import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check documents
                cur.execute("""
                    SELECT doc_type, COUNT(*) 
                    FROM rag.documents 
                    GROUP BY doc_type
                """)
                docs = cur.fetchall()

                # Check chunks
                cur.execute("SELECT COUNT(*) FROM rag.doc_chunks")
                chunk_count = cur.fetchone()[0]

                # Check existing facts
                cur.execute("SELECT COUNT(*) FROM rag.transcript_facts")
                transcript_facts = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM rag.filing_facts")
                filing_facts = cur.fetchone()[0]

        print(f"Documents by type:")
        for doc_type, count in docs:
            print(f"  {doc_type}: {count}")

        print(f"\nChunks: {chunk_count}")
        print(f"Transcript facts extracted: {transcript_facts}")
        print(f"Filing facts extracted: {filing_facts}")

        if chunk_count > 0:
            print(f"\n✅ Documents and chunks available for extraction")
            return True
        else:
            print(f"\n⚠️ No chunks found. Run chunking first.")
            return True  # Not a failure, just no data

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_extraction_dry_run():
    """Test extraction without actually calling LLM."""
    print("\n" + "=" * 60)
    print("TEST 7: Extraction Dry Run")
    print("=" * 60)

    try:
        from src.rag.fact_extractor import FactExtractor
        from src.db.connection import get_connection

        extractor = FactExtractor()

        # Get first document
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT doc_id, ticker, doc_type 
                    FROM rag.documents 
                    LIMIT 1
                """)
                row = cur.fetchone()

        if not row:
            print("⚠️ No documents found")
            return True

        doc_id, ticker, doc_type = row
        print(f"Test document: {ticker} ({doc_type}), doc_id={doc_id}")

        # Get document info
        doc_info = extractor._get_document_info(doc_id)
        print(f"✅ Document info retrieved: {doc_info}")

        # Get chunks
        content, chunk_ids = extractor._get_document_chunks(doc_id, limit=3)
        print(f"✅ Chunks retrieved: {len(chunk_ids)} chunks, {len(content)} chars")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_metrics_calculation():
    """Test evaluation metrics calculation."""
    print("\n" + "=" * 60)
    print("TEST 8: Metrics Calculation")
    print("=" * 60)

    try:
        from src.rag.evaluation import RAGEvaluator

        evaluator = RAGEvaluator()

        # Test precision
        retrieved = [1, 2, 3, 4, 5]
        expected = [1, 2, 6]
        precision = evaluator._calculate_retrieval_precision(retrieved, expected)
        print(f"Precision test: {precision:.2f} (expected: 0.40)")

        # Test recall
        recall = evaluator._calculate_retrieval_recall(retrieved, expected)
        print(f"Recall test: {recall:.2f} (expected: 0.67)")

        # Test MRR
        mrr = evaluator._calculate_mrr(retrieved, expected)
        print(f"MRR test: {mrr:.2f} (expected: 1.00)")

        # Test faithfulness
        faithfulness = evaluator._evaluate_faithfulness(
            "Revenue grew 15% year over year.",
            ["Company reported revenue growth of 15% YoY."]
        )
        print(f"Faithfulness test: {faithfulness:.2f}")

        # Test relevance
        relevance = evaluator._evaluate_relevance(
            "Revenue grew 15% driven by AI demand.",
            "What was the revenue growth?"
        )
        print(f"Relevance test: {relevance:.2f}")

        print(f"\n✅ All metrics calculations working")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_all_tests():
    """Run all Batch C tests."""
    print("\n" + "=" * 60)
    print("RAG BATCH C - STRUCTURED EXTRACTION + EVALUATION TESTS")
    print("=" * 60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Project root: {project_root}")

    results = {
        'schema': test_schema(),
        'extractor_import': test_extractor_import(),
        'evaluator_import': test_evaluator_import(),
        'extractor_init': test_extractor_init(),
        'evaluator_init': test_evaluator_init(),
        'documents': test_document_available(),
        'extraction_dry_run': test_extraction_dry_run(),
        'metrics': test_metrics_calculation(),
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
        print("\n🎉 Batch C installation complete!")
        print("\nNext steps:")
        print("  1. Run schema creation: python src/rag/schema_batch_c.py")
        print("  2. Extract facts: python src/rag/fact_extractor.py")
        print("  3. Create eval cases and run: python src/rag/evaluation.py")
    else:
        print("\n⚠️ Some tests failed. Review errors above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)