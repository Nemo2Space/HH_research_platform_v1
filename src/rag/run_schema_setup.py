"""
Run RAG Schema Setup
====================

This script properly executes the RAG schema SQL file.
Run from project root: python src/rag/run_schema_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.db.connection import get_engine
import sqlalchemy


def run_schema_setup():
    """Execute the RAG schema setup."""

    engine = get_engine()

    print("="*60)
    print("RAG SCHEMA SETUP")
    print("="*60)

    # Step 1: Check/Create pgvector extension
    print("\n[1/6] Checking pgvector extension...")
    with engine.connect() as conn:
        try:
            conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            print("✅ pgvector extension ready")
        except Exception as e:
            print(f"⚠️ pgvector: {e}")
            print("   You may need superuser to create extension")

    # Step 2: Create pg_trgm extension
    print("\n[2/6] Checking pg_trgm extension...")
    with engine.connect() as conn:
        try:
            conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            conn.commit()
            print("✅ pg_trgm extension ready")
        except Exception as e:
            print(f"⚠️ pg_trgm: {e}")

    # Step 3: Create schema
    print("\n[3/6] Creating rag schema...")
    with engine.connect() as conn:
        try:
            conn.execute(sqlalchemy.text("CREATE SCHEMA IF NOT EXISTS rag"))
            conn.commit()
            print("✅ Schema 'rag' created")
        except Exception as e:
            print(f"❌ Schema creation failed: {e}")
            return False

    # Step 4: Create tables
    print("\n[4/6] Creating tables...")

    tables_sql = """
    -- DOCUMENTS TABLE
    CREATE TABLE IF NOT EXISTS rag.documents (
        doc_id              BIGSERIAL PRIMARY KEY,
        ticker              TEXT NOT NULL,
        doc_type            TEXT NOT NULL,
        source              TEXT NOT NULL,
        source_url          TEXT,
        source_fetched_at_utc TIMESTAMPTZ,
        asof_ts_utc         TIMESTAMPTZ NOT NULL,
        period_label        TEXT NOT NULL,
        fiscal_year         INT,
        fiscal_quarter      INT,
        title               TEXT,
        raw_text            TEXT NOT NULL,
        raw_text_sha256     TEXT NOT NULL,
        char_count          INT NOT NULL,
        word_count          INT NOT NULL,
        section_count       INT,
        metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
        ingested_at_utc     TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_documents_hash UNIQUE (raw_text_sha256),
        CONSTRAINT uq_documents_natural_key UNIQUE (ticker, doc_type, period_label, source),
        CONSTRAINT chk_documents_doc_type CHECK (
            doc_type IN ('transcript', '10k', '10q', '8k', 'press_release', 'news_fulltext', 'other')
        ),
        CONSTRAINT chk_documents_quarter CHECK (
            fiscal_quarter IS NULL OR fiscal_quarter BETWEEN 1 AND 4
        ),
        CONSTRAINT chk_documents_period_required CHECK (
            period_label IS NOT NULL AND period_label != '' AND length(period_label) >= 4
        ),
        CONSTRAINT chk_documents_char_count CHECK (char_count > 0)
    );

    -- DOC_CHUNKS TABLE
    CREATE TABLE IF NOT EXISTS rag.doc_chunks (
        chunk_id            BIGSERIAL PRIMARY KEY,
        doc_id              BIGINT NOT NULL REFERENCES rag.documents(doc_id) ON DELETE CASCADE,
        ticker              TEXT NOT NULL,
        doc_type            TEXT NOT NULL,
        asof_ts_utc         TIMESTAMPTZ NOT NULL,
        section             TEXT,
        speaker             TEXT,
        speaker_role        TEXT,
        chunk_index         INT NOT NULL,
        text                TEXT NOT NULL,
        char_len            INT NOT NULL,
        approx_token_count  INT,
        char_start          INT NOT NULL,
        char_end            INT NOT NULL,
        chunk_hash          TEXT NOT NULL,
        embedding           vector(1024),
        embedding_model     TEXT,
        embedding_dim       INT,
        embedded_at_utc     TIMESTAMPTZ,
        text_tsv            tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
        metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_doc_chunks_index UNIQUE (doc_id, chunk_index),
        CONSTRAINT uq_doc_chunks_range UNIQUE (doc_id, char_start, char_end),
        CONSTRAINT chk_chunks_positions CHECK (char_start >= 0 AND char_end > char_start),
        CONSTRAINT chk_chunks_char_len CHECK (char_len > 0),
        CONSTRAINT chk_chunks_hash_required CHECK (chunk_hash IS NOT NULL AND chunk_hash != ''),
        CONSTRAINT chk_chunks_embedding_provenance CHECK (
            (embedding IS NULL AND embedded_at_utc IS NULL AND embedding_model IS NULL AND embedding_dim IS NULL)
            OR
            (embedding IS NOT NULL AND embedded_at_utc IS NOT NULL AND embedding_model IS NOT NULL AND embedding_dim IS NOT NULL)
        ),
        CONSTRAINT chk_chunks_embedding_dim CHECK (embedding_dim IS NULL OR embedding_dim = 1024)
    );

    -- REQUEST_SNAPSHOTS TABLE
    CREATE TABLE IF NOT EXISTS rag.request_snapshots (
        snapshot_id         BIGSERIAL PRIMARY KEY,
        ticker              TEXT NOT NULL,
        requested_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
        market_snapshot     JSONB NOT NULL,
        fundamentals_snapshot JSONB,
        options_snapshot    JSONB,
        signals_snapshot    JSONB,
        document_snapshot   JSONB NOT NULL,
        computed_metrics    JSONB NOT NULL,
        staleness_flags     JSONB NOT NULL,
        data_quality_score  FLOAT,
        policy_decision     JSONB NOT NULL,
        user_query          TEXT,
        query_type          TEXT
    );

    -- RAG_RETRIEVALS TABLE
    CREATE TABLE IF NOT EXISTS rag.rag_retrievals (
        retrieval_id        BIGSERIAL PRIMARY KEY,
        snapshot_id         BIGINT NOT NULL REFERENCES rag.request_snapshots(snapshot_id) ON DELETE CASCADE,
        query_text          TEXT NOT NULL,
        query_embedding     vector(1024),
        query_embedding_model TEXT NOT NULL DEFAULT 'BAAI/bge-large-en-v1.5',
        retrieval_method    TEXT NOT NULL,
        k_vector            INT NOT NULL DEFAULT 40,
        k_lexical           INT NOT NULL DEFAULT 40,
        rrf_k               INT NOT NULL DEFAULT 60,
        top_n_final         INT NOT NULL DEFAULT 12,
        filters             JSONB NOT NULL,
        retrieved_chunk_ids BIGINT[] NOT NULL,
        scores              JSONB NOT NULL,
        gating_decision     TEXT NOT NULL,
        gating_details      JSONB,
        created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    -- AI_RESPONSES TABLE
    CREATE TABLE IF NOT EXISTS rag.ai_responses (
        response_id         BIGSERIAL PRIMARY KEY,
        snapshot_id         BIGINT NOT NULL REFERENCES rag.request_snapshots(snapshot_id) ON DELETE CASCADE,
        retrieval_id        BIGINT REFERENCES rag.rag_retrievals(retrieval_id) ON DELETE SET NULL,
        model_name          TEXT NOT NULL,
        model_endpoint      TEXT,
        prompt_version      TEXT NOT NULL,
        prompt_hash         TEXT,
        prompt_tokens       INT,
        context_tokens      INT,
        response_text       TEXT NOT NULL,
        response_tokens     INT,
        citations           JSONB NOT NULL,
        citation_count      INT GENERATED ALWAYS AS (jsonb_array_length(citations)) STORED,
        unknown_claims      JSONB,
        confidence_score    FLOAT,
        latency_ms          INT,
        created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    -- TRANSCRIPT_FACTS TABLE
    CREATE TABLE IF NOT EXISTS rag.transcript_facts (
        fact_id             BIGSERIAL PRIMARY KEY,
        ticker              TEXT NOT NULL,
        period_label        TEXT NOT NULL,
        doc_id              BIGINT NOT NULL REFERENCES rag.documents(doc_id) ON DELETE CASCADE,
        extracted_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
        extractor_model     TEXT NOT NULL,
        extractor_version   TEXT,
        extraction_confidence FLOAT,
        guidance_direction  TEXT,
        guidance_revenue_low NUMERIC,
        guidance_revenue_high NUMERIC,
        guidance_eps_low    NUMERIC,
        guidance_eps_high   NUMERIC,
        guidance_notes      TEXT,
        demand_tone         TEXT,
        demand_drivers      TEXT[],
        ai_mentions_count   INT,
        ai_sentiment        TEXT,
        ai_commentary       TEXT,
        capex_direction     TEXT,
        capex_amount        NUMERIC,
        capex_notes         TEXT,
        margin_outlook      TEXT,
        margin_notes        TEXT,
        risk_themes         TEXT[],
        risk_phrases_raw    TEXT[],
        evidence            JSONB NOT NULL,
        raw_json            JSONB NOT NULL,
        CONSTRAINT uq_transcript_facts UNIQUE (ticker, period_label, doc_id)
    );

    -- FILING_FACTS TABLE
    CREATE TABLE IF NOT EXISTS rag.filing_facts (
        fact_id             BIGSERIAL PRIMARY KEY,
        ticker              TEXT NOT NULL,
        doc_id              BIGINT NOT NULL REFERENCES rag.documents(doc_id) ON DELETE CASCADE,
        filing_type         TEXT NOT NULL,
        asof_ts_utc         TIMESTAMPTZ NOT NULL,
        period_label        TEXT,
        extracted_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
        extractor_model     TEXT NOT NULL,
        extractor_version   TEXT,
        key_risks           TEXT[],
        risk_severity       JSONB,
        new_risks           TEXT[],
        removed_risks       TEXT[],
        segment_notes       JSONB,
        litigation_notes    TEXT,
        material_litigation BOOLEAN,
        cybersecurity_notes TEXT,
        cyber_incidents     BOOLEAN,
        china_exposure      TEXT,
        geographic_risks    JSONB,
        capex_plan          JSONB,
        evidence            JSONB NOT NULL,
        raw_json            JSONB NOT NULL
    );

    -- RAG_EVAL_CASES TABLE
    CREATE TABLE IF NOT EXISTS rag.rag_eval_cases (
        case_id             BIGSERIAL PRIMARY KEY,
        ticker              TEXT,
        question            TEXT NOT NULL,
        question_type       TEXT,
        expected_answer     TEXT,
        expected_citations  BIGINT[],
        expected_doc_types  TEXT[],
        must_refuse         BOOLEAN NOT NULL DEFAULT FALSE,
        refuse_reason       TEXT,
        difficulty          TEXT DEFAULT 'medium',
        created_by          TEXT,
        verified            BOOLEAN DEFAULT FALSE,
        created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    -- RAG_EVAL_RUNS TABLE
    CREATE TABLE IF NOT EXISTS rag.rag_eval_runs (
        run_id              BIGSERIAL PRIMARY KEY,
        run_ts_utc          TIMESTAMPTZ NOT NULL DEFAULT now(),
        embedding_model     TEXT NOT NULL,
        retrieval_method    TEXT NOT NULL,
        prompt_version      TEXT NOT NULL,
        llm_model           TEXT NOT NULL,
        parameters          JSONB NOT NULL,
        metrics             JSONB NOT NULL,
        case_results        JSONB NOT NULL,
        failures            JSONB NOT NULL,
        failure_count       INT GENERATED ALWAYS AS (jsonb_array_length(failures)) STORED,
        total_cases         INT NOT NULL,
        passed_cases        INT NOT NULL,
        pass_rate           FLOAT GENERATED ALWAYS AS (passed_cases::float / NULLIF(total_cases, 0)) STORED,
        notes               TEXT
    );
    """

    with engine.connect() as conn:
        try:
            conn.execute(sqlalchemy.text(tables_sql))
            conn.commit()
            print("✅ All 9 tables created")
        except Exception as e:
            print(f"❌ Table creation error: {e}")
            return False

    # Step 5: Create indexes
    print("\n[5/6] Creating indexes...")

    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_documents_ticker_type_asof ON rag.documents(ticker, doc_type, asof_ts_utc DESC)",
        "CREATE INDEX IF NOT EXISTS idx_documents_period ON rag.documents(period_label)",
        "CREATE INDEX IF NOT EXISTS idx_documents_ticker ON rag.documents(ticker)",
        "CREATE INDEX IF NOT EXISTS idx_documents_ingested ON rag.documents(ingested_at_utc DESC)",
        "CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id ON rag.doc_chunks(doc_id)",
        "CREATE INDEX IF NOT EXISTS idx_doc_chunks_ticker_type_asof ON rag.doc_chunks(ticker, doc_type, asof_ts_utc DESC)",
        "CREATE INDEX IF NOT EXISTS idx_doc_chunks_tsv ON rag.doc_chunks USING GIN(text_tsv)",
        "CREATE INDEX IF NOT EXISTS idx_request_snapshots_ticker ON rag.request_snapshots(ticker)",
        "CREATE INDEX IF NOT EXISTS idx_request_snapshots_time ON rag.request_snapshots(requested_at_utc DESC)",
        "CREATE INDEX IF NOT EXISTS idx_rag_retrievals_snapshot ON rag.rag_retrievals(snapshot_id)",
        "CREATE INDEX IF NOT EXISTS idx_ai_responses_snapshot ON rag.ai_responses(snapshot_id)",
        "CREATE INDEX IF NOT EXISTS idx_transcript_facts_ticker_period ON rag.transcript_facts(ticker, period_label)",
        "CREATE INDEX IF NOT EXISTS idx_filing_facts_ticker ON rag.filing_facts(ticker)",
    ]

    with engine.connect() as conn:
        for idx_sql in indexes:
            try:
                conn.execute(sqlalchemy.text(idx_sql))
                conn.commit()
            except Exception as e:
                print(f"⚠️ Index warning: {e}")
        print("✅ Indexes created")

    # Step 6: Create HNSW vector index (if pgvector available)
    print("\n[6/6] Creating vector index...")
    with engine.connect() as conn:
        try:
            conn.execute(sqlalchemy.text("""
                CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding_hnsw 
                ON rag.doc_chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))
            conn.commit()
            print("✅ HNSW vector index created")
        except Exception as e:
            print(f"⚠️ Vector index: {e}")
            print("   (This is OK if pgvector is not installed yet)")

    # Verify
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    with engine.connect() as conn:
        # Check tables
        result = conn.execute(sqlalchemy.text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'rag'
            ORDER BY table_name
        """))
        tables = [row[0] for row in result]

        print(f"\nTables in 'rag' schema: {len(tables)}")
        for t in tables:
            print(f"  ✅ rag.{t}")

        # Check pgvector
        result = conn.execute(sqlalchemy.text("""
            SELECT 1 FROM pg_extension WHERE extname = 'vector'
        """))
        if result.fetchone():
            print("\n✅ pgvector extension: installed")
        else:
            print("\n⚠️ pgvector extension: NOT installed")
            print("   Run as superuser: CREATE EXTENSION vector;")

    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext step: Run the test script:")
    print("  python src/rag/test_batch_a.py")

    return True


if __name__ == "__main__":
    success = run_schema_setup()
    sys.exit(0 if success else 1)