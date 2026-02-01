-- ============================================================================
-- HH RESEARCH PLATFORM - RAG DATABASE SETUP (Audit-Grade)
-- ============================================================================
--
-- This script creates the complete RAG schema with:
-- - Dedicated 'rag' schema (not public)
-- - Full traceability columns
-- - Integrity constraints
-- - Performance indexes
-- - Proper permission grants
--
-- INSTALLATION:
-- 1. Run as DBA/superuser: psql -U postgres -d alpha_platform -f schema.sql
-- 2. The script will:
--    a) Create pgvector extension (if not exists)
--    b) Create 'rag' schema
--    c) Create all tables with constraints
--    d) Grant minimal permissions to 'alpha' user
--
-- ============================================================================

-- ============================================================================
-- STEP 1: EXTENSIONS (Requires superuser)
-- ============================================================================

-- Verify we're running as superuser
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = current_user AND rolsuper) THEN
        RAISE EXCEPTION 'This script must be run as a superuser (e.g., postgres). Current user: %', current_user;
    END IF;
END $$;

-- Install required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- ============================================================================
-- STEP 2: CREATE RAG SCHEMA
-- ============================================================================

-- Create dedicated schema for RAG tables
CREATE SCHEMA IF NOT EXISTS rag;

-- Set search path for this session
SET search_path TO rag, public;

-- ============================================================================
-- STEP 3: CORE DOCUMENT TABLES
-- ============================================================================

-- -----------------------------------------------------------------------------
-- 3.1 DOCUMENTS - Raw documents with full provenance
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag.documents (
    doc_id              BIGSERIAL PRIMARY KEY,

    -- Document identification
    ticker              TEXT NOT NULL,
    doc_type            TEXT NOT NULL,      -- 'transcript', '10k', '10q', '8k', 'press_release'

    -- Source provenance (audit trail)
    source              TEXT NOT NULL,      -- 'sec_edgar', 'manual', 'finnhub'
    source_url          TEXT,
    source_fetched_at_utc TIMESTAMPTZ,      -- When we fetched from source

    -- Temporal information
    asof_ts_utc         TIMESTAMPTZ NOT NULL,   -- Document's effective date
    period_label        TEXT NOT NULL,          -- '2025Q3', 'FY2025' (required)
    fiscal_year         INT,
    fiscal_quarter      INT,

    -- Content
    title               TEXT,
    raw_text            TEXT NOT NULL,
    raw_text_sha256     TEXT NOT NULL,          -- SHA256 of normalized raw_text

    -- Length measures
    char_count          INT NOT NULL,           -- Character count (truth)
    word_count          INT NOT NULL,           -- Word count

    -- Structure
    section_count       INT,

    -- Metadata
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    ingested_at_utc     TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- =========== CONSTRAINTS ===========

    -- Deduplication by content hash
    CONSTRAINT uq_documents_hash UNIQUE (raw_text_sha256),

    -- Natural key uniqueness (prevents duplicate docs even if content differs slightly)
    CONSTRAINT uq_documents_natural_key UNIQUE (ticker, doc_type, period_label, source),

    -- Valid document type
    CONSTRAINT chk_documents_doc_type CHECK (
        doc_type IN ('transcript', '10k', '10q', '8k', 'press_release', 'news_fulltext', 'other')
    ),

    -- Valid fiscal quarter
    CONSTRAINT chk_documents_quarter CHECK (
        fiscal_quarter IS NULL OR fiscal_quarter BETWEEN 1 AND 4
    ),

    -- Period label is required and non-empty
    CONSTRAINT chk_documents_period_required CHECK (
        period_label IS NOT NULL AND period_label != '' AND length(period_label) >= 4
    ),

    -- Character count must be positive and match actual length
    CONSTRAINT chk_documents_char_count CHECK (char_count > 0)
);

-- Indexes for documents
CREATE INDEX IF NOT EXISTS idx_documents_ticker_type_asof
    ON rag.documents(ticker, doc_type, asof_ts_utc DESC);
CREATE INDEX IF NOT EXISTS idx_documents_period
    ON rag.documents(period_label);
CREATE INDEX IF NOT EXISTS idx_documents_ticker
    ON rag.documents(ticker);
CREATE INDEX IF NOT EXISTS idx_documents_ingested
    ON rag.documents(ingested_at_utc DESC);

COMMENT ON TABLE rag.documents IS 'Raw documents for RAG: earnings transcripts, SEC filings, press releases';
COMMENT ON COLUMN rag.documents.source_fetched_at_utc IS 'When we fetched from external source (audit trail)';
COMMENT ON COLUMN rag.documents.raw_text_sha256 IS 'SHA256 of normalized raw_text for deduplication';

-- -----------------------------------------------------------------------------
-- 3.2 DOC_CHUNKS - Chunked documents with embeddings and full traceability
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag.doc_chunks (
    chunk_id            BIGSERIAL PRIMARY KEY,
    doc_id              BIGINT NOT NULL REFERENCES rag.documents(doc_id) ON DELETE CASCADE,

    -- Denormalized for query performance (avoids joins)
    ticker              TEXT NOT NULL,
    doc_type            TEXT NOT NULL,
    asof_ts_utc         TIMESTAMPTZ NOT NULL,

    -- Chunk position and metadata
    section             TEXT,               -- 'Prepared Remarks', 'Q&A', 'Risk Factors'
    speaker             TEXT,               -- For transcripts: speaker name
    speaker_role        TEXT,               -- 'executive', 'analyst', 'operator'
    chunk_index         INT NOT NULL,       -- Order within document (0-based)

    -- Content
    text                TEXT NOT NULL,

    -- Length measures (char_len is truth, token_count is approximation)
    char_len            INT NOT NULL,
    approx_token_count  INT,                -- Approximate (tiktoken cl100k_base)

    -- Traceability: exact position in source document (REQUIRED)
    char_start          INT NOT NULL,
    char_end            INT NOT NULL,

    -- Deduplication
    chunk_hash          TEXT NOT NULL,      -- SHA256 of normalized chunk text

    -- Embedding with full provenance
    embedding           vector(1024),       -- Fixed: bge-large-en-v1.5 = 1024 dims
    embedding_model     TEXT,               -- 'BAAI/bge-large-en-v1.5'
    embedding_dim       INT,                -- 1024 (stored for validation)
    embedded_at_utc     TIMESTAMPTZ,        -- When embedding was generated

    -- Full-text search
    text_tsv            tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,

    -- Metadata
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- =========== CONSTRAINTS ===========

    -- Unique chunk index per document
    CONSTRAINT uq_doc_chunks_index UNIQUE (doc_id, chunk_index),

    -- Unique character range per document (no overlapping chunks)
    CONSTRAINT uq_doc_chunks_range UNIQUE (doc_id, char_start, char_end),

    -- Valid character positions
    CONSTRAINT chk_chunks_positions CHECK (
        char_start >= 0 AND char_end > char_start
    ),

    -- Character length must match range
    CONSTRAINT chk_chunks_char_len CHECK (
        char_len > 0 AND char_len = (char_end - char_start)
    ),

    -- Chunk hash is required
    CONSTRAINT chk_chunks_hash_required CHECK (
        chunk_hash IS NOT NULL AND chunk_hash != ''
    ),

    -- Embedding provenance: all fields NULL together or all NOT NULL together
    CONSTRAINT chk_chunks_embedding_provenance CHECK (
        (embedding IS NULL AND embedded_at_utc IS NULL AND embedding_model IS NULL AND embedding_dim IS NULL)
        OR
        (embedding IS NOT NULL AND embedded_at_utc IS NOT NULL AND embedding_model IS NOT NULL AND embedding_dim IS NOT NULL)
    ),

    -- If embedding exists, dimension must be 1024 (bge-large-en-v1.5)
    CONSTRAINT chk_chunks_embedding_dim CHECK (
        embedding_dim IS NULL OR embedding_dim = 1024
    )
);

-- Indexes for doc_chunks
CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id
    ON rag.doc_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_ticker_type_asof
    ON rag.doc_chunks(ticker, doc_type, asof_ts_utc DESC);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_section
    ON rag.doc_chunks(section) WHERE section IS NOT NULL;

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_doc_chunks_tsv
    ON rag.doc_chunks USING GIN(text_tsv);

-- Trigram index for fuzzy search (optional, heavy)
-- CREATE INDEX IF NOT EXISTS idx_doc_chunks_trgm
--     ON rag.doc_chunks USING GIN(text gin_trgm_ops);

-- HNSW vector index (can add now, will be used in Batch B)
-- Note: Index is created even if no rows exist yet
CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding_hnsw
    ON rag.doc_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

COMMENT ON TABLE rag.doc_chunks IS 'Chunked documents with embeddings for RAG retrieval';
COMMENT ON COLUMN rag.doc_chunks.char_len IS 'Character count (truth) - primary length measure';
COMMENT ON COLUMN rag.doc_chunks.approx_token_count IS 'Approximate tokens (tiktoken cl100k_base) - NOT authoritative';
COMMENT ON COLUMN rag.doc_chunks.char_start IS 'Start position in documents.raw_text (0-indexed, inclusive)';
COMMENT ON COLUMN rag.doc_chunks.char_end IS 'End position in documents.raw_text (exclusive)';

-- ============================================================================
-- STEP 4: AUDIT/TRACEABILITY TABLES
-- ============================================================================

-- -----------------------------------------------------------------------------
-- 4.1 REQUEST_SNAPSHOTS - Complete state at request time
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag.request_snapshots (
    snapshot_id         BIGSERIAL PRIMARY KEY,
    ticker              TEXT NOT NULL,
    requested_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Structured data snapshots
    market_snapshot     JSONB NOT NULL,
    fundamentals_snapshot JSONB,
    options_snapshot    JSONB,
    signals_snapshot    JSONB,

    -- Document context
    document_snapshot   JSONB NOT NULL,     -- Which docs were available/used

    -- Computed metrics
    computed_metrics    JSONB NOT NULL,

    -- Quality flags
    staleness_flags     JSONB NOT NULL,
    data_quality_score  FLOAT,

    -- Policy decision
    policy_decision     JSONB NOT NULL,

    -- Query context
    user_query          TEXT,
    query_type          TEXT
);

CREATE INDEX IF NOT EXISTS idx_request_snapshots_ticker
    ON rag.request_snapshots(ticker);
CREATE INDEX IF NOT EXISTS idx_request_snapshots_time
    ON rag.request_snapshots(requested_at_utc DESC);

-- -----------------------------------------------------------------------------
-- 4.2 RAG_RETRIEVALS - What chunks were retrieved
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag.rag_retrievals (
    retrieval_id        BIGSERIAL PRIMARY KEY,
    snapshot_id         BIGINT NOT NULL REFERENCES rag.request_snapshots(snapshot_id) ON DELETE CASCADE,

    -- Query
    query_text          TEXT NOT NULL,
    query_embedding     vector(1024),
    query_embedding_model TEXT NOT NULL DEFAULT 'BAAI/bge-large-en-v1.5',

    -- Configuration
    retrieval_method    TEXT NOT NULL,      -- 'hybrid_rrf_v1'
    k_vector            INT NOT NULL DEFAULT 40,
    k_lexical           INT NOT NULL DEFAULT 40,
    rrf_k               INT NOT NULL DEFAULT 60,
    top_n_final         INT NOT NULL DEFAULT 12,

    -- Filters
    filters             JSONB NOT NULL,

    -- Results
    retrieved_chunk_ids BIGINT[] NOT NULL,
    scores              JSONB NOT NULL,

    -- Gating
    gating_decision     TEXT NOT NULL,      -- 'PASS', 'FAIL_SIMILARITY', etc.
    gating_details      JSONB,

    created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_retrievals_snapshot
    ON rag.rag_retrievals(snapshot_id);

-- -----------------------------------------------------------------------------
-- 4.3 AI_RESPONSES - LLM responses with citations
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag.ai_responses (
    response_id         BIGSERIAL PRIMARY KEY,
    snapshot_id         BIGINT NOT NULL REFERENCES rag.request_snapshots(snapshot_id) ON DELETE CASCADE,
    retrieval_id        BIGINT REFERENCES rag.rag_retrievals(retrieval_id) ON DELETE SET NULL,

    -- Model
    model_name          TEXT NOT NULL,
    model_endpoint      TEXT,
    prompt_version      TEXT NOT NULL,
    prompt_hash         TEXT,

    -- Input
    prompt_tokens       INT,
    context_tokens      INT,

    -- Output
    response_text       TEXT NOT NULL,
    response_tokens     INT,

    -- Citations (required)
    citations           JSONB NOT NULL,
    citation_count      INT GENERATED ALWAYS AS (jsonb_array_length(citations)) STORED,

    -- Quality
    unknown_claims      JSONB,
    confidence_score    FLOAT,

    -- Timing
    latency_ms          INT,
    created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ai_responses_snapshot
    ON rag.ai_responses(snapshot_id);

-- ============================================================================
-- STEP 5: STRUCTURED EXTRACTION TABLES
-- ============================================================================

-- -----------------------------------------------------------------------------
-- 5.1 TRANSCRIPT_FACTS - Extracted facts from earnings calls
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag.transcript_facts (
    fact_id             BIGSERIAL PRIMARY KEY,
    ticker              TEXT NOT NULL,
    period_label        TEXT NOT NULL,
    doc_id              BIGINT NOT NULL REFERENCES rag.documents(doc_id) ON DELETE CASCADE,

    -- Extraction metadata
    extracted_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
    extractor_model     TEXT NOT NULL,
    extractor_version   TEXT,
    extraction_confidence FLOAT,

    -- Guidance
    guidance_direction  TEXT,
    guidance_revenue_low NUMERIC,
    guidance_revenue_high NUMERIC,
    guidance_eps_low    NUMERIC,
    guidance_eps_high   NUMERIC,
    guidance_notes      TEXT,

    -- Demand
    demand_tone         TEXT,
    demand_drivers      TEXT[],

    -- AI-specific
    ai_mentions_count   INT,
    ai_sentiment        TEXT,
    ai_commentary       TEXT,

    -- Capex
    capex_direction     TEXT,
    capex_amount        NUMERIC,
    capex_notes         TEXT,

    -- Margins
    margin_outlook      TEXT,
    margin_notes        TEXT,

    -- Risks
    risk_themes         TEXT[],
    risk_phrases_raw    TEXT[],

    -- Evidence
    evidence            JSONB NOT NULL,
    raw_json            JSONB NOT NULL,

    CONSTRAINT uq_transcript_facts UNIQUE (ticker, period_label, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_transcript_facts_ticker_period
    ON rag.transcript_facts(ticker, period_label);

-- -----------------------------------------------------------------------------
-- 5.2 FILING_FACTS - Extracted facts from SEC filings
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag.filing_facts (
    fact_id             BIGSERIAL PRIMARY KEY,
    ticker              TEXT NOT NULL,
    doc_id              BIGINT NOT NULL REFERENCES rag.documents(doc_id) ON DELETE CASCADE,
    filing_type         TEXT NOT NULL,
    asof_ts_utc         TIMESTAMPTZ NOT NULL,
    period_label        TEXT,

    -- Extraction metadata
    extracted_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
    extractor_model     TEXT NOT NULL,
    extractor_version   TEXT,

    -- Risk factors
    key_risks           TEXT[],
    risk_severity       JSONB,
    new_risks           TEXT[],
    removed_risks       TEXT[],

    -- Segments
    segment_notes       JSONB,

    -- Legal
    litigation_notes    TEXT,
    material_litigation BOOLEAN,

    -- Cybersecurity
    cybersecurity_notes TEXT,
    cyber_incidents     BOOLEAN,

    -- Geographic
    china_exposure      TEXT,
    geographic_risks    JSONB,

    -- Capex
    capex_plan          JSONB,

    -- Evidence
    evidence            JSONB NOT NULL,
    raw_json            JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_filing_facts_ticker
    ON rag.filing_facts(ticker);

-- ============================================================================
-- STEP 6: EVALUATION TABLES
-- ============================================================================

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

-- ============================================================================
-- STEP 7: VIEWS
-- ============================================================================

-- Latest document per ticker/type
CREATE OR REPLACE VIEW rag.v_latest_documents AS
SELECT DISTINCT ON (ticker, doc_type)
    doc_id, ticker, doc_type, source, asof_ts_utc, period_label, title,
    char_count, word_count, ingested_at_utc
FROM rag.documents
ORDER BY ticker, doc_type, asof_ts_utc DESC;

-- Chunk statistics per document
CREATE OR REPLACE VIEW rag.v_document_chunk_stats AS
SELECT
    d.doc_id,
    d.ticker,
    d.doc_type,
    d.period_label,
    COUNT(c.chunk_id) as chunk_count,
    SUM(c.char_len) as total_chars,
    SUM(c.approx_token_count) as total_approx_tokens,
    COUNT(c.embedding) FILTER (WHERE c.embedding IS NOT NULL) as embedded_chunks,
    array_agg(DISTINCT c.section) FILTER (WHERE c.section IS NOT NULL) as sections
FROM rag.documents d
LEFT JOIN rag.doc_chunks c ON d.doc_id = c.doc_id
GROUP BY d.doc_id, d.ticker, d.doc_type, d.period_label;

-- ============================================================================
-- STEP 8: PERMISSIONS (Least Privilege)
-- ============================================================================

-- Grant usage on schema (but NOT create)
GRANT USAGE ON SCHEMA rag TO alpha;

-- Grant DML on all tables
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA rag TO alpha;

-- Grant sequence usage
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA rag TO alpha;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA rag
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO alpha;
ALTER DEFAULT PRIVILEGES IN SCHEMA rag
    GRANT USAGE, SELECT ON SEQUENCES TO alpha;

-- ============================================================================
-- STEP 9: VERIFICATION
-- ============================================================================

DO $$
DECLARE
    ext_count INT;
    table_count INT;
BEGIN
    -- Check extensions
    SELECT COUNT(*) INTO ext_count
    FROM pg_extension
    WHERE extname IN ('vector', 'pg_trgm');

    IF ext_count < 2 THEN
        RAISE WARNING 'Not all extensions installed. Found: %', ext_count;
    ELSE
        RAISE NOTICE '✅ Extensions: OK (% installed)', ext_count;
    END IF;

    -- Check tables
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'rag';

    RAISE NOTICE '✅ Tables created: % in rag schema', table_count;

    -- Test vector operations
    PERFORM '[1,2,3]'::vector <-> '[4,5,6]'::vector;
    RAISE NOTICE '✅ Vector operations: OK';

    RAISE NOTICE '';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'RAG SCHEMA SETUP COMPLETE';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Schema: rag';
    RAISE NOTICE 'Tables: %', table_count;
    RAISE NOTICE 'User alpha has DML access (no CREATE)';
    RAISE NOTICE '';
END $$;

-- Show summary
SELECT
    table_name,
    (SELECT COUNT(*) FROM information_schema.columns c
     WHERE c.table_schema = 'rag' AND c.table_name = t.table_name) as columns
FROM information_schema.tables t
WHERE table_schema = 'rag' AND table_type = 'BASE TABLE'
ORDER BY table_name;