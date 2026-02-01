"""
RAG Batch C - Schema Additions
==============================

Adds tables for:
- transcript_facts: Structured facts from earnings transcripts
- filing_facts: Structured facts from SEC filings
- rag_eval_cases: Evaluation test cases
- rag_eval_runs: Evaluation run results

Run:
    python src/rag/schema_batch_c.py

Author: HH Research Platform
"""

from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)


SCHEMA_DDL = """
-- ============================================================
-- RAG BATCH C: Structured Extraction + Evaluation Tables
-- ============================================================

-- Transcript Facts Table
-- Stores structured facts extracted from earnings transcripts
CREATE TABLE IF NOT EXISTS rag.transcript_facts (
    fact_id SERIAL PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES rag.documents(doc_id),
    ticker VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_quarter INTEGER NOT NULL,
    
    -- Guidance
    guidance_direction VARCHAR(20), -- raised/maintained/lowered/initiated/withdrawn/not_mentioned
    guidance_revenue_low NUMERIC(15,2),
    guidance_revenue_high NUMERIC(15,2),
    guidance_eps_low NUMERIC(10,2),
    guidance_eps_high NUMERIC(10,2),
    guidance_margin_low NUMERIC(5,4),
    guidance_margin_high NUMERIC(5,4),
    
    -- Sentiment
    mgmt_tone VARCHAR(20), -- bullish/neutral/cautious/bearish
    confidence_level NUMERIC(3,2), -- 0.00-1.00
    
    -- Key themes (controlled taxonomy)
    risk_themes TEXT[],
    growth_drivers TEXT[],
    
    -- Key quotes
    ceo_key_quote TEXT,
    cfo_key_quote TEXT,
    
    -- Metrics mentioned
    metrics_mentioned JSONB,
    
    -- Extraction metadata
    extractor_model VARCHAR(100) NOT NULL,
    extraction_confidence NUMERIC(3,2) NOT NULL,
    chunk_ids_used INTEGER[],
    extracted_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(doc_id)
);

-- Filing Facts Table  
-- Stores structured facts extracted from SEC filings
CREATE TABLE IF NOT EXISTS rag.filing_facts (
    fact_id SERIAL PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES rag.documents(doc_id),
    ticker VARCHAR(10) NOT NULL,
    filing_type VARCHAR(10) NOT NULL, -- 10k, 10q, 8k
    fiscal_year INTEGER NOT NULL,
    fiscal_quarter INTEGER,
    
    -- Risk factors (controlled taxonomy)
    risk_themes TEXT[],
    new_risks_vs_prior TEXT[],
    removed_risks_vs_prior TEXT[],
    
    -- Financial highlights
    revenue_reported NUMERIC(15,2),
    net_income_reported NUMERIC(15,2),
    eps_reported NUMERIC(10,2),
    
    -- Key changes
    significant_changes TEXT[],
    
    -- Extraction metadata
    extractor_model VARCHAR(100) NOT NULL,
    extraction_confidence NUMERIC(3,2) NOT NULL,
    extracted_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(doc_id)
);

-- Evaluation Cases Table
-- Stores test cases for evaluating RAG quality
CREATE TABLE IF NOT EXISTS rag.rag_eval_cases (
    case_id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    question TEXT NOT NULL,
    expected_answer TEXT NOT NULL,
    expected_chunk_ids INTEGER[],
    
    -- Classification
    category VARCHAR(50) NOT NULL, -- guidance, risk, financial, general
    difficulty VARCHAR(20) NOT NULL DEFAULT 'medium', -- easy, medium, hard
    
    -- Source
    source_doc_ids INTEGER[],
    tags TEXT[],
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Evaluation Runs Table
-- Stores results of evaluation runs
CREATE TABLE IF NOT EXISTS rag.rag_eval_runs (
    run_id SERIAL PRIMARY KEY,
    started_at_utc TIMESTAMPTZ NOT NULL,
    completed_at_utc TIMESTAMPTZ NOT NULL,
    
    -- Counts
    total_cases INTEGER NOT NULL,
    passed_cases INTEGER NOT NULL,
    failed_cases INTEGER NOT NULL,
    
    -- Aggregate metrics
    avg_retrieval_precision NUMERIC(5,4),
    avg_retrieval_recall NUMERIC(5,4),
    avg_retrieval_mrr NUMERIC(5,4),
    avg_answer_faithfulness NUMERIC(5,4),
    avg_answer_relevance NUMERIC(5,4),
    avg_citation_accuracy NUMERIC(5,4),
    avg_latency_ms NUMERIC(10,2),
    
    -- Results by category
    results_by_category JSONB,
    
    -- Config snapshot
    config_snapshot JSONB,
    
    -- Metadata
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_transcript_facts_ticker 
ON rag.transcript_facts(ticker, fiscal_year, fiscal_quarter);

CREATE INDEX IF NOT EXISTS idx_transcript_facts_guidance 
ON rag.transcript_facts(guidance_direction);

CREATE INDEX IF NOT EXISTS idx_filing_facts_ticker 
ON rag.filing_facts(ticker, fiscal_year);

CREATE INDEX IF NOT EXISTS idx_filing_facts_risks 
ON rag.filing_facts USING GIN(risk_themes);

CREATE INDEX IF NOT EXISTS idx_eval_cases_category 
ON rag.rag_eval_cases(category, is_active);

CREATE INDEX IF NOT EXISTS idx_eval_runs_date 
ON rag.rag_eval_runs(started_at_utc DESC);
"""


def create_batch_c_schema():
    """Create Batch C schema tables."""
    print("Creating Batch C schema...")

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Split and execute statements
            statements = [s.strip() for s in SCHEMA_DDL.split(';') if s.strip()]

            for stmt in statements:
                if stmt and not stmt.startswith('--'):
                    try:
                        cur.execute(stmt)
                    except Exception as e:
                        if 'already exists' in str(e):
                            pass  # Ignore already exists errors
                        else:
                            print(f"Warning: {e}")

            conn.commit()

    print("✅ Batch C schema created successfully!")


def verify_schema():
    """Verify Batch C tables exist."""
    print("\nVerifying Batch C schema...")

    tables = [
        'rag.transcript_facts',
        'rag.filing_facts',
        'rag.rag_eval_cases',
        'rag.rag_eval_runs',
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            for table in tables:
                schema, name = table.split('.')
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = %s AND table_name = %s
                    )
                """, (schema, name))
                exists = cur.fetchone()[0]

                status = "✅" if exists else "❌"
                print(f"  {status} {table}")

    print("\nBatch C schema verification complete!")


if __name__ == "__main__":
    create_batch_c_schema()
    verify_schema()