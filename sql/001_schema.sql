-- =============================================================================
-- ALPHA PLATFORM - Database Schema
-- =============================================================================
-- This file is auto-executed when TimescaleDB container starts
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector for embeddings

-- =============================================================================
-- MARKET DATA TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Prices (OHLCV) - TimescaleDB hypertable
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prices (
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open NUMERIC(12, 4),
    high NUMERIC(12, 4),
    low NUMERIC(12, 4),
    close NUMERIC(12, 4),
    adj_close NUMERIC(12, 4),
    volume BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (ticker, date)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('prices', 'date', if_not_exists => TRUE);

-- Index for fast ticker lookups
CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices (ticker, date DESC);

-- -----------------------------------------------------------------------------
-- Fundamentals
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fundamentals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    
    -- Valuation
    market_cap BIGINT,
    pe_ratio NUMERIC(10, 2),
    forward_pe NUMERIC(10, 2),
    pb_ratio NUMERIC(10, 2),
    ps_ratio NUMERIC(10, 2),
    peg_ratio NUMERIC(10, 2),
    
    -- Profitability
    profit_margin NUMERIC(8, 4),
    operating_margin NUMERIC(8, 4),
    gross_margin NUMERIC(8, 4),
    roe NUMERIC(8, 4),
    roa NUMERIC(8, 4),
    
    -- Growth
    revenue_growth NUMERIC(8, 4),
    earnings_growth NUMERIC(8, 4),
    eps_growth NUMERIC(8, 4),
    
    -- Dividend
    dividend_yield NUMERIC(8, 4),
    dividend_payout_ratio NUMERIC(8, 4),
    
    -- Financial Health
    current_ratio NUMERIC(8, 4),
    debt_to_equity NUMERIC(10, 2),
    free_cash_flow BIGINT,
    
    -- Per Share
    eps NUMERIC(10, 4),
    book_value_per_share NUMERIC(10, 4),
    revenue_per_share NUMERIC(10, 4),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker ON fundamentals (ticker, date DESC);

-- -----------------------------------------------------------------------------
-- Analyst Ratings (from Project 1)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analyst_ratings (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    
    -- Counts
    analyst_total INTEGER DEFAULT 0,
    analyst_buy INTEGER DEFAULT 0,
    analyst_hold INTEGER DEFAULT 0,
    analyst_sell INTEGER DEFAULT 0,
    analyst_strong_buy INTEGER DEFAULT 0,
    analyst_strong_sell INTEGER DEFAULT 0,
    
    -- Derived
    analyst_positive INTEGER DEFAULT 0,  -- strong_buy + buy
    analyst_positivity NUMERIC(5, 2),    -- positive / total * 100
    
    -- Consensus
    consensus_rating VARCHAR(20),  -- Strong Buy, Buy, Hold, Sell, Strong Sell
    
    source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (ticker, date, source)
);

CREATE INDEX IF NOT EXISTS idx_analyst_ratings_ticker ON analyst_ratings (ticker, date DESC);

-- -----------------------------------------------------------------------------
-- Price Targets (from Project 1)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS price_targets (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    
    current_price NUMERIC(12, 4),
    target_mean NUMERIC(12, 4),
    target_high NUMERIC(12, 4),
    target_low NUMERIC(12, 4),
    target_median NUMERIC(12, 4),
    
    -- Derived
    target_upside_pct NUMERIC(8, 4),  -- (target_mean - current) / current * 100
    analyst_count INTEGER,
    
    source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (ticker, date, source)
);

CREATE INDEX IF NOT EXISTS idx_price_targets_ticker ON price_targets (ticker, date DESC);

-- =============================================================================
-- NEWS & SENTIMENT TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- News Source Credibility (from Project 1)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_source_credibility (
    source VARCHAR(100) PRIMARY KEY,
    credibility_score INTEGER NOT NULL CHECK (credibility_score >= 1 AND credibility_score <= 10),
    category VARCHAR(50),  -- financial, general, social
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default credibility scores
INSERT INTO news_source_credibility (source, credibility_score, category) VALUES
    ('bloomberg', 10, 'financial'),
    ('wsj', 10, 'financial'),
    ('ft', 10, 'financial'),
    ('reuters', 10, 'financial'),
    ('cnbc', 9, 'financial'),
    ('barrons', 9, 'financial'),
    ('forbes', 8, 'financial'),
    ('marketwatch', 8, 'financial'),
    ('morningstar', 8, 'financial'),
    ('seekingalpha', 7, 'financial'),
    ('investors.com', 7, 'financial'),
    ('yahoo finance', 7, 'financial'),
    ('thestreet', 6, 'financial'),
    ('motley fool', 6, 'financial'),
    ('benzinga', 6, 'financial'),
    ('businessinsider', 5, 'general'),
    ('cnet', 5, 'general'),
    ('reddit', 4, 'social'),
    ('default', 5, 'general')
ON CONFLICT (source) DO NOTHING;

-- -----------------------------------------------------------------------------
-- News Articles
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_articles (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),  -- Can be NULL for market-wide news
    
    headline TEXT NOT NULL,
    snippet TEXT,
    url TEXT,
    source VARCHAR(100),
    author VARCHAR(200),
    
    published_at TIMESTAMPTZ,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Credibility (joined from news_source_credibility)
    credibility_score INTEGER DEFAULT 5,
    
    -- AI-generated fast sentiment (from GPT-OSS-20B)
    ai_sentiment_fast INTEGER,  -- 0-100
    
    -- Relevance
    is_relevant BOOLEAN DEFAULT TRUE,
    relevance_score NUMERIC(5, 4),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_news_articles_ticker ON news_articles (ticker, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_articles_published ON news_articles (published_at DESC);

-- -----------------------------------------------------------------------------
-- Sentiment Scores (aggregated per ticker per day)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    
    -- Raw scores
    sentiment_raw INTEGER,          -- Simple average (0-100)
    sentiment_weighted INTEGER,     -- Credibility-weighted average (0-100)
    
    -- AI sentiment
    ai_sentiment_fast INTEGER,      -- Fast model average
    ai_sentiment_deep INTEGER,      -- Deep model (Qwen) score
    
    -- Article counts
    article_count INTEGER DEFAULT 0,
    relevant_article_count INTEGER DEFAULT 0,
    
    -- Classification
    sentiment_class VARCHAR(20),    -- Very Bearish, Bearish, Neutral, Bullish, Very Bullish
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_scores_ticker ON sentiment_scores (ticker, date DESC);

-- -----------------------------------------------------------------------------
-- Market-wide Sentiment (from Project 1: polarity_scores_mkt_allnews)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS market_sentiment (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    
    sentiment_score INTEGER,  -- 0-100
    sentiment_class VARCHAR(20),  -- positive, negative, neutral
    
    headline_count INTEGER,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- SEC FILINGS TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- SEC Filings Metadata
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sec_filings (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    cik VARCHAR(20) NOT NULL,
    
    form_type VARCHAR(20) NOT NULL,  -- 10-K, 10-Q, 8-K, 13F-HR, 4
    accession_number VARCHAR(30) NOT NULL,
    filing_date DATE NOT NULL,
    
    document_url TEXT,
    document_title TEXT,
    
    -- Processing status
    is_processed BOOLEAN DEFAULT FALSE,
    chunk_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (accession_number)
);

CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker ON sec_filings (ticker, filing_date DESC);
CREATE INDEX IF NOT EXISTS idx_sec_filings_form ON sec_filings (form_type, filing_date DESC);

-- -----------------------------------------------------------------------------
-- SEC Filing Chunks (for RAG)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sec_chunks (
    id SERIAL PRIMARY KEY,
    filing_id INTEGER REFERENCES sec_filings(id) ON DELETE CASCADE,
    
    ticker VARCHAR(10) NOT NULL,
    form_type VARCHAR(20) NOT NULL,
    accession_number VARCHAR(30) NOT NULL,
    filing_date DATE NOT NULL,
    
    section VARCHAR(100),  -- ITEM 1A Risk Factors, ITEM 7 MD&A, etc.
    chunk_index INTEGER,
    chunk_text TEXT NOT NULL,
    
    -- pgvector embedding (1536 dimensions for OpenAI, 384 for sentence-transformers)
    embedding vector(384),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sec_chunks_ticker ON sec_chunks (ticker, filing_date DESC);
CREATE INDEX IF NOT EXISTS idx_sec_chunks_embedding ON sec_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- -----------------------------------------------------------------------------
-- Insider Transactions (Form 4)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS insider_transactions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    cik VARCHAR(20),
    
    -- Insider info
    insider_name VARCHAR(200) NOT NULL,
    insider_title VARCHAR(100),
    insider_relationship VARCHAR(50),  -- Director, Officer, 10% Owner, etc.
    
    -- Transaction details
    transaction_date DATE NOT NULL,
    transaction_type VARCHAR(20),  -- P (Purchase), S (Sale), A (Award), etc.
    transaction_code VARCHAR(10),
    
    shares_transacted BIGINT,
    price_per_share NUMERIC(12, 4),
    total_value NUMERIC(16, 2),
    
    shares_owned_after BIGINT,
    
    -- Filing info
    filing_date DATE,
    accession_number VARCHAR(30),
    
    -- Derived signal
    signal INTEGER,  -- +1 (bullish buy), -1 (bearish sell), 0 (neutral)
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_insider_transactions_ticker ON insider_transactions (ticker, transaction_date DESC);
CREATE INDEX IF NOT EXISTS idx_insider_transactions_date ON insider_transactions (transaction_date DESC);

-- -----------------------------------------------------------------------------
-- Institutional Holdings (13F)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS institutional_holdings (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    
    -- Institution info
    institution_cik VARCHAR(20) NOT NULL,
    institution_name VARCHAR(300) NOT NULL,
    
    -- Quarter info
    report_date DATE NOT NULL,  -- Quarter end date
    filing_date DATE,
    
    -- Position details
    shares_held BIGINT,
    market_value BIGINT,  -- In thousands USD
    
    -- Change from previous quarter
    shares_change BIGINT,
    shares_change_pct NUMERIC(10, 4),
    
    -- Position type
    position_type VARCHAR(20),  -- NEW, INCREASED, DECREASED, UNCHANGED, SOLD_OUT
    
    accession_number VARCHAR(30),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (ticker, institution_cik, report_date)
);

CREATE INDEX IF NOT EXISTS idx_inst_holdings_ticker ON institutional_holdings (ticker, report_date DESC);
CREATE INDEX IF NOT EXISTS idx_inst_holdings_institution ON institutional_holdings (institution_cik, report_date DESC);

-- =============================================================================
-- ANALYSIS RESULTS TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Screener Scores (all Project 1 scores)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS screener_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    
    -- Sentiment scores
    sentiment_score INTEGER,
    sentiment_weighted INTEGER,
    
    -- Fundamental scores
    fundamental_score INTEGER,
    growth_score INTEGER,
    dividend_score INTEGER,
    
    -- Technical scores
    technical_score INTEGER,
    gap_score INTEGER,
    gap_type VARCHAR(20),
    likelihood_score INTEGER,
    
    -- Analyst scores
    analyst_positivity NUMERIC(5, 2),
    target_upside_pct NUMERIC(8, 4),
    
    -- Smart money signals (NEW)
    insider_signal INTEGER,       -- -1, 0, +1
    institutional_signal INTEGER, -- -1, 0, +1
    
    -- Composite
    composite_score INTEGER,  -- Weighted combination
    total_score INTEGER,      -- Final score for ranking
    
    -- Metadata
    article_count INTEGER,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_screener_scores_ticker ON screener_scores (ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_screener_scores_date ON screener_scores (date DESC, total_score DESC);

-- -----------------------------------------------------------------------------
-- Trading Signals (from Project 1 signal_generation)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trading_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    
    signal_type VARCHAR(20) NOT NULL,  -- STRONG BUY, BUY, WEAK BUY, NEUTRAL+, NEUTRAL, NEUTRAL-, WEAK SELL, SELL, STRONG SELL, INCOME BUY, GROWTH BUY
    signal_strength INTEGER,  -- -5 to +5
    signal_color VARCHAR(10),  -- Hex color code
    signal_reason TEXT,
    
    -- Underlying scores (for audit)
    sentiment_score INTEGER,
    fundamental_score INTEGER,
    gap_score INTEGER,
    likelihood_score INTEGER,
    analyst_positivity NUMERIC(5, 2),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_trading_signals_ticker ON trading_signals (ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_type ON trading_signals (signal_type, date DESC);

-- -----------------------------------------------------------------------------
-- Committee Decisions (from Project 2)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS committee_decisions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    persona VARCHAR(20) DEFAULT 'neutral',  -- neutral, risk_averse, risk_seeking
    
    -- Verdict
    verdict VARCHAR(10) NOT NULL,  -- BUY, HOLD, SELL
    conviction INTEGER,  -- 0-100
    
    -- Expected outcome
    expected_alpha_bps INTEGER,
    horizon_days INTEGER,
    
    -- Aggregated probabilities
    buy_prob NUMERIC(5, 4),
    confidence NUMERIC(5, 4),
    
    -- Risks (JSON array)
    risks JSONB,
    
    -- Rationale
    rationale TEXT,
    
    -- Transcript reference
    transcript_id VARCHAR(50),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (ticker, date, persona)
);

CREATE INDEX IF NOT EXISTS idx_committee_decisions_ticker ON committee_decisions (ticker, date DESC);

-- -----------------------------------------------------------------------------
-- Agent Votes (individual agent outputs)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_votes (
    id SERIAL PRIMARY KEY,
    decision_id INTEGER REFERENCES committee_decisions(id) ON DELETE CASCADE,
    
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    
    agent_role VARCHAR(20) NOT NULL,  -- fundamental, valuation, sentiment, technical, insider, institutional
    
    buy_prob NUMERIC(5, 4),
    expected_alpha_bps INTEGER,
    confidence NUMERIC(5, 4),
    horizon_days INTEGER,
    
    rationale TEXT,
    risks JSONB,
    evidence_refs JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_votes_decision ON agent_votes (decision_id);
CREATE INDEX IF NOT EXISTS idx_agent_votes_ticker ON agent_votes (ticker, date DESC);

-- -----------------------------------------------------------------------------
-- Debate Transcripts
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS debate_transcripts (
    id VARCHAR(50) PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    persona VARCHAR(20) DEFAULT 'neutral',
    
    transcript JSONB NOT NULL,  -- Full debate conversation
    
    total_turns INTEGER,
    duration_seconds NUMERIC(10, 2),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_debate_transcripts_ticker ON debate_transcripts (ticker, date DESC);

-- =============================================================================
-- PORTFOLIO TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Portfolio Positions
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    
    shares NUMERIC(16, 4) NOT NULL,
    cost_basis NUMERIC(12, 4),
    avg_cost_per_share NUMERIC(12, 4),
    
    entry_date DATE,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    
    -- Current values (updated periodically)
    current_price NUMERIC(12, 4),
    market_value NUMERIC(16, 2),
    unrealized_pnl NUMERIC(16, 2),
    unrealized_pnl_pct NUMERIC(8, 4),
    
    UNIQUE (ticker)
);

-- -----------------------------------------------------------------------------
-- Portfolio History (daily snapshots)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_history (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    
    total_value NUMERIC(16, 2),
    cash NUMERIC(16, 2),
    invested NUMERIC(16, 2),
    
    daily_pnl NUMERIC(16, 2),
    daily_return_pct NUMERIC(8, 4),
    
    cumulative_return_pct NUMERIC(10, 4),
    
    positions_count INTEGER,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (date)
);

SELECT create_hypertable('portfolio_history', 'date', if_not_exists => TRUE);

-- -----------------------------------------------------------------------------
-- Rebalance Proposals
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rebalance_proposals (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    
    status VARCHAR(20) DEFAULT 'proposed',  -- proposed, approved, executed, cancelled
    
    -- Summary
    trades_count INTEGER,
    total_buy_value NUMERIC(16, 2),
    total_sell_value NUMERIC(16, 2),
    estimated_cost NUMERIC(12, 2),
    
    -- Detail (JSON array of trades)
    trades JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    executed_at TIMESTAMPTZ
);

-- =============================================================================
-- BACKTEST TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Backtest Runs
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_runs (
    id SERIAL PRIMARY KEY,
    
    -- Parameters
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital NUMERIC(16, 2),
    strategy_name VARCHAR(100),
    parameters JSONB,
    
    -- Results
    final_value NUMERIC(16, 2),
    total_return_pct NUMERIC(10, 4),
    cagr_pct NUMERIC(8, 4),
    sharpe_ratio NUMERIC(8, 4),
    sortino_ratio NUMERIC(8, 4),
    max_drawdown_pct NUMERIC(8, 4),
    win_rate NUMERIC(5, 4),
    
    -- Benchmark comparison
    benchmark VARCHAR(10),
    benchmark_return_pct NUMERIC(10, 4),
    alpha NUMERIC(8, 4),
    beta NUMERIC(8, 4),
    
    -- Trade stats
    total_trades INTEGER,
    avg_trade_return_pct NUMERIC(8, 4),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- SYSTEM TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- System Status (for pause/resume tracking)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS system_status (
    id SERIAL PRIMARY KEY,
    
    component VARCHAR(50) NOT NULL,  -- screener, committee, ingestion
    status VARCHAR(20) NOT NULL,     -- running, paused, stopped, error
    
    last_run_at TIMESTAMPTZ,
    last_completed_at TIMESTAMPTZ,
    
    progress_pct INTEGER,
    progress_message TEXT,
    
    error_message TEXT,
    
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (component)
);

-- Insert default status
INSERT INTO system_status (component, status) VALUES
    ('screener', 'stopped'),
    ('committee', 'stopped'),
    ('ingestion', 'stopped')
ON CONFLICT (component) DO NOTHING;

-- -----------------------------------------------------------------------------
-- Job History
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_history (
    id SERIAL PRIMARY KEY,
    
    job_type VARCHAR(50) NOT NULL,  -- screener_full, screener_ticker, committee, ingestion
    ticker VARCHAR(10),
    
    status VARCHAR(20) NOT NULL,  -- started, completed, failed
    
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    duration_seconds NUMERIC(10, 2),
    
    records_processed INTEGER,
    error_message TEXT,
    
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_job_history_type ON job_history (job_type, started_at DESC);

-- =============================================================================
-- HELPER VIEWS
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Latest screener scores with signals
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_latest_scores AS
SELECT 
    s.ticker,
    s.date,
    s.sentiment_score,
    s.fundamental_score,
    s.growth_score,
    s.dividend_score,
    s.technical_score,
    s.gap_score,
    s.likelihood_score,
    s.analyst_positivity,
    s.target_upside_pct,
    s.insider_signal,
    s.institutional_signal,
    s.composite_score,
    s.total_score,
    t.signal_type,
    t.signal_strength,
    t.signal_color
FROM screener_scores s
LEFT JOIN trading_signals t ON s.ticker = t.ticker AND s.date = t.date
WHERE s.date = (SELECT MAX(date) FROM screener_scores WHERE ticker = s.ticker);

-- -----------------------------------------------------------------------------
-- Latest committee decisions
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_latest_decisions AS
SELECT 
    c.*,
    s.total_score as screener_score,
    t.signal_type as screener_signal
FROM committee_decisions c
LEFT JOIN screener_scores s ON c.ticker = s.ticker AND c.date = s.date
LEFT JOIN trading_signals t ON c.ticker = t.ticker AND c.date = t.date
WHERE c.date = (SELECT MAX(date) FROM committee_decisions WHERE ticker = c.ticker AND persona = c.persona);

-- =============================================================================
-- GRANTS (if needed for separate app user)
-- =============================================================================
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO alpha;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alpha;

-- =============================================================================
-- Done!
-- =============================================================================
