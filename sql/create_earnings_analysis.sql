-- Earnings Analysis Table Migration
-- Run this to create the earnings_analysis table

CREATE TABLE IF NOT EXISTS earnings_analysis (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    filing_date DATE NOT NULL,
    filing_type VARCHAR(20),
    fiscal_period VARCHAR(20),

    -- Quantitative data
    eps_actual FLOAT,
    eps_estimate FLOAT,
    eps_surprise_pct FLOAT,
    revenue_actual FLOAT,
    revenue_estimate FLOAT,
    revenue_surprise_pct FLOAT,

    -- Guidance
    guidance_direction VARCHAR(20),  -- RAISED, LOWERED, MAINTAINED, NOT_PROVIDED
    guidance_summary TEXT,

    -- AI Analysis
    overall_sentiment VARCHAR(20),  -- VERY_BULLISH, BULLISH, NEUTRAL, BEARISH, VERY_BEARISH
    sentiment_score INT,  -- 0-100
    management_tone VARCHAR(20),  -- CONFIDENT, CAUTIOUS, DEFENSIVE, OPTIMISTIC
    key_highlights JSONB,
    concerns JSONB,

    -- Score Impact
    score_adjustment INT DEFAULT 0,  -- -20 to +20
    adjustment_reason TEXT,

    -- Raw data
    transcript_url TEXT,
    transcript_text TEXT,
    analysis_date TIMESTAMP DEFAULT NOW(),

    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(ticker, filing_date)
);

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_earnings_ticker ON earnings_analysis(ticker);
CREATE INDEX IF NOT EXISTS idx_earnings_date ON earnings_analysis(filing_date);
CREATE INDEX IF NOT EXISTS idx_earnings_sentiment ON earnings_analysis(overall_sentiment);

-- Add earnings_adjustment column to screener_scores if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'screener_scores' AND column_name = 'earnings_adjustment'
    ) THEN
        ALTER TABLE screener_scores ADD COLUMN earnings_adjustment INT DEFAULT 0;
    END IF;
END $$;

-- Comments
COMMENT ON TABLE earnings_analysis IS 'AI-analyzed earnings transcripts and their impact on signals';
COMMENT ON COLUMN earnings_analysis.score_adjustment IS 'Impact on screener score: -20 (disaster) to +20 (exceptional)';
COMMENT ON COLUMN earnings_analysis.guidance_direction IS 'RAISED, LOWERED, MAINTAINED, or NOT_PROVIDED';