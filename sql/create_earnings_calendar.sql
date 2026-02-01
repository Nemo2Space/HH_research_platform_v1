-- Earnings Intelligence Database Schema
-- Run this to create missing tables for earnings tracking

-- 1. Earnings Calendar - stores upcoming and past earnings dates
CREATE TABLE IF NOT EXISTS earnings_calendar (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    earnings_date DATE NOT NULL,
    earnings_time VARCHAR(10),  -- 'BMO' (before market open), 'AMC' (after market close)

    -- Actual results (filled after earnings)
    eps_estimate DECIMAL(10,4),
    eps_actual DECIMAL(10,4),
    eps_surprise_pct DECIMAL(10,4),
    revenue_estimate BIGINT,
    revenue_actual BIGINT,
    revenue_surprise_pct DECIMAL(10,4),

    -- Guidance
    guidance_direction VARCHAR(20),  -- 'raised', 'lowered', 'maintained', 'withdrawn'

    -- Market reaction
    price_before DECIMAL(10,2),
    price_after DECIMAL(10,2),
    reaction_pct DECIMAL(10,4),

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(ticker, earnings_date)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_earnings_calendar_ticker ON earnings_calendar(ticker);
CREATE INDEX IF NOT EXISTS idx_earnings_calendar_date ON earnings_calendar(earnings_date);
CREATE INDEX IF NOT EXISTS idx_earnings_calendar_ticker_date ON earnings_calendar(ticker, earnings_date DESC);

-- 2. Insert NKE earnings (the one we just found)
INSERT INTO earnings_calendar (
    ticker,
    earnings_date,
    earnings_time,
    eps_estimate,
    eps_actual,
    eps_surprise_pct,
    price_before,
    reaction_pct,
    updated_at
) VALUES (
    'NKE',
    '2024-12-18',  -- Adjust if different
    'AMC',
    0.38,
    0.53,
    41.33,
    65.69,
    -10.95,
    NOW()
) ON CONFLICT (ticker, earnings_date) DO UPDATE SET
    eps_estimate = EXCLUDED.eps_estimate,
    eps_actual = EXCLUDED.eps_actual,
    eps_surprise_pct = EXCLUDED.eps_surprise_pct,
    price_before = EXCLUDED.price_before,
    reaction_pct = EXCLUDED.reaction_pct,
    updated_at = NOW();

-- 3. Verify
SELECT * FROM earnings_calendar WHERE ticker = 'NKE';

-- 4. Check news articles for NKE
SELECT
    headline,
    source,
    published_at,
    ai_sentiment_fast,
    created_at
FROM news_articles
WHERE ticker = 'NKE'
ORDER BY COALESCE(published_at, created_at) DESC
LIMIT 15;

-- 5. Check sentiment scores
SELECT * FROM screener_scores WHERE ticker = 'NKE';