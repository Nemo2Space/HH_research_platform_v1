-- ============================================================
-- SIGNAL HUB DATABASE MIGRATION
-- Run this to add all missing tables and columns
-- ============================================================

-- 1. Create earnings_calendar table
CREATE TABLE IF NOT EXISTS earnings_calendar (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    earnings_date DATE NOT NULL,
    earnings_time VARCHAR(10),  -- 'BMO' (before market open), 'AMC' (after market close), 'TAS' (time after session)

    -- Estimates (before earnings)
    eps_estimate DECIMAL(10,4),
    revenue_estimate BIGINT,

    -- Actual results (after earnings)
    eps_actual DECIMAL(10,4),
    eps_surprise_pct DECIMAL(10,4),
    revenue_actual BIGINT,
    revenue_surprise_pct DECIMAL(10,4),

    -- Guidance
    guidance_direction VARCHAR(20),  -- 'raised', 'lowered', 'maintained', 'withdrawn', NULL

    -- Market reaction
    price_before DECIMAL(10,2),
    price_after DECIMAL(10,2),
    reaction_pct DECIMAL(10,4),

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(ticker, earnings_date)
);

-- Indexes for earnings_calendar
CREATE INDEX IF NOT EXISTS idx_earnings_calendar_ticker ON earnings_calendar(ticker);
CREATE INDEX IF NOT EXISTS idx_earnings_calendar_date ON earnings_calendar(earnings_date);
CREATE INDEX IF NOT EXISTS idx_earnings_calendar_upcoming ON earnings_calendar(earnings_date)
    WHERE earnings_date >= CURRENT_DATE;

-- 2. Add missing columns to screener_scores (if they don't exist)
DO $$
BEGIN
    -- Add sentiment_signal column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'screener_scores' AND column_name = 'sentiment_signal') THEN
        ALTER TABLE screener_scores ADD COLUMN sentiment_signal VARCHAR(20);
        RAISE NOTICE 'Added sentiment_signal column';
    END IF;

    -- Add article_count column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'screener_scores' AND column_name = 'article_count') THEN
        ALTER TABLE screener_scores ADD COLUMN article_count INTEGER DEFAULT 0;
        RAISE NOTICE 'Added article_count column';
    END IF;

    -- Add earnings-related columns
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'screener_scores' AND column_name = 'earnings_score') THEN
        ALTER TABLE screener_scores ADD COLUMN earnings_score INTEGER;
        RAISE NOTICE 'Added earnings_score column';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'screener_scores' AND column_name = 'earnings_signal') THEN
        ALTER TABLE screener_scores ADD COLUMN earnings_signal VARCHAR(20);
        RAISE NOTICE 'Added earnings_signal column';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'screener_scores' AND column_name = 'days_to_earnings') THEN
        ALTER TABLE screener_scores ADD COLUMN days_to_earnings INTEGER;
        RAISE NOTICE 'Added days_to_earnings column';
    END IF;
END $$;

-- 3. Add relevance_score column to news_articles (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'news_articles' AND column_name = 'relevance_score') THEN
        ALTER TABLE news_articles ADD COLUMN relevance_score INTEGER;
        RAISE NOTICE 'Added relevance_score column to news_articles';
    END IF;
END $$;

-- 4. Insert NKE earnings data (from yfinance)
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
) VALUES
    ('NKE', '2025-12-18', 'AMC', 0.38, 0.53, 41.33, 65.69, -11.04, NOW()),
    ('NKE', '2025-09-30', 'AMC', 0.27, 0.49, 81.32, NULL, NULL, NOW()),
    ('NKE', '2025-06-26', 'AMC', 0.12, 0.14, 13.48, NULL, NULL, NOW()),
    ('NKE', '2025-03-20', 'AMC', 0.29, 0.54, 85.52, NULL, NULL, NOW()),
    ('NKE', '2026-03-19', 'AMC', 0.48, NULL, NULL, NULL, NULL, NOW())  -- Next upcoming
ON CONFLICT (ticker, earnings_date) DO UPDATE SET
    eps_estimate = EXCLUDED.eps_estimate,
    eps_actual = EXCLUDED.eps_actual,
    eps_surprise_pct = EXCLUDED.eps_surprise_pct,
    price_before = EXCLUDED.price_before,
    reaction_pct = EXCLUDED.reaction_pct,
    updated_at = NOW();

-- 5. Verify everything
SELECT 'earnings_calendar' as table_name, COUNT(*) as row_count FROM earnings_calendar
UNION ALL
SELECT 'NKE earnings', COUNT(*) FROM earnings_calendar WHERE ticker = 'NKE';

-- Show NKE earnings
SELECT ticker, earnings_date, eps_estimate, eps_actual, eps_surprise_pct, reaction_pct
FROM earnings_calendar
WHERE ticker = 'NKE'
ORDER BY earnings_date DESC;

-- Check screener_scores columns
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'screener_scores'
ORDER BY ordinal_position;