-- =============================================================================
-- Alpha Enhancements - Database Setup
-- =============================================================================
-- Run this script to create/update tables required for alpha enhancements
--
-- Usage: psql -U alpha -d alpha_platform -f setup_alpha_tables.sql
-- =============================================================================

-- 1. Create alpha_predictions table if not exists
CREATE TABLE IF NOT EXISTS alpha_predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    prediction_date DATE NOT NULL,
    expected_return_5d DECIMAL(8, 4),
    expected_return_10d DECIMAL(8, 4),
    expected_return_20d DECIMAL(8, 4),
    ci_lower_5d DECIMAL(8, 4),
    ci_upper_5d DECIMAL(8, 4),
    ci_lower_10d DECIMAL(8, 4),
    ci_upper_10d DECIMAL(8, 4),
    prob_positive_5d DECIMAL(5, 4),
    prob_positive_10d DECIMAL(5, 4),
    prob_beat_market_5d DECIMAL(5, 4),
    sharpe_implied DECIMAL(6, 3),
    information_ratio DECIMAL(6, 3),
    model_uncertainty DECIMAL(6, 3),
    signal VARCHAR(20),
    conviction DECIMAL(4, 3),
    confidence VARCHAR(10),
    position_size_recommended DECIMAL(4, 2),
    regime VARCHAR(20),
    sector VARCHAR(50),
    sector_group VARCHAR(20),
    factor_contributions JSONB,
    top_bullish_factors JSONB,
    top_bearish_factors JSONB,
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),

    -- Outcome tracking columns (added for enhancements)
    actual_return_5d DECIMAL(8, 4),
    actual_return_10d DECIMAL(8, 4),
    actual_return_20d DECIMAL(8, 4),
    direction_correct_5d BOOLEAN,
    direction_correct_10d BOOLEAN,
    within_ci_5d BOOLEAN,
    prediction_error_5d DECIMAL(8, 4),
    updated_at TIMESTAMP,

    UNIQUE(ticker, prediction_date)
);

-- 2. Add outcome columns if table exists but columns don't
DO $$
BEGIN
    -- Add actual_return_5d if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'alpha_predictions' AND column_name = 'actual_return_5d') THEN
        ALTER TABLE alpha_predictions ADD COLUMN actual_return_5d DECIMAL(8, 4);
    END IF;

    -- Add actual_return_10d if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'alpha_predictions' AND column_name = 'actual_return_10d') THEN
        ALTER TABLE alpha_predictions ADD COLUMN actual_return_10d DECIMAL(8, 4);
    END IF;

    -- Add actual_return_20d if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'alpha_predictions' AND column_name = 'actual_return_20d') THEN
        ALTER TABLE alpha_predictions ADD COLUMN actual_return_20d DECIMAL(8, 4);
    END IF;

    -- Add direction_correct_5d if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'alpha_predictions' AND column_name = 'direction_correct_5d') THEN
        ALTER TABLE alpha_predictions ADD COLUMN direction_correct_5d BOOLEAN;
    END IF;

    -- Add direction_correct_10d if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'alpha_predictions' AND column_name = 'direction_correct_10d') THEN
        ALTER TABLE alpha_predictions ADD COLUMN direction_correct_10d BOOLEAN;
    END IF;

    -- Add within_ci_5d if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'alpha_predictions' AND column_name = 'within_ci_5d') THEN
        ALTER TABLE alpha_predictions ADD COLUMN within_ci_5d BOOLEAN;
    END IF;

    -- Add prediction_error_5d if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'alpha_predictions' AND column_name = 'prediction_error_5d') THEN
        ALTER TABLE alpha_predictions ADD COLUMN prediction_error_5d DECIMAL(8, 4);
    END IF;

    -- Add updated_at if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'alpha_predictions' AND column_name = 'updated_at') THEN
        ALTER TABLE alpha_predictions ADD COLUMN updated_at TIMESTAMP;
    END IF;
END $$;

-- 3. Create indexes
CREATE INDEX IF NOT EXISTS idx_alpha_predictions_ticker_date
    ON alpha_predictions(ticker, prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_alpha_predictions_signal
    ON alpha_predictions(signal, prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_alpha_predictions_actual
    ON alpha_predictions(prediction_date) WHERE actual_return_5d IS NOT NULL;

-- 4. Create earnings_calendar table if not exists (for catalyst detection)
CREATE TABLE IF NOT EXISTS earnings_calendar (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    earnings_date DATE NOT NULL,
    earnings_time VARCHAR(10),  -- 'BMO', 'AMC', 'Unknown'
    eps_estimate DECIMAL(10, 4),
    revenue_estimate DECIMAL(20, 2),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, earnings_date)
);

CREATE INDEX IF NOT EXISTS idx_earnings_calendar_date
    ON earnings_calendar(earnings_date, ticker);

-- 5. Create options_summary table if not exists (for IV rank)
CREATE TABLE IF NOT EXISTS options_summary (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    iv_rank DECIMAL(5, 2),
    iv_percentile DECIMAL(5, 2),
    implied_volatility DECIMAL(8, 4),
    put_call_ratio DECIMAL(6, 3),
    total_volume INTEGER,
    total_open_interest INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_options_summary_ticker_date
    ON options_summary(ticker, date DESC);

-- 6. Verify setup
DO $$
DECLARE
    tbl_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO tbl_count
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name IN ('alpha_predictions', 'earnings_calendar', 'options_summary');

    RAISE NOTICE 'Setup complete. Tables created/verified: %', tbl_count;
END $$;

-- Show table status
SELECT
    table_name,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
FROM information_schema.tables t
WHERE table_schema = 'public'
AND table_name IN ('alpha_predictions', 'earnings_calendar', 'options_summary', 'market_data', 'news')
ORDER BY table_name;