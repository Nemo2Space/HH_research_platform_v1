-- =============================================================================
-- SIGNAL HUB - Database Schema
-- =============================================================================
-- Signal snapshots for historical tracking and performance analysis
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Signal Snapshots - Historical signal data for performance tracking
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signal_snapshots (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    snapshot_date DATE NOT NULL,

    -- Signal at snapshot time
    today_signal VARCHAR(20) NOT NULL,  -- STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    today_score INTEGER NOT NULL,        -- 0-100
    longterm_score INTEGER NOT NULL,     -- 0-100
    risk_level VARCHAR(20),              -- LOW, MEDIUM, HIGH, EXTREME

    -- Price at snapshot
    price_at_snapshot NUMERIC(12, 4),

    -- Component scores (all 0-100)
    technical_score INTEGER DEFAULT 50,
    fundamental_score INTEGER DEFAULT 50,
    sentiment_score INTEGER DEFAULT 50,
    options_score INTEGER DEFAULT 50,
    earnings_score INTEGER DEFAULT 50,

    -- Prices for later comparison (filled by background job)
    price_1d_later NUMERIC(12, 4),
    price_7d_later NUMERIC(12, 4),
    price_30d_later NUMERIC(12, 4),

    -- Was signal correct? (filled by background job)
    correct_1d BOOLEAN,
    correct_7d BOOLEAN,
    correct_30d BOOLEAN,

    -- Full signal JSON for deep investigation
    full_signal_json JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- One snapshot per ticker per day
    UNIQUE (ticker, snapshot_date)
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_signal_snapshots_ticker ON signal_snapshots (ticker, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_signal_snapshots_date ON signal_snapshots (snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_signal_snapshots_signal ON signal_snapshots (today_signal, snapshot_date DESC);

-- Index for finding snapshots needing price updates
CREATE INDEX IF NOT EXISTS idx_signal_snapshots_needs_update
ON signal_snapshots (snapshot_date)
WHERE price_1d_later IS NULL OR price_7d_later IS NULL OR price_30d_later IS NULL;


-- -----------------------------------------------------------------------------
-- Market Overview Snapshots - Daily market state
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS market_overview_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL UNIQUE,

    -- Regime
    regime VARCHAR(20),           -- RISK_ON, RISK_OFF, NEUTRAL
    regime_score INTEGER,         -- 0-100

    -- Key metrics
    vix NUMERIC(8, 2),
    spy_change NUMERIC(8, 4),
    qqq_change NUMERIC(8, 4),

    -- Sector data
    leading_sector VARCHAR(50),
    lagging_sector VARCHAR(50),

    -- Economic events
    economic_events_json JSONB,

    -- AI Summary
    ai_summary TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_market_overview_date ON market_overview_snapshots (snapshot_date DESC);


-- -----------------------------------------------------------------------------
-- Signal Performance Summary - Aggregated accuracy stats
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signal_performance_summary (
    id SERIAL PRIMARY KEY,

    -- Time period
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,

    -- Overall stats
    total_signals INTEGER DEFAULT 0,
    correct_1d INTEGER DEFAULT 0,
    correct_7d INTEGER DEFAULT 0,
    correct_30d INTEGER DEFAULT 0,

    -- By signal type
    strong_buy_count INTEGER DEFAULT 0,
    strong_buy_correct_7d INTEGER DEFAULT 0,
    buy_count INTEGER DEFAULT 0,
    buy_correct_7d INTEGER DEFAULT 0,
    sell_count INTEGER DEFAULT 0,
    sell_correct_7d INTEGER DEFAULT 0,
    strong_sell_count INTEGER DEFAULT 0,
    strong_sell_correct_7d INTEGER DEFAULT 0,

    -- By component (which component was most accurate?)
    technical_accuracy NUMERIC(5, 2),
    fundamental_accuracy NUMERIC(5, 2),
    sentiment_accuracy NUMERIC(5, 2),
    options_accuracy NUMERIC(5, 2),
    earnings_accuracy NUMERIC(5, 2),

    -- Average returns
    avg_return_1d NUMERIC(8, 4),
    avg_return_7d NUMERIC(8, 4),
    avg_return_30d NUMERIC(8, 4),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (period_start, period_end)
);


-- -----------------------------------------------------------------------------
-- Function to update price_X_later fields
-- Run daily as a scheduled job
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_signal_snapshot_prices()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER := 0;
BEGIN
    -- Update 1-day later prices (for snapshots from yesterday)
    UPDATE signal_snapshots ss
    SET price_1d_later = p.close,
        correct_1d = CASE
            WHEN ss.today_signal IN ('STRONG_BUY', 'BUY') AND p.close > ss.price_at_snapshot THEN TRUE
            WHEN ss.today_signal IN ('STRONG_SELL', 'SELL') AND p.close < ss.price_at_snapshot THEN TRUE
            WHEN ss.today_signal = 'HOLD' THEN NULL
            ELSE FALSE
        END
    FROM prices p
    WHERE ss.ticker = p.ticker
      AND ss.snapshot_date = p.date - INTERVAL '1 day'
      AND ss.price_1d_later IS NULL;

    GET DIAGNOSTICS updated_count = ROW_COUNT;

    -- Update 7-day later prices
    UPDATE signal_snapshots ss
    SET price_7d_later = p.close,
        correct_7d = CASE
            WHEN ss.today_signal IN ('STRONG_BUY', 'BUY') AND p.close > ss.price_at_snapshot THEN TRUE
            WHEN ss.today_signal IN ('STRONG_SELL', 'SELL') AND p.close < ss.price_at_snapshot THEN TRUE
            WHEN ss.today_signal = 'HOLD' THEN NULL
            ELSE FALSE
        END
    FROM prices p
    WHERE ss.ticker = p.ticker
      AND ss.snapshot_date = p.date - INTERVAL '7 days'
      AND ss.price_7d_later IS NULL;

    -- Update 30-day later prices
    UPDATE signal_snapshots ss
    SET price_30d_later = p.close,
        correct_30d = CASE
            WHEN ss.today_signal IN ('STRONG_BUY', 'BUY') AND p.close > ss.price_at_snapshot THEN TRUE
            WHEN ss.today_signal IN ('STRONG_SELL', 'SELL') AND p.close < ss.price_at_snapshot THEN TRUE
            WHEN ss.today_signal = 'HOLD' THEN NULL
            ELSE FALSE
        END
    FROM prices p
    WHERE ss.ticker = p.ticker
      AND ss.snapshot_date = p.date - INTERVAL '30 days'
      AND ss.price_30d_later IS NULL;

    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;


-- -----------------------------------------------------------------------------
-- View for quick signal performance stats
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW signal_performance_view AS
SELECT
    today_signal,
    COUNT(*) as total_signals,

    -- 1-day accuracy
    ROUND(100.0 * SUM(CASE WHEN correct_1d = TRUE THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN correct_1d IS NOT NULL THEN 1 ELSE 0 END), 0), 1) as accuracy_1d,

    -- 7-day accuracy
    ROUND(100.0 * SUM(CASE WHEN correct_7d = TRUE THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN correct_7d IS NOT NULL THEN 1 ELSE 0 END), 0), 1) as accuracy_7d,

    -- 30-day accuracy
    ROUND(100.0 * SUM(CASE WHEN correct_30d = TRUE THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN correct_30d IS NOT NULL THEN 1 ELSE 0 END), 0), 1) as accuracy_30d,

    -- Average returns
    ROUND(AVG(CASE WHEN price_7d_later IS NOT NULL
              THEN (price_7d_later - price_at_snapshot) / price_at_snapshot * 100
              END)::numeric, 2) as avg_return_7d_pct

FROM signal_snapshots
WHERE snapshot_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY today_signal
ORDER BY today_signal;


-- -----------------------------------------------------------------------------
-- View for recent signal history per ticker
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW ticker_signal_history AS
SELECT
    ticker,
    snapshot_date,
    today_signal,
    today_score,
    price_at_snapshot,
    price_7d_later,
    correct_7d,
    ROUND(((price_7d_later - price_at_snapshot) / price_at_snapshot * 100)::numeric, 2) as return_7d_pct
FROM signal_snapshots
WHERE snapshot_date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY ticker, snapshot_date DESC;


-- =============================================================================
-- COMMENTS
-- =============================================================================
COMMENT ON TABLE signal_snapshots IS 'Daily snapshots of signals for historical tracking and performance analysis';
COMMENT ON TABLE market_overview_snapshots IS 'Daily market state snapshots';
COMMENT ON TABLE signal_performance_summary IS 'Aggregated signal performance statistics';
COMMENT ON VIEW signal_performance_view IS 'Quick view of signal accuracy by signal type';
COMMENT ON VIEW ticker_signal_history IS 'Signal history for individual tickers';