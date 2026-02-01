-- =============================================================================
-- AI TRADING SYSTEM - DATABASE MIGRATIONS
-- =============================================================================
--
-- Run this file to create all tables needed for the AI trading system.
--
-- Usage:
--   psql -d alpha_platform -f migrations/ai_system_tables.sql
--
-- Tables created:
--   1. setup_cards - RAG memory for similar setups
--   2. ai_recommendations - Logged recommendations with outcomes
--   3. model_performance - Performance snapshots over time
--   4. drift_alerts - Detected drift and alerts
--   5. ml_model_metadata - Model versioning and training info
--
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. SETUP CARDS (RAG Memory)
-- -----------------------------------------------------------------------------
-- Stores historical trading setups for similarity search.
-- CRITICAL: Only contains data known at T0 (no future leakage).

CREATE TABLE IF NOT EXISTS setup_cards (
    card_id VARCHAR(64) PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    setup_date DATE NOT NULL,

    -- Features at T0 (decision time)
    sentiment_score DECIMAL(5,2),
    fundamental_score DECIMAL(5,2),
    technical_score DECIMAL(5,2),
    options_flow_score DECIMAL(5,2),
    short_squeeze_score DECIMAL(5,2),
    total_score DECIMAL(5,2),

    -- Market context at T0
    sector VARCHAR(50),
    market_cap VARCHAR(10),  -- MEGA, LARGE, MID, SMALL
    vix_level DECIMAL(5,2),
    regime_score DECIMAL(5,2),

    -- Signal at T0
    signal_type VARCHAR(20),
    ml_probability DECIMAL(5,4),
    expected_value DECIMAL(8,6),

    -- Event context at T0
    days_to_earnings INTEGER,
    had_recent_earnings BOOLEAN DEFAULT FALSE,

    -- Trade plan at T0
    entry_price DECIMAL(12,4),
    planned_stop_pct DECIMAL(5,4),
    planned_target_pct DECIMAL(5,4),
    planned_horizon INTEGER DEFAULT 5,

    -- Outcome (filled AFTER trade closes - never used for retrieval)
    outcome VARCHAR(10),  -- WIN, LOSS, SCRATCH, NULL=open
    actual_return DECIMAL(8,4),
    actual_holding_days INTEGER,
    exit_reason VARCHAR(20),  -- TARGET, STOP, TIME, MANUAL
    exit_date DATE,

    -- Feature vector for similarity search (optional, for optimization)
    feature_vector FLOAT[],

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    notes TEXT,

    UNIQUE(ticker, setup_date)
);

CREATE INDEX IF NOT EXISTS idx_setup_cards_date ON setup_cards(setup_date DESC);
CREATE INDEX IF NOT EXISTS idx_setup_cards_ticker ON setup_cards(ticker);
CREATE INDEX IF NOT EXISTS idx_setup_cards_outcome ON setup_cards(outcome);
CREATE INDEX IF NOT EXISTS idx_setup_cards_sector ON setup_cards(sector);
CREATE INDEX IF NOT EXISTS idx_setup_cards_signal ON setup_cards(signal_type);

-- -----------------------------------------------------------------------------
-- 2. AI RECOMMENDATIONS
-- -----------------------------------------------------------------------------
-- Logs all AI recommendations for feedback loop.

CREATE TABLE IF NOT EXISTS ai_recommendations (
    id SERIAL PRIMARY KEY,

    -- Recommendation details
    ticker VARCHAR(10) NOT NULL,
    recommendation_date DATE NOT NULL,
    recommendation_type VARCHAR(20),  -- STRONG_BUY, BUY, HOLD, SELL, SKIP

    -- ML model outputs
    ml_probability DECIMAL(5,4),
    ml_ev DECIMAL(8,6),
    ml_confidence VARCHAR(20),  -- HIGH, MEDIUM, LOW

    -- Meta-labeler outputs
    meta_probability DECIMAL(5,4),
    combined_probability DECIMAL(5,4),

    -- Decision layer outputs
    approved BOOLEAN,
    rejection_reasons TEXT[],

    -- Trade plan
    entry_price DECIMAL(12,4),
    stop_loss DECIMAL(12,4),
    target_price DECIMAL(12,4),
    position_size_pct DECIMAL(5,4),
    position_shares INTEGER,
    horizon_days INTEGER DEFAULT 5,

    -- Context at time
    vix_level DECIMAL(5,2),
    regime_score DECIMAL(5,2),
    similar_setups_count INTEGER,
    similar_setups_win_rate DECIMAL(5,4),

    -- Outcome (filled later by update_outcome)
    outcome VARCHAR(10),  -- WIN, LOSS, SCRATCH, OPEN
    actual_return DECIMAL(8,4),
    actual_holding_days INTEGER,
    exit_reason VARCHAR(20),
    exit_date DATE,

    -- LLM analysis summary
    llm_summary TEXT,
    llm_recommendation VARCHAR(20),
    llm_confidence VARCHAR(20),

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(ticker, recommendation_date)
);

CREATE INDEX IF NOT EXISTS idx_ai_recs_date ON ai_recommendations(recommendation_date DESC);
CREATE INDEX IF NOT EXISTS idx_ai_recs_ticker ON ai_recommendations(ticker);
CREATE INDEX IF NOT EXISTS idx_ai_recs_outcome ON ai_recommendations(outcome);
CREATE INDEX IF NOT EXISTS idx_ai_recs_type ON ai_recommendations(recommendation_type);
CREATE INDEX IF NOT EXISTS idx_ai_recs_approved ON ai_recommendations(approved);

-- -----------------------------------------------------------------------------
-- 3. MODEL PERFORMANCE SNAPSHOTS
-- -----------------------------------------------------------------------------
-- Tracks model performance over time for monitoring.

CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    period_days INTEGER NOT NULL,  -- e.g., 7, 14, 30

    -- Volume metrics
    total_recommendations INTEGER,
    trades_taken INTEGER,
    trades_skipped INTEGER,
    closed_trades INTEGER,

    -- Outcome metrics
    wins INTEGER,
    losses INTEGER,
    scratches INTEGER,
    win_rate DECIMAL(5,4),

    -- Return metrics
    avg_return DECIMAL(8,4),
    median_return DECIMAL(8,4),
    total_return DECIMAL(10,4),
    best_trade DECIMAL(8,4),
    worst_trade DECIMAL(8,4),

    -- Calibration metrics
    avg_predicted_prob DECIMAL(5,4),
    actual_hit_rate DECIMAL(5,4),
    calibration_error DECIMAL(6,4),
    brier_score DECIMAL(6,4),

    -- By regime breakdown (JSON)
    metrics_by_regime JSONB,

    -- By signal type breakdown (JSON)
    metrics_by_signal JSONB,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_perf_date ON model_performance(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_perf_period ON model_performance(period_days);

-- -----------------------------------------------------------------------------
-- 4. DRIFT ALERTS
-- -----------------------------------------------------------------------------
-- Records detected drift and alerts.

CREATE TABLE IF NOT EXISTS drift_alerts (
    id SERIAL PRIMARY KEY,
    alert_time TIMESTAMP NOT NULL,

    -- Alert details
    drift_type VARCHAR(50),  -- CALIBRATION, PERFORMANCE, FEATURE, REGIME
    alert_level VARCHAR(20),  -- INFO, WARNING, CRITICAL
    metric_name VARCHAR(50),

    -- Values
    current_value DECIMAL(10,4),
    expected_value DECIMAL(10,4),
    threshold DECIMAL(10,4),
    deviation_pct DECIMAL(8,4),

    -- Context
    message TEXT,
    recommended_action TEXT,

    -- Resolution
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolution_notes TEXT,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_time ON drift_alerts(alert_time DESC);
CREATE INDEX IF NOT EXISTS idx_drift_type ON drift_alerts(drift_type);
CREATE INDEX IF NOT EXISTS idx_drift_level ON drift_alerts(alert_level);
CREATE INDEX IF NOT EXISTS idx_drift_resolved ON drift_alerts(resolved);

-- -----------------------------------------------------------------------------
-- 5. ML MODEL METADATA
-- -----------------------------------------------------------------------------
-- Tracks model versions and training info.

CREATE TABLE IF NOT EXISTS ml_model_metadata (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,

    -- Training info
    trained_at TIMESTAMP NOT NULL,
    training_samples INTEGER,
    feature_count INTEGER,
    target_horizon INTEGER,

    -- Validation metrics
    mean_auc DECIMAL(5,4),
    mean_accuracy DECIMAL(5,4),
    mean_win_rate DECIMAL(5,4),
    mean_return DECIMAL(8,4),
    brier_score DECIMAL(6,4),

    -- Calibration
    is_well_calibrated BOOLEAN,
    calibration_error DECIMAL(6,4),

    -- Baseline comparison
    beats_baseline BOOLEAN,
    baseline_auc DECIMAL(5,4),
    improvement_vs_baseline DECIMAL(6,4),

    -- Feature importance (JSON)
    feature_importance JSONB,

    -- Model file path
    model_path VARCHAR(255),

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    deactivated_at TIMESTAMP,
    deactivation_reason TEXT,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(model_name, version)
);

CREATE INDEX IF NOT EXISTS idx_model_name ON ml_model_metadata(model_name);
CREATE INDEX IF NOT EXISTS idx_model_active ON ml_model_metadata(is_active);
CREATE INDEX IF NOT EXISTS idx_model_trained ON ml_model_metadata(trained_at DESC);

-- -----------------------------------------------------------------------------
-- 6. TRADE JOURNAL (Enhanced)
-- -----------------------------------------------------------------------------
-- Enhanced trade journal with AI attribution.

CREATE TABLE IF NOT EXISTS trade_journal_ai (
    id SERIAL PRIMARY KEY,

    -- Trade details
    ticker VARCHAR(10) NOT NULL,
    entry_date DATE NOT NULL,
    entry_price DECIMAL(12,4) NOT NULL,
    direction VARCHAR(10) DEFAULT 'LONG',  -- LONG, SHORT
    shares INTEGER DEFAULT 1,

    -- Exit details
    exit_date DATE,
    exit_price DECIMAL(12,4),
    exit_reason VARCHAR(20),  -- TARGET, STOP, TIME, MANUAL

    -- Returns
    gross_return DECIMAL(8,4),
    net_return DECIMAL(8,4),  -- After costs
    dollar_pnl DECIMAL(12,2),

    -- AI Attribution
    ai_recommendation_id INTEGER REFERENCES ai_recommendations(id),
    ml_probability_at_entry DECIMAL(5,4),
    ml_ev_at_entry DECIMAL(8,6),
    similar_setups_win_rate DECIMAL(5,4),

    -- Signals at entry
    entry_signals JSONB,  -- All signal values at entry

    -- Analysis
    followed_ai BOOLEAN,  -- Did trader follow AI recommendation?
    ai_was_correct BOOLEAN,  -- Was AI right?

    -- Notes
    entry_notes TEXT,
    exit_notes TEXT,
    lessons_learned TEXT,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(ticker, entry_date, direction)
);

CREATE INDEX IF NOT EXISTS idx_journal_ticker ON trade_journal_ai(ticker);
CREATE INDEX IF NOT EXISTS idx_journal_entry ON trade_journal_ai(entry_date DESC);
CREATE INDEX IF NOT EXISTS idx_journal_ai_rec ON trade_journal_ai(ai_recommendation_id);

-- -----------------------------------------------------------------------------
-- HELPER VIEWS
-- -----------------------------------------------------------------------------

-- Recent recommendations with outcomes
CREATE OR REPLACE VIEW v_recent_recommendations AS
SELECT
    ticker,
    recommendation_date,
    recommendation_type,
    ml_probability,
    ml_ev,
    approved,
    outcome,
    actual_return,
    CASE WHEN approved AND outcome = 'WIN' THEN 1 ELSE 0 END as was_correct
FROM ai_recommendations
WHERE recommendation_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY recommendation_date DESC;

-- Model performance trend
CREATE OR REPLACE VIEW v_performance_trend AS
SELECT
    snapshot_date,
    period_days,
    win_rate,
    avg_return,
    calibration_error,
    LAG(win_rate) OVER (ORDER BY snapshot_date) as prev_win_rate,
    win_rate - LAG(win_rate) OVER (ORDER BY snapshot_date) as win_rate_change
FROM model_performance
WHERE period_days = 30
ORDER BY snapshot_date DESC;

-- Signal effectiveness by type
CREATE OR REPLACE VIEW v_signal_effectiveness AS
SELECT
    recommendation_type,
    COUNT(*) as total_count,
    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
    AVG(CASE WHEN outcome IS NOT NULL THEN actual_return END) as avg_return,
    AVG(ml_probability) as avg_predicted_prob,
    AVG(CASE WHEN outcome = 'WIN' THEN 1.0 ELSE 0.0 END) as actual_win_rate,
    AVG(ml_probability) - AVG(CASE WHEN outcome = 'WIN' THEN 1.0 ELSE 0.0 END) as calibration_error
FROM ai_recommendations
WHERE outcome IS NOT NULL
GROUP BY recommendation_type
ORDER BY actual_win_rate DESC;

-- -----------------------------------------------------------------------------
-- FUNCTIONS
-- -----------------------------------------------------------------------------

-- Function to update recommendation outcome
CREATE OR REPLACE FUNCTION update_recommendation_outcome(
    p_ticker VARCHAR(10),
    p_rec_date DATE,
    p_outcome VARCHAR(10),
    p_return DECIMAL(8,4),
    p_holding_days INTEGER,
    p_exit_reason VARCHAR(20)
) RETURNS BOOLEAN AS $$
BEGIN
    UPDATE ai_recommendations
    SET
        outcome = p_outcome,
        actual_return = p_return,
        actual_holding_days = p_holding_days,
        exit_reason = p_exit_reason,
        exit_date = CURRENT_DATE,
        updated_at = NOW()
    WHERE ticker = p_ticker AND recommendation_date = p_rec_date;

    -- Also update setup card if exists
    UPDATE setup_cards
    SET
        outcome = p_outcome,
        actual_return = p_return,
        actual_holding_days = p_holding_days,
        exit_reason = p_exit_reason,
        exit_date = CURRENT_DATE,
        updated_at = NOW()
    WHERE ticker = p_ticker AND setup_date = p_rec_date;

    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Function to get calibration metrics
CREATE OR REPLACE FUNCTION get_calibration_metrics(p_days INTEGER DEFAULT 30)
RETURNS TABLE (
    probability_bucket TEXT,
    predicted_prob DECIMAL,
    actual_rate DECIMAL,
    sample_count BIGINT,
    calibration_error DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        CASE
            WHEN ml_probability < 0.55 THEN '50-55%'
            WHEN ml_probability < 0.60 THEN '55-60%'
            WHEN ml_probability < 0.65 THEN '60-65%'
            WHEN ml_probability < 0.70 THEN '65-70%'
            WHEN ml_probability < 0.75 THEN '70-75%'
            ELSE '75%+'
        END as probability_bucket,
        AVG(ml_probability) as predicted_prob,
        AVG(CASE WHEN outcome = 'WIN' THEN 1.0 ELSE 0.0 END) as actual_rate,
        COUNT(*) as sample_count,
        AVG(ml_probability) - AVG(CASE WHEN outcome = 'WIN' THEN 1.0 ELSE 0.0 END) as calibration_error
    FROM ai_recommendations
    WHERE recommendation_date >= CURRENT_DATE - (p_days || ' days')::INTERVAL
      AND outcome IS NOT NULL
      AND approved = TRUE
    GROUP BY 1
    ORDER BY 2;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- GRANTS (adjust as needed)
-- -----------------------------------------------------------------------------

-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_app_user;

-- -----------------------------------------------------------------------------
-- DONE
-- -----------------------------------------------------------------------------

SELECT 'AI Trading System tables created successfully!' as status;