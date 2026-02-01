-- ============================================================================
-- MULTI-FACTOR ALPHA MODEL - DATABASE TABLES
-- ============================================================================
--
-- Run this SQL directly in your PostgreSQL database:
--   psql -U alpha -d alpha_platform -f alpha_model_tables.sql
--
-- Or run in DBeaver/pgAdmin by copying and pasting
--
-- Author: HH Research Platform
-- Date: December 2025
-- ============================================================================

-- Store alpha predictions for each ticker
CREATE TABLE IF NOT EXISTS alpha_predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    prediction_date DATE NOT NULL,

    -- Expected returns (as percentages, e.g., 2.5 = 2.5%)
    expected_return_5d DECIMAL(8, 4),
    expected_return_10d DECIMAL(8, 4),
    expected_return_20d DECIMAL(8, 4),

    -- Confidence intervals
    ci_lower_5d DECIMAL(8, 4),
    ci_upper_5d DECIMAL(8, 4),
    ci_lower_10d DECIMAL(8, 4),
    ci_upper_10d DECIMAL(8, 4),

    -- Probabilities (0-1)
    prob_positive_5d DECIMAL(5, 4),
    prob_positive_10d DECIMAL(5, 4),
    prob_beat_market_5d DECIMAL(5, 4),

    -- Risk metrics
    sharpe_implied DECIMAL(6, 3),
    information_ratio DECIMAL(6, 3),
    model_uncertainty DECIMAL(6, 3),

    -- Signal
    signal VARCHAR(20),  -- STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    conviction DECIMAL(4, 3),  -- 0-1
    confidence VARCHAR(10),  -- HIGH, MEDIUM, LOW
    position_size_recommended DECIMAL(4, 2),  -- Multiplier (e.g., 1.5x)

    -- Context
    regime VARCHAR(20),
    sector VARCHAR(50),
    sector_group VARCHAR(20),

    -- Factor contributions (JSON)
    factor_contributions JSONB,
    top_bullish_factors JSONB,
    top_bearish_factors JSONB,

    -- Metadata
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    UNIQUE(ticker, prediction_date)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_alpha_predictions_ticker_date
    ON alpha_predictions(ticker, prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_alpha_predictions_signal
    ON alpha_predictions(signal, prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_alpha_predictions_date
    ON alpha_predictions(prediction_date DESC);


-- Store actual returns for tracking prediction accuracy
CREATE TABLE IF NOT EXISTS alpha_prediction_outcomes (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES alpha_predictions(id),
    ticker VARCHAR(10) NOT NULL,
    prediction_date DATE NOT NULL,

    -- Actual returns
    actual_return_5d DECIMAL(8, 4),
    actual_return_10d DECIMAL(8, 4),
    actual_return_20d DECIMAL(8, 4),

    -- Was prediction correct?
    correct_direction_5d BOOLEAN,
    correct_direction_10d BOOLEAN,

    -- Prediction vs Actual
    prediction_error_5d DECIMAL(8, 4),
    prediction_error_10d DECIMAL(8, 4),

    -- Calculated after returns are known
    calculated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(ticker, prediction_date)
);

CREATE INDEX IF NOT EXISTS idx_alpha_outcomes_ticker_date
    ON alpha_prediction_outcomes(ticker, prediction_date DESC);


-- Store model validation history
CREATE TABLE IF NOT EXISTS alpha_model_validations (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(20) NOT NULL,
    trained_at TIMESTAMP NOT NULL,

    -- Overall metrics
    overall_ic DECIMAL(6, 4),
    overall_icir DECIMAL(6, 4),
    overall_r2 DECIMAL(6, 4),

    -- Walk-forward metrics
    n_folds INTEGER,
    mean_ic_oos DECIMAL(6, 4),
    std_ic_oos DECIMAL(6, 4),

    -- Statistical significance
    ic_tstat DECIMAL(6, 3),
    ic_pvalue DECIMAL(6, 4),
    beats_baseline BOOLEAN,

    -- Training details
    n_samples INTEGER,
    n_features INTEGER,
    train_start DATE,
    train_end DATE,

    -- Full report (JSON)
    full_report JSONB,

    created_at TIMESTAMP DEFAULT NOW()
);


-- Store factor weights for each model version
CREATE TABLE IF NOT EXISTS alpha_factor_weights (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(20) NOT NULL,

    -- Context
    regime VARCHAR(20),  -- NULL for global model
    sector_group VARCHAR(20),  -- NULL for global model
    horizon_days INTEGER NOT NULL,

    -- Weights (JSON: factor_name -> weight)
    weights JSONB NOT NULL,
    intercept DECIMAL(8, 4),

    -- Performance metrics
    r_squared DECIMAL(6, 4),
    mse DECIMAL(10, 6),
    information_coefficient DECIMAL(6, 4),
    n_samples INTEGER,

    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(model_version, regime, sector_group, horizon_days)
);

CREATE INDEX IF NOT EXISTS idx_alpha_weights_version
    ON alpha_factor_weights(model_version);


-- ============================================================================
-- VIEWS FOR ANALYSIS
-- ============================================================================

-- View for tracking prediction accuracy over time
CREATE OR REPLACE VIEW v_alpha_prediction_accuracy AS
SELECT
    DATE_TRUNC('week', p.prediction_date) as week,
    p.signal,
    COUNT(*) as n_predictions,
    AVG(CASE WHEN o.correct_direction_5d THEN 1 ELSE 0 END) as accuracy_5d,
    AVG(CASE WHEN o.correct_direction_10d THEN 1 ELSE 0 END) as accuracy_10d,
    AVG(o.actual_return_5d) as avg_actual_return_5d,
    AVG(p.expected_return_5d) as avg_predicted_return_5d,
    AVG(ABS(o.prediction_error_5d)) as mae_5d,
    CORR(p.expected_return_5d, o.actual_return_5d) as ic_5d
FROM alpha_predictions p
LEFT JOIN alpha_prediction_outcomes o
    ON p.ticker = o.ticker AND p.prediction_date = o.prediction_date
WHERE o.actual_return_5d IS NOT NULL
GROUP BY DATE_TRUNC('week', p.prediction_date), p.signal
ORDER BY week DESC, signal;


-- View for factor performance analysis
CREATE OR REPLACE VIEW v_alpha_factor_performance AS
WITH latest_weights AS (
    SELECT DISTINCT ON (horizon_days)
        model_version,
        horizon_days,
        weights,
        information_coefficient,
        n_samples
    FROM alpha_factor_weights
    WHERE regime IS NULL AND sector_group IS NULL
    ORDER BY horizon_days, created_at DESC
)
SELECT
    lw.model_version,
    lw.horizon_days,
    key as factor_name,
    value::decimal as weight,
    lw.information_coefficient,
    lw.n_samples
FROM latest_weights lw,
LATERAL jsonb_each_text(lw.weights);


-- ============================================================================
-- HELPER FUNCTION
-- ============================================================================

-- Function to update prediction outcomes (call daily after market close)
CREATE OR REPLACE FUNCTION update_alpha_outcomes()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER := 0;
BEGIN
    -- Update 5-day outcomes
    INSERT INTO alpha_prediction_outcomes (
        prediction_id, ticker, prediction_date,
        actual_return_5d, correct_direction_5d, prediction_error_5d
    )
    SELECT
        p.id,
        p.ticker,
        p.prediction_date,
        ((pr_future.close - pr_base.close) / pr_base.close * 100) as actual_return,
        CASE
            WHEN p.expected_return_5d > 0 AND pr_future.close > pr_base.close THEN TRUE
            WHEN p.expected_return_5d < 0 AND pr_future.close < pr_base.close THEN TRUE
            WHEN p.expected_return_5d = 0 THEN NULL
            ELSE FALSE
        END as correct_direction,
        ((pr_future.close - pr_base.close) / pr_base.close * 100) - p.expected_return_5d as error
    FROM alpha_predictions p
    JOIN prices pr_base ON p.ticker = pr_base.ticker AND p.prediction_date = pr_base.date
    JOIN prices pr_future ON p.ticker = pr_future.ticker
        AND pr_future.date = (
            SELECT MIN(date) FROM prices
            WHERE ticker = p.ticker AND date >= p.prediction_date + INTERVAL '5 days'
        )
    WHERE NOT EXISTS (
        SELECT 1 FROM alpha_prediction_outcomes o
        WHERE o.ticker = p.ticker AND o.prediction_date = p.prediction_date
        AND o.actual_return_5d IS NOT NULL
    )
    AND p.prediction_date <= CURRENT_DATE - INTERVAL '5 days'
    ON CONFLICT (ticker, prediction_date)
    DO UPDATE SET
        actual_return_5d = EXCLUDED.actual_return_5d,
        correct_direction_5d = EXCLUDED.correct_direction_5d,
        prediction_error_5d = EXCLUDED.prediction_error_5d,
        calculated_at = NOW();

    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- VERIFY INSTALLATION
-- ============================================================================

-- Run this to verify tables were created:
SELECT
    table_name,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as columns
FROM information_schema.tables t
WHERE table_schema = 'public'
AND table_name LIKE 'alpha_%'
ORDER BY table_name;