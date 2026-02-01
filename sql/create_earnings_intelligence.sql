-- Earnings Intelligence System - Database Schema
-- ============================================================
-- Creates tables for IES (Implied Expectations Score) and
-- ECS (Expectations Clearance Score) system.
-- Safe to re-run (uses IF NOT EXISTS).
-- ============================================================

-- MAIN TABLE: earnings_intelligence
CREATE TABLE IF NOT EXISTS earnings_intelligence (
    id SERIAL PRIMARY KEY,
    
    -- IDENTIFICATION
    ticker VARCHAR(20) NOT NULL,
    earnings_date DATE NOT NULL,
    earnings_timestamp TIMESTAMP,
    sector VARCHAR(50),
    market_cap BIGINT,
    
    -- IES COMPONENTS (Pre-Earnings Inputs)
    drift_20d FLOAT,
    rel_drift_20d FLOAT,
    iv FLOAT,
    iv_pctl FLOAT,
    implied_move_pct FLOAT,
    implied_move_pctl FLOAT,
    skew_shift FLOAT,
    revision_score FLOAT,
    confidence_lang_score FLOAT,
    
    -- IES OUTPUT
    ies FLOAT,
    ies_compute_timestamp TIMESTAMP,
    
    -- EXPECTATIONS REGIME
    regime VARCHAR(20),
    
    -- RAW EARNINGS DATA (Post-Earnings)
    eps_actual FLOAT,
    eps_consensus FLOAT,
    eps_surprise_pct FLOAT,
    revenue_actual BIGINT,
    revenue_consensus BIGINT,
    revenue_surprise_pct FLOAT,
    guidance_direction VARCHAR(20),
    guidance_numeric FLOAT,
    
    -- Z-SCORES (Post-Earnings)
    eps_z FLOAT,
    rev_z FLOAT,
    guidance_z FLOAT,
    event_z FLOAT,
    
    -- EQS COMPONENTS (Post-Earnings)
    guidance_score FLOAT,
    margin_score FLOAT,
    tone_score FLOAT,
    eqs FLOAT,
    
    -- ECS CALCULATION (Post-Earnings)
    required_z FLOAT,
    ecs VARCHAR(20),
    
    -- POSITION SCALING
    position_scale FLOAT,
    ies_penalty FLOAT,
    implied_move_penalty FLOAT,
    
    -- DATA QUALITY
    data_quality VARCHAR(10),
    missing_inputs TEXT[],
    suppression_reason VARCHAR(10),
    
    -- OUTCOME TRACKING
    pre_earnings_close FLOAT,
    post_earnings_open FLOAT,
    post_earnings_close FLOAT,
    gap_reaction FLOAT,
    intraday_reaction FLOAT,
    total_reaction FLOAT,
    reaction_5d FLOAT,
    reaction_10d FLOAT,
    
    -- METADATA
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(ticker, earnings_date)
);

-- INDEXES for earnings_intelligence
CREATE INDEX IF NOT EXISTS idx_ei_ticker ON earnings_intelligence(ticker);
CREATE INDEX IF NOT EXISTS idx_ei_earnings_date ON earnings_intelligence(earnings_date);
CREATE INDEX IF NOT EXISTS idx_ei_ticker_date ON earnings_intelligence(ticker, earnings_date DESC);
CREATE INDEX IF NOT EXISTS idx_ei_regime ON earnings_intelligence(regime);
CREATE INDEX IF NOT EXISTS idx_ei_ecs ON earnings_intelligence(ecs);
CREATE INDEX IF NOT EXISTS idx_ei_data_quality ON earnings_intelligence(data_quality);
CREATE INDEX IF NOT EXISTS idx_ei_sector ON earnings_intelligence(sector);
CREATE INDEX IF NOT EXISTS idx_ei_created ON earnings_intelligence(created_at DESC);

-- HISTORICAL Z-SCORE REFERENCE TABLE
CREATE TABLE IF NOT EXISTS earnings_history (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    earnings_date DATE NOT NULL,
    eps_surprise_pct FLOAT,
    revenue_surprise_pct FLOAT,
    guidance_direction VARCHAR(20),
    guidance_numeric FLOAT,
    total_reaction FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, earnings_date)
);

CREATE INDEX IF NOT EXISTS idx_eh_ticker ON earnings_history(ticker);
CREATE INDEX IF NOT EXISTS idx_eh_ticker_date ON earnings_history(ticker, earnings_date DESC);

-- IES CACHE TABLE
CREATE TABLE IF NOT EXISTS ies_cache (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    ies FLOAT,
    regime VARCHAR(20),
    implied_move_pctl FLOAT,
    position_scale FLOAT,
    data_quality VARCHAR(10),
    earnings_date DATE,
    days_to_earnings INT,
    in_compute_window BOOLEAN DEFAULT FALSE,
    in_action_window BOOLEAN DEFAULT FALSE,
    computed_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    UNIQUE(ticker)
);

CREATE INDEX IF NOT EXISTS idx_ies_cache_ticker ON ies_cache(ticker);
CREATE INDEX IF NOT EXISTS idx_ies_cache_expires ON ies_cache(expires_at);

-- UPDATE TRIGGER
CREATE OR REPLACE FUNCTION update_earnings_intelligence_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_ei_updated_at ON earnings_intelligence;
CREATE TRIGGER trigger_ei_updated_at
    BEFORE UPDATE ON earnings_intelligence
    FOR EACH ROW
    EXECUTE FUNCTION update_earnings_intelligence_timestamp();

-- TABLE COMMENTS
COMMENT ON TABLE earnings_intelligence IS 
    'Core table for Earnings Intelligence System - stores IES, ECS, and outcome tracking';
COMMENT ON COLUMN earnings_intelligence.ies IS 
    'Implied Expectations Score (0-100). Higher = higher expectations priced in';
COMMENT ON COLUMN earnings_intelligence.ecs IS 
    'Expectations Clearance Score: STRONG_BEAT, BEAT, INLINE, MISS, STRONG_MISS';
COMMENT ON COLUMN earnings_intelligence.regime IS 
    'Expectations regime: HYPED, FEARED, VOLATILE, NORMAL';
COMMENT ON COLUMN earnings_intelligence.event_z IS 
    'Blended event surprise z-score: 35% eps_z + 30% rev_z + 35% guidance_z';
COMMENT ON COLUMN earnings_intelligence.total_reaction IS 
    'Canonical outcome: total return from pre-earnings close to post-earnings close';
COMMENT ON COLUMN earnings_intelligence.position_scale IS 
    'Recommended position size multiplier (0.2 to 1.0)';
COMMENT ON COLUMN earnings_intelligence.suppression_reason IS 
    'LOGIC = good data but poor setup, DATA = insufficient data quality';
COMMENT ON TABLE earnings_history IS 
    'Historical earnings data for computing ticker-specific z-scores';
COMMENT ON TABLE ies_cache IS 
    'IES calculation cache for cross-module consistency';

-- END OF MIGRATION