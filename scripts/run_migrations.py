"""
Run AI System Database Migrations

Use this script on Windows when psql is not in PATH.

Usage:
    python run_migrations.py
"""

import os
import sys

# Database connection
def get_connection():
    """Get database connection."""

    # Your actual connection details from .env
    DB_CONFIG = {
        'host': 'localhost',
        'port': '5432',
        'dbname': 'alpha_platform',
        'user': 'alpha',
        'password': 'alpha_secure_2024'
    }

    # Method 1: Try your existing connection module
    try:
        from src.db.connection import get_connection as get_conn
        conn = get_conn()
        print("✅ Connected using src.db.connection")
        return conn
    except Exception as e:
        print(f"   (src.db.connection not available: {e})")

    # Method 2: Use psycopg2 directly with your credentials
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            dbname=DB_CONFIG['dbname'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        print(f"✅ Connected to {DB_CONFIG['dbname']}@{DB_CONFIG['host']}")
        return conn

    except ImportError:
        print("❌ psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)


# SQL Migrations
MIGRATIONS = """
-- =============================================================================
-- AI TRADING SYSTEM - DATABASE MIGRATIONS
-- =============================================================================

-- 1. SETUP CARDS (RAG Memory)
CREATE TABLE IF NOT EXISTS setup_cards (
    card_id VARCHAR(64) PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    setup_date DATE NOT NULL,
    
    sentiment_score DECIMAL(5,2),
    fundamental_score DECIMAL(5,2),
    technical_score DECIMAL(5,2),
    options_flow_score DECIMAL(5,2),
    short_squeeze_score DECIMAL(5,2),
    total_score DECIMAL(5,2),
    
    sector VARCHAR(50),
    market_cap VARCHAR(10),
    vix_level DECIMAL(5,2),
    regime_score DECIMAL(5,2),
    
    signal_type VARCHAR(20),
    ml_probability DECIMAL(5,4),
    expected_value DECIMAL(8,6),
    
    days_to_earnings INTEGER,
    had_recent_earnings BOOLEAN DEFAULT FALSE,
    
    entry_price DECIMAL(12,4),
    planned_stop_pct DECIMAL(5,4),
    planned_target_pct DECIMAL(5,4),
    planned_horizon INTEGER DEFAULT 5,
    
    outcome VARCHAR(10),
    actual_return DECIMAL(8,4),
    actual_holding_days INTEGER,
    exit_reason VARCHAR(20),
    exit_date DATE,
    
    feature_vector FLOAT[],
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    notes TEXT,
    
    UNIQUE(ticker, setup_date)
);

-- 2. AI RECOMMENDATIONS
CREATE TABLE IF NOT EXISTS ai_recommendations (
    id SERIAL PRIMARY KEY,
    
    ticker VARCHAR(10) NOT NULL,
    recommendation_date DATE NOT NULL,
    recommendation_type VARCHAR(20),
    
    ml_probability DECIMAL(5,4),
    ml_ev DECIMAL(8,6),
    ml_confidence VARCHAR(20),
    
    meta_probability DECIMAL(5,4),
    combined_probability DECIMAL(5,4),
    
    approved BOOLEAN,
    rejection_reasons TEXT[],
    
    entry_price DECIMAL(12,4),
    stop_loss DECIMAL(12,4),
    target_price DECIMAL(12,4),
    position_size_pct DECIMAL(5,4),
    position_shares INTEGER,
    horizon_days INTEGER DEFAULT 5,
    
    vix_level DECIMAL(5,2),
    regime_score DECIMAL(5,2),
    similar_setups_count INTEGER,
    similar_setups_win_rate DECIMAL(5,4),
    
    outcome VARCHAR(10),
    actual_return DECIMAL(8,4),
    actual_holding_days INTEGER,
    exit_reason VARCHAR(20),
    exit_date DATE,
    
    llm_summary TEXT,
    llm_recommendation VARCHAR(20),
    llm_confidence VARCHAR(20),
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(ticker, recommendation_date)
);

-- 3. MODEL PERFORMANCE SNAPSHOTS
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    period_days INTEGER NOT NULL,
    
    total_recommendations INTEGER,
    trades_taken INTEGER,
    trades_skipped INTEGER,
    closed_trades INTEGER,
    
    wins INTEGER,
    losses INTEGER,
    scratches INTEGER,
    win_rate DECIMAL(5,4),
    
    avg_return DECIMAL(8,4),
    median_return DECIMAL(8,4),
    total_return DECIMAL(10,4),
    best_trade DECIMAL(8,4),
    worst_trade DECIMAL(8,4),
    
    avg_predicted_prob DECIMAL(5,4),
    actual_hit_rate DECIMAL(5,4),
    calibration_error DECIMAL(6,4),
    brier_score DECIMAL(6,4),
    
    metrics_by_regime JSONB,
    metrics_by_signal JSONB,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- 4. DRIFT ALERTS
CREATE TABLE IF NOT EXISTS drift_alerts (
    id SERIAL PRIMARY KEY,
    alert_time TIMESTAMP NOT NULL,
    
    drift_type VARCHAR(50),
    alert_level VARCHAR(20),
    metric_name VARCHAR(50),
    
    current_value DECIMAL(10,4),
    expected_value DECIMAL(10,4),
    threshold DECIMAL(10,4),
    deviation_pct DECIMAL(8,4),
    
    message TEXT,
    recommended_action TEXT,
    
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolution_notes TEXT,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- 5. ML MODEL METADATA
CREATE TABLE IF NOT EXISTS ml_model_metadata (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    
    trained_at TIMESTAMP NOT NULL,
    training_samples INTEGER,
    feature_count INTEGER,
    target_horizon INTEGER,
    
    mean_auc DECIMAL(5,4),
    mean_accuracy DECIMAL(5,4),
    mean_win_rate DECIMAL(5,4),
    mean_return DECIMAL(8,4),
    brier_score DECIMAL(6,4),
    
    is_well_calibrated BOOLEAN,
    calibration_error DECIMAL(6,4),
    
    beats_baseline BOOLEAN,
    baseline_auc DECIMAL(5,4),
    improvement_vs_baseline DECIMAL(6,4),
    
    feature_importance JSONB,
    model_path VARCHAR(255),
    
    is_active BOOLEAN DEFAULT TRUE,
    deactivated_at TIMESTAMP,
    deactivation_reason TEXT,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(model_name, version)
);

-- 6. TRADE JOURNAL AI
CREATE TABLE IF NOT EXISTS trade_journal_ai (
    id SERIAL PRIMARY KEY,
    
    ticker VARCHAR(10) NOT NULL,
    entry_date DATE NOT NULL,
    entry_price DECIMAL(12,4) NOT NULL,
    direction VARCHAR(10) DEFAULT 'LONG',
    shares INTEGER DEFAULT 1,
    
    exit_date DATE,
    exit_price DECIMAL(12,4),
    exit_reason VARCHAR(20),
    
    gross_return DECIMAL(8,4),
    net_return DECIMAL(8,4),
    dollar_pnl DECIMAL(12,2),
    
    ai_recommendation_id INTEGER,
    ml_probability_at_entry DECIMAL(5,4),
    ml_ev_at_entry DECIMAL(8,6),
    similar_setups_win_rate DECIMAL(5,4),
    
    entry_signals JSONB,
    
    followed_ai BOOLEAN,
    ai_was_correct BOOLEAN,
    
    entry_notes TEXT,
    exit_notes TEXT,
    lessons_learned TEXT,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(ticker, entry_date, direction)
);
"""

INDEX_SQL = """
-- Create indexes
CREATE INDEX IF NOT EXISTS idx_setup_cards_date ON setup_cards(setup_date DESC);
CREATE INDEX IF NOT EXISTS idx_setup_cards_ticker ON setup_cards(ticker);
CREATE INDEX IF NOT EXISTS idx_setup_cards_outcome ON setup_cards(outcome);
CREATE INDEX IF NOT EXISTS idx_setup_cards_sector ON setup_cards(sector);

CREATE INDEX IF NOT EXISTS idx_ai_recs_date ON ai_recommendations(recommendation_date DESC);
CREATE INDEX IF NOT EXISTS idx_ai_recs_ticker ON ai_recommendations(ticker);
CREATE INDEX IF NOT EXISTS idx_ai_recs_outcome ON ai_recommendations(outcome);
CREATE INDEX IF NOT EXISTS idx_ai_recs_approved ON ai_recommendations(approved);

CREATE INDEX IF NOT EXISTS idx_perf_date ON model_performance(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_drift_time ON drift_alerts(alert_time DESC);
CREATE INDEX IF NOT EXISTS idx_model_active ON ml_model_metadata(is_active);
"""

VIEW_SQL = """
-- Create helper views
CREATE OR REPLACE VIEW v_recent_recommendations AS
SELECT 
    ticker,
    recommendation_date,
    recommendation_type,
    ml_probability,
    ml_ev,
    approved,
    outcome,
    actual_return
FROM ai_recommendations
WHERE recommendation_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY recommendation_date DESC;

CREATE OR REPLACE VIEW v_signal_effectiveness AS
SELECT 
    recommendation_type,
    COUNT(*) as total_count,
    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
    AVG(CASE WHEN outcome IS NOT NULL THEN actual_return END) as avg_return,
    AVG(ml_probability) as avg_predicted_prob,
    AVG(CASE WHEN outcome = 'WIN' THEN 1.0 ELSE 0.0 END) as actual_win_rate
FROM ai_recommendations
WHERE outcome IS NOT NULL
GROUP BY recommendation_type;
"""


def run_migrations():
    """Run all migrations."""
    print("=" * 60)
    print("AI Trading System - Database Migrations")
    print("=" * 60)

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Run main tables
        print("\n📦 Creating tables...")
        cursor.execute(MIGRATIONS)
        conn.commit()
        print("   ✅ Tables created")

        # Run indexes
        print("\n📇 Creating indexes...")
        cursor.execute(INDEX_SQL)
        conn.commit()
        print("   ✅ Indexes created")

        # Run views
        print("\n👁️ Creating views...")
        cursor.execute(VIEW_SQL)
        conn.commit()
        print("   ✅ Views created")

        # Verify
        print("\n🔍 Verifying tables...")
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN (
                'setup_cards', 'ai_recommendations', 'model_performance',
                'drift_alerts', 'ml_model_metadata', 'trade_journal_ai'
            )
        """)
        tables = cursor.fetchall()

        for table in tables:
            print(f"   ✅ {table[0]}")

        print("\n" + "=" * 60)
        print("✅ All migrations completed successfully!")
        print("=" * 60)

    except Exception as e:
        conn.rollback()
        print(f"\n❌ Migration failed: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    run_migrations()