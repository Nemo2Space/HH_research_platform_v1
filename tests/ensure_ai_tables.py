"""
Ensure AI Tables Schema
=======================
Creates or updates all AI-related tables with proper structure.
Run this BEFORE populate_biotech_ai_data.py

Run: python ensure_ai_tables.py
"""

import sys

sys.path.insert(0, '..')


def get_db_connection():
    try:
        from src.db.connection import get_connection
        cm = get_connection()
        return cm.__enter__()
    except:
        import psycopg2
        return psycopg2.connect(
            host="localhost", port=5432, dbname="alpha_platform",
            user="alpha", password="alpha_secure_2024"
        )


print("=" * 80)
print("ENSURING AI TABLES SCHEMA")
print("=" * 80)

conn = get_db_connection()

# =============================================================================
# TABLE DEFINITIONS
# =============================================================================

tables = [
    # ai_analysis - AI recommendations with reasoning
    """
    CREATE TABLE IF NOT EXISTS ai_analysis (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(20) NOT NULL,
        analysis_date DATE NOT NULL DEFAULT CURRENT_DATE,
        ai_action VARCHAR(20),        -- BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
        ai_confidence VARCHAR(20),    -- HIGH, MEDIUM, LOW
        bull_case TEXT,
        bear_case TEXT,
        key_risks TEXT,
        one_line_summary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, analysis_date)
    )
    """,

    # ai_recommendations - probability scores
    """
    CREATE TABLE IF NOT EXISTS ai_recommendations (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(20) NOT NULL,
        ai_probability FLOAT,         -- 0-1 probability
        ai_ev FLOAT,                  -- Expected value
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # committee_decisions - committee consensus
    """
    CREATE TABLE IF NOT EXISTS committee_decisions (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(20) NOT NULL,
        date DATE NOT NULL DEFAULT CURRENT_DATE,
        verdict VARCHAR(20),          -- STRONG BUY, BUY, HOLD, SELL, STRONG SELL
        conviction FLOAT,             -- 0-1 conviction level
        expected_alpha_bps INTEGER,   -- Expected alpha in basis points
        horizon_days INTEGER,         -- Investment horizon
        rationale TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date)
    )
    """,

    # agent_votes - individual agent recommendations
    """
    CREATE TABLE IF NOT EXISTS agent_votes (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(20) NOT NULL,
        date DATE NOT NULL DEFAULT CURRENT_DATE,
        agent_role VARCHAR(50),       -- fundamental, sentiment, technical, valuation, risk, macro
        buy_prob FLOAT,               -- 0-1 buy probability
        confidence FLOAT,             -- 0-1 confidence level
        rationale TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date, agent_role)
    )
    """,

    # alpha_predictions - ML model predictions
    """
    CREATE TABLE IF NOT EXISTS alpha_predictions (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(20) NOT NULL,
        prediction_date DATE NOT NULL DEFAULT CURRENT_DATE,
        predicted_probability FLOAT,  -- 0-100 probability
        alpha_signal VARCHAR(20),     -- Model signal
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, prediction_date)
    )
    """,

    # enhanced_scores - additional scoring factors
    """
    CREATE TABLE IF NOT EXISTS enhanced_scores (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(20) NOT NULL,
        date DATE NOT NULL DEFAULT CURRENT_DATE,
        insider_score INTEGER,        -- 0-100
        revision_score INTEGER,       -- 0-100
        earnings_surprise_score INTEGER,  -- 0-100
        volume_score INTEGER,
        pe_relative_score INTEGER,
        peg_score INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date)
    )
    """,

    # trading_signals - generated signals
    """
    CREATE TABLE IF NOT EXISTS trading_signals (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(20) NOT NULL,
        date DATE NOT NULL DEFAULT CURRENT_DATE,
        signal_type VARCHAR(20),      -- STRONG BUY, BUY, HOLD, SELL, STRONG SELL
        signal_strength INTEGER,      -- 0-100
        signal_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date)
    )
    """,

    # fda_calendar - FDA catalysts
    """
    CREATE TABLE IF NOT EXISTS fda_calendar (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(20) NOT NULL,
        company_name VARCHAR(200),
        drug_name VARCHAR(200),
        indication TEXT,
        catalyst_type VARCHAR(50),    -- PDUFA, sNDA, BLA, Phase3 Data, etc.
        expected_date DATE,
        date_confirmed BOOLEAN DEFAULT FALSE,
        priority VARCHAR(20),         -- HIGH, MEDIUM, LOW
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # earnings_calendar - earnings dates
    """
    CREATE TABLE IF NOT EXISTS earnings_calendar (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(20) NOT NULL,
        earnings_date DATE,
        earnings_time VARCHAR(20),    -- BMO, AMC, DURING
        eps_estimate FLOAT,
        eps_actual FLOAT,
        revenue_estimate BIGINT,
        revenue_actual BIGINT,
        guidance_direction VARCHAR(20),  -- RAISED, MAINTAINED, LOWERED, WITHDRAWN
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, earnings_date)
    )
    """,
]

# =============================================================================
# CREATE TABLES
# =============================================================================

print("\nCreating/updating tables...")

with conn.cursor() as cur:
    for sql in tables:
        # Extract table name for logging
        import re

        match = re.search(r'CREATE TABLE IF NOT EXISTS (\w+)', sql)
        table_name = match.group(1) if match else 'unknown'

        try:
            cur.execute(sql)
            conn.commit()
            print(f"  ✓ {table_name}")
        except Exception as e:
            print(f"  ✗ {table_name}: {e}")
            conn.rollback()

# =============================================================================
# CREATE INDEXES
# =============================================================================

print("\nCreating indexes...")

indexes = [
    "CREATE INDEX IF NOT EXISTS idx_ai_analysis_ticker ON ai_analysis(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_ai_analysis_date ON ai_analysis(analysis_date)",
    "CREATE INDEX IF NOT EXISTS idx_ai_rec_ticker ON ai_recommendations(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_committee_ticker ON committee_decisions(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_committee_date ON committee_decisions(date)",
    "CREATE INDEX IF NOT EXISTS idx_agent_ticker ON agent_votes(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_agent_role ON agent_votes(agent_role)",
    "CREATE INDEX IF NOT EXISTS idx_alpha_ticker ON alpha_predictions(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_enhanced_ticker ON enhanced_scores(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_signals_ticker ON trading_signals(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_signals_type ON trading_signals(signal_type)",
    "CREATE INDEX IF NOT EXISTS idx_fda_ticker ON fda_calendar(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_fda_date ON fda_calendar(expected_date)",
    "CREATE INDEX IF NOT EXISTS idx_earnings_ticker ON earnings_calendar(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_earnings_date ON earnings_calendar(earnings_date)",
]

with conn.cursor() as cur:
    for sql in indexes:
        try:
            cur.execute(sql)
            conn.commit()
            # Extract index name
            match = re.search(r'CREATE INDEX IF NOT EXISTS (\w+)', sql)
            idx_name = match.group(1) if match else 'unknown'
            print(f"  ✓ {idx_name}")
        except Exception as e:
            print(f"  ✗ {sql[:50]}...: {e}")
            conn.rollback()

# =============================================================================
# VERIFY TABLES
# =============================================================================

print("\nVerifying tables...")

required_tables = [
    'ai_analysis', 'ai_recommendations', 'committee_decisions', 'agent_votes',
    'alpha_predictions', 'enhanced_scores', 'trading_signals', 'fda_calendar',
    'earnings_calendar'
]

with conn.cursor() as cur:
    for table in required_tables:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            )
        """, (table,))
        exists = cur.fetchone()[0]

        if exists:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"  ✓ {table}: {count:,} rows")
        else:
            print(f"  ✗ {table}: MISSING!")

conn.close()

print("\n" + "=" * 80)
print("SCHEMA SETUP COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("  1. python diagnose_biotech_data.py        # Analyze data gaps")
print("  2. python populate_biotech_ai_data.py --all  # Populate AI data")
print("  3. python test_portfolio_v4.py            # Test portfolio engine")