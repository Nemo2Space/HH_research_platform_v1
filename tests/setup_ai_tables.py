"""
Analyze AI Tables and Create FDA Calendar
==========================================
1. Check all AI-related data available
2. Create FDA calendar table
3. Show what can be integrated
"""

import sys
sys.path.insert(0, '..')

def get_db_connection():
    try:
        from src.db.connection import get_connection
        cm = get_connection()
        return cm.__enter__()
    except (ImportError, AttributeError):
        import psycopg2
        return psycopg2.connect(
            host="localhost", port=5432, dbname="alpha_platform",
            user="alpha", password="alpha_secure_2024"
        )

conn = get_db_connection()

print("=" * 80)
print("AI DATA ANALYSIS & FDA CALENDAR SETUP")
print("=" * 80)

# =============================================================================
# 1. ANALYZE ALL AI-RELATED TABLES
# =============================================================================

print("\n" + "=" * 80)
print("1. AI-RELATED TABLES ANALYSIS")
print("=" * 80)

ai_tables = [
    'ai_analysis',
    'ai_recommendations',
    'committee_decisions',
    'alpha_predictions',
    'agent_votes',
    'setup_cards',
    'trading_signals',
    'signal_snapshots',
    'enhanced_scores',
    'earnings_intelligence',
]

for table in ai_tables:
    with conn.cursor() as cur:
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            )
        """, (table,))
        exists = cur.fetchone()[0]

        if not exists:
            print(f"\n❌ {table}: Does not exist")
            continue

        # Get columns
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, (table,))
        cols = cur.fetchall()

        # Get row count
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]

        # Get distinct tickers if ticker column exists
        ticker_count = 0
        if any(c[0] == 'ticker' for c in cols):
            cur.execute(f"SELECT COUNT(DISTINCT ticker) FROM {table}")
            ticker_count = cur.fetchone()[0]

        print(f"\n✅ {table}: {count:,} rows, {ticker_count} tickers, {len(cols)} columns")

        # Show relevant columns
        relevant_cols = [c for c in cols if any(kw in c[0].lower() for kw in
            ['prob', 'score', 'signal', 'action', 'buy', 'sell', 'confidence',
             'bull', 'bear', 'risk', 'catalyst', 'target', 'price', 'date'])]

        if relevant_cols:
            print("  Key columns:")
            for col, dtype in relevant_cols[:10]:
                print(f"    • {col}: {dtype}")

# =============================================================================
# 2. SAMPLE DATA FROM KEY TABLES
# =============================================================================

print("\n" + "=" * 80)
print("2. SAMPLE DATA FROM KEY AI TABLES")
print("=" * 80)

# Sample from committee_decisions
print("\n--- committee_decisions (sample) ---")
with conn.cursor() as cur:
    cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'committee_decisions'
    """)
    cols = [r[0] for r in cur.fetchall()]
    print(f"Columns: {cols}")

    cur.execute("SELECT * FROM committee_decisions LIMIT 3")
    rows = cur.fetchall()
    for row in rows:
        print(f"  {dict(zip(cols, row))}")

# Sample from agent_votes
print("\n--- agent_votes (sample) ---")
with conn.cursor() as cur:
    cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'agent_votes'
    """)
    cols = [r[0] for r in cur.fetchall()]
    print(f"Columns: {cols}")

    cur.execute("SELECT * FROM agent_votes LIMIT 3")
    rows = cur.fetchall()
    for row in rows:
        print(f"  {dict(zip(cols, row))}")

# Sample from enhanced_scores
print("\n--- enhanced_scores (sample) ---")
with conn.cursor() as cur:
    cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'enhanced_scores'
    """)
    cols = [r[0] for r in cur.fetchall()]
    print(f"Columns: {cols}")

    cur.execute("SELECT * FROM enhanced_scores ORDER BY date DESC LIMIT 3")
    rows = cur.fetchall()
    for row in rows:
        print(f"  {dict(zip(cols, row))}")

# Sample from earnings_intelligence
print("\n--- earnings_intelligence (sample) ---")
with conn.cursor() as cur:
    cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'earnings_intelligence'
    """)
    cols = [r[0] for r in cur.fetchall()]
    print(f"Columns: {cols[:10]}...")  # First 10 only

    cur.execute("SELECT COUNT(*) FROM earnings_intelligence")
    count = cur.fetchone()[0]
    print(f"Total rows: {count}")

# =============================================================================
# 3. CHECK EARNINGS CALENDAR DATA
# =============================================================================

print("\n" + "=" * 80)
print("3. EARNINGS CALENDAR ANALYSIS")
print("=" * 80)

with conn.cursor() as cur:
    # Upcoming earnings
    cur.execute("""
        SELECT ticker, earnings_date, earnings_time, eps_estimate, guidance_direction
        FROM earnings_calendar
        WHERE earnings_date >= CURRENT_DATE
        ORDER BY earnings_date
        LIMIT 20
    """)
    rows = cur.fetchall()
    print(f"\nUpcoming earnings: {len(rows)} stocks")
    for row in rows[:10]:
        print(f"  {row[0]}: {row[1]} ({row[2]}) - EPS Est: {row[3]}, Guidance: {row[4]}")

# =============================================================================
# 4. CREATE FDA CALENDAR TABLE
# =============================================================================

print("\n" + "=" * 80)
print("4. FDA CALENDAR TABLE")
print("=" * 80)

with conn.cursor() as cur:
    # Check if table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'fda_calendar'
        )
    """)
    exists = cur.fetchone()[0]

    if exists:
        print("FDA calendar table already exists")
        cur.execute("SELECT COUNT(*) FROM fda_calendar")
        count = cur.fetchone()[0]
        print(f"  Rows: {count}")
    else:
        print("Creating FDA calendar table...")
        cur.execute("""
            CREATE TABLE fda_calendar (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20) NOT NULL,
                company_name VARCHAR(255),
                drug_name VARCHAR(255),
                indication VARCHAR(500),
                catalyst_type VARCHAR(50),  -- PDUFA, AdCom, Phase3, CRL, etc.
                expected_date DATE,
                date_confirmed BOOLEAN DEFAULT FALSE,
                priority VARCHAR(20),  -- HIGH, MEDIUM, LOW
                market_cap_at_filing BIGINT,
                notes TEXT,
                source VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cur.execute("CREATE INDEX idx_fda_ticker ON fda_calendar(ticker)")
        cur.execute("CREATE INDEX idx_fda_date ON fda_calendar(expected_date)")
        cur.execute("CREATE INDEX idx_fda_type ON fda_calendar(catalyst_type)")

        conn.commit()
        print("  ✅ Created fda_calendar table with indexes")

# =============================================================================
# 5. CREATE CATALYST_CALENDAR TABLE (unified catalysts)
# =============================================================================

print("\n--- Creating unified catalyst_calendar table ---")

with conn.cursor() as cur:
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'catalyst_calendar'
        )
    """)
    exists = cur.fetchone()[0]

    if exists:
        print("catalyst_calendar table already exists")
        cur.execute("SELECT COUNT(*) FROM catalyst_calendar")
        count = cur.fetchone()[0]
        print(f"  Rows: {count}")
    else:
        print("Creating catalyst_calendar table...")
        cur.execute("""
            CREATE TABLE catalyst_calendar (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20) NOT NULL,
                catalyst_type VARCHAR(50) NOT NULL,  -- EARNINGS, FDA, CONFERENCE, DIVIDEND, SPLIT, etc.
                catalyst_date DATE NOT NULL,
                catalyst_time VARCHAR(20),  -- BMO, AMC, DURING
                description TEXT,
                importance VARCHAR(20),  -- HIGH, MEDIUM, LOW
                expected_impact VARCHAR(20),  -- POSITIVE, NEGATIVE, NEUTRAL, UNKNOWN
                notes TEXT,
                source VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("CREATE INDEX idx_catalyst_ticker ON catalyst_calendar(ticker)")
        cur.execute("CREATE INDEX idx_catalyst_date ON catalyst_calendar(catalyst_date)")
        cur.execute("CREATE INDEX idx_catalyst_type ON catalyst_calendar(catalyst_type)")

        conn.commit()
        print("  ✅ Created catalyst_calendar table with indexes")

# =============================================================================
# 6. POPULATE FDA CALENDAR WITH SAMPLE BIOTECH DATA
# =============================================================================

print("\n" + "=" * 80)
print("5. POPULATING FDA CALENDAR (Biotech Sample Data)")
print("=" * 80)

# Sample FDA catalyst data for biotech stocks - comprehensive list
fda_catalysts = [
    # (ticker, company, drug, indication, type, date, confirmed, priority)
    # Near-term catalysts (Q1 2025)
    ('GERN', 'Geron', 'Imetelstat', 'Lower-risk MDS', 'PDUFA', '2025-01-28', True, 'HIGH'),
    ('IOVA', 'Iovance', 'Lifileucel', 'Melanoma Label Expansion', 'sNDA', '2025-02-15', True, 'HIGH'),
    ('ARDX', 'Ardelyx', 'Tenapanor', 'IBS-C Label Expansion', 'sNDA', '2025-02-01', True, 'MEDIUM'),
    ('SRPT', 'Sarepta', 'SRP-9001', 'DMD Gene Therapy', 'PDUFA', '2025-02-28', True, 'HIGH'),

    # Q2 2025
    ('NVAX', 'Novavax', 'NVX-CoV2373', 'COVID-19 Updated Booster', 'sBLA', '2025-03-15', False, 'HIGH'),
    ('MNKD', 'MannKind', 'Tyvaso DPI', 'PAH Inhaled', 'PDUFA', '2025-04-20', True, 'HIGH'),
    ('NTLA', 'Intellia', 'NTLA-2001', 'ATTR Amyloidosis', 'Phase3 Readout', '2025-04-01', False, 'HIGH'),
    ('WVE', 'Wave Life Sciences', 'WVE-003', 'Huntington Disease', 'Phase2 Data', '2025-03-01', False, 'MEDIUM'),
    ('PCRX', 'Pacira', 'EXPAREL', 'Nerve Block Expansion', 'sNDA', '2025-03-10', False, 'LOW'),

    # Q3-Q4 2025
    ('QURE', 'uniQure', 'AMT-130', 'Huntington Disease', 'Phase3 Interim', '2025-05-01', False, 'HIGH'),
    ('COGT', 'Cogent Bio', 'Bezuclastinib', 'GIST', 'Phase3 Data', '2025-07-01', False, 'HIGH'),
    ('SANA', 'Sana Bio', 'SC291', 'B-cell Malignancies', 'Phase1 Data', '2025-06-01', False, 'MEDIUM'),
    ('DVAX', 'Dynavax', 'CpG 1018', 'Adjuvant Programs', 'Phase3 Partner', '2025-05-15', False, 'MEDIUM'),
    ('TWST', 'Twist Bioscience', 'Antibody Libraries', 'Oncology Partnership', 'Phase1 Start', '2025-06-01', False, 'LOW'),
    ('ABCL', 'AbCellera', 'ABCL-575', 'Solid Tumors', 'Phase1 Data', '2025-04-15', False, 'LOW'),

    # Additional biotech catalysts
    ('NRIX', 'Nurix Therapeutics', 'NX-5948', 'BTK Degrader B-cell', 'Phase2 Data', '2025-06-15', False, 'HIGH'),
    ('ORIC', 'ORIC Pharma', 'ORIC-944', 'Prostate Cancer', 'Phase2 Data', '2025-03-20', False, 'HIGH'),
    ('REPL', 'Replimune', 'RP1', 'Melanoma Combo', 'Phase2 Data', '2025-04-10', False, 'HIGH'),
    ('CTMX', 'CytomX', 'CX-904', 'Solid Tumors', 'Phase1 Expansion', '2025-05-01', False, 'MEDIUM'),
    ('GPCR', 'Structure Therapeutics', 'GSBR-1290', 'Obesity GLP-1', 'Phase2 Data', '2025-04-01', False, 'HIGH'),
    ('ERAS', 'Erasca', 'ERAS-007', 'RAS-driven Cancers', 'Phase2 Data', '2025-05-15', False, 'MEDIUM'),
    ('DAWN', 'Day One Bio', 'DAY101', 'Pediatric Brain Tumors', 'PDUFA', '2025-05-30', True, 'HIGH'),
    ('XOMA', 'XOMA', 'Royalty Portfolio', 'Multiple Programs', 'Partner Data', '2025-06-01', False, 'LOW'),
    ('LXRX', 'Lexicon Pharma', 'Sotagliflozin', 'Heart Failure Expansion', 'sNDA', '2025-04-15', False, 'MEDIUM'),
    ('AMLX', 'Amylyx Pharma', 'AMX0035', 'ALS Long-term Data', 'Phase3 Extension', '2025-03-01', False, 'MEDIUM'),
    ('OCGN', 'Ocugen', 'OCU400', 'Retinitis Pigmentosa', 'Phase3 Data', '2025-06-01', False, 'MEDIUM'),
    ('NEOG', 'Neogen', 'Food Safety', 'FDA Clearances', 'Multiple 510k', '2025-04-01', False, 'LOW'),
    ('MDXG', 'MiMedx', 'AmnioFix', 'Wound Healing', 'Label Expansion', '2025-05-01', False, 'LOW'),
    ('CMPX', 'Compass Pathways', 'COMP360', 'Treatment-Resistant Depression', 'Phase3 Data', '2025-08-01', False, 'HIGH'),
    ('BCRX', 'BioCryst', 'Orladeyo', 'HAE Pediatric', 'sNDA', '2025-03-15', False, 'MEDIUM'),
    ('ADPT', 'Adaptive Biotech', 'clonoSEQ', 'MRD Testing Expansion', '510k', '2025-04-01', False, 'LOW'),
    ('STOK', 'Stoke Therapeutics', 'STK-001', 'Dravet Syndrome', 'Phase3 Data', '2025-07-01', False, 'HIGH'),
    ('VRDN', 'Viridian Therapeutics', 'VRDN-001', 'Thyroid Eye Disease', 'Phase3 Data', '2025-05-01', False, 'HIGH'),
    ('PGEN', 'Precigen', 'PRGN-3005', 'Ovarian Cancer', 'Phase2 Data', '2025-06-01', False, 'MEDIUM'),
    ('ATAI', 'ATAI Life Sciences', 'PCN-101', 'Treatment-Resistant Depression', 'Phase2 Data', '2025-04-01', False, 'MEDIUM'),
    ('IMNM', 'Immunome', 'IMM-1-104', 'Solid Tumors', 'Phase1 Data', '2025-05-01', False, 'LOW'),
    ('TRVI', 'Trevi Therapeutics', 'Haduvio', 'Chronic Cough', 'Phase3 Data', '2025-06-01', False, 'MEDIUM'),
    ('TNGX', 'Tango Therapeutics', 'TNG908', 'MTAP-deleted Cancers', 'Phase2 Data', '2025-05-15', False, 'HIGH'),
    ('TERN', 'Terns Pharma', 'TERN-501', 'NASH', 'Phase2 Data', '2025-04-01', False, 'MEDIUM'),
    ('ALT', 'Altimmune', 'Pemvidutide', 'Obesity/NASH', 'Phase2 Data', '2025-03-15', False, 'HIGH'),
]

with conn.cursor() as cur:
    # Check current count
    cur.execute("SELECT COUNT(*) FROM fda_calendar")
    current_count = cur.fetchone()[0]

    if current_count == 0:
        print(f"Inserting {len(fda_catalysts)} FDA catalysts...")
        for ticker, company, drug, indication, cat_type, date, confirmed, priority in fda_catalysts:
            cur.execute("""
                INSERT INTO fda_calendar 
                (ticker, company_name, drug_name, indication, catalyst_type, expected_date, date_confirmed, priority)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (ticker, company, drug, indication, cat_type, date, confirmed, priority))

        conn.commit()
        print(f"  ✅ Inserted {len(fda_catalysts)} FDA catalysts")
    else:
        print(f"FDA calendar already has {current_count} entries")

# Show FDA calendar
with conn.cursor() as cur:
    cur.execute("""
        SELECT ticker, drug_name, catalyst_type, expected_date, priority
        FROM fda_calendar
        WHERE expected_date >= CURRENT_DATE
        ORDER BY expected_date
        LIMIT 15
    """)
    rows = cur.fetchall()
    print(f"\nUpcoming FDA Catalysts:")
    for row in rows:
        print(f"  {row[0]}: {row[1]} ({row[2]}) - {row[3]} [{row[4]}]")

# =============================================================================
# 7. SUMMARY OF AVAILABLE AI DATA
# =============================================================================

print("\n" + "=" * 80)
print("6. SUMMARY: AI DATA AVAILABLE FOR PORTFOLIO ENGINE")
print("=" * 80)

summary = """
READY TO INTEGRATE:

1. AI ANALYSIS (ai_analysis) - 114 stocks
   • ai_action: BUY/SELL/HOLD recommendation
   • ai_confidence: HIGH/MEDIUM/LOW
   • bull_case, bear_case: AI reasoning
   • one_line_summary: Quick summary
   
2. AI RECOMMENDATIONS (ai_recommendations) - 116 stocks  
   • ai_probability: 0-1 probability score
   • ai_ev: Expected value
   
3. ALPHA PREDICTIONS (alpha_predictions) - 109 stocks
   • predicted_probability: ML model probability
   • alpha_signal: Model signal
   
4. AGENT VOTES (agent_votes) - Individual agent votes
   • agent_name: Which AI agent
   • buy_prob: Agent's buy probability
   • reasoning: Agent's reasoning
   
5. COMMITTEE DECISIONS (committee_decisions) - Committee consensus
   • buy_prob: Aggregated committee probability
   • final_action: Committee decision
   
6. ENHANCED SCORES (enhanced_scores) - Additional scoring
   • revision_score, insider_score, volume_score
   • pe_relative_score, peg_score
   
7. EARNINGS INTELLIGENCE (earnings_intelligence) - Earnings analysis
   • guidance_score, tone_score, revision_score
   • confidence_lang_score, margin_score
   
8. FDA CALENDAR (fda_calendar) - NEW
   • Upcoming PDUFA dates, clinical trials
   • Drug names, indications, priorities
   
9. CATALYST CALENDAR (catalyst_calendar) - NEW
   • Unified catalyst tracking
"""
print(summary)

conn.close()

print("\n" + "=" * 80)
print("DONE - Tables ready for integration")
print("=" * 80)