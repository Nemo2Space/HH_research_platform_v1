"""
Check Database Schema for AI and Catalyst Data
==============================================
Lists all available columns that can be used in portfolio scoring.
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
print("DATABASE SCHEMA ANALYSIS")
print("=" * 80)

# Check all tables
with conn.cursor() as cur:
    cur.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = [r[0] for r in cur.fetchall()]
    print(f"\nTables in database: {len(tables)}")
    for t in tables:
        print(f"  • {t}")

# Check key tables for AI/catalyst columns
key_tables = ['screener_scores', 'fundamentals', 'ai_analysis', 'ai_signals',
              'signals', 'earnings_calendar', 'fda_calendar', 'catalyst_calendar',
              'options_flow', 'institutional_holdings']

print("\n" + "=" * 80)
print("KEY TABLE COLUMNS")
print("=" * 80)

for table in key_tables:
    if table in tables:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table}'
                ORDER BY ordinal_position
            """)
            cols = cur.fetchall()
            print(f"\n{table.upper()} ({len(cols)} columns):")
            for col, dtype in cols:
                print(f"  • {col}: {dtype}")
    else:
        print(f"\n{table.upper()}: ❌ Table does not exist")

# Check for AI-related columns anywhere
print("\n" + "=" * 80)
print("AI-RELATED COLUMNS (searching all tables)")
print("=" * 80)

ai_keywords = ['ai_', 'prob', 'prediction', 'committee', 'llm', 'model', 'signal', 'score']
with conn.cursor() as cur:
    for keyword in ai_keywords:
        cur.execute(f"""
            SELECT table_name, column_name 
            FROM information_schema.columns 
            WHERE column_name LIKE '%{keyword}%'
            ORDER BY table_name, column_name
        """)
        results = cur.fetchall()
        if results:
            print(f"\nColumns containing '{keyword}':")
            for table, col in results:
                print(f"  • {table}.{col}")

# Check sample data from screener_scores
print("\n" + "=" * 80)
print("SAMPLE DATA FROM SCREENER_SCORES")
print("=" * 80)

with conn.cursor() as cur:
    cur.execute("""
        SELECT * FROM screener_scores 
        WHERE ticker = 'NVAX' OR ticker = 'AAPL' OR ticker = 'MSFT'
        ORDER BY date DESC
        LIMIT 3
    """)
    cols = [desc[0] for desc in cur.description]
    rows = cur.fetchall()

    print(f"\nColumns: {len(cols)}")
    for i, col in enumerate(cols):
        if rows:
            val = rows[0][i]
            print(f"  {i + 1}. {col}: {val}")

conn.close()
print("\n" + "=" * 80)
print("DONE")
print("=" * 80)