"""Check column types for filing_facts."""
from src.db.connection import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type, udt_name
            FROM information_schema.columns 
            WHERE table_schema = 'rag' AND table_name = 'filing_facts'
            ORDER BY ordinal_position
        """)
        print('filing_facts column types:')
        for row in cur.fetchall():
            print(f'  {row[0]}: {row[1]} ({row[2]})')