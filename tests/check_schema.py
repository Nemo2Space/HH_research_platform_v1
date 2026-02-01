"""Check schema columns."""
from src.db.connection import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        # Check filing_facts columns
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_schema = 'rag' AND table_name = 'filing_facts'
            ORDER BY ordinal_position
        """)
        print('filing_facts columns:')
        for row in cur.fetchall():
            print(f'  {row[0]}')

        # Check documents columns
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_schema = 'rag' AND table_name = 'documents'
            ORDER BY ordinal_position
        """)
        print('\ndocuments columns:')
        for row in cur.fetchall():
            print(f'  {row[0]}')

        # Check transcript_facts columns
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_schema = 'rag' AND table_name = 'transcript_facts'
            ORDER BY ordinal_position
        """)
        print('\ntranscript_facts columns:')
        for row in cur.fetchall():
            print(f'  {row[0]}')