"""Check options_flow_daily columns."""
from src.db.connection import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'options_flow_daily'
            ORDER BY ordinal_position
        """)
        cols = [row[0] for row in cur.fetchall()]
        print(f"options_flow_daily columns: {cols}")

        # Also check options_summary
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'options_summary'
            ORDER BY ordinal_position
        """)
        cols = [row[0] for row in cur.fetchall()]
        print(f"options_summary columns: {cols}")