# Quick diagnostic - run in Python or add to a test file
from src.db.connection import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        # Check total count
        cur.execute("SELECT COUNT(*) FROM trading_signals")
        total = cur.fetchone()[0]
        print(f"Total signals: {total}")

        # Check date range
        cur.execute("SELECT MIN(date), MAX(date) FROM trading_signals")
        min_date, max_date = cur.fetchone()
        print(f"Date range: {min_date} to {max_date}")

        # Check recent signals
        cur.execute("""
            SELECT date, COUNT(*) 
            FROM trading_signals 
            GROUP BY date 
            ORDER BY date DESC 
            LIMIT 10
        """)
        print("\nRecent signals by date:")
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]} signals")