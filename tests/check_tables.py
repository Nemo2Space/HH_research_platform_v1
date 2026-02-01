import psycopg2
import pandas as pd

conn = psycopg2.connect(host='localhost', port=5432, dbname='alpha_platform', user='alpha', password='alpha_secure_2024')

# Check what scoring columns exist in fundamentals
cur = conn.cursor()
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'fundamentals'")
print("FUNDAMENTALS columns:")
print([row[0] for row in cur.fetchall()])

# Check if there's a screener_scores or similar table
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'screener_scores'")
print("\nSCREENER_SCORES columns:")
print([row[0] for row in cur.fetchall()])

# Check trading_signals
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'trading_signals'")
print("\nTRADING_SIGNALS columns:")
print([row[0] for row in cur.fetchall()])

# Check ai_recommendations
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'ai_recommendations'")
print("\nAI_RECOMMENDATIONS columns:")
print([row[0] for row in cur.fetchall()])

conn.close()
