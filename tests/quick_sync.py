from src.db.connection import get_connection, get_engine
import pandas as pd

engine = get_engine()

query = """
    SELECT 
        date as score_date, ticker, sentiment_score as sentiment,
        fundamental_score, growth_score, dividend_score, total_score,
        gap_score, composite_score as mkt_score, signal_type
    FROM screener_scores
    WHERE date > '2025-12-10'
"""
df = pd.read_sql(query, engine)
print(f"Found {len(df)} records to sync")

sectors = pd.read_sql("SELECT DISTINCT ON (ticker) ticker, sector FROM fundamentals ORDER BY ticker, date DESC", engine)
df = df.merge(sectors, on='ticker', how='left')

with get_connection() as conn:
    cur = conn.cursor()
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO historical_scores (score_date, ticker, sector, sentiment, fundamental_score, 
                growth_score, dividend_score, total_score, gap_score, mkt_score, signal_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (score_date, ticker) DO UPDATE SET
                total_score = EXCLUDED.total_score, signal_type = EXCLUDED.signal_type
        """, (row['score_date'], row['ticker'], row.get('sector'), row.get('sentiment'),
              row.get('fundamental_score'), row.get('growth_score'), row.get('dividend_score'),
              row.get('total_score'), row.get('gap_score'), row.get('mkt_score'), row.get('signal_type')))
    conn.commit()
    print(f"Synced {len(df)} records!")

verify = pd.read_sql("SELECT MAX(score_date) as max_date, COUNT(*) as total FROM historical_scores", engine)
print(f"Now has {verify['total'].iloc[0]} rows through {verify['max_date'].iloc[0]}")
