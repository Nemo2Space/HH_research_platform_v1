from src.ml.db_helper import get_engine
import pandas as pd

# Check what columns exist
e = get_engine()
q = "SELECT column_name FROM information_schema.columns WHERE table_name = 'historical_scores' ORDER BY ordinal_position"
cols = pd.read_sql(q, e)
print("=== historical_scores columns ===")
for _, r in cols.iterrows():
    print(f"  {r['column_name']}")

# Check if there's a backfill function/table
q2 = "SELECT score_date, ticker, op_price, return_1d, return_5d FROM historical_scores WHERE score_date = '2026-01-02' LIMIT 5"
df = pd.read_sql(q2, e)
print("\n=== Sample Jan 2 rows ===")
print(df.to_string())
