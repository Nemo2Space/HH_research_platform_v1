import yfinance as yf
from datetime import date, timedelta
from src.ml.db_helper import get_engine
import pandas as pd

e = get_engine()
q = "SELECT id, score_date, ticker, op_price, return_1d FROM historical_scores WHERE ticker='DGX' AND return_1d IS NULL ORDER BY score_date"
rows = pd.read_sql(q, e)
print(f"DGX rows needing backfill: {len(rows)}")
for _, r in rows.iterrows():
    print(f"  {r['score_date']}")

min_dt = pd.Timestamp(rows['score_date'].min())
max_dt = pd.Timestamp(rows['score_date'].max())
print(f"\nFetching prices from {min_dt - timedelta(days=5)} to {max_dt + timedelta(days=46)}")

closes = yf.Ticker('DGX').history(
    start=(min_dt - timedelta(days=5)).strftime('%Y-%m-%d'),
    end=(max_dt + timedelta(days=46)).strftime('%Y-%m-%d')
)
closes.index = closes.index.tz_localize(None).date
print(f"Got {len(closes)} days of prices")
print(f"Date range: {min(closes.index)} to {max(closes.index)}")

# Test first and last row
for test_date in [rows['score_date'].iloc[0], rows['score_date'].iloc[-1]]:
    future = sorted([d for d in closes.index if d > test_date])
    print(f"\n  score_date={test_date}: {len(future)} future trading days")
    if future:
        print(f"    1d={future[0]}, 5d={future[4] if len(future)>4 else 'N/A'}")
