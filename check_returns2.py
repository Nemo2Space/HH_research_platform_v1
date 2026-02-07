from src.ml.db_helper import get_engine
import pandas as pd
e = get_engine()

# Reverse-engineer the return calculation from existing data
q = """
SELECT score_date, ticker, op_price, price_1d, price_5d, return_1d, return_5d
FROM historical_scores
WHERE return_5d IS NOT NULL AND price_5d IS NOT NULL AND op_price > 0
ORDER BY score_date DESC
LIMIT 10
"""
df = pd.read_sql(q, e)
for _, r in df.iterrows():
    calc_1d = (float(r['price_1d']) / float(r['op_price']) - 1) * 100 if r['price_1d'] else None
    calc_5d = (float(r['price_5d']) / float(r['op_price']) - 1) * 100 if r['price_5d'] else None
    print(f"  {str(r['score_date'])[:10]} {r['ticker']:5s}  op={float(r['op_price']):8.2f}  p5d={float(r['price_5d']):8.2f}  stored_ret5d={float(r['return_5d']):+7.3f}  calc_ret5d={calc_5d:+7.3f}" if calc_5d else f"  {str(r['score_date'])[:10]} {r['ticker']:5s}  no price_5d")

# Count what needs backfilling
q2 = """
SELECT
    COUNT(*) as total_null,
    COUNT(DISTINCT ticker) as tickers,
    MIN(score_date) as earliest,
    MAX(score_date) as latest
FROM historical_scores
WHERE return_1d IS NULL AND op_price IS NOT NULL AND op_price > 0
AND score_date <= '2026-01-25'
"""
print("\n=== Rows needing backfill ===")
print(pd.read_sql(q2, e).to_string())
