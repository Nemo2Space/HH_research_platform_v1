from src.ml.db_helper import get_engine
import pandas as pd
e = get_engine()

# Why are Jan returns missing?
q = """
SELECT score_date, COUNT(*) as n,
       COUNT(return_1d) as has_1d,
       COUNT(return_5d) as has_5d
FROM historical_scores
WHERE score_date >= '2025-12-20'
GROUP BY 1 ORDER BY 1
"""
df = pd.read_sql(q, e)
for _, r in df.iterrows():
    d = str(r['score_date'])[:10]
    print(f"  {d}  n={int(r['n']):4d}  1d={int(r['has_1d']):4d}  5d={int(r['has_5d']):4d}")
