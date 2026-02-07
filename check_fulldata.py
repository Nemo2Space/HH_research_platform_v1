from src.ml.db_helper import get_engine
import pandas as pd
e = get_engine()

q = """
SELECT COUNT(*) as usable,
       COUNT(DISTINCT h.ticker) as tickers,
       MIN(h.score_date) as earliest,
       MAX(h.score_date) as latest
FROM historical_scores h
JOIN screener_scores s ON h.ticker = s.ticker AND h.score_date = s.date
WHERE h.return_5d IS NOT NULL
  AND h.op_price > 0
  AND s.technical_score IS NOT NULL
  AND s.options_flow_score IS NOT NULL
"""
print("=== Full-feature samples with returns ===")
print(pd.read_sql(q, e).to_string())

q2 = """
SELECT date_trunc('month', h.score_date) as month,
       COUNT(*) as usable,
       AVG(h.return_5d) as avg_5d,
       SUM(CASE WHEN h.return_5d > 0.15 THEN 1 ELSE 0 END)::float / COUNT(*) as win_rate
FROM historical_scores h
JOIN screener_scores s ON h.ticker = s.ticker AND h.score_date = s.date
WHERE h.return_5d IS NOT NULL AND h.op_price > 0
  AND s.technical_score IS NOT NULL
  AND s.options_flow_score IS NOT NULL
GROUP BY 1 ORDER BY 1
"""
print("\n=== By month ===")
df = pd.read_sql(q2, e)
for _, r in df.iterrows():
    m = str(r['month'])[:7]
    print(f"  {m}  usable={int(r['usable']):5d}  avg_5d={r['avg_5d']:+.3f}%  win_rate={r['win_rate']:.1%}")
