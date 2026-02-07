from src.ml.db_helper import get_engine
import pandas as pd
e = get_engine()

# Check overlap: historical_scores WITH returns AND screener_scores
q = """
SELECT date_trunc('month', h.score_date) as month,
       COUNT(*) as total,
       COUNT(s.technical_score) as has_full_features,
       COUNT(h.return_5d) as has_5d_return,
       SUM(CASE WHEN s.technical_score IS NOT NULL AND h.return_5d IS NOT NULL THEN 1 ELSE 0 END) as usable
FROM historical_scores h
LEFT JOIN screener_scores s ON h.ticker = s.ticker AND h.score_date = s.date
WHERE h.score_date >= '2025-11-01'
GROUP BY 1 ORDER BY 1
"""
print("=== OVERLAP: full features + returns ===")
df = pd.read_sql(q, e)
for _, r in df.iterrows():
    m = str(r['month'])[:7]
    print(f"  {m}  total={int(r['total']):5d}  full_feat={int(r['has_full_features']):5d}  has_return={int(r['has_5d_return']):5d}  usable={int(r['usable']):5d}")

# Check: do the 3 available features have ANY signal?
q2 = """
SELECT
    CASE WHEN h.sentiment >= 65 THEN 'bullish' WHEN h.sentiment <= 35 THEN 'bearish' ELSE 'neutral' END as sent_bucket,
    COUNT(*) as n,
    AVG(h.return_5d) as avg_5d,
    SUM(CASE WHEN h.return_5d > 0.15 THEN 1 ELSE 0 END)::float / COUNT(*) as win_rate
FROM historical_scores h
WHERE h.return_5d IS NOT NULL AND h.sentiment IS NOT NULL
GROUP BY 1 ORDER BY avg_5d DESC
"""
print("\n=== SIGNAL CHECK: sentiment buckets ===")
df2 = pd.read_sql(q2, e)
for _, r in df2.iterrows():
    print(f"  {r['sent_bucket']:10s}  n={int(r['n']):5d}  avg_5d={r['avg_5d']:+.3f}%  win_rate={r['win_rate']:.1%}")

q3 = """
SELECT
    CASE WHEN gap_score ~ '^[0-9.]+$' AND gap_score::numeric >= 65 THEN 'high'
         WHEN gap_score ~ '^[0-9.]+$' AND gap_score::numeric <= 35 THEN 'low'
         ELSE 'mid' END as gap_bucket,
    COUNT(*) as n,
    AVG(h.return_5d) as avg_5d,
    SUM(CASE WHEN h.return_5d > 0.15 THEN 1 ELSE 0 END)::float / COUNT(*) as win_rate
FROM historical_scores h
WHERE h.return_5d IS NOT NULL
GROUP BY 1 ORDER BY avg_5d DESC
"""
print("\n=== SIGNAL CHECK: gap_score buckets ===")
df3 = pd.read_sql(q3, e)
for _, r in df3.iterrows():
    print(f"  {r['gap_bucket']:10s}  n={int(r['n']):5d}  avg_5d={r['avg_5d']:+.3f}%  win_rate={r['win_rate']:.1%}")
