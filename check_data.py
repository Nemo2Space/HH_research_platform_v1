from src.ml.db_helper import get_engine
import pandas as pd
e = get_engine()
q = """
SELECT date_trunc('month', date) as month,
       COUNT(*) as rows,
       COUNT(technical_score) as has_tech,
       COUNT(options_flow_score) as has_opts,
       COUNT(short_squeeze_score) as has_squeeze,
       COUNT(target_upside_pct) as has_upside
FROM screener_scores
GROUP BY 1 ORDER BY 1
"""
df = pd.read_sql(q, e)
for _, r in df.iterrows():
    m = str(r['month'])[:7]
    print(f"{m}  rows={int(r['rows']):5d}  tech={int(r['has_tech']):5d}  opts={int(r['has_opts']):5d}  squeeze={int(r['has_squeeze']):5d}  upside={int(r['has_upside']):5d}")
