from src.ml.db_helper import get_engine
import pandas as pd

e = get_engine()
q = "SELECT id, score_date, ticker, op_price FROM historical_scores WHERE ticker='DGX' AND return_1d IS NULL LIMIT 3"
df = pd.read_sql(q, e)
for _, r in df.iterrows():
    sd = r['score_date']
    print(f"  score_date={sd}  type={type(sd)}  hasattr_date={hasattr(sd, 'date')}")
    if hasattr(sd, 'date'):
        print(f"    .date() = {sd.date()}  type={type(sd.date())}")
    else:
        print(f"    raw value used directly")
