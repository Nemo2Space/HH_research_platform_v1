import psycopg2
import pandas as pd
import sys
sys.path.insert(0, '../dashboard')

from portfolio_engine import PortfolioEngine, PortfolioIntent

conn = psycopg2.connect(host='localhost', port=5432, dbname='alpha_platform', user='alpha', password='alpha_secure_2024')
df = pd.read_sql('SELECT DISTINCT ON (ticker) * FROM fundamentals ORDER BY ticker, date DESC', conn)
conn.close()

tickers = ['NVDA', 'AMD', 'AVGO', 'MRVL', 'ARM', 'MU', 'HXSCL', 'TSM', 'ASML', 'SNPS', 
           'SMCI', 'ANET', 'DELL', 'VRT', 'ETN', 'SBGSY', 'ABB', 'CEG', 'MSFT', 'META', 
           'GOOGL', 'AMZN', 'PLTR', 'CRWD', 'SNOW', 'DDOG', 'PANW', 'PATH', 'NET']

intent = PortfolioIntent(
    objective='tech_growth',
    risk_level='aggressive',
    portfolio_value=100000,
    tickers_include=tickers,
    restrict_to_tickers=False,
    fully_invested=True,
    equal_weight=False,  # Score-based weighting
    max_position_pct=10  # Cap at 10%
)

engine = PortfolioEngine(df)
result = engine.build_portfolio(intent, user_request='Aggressive AI growth portfolio')

print(f"Holdings: {result.num_holdings}")
print(f"\n--- TOP 10 BY WEIGHT ---")
for h in sorted(result.holdings, key=lambda x: -x.weight_pct)[:10]:
    print(f"{h.ticker:6} {h.weight_pct:5.1f}%  Score: {h.composite_score:.0f}  {h.conviction}")

print(f"\n--- BOTTOM 5 ---")
for h in sorted(result.holdings, key=lambda x: x.weight_pct)[:5]:
    print(f"{h.ticker:6} {h.weight_pct:5.1f}%  Score: {h.composite_score:.0f}  {h.conviction}")
