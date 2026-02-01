import psycopg2
import pandas as pd
import sys
sys.path.insert(0, '../dashboard')

from portfolio_engine import PortfolioEngine, PortfolioIntent

# Connect and load data
conn = psycopg2.connect(host='localhost', port=5432, dbname='alpha_platform', user='alpha', password='alpha_secure_2024')

# Load universe
df = pd.read_sql('''
    SELECT DISTINCT ON (ticker) * 
    FROM fundamentals 
    ORDER BY ticker, date DESC
''', conn)
conn.close()

print(f"Universe: {len(df)} stocks")

# Your 29 tickers
tickers = ['NVDA', 'AMD', 'AVGO', 'MRVL', 'ARM', 'MU', 'HXSCL', 'TSM', 'ASML', 'SNPS', 
           'SMCI', 'ANET', 'DELL', 'VRT', 'ETN', 'SBGSY', 'ABB', 'CEG', 'MSFT', 'META', 
           'GOOGL', 'AMZN', 'PLTR', 'CRWD', 'SNOW', 'DDOG', 'PANW', 'PATH', 'NET']

# Check which are in DB
found = df[df['ticker'].isin(tickers)]['ticker'].unique().tolist()
missing = [t for t in tickers if t not in found]

print(f"\nFound in DB: {len(found)}")
print(f"Missing from DB: {missing}")

# Build portfolio with found tickers
intent = PortfolioIntent(
    objective='tech_growth',
    risk_level='aggressive',
    portfolio_value=100000,
    tickers_include=found,
    restrict_to_tickers=False,
    fully_invested=True,
    equal_weight=True
)

engine = PortfolioEngine(df)
result = engine.build_portfolio(intent, user_request='AI growth portfolio with all specified tickers')

print(f"\n--- PORTFOLIO RESULT ---")
print(f"Success: {result.success}")
print(f"Holdings: {result.num_holdings}")
print(f"Warnings: {result.warnings}")

print(f"\n--- HOLDINGS ---")
for h in sorted(result.holdings, key=lambda x: -x.weight_pct):
    print(f"{h.ticker:6} {h.weight_pct:5.1f}%  {h.sector}")
