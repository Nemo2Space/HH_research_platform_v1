import psycopg2
import pandas as pd
import sys
sys.path.insert(0, '../dashboard')

from portfolio_engine import PortfolioEngine, PortfolioIntent

conn = psycopg2.connect(host='localhost', port=5432, dbname='alpha_platform', user='alpha', password='alpha_secure_2024')
query = '''
SELECT DISTINCT ON (f.ticker)
    f.ticker, f.sector, f.market_cap, f.revenue_growth,
    s.sentiment_score, s.fundamental_score, s.growth_score,
    t.signal_type, a.ai_probability
FROM fundamentals f
LEFT JOIN screener_scores s ON f.ticker = s.ticker AND s.date = (SELECT MAX(date) FROM screener_scores WHERE ticker = f.ticker)
LEFT JOIN trading_signals t ON f.ticker = t.ticker AND t.date = (SELECT MAX(date) FROM trading_signals WHERE ticker = f.ticker)
LEFT JOIN ai_recommendations a ON f.ticker = a.ticker AND a.recommendation_date = (SELECT MAX(recommendation_date) FROM ai_recommendations WHERE ticker = f.ticker)
ORDER BY f.ticker, f.date DESC
'''
df = pd.read_sql(query, conn)
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
    equal_weight=False,
    max_position_pct=10
)

engine = PortfolioEngine(df)
result = engine.build_portfolio(intent, user_request='Aggressive AI growth portfolio')

print("="*60)
print("WARNINGS (should explain missing tickers):")
print("="*60)
if result.warnings:
    for w in result.warnings:
        print(f"  {w}")
else:
    print("  No warnings")

print(f"\nHoldings: {result.num_holdings}/29")
print(f"Success: {result.success}")
