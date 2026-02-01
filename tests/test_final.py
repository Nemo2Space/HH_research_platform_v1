import psycopg2
import pandas as pd
import sys
sys.path.insert(0, '../dashboard')

from portfolio_engine import PortfolioEngine, PortfolioIntent

conn = psycopg2.connect(host='localhost', port=5432, dbname='alpha_platform', user='alpha', password='alpha_secure_2024')
query = '''
SELECT DISTINCT ON (f.ticker)
    f.ticker, f.sector, f.market_cap, f.revenue_growth,
    f.ai_exposure_score, f.is_ai_company, f.ai_category,
    s.sentiment_score, s.fundamental_score, s.growth_score,
    s.options_flow_score, s.composite_score as screener_composite,
    t.signal_type, t.signal_strength,
    a.ai_probability, a.recommendation as ai_action
FROM fundamentals f
LEFT JOIN screener_scores s ON f.ticker = s.ticker AND s.date = (SELECT MAX(date) FROM screener_scores WHERE ticker = f.ticker)
LEFT JOIN trading_signals t ON f.ticker = t.ticker AND t.date = (SELECT MAX(date) FROM trading_signals WHERE ticker = f.ticker)
LEFT JOIN ai_recommendations a ON f.ticker = a.ticker AND a.recommendation_date = (SELECT MAX(recommendation_date) FROM ai_recommendations WHERE ticker = f.ticker)
ORDER BY f.ticker, f.date DESC
'''
df = pd.read_sql(query, conn)
conn.close()

print(f"Universe: {len(df)} stocks")

tickers = ['NVDA', 'AMD', 'AVGO', 'MRVL', 'ARM', 'MU', 'HXSCL', 'TSM', 'ASML', 'SNPS', 
           'SMCI', 'ANET', 'DELL', 'VRT', 'ETN', 'SBGSY', 'ABB', 'CEG', 'MSFT', 'META', 
           'GOOGL', 'AMZN', 'PLTR', 'CRWD', 'SNOW', 'DDOG', 'PANW', 'PATH', 'NET']

# TEST: restrict_to_tickers=False but 29 tickers provided
# This simulates what Qwen does (doesn't set restrict_to_tickers)
print("\n" + "="*60)
print("TEST: restrict_to_tickers=FALSE, but 29 tickers provided")
print("(This simulates Qwen not setting the flag)")
print("="*60)

intent = PortfolioIntent(
    objective='tech_growth',
    risk_level='aggressive',
    portfolio_value=100000,
    tickers_include=tickers,
    restrict_to_tickers=False,  # Qwen doesn't set this!
    fully_invested=True,
    equal_weight=False,
    max_position_pct=10
)

engine = PortfolioEngine(df)
result = engine.build_portfolio(intent, user_request='Aggressive AI growth portfolio')

print(f"Success: {result.success}")
print(f"Holdings: {result.num_holdings}")

if result.num_holdings == 29:
    print("\n✅ SUCCESS! All 29 stocks included even without restrict_to_tickers!")
else:
    print(f"\n❌ FAILED! Only {result.num_holdings}/29 stocks")
    included = [h.ticker for h in result.holdings]
    missing = [t for t in tickers if t not in included]
    print(f"Missing: {missing}")

print("\n--- TOP 10 ---")
for h in sorted(result.holdings, key=lambda x: -x.weight_pct)[:10]:
    print(f"{h.ticker:6} {h.weight_pct:5.1f}%  Score: {h.composite_score:5.1f}")
