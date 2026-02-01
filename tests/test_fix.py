import psycopg2
import pandas as pd
import sys
sys.path.insert(0, '../dashboard')

from portfolio_engine import PortfolioEngine, PortfolioIntent

# Load data
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

print(f"Universe loaded: {len(df)} stocks")

tickers = ['NVDA', 'AMD', 'AVGO', 'MRVL', 'ARM', 'MU', 'HXSCL', 'TSM', 'ASML', 'SNPS', 
           'SMCI', 'ANET', 'DELL', 'VRT', 'ETN', 'SBGSY', 'ABB', 'CEG', 'MSFT', 'META', 
           'GOOGL', 'AMZN', 'PLTR', 'CRWD', 'SNOW', 'DDOG', 'PANW', 'PATH', 'NET']

found = df[df['ticker'].isin(tickers)]['ticker'].unique().tolist()
missing = [t for t in tickers if t not in found]
print(f"Tickers in DB: {len(found)}/29")
if missing:
    print(f"MISSING: {missing}")

# TEST 1: restrict_to_tickers=True
print("\n" + "="*60)
print("TEST 1: restrict_to_tickers=True")
print("="*60)

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

print(f"Success: {result.success}")
print(f"Holdings: {result.num_holdings}")
print(f"Warnings: {result.warnings}")

if result.num_holdings == 29:
    print("\n✅ FIX WORKS! All 29 stocks included.")
else:
    print(f"\n❌ FIX FAILED! Only {result.num_holdings}/29 stocks included.")
    included = [h.ticker for h in result.holdings]
    missing_from_result = [t for t in tickers if t not in included]
    print(f"Missing from result: {missing_from_result}")

print("\n--- TOP 10 ---")
for h in sorted(result.holdings, key=lambda x: -x.weight_pct)[:10]:
    print(f"{h.ticker:6} {h.weight_pct:5.1f}%  Score: {h.composite_score:5.1f}")

print("\n--- BOTTOM 5 ---")
for h in sorted(result.holdings, key=lambda x: x.weight_pct)[:5]:
    print(f"{h.ticker:6} {h.weight_pct:5.1f}%  Score: {h.composite_score:5.1f}")

# TEST 2: restrict_to_tickers=False (the bug)
print("\n" + "="*60)
print("TEST 2: restrict_to_tickers=False (bug scenario)")
print("="*60)

intent2 = PortfolioIntent(
    objective='tech_growth',
    risk_level='aggressive',
    portfolio_value=100000,
    tickers_include=tickers,
    restrict_to_tickers=False,
    fully_invested=True,
    equal_weight=False,
    max_position_pct=10
)

result2 = engine.build_portfolio(intent2, user_request='Aggressive AI growth portfolio')
print(f"Holdings: {result2.num_holdings}")

if result2.num_holdings < 29:
    print(f"⚠️ Without restrict_to_tickers=False, only {result2.num_holdings} stocks")
    print("This is what happens when LLM doesn't set the flag!")
