import psycopg2
import pandas as pd
import sys
sys.path.insert(0, '../dashboard')

from portfolio_engine import PortfolioEngine, PortfolioIntent

conn = psycopg2.connect(host='localhost', port=5432, dbname='alpha_platform', user='alpha', password='alpha_secure_2024')

# Join all relevant tables
query = '''
SELECT DISTINCT ON (f.ticker)
    f.ticker,
    f.sector,
    f.market_cap,
    f.pe_ratio,
    f.dividend_yield,
    f.revenue_growth,
    f.ai_exposure_score,
    f.is_ai_company,
    f.ai_category,
    -- Screener scores
    s.sentiment_score,
    s.fundamental_score,
    s.growth_score,
    s.technical_score,
    s.dividend_score,
    s.options_flow_score,
    s.short_squeeze_score,
    s.options_sentiment,
    s.squeeze_risk,
    s.target_upside_pct,
    s.days_to_earnings,
    s.composite_score as screener_composite,
    -- Trading signals
    t.signal_type,
    t.signal_strength,
    t.signal_reason,
    -- AI recommendations
    a.ai_probability,
    a.recommendation as ai_action
FROM fundamentals f
LEFT JOIN screener_scores s ON f.ticker = s.ticker AND s.date = (SELECT MAX(date) FROM screener_scores WHERE ticker = f.ticker)
LEFT JOIN trading_signals t ON f.ticker = t.ticker AND t.date = (SELECT MAX(date) FROM trading_signals WHERE ticker = f.ticker)
LEFT JOIN ai_recommendations a ON f.ticker = a.ticker AND a.recommendation_date = (SELECT MAX(recommendation_date) FROM ai_recommendations WHERE ticker = f.ticker)
ORDER BY f.ticker, f.date DESC
'''

df = pd.read_sql(query, conn)
conn.close()

print(f"Universe: {len(df)} stocks")
print(f"\nSample data for NVDA:")
nvda = df[df['ticker'] == 'NVDA'].iloc[0]
print(f"  Sector: {nvda['sector']}")
print(f"  Sentiment Score: {nvda['sentiment_score']}")
print(f"  Fundamental Score: {nvda['fundamental_score']}")
print(f"  Growth Score: {nvda['growth_score']}")
print(f"  AI Probability: {nvda['ai_probability']}")
print(f"  Signal Type: {nvda['signal_type']}")

# Build portfolio
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

print(f"\n--- TOP 15 BY WEIGHT ---")
for h in sorted(result.holdings, key=lambda x: -x.weight_pct)[:15]:
    print(f"{h.ticker:6} {h.weight_pct:5.1f}%  Score: {h.composite_score:5.1f}  {h.conviction}")

print(f"\n--- BOTTOM 5 ---")
for h in sorted(result.holdings, key=lambda x: x.weight_pct)[:5]:
    print(f"{h.ticker:6} {h.weight_pct:5.1f}%  Score: {h.composite_score:5.1f}  {h.conviction}")
