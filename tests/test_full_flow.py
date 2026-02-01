import sys
sys.path.insert(0, '../dashboard')
import pandas as pd
import psycopg2
import logging

logging.basicConfig(level=logging.INFO)

from portfolio_builder import get_llm_client, get_intent_extraction_prompt, parse_llm_intent
from portfolio_engine import PortfolioEngine

# Get real data with joins
conn = psycopg2.connect(host='localhost', port=5432, dbname='alpha_platform', user='alpha', password='alpha_secure_2024')

# This is the SAME query used in portfolio_builder.py
query = '''
SELECT DISTINCT ON (f.ticker)
    f.ticker, f.sector, f.market_cap, f.revenue_growth,
    f.ai_exposure_score, f.is_ai_company, f.ai_category,
    s.sentiment_score, s.fundamental_score, s.growth_score,
    s.options_flow_score,
    t.signal_type, t.signal_strength,
    a.ai_probability, a.recommendation as ai_action
FROM fundamentals f
LEFT JOIN screener_scores s ON f.ticker = s.ticker AND s.date = (SELECT MAX(date) FROM screener_scores WHERE ticker = f.ticker)
LEFT JOIN trading_signals t ON f.ticker = t.ticker AND t.date = (SELECT MAX(date) FROM trading_signals WHERE ticker = f.ticker)
LEFT JOIN ai_recommendations a ON f.ticker = a.ticker AND a.recommendation_date = (SELECT MAX(recommendation_date) FROM ai_recommendations WHERE ticker = f.ticker)
ORDER BY f.ticker, f.date DESC
'''
df = pd.read_sql(query, conn)

cur = conn.cursor()
cur.execute("SELECT DISTINCT ticker FROM fundamentals")
tickers = [row[0] for row in cur.fetchall()]
cur.execute("SELECT DISTINCT sector FROM fundamentals WHERE sector IS NOT NULL")
sectors = [row[0] for row in cur.fetchall()]
conn.close()

print(f"DataFrame: {len(df)} rows")

user_request = '''Build an aggressive AI growth portfolio using ONLY these 27 stocks:
NVDA, AMD, AVGO, MRVL, ARM, MU, TSM, ASML, SNPS, SMCI, ANET, DELL, VRT, ETN, SBGSY, CEG, MSFT, META, GOOGL, AMZN, PLTR, CRWD, SNOW, DDOG, PANW, PATH, NET
\,000 fully invested. Max 10% per position.'''

# Get LLM response
client = get_llm_client()
prompt = get_intent_extraction_prompt(user_request, sectors=sectors, tickers=tickers)
llm_output = client.chat([{"role": "user", "content": prompt}])

# Parse intent
intent, errors = parse_llm_intent(llm_output, valid_tickers=tickers, valid_sectors=sectors)

print(f"\n--- INTENT ---")
print(f"tickers_include: {len(intent.tickers_include)} tickers")
print(f"restrict_to_tickers: {intent.restrict_to_tickers}")
print(f"theme: {intent.theme}")
print(f"require_theme_match: {intent.require_theme_match}")

# NOW build portfolio with the SAME engine used in portfolio_builder.py
print(f"\n--- BUILDING PORTFOLIO ---")
engine = PortfolioEngine(df)
result = engine.build_portfolio(intent, user_request=user_request)

print(f"\nResult:")
print(f"  Success: {result.success}")
print(f"  Holdings: {result.num_holdings}")
print(f"  Warnings: {result.warnings}")

if result.num_holdings < 27:
    included = [h.ticker for h in result.holdings]
    requested = ['NVDA', 'AMD', 'AVGO', 'MRVL', 'ARM', 'MU', 'TSM', 'ASML', 'SNPS', 'SMCI', 
                 'ANET', 'DELL', 'VRT', 'ETN', 'SBGSY', 'CEG', 'MSFT', 'META', 'GOOGL', 
                 'AMZN', 'PLTR', 'CRWD', 'SNOW', 'DDOG', 'PANW', 'PATH', 'NET']
    missing = [t for t in requested if t not in included]
    print(f"  Missing tickers: {missing}")

print(f"\n--- TOP 10 HOLDINGS ---")
for h in sorted(result.holdings, key=lambda x: -x.weight_pct)[:10]:
    print(f"  {h.ticker:6} {h.weight_pct:5.1f}%")
