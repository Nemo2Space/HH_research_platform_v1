import sys
sys.path.insert(0, '../dashboard')

# Check what the LLM validation does
from portfolio_engine import parse_llm_intent
import psycopg2

conn = psycopg2.connect(host='localhost', port=5432, dbname='alpha_platform', user='alpha', password='alpha_secure_2024')
cur = conn.cursor()
cur.execute("SELECT DISTINCT ticker FROM fundamentals")
valid_tickers = [row[0] for row in cur.fetchall()]
conn.close()

# Simulate what Qwen might return
fake_llm_response = '''{
    "objective": "tech_growth",
    "risk_level": "aggressive",
    "portfolio_value": 100000,
    "fully_invested": true,
    "max_position_pct": 10,
    "tickers_include": ["NVDA", "AMD", "AVGO", "MRVL", "ARM", "MU", "TSM", "ASML", "SNPS", "SMCI", "ANET", "DELL", "VRT", "ETN", "SBGSY", "CEG", "MSFT", "META", "GOOGL", "AMZN", "PLTR", "CRWD", "SNOW", "DDOG", "PANW", "PATH", "NET"],
    "restrict_to_tickers": true
}'''

intent, errors = parse_llm_intent(fake_llm_response, valid_tickers)

print("="*60)
print("PARSED INTENT:")
print("="*60)
print(f"tickers_include: {intent.tickers_include}")
print(f"restrict_to_tickers: {intent.restrict_to_tickers}")
print(f"Number of tickers: {len(intent.tickers_include)}")
print(f"\nErrors: {errors}")

# Check which tickers are in valid_tickers
requested = ['NVDA', 'AMD', 'AVGO', 'MRVL', 'ARM', 'MU', 'TSM', 'ASML', 'SNPS', 'SMCI', 
             'ANET', 'DELL', 'VRT', 'ETN', 'SBGSY', 'CEG', 'MSFT', 'META', 'GOOGL', 
             'AMZN', 'PLTR', 'CRWD', 'SNOW', 'DDOG', 'PANW', 'PATH', 'NET']

valid_upper = [t.upper() for t in valid_tickers]
invalid = [t for t in requested if t.upper() not in valid_upper]
print(f"\nTickers NOT in valid_tickers list: {invalid}")
