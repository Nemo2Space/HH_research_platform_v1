import json

with open(r'/ai_growth_etf_2026.json', 'r') as f:
    data1 = json.load(f)
print(f"ai_growth_etf_2026.json: {len(data1)} stocks")

# Check the IBKR file format
ibkr_content = open('KAIF_holdings_etfDB_ibkr_latest.json', 'r').read() if __import__('os').path.exists('KAIF_holdings_etfDB_ibkr_latest.json') else None
