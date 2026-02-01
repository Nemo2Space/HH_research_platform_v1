import sys
sys.path.insert(0, '../dashboard')
import requests
import json

# Simulate what your app sends to Qwen
user_request = '''Build an aggressive AI growth portfolio using ONLY these 29 stocks - include ALL of them:

NVDA, AMD, AVGO, MRVL, ARM, MU, HXSCL, TSM, ASML, SNPS, SMCI, ANET, DELL, VRT, ETN, SBGSY, ABB, CEG, MSFT, META, GOOGL, AMZN, PLTR, CRWD, SNOW, DDOG, PANW, PATH, NET

100000 fully invested. Max 10% per position.'''

from portfolio_builder import get_intent_extraction_prompt

extraction_prompt = get_intent_extraction_prompt(user_request)

# Call Qwen locally
response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'qwen3:32b',
        'prompt': extraction_prompt,
        'stream': False
    },
    timeout=120
)

result = response.json()
llm_response = result.get('response', '')

print("="*60)
print("QWEN RAW RESPONSE:")
print("="*60)
print(llm_response)

# Try to parse the JSON
print("\n" + "="*60)
print("PARSED VALUES:")
print("="*60)

try:
    # Find JSON in response
    import re
    json_match = re.search(r'\{[\s\S]*\}', llm_response)
    if json_match:
        data = json.loads(json_match.group())
        print(f"tickers_include: {data.get('tickers_include', 'NOT SET')}")
        print(f"restrict_to_tickers: {data.get('restrict_to_tickers', 'NOT SET')}")
        print(f"max_holdings: {data.get('max_holdings', 'NOT SET')}")
        print(f"max_position_pct: {data.get('max_position_pct', 'NOT SET')}")
        print(f"fully_invested: {data.get('fully_invested', 'NOT SET')}")
        
        if data.get('restrict_to_tickers') == True:
            print("\n✅ Qwen correctly set restrict_to_tickers=True")
        else:
            print("\n❌ Qwen did NOT set restrict_to_tickers=True - THIS IS THE BUG!")
except Exception as e:
    print(f"Parse error: {e}")
