import sys
sys.path.insert(0, '../dashboard')

from portfolio_builder import get_intent_extraction_prompt, parse_llm_intent

# This is your prompt
user_request = '''Build an aggressive AI growth portfolio using ONLY these 29 stocks - include ALL of them:

NVDA, AMD, AVGO, MRVL, ARM, MU, HXSCL, TSM, ASML, SNPS, SMCI, ANET, DELL, VRT, ETN, SBGSY, ABB, CEG, MSFT, META, GOOGL, AMZN, PLTR, CRWD, SNOW, DDOG, PANW, PATH, NET

\,000 fully invested. Max 10% per position.'''

# Get the prompt that goes to Qwen
extraction_prompt = get_intent_extraction_prompt(user_request)

print("="*60)
print("PROMPT SENT TO QWEN:")
print("="*60)
print(extraction_prompt[:2000])
print("\n... [truncated]")

# Check if restrict_to_tickers is even in the schema
if 'restrict_to_tickers' in extraction_prompt:
    print("\n✅ restrict_to_tickers IS in the extraction prompt")
else:
    print("\n❌ restrict_to_tickers is NOT in the extraction prompt - THIS IS THE BUG!")
