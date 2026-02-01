import sys
sys.path.insert(0, '../dashboard')
import pandas as pd
import psycopg2
import logging

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO)

from portfolio_builder import get_llm_client, get_intent_extraction_prompt, parse_llm_intent

# Get real data
conn = psycopg2.connect(host='localhost', port=5432, dbname='alpha_platform', user='alpha', password='alpha_secure_2024')
cur = conn.cursor()
cur.execute("SELECT DISTINCT ticker FROM fundamentals")
tickers = [row[0] for row in cur.fetchall()]
cur.execute("SELECT DISTINCT sector FROM fundamentals WHERE sector IS NOT NULL")
sectors = [row[0] for row in cur.fetchall()]
conn.close()

print(f"Valid tickers: {len(tickers)}")
print(f"Valid sectors: {sectors}")

# User request
user_request = '''Build an aggressive AI growth portfolio using ONLY these 27 stocks:

NVDA, AMD, AVGO, MRVL, ARM, MU, TSM, ASML, SNPS, SMCI, ANET, DELL, VRT, ETN, SBGSY, CEG, MSFT, META, GOOGL, AMZN, PLTR, CRWD, SNOW, DDOG, PANW, PATH, NET

\,000 fully invested. Max 10% per position.'''

# Get the prompt
prompt = get_intent_extraction_prompt(user_request, sectors=sectors, tickers=tickers)

print("\n" + "="*60)
print("STEP 1: Getting LLM client...")
print("="*60)

client = get_llm_client()
if client:
    print(f"Client: {client.model_name}")
else:
    print("ERROR: No client!")
    exit()

print("\n" + "="*60)
print("STEP 2: Calling Qwen...")
print("="*60)

messages = [{"role": "user", "content": prompt}]

try:
    llm_output = client.chat(messages)
    print(f"Qwen response length: {len(llm_output)}")
    print("\n--- RAW QWEN OUTPUT ---")
    print(llm_output[:2000])
    if len(llm_output) > 2000:
        print("... [truncated]")
except Exception as e:
    print(f"ERROR calling Qwen: {e}")
    exit()

print("\n" + "="*60)
print("STEP 3: Parsing intent...")
print("="*60)

intent, errors = parse_llm_intent(llm_output, valid_tickers=tickers, valid_sectors=sectors)

print(f"Errors: {errors}")
print(f"\nParsed intent:")
print(f"  tickers_include: {intent.tickers_include}")
print(f"  restrict_to_tickers: {intent.restrict_to_tickers}")
print(f"  max_holdings: {intent.max_holdings}")
print(f"  fully_invested: {intent.fully_invested}")
print(f"  Number of tickers: {len(intent.tickers_include) if intent.tickers_include else 0}")
