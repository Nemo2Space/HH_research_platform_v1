import psycopg2
import pandas as pd
import sys
sys.path.insert(0, '../dashboard')

conn = psycopg2.connect(host='localhost', port=5432, dbname='alpha_platform', user='alpha', password='alpha_secure_2024')
query = '''
SELECT DISTINCT ON (f.ticker)
    f.ticker, f.sector, f.market_cap,
    s.sentiment_score, s.fundamental_score, s.growth_score
FROM fundamentals f
LEFT JOIN screener_scores s ON f.ticker = s.ticker AND s.date = (SELECT MAX(date) FROM screener_scores WHERE ticker = f.ticker)
ORDER BY f.ticker, f.date DESC
'''
df = pd.read_sql(query, conn)
conn.close()

tickers = ['NVDA', 'AMD', 'AVGO', 'MRVL', 'ARM', 'MU', 'HXSCL', 'TSM', 'ASML', 'SNPS', 
           'SMCI', 'ANET', 'DELL', 'VRT', 'ETN', 'SBGSY', 'ABB', 'CEG', 'MSFT', 'META', 
           'GOOGL', 'AMZN', 'PLTR', 'CRWD', 'SNOW', 'DDOG', 'PANW', 'PATH', 'NET']

whitelist = [t.upper() for t in tickers]

print("STEP 1: Filter to whitelist")
filtered = df[df['ticker'].str.upper().isin(whitelist)]
print(f"After whitelist filter: {len(filtered)} stocks")
print(f"Tickers: {sorted(filtered['ticker'].tolist())}")

print("\nSTEP 2: Check sectors")
for sector in filtered['sector'].unique():
    count = len(filtered[filtered['sector'] == sector])
    print(f"  {sector}: {count} stocks")

print("\nSTEP 3: Check for NULL values that might cause filtering")
for col in ['market_cap', 'sentiment_score', 'fundamental_score']:
    null_count = filtered[col].isna().sum()
    if null_count > 0:
        null_tickers = filtered[filtered[col].isna()]['ticker'].tolist()
        print(f"  {col} NULL: {null_count} - {null_tickers}")

print("\nSTEP 4: Check which tickers have screener data")
has_scores = filtered[filtered['fundamental_score'].notna()]['ticker'].tolist()
no_scores = filtered[filtered['fundamental_score'].isna()]['ticker'].tolist()
print(f"  With scores: {len(has_scores)} - {has_scores}")
print(f"  No scores: {len(no_scores)} - {no_scores}")
