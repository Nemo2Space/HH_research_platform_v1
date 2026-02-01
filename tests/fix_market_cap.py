import psycopg2
import yfinance as yf

conn = psycopg2.connect(host='localhost', port=5432, dbname='alpha_platform', user='alpha', password='alpha_secure_2024')
cur = conn.cursor()

# Tickers missing market_cap
tickers = ['AMD', 'AMZN', 'ANET', 'ARM', 'ASML', 'AVGO', 'CRWD', 'DDOG', 'DELL', 
           'GOOGL', 'HXSCL', 'META', 'MRVL', 'MSFT', 'MU', 'NET', 'NVDA', 'PANW', 
           'PATH', 'PLTR', 'SMCI', 'SNOW', 'SNPS', 'TSM']

print("Updating market_cap for tickers...")
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get('marketCap')
        
        if market_cap:
            cur.execute("""
                UPDATE fundamentals 
                SET market_cap = %s 
                WHERE ticker = %s
            """, (market_cap, ticker))
            print(f"  {ticker}: {market_cap:,.0f}")
        else:
            print(f"  {ticker}: No market cap from yfinance")
    except Exception as e:
        print(f"  {ticker}: Error - {e}")

conn.commit()
print("\nDone! Verifying...")

# Verify
cur.execute("""
    SELECT ticker, market_cap 
    FROM fundamentals 
    WHERE ticker IN %s 
    ORDER BY ticker
""", (tuple(tickers),))

for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]}")

conn.close()
