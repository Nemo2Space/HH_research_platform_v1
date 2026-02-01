"""
Diagnose Sector Data Issue
===========================
Check why sector is not being populated for stocks.
"""

import yfinance as yf
from datetime import date

# Test tickers
TICKERS = ['NVAX', 'CERT', 'ADPT', 'ABEO', 'CRMD']

print("=" * 60)
print("SECTOR DATA DIAGNOSIS")
print("=" * 60)

# Step 1: Check yfinance data
print("\n1. YFINANCE DATA:")
print("-" * 40)
for ticker in TICKERS:
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        sector = info.get('sector')
        industry = info.get('industry')
        print(f"{ticker}: Sector='{sector}', Industry='{industry}'")
    except Exception as e:
        print(f"{ticker}: ERROR - {e}")

# Step 2: Check Finviz data
print("\n2. FINVIZ DATA:")
print("-" * 40)
try:
    from finvizfinance.quote import finvizfinance

    for ticker in TICKERS:
        try:
            stock = finvizfinance(ticker)
            data = stock.ticker_fundament()
            sector = data.get('Sector')
            industry = data.get('Industry')
            print(f"{ticker}: Sector='{sector}', Industry='{industry}'")
        except Exception as e:
            print(f"{ticker}: ERROR - {e}")
except ImportError:
    print("Finviz not installed")

# Step 3: Check database - screener_scores table
print("\n3. DATABASE - screener_scores:")
print("-" * 40)
try:
    from src.db.connection import get_connection

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Check if sector column exists
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'screener_scores'
                ORDER BY ordinal_position
            """)
            columns = [row[0] for row in cur.fetchall()]
            print(f"Columns in screener_scores: {columns}")

            if 'sector' in columns:
                print("\n✓ 'sector' column EXISTS")
                for ticker in TICKERS:
                    cur.execute("SELECT sector FROM screener_scores WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                                (ticker,))
                    row = cur.fetchone()
                    print(f"  {ticker}: sector = {row[0] if row else 'NO ROW'}")
            else:
                print("\n✗ 'sector' column DOES NOT EXIST in screener_scores")
except Exception as e:
    print(f"DB Error: {e}")

# Step 4: Check database - fundamentals table
print("\n4. DATABASE - fundamentals:")
print("-" * 40)
try:
    from src.db.connection import get_connection

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'fundamentals'
                ORDER BY ordinal_position
            """)
            columns = [row[0] for row in cur.fetchall()]
            print(f"Columns in fundamentals: {columns}")

            if 'sector' in columns:
                print("\n✓ 'sector' column EXISTS")
                for ticker in TICKERS:
                    cur.execute("SELECT sector FROM fundamentals WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                                (ticker,))
                    row = cur.fetchone()
                    print(f"  {ticker}: sector = {row[0] if row else 'NO ROW'}")
            else:
                print("\n✗ 'sector' column DOES NOT EXIST in fundamentals")
except Exception as e:
    print(f"DB Error: {e}")

# Step 5: Check universe.csv
print("\n5. UNIVERSE.CSV:")
print("-" * 40)
try:
    import pandas as pd
    import os

    # Try common paths
    paths = [
        'config/universe.csv',
        '../config/universe.csv',
        'universe.csv'
    ]

    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Found at: {path}")
            print(f"Columns: {list(df.columns)}")

            if 'sector' in df.columns:
                for ticker in TICKERS:
                    row = df[df['ticker'].str.upper() == ticker.upper()]
                    if not row.empty:
                        print(f"  {ticker}: sector = {row['sector'].values[0]}")
                    else:
                        print(f"  {ticker}: NOT IN universe.csv")
            else:
                print("✗ 'sector' column NOT in universe.csv")
            break
    else:
        print("universe.csv not found in common paths")
except Exception as e:
    print(f"Error: {e}")

# Step 6: Check signal generation
print("\n6. SIGNAL GENERATION:")
print("-" * 40)
try:
    from src.core import get_signal_engine

    engine = get_signal_engine()

    for ticker in TICKERS[:2]:  # Just test 2 to save time
        signal = engine.generate_signal(ticker, force_refresh=False)
        if signal:
            print(f"{ticker}: signal.sector = '{signal.sector}'")
        else:
            print(f"{ticker}: No signal generated")
except Exception as e:
    print(f"Signal engine error: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)