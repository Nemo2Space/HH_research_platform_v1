"""
Check what's in the earnings_calendar table
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

import pandas as pd

try:
    from src.db.connection import get_engine
    engine = get_engine()

    # Check total rows
    total = pd.read_sql("SELECT COUNT(*) as cnt FROM earnings_calendar", engine)
    print(f"Total rows in earnings_calendar: {total.iloc[0]['cnt']}")

    # Check date range
    dates = pd.read_sql("""
        SELECT MIN(earnings_date) as min_date, MAX(earnings_date) as max_date 
        FROM earnings_calendar
    """, engine)
    print(f"Date range: {dates.iloc[0]['min_date']} to {dates.iloc[0]['max_date']}")

    # Check how many have future earnings
    future = pd.read_sql("""
        SELECT COUNT(*) as cnt FROM earnings_calendar 
        WHERE earnings_date >= CURRENT_DATE
    """, engine)
    print(f"Stocks with future earnings: {future.iloc[0]['cnt']}")

    # Sample of data
    print("\nSample of upcoming earnings:")
    sample = pd.read_sql("""
        SELECT ticker, earnings_date, earnings_time 
        FROM earnings_calendar 
        WHERE earnings_date >= CURRENT_DATE
        ORDER BY earnings_date
        LIMIT 20
    """, engine)
    print(sample.to_string())

    # Check specific tickers from universe
    print("\n\nChecking specific tickers from universe:")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'MU', 'TSLA', 'META']
    for t in tickers:
        df = pd.read_sql(f"SELECT * FROM earnings_calendar WHERE ticker = '{t}'", engine)
        if df.empty:
            print(f"  {t}: NO ENTRY in earnings_calendar")
        else:
            print(f"  {t}: {df.iloc[0]['earnings_date']}")

except Exception as e:
    print(f"Error: {e}")