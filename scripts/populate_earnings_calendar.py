"""
Bulk populate earnings_calendar from yfinance for all tickers in universe.
Run: python scripts/populate_earnings_calendar.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

import yfinance as yf
import pandas as pd
from datetime import date
import time
from sqlalchemy import text

def populate_earnings_calendar():
    """Fetch and save earnings dates for all tickers in screener_scores."""

    from src.db.connection import get_engine

    engine = get_engine()

    # Get all unique tickers from screener_scores
    tickers_df = pd.read_sql("SELECT DISTINCT ticker FROM screener_scores", engine)
    tickers = tickers_df['ticker'].tolist()

    print(f"Found {len(tickers)} tickers to update")

    # First, check/fix table structure
    print("\nChecking table structure...")
    try:
        with engine.connect() as conn:
            # Check if unique constraint exists
            result = conn.execute(text("""
                SELECT COUNT(*) as cnt FROM pg_constraint 
                WHERE conname = 'earnings_calendar_ticker_key'
            """))
            row = result.fetchone()

            if row[0] == 0:
                print("Adding unique constraint on ticker...")
                try:
                    conn.execute(text("ALTER TABLE earnings_calendar ADD CONSTRAINT earnings_calendar_ticker_key UNIQUE (ticker)"))
                    conn.commit()
                    print("  Done!")
                except Exception as e:
                    conn.rollback()
                    print(f"  Could not add constraint: {e}")
    except Exception as e:
        print(f"  Note: {e}")

    today = date.today()
    updated = 0
    failed = 0
    no_data = 0

    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            ed = stock.earnings_dates

            if ed is not None and not ed.empty:
                # Get the next upcoming earnings date
                found_date = None
                for idx in ed.index:
                    earnings_dt = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                    if earnings_dt >= today:
                        found_date = earnings_dt
                        break

                if found_date:
                    # Use fresh connection for each insert
                    try:
                        with engine.connect() as conn:
                            # Delete existing and insert new
                            conn.execute(text(f"DELETE FROM earnings_calendar WHERE ticker = '{ticker}'"))
                            conn.execute(text(f"INSERT INTO earnings_calendar (ticker, earnings_date) VALUES ('{ticker}', '{found_date}')"))
                            conn.commit()
                        print(f"[{i+1}/{len(tickers)}] {ticker}: {found_date}")
                        updated += 1
                    except Exception as db_err:
                        print(f"[{i+1}/{len(tickers)}] {ticker}: DB ERROR - {db_err}")
                        failed += 1
                else:
                    print(f"[{i+1}/{len(tickers)}] {ticker}: No future earnings")
                    no_data += 1
            else:
                print(f"[{i+1}/{len(tickers)}] {ticker}: No earnings data")
                no_data += 1

        except Exception as e:
            err_str = str(e)
            if "No fundamentals data" in err_str or "symbol may be delisted" in err_str:
                print(f"[{i+1}/{len(tickers)}] {ticker}: No data (ETF/delisted)")
                no_data += 1
            else:
                print(f"[{i+1}/{len(tickers)}] {ticker}: ERROR - {e}")
                failed += 1

        # Rate limiting - avoid hitting Yahoo too fast
        if i > 0 and i % 10 == 0:
            time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"DONE: Updated {updated}, No data {no_data}, Failed {failed}")
    print(f"{'='*50}")

if __name__ == "__main__":
    populate_earnings_calendar()