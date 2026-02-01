"""
Fetch Historical Prices for Backtesting

Downloads price data for universe + benchmarks.

Usage:
    python scripts/fetch_historical_prices.py
    python scripts/fetch_historical_prices.py --benchmarks-only
"""

import sys
import os
import argparse
import time
from datetime import datetime, timedelta

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
import pandas as pd
from src.db.connection import get_connection
from src.db.repository import Repository

BENCHMARKS = ['SPY', 'VOO', 'QQQ', 'IWM', 'DIA']


def fetch_prices(ticker: str, years: int = 3) -> pd.DataFrame:
    """Fetch historical prices from Yahoo Finance."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            print(f"  {ticker}: No data returned")
            return pd.DataFrame()

        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        })

        df['date'] = pd.to_datetime(df['date']).dt.date
        df['ticker'] = ticker

        return df

    except Exception as e:
        print(f"  {ticker}: Error - {e}")
        return pd.DataFrame()


def save_prices(df: pd.DataFrame) -> int:
    """Save prices to database."""
    if df.empty:
        return 0

    saved = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                try:
                    cur.execute("""
                        INSERT INTO prices (ticker, date, open, high, low, close, adj_close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            adj_close = EXCLUDED.adj_close,
                            volume = EXCLUDED.volume
                    """, (
                        row['ticker'],
                        row['date'],
                        float(row['open']) if pd.notna(row['open']) else None,
                        float(row['high']) if pd.notna(row['high']) else None,
                        float(row['low']) if pd.notna(row['low']) else None,
                        float(row['close']) if pd.notna(row['close']) else None,
                        float(row.get('adj_close', row['close'])) if pd.notna(row.get('adj_close', row['close'])) else None,
                        int(row['volume']) if pd.notna(row['volume']) else 0,
                    ))
                    saved += 1
                except Exception as e:
                    pass
    return saved


def main():
    parser = argparse.ArgumentParser(description="Fetch historical prices")
    parser.add_argument("--years", type=int, default=3, help="Years of history")
    parser.add_argument("--benchmarks-only", action="store_true", help="Only fetch benchmarks")
    parser.add_argument("--ticker", "-t", nargs='+', help="Specific ticker(s)")
    args = parser.parse_args()

    print("=" * 60)
    print("FETCH HISTORICAL PRICES")
    print("=" * 60)

    if args.ticker:
        tickers = [t.upper() for t in args.ticker]
    elif args.benchmarks_only:
        tickers = BENCHMARKS
    else:
        repo = Repository()
        universe = repo.get_universe()
        tickers = list(set(universe + BENCHMARKS))

    print(f"Fetching {len(tickers)} tickers, {args.years} years of history\n")

    total_saved = 0
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:3}/{len(tickers)}] {ticker}...", end=" ", flush=True)

        df = fetch_prices(ticker, years=args.years)
        if df.empty:
            print("NO DATA")
            continue

        saved = save_prices(df)
        total_saved += saved
        print(f"OK ({saved} days)")

        time.sleep(0.3)

    print(f"\nTotal saved: {total_saved:,} records")


if __name__ == "__main__":
    main()