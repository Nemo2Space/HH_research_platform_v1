"""Fetch insider transactions for all tickers"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.data.insider import InsiderDataFetcher

def main():
    fetcher = InsiderDataFetcher()

    def progress(current, total, ticker):
        print(f"[{current:3}/{total}] {ticker}...", end=" ", flush=True)

    print("Fetching insider transactions...")
    print("=" * 60)

    results = fetcher.fetch_universe(days_back=90, delay=0.3, progress_callback=progress)

    print("\n" + "=" * 60)

    # Summary
    total = sum(results.values())
    with_data = sum(1 for v in results.values() if v > 0)

    print(f"Total transactions: {total}")
    print(f"Tickers with insider data: {with_data}/{len(results)}")

if __name__ == "__main__":
    main()