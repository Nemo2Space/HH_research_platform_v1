"""
Ingest Prices for Full Universe

Run this script to fetch prices for all 100 tickers.

Usage:
    python scripts/ingest_all_prices.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.data.market import MarketDataIngester
from src.db.repository import Repository


def main():
    print("=" * 50)
    print("Alpha Platform - Full Universe Price Ingestion")
    print("=" * 50)
    print()

    repo = Repository()
    ingester = MarketDataIngester(repo)

    # Update system status
    repo.update_system_status('ingestion', 'running', progress_pct=0, progress_message='Starting...')

    def progress(current, total, ticker):
        pct = int(current / total * 100)
        print(f"[{pct:3d}%] ({current:3d}/{total}) {ticker}")
        repo.update_system_status('ingestion', 'running', progress_pct=pct, progress_message=f'Processing {ticker}')

    try:
        results = ingester.ingest_universe(days=60, delay=0.3, progress_callback=progress)

        print()
        print("=" * 50)
        print(f"SUCCESS: {len(results['success'])} tickers")
        print(f"FAILED:  {len(results['failed'])} tickers")

        if results['failed']:
            print(f"\nFailed: {results['failed']}")

        repo.update_system_status('ingestion', 'completed', progress_message=f"Done: {len(results['success'])} success")

    except Exception as e:
        print(f"\nERROR: {e}")
        repo.update_system_status('ingestion', 'error', error_message=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()