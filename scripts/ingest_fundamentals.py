"""
Ingest Fundamental Data for Full Universe

Usage:
    python scripts/ingest_fundamentals.py
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.data.fundamentals import FundamentalDataIngester
from src.db.repository import Repository


def main():
    print("=" * 50)
    print("Alpha Platform - Fundamental Data Ingestion")
    print("=" * 50)
    print()

    repo = Repository()
    ingester = FundamentalDataIngester(repo)

    def progress(current, total, ticker):
        pct = int(current / total * 100)
        print(f"[{pct:3d}%] ({current:3d}/{total}) {ticker}")

    results = ingester.ingest_universe(delay=0.3, progress_callback=progress)

    print()
    print("=" * 50)
    print(f"SUCCESS: {len(results['success'])} tickers")
    print(f"FAILED:  {len(results['failed'])} tickers")

    if results['failed']:
        print(f"\nFailed: {results['failed']}")


if __name__ == "__main__":
    main()