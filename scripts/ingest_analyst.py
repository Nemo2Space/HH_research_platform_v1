"""
Ingest Analyst Ratings & Price Targets

Usage:
    python scripts/ingest_analyst.py
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.data.analyst import AnalystDataIngester


def main():
    print("=" * 50)
    print("Alpha Platform - Analyst Data Ingestion")
    print("=" * 50)
    print()

    ingester = AnalystDataIngester()

    def progress(current, total, ticker):
        pct = int(current / total * 100)
        print(f"[{pct:3d}%] ({current:3d}/{total}) {ticker}")

    results = ingester.ingest_universe(delay=0.3, progress_callback=progress)

    print()
    print("=" * 50)
    print(f"SUCCESS: {len(results['success'])} tickers")
    print(f"FAILED:  {len(results['failed'])} tickers")


if __name__ == "__main__":
    main()