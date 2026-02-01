"""
Test Market Data Ingestion

Run this script to fetch prices for a few test tickers.

Usage:
    python scripts/test_market_data.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.data.market import MarketDataIngester


def main():
    print("=" * 50)
    print("Alpha Platform - Market Data Test")
    print("=" * 50)
    print()

    ingester = MarketDataIngester()

    # Test with 5 tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']

    print(f"Testing {len(test_tickers)} tickers: {test_tickers}")
    print()

    for ticker in test_tickers:
        print(f"Fetching {ticker}...", end=" ")
        success = ingester.ingest_ticker(ticker, days=30)

        if success:
            print("OK")
        else:
            print("FAILED")

    print()

    # Verify data was saved
    print("Verifying saved data...")
    latest_df = ingester.get_latest_prices()

    if len(latest_df) > 0:
        print(f"Found {len(latest_df)} tickers with price data:")
        print()
        print(latest_df[['ticker', 'date', 'close', 'volume']].to_string(index=False))
    else:
        print("No price data found!")

    print()
    print("=" * 50)
    print("SUCCESS - Market data test complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()