"""
Test Insider Trading Data

Usage:
    python scripts/test_insider.py
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.data.insider import InsiderDataFetcher


def main():
    print("=" * 60)
    print("Alpha Platform - Insider Trading Test")
    print("=" * 60)
    print()

    fetcher = InsiderDataFetcher()

    # Check if Finnhub key is configured
    if not fetcher.finnhub_key:
        print("WARNING: FINNHUB_API_KEY not set in .env")
        print("Insider trading data requires a Finnhub API key.")
        print()

    # Test with a few tickers
    test_tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA']

    print("Fetching insider transactions...")
    print("-" * 60)

    for ticker in test_tickers:
        transactions = fetcher.get_finnhub_insider(ticker, days_back=90)

        if transactions:
            print(f"\n{ticker}: {len(transactions)} transactions")

            # Show recent transactions
            for tx in transactions[:3]:
                print(f"  {tx.transaction_date} | {tx.insider_name[:20]:<20} | "
                      f"{tx.transaction_type:>4} | {tx.shares:>10,} shares @ ${tx.price:.2f}")

            # Save to database
            saved = fetcher.fetch_and_save(ticker, days_back=90)
            print(f"  Saved {saved} transactions")

            # Get signal
            signal = fetcher.get_insider_signal(ticker, days_back=30)
            print(f"  Signal: {signal['insider_signal']} - {signal['summary']}")
        else:
            print(f"\n{ticker}: No insider transactions found")

    print()
    print("=" * 60)
    print("SUCCESS - Insider trading test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()