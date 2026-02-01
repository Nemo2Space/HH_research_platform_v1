"""Test Institutional Holdings Data"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.data.institutional import InstitutionalDataFetcher


def main():
    print("=" * 60)
    print("Alpha Platform - Institutional Holdings Test")
    print("=" * 60)

    fetcher = InstitutionalDataFetcher()

    if not fetcher.finnhub_key:
        print("ERROR: FINNHUB_API_KEY not set in .env")
        return

    test_tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA']

    print("\nFetching institutional holdings...")
    print("-" * 60)

    for ticker in test_tickers:
        holdings = fetcher.get_finnhub_institutional(ticker)

        if holdings:
            print(f"\n{ticker}: {len(holdings)} institutional holders")

            for h in holdings[:3]:
                change_str = f"+{h.shares_change:,}" if h.shares_change > 0 else f"{h.shares_change:,}"
                print(f"  {h.institution_name[:35]:<35} | {h.shares_held:>12,} | {change_str:>12} | {h.position_type}")

            saved = fetcher.fetch_and_save(ticker)
            print(f"  Saved: {saved}")

            signal = fetcher.get_institutional_signal(ticker)
            print(f"  Signal: {signal['institutional_signal']} - {signal['summary']}")
        else:
            print(f"\n{ticker}: No data")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()