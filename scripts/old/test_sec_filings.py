"""
Test SEC Filing Fetcher and RAG

Usage:
    python scripts/test_sec_filings.py
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.data.sec_filings import SECFilingFetcher
from src.data.sec_rag import SECChunker, SECRetriever


def main():
    print("=" * 60)
    print("Alpha Platform - SEC Filing Test")
    print("=" * 60)
    print()

    fetcher = SECFilingFetcher()
    chunker = SECChunker()
    retriever = SECRetriever()

    # Test with a few tickers
    test_tickers = ['AAPL', 'MSFT', 'NVDA']

    print("Fetching SEC filings...")
    print("-" * 40)

    for ticker in test_tickers:
        print(f"\n{ticker}:")

        # Get CIK
        cik = fetcher.get_cik(ticker)
        print(f"  CIK: {cik}")

        # Get recent filings
        filings = fetcher.get_recent_filings(ticker, limit=3)
        print(f"  Found {len(filings)} filings:")

        for filing in filings[:3]:
            print(f"    - {filing.form_type} ({filing.filing_date})")

        # Fetch and save one filing
        if filings:
            print(f"  Fetching content for {filings[0].form_type}...")
            saved = fetcher.fetch_and_save(ticker, limit=1)
            print(f"  Saved {saved} filings")

    print()
    print("Processing chunks...")
    print("-" * 40)

    # Process all filings into chunks
    total_chunks = chunker.process_all_filings()
    print(f"Created {total_chunks} chunks total")

    print()
    print("Testing retrieval...")
    print("-" * 40)

    for ticker in test_tickers:
        context = retriever.get_context_for_analysis(ticker, max_chunks=5)
        if context:
            print(f"\n{ticker} SEC Context ({len(context)} chars):")
            print(context[:500] + "..." if len(context) > 500 else context)
        else:
            print(f"\n{ticker}: No SEC context available")

    print()
    print("=" * 60)
    print("SUCCESS - SEC filing test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()