"""
Run Full Screener

Runs the screener on all tickers and generates signals.

Usage:
    python scripts/run_screener.py
    python scripts/run_screener.py --no-llm    # Skip LLM sentiment (faster)
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.screener.worker import ScreenerWorker


def main():
    # Check for --no-llm flag
    use_llm = '--no-llm' not in sys.argv

    print("=" * 50)
    print("Alpha Platform - Full Screener Run")
    print("=" * 50)
    print()
    print(f"LLM Sentiment: {'Enabled' if use_llm else 'Disabled (using defaults)'}")
    print()

    worker = ScreenerWorker(use_llm=use_llm)

    def progress(current, total, ticker):
        pct = int(current / total * 100)
        print(f"[{pct:3d}%] ({current:3d}/{total}) {ticker}")

    results = worker.run_full_screen(
        collect_news=False,  # Use existing news in DB
        delay=0.1,
        progress_callback=progress
    )

    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Processed: {results['processed']}/{results['total']}")
    print(f"Time: {results['elapsed_seconds']:.1f}s")
    print()
    print("Signal Summary:")
    for signal, count in sorted(results['signal_summary'].items(), key=lambda x: -x[1]):
        print(f"  {signal}: {count}")

    if results['errors']:
        print()
        print(f"Errors: {len(results['errors'])}")
        for err in results['errors'][:5]:
            print(f"  {err['ticker']}: {err['error']}")


if __name__ == "__main__":
    main()