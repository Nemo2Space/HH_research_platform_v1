
import sys
import os
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.screener.worker import ScreenerWorker
from src.db.connection import get_connection
from collections import Counter


def get_all_tickers():
    """Get all tickers from fundamentals table."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT ticker FROM fundamentals ORDER BY ticker")
            return [row[0] for row in cur.fetchall()]


def run_screener(tickers: list, use_llm: bool = True, force_news: bool = False):
    """Run screener on list of tickers."""
    mode_str = []
    if use_llm:
        mode_str.append("LLM=ON")
    else:
        mode_str.append("LLM=OFF")

    if force_news:
        mode_str.append("FRESH_NEWS=ON")
    else:
        mode_str.append("SMART_CACHE=ON (6h)")

    print(f"Running screener on {len(tickers)} ticker(s) [{', '.join(mode_str)}]")
    print("=" * 70)

    # Initialize worker with LLM enabled
    worker = ScreenerWorker(use_llm=use_llm)

    # Process tickers
    results = []
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:3}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        try:
            result = worker.process_ticker(ticker, force_news=force_news, days_back=7)
            sent = result.get('sentiment_score')
            sig = result.get('signal', {}).get('type', 'N/A')
            print(f"Sentiment={sent}, Signal={sig}")
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")

    print("=" * 70)
    print(f"Completed {len(results)}/{len(tickers)} tickers")

    # Summary
    if results:
        signals = Counter(r.get('signal', {}).get('type', 'N/A') for r in results)
        print("\nSignal Summary:")
        for sig, count in signals.most_common():
            print(f"  {sig}: {count}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run screener with LLM sentiment")
    parser.add_argument("--ticker", "-t", nargs='+', help="Specific ticker(s) to process")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM sentiment (faster)")
    parser.add_argument("--fresh-news", action="store_true",
                        help="Force fresh news collection (bypass 6h cache)")

    args = parser.parse_args()

    if args.ticker:
        # Run specific tickers
        tickers = [t.upper() for t in args.ticker]
    else:
        # Run all tickers
        tickers = get_all_tickers()

    run_screener(
        tickers,
        use_llm=not args.no_llm,
        force_news=args.fresh_news
    )


if __name__ == "__main__":
    main()