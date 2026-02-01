"""Run committee analysis for all tickers or a specific ticker"""

import sys
import os
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.committee.coordinator import CommitteeCoordinator


def main():
    parser = argparse.ArgumentParser(description='Run committee analysis')
    parser.add_argument('--ticker', type=str, help='Single ticker to analyze')
    parser.add_argument('--all', action='store_true', help='Analyze all tickers')
    args = parser.parse_args()

    coordinator = CommitteeCoordinator()

    if args.ticker:
        # Single ticker
        print(f"Running committee analysis for {args.ticker}...")
        print("=" * 60)

        decision = coordinator.analyze_ticker(args.ticker)

        print(f"\nVerdict: {decision.verdict}")
        print(f"Conviction: {decision.conviction}%")
        print(f"Expected Alpha: {decision.expected_alpha_bps:.0f} bps")
        print(f"Horizon: {decision.horizon_days} days")
        print(f"\nRationale: {decision.rationale}")
        print(f"\nRisks: {', '.join(decision.risks)}")

        print("\nAgent Votes:")
        for vote in decision.votes:
            print(f"  {vote.role}: buy_prob={vote.buy_prob:.2f}, alpha={vote.expected_alpha_bps:.0f}bps")

    elif args.all:
        # All tickers
        print("Running committee analysis for all tickers...")
        print("=" * 60)

        def progress(current, total, ticker):
            print(f"[{current:3}/{total}] {ticker}...", flush=True)

        results = coordinator.analyze_universe(progress_callback=progress)

        print("=" * 60)
        print(f"\nCompleted {len(results)} tickers")

        # Summary
        verdicts = {}
        for ticker, decision in results.items():
            v = decision.verdict
            verdicts[v] = verdicts.get(v, 0) + 1

        print("\nVerdict Summary:")
        for verdict, count in sorted(verdicts.items(), key=lambda x: -x[1]):
            print(f"  {verdict}: {count}")

    else:
        print("Usage:")
        print("  python scripts/run_committee.py --ticker AAPL")
        print("  python scripts/run_committee.py --all")


if __name__ == "__main__":
    main()