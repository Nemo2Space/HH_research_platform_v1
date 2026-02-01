"""
Test Committee Analysis

Usage:
    python scripts/test_committee.py
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.committee.coordinator import CommitteeCoordinator


def main():
    print("=" * 60)
    print("Alpha Platform - Committee Analysis Test")
    print("=" * 60)
    print()

    coordinator = CommitteeCoordinator()

    # Test with a few tickers
    test_tickers = ['WMT']

    print(f"{'Ticker':<8} {'Verdict':<12} {'Alpha':>8} {'Conv':>6} {'Rationale'}")
    print("-" * 80)

    for ticker in test_tickers:
        decision = coordinator.analyze_ticker(ticker)

        rationale_short = decision.rationale[:40] + "..." if len(decision.rationale) > 40 else decision.rationale

        print(f"{ticker:<8} {decision.verdict:<12} {decision.expected_alpha_bps:>7.0f}bp {decision.conviction:>5}% {rationale_short}")

    print()

    # Show detailed for AAPL
    print("Detailed Analysis for AAPL:")
    print("-" * 40)
    decision = coordinator.analyze_ticker('AAPL')

    print(f"Verdict: {decision.verdict}")
    print(f"Conviction: {decision.conviction}%")
    print(f"Expected Alpha: {decision.expected_alpha_bps:.0f} bps")
    print(f"Horizon: {decision.horizon_days} days")
    print()
    print("Agent Votes:")
    for vote in decision.votes:
        print(f"  {vote.role}: buy_prob={vote.buy_prob:.2f}, alpha={vote.expected_alpha_bps:.0f}bps, conf={vote.confidence:.2f}")
        print(f"    → {vote.rationale[:60]}...")
    print()
    print(f"Risks: {', '.join(decision.risks)}")

    print()
    print("=" * 60)
    print("SUCCESS - Committee analysis test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()