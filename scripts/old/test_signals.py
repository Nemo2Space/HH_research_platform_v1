"""
Test Signal Generation

Tests signal generation with sample scores.
Does NOT require LLM to be running.

Usage:
    python scripts/test_signals.py
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.screener.signals import SignalGenerator, calculate_composite_score, calculate_likelihood_score


def main():
    print("=" * 50)
    print("Alpha Platform - Signal Generation Test")
    print("=" * 50)
    print()

    generator = SignalGenerator()

    # Test cases with different score profiles
    test_cases = [
        {
            'ticker': 'STRONG_BUY_TEST',
            'sentiment_score': 75,
            'fundamental_score': 80,
            'growth_score': 70,
            'dividend_score': 50,
            'gap_score': 65,
            'likelihood_score': 80,
            'analyst_positivity': 85,
            'total_score': 75,
        },
        {
            'ticker': 'BUY_TEST',
            'sentiment_score': 60,
            'fundamental_score': 65,
            'growth_score': 60,
            'dividend_score': 40,
            'gap_score': 55,
            'likelihood_score': 65,
            'analyst_positivity': 70,
            'total_score': 60,
        },
        {
            'ticker': 'NEUTRAL_TEST',
            'sentiment_score': 50,
            'fundamental_score': 50,
            'growth_score': 50,
            'dividend_score': 50,
            'gap_score': 50,
            'likelihood_score': 50,
            'analyst_positivity': 50,
            'total_score': 50,
        },
        {
            'ticker': 'SELL_TEST',
            'sentiment_score': 35,
            'fundamental_score': 35,
            'growth_score': 40,
            'dividend_score': 30,
            'gap_score': 40,
            'likelihood_score': 35,
            'analyst_positivity': 30,
            'total_score': 35,
        },
        {
            'ticker': 'INCOME_BUY_TEST',
            'sentiment_score': 55,
            'fundamental_score': 70,
            'growth_score': 45,
            'dividend_score': 85,
            'gap_score': 50,
            'likelihood_score': 55,
            'analyst_positivity': 60,
            'total_score': 55,
        },
        {
            'ticker': 'GROWTH_BUY_TEST',
            'sentiment_score': 65,
            'fundamental_score': 60,
            'growth_score': 85,
            'dividend_score': 20,
            'gap_score': 55,
            'likelihood_score': 60,
            'analyst_positivity': 65,
            'total_score': 60,
        },
    ]

    print("Testing signal generation:")
    print("-" * 70)
    print(f"{'Ticker':<20} {'Signal':<15} {'Strength':>8} {'Color':<10} {'Reason'}")
    print("-" * 70)

    for scores in test_cases:
        # Calculate additional scores
        scores['likelihood_score'] = calculate_likelihood_score(scores)
        scores['composite_score'] = calculate_composite_score(scores)

        signal = generator.generate_signal(scores)

        print(f"{scores['ticker']:<20} {signal.type:<15} {signal.strength:>8} {signal.color:<10} {signal.reasons[0][:30]}")

    print()
    print("=" * 50)
    print("SUCCESS - Signal generation test complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()