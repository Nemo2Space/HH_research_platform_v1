"""
Test Technical Analysis

Usage:
    python scripts/test_technicals.py
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.screener.technicals import TechnicalAnalyzer


def main():
    print("=" * 60)
    print("Alpha Platform - Technical Analysis Test")
    print("=" * 60)
    print()

    analyzer = TechnicalAnalyzer()

    # Test with a few tickers
    test_tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD']

    print(f"{'Ticker':<8} {'Score':>6} {'RSI':>6} {'MACD':>10} {'Trend':>10} {'Mom 5d':>8}")
    print("-" * 60)

    for ticker in test_tickers:
        result = analyzer.analyze_ticker(ticker)
        print(f"{ticker:<8} {result['technical_score']:>6} {result['rsi']:>6.1f} "
              f"{result['macd_signal']:>10} {result['trend']:>10} {result['momentum_5d']:>7.1f}%")

    print()

    # Show detailed analysis for AAPL
    print("Detailed Analysis for AAPL:")
    print("-" * 40)
    aapl = analyzer.analyze_ticker('AAPL')
    for key, value in aapl.items():
        print(f"  {key}: {value}")

    print()
    print("=" * 60)
    print("SUCCESS - Technical analysis test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()