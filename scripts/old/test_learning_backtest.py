"""Test AI learning with backtest insights."""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv()

from src.ai.learning import SignalLearner


def main():
    print("=" * 60)
    print("AI LEARNING + BACKTEST TEST")
    print("=" * 60)

    learner = SignalLearner()

    # Test 1: Get backtest insights
    print("\n1. Backtest Insights:")
    insights = learner.get_backtest_insights()
    print(f"   Has data: {insights.get('has_data')}")

    if insights.get('has_data'):
        print(f"   Total backtests: {insights.get('total_backtests')}")
        if insights.get('best_strategy'):
            best = insights['best_strategy']
            print(f"   Best strategy: {best['strategy']}")
            print(f"   Best Sharpe: {best['sharpe']:.2f}")
            print(f"   Best Win Rate: {best['win_rate']:.1%}")
        print(f"   Recommendation: {insights.get('recommendation')}")
    else:
        print(f"   Message: {insights.get('message')}")

    # Test 2: Adjust signal confidence (now includes backtest data)
    print("\n2. Signal Confidence Adjustment (with backtest):")
    for signal in ['BUY', 'STRONG_BUY', 'WEAK_BUY']:
        adj = learner.adjust_signal_confidence(signal, 'Technology')
        print(f"   {signal} -> multiplier={adj.confidence_multiplier}, reason={adj.reason[:80]}...")

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()