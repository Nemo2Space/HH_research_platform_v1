"""Quick test of backtest module."""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.backtest.engine import BacktestEngine
from src.backtest.strategies import STRATEGIES

def main():
    print("=" * 60)
    print("BACKTEST MODULE TEST")
    print("=" * 60)

    # Test 1: Check strategies loaded
    print(f"\n1. Strategies available: {list(STRATEGIES.keys())}")

    # Test 2: Initialize engine
    print("\n2. Initializing BacktestEngine...")
    engine = BacktestEngine()
    print("   OK")

    # Test 3: Load historical data
    print("\n3. Loading historical scores...")
    df = engine.load_historical_scores()
    print(f"   Loaded {len(df)} records")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Signals: {df['signal_type'].value_counts().to_dict()}")

    # Test 4: Run a simple backtest
    print("\n4. Running signal_based backtest (10-day hold)...")
    result = engine.run_backtest(
        strategy='signal_based',
        holding_period=10,
        benchmark='SPY',
        params={'buy_signals': ['STRONG_BUY', 'BUY']}
    )

    print(f"   Trades: {result.total_trades}")
    print(f"   Win Rate: {result.win_rate:.1%}")
    print(f"   Avg Return: {result.avg_return:+.2f}%")
    print(f"   Total Return: {result.total_return:+.2f}%")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Benchmark (SPY): {result.benchmark_return:+.2f}%")
    print(f"   Alpha: {result.alpha:+.2f}%")

    # Test 5: Show returns by signal
    print("\n5. Returns by Signal Type:")
    for sig, stats in result.returns_by_signal.items():
        print(f"   {sig}: {stats['avg_return']:+.2f}% avg, {stats['win_rate']:.0%} win, n={stats['count']}")

    print("\n" + "=" * 60)
    print("SUCCESS - Backtest module working!")
    print("=" * 60)

if __name__ == "__main__":
    main()