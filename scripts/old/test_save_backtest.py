"""Test saving backtest results."""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.backtest.engine import BacktestEngine

def main():
    print("Testing backtest save...")

    engine = BacktestEngine()

    result = engine.run_backtest(
        strategy='signal_based',
        holding_period=5,
        params={'buy_signals': ['BUY', 'STRONG_BUY']}
    )

    print(f"Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Sharpe: {result.sharpe_ratio:.2f}")

    print("\nSaving to database...")
    saved = engine.save_results_for_learning(result)
    print(f"Saved: {saved}")

if __name__ == "__main__":
    main()