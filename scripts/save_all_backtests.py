"""Save all strategy backtests for AI learning."""

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
    print("SAVING ALL BACKTESTS FOR AI LEARNING")
    print("=" * 60)

    engine = BacktestEngine()
    saved_count = 0

    # Test each holding period
    for holding_period in [5, 10, 20]:
        print(f"\n--- Holding Period: {holding_period} days ---")

        for name, config in STRATEGIES.items():
            print(f"  {name}...", end=" ", flush=True)

            try:
                result = engine.run_backtest(
                    strategy=config.strategy_type,
                    holding_period=holding_period,
                    benchmark='SPY',
                    params=config.default_params
                )

                if result.total_trades >= 3:
                    saved = engine.save_results_for_learning(result)
                    if saved:
                        saved_count += 1
                        print(f"OK (trades={result.total_trades}, sharpe={result.sharpe_ratio:.2f})")
                    else:
                        print("SAVE FAILED")
                else:
                    print(f"SKIP (only {result.total_trades} trades)")

            except Exception as e:
                print(f"ERROR: {e}")

    print(f"\n{'=' * 60}")
    print(f"Saved {saved_count} backtest results")
    print("=" * 60)


if __name__ == "__main__":
    main()