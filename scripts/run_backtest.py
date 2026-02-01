"""
Run Backtest from Command Line

Usage:
    # Run with default settings
    python scripts/run_backtest.py

    # Run specific strategy
    python scripts/run_backtest.py --strategy signal_based

    # Run with custom parameters
    python scripts/run_backtest.py --strategy score_threshold --holding-period 20 --buy-threshold 65

    # Run optimization
    python scripts/run_backtest.py --optimize --strategy sentiment_only

    # Compare all strategies
    python scripts/run_backtest.py --compare-all
"""

import sys
import os
import argparse
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv()

from src.backtest.engine import BacktestEngine
from src.backtest.strategies import (
    STRATEGIES, get_strategy, get_all_strategies,
    HOLDING_PERIODS, BENCHMARKS, OPTIMIZATION_GRIDS
)
from src.backtest.metrics import format_metrics_report, compare_strategies, rank_strategies


def run_single_backtest(args):
    """Run a single backtest."""
    engine = BacktestEngine()

    # Build params from args
    params = {}

    if args.buy_signals:
        params['buy_signals'] = args.buy_signals
    if args.sell_signals:
        params['sell_signals'] = args.sell_signals
    if args.buy_threshold:
        params['buy_threshold'] = args.buy_threshold
    if args.sell_threshold:
        params['sell_threshold'] = args.sell_threshold
    if args.min_sentiment:
        params['min_sentiment'] = args.min_sentiment
    if args.min_fundamental:
        params['min_fundamental'] = args.min_fundamental
    if args.require_all:
        params['require_all'] = args.require_all

    # Get strategy defaults if not overridden
    if args.strategy in STRATEGIES and not params:
        strategy_config = get_strategy(args.strategy)
        params = strategy_config.default_params
        strategy_type = strategy_config.strategy_type
    else:
        strategy_type = args.strategy

    print("=" * 60)
    print("RUNNING BACKTEST")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Holding Period: {args.holding_period} days")
    print(f"Benchmark: {args.benchmark}")
    print(f"Parameters: {json.dumps(params, indent=2)}")
    print()

    result = engine.run_backtest(
        strategy=strategy_type,
        start_date=args.start_date,
        end_date=args.end_date,
        holding_period=args.holding_period,
        benchmark=args.benchmark,
        params=params
    )

    # Print results
    print(f"\n📊 RESULTS: {result.strategy_name}")
    print(f"   Period: {result.start_date} to {result.end_date}")
    print(f"   Trades: {result.total_trades}")
    print(f"   Win Rate: {result.win_rate:.1%}")
    print(f"   Avg Return: {result.avg_return:+.2f}%")
    print(f"   Total Return: {result.total_return:+.2f}%")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {result.max_drawdown:.2f}%")
    print(f"   Benchmark: {result.benchmark_return:+.2f}%")
    print(f"   Alpha: {result.alpha:+.2f}%")

    # Returns by signal
    if result.returns_by_signal:
        print("\n📈 BY SIGNAL:")
        for sig, stats in result.returns_by_signal.items():
            print(f"   {sig}: {stats['avg_return']:+.2f}% avg, "
                  f"{stats['win_rate']:.0%} win, n={stats['count']}")

    # Save results if requested
    if args.save:
        engine.save_results_for_learning(result)
        print("\n✅ Results saved for AI learning")

    return result


def run_optimization(args):
    """Run parameter optimization."""
    engine = BacktestEngine()

    strategy_type = args.strategy
    if args.strategy in STRATEGIES:
        strategy_config = get_strategy(args.strategy)
        strategy_type = strategy_config.strategy_type

    # Get parameter grid
    param_grid = OPTIMIZATION_GRIDS.get(strategy_type, {})

    if not param_grid:
        print(f"No optimization grid defined for {strategy_type}")
        return

    print("=" * 60)
    print("PARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Optimizing for: {args.optimize_metric}")
    print(f"Parameter grid: {param_grid}")
    print()

    result = engine.optimize_parameters(
        strategy=strategy_type,
        param_grid=param_grid,
        holding_period=args.holding_period,
        metric=args.optimize_metric,
        start_date=args.start_date,
        end_date=args.end_date
    )

    print(f"\n🏆 BEST PARAMETERS:")
    print(f"   {json.dumps(result['best_params'], indent=4)}")
    print(f"   Score ({args.optimize_metric}): {result['best_score']:.4f}")

    if result['best_result']:
        r = result['best_result']
        print(f"\n📊 BEST RESULT:")
        print(f"   Trades: {r.total_trades}")
        print(f"   Win Rate: {r.win_rate:.1%}")
        print(f"   Avg Return: {r.avg_return:+.2f}%")
        print(f"   Sharpe: {r.sharpe_ratio:.2f}")

    print(f"\n📋 TOP 10 COMBINATIONS:")
    for i, combo in enumerate(result['all_results'][:10], 1):
        print(f"   {i}. {combo['params']} → {combo['score']:.4f} "
              f"(n={combo['trades']}, win={combo['win_rate']:.0%})")


def run_compare_all(args):
    """Compare all predefined strategies."""
    engine = BacktestEngine()

    print("=" * 60)
    print("COMPARING ALL STRATEGIES")
    print("=" * 60)
    print(f"Holding Period: {args.holding_period} days")
    print(f"Benchmark: {args.benchmark}")
    print()

    results = []

    for name, config in STRATEGIES.items():
        print(f"Running {name}...", end=" ", flush=True)

        try:
            result = engine.run_backtest(
                strategy=config.strategy_type,
                holding_period=args.holding_period,
                benchmark=args.benchmark,
                params=config.default_params
            )

            results.append({
                'strategy_name': name,
                'display_name': config.display_name,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'avg_return': result.avg_return,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'alpha': result.alpha,
            })

            print(f"OK ({result.total_trades} trades)")

        except Exception as e:
            print(f"FAILED: {e}")

    # Rank strategies
    ranked = rank_strategies(results)

    print("\n" + "=" * 60)
    print("STRATEGY RANKINGS")
    print("=" * 60)

    for r in ranked:
        print(f"\n#{r['rank']} {r['display_name']} (score: {r['composite_score']})")
        print(f"    Trades: {r['total_trades']}, Win Rate: {r['win_rate']:.1%}")
        print(f"    Return: {r['avg_return']:+.2f}% avg, {r['total_return']:+.2f}% total")
        print(f"    Sharpe: {r['sharpe_ratio']:.2f}, Alpha: {r['alpha']:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Run backtests")

    # Strategy selection
    parser.add_argument("--strategy", "-s", default="signal_based",
                        choices=list(STRATEGIES.keys()) + ['signal_based', 'score_threshold',
                                                           'composite', 'sentiment_only', 'fundamental_only'],
                        help="Strategy to test")

    # Date range
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")

    # Holding period
    parser.add_argument("--holding-period", "-hp", type=int, default=10,
                        choices=[1, 5, 10, 20],
                        help="Holding period in days")

    # Benchmark
    parser.add_argument("--benchmark", "-b", default="SPY",
                        help="Benchmark ticker")

    # Strategy parameters
    parser.add_argument("--buy-signals", nargs='+',
                        help="Signals to buy on (for signal_based)")
    parser.add_argument("--sell-signals", nargs='+',
                        help="Signals to short on (for signal_based)")
    parser.add_argument("--buy-threshold", type=int,
                        help="Buy threshold (for score_threshold)")
    parser.add_argument("--sell-threshold", type=int,
                        help="Sell threshold (for score_threshold)")
    parser.add_argument("--min-sentiment", type=int,
                        help="Min sentiment (for composite)")
    parser.add_argument("--min-fundamental", type=int,
                        help="Min fundamental (for composite)")
    parser.add_argument("--require-all", action="store_true",
                        help="Require all conditions (for composite)")

    # Modes
    parser.add_argument("--optimize", action="store_true",
                        help="Run parameter optimization")
    parser.add_argument("--optimize-metric", default="sharpe_ratio",
                        choices=['sharpe_ratio', 'win_rate', 'avg_return', 'total_return'],
                        help="Metric to optimize")
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all predefined strategies")

    # Output
    parser.add_argument("--save", action="store_true",
                        help="Save results for AI learning")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    if args.compare_all:
        run_compare_all(args)
    elif args.optimize:
        run_optimization(args)
    else:
        run_single_backtest(args)


if __name__ == "__main__":
    main()