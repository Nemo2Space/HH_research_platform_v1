#!/usr/bin/env python3
"""
Multi-Factor Alpha Model - Command Line Interface

Usage:
    python alpha_model_cli.py train [--min-date DATE] [--folds N]
    python alpha_model_cli.py predict [--ticker TICKER] [--top N]
    python alpha_model_cli.py factors
    python alpha_model_cli.py report

Location: scripts/alpha_model_cli.py

Author: HH Research Platform
Date: December 2025
"""

import argparse
import sys
import os
from datetime import datetime, date
from pathlib import Path

# Load environment variables FIRST
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Default model path - defined BEFORE imports
MODEL_PATH = 'models/multi_factor_alpha.pkl'

# Import the alpha model (simple imports only)
from src.ml.multi_factor_alpha import MultiFactorAlphaModel


def load_alpha_model(path: str) -> MultiFactorAlphaModel:
    """Load a trained alpha model from disk."""
    model = MultiFactorAlphaModel()
    model.load(path)
    return model


def train_alpha_model(min_date: str = None, save_path: str = MODEL_PATH):
    """Train and save the alpha model."""
    model = MultiFactorAlphaModel()
    report = model.train(min_date=min_date)
    model.save(save_path)
    return report


def cmd_train(args):
    """Train the multi-factor alpha model."""
    print("=" * 70)
    print("MULTI-FACTOR ALPHA MODEL - TRAINING")
    print("=" * 70)
    print(f"Min Date: {args.min_date or 'All available'}")
    print(f"Folds: {args.folds}")
    print(f"Output: {args.output}")
    print("=" * 70)

    model = MultiFactorAlphaModel(
        target_horizons=[5, 10, 20],
        use_regime_models=True,
        use_sector_models=True
    )

    try:
        report = model.train(
            min_date=args.min_date,
            n_folds=args.folds
        )

        model.save(args.output)

        print("\n✅ Training complete!")
        print(f"Model saved to: {args.output}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_predict(args):
    """Generate predictions."""
    print("=" * 70)
    print("MULTI-FACTOR ALPHA MODEL - PREDICTIONS")
    print("=" * 70)

    # Load model
    try:
        model = load_alpha_model(args.model)
        print(f"✅ Model loaded from {args.model}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Run 'python alpha_model_cli.py train' first")
        sys.exit(1)

    if args.ticker:
        # Single ticker prediction
        _predict_single(model, args.ticker)
    else:
        # Batch predictions
        _predict_batch(model, args.top, args.signal)


def _predict_single(model: MultiFactorAlphaModel, ticker: str):
    """Predict for a single ticker."""
    print(f"\n📊 Analyzing {ticker}...")

    df = model.data_loader.load_live_data([ticker])

    if df.empty:
        print(f"❌ No data found for {ticker}")
        return

    row = df.iloc[0]
    factor_values = {f: row.get(f, 50) for f in model.feature_names}

    pred = model.predict(
        ticker=ticker,
        factor_values=factor_values,
        sector=row.get('sector'),
        regime=row.get('regime')
    )

    print("\n" + "=" * 50)
    print(f"  {ticker} - {pred.signal}")
    print("=" * 50)

    print(f"\n📈 Expected Returns:")
    print(
        f"   5-day:  {pred.expected_return_5d * 100:+.2f}% (CI: {pred.ci_lower_5d * 100:+.1f}% to {pred.ci_upper_5d * 100:+.1f}%)")
    print(f"   10-day: {pred.expected_return_10d * 100:+.2f}%")
    print(f"   20-day: {pred.expected_return_20d * 100:+.2f}%")

    print(f"\n📊 Probabilities:")
    print(f"   P(Positive 5d):     {pred.prob_positive_5d * 100:.1f}%")
    print(f"   P(Beat Market 5d):  {pred.prob_beat_market_5d * 100:.1f}%")

    print(f"\n⚖️ Risk Metrics:")
    print(f"   Implied Sharpe:     {pred.sharpe_ratio_implied:.2f}")
    print(f"   Information Ratio:  {pred.information_ratio:.2f}")
    print(f"   Model Uncertainty:  {pred.model_uncertainty:.2f}")

    print(f"\n🎯 Trading Recommendation:")
    print(f"   Signal:           {pred.signal}")
    print(f"   Conviction:       {pred.conviction:.0%}")
    print(f"   Confidence:       {pred.prediction_confidence}")
    print(f"   Position Size:    {pred.recommended_position_size:.2f}x")

    print(f"\n📋 Context:")
    print(f"   Market Regime:    {pred.regime}")
    print(f"   Sector:           {pred.sector}")

    print(f"\n🔍 Top Factors:")
    print("   Bullish:")
    for factor, contrib in pred.top_bullish_factors[:3]:
        print(f"     ✅ {factor}: +{contrib:.3f}")
    print("   Bearish:")
    for factor, contrib in pred.top_bearish_factors[:3]:
        print(f"     ❌ {factor}: {contrib:.3f}")


def _predict_batch(model: MultiFactorAlphaModel, top_n: int, signal_filter: str):
    """Generate batch predictions."""
    print(f"\n📊 Generating predictions...")

    predictions = model.predict_live()

    if predictions.empty:
        print("❌ No predictions available")
        return

    # Apply signal filter
    if signal_filter and signal_filter != 'ALL':
        predictions = predictions[predictions['signal'] == signal_filter]

    # Summary
    print(f"\n📈 Summary:")
    print(f"   Total stocks:     {len(predictions)}")
    print(f"   STRONG_BUY:       {len(predictions[predictions['signal'] == 'STRONG_BUY'])}")
    print(f"   BUY:              {len(predictions[predictions['signal'] == 'BUY'])}")
    print(f"   HOLD:             {len(predictions[predictions['signal'] == 'HOLD'])}")
    print(f"   SELL:             {len(predictions[predictions['signal'] == 'SELL'])}")
    print(f"   STRONG_SELL:      {len(predictions[predictions['signal'] == 'STRONG_SELL'])}")

    # Top buys
    print(f"\n🟢 TOP {top_n} BUY SIGNALS:")
    print("-" * 90)
    print(f"{'Ticker':<8} {'Signal':<12} {'E[R] 5d':<10} {'P(Win)':<10} {'Conviction':<10} {'Sector':<15}")
    print("-" * 90)

    top_buys = predictions[predictions['signal'].isin(['STRONG_BUY', 'BUY'])].head(top_n)

    for _, row in top_buys.iterrows():
        print(f"{row['ticker']:<8} {row['signal']:<12} {row['expected_return_5d'] * 100:+.2f}%     "
              f"{row['prob_positive_5d'] * 100:.1f}%      {row['conviction']:.2f}       {row['sector']:<15}")

    # Top sells
    print(f"\n🔴 TOP {top_n} SELL SIGNALS:")
    print("-" * 90)
    print(f"{'Ticker':<8} {'Signal':<12} {'E[R] 5d':<10} {'P(Win)':<10} {'Conviction':<10} {'Sector':<15}")
    print("-" * 90)

    top_sells = predictions[predictions['signal'].isin(['STRONG_SELL', 'SELL'])].tail(top_n)

    for _, row in top_sells.iterrows():
        print(f"{row['ticker']:<8} {row['signal']:<12} {row['expected_return_5d'] * 100:+.2f}%     "
              f"{row['prob_positive_5d'] * 100:.1f}%      {row['conviction']:.2f}       {row['sector']:<15}")


def cmd_factors(args):
    """Show factor analysis."""
    print("=" * 70)
    print("MULTI-FACTOR ALPHA MODEL - FACTOR ANALYSIS")
    print("=" * 70)

    # Load model
    try:
        model = load_alpha_model(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    factor_df = model.get_factor_report()

    print("\n📊 Factor Importance & Weights:")
    print("-" * 80)
    print(f"{'Factor':<25} {'Category':<12} {'Importance':<12} {'Weight 5d':<12} {'Sign OK':<8}")
    print("-" * 80)

    for _, row in factor_df.iterrows():
        sign_ok = "✅" if row['sign_match'] else "❌"
        print(f"{row['factor']:<25} {row['category']:<12} {row['importance']:.4f}       "
              f"{row['weight_5d']:+.4f}       {sign_ok}")

    print("-" * 80)

    # Category summary
    print("\n📈 Category Summary:")
    category_importance = factor_df.groupby('category')['importance'].sum().sort_values(ascending=False)
    for cat, imp in category_importance.items():
        print(f"   {cat}: {imp:.3f}")


def cmd_report(args):
    """Show model validation report."""
    print("=" * 70)
    print("MULTI-FACTOR ALPHA MODEL - VALIDATION REPORT")
    print("=" * 70)

    # Load model
    try:
        model = load_alpha_model(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    if model.validation_report:
        model.validation_report.print_report()
    else:
        print("❌ No validation report available. Retrain the model.")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Factor Alpha Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train model:       python alpha_model_cli.py train --min-date 2023-01-01
  Predict single:    python alpha_model_cli.py predict --ticker AAPL
  Predict batch:     python alpha_model_cli.py predict --top 20
  Factor analysis:   python alpha_model_cli.py factors
  Validation report: python alpha_model_cli.py report
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--min-date', type=str, default=None,
                              help='Minimum date for training data (YYYY-MM-DD)')
    train_parser.add_argument('--folds', type=int, default=5,
                              help='Number of walk-forward folds')
    train_parser.add_argument('--output', type=str, default=MODEL_PATH,
                              help='Output path for model')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--ticker', type=str, default=None,
                                help='Single ticker to analyze')
    predict_parser.add_argument('--top', type=int, default=10,
                                help='Number of top predictions to show')
    predict_parser.add_argument('--signal', type=str, default='ALL',
                                choices=['ALL', 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'],
                                help='Filter by signal type')
    predict_parser.add_argument('--model', type=str, default=MODEL_PATH,
                                help='Path to trained model')

    # Factors command
    factors_parser = subparsers.add_parser('factors', help='Show factor analysis')
    factors_parser.add_argument('--model', type=str, default=MODEL_PATH,
                                help='Path to trained model')

    # Report command
    report_parser = subparsers.add_parser('report', help='Show validation report')
    report_parser.add_argument('--model', type=str, default=MODEL_PATH,
                               help='Path to trained model')

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'factors':
        cmd_factors(args)
    elif args.command == 'report':
        cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()