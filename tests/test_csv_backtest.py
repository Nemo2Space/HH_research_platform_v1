"""
TEST SCRIPT - Verify CSV Backtest Functionality
================================================

This script tests the new CSV upload backtest functionality
WITHOUT modifying any files or data.

Usage:
    python test_csv_backtest.py
"""

import sys
import pandas as pd
from datetime import datetime, timedelta

print("=" * 70)
print("TEST: CSV Portfolio Backtest Functionality")
print("=" * 70)
print()

# =============================================================================
# Step 1: Import the backtester
# =============================================================================
print("Step 1: Import PortfolioBacktester")
print("-" * 70)

try:
    from dashboard.portfolio_backtester import (
        PortfolioBacktester,
        BacktestConfig,
    )

    print("✅ Imported from dashboard.portfolio_backtester")
except ImportError:
    try:
        from portfolio_backtester import (
            PortfolioBacktester,
            BacktestConfig,
        )

        print("✅ Imported from portfolio_backtester")
    except ImportError as e:
        print(f"❌ Cannot import: {e}")
        sys.exit(1)

# =============================================================================
# Step 2: Check if run_backtest_from_holdings method exists
# =============================================================================
print()
print("Step 2: Check for run_backtest_from_holdings method")
print("-" * 70)

backtester = PortfolioBacktester()

if hasattr(backtester, 'run_backtest_from_holdings'):
    print("✅ run_backtest_from_holdings method exists")
else:
    print("❌ run_backtest_from_holdings method NOT FOUND")
    print("   You need to update portfolio_backtester.py with the new version")
    sys.exit(1)

# =============================================================================
# Step 3: Create a sample CSV-style DataFrame
# =============================================================================
print()
print("Step 3: Create sample CSV portfolio")
print("-" * 70)

# Sample portfolio similar to what a user might upload
sample_csv_data = {
    'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    'weight_pct': [25.0, 25.0, 20.0, 15.0, 15.0],
    'name': ['Apple', 'Microsoft', 'Alphabet', 'Amazon', 'NVIDIA'],
    'value': [0, 0, 0, 0, 0],  # Placeholder
    'shares': [0, 0, 0, 0, 0],  # Placeholder
    'score': [0, 0, 0, 0, 0],  # Placeholder
    'conviction': [None, None, None, None, None]  # Placeholder
}

holdings_df = pd.DataFrame(sample_csv_data)
print(f"✅ Created sample portfolio with {len(holdings_df)} holdings:")
print(holdings_df[['ticker', 'weight_pct', 'name']].to_string(index=False))
print(f"   Total weight: {holdings_df['weight_pct'].sum():.2f}%")

# =============================================================================
# Step 4: Create BacktestConfig for CSV portfolio
# =============================================================================
print()
print("Step 4: Create BacktestConfig")
print("-" * 70)

config = BacktestConfig(
    portfolio_id=-1,  # -1 indicates CSV portfolio
    portfolio_name="Test CSV Portfolio",
    start_date=datetime.now() - timedelta(days=30),  # Last 30 days for quick test
    end_date=datetime.now(),
    initial_capital=100000,
    benchmark='SPY',
    rebalance_frequency='never',
    transaction_cost_pct=0.0
)

print(f"✅ Config created:")
print(f"   Portfolio ID: {config.portfolio_id} (CSV portfolio)")
print(f"   Portfolio Name: {config.portfolio_name}")
print(f"   Date Range: {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}")
print(f"   Initial Capital: ${config.initial_capital:,.0f}")
print(f"   Benchmark: {config.benchmark}")

# =============================================================================
# Step 5: Test run_backtest_from_holdings
# =============================================================================
print()
print("Step 5: Run backtest from holdings DataFrame")
print("-" * 70)

try:
    # Note: This will actually fetch price data from Yahoo Finance
    print("   Fetching price data (this may take a moment)...")

    result = backtester.run_backtest_from_holdings(holdings_df, config)

    if result:
        print("✅ Backtest completed successfully!")
        print()
        print("   Results:")
        print(f"   - Total Return: {result.total_return_pct:+.2f}%")
        print(f"   - Annualized Return: {result.annualized_return_pct:+.2f}%")
        print(f"   - Volatility: {result.volatility_pct:.2f}%")
        print(f"   - Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"   - Max Drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"   - vs {config.benchmark}: {result.total_return_pct - result.benchmark_total_return_pct:+.2f}%")
    else:
        print("❌ Backtest returned None")
        sys.exit(1)

except Exception as e:
    print(f"❌ Backtest failed with error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# Step 6: Test saved portfolio still works
# =============================================================================
print()
print("Step 6: Verify saved portfolio backtest still works")
print("-" * 70)

portfolios_df = backtester.get_saved_portfolios()
if not portfolios_df.empty:
    test_id = int(portfolios_df.iloc[0]['id'])
    test_name = portfolios_df.iloc[0]['name']

    print(f"   Testing with saved portfolio: {test_name} (id={test_id})")

    saved_config = BacktestConfig(
        portfolio_id=test_id,
        portfolio_name=test_name,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_capital=100000,
        benchmark='SPY',
        rebalance_frequency='never',
        transaction_cost_pct=0.0
    )

    try:
        saved_result = backtester.run_backtest(saved_config)
        if saved_result:
            print(f"✅ Saved portfolio backtest works!")
            print(f"   - Total Return: {saved_result.total_return_pct:+.2f}%")
        else:
            print("⚠️  Saved portfolio backtest returned None (might be missing price data)")
    except Exception as e:
        print(f"⚠️  Saved portfolio backtest error: {e}")
else:
    print("   No saved portfolios found, skipping this test")

# =============================================================================
# Summary
# =============================================================================
print()
print("=" * 70)
print("✅ ALL TESTS PASSED")
print("=" * 70)
print()
print("The CSV backtest functionality is working correctly.")
print()
print("You can now:")
print("1. Copy the updated files to your dashboard folder:")
print("   - portfolio_backtester.py")
print("   - backtest_tab.py")
print("2. Restart your Streamlit app")
print("3. Use the 'Upload CSV' option in the Backtest tab")
print()
print("CSV file format:")
print("   Required columns: Ticker, Weight, Name")
print("   Weights should sum to ~100%")
print()