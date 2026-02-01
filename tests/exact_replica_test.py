"""
EXACT REPLICA TEST - Mimics backtest_tab.py line 96-99
=======================================================

This test does EXACTLY what the failing code does.
If this passes, the app should work.

Usage:
    python exact_replica_test.py
"""

import sys

print("=" * 70)
print("EXACT REPLICA TEST - backtest_tab.py line 96-99")
print("=" * 70)
print()

# =============================================================================
# Step 1: Import the ACTUAL backtester class (same as backtest_tab.py line 15-16)
# =============================================================================
print("Step 1: Import PortfolioBacktester (same as backtest_tab.py)")
print("-" * 70)

try:
    from dashboard.portfolio_backtester import PortfolioBacktester

    print("✅ Imported from dashboard.portfolio_backtester")
except ImportError:
    try:
        from portfolio_backtester import PortfolioBacktester

        print("✅ Imported from portfolio_backtester")
    except ImportError as e:
        print(f"❌ Cannot import PortfolioBacktester: {e}")
        sys.exit(1)

# =============================================================================
# Step 2: Create backtester instance (same as backtest_tab.py line 44)
# =============================================================================
print()
print("Step 2: Create backtester instance")
print("-" * 70)

backtester = PortfolioBacktester()
print("✅ backtester = PortfolioBacktester()")

# =============================================================================
# Step 3: Get saved portfolios (same as backtest_tab.py line 58)
# =============================================================================
print()
print("Step 3: Get saved portfolios")
print("-" * 70)

portfolios_df = backtester.get_saved_portfolios()
print(f"✅ portfolios_df = backtester.get_saved_portfolios()")
print(f"   Found {len(portfolios_df)} portfolios")

if portfolios_df.empty:
    print("❌ No portfolios found!")
    sys.exit(1)

# Get first portfolio ID
selected_id = int(portfolios_df.iloc[0]['id'])
portfolio_name = portfolios_df.iloc[0]['name']
print(f"   Using: {portfolio_name} (id={selected_id})")

# =============================================================================
# Step 4: Load portfolio holdings (EXACT LINE 96)
# =============================================================================
print()
print("Step 4: Load portfolio holdings (backtest_tab.py LINE 96)")
print("-" * 70)

holdings_df = backtester.load_portfolio_holdings(selected_id)
print(f"✅ holdings_df = backtester.load_portfolio_holdings({selected_id})")
print(f"   DataFrame shape: {holdings_df.shape}")
print(f"   Columns: {list(holdings_df.columns)}")

if holdings_df.empty:
    print("❌ holdings_df is empty!")
    sys.exit(1)

# =============================================================================
# Step 5: The EXACT failing line (LINE 99)
# =============================================================================
print()
print("Step 5: The EXACT failing operation (backtest_tab.py LINE 99)")
print("-" * 70)

print("Executing: holdings_df[['ticker', 'weight_pct', 'value', 'score', 'conviction']]")
print()

try:
    display_df = holdings_df[['ticker', 'weight_pct', 'value', 'score', 'conviction']]
    print("✅ SUCCESS! Column selection worked!")
    print()
    print("First 10 rows:")
    print(display_df.head(10).to_string(index=False))

except KeyError as e:
    print(f"❌ FAILED with KeyError: {e}")
    print()
    print("DataFrame has these columns:")
    for col in holdings_df.columns:
        print(f"   - {col}")
    print()
    print("But line 99 needs: ['ticker', 'weight_pct', 'value', 'score', 'conviction']")
    print()
    missing = [c for c in ['ticker', 'weight_pct', 'value', 'score', 'conviction'] if c not in holdings_df.columns]
    print(f"MISSING COLUMNS: {missing}")
    sys.exit(1)

# =============================================================================
# Summary
# =============================================================================
print()
print("=" * 70)
print("✅ ALL TESTS PASSED")
print("=" * 70)
print()
print("If this test passes but the app still fails, try:")
print("1. Delete __pycache__ folders: ")
print("   rd /s /q dashboard\\__pycache__")
print("   rd /s /q __pycache__")
print("2. Restart Streamlit completely (close terminal, open new one)")
print("3. Run: streamlit run dashboard/app.py")
print()