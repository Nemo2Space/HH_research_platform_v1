"""
FINAL VERIFICATION TEST
========================

This simulates the COMPLETE flow from selecting a portfolio to running backtest.
Run this to be 100% sure the fix will work.

Usage:
    python final_verification_test.py
"""

import sys
import pandas as pd

print("=" * 70)
print("FINAL VERIFICATION - Complete Backtest Flow Simulation")
print("=" * 70)
print()

# =============================================================================
# Step 1: Connect to database
# =============================================================================
print("Step 1: Database connection")
print("-" * 70)

try:
    from src.db.connection import get_connection
    conn = get_connection()
    if hasattr(conn, '__enter__'):
        conn = conn.__enter__()
    print("✅ Connected")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# =============================================================================
# Step 2: List saved portfolios (like the dropdown does)
# =============================================================================
print()
print("Step 2: List saved portfolios (simulates dropdown)")
print("-" * 70)

try:
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, description, created_at, num_holdings, total_value
        FROM saved_portfolios
        ORDER BY created_at DESC
    """)
    portfolios = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()

    portfolios_df = pd.DataFrame(portfolios, columns=columns)
    print(f"✅ Found {len(portfolios_df)} portfolios")
    print(portfolios_df[['id', 'name', 'num_holdings']].to_string(index=False))

    if portfolios_df.empty:
        print("❌ No portfolios to test with!")
        conn.close()
        sys.exit(1)

    test_portfolio_id = int(portfolios_df.iloc[0]['id'])
    test_portfolio_name = portfolios_df.iloc[0]['name']
    print(f"\nUsing portfolio: {test_portfolio_name} (id={test_portfolio_id})")

except Exception as e:
    print(f"❌ Failed: {e}")
    conn.close()
    sys.exit(1)

# =============================================================================
# Step 3: Load portfolio holdings (THE FIXED QUERY)
# =============================================================================
print()
print("Step 3: Load portfolio holdings (FIXED query from portfolio_builder.py)")
print("-" * 70)

# This is the FIXED query - what portfolio_builder.py line 385 WILL look like
fixed_holdings_query = """
    SELECT ticker, weight_pct, shares, value, sector, score, signal_type, rationale, conviction
    FROM saved_portfolio_holdings
    WHERE portfolio_id = %s
    ORDER BY weight_pct DESC
"""

try:
    cur = conn.cursor()
    cur.execute(fixed_holdings_query, (test_portfolio_id,))
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()

    holdings_df = pd.DataFrame(rows, columns=columns)
    print(f"✅ Loaded {len(holdings_df)} holdings")
    print(f"   Columns: {list(holdings_df.columns)}")

except Exception as e:
    print(f"❌ Failed: {e}")
    conn.close()
    sys.exit(1)

# =============================================================================
# Step 4: Simulate backtest_tab.py displaying holdings (line 99)
# =============================================================================
print()
print("Step 4: Simulate backtest_tab.py line 99 (display holdings table)")
print("-" * 70)

try:
    display_columns = ['ticker', 'weight_pct', 'value', 'score', 'conviction']
    display_df = holdings_df[display_columns]
    print(f"✅ Column selection successful")
    print(f"\nFirst 10 holdings:")
    print(display_df.head(10).to_string(index=False))

except KeyError as e:
    print(f"❌ KeyError: {e}")
    print("   This is the error you were seeing!")
    conn.close()
    sys.exit(1)

# =============================================================================
# Step 5: Simulate portfolio_backtester.py load_portfolio_holdings()
# =============================================================================
print()
print("Step 5: Simulate portfolio_backtester.py load_portfolio_holdings()")
print("-" * 70)

backtester_query = """
    SELECT 
        ticker,
        weight_pct,
        value,
        shares,
        score,
        conviction
    FROM saved_portfolio_holdings
    WHERE portfolio_id = %s
    ORDER BY weight_pct DESC
"""

try:
    cur = conn.cursor()
    cur.execute(backtester_query, (test_portfolio_id,))
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()

    backtester_df = pd.DataFrame(rows, columns=columns)
    print(f"✅ Loaded {len(backtester_df)} holdings for backtesting")
    print(f"   Columns: {list(backtester_df.columns)}")

except Exception as e:
    print(f"❌ Failed: {e}")
    conn.close()
    sys.exit(1)

# =============================================================================
# Step 6: Verify backtester can use the data
# =============================================================================
print()
print("Step 6: Verify backtester can calculate weights")
print("-" * 70)

try:
    # This is what the backtester does with the data
    tickers = backtester_df['ticker'].tolist()
    weights = backtester_df.set_index('ticker')['weight_pct'] / 100

    print(f"✅ Extracted {len(tickers)} tickers")
    print(f"✅ Total weight: {weights.sum()*100:.2f}%")
    print(f"\nTop 5 by weight:")
    print(weights.head().to_string())

except Exception as e:
    print(f"❌ Failed: {e}")
    conn.close()
    sys.exit(1)

# =============================================================================
# Summary
# =============================================================================
print()
print("=" * 70)
print("✅ ALL TESTS PASSED - THE FIX IS VERIFIED")
print("=" * 70)
print()
print("You can now safely apply the fix to portfolio_builder.py line 385:")
print()
print("CHANGE:")
print("  SELECT ticker, weight_pct, shares, value, sector, score, signal_type, rationale")
print()
print("TO:")
print("  SELECT ticker, weight_pct, shares, value, sector, score, signal_type, rationale, conviction")
print()
print("Then restart your Streamlit app.")
print()

conn.close()
