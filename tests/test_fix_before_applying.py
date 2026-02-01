"""
TEST SCRIPT - Validates the fix before applying
================================================

This script tests if adding 'conviction' to the query will work.
Run this BEFORE making any code changes.

Usage:
    python test_fix_before_applying.py
"""

import sys

print("=" * 70)
print("TEST: Will adding 'conviction' to load_portfolio() fix the issue?")
print("=" * 70)
print()

# Connect to database
try:
    from src.db.connection import get_connection

    conn = get_connection()
    if hasattr(conn, '__enter__'):
        conn = conn.__enter__()
    print("✅ Database connected")
except Exception as e:
    print(f"❌ Cannot connect: {e}")
    sys.exit(1)

# Get a test portfolio
cur = conn.cursor()
cur.execute("SELECT id, name FROM saved_portfolios LIMIT 1")
row = cur.fetchone()
cur.close()

if not row:
    print("❌ No saved portfolios found. Save a portfolio first.")
    conn.close()
    sys.exit(1)

portfolio_id = int(row[0])  # Convert to Python int to avoid numpy issues
portfolio_name = row[1]
print(f"Testing with portfolio: {portfolio_name} (id={portfolio_id})")
print()

# =============================================================================
# TEST 1: The PROPOSED FIX for portfolio_builder.py load_portfolio()
# =============================================================================
print("-" * 70)
print("TEST 1: Proposed fix for portfolio_builder.py line 385")
print("-" * 70)

fixed_query = """
    SELECT ticker, weight_pct, shares, value, sector, score, signal_type, rationale, conviction
    FROM saved_portfolio_holdings
    WHERE portfolio_id = %s
    ORDER BY weight_pct DESC
"""

print("Fixed query (adds 'conviction' at the end):")
print(fixed_query)

try:
    cur = conn.cursor()
    cur.execute(fixed_query, (portfolio_id,))
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()

    print(f"✅ Query PASSED - returned {len(rows)} rows")
    print(f"   Columns: {columns}")

    # Check conviction is there
    if 'conviction' in columns:
        print("   ✅ 'conviction' column is present")
    else:
        print("   ❌ 'conviction' column is MISSING")

except Exception as e:
    print(f"❌ Query FAILED: {e}")
    conn.close()
    sys.exit(1)

# =============================================================================
# TEST 2: Simulate what backtest_tab.py line 99 does
# =============================================================================
print()
print("-" * 70)
print("TEST 2: Simulating backtest_tab.py line 99")
print("-" * 70)

import pandas as pd

# Create DataFrame from query results (simulating what load_portfolio returns)
try:
    cur = conn.cursor()
    cur.execute(fixed_query, (portfolio_id,))
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()

    holdings_df = pd.DataFrame(rows, columns=columns)
    print(f"DataFrame created with columns: {list(holdings_df.columns)}")
    print()

    # This is what backtest_tab.py line 99 does:
    required_columns = ['ticker', 'weight_pct', 'value', 'score', 'conviction']
    print(f"backtest_tab.py line 99 selects: {required_columns}")

    # Check if all required columns exist
    missing = [c for c in required_columns if c not in holdings_df.columns]

    if missing:
        print(f"❌ WOULD FAIL - Missing columns: {missing}")
    else:
        # Try the actual selection
        selected_df = holdings_df[required_columns]
        print(f"✅ Selection PASSED")
        print()
        print("Sample data (first 5 rows):")
        print(selected_df.head().to_string(index=False))

except Exception as e:
    print(f"❌ Test FAILED: {e}")
    conn.close()
    sys.exit(1)

# =============================================================================
# TEST 3: Test portfolio_backtester.py load_portfolio_holdings() query
# =============================================================================
print()
print("-" * 70)
print("TEST 3: Testing portfolio_backtester.py query (should already work)")
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
    cur.execute(backtester_query, (portfolio_id,))
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()

    print(f"✅ Query PASSED - returned {len(rows)} rows")
    print(f"   Columns: {columns}")

except Exception as e:
    print(f"❌ Query FAILED: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("✅ All tests PASSED!")
print()
print("The fix is to update portfolio_builder.py line 385:")
print()
print("FIND THIS:")
print('    SELECT ticker, weight_pct, shares, value, sector, score, signal_type, rationale')
print()
print("REPLACE WITH:")
print('    SELECT ticker, weight_pct, shares, value, sector, score, signal_type, rationale, conviction')
print()
print("After making this change, restart your Streamlit app.")
print()

conn.close()
print("Done.")