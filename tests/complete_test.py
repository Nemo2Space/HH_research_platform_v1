# EXHAUSTIVE_TEST.py - Test EVERY column reference in get_top_stocks_by_category

from dashboard.portfolio_builder import get_latest_stock_universe
import pandas as pd

print("=" * 80)
print("EXHAUSTIVE TEST - EVERY FUNCTION IN get_top_stocks_by_category")
print("=" * 80)

df = get_latest_stock_universe(force_refresh=True)
print(f"\n✓ Loaded {len(df)} stocks, {len(df.columns)} columns")

print("\n" + "=" * 80)
print("ALL COLUMNS IN DATAFRAME")
print("=" * 80)
for col in sorted(df.columns):
    has_data = df[col].notna().sum()
    pct = (has_data / len(df) * 100)
    print(f"  • {col}: {has_data}/{len(df)} ({pct:.0f}%)")

errors = []
n = 10  # Test value for nlargest

# ============================================================================
# TEST 1: best_overall
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: best_overall (line ~1080)")
print("=" * 80)

test_cols = ['ticker', 'sector', 'total_score', 'trading_signal_type', 'market_cap']
print("Required columns:", test_cols)

for col in test_cols:
    if col not in df.columns:
        print(f"  ✗ {col}: MISSING")
        errors.append(f"best_overall: missing '{col}'")
    else:
        print(f"  ✓ {col}: exists")

try:
    result = df.nlargest(n, 'total_score')[test_cols].copy()
    print(f"\n✓ best_overall WORKS - {len(result)} stocks")
except Exception as e:
    print(f"\n✗ best_overall FAILS: {e}")
    errors.append(f"best_overall: {e}")

# ============================================================================
# TEST 2: best_growth
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: best_growth (line ~1093)")
print("=" * 80)

test_cols = ['ticker', 'sector', 'growth_score', 'revenue_growth']
print("Required columns:", test_cols)

for col in test_cols:
    if col not in df.columns:
        print(f"  ✗ {col}: MISSING")
        errors.append(f"best_growth: missing '{col}'")
    else:
        print(f"  ✓ {col}: exists")

try:
    growth_df = df[df['growth_score'] > 70]
    result = growth_df.nlargest(n, 'growth_score')[test_cols].copy()
    print(f"\n✓ best_growth WORKS - {len(result)} stocks")
except Exception as e:
    print(f"\n✗ best_growth FAILS: {e}")
    errors.append(f"best_growth: {e}")

# ============================================================================
# TEST 3: best_value
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: best_value (line ~1096)")
print("=" * 80)

test_cols = ['ticker', 'sector', 'pe_ratio', 'pb_ratio', 'dividend_yield']
print("Required columns:", test_cols)

for col in test_cols:
    if col not in df.columns:
        print(f"  ✗ {col}: MISSING")
        errors.append(f"best_value: missing '{col}'")
    else:
        print(f"  ✓ {col}: exists")

try:
    value_df = df[(df['pe_ratio'] > 0) & (df['pe_ratio'] < 20)]
    result = value_df.nlargest(n, 'total_score')[test_cols].copy()
    print(f"\n✓ best_value WORKS - {len(result)} stocks")
except Exception as e:
    print(f"\n✗ best_value FAILS: {e}")
    errors.append(f"best_value: {e}")

# ============================================================================
# TEST 4: best_dividend
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: best_dividend (line ~1099)")
print("=" * 80)

# First find what columns are actually used
print("\nSearching for best_dividend column list in code...")
with open('../dashboard/portfolio_builder.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if 'best_dividend' in line and 'nlargest' in line:
            # Get next line which has column list
            if i < len(lines):
                next_line = lines[i].strip()
                print(f"Line {i + 1}: {next_line}")

# Test what's actually in the code
test_cols_possibilities = [
    ['ticker', 'sector', 'dividend_yield', 'dividend_score'],
    ['ticker', 'sector', 'dividend_yield', 'peg_ratio'],
    ['ticker', 'sector', 'dividend_yield', 'dividend_score', 'peg_ratio']
]

print("\nTesting possible column combinations:")
for test_cols in test_cols_possibilities:
    print(f"\n  Testing: {test_cols}")
    missing = [col for col in test_cols if col not in df.columns]
    if missing:
        print(f"    ✗ Missing: {missing}")
    else:
        print(f"    ✓ All columns exist")
        try:
            div_df = df[df['dividend_yield'] > 2]
            result = div_df.nlargest(n, 'dividend_yield')[test_cols].copy()
            print(f"    ✓ WORKS with {len(result)} stocks")
        except Exception as e:
            print(f"    ✗ FAILS: {e}")

# ============================================================================
# TEST 5: strong_signals
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: strong_signals (line ~1102+)")
print("=" * 80)

test_cols = ['ticker', 'sector', 'trading_signal_type', 'trading_signal_strength', 'total_score']
print("Required columns:", test_cols)

for col in test_cols:
    if col not in df.columns:
        print(f"  ✗ {col}: MISSING")
        errors.append(f"strong_signals: missing '{col}'")
    else:
        print(f"  ✓ {col}: exists")

try:
    signal_df = df[df['trading_signal_type'].isin(['BUY', 'STRONG BUY'])]
    result = signal_df.head(n)[test_cols].copy()
    print(f"\n✓ strong_signals WORKS - {len(result)} stocks")
except Exception as e:
    print(f"\n✗ strong_signals FAILS: {e}")
    errors.append(f"strong_signals: {e}")

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "=" * 80)
print("FINAL REPORT")
print("=" * 80)

if not errors:
    print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
else:
    print(f"\n✗✗✗ {len(errors)} ERRORS ✗✗✗")
    for err in errors:
        print(f"  • {err}")

    print("\n" + "=" * 80)
    print("REQUIRED FIXES")
    print("=" * 80)

    if any('peg_ratio' in str(e) for e in errors):
        print("\n1. Remove 'peg_ratio' from best_dividend column list (line ~1099)")
        print("   Change to: ['ticker', 'sector', 'dividend_yield', 'dividend_score']")

print("\n" + "=" * 80)