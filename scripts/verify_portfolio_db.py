"""
Portfolio Database Verification & Cleanup Script
=================================================

Run this BEFORE using the platform to ensure data integrity.

Usage:
    python scripts/verify_portfolio_db.py check      # Check current state
    python scripts/verify_portfolio_db.py cleanup    # Clean and re-import
    python scripts/verify_portfolio_db.py verify     # Verify calculations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.portfolio_db import PortfolioRepository
from src.db.connection import get_connection
import pandas as pd


def check_database():
    """Check current database state and identify issues."""
    print("=" * 70)
    print("  DATABASE CHECK")
    print("=" * 70)

    repo = PortfolioRepository()

    # Get all accounts
    accounts = repo.get_all_accounts()
    print(f"\n📊 Accounts in database: {len(accounts)}")

    issues = []

    for acc in accounts:
        acc_id = acc['account_id']
        print(f"\n{'─' * 60}")
        print(f"  Account: {acc_id}")
        print(f"{'─' * 60}")

        # NAV
        nav = repo.get_latest_nav(acc_id)
        if nav:
            print(f"  NAV Total:     ${float(nav['nav_total']):>15,.2f}")
            print(f"  NAV Cash:      ${float(nav['nav_cash'] or 0):>15,.2f}")
            print(f"  NAV Stock:     ${float(nav['nav_stock'] or 0):>15,.2f}")
        else:
            print(f"  NAV: NOT FOUND")
            issues.append(f"{acc_id}: No NAV data")

        # Deposits
        dw_df = repo.get_deposits_withdrawals(acc_id)
        total_deposits = dw_df['amount'].sum() if not dw_df.empty else 0

        # Separate internal vs external
        external_deposits = 0
        internal_in = 0
        internal_out = 0

        print(f"\n  Deposits/Withdrawals ({len(dw_df)} records):")
        for _, row in dw_df.iterrows():
            desc = row.get('description', '')
            amt = float(row['amount'])

            # Check for self-transfers (bad data from MULTI CSV)
            is_self_transfer = f'From Account {acc_id}' in desc or f'To Account {acc_id}' in desc

            if is_self_transfer:
                marker = " [SELF-TRANSFER - IGNORED]"
            elif 'Transfer In From Account' in desc or 'Transfer In To Account' in desc:
                internal_in += amt
                marker = " [INTERNAL IN]"
            elif 'Transfer Out To Account' in desc or 'Transfer Out From Account' in desc:
                internal_out += amt
                marker = " [INTERNAL OUT]"
            else:
                external_deposits += amt
                marker = ""

            print(f"    {row['date']}: {desc[:45]:45} ${amt:>12,.2f}{marker}")

        print(f"\n  Summary:")
        print(f"    External deposits:    ${external_deposits:>15,.2f}")
        print(f"    Internal transfers:   ${internal_in + internal_out:>15,.2f} (in: ${internal_in:,.2f}, out: ${internal_out:,.2f})")
        print(f"    Total in DB:          ${total_deposits:>15,.2f}")

        # Dividends
        div_total = repo.get_total_dividends(acc_id)
        div_df = repo.get_dividends(acc_id)
        print(f"\n  Dividends: {len(div_df)} records, Total: ${div_total:,.2f}")

        # Interest
        int_total = repo.get_total_interest(acc_id)
        print(f"  Interest: ${int_total:,.2f}")

        # Positions
        pos_df = repo.get_positions(acc_id)
        pos_value = pos_df['market_value'].sum() if not pos_df.empty else 0
        print(f"  Positions: {len(pos_df)} holdings, Market Value: ${pos_value:,.2f}")

        # True Profit calculation - CORRECT METHOD
        # Capital for this account = external deposits + net internal transfers
        if nav:
            nav_total = float(nav['nav_total'])

            # Method 1: External only (for consolidated view)
            true_profit_external = nav_total - external_deposits

            # Method 2: Account-specific capital (external + internal net)
            account_capital = external_deposits + internal_in + internal_out
            true_profit_account = nav_total - account_capital
            true_profit_account_pct = (true_profit_account / abs(account_capital) * 100) if account_capital != 0 else 0

            print(f"\n  📈 ACCOUNT-SPECIFIC TRUE PROFIT:")
            print(f"    Capital allocated to this account:")
            print(f"      External deposits:  ${external_deposits:>12,.2f}")
            print(f"      + Internal in:      ${internal_in:>12,.2f}")
            print(f"      + Internal out:     ${internal_out:>12,.2f}")
            print(f"      = Account Capital:  ${account_capital:>12,.2f}")
            print(f"    NAV:                  ${nav_total:>12,.2f}")
            print(f"    TRUE PROFIT:          ${true_profit_account:>12,.2f} ({true_profit_account_pct:+.2f}%)")

            # Check for issues
            if account_capital > 0 and nav_total > 0:
                if true_profit_account_pct < -50:
                    issues.append(f"{acc_id}: Very negative return ({true_profit_account_pct:.1f}%) - check data")

    # Summary
    print(f"\n{'=' * 70}")
    print("  CONSOLIDATED TOTALS (All Accounts)")
    print(f"{'=' * 70}")

    total_nav = 0
    total_external = 0
    total_dividends = 0
    total_interest = 0
    total_positions = 0

    for acc in accounts:
        acc_id = acc['account_id']
        nav = repo.get_latest_nav(acc_id)
        if nav:
            total_nav += float(nav['nav_total'])

        dw_df = repo.get_deposits_withdrawals(acc_id)
        for _, row in dw_df.iterrows():
            desc = row.get('description', '')
            if not ('Transfer In From Account' in desc or
                    'Transfer Out To Account' in desc or
                    'Transfer In To Account' in desc or
                    'Transfer Out From Account' in desc):
                total_external += float(row['amount'])

        total_dividends += repo.get_total_dividends(acc_id)
        total_interest += repo.get_total_interest(acc_id)

        pos_df = repo.get_positions(acc_id)
        total_positions += len(pos_df)

    print(f"\n  Total NAV:              ${total_nav:>15,.2f}")
    print(f"  Total External Deposits:${total_external:>15,.2f}")
    print(f"  Total Dividends:        ${total_dividends:>15,.2f}")
    print(f"  Total Interest:         ${total_interest:>15,.2f}")
    print(f"  Total Positions:        {total_positions:>15}")

    true_profit = total_nav - total_external
    true_profit_pct = (true_profit / abs(total_external) * 100) if total_external != 0 else 0
    print(f"\n  📈 CONSOLIDATED TRUE PROFIT:")
    print(f"    ${true_profit:,.2f} ({true_profit_pct:+.2f}%)")

    # Issues
    if issues:
        print(f"\n{'=' * 70}")
        print("  ⚠️  ISSUES DETECTED")
        print(f"{'=' * 70}")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✅ No issues detected")

    return issues


def cleanup_database():
    """
    Clean up database - remove all portfolio data and prepare for fresh import.
    """
    print("=" * 70)
    print("  DATABASE CLEANUP")
    print("=" * 70)

    confirm = input("\n⚠️  This will DELETE all portfolio data. Type 'YES' to confirm: ")
    if confirm != 'YES':
        print("Cancelled.")
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            tables = [
                'portfolio_import_history',
                'portfolio_positions',
                'portfolio_nav_history',
                'portfolio_fees',
                'portfolio_trades',
                'portfolio_withholding_tax',
                'portfolio_interest',
                'portfolio_dividends',
                'portfolio_deposits_withdrawals',
                'portfolio_accounts',
            ]

            for table in tables:
                cur.execute(f"DELETE FROM {table}")
                print(f"  Cleared: {table}")

    print("\n✅ Database cleaned. Now import your CSV files:")
    print("   python scripts/import_ibkr_csv.py <your_file.csv>")


def verify_expected_values():
    """
    Verify database values match expected calculations.
    """
    print("=" * 70)
    print("  VERIFICATION")
    print("=" * 70)

    repo = PortfolioRepository()

    # Expected values (from your IBKR statements)
    expected = {
        'consolidated': {
            'nav': 2208307.39,
            'deposits': 2131226.96,
            'dividends': 35240.33,
            'interest': 11380.23,
            'true_profit': 77080.43,
            'true_profit_pct': 3.62,
        },
        'U20993660': {
            'nav': 753014.87,
            'deposits': 717000.00,  # But these are internal transfers!
            'dividends': 475.79,
            'interest': 2696.81,
        }
    }

    print("\n📋 Expected vs Actual:")

    # Check consolidated
    accounts = repo.get_all_accounts()
    total_nav = 0
    total_external = 0
    total_dividends = 0
    total_interest = 0

    for acc in accounts:
        acc_id = acc['account_id']
        nav = repo.get_latest_nav(acc_id)
        if nav:
            total_nav += float(nav['nav_total'])

        dw_df = repo.get_deposits_withdrawals(acc_id)
        for _, row in dw_df.iterrows():
            desc = row.get('description', '')
            if not ('Transfer In From Account' in desc or
                    'Transfer Out To Account' in desc):
                total_external += float(row['amount'])

        total_dividends += repo.get_total_dividends(acc_id)
        total_interest += repo.get_total_interest(acc_id)

    print(f"\n  CONSOLIDATED:")
    print(f"    NAV:       Expected ${expected['consolidated']['nav']:>12,.2f}  |  Actual ${total_nav:>12,.2f}  {'✓' if abs(total_nav - expected['consolidated']['nav']) < 1 else '✗'}")
    print(f"    Deposits:  Expected ${expected['consolidated']['deposits']:>12,.2f}  |  Actual ${total_external:>12,.2f}  {'✓' if abs(total_external - expected['consolidated']['deposits']) < 1 else '✗'}")
    print(f"    Dividends: Expected ${expected['consolidated']['dividends']:>12,.2f}  |  Actual ${total_dividends:>12,.2f}  {'✓' if abs(total_dividends - expected['consolidated']['dividends']) < 100 else '✗'}")
    print(f"    Interest:  Expected ${expected['consolidated']['interest']:>12,.2f}  |  Actual ${total_interest:>12,.2f}  {'✓' if abs(total_interest - expected['consolidated']['interest']) < 100 else '✗'}")

    true_profit = total_nav - total_external
    true_profit_pct = (true_profit / total_external * 100) if total_external else 0
    print(f"\n    True Profit: ${true_profit:,.2f} ({true_profit_pct:.2f}%)")
    print(f"    Expected:    ${expected['consolidated']['true_profit']:,.2f} ({expected['consolidated']['true_profit_pct']:.2f}%)")

    if abs(true_profit - expected['consolidated']['true_profit']) < 100:
        print(f"\n  ✅ CONSOLIDATED VALUES MATCH!")
    else:
        print(f"\n  ⚠️  Values differ - check data imports")


def show_recommended_setup():
    """Show recommended database setup."""
    print("=" * 70)
    print("  RECOMMENDED SETUP")
    print("=" * 70)

    print("""
    The cleanest approach for your multi-account setup:
    
    OPTION 1: Use MULTI CSV only (Recommended)
    ──────────────────────────────────────────
    1. Clean database: python scripts/verify_portfolio_db.py cleanup
    2. Import MULTI:   python scripts/import_ibkr_csv.py MULTI_*.csv
    3. The MULTI data goes to U17994267, which represents "All"
    4. In the platform, "All" and "U17994267" will show consolidated data
    5. U20993660 will have no data (use live IBKR for that account)
    
    OPTION 2: Use individual CSVs only
    ──────────────────────────────────────────
    1. Clean database: python scripts/verify_portfolio_db.py cleanup
    2. Import each account's individual CSV separately
    3. Each account shows its own data
    4. "All" combines them (but handles internal transfers)
    
    OPTION 3: Current setup (mixed)
    ──────────────────────────────────────────
    - U17994267 has MULTI data (all deposits, dividends)
    - U20993660 has its own data
    - "All" should work correctly by combining and filtering transfers
    - Individual accounts may show mixed data
    
    Current recommendation: Use "All" for accurate consolidated view,
    and connect to live IBKR for individual account data.
    """)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nRunning check by default...\n")
        check_database()
        print("\n")
        show_recommended_setup()
    elif sys.argv[1] == 'check':
        check_database()
    elif sys.argv[1] == 'cleanup':
        cleanup_database()
    elif sys.argv[1] == 'verify':
        verify_expected_values()
    elif sys.argv[1] == 'setup':
        show_recommended_setup()
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Use: check, cleanup, verify, or setup")