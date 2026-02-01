#!/usr/bin/env python3
"""
Debug Script: Test IBKR CSV Statement Parsing
==============================================

Run: python debug_csv_parser.py /path/to/statement.csv
"""

import sys
import os
from pprint import pprint
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add parent directory
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def separator(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_csv_parsing(csv_path: str):
    """Test parsing the CSV file."""

    separator("IBKR CSV PARSER DEBUG")
    print(f"File: {csv_path}")
    print(f"Exists: {os.path.exists(csv_path)}")

    if not os.path.exists(csv_path):
        print("ERROR: File not found!")
        return

    # Read file
    with open(csv_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
        content = f.read()

    print(f"File size: {len(content)} bytes")
    print(f"First 500 chars:\n{content[:500]}")

    # Try to import parser
    separator("1. IMPORTING PARSER")
    try:
        # Try multiple import paths
        import_success = False

        # Try 1: Direct from dashboard folder
        dashboard_path = os.path.join(project_root, 'dashboard')
        if os.path.exists(dashboard_path):
            sys.path.insert(0, dashboard_path)

        # Try 2: From parent/dashboard
        parent_dashboard = os.path.join(parent_dir, 'dashboard')
        if os.path.exists(parent_dashboard):
            sys.path.insert(0, parent_dashboard)

        try:
            from portfolio_tab import IBKRStatementParser, ParsedStatement
            print("✓ Imported IBKRStatementParser from portfolio_tab")
            import_success = True
        except ImportError:
            pass

        if not import_success:
            # Try importing from current directory if portfolio_tab.py exists
            if os.path.exists(os.path.join(project_root, 'portfolio_tab.py')):
                sys.path.insert(0, project_root)
                from portfolio_tab import IBKRStatementParser, ParsedStatement
                print("✓ Imported IBKRStatementParser from local portfolio_tab.py")
                import_success = True

        if not import_success:
            raise ImportError("Could not find portfolio_tab module")

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nLooking for portfolio_tab.py...")

        # Search for the file
        for root, dirs, files in os.walk(project_root):
            if 'portfolio_tab.py' in files:
                print(f"  Found: {os.path.join(root, 'portfolio_tab.py')}")

        for root, dirs, files in os.walk(parent_dir):
            if 'portfolio_tab.py' in files:
                filepath = os.path.join(root, 'portfolio_tab.py')
                print(f"  Found: {filepath}")
                # Try to import from this path
                sys.path.insert(0, root)
                try:
                    from portfolio_tab import IBKRStatementParser, ParsedStatement
                    print(f"✓ Successfully imported from {root}")
                    import_success = True
                    break
                except Exception as e2:
                    print(f"  Failed to import: {e2}")

        if not import_success:
            print(
                "\nCannot import parser. Please copy portfolio_tab.py to current directory or run from dashboard folder.")
            return

    # Parse the file
    separator("2. PARSING CSV")
    try:
        parser = IBKRStatementParser()
        statement = parser.parse(content)
        print("✓ Parsing completed")
    except Exception as e:
        import traceback
        print(f"✗ Parsing error: {e}")
        traceback.print_exc()
        return

    # Show results
    separator("3. ACCOUNT INFO")
    acc = statement.account_info
    print(f"Name: {acc.name}")
    print(f"Account ID: {acc.account_id}")
    print(f"Type: {acc.account_type}")
    print(f"Currency: {acc.base_currency}")

    separator("4. PERIOD")
    print(f"Start: {statement.period_start}")
    print(f"End: {statement.period_end}")

    separator("5. NAV BREAKDOWN")
    nav = statement.nav
    print(f"Cash: ${nav.cash:,.2f}")
    print(f"Stock: ${nav.stock:,.2f}")
    print(f"Total: ${nav.total:,.2f}")
    print(f"TWR %: {nav.twr_percent}%")

    separator("6. CHANGE IN NAV")
    change = statement.change_in_nav
    print(f"Starting Value: ${change.starting_value:,.2f}")
    print(f"Ending Value: ${change.ending_value:,.2f}")
    print(f"Mark-to-Market: ${change.mark_to_market:,.2f}")
    print(f"Deposits/Withdrawals: ${change.deposits_withdrawals:,.2f}")
    print(f"Dividends: ${change.dividends:,.2f}")
    print(f"Withholding Tax: ${change.withholding_tax:,.2f}")
    print(f"Interest: ${change.interest:,.2f}")
    print(f"Commissions: ${change.commissions:,.2f}")
    print(f"Other Fees: ${change.other_fees:,.2f}")

    separator("7. DEPOSITS & WITHDRAWALS")
    dw_list = statement.deposits_withdrawals
    print(f"Total records: {len(dw_list)}")
    total_deposits = sum(d.amount for d in dw_list if d.amount > 0)
    total_withdrawals = sum(d.amount for d in dw_list if d.amount < 0)
    print(f"Total Deposits: ${total_deposits:,.2f}")
    print(f"Total Withdrawals: ${total_withdrawals:,.2f}")
    print(f"Net: ${total_deposits + total_withdrawals:,.2f}")
    if dw_list:
        print("\nFirst 5 records:")
        for i, dw in enumerate(dw_list[:5]):
            print(f"  [{i}] {dw.date} | {dw.description[:40]} | ${dw.amount:,.2f}")

    separator("8. DIVIDENDS")
    div_list = statement.dividends
    print(f"Total records: {len(div_list)}")
    total_div = sum(d.amount for d in div_list)
    print(f"Total Dividends: ${total_div:,.2f}")
    if div_list:
        print("\nFirst 5 records:")
        for i, d in enumerate(div_list[:5]):
            print(f"  [{i}] {d.date} | {d.symbol} | ${d.amount:,.2f}")

    separator("9. INTEREST")
    int_list = statement.interest
    print(f"Total records: {len(int_list)}")
    total_int = sum(i.amount for i in int_list)
    print(f"Total Interest: ${total_int:,.2f}")
    if int_list:
        print("\nFirst 5 records:")
        for i, rec in enumerate(int_list[:5]):
            print(f"  [{i}] {rec.date} | {rec.description[:40]} | ${rec.amount:,.2f}")

    separator("10. WITHHOLDING TAX")
    wht_list = statement.withholding_tax
    print(f"Total records: {len(wht_list)}")
    total_wht = sum(w.amount for w in wht_list)
    print(f"Total Withholding Tax: ${total_wht:,.2f}")

    separator("11. FEES")
    fee_list = statement.fees
    print(f"Total records: {len(fee_list)}")
    total_fees = sum(f.amount for f in fee_list)
    print(f"Total Fees: ${total_fees:,.2f}")

    separator("12. OPEN POSITIONS")
    pos_list = statement.open_positions
    print(f"Total positions: {len(pos_list)}")
    total_value = sum(p.market_value for p in pos_list)
    print(f"Total Market Value: ${total_value:,.2f}")
    if pos_list:
        print("\nFirst 10 positions:")
        for i, p in enumerate(pos_list[:10]):
            print(
                f"  [{i}] {p.symbol:6} | Qty: {p.quantity:>8.2f} | Value: ${p.market_value:>12,.2f} | P&L: ${p.unrealized_pnl:>10,.2f}")

    separator("13. TRADES")
    trade_list = statement.trades
    print(f"Total trades: {len(trade_list)}")

    separator("14. PARSE ERRORS")
    errors = statement.parse_errors
    print(f"Total errors: {len(errors)}")
    if errors:
        print("\nFirst 10 errors:")
        for i, err in enumerate(errors[:10]):
            print(f"  [{i}] {err}")

    separator("SUMMARY")
    print(f"Account: {acc.account_id} - {acc.name}")
    print(f"NAV: ${nav.total:,.2f}")
    print(f"Deposits: ${change.deposits_withdrawals:,.2f}")
    print(f"True Profit: ${nav.total - change.deposits_withdrawals:,.2f}")
    print(f"Dividends: ${total_div:,.2f}")
    print(f"Interest: ${total_int:,.2f}")
    print(f"Positions: {len(pos_list)}")

    separator("DEBUG COMPLETE")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_csv_parser.py /path/to/statement.csv")
        print("\nAlternatively, provide path directly:")
        csv_path = input("Enter CSV path: ").strip()
        if csv_path:
            test_csv_parsing(csv_path)
    else:
        test_csv_parsing(sys.argv[1])