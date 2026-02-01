"""
Test script for IBKR connection.
Run directly (not as pytest): python scripts/test_broker.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.broker.ibkr_utils import get_ibkr_accounts, get_ibkr_account_summary, get_ibkr_positions


def main():
    print("=" * 50)
    print("Testing IBKR Connection...")
    print("=" * 50)

    # Test 1: Get accounts
    print("\n[1] Getting accounts...")
    accounts, error = get_ibkr_accounts()

    if error:
        print(f"❌ Error: {error}")
        print("\nTroubleshooting:")
        print("  - Is TWS or IB Gateway running?")
        print("  - Is API enabled in TWS? (File > Global Configuration > API > Settings)")
        print("  - Is 'Enable ActiveX and Socket Clients' checked?")
        print("  - Is port 7496 (TWS) or 4001 (Gateway) correct?")
        return

    print(f"✅ Found {len(accounts)} account(s): {accounts}")

    if not accounts:
        print("No accounts found!")
        return

    # Test 2: Get account summary
    account_id = accounts[0]
    print(f"\n[2] Getting summary for account: {account_id}...")

    summary, error = get_ibkr_account_summary(account_id)
    if error:
        print(f"❌ Error: {error}")
    else:
        print(f"✅ Account Summary:")
        print(f"   Net Liquidation: ${summary['net_liquidation']:,.2f}")
        print(f"   Total Cash:      ${summary['total_cash']:,.2f}")
        print(f"   Buying Power:    ${summary['buying_power']:,.2f}")
        print(f"   Position Value:  ${summary['gross_position_value']:,.2f}")

    # Test 3: Get positions
    print(f"\n[3] Getting positions for account: {account_id}...")
    positions, error = get_ibkr_positions(account_id, fetch_prices=False)

    if error:
        print(f"❌ Error: {error}")
    elif not positions:
        print("ℹ️  No positions found (empty portfolio)")
    else:
        print(f"✅ Found {len(positions)} position(s):")
        print("-" * 70)
        print(f"{'Symbol':<10} {'Qty':>10} {'Avg Cost':>12} {'Mkt Value':>14}")
        print("-" * 70)
        for p in positions:
            print(f"{p['symbol']:<10} {p['quantity']:>10.0f} ${p['avg_cost']:>10.2f} ${p['market_value']:>12.2f}")
        print("-" * 70)

    print("\n" + "=" * 50)
    print("Test complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()