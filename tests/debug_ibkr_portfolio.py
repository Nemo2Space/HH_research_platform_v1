#!/usr/bin/env python3
"""
Debug Script for Portfolio Tab IBKR Integration
================================================

This script directly tests the IBKR client methods used by portfolio_tab.py
to understand exactly what data is being returned and in what format.

Run this script from your project root:
    python debug_ibkr_portfolio.py

Author: Debug Script
"""

import sys
import os
from pprint import pprint
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def separator(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_ibkr_client():
    """Test the IBKRClient from src.broker.ibkr_client"""
    separator("TEST 1: IBKRClient from src.broker.ibkr_client")

    try:
        from src.broker.ibkr_client import IBKRClient
        print("✓ IBKRClient imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import IBKRClient: {e}")
        return None, None, None

    host = "127.0.0.1"
    port = 7496

    print(f"\nConnecting to {host}:{port}...")
    client = IBKRClient(host=host, port=port)

    if not client.connect():
        print("✗ Failed to connect to IBKR")
        return None, None, None

    print("✓ Connected to IBKR")

    # Test 1: Get managed accounts
    separator("1.1: get_managed_accounts()")
    try:
        accounts = client.get_managed_accounts()
        print(f"Type: {type(accounts)}")
        print(f"Value: {accounts}")
        print(f"Length: {len(accounts) if accounts else 0}")
    except Exception as e:
        print(f"✗ Error: {e}")
        accounts = []

    if not accounts:
        print("No accounts found, disconnecting...")
        client.disconnect()
        return None, None, None

    # Test with first account
    test_account = accounts[0]
    print(f"\n>>> Using test account: {test_account}")

    # Test 2: Get positions
    separator("1.2: get_positions(account_id)")
    try:
        positions = client.get_positions(test_account)
        print(f"Type: {type(positions)}")
        print(f"Length: {len(positions) if positions else 0}")

        if positions:
            print(f"\nFirst position:")
            pos = positions[0]
            print(f"  Type: {type(pos)}")

            # Check if it's a dict
            if isinstance(pos, dict):
                print(f"  Keys: {list(pos.keys())}")
                print(f"\n  Full position dict:")
                pprint(pos, indent=4)
            # Check if it's an ib_insync Position object
            elif hasattr(pos, 'contract'):
                print(f"  Has 'contract' attribute: Yes")
                print(f"    contract type: {type(pos.contract)}")
                print(f"    contract.symbol: {getattr(pos.contract, 'symbol', 'N/A')}")
                print(f"    contract.conId: {getattr(pos.contract, 'conId', 'N/A')}")
            else:
                print(f"  Has 'contract' attribute: No")

            if hasattr(pos, 'position'):
                print(f"  pos.position (quantity): {pos.position}")

            if hasattr(pos, 'avgCost'):
                print(f"  pos.avgCost: {pos.avgCost}")

            if hasattr(pos, 'marketPrice'):
                print(f"  pos.marketPrice: {getattr(pos, 'marketPrice', 'N/A')}")

            if hasattr(pos, 'marketValue'):
                print(f"  pos.marketValue: {getattr(pos, 'marketValue', 'N/A')}")

            # Check __dict__
            if hasattr(pos, '__dict__'):
                print(f"\n  pos.__dict__:")
                pprint(pos.__dict__, indent=4)

            # Print ALL positions
            print(f"\n  ALL POSITIONS:")
            for i, p in enumerate(positions):
                if isinstance(p, dict):
                    print(
                        f"    [{i}] {p.get('symbol', 'N/A')}: qty={p.get('quantity', p.get('position', 'N/A'))}, value={p.get('market_value', p.get('marketValue', 'N/A'))}")
                else:
                    print(f"    [{i}] {p}")
    except Exception as e:
        import traceback
        print(f"✗ Error: {e}")
        traceback.print_exc()
        positions = []

    # Test 3: Get account summary
    separator("1.3: get_account_summary(account_id)")
    try:
        summary = client.get_account_summary(test_account)
        print(f"Type: {type(summary)}")

        if isinstance(summary, dict):
            print(f"Keys: {list(summary.keys())}")
            print(f"\nFull summary dict:")
            pprint(summary, indent=2)
        elif isinstance(summary, list):
            print(f"Length: {len(summary)}")
            if summary:
                print(f"\nFirst item type: {type(summary[0])}")
                print(f"First item: {summary[0]}")

                # Check if it has tag/value structure
                item = summary[0]
                if hasattr(item, 'tag'):
                    print(f"\n  Has 'tag' attribute: {item.tag}")
                if hasattr(item, 'value'):
                    print(f"  Has 'value' attribute: {item.value}")
                if hasattr(item, 'account'):
                    print(f"  Has 'account' attribute: {item.account}")

                # Print all items
                print(f"\nAll summary items:")
                for i, item in enumerate(summary):
                    if hasattr(item, 'tag') and hasattr(item, 'value'):
                        print(f"  [{i}] {item.tag}: {item.value}")
                    else:
                        print(f"  [{i}] {item}")
        else:
            print(f"Value: {summary}")
    except Exception as e:
        import traceback
        print(f"✗ Error: {e}")
        traceback.print_exc()
        summary = None

    # Disconnect
    print("\nDisconnecting...")
    client.disconnect()
    print("✓ Disconnected")

    return accounts, positions, summary


def test_ibkr_gateway():
    """Test the IbkrGateway from dashboard.ai_pm.ibkr_gateway"""
    separator("TEST 2: IbkrGateway from dashboard.ai_pm.ibkr_gateway")

    try:
        from dashboard.ai_pm.ibkr_gateway import IbkrGateway
        print("✓ IbkrGateway imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import IbkrGateway: {e}")
        return None, None, None

    print("\nCreating gateway and connecting...")
    gw = IbkrGateway()

    if not gw.is_connected():
        gw.connect()

    if not gw.is_connected():
        print("✗ Failed to connect via IbkrGateway")
        return None, None, None

    print("✓ Connected via IbkrGateway")

    # Test 1: List accounts
    separator("2.1: list_accounts()")
    try:
        accounts = gw.list_accounts()
        print(f"Type: {type(accounts)}")
        print(f"Value: {accounts}")
    except Exception as e:
        print(f"✗ Error: {e}")
        accounts = []

    if not accounts:
        print("No accounts found")
        return None, None, None

    test_account = accounts[0]
    print(f"\n>>> Using test account: {test_account}")
    gw.set_account(test_account)

    # Test 2: Get positions
    separator("2.2: positions()")
    try:
        positions = gw.positions()
        print(f"Type: {type(positions)}")
        print(f"Length: {len(positions) if positions else 0}")

        if positions:
            pos = positions[0]
            print(f"\nFirst position type: {type(pos)}")
            if hasattr(pos, '__dict__'):
                pprint(pos.__dict__, indent=2)
    except Exception as e:
        import traceback
        print(f"✗ Error: {e}")
        traceback.print_exc()
        positions = []

    # Test 3: Account summary
    separator("2.3: account_summary()")
    try:
        summary = gw.account_summary(test_account)
        print(f"Type: {type(summary)}")
        if summary:
            if hasattr(summary, '__dict__'):
                pprint(summary.__dict__, indent=2)
            else:
                print(f"Value: {summary}")
    except Exception as e:
        import traceback
        print(f"✗ Error: {e}")
        traceback.print_exc()
        summary = None

    return accounts, positions, summary


def analyze_results(client_data, gateway_data):
    """Compare and analyze results from both methods"""
    separator("ANALYSIS: Comparing IBKRClient vs IbkrGateway")

    client_accounts, client_positions, client_summary = client_data
    gateway_accounts, gateway_positions, gateway_summary = gateway_data

    print("\n--- Accounts ---")
    print(f"IBKRClient accounts: {client_accounts}")
    print(f"IbkrGateway accounts: {gateway_accounts}")

    print("\n--- Positions ---")
    print(f"IBKRClient positions count: {len(client_positions) if client_positions else 0}")
    print(f"IbkrGateway positions count: {len(gateway_positions) if gateway_positions else 0}")

    print("\n--- Summary Structure ---")
    if client_summary:
        print(f"IBKRClient summary type: {type(client_summary)}")
        if isinstance(client_summary, list) and client_summary:
            # Find NetLiquidation
            for item in client_summary:
                if hasattr(item, 'tag') and item.tag == 'NetLiquidation':
                    print(f"  NetLiquidation: {item.value}")
                if hasattr(item, 'tag') and item.tag == 'TotalCashValue':
                    print(f"  TotalCashValue: {item.value}")
    else:
        print("IBKRClient summary: None")

    if gateway_summary:
        print(f"IbkrGateway summary type: {type(gateway_summary)}")
        if hasattr(gateway_summary, 'net_liquidation'):
            print(f"  net_liquidation: {gateway_summary.net_liquidation}")
        if hasattr(gateway_summary, 'total_cash_value'):
            print(f"  total_cash_value: {gateway_summary.total_cash_value}")
    else:
        print("IbkrGateway summary: None")


def main():
    print("\n" + "=" * 70)
    print("  IBKR PORTFOLIO DEBUG SCRIPT")
    print("  Testing data retrieval methods used by portfolio_tab.py")
    print("=" * 70)
    print(f"  Timestamp: {datetime.now()}")
    print("=" * 70)

    # Test IBKRClient (used by portfolio_tab.py)
    client_data = test_ibkr_client()

    print("\n\n")

    # Test IbkrGateway (used by AI PM)
    gateway_data = test_ibkr_gateway()

    # Compare results
    if client_data[0] or gateway_data[0]:
        analyze_results(client_data, gateway_data)

    separator("RECOMMENDATIONS")

    if client_data[1] is None or len(client_data[1]) == 0:
        print("⚠️  IBKRClient.get_positions() returned no positions!")
        print("    This is why portfolio_tab.py shows no data.")

    if client_data[2] is None:
        print("⚠️  IBKRClient.get_account_summary() returned None!")
        print("    NAV values won't be available.")
    elif isinstance(client_data[2], list):
        print("ℹ️  IBKRClient.get_account_summary() returns a LIST of AccountValue objects")
        print("    Need to parse it correctly in _create_statement_from_live_ibkr()")

    print("\n" + "=" * 70)
    print("  DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()