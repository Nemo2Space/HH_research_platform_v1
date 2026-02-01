"""
IBKR Data Fetch Debug Script
=============================
Run this OUTSIDE of Streamlit to test IBKR data fetching.

Usage:
    cd C:\Develop\Latest_2025\HH_research_platform_v1
    python debug_ibkr_fetch.py

This tests:
1. Connection to IBKR
2. Fetching positions (with timing)
3. Fetching open orders (with timing)
4. Building a snapshot like the AI PM does
"""

import os
import sys
import time
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "7496"))
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "999"))  # Use different ID than Streamlit
ACCOUNT_ID = os.getenv("IBKR_ACCOUNT", "DUK415187")  # Your account

print(f"""
================================================================================
IBKR Data Fetch Debug Script
================================================================================
Host:      {IBKR_HOST}
Port:      {IBKR_PORT}
Client ID: {IBKR_CLIENT_ID}
Account:   {ACCOUNT_ID}
================================================================================
""")

# ============================================================================
# STEP 1: Import ib_insync
# ============================================================================
print("\n[STEP 1] Importing ib_insync...")
try:
    from ib_insync import IB, util, Stock

    print("  ✅ ib_insync imported successfully")
except ImportError as e:
    print(f"  ❌ Failed to import ib_insync: {e}")
    sys.exit(1)

# Start the event loop (required for ib_insync)
try:
    util.startLoop()
    print("  ✅ Event loop started")
except Exception as e:
    print(f"  ⚠️ Event loop warning: {e}")

# ============================================================================
# STEP 2: Connect to IBKR
# ============================================================================
print("\n[STEP 2] Connecting to IBKR...")
ib = IB()

start_time = time.time()
try:
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=10)
    connect_time = time.time() - start_time
    print(f"  ✅ Connected in {connect_time:.2f}s")
    print(f"  ✅ isConnected: {ib.isConnected()}")
except Exception as e:
    print(f"  ❌ Connection failed: {e}")
    print("\n  Possible causes:")
    print("    - TWS/Gateway not running")
    print("    - Wrong port (TWS=7496, Gateway=4001)")
    print("    - API not enabled in TWS settings")
    print("    - Another client using same clientId")
    sys.exit(1)

# ============================================================================
# STEP 3: Get Accounts
# ============================================================================
print("\n[STEP 3] Getting accounts...")
start_time = time.time()
try:
    accounts = ib.managedAccounts()
    fetch_time = time.time() - start_time
    print(f"  ✅ Got {len(accounts)} accounts in {fetch_time:.2f}s")
    for acc in accounts:
        print(f"      - {acc}")
except Exception as e:
    print(f"  ❌ Failed to get accounts: {e}")

# ============================================================================
# STEP 4: Get Account Summary
# ============================================================================
print(f"\n[STEP 4] Getting account summary for {ACCOUNT_ID}...")
start_time = time.time()
try:
    # Method 1: reqAccountSummary (async)
    ib.reqAccountSummary()
    ib.sleep(2)  # Wait for data

    summary = ib.accountSummary()
    fetch_time = time.time() - start_time
    print(f"  ✅ Got {len(summary)} summary items in {fetch_time:.2f}s")

    # Extract key values
    nav = None
    cash = None
    for item in summary:
        if item.account == ACCOUNT_ID:
            if item.tag == 'NetLiquidation':
                nav = float(item.value)
            elif item.tag == 'TotalCashValue':
                cash = float(item.value)

    if nav:
        print(f"      NAV: ${nav:,.2f}")
    if cash:
        print(f"      Cash: ${cash:,.2f}")

except Exception as e:
    print(f"  ❌ Failed to get account summary: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# STEP 5: Get Positions (Method 1 - direct)
# ============================================================================
print("\n[STEP 5] Getting positions (direct call)...")
start_time = time.time()
try:
    positions = ib.positions()
    fetch_time = time.time() - start_time
    print(f"  ✅ Got {len(positions)} positions in {fetch_time:.2f}s")

    # Show first 5
    for i, pos in enumerate(positions[:5]):
        sym = pos.contract.symbol
        qty = pos.position
        avg_cost = pos.avgCost
        print(f"      {i + 1}. {sym}: {qty} shares @ ${avg_cost:.2f}")
    if len(positions) > 5:
        print(f"      ... and {len(positions) - 5} more")

except Exception as e:
    print(f"  ❌ Failed to get positions: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# STEP 6: Get Positions (Method 2 - with reqPositions)
# ============================================================================
print("\n[STEP 6] Getting positions (with reqPositions)...")
start_time = time.time()
try:
    ib.reqPositions()
    ib.sleep(2)  # Wait for data
    positions2 = ib.positions()
    fetch_time = time.time() - start_time
    print(f"  ✅ Got {len(positions2)} positions in {fetch_time:.2f}s (after reqPositions)")

except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# STEP 7: Get Open Orders (Method 1 - openTrades directly)
# ============================================================================
print("\n[STEP 7] Getting open orders (openTrades direct)...")
start_time = time.time()
try:
    trades = ib.openTrades()
    fetch_time = time.time() - start_time
    print(f"  ✅ Got {len(trades)} open trades in {fetch_time:.2f}s")

    # Show first 5
    for i, trade in enumerate(trades[:5]):
        sym = trade.contract.symbol
        action = trade.order.action
        qty = trade.order.totalQuantity
        status = trade.orderStatus.status
        print(f"      {i + 1}. {action} {qty} {sym} - {status}")
    if len(trades) > 5:
        print(f"      ... and {len(trades) - 5} more")

except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# STEP 8: Get Open Orders (Method 2 - reqOpenOrders) - THIS MIGHT FREEZE
# ============================================================================
print("\n[STEP 8] Getting open orders (reqOpenOrders)...")
print("  ⏳ This is the call that might freeze in Streamlit...")
start_time = time.time()
try:
    ib.reqOpenOrders()
    print(f"  ... reqOpenOrders() returned after {time.time() - start_time:.2f}s")

    ib.sleep(2)  # Wait for data
    print(f"  ... sleep(2) completed after {time.time() - start_time:.2f}s")

    trades2 = ib.openTrades()
    fetch_time = time.time() - start_time
    print(f"  ✅ Got {len(trades2)} open trades in {fetch_time:.2f}s (after reqOpenOrders)")

except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# STEP 9: Get Open Orders (Method 3 - reqAllOpenOrders) - THIS MIGHT FREEZE
# ============================================================================
print("\n[STEP 9] Getting open orders (reqAllOpenOrders)...")
print("  ⏳ This is the call that freezes in Streamlit...")
start_time = time.time()
try:
    ib.reqAllOpenOrders()
    print(f"  ... reqAllOpenOrders() returned after {time.time() - start_time:.2f}s")

    ib.sleep(2)  # Wait for data
    print(f"  ... sleep(2) completed after {time.time() - start_time:.2f}s")

    trades3 = ib.openTrades()
    fetch_time = time.time() - start_time
    print(f"  ✅ Got {len(trades3)} open trades in {fetch_time:.2f}s (after reqAllOpenOrders)")

except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# STEP 10: Test time.sleep vs ib.sleep
# ============================================================================
print("\n[STEP 10] Testing time.sleep vs ib.sleep...")

print("  Testing time.sleep(2)...")
start_time = time.time()
time.sleep(2)
print(f"  ✅ time.sleep(2) completed in {time.time() - start_time:.2f}s")

print("  Testing ib.sleep(2)...")
start_time = time.time()
ib.sleep(2)
print(f"  ✅ ib.sleep(2) completed in {time.time() - start_time:.2f}s")

# ============================================================================
# STEP 11: Build Complete Snapshot (like AI PM does)
# ============================================================================
print("\n[STEP 11] Building complete snapshot (like AI PM)...")
start_time = time.time()

snapshot_data = {
    'account': ACCOUNT_ID,
    'nav': None,
    'cash': None,
    'positions': [],
    'open_orders': [],
}

try:
    # 1. Account summary
    print("  1. Getting account summary...")
    t0 = time.time()
    ib.reqAccountSummary()
    ib.sleep(1)
    summary = ib.accountSummary()
    print(f"     ... done in {time.time() - t0:.2f}s")

    for item in summary:
        if item.account == ACCOUNT_ID:
            if item.tag == 'NetLiquidation':
                snapshot_data['nav'] = float(item.value)
            elif item.tag == 'TotalCashValue':
                snapshot_data['cash'] = float(item.value)

    # 2. Positions
    print("  2. Getting positions...")
    t0 = time.time()
    ib.reqPositions()
    ib.sleep(1)
    positions = ib.positions()
    print(f"     ... done in {time.time() - t0:.2f}s, got {len(positions)} positions")

    for p in positions:
        if p.account == ACCOUNT_ID:
            snapshot_data['positions'].append({
                'symbol': p.contract.symbol,
                'quantity': p.position,
                'avg_cost': p.avgCost,
            })

    # 3. Open orders - TRY WITHOUT reqAllOpenOrders first
    print("  3. Getting open orders (WITHOUT reqAllOpenOrders)...")
    t0 = time.time()
    trades = ib.openTrades()
    print(f"     ... done in {time.time() - t0:.2f}s, got {len(trades)} orders")

    for t in trades:
        snapshot_data['open_orders'].append({
            'symbol': t.contract.symbol,
            'action': t.order.action,
            'quantity': t.order.totalQuantity,
            'status': t.orderStatus.status,
        })

    total_time = time.time() - start_time
    print(f"\n  ✅ Complete snapshot built in {total_time:.2f}s")
    print(f"     NAV: ${snapshot_data['nav']:,.2f}" if snapshot_data['nav'] else "     NAV: None")
    print(f"     Cash: ${snapshot_data['cash']:,.2f}" if snapshot_data['cash'] else "     Cash: None")
    print(f"     Positions: {len(snapshot_data['positions'])}")
    print(f"     Open Orders: {len(snapshot_data['open_orders'])}")

except Exception as e:
    print(f"  ❌ Failed to build snapshot: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# CLEANUP
# ============================================================================
print("\n[CLEANUP] Disconnecting...")
try:
    ib.disconnect()
    print("  ✅ Disconnected")
except Exception as e:
    print(f"  ⚠️ Disconnect warning: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"""
================================================================================
SUMMARY
================================================================================
If all steps completed without freezing, the issue is specific to Streamlit.

RECOMMENDATION for Streamlit:
- DO NOT use reqAllOpenOrders() - it blocks the event loop
- DO NOT use reqOpenOrders() - same issue  
- Just use ib.openTrades() directly - it returns cached data
- If you need fresh data, disconnect and reconnect

The cached data from openTrades() is usually fine because:
- It auto-updates via the TWS subscription
- The cache is updated whenever orders change

================================================================================
""")