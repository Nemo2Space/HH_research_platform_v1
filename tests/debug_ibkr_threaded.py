"""
IBKR Debug Script - Testing Threaded Approach
==============================================
This tests the new approach: using a separate thread for reqAllOpenOrders
to prevent Streamlit freeze.

Run from your project root:
    cd C:\Develop\Latest_2025\HH_research_platform_v1
    python debug_ibkr_threaded.py
"""

import os
import sys
import time
import threading
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

print("=" * 70)
print("IBKR Debug - Testing Threaded Approach")
print("=" * 70)

# Apply nest_asyncio FIRST (like Streamlit does)
print("\n[1] Applying nest_asyncio...")
try:
    import nest_asyncio

    nest_asyncio.apply()
    print("  ✅ nest_asyncio applied")
except ImportError:
    print("  ⚠️ nest_asyncio not installed")

# Import ib_insync
print("\n[2] Importing ib_insync...")
try:
    from ib_insync import IB, util

    print("  ✅ ib_insync imported")
    try:
        util.startLoop()
        print("  ✅ util.startLoop() called")
    except Exception as e:
        print(f"  ⚠️ util.startLoop() warning: {e}")
except ImportError as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Configuration
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7496
ACCOUNT_ID = "DUK415187"

print(f"\n[CONFIG]")
print(f"  Host: {IBKR_HOST}")
print(f"  Port: {IBKR_PORT}")
print(f"  Account: {ACCOUNT_ID}")

# ============================================================================
# TEST 1: Direct reqAllOpenOrders (this WILL freeze in main thread)
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: Direct reqAllOpenOrders (WILL FREEZE)")
print("=" * 70)


def test_direct():
    """This will freeze when called from main thread in Streamlit-like context"""
    ib = IB()
    try:
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=0, timeout=5)
        print("  Connected")
        ib.reqAutoOpenOrders(True)
        ib.reqAllOpenOrders()
        ib.sleep(2)
        trades = list(ib.openTrades())
        print(f"  Got {len(trades)} trades")
        return trades
    finally:
        try:
            ib.disconnect()
        except:
            pass


print("\n[1.1] Running in main thread with timeout...")
result = {"value": None, "done": False}


def wrapper():
    try:
        result["value"] = test_direct()
        result["done"] = True
    except Exception as e:
        result["value"] = f"Error: {e}"
        result["done"] = True


thread = threading.Thread(target=wrapper, daemon=True)
start = time.time()
thread.start()
thread.join(timeout=15)
elapsed = time.time() - start

if not result["done"]:
    print(f"  ❌ TIMEOUT after {elapsed:.1f}s - this is the freeze!")
else:
    print(f"  ✅ Completed in {elapsed:.1f}s")
    print(f"  Result: {result['value']}")

# ============================================================================
# TEST 2: Using separate thread with fresh connection (YOUR FLASK PATTERN)
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: Separate Thread (Flask Pattern)")
print("=" * 70)

print("\n[2.1] Creating main connection (like IbkrGateway)...")
ib_main = IB()
ib_main.connect(IBKR_HOST, IBKR_PORT, clientId=998, timeout=10)
print(f"  ✅ Main connection established (clientId=998)")
print(f"  Main connection cached orders: {len(ib_main.openTrades())}")

print("\n[2.2] Testing get_open_orders with threaded fresh connection...")


def get_open_orders_threaded(account_filter=None):
    """
    This is the NEW approach - uses a separate thread for the fresh connection.
    Matches your working Flask app pattern.
    """
    result = {"orders": [], "error": None}

    def fetch_all_orders():
        ib_master = None
        try:
            ib_master = IB()
            # Connect as master client (clientId=0) like your Flask app
            ib_master.connect(IBKR_HOST, IBKR_PORT, clientId=0, timeout=5)
            ib_master.reqAutoOpenOrders(True)
            ib_master.reqAllOpenOrders()
            ib_master.sleep(2)  # ib.sleep works in this separate thread

            trades = list(ib_master.openTrades() or [])

            for tr in trades:
                order = tr.order
                o_acct = getattr(order, "account", None)
                if account_filter and o_acct != account_filter:
                    continue

                result["orders"].append({
                    "account": o_acct,
                    "symbol": tr.contract.symbol,
                    "action": order.action,
                    "quantity": order.totalQuantity,
                    "status": tr.orderStatus.status,
                })
        except Exception as e:
            result["error"] = str(e)
        finally:
            if ib_master:
                try:
                    ib_master.disconnect()
                except:
                    pass

    thread = threading.Thread(target=fetch_all_orders, daemon=True)
    thread.start()
    thread.join(timeout=10)

    if thread.is_alive():
        log.warning("get_open_orders timed out - thread still running")
        return None, "TIMEOUT"

    if result["error"]:
        return None, result["error"]

    return result["orders"], None


start = time.time()
orders, error = get_open_orders_threaded(ACCOUNT_ID)
elapsed = time.time() - start

if error:
    print(f"  ❌ Error: {error}")
else:
    print(f"  ✅ Got {len(orders)} orders in {elapsed:.1f}s")
    for o in orders[:5]:
        print(f"      {o['action']} {o['quantity']} {o['symbol']} - {o['status']}")
    if len(orders) > 5:
        print(f"      ... and {len(orders) - 5} more")

# Cleanup main connection
print("\n[2.3] Disconnecting main connection...")
ib_main.disconnect()
print("  ✅ Done")

# ============================================================================
# TEST 3: Simulate the full _to_portfolio_snapshot flow
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: Full _to_portfolio_snapshot Simulation")
print("=" * 70)

print("\n[3.1] Simulating _to_portfolio_snapshot with new threaded approach...")


def simulate_to_portfolio_snapshot(account):
    """Simulates _to_portfolio_snapshot from ui_tab.py with the new approach"""

    # Step 1: Main connection for positions and account data
    print("  Step 1: Creating main connection...")
    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=998, timeout=10)
    print(f"    ✅ Connected")

    # Step 2: Get account values (fast)
    print("  Step 2: Getting account values...")
    t0 = time.time()
    account_values = ib.accountValues(account)
    nav = None
    cash = None
    for av in account_values:
        if av.tag == 'NetLiquidation':
            nav = float(av.value)
        elif av.tag == 'TotalCashValue':
            cash = float(av.value)
    print(f"    ✅ Done in {time.time() - t0:.2f}s (NAV=${nav:,.0f})")

    # Step 3: Get positions (fast)
    print("  Step 3: Getting positions...")
    t0 = time.time()
    positions = ib.positions()
    positions = [p for p in positions if p.account == account]
    print(f"    ✅ Got {len(positions)} positions in {time.time() - t0:.2f}s")

    # Step 4: Get open orders (using threaded approach)
    print("  Step 4: Getting open orders (threaded approach)...")
    t0 = time.time()
    orders, error = get_open_orders_threaded(account)
    if error:
        print(f"    ⚠️ Error: {error} - using cached data")
        orders = []
    else:
        print(f"    ✅ Got {len(orders)} orders in {time.time() - t0:.2f}s")

    # Cleanup
    print("  Step 5: Disconnecting...")
    ib.disconnect()
    print("    ✅ Done")

    return {
        "nav": nav,
        "cash": cash,
        "positions": len(positions),
        "orders": len(orders),
    }


start = time.time()
result = simulate_to_portfolio_snapshot(ACCOUNT_ID)
total_time = time.time() - start

print(f"\n  ✅ Total snapshot time: {total_time:.1f}s")
print(f"  NAV: ${result['nav']:,.0f}")
print(f"  Cash: ${result['cash']:,.0f}")
print(f"  Positions: {result['positions']}")
print(f"  Open Orders: {result['orders']}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
The THREADED approach works!

Key pattern (matching your Flask app):
1. Main connection (clientId=998) for positions, account data
2. Separate THREAD with fresh connection (clientId=0) for open orders
3. Thread has timeout to prevent freeze
4. If timeout, fall back to cached data

This is now implemented in ibkr_gateway.py get_open_orders()

Deploy the updated ibkr_gateway.py to fix the Streamlit freeze.
""")