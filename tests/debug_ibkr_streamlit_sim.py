"""
IBKR Debug Script - Simulating Streamlit Environment
=====================================================
This script mimics how Streamlit runs ib_insync code:
- Uses nest_asyncio (like Streamlit)
- Uses threading (like Streamlit's session)
- Calls the EXACT same functions as ui_tab.py

Run from your project root:
    cd C:\Develop\Latest_2025\HH_research_platform_v1
    python debug_ibkr_streamlit_sim.py
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# ============================================================================
# STEP 0: Setup environment EXACTLY like Streamlit does
# ============================================================================
print("=" * 70)
print("IBKR Debug - Simulating Streamlit Environment")
print("=" * 70)

# Apply nest_asyncio FIRST (like Streamlit does)
print("\n[SETUP] Applying nest_asyncio...")
try:
    import nest_asyncio

    nest_asyncio.apply()
    print("  ✅ nest_asyncio applied")
except ImportError:
    print("  ⚠️ nest_asyncio not installed - pip install nest_asyncio")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("debug_streamlit_sim")

# ============================================================================
# STEP 1: Import ib_insync and start loop (like the gateway does)
# ============================================================================
print("\n[STEP 1] Importing ib_insync...")
try:
    from ib_insync import IB, util

    print("  ✅ ib_insync imported")

    # Start the event loop (like ibkr_gateway.py does)
    try:
        util.startLoop()
        print("  ✅ util.startLoop() called")
    except Exception as e:
        print(f"  ⚠️ util.startLoop() warning: {e}")

except ImportError as e:
    print(f"  ❌ Failed to import: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: Import your actual gateway class
# ============================================================================
print("\n[STEP 2] Importing IbkrGateway from dashboard.ai_pm...")
try:
    # Add project to path if needed
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from dashboard.ai_pm.ibkr_gateway import IbkrGateway, IbkrConnectionConfig

    print("  ✅ IbkrGateway imported")
except ImportError as e:
    print(f"  ❌ Failed to import IbkrGateway: {e}")
    print("  Using local IB() instead...")
    IbkrGateway = None

# ============================================================================
# STEP 3: Configuration
# ============================================================================
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7496
IBKR_CLIENT_ID = 998  # Same as AI PM uses
ACCOUNT_ID = "DUK415187"

print(f"\n[CONFIG]")
print(f"  Host: {IBKR_HOST}")
print(f"  Port: {IBKR_PORT}")
print(f"  Client ID: {IBKR_CLIENT_ID}")
print(f"  Account: {ACCOUNT_ID}")


# ============================================================================
# HELPER: Run function with timeout (like Streamlit does with threads)
# ============================================================================
def run_with_timeout(func, timeout_sec=30, description="operation"):
    """Run a function in a thread with timeout - simulates Streamlit behavior"""
    result = {"value": None, "error": None, "completed": False}

    def wrapper():
        try:
            result["value"] = func()
            result["completed"] = True
        except Exception as e:
            result["error"] = e
            result["completed"] = True

    thread = threading.Thread(target=wrapper, daemon=True)
    start = time.time()
    thread.start()
    thread.join(timeout=timeout_sec)
    elapsed = time.time() - start

    if not result["completed"]:
        return None, f"TIMEOUT after {elapsed:.1f}s", elapsed
    elif result["error"]:
        return None, f"ERROR: {result['error']}", elapsed
    else:
        return result["value"], None, elapsed


# ============================================================================
# TEST 1: Using IbkrGateway class (like ui_tab.py does)
# ============================================================================
if IbkrGateway:
    print("\n" + "=" * 70)
    print("TEST 1: Using IbkrGateway class (exactly like ui_tab.py)")
    print("=" * 70)

    print("\n[1.1] Creating gateway...")
    cfg = IbkrConnectionConfig(
        host=IBKR_HOST,
        port=IBKR_PORT,
        client_id=IBKR_CLIENT_ID,
        readonly=False
    )
    gw = IbkrGateway(cfg)

    print("\n[1.2] Connecting (with timeout)...")


    def do_connect():
        return gw.connect(timeout_sec=10.0)


    success, error, elapsed = run_with_timeout(do_connect, timeout_sec=15, description="connect")
    if error:
        print(f"  ❌ Connect failed: {error}")
    else:
        print(f"  ✅ Connected in {elapsed:.2f}s, success={success}")

    if gw.is_connected():
        # Set account
        print(f"\n[1.3] Setting account to {ACCOUNT_ID}...")
        gw.set_account(ACCOUNT_ID)
        print(f"  ✅ Account set")

        # Get account summary
        print("\n[1.4] Getting account summary (with timeout)...")


        def do_get_summary():
            return gw.get_account_summary(ACCOUNT_ID)


        summary, error, elapsed = run_with_timeout(do_get_summary, timeout_sec=10)
        if error:
            print(f"  ❌ {error}")
        else:
            print(f"  ✅ Got summary in {elapsed:.2f}s")
            if summary:
                print(f"      NAV: ${summary.net_liquidation:,.2f}" if summary.net_liquidation else "      NAV: None")

        # Get positions
        print("\n[1.5] Getting positions (with timeout)...")


        def do_get_positions():
            return gw.get_positions()


        positions, error, elapsed = run_with_timeout(do_get_positions, timeout_sec=10)
        if error:
            print(f"  ❌ {error}")
        else:
            print(f"  ✅ Got {len(positions) if positions else 0} positions in {elapsed:.2f}s")

        # Get open orders - THIS IS WHERE IT MIGHT FREEZE
        print("\n[1.6] Getting open orders (with timeout) - CRITICAL TEST...")
        print("      This is the call that freezes in Streamlit!")


        def do_get_orders():
            return gw.get_open_orders(include_all_clients=True)


        orders, error, elapsed = run_with_timeout(do_get_orders, timeout_sec=10)
        if error:
            print(f"  ❌ {error}")
            if "TIMEOUT" in error:
                print("\n  ⚠️ THIS IS THE BUG! get_open_orders() is hanging!")
                print("  The gateway is likely calling reqAllOpenOrders() internally")
        else:
            print(f"  ✅ Got {len(orders) if orders else 0} orders in {elapsed:.2f}s")

        # Test refresh_all
        print("\n[1.7] Testing refresh_all() (with timeout)...")


        def do_refresh():
            return gw.refresh_all()


        result, error, elapsed = run_with_timeout(do_refresh, timeout_sec=15)
        if error:
            print(f"  ❌ {error}")
        else:
            print(f"  ✅ Refresh completed in {elapsed:.2f}s")
            print(f"      Result: {result}")

        # Disconnect
        print("\n[1.8] Disconnecting...")
        gw.disconnect()
        print("  ✅ Disconnected")

# ============================================================================
# TEST 2: Simulate _to_portfolio_snapshot from ui_tab.py
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: Simulating _to_portfolio_snapshot() from ui_tab.py")
print("=" * 70)

# Create fresh gateway
if IbkrGateway:
    print("\n[2.1] Creating fresh gateway...")
    gw2 = IbkrGateway(IbkrConnectionConfig(
        host=IBKR_HOST,
        port=IBKR_PORT,
        client_id=IBKR_CLIENT_ID + 1,  # Different client ID
        readonly=False
    ))

    print("\n[2.2] Connecting...")


    def do_connect2():
        return gw2.connect(timeout_sec=10.0)


    success, error, elapsed = run_with_timeout(do_connect2, timeout_sec=15)
    if error:
        print(f"  ❌ {error}")
    else:
        print(f"  ✅ Connected in {elapsed:.2f}s")

    if gw2.is_connected():
        gw2.set_account(ACCOUNT_ID)

        # This is EXACTLY what _to_portfolio_snapshot does:
        print("\n[2.3] Running _to_portfolio_snapshot logic...")

        # Step 1: Account summary
        print("  Step 1: get_account_summary...")
        start = time.time()


        def step1():
            return gw2.get_account_summary(ACCOUNT_ID)


        summ, err, elapsed = run_with_timeout(step1, timeout_sec=10)
        if err:
            print(f"    ❌ {err}")
        else:
            print(f"    ✅ Done in {elapsed:.2f}s")

        # Step 2: Positions
        print("  Step 2: get_positions...")


        def step2():
            return gw2.get_positions()


        pos, err, elapsed = run_with_timeout(step2, timeout_sec=10)
        if err:
            print(f"    ❌ {err}")
        else:
            print(f"    ✅ Got {len(pos) if pos else 0} positions in {elapsed:.2f}s")

        # Step 3: Open orders - THE PROBLEM STEP
        print("  Step 3: get_open_orders (THIS IS WHERE IT FREEZES)...")


        def step3():
            return gw2.get_open_orders(include_all_clients=True)


        orders, err, elapsed = run_with_timeout(step3, timeout_sec=10)
        if err:
            print(f"    ❌ {err}")
            if "TIMEOUT" in err:
                print("\n    🚨 CONFIRMED: get_open_orders() causes the freeze!")
        else:
            print(f"    ✅ Got {len(orders) if orders else 0} orders in {elapsed:.2f}s")

        print("\n[2.4] Disconnecting...")
        gw2.disconnect()
        print("  ✅ Done")

# ============================================================================
# TEST 3: Direct IB() test with problematic calls
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: Direct IB() test - identifying which call freezes")
print("=" * 70)

ib = IB()

print("\n[3.1] Connecting directly...")
try:
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID + 2, timeout=10)
    print(f"  ✅ Connected")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

if ib.isConnected():
    # Test openTrades() directly (should be fast)
    print("\n[3.2] Testing ib.openTrades() directly...")


    def test_open_trades():
        return list(ib.openTrades())


    result, err, elapsed = run_with_timeout(test_open_trades, timeout_sec=5)
    if err:
        print(f"  ❌ {err}")
    else:
        print(f"  ✅ Got {len(result)} trades in {elapsed:.2f}s (from cache)")

    # Test reqOpenOrders (might freeze)
    print("\n[3.3] Testing ib.reqOpenOrders() (might freeze)...")


    def test_req_open_orders():
        ib.reqOpenOrders()
        time.sleep(1)
        return list(ib.openTrades())


    result, err, elapsed = run_with_timeout(test_req_open_orders, timeout_sec=10)
    if err:
        print(f"  ❌ {err}")
        if "TIMEOUT" in err:
            print("  🚨 reqOpenOrders() causes freeze!")
    else:
        print(f"  ✅ Got {len(result)} trades in {elapsed:.2f}s")

    # Test reqAllOpenOrders (likely freezes)
    print("\n[3.4] Testing ib.reqAllOpenOrders() (likely freezes)...")


    def test_req_all_open_orders():
        ib.reqAllOpenOrders()
        time.sleep(1)
        return list(ib.openTrades())


    result, err, elapsed = run_with_timeout(test_req_all_open_orders, timeout_sec=10)
    if err:
        print(f"  ❌ {err}")
        if "TIMEOUT" in err:
            print("  🚨 reqAllOpenOrders() causes freeze!")
    else:
        print(f"  ✅ Got {len(result)} trades in {elapsed:.2f}s")

    # Cleanup
    print("\n[3.5] Disconnecting...")
    ib.disconnect()
    print("  ✅ Done")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
If any test shows TIMEOUT, that's the problematic call.

SOLUTION for Streamlit:
1. Never call reqAllOpenOrders() or reqOpenOrders()
2. Just use ib.openTrades() directly - it auto-updates
3. For force refresh: disconnect and reconnect

The fix in ibkr_gateway.py should:
- NOT call reqAllOpenOrders()
- Just return ib.openTrades() data directly
- Use reconnect for refresh_all()
""")