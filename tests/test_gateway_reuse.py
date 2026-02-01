import sys
import time
import os
sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '996'

print("="*70)
print("TESTING GATEWAY REUSE (Simulating Streamlit Session)")
print("="*70)

from dashboard.ai_pm.ibkr_gateway import IbkrGateway

# Create gateway like Streamlit does
print("\n[1] Creating gateway...")
gw = IbkrGateway()
print(f"    ClientId: {gw.cfg.client_id}")

print("\n[2] Connecting...")
success = gw.connect(timeout_sec=10)
print(f"    Success: {success}, is_connected: {gw.is_connected()}")

if not gw.is_connected():
    print("❌ Cannot continue")
    exit(1)

accounts = gw.list_accounts()
account = accounts[0] if accounts else None
print(f"    Account: {account}")

# First call - should work
print("\n[3] First get_account_summary call...")
t = time.time()
try:
    summ = gw.get_account_summary(account)
    print(f"    Got summary in {time.time()-t:.1f}s: NAV={summ.net_liquidation if summ else 'None'}")
except Exception as e:
    print(f"    Error: {e}")

# Simulate time passing (like user doing other things in Streamlit)
print("\n[4] Sleeping 2 seconds (simulating session idle)...")
time.sleep(2)

# Second call - this is what fails in Streamlit
print("\n[5] Second get_account_summary call (after idle)...")
t = time.time()
try:
    summ = gw.get_account_summary(account)
    print(f"    Got summary in {time.time()-t:.1f}s: NAV={summ.net_liquidation if summ else 'None'}")
except Exception as e:
    print(f"    Error: {e}")

# Check connection status
print(f"\n[6] Connection status: {gw.is_connected()}")

# Try positions
print("\n[7] get_positions call...")
t = time.time()
try:
    pos = gw.get_positions()
    print(f"    Got {len(pos)} positions in {time.time()-t:.1f}s")
except Exception as e:
    print(f"    Error: {e}")

# Test the full snapshot
print("\n[8] Full snapshot test...")
t = time.time()
try:
    from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
    snapshot = _to_portfolio_snapshot(gw, account)
    print(f"    Snapshot in {time.time()-t:.1f}s: {len(snapshot.positions) if snapshot else 0} positions")
except Exception as e:
    print(f"    Error: {e}")
    import traceback
    traceback.print_exc()

print("\n[9] Cleanup...")
gw.disconnect()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
