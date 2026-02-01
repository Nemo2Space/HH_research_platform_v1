import sys
import os
import time
import json
import logging

# Configure logging to show the exact same messages as Streamlit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.insert(0, '..')

# Apply nest_asyncio FIRST - exactly like app.py does now
import nest_asyncio
nest_asyncio.apply()

os.environ['IBKR_CLIENT_ID'] = '994'

print("="*70)
print("EXACT STREAMLIT SIMULATION")
print("Testing: INFO:dashboard.ai_pm.ui_tab:Snapshot: getting account summary...")
print("="*70)

# Load the target portfolio JSON
json_path = '../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json'
with open(json_path, 'r') as f:
    target_portfolio = json.load(f)
print(f"\n✅ Loaded target portfolio: {len(target_portfolio)} positions from {json_path}")

# Import gateway - this is what Streamlit does
print("\n[1] Importing IbkrGateway...")
from dashboard.ai_pm.ibkr_gateway import IbkrGateway

# Create gateway - simulating session_state
print("[2] Creating gateway (simulating st.session_state.ai_pm_gateway)...")
gw = IbkrGateway()
print(f"    ClientId: {gw.cfg.client_id}")

# Connect - simulating the Reconnect button
print("[3] Connecting (simulating Reconnect button)...")
success = gw.connect(timeout_sec=10)
print(f"    Success: {success}, is_connected: {gw.is_connected()}")

if not gw.is_connected():
    print("❌ FAILED: Cannot connect")
    exit(1)

accounts = gw.list_accounts()
account = accounts[0] if accounts else None
print(f"    Account: {account}")

# NOW THE CRITICAL PART - this is what freezes
print("\n[4] Calling _to_portfolio_snapshot (THIS IS WHERE IT FREEZES)...")
print("    You should see: INFO:dashboard.ai_pm.ui_tab:Snapshot: getting account summary...")
print("-"*70)

start = time.time()

# Import and call exactly like ui_tab does
from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot

snapshot = _to_portfolio_snapshot(gw, account)

elapsed = time.time() - start
print("-"*70)

if snapshot and snapshot.positions:
    print(f"\n✅ SUCCESS! Snapshot completed in {elapsed:.1f}s")
    print(f"   Positions: {len(snapshot.positions)}")
    print(f"   NAV: {snapshot.net_liquidation:,.0f}")
else:
    print(f"\n❌ FAILED: Snapshot is empty")

# Cleanup
print("\n[5] Cleanup...")
gw.disconnect()

print("\n" + "="*70)
print("TEST COMPLETE - If you saw the log messages and no freeze, Streamlit will work!")
print("="*70)
