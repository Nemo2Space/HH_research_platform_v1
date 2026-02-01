import sys
import os
import time
import json
import logging

# CRITICAL: This must be FIRST - exactly like app.py
import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '993'

print("="*70)
print("EXACT STREAMLIT FLOW SIMULATION")
print("="*70)

# ============================================
# SIMULATE: Page Load - Gateway Initialization
# ============================================
print("\n[SIMULATE PAGE LOAD]")
print("Creating gateway (like session_state init)...")

from dashboard.ai_pm.ibkr_gateway import IbkrGateway

# This is what happens on page load
gw = IbkrGateway()
print(f"Gateway created with clientId: {gw.cfg.client_id}")

# Connect (like the auto-connect on page load)
print("Connecting...")
success = gw.connect(timeout_sec=10)
print(f"Connected: {success}, is_connected: {gw.is_connected()}")

if not gw.is_connected():
    print("❌ Connection failed")
    exit(1)

accounts = gw.list_accounts()
sel_account = accounts[0] if accounts else None
print(f"Account: {sel_account}")

# ============================================
# SIMULATE: User Loads JSON Portfolio
# ============================================
print("\n[SIMULATE USER LOADS JSON]")
json_path = '../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json'
with open(json_path, 'r') as f:
    target_portfolio = json.load(f)
print(f"Loaded: {len(target_portfolio)} positions from {json_path}")

# Build target weights (like Streamlit does)
target_weights = {p['symbol']: p['weight'] / 100.0 for p in target_portfolio}
print(f"Built target weights for {len(target_weights)} stocks")

# Simulate some time passing (user looking at UI)
print("\nSimulating 3 seconds of user browsing...")
time.sleep(3)

# ============================================
# SIMULATE: User Clicks "Run Now"
# ============================================
print("\n[SIMULATE USER CLICKS 'RUN NOW']")
print("-"*70)

# This is the exact code from ui_tab.py lines 3454-3482
print("Verifying connection...")
try:
    test_accounts = gw.list_accounts()
    if not test_accounts:
        print("❌ IBKR connection stale")
        exit(1)
    print(f"Connection OK, accounts: {test_accounts}")
except Exception as e:
    print(f"❌ Connection error: {e}")
    exit(1)

print("\nVerifying with ping...")
try:
    if not gw.ping():
        print("❌ Ping failed")
        exit(1)
    print("Ping OK")
except Exception as e:
    print(f"❌ Ping error: {e}")
    exit(1)

print("\n⏳ Step 1/5: Loading portfolio snapshot...")
print(">>> CALLING _to_portfolio_snapshot - THIS IS WHERE STREAMLIT FREEZES <<<")
print("-"*70)

start = time.time()
try:
    from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
    snapshot = _to_portfolio_snapshot(gw, sel_account)
    elapsed = time.time() - start
    print("-"*70)
    print(f"✅ Snapshot completed in {elapsed:.1f}s")
    print(f"   Positions: {len(snapshot.positions) if snapshot and snapshot.positions else 0}")
    print(f"   NAV: {snapshot.net_liquidation:,.0f}" if snapshot else "   NAV: N/A")
except Exception as e:
    print(f"❌ Snapshot FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================
# Continue with rest of pipeline
# ============================================
print("\n⏳ Step 2/5: Loading signals...")
from dashboard.ai_pm.signal_adapter import load_platform_signals_snapshot
signals, diag = load_platform_signals_snapshot(limit=500)
print(f"✅ Signals: {len(signals.rows) if signals else 0}")

print("\n⏳ Step 3/5: Fetching prices...")
symbols = [p.symbol for p in snapshot.positions if p.symbol]
from src.utils.price_fetcher import get_prices
prices = get_prices(symbols, price_type='close')
print(f"✅ Prices: {len(prices)}/{len(symbols)}")

# Cleanup
print("\n[CLEANUP]")
gw.disconnect()

print("\n" + "="*70)
print("✅ FULL SIMULATION COMPLETE - Streamlit should work now!")
print("="*70)
