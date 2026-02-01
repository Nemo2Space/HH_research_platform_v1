import sys
import time
import os
import json
sys.path.insert(0, '..')

# Apply nest_asyncio FIRST - just like app.py now does
import nest_asyncio
nest_asyncio.apply()

os.environ['IBKR_CLIENT_ID'] = '995'

print("="*70)
print("STREAMLIT SIMULATION TEST (with loaded portfolio)")
print("="*70)

# Load the target portfolio
with open('../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json', 'r') as f:
    target_portfolio = json.load(f)
print(f"Loaded target portfolio: {len(target_portfolio)} positions")

total_start = time.time()

# Step 1: Connection + Snapshot (this is where it freezes)
print("\n⏳ Step 1/5: Connecting to IBKR and loading snapshot...")
step_start = time.time()

from dashboard.ai_pm.ibkr_gateway import IbkrGateway
print("   Gateway imported...")

gw = IbkrGateway()
print(f"   Gateway created with clientId {gw.cfg.client_id}...")

success = gw.connect(timeout_sec=10)
print(f"   Connect result: {success}, is_connected: {gw.is_connected()}")

if not gw.is_connected():
    print("❌ FAILED: Cannot connect to IBKR")
    exit(1)

accounts = gw.list_accounts()
print(f"   Accounts: {accounts}")
account = accounts[0]

# This is the exact call that freezes
print("   Calling _to_portfolio_snapshot...")
from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
snapshot = _to_portfolio_snapshot(gw, account)
print(f"✅ Step 1: {time.time()-step_start:.1f}s | {len(snapshot.positions)} positions, NAV={snapshot.net_liquidation:,.0f}")

# Step 2-3: Signals
print("\n⏳ Step 2-3/5: Loading signals...")
step_start = time.time()
from dashboard.ai_pm.signal_adapter import load_platform_signals_snapshot
signals, diag = load_platform_signals_snapshot(limit=500)
print(f"✅ Step 2-3: {time.time()-step_start:.1f}s | {len(signals.rows)} signals")

# Step 4: Prices
print("\n⏳ Step 4/5: Fetching prices...")
step_start = time.time()
symbols = [p.symbol for p in snapshot.positions if p.symbol]
from src.utils.price_fetcher import get_prices
prices = get_prices(symbols, price_type='close')
print(f"✅ Step 4: {time.time()-step_start:.1f}s | {len(prices)}/{len(symbols)} prices")

# Step 5: Build Plan with loaded targets
print("\n⏳ Step 5/5: Building trade plan (using loaded portfolio)...")
step_start = time.time()
try:
    from dashboard.ai_pm.trade_planner import build_trade_plan
    from dashboard.ai_pm.target_builder import TargetWeights
    from dashboard.ai_pm.ui_tab import DEFAULT_CONSTRAINTS
    
    # Build targets from loaded JSON
    weights = {p['symbol']: p['weight'] / 100.0 for p in target_portfolio}
    targets = TargetWeights(weights=weights, diagnostics={})
    
    plan = build_trade_plan(
        snapshot=snapshot,
        targets=targets,
        constraints=DEFAULT_CONSTRAINTS,
        price_map=prices,
    )
    
    order_count = len(plan.orders) if plan and plan.orders else 0
    print(f"✅ Step 5: {time.time()-step_start:.1f}s | {len(weights)} targets, {order_count} orders")
except Exception as e:
    print(f"❌ Step 5: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
gw.disconnect()

total = time.time() - total_start
print("\n" + "="*70)
print(f"TOTAL: {total:.1f}s")
if total < 15:
    print("🚀 EXCELLENT - Ready for Streamlit!")
print("="*70)
