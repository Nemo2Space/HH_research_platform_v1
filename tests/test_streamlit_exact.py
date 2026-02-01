import sys
import os
import time
import json
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '965'

print("="*70)
print("EXACT STREAMLIT FLOW SIMULATION")
print("This simulates what happens when you click Confirm & Send")
print("="*70)

from dashboard.ai_pm.ibkr_gateway import IbkrGateway

# Step 1: Connect (like session_state)
print("\n[1] Connecting...")
gw = IbkrGateway()
gw.connect(timeout_sec=10)
sel_account = gw.list_accounts()[0]
print(f"    Connected: {sel_account}")

# Step 2: Load JSON portfolio
print("\n[2] Loading JSON portfolio...")
json_path = '../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json'
with open(json_path, 'r') as f:
    target_portfolio = json.load(f)
target_weights = {p['symbol']: p['weight'] / 100.0 for p in target_portfolio}
print(f"    {len(target_weights)} targets")

# Step 3: Get snapshot (EXACT code from ui_tab.py)
print("\n[3] Getting snapshot...")
from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
snapshot = _to_portfolio_snapshot(gw, sel_account)
print(f"    {len(snapshot.positions)} positions, NAV={snapshot.net_liquidation}")

# Step 4: Load signals
print("\n[4] Loading signals...")
from dashboard.ai_pm.signal_adapter import load_platform_signals_snapshot
signals, diag = load_platform_signals_snapshot(limit=5000)
print(f"    {len(signals.rows)} signals")

# Step 5: Build targets
print("\n[5] Building targets...")
from dashboard.ai_pm.models import TargetWeights
from datetime import datetime, timezone
targets = TargetWeights(
    weights=target_weights,
    strategy_key='saved_portfolio',
    ts_utc=datetime.now(timezone.utc),
    notes=["Test"]
)

# Step 6: Fetch prices (EXACT code from ui_tab.py)
print("\n[6] Fetching prices...")
universe = sorted(
    set([p.symbol.strip().upper() for p in snapshot.positions if getattr(p, "symbol", None)])
    | set([k.strip().upper() for k in targets.weights.keys() if k])
)
from src.utils.price_fetcher import get_prices
price_map = get_prices(universe, price_type='close')
print(f"    {len(price_map)}/{len(universe)} prices")

# Step 7: Build trade plan
print("\n[7] Building trade plan...")
from dashboard.ai_pm.trade_planner import build_trade_plan
from dashboard.ai_pm.config import DEFAULT_CONSTRAINTS
plan = build_trade_plan(
    snapshot=snapshot,
    targets=targets,
    constraints=DEFAULT_CONSTRAINTS,
    price_map=price_map,
)
print(f"    {len(plan.orders)} orders")

# Step 8: Evaluate gates
print("\n[8] Evaluating gates...")
from dashboard.ai_pm.risk_engine import evaluate_trade_plan_gates
gates = evaluate_trade_plan_gates(
    snapshot=snapshot,
    signals=signals,
    plan=plan,
    constraints=DEFAULT_CONSTRAINTS
)
print(f"    blocked={gates.blocked}")

# Step 9: EXECUTE (this is where it freezes in Streamlit)
print("\n" + "="*70)
print("[9] EXECUTING - THIS IS WHERE STREAMLIT FREEZES")
print("="*70)

from dashboard.ai_pm.execution_engine import execute_trade_plan

print(f"\n    Pre-checks:")
print(f"      gw.ib.isConnected(): {gw.ib.isConnected()}")
print(f"      gates.blocked: {gates.blocked}")
print(f"      orders: {len(plan.orders)}")
print(f"      price_map: {len(price_map)}")

print(f"\n    >>> Calling execute_trade_plan... <<<")
t = time.time()

execution = execute_trade_plan(
    ib=gw.ib,
    snapshot=snapshot,
    plan=plan,
    account=sel_account,
    constraints=DEFAULT_CONSTRAINTS,
    dry_run=False,
    kill_switch=False,
    auto_trade_enabled=False,
    armed=False,
    price_map=price_map,
    skip_live_quotes=True,
)

elapsed = time.time() - t
print(f"\n    >>> execute_trade_plan returned in {elapsed:.1f}s <<<")
print(f"    submitted: {execution.submitted}")
print(f"    orders: {len(execution.submitted_orders) if execution.submitted_orders else 0}")
print(f"    errors: {execution.errors}")

# Step 10: Verify orders in TWS
print("\n[10] Verifying orders...")
time.sleep(1)
open_trades = gw.ib.openTrades()
print(f"    Open orders: {len(open_trades)}")

# Cancel all
print("\n[11] Cancelling all orders...")
gw.ib.reqGlobalCancel()
gw.ib.sleep(3)

gw.disconnect()
print("\n✅ TEST COMPLETE")
