import sys
import os
import time
import json
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '986'

print("="*70)
print("STREAMLIT SESSION SIMULATION")
print("Simulating: Gateway in session_state, then Confirm clicked")
print("="*70)

# ============================================================
# SIMULATE SESSION START: Gateway connects and stays connected
# ============================================================
print("\n[SESSION START] Gateway connects (like page load)...")

from dashboard.ai_pm.ibkr_gateway import IbkrGateway
gw = IbkrGateway()
gw.connect(timeout_sec=10)
sel_account = gw.list_accounts()[0]
print(f"    Connected, account: {sel_account}")

# Simulate session_state variables
print("\n[SESSION STATE] Storing gateway and loading portfolio...")
session_state = {
    'ai_pm_gateway': gw,
    'ai_pm_connection_ok': True,
}

# Load JSON portfolio into session_state
json_path = '../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json'
with open(json_path, 'r') as f:
    target_portfolio = json.load(f)
session_state['ai_pm_target_weights'] = {p['symbol']: p['weight'] / 100.0 for p in target_portfolio}
session_state['ai_pm_target_portfolio'] = {'name': 'KAIF_SC_2026_Q1'}

# ============================================================
# SIMULATE USER ACTIVITY: Time passes, maybe some other actions
# ============================================================
print("\n[USER ACTIVITY] Simulating 5 seconds of browsing...")
time.sleep(5)

# ============================================================
# SIMULATE CONFIRM BUTTON: Get gw from session_state
# ============================================================
print("\n[CONFIRM CLICKED] Getting gateway from session_state...")

# This is what Streamlit does - gets gw from session_state
gw = session_state['ai_pm_gateway']
print(f"    Gateway from session_state: {gw}")
print(f"    is_connected: {gw.is_connected()}")
print(f"    ib.isConnected: {gw.ib.isConnected()}")

# ============================================================
# NOW RUN THE EXACT EXECUTION CODE
# ============================================================
print("\n[EXECUTION] Running exact Streamlit code...")

from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
from dashboard.ai_pm.signal_adapter import load_platform_signals_snapshot
from dashboard.ai_pm.models import TargetWeights
from dashboard.ai_pm.trade_planner import build_trade_plan
from dashboard.ai_pm.config import DEFAULT_CONSTRAINTS
from dashboard.ai_pm.risk_engine import evaluate_trade_plan_gates
from dashboard.ai_pm.execution_engine import execute_trade_plan
from datetime import datetime, timezone

print("    Getting snapshot...")
snapshot = _to_portfolio_snapshot(gw, sel_account)
print(f"    Snapshot: {len(snapshot.positions)} positions")

print("    Loading signals...")
signals, diag = load_platform_signals_snapshot(limit=5000)
print(f"    Signals: {len(signals.rows)}")

print("    Building targets...")
targets = TargetWeights(
    weights=session_state['ai_pm_target_weights'],
    strategy_key='saved_portfolio',
    ts_utc=datetime.now(timezone.utc),
    notes=["Test"]
)

print("    Fetching prices...")
universe = sorted(
    set([p.symbol.strip().upper() for p in snapshot.positions if getattr(p, "symbol", None)])
    | set([k.strip().upper() for k in targets.weights.keys() if k])
)
from src.utils.price_fetcher import get_prices
price_map = get_prices(universe, price_type='close')
print(f"    Prices: {len(price_map)}/{len(universe)}")

print("    Building trade plan...")
plan = build_trade_plan(
    snapshot=snapshot,
    targets=targets,
    constraints=DEFAULT_CONSTRAINTS,
    price_map=price_map,
)
print(f"    Plan: {len(plan.orders) if plan and plan.orders else 0} orders")

print("    Evaluating gates...")
gates = evaluate_trade_plan_gates(
    snapshot=snapshot,
    signals=signals,
    plan=plan,
    constraints=DEFAULT_CONSTRAINTS
)
print(f"    Gates: blocked={gates.blocked}")

print("\n    >>> CALLING execute_trade_plan (this is where Streamlit freezes) <<<")
print(f"    IB connection check: {gw.ib.isConnected()}")
print(f"    IB clientId: {gw.ib.client.clientId}")

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
print(f"    Execution returned in {time.time()-t:.1f}s")
print(f"    submitted={execution.submitted}, orders={len(execution.submitted_orders) if execution.submitted_orders else 0}")

gw.disconnect()
print("\n✅ TEST COMPLETE")
