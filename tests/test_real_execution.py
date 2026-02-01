import sys
import os
import time
import json
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '988'

print("="*70)
print("REAL EXECUTION TEST (dry_run=FALSE - WILL ACTUALLY SEND ORDERS)")
print("="*70)
print("\n⚠️  WARNING: This will send REAL orders to IBKR!")
print("    Press Ctrl+C within 5 seconds to abort...")
time.sleep(5)

# Connect
print("\n[1] Connecting...")
from dashboard.ai_pm.ibkr_gateway import IbkrGateway
gw = IbkrGateway()
gw.connect(timeout_sec=10)
sel_account = gw.list_accounts()[0]
print(f"    Account: {sel_account}")

# Snapshot
print("\n[2] Snapshot...")
t = time.time()
from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
snapshot = _to_portfolio_snapshot(gw, sel_account)
print(f"    {len(snapshot.positions)} positions in {time.time()-t:.1f}s")

# Signals
print("\n[3] Signals...")
t = time.time()
from dashboard.ai_pm.signal_adapter import load_platform_signals_snapshot
signals, diag = load_platform_signals_snapshot(limit=5000)
print(f"    {len(signals.rows)} signals in {time.time()-t:.1f}s")

# Targets from JSON
print("\n[4] Loading targets from JSON...")
with open('../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json', 'r') as f:
    target_portfolio = json.load(f)
target_weights = {p['symbol']: p['weight'] / 100.0 for p in target_portfolio}

from dashboard.ai_pm.models import TargetWeights
from datetime import datetime, timezone
targets = TargetWeights(
    weights=target_weights,
    strategy_key='saved_portfolio',
    ts_utc=datetime.now(timezone.utc),
    notes=["Test"]
)
print(f"    {len(targets.weights)} targets")

# Prices
print("\n[5] Fetching prices...")
t = time.time()
universe = sorted(
    set([p.symbol.strip().upper() for p in snapshot.positions if getattr(p, "symbol", None)])
    | set([k.strip().upper() for k in targets.weights.keys() if k])
)
from src.utils.price_fetcher import get_prices
price_map = get_prices(universe, price_type='close')
print(f"    {len(price_map)}/{len(universe)} prices in {time.time()-t:.1f}s")

# Trade Plan
print("\n[6] Building trade plan...")
t = time.time()
from dashboard.ai_pm.trade_planner import build_trade_plan
from dashboard.ai_pm.config import DEFAULT_CONSTRAINTS
plan = build_trade_plan(
    snapshot=snapshot,
    targets=targets,
    constraints=DEFAULT_CONSTRAINTS,
    price_map=price_map,
)
print(f"    {len(plan.orders) if plan and plan.orders else 0} orders in {time.time()-t:.1f}s")

# Gates
print("\n[7] Evaluating gates...")
t = time.time()
from dashboard.ai_pm.risk_engine import evaluate_trade_plan_gates
gates = evaluate_trade_plan_gates(
    snapshot=snapshot,
    signals=signals,
    plan=plan,
    constraints=DEFAULT_CONSTRAINTS
)
print(f"    blocked={gates.blocked} in {time.time()-t:.1f}s")

if gates.blocked:
    print("    ❌ BLOCKED - cannot execute")
    print(f"    Reasons: {gates.block_reasons}")
    gw.disconnect()
    exit(1)

# Execute with dry_run=FALSE (REAL ORDERS)
print("\n[8] Executing (dry_run=FALSE - REAL ORDERS)...")
print("    >>> THIS IS EXACTLY WHAT STREAMLIT DOES <<<")
t = time.time()

from dashboard.ai_pm.execution_engine import execute_trade_plan

print("    Calling execute_trade_plan...")
execution = execute_trade_plan(
    ib=gw.ib,
    snapshot=snapshot,
    plan=plan,
    account=sel_account,
    constraints=DEFAULT_CONSTRAINTS,
    dry_run=False,  # REAL ORDERS
    kill_switch=False,
    auto_trade_enabled=False,
    armed=False,
    price_map=price_map,
    skip_live_quotes=True,
)
exec_time = time.time() - t
print(f"    completed in {exec_time:.1f}s")
print(f"    submitted={execution.submitted}")
print(f"    orders sent: {len(execution.submitted_orders) if execution.submitted_orders else 0}")

if execution.errors:
    print(f"    Errors: {execution.errors[:5]}")
if execution.notes:
    print(f"    Notes: {execution.notes[:5]}")

gw.disconnect()

print("\n" + "="*70)
print(f"Execution time: {exec_time:.1f}s")
print("="*70)
