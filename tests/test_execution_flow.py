import sys
import os
import time
import json
import logging

# CRITICAL: nest_asyncio FIRST
import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '990'

print("="*70)
print("EXECUTION FLOW SIMULATION (Confirm & Send)")
print("="*70)

# Step 1: Connect
print("\n[1] Connecting to IBKR...")
from dashboard.ai_pm.ibkr_gateway import IbkrGateway
gw = IbkrGateway()
gw.connect(timeout_sec=10)
accounts = gw.list_accounts()
sel_account = accounts[0]
print(f"    Connected, account: {sel_account}")

# Step 2: Load JSON portfolio (simulate session state)
print("\n[2] Loading target portfolio...")
json_path = '../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json'
with open(json_path, 'r') as f:
    target_portfolio = json.load(f)
target_weights = {p['symbol']: p['weight'] / 100.0 for p in target_portfolio}
print(f"    Loaded {len(target_weights)} target weights")

# Step 3: Get snapshot (this part works now)
print("\n[3] Getting portfolio snapshot...")
t = time.time()
from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
snapshot = _to_portfolio_snapshot(gw, sel_account)
print(f"    Snapshot: {len(snapshot.positions)} positions in {time.time()-t:.1f}s")

# Step 4: Load signals
print("\n[4] Loading signals...")
t = time.time()
from dashboard.ai_pm.signal_adapter import load_platform_signals_snapshot
signals, diag = load_platform_signals_snapshot(limit=5000)
print(f"    Signals: {len(signals.rows)} in {time.time()-t:.1f}s")

# Step 5: Build targets from loaded portfolio
print("\n[5] Building targets...")
t = time.time()
from dashboard.ai_pm.models import TargetWeights
from datetime import datetime
targets = TargetWeights(
    weights=target_weights,
    strategy_key='saved_portfolio',
    ts_utc=datetime.utcnow(),
    notes=["Test simulation"]
)
print(f"    Targets: {len(targets.weights)} in {time.time()-t:.1f}s")

# Step 6: Get prices
print("\n[6] Fetching prices...")
t = time.time()
universe = sorted(
    set([p.symbol.strip().upper() for p in snapshot.positions if getattr(p, "symbol", None)])
    | set([k.strip().upper() for k in targets.weights.keys() if k])
)
print(f"    Universe: {len(universe)} symbols")

from src.utils.price_fetcher import get_prices
price_map = get_prices(universe, price_type='close')
print(f"    Prices: {len(price_map)}/{len(universe)} in {time.time()-t:.1f}s")

# Step 7: Build trade plan (THIS MAY BE WHERE IT FREEZES)
print("\n[7] Building trade plan...")
print("    >>> THIS IS WHERE STREAMLIT MAY FREEZE <<<")
t = time.time()
from dashboard.ai_pm.trade_planner import build_trade_plan
from dashboard.ai_pm.config import DEFAULT_CONSTRAINTS

plan = build_trade_plan(
    snapshot=snapshot,
    targets=targets,
    constraints=DEFAULT_CONSTRAINTS,
    price_map=price_map,
)
print(f"    Plan: {len(plan.orders) if plan and plan.orders else 0} orders in {time.time()-t:.1f}s")

# Step 8: Evaluate gates
print("\n[8] Evaluating gates...")
t = time.time()
from dashboard.ai_pm.risk_engine import evaluate_trade_plan_gates

gates = evaluate_trade_plan_gates(
    snapshot=snapshot,
    signals=signals,
    plan=plan,
    constraints=DEFAULT_CONSTRAINTS
)
print(f"    Gates: blocked={gates.blocked} in {time.time()-t:.1f}s")
if gates.block_reasons:
    print(f"    Block reasons: {gates.block_reasons}")
if gates.warnings:
    print(f"    Warnings: {gates.warnings[:3]}")

# Step 9: Execute (DRY RUN - won't actually send orders)
print("\n[9] Executing trade plan (DRY RUN)...")
t = time.time()
from dashboard.ai_pm.execution_engine import execute_trade_plan

execution = execute_trade_plan(
    ib=gw.ib,
    snapshot=snapshot,
    plan=plan,
    account=sel_account,
    constraints=DEFAULT_CONSTRAINTS,
    dry_run=True,  # DRY RUN - no real orders
    kill_switch=False,
    auto_trade_enabled=False,
    armed=False,
)
print(f"    Execution: submitted={execution.submitted} in {time.time()-t:.1f}s")
if execution.errors:
    print(f"    Errors: {execution.errors}")
if execution.notes:
    print(f"    Notes: {execution.notes[:3]}")

# Cleanup
print("\n[10] Cleanup...")
gw.disconnect()

print("\n" + "="*70)
print("✅ EXECUTION SIMULATION COMPLETE")
print("="*70)
