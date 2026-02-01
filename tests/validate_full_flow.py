import sys
import os
import time
import json
import logging

import nest_asyncio
nest_asyncio.apply()

# Enable DEBUG logging to see EVERYTHING
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '964'

print("="*70)
print("COMPLETE VALIDATION TEST - SIMULATING STREAMLIT")
print("="*70)

# Import everything upfront like Streamlit would
from dashboard.ai_pm.ibkr_gateway import IbkrGateway
from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
from dashboard.ai_pm.signal_adapter import load_platform_signals_snapshot
from dashboard.ai_pm.models import TargetWeights
from dashboard.ai_pm.trade_planner import build_trade_plan
from dashboard.ai_pm.config import DEFAULT_CONSTRAINTS
from dashboard.ai_pm.risk_engine import evaluate_trade_plan_gates
from dashboard.ai_pm.execution_engine import execute_trade_plan
from src.utils.price_fetcher import get_prices
from datetime import datetime, timezone

print("\n[1] CONNECT")
print("-"*50)
t0 = time.time()
gw = IbkrGateway()
gw.connect(timeout_sec=10)
sel_account = gw.list_accounts()[0]
print(f"    ✅ Connected in {time.time()-t0:.1f}s: {sel_account}")

print("\n[2] LOAD JSON")
print("-"*50)
t0 = time.time()
json_path = '../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json'
with open(json_path, 'r') as f:
    target_portfolio = json.load(f)
target_weights = {p['symbol']: p['weight'] / 100.0 for p in target_portfolio}
print(f"    ✅ Loaded in {time.time()-t0:.1f}s: {len(target_weights)} targets")

print("\n[3] SNAPSHOT")
print("-"*50)
t0 = time.time()
snapshot = _to_portfolio_snapshot(gw, sel_account)
print(f"    ✅ Snapshot in {time.time()-t0:.1f}s: {len(snapshot.positions)} positions, NAV={snapshot.net_liquidation}")

print("\n[4] SIGNALS")
print("-"*50)
t0 = time.time()
signals, diag = load_platform_signals_snapshot(limit=5000)
print(f"    ✅ Signals in {time.time()-t0:.1f}s: {len(signals.rows)} signals")

print("\n[5] TARGETS")
print("-"*50)
t0 = time.time()
targets = TargetWeights(
    weights=target_weights,
    strategy_key='saved_portfolio',
    ts_utc=datetime.now(timezone.utc),
    notes=["Test"]
)
print(f"    ✅ Targets in {time.time()-t0:.1f}s")

print("\n[6] PRICES")
print("-"*50)
t0 = time.time()
universe = sorted(
    set([p.symbol.strip().upper() for p in snapshot.positions if getattr(p, "symbol", None)])
    | set([k.strip().upper() for k in targets.weights.keys() if k])
)
price_map = get_prices(universe, price_type='close')
print(f"    ✅ Prices in {time.time()-t0:.1f}s: {len(price_map)}/{len(universe)}")

print("\n[7] TRADE PLAN")
print("-"*50)
t0 = time.time()
plan = build_trade_plan(
    snapshot=snapshot,
    targets=targets,
    constraints=DEFAULT_CONSTRAINTS,
    price_map=price_map,
)
print(f"    ✅ Plan in {time.time()-t0:.1f}s: {len(plan.orders)} orders")

print("\n[8] GATES")
print("-"*50)
t0 = time.time()
gates = evaluate_trade_plan_gates(
    snapshot=snapshot,
    signals=signals,
    plan=plan,
    constraints=DEFAULT_CONSTRAINTS
)
print(f"    ✅ Gates in {time.time()-t0:.1f}s: blocked={gates.blocked}")

print("\n[9] EXECUTE")
print("-"*50)
print(f"    Pre-flight checks:")
print(f"      - IB connected: {gw.ib.isConnected()}")
print(f"      - Client ID: {gw.ib.client.clientId}")
print(f"      - gates.blocked: {gates.blocked}")
print(f"      - orders count: {len(plan.orders)}")
print(f"      - price_map count: {len(price_map)}")
print(f"      - skip_live_quotes: True")

print(f"\n    >>> Calling execute_trade_plan NOW <<<")
t0 = time.time()

try:
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
    elapsed = time.time() - t0
    print(f"\n    ✅ Execution completed in {elapsed:.1f}s")
    print(f"       submitted: {execution.submitted}")
    print(f"       orders: {len(execution.submitted_orders) if execution.submitted_orders else 0}")
    print(f"       errors: {execution.errors[:3] if execution.errors else []}")
except Exception as e:
    elapsed = time.time() - t0
    print(f"\n    ❌ Execution FAILED after {elapsed:.1f}s")
    print(f"       Error: {e}")
    import traceback
    traceback.print_exc()

print("\n[10] VERIFY TWS")
print("-"*50)
time.sleep(2)
open_trades = gw.ib.openTrades()
print(f"    Open orders in TWS: {len(open_trades)}")
if open_trades:
    for t in open_trades[:5]:
        print(f"      {t.contract.symbol} {t.order.action} {t.order.totalQuantity} - {t.orderStatus.status}")

print("\n[11] CANCEL ALL")
print("-"*50)
gw.ib.reqGlobalCancel()
gw.ib.sleep(3)
print(f"    Cancelled")

gw.disconnect()

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)
