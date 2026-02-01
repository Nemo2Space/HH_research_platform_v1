import sys
import os
import time
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '975'

print("="*70)
print("TRACE EVERY ORDER SUBMISSION")
print("="*70)

from dashboard.ai_pm.ibkr_gateway import IbkrGateway
import json

gw = IbkrGateway()
gw.connect(timeout_sec=10)
sel_account = gw.list_accounts()[0]
print(f"Connected, account: {sel_account}")

# Build full plan
from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
snapshot = _to_portfolio_snapshot(gw, sel_account)

json_path = '../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json'
with open(json_path, 'r') as f:
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

universe = sorted(
    set([p.symbol.strip().upper() for p in snapshot.positions if getattr(p, "symbol", None)])
    | set([k.strip().upper() for k in targets.weights.keys() if k])
)
from src.utils.price_fetcher import get_prices
price_map = get_prices(universe, price_type='close')

from dashboard.ai_pm.trade_planner import build_trade_plan
from dashboard.ai_pm.config import DEFAULT_CONSTRAINTS
plan = build_trade_plan(
    snapshot=snapshot,
    targets=targets,
    constraints=DEFAULT_CONSTRAINTS,
    price_map=price_map,
)

print(f"\nPlan has {len(plan.orders)} orders")

# Check open orders BEFORE
print(f"Open orders BEFORE: {len(gw.ib.openTrades())}")

# Execute
from dashboard.ai_pm.execution_engine import execute_trade_plan

print("\n>>> Executing... <<<")
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

print(f"\n>>> Execution returned <<<")
print(f"submitted: {execution.submitted}")
print(f"submitted_orders count: {len(execution.submitted_orders) if execution.submitted_orders else 0}")
print(f"errors: {execution.errors}")

# Show first 10 submitted orders
if execution.submitted_orders:
    print(f"\nFirst 10 submitted orders:")
    for o in execution.submitted_orders[:10]:
        print(f"  {o}")

# Wait and check
print(f"\nWaiting 3 seconds...")
time.sleep(3)

# Check open orders AFTER
open_trades = gw.ib.openTrades()
print(f"\nOpen orders AFTER: {len(open_trades)}")

if len(open_trades) < len(execution.submitted_orders or []):
    print(f"\n⚠️ MISMATCH: submitted {len(execution.submitted_orders or [])} but only {len(open_trades)} open")

for t in open_trades[:20]:
    print(f"  {t.contract.symbol} {t.order.action} {t.order.totalQuantity} - {t.orderStatus.status}")

# DON'T cancel - let user verify in TWS
print("\n" + "="*70)
print("NOT CANCELLING - Check TWS now!")
print("Press Enter to cancel all orders and disconnect...")
print("="*70)

input()

# Now cancel
print("Cancelling all orders...")
for t in gw.ib.openTrades():
    try:
        gw.ib.cancelOrder(t.order)
    except:
        pass

time.sleep(2)
gw.disconnect()
print("Done")
