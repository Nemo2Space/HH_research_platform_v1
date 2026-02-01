import sys
import os
import time
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '970'

print("="*70)
print("DIRECT TEST WITH EXPLICIT dry_run=False")
print("="*70)

from dashboard.ai_pm.ibkr_gateway import IbkrGateway
import json

gw = IbkrGateway()
gw.connect(timeout_sec=10)
sel_account = gw.list_accounts()[0]
print(f"Connected: {sel_account}")

# Build plan
from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
snapshot = _to_portfolio_snapshot(gw, sel_account)

json_path = '../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json'
with open(json_path, 'r') as f:
    target_portfolio = json.load(f)
target_weights = {p['symbol']: p['weight'] / 100.0 for p in target_portfolio}

from dashboard.ai_pm.models import TargetWeights, TradePlan
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
from dashboard.ai_pm.config import DEFAULT_CONSTRAINTS, RiskConstraints

# Use very relaxed constraints
relaxed = RiskConstraints(
    max_position_weight=0.20,
    max_sector_weight=1.00,
    cash_min=-1.00,  # Allow -100% cash
    cash_max=1.00,
    max_turnover_per_cycle=1.00,
    max_trades_per_cycle=200,
)

full_plan = build_trade_plan(
    snapshot=snapshot,
    targets=targets,
    constraints=relaxed,
    price_map=price_map,
)

# Create a new plan with just 3 orders
limited_orders = list(full_plan.orders)[:3]
plan = TradePlan(
    ts_utc=full_plan.ts_utc,
    account=full_plan.account,
    strategy_key=full_plan.strategy_key,
    nav=full_plan.nav,
    cash=full_plan.cash,
    orders=limited_orders,
)

print(f"\nTesting with {len(plan.orders)} orders:")
for o in plan.orders:
    print(f"  {o.action} {o.quantity} {o.symbol}")

print(f"\nOpen orders BEFORE: {len(gw.ib.openTrades())}")

# Execute with EXPLICIT dry_run=False
print("\n>>> Calling execute_trade_plan with dry_run=False <<<")

from dashboard.ai_pm.execution_engine import execute_trade_plan

execution = execute_trade_plan(
    ib=gw.ib,
    snapshot=snapshot,
    plan=plan,
    account=sel_account,
    constraints=relaxed,
    dry_run=False,  # EXPLICIT False
    kill_switch=False,
    auto_trade_enabled=False,
    armed=False,
    price_map=price_map,
    skip_live_quotes=True,
)

print(f"\n>>> Result <<<")
print(f"submitted: {execution.submitted}")
print(f"submitted_orders: {len(execution.submitted_orders) if execution.submitted_orders else 0}")

if execution.submitted_orders:
    print("\nOrder details:")
    for o in execution.submitted_orders:
        print(f"  {o.get('symbol')}: submitted={o.get('submitted')}, dry_run={o.get('dry_run')}, status={o.get('status')}")

print(f"errors: {execution.errors}")
print(f"notes: {execution.notes}")

time.sleep(2)

open_trades = gw.ib.openTrades()
print(f"\nOpen orders AFTER: {len(open_trades)}")
for t in open_trades:
    print(f"  {t.contract.symbol} {t.order.action} {t.order.totalQuantity} - {t.orderStatus.status}")

print("\nCancelling...")
for t in open_trades:
    try:
        gw.ib.cancelOrder(t.order)
    except:
        pass

time.sleep(1)
gw.disconnect()
print("Done")
