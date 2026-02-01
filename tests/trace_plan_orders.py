import sys
import os
import time
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '978'

print("="*70)
print("TRACE WHAT execute_trade_plan RECEIVES AND DOES")
print("="*70)

# First, let's see what a real TradePlan looks like
from dashboard.ai_pm.ibkr_gateway import IbkrGateway
import json

gw = IbkrGateway()
gw.connect(timeout_sec=10)
sel_account = gw.list_accounts()[0]
print(f"Connected, account: {sel_account}")

# Build a real plan
print("\n[1] Building real plan...")
from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
snapshot = _to_portfolio_snapshot(gw, sel_account)
print(f"    Snapshot: {len(snapshot.positions)} positions")

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
print(f"    Prices: {len(price_map)}")

from dashboard.ai_pm.trade_planner import build_trade_plan
from dashboard.ai_pm.config import DEFAULT_CONSTRAINTS
plan = build_trade_plan(
    snapshot=snapshot,
    targets=targets,
    constraints=DEFAULT_CONSTRAINTS,
    price_map=price_map,
)

print(f"\n[2] Plan details:")
print(f"    Type of plan: {type(plan)}")
print(f"    Type of plan.orders: {type(plan.orders)}")
print(f"    Number of orders: {len(plan.orders) if plan.orders else 0}")

if plan.orders:
    print(f"\n[3] First order details:")
    o = plan.orders[0]
    print(f"    Type: {type(o)}")
    print(f"    Dir: {dir(o)}")
    print(f"    Repr: {o}")
    
    # Try to access attributes
    for attr in ['symbol', 'action', 'quantity', 'order_type', 'side', 'qty', 'ticker']:
        val = getattr(o, attr, 'NOT_FOUND')
        print(f"    {attr}: {val}")

# Now let's trace into execute_trade_plan
print(f"\n[4] Tracing execute_trade_plan...")

# Read the execution engine to see the order processing loop
with open('../dashboard/ai_pm/execution_engine.py', 'r') as f:
    content = f.read()

# Find the loop that processes orders
idx = content.find('for o in orders_all')
if idx > 0:
    print("Order processing loop:")
    print(content[idx:idx+1000])

gw.disconnect()
