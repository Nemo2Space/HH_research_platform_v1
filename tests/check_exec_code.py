import sys
import os
import time
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '973'

print("="*70)
print("COMPARE: MANUAL vs EXECUTION ENGINE")
print("="*70)

from ib_insync import Stock, MarketOrder
from dashboard.ai_pm.ibkr_gateway import IbkrGateway
import json

gw = IbkrGateway()
gw.connect(timeout_sec=10)
sel_account = gw.list_accounts()[0]
print(f"Connected, account: {sel_account}")

# Build plan
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

# Read the execution_engine source to understand the flow
print("\n" + "="*70)
print("CHECKING EXECUTION ENGINE CODE")
print("="*70)

with open('../dashboard/ai_pm/execution_engine.py', 'r') as f:
    content = f.read()

# Find where placeOrder is called
idx = content.find('placeOrder')
print(f"\nplaceOrder found at index: {idx}")

# Show surrounding code
if idx > 0:
    start = max(0, idx - 200)
    end = min(len(content), idx + 300)
    print(f"\nCode around placeOrder:")
    print(content[start:end])

# Now check if dry_run is being respected
print("\n" + "="*70)
print("CHECKING dry_run LOGIC")
print("="*70)

idx = content.find('dry_run')
print(f"\ndry_run found at index: {idx}")

# Find all occurrences
import re
dry_run_matches = list(re.finditer(r'dry_run', content))
print(f"Total dry_run occurrences: {len(dry_run_matches)}")

for m in dry_run_matches[:10]:
    start = max(0, m.start() - 50)
    end = min(len(content), m.end() + 100)
    line = content[start:end].replace('\n', ' ')
    print(f"  ...{line}...")

gw.disconnect()
