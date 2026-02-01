import sys
import os
import time
import json
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '969'

print("="*70)
print("FULL PIPELINE TEST WITH RELAXED CONSTRAINTS")
print("="*70)

from dashboard.ai_pm.ibkr_gateway import IbkrGateway

gw = IbkrGateway()
gw.connect(timeout_sec=10)
sel_account = gw.list_accounts()[0]
print(f"Connected: {sel_account}")

# Snapshot
from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
snapshot = _to_portfolio_snapshot(gw, sel_account)
print(f"Snapshot: {len(snapshot.positions)} positions, NAV={snapshot.net_liquidation}")

# Targets
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
print(f"Targets: {len(targets.weights)}")

# Prices
universe = sorted(
    set([p.symbol.strip().upper() for p in snapshot.positions if getattr(p, "symbol", None)])
    | set([k.strip().upper() for k in targets.weights.keys() if k])
)
from src.utils.price_fetcher import get_prices
price_map = get_prices(universe, price_type='close')
print(f"Prices: {len(price_map)}/{len(universe)}")

# RELAXED CONSTRAINTS - This is the key!
from dashboard.ai_pm.config import RiskConstraints

RELAXED = RiskConstraints(
    max_position_weight=0.20,
    max_sector_weight=1.00,
    cash_min=-1.00,  # Allow -100% cash (no cash constraint)
    cash_max=1.00,
    max_turnover_per_cycle=1.00,
    max_trades_per_cycle=200,
)

# Build plan
from dashboard.ai_pm.trade_planner import build_trade_plan
plan = build_trade_plan(
    snapshot=snapshot,
    targets=targets,
    constraints=RELAXED,
    price_map=price_map,
)
print(f"Plan: {len(plan.orders)} orders")

# Gates
from dashboard.ai_pm.signal_adapter import load_platform_signals_snapshot
signals, _ = load_platform_signals_snapshot(limit=5000)

from dashboard.ai_pm.risk_engine import evaluate_trade_plan_gates
gates = evaluate_trade_plan_gates(
    snapshot=snapshot,
    signals=signals,
    plan=plan,
    constraints=RELAXED
)
print(f"Gates: blocked={gates.blocked}")

if gates.blocked:
    print(f"Block reasons: {gates.block_reasons}")

# Execute
print(f"\nOpen orders BEFORE: {len(gw.ib.openTrades())}")

from dashboard.ai_pm.execution_engine import execute_trade_plan

print("\n>>> EXECUTING ALL ORDERS <<<")
t = time.time()

execution = execute_trade_plan(
    ib=gw.ib,
    snapshot=snapshot,
    plan=plan,
    account=sel_account,
    constraints=RELAXED,
    dry_run=False,
    kill_switch=False,
    auto_trade_enabled=False,
    armed=False,
    price_map=price_map,
    skip_live_quotes=True,
)

elapsed = time.time() - t
print(f"\n>>> Execution completed in {elapsed:.1f}s <<<")
print(f"submitted: {execution.submitted}")
print(f"submitted_orders: {len(execution.submitted_orders) if execution.submitted_orders else 0}")
print(f"errors: {execution.errors}")

# Wait and verify
time.sleep(3)

open_trades = gw.ib.openTrades()
print(f"\n>>> Open orders in TWS: {len(open_trades)} <<<")

if open_trades:
    print("\nFirst 20 orders:")
    for t in open_trades[:20]:
        print(f"  {t.contract.symbol} {t.order.action} {t.order.totalQuantity} - {t.orderStatus.status}")

print("\n" + "="*70)
print("CHECK TWS NOW! Press Enter to cancel all orders...")
print("="*70)
input()

# Cancel
print("Cancelling all orders...")
for t in gw.ib.openTrades():
    try:
        gw.ib.cancelOrder(t.order)
    except:
        pass

time.sleep(2)
gw.disconnect()
print("Done")
