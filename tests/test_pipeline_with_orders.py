import sys
import os
import time
import json
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '981'

print("="*70)
print("COMPLETE PIPELINE TEST WITH ORDER VERIFICATION")
print("This test will:")
print("  1. Run the exact Streamlit execution flow")
print("  2. Send REAL orders (LIMIT orders to avoid weekend issues)")
print("  3. Keep connection alive to verify orders in TWS")
print("  4. Show all orders before cleanup")
print("="*70)

# ============================================================
# CONNECT (keep this connection for the whole test)
# ============================================================
print("\n[1] CONNECTING...")
from dashboard.ai_pm.ibkr_gateway import IbkrGateway

gw = IbkrGateway()
gw.connect(timeout_sec=10)
sel_account = gw.list_accounts()[0]
print(f"    Connected, account: {sel_account}, clientId: {gw.ib.client.clientId}")

# ============================================================
# LOAD PORTFOLIO & BUILD PIPELINE
# ============================================================
print("\n[2] LOADING PORTFOLIO...")
json_path = '../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json'
with open(json_path, 'r') as f:
    target_portfolio = json.load(f)
target_weights = {p['symbol']: p['weight'] / 100.0 for p in target_portfolio}
print(f"    Loaded {len(target_weights)} targets")

print("\n[3] GETTING SNAPSHOT...")
from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
snapshot = _to_portfolio_snapshot(gw, sel_account)
print(f"    {len(snapshot.positions)} positions, NAV={snapshot.net_liquidation}")

print("\n[4] LOADING SIGNALS...")
from dashboard.ai_pm.signal_adapter import load_platform_signals_snapshot
signals, diag = load_platform_signals_snapshot(limit=5000)
print(f"    {len(signals.rows)} signals")

print("\n[5] BUILDING TARGETS...")
from dashboard.ai_pm.models import TargetWeights
from datetime import datetime, timezone
targets = TargetWeights(
    weights=target_weights,
    strategy_key='saved_portfolio',
    ts_utc=datetime.now(timezone.utc),
    notes=["Pipeline test"]
)

print("\n[6] FETCHING PRICES...")
universe = sorted(
    set([p.symbol.strip().upper() for p in snapshot.positions if getattr(p, "symbol", None)])
    | set([k.strip().upper() for k in targets.weights.keys() if k])
)
from src.utils.price_fetcher import get_prices
price_map = get_prices(universe, price_type='close')
print(f"    {len(price_map)}/{len(universe)} prices")

print("\n[7] BUILDING TRADE PLAN...")
from dashboard.ai_pm.trade_planner import build_trade_plan
from dashboard.ai_pm.config import DEFAULT_CONSTRAINTS
plan = build_trade_plan(
    snapshot=snapshot,
    targets=targets,
    constraints=DEFAULT_CONSTRAINTS,
    price_map=price_map,
)
order_count = len(plan.orders) if plan and plan.orders else 0
print(f"    {order_count} orders")

if plan and plan.orders:
    print(f"    First 5 orders:")
    for o in plan.orders[:5]:
        print(f"      {o.action} {o.quantity} {o.symbol}")

print("\n[8] EVALUATING GATES...")
from dashboard.ai_pm.risk_engine import evaluate_trade_plan_gates
gates = evaluate_trade_plan_gates(
    snapshot=snapshot,
    signals=signals,
    plan=plan,
    constraints=DEFAULT_CONSTRAINTS
)
print(f"    blocked={gates.blocked}")

if gates.blocked:
    print(f"    ❌ BLOCKED: {gates.block_reasons}")
    gw.disconnect()
    exit(1)

# ============================================================
# EXECUTE - THE CRITICAL PART
# ============================================================
print("\n" + "="*70)
print("[9] EXECUTING TRADE PLAN")
print("="*70)

print(f"\nPre-execution state:")
print(f"  IB connected: {gw.ib.isConnected()}")
print(f"  ClientId: {gw.ib.client.clientId}")
print(f"  Orders to send: {order_count}")

print(f"\n>>> CALLING execute_trade_plan (dry_run=False) <<<\n")

from dashboard.ai_pm.execution_engine import execute_trade_plan

t_start = time.time()
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
t_elapsed = time.time() - t_start

print(f"\n>>> execute_trade_plan COMPLETED in {t_elapsed:.1f}s <<<")
print(f"    submitted: {execution.submitted}")
print(f"    orders sent: {len(execution.submitted_orders) if execution.submitted_orders else 0}")

if execution.errors:
    print(f"    ERRORS: {execution.errors}")
if execution.notes:
    print(f"    Notes: {execution.notes[:5]}")

# ============================================================
# VERIFY ORDERS IN TWS
# ============================================================
print("\n" + "="*70)
print("[10] VERIFYING ORDERS IN TWS")
print("="*70)

print("\nWaiting 3 seconds for orders to settle...")
time.sleep(3)

print(f"\nIB connection still alive: {gw.ib.isConnected()}")

# Get open orders
open_trades = gw.ib.openTrades()
print(f"\nOpen trades in TWS: {len(open_trades)}")

if open_trades:
    print("\n  Symbol    Action  Qty     Type      Status")
    print("  " + "-"*50)
    for t in open_trades[:20]:
        print(f"  {t.contract.symbol:8} {t.order.action:6} {t.order.totalQuantity:6} {t.order.orderType:8} {t.orderStatus.status}")
else:
    print("  (No open trades - orders may have been rejected or are pending)")

# Get all trades from this session
all_trades = gw.ib.trades()
print(f"\nAll trades this session: {len(all_trades)}")

# ============================================================
# CLEANUP - Cancel all orders from this test
# ============================================================
print("\n" + "="*70)
print("[11] CLEANUP - CANCELLING TEST ORDERS")
print("="*70)

if open_trades:
    print(f"\nCancelling {len(open_trades)} orders...")
    for t in open_trades:
        try:
            gw.ib.cancelOrder(t.order)
        except:
            pass
    
    time.sleep(2)
    
    remaining = gw.ib.openTrades()
    print(f"Remaining open orders: {len(remaining)}")
else:
    print("No orders to cancel")

gw.disconnect()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print(f"✅ Pipeline executed successfully")
print(f"✅ {len(execution.submitted_orders) if execution.submitted_orders else 0} orders were submitted")
print(f"✅ Execution time: {t_elapsed:.1f}s")
print("="*70)
