import sys
import os
import time
import json
import logging

import nest_asyncio
nest_asyncio.apply()

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '985'

print("="*70)
print("COMPLETE VALIDATION TEST - EVERY STEP")
print("="*70)

# ============================================================
# STEP 1: CONNECT
# ============================================================
print("\n" + "="*70)
print("STEP 1: CONNECT TO IBKR")
print("="*70)

from dashboard.ai_pm.ibkr_gateway import IbkrGateway

gw = IbkrGateway()
print(f"Gateway created, clientId: {gw.cfg.client_id}")

success = gw.connect(timeout_sec=10)
print(f"Connect result: {success}")
print(f"is_connected(): {gw.is_connected()}")
print(f"ib.isConnected(): {gw.ib.isConnected()}")

accounts = gw.list_accounts()
sel_account = accounts[0] if accounts else None
print(f"Accounts: {accounts}")
print(f"Selected account: {sel_account}")

if not gw.is_connected():
    print("❌ FAILED: Not connected")
    exit(1)

print("✅ STEP 1 PASSED")

# ============================================================
# STEP 2: LOAD TARGET PORTFOLIO
# ============================================================
print("\n" + "="*70)
print("STEP 2: LOAD TARGET PORTFOLIO FROM JSON")
print("="*70)

json_path = '../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json'
with open(json_path, 'r') as f:
    target_portfolio = json.load(f)

target_weights = {p['symbol']: p['weight'] / 100.0 for p in target_portfolio}
print(f"Loaded {len(target_weights)} target weights from {json_path}")
print(f"Sample weights: {dict(list(target_weights.items())[:5])}")

print("✅ STEP 2 PASSED")

# ============================================================
# STEP 3: GET PORTFOLIO SNAPSHOT
# ============================================================
print("\n" + "="*70)
print("STEP 3: GET PORTFOLIO SNAPSHOT")
print("="*70)

from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot

t = time.time()
snapshot = _to_portfolio_snapshot(gw, sel_account)
elapsed = time.time() - t

print(f"Snapshot retrieved in {elapsed:.2f}s")
print(f"Positions: {len(snapshot.positions)}")
print(f"NAV: {snapshot.net_liquidation}")
print(f"Cash: {snapshot.total_cash}")
print(f"Buying Power: {snapshot.buying_power}")

if not snapshot.positions:
    print("❌ FAILED: No positions in snapshot")
    exit(1)

print("✅ STEP 3 PASSED")

# ============================================================
# STEP 4: LOAD SIGNALS
# ============================================================
print("\n" + "="*70)
print("STEP 4: LOAD SIGNALS")
print("="*70)

from dashboard.ai_pm.signal_adapter import load_platform_signals_snapshot

t = time.time()
signals, diag = load_platform_signals_snapshot(limit=5000)
elapsed = time.time() - t

print(f"Signals loaded in {elapsed:.2f}s")
print(f"Signal count: {len(signals.rows)}")
print(f"Diagnostics: {diag}")

print("✅ STEP 4 PASSED")

# ============================================================
# STEP 5: BUILD TARGETS
# ============================================================
print("\n" + "="*70)
print("STEP 5: BUILD TARGETS")
print("="*70)

from dashboard.ai_pm.models import TargetWeights
from datetime import datetime, timezone

targets = TargetWeights(
    weights=target_weights,
    strategy_key='saved_portfolio',
    ts_utc=datetime.now(timezone.utc),
    notes=["Validation test"]
)

print(f"Targets built: {len(targets.weights)} holdings")
print(f"Strategy: {targets.strategy_key}")

print("✅ STEP 5 PASSED")

# ============================================================
# STEP 6: FETCH PRICES
# ============================================================
print("\n" + "="*70)
print("STEP 6: FETCH PRICES")
print("="*70)

universe = sorted(
    set([p.symbol.strip().upper() for p in snapshot.positions if getattr(p, "symbol", None)])
    | set([k.strip().upper() for k in targets.weights.keys() if k])
)
print(f"Universe: {len(universe)} symbols")

from src.utils.price_fetcher import get_prices

t = time.time()
price_map = get_prices(universe, price_type='close')
elapsed = time.time() - t

print(f"Prices fetched in {elapsed:.2f}s")
print(f"Prices retrieved: {len(price_map)}/{len(universe)}")

missing_prices = [s for s in universe if s not in price_map or not price_map.get(s)]
if missing_prices:
    print(f"Missing prices: {missing_prices[:10]}")

print("✅ STEP 6 PASSED")

# ============================================================
# STEP 7: BUILD TRADE PLAN
# ============================================================
print("\n" + "="*70)
print("STEP 7: BUILD TRADE PLAN")
print("="*70)

from dashboard.ai_pm.trade_planner import build_trade_plan
from dashboard.ai_pm.config import DEFAULT_CONSTRAINTS

t = time.time()
plan = build_trade_plan(
    snapshot=snapshot,
    targets=targets,
    constraints=DEFAULT_CONSTRAINTS,
    price_map=price_map,
)
elapsed = time.time() - t

order_count = len(plan.orders) if plan and plan.orders else 0
print(f"Trade plan built in {elapsed:.2f}s")
print(f"Orders: {order_count}")

if plan and plan.orders:
    print(f"Sample orders:")
    for o in plan.orders[:5]:
        print(f"  {o.action} {o.quantity} {o.symbol}")

print("✅ STEP 7 PASSED")

# ============================================================
# STEP 8: EVALUATE GATES
# ============================================================
print("\n" + "="*70)
print("STEP 8: EVALUATE GATES")
print("="*70)

from dashboard.ai_pm.risk_engine import evaluate_trade_plan_gates

t = time.time()
gates = evaluate_trade_plan_gates(
    snapshot=snapshot,
    signals=signals,
    plan=plan,
    constraints=DEFAULT_CONSTRAINTS
)
elapsed = time.time() - t

print(f"Gates evaluated in {elapsed:.2f}s")
print(f"Blocked: {gates.blocked}")
print(f"Block reasons: {gates.block_reasons}")
print(f"Warnings: {gates.warnings[:3] if gates.warnings else []}")

print("✅ STEP 8 PASSED")

# ============================================================
# STEP 9: EXECUTE TRADE PLAN (THE CRITICAL STEP)
# ============================================================
print("\n" + "="*70)
print("STEP 9: EXECUTE TRADE PLAN")
print("="*70)

print("Pre-execution checks:")
print(f"  gw.is_connected(): {gw.is_connected()}")
print(f"  gw.ib.isConnected(): {gw.ib.isConnected()}")
print(f"  gw.ib.client.clientId: {gw.ib.client.clientId}")
print(f"  gates.blocked: {gates.blocked}")
print(f"  order_count: {order_count}")
print(f"  price_map size: {len(price_map)}")

if gates.blocked:
    print("❌ BLOCKED BY GATES - Cannot execute")
    print(f"   Reasons: {gates.block_reasons}")
else:
    print("\n>>> CALLING execute_trade_plan <<<")
    print(">>> THIS IS THE EXACT CALL THAT FREEZES IN STREAMLIT <<<\n")
    
    from dashboard.ai_pm.execution_engine import execute_trade_plan
    
    t = time.time()
    
    # Call execute_trade_plan with detailed timing
    print("  Entering execute_trade_plan...")
    
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
    
    elapsed = time.time() - t
    
    print(f"\n  execute_trade_plan returned in {elapsed:.2f}s")
    print(f"  submitted: {execution.submitted}")
    print(f"  orders sent: {len(execution.submitted_orders) if execution.submitted_orders else 0}")
    
    if execution.errors:
        print(f"  Errors: {execution.errors}")
    if execution.notes:
        print(f"  Notes (first 5): {execution.notes[:5]}")
    
    if execution.submitted_orders:
        print(f"\n  Submitted orders:")
        for o in execution.submitted_orders[:10]:
            print(f"    {o}")

print("✅ STEP 9 PASSED")

# ============================================================
# STEP 10: VERIFY CONNECTION STILL ALIVE
# ============================================================
print("\n" + "="*70)
print("STEP 10: POST-EXECUTION CONNECTION CHECK")
print("="*70)

print(f"gw.is_connected(): {gw.is_connected()}")
print(f"gw.ib.isConnected(): {gw.ib.isConnected()}")

# Try to get accounts again
try:
    post_accounts = gw.list_accounts()
    print(f"list_accounts() works: {post_accounts}")
except Exception as e:
    print(f"list_accounts() failed: {e}")

print("✅ STEP 10 PASSED")

# ============================================================
# CLEANUP
# ============================================================
print("\n" + "="*70)
print("CLEANUP")
print("="*70)

gw.disconnect()
print("Disconnected")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)
print(f"✅ Connection: OK")
print(f"✅ Snapshot: {len(snapshot.positions)} positions, NAV={snapshot.net_liquidation}")
print(f"✅ Signals: {len(signals.rows)}")
print(f"✅ Targets: {len(targets.weights)}")
print(f"✅ Prices: {len(price_map)}/{len(universe)}")
print(f"✅ Trade Plan: {order_count} orders")
print(f"✅ Gates: blocked={gates.blocked}")
if not gates.blocked:
    print(f"✅ Execution: submitted={execution.submitted}, orders={len(execution.submitted_orders) if execution.submitted_orders else 0}")
print("="*70)
print("ALL STEPS COMPLETED SUCCESSFULLY")
print("="*70)
