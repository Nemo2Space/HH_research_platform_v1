import sys
import os
import time
import json
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '987'

print("="*70)
print("COMPLETE STREAMLIT FLOW SIMULATION")
print("Simulating: Confirm & Send button click")
print("="*70)

# ============================================================
# STEP 1: GATEWAY CONNECTION (like session_state init)
# ============================================================
print("\n[STEP 1] Gateway connection...")
t = time.time()

from dashboard.ai_pm.ibkr_gateway import IbkrGateway
gw = IbkrGateway()
success = gw.connect(timeout_sec=10)
sel_account = gw.list_accounts()[0] if gw.list_accounts() else None

print(f"    Connected: {success}, account: {sel_account} ({time.time()-t:.1f}s)")

if not gw.is_connected():
    print("❌ FAILED: Not connected")
    exit(1)

# ============================================================
# STEP 2: SIMULATE SESSION STATE (loaded JSON portfolio)
# ============================================================
print("\n[STEP 2] Loading session state (JSON portfolio)...")

json_path = '../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json'
with open(json_path, 'r') as f:
    target_portfolio = json.load(f)

# This is what Streamlit stores in session_state
ai_pm_target_weights = {p['symbol']: p['weight'] / 100.0 for p in target_portfolio}
ai_pm_target_portfolio = {'name': 'KAIF_SC_2026_Q1'}

print(f"    Loaded {len(ai_pm_target_weights)} target weights")

# ============================================================
# STEP 3: CONFIRM BUTTON CLICKED - START EXECUTION FLOW
# ============================================================
print("\n[STEP 3] >>> CONFIRM & SEND CLICKED <<<")
print("=" * 60)

# This is the exact code from ui_tab.py lines 3926+

print("\n    [3a] Getting portfolio snapshot...")
t = time.time()
from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
snapshot = _to_portfolio_snapshot(gw, sel_account)
print(f"        {len(snapshot.positions)} positions ({time.time()-t:.1f}s)")

# Debug valuation coverage
missing_val = [p.symbol for p in (snapshot.positions or []) if
               (p.market_price is None and p.market_value is None)]
print(f"        Valuation coverage: {len(snapshot.positions) - len(missing_val)}/{len(snapshot.positions)}")

print("\n    [3b] Loading signals...")
t = time.time()
from dashboard.ai_pm.signal_adapter import load_platform_signals_snapshot
signals, diag = load_platform_signals_snapshot(limit=5000)
print(f"        {len(signals.rows)} signals ({time.time()-t:.1f}s)")

print("\n    [3c] Building targets from saved portfolio...")
t = time.time()
from dashboard.ai_pm.models import TargetWeights
from datetime import datetime, timezone
targets = TargetWeights(
    weights=ai_pm_target_weights,
    strategy_key='saved_portfolio',
    ts_utc=datetime.now(timezone.utc),
    notes=[f"Using saved portfolio: {ai_pm_target_portfolio.get('name', 'Unknown')}"]
)
print(f"        {len(targets.weights)} targets ({time.time()-t:.1f}s)")

print("\n    [3d] Building universe and fetching prices...")
t = time.time()
universe = sorted(
    set([p.symbol.strip().upper() for p in (snapshot.positions or []) if getattr(p, "symbol", None)])
    | set([k.strip().upper() for k in ((targets.weights or {}) or {}).keys() if k])
)
print(f"        Universe: {len(universe)} symbols")

from src.utils.price_fetcher import get_prices
price_map = get_prices(universe, price_type='close')
print(f"        Prices: {len(price_map)}/{len(universe)} ({time.time()-t:.1f}s)")

print("\n    [3e] Building trade plan...")
t = time.time()
from dashboard.ai_pm.trade_planner import build_trade_plan
from dashboard.ai_pm.config import DEFAULT_CONSTRAINTS

plan = build_trade_plan(
    snapshot=snapshot,
    targets=targets,
    constraints=DEFAULT_CONSTRAINTS,
    price_map=price_map,
)
order_count = len(plan.orders) if plan and plan.orders else 0
print(f"        {order_count} orders ({time.time()-t:.1f}s)")

print("\n    [3f] Evaluating gates...")
t = time.time()
from dashboard.ai_pm.risk_engine import evaluate_trade_plan_gates

gates = evaluate_trade_plan_gates(
    snapshot=snapshot,
    signals=signals,
    plan=plan,
    constraints=DEFAULT_CONSTRAINTS
)
print(f"        blocked={gates.blocked} ({time.time()-t:.1f}s)")
if gates.block_reasons:
    print(f"        Block reasons: {gates.block_reasons}")

# ============================================================
# STEP 4: EXECUTE TRADE PLAN (THIS IS WHERE IT FREEZES)
# ============================================================
print("\n    [3g] >>> EXECUTING TRADE PLAN (dry_run=False) <<<")
print("        THIS IS THE EXACT CALL THAT STREAMLIT MAKES")
t = time.time()

from dashboard.ai_pm.execution_engine import execute_trade_plan

# Check conditions (same as ui_tab.py)
if gates.blocked:
    print("        ❌ Blocked by hard gates; not sending orders.")
    execution = None
else:
    # Simulate is_kill_switch() = False
    kill_switch = False
    
    if kill_switch:
        print("        ❌ Kill switch enabled; not sending orders.")
        execution = None
    else:
        print("        Calling execute_trade_plan...")
        
        execution = execute_trade_plan(
            ib=gw.ib,
            snapshot=snapshot,
            plan=plan,
            account=sel_account,
            constraints=DEFAULT_CONSTRAINTS,
            dry_run=False,  # REAL - same as Streamlit
            kill_switch=False,
            auto_trade_enabled=False,
            armed=False,
            price_map=price_map,
            skip_live_quotes=True,
        )
        
        print(f"        execute_trade_plan returned ({time.time()-t:.1f}s)")
        print(f"        submitted={execution.submitted}")
        print(f"        orders sent: {len(execution.submitted_orders) if execution.submitted_orders else 0}")
        
        if execution.errors:
            print(f"        Errors: {execution.errors[:3]}")
        if execution.notes:
            print(f"        Notes: {execution.notes[:3]}")

# ============================================================
# STEP 5: POST-EXECUTION (audit write, session updates)
# ============================================================
print("\n    [3h] Writing audit log...")
t = time.time()

if execution:
    try:
        from dashboard.ai_pm.audit import write_ai_pm_audit
        
        audit_path = write_ai_pm_audit(
            account=sel_account,
            strategy_key='saved_portfolio',
            snapshot=snapshot,
            signals=signals,
            signals_diagnostics=diag,
            targets=targets,
            plan=plan,
            gates=gates,
            execution=execution,
            extra={"dry_run": False, "manual_confirm": True},
        )
        print(f"        Audit written: {audit_path} ({time.time()-t:.1f}s)")
    except Exception as e:
        print(f"        Audit error (non-fatal): {e}")

# ============================================================
# CLEANUP
# ============================================================
print("\n[CLEANUP] Disconnecting...")
gw.disconnect()

print("\n" + "="*70)
print("✅ COMPLETE FLOW SIMULATION FINISHED")
print("="*70)
