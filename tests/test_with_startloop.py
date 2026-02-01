import sys
import os
import time
import json
import logging
import threading

sys.path.insert(0, '..')

# CRITICAL: Start ib_insync event loop FIRST before anything else
from ib_insync import util
util.startLoop()

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

os.environ['IBKR_CLIENT_ID'] = '962'

print("="*70)
print("TEST WITH util.startLoop()")
print("="*70)

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

print("\n[1] Connect...")
gw = IbkrGateway()
gw.connect(timeout_sec=10)
sel_account = gw.list_accounts()[0]
print(f"    Connected: {sel_account}")

print("\n[2] Load JSON...")
json_path = '../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json'
with open(json_path, 'r') as f:
    target_portfolio = json.load(f)
target_weights = {p['symbol']: p['weight'] / 100.0 for p in target_portfolio}

def simulate_confirm():
    print("\n>>> SIMULATING CONFIRM (in thread) <<<")
    
    snapshot = _to_portfolio_snapshot(gw, sel_account)
    print(f"    Snapshot: {len(snapshot.positions)} positions")
    
    signals, diag = load_platform_signals_snapshot(limit=5000)
    
    targets = TargetWeights(
        weights=target_weights,
        strategy_key='saved_portfolio',
        ts_utc=datetime.now(timezone.utc),
        notes=["Test"]
    )
    
    universe = sorted(
        set([p.symbol.strip().upper() for p in (snapshot.positions or []) if getattr(p, "symbol", None)])
        | set([k.strip().upper() for k in ((targets.weights or {}) or {}).keys() if k])
    )
    
    # Yahoo prices
    import yfinance as yf
    price_map = {}
    for i in range(0, len(universe), 100):
        chunk = universe[i:i+100]
        try:
            yf_data = yf.download(' '.join(chunk), period='1d', progress=False, threads=True)
            if not yf_data.empty and 'Close' in yf_data.columns:
                if len(chunk) == 1:
                    val = yf_data['Close'].iloc[-1]
                    if val and val > 0:
                        price_map[chunk[0]] = float(val)
                else:
                    for sym in chunk:
                        try:
                            if sym in yf_data['Close'].columns:
                                val = yf_data['Close'][sym].iloc[-1]
                                if val and val > 0:
                                    price_map[sym] = float(val)
                        except:
                            pass
        except Exception as e:
            print(f"    Yahoo error: {e}")
    
    print(f"    Prices: {len(price_map)}/{len(universe)}")
    
    plan = build_trade_plan(
        snapshot=snapshot,
        targets=targets,
        constraints=DEFAULT_CONSTRAINTS,
        price_map=price_map,
    )
    print(f"    Plan: {len(plan.orders)} orders")
    
    gates = evaluate_trade_plan_gates(
        snapshot=snapshot,
        signals=signals,
        plan=plan,
        constraints=DEFAULT_CONSTRAINTS
    )
    print(f"    Gates blocked: {gates.blocked}")
    
    print("\n    >>> execute_trade_plan <<<")
    t0 = time.time()
    
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
    
    print(f"\n    ✅ Completed in {time.time()-t0:.1f}s")
    print(f"    submitted: {execution.submitted}")
    print(f"    orders: {len(execution.submitted_orders) if execution.submitted_orders else 0}")
    return execution

# Run in thread
print("\n[3] Running in thread...")
result = [None]
def run_thread():
    result[0] = simulate_confirm()

thread = threading.Thread(target=run_thread)
thread.start()
thread.join(timeout=60)

if thread.is_alive():
    print("\n❌ THREAD TIMED OUT")
else:
    print("\n✅ Thread completed")

# Cleanup
print("\n[4] Cleanup...")
gw.ib.reqGlobalCancel()
gw.ib.sleep(2)
gw.disconnect()
print("Done")
