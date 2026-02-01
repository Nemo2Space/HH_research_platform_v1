import sys
import os
import time
import json
import logging
import threading

# This simulates Streamlit's threading environment
import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '963'

print("="*70)
print("STREAMLIT-LIKE THREADING TEST")
print("="*70)

# Import like Streamlit does
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

# Create gateway like Streamlit session_state does
print("\n[1] Creating gateway (like session_state)...")
gw = IbkrGateway()
gw.connect(timeout_sec=10)
sel_account = gw.list_accounts()[0]
print(f"    Connected: {sel_account}")

# Load data
print("\n[2] Loading JSON...")
json_path = '../json/KAIF_SC_2026_Q1_IBKR_20260125_061640.json'
with open(json_path, 'r') as f:
    target_portfolio = json.load(f)
target_weights = {p['symbol']: p['weight'] / 100.0 for p in target_portfolio}

# Now simulate what happens when confirm button is clicked
# This runs in the Streamlit rerun context

def simulate_confirm_click():
    """This simulates what happens when 'Confirm & Send' is clicked"""
    print("\n" + "="*70)
    print("SIMULATING CONFIRM CLICK (in thread like Streamlit)")
    print("="*70)
    
    # This is the exact flow from ui_tab.py when confirm is clicked
    print("\n    Getting snapshot...")
    snapshot = _to_portfolio_snapshot(gw, sel_account)
    print(f"    Snapshot: {len(snapshot.positions)} positions")
    
    print("\n    Loading signals...")
    signals, diag = load_platform_signals_snapshot(limit=5000)
    print(f"    Signals: {len(signals.rows)}")
    
    print("\n    Building targets...")
    targets = TargetWeights(
        weights=target_weights,
        strategy_key='saved_portfolio',
        ts_utc=datetime.now(timezone.utc),
        notes=["Test"]
    )
    
    print("\n    Building universe...")
    universe = sorted(
        set([p.symbol.strip().upper() for p in (snapshot.positions or []) if getattr(p, "symbol", None)])
        | set([k.strip().upper() for k in ((targets.weights or {}) or {}).keys() if k])
    )
    print(f"    Universe: {len(universe)} symbols")
    
    # This is the Yahoo Finance fetch that happens in Streamlit
    print("\n    Fetching prices (Yahoo)...")
    import yfinance as yf
    price_map = {}
    
    for i in range(0, len(universe), 100):
        chunk = universe[i:i+100]
        tickers_str = ' '.join(chunk)
        try:
            yf_data = yf.download(tickers_str, period='1d', progress=False, threads=True)
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
    
    print("\n    Building trade plan...")
    plan = build_trade_plan(
        snapshot=snapshot,
        targets=targets,
        constraints=DEFAULT_CONSTRAINTS,
        price_map=price_map,
    )
    print(f"    Plan: {len(plan.orders)} orders")
    
    print("\n    Evaluating gates...")
    gates = evaluate_trade_plan_gates(
        snapshot=snapshot,
        signals=signals,
        plan=plan,
        constraints=DEFAULT_CONSTRAINTS
    )
    print(f"    Gates blocked: {gates.blocked}")
    
    print("\n    >>> CALLING execute_trade_plan <<<")
    print(f"    IB connected: {gw.ib.isConnected()}")
    print(f"    price_map size: {len(price_map)}")
    print(f"    skip_live_quotes: True")
    
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
        print(f"\n    ✅ execute_trade_plan returned in {elapsed:.1f}s")
        print(f"    submitted: {execution.submitted}")
        print(f"    orders: {len(execution.submitted_orders) if execution.submitted_orders else 0}")
        return execution
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n    ❌ FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run in a thread to simulate Streamlit's behavior
print("\n[3] Running in thread (like Streamlit)...")
result = [None]
def run_in_thread():
    result[0] = simulate_confirm_click()

thread = threading.Thread(target=run_in_thread)
thread.start()

# Wait with timeout
print("    Waiting for thread (timeout 60s)...")
thread.join(timeout=60)

if thread.is_alive():
    print("\n❌ THREAD TIMED OUT - This is the freeze!")
    print("    The execution is stuck somewhere")
else:
    print("\n✅ Thread completed")
    if result[0]:
        print(f"    Result: submitted={result[0].submitted}")

# Cleanup
print("\n[4] Cleanup...")
gw.ib.reqGlobalCancel()
gw.ib.sleep(2)
gw.disconnect()
print("Done")
