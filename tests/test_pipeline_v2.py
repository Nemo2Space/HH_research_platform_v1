import sys
import time
import os
sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '996'

print("="*70)
print("FULL RUN NOW PIPELINE TEST")
print("="*70)

total_start = time.time()

# Step 1: Snapshot
print("\n⏳ Step 1/5: Loading portfolio snapshot...")
step_start = time.time()
from dashboard.ai_pm.ibkr_gateway import IbkrGateway
gw = IbkrGateway()
gw.connect(timeout_sec=10)
accounts = gw.list_accounts()
account = accounts[0]

from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
snapshot = _to_portfolio_snapshot(gw, account)
print(f"✅ Step 1: {time.time()-step_start:.1f}s | {len(snapshot.positions)} positions")

# Step 2-3: Signals
print("\n⏳ Step 2-3/5: Loading signals...")
step_start = time.time()
from dashboard.ai_pm.signal_adapter import load_platform_signals_snapshot
signals, diag = load_platform_signals_snapshot(limit=500)
print(f"✅ Step 2-3: {time.time()-step_start:.1f}s | {len(signals.rows)} signals")

# Step 4: Prices
print("\n⏳ Step 4/5: Fetching prices...")
step_start = time.time()
symbols = [p.symbol for p in snapshot.positions if p.symbol]
from src.utils.price_fetcher import get_prices
prices = get_prices(symbols, price_type='close')
print(f"✅ Step 4: {time.time()-step_start:.1f}s | {len(prices)}/{len(symbols)} prices")

# Step 5: Build Plan
print("\n⏳ Step 5/5: Building trade plan...")
step_start = time.time()
try:
    from dashboard.ai_pm.trade_planner import build_trade_plan
    from dashboard.ai_pm.target_builder import build_target_weights
    from dashboard.ai_pm.ui_tab import DEFAULT_CONSTRAINTS
    
    targets = build_target_weights(
        signals=signals,
        strategy_key='core_passive_drift',
        constraints=DEFAULT_CONSTRAINTS,
        max_holdings_override=25,
    )
    
    plan = build_trade_plan(
        snapshot=snapshot,
        targets=targets,
        constraints=DEFAULT_CONSTRAINTS,
        price_map=prices,
    )
    
    order_count = len(plan.orders) if plan and plan.orders else 0
    print(f"✅ Step 5: {time.time()-step_start:.1f}s | {len(targets.weights)} targets, {order_count} orders")
except Exception as e:
    print(f"❌ Step 5: {e}")
    plan = None

# Bonus: Pre-Trade Analysis
print("\n⏳ Bonus: Pre-Trade Analysis...")
step_start = time.time()
try:
    from dashboard.ai_pm.pre_trade_analysis import analyze_proposed_trades
    if plan and plan.orders:
        orders_list = [{'symbol': o.symbol, 'action': o.action, 'quantity': o.quantity} for o in plan.orders[:20]]
        analysis = analyze_proposed_trades(orders_list, signals)
        print(f"✅ Analysis: {time.time()-step_start:.1f}s | {analysis.overall_recommendation} | {len(analysis.warnings)} warnings")
except Exception as e:
    print(f"❌ Analysis: {e}")

gw.disconnect()

total = time.time() - total_start
print("\n" + "="*70)
print(f"TOTAL: {total:.1f}s")
if total < 15:
    print("🚀 EXCELLENT - Ready for Streamlit!")
elif total < 25:
    print("✅ GOOD - Acceptable performance")
else:
    print("⚠️ SLOW - Check network")
print("="*70)
