import sys
import os
import time
import logging

import nest_asyncio
nest_asyncio.apply()

# Enable DEBUG logging to see everything
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '974'

print("="*70)
print("MANUAL ORDER LOOP - BYPASSING EXECUTION ENGINE")
print("="*70)

from ib_insync import IB, Stock, MarketOrder
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
print(f"\nFirst 5 orders:")
for o in plan.orders[:5]:
    print(f"  {o.action} {o.quantity} {o.symbol}")

# NOW MANUALLY SEND JUST 3 ORDERS
print("\n" + "="*70)
print("MANUALLY SENDING 3 ORDERS (bypassing execution engine)")
print("="*70)

ib = gw.ib

orders_to_send = plan.orders[:3]
sent_count = 0

for o in orders_to_send:
    sym = o.symbol
    action = o.action
    qty = int(o.quantity)
    
    print(f"\n[{sent_count+1}] Sending: {action} {qty} {sym}")
    
    # Create contract
    contract = Stock(sym, 'SMART', 'USD')
    try:
        ib.qualifyContracts(contract)
        print(f"    Contract qualified: conId={contract.conId}")
    except Exception as e:
        print(f"    ❌ Failed to qualify contract: {e}")
        continue
    
    # Create order
    order = MarketOrder(action, qty)
    order.account = sel_account
    order.tif = 'DAY'
    
    # Place order
    print(f"    Placing order...")
    trade = ib.placeOrder(contract, order)
    print(f"    Order placed: orderId={trade.order.orderId}, status={trade.orderStatus.status}")
    
    sent_count += 1
    time.sleep(0.5)  # Small delay between orders

print(f"\n>>> Sent {sent_count} orders manually <<<")

# Wait and check
print(f"\nWaiting 3 seconds...")
time.sleep(3)

# Check open orders
open_trades = ib.openTrades()
print(f"\nOpen orders in TWS: {len(open_trades)}")
for t in open_trades:
    print(f"  {t.contract.symbol} {t.order.action} {t.order.totalQuantity} - {t.orderStatus.status}")

print("\n" + "="*70)
print("Check TWS now! Press Enter to cancel and exit...")
print("="*70)
input()

# Cancel
for t in ib.openTrades():
    try:
        ib.cancelOrder(t.order)
    except:
        pass

time.sleep(1)
gw.disconnect()
print("Done")
