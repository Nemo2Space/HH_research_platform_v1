import sys
import os
import time
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '976'

print("="*70)
print("DETAILED EXECUTION TRACE - SINGLE ORDER")
print("="*70)

from dashboard.ai_pm.ibkr_gateway import IbkrGateway

gw = IbkrGateway()
gw.connect(timeout_sec=10)
sel_account = gw.list_accounts()[0]
print(f"Connected, account: {sel_account}")

# Get snapshot
from dashboard.ai_pm.ui_tab import _to_portfolio_snapshot
snapshot = _to_portfolio_snapshot(gw, sel_account)

# Build a minimal plan with JUST ONE order
from dashboard.ai_pm.models import TradePlan, OrderTicket
from datetime import datetime, timezone

# Find a stock we own
owned_symbol = None
for p in snapshot.positions:
    if p.symbol and p.quantity and p.quantity > 0:
        owned_symbol = p.symbol
        break

print(f"\nUsing symbol: {owned_symbol}")

plan = TradePlan(
    ts_utc=datetime.now(timezone.utc),
    account=sel_account,
    strategy_key="test",
    nav=snapshot.net_liquidation or 1000000,
    orders=[
        OrderTicket(
            symbol=owned_symbol,
            action="SELL",
            quantity=1,
            order_type="MKT",
            limit_price=None,
            tif="DAY",
            reason="Test order"
        ),
    ],
)

print(f"Plan has {len(plan.orders)} order(s)")
print(f"Order: {plan.orders[0]}")

# Get price for this symbol
from src.utils.price_fetcher import get_prices
price_map = get_prices([owned_symbol], price_type='close')
print(f"Price for {owned_symbol}: {price_map.get(owned_symbol)}")

# Check open orders BEFORE
print(f"\nOpen orders BEFORE: {len(gw.ib.openTrades())}")

# Now call execute_trade_plan
print("\n>>> Calling execute_trade_plan <<<")

from dashboard.ai_pm.execution_engine import execute_trade_plan
from dashboard.ai_pm.config import DEFAULT_CONSTRAINTS

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

print(f"\n>>> execute_trade_plan returned <<<")
print(f"submitted: {execution.submitted}")
print(f"submitted_orders: {execution.submitted_orders}")
print(f"errors: {execution.errors}")
print(f"notes: {execution.notes}")

# Wait and check
time.sleep(2)

# Check open orders AFTER
open_trades = gw.ib.openTrades()
print(f"\nOpen orders AFTER: {len(open_trades)}")
for t in open_trades:
    print(f"  {t.contract.symbol} {t.order.action} {t.order.totalQuantity} - {t.orderStatus.status}")

# Cancel if any
if open_trades:
    print("\nCancelling...")
    for t in open_trades:
        try:
            gw.ib.cancelOrder(t.order)
        except:
            pass

gw.disconnect()
print("\nDone")
