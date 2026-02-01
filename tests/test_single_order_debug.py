import sys
import os
import time
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '980'

print("="*70)
print("DEBUG: SINGLE ORDER TEST")
print("="*70)

from ib_insync import IB, Stock, MarketOrder

# Connect directly with ib_insync
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=980, timeout=10)
print(f"Connected: {ib.isConnected()}, clientId: {ib.client.clientId}")

# Check open orders BEFORE
print(f"\nOpen orders BEFORE: {len(ib.openTrades())}")

# Create and send ONE order manually
contract = Stock('MSFT', 'SMART', 'USD')
ib.qualifyContracts(contract)

order = MarketOrder('BUY', 1)
order.account = 'DUK415187'
order.tif = 'DAY'

print(f"\nPlacing order: BUY 1 MSFT MKT...")
trade = ib.placeOrder(contract, order)

print("Waiting 2 seconds...")
time.sleep(2)

print(f"\nOrder status: {trade.orderStatus.status}")
print(f"Order ID: {trade.order.orderId}")

# Check open orders AFTER
open_now = ib.openTrades()
print(f"\nOpen orders AFTER: {len(open_now)}")
for t in open_now:
    print(f"  {t.contract.symbol} {t.order.action} {t.order.totalQuantity} - {t.orderStatus.status}")

# NOW test with execution_engine
print("\n" + "="*70)
print("NOW TESTING WITH EXECUTION ENGINE")
print("="*70)

# Build a minimal plan with just one order
from dashboard.ai_pm.models import PortfolioSnapshot, Position, TradePlan, Order, TargetWeights
from dashboard.ai_pm.config import DEFAULT_CONSTRAINTS
from dashboard.ai_pm.execution_engine import execute_trade_plan
from datetime import datetime, timezone

# Minimal snapshot
snapshot = PortfolioSnapshot(
    net_liquidation=1000000,
    total_cash=50000,
    buying_power=200000,
    available_funds=100000,
    currency="USD",
    positions=[],
)

# Minimal plan with ONE order
plan = TradePlan(
    strategy_key="test",
    ts_utc=datetime.now(timezone.utc),
    nav=1000000,
    orders=[
        Order(symbol="AAPL", action="BUY", quantity=1, order_type="MKT", reason="test"),
    ],
    notes=[],
    warnings=[],
)

# Price map
price_map = {"AAPL": 180.0}

print(f"\nCalling execute_trade_plan with 1 order...")
print(f"  dry_run=False")
print(f"  skip_live_quotes=True")
print(f"  price_map={price_map}")

execution = execute_trade_plan(
    ib=ib,
    snapshot=snapshot,
    plan=plan,
    account='DUK415187',
    constraints=DEFAULT_CONSTRAINTS,
    dry_run=False,
    kill_switch=False,
    auto_trade_enabled=False,
    armed=False,
    price_map=price_map,
    skip_live_quotes=True,
)

print(f"\nExecution result:")
print(f"  submitted: {execution.submitted}")
print(f"  submitted_orders: {execution.submitted_orders}")
print(f"  errors: {execution.errors}")
print(f"  notes: {execution.notes}")

# Check open orders again
print("\nWaiting 2 seconds...")
time.sleep(2)

open_final = ib.openTrades()
print(f"\nOpen orders FINAL: {len(open_final)}")
for t in open_final:
    print(f"  {t.contract.symbol} {t.order.action} {t.order.totalQuantity} - {t.orderStatus.status}")

# Cancel all
print("\nCancelling all orders...")
for t in open_final:
    try:
        ib.cancelOrder(t.order)
    except:
        pass

time.sleep(1)
ib.disconnect()
print("\nDone")
