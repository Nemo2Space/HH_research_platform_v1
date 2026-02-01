import sys
import os
sys.path.insert(0, '..')

import nest_asyncio
nest_asyncio.apply()

os.environ['IBKR_CLIENT_ID'] = '982'

from ib_insync import IB, Stock, MarketOrder
import time

print("="*70)
print("TEST MARKET ORDER ON WEEKEND")
print("="*70)

ib = IB()
ib.connect('127.0.0.1', 7496, clientId=982, timeout=10)
print(f"Connected: {ib.isConnected()}")

contract = Stock('AAPL', 'SMART', 'USD')
ib.qualifyContracts(contract)

# Try a MARKET order like our execution engine does
order = MarketOrder('BUY', 1)
order.account = 'DUK415187'
order.tif = 'DAY'

print(f"\nPlacing MARKET order: BUY 1 AAPL MKT DAY")
trade = ib.placeOrder(contract, order)

print(f"Order placed, waiting for status...")
time.sleep(3)

print(f"\nTrade status:")
print(f"  Order ID: {trade.order.orderId}")
print(f"  Status: {trade.orderStatus.status}")
print(f"  Why Held: {trade.orderStatus.whyHeld}")

# Check log entries for rejection
print(f"\nTrade log:")
for log in trade.log:
    print(f"  {log.time}: {log.status} - {log.message} (errorCode={log.errorCode})")

# Check open orders
print(f"\nOpen trades: {len(ib.openTrades())}")
for t in ib.openTrades():
    print(f"  {t.contract.symbol} {t.order.action} {t.order.totalQuantity} {t.order.orderType} - {t.orderStatus.status}")

ib.disconnect()
print("\nDone")
