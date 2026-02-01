import sys
import os
sys.path.insert(0, '..')

import nest_asyncio
nest_asyncio.apply()

os.environ['IBKR_CLIENT_ID'] = '983'

from ib_insync import IB, Stock, MarketOrder, LimitOrder
import time

print("="*70)
print("TEST ORDER SUBMISSION (Weekend)")
print("="*70)

ib = IB()
ib.connect('127.0.0.1', 7496, clientId=983, timeout=10)
print(f"Connected: {ib.isConnected()}")

# Create a small test order
contract = Stock('AAPL', 'SMART', 'USD')
ib.qualifyContracts(contract)

print(f"\nContract qualified: {contract}")

# Try a LIMIT order (more likely to be accepted on weekend)
# Set limit price way below market so it won't fill
order = LimitOrder('BUY', 1, 100.00)  # Buy 1 AAPL at  (won't fill)
order.account = 'DUK415187'
order.tif = 'GTC'  # Good Till Cancelled

print(f"\nPlacing test order: BUY 1 AAPL @  LIMIT GTC")
trade = ib.placeOrder(contract, order)

print(f"Order placed, waiting for status...")
time.sleep(2)

print(f"\nTrade status:")
print(f"  Order ID: {trade.order.orderId}")
print(f"  Status: {trade.orderStatus.status}")
print(f"  Filled: {trade.orderStatus.filled}")

# Check if it shows in open orders
print(f"\nOpen trades: {len(ib.openTrades())}")
for t in ib.openTrades():
    print(f"  {t.contract.symbol} {t.order.action} {t.order.totalQuantity} @ {t.order.lmtPrice} - {t.orderStatus.status}")

# Cancel the test order
print(f"\nCancelling test order...")
ib.cancelOrder(order)
time.sleep(1)

print(f"Final status: {trade.orderStatus.status}")

ib.disconnect()
print("\nDone")
