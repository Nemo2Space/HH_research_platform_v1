import sys
import os
import time
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '968'

print("="*70)
print("TEST: VERIFY ORDERS ACTUALLY REACH TWS")
print("="*70)

from ib_insync import IB, Stock, MarketOrder

ib = IB()
ib.connect('127.0.0.1', 7496, clientId=968, timeout=10)
print(f"Connected: {ib.isConnected()}, clientId: {ib.client.clientId}")

# Send 3 simple orders and WAIT for status
symbols = ['AAPL', 'MSFT', 'GOOGL']
trades = []

for sym in symbols:
    contract = Stock(sym, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    
    order = MarketOrder('BUY', 1)
    order.account = 'DUK415187'
    order.tif = 'DAY'
    
    print(f"\nPlacing order: BUY 1 {sym}...")
    trade = ib.placeOrder(contract, order)
    trades.append(trade)
    
    # Wait for status update
    print(f"  Initial status: {trade.orderStatus.status}")

# IMPORTANT: Give time for IBKR to process
print("\n>>> Waiting 5 seconds for IBKR to process... <<<")
ib.sleep(5)  # This is important - allows event loop to process callbacks

# Check status after waiting
print("\nOrder statuses after waiting:")
for trade in trades:
    print(f"  {trade.contract.symbol}: {trade.orderStatus.status}")
    if trade.log:
        for entry in trade.log:
            print(f"    Log: {entry.status} - {entry.message} (error={entry.errorCode})")

# Check what IBKR reports
print(f"\nOpen trades from IBKR: {len(ib.openTrades())}")
for t in ib.openTrades():
    print(f"  {t.contract.symbol} {t.order.action} {t.order.totalQuantity} - {t.orderStatus.status}")

print("\n>>> CHECK TWS NOW! You should see 3 orders <<<")
print(">>> Press Enter to cancel... <<<")
input()

# Cancel
for trade in trades:
    ib.cancelOrder(trade.order)

ib.sleep(2)
ib.disconnect()
print("Done")
