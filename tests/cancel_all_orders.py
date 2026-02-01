import sys
import os
sys.path.insert(0, '..')

import nest_asyncio
nest_asyncio.apply()

os.environ['IBKR_CLIENT_ID'] = '967'

from ib_insync import IB

print("Cancelling all open orders...")

ib = IB()
ib.connect('127.0.0.1', 7496, clientId=967, timeout=10)

open_trades = ib.openTrades()
print(f"Found {len(open_trades)} open orders")

for t in open_trades:
    print(f"  Cancelling {t.contract.symbol}...")
    ib.cancelOrder(t.order)

# CRITICAL: Wait for cancellations to transmit
print("Waiting for cancellations to transmit...")
ib.sleep(5)

remaining = ib.openTrades()
print(f"Remaining orders: {len(remaining)}")

ib.disconnect()
print("Done")
