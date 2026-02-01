import sys
import os
sys.path.insert(0, '..')

import nest_asyncio
nest_asyncio.apply()

os.environ['IBKR_CLIENT_ID'] = '966'

from ib_insync import IB

print("Cancelling ALL orders (global cancel)...")

ib = IB()
ib.connect('127.0.0.1', 7496, clientId=966, timeout=10)

# Global cancel - cancels orders from ALL clients
print("Requesting global cancel...")
ib.reqGlobalCancel()

print("Waiting for cancellations...")
ib.sleep(5)

print("Done - check TWS")
ib.disconnect()
