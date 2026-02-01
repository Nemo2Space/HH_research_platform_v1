import sys
import os
import time
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '979'

print("="*70)
print("TRACE EXECUTION ENGINE")
print("="*70)

from ib_insync import IB

# Connect
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=979, timeout=10)
print(f"Connected: {ib.isConnected()}")

# Check what Order class looks like
from dashboard.ai_pm.models import TradePlan
import inspect

# Find Order class
print("\nLooking for Order class...")
with open('../dashboard/ai_pm/models.py', 'r') as f:
    content = f.read()
    if 'class Order' in content:
        idx = content.find('class Order')
        print(content[idx:idx+500])
    else:
        print("No Order class found")
        # Find what's in TradePlan
        print("\nTradePlan definition:")
        idx = content.find('class TradePlan')
        print(content[idx:idx+500])

ib.disconnect()
