import sys
import os
sys.path.insert(0, '..')

import nest_asyncio
nest_asyncio.apply()

os.environ['IBKR_CLIENT_ID'] = '991'

from ib_insync import IB, util

print("Connecting directly with ib_insync...")
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=991, timeout=10)

print(f"Connected: {ib.isConnected()}")

accounts = ib.managedAccounts()
print(f"Accounts: {accounts}")
acct = accounts[0]

print("\n1. Testing accountSummary()...")
try:
    rows = ib.accountSummary()
    print(f"   Got {len(rows) if rows else 0} rows")
    if rows:
        for r in rows[:10]:
            print(f"   {r}")
except Exception as e:
    print(f"   Error: {e}")

print("\n2. Testing accountSummary(acct)...")
try:
    rows = ib.accountSummary(acct)
    print(f"   Got {len(rows) if rows else 0} rows")
    if rows:
        for r in rows[:10]:
            print(f"   {r}")
except Exception as e:
    print(f"   Error: {e}")

print("\n3. Testing accountValues()...")
try:
    rows = ib.accountValues(acct)
    print(f"   Got {len(rows) if rows else 0} rows")
    # Find NetLiquidation
    for r in rows or []:
        if getattr(r, 'tag', '') == 'NetLiquidation':
            print(f"   NetLiquidation: {r}")
            break
except Exception as e:
    print(f"   Error: {e}")

print("\n4. Testing portfolio()...")
try:
    items = ib.portfolio(acct)
    print(f"   Got {len(items) if items else 0} portfolio items")
except Exception as e:
    print(f"   Error: {e}")

ib.disconnect()
print("\nDone.")
