# Fix 1: Check why NAV is None
import sys
import os
sys.path.insert(0, '..')

import nest_asyncio
nest_asyncio.apply()

os.environ['IBKR_CLIENT_ID'] = '992'

from dashboard.ai_pm.ibkr_gateway import IbkrGateway

gw = IbkrGateway()
gw.connect(timeout_sec=10)
accounts = gw.list_accounts()
print(f"Accounts: {accounts}")

# Test account summary directly
print("\nTesting get_account_summary directly...")
summ = gw.get_account_summary(accounts[0])

print(f"\nResult: {summ}")
if summ:
    print(f"  net_liquidation: {summ.net_liquidation}")
    print(f"  total_cash_value: {summ.total_cash_value}")
    print(f"  buying_power: {summ.buying_power}")
    print(f"  available_funds: {summ.available_funds}")
else:
    print("  summ is None!")

gw.disconnect()
