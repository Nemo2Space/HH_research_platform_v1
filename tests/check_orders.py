import sys
import os
sys.path.insert(0, '..')

import nest_asyncio
nest_asyncio.apply()

os.environ['IBKR_CLIENT_ID'] = '984'

from dashboard.ai_pm.ibkr_gateway import IbkrGateway

print("Checking open orders and recent trades...")

gw = IbkrGateway()
gw.connect(timeout_sec=10)

print(f"\nConnected: {gw.is_connected()}")

# Check open orders
print("\n--- OPEN ORDERS ---")
trades = gw.ib.openTrades()
print(f"Open trades count: {len(trades) if trades else 0}")
for t in (trades or [])[:10]:
    print(f"  {t.contract.symbol} {t.order.action} {t.order.totalQuantity} - Status: {t.orderStatus.status}")

# Check all orders (including filled/cancelled)
print("\n--- ALL TRADES ---")
all_trades = gw.ib.trades()
print(f"All trades count: {len(all_trades) if all_trades else 0}")
for t in (all_trades or [])[:10]:
    print(f"  {t.contract.symbol} {t.order.action} {t.order.totalQuantity} - Status: {t.orderStatus.status}")

# Check executions
print("\n--- EXECUTIONS ---")
executions = gw.ib.executions()
print(f"Executions count: {len(executions) if executions else 0}")
for e in (executions or [])[:10]:
    print(f"  {e}")

gw.disconnect()
print("\nNote: Today is SUNDAY - market is CLOSED. Orders won't execute until Monday.")
