import sys
import os
import time
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '..')

os.environ['IBKR_CLIENT_ID'] = '972'

print("="*70)
print("TRACE dry_run VALUE")
print("="*70)

# Patch execute_trade_plan to print dry_run value
original_file = '../dashboard/ai_pm/execution_engine.py'

with open(original_file, 'r') as f:
    content = f.read()

# Check what the START log shows
print("Looking for dry_run logging...")

# The log shows: execute_trade_plan: START dry_run=False
# But the orders show dry_run: True

# Let me find where the order append happens with dry_run
idx = content.find('"dry_run": True')
if idx > 0:
    print(f"\nFound 'dry_run: True' at index {idx}")
    print("Context:")
    print(content[idx-100:idx+100])

# The issue is this block:
# if dry_run:
#     submitted_orders.append({**ticket_repr, "submitted": False, "dry_run": True})
#     continue

# So if dry_run=True, it appends and continues (skips placeOrder)

# Let me check test_full_validation.py to see what we're passing
print("\n" + "="*70)
print("Checking test_full_validation.py")
print("="*70)

with open('test_full_validation.py', 'r') as f:
    test_content = f.read()

# Find the execute_trade_plan call
idx = test_content.find('execute_trade_plan(')
if idx > 0:
    print(test_content[idx:idx+500])
