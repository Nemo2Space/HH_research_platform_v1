import time, sys

# Test 1: Direct subprocess call
print("Test 1: Direct subprocess worker (AAPL)...")
start = time.time()
result = __import__('subprocess').run(
    [sys.executable, '-m', 'src.analytics.yahoo_options_subprocess', 'AAPL', '4'],
    capture_output=True, text=True, timeout=15, check=False
)
elapsed = time.time() - start
import json
payload = json.loads(result.stdout.strip().splitlines()[-1])
print(f"  ok={payload.get('ok')}, price={payload.get('stock_price')}, time={elapsed:.1f}s")

# Test 2: Through the class
print("\nTest 2: Through OptionsFlowAnalyzer (skip_ibkr=True)...")
from src.analytics.options_flow import OptionsFlowAnalyzer
ofa = OptionsFlowAnalyzer()
start = time.time()
calls, puts, price, source = ofa.get_options_chain('AAPL', skip_ibkr=True)
elapsed = time.time() - start
print(f"  calls={len(calls)}, puts={len(puts)}, price={price}, source={source}, time={elapsed:.1f}s")

# Test 3: Ticker that might hang (PCAR)
print("\nTest 3: PCAR (the problematic ticker)...")
start = time.time()
calls, puts, price, source = ofa.get_options_chain('PCAR', skip_ibkr=True)
elapsed = time.time() - start
print(f"  calls={len(calls)}, puts={len(puts)}, price={price}, source={source}, time={elapsed:.1f}s")

print("\nAll tests passed - script exited cleanly")
