"""
Debug script to test JSON-based caching for bond dashboard.

Run: python scripts/debug_cache_test.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.getcwd())

print("=" * 70)
print("BOND CACHE DEBUG (JSON)")
print("=" * 70)

CACHE_DIR = Path("data/cache")
CACHE_FILE = CACHE_DIR / "bond_analysis_cache.json"

# Test 1: Directory
print("\n[TEST 1] Cache directory...")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"  ✅ {CACHE_DIR.absolute()}")

# Test 2: Write JSON
print("\n[TEST 2] Write JSON cache...")
test_data = {
    'cache_time': datetime.now().isoformat(),
    'signals': {
        'TLT': {'ticker': 'TLT', 'name': 'Test', 'current_price': 88.0, 'target_price': 90.0,
                'composite_score': 60, 'signal_value': 'BUY'},
    },
    'yields': {'yield_10y': 4.14, 'yield_30y': 4.80},
}

with open(CACHE_FILE, 'w') as f:
    json.dump(test_data, f, indent=2)
print(f"  ✅ Wrote {CACHE_FILE.stat().st_size} bytes")

# Test 3: Read JSON
print("\n[TEST 3] Read JSON cache...")
with open(CACHE_FILE, 'r') as f:
    loaded = json.load(f)
print(f"  ✅ Signals: {list(loaded['signals'].keys())}")

# Test 4: Check dashboard code
print("\n[TEST 4] Check dashboard...")
dash = Path("dashboard/bond_signals_dashboard.py")
if dash.exists():
    content = dash.read_text(encoding='utf-8')
    checks = [
        ("bond_analysis_cache.json", "bond_analysis_cache.json" in content),
        ("json.dump", "json.dump" in content),
        ("json.load", "json.load" in content),
    ]
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")

    if all(c[1] for c in checks):
        print("  ✅ Dashboard has JSON caching!")
    else:
        print("  ❌ Dashboard missing JSON - copy latest file!")

# Test 5: Import and test
print("\n[TEST 5] Test dashboard functions...")
try:
    from dashboard.bond_signals_dashboard import save_bond_cache, load_bond_cache, get_cache_info

    exists, age, n = get_cache_info()
    print(f"  Cache: exists={exists}, age={age}m, signals={n}")

    if exists:
        data = load_bond_cache()
        print(f"  ✅ Loaded {len(data.get('signals', {}))} signals")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 6: Real signal
print("\n[TEST 6] Real signal test...")
try:
    from src.analytics.bond_signals_institutional import InstitutionalBondSignalGenerator
    from dashboard.bond_signals_dashboard import save_bond_cache

    gen = InstitutionalBondSignalGenerator()
    sig = gen.generate_signal('TLT')

    if sig:
        print(f"  Signal: ${sig.current_price:.2f} → ${sig.target_price:.2f}")

        result = save_bond_cache({'signals': {'TLT': sig}, 'yields': None})

        if result:
            size = CACHE_FILE.stat().st_size
            print(f"  ✅ Saved! ({size} bytes)")

            # Verify
            with open(CACHE_FILE) as f:
                verify = json.load(f)
            print(f"  ✅ Verified: {list(verify['signals'].keys())}")
        else:
            print("  ❌ Save failed")
except Exception as e:
    print(f"  ⚠️ Skipped: {e}")

print("\n" + "=" * 70)
print("Copy command:")
print("  Copy-Item 'downloads\\bond_dashboard_complete.py' 'dashboard\\bond_signals_dashboard.py' -Force")