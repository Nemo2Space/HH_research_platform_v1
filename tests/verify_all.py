"""
MASTER VERIFICATION — Run All Step Tests
"""
import os
import sys
import subprocess

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(TEST_DIR)

tests = [
    ("Step 1: Kill Switch Enhancement", "verify_step1_kill_switch.py"),
    ("Step 2: Position Concentration Hard Limits", "verify_step2_concentration.py"),
    ("Step 3: Order Lifecycle & Session Reconciliation", "verify_step3_reconciliation.py"),
    ("Step 4: Point-in-Time Leakage Fix", "verify_step4_pit_fix.py"),
    ("Step 5: Contract Validation Fix", "verify_step5_contracts.py"),
    ("Step 6: Backtest Engine Fixes", "verify_step6_backtest.py"),
    ("Step 7: Exposure Control — Remove Hardcoded Values", "verify_step7_exposure.py"),
]

print("=" * 70)
print("  HH RESEARCH PLATFORM v1 — MASTER VERIFICATION SUITE")
print("=" * 70)

passed = 0
failed = 0

for step_name, test_file in tests:
    path = os.path.join(TEST_DIR, test_file)
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True, cwd=PROJ_DIR
    )
    
    if result.returncode == 0:
        passed += 1
        print(f"  ✅ {step_name}")
    else:
        failed += 1
        print(f"  ❌ {step_name}")
        # Show failure detail
        for line in result.stdout.split('\n'):
            if '❌' in line:
                print(f"     {line.strip()}")

print(f"\n{'=' * 70}")
print(f"  RESULTS: {passed}/{len(tests)} steps passed, {failed} failed")
if failed == 0:
    print(f"  ALL GATE 1 & GATE 2 FIXES VERIFIED ✅")
else:
    print(f"  ⚠️  {failed} step(s) need attention")
print(f"{'=' * 70}")

sys.exit(0 if failed == 0 else 1)
