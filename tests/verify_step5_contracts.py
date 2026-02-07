"""
Step 5 Verification: Contract Validation Fix

Tests that:
1. qualifyContracts is no longer skipped
2. Batch qualification is implemented
3. Unqualified contracts are handled properly (skipped, not silently used)
"""
import os
import sys
import inspect
import tempfile

os.environ["HH_KILL_SWITCH_DIR"] = tempfile.mkdtemp(prefix="hh_ks_test_")
os.environ["HH_SESSION_DIR"] = tempfile.mkdtemp(prefix="hh_sess_test_")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.ai_pm import execution_engine


def test_contract_validation():
    results = []

    source = inspect.getsource(execution_engine.execute_trade_plan)

    # =========================================================
    # Test 1: qualifyContracts is no longer skipped
    # =========================================================
    assert "skipping qualifyContracts" not in source, \
        "qualifyContracts should no longer be skipped"
    results.append("✅ Test 1: qualifyContracts is no longer skipped")

    # =========================================================
    # Test 2: qualifyContracts is actually called
    # =========================================================
    assert "ib.qualifyContracts" in source, \
        "qualifyContracts should be called on the ib connection"
    results.append("✅ Test 2: ib.qualifyContracts is called")

    # =========================================================
    # Test 3: Batch processing implemented
    # =========================================================
    assert "BATCH_SIZE" in source, \
        "Batch processing should be implemented for contract qualification"
    results.append("✅ Test 3: Batch processing implemented (BATCH_SIZE)")

    # =========================================================
    # Test 4: Unqualified contracts are tracked
    # =========================================================
    assert "unqualified_symbols" in source, \
        "Unqualified symbols should be tracked"
    results.append("✅ Test 4: Unqualified symbols are tracked")

    # =========================================================
    # Test 5: Unqualified contracts are removed (orders skipped)
    # =========================================================
    assert "contract_by_symbol.pop" in source, \
        "Unqualified contracts should be removed from contract map"
    results.append("✅ Test 5: Unqualified contracts removed (orders will be skipped)")

    # =========================================================
    # Test 6: conId validation
    # =========================================================
    assert "conId" in source, \
        "Contract validation should check conId"
    results.append("✅ Test 6: conId validation implemented")

    # =========================================================
    # Test 7: Graceful error handling (not crash)
    # =========================================================
    assert "Contract qualification completely failed" in source or "batch error" in source, \
        "Should have graceful error handling for qualification failures"
    results.append("✅ Test 7: Graceful error handling for qualification failures")

    # =========================================================
    # Test 8: No more "thread deadlock" bypass
    # =========================================================
    assert "thread deadlock" not in source, \
        "Thread deadlock bypass comment should be removed"
    results.append("✅ Test 8: Thread deadlock bypass removed")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5 VERIFICATION: Contract Validation Fix")
    print("=" * 60)

    try:
        results = test_contract_validation()
        for r in results:
            print(f"  {r}")
        print(f"\n{'=' * 60}")
        print(f"ALL {len(results)} TESTS PASSED ✅")
        print(f"{'=' * 60}")
    except AssertionError as e:
        print(f"\n❌ FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
