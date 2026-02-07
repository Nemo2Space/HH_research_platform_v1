"""
Step 1 Verification: Kill Switch Enhancement
Tests that the kill switch module works correctly.
"""
import json
import os
import sys
import tempfile

# Temporarily override kill switch dir to avoid polluting real config
TEST_DIR = tempfile.mkdtemp(prefix="hh_kill_test_")
os.environ["HH_KILL_SWITCH_DIR"] = TEST_DIR

# Now import (must be after env var is set)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard.ai_pm.kill_switch import (
    activate, deactivate, is_active, get_status, check_and_block,
    _KILL_SWITCH_FILE
)

def test_kill_switch():
    results = []
    
    # Test 1: Initially inactive
    assert not is_active(), "Kill switch should be inactive initially"
    results.append("✅ Test 1: Initially inactive")
    
    # Test 2: check_and_block returns None when inactive
    block = check_and_block("test")
    assert block is None, f"Expected None, got: {block}"
    results.append("✅ Test 2: check_and_block returns None when inactive")
    
    # Test 3: Activate
    info = activate(reason="Unit test activation", cancel_orders=False)
    assert is_active(), "Kill switch should be active after activation"
    assert "activated_at" in info
    assert info["reason"] == "Unit test activation"
    results.append("✅ Test 3: Activation works")
    
    # Test 4: File exists on disk
    ks_file = os.path.join(TEST_DIR, "KILL_SWITCH")
    assert os.path.exists(ks_file), f"Kill switch file should exist at {ks_file}"
    with open(ks_file) as f:
        content = json.load(f)
    assert content["reason"] == "Unit test activation"
    results.append("✅ Test 4: File-based persistence works")
    
    # Test 5: check_and_block returns error when active
    block = check_and_block("test_context")
    assert block is not None, "Should return block message when active"
    assert "KILL SWITCH ACTIVE" in block
    assert "Unit test activation" in block
    results.append("✅ Test 5: check_and_block blocks when active")
    
    # Test 6: Status shows details
    status = get_status()
    assert status["active"] is True
    assert status["reason"] == "Unit test activation"
    results.append("✅ Test 6: Status reports correctly")
    
    # Test 7: Deactivate
    info = deactivate(reason="Test complete")
    assert not is_active(), "Kill switch should be inactive after deactivation"
    results.append("✅ Test 7: Deactivation works")
    
    # Test 8: check_and_block returns None after deactivation
    block = check_and_block("test")
    assert block is None
    results.append("✅ Test 8: Trading resumes after deactivation")
    
    # Test 9: Log file was created
    log_file = os.path.join(TEST_DIR, "kill_switch_log.jsonl")
    assert os.path.exists(log_file), "Log file should exist"
    with open(log_file) as f:
        lines = f.readlines()
    assert len(lines) >= 2, f"Expected at least 2 log entries, got {len(lines)}"
    results.append("✅ Test 9: Audit log created with entries")
    
    # Test 10: Double deactivation is safe
    info = deactivate(reason="Already inactive")
    assert not is_active()
    assert info.get("note") == "Was not active"
    results.append("✅ Test 10: Double deactivation is safe (idempotent)")
    
    # Cleanup
    import shutil
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1 VERIFICATION: Kill Switch Enhancement")
    print("=" * 60)
    
    try:
        results = test_kill_switch()
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
