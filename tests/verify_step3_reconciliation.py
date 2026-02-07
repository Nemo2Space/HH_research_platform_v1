"""
Step 3 Verification: Order Lifecycle & Session Reconciliation

Tests that the order reconciliation module correctly:
- Creates unique session IDs
- Tags orders with session references
- Tracks orders placed in a session
- Persists session data to disk
"""
import os
import sys
import tempfile
import shutil

# Set dirs to temp
TEST_DIR = tempfile.mkdtemp(prefix="hh_recon_test_")
os.environ["HH_KILL_SWITCH_DIR"] = os.path.join(TEST_DIR, "ks")
os.environ["HH_SESSION_DIR"] = os.path.join(TEST_DIR, "sessions")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.ai_pm.order_reconciliation import (
    OrderSession, tag_ib_order, get_previous_sessions, cleanup_old_sessions,
    _ORDER_REF_PREFIX
)


class MockOrder:
    """Mock IB order for testing."""
    def __init__(self):
        self.orderRef = ""
        self.ocaGroup = ""


def test_order_reconciliation():
    results = []

    # =========================================================
    # Test 1: Session creation
    # =========================================================
    session = OrderSession(account="TEST123")
    assert session.session_id.startswith(_ORDER_REF_PREFIX), \
        f"Session ID should start with '{_ORDER_REF_PREFIX}', got: {session.session_id}"
    assert session.account == "TEST123"
    assert session.order_count == 0
    results.append(f"✅ Test 1: Session created with ID: {session.session_id}")

    # =========================================================
    # Test 2: Unique session IDs
    # =========================================================
    session2 = OrderSession(account="TEST123")
    assert session.session_id != session2.session_id, "Each session should have unique ID"
    results.append("✅ Test 2: Session IDs are unique")

    # =========================================================
    # Test 3: Order reference generation
    # =========================================================
    ref1 = session.get_order_ref("AAPL", "BUY")
    ref2 = session.get_order_ref("MSFT", "SELL")
    assert session.session_id in ref1, f"Order ref should contain session ID: {ref1}"
    assert "AAPL" in ref1 and "BUY" in ref1
    assert "MSFT" in ref2 and "SELL" in ref2
    assert ref1 != ref2, "Order refs should be unique"
    results.append(f"✅ Test 3: Order refs generated — {ref1}, {ref2}")

    # =========================================================
    # Test 4: Order tagging
    # =========================================================
    mock_order = MockOrder()
    tag_ib_order(mock_order, ref1)
    assert mock_order.orderRef == ref1, f"Order should be tagged: {mock_order.orderRef}"
    results.append("✅ Test 4: IB order tagged with session reference")

    # =========================================================
    # Test 5: Order recording
    # =========================================================
    session.record_order({
        "symbol": "AAPL", "action": "BUY", "quantity": 10,
        "order_id": 1001, "order_ref": ref1,
    })
    assert session.order_count == 1
    session.record_order({
        "symbol": "MSFT", "action": "SELL", "quantity": 5,
        "order_id": 1002, "order_ref": ref2,
    })
    assert session.order_count == 2
    results.append(f"✅ Test 5: {session.order_count} orders recorded in session")

    # =========================================================
    # Test 6: Session persistence
    # =========================================================
    sessions_dir = os.environ["HH_SESSION_DIR"]
    session_files = [f for f in os.listdir(sessions_dir) if f.startswith("session_")]
    assert len(session_files) >= 2, f"Expected at least 2 session files, found {len(session_files)}"
    results.append(f"✅ Test 6: {len(session_files)} session files persisted to disk")

    # =========================================================
    # Test 7: Previous sessions retrieval
    # =========================================================
    prev = get_previous_sessions()
    assert len(prev) >= 2, f"Expected at least 2 previous sessions, got {len(prev)}"
    # Most recent should be first
    assert prev[0]["session_id"] == session2.session_id or prev[0]["session_id"] == session.session_id
    results.append(f"✅ Test 7: {len(prev)} previous sessions retrieved")

    # =========================================================
    # Test 8: Session cleanup
    # =========================================================
    cleanup_old_sessions(keep_last_n=1)
    remaining = get_previous_sessions()
    assert len(remaining) <= 1, f"Expected <= 1 after cleanup, got {len(remaining)}"
    results.append(f"✅ Test 8: Cleanup kept {len(remaining)} session(s)")

    # =========================================================
    # Test 9: Verify execution_engine has session support
    # =========================================================
    import inspect
    from dashboard.ai_pm import execution_engine
    sig = inspect.signature(execution_engine.execute_trade_plan)
    assert "session" in sig.parameters, "execute_trade_plan should accept 'session' parameter"
    results.append("✅ Test 9: execute_trade_plan accepts 'session' parameter")

    # =========================================================
    # Test 10: Verify order tagging code in execution_engine
    # =========================================================
    source = inspect.getsource(execution_engine.execute_trade_plan)
    assert "tag_ib_order" in source, "Order tagging not found in execution_engine"
    assert "session.record_order" in source, "Order recording not found in execution_engine"
    assert "order_ref" in source, "order_ref not found in execution_engine"
    results.append("✅ Test 10: Order tagging & recording wired into execution_engine")

    # Cleanup
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3 VERIFICATION: Order Lifecycle & Session Reconciliation")
    print("=" * 60)

    try:
        results = test_order_reconciliation()
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
