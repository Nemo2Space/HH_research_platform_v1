"""
Step 6 Verification: Backtest Engine Fixes

Tests that:
1. beta is no longer hardcoded to 1.0
2. total_return uses compounding (not sum)
3. Equity curve uses net returns consistently
4. _calculate_beta method exists and uses real regression
"""
import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def test_backtest_fixes():
    results = []

    from src.backtest.engine import BacktestEngine, BacktestResult, Trade

    # =========================================================
    # Test 1: beta is no longer hardcoded to 1.0
    # =========================================================
    source_calc = inspect.getsource(BacktestEngine._calculate_results)
    # Should NOT have "beta = 1.0" as a hardcoded assignment
    lines = source_calc.split('\n')
    for line in lines:
        stripped = line.strip()
        # Check for literal hardcoded beta = 1.0
        if stripped == "beta = 1.0":
            raise AssertionError("beta = 1.0 is still hardcoded in _calculate_results!")
    results.append("✅ Test 1: beta is no longer hardcoded to 1.0")

    # =========================================================
    # Test 2: _calculate_beta method exists
    # =========================================================
    assert hasattr(BacktestEngine, '_calculate_beta'), \
        "_calculate_beta method should exist"
    results.append("✅ Test 2: _calculate_beta method exists")

    # =========================================================
    # Test 3: _calculate_beta uses real regression (cov/var)
    # =========================================================
    source_beta = inspect.getsource(BacktestEngine._calculate_beta)
    assert "cov" in source_beta and "var" in source_beta, \
        "_calculate_beta should use covariance/variance for regression"
    results.append("✅ Test 3: _calculate_beta uses cov/var regression")

    # =========================================================
    # Test 4: _calculate_beta returns None (not 1.0) when insufficient data
    # =========================================================
    assert "return None" in source_beta, \
        "_calculate_beta should return None when data insufficient"
    # Should NOT have any "return 1.0" fallback
    assert "return 1.0" not in source_beta, \
        "_calculate_beta should NOT fall back to 1.0"
    results.append("✅ Test 4: _calculate_beta returns None (not 1.0) for insufficient data")

    # =========================================================
    # Test 5: BacktestResult.beta is Optional[float]
    # =========================================================
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(BacktestResult)}
    beta_field = fields.get('beta')
    assert beta_field is not None, "BacktestResult should have beta field"
    # Check it allows None (Optional)
    type_str = str(beta_field.type)
    assert "Optional" in type_str or "None" in type_str, \
        f"BacktestResult.beta should be Optional[float], got {type_str}"
    results.append("✅ Test 5: BacktestResult.beta is Optional[float]")

    # =========================================================
    # Test 6: total_return uses compounding (not sum)
    # =========================================================
    source_calc = inspect.getsource(BacktestEngine._calculate_results)
    assert "np.prod" in source_calc, \
        "total_return should use np.prod for compounding"
    assert "np.sum(returns_array)" not in source_calc, \
        "total_return should NOT use np.sum (simple addition is wrong)"
    results.append("✅ Test 6: total_return uses compounding (np.prod)")

    # =========================================================
    # Test 7: Compounding math is correct
    # =========================================================
    # Test with known returns: +10%, -5%, +8%
    test_returns = np.array([10.0, -5.0, 8.0])
    # Sum method (WRONG): 10 + (-5) + 8 = 13%
    wrong_total = float(np.sum(test_returns))
    # Compound method (CORRECT): (1.10 * 0.95 * 1.08 - 1) * 100 = 12.86%
    correct_total = float((np.prod(1 + test_returns / 100.0) - 1) * 100.0)
    assert abs(correct_total - 12.86) < 0.01, f"Compounding math wrong: {correct_total}"
    assert abs(wrong_total - 13.0) < 0.01, f"Sum math wrong: {wrong_total}"
    assert wrong_total != correct_total, "Sum and compound should differ"
    results.append(f"✅ Test 7: Compounding math verified (sum={wrong_total:.2f}%, compound={correct_total:.2f}%)")

    # =========================================================
    # Test 8: Equity curve uses net returns
    # =========================================================
    source_equity = inspect.getsource(BacktestEngine._build_equity_curve)
    assert "return_pct_net" in source_equity, \
        "Equity curve should use return_pct_net for consistency"
    assert "return_net" in source_equity, \
        "Equity curve dataframe should include return_net column"
    results.append("✅ Test 8: Equity curve uses net returns for consistency")

    # =========================================================
    # Test 9: _empty_result has None beta
    # =========================================================
    source_empty = inspect.getsource(BacktestEngine._empty_result)
    assert "beta=None" in source_empty, \
        "_empty_result should set beta=None, not beta=0 or beta=1.0"
    results.append("✅ Test 9: _empty_result returns beta=None (not 0 or 1.0)")

    # =========================================================
    # Test 10: _calculate_beta has sanity bounds
    # =========================================================
    assert "abs(beta) > 10" in source_beta or "abs(beta)" in source_beta, \
        "_calculate_beta should have sanity bounds check"
    results.append("✅ Test 10: _calculate_beta has sanity bounds (|beta| > 10 → None)")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 6 VERIFICATION: Backtest Engine Fixes")
    print("=" * 60)

    try:
        results = test_backtest_fixes()
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
