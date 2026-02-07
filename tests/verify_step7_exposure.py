"""
Step 7 Verification: Exposure Control — Remove Hardcoded Fallback Values

Tests that:
1. _calculate_portfolio_beta returns None (not 1.0) when data unavailable
2. _calculate_portfolio_volatility returns None (not 0.15) when data unavailable
3. _calculate_factor_exposure returns None (not 0.0) when data unavailable
4. ExposureReport fields are Optional where appropriate
"""
import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def test_exposure_control_fixes():
    results = []

    from src.portfolio.exposure_control import (
        ExposureController, ExposureReport, ExposureLimits, ExposureStatus
    )

    # =========================================================
    # Test 1: _calculate_portfolio_beta returns None on empty data
    # =========================================================
    source = inspect.getsource(ExposureController._calculate_portfolio_beta)
    # Should NOT have "return 1.0" as actual code (not in comments)
    code_lines = [line for line in source.split('\n') 
                  if not line.strip().startswith('#') 
                  and 'FIXED:' not in line
                  and 'was return' not in line]
    code_only = '\n'.join(code_lines)
    assert "return 1.0" not in code_only, \
        "_calculate_portfolio_beta should NOT return hardcoded 1.0"
    assert "return None" in source, \
        "_calculate_portfolio_beta should return None when data unavailable"
    results.append("✅ Test 1: _calculate_portfolio_beta returns None (not 1.0)")

    # =========================================================
    # Test 2: _calculate_portfolio_volatility returns None on empty data
    # =========================================================
    source_vol = inspect.getsource(ExposureController._calculate_portfolio_volatility)
    vol_code_lines = [line for line in source_vol.split('\n') 
                      if not line.strip().startswith('#')
                      and 'FIXED:' not in line
                      and 'was return' not in line]
    vol_code_only = '\n'.join(vol_code_lines)
    assert "return 0.15" not in vol_code_only, \
        "_calculate_portfolio_volatility should NOT return hardcoded 0.15"
    assert "return None" in source_vol, \
        "_calculate_portfolio_volatility should return None when data unavailable"
    results.append("✅ Test 2: _calculate_portfolio_volatility returns None (not 0.15)")

    # =========================================================
    # Test 3: _calculate_factor_exposure returns None on empty data
    # =========================================================
    source_factor = inspect.getsource(ExposureController._calculate_factor_exposure)
    factor_code_lines = [line for line in source_factor.split('\n')
                         if not line.strip().startswith('#')
                         and 'FIXED:' not in line
                         and 'was return' not in line]
    factor_code_only = '\n'.join(factor_code_lines)
    assert "return 0.0" not in factor_code_only, \
        "_calculate_factor_exposure should NOT return hardcoded 0.0"
    assert "return None" in source_factor, \
        "_calculate_factor_exposure should return None when data unavailable"
    results.append("✅ Test 3: _calculate_factor_exposure returns None (not 0.0)")

    # =========================================================
    # Test 4: ExposureReport fields are Optional
    # =========================================================
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(ExposureReport)}
    
    for field_name in ['portfolio_beta', 'portfolio_volatility', 'vol_scaling_factor',
                       'var_95_pct', 'max_drawdown_estimate']:
        f = fields.get(field_name)
        assert f is not None, f"ExposureReport missing field: {field_name}"
        type_str = str(f.type)
        assert "Optional" in type_str or "None" in type_str, \
            f"ExposureReport.{field_name} should be Optional, got: {type_str}"
    results.append("✅ Test 4: ExposureReport fields are Optional where appropriate")

    # =========================================================
    # Test 5: Functional test — beta returns None on empty DataFrame
    # =========================================================
    controller = ExposureController()
    # Override _get_returns to return empty DataFrame
    controller._get_returns = lambda symbols, days: pd.DataFrame()
    
    beta = controller._calculate_portfolio_beta(["AAPL"], np.array([1.0]))
    assert beta is None, f"Beta should be None when data empty, got: {beta}"
    results.append("✅ Test 5: Functional — beta is None when data is empty")

    # =========================================================
    # Test 6: Functional test — volatility returns None on empty DataFrame
    # =========================================================
    vol = controller._calculate_portfolio_volatility(["AAPL"], np.array([1.0]))
    assert vol is None, f"Vol should be None when data empty, got: {vol}"
    results.append("✅ Test 6: Functional — volatility is None when data is empty")

    # =========================================================
    # Test 7: Functional test — factor exposure returns None on empty DataFrame
    # =========================================================
    factor_exp = controller._calculate_factor_exposure(
        ["AAPL"], np.array([1.0]), "QQQ"
    )
    assert factor_exp is None, f"Factor exposure should be None when data empty, got: {factor_exp}"
    results.append("✅ Test 7: Functional — factor exposure is None when data is empty")

    # =========================================================
    # Test 8: analyze_portfolio handles None beta gracefully
    # =========================================================
    report = controller.analyze_portfolio([{"symbol": "AAPL", "weight": 1.0}])
    assert report.portfolio_beta is None, \
        f"Report beta should be None when data unavailable, got: {report.portfolio_beta}"
    # Should have a warning about unavailable data
    has_beta_warning = any("not available" in w.lower() or "data" in w.lower() 
                          for w in report.warnings)
    assert has_beta_warning, f"Should warn about missing beta data. Warnings: {report.warnings}"
    results.append("✅ Test 8: analyze_portfolio shows 'Data not available' for beta")

    # =========================================================
    # Test 9: VaR and drawdown are None when vol unavailable
    # =========================================================
    assert report.var_95_pct is None, \
        f"VaR should be None when vol unavailable, got: {report.var_95_pct}"
    assert report.max_drawdown_estimate is None, \
        f"Drawdown should be None when vol unavailable, got: {report.max_drawdown_estimate}"
    results.append("✅ Test 9: VaR and drawdown are None when vol unavailable")

    # =========================================================
    # Test 10: No hardcoded defaults anywhere in key methods
    # =========================================================
    full_source = inspect.getsource(ExposureController)
    # Scan for suspicious hardcoded defaults in return statements
    suspicious = []
    for line_num, line in enumerate(full_source.split('\n'), 1):
        stripped = line.strip()
        # Only check actual return statements, not comments or FIXED annotations
        if (stripped.startswith('return') 
            and not '#' in stripped  # Exclude lines with comments
            and not 'FIXED:' in line
            and not 'was return' in line):
            if 'return 1.0' in stripped or 'return 0.15' in stripped:
                suspicious.append(f"Line ~{line_num}: {stripped}")
    
    assert len(suspicious) == 0, \
        f"Found suspicious hardcoded returns: {suspicious}"
    results.append("✅ Test 10: No hardcoded fallback values (1.0, 0.15) in return statements")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 7 VERIFICATION: Exposure Control — Remove Hardcoded Values")
    print("=" * 60)

    try:
        results = test_exposure_control_fixes()
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
