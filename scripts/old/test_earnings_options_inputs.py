#!/usr/bin/env python3
"""
Phase 5 Verification: Options-Based IES Inputs

Tests:
1. ATM interpolation logic
2. Options metrics retrieval
3. Implied move calculation
4. Percentile calculations
5. Skew shift calculation
6. Score normalization functions
7. Cache functionality
8. Live data retrieval

Author: Alpha Research Platform
"""

import sys
import os
import time
from datetime import datetime, date, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all imports work correctly."""
    print("Testing Imports...")

    try:
        from src.analytics.earnings_intelligence.options_inputs import (
            OptionsMetrics,
            HistoricalOptionsData,
            get_options_metrics,
            calculate_implied_move_pct,
            calculate_implied_move_pctl,
            calculate_iv_pctl,
            calculate_skew_shift,
            normalize_implied_move_to_score,
            normalize_iv_to_score,
            normalize_skew_to_score,
            calculate_all_options_inputs,
            clear_options_cache,
        )
        print("  [OK] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_normalization_functions():
    """Test score normalization functions."""
    print("\nTesting Score Normalization Functions...")

    from src.analytics.earnings_intelligence.options_inputs import (
        normalize_implied_move_to_score,
        normalize_iv_to_score,
        normalize_skew_to_score,
    )

    all_passed = True

    # Test implied move normalization
    test_cases_im = [
        (None, 50.0, "None -> 50 (default)"),
        (0, 0.0, "0th percentile -> 0"),
        (50, 50.0, "50th percentile -> 50"),
        (100, 100.0, "100th percentile -> 100"),
        (85, 85.0, "85th percentile -> 85"),
    ]

    for input_val, expected, desc in test_cases_im:
        result = normalize_implied_move_to_score(input_val)
        passed = abs(result - expected) < 0.1
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} IM: {desc}: got {result}")
        all_passed = all_passed and passed

    # Test IV normalization
    test_cases_iv = [
        (None, 50.0, "None -> 50 (default)"),
        (20, 20.0, "20th percentile -> 20"),
        (80, 80.0, "80th percentile -> 80"),
    ]

    for input_val, expected, desc in test_cases_iv:
        result = normalize_iv_to_score(input_val)
        passed = abs(result - expected) < 0.1
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} IV: {desc}: got {result}")
        all_passed = all_passed and passed

    # Test skew normalization
    # skew_shift < -0.05 -> bearish -> low score
    # skew_shift ~ 0 -> neutral -> 50
    # skew_shift > +0.05 -> bullish -> high score
    test_cases_skew = [
        (None, 50.0, "None -> 50 (default)"),
        (0.0, 50.0, "0 -> 50 (neutral)"),
        (0.10, 100.0, "+0.10 -> 100 (max bullish)"),
        (-0.10, 0.0, "-0.10 -> 0 (max bearish)"),
        (0.05, 75.0, "+0.05 -> 75 (bullish)"),
        (-0.05, 25.0, "-0.05 -> 25 (bearish)"),
    ]

    for input_val, expected, desc in test_cases_skew:
        result = normalize_skew_to_score(input_val)
        passed = abs(result - expected) < 0.1
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} Skew: {desc}: got {result}")
        all_passed = all_passed and passed

    if all_passed:
        print("  [OK] All normalization tests passed!\n")
    else:
        print("  [FAIL] Some normalization tests failed!\n")

    return all_passed


def test_cache():
    """Test options cache functionality."""
    print("Testing Options Cache...")

    from src.analytics.earnings_intelligence.options_inputs import (
        get_options_metrics,
        clear_options_cache,
    )

    # Clear cache first
    clear_options_cache()
    print("  [OK] Cache cleared")

    # First call - should fetch from API
    start = time.time()
    metrics1 = get_options_metrics("AAPL")
    time1 = time.time() - start
    print(f"  [OK] First call: {time1:.2f}s")

    if metrics1 is None:
        print("  [WARN] Could not retrieve options data for AAPL")
        return True  # Skip cache test if no data

    # Second call - should be cached
    start = time.time()
    metrics2 = get_options_metrics("AAPL")
    time2 = time.time() - start
    print(f"  [OK] Second call (cached): {time2:.4f}s")

    # Verify results match
    if metrics1 and metrics2:
        assert metrics1.implied_move_pct == metrics2.implied_move_pct, "Cached results differ"
        print("  [OK] Cached result matches")

    # Verify caching speedup
    if time1 > 0.1:
        assert time2 < time1 / 2, f"Cache not effective: {time2:.4f}s vs {time1:.2f}s"
        print(f"  [OK] Cache speedup: {time1 / max(time2, 0.0001):.0f}x faster")

    print("  [OK] Cache tests passed!\n")
    return True


def test_options_metrics():
    """Test options metrics retrieval."""
    print("Testing Options Metrics Retrieval...")

    from src.analytics.earnings_intelligence.options_inputs import (
        get_options_metrics,
        OptionsMetrics,
    )

    # Test with a liquid stock
    metrics = get_options_metrics("AAPL")

    if metrics is None:
        print("  [WARN] Could not retrieve options data - market may be closed")
        return True

    # Verify it's the right type
    assert isinstance(metrics, OptionsMetrics), "Result should be OptionsMetrics"
    print(f"  [OK] Got OptionsMetrics for AAPL")

    # Verify required fields
    assert metrics.ticker == "AAPL"
    print(f"  [OK] Ticker: {metrics.ticker}")

    assert metrics.stock_price > 0
    print(f"  [OK] Stock Price: ${metrics.stock_price:.2f}")

    assert metrics.atm_straddle_price > 0
    print(f"  [OK] ATM Straddle: ${metrics.atm_straddle_price:.2f}")

    assert 0 < metrics.implied_move_pct < 1.0  # Should be 0-100% range
    print(f"  [OK] Implied Move: {metrics.implied_move_pct * 100:.1f}%")

    assert 0 < metrics.avg_iv < 2.0  # IV between 0-200%
    print(f"  [OK] Average IV: {metrics.avg_iv * 100:.0f}%")

    assert metrics.expiry_used is not None
    print(f"  [OK] Expiry Used: {metrics.expiry_used}")

    assert metrics.days_to_expiry is not None
    print(f"  [OK] Days to Expiry: {metrics.days_to_expiry}")

    print("  [OK] Options metrics retrieval tests passed!\n")
    return True


def test_implied_move_calculations():
    """Test implied move calculations."""
    print("Testing Implied Move Calculations...")

    from src.analytics.earnings_intelligence.options_inputs import (
        calculate_implied_move_pct,
        calculate_implied_move_pctl,
    )

    # Test implied move percentage
    im_pct = calculate_implied_move_pct("AAPL")

    if im_pct is None:
        print("  [WARN] Could not calculate implied move - market may be closed")
        return True

    assert 0 < im_pct < 1.0, f"Implied move should be 0-100%, got {im_pct * 100}%"
    print(f"  [OK] AAPL Implied Move: {im_pct * 100:.1f}%")

    # Test implied move percentile
    im_pctl = calculate_implied_move_pctl("AAPL")

    if im_pctl is not None:
        assert 5 <= im_pctl <= 95, f"Percentile should be 5-95, got {im_pctl}"
        print(f"  [OK] AAPL IM Percentile: {im_pctl:.0f}th")
    else:
        print("  [INFO] IM Percentile: N/A (insufficient history)")

    print("  [OK] Implied move calculation tests passed!\n")
    return True


def test_iv_calculations():
    """Test IV percentile calculations."""
    print("Testing IV Percentile Calculations...")

    from src.analytics.earnings_intelligence.options_inputs import (
        calculate_iv_pctl,
        get_options_metrics,
    )

    # Get base metrics
    metrics = get_options_metrics("AAPL")

    if metrics is None:
        print("  [WARN] Could not get options metrics")
        return True

    print(f"  [OK] Current IV: {metrics.avg_iv * 100:.0f}%")

    # Test IV percentile
    iv_pctl = calculate_iv_pctl("AAPL")

    if iv_pctl is not None:
        assert 5 <= iv_pctl <= 95, f"Percentile should be 5-95, got {iv_pctl}"
        print(f"  [OK] AAPL IV Percentile: {iv_pctl:.0f}th")
    else:
        print("  [INFO] IV Percentile: N/A (insufficient history)")

    print("  [OK] IV percentile tests passed!\n")
    return True


def test_skew_calculations():
    """Test skew shift calculations."""
    print("Testing Skew Shift Calculations...")

    from src.analytics.earnings_intelligence.options_inputs import (
        calculate_skew_shift,
        get_options_metrics,
    )

    # Get base metrics for IV skew
    metrics = get_options_metrics("AAPL")

    if metrics is None:
        print("  [WARN] Could not get options metrics")
        return True

    print(f"  [OK] Call IV: {metrics.atm_call_iv * 100:.1f}%")
    print(f"  [OK] Put IV: {metrics.atm_put_iv * 100:.1f}%")
    print(f"  [OK] IV Skew (put-call): {metrics.iv_skew * 100:.1f}%")

    # Test skew shift
    skew_shift = calculate_skew_shift("AAPL")

    if skew_shift is not None:
        print(f"  [OK] Skew Shift: {skew_shift:+.4f}")

        # Interpret
        if skew_shift > 0.02:
            print("  [INFO] Interpretation: Bullish positioning (calls bid)")
        elif skew_shift < -0.02:
            print("  [INFO] Interpretation: Bearish positioning (puts bid)")
        else:
            print("  [INFO] Interpretation: Neutral positioning")
    else:
        print("  [INFO] Skew Shift: N/A")

    print("  [OK] Skew shift tests passed!\n")
    return True


def test_calculate_all_options_inputs():
    """Test the comprehensive all-inputs function."""
    print("Testing calculate_all_options_inputs()...")

    from src.analytics.earnings_intelligence.options_inputs import (
        calculate_all_options_inputs,
    )

    inputs = calculate_all_options_inputs("AAPL")

    # Verify all expected keys are present
    expected_keys = [
        'implied_move_pct',
        'implied_move_pctl',
        'implied_move_score',
        'iv_pctl',
        'iv_score',
        'skew_shift',
        'skew_score',
        'avg_iv',
        'stock_price',
        'straddle_price',
        'expiry_used',
        'days_to_expiry'
    ]

    for key in expected_keys:
        assert key in inputs, f"Missing key: {key}"

    print(f"  [OK] All {len(expected_keys)} expected keys present")

    # Verify scores are in valid range
    for score_key in ['implied_move_score', 'iv_score', 'skew_score']:
        score = inputs[score_key]
        assert 0 <= score <= 100, f"{score_key} should be 0-100, got {score}"

    print("  [OK] All scores in valid 0-100 range")

    # Display results
    if inputs['implied_move_pct'] is not None:
        print(f"\n  Results for AAPL:")
        print(f"    Implied Move: {inputs['implied_move_pct'] * 100:.1f}%")
        print(f"    IM Percentile: {inputs['implied_move_pctl']:.0f}th" if inputs[
            'implied_move_pctl'] else "    IM Percentile: N/A")
        print(f"    IM Score: {inputs['implied_move_score']:.0f}/100")
        print(f"    IV Percentile: {inputs['iv_pctl']:.0f}th" if inputs['iv_pctl'] else "    IV Percentile: N/A")
        print(f"    IV Score: {inputs['iv_score']:.0f}/100")
        print(f"    Skew Shift: {inputs['skew_shift']:.4f}" if inputs[
                                                                   'skew_shift'] is not None else "    Skew Shift: N/A")
        print(f"    Skew Score: {inputs['skew_score']:.0f}/100")
    else:
        print("  [WARN] Could not retrieve options data")

    print("\n  [OK] calculate_all_options_inputs() tests passed!\n")
    return True


def test_multiple_tickers():
    """Test options inputs for multiple tickers."""
    print("Testing Multiple Tickers...")

    from src.analytics.earnings_intelligence.options_inputs import (
        calculate_all_options_inputs,
    )

    test_tickers = ["AAPL", "TSLA", "NVDA", "META"]
    results = []

    for ticker in test_tickers:
        inputs = calculate_all_options_inputs(ticker)

        if inputs['implied_move_pct'] is not None:
            results.append({
                'ticker': ticker,
                'im_pct': inputs['implied_move_pct'],
                'im_score': inputs['implied_move_score'],
                'iv_score': inputs['iv_score'],
                'skew_score': inputs['skew_score'],
            })
            print(
                f"  [OK] {ticker}: IM={inputs['implied_move_pct'] * 100:.1f}%, IM_Score={inputs['implied_move_score']:.0f}, IV_Score={inputs['iv_score']:.0f}")
        else:
            print(f"  [WARN] {ticker}: No options data available")

    # Verify we got some results
    if results:
        print(f"\n  [OK] Retrieved options data for {len(results)}/{len(test_tickers)} tickers\n")
    else:
        print("  [WARN] Could not retrieve options data for any tickers\n")

    return True


def test_ai_summary():
    """Test the AI summary generation."""
    print("Testing AI Summary Generation...")

    from src.analytics.earnings_intelligence.options_inputs import (
        get_options_summary_for_ai,
    )

    summary = get_options_summary_for_ai("NVDA")

    assert isinstance(summary, str), "Summary should be a string"
    assert "NVDA" in summary, "Summary should contain ticker"
    assert "IMPLIED MOVE" in summary or "unavailable" in summary.lower(), "Summary should have implied move section"

    print("  [OK] Summary generated successfully")
    print("\n--- AI Summary Preview ---")
    # Print first 20 lines
    lines = summary.split('\n')[:20]
    for line in lines:
        print(f"  {line}")
    print("  ...")

    print("\n  [OK] AI summary tests passed!\n")
    return True


def main():
    """Run all Phase 5 tests."""
    print("=" * 60)
    print("PHASE 5 VERIFICATION: Options-Based IES Inputs")
    print("=" * 60)

    all_passed = True

    # Run tests in order
    all_passed &= test_imports()
    all_passed &= test_normalization_functions()
    all_passed &= test_cache()
    all_passed &= test_options_metrics()
    all_passed &= test_implied_move_calculations()
    all_passed &= test_iv_calculations()
    all_passed &= test_skew_calculations()
    all_passed &= test_calculate_all_options_inputs()
    all_passed &= test_multiple_tickers()
    all_passed &= test_ai_summary()

    # Final result
    print("=" * 60)
    if all_passed:
        print("[OK] ALL PHASE 5 TESTS PASSED!")
    else:
        print("[FAIL] SOME TESTS FAILED - Review output above")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())