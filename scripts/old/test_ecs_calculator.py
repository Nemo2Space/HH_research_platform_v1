#!/usr/bin/env python3
"""
Phase 9 Verification: ECS Calculator (Expectations Clearance Score)

Tests:
1. Required Z calculation
2. ECS category classification
3. Clearance margin calculation
4. Combined IES + EQS integration
5. Score adjustments
6. Data quality assessment
7. AI summary generation

Author: Alpha Research Platform
"""

# CRITICAL: Set path before ANY other imports
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime, date, timedelta


def test_imports():
    """Test that all imports work correctly."""
    print("Testing Imports...")

    try:
        from src.analytics.earnings_intelligence.ecs_calculator import (
            calculate_ecs,
            calculate_ecs_batch,
            ECSCalculationResult,
            calculate_required_z,
            calculate_clearance_margin,
            get_ecs_summary_for_ai,
            get_ecs_for_screener,
            get_full_earnings_analysis,
        )

        from src.analytics.earnings_intelligence.models import (
            ECSCategory,
            ExpectationsRegime,
            DataQuality,
            REQUIRED_Z_BASE,
            REQUIRED_Z_SLOPE,
            REQUIRED_Z_FLOOR,
            REQUIRED_Z_CEILING,
        )

        print("  [OK] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_required_z_calculation():
    """Test required_z calculation."""
    print("\nTesting Required Z Calculation...")

    from src.analytics.earnings_intelligence.ecs_calculator import calculate_required_z
    from src.analytics.earnings_intelligence.models import (
        REQUIRED_Z_BASE,
        REQUIRED_Z_SLOPE,
        REQUIRED_Z_FLOOR,
        REQUIRED_Z_CEILING,
    )

    print(
        f"  Parameters: BASE={REQUIRED_Z_BASE}, SLOPE={REQUIRED_Z_SLOPE}, FLOOR={REQUIRED_Z_FLOOR}, CEILING={REQUIRED_Z_CEILING}")

    # Test various implied move percentiles
    test_cases = [
        (0, REQUIRED_Z_FLOOR),  # Low expectations -> floor
        (25, None),  # Low-mid
        (50, None),  # Median
        (75, None),  # High
        (100, REQUIRED_Z_CEILING),  # Extreme -> ceiling
    ]

    for implied_move_pctl, expected in test_cases:
        required_z = calculate_required_z(implied_move_pctl=implied_move_pctl)

        # Calculate expected (before clamping)
        raw = REQUIRED_Z_BASE + (implied_move_pctl * REQUIRED_Z_SLOPE)
        clamped = max(REQUIRED_Z_FLOOR, min(REQUIRED_Z_CEILING, raw))

        print(f"  IM_pctl={implied_move_pctl:3d}: required_z={required_z:.2f} (raw={raw:.2f})")

        assert abs(required_z - clamped) < 0.01, f"Mismatch for IM={implied_move_pctl}"

    # Test with IES as fallback
    required_z_ies = calculate_required_z(ies=70, implied_move_pctl=None)
    print(f"  IES=70 (no IM): required_z={required_z_ies:.2f}")

    # Test with nothing
    required_z_default = calculate_required_z()
    print(f"  Default (50): required_z={required_z_default:.2f}")
    assert abs(required_z_default - 1.5) < 0.01, "Default should be ~1.5"

    print("  [OK] All required_z tests passed!\n")
    return True


def test_ecs_category_classification():
    """Test ECS category classification from event_z and required_z."""
    print("Testing ECS Category Classification...")

    from src.analytics.earnings_intelligence.models import ECSCategory

    required_z = 1.5  # Fixed bar for testing

    # Test cases: (event_z, expected_category)
    # Note: Floating point precision can affect boundary cases
    test_cases = [
        (2.5, ECSCategory.STRONG_BEAT),  # diff = 1.0 > 0.5 ✓
        (2.01, ECSCategory.STRONG_BEAT),  # diff = 0.51 > 0.5 ✓
        (2.0, ECSCategory.BEAT),  # diff = 0.5 (boundary - NOT > 0.5)
        (1.5, ECSCategory.BEAT),  # diff = 0 (exact)
        (1.3, ECSCategory.INLINE),  # diff = -0.2
        (1.21, ECSCategory.INLINE),  # diff = -0.29 (safely INLINE)
        (1.0, ECSCategory.MISS),  # diff = -0.5
        (0.51, ECSCategory.MISS),  # diff = -0.99 (safely MISS)
        (0.4, ECSCategory.STRONG_MISS),  # diff = -1.1
        (-0.5, ECSCategory.STRONG_MISS),  # diff = -2.0
    ]

    for event_z, expected in test_cases:
        category = ECSCategory.from_event_z(event_z, required_z)
        diff = event_z - required_z
        print(f"  event_z={event_z:+.1f}, diff={diff:+.2f}: {category.value} {'✓' if category == expected else '✗'}")
        assert category == expected, f"Expected {expected.value}, got {category.value}"

    # Test with None values
    unknown = ECSCategory.from_event_z(None, required_z)
    assert unknown == ECSCategory.UNKNOWN, "None event_z should be UNKNOWN"

    unknown2 = ECSCategory.from_event_z(1.0, None)
    assert unknown2 == ECSCategory.UNKNOWN, "None required_z should be UNKNOWN"

    print("  [OK] All ECS category tests passed!\n")
    return True


def test_ecs_category_properties():
    """Test ECS category properties (is_positive, score_adjustment)."""
    print("Testing ECS Category Properties...")

    from src.analytics.earnings_intelligence.models import ECSCategory

    # Test is_positive
    assert ECSCategory.STRONG_BEAT.is_positive == True
    assert ECSCategory.BEAT.is_positive == True
    assert ECSCategory.INLINE.is_positive == False
    assert ECSCategory.MISS.is_positive == False
    assert ECSCategory.STRONG_MISS.is_positive == False
    print("  [OK] is_positive correct")

    # Test is_negative
    assert ECSCategory.STRONG_BEAT.is_negative == False
    assert ECSCategory.BEAT.is_negative == False
    assert ECSCategory.INLINE.is_negative == False
    assert ECSCategory.MISS.is_negative == True
    assert ECSCategory.STRONG_MISS.is_negative == True
    print("  [OK] is_negative correct")

    # Test score_adjustment
    adjustments = {
        ECSCategory.STRONG_BEAT: 18,
        ECSCategory.BEAT: 8,
        ECSCategory.INLINE: 0,
        ECSCategory.MISS: -12,
        ECSCategory.STRONG_MISS: -22,
    }

    for cat, expected in adjustments.items():
        assert cat.score_adjustment == expected, f"{cat.value} adjustment wrong"
        print(f"  {cat.value}: {cat.score_adjustment:+d}")

    print("  [OK] All category property tests passed!\n")
    return True


def test_clearance_margin():
    """Test clearance margin calculation."""
    print("Testing Clearance Margin Calculation...")

    from src.analytics.earnings_intelligence.ecs_calculator import calculate_clearance_margin

    # Normal case
    margin = calculate_clearance_margin(2.0, 1.5)
    assert margin == 0.5, f"Expected 0.5, got {margin}"
    print(f"  event_z=2.0, required_z=1.5: margin={margin:+.2f}")

    # Negative margin (miss)
    margin2 = calculate_clearance_margin(1.0, 1.5)
    assert margin2 == -0.5, f"Expected -0.5, got {margin2}"
    print(f"  event_z=1.0, required_z=1.5: margin={margin2:+.2f}")

    # None handling
    margin3 = calculate_clearance_margin(None, 1.5)
    assert margin3 is None, "Should return None if event_z is None"

    margin4 = calculate_clearance_margin(2.0, None)
    assert margin4 is None, "Should return None if required_z is None"

    print("  [OK] All clearance margin tests passed!\n")
    return True


def test_single_ticker_calculation():
    """Test ECS calculation for a single ticker."""
    print("Testing Single Ticker ECS Calculation...")

    from src.analytics.earnings_intelligence.ecs_calculator import calculate_ecs

    ticker = "AAPL"
    print(f"  Calculating ECS for {ticker}...")

    start = time.time()
    result = calculate_ecs(ticker)
    elapsed = time.time() - start

    print(f"  [OK] Calculation completed in {elapsed:.2f}s")

    # Verify result structure
    assert result.ticker == ticker
    print(f"  [OK] Ticker: {result.ticker}")

    # ECS category
    assert result.ecs_category is not None
    print(f"  [OK] ECS Category: {result.ecs_category.value}")

    # Cleared bar
    print(f"  [OK] Cleared Bar: {'YES' if result.cleared_bar else 'NO'}")

    # Required Z should always be calculated
    assert result.required_z is not None
    print(f"  [OK] Required Z: {result.required_z:+.2f}")

    # Event Z (may be None if no earnings data)
    if result.event_z is not None:
        print(f"  [OK] Event Z: {result.event_z:+.2f}")
        print(f"  [OK] Clearance Margin: {result.clearance_margin:+.2f}")
    else:
        print(f"  [WARN] Event Z: N/A")

    # IES
    if result.ies is not None:
        print(f"  [OK] IES: {result.ies:.0f}/100")

    # EQS
    if result.eqs is not None:
        print(f"  [OK] EQS: {result.eqs:.0f}/100")

    # Score adjustment
    print(f"  [OK] Score Adjustment: {result.score_adjustment:+d}")

    # Data quality
    print(f"  [OK] Data Quality: {result.data_quality.value}")

    print("  [OK] Single ticker calculation tests passed!\n")
    return True


def test_multiple_tickers():
    """Test ECS calculation for multiple tickers."""
    print("Testing Multiple Tickers...")

    from src.analytics.earnings_intelligence.ecs_calculator import calculate_ecs

    test_tickers = ["AAPL", "MSFT", "NVDA"]

    for ticker in test_tickers:
        result = calculate_ecs(ticker)

        event_str = f"{result.event_z:+.2f}" if result.event_z else "N/A"
        margin_str = f"{result.clearance_margin:+.2f}" if result.clearance_margin else "N/A"

        print(
            f"  [OK] {ticker}: ECS={result.ecs_category.value}, Event={event_str}, Margin={margin_str}, Adj={result.score_adjustment:+d}")

    print(f"\n  [OK] Processed {len(test_tickers)} tickers\n")
    return True


def test_batch_calculation():
    """Test batch ECS calculation."""
    print("Testing Batch Calculation...")

    from src.analytics.earnings_intelligence.ecs_calculator import calculate_ecs_batch

    test_tickers = ["AAPL", "GOOGL"]

    def progress(current, total, ticker):
        print(f"  [{current}/{total}] {ticker}")

    start = time.time()
    results = calculate_ecs_batch(test_tickers, progress_callback=progress)
    elapsed = time.time() - start

    assert len(results) == len(test_tickers), "Batch incomplete"
    print(f"  [OK] Batch completed in {elapsed:.2f}s")

    for ticker, result in results.items():
        print(f"  [OK] {ticker}: {result.ecs_category.value}")

    print("  [OK] Batch calculation tests passed!\n")
    return True


def test_ai_summary():
    """Test AI summary generation."""
    print("Testing AI Summary Generation...")

    from src.analytics.earnings_intelligence.ecs_calculator import get_ecs_summary_for_ai

    summary = get_ecs_summary_for_ai("NVDA")

    assert isinstance(summary, str), "Summary should be a string"
    assert "NVDA" in summary, "Summary should contain ticker"
    assert "ECS" in summary, "Summary should mention ECS"
    assert "Cleared" in summary, "Summary should mention cleared"

    print("  [OK] Summary generated successfully")
    print("\n--- AI Summary Preview ---")
    lines = summary.split('\n')[:40]
    for line in lines:
        print(f"  {line}")
    if len(summary.split('\n')) > 40:
        print("  ...")

    print("\n  [OK] AI summary tests passed!\n")
    return True


def test_screener_integration():
    """Test screener integration format."""
    print("Testing Screener Integration...")

    from src.analytics.earnings_intelligence.ecs_calculator import get_ecs_for_screener

    data = get_ecs_for_screener("AAPL")

    required_fields = ['ecs_category', 'cleared_bar', 'event_z', 'required_z',
                       'clearance_margin', 'score_adjustment', 'ies', 'eqs',
                       'regime', 'data_quality']

    for field in required_fields:
        assert field in data, f"Missing field: {field}"

    print(f"  [OK] All required fields present")
    print(f"  Data: {data}")

    print("  [OK] Screener integration tests passed!\n")
    return True


def test_full_analysis():
    """Test full earnings analysis function."""
    print("Testing Full Earnings Analysis...")

    from src.analytics.earnings_intelligence.ecs_calculator import get_full_earnings_analysis

    analysis = get_full_earnings_analysis("NVDA")

    # Check structure
    assert 'ticker' in analysis
    assert 'pre_earnings' in analysis
    assert 'post_earnings' in analysis
    assert 'clearance' in analysis
    assert 'data_quality' in analysis

    print(f"  [OK] Ticker: {analysis['ticker']}")
    print(f"  [OK] Pre-earnings IES: {analysis['pre_earnings'].get('ies')}")
    print(f"  [OK] Post-earnings EQS: {analysis['post_earnings'].get('eqs')}")
    print(f"  [OK] Clearance ECS: {analysis['clearance'].get('ecs_category')}")
    print(f"  [OK] Data Quality: {analysis['data_quality']}")

    print("  [OK] Full analysis tests passed!\n")
    return True


def test_result_serialization():
    """Test result JSON serialization."""
    print("Testing Result Serialization...")

    from src.analytics.earnings_intelligence.ecs_calculator import calculate_ecs
    import json

    result = calculate_ecs("AAPL")
    data = result.to_dict()

    # Should be JSON serializable
    json_str = json.dumps(data)
    assert len(json_str) > 100, "Serialized data too small"

    # Deserialize and verify
    parsed = json.loads(json_str)
    assert parsed['ticker'] == 'AAPL', "Ticker mismatch"

    print(f"  [OK] Serialized to {len(json_str)} bytes")
    print(f"  [OK] Keys: {list(data.keys())[:8]}...")

    print("  [OK] Serialization tests passed!\n")
    return True


def main():
    """Run all Phase 9 tests."""
    print("=" * 60)
    print("PHASE 9 VERIFICATION: ECS Calculator")
    print("(Expectations Clearance Score)")
    print("=" * 60)

    all_passed = True

    # Run tests in order
    all_passed &= test_imports()
    all_passed &= test_required_z_calculation()
    all_passed &= test_ecs_category_classification()
    all_passed &= test_ecs_category_properties()
    all_passed &= test_clearance_margin()
    all_passed &= test_single_ticker_calculation()
    all_passed &= test_multiple_tickers()
    all_passed &= test_batch_calculation()
    all_passed &= test_ai_summary()
    all_passed &= test_screener_integration()
    all_passed &= test_full_analysis()
    all_passed &= test_result_serialization()

    # Final result
    print("=" * 60)
    if all_passed:
        print("[OK] ALL PHASE 9 TESTS PASSED!")
    else:
        print("[FAIL] SOME TESTS FAILED - Review output above")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())