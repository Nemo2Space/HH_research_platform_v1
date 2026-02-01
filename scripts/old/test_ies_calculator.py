#!/usr/bin/env python3
"""
Phase 7 Verification: IES Calculator

Tests:
1. IES calculation for single ticker
2. Component score integration (Phases 4-6)
3. Weighted average calculation
4. Regime classification
5. Position scaling
6. Data quality assessment
7. Batch calculation
8. AI summary generation

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
        from src.analytics.earnings_intelligence.ies_calculator import (
            calculate_ies,
            calculate_ies_batch,
            IESCalculationResult,
            IES_WEIGHTS_V2,
            get_ies_summary_for_ai,
            get_ies_for_screener,
            _calculate_weighted_ies,
            _assess_ies_data_quality,
        )

        from src.analytics.earnings_intelligence.models import (
            ExpectationsRegime,
            DataQuality,
        )

        print("  [OK] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_weights():
    """Test that IES weights sum to 1.0."""
    print("\nTesting IES Weights...")

    from src.analytics.earnings_intelligence.ies_calculator import IES_WEIGHTS_V2

    total = sum(IES_WEIGHTS_V2.values())

    print(f"  Weights: {IES_WEIGHTS_V2}")
    print(f"  Total: {total:.2f}")

    if abs(total - 1.0) < 0.001:
        print("  [OK] Weights sum to 1.0")
        return True
    else:
        print(f"  [FAIL] Weights sum to {total}, expected 1.0")
        return False


def test_weighted_calculation():
    """Test the weighted IES calculation with missing inputs."""
    print("\nTesting Weighted IES Calculation...")

    from src.analytics.earnings_intelligence.ies_calculator import (
        _calculate_weighted_ies,
        IES_WEIGHTS_V2,
    )

    # Test 1: All inputs available
    all_scores = {
        'drift_20d': 60,
        'rel_drift_20d': 55,
        'implied_move_pctl': 70,
        'iv_pctl': 65,
        'skew_shift': 50,
        'revision_score': 80,
        'estimate_momentum': 40,
    }

    ies, count = _calculate_weighted_ies(all_scores, IES_WEIGHTS_V2)
    print(f"  All inputs (7/7): IES={ies:.1f}, count={count}")
    assert count == 7, f"Expected 7 inputs, got {count}"
    assert 50 <= ies <= 70, f"IES should be moderate, got {ies}"
    print("  [OK] All inputs test passed")

    # Test 2: Missing some inputs
    partial_scores = {
        'drift_20d': 60,
        'rel_drift_20d': None,  # Missing
        'implied_move_pctl': 70,
        'iv_pctl': 65,
        'skew_shift': None,  # Missing
        'revision_score': 80,
        'estimate_momentum': None,  # Missing
    }

    ies2, count2 = _calculate_weighted_ies(partial_scores, IES_WEIGHTS_V2)
    print(f"  Partial inputs (4/7): IES={ies2:.1f}, count={count2}")
    assert count2 == 4, f"Expected 4 inputs, got {count2}"
    print("  [OK] Partial inputs test passed")

    # Test 3: No inputs
    no_scores = {k: None for k in IES_WEIGHTS_V2.keys()}
    ies3, count3 = _calculate_weighted_ies(no_scores, IES_WEIGHTS_V2)
    print(f"  No inputs (0/7): IES={'N/A' if ies3 is None else ies3}, count={count3}")
    assert count3 == 0, f"Expected 0 inputs, got {count3}"
    assert ies3 is None, "IES should be None with no inputs"
    print("  [OK] No inputs test passed")

    # Test 4: Extreme high scores
    high_scores = {k: 95 for k in IES_WEIGHTS_V2.keys()}
    ies4, count4 = _calculate_weighted_ies(high_scores, IES_WEIGHTS_V2)
    print(f"  All high (95): IES={ies4:.1f}")
    assert 90 <= ies4 <= 100, f"IES should be ~95, got {ies4}"
    print("  [OK] High scores test passed")

    # Test 5: Extreme low scores
    low_scores = {k: 10 for k in IES_WEIGHTS_V2.keys()}
    ies5, count5 = _calculate_weighted_ies(low_scores, IES_WEIGHTS_V2)
    print(f"  All low (10): IES={ies5:.1f}")
    assert 0 <= ies5 <= 20, f"IES should be ~10, got {ies5}"
    print("  [OK] Low scores test passed")

    print("  [OK] All weighted calculation tests passed!\n")
    return True


def test_data_quality_assessment():
    """Test data quality assessment logic."""
    print("Testing Data Quality Assessment...")

    from src.analytics.earnings_intelligence.ies_calculator import _assess_ies_data_quality
    from src.analytics.earnings_intelligence.models import DataQuality

    # Test 1: All inputs available
    quality1 = _assess_ies_data_quality([], 7)
    assert quality1 == DataQuality.HIGH, f"Expected HIGH, got {quality1}"
    print(f"  [OK] 7/7 inputs -> {quality1.value}")

    # Test 2: Most inputs available
    quality2 = _assess_ies_data_quality(['skew_shift'], 6)
    assert quality2 == DataQuality.HIGH, f"Expected HIGH, got {quality2}"
    print(f"  [OK] 6/7 inputs -> {quality2.value}")

    # Test 3: Missing options data (critical)
    quality3 = _assess_ies_data_quality(['implied_move_pctl', 'iv_pctl'], 5)
    assert quality3 == DataQuality.MEDIUM, f"Expected MEDIUM, got {quality3}"
    print(f"  [OK] Missing options -> {quality3.value}")

    # Test 4: Few inputs
    quality4 = _assess_ies_data_quality(['drift_20d', 'rel_drift_20d', 'implied_move_pctl', 'iv_pctl'], 3)
    assert quality4 == DataQuality.MEDIUM, f"Expected MEDIUM, got {quality4}"
    print(f"  [OK] 3/7 inputs -> {quality4.value}")

    # Test 5: Very few inputs
    quality5 = _assess_ies_data_quality(['drift_20d', 'rel_drift_20d', 'implied_move_pctl', 'iv_pctl', 'skew_shift'], 2)
    assert quality5 == DataQuality.LOW, f"Expected LOW, got {quality5}"
    print(f"  [OK] 2/7 inputs -> {quality5.value}")

    print("  [OK] All data quality tests passed!\n")
    return True


def test_regime_classification():
    """Test regime classification logic."""
    print("Testing Regime Classification...")

    from src.analytics.earnings_intelligence.models import ExpectationsRegime

    # Test HYPED: High IES, high implied move, positive drift
    regime1 = ExpectationsRegime.classify(ies=75, implied_move_pctl=80, drift_20d=0.15)
    assert regime1 == ExpectationsRegime.HYPED, f"Expected HYPED, got {regime1}"
    print(f"  [OK] IES=75, IM=80, Drift=15% -> {regime1.value}")

    # Test FEARED: Low IES, negative drift
    regime2 = ExpectationsRegime.classify(ies=40, implied_move_pctl=60, drift_20d=-0.10)
    assert regime2 == ExpectationsRegime.FEARED, f"Expected FEARED, got {regime2}"
    print(f"  [OK] IES=40, IM=60, Drift=-10% -> {regime2.value}")

    # Test VOLATILE: High implied move, neutral IES
    regime3 = ExpectationsRegime.classify(ies=50, implied_move_pctl=85, drift_20d=0.02)
    assert regime3 == ExpectationsRegime.VOLATILE, f"Expected VOLATILE, got {regime3}"
    print(f"  [OK] IES=50, IM=85, Drift=2% -> {regime3.value}")

    # Test NORMAL: Average everything
    regime4 = ExpectationsRegime.classify(ies=55, implied_move_pctl=55, drift_20d=0.05)
    assert regime4 == ExpectationsRegime.NORMAL, f"Expected NORMAL, got {regime4}"
    print(f"  [OK] IES=55, IM=55, Drift=5% -> {regime4.value}")

    print("  [OK] All regime classification tests passed!\n")
    return True


def test_position_scaling():
    """Test position scaling calculation."""
    print("Testing Position Scaling...")

    from src.analytics.earnings_intelligence.models import PositionScaling

    # Test 1: Normal conditions
    scale1 = PositionScaling.calculate(ies=50, implied_move_pctl=50)
    print(f"  Normal (IES=50, IM=50): scale={scale1.final_scale:.0%}")
    assert 0.8 <= scale1.final_scale <= 1.0, f"Expected ~100%, got {scale1.final_scale}"

    # Test 2: High expectations
    scale2 = PositionScaling.calculate(ies=85, implied_move_pctl=80)
    print(f"  High (IES=85, IM=80): scale={scale2.final_scale:.0%}")
    assert scale2.final_scale <= 0.75, f"Expected reduced scale, got {scale2.final_scale}"

    # Test 3: Extreme expectations
    scale3 = PositionScaling.calculate(ies=95, implied_move_pctl=95)
    print(f"  Extreme (IES=95, IM=95): scale={scale3.final_scale:.0%}")
    assert scale3.final_scale <= 0.55, f"Expected very reduced scale, got {scale3.final_scale}"

    # Test 4: Low expectations
    scale4 = PositionScaling.calculate(ies=30, implied_move_pctl=40)
    print(f"  Low (IES=30, IM=40): scale={scale4.final_scale:.0%}")
    assert scale4.final_scale >= 0.9, f"Expected full scale, got {scale4.final_scale}"

    print("  [OK] All position scaling tests passed!\n")
    return True


def test_single_ticker_calculation():
    """Test IES calculation for a single ticker."""
    print("Testing Single Ticker IES Calculation...")

    from src.analytics.earnings_intelligence.ies_calculator import calculate_ies

    ticker = "AAPL"
    print(f"  Calculating IES for {ticker}...")

    start = time.time()
    result = calculate_ies(ticker)
    elapsed = time.time() - start

    print(f"  [OK] Calculation completed in {elapsed:.2f}s")

    # Verify result structure
    assert result.ticker == ticker, f"Ticker mismatch: {result.ticker}"
    print(f"  [OK] Ticker: {result.ticker}")

    # Check IES is valid
    if result.ies is not None:
        assert 0 <= result.ies <= 100, f"IES out of range: {result.ies}"
        print(f"  [OK] IES: {result.ies:.0f}/100")
    else:
        print(f"  [WARN] IES: N/A (insufficient data)")

    # Check regime
    assert result.regime is not None
    print(f"  [OK] Regime: {result.regime.value}")

    # Check position scale
    assert 0.2 <= result.position_scale <= 1.0, f"Position scale out of range: {result.position_scale}"
    print(f"  [OK] Position Scale: {result.position_scale:.0%}")

    # Check data quality
    assert result.data_quality is not None
    print(f"  [OK] Data Quality: {result.data_quality.value} ({result.input_count}/7 inputs)")

    # Check component scores
    available_components = [
        ('drift_score', result.drift_score),
        ('rel_drift_score', result.rel_drift_score),
        ('implied_move_score', result.implied_move_score),
        ('iv_score', result.iv_score),
        ('skew_score', result.skew_score),
        ('revision_score', result.revision_score),
        ('momentum_score', result.momentum_score),
    ]

    for name, score in available_components:
        if score is not None:
            assert 0 <= score <= 100, f"{name} out of range: {score}"
            print(f"  [OK] {name}: {score:.0f}")
        else:
            print(f"  [--] {name}: N/A")

    print("  [OK] Single ticker calculation tests passed!\n")
    return True


def test_multiple_tickers():
    """Test IES calculation for multiple tickers."""
    print("Testing Multiple Tickers...")

    from src.analytics.earnings_intelligence.ies_calculator import calculate_ies

    test_tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
    results = {}

    for ticker in test_tickers:
        result = calculate_ies(ticker)
        results[ticker] = result

        ies_str = f"{result.ies:.0f}" if result.ies else "N/A"
        print(f"  [OK] {ticker}: IES={ies_str}, Regime={result.regime.value}, Quality={result.data_quality.value}")

    # Verify all returned valid results
    assert len(results) == len(test_tickers), "Not all tickers processed"

    print(f"\n  [OK] Processed {len(results)}/{len(test_tickers)} tickers\n")
    return True


def test_batch_calculation():
    """Test batch IES calculation."""
    print("Testing Batch Calculation...")

    from src.analytics.earnings_intelligence.ies_calculator import calculate_ies_batch

    test_tickers = ["AAPL", "GOOGL"]

    def progress(current, total, ticker):
        print(f"  [{current}/{total}] {ticker}")

    start = time.time()
    results = calculate_ies_batch(test_tickers, progress_callback=progress)
    elapsed = time.time() - start

    assert len(results) == len(test_tickers), "Batch incomplete"
    print(f"  [OK] Batch completed in {elapsed:.2f}s")

    for ticker, result in results.items():
        ies_str = f"{result.ies:.0f}" if result.ies else "N/A"
        print(f"  [OK] {ticker}: IES={ies_str}")

    print("  [OK] Batch calculation tests passed!\n")
    return True


def test_ai_summary():
    """Test AI summary generation."""
    print("Testing AI Summary Generation...")

    from src.analytics.earnings_intelligence.ies_calculator import (
        calculate_ies,
        get_ies_summary_for_ai,
    )

    ticker = "NVDA"
    summary = get_ies_summary_for_ai(ticker)

    assert isinstance(summary, str), "Summary should be a string"
    assert ticker in summary, "Summary should contain ticker"
    assert "IES" in summary, "Summary should mention IES"
    assert "Regime" in summary, "Summary should mention regime"

    print("  [OK] Summary generated successfully")
    print("\n--- AI Summary Preview ---")
    # Print first 35 lines
    lines = summary.split('\n')[:35]
    for line in lines:
        print(f"  {line}")
    if len(summary.split('\n')) > 35:
        print("  ...")

    print("\n  [OK] AI summary tests passed!\n")
    return True


def test_screener_integration():
    """Test screener integration format."""
    print("Testing Screener Integration...")

    from src.analytics.earnings_intelligence.ies_calculator import get_ies_for_screener

    ticker = "AAPL"
    data = get_ies_for_screener(ticker)

    # Check required fields
    required_fields = ['ies', 'ies_level', 'regime', 'position_scale',
                       'days_to_earnings', 'in_action_window', 'data_quality']

    for field in required_fields:
        assert field in data, f"Missing field: {field}"

    print(f"  [OK] All required fields present")
    print(f"  Data: {data}")

    print("  [OK] Screener integration tests passed!\n")
    return True


def test_to_dict():
    """Test result serialization."""
    print("Testing Result Serialization...")

    from src.analytics.earnings_intelligence.ies_calculator import calculate_ies
    import json

    result = calculate_ies("AAPL")
    data = result.to_dict()

    # Should be JSON serializable
    json_str = json.dumps(data)
    assert len(json_str) > 100, "Serialized data too small"

    # Deserialize and verify
    parsed = json.loads(json_str)
    assert parsed['ticker'] == 'AAPL', "Ticker mismatch after serialization"

    print(f"  [OK] Serialized to {len(json_str)} bytes")
    print(f"  [OK] Keys: {list(data.keys())[:10]}...")

    print("  [OK] Serialization tests passed!\n")
    return True


def main():
    """Run all Phase 7 tests."""
    print("=" * 60)
    print("PHASE 7 VERIFICATION: IES Calculator")
    print("=" * 60)

    all_passed = True

    # Run tests in order
    all_passed &= test_imports()
    all_passed &= test_weights()
    all_passed &= test_weighted_calculation()
    all_passed &= test_data_quality_assessment()
    all_passed &= test_regime_classification()
    all_passed &= test_position_scaling()
    all_passed &= test_single_ticker_calculation()
    all_passed &= test_multiple_tickers()
    all_passed &= test_batch_calculation()
    all_passed &= test_ai_summary()
    all_passed &= test_screener_integration()
    all_passed &= test_to_dict()

    # Final result
    print("=" * 60)
    if all_passed:
        print("[OK] ALL PHASE 7 TESTS PASSED!")
    else:
        print("[FAIL] SOME TESTS FAILED - Review output above")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())