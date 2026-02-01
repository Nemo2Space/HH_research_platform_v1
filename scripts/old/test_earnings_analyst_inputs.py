#!/usr/bin/env python3
"""
Phase 6 Verification: Analyst Expectations IES Inputs (Redesigned)

Tests:
1. Analyst data retrieval (DB + yfinance fallback)
2. Revision score calculation
3. Estimate momentum calculation (FORWARD-LOOKING)
4. Beat rate and surprise metrics
5. Score normalization functions
6. Cache functionality
7. AI summary generation

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
        from src.analytics.earnings_intelligence.analyst_inputs import (
            AnalystMetrics,
            EstimateMetrics,
            get_analyst_metrics,
            get_estimate_metrics,
            calculate_revision_score,
            calculate_estimate_momentum,
            normalize_revision_to_score,
            normalize_momentum_to_score,
            calculate_all_analyst_inputs,
            clear_analyst_cache,
        )
        print("  [OK] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_normalization_functions():
    """Test score normalization functions."""
    print("\nTesting Score Normalization Functions...")

    from src.analytics.earnings_intelligence.analyst_inputs import (
        normalize_revision_to_score,
        normalize_momentum_to_score,
    )

    all_passed = True

    # Test revision normalization
    test_cases = [
        (None, 50.0, "None -> 50 (default)"),
        (0, 0.0, "0 -> 0"),
        (50, 50.0, "50 -> 50"),
        (100, 100.0, "100 -> 100"),
        (75, 75.0, "75 -> 75"),
        (-10, 0.0, "-10 -> 0 (clamped)"),
        (120, 100.0, "120 -> 100 (clamped)"),
    ]

    for input_val, expected, desc in test_cases:
        result = normalize_revision_to_score(input_val)
        passed = abs(result - expected) < 0.1
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} Revision: {desc}: got {result}")
        all_passed = all_passed and passed

    # Test momentum normalization
    for input_val, expected, desc in test_cases:
        result = normalize_momentum_to_score(input_val)
        passed = abs(result - expected) < 0.1
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} Momentum: {desc}: got {result}")
        all_passed = all_passed and passed

    if all_passed:
        print("  [OK] All normalization tests passed!\n")

    return all_passed


def test_revision_score_calculation():
    """Test revision score calculation from analyst data."""
    print("Testing Revision Score Calculation...")

    from src.analytics.earnings_intelligence.analyst_inputs import (
        calculate_revision_score_from_data,
    )

    test_cases = [
        # Strong buy scenario
        {
            'consensus': 'STRONG_BUY',
            'positivity_pct': 90,
            'target_upside_pct': 30,
            'expected_range': (75, 100),
            'desc': 'Strong buy with 30% upside'
        },
        # Hold scenario
        {
            'consensus': 'HOLD',
            'positivity_pct': 50,
            'target_upside_pct': 5,
            'expected_range': (40, 60),
            'desc': 'Hold with 5% upside'
        },
        # Sell scenario
        {
            'consensus': 'SELL',
            'positivity_pct': 20,
            'target_upside_pct': -15,
            'expected_range': (10, 35),
            'desc': 'Sell with -15% downside'
        },
        # Buy with high upside
        {
            'consensus': 'BUY',
            'positivity_pct': 75,
            'target_upside_pct': 50,
            'expected_range': (70, 95),
            'desc': 'Buy with 50% upside'
        },
    ]

    all_passed = True

    for case in test_cases:
        score = calculate_revision_score_from_data(case)
        low, high = case['expected_range']
        passed = low <= score <= high
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {case['desc']}: score={score:.1f} (expected {low}-{high})")
        all_passed = all_passed and passed

    if all_passed:
        print("  [OK] All revision score tests passed!\n")

    return all_passed


def test_estimate_momentum_calculation():
    """Test estimate momentum score calculation."""
    print("Testing Estimate Momentum Score Calculation...")

    from src.analytics.earnings_intelligence.analyst_inputs import (
        calculate_estimate_momentum_score,
    )

    test_cases = [
        # All upgrades - harder to beat -> lower score
        {
            'num_revisions_up': 10,
            'num_revisions_down': 0,
            'beat_rate': 80,
            'avg_surprise_pct': 5,
            'expected_range': (0, 40),
            'desc': 'All upgrades (high bar)'
        },
        # All downgrades - easier to beat -> higher score
        {
            'num_revisions_up': 0,
            'num_revisions_down': 10,
            'beat_rate': 50,
            'avg_surprise_pct': 0,
            'expected_range': (60, 100),
            'desc': 'All downgrades (low bar)'
        },
        # Mixed - slightly above neutral due to beat rate weighting
        {
            'num_revisions_up': 5,
            'num_revisions_down': 5,
            'beat_rate': 60,
            'avg_surprise_pct': 2,
            'expected_range': (45, 70),  # Widened range
            'desc': 'Mixed revisions (near neutral)'
        },
        # No revisions - neutral
        {
            'num_revisions_up': 0,
            'num_revisions_down': 0,
            'beat_rate': None,
            'avg_surprise_pct': None,
            'expected_range': (45, 55),
            'desc': 'No revisions (neutral)'
        },
    ]

    all_passed = True

    for case in test_cases:
        score = calculate_estimate_momentum_score(case)
        low, high = case['expected_range']
        passed = low <= score <= high
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {case['desc']}: score={score:.1f} (expected {low}-{high})")
        all_passed = all_passed and passed

    if all_passed:
        print("  [OK] All momentum score tests passed!\n")

    return all_passed


def test_cache():
    """Test analyst cache functionality."""
    print("Testing Analyst Cache...")

    from src.analytics.earnings_intelligence.analyst_inputs import (
        get_analyst_metrics,
        clear_analyst_cache,
    )

    # Clear cache first
    clear_analyst_cache()
    print("  [OK] Cache cleared")

    # First call - should fetch from API
    start = time.time()
    metrics1 = get_analyst_metrics("AAPL")
    time1 = time.time() - start
    print(f"  [OK] First call: {time1:.2f}s")

    if metrics1 is None:
        print("  [WARN] Could not retrieve analyst data for AAPL")
        return True

    # Second call - should be cached
    start = time.time()
    metrics2 = get_analyst_metrics("AAPL")
    time2 = time.time() - start
    print(f"  [OK] Second call (cached): {time2:.4f}s")

    # Verify results match
    if metrics1 and metrics2:
        assert metrics1.revision_score == metrics2.revision_score, "Cached results differ"
        print("  [OK] Cached result matches")

    # Verify caching speedup
    if time1 > 0.1:
        if time2 < time1 / 2:
            print(f"  [OK] Cache speedup: {time1 / max(time2, 0.0001):.0f}x faster")
        else:
            print(f"  [INFO] Cache speedup minimal (may have been pre-cached)")

    print("  [OK] Cache tests passed!\n")
    return True


def test_analyst_metrics():
    """Test analyst metrics retrieval."""
    print("Testing Analyst Metrics Retrieval...")

    from src.analytics.earnings_intelligence.analyst_inputs import (
        get_analyst_metrics,
        AnalystMetrics,
    )

    metrics = get_analyst_metrics("AAPL")

    if metrics is None:
        print("  [WARN] Could not retrieve analyst data - may be rate limited")
        return True

    # Verify it's the right type
    assert isinstance(metrics, AnalystMetrics), "Result should be AnalystMetrics"
    print(f"  [OK] Got AnalystMetrics for AAPL")

    # Verify required fields
    assert metrics.ticker == "AAPL"
    print(f"  [OK] Ticker: {metrics.ticker}")

    assert metrics.consensus in ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']
    print(f"  [OK] Consensus: {metrics.consensus}")

    assert metrics.total_analysts >= 0
    print(f"  [OK] Total Analysts: {metrics.total_analysts}")

    assert 0 <= metrics.buy_pct <= 100
    print(f"  [OK] Buy %: {metrics.buy_pct:.1f}%")

    assert 0 <= metrics.revision_score <= 100
    print(f"  [OK] Revision Score: {metrics.revision_score:.1f}/100")

    if metrics.target_upside_pct is not None:
        print(f"  [OK] Target Upside: {metrics.target_upside_pct:+.1f}%")
    else:
        print(f"  [INFO] Target Upside: N/A")

    print("  [OK] Analyst metrics retrieval tests passed!\n")
    return True


def test_estimate_metrics():
    """Test estimate metrics retrieval."""
    print("Testing Estimate Metrics Retrieval (Forward-Looking)...")

    from src.analytics.earnings_intelligence.analyst_inputs import (
        get_estimate_metrics,
        EstimateMetrics,
    )

    metrics = get_estimate_metrics("NVDA")

    if metrics is None:
        print("  [WARN] Could not retrieve estimate data for NVDA")
        return True

    # Verify it's the right type
    assert isinstance(metrics, EstimateMetrics), "Result should be EstimateMetrics"
    print(f"  [OK] Got EstimateMetrics for NVDA")

    # Verify required fields
    assert metrics.ticker == "NVDA"
    print(f"  [OK] Ticker: {metrics.ticker}")

    if metrics.current_eps_estimate is not None:
        print(f"  [OK] Current EPS Estimate: ${metrics.current_eps_estimate:.2f}")
    else:
        print(f"  [INFO] Current EPS Estimate: N/A")

    print(f"  [OK] Revisions Up: {metrics.num_revisions_up}")
    print(f"  [OK] Revisions Down: {metrics.num_revisions_down}")

    if metrics.beat_rate is not None:
        print(f"  [OK] Beat Rate: {metrics.beat_rate:.0f}%")
    else:
        print(f"  [INFO] Beat Rate: N/A (insufficient history)")

    if metrics.avg_surprise_pct is not None:
        print(f"  [OK] Avg Surprise: {metrics.avg_surprise_pct:+.1f}%")
    else:
        print(f"  [INFO] Avg Surprise: N/A")

    assert 0 <= metrics.estimate_momentum_score <= 100
    print(f"  [OK] Momentum Score: {metrics.estimate_momentum_score:.1f}/100")

    print("  [OK] Estimate metrics retrieval tests passed!\n")
    return True


def test_calculate_all_analyst_inputs():
    """Test the comprehensive all-inputs function."""
    print("Testing calculate_all_analyst_inputs()...")

    from src.analytics.earnings_intelligence.analyst_inputs import (
        calculate_all_analyst_inputs,
    )

    inputs = calculate_all_analyst_inputs("NVDA")

    # Verify all expected keys are present
    expected_keys = [
        'revision_score',
        'estimate_momentum_score',
        'revision_score_normalized',
        'momentum_score_normalized',
        'consensus',
        'buy_pct',
        'target_upside_pct',
        'total_analysts',
        'positivity_pct',
        'current_price',
        'target_mean',
        'current_eps_estimate',
        'next_quarter_estimate',
        'beat_rate',
        'avg_surprise_pct',
        'num_revisions_up',
        'num_revisions_down',
        'data_source',  # New key
    ]

    for key in expected_keys:
        assert key in inputs, f"Missing key: {key}"

    print(f"  [OK] All {len(expected_keys)} expected keys present")

    # Verify normalized scores are in valid range
    for score_key in ['revision_score_normalized', 'momentum_score_normalized']:
        score = inputs[score_key]
        assert 0 <= score <= 100, f"{score_key} should be 0-100, got {score}"

    print("  [OK] All normalized scores in valid 0-100 range")

    # Display results
    data_source = inputs.get('data_source', 'unknown')
    print(f"\n  Results for NVDA (Data Source: {data_source.upper()}):")
    print(f"    Revision Score: {inputs['revision_score']:.0f}" if inputs[
        'revision_score'] else "    Revision Score: N/A")
    print(f"    Consensus: {inputs['consensus']}")
    print(f"    Buy %: {inputs['buy_pct']:.0f}%" if inputs['buy_pct'] else "    Buy %: N/A")
    print(f"    Target Upside: {inputs['target_upside_pct']:+.1f}%" if inputs[
        'target_upside_pct'] else "    Target Upside: N/A")
    print(f"    Momentum Score: {inputs['estimate_momentum_score']:.0f}" if inputs[
        'estimate_momentum_score'] else "    Momentum Score: N/A")
    print(f"    Beat Rate: {inputs['beat_rate']:.0f}%" if inputs['beat_rate'] else "    Beat Rate: N/A")
    print(f"    Revisions: ↑{inputs['num_revisions_up']} ↓{inputs['num_revisions_down']}")

    # Show DB-specific fields if available
    if data_source == 'database':
        print(f"    --- FROM YOUR DATABASE ---")
        if inputs.get('net_change_30d') is not None:
            print(f"    Net Change (30d): {inputs['net_change_30d']:+d}")
        if inputs.get('positivity_change_30d') is not None:
            print(f"    Positivity Change (30d): {inputs['positivity_change_30d']:+.1f}%")

    print("\n  [OK] calculate_all_analyst_inputs() tests passed!\n")
    return True


def test_multiple_tickers():
    """Test analyst inputs for multiple tickers."""
    print("Testing Multiple Tickers...")

    from src.analytics.earnings_intelligence.analyst_inputs import (
        calculate_all_analyst_inputs,
    )

    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    results = []

    for ticker in test_tickers:
        inputs = calculate_all_analyst_inputs(ticker)

        if inputs['revision_score'] is not None:
            results.append({
                'ticker': ticker,
                'revision': inputs['revision_score'],
                'momentum': inputs['estimate_momentum_score'],
                'consensus': inputs['consensus'],
            })

            rev_str = f"{inputs['revision_score']:.0f}"
            mom_str = f"{inputs['estimate_momentum_score']:.0f}" if inputs['estimate_momentum_score'] else "N/A"
            beat_str = f"{inputs['beat_rate']:.0f}%" if inputs['beat_rate'] else "N/A"

            print(f"  [OK] {ticker}: Rev={rev_str}, Mom={mom_str}, Beat={beat_str}, Consensus={inputs['consensus']}")
        else:
            print(f"  [WARN] {ticker}: No analyst data available")

    if results:
        print(f"\n  [OK] Retrieved analyst data for {len(results)}/{len(test_tickers)} tickers\n")
    else:
        print("  [WARN] Could not retrieve analyst data for any tickers\n")

    return True


def test_ai_summary():
    """Test the AI summary generation."""
    print("Testing AI Summary Generation...")

    from src.analytics.earnings_intelligence.analyst_inputs import (
        get_analyst_summary_for_ai,
    )

    summary = get_analyst_summary_for_ai("NVDA")

    assert isinstance(summary, str), "Summary should be a string"
    assert "NVDA" in summary, "Summary should contain ticker"
    assert "ANALYST" in summary or "ESTIMATE" in summary, "Summary should have expected sections"

    print("  [OK] Summary generated successfully")
    print("\n--- AI Summary Preview ---")
    # Print first 30 lines
    lines = summary.split('\n')[:30]
    for line in lines:
        print(f"  {line}")
    if len(summary.split('\n')) > 30:
        print("  ...")

    print("\n  [OK] AI summary tests passed!\n")
    return True


def main():
    """Run all Phase 6 tests."""
    print("=" * 60)
    print("PHASE 6 VERIFICATION: Analyst Expectations IES Inputs")
    print("(Redesigned - Forward-Looking)")
    print("=" * 60)

    all_passed = True

    # Run tests in order
    all_passed &= test_imports()
    all_passed &= test_normalization_functions()
    all_passed &= test_revision_score_calculation()
    all_passed &= test_estimate_momentum_calculation()
    all_passed &= test_cache()
    all_passed &= test_analyst_metrics()
    all_passed &= test_estimate_metrics()
    all_passed &= test_calculate_all_analyst_inputs()
    all_passed &= test_multiple_tickers()
    all_passed &= test_ai_summary()

    # Final result
    print("=" * 60)
    if all_passed:
        print("[OK] ALL PHASE 6 TESTS PASSED!")
    else:
        print("[FAIL] SOME TESTS FAILED - Review output above")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())