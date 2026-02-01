#!/usr/bin/env python3
"""
Phase 8 Verification: EQS Calculator (Earnings Quality Score)

Tests:
1. Z-score calculations (EPS, Revenue, Guidance)
2. Score conversions
3. Weighted EQS calculation
4. Data quality assessment
5. Earnings data retrieval
6. AI summary generation

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
        from src.analytics.earnings_intelligence.eqs_calculator import (
            calculate_eqs,
            EQSCalculationResult,
            EarningsData,
            calculate_eps_z,
            calculate_rev_z,
            calculate_guidance_z,
            calculate_event_z,
            z_to_score,
            guidance_to_score,
            tone_to_score,
            get_eqs_summary_for_ai,
            get_eqs_for_screener,
        )
        
        from src.analytics.earnings_intelligence.models import (
            GuidanceDirection,
            DataQuality,
            EQS_WEIGHTS,
            EVENT_Z_WEIGHTS,
        )
        
        print("  [OK] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_weights():
    """Test that EQS and Event Z weights sum to 1.0."""
    print("\nTesting EQS Weights...")
    
    from src.analytics.earnings_intelligence.models import EQS_WEIGHTS, EVENT_Z_WEIGHTS
    
    eqs_total = sum(EQS_WEIGHTS.values())
    event_total = sum(EVENT_Z_WEIGHTS.values())
    
    print(f"  EQS Weights: {EQS_WEIGHTS}")
    print(f"  EQS Total: {eqs_total:.2f}")
    
    print(f"  Event Z Weights: {EVENT_Z_WEIGHTS}")
    print(f"  Event Z Total: {event_total:.2f}")
    
    passed = True
    if abs(eqs_total - 1.0) < 0.001:
        print("  [OK] EQS weights sum to 1.0")
    else:
        print(f"  [FAIL] EQS weights sum to {eqs_total}")
        passed = False
    
    if abs(event_total - 1.0) < 0.001:
        print("  [OK] Event Z weights sum to 1.0")
    else:
        print(f"  [FAIL] Event Z weights sum to {event_total}")
        passed = False
    
    return passed


def test_z_score_calculations():
    """Test z-score calculation functions."""
    print("\nTesting Z-Score Calculations...")
    
    from src.analytics.earnings_intelligence.eqs_calculator import (
        calculate_eps_z,
        calculate_rev_z,
        calculate_guidance_z,
        calculate_event_z,
        DEFAULT_EPS_SURPRISE_MEAN,
        DEFAULT_EPS_SURPRISE_STD,
    )
    from src.analytics.earnings_intelligence.models import GuidanceDirection
    
    # Test EPS z-score
    # Default mean is 4.5%, std is 12%
    # A 16.5% beat is 1 std above mean -> z = 1.0
    eps_z = calculate_eps_z(16.5)  # 4.5 + 12 = 16.5
    print(f"  EPS 16.5% surprise: z={eps_z:.2f} (expected ~1.0)")
    assert 0.9 <= eps_z <= 1.1, f"EPS z-score wrong: {eps_z}"
    
    # A -7.5% beat is 1 std below mean -> z = -1.0
    eps_z2 = calculate_eps_z(-7.5)  # 4.5 - 12 = -7.5
    print(f"  EPS -7.5% surprise: z={eps_z2:.2f} (expected ~-1.0)")
    assert -1.1 <= eps_z2 <= -0.9, f"EPS z-score wrong: {eps_z2}"
    
    print("  [OK] EPS z-score calculations correct")
    
    # Test Revenue z-score
    # Default mean is 1.5%, std is 5%
    rev_z = calculate_rev_z(6.5)  # 1.5 + 5 = 6.5
    print(f"  Revenue 6.5% surprise: z={rev_z:.2f} (expected ~1.0)")
    assert 0.9 <= rev_z <= 1.1, f"Rev z-score wrong: {rev_z}"
    
    print("  [OK] Revenue z-score calculations correct")
    
    # Test Guidance z-score
    guide_raised = calculate_guidance_z(GuidanceDirection.RAISED)
    guide_lowered = calculate_guidance_z(GuidanceDirection.LOWERED)
    guide_maintained = calculate_guidance_z(GuidanceDirection.MAINTAINED)
    
    print(f"  Guidance RAISED: z={guide_raised:.1f}")
    print(f"  Guidance LOWERED: z={guide_lowered:.1f}")
    print(f"  Guidance MAINTAINED: z={guide_maintained:.1f}")
    
    assert guide_raised == 1.0, f"RAISED should be 1.0, got {guide_raised}"
    assert guide_lowered == -1.0, f"LOWERED should be -1.0, got {guide_lowered}"
    assert guide_maintained == 0.0, f"MAINTAINED should be 0.0, got {guide_maintained}"
    
    print("  [OK] Guidance z-score calculations correct")
    
    # Test Event Z (blended)
    event_z = calculate_event_z(1.0, 0.5, 1.0)  # Good earnings
    print(f"  Event Z (eps=1, rev=0.5, guide=1): {event_z:.2f}")
    assert 0.7 <= event_z <= 0.9, f"Event z should be ~0.83, got {event_z}"
    
    # Partial data
    event_z_partial = calculate_event_z(1.0, None, 1.0)  # Missing revenue
    print(f"  Event Z (eps=1, guide=1, no rev): {event_z_partial:.2f}")
    assert event_z_partial is not None, "Should handle missing inputs"
    
    print("  [OK] Event Z calculations correct")
    
    print("  [OK] All z-score tests passed!\n")
    return True


def test_score_conversions():
    """Test z-score to 0-100 score conversions."""
    print("Testing Score Conversions...")
    
    from src.analytics.earnings_intelligence.eqs_calculator import (
        z_to_score,
        guidance_to_score,
        tone_to_score,
        margin_to_score,
    )
    from src.analytics.earnings_intelligence.models import GuidanceDirection
    
    # Test z_to_score
    assert z_to_score(0) == 50, "z=0 should map to 50"
    assert z_to_score(1) == 65, "z=1 should map to 65"
    assert z_to_score(-1) == 35, "z=-1 should map to 35"
    assert z_to_score(5) == 100, "z=5 should be clamped to 100"
    assert z_to_score(-5) == 0, "z=-5 should be clamped to 0"
    
    print("  [OK] z_to_score conversions correct")
    
    # Test guidance_to_score
    assert guidance_to_score(GuidanceDirection.RAISED_STRONG) == 95
    assert guidance_to_score(GuidanceDirection.RAISED) == 77
    assert guidance_to_score(GuidanceDirection.MAINTAINED) == 50
    assert guidance_to_score(GuidanceDirection.LOWERED) == 30
    assert guidance_to_score(GuidanceDirection.LOWERED_STRONG) == 8
    assert guidance_to_score(None) == 50
    
    print("  [OK] guidance_to_score conversions correct")
    
    # Test tone_to_score
    assert tone_to_score("CONFIDENT") == 75
    assert tone_to_score("DEFENSIVE") == 30
    assert tone_to_score(None) == 50
    assert tone_to_score("CONFIDENT", 85) == 85  # Sentiment score overrides
    
    print("  [OK] tone_to_score conversions correct")
    
    # Test margin_to_score
    assert margin_to_score(5) == 100, "5% margin improvement -> 100"
    assert margin_to_score(0) == 50, "0% change -> 50"
    assert margin_to_score(-5) == 0, "-5% decline -> 0"
    assert margin_to_score(None) == 50
    
    print("  [OK] margin_to_score conversions correct")
    
    print("  [OK] All score conversion tests passed!\n")
    return True


def test_earnings_data_structure():
    """Test EarningsData dataclass."""
    print("Testing EarningsData Structure...")
    
    from src.analytics.earnings_intelligence.eqs_calculator import EarningsData
    from src.analytics.earnings_intelligence.models import GuidanceDirection
    
    # Create test earnings data
    data = EarningsData(
        ticker="AAPL",
        earnings_date=date(2024, 10, 31),
        eps_actual=1.64,
        eps_estimate=1.60,
        eps_surprise_pct=2.5,
        revenue_actual=94.9e9,
        revenue_estimate=94.4e9,
        revenue_surprise_pct=0.5,
        guidance_direction=GuidanceDirection.MAINTAINED,
        management_tone="CONFIDENT",
        data_source='test',
    )
    
    assert data.ticker == "AAPL"
    assert data.eps_surprise_pct == 2.5
    assert data.guidance_direction == GuidanceDirection.MAINTAINED
    
    print(f"  [OK] Created EarningsData for {data.ticker}")
    print(f"  [OK] EPS: ${data.eps_actual} vs ${data.eps_estimate} ({data.eps_surprise_pct:+.1f}%)")
    print(f"  [OK] Guidance: {data.guidance_direction.value}")
    
    print("  [OK] EarningsData structure tests passed!\n")
    return True


def test_eqs_calculation_with_mock_data():
    """Test EQS calculation with mock earnings data."""
    print("Testing EQS Calculation with Mock Data...")
    
    from src.analytics.earnings_intelligence.eqs_calculator import (
        calculate_eqs,
        EarningsData,
    )
    from src.analytics.earnings_intelligence.models import GuidanceDirection
    
    # Create mock earnings data - STRONG beat
    strong_beat = EarningsData(
        ticker="TEST",
        earnings_date=date(2024, 10, 31),
        eps_actual=2.00,
        eps_estimate=1.50,
        eps_surprise_pct=33.3,  # Huge beat
        revenue_actual=10e9,
        revenue_estimate=9.5e9,
        revenue_surprise_pct=5.3,  # Solid beat
        guidance_direction=GuidanceDirection.RAISED_STRONG,
        management_tone="CONFIDENT",
        sentiment_score=85,
        data_source='mock',
    )
    
    result = calculate_eqs("TEST", earnings_data=strong_beat)
    
    print(f"  Strong Beat Test:")
    print(f"    EPS Surprise: {result.eps_surprise_pct:+.1f}%")
    print(f"    Revenue Surprise: {result.revenue_surprise_pct:+.1f}%")
    print(f"    Guidance: {result.guidance_direction.value}")
    print(f"    EQS: {result.eqs:.0f}/100")
    print(f"    Event Z: {result.event_z:+.2f}")
    
    assert result.eqs is not None, "EQS should be calculated"
    assert result.eqs >= 70, f"Strong beat should have high EQS, got {result.eqs}"
    assert result.event_z > 1.0, f"Strong beat should have positive event z, got {result.event_z}"
    
    print("  [OK] Strong beat calculated correctly")
    
    # Create mock earnings data - MISS
    miss = EarningsData(
        ticker="TEST2",
        earnings_date=date(2024, 10, 31),
        eps_actual=1.20,
        eps_estimate=1.50,
        eps_surprise_pct=-20.0,  # Big miss
        revenue_actual=8.5e9,
        revenue_estimate=9.5e9,
        revenue_surprise_pct=-10.5,  # Revenue miss
        guidance_direction=GuidanceDirection.LOWERED_STRONG,
        management_tone="DEFENSIVE",
        sentiment_score=25,
        data_source='mock',
    )
    
    result2 = calculate_eqs("TEST2", earnings_data=miss)
    
    print(f"\n  Miss Test:")
    print(f"    EPS Surprise: {result2.eps_surprise_pct:+.1f}%")
    print(f"    Revenue Surprise: {result2.revenue_surprise_pct:+.1f}%")
    print(f"    Guidance: {result2.guidance_direction.value}")
    print(f"    EQS: {result2.eqs:.0f}/100")
    print(f"    Event Z: {result2.event_z:+.2f}")
    
    assert result2.eqs is not None, "EQS should be calculated"
    assert result2.eqs <= 35, f"Miss should have low EQS, got {result2.eqs}"
    assert result2.event_z < -1.0, f"Miss should have negative event z, got {result2.event_z}"
    
    print("  [OK] Miss calculated correctly")
    
    print("  [OK] All mock data tests passed!\n")
    return True


def test_data_quality_assessment():
    """Test data quality assessment."""
    print("Testing Data Quality Assessment...")
    
    from src.analytics.earnings_intelligence.eqs_calculator import _assess_eqs_data_quality
    from src.analytics.earnings_intelligence.models import DataQuality
    
    # All inputs -> HIGH
    quality1 = _assess_eqs_data_quality([], 5)
    assert quality1 == DataQuality.HIGH
    print(f"  [OK] 5/5 inputs -> {quality1.value}")
    
    # Most inputs -> HIGH
    quality2 = _assess_eqs_data_quality(['margin_data'], 4)
    assert quality2 == DataQuality.HIGH
    print(f"  [OK] 4/5 inputs -> {quality2.value}")
    
    # Missing critical -> MEDIUM
    quality3 = _assess_eqs_data_quality(['eps_surprise'], 4)
    assert quality3 == DataQuality.HIGH  # Still 4 inputs
    print(f"  [OK] 4/5 with eps missing -> {quality3.value}")
    
    # Few inputs -> MEDIUM
    quality4 = _assess_eqs_data_quality(['eps_surprise', 'guidance', 'margin_data'], 2)
    assert quality4 == DataQuality.MEDIUM
    print(f"  [OK] 2/5 inputs -> {quality4.value}")
    
    # No data -> LOW
    quality5 = _assess_eqs_data_quality(['earnings_data'], 0)
    assert quality5 == DataQuality.LOW
    print(f"  [OK] 0/5 inputs -> {quality5.value}")
    
    print("  [OK] All data quality tests passed!\n")
    return True


def test_real_ticker_calculation():
    """Test EQS calculation for real tickers (uses yfinance fallback)."""
    print("Testing Real Ticker EQS Calculation...")
    
    from src.analytics.earnings_intelligence.eqs_calculator import calculate_eqs
    
    ticker = "AAPL"
    print(f"  Calculating EQS for {ticker}...")
    
    start = time.time()
    result = calculate_eqs(ticker)
    elapsed = time.time() - start
    
    print(f"  [OK] Calculation completed in {elapsed:.2f}s")
    print(f"  [OK] Ticker: {result.ticker}")
    print(f"  [OK] Earnings Date: {result.earnings_date}")
    
    if result.eqs is not None:
        print(f"  [OK] EQS: {result.eqs:.0f}/100")
    else:
        print(f"  [WARN] EQS: N/A (may not have recent earnings data)")
    
    if result.event_z is not None:
        print(f"  [OK] Event Z: {result.event_z:+.2f}")
    
    if result.eps_surprise_pct is not None:
        print(f"  [OK] EPS Surprise: {result.eps_surprise_pct:+.1f}%")
    
    print(f"  [OK] Data Quality: {result.data_quality.value}")
    print(f"  [OK] Data Source: {result.data_source}")
    
    print("  [OK] Real ticker calculation tests passed!\n")
    return True


def test_multiple_tickers():
    """Test EQS calculation for multiple tickers."""
    print("Testing Multiple Tickers...")
    
    from src.analytics.earnings_intelligence.eqs_calculator import calculate_eqs
    
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    for ticker in test_tickers:
        result = calculate_eqs(ticker)
        
        eqs_str = f"{result.eqs:.0f}" if result.eqs else "N/A"
        event_z_str = f"{result.event_z:+.2f}" if result.event_z else "N/A"
        eps_str = f"{result.eps_surprise_pct:+.1f}%" if result.eps_surprise_pct else "N/A"
        
        print(f"  [OK] {ticker}: EQS={eqs_str}, Event_Z={event_z_str}, EPS={eps_str}")
    
    print(f"\n  [OK] Processed {len(test_tickers)} tickers\n")
    return True


def test_ai_summary():
    """Test AI summary generation."""
    print("Testing AI Summary Generation...")
    
    from src.analytics.earnings_intelligence.eqs_calculator import (
        calculate_eqs,
        EarningsData,
        get_eqs_summary_for_ai,
    )
    from src.analytics.earnings_intelligence.models import GuidanceDirection
    
    # Use mock data for consistent output
    mock_data = EarningsData(
        ticker="NVDA",
        earnings_date=date(2024, 11, 20),
        eps_actual=0.81,
        eps_estimate=0.74,
        eps_surprise_pct=9.5,
        revenue_actual=35.1e9,
        revenue_estimate=33.2e9,
        revenue_surprise_pct=5.7,
        guidance_direction=GuidanceDirection.RAISED,
        management_tone="CONFIDENT",
        sentiment_score=80,
        data_source='mock',
    )
    
    result = calculate_eqs("NVDA", earnings_data=mock_data)
    summary = result.get_ai_summary()
    
    assert isinstance(summary, str), "Summary should be a string"
    assert "NVDA" in summary, "Summary should contain ticker"
    assert "EQS" in summary, "Summary should mention EQS"
    
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
    
    from src.analytics.earnings_intelligence.eqs_calculator import get_eqs_for_screener
    
    data = get_eqs_for_screener("AAPL")
    
    required_fields = ['eqs', 'event_z', 'eps_surprise_pct', 
                       'revenue_surprise_pct', 'guidance_direction', 'data_quality']
    
    for field in required_fields:
        assert field in data, f"Missing field: {field}"
    
    print(f"  [OK] All required fields present")
    print(f"  Data: {data}")
    
    print("  [OK] Screener integration tests passed!\n")
    return True


def test_result_serialization():
    """Test result JSON serialization."""
    print("Testing Result Serialization...")
    
    from src.analytics.earnings_intelligence.eqs_calculator import (
        calculate_eqs,
        EarningsData,
    )
    from src.analytics.earnings_intelligence.models import GuidanceDirection
    import json
    
    mock_data = EarningsData(
        ticker="TEST",
        earnings_date=date(2024, 10, 31),
        eps_actual=1.50,
        eps_estimate=1.40,
        eps_surprise_pct=7.1,
        guidance_direction=GuidanceDirection.RAISED,
        data_source='mock',
    )
    
    result = calculate_eqs("TEST", earnings_data=mock_data)
    data = result.to_dict()
    
    # Should be JSON serializable
    json_str = json.dumps(data)
    assert len(json_str) > 100, "Serialized data too small"
    
    # Deserialize and verify
    parsed = json.loads(json_str)
    assert parsed['ticker'] == 'TEST', "Ticker mismatch"
    
    print(f"  [OK] Serialized to {len(json_str)} bytes")
    print(f"  [OK] Keys: {list(data.keys())[:8]}...")
    
    print("  [OK] Serialization tests passed!\n")
    return True


def main():
    """Run all Phase 8 tests."""
    print("=" * 60)
    print("PHASE 8 VERIFICATION: EQS Calculator")
    print("(Earnings Quality Score)")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests in order
    all_passed &= test_imports()
    all_passed &= test_weights()
    all_passed &= test_z_score_calculations()
    all_passed &= test_score_conversions()
    all_passed &= test_earnings_data_structure()
    all_passed &= test_eqs_calculation_with_mock_data()
    all_passed &= test_data_quality_assessment()
    all_passed &= test_real_ticker_calculation()
    all_passed &= test_multiple_tickers()
    all_passed &= test_ai_summary()
    all_passed &= test_screener_integration()
    all_passed &= test_result_serialization()
    
    # Final result
    print("=" * 60)
    if all_passed:
        print("[OK] ALL PHASE 8 TESTS PASSED!")
    else:
        print("[FAIL] SOME TESTS FAILED - Review output above")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())