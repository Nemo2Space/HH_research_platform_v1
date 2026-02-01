#!/usr/bin/env python3
"""
Phase 10 Verification: Screener Integration

Tests:
1. Single ticker enrichment
2. Batch adjustments
3. Earnings window filtering
4. Score adjustment calculation
5. Position scale recommendations
6. Report generation

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
        from src.analytics.earnings_intelligence.screener_integration import (
            enrich_screener_with_earnings,
            get_earnings_adjustments,
            get_tickers_in_earnings_window,
            get_upcoming_earnings,
            process_earnings_for_screener,
            apply_earnings_adjustment,
            get_position_scale,
            should_flag_for_earnings,
            generate_earnings_report,
            EarningsEnrichment,
            SCORE_ADJUSTMENTS,
            PRE_EARNINGS_ADJUSTMENT,
        )

        print("  [OK] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_score_adjustments_config():
    """Test score adjustment configuration."""
    print("\nTesting Score Adjustment Configuration...")

    from src.analytics.earnings_intelligence.screener_integration import (
        SCORE_ADJUSTMENTS,
        PRE_EARNINGS_ADJUSTMENT,
    )
    from src.analytics.earnings_intelligence.models import ECSCategory

    # Verify all ECS categories have adjustments
    for cat in ECSCategory:
        assert cat in SCORE_ADJUSTMENTS, f"Missing adjustment for {cat.value}"
        print(f"  [OK] {cat.value}: {SCORE_ADJUSTMENTS[cat]:+d}")

    # Verify pre-earnings adjustments
    print("\n  Pre-earnings adjustments:")
    for key, val in PRE_EARNINGS_ADJUSTMENT.items():
        print(f"  [OK] {key}: {val:+d}")

    print("  [OK] Configuration tests passed!\n")
    return True


def test_single_ticker_enrichment():
    """Test enrichment for a single ticker."""
    print("Testing Single Ticker Enrichment...")

    from src.analytics.earnings_intelligence.screener_integration import (
        enrich_screener_with_earnings,
        EarningsEnrichment,
    )

    ticker = "AAPL"
    print(f"  Enriching {ticker}...")

    start = time.time()
    enrichment = enrich_screener_with_earnings(ticker)
    elapsed = time.time() - start

    print(f"  [OK] Completed in {elapsed:.2f}s")

    # Verify structure
    assert isinstance(enrichment, EarningsEnrichment)
    assert enrichment.ticker == ticker
    print(f"  [OK] Ticker: {enrichment.ticker}")

    # Check fields
    print(f"  [OK] Has Upcoming Earnings: {enrichment.has_upcoming_earnings}")
    print(f"  [OK] Days to Earnings: {enrichment.days_to_earnings}")
    print(f"  [OK] In Action Window: {enrichment.in_action_window}")

    if enrichment.ies is not None:
        print(f"  [OK] IES: {enrichment.ies:.0f}")
    else:
        print(f"  [--] IES: N/A (not in compute window)")

    if enrichment.regime:
        print(f"  [OK] Regime: {enrichment.regime.value}")

    print(f"  [OK] Score Adjustment: {enrichment.score_adjustment:+d}")
    print(f"  [OK] Position Scale: {enrichment.position_scale:.0%}")
    print(f"  [OK] Risk Flags: {enrichment.risk_flags}")

    # Test to_dict
    data = enrichment.to_dict()
    assert isinstance(data, dict)
    assert data['ticker'] == ticker
    print(f"  [OK] to_dict() works ({len(data)} keys)")

    print("  [OK] Single ticker enrichment tests passed!\n")
    return True


def test_batch_adjustments():
    """Test batch earnings adjustments."""
    print("Testing Batch Adjustments...")

    from src.analytics.earnings_intelligence.screener_integration import (
        get_earnings_adjustments,
    )

    test_tickers = ["AAPL", "MSFT", "NVDA"]

    def progress(current, total, ticker):
        print(f"  [{current}/{total}] {ticker}")

    start = time.time()
    adjustments = get_earnings_adjustments(test_tickers, progress_callback=progress)
    elapsed = time.time() - start

    print(f"  [OK] Completed in {elapsed:.2f}s")

    # Verify all tickers processed
    assert len(adjustments) == len(test_tickers)

    for ticker, adj in adjustments.items():
        assert 'score_adjustment' in adj
        assert 'position_scale' in adj
        assert 'risk_flags' in adj

        print(f"  [OK] {ticker}: adj={adj['score_adjustment']:+d}, scale={adj['position_scale']:.0%}")

    print("  [OK] Batch adjustment tests passed!\n")
    return True


def test_earnings_window_filtering():
    """Test filtering tickers by earnings window."""
    print("Testing Earnings Window Filtering...")

    from src.analytics.earnings_intelligence.screener_integration import (
        get_tickers_in_earnings_window,
        get_upcoming_earnings,
    )

    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "META", "AMZN"]

    # Test action window
    in_action = get_tickers_in_earnings_window(test_tickers, 'action')
    print(f"  [OK] Tickers in action window: {len(in_action)}")
    for ticker in in_action:
        print(f"       - {ticker}")

    # Test compute window
    in_compute = get_tickers_in_earnings_window(test_tickers, 'compute')
    print(f"  [OK] Tickers in compute window: {len(in_compute)}")

    # Test upcoming earnings
    upcoming = get_upcoming_earnings(test_tickers, days_ahead=30)
    print(f"  [OK] Upcoming earnings (30 days): {len(upcoming)}")
    for item in upcoming[:5]:
        print(f"       - {item['ticker']}: {item['earnings_date']} ({item['days_to_earnings']} days)")

    print("  [OK] Earnings window filtering tests passed!\n")
    return True


def test_helper_functions():
    """Test helper functions for screener integration."""
    print("Testing Helper Functions...")

    from src.analytics.earnings_intelligence.screener_integration import (
        apply_earnings_adjustment,
        get_position_scale,
        should_flag_for_earnings,
    )

    ticker = "AAPL"

    # Test apply_earnings_adjustment
    base_score = 75
    adjusted, flags = apply_earnings_adjustment(base_score, ticker)
    print(f"  [OK] apply_earnings_adjustment({base_score}, {ticker}): {adjusted} (flags: {flags})")

    # Test get_position_scale
    scale = get_position_scale(ticker)
    assert 0.2 <= scale <= 1.0
    print(f"  [OK] get_position_scale({ticker}): {scale:.0%}")

    # Test should_flag_for_earnings
    flag = should_flag_for_earnings(ticker)
    print(f"  [OK] should_flag_for_earnings({ticker}): {flag}")

    print("  [OK] Helper function tests passed!\n")
    return True


def test_batch_processing():
    """Test batch processing for screener."""
    print("Testing Batch Processing for Screener...")

    from src.analytics.earnings_intelligence.screener_integration import (
        process_earnings_for_screener,
    )

    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL"]

    # Test fast mode
    print("  Testing fast mode (only process tickers in window)...")
    start = time.time()
    df = process_earnings_for_screener(test_tickers, fast_mode=True)
    elapsed = time.time() - start

    print(f"  [OK] Fast mode completed in {elapsed:.2f}s")
    print(f"  [OK] DataFrame shape: {df.shape}")
    print(f"  [OK] Columns: {list(df.columns)[:8]}...")

    # Check all tickers are in result (even if not processed)
    assert len(df) >= len(test_tickers), "Some tickers missing from result"

    print("  [OK] Batch processing tests passed!\n")
    return True


def test_report_generation():
    """Test earnings report generation."""
    print("Testing Report Generation...")

    from src.analytics.earnings_intelligence.screener_integration import (
        generate_earnings_report,
    )

    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]

    report = generate_earnings_report(test_tickers)

    assert isinstance(report, str)
    assert len(report) > 100
    assert "EARNINGS INTELLIGENCE REPORT" in report

    print("  [OK] Report generated successfully")
    print("\n--- Report Preview ---")
    lines = report.split('\n')[:30]
    for line in lines:
        print(f"  {line}")
    if len(report.split('\n')) > 30:
        print("  ...")

    print("\n  [OK] Report generation tests passed!\n")
    return True


def test_enrichment_serialization():
    """Test enrichment serialization."""
    print("Testing Enrichment Serialization...")

    from src.analytics.earnings_intelligence.screener_integration import (
        enrich_screener_with_earnings,
    )
    import json

    enrichment = enrich_screener_with_earnings("AAPL")
    data = enrichment.to_dict()

    # Should be JSON serializable
    json_str = json.dumps(data)
    assert len(json_str) > 50

    # Deserialize and verify
    parsed = json.loads(json_str)
    assert parsed['ticker'] == 'AAPL'

    print(f"  [OK] Serialized to {len(json_str)} bytes")
    print(f"  [OK] Keys: {list(data.keys())}")

    print("  [OK] Serialization tests passed!\n")
    return True


def test_multiple_tickers_performance():
    """Test performance with multiple tickers."""
    print("Testing Multiple Tickers Performance...")

    from src.analytics.earnings_intelligence.screener_integration import (
        enrich_screener_with_earnings,
    )

    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "META", "AMZN", "AMD"]

    start = time.time()
    results = []
    for ticker in test_tickers:
        enrichment = enrich_screener_with_earnings(ticker)
        results.append(enrichment)
    elapsed = time.time() - start

    avg_time = elapsed / len(test_tickers)

    print(f"  [OK] Processed {len(test_tickers)} tickers in {elapsed:.2f}s")
    print(f"  [OK] Average: {avg_time:.2f}s per ticker")

    # Summarize results
    in_window = sum(1 for r in results if r.in_action_window)
    with_ies = sum(1 for r in results if r.ies is not None)
    with_adj = sum(1 for r in results if r.score_adjustment != 0)

    print(f"\n  Summary:")
    print(f"    In action window: {in_window}")
    print(f"    With IES calculated: {with_ies}")
    print(f"    With score adjustment: {with_adj}")

    print("  [OK] Performance tests passed!\n")
    return True


def main():
    """Run all Phase 10 tests."""
    print("=" * 60)
    print("PHASE 10 VERIFICATION: Screener Integration")
    print("=" * 60)

    all_passed = True

    # Run tests in order
    all_passed &= test_imports()
    all_passed &= test_score_adjustments_config()
    all_passed &= test_single_ticker_enrichment()
    all_passed &= test_batch_adjustments()
    all_passed &= test_earnings_window_filtering()
    all_passed &= test_helper_functions()
    all_passed &= test_batch_processing()
    all_passed &= test_report_generation()
    all_passed &= test_enrichment_serialization()
    all_passed &= test_multiple_tickers_performance()

    # Final result
    print("=" * 60)
    if all_passed:
        print("[OK] ALL PHASE 10 TESTS PASSED!")
    else:
        print("[FAIL] SOME TESTS FAILED - Review output above")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())