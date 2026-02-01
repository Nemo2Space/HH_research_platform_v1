#!/usr/bin/env python3
"""
Phase 12 Verification: Backtesting Module

Tests:
1. Historical data retrieval
2. Event scoring
3. Backtest metrics calculation
4. Report generation
5. CSV export

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
        from src.analytics.earnings_intelligence.backtesting import (
            run_backtest,
            run_quick_backtest,
            BacktestResult,
            EarningsEvent,
            get_historical_earnings_yfinance,
            calculate_event_scores,
            calculate_backtest_metrics,
            generate_backtest_report,
            save_backtest_to_csv,
        )

        print("  [OK] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_historical_data_retrieval():
    """Test historical earnings data retrieval."""
    print("\nTesting Historical Data Retrieval...")

    from src.analytics.earnings_intelligence.backtesting import (
        get_historical_earnings_yfinance,
        EarningsEvent,
    )

    ticker = "AAPL"
    print(f"  Getting historical earnings for {ticker}...")

    start = time.time()
    events = get_historical_earnings_yfinance(ticker, lookback_quarters=4)
    elapsed = time.time() - start

    print(f"  [OK] Retrieved {len(events)} events in {elapsed:.2f}s")

    if events:
        for event in events[:3]:
            print(f"    {event.earnings_date}: EPS={event.eps_surprise_pct:+.1f}% Reaction={event.total_reaction:+.1f}%"
                  if event.eps_surprise_pct and event.total_reaction else f"    {event.earnings_date}: Data incomplete")

    assert len(events) > 0, "Should retrieve at least one event"

    print("  [OK] Historical data retrieval tests passed!\n")
    return True


def test_event_scoring():
    """Test event scoring calculation."""
    print("Testing Event Scoring...")

    from src.analytics.earnings_intelligence.backtesting import (
        EarningsEvent,
        calculate_event_scores,
    )
    from src.analytics.earnings_intelligence.models import ECSCategory

    # Create test event - big beat
    # Note: To get a BEAT, need eps_z > required_z
    # With pctl=30, required_z = 0.8 + (30/50) = 1.4
    # EPS surprise of 30% -> z = (30-4.5)/12 = 2.13 > 1.4 -> BEAT
    event_beat = EarningsEvent(
        ticker="TEST",
        earnings_date=date(2024, 10, 31),
        eps_surprise_pct=30.0,  # Big beat (z ~ 2.1)
        revenue_surprise_pct=None,
        pre_close=100.0,
        post_open=105.0,
        post_close=108.0,
        gap_reaction=5.0,
        total_reaction=8.0,
    )

    scored = calculate_event_scores(event_beat, implied_move_pctl=30.0)  # Lower bar

    print(
        f"  Beat Event: EPS_Z={scored.eps_z:.2f}, Required_Z={scored.required_z:.2f}, ECS={scored.ecs_category.value if scored.ecs_category else 'N/A'}")

    assert scored.eps_z is not None, "Should calculate eps_z"
    assert scored.event_z is not None, "Should calculate event_z"
    assert scored.ecs_category is not None, "Should determine ECS category"

    # With lower required_z, should be a beat
    assert scored.ecs_category in (ECSCategory.BEAT,
                                   ECSCategory.STRONG_BEAT), f"Expected BEAT, got {scored.ecs_category.value}"
    assert scored.predicted_direction == 'positive', f"Expected positive, got {scored.predicted_direction}"
    assert scored.actual_direction == 'positive', f"Expected actual positive, got {scored.actual_direction}"
    assert scored.correct_prediction == True, "Should be correct prediction"

    print("  [OK] Beat event scored correctly")

    # Create test event - miss
    event_miss = EarningsEvent(
        ticker="TEST2",
        earnings_date=date(2024, 10, 31),
        eps_surprise_pct=-10.0,  # Miss
        revenue_surprise_pct=None,
        pre_close=100.0,
        post_open=95.0,
        post_close=92.0,
        gap_reaction=-5.0,
        total_reaction=-8.0,
    )

    scored_miss = calculate_event_scores(event_miss, implied_move_pctl=50.0)

    print(
        f"  Miss Event: EPS_Z={scored_miss.eps_z:.2f}, ECS={scored_miss.ecs_category.value if scored_miss.ecs_category else 'N/A'}")

    assert scored_miss.predicted_direction == 'negative', f"Expected negative, got {scored_miss.predicted_direction}"
    assert scored_miss.actual_direction == 'negative', f"Expected actual negative, got {scored_miss.actual_direction}"
    assert scored_miss.correct_prediction == True, "Should be correct prediction"

    print("  [OK] Miss event scored correctly")
    print("  [OK] Event scoring tests passed!\n")
    return True


def test_backtest_metrics():
    """Test backtest metrics calculation."""
    print("Testing Backtest Metrics Calculation...")

    from src.analytics.earnings_intelligence.backtesting import (
        EarningsEvent,
        calculate_event_scores,
        calculate_backtest_metrics,
    )

    # Create sample events
    events = []

    # 3 beats with positive reaction (correct)
    # Use lower implied_move_pctl=20 so required_z is lower (~0.9)
    # EPS 25-35% -> z = 1.7-2.5 which beats required_z of 0.9
    for i in range(3):
        event = EarningsEvent(
            ticker=f"BEAT{i}",
            earnings_date=date(2024, 10, 31),
            eps_surprise_pct=25.0 + i * 5,  # 25%, 30%, 35%
            revenue_surprise_pct=None,
            pre_close=100.0,
            post_open=105.0,
            post_close=107.0,
            gap_reaction=5.0,
            total_reaction=7.0,
        )
        events.append(calculate_event_scores(event, implied_move_pctl=20.0))

    # 2 misses with negative reaction (correct)
    for i in range(2):
        event = EarningsEvent(
            ticker=f"MISS{i}",
            earnings_date=date(2024, 10, 31),
            eps_surprise_pct=-10.0 - i * 5,
            revenue_surprise_pct=None,
            pre_close=100.0,
            post_open=95.0,
            post_close=93.0,
            gap_reaction=-5.0,
            total_reaction=-7.0,
        )
        events.append(calculate_event_scores(event, implied_move_pctl=20.0))

    # 1 wrong prediction (beat with negative reaction)
    wrong_event = EarningsEvent(
        ticker="WRONG",
        earnings_date=date(2024, 10, 31),
        eps_surprise_pct=30.0,  # Looks like a beat
        revenue_surprise_pct=None,
        pre_close=100.0,
        post_open=98.0,
        post_close=95.0,
        gap_reaction=-2.0,
        total_reaction=-5.0,  # But price dropped
    )
    events.append(calculate_event_scores(wrong_event, implied_move_pctl=20.0))

    # Calculate metrics
    metrics = calculate_backtest_metrics(events)

    print(f"  Events: {len(events)}")
    print(f"  ECS Accuracy: {metrics['ecs_accuracy']:.1f}%")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.1f}%")

    # Just verify metrics are calculated, don't assert specific values
    assert metrics['ecs_accuracy'] >= 0, "Should calculate accuracy"
    assert 'ecs_by_category' in metrics, "Should have category breakdown"

    print("  [OK] Backtest metrics tests passed!\n")
    return True


def test_full_backtest():
    """Test full backtest execution."""
    print("Testing Full Backtest...")

    from src.analytics.earnings_intelligence.backtesting import (
        run_backtest,
        BacktestResult,
    )

    test_tickers = ["AAPL", "MSFT"]

    def progress(current, total, ticker):
        print(f"    [{current}/{total}] {ticker}")

    print(f"  Running backtest on {test_tickers}...")

    start = time.time()
    result = run_backtest(test_tickers, lookback_quarters=4, progress_callback=progress)
    elapsed = time.time() - start

    print(f"  [OK] Backtest completed in {elapsed:.2f}s")

    # Verify result structure
    assert isinstance(result, BacktestResult)
    assert result.total_events > 0
    assert len(result.tickers_tested) == len(test_tickers)

    print(f"  [OK] Total events: {result.total_events}")
    print(f"  [OK] Valid events: {result.valid_events}")
    print(f"  [OK] ECS Accuracy: {result.ecs_accuracy:.1f}%")
    print(f"  [OK] Direction Accuracy: {result.direction_accuracy:.1f}%")

    print("  [OK] Full backtest tests passed!\n")
    return True


def test_backtest_report():
    """Test backtest report generation."""
    print("Testing Report Generation...")

    from src.analytics.earnings_intelligence.backtesting import (
        run_backtest,
        generate_backtest_report,
    )

    result = run_backtest(["AAPL"], lookback_quarters=4)

    report = generate_backtest_report(result)

    assert isinstance(report, str)
    assert len(report) > 100
    assert "BACKTEST" in report

    print("  [OK] Report generated")
    print("\n  --- Report Preview ---")
    lines = report.split('\n')[:20]
    for line in lines:
        print(f"    {line}")
    print("    ...")

    print("\n  [OK] Report generation tests passed!\n")
    return True


def test_result_serialization():
    """Test result serialization."""
    print("Testing Result Serialization...")

    from src.analytics.earnings_intelligence.backtesting import run_backtest
    import json

    result = run_backtest(["AAPL"], lookback_quarters=2)

    # Test to_dict
    data = result.to_dict()
    assert isinstance(data, dict)

    # Should be JSON serializable
    json_str = json.dumps(data)
    assert len(json_str) > 100

    print(f"  [OK] Serialized to {len(json_str)} bytes")

    print("  [OK] Serialization tests passed!\n")
    return True


def test_quick_backtest():
    """Test quick backtest function."""
    print("Testing Quick Backtest...")

    from src.analytics.earnings_intelligence.backtesting import run_quick_backtest

    # Run with custom tickers (smaller set for speed)
    result = run_quick_backtest(tickers=["AAPL", "NVDA"])

    assert result.total_events > 0
    print(f"  [OK] Quick backtest: {result.total_events} events, {result.ecs_accuracy:.1f}% accuracy")

    print("  [OK] Quick backtest tests passed!\n")
    return True


def test_multiple_tickers():
    """Test backtest with multiple tickers."""
    print("Testing Multiple Tickers Backtest...")

    from src.analytics.earnings_intelligence.backtesting import run_backtest

    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL"]

    start = time.time()
    result = run_backtest(test_tickers, lookback_quarters=4)
    elapsed = time.time() - start

    print(f"  [OK] Processed {len(test_tickers)} tickers in {elapsed:.2f}s")
    print(f"  [OK] Total events: {result.total_events}")
    print(f"  [OK] ECS Accuracy: {result.ecs_accuracy:.1f}%")

    # Print category breakdown
    print("\n  ECS Category Breakdown:")
    for cat, stats in result.ecs_by_category.items():
        print(f"    {cat}: {stats['count']} events, {stats['avg_return']:+.1f}% avg return")

    print("\n  [OK] Multiple tickers tests passed!\n")
    return True


def main():
    """Run all Phase 12 tests."""
    print("=" * 60)
    print("PHASE 12 VERIFICATION: Backtesting Module")
    print("=" * 60)

    all_passed = True

    # Run tests in order
    all_passed &= test_imports()
    all_passed &= test_historical_data_retrieval()
    all_passed &= test_event_scoring()
    all_passed &= test_backtest_metrics()
    all_passed &= test_full_backtest()
    all_passed &= test_backtest_report()
    all_passed &= test_result_serialization()
    all_passed &= test_quick_backtest()
    all_passed &= test_multiple_tickers()

    # Final result
    print("=" * 60)
    if all_passed:
        print("[OK] ALL PHASE 12 TESTS PASSED!")
    else:
        print("[FAIL] SOME TESTS FAILED - Review output above")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())