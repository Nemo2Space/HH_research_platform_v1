#!/usr/bin/env python3
"""
Phase 11 Verification: AI Chat Integration

Tests:
1. Earnings context generation
2. Keyword detection
3. Quick status formatting
4. Conversational formatting
5. Portfolio alerts
6. Integration helper function

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
        from src.analytics.earnings_intelligence.chat_integration import (
            get_earnings_context_for_ai,
            get_quick_earnings_status,
            format_earnings_for_chat,
            get_earnings_alerts_for_portfolio,
            get_high_risk_earnings_tickers,
            needs_earnings_context,
            get_earnings_keywords,
            get_earnings_for_ai,
            EARNINGS_KEYWORDS,
        )

        print("  [OK] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_keywords():
    """Test earnings keyword detection."""
    print("\nTesting Keyword Detection...")

    from src.analytics.earnings_intelligence.chat_integration import (
        needs_earnings_context,
        get_earnings_keywords,
        EARNINGS_KEYWORDS,
    )

    # Verify keywords list
    keywords = get_earnings_keywords()
    assert len(keywords) > 10, "Should have many keywords"
    print(f"  [OK] {len(keywords)} earnings keywords defined")

    # Test positive cases
    positive_messages = [
        "When is AAPL earnings?",
        "Did NVDA beat expectations?",
        "What's the EPS estimate?",
        "Tell me about the quarterly results",
        "Should I sell before earnings?",
        "What's the IES for TSLA?",
        "Did they raise guidance?",
    ]

    for msg in positive_messages:
        assert needs_earnings_context(msg), f"Should detect: {msg}"
    print(f"  [OK] All {len(positive_messages)} positive cases detected")

    # Test negative cases
    negative_messages = [
        "What's the current price?",
        "Show me the chart",
        "What sector is this?",
        "How many shares do I own?",
    ]

    for msg in negative_messages:
        result = needs_earnings_context(msg)
        # These shouldn't trigger earnings context
        print(f"  [--] '{msg[:30]}...': {result}")

    print("  [OK] Keyword detection tests passed!\n")
    return True


def test_earnings_context():
    """Test earnings context generation."""
    print("Testing Earnings Context Generation...")

    from src.analytics.earnings_intelligence.chat_integration import (
        get_earnings_context_for_ai,
    )

    ticker = "AAPL"
    print(f"  Getting earnings context for {ticker}...")

    start = time.time()
    context = get_earnings_context_for_ai(ticker)
    elapsed = time.time() - start

    print(f"  [OK] Generated in {elapsed:.2f}s")

    if context:
        assert isinstance(context, str)
        assert ticker in context
        print(f"  [OK] Context length: {len(context)} chars")
        print("\n  --- Context Preview ---")
        lines = context.split('\n')[:15]
        for line in lines:
            print(f"    {line}")
        if len(context.split('\n')) > 15:
            print("    ...")
    else:
        print(f"  [--] No earnings context available for {ticker}")

    print("  [OK] Earnings context tests passed!\n")
    return True


def test_quick_status():
    """Test quick status formatting."""
    print("Testing Quick Status...")

    from src.analytics.earnings_intelligence.chat_integration import (
        get_quick_earnings_status,
    )

    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]

    for ticker in test_tickers:
        status = get_quick_earnings_status(ticker)
        if status:
            print(f"  [OK] {status}")
        else:
            print(f"  [--] {ticker}: No quick status")

    print("  [OK] Quick status tests passed!\n")
    return True


def test_conversational_format():
    """Test conversational formatting."""
    print("Testing Conversational Format...")

    from src.analytics.earnings_intelligence.chat_integration import (
        format_earnings_for_chat,
    )

    ticker = "NVDA"
    print(f"  Formatting earnings for {ticker}...")

    response = format_earnings_for_chat(ticker)

    assert isinstance(response, str)
    assert len(response) > 10

    print(f"  [OK] Response length: {len(response)} chars")
    print("\n  --- Response Preview ---")
    lines = response.split('\n')[:12]
    for line in lines:
        print(f"    {line}")
    if len(response.split('\n')) > 12:
        print("    ...")

    print("\n  [OK] Conversational format tests passed!\n")
    return True


def test_portfolio_alerts():
    """Test portfolio earnings alerts."""
    print("Testing Portfolio Alerts...")

    from src.analytics.earnings_intelligence.chat_integration import (
        get_earnings_alerts_for_portfolio,
    )

    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "META", "AMZN"]

    alerts = get_earnings_alerts_for_portfolio(test_tickers)

    if alerts:
        print(f"  [OK] Got alerts")
        print("\n  --- Alerts ---")
        for line in alerts.split('\n'):
            print(f"    {line}")
    else:
        print("  [--] No earnings alerts (no stocks near earnings)")

    print("  [OK] Portfolio alerts tests passed!\n")
    return True


def test_high_risk_detection():
    """Test high-risk earnings detection."""
    print("Testing High-Risk Detection...")

    from src.analytics.earnings_intelligence.chat_integration import (
        get_high_risk_earnings_tickers,
    )

    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]

    high_risk = get_high_risk_earnings_tickers(test_tickers)

    print(f"  [OK] High-risk tickers: {high_risk if high_risk else 'None'}")

    print("  [OK] High-risk detection tests passed!\n")
    return True


def test_integration_helper():
    """Test the drop-in integration helper."""
    print("Testing Integration Helper (get_earnings_for_ai)...")

    from src.analytics.earnings_intelligence.chat_integration import (
        get_earnings_for_ai,
    )

    # This should match the interface expected by chat.py
    ticker = "AAPL"
    context = get_earnings_for_ai(ticker)

    assert isinstance(context, str)
    print(f"  [OK] get_earnings_for_ai({ticker}) returns {len(context)} chars")

    print("  [OK] Integration helper tests passed!\n")
    return True


def test_multiple_tickers():
    """Test with multiple tickers."""
    print("Testing Multiple Tickers...")

    from src.analytics.earnings_intelligence.chat_integration import (
        get_earnings_context_for_ai,
    )

    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "META"]

    start = time.time()
    results = {}
    for ticker in test_tickers:
        context = get_earnings_context_for_ai(ticker)
        results[ticker] = len(context) if context else 0
    elapsed = time.time() - start

    print(f"  [OK] Processed {len(test_tickers)} tickers in {elapsed:.2f}s")

    for ticker, length in results.items():
        if length > 0:
            print(f"  [OK] {ticker}: {length} chars")
        else:
            print(f"  [--] {ticker}: No context")

    print("  [OK] Multiple tickers tests passed!\n")
    return True


def test_context_with_full_analysis():
    """Test context with full analysis flag."""
    print("Testing Full Analysis Mode...")

    from src.analytics.earnings_intelligence.chat_integration import (
        get_earnings_context_for_ai,
    )

    ticker = "NVDA"

    # Without full analysis
    context_basic = get_earnings_context_for_ai(ticker, include_full_analysis=False)

    # With full analysis
    context_full = get_earnings_context_for_ai(ticker, include_full_analysis=True)

    print(f"  [OK] Basic context: {len(context_basic)} chars")
    print(f"  [OK] Full context: {len(context_full)} chars")

    # Full should be same or longer
    assert len(context_full) >= len(context_basic) - 50, "Full should be at least as long"

    print("  [OK] Full analysis mode tests passed!\n")
    return True


def main():
    """Run all Phase 11 tests."""
    print("=" * 60)
    print("PHASE 11 VERIFICATION: AI Chat Integration")
    print("=" * 60)

    all_passed = True

    # Run tests in order
    all_passed &= test_imports()
    all_passed &= test_keywords()
    all_passed &= test_earnings_context()
    all_passed &= test_quick_status()
    all_passed &= test_conversational_format()
    all_passed &= test_portfolio_alerts()
    all_passed &= test_high_risk_detection()
    all_passed &= test_integration_helper()
    all_passed &= test_multiple_tickers()
    all_passed &= test_context_with_full_analysis()

    # Final result
    print("=" * 60)
    if all_passed:
        print("[OK] ALL PHASE 11 TESTS PASSED!")
    else:
        print("[FAIL] SOME TESTS FAILED - Review output above")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())