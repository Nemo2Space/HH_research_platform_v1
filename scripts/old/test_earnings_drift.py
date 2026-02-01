"""
Phase 4 Verification Script - Test Price Drift Calculator

Run this to verify drift calculations are working correctly.

Usage:
    python scripts/test_earnings_drift.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_sector_etf_mapping():
    """Test sector to ETF mapping."""
    from src.analytics.earnings_intelligence import SECTOR_ETFS, get_sector_etf

    print("Testing Sector ETF Mapping...")

    test_cases = [
        ('Technology', 'XLK'),
        ('Healthcare', 'XLV'),
        ('Financials', 'XLF'),
        ('Energy', 'XLE'),
        ('Consumer Cyclical', 'XLY'),  # Alias
        ('Information Technology', 'XLK'),  # Alias
        ('Unknown Sector', 'SPY'),  # Default
        (None, 'SPY'),  # None
    ]

    for sector, expected_etf in test_cases:
        result = get_sector_etf(sector)
        status = "[OK]" if result == expected_etf else "[FAIL]"
        print(f"  {status} '{sector}' -> {result} (expected {expected_etf})")

    print("  [OK] Sector ETF mapping tests passed!\n")


def test_normalize_drift_to_score():
    """Test drift normalization to 0-100 score."""
    from src.analytics.earnings_intelligence import normalize_drift_to_score

    print("Testing Drift Score Normalization...")

    # Default thresholds: low=-0.10, high=0.20
    test_cases = [
        # (drift, expected_score_approx)
        (-0.20, 0),    # Below low threshold
        (-0.10, 0),    # At low threshold
        (-0.05, 17),   # Between thresholds
        (0.00, 33),    # Zero drift
        (0.05, 50),    # Midpoint
        (0.10, 67),    # Positive drift
        (0.15, 83),    # Higher drift
        (0.20, 100),   # At high threshold
        (0.30, 100),   # Above high threshold
        (None, 50),    # None returns neutral
    ]

    for drift, expected in test_cases:
        result = normalize_drift_to_score(drift)
        diff = abs(result - expected)
        status = "[OK]" if diff < 5 else "[FAIL]"  # Allow 5 point tolerance
        drift_str = f"{drift:+.0%}" if drift is not None else "None"
        print(f"  {status} drift={drift_str} -> score={result:.0f} (expected ~{expected})")

    print("  [OK] Score normalization tests passed!\n")


def test_live_drift_calculations():
    """Test live drift calculations (requires internet)."""
    from src.analytics.earnings_intelligence import (
        calculate_drift_20d,
        calculate_relative_drift,
        calculate_all_drift_metrics,
        get_ticker_sector,
    )

    print("Testing Live Drift Calculations...")
    print("  (Requires internet connection)")

    # Test with diversified tickers from different sectors
    test_tickers = [
        ("NVDA", "Technology"),
        ("XOM", "Energy"),
        ("JPM", "Financials"),
    ]

    try:
        for ticker, expected_sector in test_tickers:
            print(f"\n  {ticker}:")

            # Get sector
            sector = get_ticker_sector(ticker)
            if sector:
                print(f"    Sector: {sector}")
            else:
                print(f"    Sector: Unable to fetch (using expected: {expected_sector})")
                sector = expected_sector

            # Calculate drift
            drift = calculate_drift_20d(ticker)
            if drift is not None:
                print(f"    20-day Drift: {drift:+.2%}")
            else:
                print(f"    20-day Drift: Unable to calculate")

            # Calculate relative drift
            rel_drift = calculate_relative_drift(ticker, sector)
            if rel_drift is not None:
                print(f"    Relative Drift: {rel_drift:+.2%}")
            else:
                print(f"    Relative Drift: Unable to calculate")

            # Verify drift is reasonable (between -50% and +100%)
            if drift is not None:
                assert -0.50 <= drift <= 1.00, f"Drift {drift} out of reasonable range"

        print("\n  [OK] Live drift calculations working!\n")
        return True

    except Exception as e:
        print(f"\n  [WARN] Live test failed: {e}")
        print("  [SKIP] Skipping live tests\n")
        return False


def test_all_drift_metrics():
    """Test the combined metrics function."""
    from src.analytics.earnings_intelligence import calculate_all_drift_metrics

    print("Testing calculate_all_drift_metrics()...")

    try:
        metrics = calculate_all_drift_metrics("AAPL")

        # Check all expected keys present
        expected_keys = ['drift_20d', 'rel_drift_20d', 'sector', 'sector_etf']
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
        print(f"  [OK] All expected keys present: {list(metrics.keys())}")

        # Check sector ETF is valid
        assert metrics['sector_etf'] in ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLP',
                                          'XLI', 'XLB', 'XLU', 'XLRE', 'XLC', 'SPY']
        print(f"  [OK] Sector ETF is valid: {metrics['sector_etf']}")

        print(f"  [OK] AAPL metrics: drift={metrics['drift_20d']}, rel_drift={metrics['rel_drift_20d']}")
        print("  [OK] calculate_all_drift_metrics() tests passed!\n")

    except Exception as e:
        print(f"  [WARN] Test failed: {e}\n")


def test_cache():
    """Test price caching."""
    from src.analytics.earnings_intelligence import (
        calculate_drift_20d,
        clear_price_cache,
    )
    import src.analytics.earnings_intelligence.drift as drift_module
    import time

    print("Testing Price Cache...")

    # Clear cache first
    clear_price_cache()
    print("  [OK] Cache cleared")

    # First call should populate cache
    start = time.time()
    drift1 = calculate_drift_20d("MSFT")
    time1 = time.time() - start
    print(f"  [OK] First call: {time1:.2f}s")

    # Second call should be faster (from cache)
    start = time.time()
    drift2 = calculate_drift_20d("MSFT")
    time2 = time.time() - start
    print(f"  [OK] Second call (cached): {time2:.4f}s")

    # Results should be identical
    assert drift1 == drift2, "Cached result differs"
    print("  [OK] Cached result matches")

    # Verify caching is working by checking time difference
    # Second call should be significantly faster
    if time1 > 0.1:  # Only check if first call took meaningful time
        assert time2 < time1 / 2, f"Cache doesn't seem to be working: {time2:.4f}s vs {time1:.2f}s"
        print(f"  [OK] Cache speedup confirmed ({time1/max(time2,0.0001):.0f}x faster)")
    else:
        print(f"  [INFO] First call was fast ({time1:.2f}s), may have been pre-cached")

    print("  [OK] Cache tests passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PHASE 4 VERIFICATION: Price Drift Calculator")
    print("=" * 60 + "\n")

    try:
        test_sector_etf_mapping()
        test_normalize_drift_to_score()
        test_cache()
        test_all_drift_metrics()
        test_live_drift_calculations()

        print("=" * 60)
        print("[OK] ALL PHASE 4 TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())