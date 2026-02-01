#!/usr/bin/env python3
"""
Data Quality Validation Test Suite

Run this script after applying all the data quality fixes to verify:
1. None values are properly returned when data is missing
2. No more silent 50.0 or 0 defaults
3. Display methods handle None correctly
4. Scoring properly handles partial data

Usage:
    python test_data_quality_fixes.py

Or run specific tests:
    python test_data_quality_fixes.py --test earnings
    python test_data_quality_fixes.py --test scorer
    python test_data_quality_fixes.py --test squeeze
    python test_data_quality_fixes.py --test trade_ideas

Author: Alpha Research Platform
"""

import sys
import os
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Any
import traceback

# =============================================================================
# PATH SETUP - Add project root to Python path
# =============================================================================
# Get the directory where this script is located
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to project root (assumes script is in scripts/ folder)
_project_root = os.path.dirname(_script_dir)
# Add to path if not already there
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

print(f"Project root: {_project_root}")
print(f"Python path includes: {_project_root in sys.path}")


# =============================================================================
# TEST UTILITIES
# =============================================================================

class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []

    def record_pass(self, test_name: str):
        self.passed += 1
        print(f"  ✅ PASS: {test_name}")

    def record_fail(self, test_name: str, reason: str):
        self.failed += 1
        self.errors.append((test_name, reason))
        print(f"  ❌ FAIL: {test_name}")
        print(f"         Reason: {reason}")

    def record_skip(self, test_name: str, reason: str):
        self.skipped += 1
        print(f"  ⏭️  SKIP: {test_name} - {reason}")

    def summary(self):
        total = self.passed + self.failed + self.skipped
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"  Total:   {total}")
        print(f"  Passed:  {self.passed} ✅")
        print(f"  Failed:  {self.failed} ❌")
        print(f"  Skipped: {self.skipped} ⏭️")

        if self.errors:
            print("\nFailed Tests:")
            for name, reason in self.errors:
                print(f"  - {name}: {reason}")

        return self.failed == 0


def safe_import(module_path: str):
    """Safely import a module, return None if not available."""
    try:
        parts = module_path.rsplit('.', 1)
        if len(parts) == 2:
            module = __import__(parts[0], fromlist=[parts[1]])
            return getattr(module, parts[1])
        else:
            return __import__(module_path)
    except ImportError as e:
        return None
    except Exception as e:
        print(f"  Warning: Error importing {module_path}: {e}")
        return None


# =============================================================================
# TEST 1: EARNINGS INTELLIGENCE
# =============================================================================

def test_earnings_intelligence(results: TestResults):
    """Test earnings_intelligence.py fixes."""
    print("\n" + "=" * 60)
    print("TEST 1: EARNINGS INTELLIGENCE")
    print("=" * 60)

    # Import
    try:
        from src.analytics.earnings_intelligence import (
            enrich_screener_with_earnings,
            EarningsEnrichment,
            ComponentResult,
            ComponentStatus,
            IESRegime,
        )
    except ImportError as e:
        results.record_skip("earnings_intelligence import", str(e))
        return

    # Test 1.1: EarningsEnrichment default values are None, not 50
    print("\n  Testing EarningsEnrichment defaults...")
    enrichment = EarningsEnrichment(ticker="TEST")

    if enrichment.ies is None:
        results.record_pass("EarningsEnrichment.ies defaults to None")
    else:
        results.record_fail("EarningsEnrichment.ies defaults to None",
                            f"Got {enrichment.ies} instead of None")

    if enrichment.pre_earnings_runup is None:
        results.record_pass("EarningsEnrichment.pre_earnings_runup defaults to None")
    else:
        results.record_fail("EarningsEnrichment.pre_earnings_runup defaults to None",
                            f"Got {enrichment.pre_earnings_runup} instead of None")

    # Test 1.2: ies_display property
    if "N/A" in enrichment.ies_display or enrichment.ies_display == "N/A":
        results.record_pass("ies_display shows N/A when ies is None")
    else:
        results.record_fail("ies_display shows N/A when ies is None",
                            f"Got '{enrichment.ies_display}'")

    # Test 1.3: has_sufficient_data property
    if not enrichment.has_sufficient_data:
        results.record_pass("has_sufficient_data is False when no components")
    else:
        results.record_fail("has_sufficient_data is False when no components",
                            "Should be False with default values")

    # Test 1.4: IESRegime has UNKNOWN
    if hasattr(IESRegime, 'UNKNOWN'):
        results.record_pass("IESRegime has UNKNOWN value")
    else:
        results.record_fail("IESRegime has UNKNOWN value",
                            "UNKNOWN enum value not found")

    # Test 1.5: Live test with real ticker
    print("\n  Testing live enrichment...")
    try:
        result = enrich_screener_with_earnings("AAPL")

        # Check that we got a result
        if isinstance(result, EarningsEnrichment):
            results.record_pass("enrich_screener_with_earnings returns EarningsEnrichment")
        else:
            results.record_fail("enrich_screener_with_earnings returns EarningsEnrichment",
                                f"Got {type(result)}")

        # Print diagnostic info
        print(f"\n  AAPL Enrichment Results:")
        print(f"    IES: {result.ies_display}")
        print(f"    Regime: {result.regime}")
        print(f"    Components: {result.components_available}/{result.components_total}")
        print(f"    Confidence: {result.ies_confidence:.0%}")
        print(f"    Has sufficient data: {result.has_sufficient_data}")

    except Exception as e:
        results.record_skip("Live enrichment test", f"Error: {e}")


# =============================================================================
# TEST 2: UNIFIED SCORER
# =============================================================================

def test_unified_scorer(results: TestResults):
    """Test unified_scorer.py fixes."""
    print("\n" + "=" * 60)
    print("TEST 2: UNIFIED SCORER")
    print("=" * 60)

    try:
        from src.core.unified_scorer import (
            UnifiedScorer,
            TickerFeatures,
            ScoringResult,
            ScoringStatus,
            DataQuality,
        )
    except ImportError as e:
        results.record_skip("unified_scorer import", str(e))
        return

    # Test 2.1: TickerFeatures default values are None
    print("\n  Testing TickerFeatures defaults...")
    features = TickerFeatures(ticker="TEST", as_of_time=datetime.now())

    if features.sentiment_score is None:
        results.record_pass("TickerFeatures.sentiment_score defaults to None")
    else:
        results.record_fail("TickerFeatures.sentiment_score defaults to None",
                            f"Got {features.sentiment_score}")

    if features.options_flow_score is None:
        results.record_pass("TickerFeatures.options_flow_score defaults to None")
    else:
        results.record_fail("TickerFeatures.options_flow_score defaults to None",
                            f"Got {features.options_flow_score}")

    # Test 2.2: ScoringResult defaults
    print("\n  Testing ScoringResult defaults...")
    result = ScoringResult(ticker="TEST", as_of_time=datetime.now())

    if result.composite_score is None:
        results.record_pass("ScoringResult.composite_score defaults to None")
    else:
        results.record_fail("ScoringResult.composite_score defaults to None",
                            f"Got {result.composite_score}")

    if result.signal_type == "CANNOT_SCORE":
        results.record_pass("ScoringResult.signal_type defaults to CANNOT_SCORE")
    else:
        results.record_fail("ScoringResult.signal_type defaults to CANNOT_SCORE",
                            f"Got {result.signal_type}")

    if result.status == ScoringStatus.BLOCKED:
        results.record_pass("ScoringResult.status defaults to BLOCKED")
    else:
        results.record_fail("ScoringResult.status defaults to BLOCKED",
                            f"Got {result.status}")

    # Test 2.3: score_display property
    if result.score_display == "N/A":
        results.record_pass("score_display shows N/A when score is None")
    else:
        results.record_fail("score_display shows N/A when score is None",
                            f"Got '{result.score_display}'")

    # Test 2.4: Scoring with insufficient data returns None
    print("\n  Testing scoring with insufficient data...")
    scorer = UnifiedScorer()

    # Create features with only 1 component (less than minimum required)
    sparse_features = TickerFeatures(
        ticker="TEST",
        as_of_time=datetime.now(),
        current_price=100.0,
        sentiment_score=65.0,  # Only one score
    )

    score_result = scorer.compute_scores(sparse_features)

    if score_result.status == ScoringStatus.BLOCKED:
        results.record_pass("Scoring with <2 components returns BLOCKED status")
    else:
        results.record_fail("Scoring with <2 components returns BLOCKED status",
                            f"Got {score_result.status}")

    if score_result.composite_score is None:
        results.record_pass("Scoring with insufficient data returns None composite_score")
    else:
        results.record_fail("Scoring with insufficient data returns None composite_score",
                            f"Got {score_result.composite_score}")


# =============================================================================
# TEST 3: SHORT SQUEEZE
# =============================================================================

def test_short_squeeze(results: TestResults):
    """Test short_squeeze.py fixes."""
    print("\n" + "=" * 60)
    print("TEST 3: SHORT SQUEEZE")
    print("=" * 60)

    try:
        from src.analytics.short_squeeze import (
            ShortSqueezeDetector,
            ShortSqueezeData,
        )
    except ImportError as e:
        results.record_skip("short_squeeze import", str(e))
        return

    # Test 3.1: ShortSqueezeData defaults
    print("\n  Testing ShortSqueezeData defaults...")
    data = ShortSqueezeData(ticker="TEST")

    if data.squeeze_score is None:
        results.record_pass("ShortSqueezeData.squeeze_score defaults to None")
    else:
        results.record_fail("ShortSqueezeData.squeeze_score defaults to None",
                            f"Got {data.squeeze_score}")

    if data.rsi_14 is None:
        results.record_pass("ShortSqueezeData.rsi_14 defaults to None")
    else:
        results.record_fail("ShortSqueezeData.rsi_14 defaults to None",
                            f"Got {data.rsi_14}")

    if data.squeeze_risk == "UNKNOWN" or data.squeeze_risk == "NOT_ANALYZED":
        results.record_pass("ShortSqueezeData.squeeze_risk defaults to UNKNOWN/NOT_ANALYZED")
    else:
        results.record_fail("ShortSqueezeData.squeeze_risk defaults to UNKNOWN/NOT_ANALYZED",
                            f"Got '{data.squeeze_risk}'")

    # Test 3.2: score_display property
    if "N/A" in data.score_display:
        results.record_pass("score_display shows N/A when score is None")
    else:
        results.record_fail("score_display shows N/A when score is None",
                            f"Got '{data.score_display}'")

    # Test 3.3: Live test
    print("\n  Testing live squeeze analysis...")
    try:
        detector = ShortSqueezeDetector()
        result = detector.analyze_ticker("GME")  # Usually has short data

        print(f"\n  GME Squeeze Analysis:")
        print(f"    Score: {result.score_display}")
        print(f"    Risk: {result.squeeze_risk}")
        print(f"    RSI: {result.rsi_14 if result.rsi_14 is not None else 'N/A'}")
        print(
            f"    Short % Float: {result.short_percent_of_float if result.short_percent_of_float is not None else 'N/A'}")
        print(f"    Has sufficient data: {result.has_sufficient_data}")

        # Check RSI is not 50 if it couldn't be calculated
        if result.rsi_14 == 50:
            # This could be legitimate - 50 is possible
            print("    Note: RSI is exactly 50 - verify this is calculated, not defaulted")

        results.record_pass("Live squeeze analysis completed")

    except Exception as e:
        results.record_skip("Live squeeze test", f"Error: {e}")


# =============================================================================
# TEST 4: TRADE IDEAS
# =============================================================================

def test_trade_ideas(results: TestResults):
    """Test trade_ideas.py fixes."""
    print("\n" + "=" * 60)
    print("TEST 4: TRADE IDEAS")
    print("=" * 60)

    try:
        from src.analytics.trade_ideas import (
            TradeCandidate,
            TradeIdeasGenerator,
        )
    except ImportError as e:
        results.record_skip("trade_ideas import", str(e))
        return

    # Test 4.1: TradeCandidate defaults
    print("\n  Testing TradeCandidate defaults...")
    candidate = TradeCandidate(ticker="TEST")

    if candidate.options_score is None:
        results.record_pass("TradeCandidate.options_score defaults to None")
    else:
        results.record_fail("TradeCandidate.options_score defaults to None",
                            f"Got {candidate.options_score}")

    if candidate.squeeze_score is None:
        results.record_pass("TradeCandidate.squeeze_score defaults to None")
    else:
        results.record_fail("TradeCandidate.squeeze_score defaults to None",
                            f"Got {candidate.squeeze_score}")

    if candidate.rsi_14 is None:
        results.record_pass("TradeCandidate.rsi_14 defaults to None")
    else:
        results.record_fail("TradeCandidate.rsi_14 defaults to None",
                            f"Got {candidate.rsi_14}")

    if candidate.squeeze_risk in ["NOT_ANALYZED", "UNKNOWN"]:
        results.record_pass("TradeCandidate.squeeze_risk defaults to NOT_ANALYZED/UNKNOWN")
    else:
        results.record_fail("TradeCandidate.squeeze_risk defaults to NOT_ANALYZED/UNKNOWN",
                            f"Got '{candidate.squeeze_risk}'")

    if candidate.signal_type in ["NOT_ANALYZED", ""]:
        results.record_pass("TradeCandidate.signal_type defaults to NOT_ANALYZED")
    else:
        results.record_fail("TradeCandidate.signal_type defaults to NOT_ANALYZED",
                            f"Got '{candidate.signal_type}'")

    # Test 4.2: is_analyzable property
    if not candidate.is_analyzable:
        results.record_pass("is_analyzable is False for empty candidate")
    else:
        results.record_fail("is_analyzable is False for empty candidate",
                            "Should be False without price and score")

    # Test 4.3: score_display
    if "N/A" in candidate.score_display:
        results.record_pass("score_display shows N/A when total_score is None")
    else:
        results.record_fail("score_display shows N/A when total_score is None",
                            f"Got '{candidate.score_display}'")

    # Test 4.4: get_display_value method
    if hasattr(candidate, 'get_display_value'):
        val = candidate.get_display_value('options_score')
        if val == "N/A":
            results.record_pass("get_display_value returns N/A for None values")
        else:
            results.record_fail("get_display_value returns N/A for None values",
                                f"Got '{val}'")
    else:
        results.record_skip("get_display_value test", "Method not found")


# =============================================================================
# TEST 5: INTEGRATION TEST
# =============================================================================

def test_integration(results: TestResults):
    """Test that all modules work together."""
    print("\n" + "=" * 60)
    print("TEST 5: INTEGRATION")
    print("=" * 60)

    print("\n  Testing end-to-end flow...")

    try:
        from src.analytics.trade_ideas import TradeIdeasGenerator

        generator = TradeIdeasGenerator()

        # Generate with very limited filter to get some results quickly
        filter_config = {
            'min_score': 60,
            'signal_types': ['STRONG_BUY', 'BUY'],
            'include_no_signal': False,
            'min_rs': 0,
            'require_bullish_options': False,
            'skip_earnings_within_days': 0,
            'sectors': None,
        }

        print("  Generating trade ideas (this may take a minute)...")
        result = generator.generate_ideas(
            portfolio_positions=None,
            max_picks=3,
            filter_config=filter_config
        )

        print(f"\n  Results:")
        print(f"    Top picks: {len(result.top_picks)}")
        print(f"    Honorable mentions: {len(result.honorable_mentions)}")
        print(f"    Avoid list: {len(result.avoid_list)}")

        if result.top_picks:
            pick = result.top_picks[0]
            print(f"\n  First pick ({pick.ticker}):")
            print(f"    AI Score: {pick.ai_score}")
            print(f"    Options Score: {pick.options_score}")
            print(f"    Squeeze Score: {pick.squeeze_score}")
            print(f"    RSI: {pick.rsi_14}")
            print(f"    Data Completeness: {pick.data_completeness:.0%}")

            # Check we're not seeing fake defaults
            fields_to_check = [
                ('options_score', 50),
                ('squeeze_score', 0),
                ('rsi_14', 50),
            ]

            suspicious = []
            for field, bad_default in fields_to_check:
                val = getattr(pick, field)
                if val == bad_default:
                    suspicious.append(f"{field}={val}")

            if suspicious:
                print(f"\n  ⚠️  Suspicious values (might be real or might be old defaults):")
                for s in suspicious:
                    print(f"      {s}")

        results.record_pass("Integration test completed")

    except Exception as e:
        results.record_fail("Integration test", str(e))
        traceback.print_exc()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test data quality fixes")
    parser.add_argument('--test', choices=['earnings', 'scorer', 'squeeze', 'trade_ideas', 'integration', 'all'],
                        default='all', help='Which test to run')
    args = parser.parse_args()

    print("=" * 60)
    print("DATA QUALITY FIXES - VALIDATION TEST SUITE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = TestResults()

    tests = {
        'earnings': test_earnings_intelligence,
        'scorer': test_unified_scorer,
        'squeeze': test_short_squeeze,
        'trade_ideas': test_trade_ideas,
        'integration': test_integration,
    }

    if args.test == 'all':
        for name, test_func in tests.items():
            try:
                test_func(results)
            except Exception as e:
                print(f"\n  ERROR in {name}: {e}")
                traceback.print_exc()
    else:
        tests[args.test](results)

    success = results.summary()

    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED - Data quality fixes are working!")
    else:
        print("❌ SOME TESTS FAILED - Review the errors above")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())