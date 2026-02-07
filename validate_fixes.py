"""
HH Research Platform — Post-Fix Validation Script
===================================================
Validates F-01/F-02 (default-50 elimination), F-03 (likelihood circular dep),
and F-10 (sentiment 50 written to DB on failure).

Run from project root:
    python validate_fixes.py
"""

import sys
import traceback

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
results = []


def test(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, passed, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# TEST 1: UnifiedSignal defaults are None, not 50
# ============================================================
section("TEST 1: UnifiedSignal dataclass defaults")

try:
    from src.core.unified_signal import UnifiedSignal

    sig = UnifiedSignal(ticker="TEST")

    score_fields = [
        'technical_score', 'fundamental_score', 'sentiment_score',
        'options_score', 'earnings_score', 'today_score', 'longterm_score',
        'risk_score',
    ]

    all_none = True
    for field in score_fields:
        val = getattr(sig, field, "MISSING")
        if val is not None and val != "MISSING":
            test(f"{field} default", False, f"Expected None, got {val}")
            all_none = False

    if all_none:
        test("All score fields default to None", True)

    # Test helper methods
    test("has_score() returns False for missing",
         not sig.has_score('technical_score'))
    test("score_value() returns 0 for missing",
         sig.score_value('technical_score') == 0)
    test("score_display() returns 'N/A' for missing",
         sig.score_display('technical_score') == 'N/A')
    test("data_completeness is 0.0 for empty signal",
         sig.data_completeness == 0.0)

    # Test with real data
    sig.technical_score = 75
    sig.sentiment_score = 80
    test("has_score() returns True when set",
         sig.has_score('technical_score'))
    test("score_value() returns real value",
         sig.score_value('technical_score') == 75)
    test("data_completeness reflects available data",
         0.0 < sig.data_completeness < 1.0,
         f"completeness={sig.data_completeness:.0%}")

except Exception as e:
    test("UnifiedSignal import/test", False, str(e))
    traceback.print_exc()


# ============================================================
# TEST 2: Sentiment returns None on failure, not 50
# ============================================================
section("TEST 2: Sentiment failure returns None")

try:
    from src.screener.sentiment import SentimentAnalyzer

    # Create analyzer (may fail if no config — that's OK)
    try:
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer.sentiment_available = False

        # Test: no articles → should return None
        result_empty = analyzer.analyze_sentiment("TEST", [])
        test("No articles → sentiment_score is None",
             result_empty['sentiment_score'] is None,
             f"got {result_empty['sentiment_score']}")

        # Test: model unavailable → should return None
        result_unavail = analyzer.analyze_sentiment("TEST", [{"title": "test"}])
        test("Model unavailable → sentiment_score is None",
             result_unavail['sentiment_score'] is None,
             f"got {result_unavail['sentiment_score']}")

    except Exception as e:
        test("SentimentAnalyzer failure paths", False, str(e))

except ImportError as e:
    test("SentimentAnalyzer import", False, str(e))


# ============================================================
# TEST 3: signals.py — no phantom 50s in composite/likelihood
# ============================================================
section("TEST 3: Signal generation (no phantom 50s)")

try:
    from src.screener.signals import (
        generate_trading_signal,
        calculate_composite_score,
        calculate_likelihood_score,
    )

    # Test 3a: Composite with partial data — should NOT dilute with 50
    scores_partial = {
        'ticker': 'TEST',
        'sentiment_score': 80,
        'fundamental_score': 75,
        # growth, dividend, gap, analyst all missing
    }
    composite = calculate_composite_score(scores_partial)
    test("Partial data composite uses only available scores",
         composite is not None and composite > 70,
         f"composite={composite} (expected ~78, NOT ~55 from phantom 50s)")

    # Test 3b: All-None → should return None
    scores_empty = {'ticker': 'TEST'}
    composite_empty = calculate_composite_score(scores_empty)
    test("All-None scores → composite is None",
         composite_empty is None,
         f"got {composite_empty}")

    # Test 3c: Likelihood with partial data
    scores_like = {
        'sentiment_score': 80,
        # fundamental and analyst missing
    }
    likelihood = calculate_likelihood_score(scores_like)
    test("Likelihood with only sentiment → uses only that component",
         likelihood is not None and likelihood == 80,
         f"likelihood={likelihood} (expected 80, NOT ~55)")

    # Test 3d: F-03 — likelihood NOT in composite weights
    scores_full = {
        'ticker': 'TEST',
        'sentiment_score': 80,
        'fundamental_score': 75,
        'growth_score': 70,
        'dividend_score': 60,
        'gap_score': 65,
        'analyst_positivity': 72,
        'likelihood_score': 90,  # This should be IGNORED by composite
    }
    composite_full = calculate_composite_score(scores_full)
    # Recalculate manually without likelihood:
    # 80*0.25 + 75*0.30 + 70*0.15 + 60*0.05 + 65*0.10 + 72*0.15 = 73.8
    expected = int(80*0.25 + 75*0.30 + 70*0.15 + 60*0.05 + 65*0.10 + 72*0.15)
    test("F-03: likelihood_score excluded from composite",
         composite_full == expected,
         f"composite={composite_full}, expected={expected} (likelihood=90 should be ignored)")

    # Test 3e: Signal generation with no data
    signal_empty = generate_trading_signal({'ticker': 'TEST'})
    test("No-data signal returns NEUTRAL with reason",
         signal_empty.type == "NEUTRAL" and "Insufficient" in signal_empty.reasons[0],
         f"type={signal_empty.type}, reason={signal_empty.reasons}")

    # Test 3f: Signal generation with real data doesn't crash
    signal_real = generate_trading_signal(scores_full)
    test("Real data signal generates without error",
         signal_real.type in ['STRONG BUY', 'BUY', 'WEAK BUY', 'NEUTRAL+',
                               'NEUTRAL', 'NEUTRAL-', 'WEAK SELL', 'SELL',
                               'STRONG SELL', 'INCOME BUY', 'GROWTH BUY'],
         f"type={signal_real.type}")

except Exception as e:
    test("Signal generation tests", False, str(e))
    traceback.print_exc()


# ============================================================
# TEST 4: signal_engine.py — DB loading preserves None
# ============================================================
section("TEST 4: SignalEngine None-awareness")

try:
    from src.core.signal_engine import SignalEngine

    # Check _safe_int helper exists and works
    engine = SignalEngine.__new__(SignalEngine)

    # Look for _safe_int in the module
    import inspect
    source = inspect.getsource(type(engine).generate_signal)

    has_safe_int = '_safe_int' in source or 'is not None' in source
    test("signal_engine uses None-aware loading (not 'or 50')",
         has_safe_int,
         "Found None-aware pattern" if has_safe_int else "Still using 'or 50'")

    # Check that != 50 patterns are gone
    has_neq_50 = '!= 50' in source
    no_neq_50 = not has_neq_50
    test("signal_engine removed '!= 50' comparisons",
         no_neq_50,
         "Clean" if no_neq_50 else "Still has '!= 50' — may cause issues")

except Exception as e:
    # If we can't inspect, just check import works
    test("SignalEngine import", False, str(e))


# ============================================================
# TEST 5: worker.py — no hardcoded 50 defaults
# ============================================================
section("TEST 5: Worker score defaults")

try:
    import inspect
    from src.screener.worker import ScreenerWorker

    source = inspect.getsource(ScreenerWorker.process_ticker)

    # Check sentiment init
    has_sent_50 = "sentiment_score = 50" in source
    test("Worker: sentiment_score not initialized to 50",
         not has_sent_50,
         "Clean" if not has_sent_50 else "Still has 'sentiment_score = 50'")

    # Check analyst_positivity default
    has_analyst_50 = "'analyst_positivity', 50)" in source
    test("Worker: analyst_positivity not defaulting to 50",
         not has_analyst_50,
         "Clean" if not has_analyst_50 else "Still has default 50")

except Exception as e:
    test("Worker inspection", False, str(e))


# ============================================================
# TEST 6: Live smoke test (optional — needs DB)
# ============================================================
section("TEST 6: Live smoke test (requires DB)")

try:
    from src.core.signal_engine import SignalEngine

    engine = SignalEngine()
    signal = engine.generate_signal('AAPL')

    if signal:
        print(f"  Ticker: AAPL")
        print(f"  Today score: {signal.today_score}")
        print(f"  Technical: {signal.technical_score}")
        print(f"  Fundamental: {signal.fundamental_score}")
        print(f"  Sentiment: {signal.sentiment_score}")
        print(f"  Options: {signal.options_score}")
        print(f"  Completeness: {signal.data_completeness:.0%}")

        # Check no phantom 50s
        scores = [signal.technical_score, signal.fundamental_score,
                  signal.sentiment_score, signal.options_score]
        count_50 = sum(1 for s in scores if s == 50)
        count_none = sum(1 for s in scores if s is None)

        test("Live signal generated successfully", True,
             f"today={signal.today_score}, completeness={signal.data_completeness:.0%}")

        if count_50 >= 3:
            test("Suspiciously many 50s in live data", False,
                 f"{count_50}/4 scores are exactly 50 — phantom 50s may remain")
        else:
            test("No suspicious phantom 50 pattern", True,
                 f"50s={count_50}, Nones={count_none}")
    else:
        test("Live signal generation", False, "Returned None/empty")

except Exception as e:
    print(f"  {WARN} Skipped (DB not available): {e}")


# ============================================================
# SUMMARY
# ============================================================
section("SUMMARY")

passed = sum(1 for _, p, _ in results if p)
failed = sum(1 for _, p, _ in results if not p)
total = len(results)

print(f"\n  Passed: {passed}/{total}")
print(f"  Failed: {failed}/{total}")

if failed > 0:
    print(f"\n  Failed tests:")
    for name, p, detail in results:
        if not p:
            print(f"    {FAIL} {name}: {detail}")

print()
sys.exit(0 if failed == 0 else 1)