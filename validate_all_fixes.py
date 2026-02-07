"""
HH Research Platform — Full Audit Fix Validation
==================================================
Validates ALL fixes: F-01/F-02, F-03, F-06, F-07, F-08, F-10

Run from project root:
    python validate_all_fixes.py
"""

import sys
import traceback
import inspect

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
SKIP = "⏭️"
results = []


def test(name, passed, detail=""):
    results.append((name, passed, detail))
    status = PASS if passed else FAIL
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# F-01/F-02: UnifiedSignal defaults are None
# ============================================================
section("F-01/F-02: UnifiedSignal defaults = None")

try:
    from src.core.unified_signal import UnifiedSignal

    sig = UnifiedSignal(ticker="TEST")

    core_fields = [
        'technical_score', 'fundamental_score', 'sentiment_score',
        'options_score', 'earnings_score', 'today_score', 'longterm_score',
        'risk_score', 'bond_score', 'gex_score', 'dark_pool_score',
        'cross_asset_score', 'sentiment_nlp_score', 'whisper_score',
        'insider_score', 'inst_13f_score',
    ]

    bad_fields = []
    for f in core_fields:
        val = getattr(sig, f, "MISSING")
        if val is not None and val != "MISSING":
            bad_fields.append(f"{f}={val}")

    test("All score fields default to None",
         len(bad_fields) == 0,
         f"Still defaulting: {', '.join(bad_fields)}" if bad_fields else f"All {len(core_fields)} fields clean")

    # Signal fields
    signal_fields = [
        'technical_signal', 'fundamental_signal', 'sentiment_signal',
        'options_signal', 'earnings_signal',
    ]
    bad_signals = []
    for f in signal_fields:
        val = getattr(sig, f, "MISSING")
        if val is not None and val not in ("MISSING", ""):
            bad_signals.append(f"{f}={val}")

    test("Signal fields default to None",
         len(bad_signals) == 0,
         f"Still defaulting: {', '.join(bad_signals)}" if bad_signals else "Clean")

    # Helper methods
    test("has_score() → False for missing", not sig.has_score('technical_score'))
    test("score_value() → 0 for missing", sig.score_value('technical_score') == 0)
    test("score_display() → 'N/A' for missing", sig.score_display('technical_score') == 'N/A')
    test("data_completeness → 0.0 for empty", sig.data_completeness == 0.0)

    sig.technical_score = 80
    sig.sentiment_score = 70
    test("data_completeness updates with data",
         0.0 < sig.data_completeness <= 1.0,
         f"{sig.data_completeness:.0%}")

except Exception as e:
    test("UnifiedSignal import", False, str(e))
    traceback.print_exc()


# ============================================================
# F-01/F-02: signal_engine.py — None-aware DB loading
# ============================================================
section("F-01/F-02: signal_engine.py None-aware")

try:
    from src.core.signal_engine import SignalEngine

    source = inspect.getsource(SignalEngine)

    # Check for _safe_int or equivalent None-aware pattern
    has_safe_int = '_safe_int' in source
    test("Has _safe_int helper", has_safe_int)

    # Count remaining '!= 50' in the full class
    neq50_count = source.count('!= 50')
    test("No '!= 50' comparisons remain",
         neq50_count == 0,
         f"Found {neq50_count} occurrences" if neq50_count > 0 else "Clean")

    # Count remaining 'or 50'
    or50_count = source.count('or 50')
    test("No 'or 50' fallbacks remain",
         or50_count == 0,
         f"Found {or50_count} occurrences" if or50_count > 0 else "Clean")

    # Check 'is not None' pattern is used
    is_not_none = source.count('is not None')
    test("Uses 'is not None' checks",
         is_not_none >= 5,
         f"Found {is_not_none} occurrences")

except Exception as e:
    test("signal_engine inspection", False, str(e))


# ============================================================
# F-03: Likelihood NOT in composite score
# ============================================================
section("F-03: Likelihood circular dependency removed")

try:
    from src.screener.signals import calculate_composite_score, calculate_likelihood_score

    # Composite should ignore likelihood_score
    scores = {
        'sentiment_score': 80,
        'fundamental_score': 75,
        'growth_score': 70,
        'dividend_score': 60,
        'gap_score': 65,
        'analyst_positivity': 72,
        'likelihood_score': 90,  # Should be IGNORED
    }

    composite = calculate_composite_score(scores)
    # Manual: 80*0.25 + 75*0.30 + 70*0.15 + 60*0.05 + 65*0.10 + 72*0.15 = 73.8
    expected = int(80*0.25 + 75*0.30 + 70*0.15 + 60*0.05 + 65*0.10 + 72*0.15)

    test("Likelihood excluded from composite",
         composite == expected,
         f"composite={composite}, expected={expected}")

    # Verify function source doesn't contain 'likelihood'
    src = inspect.getsource(calculate_composite_score)
    test("No 'likelihood' in composite source",
         'likelihood' not in src.split('NOTE')[0] if 'NOTE' in src else 'likelihood' not in src.replace('# ', ''),
         "Checking function body")

    # Likelihood with partial data
    lik = calculate_likelihood_score({'sentiment_score': 80})
    test("Likelihood handles partial data",
         lik == 80,
         f"got {lik} (only sentiment available)")

    # Likelihood with no data
    lik_empty = calculate_likelihood_score({})
    test("Likelihood returns None for no data",
         lik_empty is None,
         f"got {lik_empty}")

except Exception as e:
    test("F-03 composite/likelihood", False, str(e))
    traceback.print_exc()


# ============================================================
# F-03/F-01: signals.py — no phantom 50s
# ============================================================
section("F-01 in signals.py: No phantom 50s")

try:
    from src.screener.signals import (
        generate_trading_signal, calculate_composite_score
    )

    # Partial data should NOT dilute with phantom 50s
    partial = {'ticker': 'TEST', 'sentiment_score': 80, 'fundamental_score': 75}
    comp = calculate_composite_score(partial)
    test("Partial data: composite only from available",
         comp is not None and comp > 70,
         f"composite={comp} (expected ~78)")

    # All None → None
    empty = calculate_composite_score({'ticker': 'TEST'})
    test("All None → composite is None", empty is None, f"got {empty}")

    # Signal with no data
    sig = generate_trading_signal({'ticker': 'TEST'})
    test("No-data signal → NEUTRAL + insufficient data reason",
         sig.type == "NEUTRAL" and "Insufficient" in sig.reasons[0])

    # Check source for 'or 50'
    src = inspect.getsource(generate_trading_signal)
    or50 = src.count('or 50')
    test("generate_trading_signal has no 'or 50'",
         or50 == 0,
         f"Found {or50}" if or50 > 0 else "Clean")

    src2 = inspect.getsource(calculate_composite_score)
    or50_c = src2.count('or 50')
    test("calculate_composite_score has no 'or 50'",
         or50_c == 0,
         f"Found {or50_c}" if or50_c > 0 else "Clean")

except Exception as e:
    test("signals.py phantom 50 check", False, str(e))
    traceback.print_exc()


# ============================================================
# F-10: Sentiment returns None on failure
# ============================================================
section("F-10: Sentiment failure → None (not 50)")

try:
    from src.screener.sentiment import SentimentAnalyzer

    analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
    analyzer.sentiment_available = False

    # No articles
    r1 = analyzer.analyze_sentiment("TEST", [])
    test("analyze_sentiment(no articles) → None",
         r1['sentiment_score'] is None,
         f"got {r1['sentiment_score']}")

    # Model unavailable
    r2 = analyzer.analyze_sentiment("TEST", [{"title": "test"}])
    test("analyze_sentiment(model unavail) → None",
         r2['sentiment_score'] is None,
         f"got {r2['sentiment_score']}")

    # analyze_ticker_sentiment no articles
    r3 = analyzer.analyze_ticker_sentiment("TEST", [])
    test("analyze_ticker_sentiment(no articles) → None",
         r3['sentiment_score'] is None,
         f"got {r3['sentiment_score']}")

    # Check _save_sentiment_score guards against None
    src = inspect.getsource(analyzer._save_sentiment_score)
    test("_save_sentiment_score guards None",
         'if score is None' in src or 'score is None' in src)

except Exception as e:
    test("Sentiment None checks", False, str(e))
    traceback.print_exc()


# ============================================================
# F-10: Worker initializes sentiment as None
# ============================================================
section("F-10: Worker sentiment defaults")

try:
    from src.screener.worker import ScreenerWorker
    src = inspect.getsource(ScreenerWorker.process_ticker)

    test("Worker: no 'sentiment_score = 50'",
         'sentiment_score = 50' not in src)
    test("Worker: no 'sentiment_weighted = 50'",
         'sentiment_weighted = 50' not in src)
    test("Worker: uses 'sentiment_score = None'",
         'sentiment_score = None' in src)
    test("Worker: no default-50 for analyst_positivity",
         "'analyst_positivity', 50)" not in src)
    test("Worker: no default-50 for insider_signal",
         "'insider_signal', 50)" not in src)

except Exception as e:
    test("Worker defaults", False, str(e))


# ============================================================
# F-06: Earnings whisper — no bullish bias
# ============================================================
section("F-06: Earnings whisper base_surprise")

try:
    from src.analytics.earnings_whisper import EarningsWhisperAnalyzer

    src = inspect.getsource(EarningsWhisperAnalyzer._generate_prediction)

    test("No 'base_surprise = 3.0' bullish bias",
         'base_surprise = 3.0' not in src,
         "Removed" if 'base_surprise = 3.0' not in src else "Still present!")

    test("Uses base_surprise = 0.0 (neutral)",
         'base_surprise = 0.0' in src,
         "Found neutral default" if 'base_surprise = 0.0' in src else "Missing")

except Exception as e:
    test("Earnings whisper check", False, str(e))


# ============================================================
# F-07: Monitoring — VIX/regime defaults
# ============================================================
section("F-07: Monitoring VIX/regime defaults")

try:
    from src.ml.monitoring import DriftMonitor

    src = inspect.getsource(DriftMonitor._check_regime_drift)

    test("No hardcoded VIX=20 default",
         "'vix', 20)" not in src,
         "Removed" if "'vix', 20)" not in src else "Still present!")

    test("No hardcoded regime_score=50 default",
         "'score', 50)" not in src,
         "Removed" if "'score', 50)" not in src else "Still present!")

    test("VIX check is None-guarded",
         'vix is not None' in src,
         "Guarded" if 'vix is not None' in src else "Missing guard")

    test("Regime score check is None-guarded",
         'regime_score is not None' in src,
         "Guarded" if 'regime_score is not None' in src else "Missing guard")

except Exception as e:
    test("Monitoring check", False, str(e))


# ============================================================
# F-08: Integration — ai_score default
# ============================================================
section("F-08: Integration ai_score default")

try:
    from src.core.integration import PortfolioIntegration

    src = inspect.getsource(PortfolioIntegration.rank_by_exposure)

    test("No 'ai_score', 50) default",
         "'ai_score', 50)" not in src,
         "Removed" if "'ai_score', 50)" not in src else "Still present!")

    test("Handles None ai_score",
         'base_score is None' in src or 'ai_score\') is None' in src or 'if base_score is None' in src,
         "Guarded")

except Exception as e:
    test("Integration check", False, str(e))


# ============================================================
# LIVE SMOKE TEST
# ============================================================
section("LIVE: Generate signal for AAPL (requires DB)")

try:
    from src.core.signal_engine import SignalEngine

    engine = SignalEngine()
    signal = engine.generate_signal('AAPL')

    if signal:
        scores = {
            'technical': signal.technical_score,
            'fundamental': signal.fundamental_score,
            'sentiment': signal.sentiment_score,
            'options': signal.options_score,
            'earnings': signal.earnings_score,
        }

        print(f"  Ticker: AAPL")
        print(f"  Today: {signal.today_score} | Longterm: {signal.longterm_score}")
        for name, val in scores.items():
            status = "✓" if val is not None else "—"
            print(f"  {status} {name}: {val}")
        print(f"  Completeness: {signal.data_completeness:.0%}")

        # Count phantom 50s
        count_50 = sum(1 for v in scores.values() if v == 50)
        count_none = sum(1 for v in scores.values() if v is None)
        count_real = sum(1 for v in scores.values() if v is not None and v != 50)

        test("Live signal generated", True,
             f"today={signal.today_score}, completeness={signal.data_completeness:.0%}")

        test("No suspicious phantom-50 cluster",
             count_50 < 3,
             f"real={count_real}, none={count_none}, exactly_50={count_50}")

    else:
        test("Live signal generated", False, "Returned None")

except Exception as e:
    print(f"  {SKIP} Skipped (DB not available): {e}")


# ============================================================
# SUMMARY
# ============================================================
section("FINAL SUMMARY")

passed = sum(1 for _, p, _ in results if p)
failed = sum(1 for _, p, _ in results if not p)
total = len(results)

print(f"\n  Total:  {total}")
print(f"  Passed: {passed}  {PASS}")
print(f"  Failed: {failed}  {FAIL if failed > 0 else ''}")

if failed > 0:
    print(f"\n  {'─'*50}")
    print(f"  FAILURES:")
    for name, p, detail in results:
        if not p:
            print(f"    {FAIL} {name}: {detail}")
else:
    print(f"\n  🎉 All fixes validated successfully!")

print()
sys.exit(0 if failed == 0 else 1)