"""
Debug script to test Options Flow and Squeeze scoring step by step.
Run this from your project root: python debug_universe_scores.py
"""

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from datetime import date

print("=" * 60)
print("STEP 1: Test Database Connection")
print("=" * 60)

try:
    from src.db.connection import get_connection, get_engine
    print("✅ Database connection imported")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM screener_scores")
            count = cur.fetchone()[0]
            print(f"✅ Connected! screener_scores has {count} rows")

            # Check if columns exist
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'screener_scores' 
                AND column_name IN ('options_flow_score', 'short_squeeze_score')
            """)
            cols = [row[0] for row in cur.fetchall()]
            print(f"✅ New columns found: {cols}")

            if len(cols) < 2:
                print("❌ Missing columns! Need to run migration.")
except Exception as e:
    print(f"❌ Database error: {e}")
    exit(1)

print()
print("=" * 60)
print("STEP 2: Test Options Flow Analyzer (single ticker)")
print("=" * 60)

try:
    from src.analytics.options_flow import OptionsFlowAnalyzer

    analyzer = OptionsFlowAnalyzer()
    result = analyzer.analyze_ticker("AAPL")

    print(f"✅ AAPL Options Flow:")
    print(f"   - Sentiment: {result.overall_sentiment}")
    print(f"   - Sentiment Score (raw): {result.sentiment_score}")
    print(f"   - Call Volume: {result.total_call_volume:,}")
    print(f"   - Put Volume: {result.total_put_volume:,}")
    print(f"   - P/C Ratio: {result.put_call_volume_ratio:.2f}")

    # Normalize score
    normalized = (result.sentiment_score + 100) / 2
    print(f"   - Normalized Score (0-100): {normalized:.1f}")

except Exception as e:
    print(f"❌ Options Flow error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("STEP 3: Test Short Squeeze Detector (single ticker)")
print("=" * 60)

try:
    from src.analytics.short_squeeze import ShortSqueezeDetector

    detector = ShortSqueezeDetector()
    result = detector.analyze_ticker("AAPL")

    print(f"✅ AAPL Short Squeeze:")
    print(f"   - Short % Float: {result.short_percent_float:.2f}%" if result.short_percent_float else "   - Short % Float: N/A")
    print(f"   - Squeeze Score: {result.squeeze_score:.1f}")
    print(f"   - Squeeze Risk: {result.squeeze_risk}")

except Exception as e:
    print(f"❌ Short Squeeze error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("STEP 4: Test Universe Scorer (single ticker)")
print("=" * 60)

try:
    from src.analytics.universe_scorer import UniverseScorer, UniverseScores

    scorer = UniverseScorer()
    score = scorer.score_ticker("AAPL")

    print(f"✅ AAPL Universe Score:")
    print(f"   - Options Flow Score: {score.options_flow_score}")
    print(f"   - Options Sentiment: {score.options_sentiment}")
    print(f"   - Squeeze Score: {score.short_squeeze_score}")
    print(f"   - Squeeze Risk: {score.squeeze_risk}")
    print(f"   - Error: {score.error}")

except Exception as e:
    print(f"❌ Universe Scorer error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("STEP 5: Test Database UPDATE (single ticker)")
print("=" * 60)

try:
    # First check current value
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ticker, date, options_flow_score, short_squeeze_score 
                FROM screener_scores 
                WHERE ticker = 'AAPL' 
                ORDER BY date DESC LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                print(f"Before update: {row}")
            else:
                print("❌ No AAPL in screener_scores!")
                exit(1)

    # Now update
    from src.analytics.universe_scorer import UniverseScorer
    scorer = UniverseScorer()
    scores = [scorer.score_ticker("AAPL")]
    updated = scorer.update_screener_scores(scores)
    print(f"Updated {updated} rows")

    # Check after update
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ticker, date, options_flow_score, short_squeeze_score,
                       options_sentiment, squeeze_risk
                FROM screener_scores 
                WHERE ticker = 'AAPL' 
                ORDER BY date DESC LIMIT 1
            """)
            row = cur.fetchone()
            print(f"After update: {row}")

            if row[2] is not None or row[3] is not None:
                print("✅ UPDATE WORKED!")
            else:
                print("❌ UPDATE FAILED - values still NULL")

except Exception as e:
    print(f"❌ Update error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("STEP 6: Test SQL Query (what app.py uses)")
print("=" * 60)

try:
    query = """
        SELECT DISTINCT ON (ticker)
            ticker, date, sentiment_score, options_flow_score, short_squeeze_score,
            options_sentiment, squeeze_risk
        FROM screener_scores
        ORDER BY ticker, date DESC, created_at DESC
        LIMIT 10
    """
    df = pd.read_sql(query, get_engine())
    print(f"Query returned {len(df)} rows:")
    print(df.to_string())

    # Check if any have values
    has_optflow = df['options_flow_score'].notna().sum()
    has_squeeze = df['short_squeeze_score'].notna().sum()
    print(f"\nRows with OptFlow: {has_optflow}")
    print(f"Rows with Squeeze: {has_squeeze}")

except Exception as e:
    print(f"❌ Query error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)