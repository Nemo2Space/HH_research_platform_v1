"""End-to-end test - skip options (Yahoo rate-limited from testing)."""
import time, sys
from datetime import date

total_start = time.time()
ticker = sys.argv[1] if len(sys.argv) > 1 else 'VRTX'
today = date.today()
print(f"\n  FULL PIPELINE TEST: {ticker} (options skipped - Yahoo rate-limited)\n")

# STEP 1: News
print("STEP 1: News Collection...")
t1 = time.time()
from src.data.news import NewsCollector
nc = NewsCollector()
result = nc.collect_and_save(ticker, days_back=7, force_refresh=True)
articles = result.get('articles', [])
step1 = time.time() - t1
print(f"  -> {step1:.1f}s | {len(articles)} articles\n")

# STEP 2: Sentiment
print("STEP 2: Sentiment Analysis...")
t2 = time.time()
sentiment = None
if articles:
    from src.screener.sentiment import SentimentAnalyzer
    sa = SentimentAnalyzer()
    sentiment_data = sa.analyze_ticker_sentiment(ticker, articles)
    sentiment = sentiment_data.get('sentiment_score')
step2 = time.time() - t2
print(f"  -> {step2:.1f}s | sentiment={sentiment}\n")

# STEP 3: Options (known: 1.5s, skipping due to rate limit)
print("STEP 3: Options Flow... SKIPPED (proven 1.5s)\n")

# STEP 3c: Technical
print("STEP 3c: Technical Analysis...")
t3c = time.time()
technical = None
try:
    from src.tabs.signals_tab.shared import TechnicalAnalyzer
    ta = TechnicalAnalyzer()
    tech_result = ta.analyze_ticker(ticker)
    if tech_result:
        technical = tech_result.get('technical_score', tech_result.get('score'))
except Exception as e:
    print(f"  Skipped: {e}")
step3c = time.time() - t3c
print(f"  -> {step3c:.1f}s | technical={technical}\n")

# STEP 6: Earnings (cache check)
print("STEP 6: Earnings Calendar...")
t6 = time.time()
try:
    from src.db.connection import get_connection
    import pandas as pd
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT earnings_date, updated_at FROM earnings_calendar
                WHERE ticker = %s AND earnings_date >= CURRENT_DATE
                AND updated_at >= NOW() - INTERVAL '3 days' LIMIT 1
            """, (ticker,))
            row = cur.fetchone()
            if row:
                print(f"  Cache HIT: {row[0]}")
            else:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                ed = stock.earnings_dates
                print(f"  Fetched from yfinance")
except Exception as e:
    print(f"  Error: {e}")
step6 = time.time() - t6
print(f"  -> {step6:.1f}s\n")

# SUMMARY
total = time.time() - total_start
options_est = 1.5
print(f"  Step 1 - News:       {step1:5.1f}s")
print(f"  Step 2 - Sentiment:  {step2:5.1f}s")
print(f"  Step 3 - Options:    {options_est:5.1f}s (proven earlier)")
print(f"  Step 3c- Technical:  {step3c:5.1f}s")
print(f"  Step 6 - Earnings:   {step6:5.1f}s")
print(f"  ──────────────────────────────")
print(f"  TOTAL (estimated):   {total + options_est:5.1f}s")
