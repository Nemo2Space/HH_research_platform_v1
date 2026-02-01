"""
Earnings Diagnostic Tool

Run this script to diagnose why earnings data isn't showing:

    python -m src.analytics.earnings_diagnostic NKE

This will check:
1. earnings_calendar table - do we know about the earnings?
2. news_articles table - do we have earnings news?
3. yfinance data - can we fetch fresh earnings?
4. sentiment data - what sentiment score is stored?
"""

import sys
import pandas as pd
from datetime import datetime, date, timedelta


def diagnose_ticker(ticker: str):
    """Full diagnostic for a ticker's earnings data."""

    print(f"\n{'=' * 60}")
    print(f"EARNINGS DIAGNOSTIC: {ticker}")
    print(f"{'=' * 60}\n")

    issues = []

    # 1. Check Database Connection
    print("1. DATABASE CONNECTION")
    print("-" * 40)
    try:
        from src.db.connection import get_engine
        engine = get_engine()
        print("   ✅ Database connected")
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        issues.append("Cannot connect to database")
        return issues

    # 2. Check Earnings Calendar
    print("\n2. EARNINGS CALENDAR")
    print("-" * 40)
    try:
        df = pd.read_sql(f"""
            SELECT * FROM earnings_calendar 
            WHERE ticker = '{ticker}'
            ORDER BY earnings_date DESC
            LIMIT 5
        """, engine)

        if df.empty:
            print(f"   ❌ No earnings dates found for {ticker}")
            issues.append("No earnings dates in earnings_calendar table")
        else:
            print(f"   ✅ Found {len(df)} earnings dates:")
            for _, row in df.iterrows():
                ed = row.get('earnings_date')
                print(f"      - {ed}")

            # Check if recent
            latest = pd.to_datetime(df.iloc[0]['earnings_date']).date()
            days_ago = (date.today() - latest).days

            if days_ago <= 7 and days_ago >= 0:
                print(f"   ✅ Recent earnings: {latest} ({days_ago} days ago)")
            elif days_ago < 0:
                print(f"   📅 Upcoming earnings: {latest} (in {abs(days_ago)} days)")
            else:
                print(f"   ⚠️ Last earnings was {days_ago} days ago - may be stale")
                issues.append(f"Earnings calendar may be outdated (last: {days_ago} days ago)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        issues.append(f"Cannot query earnings_calendar: {e}")

    # 3. Check News Articles
    print("\n3. NEWS ARTICLES")
    print("-" * 40)
    try:
        # Recent articles
        df = pd.read_sql(f"""
            SELECT headline, source, published_at, ai_sentiment_fast, created_at
            FROM news_articles 
            WHERE ticker = '{ticker}'
            ORDER BY COALESCE(published_at, created_at) DESC
            LIMIT 10
        """, engine)

        if df.empty:
            print(f"   ❌ No news articles for {ticker}")
            issues.append("No news articles in database")
        else:
            print(f"   ✅ Found {len(df)} recent articles:")

            # Check for earnings-related
            earnings_keywords = ['earning', 'quarter', 'revenue', 'eps', 'profit', 'result', 'beat', 'miss', 'guidance']
            earnings_articles = []

            for _, row in df.iterrows():
                headline = str(row.get('headline', '')).lower()
                is_earnings = any(kw in headline for kw in earnings_keywords)
                emoji = "📊" if is_earnings else "📰"

                if is_earnings:
                    earnings_articles.append(row)

                sent = row.get('ai_sentiment_fast')
                sent_str = f"({sent:.0f})" if pd.notna(sent) else "(no score)"

                pub_date = row.get('published_at') or row.get('created_at')
                date_str = str(pub_date)[:10] if pub_date else "?"

                print(f"      {emoji} {date_str} {sent_str}: {str(row['headline'])[:50]}...")

            if not earnings_articles:
                print(f"   ⚠️ No earnings-specific news found!")
                issues.append("No earnings-related news in recent articles")
            else:
                print(f"   ✅ {len(earnings_articles)} earnings-related articles")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        issues.append(f"Cannot query news_articles: {e}")

    # 4. Check Sentiment Scores
    print("\n4. SENTIMENT SCORES")
    print("-" * 40)
    try:
        df = pd.read_sql(f"""
            SELECT ticker, sentiment_score, sentiment_signal, article_count, score_date
            FROM screener_scores 
            WHERE ticker = '{ticker}'
            ORDER BY score_date DESC
            LIMIT 1
        """, engine)

        if df.empty:
            print(f"   ❌ No sentiment score for {ticker}")
            issues.append("No sentiment score in screener_scores")
        else:
            row = df.iloc[0]
            score = row.get('sentiment_score')
            signal = row.get('sentiment_signal')
            article_count = row.get('article_count')
            score_date = row.get('score_date')

            print(f"   Score: {score}")
            print(f"   Signal: {signal}")
            print(f"   Articles: {article_count}")
            print(f"   Date: {score_date}")

            # Is it stale?
            if score_date:
                age = (date.today() - pd.to_datetime(score_date).date()).days
                if age > 1:
                    print(f"   ⚠️ Sentiment is {age} days old!")
                    issues.append(f"Sentiment score is {age} days stale")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # 5. Check yfinance
    print("\n5. YFINANCE DATA (Live)")
    print("-" * 40)
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        info = stock.info

        # Price
        current = info.get('currentPrice') or info.get('regularMarketPrice')
        prev_close = info.get('previousClose')
        pre_market = info.get('preMarketPrice')
        post_market = info.get('postMarketPrice')

        print(f"   Current Price: ${current}")
        print(f"   Prev Close: ${prev_close}")
        if pre_market:
            pct = ((pre_market - prev_close) / prev_close) * 100
            print(f"   Pre-Market: ${pre_market} ({pct:+.2f}%)")
        if post_market:
            pct = ((post_market - prev_close) / prev_close) * 100
            print(f"   After-Hours: ${post_market} ({pct:+.2f}%)")

        # Earnings dates
        try:
            earnings_dates = stock.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                print(f"\n   Earnings Dates from yfinance:")
                for idx in earnings_dates.index[:5]:
                    row = earnings_dates.loc[idx]
                    eps_est = row.get('EPS Estimate')
                    eps_act = row.get('Reported EPS')
                    surprise = row.get('Surprise(%)')
                    print(f"      {idx.date()}: Est ${eps_est}, Actual ${eps_act}, Surprise {surprise}%")
        except Exception as e:
            print(f"   ⚠️ earnings_dates error: {e}")

        # Earnings history
        try:
            hist = stock.earnings_history
            if hist is not None and not hist.empty:
                print(f"\n   Earnings History:")
                print(hist.head())
        except Exception as e:
            print(f"   ⚠️ earnings_history error: {e}")

    except Exception as e:
        print(f"   ❌ yfinance error: {e}")
        issues.append(f"Cannot fetch yfinance data: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    if not issues:
        print("✅ All checks passed! Data should be available.")
    else:
        print(f"❌ Found {len(issues)} issues:\n")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

        print("\n📋 RECOMMENDED ACTIONS:")

        if any("earnings_calendar" in i for i in issues):
            print("   - Update earnings calendar: Run earnings calendar sync job")
            print("     Or manually INSERT into earnings_calendar")

        if any("news" in i.lower() for i in issues):
            print("   - Fetch fresh news: Click 'Refresh Data' for this ticker")
            print("     Or run: nc = NewsCollector(); nc.collect_and_save('NKE', force_refresh=True)")

        if any("sentiment" in i.lower() or "stale" in i.lower() for i in issues):
            print("   - Re-run sentiment analysis after fetching news")

    return issues


def update_earnings_calendar(ticker: str, earnings_date: str):
    """Manually add/update earnings date."""
    from src.db.connection import get_connection

    query = """
            INSERT INTO earnings_calendar (ticker, earnings_date, updated_at)
            VALUES (%(ticker)s, %(earnings_date)s, NOW()) ON CONFLICT (ticker, earnings_date) 
        DO \
            UPDATE SET updated_at = NOW() \
            """

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, {'ticker': ticker, 'earnings_date': earnings_date})
            conn.commit()
        print(f"✅ Added earnings date {earnings_date} for {ticker}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.analytics.earnings_diagnostic TICKER")
        print("       python -m src.analytics.earnings_diagnostic NKE")
        print("       python -m src.analytics.earnings_diagnostic NKE --add-date 2024-12-19")
        sys.exit(1)

    ticker = sys.argv[1].upper()

    if len(sys.argv) >= 4 and sys.argv[2] == "--add-date":
        update_earnings_calendar(ticker, sys.argv[3])
    else:
        diagnose_ticker(ticker)