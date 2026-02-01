"""
Earnings Data Fixer

Properly fetches and stores earnings data for a ticker.

Usage:
    python -m src.analytics.earnings_fixer NKE

This will:
1. Add earnings date to calendar (if missing)
2. Fetch earnings-specific news
3. Re-run sentiment analysis
4. Update screener scores
"""

import sys
import pandas as pd
from datetime import datetime, date, timedelta


def fix_earnings_data(ticker: str, earnings_date_str: str = None):
    """Completely refresh earnings data for a ticker."""

    print(f"\n{'='*60}")
    print(f"FIXING EARNINGS DATA: {ticker}")
    print(f"{'='*60}\n")

    # 1. Get/Set Earnings Date
    print("Step 1: Earnings Date")
    print("-" * 40)

    earnings_date = None

    if earnings_date_str:
        earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
        print(f"   Using provided date: {earnings_date}")
    else:
        # Try to get from yfinance
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)

            # Try earnings_dates
            ed = stock.earnings_dates
            if ed is not None and not ed.empty:
                # Find most recent past earnings
                for idx in ed.index:
                    d = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                    if d <= date.today():
                        earnings_date = d
                        print(f"   Found from yfinance: {earnings_date}")
                        break
        except Exception as e:
            print(f"   ⚠️ yfinance error: {e}")

    if not earnings_date:
        print("   ❌ Could not determine earnings date!")
        print("   Please provide it: python -m src.analytics.earnings_fixer NKE 2024-12-19")
        return

    # Insert into database
    try:
        from src.db.connection import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO earnings_calendar (ticker, earnings_date, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (ticker, earnings_date) DO UPDATE SET updated_at = NOW()
                """, (ticker, earnings_date))
            conn.commit()
        print(f"   ✅ Saved earnings date: {earnings_date}")
    except Exception as e:
        print(f"   ⚠️ DB error (table may not exist): {e}")

    # 2. Fetch Earnings News
    print("\nStep 2: Fetching Earnings News")
    print("-" * 40)

    try:
        from src.data.news import NewsCollector

        nc = NewsCollector()

        # Get company name
        company_name = ticker
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            company_name = info.get('shortName', info.get('longName', ticker))
        except:
            pass

        all_articles = []

        # Earnings-specific queries
        queries = [
            f"{ticker} earnings",
            f"{ticker} quarterly results",
            f"{company_name} earnings report",
            f"{ticker} EPS revenue",
        ]

        for query in queries:
            print(f"   Searching: {query}")
            try:
                articles = nc.collect_ai_search(ticker, company_name=query)
                for a in articles:
                    a['ticker'] = ticker
                all_articles.extend(articles)
                print(f"      Found {len(articles)} articles")
            except Exception as e:
                print(f"      Error: {e}")

        # Also standard news with force refresh
        print(f"   Fetching standard news (force refresh)...")
        try:
            result = nc.collect_and_save(ticker, days_back=5, force_refresh=True)
            standard_count = result.get('collected', 0)
            print(f"      Got {standard_count} standard articles")
        except Exception as e:
            print(f"      Error: {e}")

        # Dedupe and save earnings-specific articles
        seen = set()
        unique = []
        for a in all_articles:
            title = str(a.get('title', '')).lower()[:40]
            if title and title not in seen:
                seen.add(title)
                unique.append(a)

        if unique:
            saved = nc.save_articles(unique)
            print(f"   ✅ Saved {saved} earnings articles")

    except Exception as e:
        print(f"   ❌ News error: {e}")

    # 3. Re-run Sentiment Analysis
    print("\nStep 3: Sentiment Analysis")
    print("-" * 40)

    try:
        from src.db.connection import get_engine
        from src.screener.sentiment import SentimentAnalyzer

        engine = get_engine()

        # Get recent articles
        df = pd.read_sql(f"""
            SELECT * FROM news_articles 
            WHERE ticker = '{ticker}'
            ORDER BY COALESCE(published_at, created_at) DESC
            LIMIT 30
        """, engine)

        if df.empty:
            print("   ❌ No articles to analyze")
        else:
            print(f"   Analyzing {len(df)} articles...")

            # Convert to list of dicts
            articles = df.to_dict('records')

            sa = SentimentAnalyzer()
            result = sa.analyze_ticker_sentiment(ticker, articles)

            if result:
                print(f"   ✅ Sentiment Score: {result.get('sentiment_score')}")
                print(f"   ✅ Signal: {result.get('signal')}")
            else:
                print("   ⚠️ Sentiment analysis returned no result")

    except Exception as e:
        print(f"   ❌ Sentiment error: {e}")

    # 4. Get actual earnings results
    print("\nStep 4: Earnings Results")
    print("-" * 40)

    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)

        # Try earnings_dates first
        ed = stock.earnings_dates
        if ed is not None and not ed.empty:
            for idx in ed.index:
                d = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                if d <= date.today():
                    row = ed.loc[idx]
                    eps_est = row.get('EPS Estimate')
                    eps_act = row.get('Reported EPS')
                    surprise = row.get('Surprise(%)')

                    print(f"   Date: {d}")
                    print(f"   EPS Estimate: ${eps_est}")
                    print(f"   EPS Actual: ${eps_act}")
                    print(f"   Surprise: {surprise}%")

                    # Determine beat/miss
                    if eps_act and eps_est:
                        if float(eps_act) > float(eps_est):
                            print("   Result: ✅ BEAT")
                        elif float(eps_act) < float(eps_est):
                            print("   Result: ❌ MISS")
                        else:
                            print("   Result: ➡️ INLINE")
                    break

        # Price reaction
        print("\n   Price Reaction:")
        hist = stock.history(period="5d")
        if len(hist) >= 2:
            prev = hist['Close'].iloc[-2]
            curr = hist['Close'].iloc[-1]
            pct = ((curr - prev) / prev) * 100
            print(f"   Previous Close: ${prev:.2f}")
            print(f"   Latest Close: ${curr:.2f}")
            print(f"   Change: {pct:+.2f}%")

            if pct <= -5:
                print("   📉 MAJOR SELLOFF")
            elif pct >= 5:
                print("   📈 MAJOR RALLY")

        # Pre-market
        info = stock.info
        pre = info.get('preMarketPrice')
        if pre:
            prev_close = info.get('previousClose', prev)
            pct = ((pre - prev_close) / prev_close) * 100
            print(f"\n   Pre-Market: ${pre:.2f} ({pct:+.2f}%)")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    # 5. Summary
    print(f"\n{'='*60}")
    print("DONE - Refresh the Deep Dive page to see updated data")
    print(f"{'='*60}\n")

    print("If still not showing:")
    print("1. Make sure you're loading fresh signals (not cached)")
    print("2. The Signal Hub may need to regenerate the UnifiedSignal")
    print("3. Check that sentiment score updated in screener_scores table")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.analytics.earnings_fixer TICKER [EARNINGS_DATE]")
        print("       python -m src.analytics.earnings_fixer NKE")
        print("       python -m src.analytics.earnings_fixer NKE 2024-12-19")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    earnings_date = sys.argv[2] if len(sys.argv) > 2 else None

    fix_earnings_data(ticker, earnings_date)