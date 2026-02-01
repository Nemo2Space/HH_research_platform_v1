"""
Debug Sentiment Analysis

Check what headlines are being sent to the sentiment model
and why it's returning neutral/0 relevant.
"""

import pandas as pd
from datetime import datetime


def debug_sentiment(ticker: str):
    """Debug sentiment analysis for a ticker."""

    print(f"\n{'=' * 60}")
    print(f"DEBUG SENTIMENT: {ticker}")
    print(f"{'=' * 60}\n")

    # 1. Get articles from DB
    print("1. ARTICLES IN DATABASE")
    print("-" * 40)

    from src.db.connection import get_engine
    engine = get_engine()

    df = pd.read_sql(f"""
        SELECT 
            headline,
            source,
            published_at,
            ai_sentiment_fast,
            relevance_score,
            created_at
        FROM news_articles 
        WHERE ticker = '{ticker}'
        ORDER BY COALESCE(published_at, created_at) DESC
        LIMIT 30
    """, engine)

    if df.empty:
        print("   ❌ No articles found!")
        return

    print(f"   Found {len(df)} articles\n")

    # Show headlines
    print("   Headlines being analyzed:")
    for i, row in df.iterrows():
        headline = str(row.get('headline', ''))[:70]
        source = str(row.get('source', ''))[:15]
        rel = row.get('relevance_score')
        sent = row.get('ai_sentiment_fast')

        rel_str = f"R:{rel:.0f}" if pd.notna(rel) else "R:--"
        sent_str = f"S:{sent:.0f}" if pd.notna(sent) else "S:--"

        print(f"   {i + 1}. [{source}] {rel_str} {sent_str} | {headline}")

    # 2. Check what sentiment is getting
    print(f"\n\n2. RUNNING SENTIMENT ANALYSIS")
    print("-" * 40)

    from src.screener.sentiment import SentimentAnalyzer

    # Convert to list of dicts (as the analyzer expects)
    articles = df.to_dict('records')

    # Add required fields
    for a in articles:
        # Map headline to title (for compatibility)
        if 'headline' in a and 'title' not in a:
            a['title'] = a['headline']

    print(f"   Prepared {len(articles)} articles")
    print(f"   Sample article keys: {list(articles[0].keys()) if articles else 'none'}")

    # Run analysis
    sa = SentimentAnalyzer()
    result = sa.analyze_ticker_sentiment(ticker, articles)

    print(f"\n   RESULT:")
    print(f"   Sentiment Score: {result.get('sentiment_score')}")
    print(f"   Signal: {result.get('signal')}")
    print(f"   Sentiment Class: {result.get('sentiment_class')}")
    print(f"   Relevant Count: {result.get('relevant_count')}")
    print(f"   Article Count: {result.get('article_count')}")

    # 3. Check screener_scores
    print(f"\n\n3. SCREENER_SCORES TABLE")
    print("-" * 40)

    df_scores = pd.read_sql(f"""
        SELECT * FROM screener_scores 
        WHERE ticker = '{ticker}'
        ORDER BY score_date DESC
        LIMIT 1
    """, engine)

    if df_scores.empty:
        print("   ❌ No scores found!")
    else:
        row = df_scores.iloc[0]
        print(f"   Score Date: {row.get('score_date')}")
        print(f"   Sentiment Score: {row.get('sentiment_score')}")
        print(f"   Sentiment Signal: {row.get('sentiment_signal')}")
        print(f"   Article Count: {row.get('article_count')}")
        print(f"   Total Score: {row.get('total_score')}")

    # 4. Analysis
    print(f"\n\n4. DIAGNOSIS")
    print("-" * 40)

    # Check for earnings-related headlines
    earnings_keywords = ['earning', 'quarter', 'revenue', 'eps', 'profit', 'result', 'beat', 'miss', 'guidance',
                         'outlook']
    earnings_headlines = []

    for _, row in df.iterrows():
        headline = str(row.get('headline', '')).lower()
        if any(kw in headline for kw in earnings_keywords):
            earnings_headlines.append(row.get('headline'))

    print(f"   Earnings-related headlines: {len(earnings_headlines)}/{len(df)}")

    if earnings_headlines:
        print("\n   Earnings headlines found:")
        for h in earnings_headlines[:5]:
            print(f"   📊 {h[:70]}")
    else:
        print("   ⚠️ NO earnings-related headlines found!")
        print("   This may explain neutral sentiment.")

    # Check relevance scores
    high_rel = df[df['relevance_score'] >= 80] if 'relevance_score' in df.columns else pd.DataFrame()
    if not high_rel.empty:
        print(f"\n   High relevance articles ({len(high_rel)}):")
        for _, row in high_rel.head(3).iterrows():
            print(f"   ⭐ {row.get('headline', '')[:60]}")


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NKE"
    debug_sentiment(ticker)