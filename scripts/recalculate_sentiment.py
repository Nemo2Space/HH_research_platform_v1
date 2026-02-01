"""
Recalculate Sentiment from Existing News Articles

This script reads news articles already in the database and recalculates
sentiment scores without re-fetching news or other data.

Usage:
    python recalculate_sentiment.py                    # All tickers
    python recalculate_sentiment.py --ticker AAPL     # Single ticker
    python recalculate_sentiment.py --days 7          # Last 7 days of news
    python recalculate_sentiment.py --dry-run         # Preview without saving

Author: HH Research Platform
"""

import argparse
import sys
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import pandas as pd

# Add project root to path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.connection import get_connection, get_engine
from src.screener.sentiment import SentimentAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_tickers_to_process(ticker: Optional[str] = None) -> List[str]:
    """Get list of tickers that have news articles."""
    query = """
        SELECT DISTINCT ticker 
        FROM news_articles 
        WHERE ticker IS NOT NULL
        ORDER BY ticker
    """
    if ticker:
        query = f"""
            SELECT DISTINCT ticker 
            FROM news_articles 
            WHERE ticker = '{ticker.upper()}'
        """

    df = pd.read_sql(query, get_engine())
    return df['ticker'].tolist()


def get_articles_for_ticker(ticker: str, days: int = 5) -> List[Dict]:
    """Get news articles for a ticker from the database."""
    query = """
        SELECT 
            id, ticker, headline, source, url, published_at
        FROM news_articles
        WHERE ticker = %(ticker)s
          AND published_at >= %(since)s
        ORDER BY published_at DESC
    """

    since = datetime.now() - timedelta(days=days)
    df = pd.read_sql(query, get_engine(), params={'ticker': ticker, 'since': since})

    # Convert to list of dicts matching what SentimentAnalyzer expects
    articles = []
    for _, row in df.iterrows():
        articles.append({
            'headline': row.get('headline', ''),
            'summary': '',  # Not in your schema
            'source': row.get('source', ''),
            'url': row.get('url', ''),
            'published_at': row.get('published_at'),
        })

    return articles


def update_screener_scores(ticker: str, sentiment_score: Optional[int], article_count: int, dry_run: bool = False):
    """Update screener_scores with new sentiment value."""
    today = date.today()

    if dry_run:
        print(f"  [DRY RUN] Would update {ticker}: sentiment={sentiment_score}, articles={article_count}")
        return True

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if row exists for today
                cur.execute("""
                    SELECT id FROM screener_scores 
                    WHERE ticker = %s AND date = %s
                """, (ticker, today))

                existing = cur.fetchone()

                if existing:
                    # Update existing row
                    cur.execute("""
                        UPDATE screener_scores 
                        SET sentiment_score = %s, 
                            article_count = %s,
                            sentiment_weighted = %s
                        WHERE ticker = %s AND date = %s
                    """, (sentiment_score, article_count, sentiment_score, ticker, today))
                else:
                    # Insert new row with just sentiment (other scores will be NULL)
                    cur.execute("""
                        INSERT INTO screener_scores (ticker, date, sentiment_score, article_count, sentiment_weighted)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (ticker, today, sentiment_score, article_count, sentiment_score))

                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Error updating {ticker}: {e}")
        return False


def recalculate_sentiment(
    ticker: Optional[str] = None,
    days: int = 5,
    dry_run: bool = False,
    verbose: bool = True
):
    """
    Main function to recalculate sentiment for tickers.

    Args:
        ticker: Single ticker to process, or None for all
        days: Number of days of news to consider
        dry_run: If True, don't save to database
        verbose: Print progress
    """
    # Get tickers to process
    tickers = get_tickers_to_process(ticker)

    if not tickers:
        print("No tickers found with news articles.")
        return

    print(f"\n{'='*60}")
    print(f"SENTIMENT RECALCULATION")
    print(f"{'='*60}")
    print(f"Tickers to process: {len(tickers)}")
    print(f"News window: {days} days")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE UPDATE'}")
    print(f"{'='*60}\n")

    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()

    results = {
        'updated': 0,
        'skipped_no_articles': 0,
        'skipped_no_sentiment': 0,
        'errors': 0,
    }

    for i, t in enumerate(tickers, 1):
        try:
            # Get articles
            articles = get_articles_for_ticker(t, days)

            if not articles:
                if verbose:
                    print(f"[{i}/{len(tickers)}] {t}: No articles found (skipped)")
                results['skipped_no_articles'] += 1
                continue

            # Run sentiment analysis
            sentiment_result = analyzer.analyze_ticker_sentiment(t, articles)

            if not sentiment_result or sentiment_result.get('sentiment_score') is None:
                if verbose:
                    print(f"[{i}/{len(tickers)}] {t}: Sentiment analysis returned None ({len(articles)} articles)")
                results['skipped_no_sentiment'] += 1

                # Update with NULL to indicate we tried but couldn't calculate
                update_screener_scores(t, None, len(articles), dry_run)
                continue

            sentiment_score = sentiment_result.get('sentiment_score')

            # Update database
            success = update_screener_scores(t, sentiment_score, len(articles), dry_run)

            if success:
                results['updated'] += 1
                if verbose:
                    # Show sentiment interpretation
                    if sentiment_score >= 70:
                        label = "BULLISH 🟢"
                    elif sentiment_score >= 55:
                        label = "SLIGHTLY BULLISH"
                    elif sentiment_score <= 30:
                        label = "BEARISH 🔴"
                    elif sentiment_score <= 45:
                        label = "SLIGHTLY BEARISH"
                    else:
                        label = "NEUTRAL"

                    print(f"[{i}/{len(tickers)}] {t}: {sentiment_score} ({label}) from {len(articles)} articles")
            else:
                results['errors'] += 1

        except Exception as e:
            logger.error(f"Error processing {t}: {e}")
            results['errors'] += 1
            if verbose:
                print(f"[{i}/{len(tickers)}] {t}: ERROR - {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Updated:              {results['updated']}")
    print(f"Skipped (no articles): {results['skipped_no_articles']}")
    print(f"Skipped (no sentiment): {results['skipped_no_sentiment']}")
    print(f"Errors:               {results['errors']}")
    print(f"{'='*60}\n")

    if dry_run:
        print("This was a DRY RUN. No changes were saved.")
        print("Run without --dry-run to apply changes.")


def main():
    parser = argparse.ArgumentParser(description='Recalculate sentiment from existing news articles')
    parser.add_argument('--ticker', '-t', type=str, help='Single ticker to process')
    parser.add_argument('--days', '-d', type=int, default=5, help='Days of news to consider (default: 5)')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without saving')
    parser.add_argument('--quiet', '-q', action='store_true', help='Less verbose output')

    args = parser.parse_args()

    recalculate_sentiment(
        ticker=args.ticker,
        days=args.days,
        dry_run=args.dry_run,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()