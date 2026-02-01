"""
Test News Collection

Tests news collection from available sources.

Usage:
    python scripts/test_news_collection.py
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.data.news import NewsCollector, NewsConfig


def main():
    print("=" * 50)
    print("Alpha Platform - News Collection Test")
    print("=" * 50)
    print()

    # Show config
    config = NewsConfig.from_env()
    print("Configuration:")
    print(f"  Finnhub API Key: {'Set' if config.finnhub_key else 'Not set'}")
    print(f"  Financial Datasets Key: {'Set' if config.financial_datasets_key else 'Not set'}")
    print(f"  Reddit Credentials: {'Set' if config.reddit_client_id else 'Not set'}")
    print()

    collector = NewsCollector(config)

    # Test with one ticker
    ticker = "AAPL"
    print(f"Collecting news for {ticker}...")
    print()

    articles = collector.collect_all_news(ticker, days_back=3, max_per_source=10)

    print(f"\nTotal articles collected: {len(articles)}")
    print()

    # Show sample articles
    print("Sample articles:")
    print("-" * 50)
    for i, article in enumerate(articles[:5]):
        print(f"{i+1}. [{article['source']}] (credibility: {article['credibility_score']})")
        print(f"   {article['title'][:80]}...")
        print()

    # Summary by source
    print("Articles by collection source:")
    sources = {}
    for article in articles:
        src = article.get('collection_source', 'unknown')
        sources[src] = sources.get(src, 0) + 1

    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")

    print()
    print("=" * 50)
    print("SUCCESS - News collection test complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()