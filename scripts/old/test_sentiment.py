"""
Test Sentiment Analysis

Tests the AI sentiment analyzer with sample articles.
NOTE: Requires local Qwen model running on port 8080.

Usage:
    python scripts/test_sentiment.py
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.screener.sentiment import SentimentAnalyzer, LLMConfig


def main():
    print("=" * 50)
    print("Alpha Platform - Sentiment Analysis Test")
    print("=" * 50)
    print()

    config = LLMConfig.from_env()
    print(f"Qwen URL: {config.qwen_base_url}")
    print(f"Qwen Model: {config.qwen_model}")
    print()

    # Check if LLM is available
    print("Testing LLM connection...")
    try:
        analyzer = SentimentAnalyzer(config)
        # Quick test
        _ = analyzer.client
        print("LLM client initialized successfully")
    except Exception as e:
        print(f"ERROR: Cannot connect to LLM: {e}")
        print("\nMake sure your local Qwen model is running on port 8080")
        return

    print()

    # Test with sample articles
    ticker = "AAPL"
    sample_articles = [
        {"title": "Apple Stock Surges to All-Time High on Strong iPhone Sales", "source": "Bloomberg"},
        {"title": "Apple Raises Dividend by 10%, Announces $100B Buyback", "source": "Reuters"},
        {"title": "Analysts Raise Apple Price Targets After Stellar Earnings", "source": "CNBC"},
        {"title": "Apple Faces Supply Chain Concerns in China", "source": "WSJ"},
        {"title": "Apple vs Samsung: Market Share Battle Continues", "source": "TechCrunch"},
    ]

    print(f"Analyzing {len(sample_articles)} articles for {ticker}...")
    print()

    result = analyzer.analyze_ticker_sentiment(ticker, sample_articles)

    print("Results:")
    print("-" * 30)
    print(f"Ticker:            {result['ticker']}")
    print(f"Sentiment Score:   {result['sentiment_score']}")
    print(f"Weighted Score:    {result['sentiment_weighted']}")
    print(f"Classification:    {result['sentiment_class']}")
    print(f"Article Count:     {result['article_count']}")
    print(f"Relevant Count:    {result['relevant_count']}")
    print()

    print("=" * 50)
    print("SUCCESS - Sentiment analysis test complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()