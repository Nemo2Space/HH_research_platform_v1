"""Test relevance score saving"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.screener.sentiment import SentimentAnalyzer
from src.data.news import NewsCollector

analyzer = SentimentAnalyzer()
collector = NewsCollector()

# Get articles for AAPL
articles = collector.collect_all_news('AAPL', days_back=7)
print(f'Collected {len(articles)} articles')

# Run GPT-OSS filtering
if articles:
    filtered = analyzer.filter_articles_gpt_oss('AAPL', articles)
    print(f'Filtered to {len(filtered)} articles')

    # Check relevance scores
    print('\nSample articles with scores:')
    for a in filtered[:5]:
        score = a.get('relevance_score', 'N/A')
        title = a.get('title', '')[:50]
        print(f'  {score} - {title}')

    # Save them
    print('\nSaving relevance scores...')
    analyzer.save_relevance_scores('AAPL', filtered)
    print('Done!')