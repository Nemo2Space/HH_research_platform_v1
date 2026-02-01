"""
Test Full Sentiment Pipeline

Usage:
    python scripts/test_sentiment_full.py
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.screener.worker import ScreenerWorker

worker = ScreenerWorker(use_llm=True)

test_tickers = ['AAPL']

print('Running screener with LLM sentiment...')
print('-' * 70)

for ticker in test_tickers:
    result = worker.process_ticker(ticker, collect_news=True, days_back=7)
    sent = result['sentiment_score']
    sig = result['signal']['type']
    arts = result['article_count']
    tech = result['technical_score']
    print(f'{ticker}: Sentiment={sent}, Tech={tech}, Signal={sig}, Articles={arts}')

print('-' * 70)
print('Done!')