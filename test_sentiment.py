import time
from src.data.news import NewsCollector
from src.screener.sentiment import SentimentAnalyzer, deduplicate_articles, normalize_article

nc = NewsCollector()
articles = nc.get_cached_articles('VRTX', days_back=7)
print(f"Cached articles: {len(articles)}")

articles = [normalize_article(a) for a in articles]
articles = deduplicate_articles(articles)
print(f"After dedup: {len(articles)}")

sa = SentimentAnalyzer()
start = time.time()
result = sa.analyze_ticker_sentiment('VRTX', articles)
t = time.time() - start
score = result.get('sentiment_score')
print(f"\nSentiment: {t:.1f}s, score={score}")
