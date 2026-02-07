import time
from src.data.news import NewsCollector
nc = NewsCollector()
start = time.time()
result = nc.collect_and_save('AAPL', days_back=7, force_refresh=True)
t = time.time() - start
print(f"News collection: {t:.1f}s, articles={result.get('collected')}")
