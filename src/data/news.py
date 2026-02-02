import os
import datetime
import time
import json
import pandas as pd
import requests
import feedparser
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# For robust date parsing (handles various formats from news sources)
try:
    from dateutil import parser as date_parser

    DATEUTIL_AVAILABLE = True
except ImportError:
    date_parser = None
    DATEUTIL_AVAILABLE = False

from src.db.repository import Repository
from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_news_date(date_value) -> Optional[str]:
    """
    Parse various date formats from news sources and return ISO format string.

    Handles formats like:
    - "Tue, 17 Dec 2024 09:05:23 GMT" (Google News)
    - ISO format strings
    - Unix timestamps
    - datetime objects

    Returns:
        ISO format string or None if parsing fails
    """
    if date_value is None or date_value == '':
        return None

    try:
        dt = None

        # Already a datetime
        if isinstance(date_value, datetime.datetime):
            dt = date_value
        # Unix timestamp (int or float)
        elif isinstance(date_value, (int, float)):
            dt = datetime.datetime.fromtimestamp(date_value)
        else:
            date_str = str(date_value).strip()

            # Try pandas first (handles ISO well)
            try:
                dt = pd.to_datetime(date_str)
                if isinstance(dt, pd.Timestamp):
                    dt = dt.to_pydatetime()
            except:
                pass

            # Try dateutil (handles "Tue, 17 Dec 2024" etc)
            if dt is None and DATEUTIL_AVAILABLE and date_parser:
                try:
                    dt = date_parser.parse(date_str, fuzzy=True)
                except:
                    pass

            # Manual parsing for common formats
            if dt is None:
                formats = [
                    "%a, %d %b %Y %H:%M:%S %Z",
                    "%a, %d %b %Y %H:%M:%S %z",
                    "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d",
                ]
                for fmt in formats:
                    try:
                        dt = datetime.datetime.strptime(date_str, fmt)
                        break
                    except:
                        continue

        if dt is not None:
            # Remove timezone and return ISO format
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt.isoformat()

    except Exception as e:
        logger.debug(f"Date parsing failed for '{date_value}': {e}")

    return None


@dataclass
class NewsConfig:
    """News collection configuration."""
    finnhub_key: str = ""
    financial_datasets_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "AlphaPlatform/1.0"
    cache_dir: str = "data/cache/news"
    cache_hours: int = 6  # NEW: Hours before news is considered stale

    @classmethod
    def from_env(cls) -> "NewsConfig":
        return cls(
            finnhub_key=os.getenv("FINNHUB_API_KEY", ""),
            financial_datasets_key=os.getenv("FINANCIAL_DATASETS_API_KEY", ""),
            reddit_client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
            reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "AlphaPlatform/1.0"),
            cache_dir=os.getenv("NEWS_CACHE_DIR", "data/cache/news"),
            cache_hours=int(os.getenv("NEWS_CACHE_HOURS", "6")),
        )


# Source credibility scores (from Project 1)
SOURCE_CREDIBILITY = {
    'bloomberg': 10, 'wsj': 10, 'ft': 10, 'reuters': 10,
    'cnbc': 9, 'barrons': 9,
    'forbes': 8, 'marketwatch': 8, 'morningstar': 8,
    'seekingalpha': 7, 'investors.com': 7, 'yahoo finance': 7, 'yahoo': 7,
    'thestreet': 6, 'motley fool': 6, 'benzinga': 6,
    'businessinsider': 5, 'business insider': 5,
    'nasdaq': 7, 'finnhub': 7,
    'reddit': 4,
    'default': 5
}


def get_source_credibility(source: str) -> int:
    """Get credibility score for a news source."""
    if not source:
        return SOURCE_CREDIBILITY['default']

    source_lower = source.lower()
    for key, score in SOURCE_CREDIBILITY.items():
        if key in source_lower:
            return score
    return SOURCE_CREDIBILITY['default']


class NewsCollector:
    """Collects news from multiple sources."""

    def __init__(self, config: Optional[NewsConfig] = None, repository: Optional[Repository] = None):
        self.config = config or NewsConfig.from_env()
        self.repo = repository or Repository()

        # Ensure cache directory exists
        os.makedirs(self.config.cache_dir, exist_ok=True)

        # Initialize VADER for Reddit sentiment (optional)
        self._vader = None

    @property
    def vader(self):
        """Lazy load VADER sentiment analyzer."""
        if self._vader is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
            except ImportError:
                logger.warning("vaderSentiment not installed. Reddit sentiment disabled.")
        return self._vader

    # =========================================================================
    # SMART CACHING - NEW
    # =========================================================================

    def get_last_news_fetch_time(self, ticker: str) -> Optional[datetime.datetime]:
        """
        Get the timestamp of the most recent news article collected for a ticker.
        Uses created_at (when we saved it) not published_at (when article was published).
        """
        query = """
                SELECT MAX(created_at) as last_fetch
                FROM news_articles
                WHERE ticker = %(ticker)s \
                """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, {"ticker": ticker})
                    row = cur.fetchone()
                    if row and row[0]:
                        return row[0]
        except Exception as e:
            logger.debug(f"{ticker}: Error getting last fetch time: {e}")
        return None

    def should_fetch_news(self, ticker: str, cache_hours: Optional[int] = None) -> bool:
        """
        Check if we should fetch fresh news for a ticker.

        Returns True if:
        - No news exists for this ticker
        - Last fetch was more than cache_hours ago

        Args:
            ticker: Stock ticker
            cache_hours: Hours before cache is stale (default from config)
        """
        if cache_hours is None:
            cache_hours = self.config.cache_hours

        last_fetch = self.get_last_news_fetch_time(ticker)

        if last_fetch is None:
            logger.debug(f"{ticker}: No cached news, will fetch fresh")
            return True

        # Handle timezone-aware vs naive datetime comparison
        now = datetime.datetime.now()
        if last_fetch.tzinfo is not None:
            # Convert to naive by removing timezone (assume same timezone)
            last_fetch = last_fetch.replace(tzinfo=None)

        age = now - last_fetch
        age_hours = age.total_seconds() / 3600

        if age_hours > cache_hours:
            logger.debug(f"{ticker}: News cache is {age_hours:.1f}h old (>{cache_hours}h), will fetch fresh")
            return True
        else:
            logger.debug(f"{ticker}: News cache is {age_hours:.1f}h old (<{cache_hours}h), using cached")
            return False

    def get_cached_articles(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """Get cached news articles from database."""
        query = """
                SELECT *
                FROM news_articles
                WHERE ticker = %(ticker)s
                  AND published_at >= NOW() - INTERVAL '%(days)s days'
                ORDER BY published_at DESC \
                """
        try:
            df = pd.read_sql(query, self.repo.engine, params={"ticker": ticker, "days": days_back})
            if df.empty:
                return []

            # Map column names to expected format
            articles = []
            for _, row in df.iterrows():
                article = {
                    'ticker': row.get('ticker', ticker),
                    'title': row.get('headline', row.get('title', '')),
                    'source': row.get('source', ''),
                    'published_at': row.get('published_at', ''),
                    'url': row.get('url', ''),
                    'description': row.get('content', row.get('description', '')),
                    'credibility_score': row.get('credibility_score', 5),
                    'collection_source': 'cached'
                }
                articles.append(article)

            return articles
        except Exception as e:
            logger.error(f"{ticker}: Error getting cached articles: {e}")
            return []

    # =========================================================================
    # GOOGLE NEWS
    # =========================================================================

    def get_google_news(self, ticker: str, days_back: int = 7, max_results: int = 25) -> List[Dict]:
        """Fetch news from Google News using GNews library."""
        try:
            from gnews import GNews

            today = datetime.datetime.now().date()
            start_date = today - datetime.timedelta(days=days_back)

            google_news = GNews()
            google_news.start_date = (start_date.year, start_date.month, start_date.day)
            google_news.end_date = (today.year, today.month, today.day)
            google_news.max_results = max_results

            news_list = google_news.get_news(f"{ticker} stock")

            articles = []
            for article in news_list:
                # FIX: Parse the date to ISO format before storing
                raw_date = article.get('published date', '')
                parsed_date = parse_news_date(raw_date)

                articles.append({
                    'ticker': ticker,
                    'title': article.get('title', ''),
                    'source': article.get('publisher', {}).get('title', 'Google News'),
                    'published_at': parsed_date,  # Now in ISO format
                    'url': article.get('url', ''),
                    'description': article.get('description', ''),
                    'collection_source': 'google_news'
                })

            logger.info(f"{ticker}: Google News returned {len(articles)} articles")
            return articles

        except Exception as e:
            logger.error(f"{ticker}: Google News error - {e}")
            return []

    # =========================================================================
    # FINNHUB
    # =========================================================================

    def get_finnhub_news(self, ticker: str, days_back: int = 7, max_results: int = 25) -> List[Dict]:
        """Fetch news from Finnhub API."""
        if not self.config.finnhub_key:
            logger.debug(f"{ticker}: Finnhub API key not configured")
            return []

        try:
            end_date = datetime.datetime.now().date()
            start_date = end_date - datetime.timedelta(days=days_back)

            url = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "token": self.config.finnhub_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            news_data = response.json()[:max_results]

            articles = []
            for article in news_data:
                pub_time = datetime.datetime.fromtimestamp(article.get('datetime', 0))
                articles.append({
                    'ticker': ticker,
                    'title': article.get('headline', ''),
                    'source': article.get('source', 'Finnhub'),
                    'published_at': pub_time.isoformat(),
                    'url': article.get('url', ''),
                    'description': article.get('summary', ''),
                    'collection_source': 'finnhub'
                })

            logger.info(f"{ticker}: Finnhub returned {len(articles)} articles")
            return articles

        except Exception as e:
            logger.error(f"{ticker}: Finnhub error - {e}")
            return []

    # =========================================================================
    # NASDAQ RSS
    # =========================================================================

    def get_nasdaq_news(self, ticker: str, days_back: int = 7, max_results: int = 25) -> List[Dict]:
        """Fetch news from Nasdaq RSS feed."""
        try:
            feed_url = 'https://www.nasdaq.com/feed/rssoutbound?category=Technology'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': '*/*',
            }

            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days_back)

            response = requests.get(feed_url, headers=headers, timeout=10)
            response.raise_for_status()

            feed_data = feedparser.parse(response.content)
            entries = feed_data.get('entries', [])

            articles = []
            ticker_upper = ticker.upper()

            for entry in entries:
                title = entry.get('title', '')
                summary = entry.get('summary', '')

                # Check if ticker is mentioned
                if ticker_upper not in title.upper() and ticker_upper not in summary.upper():
                    continue

                # Parse date
                published_str = entry.get('published', '')
                try:
                    published_dt = datetime.datetime.strptime(
                        published_str, "%a, %d %b %Y %H:%M:%S %Z"
                    )
                except:
                    published_dt = datetime.datetime.now()

                # Filter by date range
                if published_dt < start_date:
                    continue

                articles.append({
                    'ticker': ticker,
                    'title': title,
                    'source': 'Nasdaq',
                    'published_at': published_dt.isoformat(),
                    'url': entry.get('link', ''),
                    'description': summary,
                    'collection_source': 'nasdaq_rss'
                })

                if len(articles) >= max_results:
                    break

            logger.info(f"{ticker}: Nasdaq RSS returned {len(articles)} articles")
            return articles

        except Exception as e:
            logger.error(f"{ticker}: Nasdaq RSS error - {e}")
            return []

    # =========================================================================
    # FINANCIAL DATASETS
    # =========================================================================

    def get_financial_datasets_news(self, ticker: str, days_back: int = 7, max_results: int = 50) -> List[Dict]:
        """Fetch news from Financial Datasets API with caching."""
        end_date = datetime.datetime.now().date()
        start_date = end_date - datetime.timedelta(days=days_back)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Check cache first
        cache_file = os.path.join(
            self.config.cache_dir,
            f"{ticker}_fd_{start_str}_{end_str}.json"
        )

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                logger.info(f"{ticker}: Using cached Financial Datasets ({len(cached)} articles)")
                return cached[:max_results]
            except:
                pass

        # Fetch from API
        try:
            headers = {}
            if self.config.financial_datasets_key:
                headers["X-API-KEY"] = self.config.financial_datasets_key

            url = f"https://api.financialdatasets.ai/news/"
            params = {
                "ticker": ticker,
                "start_date": start_str,
                "end_date": end_str,
                "limit": max_results
            }

            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            news_items = data.get("news", [])

            articles = []
            for item in news_items:
                articles.append({
                    'ticker': ticker,
                    'title': item.get('title', ''),
                    'source': item.get('source', 'Financial Datasets'),
                    'published_at': item.get('date', ''),
                    'url': item.get('url', ''),
                    'description': '',
                    'collection_source': 'financial_datasets'
                })

            # Cache results
            if len(articles) > 5:
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(articles, f)
                except:
                    pass

            logger.info(f"{ticker}: Financial Datasets returned {len(articles)} articles")
            return articles

        except Exception as e:
            logger.error(f"{ticker}: Financial Datasets error - {e}")
            return []

    # =========================================================================
    # REDDIT
    # =========================================================================

    def get_reddit_news(self, ticker: str, days_back: int = 7, max_results: int = 25) -> List[Dict]:
        """Fetch posts from Reddit with VADER sentiment."""
        if not self.config.reddit_client_id or not self.config.reddit_client_secret:
            logger.debug(f"{ticker}: Reddit credentials not configured")
            return []

        try:
            import praw

            reddit = praw.Reddit(
                client_id=self.config.reddit_client_id,
                client_secret=self.config.reddit_client_secret,
                user_agent=self.config.reddit_user_agent
            )

            subreddits = ['stocks', 'investing', 'wallstreetbets', 'StockMarket']
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(days=days_back)

            all_posts = []
            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)
                    posts = subreddit.search(ticker, limit=max_results // len(subreddits))

                    for post in posts:
                        post_time = datetime.datetime.fromtimestamp(post.created_utc)
                        if post_time < start_time:
                            continue

                        # VADER sentiment
                        sentiment_score = 0.0
                        sentiment_label = "neutral"
                        if self.vader:
                            full_text = f"{post.title}\n{post.selftext}"
                            scores = self.vader.polarity_scores(full_text)
                            sentiment_score = scores['compound']
                            if sentiment_score >= 0.05:
                                sentiment_label = "positive"
                            elif sentiment_score <= -0.05:
                                sentiment_label = "negative"

                        all_posts.append({
                            'ticker': ticker,
                            'title': post.title,
                            'source': f"Reddit/r/{sub_name}",
                            'published_at': post_time.isoformat(),
                            'url': f"https://www.reddit.com{post.permalink}",
                            'description': post.selftext[:300] if post.selftext else '',
                            'collection_source': 'reddit',
                            'reddit_sentiment': sentiment_label,
                            'reddit_sentiment_score': sentiment_score
                        })
                except Exception as e:
                    logger.debug(f"{ticker}: Reddit r/{sub_name} error - {e}")
                    continue

            # Sort by date
            all_posts.sort(key=lambda x: x['published_at'], reverse=True)

            logger.info(f"{ticker}: Reddit returned {len(all_posts)} posts")
            return all_posts[:max_results]

        except Exception as e:
            logger.error(f"{ticker}: Reddit error - {e}")
            return []

    # =========================================================================
    # AI SEARCH
    # =========================================================================

    def collect_ai_search(self, ticker: str, company_name: str = None) -> List[Dict[str, Any]]:
        """
        Collect news using AI-powered web search (Brave/Tavily/DuckDuckGo).

        Args:
            ticker: Stock ticker
            company_name: Optional company name for better search

        Returns:
            List of article dicts
        """
        tool_server_url = "http://localhost:7001/search"

        # Build search query
        if company_name:
            query = f"{company_name} {ticker} stock news"
        else:
            query = f"{ticker} stock news financial"

        try:
            response = requests.post(
                tool_server_url,
                json={"query": query, "max_results": 15},
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            # URL to source name mapping
            URL_TO_SOURCE = {
                'seekingalpha.com': 'Seeking Alpha',
                'cnbc.com': 'CNBC',
                'yahoo.com': 'Yahoo Finance',
                'finance.yahoo.com': 'Yahoo Finance',
                'bloomberg.com': 'Bloomberg',
                'reuters.com': 'Reuters',
                'wsj.com': 'WSJ',
                'ft.com': 'Financial Times',
                'marketwatch.com': 'MarketWatch',
                'fool.com': 'Motley Fool',
                'barrons.com': 'Barrons',
                'investors.com': 'IBD',
                'benzinga.com': 'Benzinga',
                'thestreet.com': 'TheStreet',
                'forbes.com': 'Forbes',
                'morningstar.com': 'Morningstar',
                'nasdaq.com': 'Nasdaq',
                'investopedia.com': 'Investopedia',
                'tipranks.com': 'TipRanks',
                'zacks.com': 'Zacks',
                'investing.com': 'Investing.com',
                'businessinsider.com': 'Business Insider',
                'nytimes.com': 'NY Times',
                'cnn.com': 'CNN',
                'bbc.com': 'BBC',
                'marketbeat.com': 'MarketBeat',
                'simplywall.st': 'Simply Wall St',
                'tradingview.com': 'TradingView',
            }

            def extract_source_from_url(url: str) -> str:
                """Extract readable source name from URL."""
                if not url:
                    return ""
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc.lower()
                    if domain.startswith('www.'):
                        domain = domain[4:]

                    for url_pattern, source_name in URL_TO_SOURCE.items():
                        if url_pattern in domain:
                            return source_name

                    # Fallback: capitalize main domain part
                    parts = domain.replace('.com', '').replace('.org', '').replace('.net', '').split('.')
                    if parts:
                        main_part = parts[0] if len(parts) == 1 else parts[-2] if len(parts) > 1 else parts[0]
                        return main_part.capitalize()
                except:
                    pass
                return ""

            articles = []
            for item in results:
                url = item.get("url", "")

                # Get source - prefer extracted from URL over API-provided source
                api_source = item.get("source") or item.get("provider", "")
                extracted_source = extract_source_from_url(url)

                # Use extracted source if API source is generic (brave, tavily, etc)
                generic_sources = {'brave', 'tavily', 'ai search', 'duckduckgo', 'google', ''}
                if api_source.lower() in generic_sources:
                    source = extracted_source or api_source
                else:
                    source = api_source

                # Try to extract publication date from multiple possible fields
                pub_date = None
                date_fields = ['published_at', 'publishedAt', 'published_date', 'date', 'pubDate', 'pub_date', 'age',
                               'time']

                for field in date_fields:
                    raw_date = item.get(field)
                    if raw_date:
                        # Try to parse the date
                        parsed = parse_news_date(raw_date)
                        if parsed:
                            pub_date = parsed
                            break
                        # Handle relative dates like "2 days ago", "3 hours ago"
                        elif isinstance(raw_date, str) and 'ago' in raw_date.lower():
                            try:
                                import re
                                match = re.search(r'(\d+)\s*(hour|day|minute|min|hr|week|month)s?\s*ago',
                                                  raw_date.lower())
                                if match:
                                    num = int(match.group(1))
                                    unit = match.group(2)
                                    now = datetime.datetime.now()
                                    if 'hour' in unit or 'hr' in unit:
                                        pub_date = (now - datetime.timedelta(hours=num)).isoformat()
                                    elif 'day' in unit:
                                        pub_date = (now - datetime.timedelta(days=num)).isoformat()
                                    elif 'minute' in unit or 'min' in unit:
                                        pub_date = (now - datetime.timedelta(minutes=num)).isoformat()
                                    elif 'week' in unit:
                                        pub_date = (now - datetime.timedelta(weeks=num)).isoformat()
                                    elif 'month' in unit:
                                        pub_date = (now - datetime.timedelta(days=num * 30)).isoformat()
                                    if pub_date:
                                        break
                            except:
                                pass

                articles.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "source": source,
                    "published_at": pub_date or "",
                    "snippet": item.get("snippet", ""),
                    "credibility": 6,
                })

            logger.info(f"{ticker}: AI Search returned {len(articles)} articles")
            return articles

        except Exception as e:
            logger.error(f"{ticker}: AI Search error - {e}")
            return []

    # =========================================================================
    # COLLECT ALL (UPDATED WITH SMART CACHING)
    # =========================================================================

    def collect_all_news(self, ticker: str, days_back: int = 7, max_per_source: int = 25,
                         force_refresh: bool = False) -> List[Dict]:
        """
        Collect news from all available sources with smart caching.

        Args:
            ticker: Stock ticker
            days_back: Days of news to fetch
            max_per_source: Max articles per source
            force_refresh: If True, bypass cache and fetch fresh (NEW)

        Returns:
            Combined list of articles with credibility scores.
        """
        # Check if we should use cached news
        if not force_refresh and not self.should_fetch_news(ticker):
            cached = self.get_cached_articles(ticker, days_back)
            if cached:
                logger.info(f"{ticker}: Using {len(cached)} cached articles (< {self.config.cache_hours}h old)")
                return cached

        # Fetch fresh news
        logger.info(f"{ticker}: Fetching fresh news{'(forced)' if force_refresh else ''}")

        all_articles = []

        # Collect from each source
        # Collect from all sources in parallel (saves ~5-6s vs sequential)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch_source(source_name, fetch_func, *args):
            """Wrapper to catch errors per source."""
            try:
                return source_name, fetch_func(*args)
            except Exception as e:
                logger.error(f"{ticker}: {source_name} failed - {e}")
                return source_name, []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(_fetch_source, "Google News", self.get_google_news, ticker, days_back, max_per_source),
                executor.submit(_fetch_source, "Finnhub", self.get_finnhub_news, ticker, days_back, max_per_source),
                executor.submit(_fetch_source, "AI Search", self.collect_ai_search, ticker),
            ]

            for future in as_completed(futures):
                source_name, articles = future.result()
                if source_name == "AI Search":
                    for article in articles:
                        article['ticker'] = ticker
                all_articles.extend(articles)

        # Add credibility scores
        for article in all_articles:
            source = article.get('source', '')
            article['credibility_score'] = get_source_credibility(source)

        # Remove duplicates by title
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = article.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)

        logger.info(f"{ticker}: Collected {len(unique_articles)} unique articles from {len(all_articles)} total")

        return unique_articles

    # =========================================================================
    # SAVE TO DATABASE
    # =========================================================================

    def save_articles(self, articles: List[Dict]) -> int:
        """Save articles to database. Returns count saved."""
        saved = 0
        errors = 0
        for article in articles:
            try:
                self.repo.save_news_article(article)
                saved += 1
            except Exception as e:
                errors += 1
                # Log actual errors, not just debug
                if 'duplicate' not in str(e).lower() and 'unique' not in str(e).lower():
                    logger.warning(f"Error saving article: {e}")

        if errors > 0:
            logger.info(f"Saved {saved} articles, {errors} skipped (likely duplicates)")
        return saved

    def collect_and_save(self, ticker: str, days_back: int = 7, force_refresh: bool = False) -> Dict[str, Any]:
        """Collect news for a ticker and save to database."""
        articles = self.collect_all_news(ticker, days_back, force_refresh=force_refresh)
        saved = self.save_articles(articles)

        return {
            'ticker': ticker,
            'articles': articles,  # Include articles for sentiment analysis
            'collected': len(articles),
            'saved': saved
        }

    def collect_universe(self, days_back: int = 7, delay: float = 1.0,
                         force_refresh: bool = False, progress_callback=None) -> Dict[str, Any]:
        """Collect news for all tickers in universe."""
        tickers = self.repo.get_universe()
        total = len(tickers)

        results = {
            'total_tickers': total,
            'total_articles': 0,
            'by_ticker': {}
        }

        for i, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(i + 1, total, ticker)

            result = self.collect_and_save(ticker, days_back, force_refresh=force_refresh)
            results['by_ticker'][ticker] = result
            results['total_articles'] += result['collected']

            if delay > 0 and i < total - 1:
                time.sleep(delay)

        return results