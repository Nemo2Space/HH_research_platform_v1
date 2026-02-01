"""
Bond News Fetcher Module - Multi-Source

Fetches bond/rates news from MULTIPLE sources:
- Google News RSS (no API key needed)
- Finnhub API
- NewsAPI
- AI Search (Brave/Tavily via tool server)

Uses the same multi-source approach as news.py for stocks.
Each source failure doesn't stop the others.

Location: src/analytics/bond_news.py
Author: Alpha Research Platform
"""

import os
import json
import hashlib
import datetime
import time
import feedparser
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from urllib.parse import quote_plus

from src.utils.logging import get_logger
from src.db.connection import get_connection

logger = get_logger(__name__)

# Try to import date parser
try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    date_parser = None
    DATEUTIL_AVAILABLE = False


def parse_news_date(date_value) -> Optional[datetime.datetime]:
    """Parse various date formats from news sources."""
    if date_value is None or date_value == '':
        return None

    try:
        dt = None

        if isinstance(date_value, datetime.datetime):
            return date_value
        elif isinstance(date_value, (int, float)):
            return datetime.datetime.fromtimestamp(date_value)

        date_str = str(date_value).strip()

        # Try dateutil first
        if DATEUTIL_AVAILABLE and date_parser:
            try:
                dt = date_parser.parse(date_str, fuzzy=True)
                if dt:
                    return dt.replace(tzinfo=None) if dt.tzinfo else dt
            except:
                pass

        # Manual parsing
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
                return dt.replace(tzinfo=None) if dt.tzinfo else dt
            except:
                continue

    except Exception as e:
        logger.debug(f"Date parsing failed for '{date_value}': {e}")

    return None


@dataclass
class BondNewsArticle:
    """Single bond news article."""
    title: str
    description: str
    url: str
    source: str
    published_at: datetime.datetime
    image_url: Optional[str] = None

    # Sentiment
    sentiment: str = "NEUTRAL"
    sentiment_score: float = 0.5
    sentiment_reasoning: str = ""

    # Categorization
    category: str = "General"
    relevance_score: float = 1.0
    credibility_score: int = 5

    # Metadata
    article_id: str = ""

    def __post_init__(self):
        if not self.article_id:
            self.article_id = hashlib.md5(self.url.encode()).hexdigest()[:12]


@dataclass
class BondNewsResult:
    """Complete bond news analysis result."""
    articles: List[BondNewsArticle]
    fetch_time: datetime.datetime

    overall_sentiment: str = "NEUTRAL"
    overall_score: float = 0.5
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0

    themes: List[str] = field(default_factory=list)
    summary: str = ""
    ai_context: str = ""

    # Source stats
    sources_tried: int = 0
    sources_succeeded: int = 0


# Source credibility
SOURCE_CREDIBILITY = {
    'bloomberg': 10, 'wsj': 10, 'ft': 10, 'reuters': 10, 'financial times': 10,
    'cnbc': 9, 'barrons': 9, 'barron\'s': 9,
    'forbes': 8, 'marketwatch': 8, 'morningstar': 8,
    'seekingalpha': 7, 'yahoo finance': 7, 'yahoo': 7, 'investing.com': 7,
    'thestreet': 6, 'benzinga': 6, 'the bond buyer': 9,
    'businessinsider': 5, 'business insider': 5,
    'federal reserve': 10, 'treasury': 10,
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


# Bond-related search terms
BOND_SEARCH_TERMS = [
    "treasury yields",
    "bond market",
    "federal reserve interest rates",
    "treasury auction",
    "10-year yield",
    "FOMC decision",
    "rate cuts fed",
    "inflation CPI bonds",
    "TLT ETF",
    "treasury bonds outlook",
]


class BondNewsFetcher:
    """Fetches bond/rates news from multiple sources."""

    def __init__(self):
        # API keys from environment
        self.finnhub_key = os.getenv('FINNHUB_API_KEY', '')
        self.newsapi_key = os.getenv('NEWSAPI_API_KEY', '') or os.getenv('NEWSAPI_KEY', '')
        self.tool_server_url = os.getenv('TOOL_SERVER_URL', '')

        # Cache
        self._cache = {}
        self._cache_duration = datetime.timedelta(hours=2)

        logger.info(f"BondNewsFetcher initialized: Finnhub={'✓' if self.finnhub_key else '✗'}, "
                    f"NewsAPI={'✓' if self.newsapi_key else '✗'}, "
                    f"ToolServer={'✓' if self.tool_server_url else '✗'}")

    def fetch_news(self, max_articles: int = 25, days_back: int = 3,
                   force_refresh: bool = False) -> BondNewsResult:
        """
        Fetch bond news from ALL available sources.
        Each source failure doesn't stop others.
        """
        cache_key = f"bond_news_{days_back}"

        # Check cache
        if not force_refresh and cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if datetime.datetime.now() - cached_time < self._cache_duration:
                logger.debug("Returning cached bond news")
                return cached_result

        all_articles = []
        sources_tried = 0
        sources_succeeded = 0

        # 1. Google News RSS (no API key needed - always try)
        sources_tried += 1
        try:
            google_articles = self._fetch_google_news(days_back)
            if google_articles:
                all_articles.extend(google_articles)
                sources_succeeded += 1
                logger.info(f"Google News: {len(google_articles)} articles")
        except Exception as e:
            logger.warning(f"Google News failed: {e}")

        # 2. Finnhub
        if self.finnhub_key:
            sources_tried += 1
            try:
                finnhub_articles = self._fetch_finnhub_news(days_back)
                if finnhub_articles:
                    all_articles.extend(finnhub_articles)
                    sources_succeeded += 1
                    logger.info(f"Finnhub: {len(finnhub_articles)} articles")
            except Exception as e:
                logger.warning(f"Finnhub failed: {e}")

        # 3. NewsAPI
        if self.newsapi_key:
            sources_tried += 1
            try:
                newsapi_articles = self._fetch_newsapi(days_back)
                if newsapi_articles:
                    all_articles.extend(newsapi_articles)
                    sources_succeeded += 1
                    logger.info(f"NewsAPI: {len(newsapi_articles)} articles")
            except Exception as e:
                logger.warning(f"NewsAPI failed: {e}")

        # 4. AI Search (Brave/Tavily via tool server)
        if self.tool_server_url:
            sources_tried += 1
            try:
                ai_articles = self._fetch_ai_search()
                if ai_articles:
                    all_articles.extend(ai_articles)
                    sources_succeeded += 1
                    logger.info(f"AI Search: {len(ai_articles)} articles")
            except Exception as e:
                logger.warning(f"AI Search failed: {e}")

        # Deduplicate by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)

        # Sort by date (newest first)
        unique_articles.sort(key=lambda x: x.published_at or datetime.datetime.min, reverse=True)

        # Limit
        unique_articles = unique_articles[:max_articles]

        # Build result
        result = BondNewsResult(
            articles=unique_articles,
            fetch_time=datetime.datetime.now(),
            sources_tried=sources_tried,
            sources_succeeded=sources_succeeded
        )

        # Cache
        self._cache[cache_key] = (datetime.datetime.now(), result)

        logger.info(f"Bond news: {len(unique_articles)} articles from {sources_succeeded}/{sources_tried} sources")

        return result

    def _fetch_google_news(self, days_back: int = 3) -> List[BondNewsArticle]:
        """Fetch from Google News RSS (no API key needed)."""
        articles = []

        search_queries = [
            "treasury+yields",
            "bond+market",
            "federal+reserve+rates",
            "FOMC+decision",
            "treasury+auction",
        ]

        cutoff = datetime.datetime.now() - datetime.timedelta(days=days_back)
        seen_titles = set()

        for query in search_queries:
            try:
                url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(url)

                for entry in feed.entries[:10]:
                    title = entry.get('title', '')

                    # Skip if already seen
                    title_lower = title.lower().strip()
                    if title_lower in seen_titles:
                        continue
                    seen_titles.add(title_lower)

                    # Parse date
                    pub_date = parse_news_date(entry.get('published', ''))
                    if pub_date and pub_date < cutoff:
                        continue

                    # Extract source from title (Google News format: "Title - Source")
                    source = "Google News"
                    if ' - ' in title:
                        parts = title.rsplit(' - ', 1)
                        if len(parts) == 2:
                            title = parts[0]
                            source = parts[1]

                    # Check relevance
                    relevance = self._calculate_relevance(title, entry.get('summary', ''))
                    if relevance < 0.3:
                        continue

                    articles.append(BondNewsArticle(
                        title=title,
                        description=entry.get('summary', '')[:500],
                        url=entry.get('link', ''),
                        source=source,
                        published_at=pub_date or datetime.datetime.now(),
                        category=self._categorize(title, entry.get('summary', '')),
                        relevance_score=relevance,
                        credibility_score=get_source_credibility(source),
                    ))

                time.sleep(0.3)  # Rate limit

            except Exception as e:
                logger.debug(f"Google News query '{query}' failed: {e}")

        return articles

    def _fetch_finnhub_news(self, days_back: int = 3) -> List[BondNewsArticle]:
        """Fetch from Finnhub general news."""
        articles = []

        try:
            # Finnhub general news (includes market/economy news)
            url = f"https://finnhub.io/api/v1/news"
            params = {
                'category': 'general',
                'token': self.finnhub_key
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                cutoff = datetime.datetime.now() - datetime.timedelta(days=days_back)

                for item in data[:30]:
                    # Filter for bond-related content
                    headline = item.get('headline', '')
                    summary = item.get('summary', '')

                    relevance = self._calculate_relevance(headline, summary)
                    if relevance < 0.3:
                        continue

                    pub_date = parse_news_date(item.get('datetime'))
                    if pub_date and pub_date < cutoff:
                        continue

                    articles.append(BondNewsArticle(
                        title=headline,
                        description=summary[:500],
                        url=item.get('url', ''),
                        source=item.get('source', 'Finnhub'),
                        published_at=pub_date or datetime.datetime.now(),
                        image_url=item.get('image'),
                        category=self._categorize(headline, summary),
                        relevance_score=relevance,
                        credibility_score=get_source_credibility(item.get('source', '')),
                    ))

        except Exception as e:
            logger.debug(f"Finnhub error: {e}")

        return articles

    def _fetch_newsapi(self, days_back: int = 3) -> List[BondNewsArticle]:
        """Fetch from NewsAPI."""
        articles = []

        from_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')

        keyword_groups = [
            "treasury yields OR bond market",
            "federal reserve rates OR FOMC",
            "inflation CPI bonds OR rate cuts",
        ]

        seen_urls = set()

        for keywords in keyword_groups:
            try:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': keywords,
                    'from': from_date,
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'pageSize': 15,
                    'apiKey': self.newsapi_key,
                }

                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()

                    for item in data.get('articles', []):
                        article_url = item.get('url', '')
                        if article_url in seen_urls:
                            continue
                        seen_urls.add(article_url)

                        title = item.get('title', '') or ''
                        description = item.get('description', '') or ''

                        relevance = self._calculate_relevance(title, description)
                        if relevance < 0.3:
                            continue

                        pub_date = parse_news_date(item.get('publishedAt', ''))
                        source = item.get('source', {}).get('name', 'NewsAPI')

                        articles.append(BondNewsArticle(
                            title=title,
                            description=description[:500],
                            url=article_url,
                            source=source,
                            published_at=pub_date or datetime.datetime.now(),
                            image_url=item.get('urlToImage'),
                            category=self._categorize(title, description),
                            relevance_score=relevance,
                            credibility_score=get_source_credibility(source),
                        ))

                time.sleep(0.2)

            except Exception as e:
                logger.debug(f"NewsAPI query failed: {e}")

        return articles

    def _fetch_ai_search(self) -> List[BondNewsArticle]:
        """Fetch from AI Search (Brave/Tavily via tool server)."""
        articles = []

        search_queries = [
            "treasury bond market news today",
            "federal reserve interest rate decision",
            "treasury yields forecast",
        ]

        for query in search_queries[:2]:  # Limit queries
            try:
                response = requests.post(
                    f"{self.tool_server_url}/search",
                    json={"query": query, "max_results": 8},
                    timeout=15
                )

                if response.status_code == 200:
                    results = response.json().get('results', [])

                    for item in results:
                        title = item.get('title', '')
                        snippet = item.get('snippet', '')
                        url = item.get('url', '')

                        # Skip if not bond-related
                        relevance = self._calculate_relevance(title, snippet)
                        if relevance < 0.3:
                            continue

                        # Try to extract source from URL
                        source = item.get('source', '')
                        if not source or source.lower() in ['brave', 'tavily', 'duckduckgo']:
                            source = self._extract_source_from_url(url)

                        # Try to parse date from content
                        pub_date = None
                        for field in ['published_at', 'date', 'age']:
                            if item.get(field):
                                pub_date = parse_news_date(item.get(field))
                                if pub_date:
                                    break

                        articles.append(BondNewsArticle(
                            title=title,
                            description=snippet[:500],
                            url=url,
                            source=source or 'Web Search',
                            published_at=pub_date or datetime.datetime.now(),
                            category=self._categorize(title, snippet),
                            relevance_score=relevance,
                            credibility_score=get_source_credibility(source),
                        ))

                time.sleep(0.5)

            except Exception as e:
                logger.debug(f"AI Search query failed: {e}")

        return articles

    def _extract_source_from_url(self, url: str) -> str:
        """Extract source name from URL."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            domain = domain.replace('www.', '')

            # Known domains
            domain_map = {
                'reuters.com': 'Reuters',
                'bloomberg.com': 'Bloomberg',
                'wsj.com': 'Wall Street Journal',
                'cnbc.com': 'CNBC',
                'ft.com': 'Financial Times',
                'marketwatch.com': 'MarketWatch',
                'yahoo.com': 'Yahoo Finance',
                'finance.yahoo.com': 'Yahoo Finance',
                'barrons.com': "Barron's",
                'seekingalpha.com': 'Seeking Alpha',
                'investing.com': 'Investing.com',
                'businessinsider.com': 'Business Insider',
                'forbes.com': 'Forbes',
            }

            for key, name in domain_map.items():
                if key in domain:
                    return name

            # Generic: use first part of domain
            main_part = domain.split('.')[0]
            return main_part.capitalize()

        except:
            return ""

    def _calculate_relevance(self, title: str, description: str) -> float:
        """Calculate relevance score for bond news."""
        text = (title + ' ' + description).lower()

        high_relevance = [
            'treasury', 'bond', 'yield', 'fed ', 'fomc', 'rate cut', 'rate hike',
            'inflation', 'tlt', 'zroz', 'fixed income', '10-year', '30-year',
            '2-year', 'auction', 'quantitative', 'monetary policy', 'interest rate',
            'powell', 'federal reserve', 'dovish', 'hawkish'
        ]

        medium_relevance = [
            'economy', 'gdp', 'employment', 'jobs', 'cpi', 'pce',
            'dollar', 'central bank', 'recession', 'growth'
        ]

        score = 0.2

        for term in high_relevance:
            if term in text:
                score += 0.12

        for term in medium_relevance:
            if term in text:
                score += 0.05

        return min(1.0, score)

    def _categorize(self, title: str, description: str) -> str:
        """Categorize the article."""
        text = (title + ' ' + description).lower()

        if any(term in text for term in ['fed ', 'fomc', 'powell', 'rate cut', 'rate hike', 'monetary']):
            return "Fed Policy"
        elif any(term in text for term in ['inflation', 'cpi', 'pce', 'prices']):
            return "Inflation"
        elif any(term in text for term in ['auction', 'issuance', 'supply', 'treasury sale']):
            return "Auctions"
        elif any(term in text for term in ['yield', 'curve', '10-year', '30-year', '2-year']):
            return "Yields"
        elif any(term in text for term in ['economy', 'gdp', 'jobs', 'employment', 'recession']):
            return "Macro"
        else:
            return "General"


class BondSentimentAnalyzer:
    """Analyzes bond news sentiment using local LLM."""

    def __init__(self):
        self._llm_available = False
        self._client = None

        try:
            from openai import OpenAI
            base_url = os.getenv('LLM_QWEN_BASE_URL', 'http://localhost:1234/v1')
            self._client = OpenAI(base_url=base_url, api_key="not-needed", timeout=60)
            self._llm_available = True
            logger.info(f"Bond sentiment analyzer using LLM at {base_url}")
        except Exception as e:
            logger.debug(f"LLM not available for sentiment: {e}")

    def analyze_article(self, article: BondNewsArticle) -> BondNewsArticle:
        """Analyze sentiment of a single article."""
        if not self._llm_available:
            return self._keyword_sentiment(article)

        try:
            # Use /no_think to prevent thinking tags
            prompt = f"""/no_think
Analyze this bond market news for Treasury bonds/TLT impact.

Title: {article.title}
Summary: {article.description[:250]}

RULES:
- Fed rate CUTS / lower rates / dovish = BULLISH (bond prices UP)
- Fed rate HIKES / higher rates / hawkish = BEARISH (bond prices DOWN)
- Lower inflation / CPI miss = BULLISH
- Higher inflation / CPI beat = BEARISH
- Weak economy / recession = BULLISH (flight to safety)
- Strong economy / jobs beat = BEARISH

OUTPUT ONLY THIS JSON (no other text):
{{"sentiment": "BULLISH", "score": 0.75, "reason": "rate cuts expected"}}

Replace values based on your analysis. Score: 0.0=very bearish, 0.5=neutral, 1.0=very bullish."""

            response = self._client.chat.completions.create(
                model=os.getenv('LLM_QWEN_MODEL', 'local-model'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
            )

            result_text = response.choices[0].message.content.strip()

            # Strip thinking tags if present
            if '</think>' in result_text:
                result_text = result_text.split('</think>')[-1].strip()
            if '<think>' in result_text:
                result_text = result_text.split('<think>')[0].strip()

            # Try to extract JSON
            import re
            json_match = re.search(r'\{[^{}]*\}', result_text)

            if json_match:
                try:
                    result = json.loads(json_match.group())
                    sentiment = result.get('sentiment', '').upper()

                    if sentiment in ['BULLISH', 'BEARISH', 'NEUTRAL']:
                        article.sentiment = sentiment
                        article.sentiment_score = float(result.get('score', 0.5))
                        article.sentiment_reasoning = result.get('reason', '')
                        logger.debug(f"LLM sentiment: {sentiment} for '{article.title[:40]}...'")
                        return article
                except json.JSONDecodeError:
                    pass

            # If LLM failed to give valid JSON, use keyword sentiment
            logger.debug(f"LLM response not parseable, using keywords: {result_text[:100]}")
            return self._keyword_sentiment(article)

        except Exception as e:
            logger.debug(f"LLM sentiment error: {e}")
            return self._keyword_sentiment(article)

    def _keyword_sentiment(self, article: BondNewsArticle) -> BondNewsArticle:
        """Fallback keyword-based sentiment with flexible matching."""
        text = (article.title + ' ' + article.description).lower()

        # More flexible patterns - check for words separately
        bull_score = 0
        bear_score = 0

        # Strong bullish signals
        if 'rate cut' in text or 'cut rate' in text or 'cuts rate' in text:
            bull_score += 3
        if 'lower rate' in text or 'rates lower' in text:
            bull_score += 3
        if 'dovish' in text:
            bull_score += 3
        if 'rate' in text and 'cut' in text:  # "cut interest rate", "rate...cut"
            bull_score += 2
        if 'pause' in text and ('hike' in text or 'rate' in text):
            bull_score += 2
        if 'inflation' in text and ('fall' in text or 'drop' in text or 'cool' in text or 'ease' in text):
            bull_score += 2
        if 'recession' in text or 'slowdown' in text:
            bull_score += 2
        if 'flight to safety' in text or 'risk off' in text or 'risk-off' in text:
            bull_score += 2
        if 'strong auction' in text or 'demand' in text and 'treasury' in text:
            bull_score += 1
        if 'yield' in text and ('fall' in text or 'drop' in text or 'decline' in text):
            bull_score += 2
        if 'bond' in text and ('rally' in text or 'gain' in text):
            bull_score += 2

        # Strong bearish signals
        if 'rate hike' in text or 'hike rate' in text or 'hikes rate' in text:
            bear_score += 3
        if 'higher rate' in text or 'rates higher' in text:
            bear_score += 3
        if 'hawkish' in text:
            bear_score += 3
        if 'rate' in text and 'raise' in text:
            bear_score += 2
        if 'inflation' in text and ('rise' in text or 'surge' in text or 'hot' in text or 'sticky' in text):
            bear_score += 2
        if 'strong economy' in text or 'robust growth' in text:
            bear_score += 2
        if 'strong jobs' in text or 'job growth' in text or 'wage growth' in text:
            bear_score += 1
        if 'risk on' in text or 'risk-on' in text:
            bear_score += 1
        if 'weak auction' in text:
            bear_score += 2
        if 'yield' in text and ('rise' in text or 'surge' in text or 'spike' in text):
            bear_score += 2
        if 'bond' in text and ('selloff' in text or 'sell-off' in text or 'drop' in text):
            bear_score += 2

        # Determine sentiment
        if bull_score >= 2 and bull_score > bear_score:
            article.sentiment = "BULLISH"
            article.sentiment_score = min(0.9, 0.6 + bull_score * 0.05)
        elif bear_score >= 2 and bear_score > bull_score:
            article.sentiment = "BEARISH"
            article.sentiment_score = max(0.1, 0.4 - bear_score * 0.05)
        else:
            article.sentiment = "NEUTRAL"
            article.sentiment_score = 0.5

        logger.debug(f"Keyword sentiment: {article.sentiment} (bull={bull_score}, bear={bear_score}) for '{article.title[:40]}...'")
        return article

    def analyze_all(self, news_result: BondNewsResult, max_to_analyze: int = 15) -> BondNewsResult:
        """Analyze sentiment for all articles and aggregate."""

        for i, article in enumerate(news_result.articles[:max_to_analyze]):
            news_result.articles[i] = self.analyze_article(article)

        if news_result.articles:
            analyzed = news_result.articles[:max_to_analyze]

            if analyzed:
                total_weight = sum(a.relevance_score for a in analyzed)
                if total_weight > 0:
                    weighted_score = sum(a.sentiment_score * a.relevance_score for a in analyzed) / total_weight
                else:
                    weighted_score = 0.5

                news_result.overall_score = weighted_score

                if weighted_score >= 0.6:
                    news_result.overall_sentiment = "BULLISH"
                elif weighted_score <= 0.4:
                    news_result.overall_sentiment = "BEARISH"
                else:
                    news_result.overall_sentiment = "NEUTRAL"

            news_result.bullish_count = sum(1 for a in news_result.articles if a.sentiment == "BULLISH")
            news_result.bearish_count = sum(1 for a in news_result.articles if a.sentiment == "BEARISH")
            news_result.neutral_count = sum(1 for a in news_result.articles if a.sentiment == "NEUTRAL")

            news_result.themes = self._extract_themes(news_result.articles)
            news_result.ai_context = self._build_ai_context(news_result)

        return news_result

    def _extract_themes(self, articles: List[BondNewsArticle]) -> List[str]:
        """Extract key themes."""
        theme_counts = {}

        themes_keywords = {
            'Fed Policy': ['fed ', 'fomc', 'powell', 'rate cut', 'rate hike', 'monetary'],
            'Inflation': ['inflation', 'cpi', 'pce', 'prices'],
            'Recession': ['recession', 'slowdown', 'contraction'],
            'Auction': ['auction', 'bid-to-cover', 'treasury sale'],
            'Yields': ['yield', 'yields rise', 'yields fall'],
        }

        for article in articles:
            text = (article.title + ' ' + article.description).lower()
            for theme, keywords in themes_keywords.items():
                if any(kw in text for kw in keywords):
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1

        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes[:5] if count >= 1]

    def _build_ai_context(self, result: BondNewsResult) -> str:
        """Build AI context string."""
        sentiment_emoji = "🟢" if result.overall_sentiment == "BULLISH" else "🔴" if result.overall_sentiment == "BEARISH" else "🟡"

        context = f"""
📰 BOND NEWS SENTIMENT ANALYSIS
{'=' * 50}
Overall: {sentiment_emoji} {result.overall_sentiment} ({result.overall_score:.0%})
Articles: {len(result.articles)} ({result.bullish_count} bullish, {result.bearish_count} bearish, {result.neutral_count} neutral)
Sources: {result.sources_succeeded}/{result.sources_tried} succeeded
Themes: {', '.join(result.themes) if result.themes else 'None'}

Headlines:
"""
        for article in result.articles[:8]:
            icon = "🟢" if article.sentiment == "BULLISH" else "🔴" if article.sentiment == "BEARISH" else "🟡"
            context += f"  {icon} [{article.source}] {article.title[:60]}...\n"

        return context


# Convenience functions
_fetcher = None
_analyzer = None


def get_bond_news(max_articles: int = 25, days_back: int = 3,
                  force_refresh: bool = False, analyze_sentiment: bool = True) -> BondNewsResult:
    """Get bond news with sentiment analysis."""
    global _fetcher, _analyzer

    if _fetcher is None:
        _fetcher = BondNewsFetcher()

    result = _fetcher.fetch_news(max_articles, days_back, force_refresh)

    if analyze_sentiment and result.articles:
        if _analyzer is None:
            _analyzer = BondSentimentAnalyzer()
        result = _analyzer.analyze_all(result)

    return result


if __name__ == "__main__":
    print("Bond News Fetcher Test (Multi-Source)\n")
    print("=" * 60)

    result = get_bond_news(max_articles=15, days_back=3)

    print(f"Fetched {len(result.articles)} articles")
    print(f"Sources: {result.sources_succeeded}/{result.sources_tried} succeeded")
    print(f"Sentiment: {result.overall_sentiment} ({result.overall_score:.0%})")
    print(f"Themes: {result.themes}")

    print("\nArticles:")
    for article in result.articles[:8]:
        icon = "🟢" if article.sentiment == "BULLISH" else "🔴" if article.sentiment == "BEARISH" else "🟡"
        print(f"\n{icon} [{article.source}] {article.title[:60]}...")
        print(f"   {article.published_at.strftime('%Y-%m-%d %H:%M') if article.published_at else 'No date'}")
        print(f"   Category: {article.category} | Relevance: {article.relevance_score:.0%}")
        print(f"   URL: {article.url[:50]}...")