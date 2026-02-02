"""
Alpha Platform - Sentiment Analysis

Two-stage sentiment analysis with dual-model architecture:
1. Filter Model: Score and rank articles by relevance (HIGH/MEDIUM/LOW/NOT)
2. Sentiment Model: Deep sentiment analysis on top-ranked articles

Both models default to Qwen-32B but can be configured independently for future use.

FIXES APPLIED:
- deduplicate_articles: Now checks both 'title' and 'headline'
- score_batch_filter: Now checks both 'title' and 'headline'
- analyze_sentiment: Now checks both 'title' and 'headline'
- analyze_ticker_sentiment: Normalizes articles to have both fields
"""

import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from openai import OpenAI

from src.utils.logging import get_logger

logger = get_logger(__name__)

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """
    LLM configuration - loads from .env

    Dual-model architecture:
    - FILTER model: Used for relevance scoring (Stage 1)
    - SENTIMENT model: Used for sentiment analysis (Stage 2)

    Both default to Qwen-32B but can be configured independently.
    """
    # Filter model (Stage 1 - relevance scoring)
    filter_base_url: str = os.getenv("LLM_FILTER_BASE_URL",
                                     os.getenv("LLM_QWEN_BASE_URL", "http://172.23.193.91:8090/v1"))
    filter_model: str = os.getenv("LLM_FILTER_MODEL", os.getenv("LLM_QWEN_MODEL", "Qwen3-32B-Q6_K.gguf"))

    # Sentiment model (Stage 2 - sentiment analysis)
    sentiment_base_url: str = os.getenv("LLM_SENTIMENT_BASE_URL",
                                        os.getenv("LLM_QWEN_BASE_URL", "http://172.23.193.91:8090/v1"))
    sentiment_model: str = os.getenv("LLM_SENTIMENT_MODEL", os.getenv("LLM_QWEN_MODEL", "Qwen3-32B-Q6_K.gguf"))

    # Legacy aliases (for backward compatibility)
    @property
    def qwen_base_url(self) -> str:
        return self.sentiment_base_url

    @property
    def qwen_model(self) -> str:
        return self.sentiment_model

    # Timeouts and batch sizes
    timeout: int = int(os.getenv("LLM_TIMEOUT", "120"))
    filter_batch_size: int = int(os.getenv("LLM_FILTER_BATCH_SIZE", "25"))
    sentiment_batch_size: int = int(os.getenv("LLM_SENTIMENT_BATCH_SIZE", "40"))


# Source credibility scores
SOURCE_CREDIBILITY = {
    'bloomberg': 10, 'wsj': 10, 'wall street journal': 10, 'financial times': 10, 'ft.com': 10,
    'reuters': 10, 'associated press': 10, 'ap news': 10,
    'cnbc': 9, 'barrons': 9, "barron's": 9, 'investor business daily': 9,
    'forbes': 8, 'marketwatch': 8, 'morningstar': 8, 'the economist': 8,
    'seeking alpha': 7, 'seekingalpha': 7, 'yahoo finance': 7, 'yahoo': 7,
    'zacks': 7, 'investopedia': 7,
    'thestreet': 6, 'the street': 6, 'motley fool': 6, 'benzinga': 6,
    'business insider': 5, 'insider': 5,
    'reddit': 4, 'stocktwits': 4,
}


def get_source_credibility(source: str) -> int:
    """Get credibility score for a news source."""
    if not source:
        return 5
    source_lower = source.lower()
    for key, score in SOURCE_CREDIBILITY.items():
        if key in source_lower:
            return score
    return 5


def get_article_title(article: Dict) -> str:
    """
    Get title from article, checking both 'title' and 'headline' fields.
    This handles articles from different sources (API vs database).
    """
    return article.get('title') or article.get('headline') or ''


def normalize_article(article: Dict) -> Dict:
    """
    Normalize article to have both 'title' and 'headline' fields.
    This ensures compatibility with all downstream code.
    """
    title = get_article_title(article)
    if title:
        article['title'] = title
        article['headline'] = title
    return article


def deduplicate_articles(articles: List[Dict]) -> List[Dict]:
    """Remove duplicate articles based on title similarity."""
    if not articles:
        return []

    seen_titles = set()
    unique = []

    for article in articles:
        # FIXED: Check both 'title' and 'headline' fields
        title = get_article_title(article)
        title = title.lower().strip() if title else ''

        # Normalize title for comparison
        normalized = re.sub(r'[^\w\s]', '', title)
        normalized = ' '.join(normalized.split()[:8])  # First 8 words

        if normalized and normalized not in seen_titles:
            seen_titles.add(normalized)
            unique.append(article)

    return unique


def clean_title(title: str, max_len: int = 80) -> str:
    """Clean title for LLM prompt - remove special characters."""
    if not title:
        return ''
    # Remove special unicode characters, keep basic ASCII
    title = title.encode('ascii', 'ignore').decode('ascii')
    # Remove extra whitespace
    title = ' '.join(title.split())
    # Truncate
    return title[:max_len]


class SentimentAnalyzer:
    """
    Two-stage sentiment analyzer using dual-model architecture.

    Stage 1 (Filter Model): Scores articles by relevance
    Stage 2 (Sentiment Model): Analyzes sentiment on filtered articles

    Both models default to Qwen-32B but can be configured independently
    via environment variables for future experimentation.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

        # Initialize Filter model client (Stage 1 - relevance scoring)
        self.filter_client = None
        self.filter_available = False
        try:
            self.filter_client = OpenAI(
                base_url=self.config.filter_base_url,
                api_key="not-needed",
                timeout=60
            )
            self.filter_client.models.list()
            logger.info(f"Filter model connected: {self.config.filter_model} @ {self.config.filter_base_url}")

            # Warmup request
            try:
                self.filter_client.chat.completions.create(
                    model=self.config.filter_model,
                    messages=[{"role": "user", "content": "Say OK"}],
                    temperature=0.1,
                    max_tokens=10
                )
                logger.info("Filter model warmup complete")
            except:
                pass

            self.filter_available = True
        except Exception as e:
            logger.warning(f"Filter model not available: {e}")

        # Initialize Sentiment model client (Stage 2 - sentiment analysis)
        self.sentiment_client = None
        self.sentiment_available = False

        # Check if both models use the same endpoint - reuse client if so
        same_endpoint = (
                self.config.filter_base_url == self.config.sentiment_base_url and
                self.config.filter_model == self.config.sentiment_model
        )

        if same_endpoint and self.filter_available:
            # Reuse filter client for sentiment (same model)
            self.sentiment_client = self.filter_client
            self.sentiment_available = True
            logger.info("Sentiment model: reusing filter model client (same endpoint)")
        else:
            # Initialize separate sentiment client
            try:
                self.sentiment_client = OpenAI(
                    base_url=self.config.sentiment_base_url,
                    api_key="not-needed",
                    timeout=self.config.timeout
                )
                self.sentiment_client.models.list()
                logger.info(
                    f"Sentiment model connected: {self.config.sentiment_model} @ {self.config.sentiment_base_url}")
                self.sentiment_available = True
            except Exception as e:
                logger.warning(f"Sentiment model not available: {e}")

        # Legacy aliases for backward compatibility
        self.qwen_client = self.sentiment_client
        self.qwen_available = self.sentiment_available
        self.gpt_oss_client = self.filter_client
        self.gpt_oss_available = self.filter_available

    def score_batch_filter(self, ticker: str, batch: List[Dict]) -> List[int]:
        """
        Score a single batch of articles using Filter model.
        Returns list of scores (95=HIGH, 70=MEDIUM, 40=LOW, 15=NOT).
        """
        scores = [50] * len(batch)  # default score

        if not self.filter_available:
            logger.warning(f"{ticker}: Filter model unavailable, using default scores")
            return scores

        # Build clean list for prompting
        lines = []
        for i, article in enumerate(batch, 1):
            # FIXED: Check both 'title' and 'headline'
            title = get_article_title(article)
            title = clean_title(title, 80)
            lines.append(f"{i}. {title}")

        prompt = f"""/no_think
Rate these headlines for relevance to {ticker} stock trading decisions.

Headlines:
{chr(10).join(lines)}

Score each headline:
- 95 = HIGH: Directly about {ticker} - earnings, products, management, analysts, legal
- 70 = MEDIUM: Industry news affecting {ticker}, competitor news, sector trends
- 40 = LOW: Tangentially related, general market, broad economic
- 15 = NOT: Unrelated, spam, clickbait, or about different company

Return ONLY a JSON object with headline numbers and scores:
{{"1": <score>, "2": <score>, ...}}"""

        try:
            response = self.filter_client.chat.completions.create(
                model=self.config.filter_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )

            result_text = response.choices[0].message.content.strip()

            # Handle think tags
            if '</think>' in result_text:
                result_text = result_text.split('</think>')[-1].strip()

            # Parse JSON
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))

                for key, value in result.items():
                    try:
                        idx = int(key) - 1
                        if 0 <= idx < len(scores):
                            scores[idx] = int(value)
                    except (ValueError, TypeError):
                        pass

                logger.info(f"{ticker}: Batch scored {len(batch)}/{len(batch)} articles")

        except Exception as e:
            logger.error(f"{ticker}: Filter model scoring failed: {e}")
            return scores

        return scores

    # Legacy alias
    def score_batch_gpt_oss(self, ticker: str, batch: List[Dict]) -> List[int]:
        """Legacy alias for score_batch_filter."""
        return self.score_batch_filter(ticker, batch)

    def save_relevance_scores(self, ticker: str, articles: List[Dict]):
        """Save relevance scores back to the news_articles table."""
        from src.db.connection import get_connection

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    updated = 0
                    for article in articles:
                        relevance = article.get('relevance_score')
                        if relevance is None:
                            continue

                        relevance_100 = relevance
                        url = article.get('url', article.get('link', ''))
                        headline = get_article_title(article)[:200]

                        # IMPROVED: Try URL match first (more reliable), fallback to headline
                        if url:
                            cur.execute("""
                                UPDATE news_articles
                                SET relevance_score = %s
                                WHERE ticker = %s AND url = %s
                            """, (relevance_100, ticker, url))

                            if cur.rowcount > 0:
                                updated += 1
                                continue

                        # Fallback to headline matching if URL didn't match
                        cur.execute("""
                            UPDATE news_articles
                            SET relevance_score = %s
                            WHERE ticker = %s
                              AND LEFT(headline, 200) = LEFT(%s, 200)
                        """, (relevance_100, ticker, headline))

                        if cur.rowcount > 0:
                            updated += 1

                conn.commit()
                logger.info(f"{ticker}: Saved relevance scores for {updated}/{len(articles)} articles")
        except Exception as e:
            logger.error(f"{ticker}: Failed to save relevance scores: {e}")

    def filter_articles(self, ticker: str, articles: List[Dict]) -> List[Dict]:
        """
        Use Filter model to score relevance, sort, and filter articles.
        """
        if not articles:
            return []

        if not self.filter_available:
            logger.warning(f"{ticker}: Filter model unavailable → skipping scoring")
            return articles

        # Set initial score
        for a in articles:
            a["relevance_score"] = 50

        batch_size = self.config.filter_batch_size

        # Batch scoring
        for start in range(0, len(articles), batch_size):
            batch = articles[start:start + batch_size]
            batch_scores = self.score_batch_filter(ticker, batch)

            for i, score in enumerate(batch_scores):
                articles[start + i]["relevance_score"] = score

        # Sort by relevance then source credibility
        articles = sorted(
            articles,
            key=lambda x: (
                x.get("relevance_score", 5),
                get_source_credibility(x.get("source", "")),
            ),
            reverse=True,
        )

        # Save relevance scores for ALL articles before filtering
        self.save_relevance_scores(ticker, articles)

        # Filter out NOT relevant (score <= 2)
        filtered = [a for a in articles if a["relevance_score"] >= 60]

        # Ensure minimum of 20 articles
        if len(filtered) < 20 and len(articles) >= 20:
            filtered = articles[:20]

        # Log distribution
        scores = [a["relevance_score"] for a in articles]
        high = sum(1 for s in scores if s >= 80)
        medium = sum(1 for s in scores if 50 <= s < 80)
        low = sum(1 for s in scores if s < 50)

        logger.info(f"{ticker}: Relevance - HIGH: {high}, MEDIUM: {medium}, LOW: {low}")
        logger.info(f"{ticker}: Keeping {len(filtered)}/{len(articles)} articles")

        return filtered

    # Legacy alias
    def filter_articles_gpt_oss(self, ticker: str, articles: List[Dict]) -> List[Dict]:
        """Legacy alias for filter_articles."""
        return self.filter_articles(ticker, articles)

    def analyze_sentiment(self, ticker: str, articles: List[Dict]) -> Dict[str, Any]:
        """
        Stage 2: Use Sentiment model to analyze sentiment of filtered articles.
        """
        if not articles:
            return {
                'sentiment_score': 50,
                'sentiment_weighted': 50,
                'sentiment_class': 'neutral',
                'relevant_count': 0
            }

        if not self.sentiment_available:
            logger.warning(f"{ticker}: Sentiment model not available, returning neutral")
            return {
                'sentiment_score': 50,
                'sentiment_weighted': 50,
                'sentiment_class': 'neutral',
                'relevant_count': len(articles)
            }

        # Add credibility scores
        for article in articles:
            if 'credibility' not in article:
                article['credibility'] = get_source_credibility(article.get('source', ''))

        # Take top articles by relevance (already sorted)
        top_articles = articles[:self.config.sentiment_batch_size]

        # Build prompt
        lines = []
        for i, article in enumerate(top_articles, 1):
            # FIXED: Check both 'title' and 'headline'
            title = get_article_title(article)
            title = (title or '')[:150]
            source = article.get('source', 'Unknown')
            lines.append(f"{i}. [{source}] {title}")

        articles_text = "\n".join(lines)

        prompt = f"""/no_think
Analyze the sentiment of these news headlines for {ticker} stock.

Headlines:
{articles_text}

Provide an OVERALL sentiment score from 0-100:
- 0-20: Very Bearish (major negative news, significant problems)
- 21-40: Bearish (negative developments, concerns)
- 41-59: Neutral (mixed or no significant news)
- 60-79: Bullish (positive developments, good news)
- 80-100: Very Bullish (major positive news, exceptional results)

Consider:
- Weight higher-credibility sources more heavily
- Concrete events matter more than speculation
- Earnings, products, and analyst actions are most important

Return ONLY a JSON object:
{{"ticker": "{ticker}", "overall_mean_sentiment": <0-100>, "relevant_count": <number>, "reasoning": "<brief explanation>"}}"""

        try:
            response = self.sentiment_client.chat.completions.create(
                model=self.config.sentiment_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()

            # Handle /think tags if present
            if '</think>' in result_text:
                result_text = result_text.split('</think>')[-1].strip()

            # Parse JSON
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))

                sentiment_score = int(result.get('overall_mean_sentiment', 50))
                sentiment_score = max(0, min(100, sentiment_score))
                relevant_count = int(result.get('relevant_count', len(top_articles)))

                # Calculate weighted sentiment
                total_weight = sum(a.get('credibility', 5) for a in top_articles)
                weighted_sentiment = sentiment_score  # Same since we're applying uniformly

                # Classify
                if sentiment_score >= 70:
                    sentiment_class = 'very_bullish'
                elif sentiment_score >= 55:
                    sentiment_class = 'bullish'
                elif sentiment_score >= 45:
                    sentiment_class = 'neutral'
                elif sentiment_score >= 30:
                    sentiment_class = 'bearish'
                else:
                    sentiment_class = 'very_bearish'

                logger.info(f"{ticker}: Sentiment={sentiment_score} ({sentiment_class}), Relevant={relevant_count}")

                # Save sentiment to database
                self._save_sentiment_score(ticker, sentiment_score, sentiment_class, len(top_articles))

                return {
                    'sentiment_score': sentiment_score,
                    'sentiment_weighted': weighted_sentiment,
                    'sentiment_class': sentiment_class,
                    'relevant_count': relevant_count,
                    'signal': sentiment_class  # Add signal for compatibility
                }

        except Exception as e:
            logger.error(f"{ticker}: Sentiment model error: {e}")

        return {
            'sentiment_score': 50,
            'sentiment_weighted': 50,
            'sentiment_class': 'neutral',
            'relevant_count': len(articles)
        }

    def _save_sentiment_score(self, ticker: str, score: int, sentiment_class: str, article_count: int):
        """Save sentiment score to screener_scores table."""
        try:
            from src.db.connection import get_connection
            from datetime import date

            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Update existing or insert new
                    cur.execute("""
                                INSERT INTO screener_scores (ticker, sentiment_score, sentiment_signal, article_count, score_date)
                                VALUES (%s, %s, %s, %s, %s) ON CONFLICT (ticker, score_date) 
                        DO
                                UPDATE SET
                                    sentiment_score = EXCLUDED.sentiment_score,
                                    sentiment_signal = EXCLUDED.sentiment_signal,
                                    article_count = EXCLUDED.article_count
                                """, (ticker, score, sentiment_class.upper(), article_count, date.today()))
                conn.commit()
                logger.info(f"{ticker}: Saved sentiment score {score} to screener_scores")
        except Exception as e:
            logger.debug(f"{ticker}: Could not save sentiment to screener_scores: {e}")

    # Legacy alias
    def analyze_sentiment_qwen(self, ticker: str, articles: List[Dict]) -> Dict[str, Any]:
        """Legacy alias for analyze_sentiment."""
        return self.analyze_sentiment(ticker, articles)

    def analyze_ticker_sentiment(self, ticker: str, articles: List[Dict]) -> Dict[str, Any]:
        """
        Two-stage sentiment analysis pipeline.

        1. Normalize articles (ensure both title and headline fields)
        2. Deduplicate articles
        3. Filter model scores and ranks by relevance
        4. Take top articles (sorted by relevance + credibility)
        5. Sentiment model analyzes sentiment on best articles
        """
        if not articles:
            return {
                'ticker': ticker,
                'sentiment_score': 50,
                'sentiment_weighted': 50,
                'sentiment_class': 'neutral',
                'article_count': 0,
                'relevant_count': 0
            }

        original_count = len(articles)

        # FIXED: Normalize articles to have both 'title' and 'headline' fields
        articles = [normalize_article(a) for a in articles]

        # Stage 1: Basic deduplication
        articles = deduplicate_articles(articles)
        dedup_count = len(articles)

        # Stage 1b: Pre-filter obviously irrelevant articles (before expensive LLM call)
        if len(articles) > 25:
            pre_filter_count = len(articles)
            ticker_upper = ticker.upper()
            ticker_lower = ticker.lower()

            def _likely_relevant(article):
                """Quick keyword check - keep if ticker appears in title or description text."""
                title = (article.get('title') or article.get('headline') or '').lower()
                desc = (article.get('description') or '')[:200].lower()
                text = title + ' ' + desc
                # Keep if ticker symbol mentioned in actual content
                if ticker_lower in text:
                    return True
                # Keep articles from high-credibility sources (they're usually relevant)
                if article.get('credibility_score', 0) >= 8:
                    return True
                return False

            articles = [a for a in articles if _likely_relevant(a)]
            removed = pre_filter_count - len(articles)
            if removed > 0:
                logger.info(
                    f"{ticker}: Pre-filtered {removed} irrelevant articles ({pre_filter_count} → {len(articles)})")

        # Stage 2: Filter model scoring and ranking
        if self.filter_available and len(articles) > 10:
            articles = self.filter_articles(ticker, articles)

        scored_count = len(articles)

        # Stage 3: Take top 30 (already sorted by relevance + credibility)
        top_articles = articles[:30]

        logger.info(
            f"{ticker}: {original_count} → {dedup_count} (dedup) → {scored_count} (scored) → {len(top_articles)} (top)")

        # Stage 4: Sentiment model analysis
        result = self.analyze_sentiment(ticker, top_articles)

        result['ticker'] = ticker
        result['article_count'] = original_count

        return result