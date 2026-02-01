"""
Sentiment NLP Module - Phase 3

Advanced sentiment analysis using local LLM (Qwen3-32B via llama.cpp).
NOT keyword matching - actual language understanding.

Analyzes:
- Earnings call transcripts (management tone, hedging, confidence)
- News articles with financial context
- Management language shifts over time
- Forward guidance tone

Returns structured sentiment scores, not just positive/negative.

Author: Alpha Research Platform
"""

import requests
import json
import re
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logging import get_logger

logger = get_logger(__name__)


class SentimentLevel(Enum):
    """Sentiment classification levels."""
    VERY_BULLISH = "VERY_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    VERY_BEARISH = "VERY_BEARISH"


class ManagementTone(Enum):
    """Management communication tone."""
    CONFIDENT = "CONFIDENT"
    CAUTIOUS = "CAUTIOUS"
    DEFENSIVE = "DEFENSIVE"
    EVASIVE = "EVASIVE"
    OPTIMISTIC = "OPTIMISTIC"


@dataclass
class SentimentAnalysis:
    """Structured sentiment analysis result."""
    ticker: str
    as_of_date: date
    source_type: str  # 'earnings_call', 'news', 'filing'

    # Core sentiment
    sentiment_level: SentimentLevel = SentimentLevel.NEUTRAL
    sentiment_score: int = 50  # 0-100
    confidence: float = 0.0  # Model confidence in analysis

    # Management tone (for earnings calls)
    management_tone: ManagementTone = ManagementTone.CAUTIOUS
    management_confidence_score: int = 50  # 0-100
    hedging_language_pct: float = 0.0  # % of hedging phrases
    forward_guidance_tone: str = "NEUTRAL"

    # Key extractions
    key_positives: List[str] = field(default_factory=list)
    key_negatives: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    key_catalysts: List[str] = field(default_factory=list)

    # Comparisons
    sentiment_vs_prior: int = 0  # Change vs last analysis
    tone_shift: str = "STABLE"  # IMPROVING, STABLE, DETERIORATING

    # Raw
    summary: str = ""
    source_text_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'sentiment_level': self.sentiment_level.value,
            'sentiment_score': self.sentiment_score,
            'management_tone': self.management_tone.value,
            'management_confidence_score': self.management_confidence_score,
            'hedging_language_pct': round(self.hedging_language_pct, 1),
            'forward_guidance_tone': self.forward_guidance_tone,
            'key_positives': self.key_positives,
            'key_negatives': self.key_negatives,
            'key_risks': self.key_risks,
            'sentiment_vs_prior': self.sentiment_vs_prior,
            'summary': self.summary,
        }


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment across multiple sources."""
    ticker: str
    as_of_date: date

    # Aggregated scores
    overall_sentiment: SentimentLevel = SentimentLevel.NEUTRAL
    overall_score: int = 50

    # Component scores
    earnings_sentiment: int = 50
    news_sentiment: int = 50
    filing_sentiment: int = 50

    # Trends
    sentiment_trend_7d: str = "STABLE"
    sentiment_trend_30d: str = "STABLE"

    # Signal
    signal: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    signal_strength: int = 50

    # Details
    analyses: List[SentimentAnalysis] = field(default_factory=list)
    source_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'overall_sentiment': self.overall_sentiment.value,
            'overall_score': self.overall_score,
            'earnings_sentiment': self.earnings_sentiment,
            'news_sentiment': self.news_sentiment,
            'signal': self.signal,
            'signal_strength': self.signal_strength,
            'sentiment_trend_7d': self.sentiment_trend_7d,
            'source_count': self.source_count,
        }


class LocalLLMClient:
    """
    Client for local LLM via llama.cpp server.
    Connects to your Qwen3-32B running on WSL2.
    """

    # Default URL from environment or fallback
    DEFAULT_URL = "http://172.23.193.91:8090/v1"

    def __init__(self, base_url: str = None):
        import os
        # Priority: passed URL > env var > default
        self.base_url = base_url or os.getenv('LLM_QWEN_BASE_URL', self.DEFAULT_URL)
        # Remove trailing /v1 if present (we add it in endpoints)
        self.base_url = self.base_url.rstrip('/v1').rstrip('/')
        self._available = None

    def is_available(self) -> bool:
        """Check if LLM server is running."""
        if self._available is not None:
            return self._available

        # Try OpenAI-compatible models endpoint first
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            self._available = response.status_code == 200
            if self._available:
                logger.info(f"LLM connected at {self.base_url}")
                return True
        except Exception:
            pass

        # Try health endpoint (some servers have this)
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            self._available = response.status_code == 200
            if self._available:
                return True
        except Exception:
            pass

        # Try a simple completion as last resort
        try:
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json={"prompt": "Hi", "max_tokens": 1},
                timeout=10
            )
            self._available = response.status_code == 200
        except Exception:
            self._available = False

        if not self._available:
            logger.warning(f"LLM server not available at {self.base_url}")

        return self._available

    def complete(self, prompt: str, max_tokens: int = 1000,
                 temperature: float = 0.3) -> Optional[str]:
        """
        Get completion from local LLM.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (lower = more focused)

        Returns:
            Generated text or None if failed
        """
        if not self.is_available():
            return None

        # Add /no_think for Qwen3 to disable extended thinking mode
        prompt_with_hint = prompt + " /no_think"

        # Try OpenAI-compatible chat endpoint first (best for instruct models)
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt_with_hint}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": ["</analysis>", "\n\n\n"],
                },
                timeout=120  # LLM can be slow
            )

            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    if content:
                        return content
        except Exception as e:
            logger.debug(f"Chat endpoint failed: {e}")

        # Fallback to completions endpoint
        try:
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "prompt": prompt_with_hint,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": ["</analysis>", "\n\n\n"],
                },
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0].get("text", "")
        except Exception as e:
            logger.debug(f"Completions endpoint failed: {e}")

        # Try native llama.cpp endpoint
        try:
            response = requests.post(
                f"{self.base_url}/completion",
                json={
                    "prompt": prompt_with_hint,
                    "n_predict": max_tokens,
                    "temperature": temperature,
                    "stop": ["</analysis>", "\n\n\n"],
                },
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("content", "")
        except Exception as e:
            logger.debug(f"Native endpoint failed: {e}")

        return None

    def chat(self, messages: List[Dict], max_tokens: int = 1000,
             temperature: float = 0.3) -> Optional[str]:
        """
        Chat completion (for chat-tuned models).
        """
        if not self.is_available():
            return None

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0].get("message", {}).get("content", "")
        except Exception as e:
            logger.debug(f"Chat endpoint failed: {e}")

        return None


class SentimentNLPAnalyzer:
    """
    Advanced sentiment analyzer using local LLM.
    """

    def __init__(self, llm_url: str = None):
        self.llm = LocalLLMClient(llm_url)
        self._llm_available = None

    def _check_llm(self) -> bool:
        """Check LLM availability once."""
        if self._llm_available is None:
            self._llm_available = self.llm.is_available()
            if not self._llm_available:
                logger.warning("LLM not available - sentiment analysis will be limited")
        return self._llm_available

    def analyze_earnings_call(self, ticker: str,
                               transcript: str) -> SentimentAnalysis:
        """
        Analyze earnings call transcript for sentiment and management tone.

        Args:
            ticker: Stock symbol
            transcript: Full or partial earnings call transcript

        Returns:
            SentimentAnalysis with detailed breakdown
        """
        analysis = SentimentAnalysis(
            ticker=ticker,
            as_of_date=date.today(),
            source_type='earnings_call',
            source_text_length=len(transcript),
        )

        if not transcript or len(transcript) < 100:
            analysis.summary = "Insufficient transcript data"
            return analysis

        if not self._check_llm():
            # Fallback to basic analysis without LLM
            return self._basic_earnings_analysis(ticker, transcript)

        # Truncate if too long (keep first and last parts)
        max_chars = 12000
        if len(transcript) > max_chars:
            half = max_chars // 2
            transcript = transcript[:half] + "\n...[truncated]...\n" + transcript[-half:]

        prompt = f"""Analyze this earnings call transcript for {ticker}. Provide a structured analysis.

TRANSCRIPT:
{transcript}

Respond in this exact JSON format:
{{
    "sentiment_score": <0-100, 50=neutral, higher=bullish>,
    "management_confidence": <0-100, how confident does management sound>,
    "hedging_pct": <0-100, percentage of hedging/uncertain language>,
    "forward_guidance": "<BULLISH/NEUTRAL/BEARISH/NOT_PROVIDED>",
    "management_tone": "<CONFIDENT/CAUTIOUS/DEFENSIVE/EVASIVE/OPTIMISTIC>",
    "key_positives": ["<positive point 1>", "<positive point 2>"],
    "key_negatives": ["<negative point 1>", "<negative point 2>"],
    "key_risks": ["<risk 1>", "<risk 2>"],
    "key_catalysts": ["<catalyst 1>", "<catalyst 2>"],
    "summary": "<2-3 sentence summary of overall tone and key takeaways>"
}}

Focus on:
1. Is management confident or hedging?
2. Are they raising, maintaining, or lowering expectations?
3. What risks did they highlight?
4. What growth drivers did they emphasize?

JSON response:"""

        response = self.llm.complete(prompt, max_tokens=800, temperature=0.2)

        if response:
            analysis = self._parse_llm_response(analysis, response)
        else:
            analysis = self._basic_earnings_analysis(ticker, transcript)

        return analysis

    def analyze_news(self, ticker: str,
                      headlines: List[str],
                      articles: List[str] = None) -> SentimentAnalysis:
        """
        Analyze news headlines and articles for sentiment.

        Args:
            ticker: Stock symbol
            headlines: List of news headlines
            articles: Optional list of full article texts

        Returns:
            SentimentAnalysis for news
        """
        analysis = SentimentAnalysis(
            ticker=ticker,
            as_of_date=date.today(),
            source_type='news',
        )

        if not headlines:
            analysis.summary = "No news data available"
            return analysis

        # Combine headlines
        headlines_text = "\n".join([f"- {h}" for h in headlines[:20]])
        analysis.source_text_length = len(headlines_text)

        if not self._check_llm():
            return self._basic_news_analysis(ticker, headlines)

        # Add article excerpts if available
        articles_text = ""
        if articles:
            for i, article in enumerate(articles[:5]):
                if article:
                    excerpt = article[:500] + "..." if len(article) > 500 else article
                    articles_text += f"\nArticle {i+1}: {excerpt}\n"

        prompt = f"""Analyze these news items for {ticker} stock. Determine the overall sentiment and key factors.

HEADLINES:
{headlines_text}

{f"ARTICLE EXCERPTS:{articles_text}" if articles_text else ""}

Respond in this exact JSON format:
{{
    "sentiment_score": <0-100, 50=neutral, higher=bullish>,
    "confidence": <0-100, how confident in this assessment>,
    "key_positives": ["<positive news 1>", "<positive news 2>"],
    "key_negatives": ["<negative news 1>", "<negative news 2>"],
    "key_catalysts": ["<upcoming catalyst 1>", "<upcoming catalyst 2>"],
    "summary": "<2-3 sentence summary of news sentiment>"
}}

Consider:
1. Is the news mostly positive, negative, or mixed?
2. Are there any major catalysts or risks mentioned?
3. How significant are these news items?

JSON response:"""

        response = self.llm.complete(prompt, max_tokens=600, temperature=0.2)

        if response:
            analysis = self._parse_llm_response(analysis, response)
        else:
            analysis = self._basic_news_analysis(ticker, headlines)

        return analysis

    def analyze_text(self, ticker: str, text: str,
                      context: str = "general") -> SentimentAnalysis:
        """
        General text analysis for any financial content.

        Args:
            ticker: Stock symbol
            text: Text to analyze
            context: Context hint ('earnings', 'news', 'filing', 'general')
        """
        analysis = SentimentAnalysis(
            ticker=ticker,
            as_of_date=date.today(),
            source_type=context,
            source_text_length=len(text),
        )

        if not text or len(text) < 50:
            analysis.summary = "Insufficient text data"
            return analysis

        if not self._check_llm():
            return self._basic_text_analysis(ticker, text)

        # Truncate if needed
        if len(text) > 8000:
            text = text[:4000] + "\n...[truncated]...\n" + text[-4000:]

        prompt = f"""Analyze this text about {ticker} for investment sentiment.

TEXT:
{text}

Respond in JSON format:
{{
    "sentiment_score": <0-100>,
    "key_positives": ["<point 1>", "<point 2>"],
    "key_negatives": ["<point 1>", "<point 2>"],
    "summary": "<brief summary>"
}}

JSON response:"""

        response = self.llm.complete(prompt, max_tokens=500, temperature=0.2)

        if response:
            analysis = self._parse_llm_response(analysis, response)
        else:
            analysis = self._basic_text_analysis(ticker, text)

        return analysis

    def _parse_llm_response(self, analysis: SentimentAnalysis,
                            response: str) -> SentimentAnalysis:
        """Parse LLM JSON response into analysis object."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if not json_match:
                # Try to find JSON with nested structures
                json_match = re.search(r'\{.*\}', response, re.DOTALL)

            if json_match:
                data = json.loads(json_match.group())

                # Parse sentiment score
                score = data.get('sentiment_score', 50)
                if isinstance(score, (int, float)):
                    analysis.sentiment_score = int(max(0, min(100, score)))

                # Determine sentiment level
                if analysis.sentiment_score >= 75:
                    analysis.sentiment_level = SentimentLevel.VERY_BULLISH
                elif analysis.sentiment_score >= 60:
                    analysis.sentiment_level = SentimentLevel.BULLISH
                elif analysis.sentiment_score >= 40:
                    analysis.sentiment_level = SentimentLevel.NEUTRAL
                elif analysis.sentiment_score >= 25:
                    analysis.sentiment_level = SentimentLevel.BEARISH
                else:
                    analysis.sentiment_level = SentimentLevel.VERY_BEARISH

                # Management metrics
                if 'management_confidence' in data:
                    analysis.management_confidence_score = int(data['management_confidence'])

                if 'hedging_pct' in data:
                    analysis.hedging_language_pct = float(data['hedging_pct'])

                if 'forward_guidance' in data:
                    analysis.forward_guidance_tone = str(data['forward_guidance'])

                if 'management_tone' in data:
                    tone = str(data['management_tone']).upper()
                    if tone in [t.value for t in ManagementTone]:
                        analysis.management_tone = ManagementTone(tone)

                # Key extractions
                if 'key_positives' in data:
                    analysis.key_positives = data['key_positives'][:5]
                if 'key_negatives' in data:
                    analysis.key_negatives = data['key_negatives'][:5]
                if 'key_risks' in data:
                    analysis.key_risks = data['key_risks'][:5]
                if 'key_catalysts' in data:
                    analysis.key_catalysts = data['key_catalysts'][:5]

                # Summary
                if 'summary' in data:
                    analysis.summary = str(data['summary'])[:500]

                # Confidence
                if 'confidence' in data:
                    analysis.confidence = float(data['confidence']) / 100
                else:
                    analysis.confidence = 0.7  # Default confidence for successful parse

        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse LLM response as JSON: {e}")
            # Try to extract sentiment from raw text
            analysis = self._extract_from_raw(analysis, response)
        except Exception as e:
            logger.debug(f"Error parsing LLM response: {e}")

        return analysis

    def _extract_from_raw(self, analysis: SentimentAnalysis,
                          text: str) -> SentimentAnalysis:
        """Extract sentiment from raw text when JSON parsing fails."""
        text_lower = text.lower()

        # Simple keyword scoring as fallback
        bullish_words = ['strong', 'growth', 'beat', 'exceed', 'raise', 'bullish',
                        'optimistic', 'confident', 'record', 'momentum']
        bearish_words = ['weak', 'miss', 'decline', 'lower', 'bearish', 'concern',
                        'challenge', 'difficult', 'risk', 'headwind']

        bullish_count = sum(1 for w in bullish_words if w in text_lower)
        bearish_count = sum(1 for w in bearish_words if w in text_lower)

        total = bullish_count + bearish_count
        if total > 0:
            analysis.sentiment_score = int(50 + (bullish_count - bearish_count) / total * 30)
            analysis.sentiment_score = max(20, min(80, analysis.sentiment_score))

        analysis.confidence = 0.3  # Low confidence for fallback
        analysis.summary = "Analysis based on keyword extraction (LLM parsing failed)"

        return analysis

    def _basic_earnings_analysis(self, ticker: str,
                                  transcript: str) -> SentimentAnalysis:
        """Fallback earnings analysis without LLM."""
        analysis = SentimentAnalysis(
            ticker=ticker,
            as_of_date=date.today(),
            source_type='earnings_call',
            source_text_length=len(transcript),
        )

        text_lower = transcript.lower()

        # Hedging language detection
        hedging_phrases = ['may', 'might', 'could', 'uncertain', 'challenging',
                          'headwind', 'difficult', 'cautious', 'conservative']
        hedging_count = sum(text_lower.count(phrase) for phrase in hedging_phrases)
        word_count = len(transcript.split())
        analysis.hedging_language_pct = min(50, hedging_count / max(1, word_count) * 1000)

        # Confidence indicators
        confident_phrases = ['confident', 'strong', 'record', 'exceeded', 'beat',
                            'momentum', 'growth', 'optimistic']
        confident_count = sum(text_lower.count(phrase) for phrase in confident_phrases)

        # Calculate sentiment
        net_sentiment = (confident_count - hedging_count) / max(1, word_count) * 500
        analysis.sentiment_score = int(50 + max(-30, min(30, net_sentiment)))

        if analysis.sentiment_score >= 60:
            analysis.sentiment_level = SentimentLevel.BULLISH
            analysis.management_tone = ManagementTone.CONFIDENT
        elif analysis.sentiment_score <= 40:
            analysis.sentiment_level = SentimentLevel.BEARISH
            analysis.management_tone = ManagementTone.CAUTIOUS
        else:
            analysis.sentiment_level = SentimentLevel.NEUTRAL
            analysis.management_tone = ManagementTone.CAUTIOUS

        analysis.confidence = 0.4
        analysis.summary = "Basic keyword analysis (LLM not available)"

        return analysis

    def _basic_news_analysis(self, ticker: str,
                              headlines: List[str]) -> SentimentAnalysis:
        """Fallback news analysis without LLM."""
        analysis = SentimentAnalysis(
            ticker=ticker,
            as_of_date=date.today(),
            source_type='news',
        )

        combined = " ".join(headlines).lower()

        bullish = ['upgrade', 'beat', 'raise', 'strong', 'growth', 'buy', 'bullish',
                  'record', 'surge', 'jump', 'soar', 'outperform']
        bearish = ['downgrade', 'miss', 'cut', 'weak', 'decline', 'sell', 'bearish',
                  'fall', 'drop', 'plunge', 'underperform', 'concern']

        bullish_count = sum(combined.count(w) for w in bullish)
        bearish_count = sum(combined.count(w) for w in bearish)

        if bullish_count + bearish_count > 0:
            analysis.sentiment_score = int(50 + (bullish_count - bearish_count) * 5)
            analysis.sentiment_score = max(20, min(80, analysis.sentiment_score))

        if analysis.sentiment_score >= 60:
            analysis.sentiment_level = SentimentLevel.BULLISH
        elif analysis.sentiment_score <= 40:
            analysis.sentiment_level = SentimentLevel.BEARISH

        analysis.confidence = 0.3
        analysis.summary = f"Basic analysis of {len(headlines)} headlines"

        return analysis

    def _basic_text_analysis(self, ticker: str,
                              text: str) -> SentimentAnalysis:
        """Fallback text analysis without LLM."""
        return self._basic_news_analysis(ticker, [text])


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources for a ticker.
    """

    def __init__(self, llm_url: str = None):
        self.analyzer = SentimentNLPAnalyzer(llm_url)

    def get_aggregated_sentiment(self, ticker: str,
                                  earnings_transcript: str = None,
                                  news_headlines: List[str] = None,
                                  news_articles: List[str] = None) -> AggregatedSentiment:
        """
        Get aggregated sentiment from all available sources.
        """
        result = AggregatedSentiment(
            ticker=ticker,
            as_of_date=date.today(),
        )

        analyses = []

        # Analyze earnings if available
        if earnings_transcript and len(earnings_transcript) > 100:
            earnings_analysis = self.analyzer.analyze_earnings_call(
                ticker, earnings_transcript
            )
            result.earnings_sentiment = earnings_analysis.sentiment_score
            analyses.append(earnings_analysis)

        # Analyze news if available
        if news_headlines:
            news_analysis = self.analyzer.analyze_news(
                ticker, news_headlines, news_articles
            )
            result.news_sentiment = news_analysis.sentiment_score
            analyses.append(news_analysis)

        result.analyses = analyses
        result.source_count = len(analyses)

        # Calculate overall sentiment
        if analyses:
            # Weighted average (earnings weighted more heavily)
            weights = []
            scores = []

            for a in analyses:
                if a.source_type == 'earnings_call':
                    weights.append(2.0)  # Earnings calls more important
                else:
                    weights.append(1.0)
                scores.append(a.sentiment_score)

            total_weight = sum(weights)
            if total_weight > 0:
                result.overall_score = int(
                    sum(s * w for s, w in zip(scores, weights)) / total_weight
                )

        # Determine overall sentiment level
        if result.overall_score >= 70:
            result.overall_sentiment = SentimentLevel.VERY_BULLISH
        elif result.overall_score >= 55:
            result.overall_sentiment = SentimentLevel.BULLISH
        elif result.overall_score >= 45:
            result.overall_sentiment = SentimentLevel.NEUTRAL
        elif result.overall_score >= 30:
            result.overall_sentiment = SentimentLevel.BEARISH
        else:
            result.overall_sentiment = SentimentLevel.VERY_BEARISH

        # Generate signal
        if result.overall_score >= 65:
            result.signal = "BULLISH"
            result.signal_strength = min(100, result.overall_score + 10)
        elif result.overall_score <= 35:
            result.signal = "BEARISH"
            result.signal_strength = min(100, 100 - result.overall_score + 10)
        else:
            result.signal = "NEUTRAL"
            result.signal_strength = 50

        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_analyzer = None
_aggregator = None

def get_sentiment_analyzer(llm_url: str = None) -> SentimentNLPAnalyzer:
    """Get singleton analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentNLPAnalyzer(llm_url)
    return _analyzer


def get_sentiment_aggregator(llm_url: str = None) -> SentimentAggregator:
    """Get singleton aggregator instance."""
    global _aggregator
    if _aggregator is None:
        _aggregator = SentimentAggregator(llm_url)
    return _aggregator


def analyze_earnings_sentiment(ticker: str, transcript: str,
                                llm_url: str = None) -> SentimentAnalysis:
    """
    Analyze earnings call transcript sentiment.

    Usage:
        analysis = analyze_earnings_sentiment('NVDA', transcript_text)
        print(f"Sentiment: {analysis.sentiment_level.value}")
        print(f"Management tone: {analysis.management_tone.value}")
    """
    analyzer = get_sentiment_analyzer(llm_url)
    return analyzer.analyze_earnings_call(ticker, transcript)


def analyze_news_sentiment(ticker: str, headlines: List[str],
                           articles: List[str] = None,
                           llm_url: str = None) -> SentimentAnalysis:
    """
    Analyze news sentiment for a ticker.

    Usage:
        headlines = ["NVDA beats estimates", "AI demand surges"]
        analysis = analyze_news_sentiment('NVDA', headlines)
        print(f"News sentiment: {analysis.sentiment_score}/100")
    """
    analyzer = get_sentiment_analyzer(llm_url)
    return analyzer.analyze_news(ticker, headlines, articles)


def get_ticker_sentiment(ticker: str,
                          earnings_transcript: str = None,
                          news_headlines: List[str] = None,
                          llm_url: str = None) -> AggregatedSentiment:
    """
    Get aggregated sentiment for a ticker.

    Usage:
        sentiment = get_ticker_sentiment('NVDA',
            earnings_transcript=transcript,
            news_headlines=['headline1', 'headline2']
        )
        print(f"Overall: {sentiment.overall_sentiment.value}")
        print(f"Signal: {sentiment.signal}")
    """
    aggregator = get_sentiment_aggregator(llm_url)
    return aggregator.get_aggregated_sentiment(
        ticker, earnings_transcript, news_headlines
    )


def is_llm_available(llm_url: str = None) -> bool:
    """
    Check if local LLM is available.

    Usage:
        if is_llm_available():
            print("LLM ready for analysis")
        else:
            print("Using fallback keyword analysis")
    """
    client = LocalLLMClient(llm_url)
    return client.is_available()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Sentiment NLP...")

    # Check LLM availability
    print(f"LLM Available: {is_llm_available()}")

    # Test with sample headlines
    headlines = [
        "NVIDIA beats Q3 estimates, raises guidance",
        "AI chip demand continues to surge",
        "Data center revenue up 200% year over year",
        "Supply constraints easing, management says",
    ]

    analysis = analyze_news_sentiment('NVDA', headlines)

    print(f"\nNVDA News Sentiment:")
    print(f"  Score: {analysis.sentiment_score}/100")
    print(f"  Level: {analysis.sentiment_level.value}")
    print(f"  Summary: {analysis.summary}")

    if analysis.key_positives:
        print(f"  Positives: {analysis.key_positives}")