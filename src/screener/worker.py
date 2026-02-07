import time
from datetime import date
from typing import Optional, Dict, Any, Callable
import pandas as pd

from src.db.repository import Repository
from src.data.news import NewsCollector
from src.screener.sentiment import SentimentAnalyzer
from src.screener.signals import SignalGenerator, calculate_composite_score, calculate_likelihood_score
from src.utils.logging import get_logger
from src.data.insider import InsiderDataFetcher
from src.data.finviz import FinvizDataFetcher
from src.screener.gap_analysis import GapAnalyzer

# Earnings Intelligence Integration
try:
    from src.analytics.earnings_intelligence import (
        apply_earnings_adjustment,
        get_position_scale,
        should_flag_for_earnings,
    )

    EARNINGS_INTELLIGENCE_AVAILABLE = True
except ImportError:
    EARNINGS_INTELLIGENCE_AVAILABLE = False

logger = get_logger(__name__)


class ScreenerWorker:
    """
    Main screener worker that processes tickers and generates signals.
    """

    def __init__(self, repository: Optional[Repository] = None,
                 use_llm: bool = True):
        self.repo = repository or Repository()
        self.news_collector = NewsCollector(repository=self.repo)
        self.sentiment_analyzer = SentimentAnalyzer() if use_llm else None
        self.signal_generator = SignalGenerator()
        self.use_llm = use_llm
        self.insider_fetcher = InsiderDataFetcher(repository=self.repo)
        self.finviz_fetcher = FinvizDataFetcher(repository=self.repo)
        self.gap_analyzer = GapAnalyzer(repository=self.repo)

    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Get fundamental data for a ticker."""
        query = """
                SELECT * \
                FROM fundamentals
                WHERE ticker = %(ticker)s
                ORDER BY date DESC LIMIT 1 \
                """
        df = pd.read_sql(query, self.repo.engine, params={"ticker": ticker})

        if len(df) > 0:
            return df.iloc[0].to_dict()
        return {}

    def get_analyst_data(self, ticker: str) -> Dict[str, Any]:
        """Get analyst ratings and price targets."""
        result = {}

        # Analyst ratings
        query = """
                SELECT * \
                FROM analyst_ratings
                WHERE ticker = %(ticker)s
                ORDER BY date DESC LIMIT 1 \
                """
        df = pd.read_sql(query, self.repo.engine, params={"ticker": ticker})
        if len(df) > 0:
            row = df.iloc[0]
            result['analyst_total'] = row.get('analyst_total', 0)
            result['analyst_positive'] = row.get('analyst_positive', 0)
            result['analyst_positivity'] = row.get('analyst_positivity', 50)

        # Price targets
        query = """
                SELECT * \
                FROM price_targets
                WHERE ticker = %(ticker)s
                ORDER BY date DESC LIMIT 1 \
                """
        df = pd.read_sql(query, self.repo.engine, params={"ticker": ticker})
        if len(df) > 0:
            row = df.iloc[0]
            result['target_mean'] = row.get('target_mean')
            result['target_high'] = row.get('target_high')
            result['target_low'] = row.get('target_low')
            result['target_upside_pct'] = row.get('target_upside_pct', 0)

        return result

    def get_news_articles(self, ticker: str, days_back: int = 7) -> list:
        """Get recent news articles for a ticker from database."""
        query = """
                SELECT * \
                FROM news_articles
                WHERE ticker = %(ticker)s
                  AND published_at >= NOW() - INTERVAL '%(days)s days'
                ORDER BY published_at DESC \
                """
        try:
            df = pd.read_sql(query, self.repo.engine, params={"ticker": ticker, "days": days_back})
            return df.to_dict('records')
        except:
            return []

    def calculate_fundamental_score(self, fundamentals: Dict[str, Any]) -> int:
        """Calculate fundamental score from raw data."""
        if not fundamentals:
            return 50

        score = 50

        # PE ratio (lower is better, but not too low)
        pe = fundamentals.get('pe_ratio')
        if pe:
            if 5 < pe < 15:
                score += 10
            elif 15 <= pe < 25:
                score += 5
            elif pe >= 40:
                score -= 10

        # Profit margin
        margin = fundamentals.get('profit_margin')
        if margin:
            if margin > 0.2:
                score += 10
            elif margin > 0.1:
                score += 5
            elif margin < 0:
                score -= 10

        # ROE
        roe = fundamentals.get('roe')
        if roe:
            if roe > 0.2:
                score += 10
            elif roe > 0.1:
                score += 5
            elif roe < 0:
                score -= 5

        # Revenue growth
        growth = fundamentals.get('revenue_growth')
        if growth:
            if growth > 0.2:
                score += 10
            elif growth > 0.1:
                score += 5
            elif growth < 0:
                score -= 5

        return max(0, min(100, score))

    def calculate_growth_score(self, fundamentals: Dict[str, Any]) -> int:
        """Calculate growth score."""
        if not fundamentals:
            return 50

        score = 50

        revenue_growth = fundamentals.get('revenue_growth')
        if revenue_growth:
            score += min(25, int(revenue_growth * 100))

        earnings_growth = fundamentals.get('earnings_growth')
        if earnings_growth:
            score += min(25, int(earnings_growth * 100))

        return max(0, min(100, score))

    def calculate_dividend_score(self, fundamentals: Dict[str, Any]) -> int:
        """Calculate dividend score."""
        if not fundamentals:
            return 50

        score = 50

        div_yield = fundamentals.get('dividend_yield')
        if div_yield:
            if div_yield > 0.04:
                score += 20
            elif div_yield > 0.02:
                score += 10
        else:
            score -= 10  # No dividend

        payout = fundamentals.get('dividend_payout_ratio')
        if payout:
            if 0.3 < payout < 0.6:
                score += 10  # Sustainable
            elif payout > 0.8:
                score -= 10  # Too high

        return max(0, min(100, score))

    def process_ticker(self, ticker: str, force_news: bool = False,
                       days_back: int = 7) -> Dict[str, Any]:
        """
        Process a single ticker: gather data, calculate scores, generate signal.

        Args:
            ticker: Stock ticker symbol
            force_news: If True, force fresh news collection (bypass cache)
            days_back: Days of news to analyze

        Returns:
            Dict with all scores and signal
        """
        from src.screener.technicals import TechnicalAnalyzer

        logger.info(f"Processing {ticker}...")

        # 1. Get fundamental data
        fundamentals = self.get_fundamentals(ticker)

        # 2. Get analyst data
        analyst_data = self.get_analyst_data(ticker)

        # 3. Calculate fundamental scores
        fundamental_score = self.calculate_fundamental_score(fundamentals)
        growth_score = self.calculate_growth_score(fundamentals)
        dividend_score = self.calculate_dividend_score(fundamentals)

        # 4. Technical analysis
        tech_analyzer = TechnicalAnalyzer(self.repo)
        tech_data = tech_analyzer.analyze_ticker(ticker)
        technical_score = tech_data.get('technical_score')

        # 5. Get news with smart caching
        # collect_all_news now handles caching internally:
        # - If force_news=True: always fetch fresh
        # - If news is < 6h old: use cached
        # - If news is > 6h old: fetch fresh
        articles = self.news_collector.collect_all_news(
            ticker,
            days_back=days_back,
            force_refresh=force_news
        )
        self.news_collector.save_articles(articles)

        # 6. Analyze sentiment (with caching)
        sentiment_score = None
        sentiment_weighted = None

        # Check if we have cached sentiment from today
        cached_sentiment = self.get_cached_sentiment(ticker)

        if cached_sentiment and cached_sentiment['sentiment_score'] not in (None, 0):
            # Use cached if article count is similar (within 10%)
            cached_count = cached_sentiment.get('article_count', 0)
            current_count = len(articles)

            if cached_count > 0 and abs(current_count - cached_count) / cached_count < 0.1:
                sentiment_score = cached_sentiment['sentiment_score']
                sentiment_weighted = cached_sentiment.get('sentiment_weighted', sentiment_score)
                logger.info(
                    f"{ticker}: Using cached sentiment={sentiment_score} (articles: {cached_count}→{current_count})")
            else:
                # News changed significantly, re-analyze
                if self.use_llm and self.sentiment_analyzer and articles:
                    sentiment_result = self.sentiment_analyzer.analyze_ticker_sentiment(ticker, articles)
                    sentiment_score = sentiment_result['sentiment_score']
                    sentiment_weighted = sentiment_result['sentiment_weighted']
        else:
            # No valid cache, analyze fresh
            if self.use_llm and self.sentiment_analyzer and articles:
                sentiment_result = self.sentiment_analyzer.analyze_ticker_sentiment(ticker, articles)
                sentiment_score = sentiment_result['sentiment_score']
                sentiment_weighted = sentiment_result['sentiment_weighted']

        # 7. Calculate target upside (analyst targets vs current price)
        target_upside = analyst_data.get('target_upside_pct', 0) or 0

        # 7b. Gap analysis (comprehensive technical analysis)
        gap_result = self.gap_analyzer.analyze(ticker)
        gap_score = gap_result.gap_score
        gap_type = gap_result.gap_type

        # 7c. Get insider signal
        insider_data = self.insider_fetcher.get_insider_signal(ticker, days_back=30)
        insider_signal = insider_data.get('insider_signal')

        # 7d. Get institutional signal from Finviz
        finviz_data = self.finviz_fetcher.get_institutional_signal(ticker)
        institutional_signal = finviz_data.get('institutional_signal')

        # 8. Build scores dict
        scores = {
            'ticker': ticker,
            'sentiment_score': sentiment_score,
            'sentiment_weighted': sentiment_weighted,
            'fundamental_score': fundamental_score,
            'growth_score': growth_score,
            'dividend_score': dividend_score,
            'technical_score': technical_score,
            'gap_score': gap_score,
            'gap_type': gap_type,
            'analyst_positivity': analyst_data.get('analyst_positivity'),
            'target_upside_pct': target_upside,
            'article_count': len(articles),
            'insider_signal': insider_signal,
            'institutional_signal': institutional_signal,

            # Technical details
            'rsi': tech_data.get('rsi'),
            'macd_signal': tech_data.get('macd_signal', 'neutral'),
            'trend': tech_data.get('trend', 'neutral'),
            'momentum_5d': tech_data.get('momentum_5d', 0),
            'volatility': tech_data.get('volatility', 0),
        }

        # 9. Calculate derived scores
        scores['likelihood_score'] = calculate_likelihood_score(scores)
        scores['composite_score'] = calculate_composite_score(scores)
        scores['total_score'] = scores['composite_score']

        # ============================================================
        # 9b. EARNINGS INTELLIGENCE ADJUSTMENT (NEW!)
        # ============================================================
        earnings_adjustment = 0
        earnings_risk_flags = []
        position_scale = 1.0

        if EARNINGS_INTELLIGENCE_AVAILABLE:
            try:
                # Apply earnings-based score adjustment
                adjusted_score, risk_flags = apply_earnings_adjustment(
                    scores['total_score'], ticker
                )
                earnings_adjustment = adjusted_score - scores['total_score']
                earnings_risk_flags = risk_flags

                # Get position scale recommendation
                position_scale = get_position_scale(ticker)

                # Update total score with earnings adjustment
                scores['total_score'] = adjusted_score
                scores['composite_score'] = adjusted_score

                # Store earnings data in scores
                scores['earnings_adjustment'] = earnings_adjustment
                scores['earnings_risk_flags'] = earnings_risk_flags
                scores['position_scale'] = position_scale

                if earnings_adjustment != 0 or risk_flags:
                    logger.info(f"{ticker}: Earnings adjustment={earnings_adjustment:+d}, "
                                f"flags={risk_flags}, scale={position_scale:.0%}")

            except Exception as e:
                logger.debug(f"{ticker}: Earnings intelligence error: {e}")
        # ============================================================

        # 10. Generate signal
        signal = self.signal_generator.generate_signal(scores)

        # 11. Save to database
        self.repo.save_screener_score(ticker, date.today(), scores)
        self.repo.save_signal(ticker, date.today(), signal.to_dict())

        logger.info(
            f"{ticker}: Score={scores['total_score']}, Signal={signal.type}, Tech={technical_score}, RSI={tech_data.get('rsi', 0):.1f}")

        return {**scores, 'signal': signal.to_dict()}

    def get_cached_sentiment(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get cached sentiment from today's screener scores."""
        query = """
                SELECT sentiment_score, sentiment_weighted, article_count
                FROM screener_scores
                WHERE ticker = %(ticker)s
                  AND date = CURRENT_DATE
                ORDER BY created_at DESC
                    LIMIT 1 \
                """
        try:
            df = pd.read_sql(query, self.repo.engine, params={"ticker": ticker})
            if len(df) > 0:
                return df.iloc[0].to_dict()
        except Exception as e:
            logger.debug(f"{ticker}: Cache lookup failed: {e}")
        return None

    def run_full_screen(self, force_news: bool = False,
                        delay: float = 0.5,
                        progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run screener on all tickers in universe.

        Args:
            force_news: If True, force fresh news collection for all tickers
            delay: Delay between tickers (rate limiting)
            progress_callback: Optional callback(current, total, ticker)

        Returns:
            Summary dict with results
        """
        tickers = self.repo.get_universe()
        total = len(tickers)

        # Update status
        self.repo.update_system_status('screener', 'running', progress_pct=0)

        results = {
            'total': total,
            'processed': 0,
            'signals': {},
            'errors': [],
            'earnings_flags': {},  # NEW: Track earnings flags
        }

        start_time = time.time()

        for i, ticker in enumerate(tickers):
            try:
                if progress_callback:
                    progress_callback(i + 1, total, ticker)

                pct = int((i + 1) / total * 100)
                self.repo.update_system_status(
                    'screener', 'running',
                    progress_pct=pct,
                    progress_message=f'Processing {ticker}'
                )

                result = self.process_ticker(ticker, force_news=force_news)
                results['processed'] += 1
                results['signals'][ticker] = result['signal']['type']

                # Track earnings flags
                if result.get('earnings_risk_flags'):
                    results['earnings_flags'][ticker] = result['earnings_risk_flags']

            except Exception as e:
                logger.error(f"{ticker}: Error - {e}")
                results['errors'].append({'ticker': ticker, 'error': str(e)})

            if delay > 0 and i < total - 1:
                time.sleep(delay)

        elapsed = time.time() - start_time

        # Update status
        self.repo.update_system_status(
            'screener', 'completed',
            progress_message=f'Done: {results["processed"]}/{total} in {elapsed:.1f}s'
        )

        # Summary
        signal_counts = {}
        for sig in results['signals'].values():
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        results['signal_summary'] = signal_counts
        results['elapsed_seconds'] = elapsed

        logger.info(f"Screener complete: {results['processed']}/{total} in {elapsed:.1f}s")
        logger.info(f"Signal summary: {signal_counts}")

        # Log earnings flags summary
        if results['earnings_flags']:
            logger.info(f"Earnings flags: {len(results['earnings_flags'])} tickers flagged")

        return results