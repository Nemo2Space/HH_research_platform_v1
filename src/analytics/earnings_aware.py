"""
Earnings-Aware Analysis Module

Handles the special case of stocks near earnings:
- Pre-earnings (within 5 days): Enhanced IES calculation
- Post-earnings (within 5 days): Fetch results, recalculate sentiment, get reaction

This module ensures that when NKE reports bad earnings yesterday,
the system KNOWS about it and reflects it in all signals.

Author: Alpha Research Platform
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, date, timedelta
from enum import Enum
import yfinance as yf
import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EarningsResult:
    """Actual earnings results after report."""
    ticker: str
    report_date: date

    # EPS
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_surprise: Optional[float] = None
    eps_surprise_pct: Optional[float] = None

    # Revenue
    revenue_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_surprise: Optional[float] = None
    revenue_surprise_pct: Optional[float] = None

    # Guidance (if available)
    guidance_direction: str = ""  # "raised", "lowered", "maintained", "withdrawn"

    # Market reaction
    pre_report_close: float = 0
    post_report_price: float = 0  # Current/pre-market/after-hours
    reaction_pct: float = 0
    gap_pct: float = 0

    # Quality assessment
    beat_eps: bool = False
    beat_revenue: bool = False
    overall_result: str = ""  # "BEAT", "MISS", "MIXED", "INLINE"

    # Data quality
    data_complete: bool = False


class EarningsAwareAnalyzer:
    """
    Analyzes stocks with earnings context.

    Key insight: A stock that just reported earnings needs DIFFERENT analysis
    than a stock with earnings far away.
    """

    def __init__(self):
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=30)  # Short cache for earnings data

    def get_earnings_context(self, ticker: str) -> Dict[str, Any]:
        """
        Get complete earnings context for a ticker.

        Returns dict with:
        - earnings_status: 'pre', 'post', or 'none'
        - days_to_earnings: int (negative if post)
        - earnings_result: EarningsResult (if post-earnings)
        - current_price: float (including pre-market/after-hours)
        - news_queries: list of search queries to use
        """

        context = {
            'earnings_status': 'none',
            'days_to_earnings': 999,
            'earnings_date': None,
            'earnings_result': None,
            'current_price': 0,
            'previous_close': 0,
            'price_change_pct': 0,
            'session': 'market',
            'news_queries': [f"{ticker} stock news"],
            'sentiment_should_refresh': False,
        }

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get current/extended price
            context.update(self._get_current_price(stock, info))

            # Get earnings date
            earnings_date, days_to = self._get_earnings_date(ticker, stock)
            context['earnings_date'] = earnings_date
            context['days_to_earnings'] = days_to

            if days_to is None or abs(days_to) > 10:
                context['earnings_status'] = 'none'
                return context

            if days_to < 0:
                # POST-EARNINGS
                context['earnings_status'] = 'post'
                context['sentiment_should_refresh'] = True

                # Get actual earnings results
                result = self._get_earnings_results(ticker, stock, earnings_date)
                context['earnings_result'] = result

                # Build earnings-focused news queries
                company_name = info.get('shortName', info.get('longName', ticker))
                context['news_queries'] = self._build_post_earnings_queries(ticker, company_name, result)

            elif days_to <= 5:
                # PRE-EARNINGS (action window)
                context['earnings_status'] = 'pre'
                company_name = info.get('shortName', info.get('longName', ticker))
                context['news_queries'] = self._build_pre_earnings_queries(ticker, company_name)

            else:
                # PRE-EARNINGS (monitoring window)
                context['earnings_status'] = 'pre_far'

        except Exception as e:
            logger.error(f"Error getting earnings context for {ticker}: {e}")

        return context

    def _get_current_price(self, stock, info: dict) -> dict:
        """Get current price including extended hours."""
        result = {
            'current_price': 0,
            'previous_close': 0,
            'price_change_pct': 0,
            'session': 'market',
        }

        try:
            result['previous_close'] = info.get('previousClose') or info.get('regularMarketPreviousClose', 0)

            # Check pre-market
            pre_price = info.get('preMarketPrice')
            if pre_price and pre_price > 0:
                result['current_price'] = pre_price
                result['session'] = 'pre-market'
                if result['previous_close'] > 0:
                    result['price_change_pct'] = ((pre_price - result['previous_close']) / result['previous_close']) * 100
                return result

            # Check after-hours
            post_price = info.get('postMarketPrice')
            if post_price and post_price > 0:
                result['current_price'] = post_price
                result['session'] = 'after-hours'
                if result['previous_close'] > 0:
                    result['price_change_pct'] = ((post_price - result['previous_close']) / result['previous_close']) * 100
                return result

            # Regular market price
            current = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            result['current_price'] = current or result['previous_close']
            result['session'] = 'market'
            if result['previous_close'] > 0 and current:
                result['price_change_pct'] = ((current - result['previous_close']) / result['previous_close']) * 100

        except Exception as e:
            logger.debug(f"Price error: {e}")

        return result

    def _get_earnings_date(self, ticker: str, stock) -> Tuple[Optional[date], Optional[int]]:
        """Get earnings date - checks both future and recent past."""

        # Try database first
        try:
            from src.db.connection import get_engine
            engine = get_engine()

            # Check upcoming
            df = pd.read_sql(f"""
                SELECT earnings_date FROM earnings_calendar 
                WHERE ticker = '{ticker}' AND earnings_date >= CURRENT_DATE
                ORDER BY earnings_date LIMIT 1
            """, engine)

            if not df.empty and df.iloc[0]['earnings_date']:
                ed = pd.to_datetime(df.iloc[0]['earnings_date']).date()
                days = (ed - date.today()).days
                return ed, days

            # Check recent past (within 7 days)
            df_past = pd.read_sql(f"""
                SELECT earnings_date FROM earnings_calendar 
                WHERE ticker = '{ticker}' 
                AND earnings_date >= CURRENT_DATE - INTERVAL '7 days'
                AND earnings_date < CURRENT_DATE
                ORDER BY earnings_date DESC LIMIT 1
            """, engine)

            if not df_past.empty and df_past.iloc[0]['earnings_date']:
                ed = pd.to_datetime(df_past.iloc[0]['earnings_date']).date()
                days = (ed - date.today()).days  # Will be negative
                return ed, days

        except Exception as e:
            logger.debug(f"DB earnings date error: {e}")

        # Fallback to yfinance calendar
        try:
            cal = stock.calendar
            if cal is not None:
                if isinstance(cal, dict) and 'Earnings Date' in cal:
                    ed = pd.to_datetime(cal['Earnings Date']).date()
                    days = (ed - date.today()).days
                    return ed, days
        except:
            pass

        # Try earnings history for recent
        try:
            hist = stock.earnings_history
            if hist is not None and not hist.empty:
                # Most recent earnings
                latest_date = pd.to_datetime(hist.index[0]).date()
                days = (latest_date - date.today()).days
                if days >= -7:  # Within last week
                    return latest_date, days
        except:
            pass

        return None, None

    def _get_earnings_results(self, ticker: str, stock, report_date: date) -> EarningsResult:
        """Get actual earnings results."""
        result = EarningsResult(ticker=ticker, report_date=report_date)

        try:
            # Get earnings history
            hist = stock.earnings_history
            if hist is not None and not hist.empty:
                latest = hist.iloc[0]

                # EPS
                result.eps_actual = latest.get('epsActual')
                result.eps_estimate = latest.get('epsEstimate')
                if result.eps_actual and result.eps_estimate:
                    result.eps_surprise = result.eps_actual - result.eps_estimate
                    if result.eps_estimate != 0:
                        result.eps_surprise_pct = (result.eps_surprise / abs(result.eps_estimate)) * 100
                    result.beat_eps = result.eps_actual > result.eps_estimate

            # Get quarterly financials for revenue
            try:
                quarterly = stock.quarterly_financials
                if quarterly is not None and not quarterly.empty:
                    if 'Total Revenue' in quarterly.index:
                        result.revenue_actual = quarterly.loc['Total Revenue'].iloc[0]
            except:
                pass

            # Get price reaction
            try:
                price_hist = stock.history(period="10d")
                if len(price_hist) >= 2:
                    # Find the earnings day
                    for i in range(len(price_hist) - 1):
                        dt = price_hist.index[i].date()
                        if dt == report_date or dt == report_date + timedelta(days=1):
                            result.pre_report_close = price_hist['Close'].iloc[max(0, i-1)]
                            result.post_report_price = price_hist['Close'].iloc[i]
                            if result.pre_report_close > 0:
                                result.reaction_pct = ((result.post_report_price - result.pre_report_close) / result.pre_report_close) * 100
                            break

                    # If we couldn't match date, use most recent change
                    if result.reaction_pct == 0:
                        result.pre_report_close = price_hist['Close'].iloc[-2]
                        result.post_report_price = price_hist['Close'].iloc[-1]
                        if result.pre_report_close > 0:
                            result.reaction_pct = ((result.post_report_price - result.pre_report_close) / result.pre_report_close) * 100
            except Exception as e:
                logger.debug(f"Price history error: {e}")

            # Determine overall result
            if result.beat_eps and result.beat_revenue:
                result.overall_result = "BEAT"
            elif not result.beat_eps and not result.beat_revenue:
                result.overall_result = "MISS"
            elif result.beat_eps or result.beat_revenue:
                result.overall_result = "MIXED"
            else:
                result.overall_result = "INLINE"

            result.data_complete = result.eps_actual is not None

        except Exception as e:
            logger.error(f"Error getting earnings results for {ticker}: {e}")

        return result

    def _build_post_earnings_queries(self, ticker: str, company_name: str, result: EarningsResult) -> List[str]:
        """Build search queries for post-earnings news."""
        queries = []

        # Primary earnings queries
        queries.append(f"{ticker} earnings results")
        queries.append(f"{company_name} quarterly earnings")
        queries.append(f"{ticker} Q4 earnings")  # or appropriate quarter

        # Result-specific queries
        if result.overall_result == "MISS":
            queries.append(f"{ticker} earnings miss")
            queries.append(f"{company_name} disappoints")
        elif result.overall_result == "BEAT":
            queries.append(f"{ticker} earnings beat")
            queries.append(f"{company_name} beats expectations")

        # Guidance queries
        queries.append(f"{ticker} guidance outlook")

        # Reaction queries
        if result.reaction_pct and abs(result.reaction_pct) > 5:
            if result.reaction_pct < 0:
                queries.append(f"{ticker} stock falls drops")
            else:
                queries.append(f"{ticker} stock rises surges")

        # General news
        queries.append(f"{ticker} stock news")

        return queries

    def _build_pre_earnings_queries(self, ticker: str, company_name: str) -> List[str]:
        """Build search queries for pre-earnings."""
        return [
            f"{ticker} earnings preview",
            f"{company_name} earnings expectations",
            f"{ticker} earnings whisper",
            f"{ticker} stock news",
        ]

    def refresh_sentiment_for_earnings(self, ticker: str, context: Dict) -> Dict[str, Any]:
        """
        Refresh sentiment analysis specifically for earnings context.

        This should be called when earnings_status is 'post' to ensure
        sentiment reflects the actual earnings news, not stale pre-earnings sentiment.
        """
        from src.data.news import NewsCollector
        from src.screener.sentiment import SentimentAnalyzer

        results = {
            'news_collected': 0,
            'sentiment_updated': False,
            'new_sentiment_score': None,
        }

        try:
            nc = NewsCollector()
            sa = SentimentAnalyzer()

            # Force fresh news collection with earnings-specific queries
            all_articles = []

            for query in context.get('news_queries', []):
                try:
                    # Use AI search with earnings query
                    articles = nc.collect_ai_search(ticker, company_name=query.replace(ticker, '').strip())
                    for a in articles:
                        a['ticker'] = ticker
                    all_articles.extend(articles)
                except:
                    pass

            # Also get from standard sources with force refresh
            standard_articles = nc.collect_all_news(ticker, days_back=3, force_refresh=True)
            all_articles.extend(standard_articles)

            # Deduplicate
            seen = set()
            unique_articles = []
            for a in all_articles:
                title = a.get('title', '').lower()[:50]
                if title and title not in seen:
                    seen.add(title)
                    unique_articles.append(a)

            results['news_collected'] = len(unique_articles)

            if unique_articles:
                # Save to database
                nc.save_articles(unique_articles)

                # Re-run sentiment analysis
                sentiment_result = sa.analyze_ticker_sentiment(ticker, unique_articles)
                results['sentiment_updated'] = True
                results['new_sentiment_score'] = sentiment_result.get('sentiment_score')

        except Exception as e:
            logger.error(f"Error refreshing sentiment for {ticker}: {e}")

        return results


def get_earnings_aware_data(ticker: str) -> Dict[str, Any]:
    """
    Convenience function to get all earnings-aware data for a ticker.

    Use this in Signal Hub when loading a ticker's deep dive.
    """
    analyzer = EarningsAwareAnalyzer()
    context = analyzer.get_earnings_context(ticker)

    # If post-earnings and sentiment should refresh, do it
    if context.get('sentiment_should_refresh'):
        refresh_result = analyzer.refresh_sentiment_for_earnings(ticker, context)
        context['sentiment_refresh_result'] = refresh_result

    return context


def analyze_ticker_with_earnings_context(ticker: str) -> Dict[str, Any]:
    """
    Full analysis for Run Analysis button - handles earnings specially.

    Returns dict with all analysis results including earnings context.
    """
    from src.data.news import NewsCollector
    from src.screener.sentiment import SentimentAnalyzer

    analyzer = EarningsAwareAnalyzer()
    context = analyzer.get_earnings_context(ticker)

    results = {
        'ticker': ticker,
        'earnings_context': context,
        'news_count': 0,
        'sentiment_score': None,
        'current_price': context.get('current_price'),
        'price_change_pct': context.get('price_change_pct'),
        'session': context.get('session'),
    }

    try:
        nc = NewsCollector()
        sa = SentimentAnalyzer()

        # Collect news with appropriate queries
        all_articles = []

        for query in context.get('news_queries', [f"{ticker} stock news"]):
            try:
                # Standard collection
                articles = nc.collect_all_news(ticker, days_back=5, force_refresh=True)
                all_articles.extend(articles)

                # If earnings query, also do AI search
                if 'earnings' in query.lower():
                    ai_articles = nc.collect_ai_search(ticker, query)
                    for a in ai_articles:
                        a['ticker'] = ticker
                    all_articles.extend(ai_articles)
            except:
                pass

        # Deduplicate
        seen = set()
        unique_articles = []
        for a in all_articles:
            title = a.get('title', '').lower()[:50]
            if title and title not in seen:
                seen.add(title)
                unique_articles.append(a)

        results['news_count'] = len(unique_articles)

        if unique_articles:
            nc.save_articles(unique_articles)

            # Sentiment analysis
            sentiment_result = sa.analyze_ticker_sentiment(ticker, unique_articles)
            results['sentiment_score'] = sentiment_result.get('sentiment_score')
            results['sentiment_signal'] = sentiment_result.get('signal')

        # Add earnings result summary
        if context.get('earnings_result'):
            er = context['earnings_result']
            results['earnings_summary'] = {
                'eps_actual': er.eps_actual,
                'eps_estimate': er.eps_estimate,
                'eps_surprise_pct': er.eps_surprise_pct,
                'reaction_pct': er.reaction_pct,
                'overall_result': er.overall_result,
            }

    except Exception as e:
        logger.error(f"Error in earnings-aware analysis for {ticker}: {e}")
        results['error'] = str(e)

    return results