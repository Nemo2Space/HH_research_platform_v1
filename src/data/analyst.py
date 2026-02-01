"""
Alpha Platform - Analyst Data Ingestion

Fetches analyst ratings and price targets using yfinance.
"""

import yfinance as yf
from datetime import date
from typing import Optional, Dict, Any
import time

from src.db.repository import Repository
from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AnalystDataIngester:
    """Fetches and stores analyst ratings and price targets."""

    def __init__(self, repository: Optional[Repository] = None):
        self.repo = repository or Repository()

    def fetch_analyst_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch analyst ratings and price targets for a ticker.

        Returns:
            Dict with analyst data or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                return None

            # Get recommendation counts
            strong_buy = info.get('recommendationKey', '')

            # Get individual counts if available
            analyst_total = info.get('numberOfAnalystOpinions', 0) or 0

            # Price targets
            target_mean = info.get('targetMeanPrice')
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            target_median = info.get('targetMedianPrice')
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')

            # Calculate upside
            target_upside_pct = None
            if target_mean and current_price and current_price > 0:
                target_upside_pct = ((target_mean - current_price) / current_price) * 100

            # Try to get recommendation breakdown
            try:
                recommendations = stock.recommendations
                if recommendations is not None and len(recommendations) > 0:
                    # Get most recent month's data
                    recent = recommendations.tail(1).iloc[0] if len(recommendations) > 0 else None
                    if recent is not None:
                        strong_buy_count = int(recent.get('strongBuy', 0) or 0)
                        buy_count = int(recent.get('buy', 0) or 0)
                        hold_count = int(recent.get('hold', 0) or 0)
                        sell_count = int(recent.get('sell', 0) or 0)
                        strong_sell_count = int(recent.get('strongSell', 0) or 0)

                        analyst_total = strong_buy_count + buy_count + hold_count + sell_count + strong_sell_count
                        analyst_positive = strong_buy_count + buy_count
                        analyst_positivity = (analyst_positive / analyst_total * 100) if analyst_total > 0 else None

                        return {
                            'ticker': ticker,
                            'date': date.today(),
                            'analyst_total': analyst_total,
                            'analyst_strong_buy': strong_buy_count,
                            'analyst_buy': buy_count,
                            'analyst_hold': hold_count,
                            'analyst_sell': sell_count,
                            'analyst_strong_sell': strong_sell_count,
                            'analyst_positive': analyst_positive,
                            'analyst_positivity': analyst_positivity,
                            'consensus_rating': info.get('recommendationKey', '').upper(),
                            'current_price': current_price,
                            'target_mean': target_mean,
                            'target_high': target_high,
                            'target_low': target_low,
                            'target_median': target_median,
                            'target_upside_pct': target_upside_pct,
                        }
            except Exception as e:
                logger.debug(f"{ticker}: No recommendation breakdown - {e}")

            # Fallback: just use what we have
            return {
                'ticker': ticker,
                'date': date.today(),
                'analyst_total': analyst_total,
                'analyst_strong_buy': 0,
                'analyst_buy': 0,
                'analyst_hold': 0,
                'analyst_sell': 0,
                'analyst_strong_sell': 0,
                'analyst_positive': 0,
                'analyst_positivity': None,
                'consensus_rating': info.get('recommendationKey', '').upper(),
                'current_price': current_price,
                'target_mean': target_mean,
                'target_high': target_high,
                'target_low': target_low,
                'target_median': target_median,
                'target_upside_pct': target_upside_pct,
            }

        except Exception as e:
            logger.error(f"{ticker}: Error fetching - {e}")
            return None

    def save_analyst_data(self, data: Dict[str, Any]) -> bool:
        """Save analyst ratings and price targets to database."""
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Save analyst ratings
                    cur.execute("""
                        INSERT INTO analyst_ratings (
                            ticker, date, analyst_total, analyst_buy, analyst_hold, analyst_sell,
                            analyst_strong_buy, analyst_strong_sell, analyst_positive, analyst_positivity,
                            consensus_rating, source
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, date, source) DO UPDATE SET
                            analyst_total = EXCLUDED.analyst_total,
                            analyst_buy = EXCLUDED.analyst_buy,
                            analyst_hold = EXCLUDED.analyst_hold,
                            analyst_sell = EXCLUDED.analyst_sell,
                            analyst_strong_buy = EXCLUDED.analyst_strong_buy,
                            analyst_strong_sell = EXCLUDED.analyst_strong_sell,
                            analyst_positive = EXCLUDED.analyst_positive,
                            analyst_positivity = EXCLUDED.analyst_positivity,
                            consensus_rating = EXCLUDED.consensus_rating
                    """, (
                        data['ticker'], data['date'],
                        data.get('analyst_total'), data.get('analyst_buy'), data.get('analyst_hold'),
                        data.get('analyst_sell'), data.get('analyst_strong_buy'), data.get('analyst_strong_sell'),
                        data.get('analyst_positive'), data.get('analyst_positivity'),
                        data.get('consensus_rating'), 'yfinance'
                    ))

                    # Save price targets
                    if data.get('target_mean'):
                        cur.execute("""
                            INSERT INTO price_targets (
                                ticker, date, current_price, target_mean, target_high, target_low,
                                target_median, target_upside_pct, analyst_count, source
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (ticker, date, source) DO UPDATE SET
                                current_price = EXCLUDED.current_price,
                                target_mean = EXCLUDED.target_mean,
                                target_high = EXCLUDED.target_high,
                                target_low = EXCLUDED.target_low,
                                target_median = EXCLUDED.target_median,
                                target_upside_pct = EXCLUDED.target_upside_pct,
                                analyst_count = EXCLUDED.analyst_count
                        """, (
                            data['ticker'], data['date'],
                            data.get('current_price'), data.get('target_mean'), data.get('target_high'),
                            data.get('target_low'), data.get('target_median'), data.get('target_upside_pct'),
                            data.get('analyst_total'), 'yfinance'
                        ))

            return True
        except Exception as e:
            logger.error(f"{data.get('ticker')}: Error saving - {e}")
            return False

    def ingest_ticker(self, ticker: str) -> bool:
        """Fetch and save analyst data for a single ticker."""
        data = self.fetch_analyst_data(ticker)

        if data is None:
            return False

        success = self.save_analyst_data(data)
        if success:
            logger.info(f"{ticker}: Saved analyst data")
        return success

    def ingest_universe(self, delay: float = 0.5, progress_callback=None) -> dict:
        """Fetch and save analyst data for all tickers."""
        tickers = self.repo.get_universe()
        total = len(tickers)

        results = {'success': [], 'failed': [], 'total': total}

        logger.info(f"Starting analyst data ingestion for {total} tickers")

        for i, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(i + 1, total, ticker)

            success = self.ingest_ticker(ticker)

            if success:
                results['success'].append(ticker)
            else:
                results['failed'].append(ticker)

            if delay > 0 and i < total - 1:
                time.sleep(delay)

        return results