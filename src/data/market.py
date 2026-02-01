"""
Alpha Platform - Market Data Ingestion

Fetches stock prices using yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import time

from src.db.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MarketDataIngester:
    """Fetches and stores market data."""

    def __init__(self, repository: Optional[Repository] = None):
        self.repo = repository or Repository()

    def fetch_prices(self, ticker: str, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Fetch price history for a single ticker.

        Args:
            ticker: Stock ticker symbol
            days: Number of days of history to fetch

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"{ticker}: No data returned")
                return None

            # Reset index to make date a column
            df = df.reset_index()
            df = df.rename(columns={'Date': 'date'})

            # Remove timezone info if present
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)

            df['date'] = df['date'].dt.date

            return df

        except Exception as e:
            logger.error(f"{ticker}: Error fetching - {e}")
            return None

    def ingest_ticker(self, ticker: str, days: int = 365) -> bool:
        """
        Fetch and save price data for a single ticker.

        Returns:
            True if successful, False otherwise
        """
        df = self.fetch_prices(ticker, days)

        if df is None or df.empty:
            return False

        try:
            self.repo.save_prices(ticker, df)
            logger.info(f"{ticker}: Saved {len(df)} price records")
            return True
        except Exception as e:
            logger.error(f"{ticker}: Error saving - {e}")
            return False

    def ingest_universe(self, days: int = 365, delay: float = 0.5,
                        progress_callback=None) -> dict:
        """
        Fetch and save price data for all tickers in universe.

        Args:
            days: Days of history to fetch
            delay: Delay between requests (rate limiting)
            progress_callback: Optional callback(current, total, ticker)

        Returns:
            Dict with success/failed counts and lists
        """
        tickers = self.repo.get_universe()
        total = len(tickers)

        results = {
            'success': [],
            'failed': [],
            'total': total
        }

        logger.info(f"Starting ingestion for {total} tickers")

        for i, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(i + 1, total, ticker)

            success = self.ingest_ticker(ticker, days)

            if success:
                results['success'].append(ticker)
            else:
                results['failed'].append(ticker)

            # Rate limiting
            if delay > 0 and i < total - 1:
                time.sleep(delay)

        logger.info(f"Ingestion complete: {len(results['success'])} success, {len(results['failed'])} failed")

        return results

    def get_latest_prices(self) -> pd.DataFrame:
        """Get the most recent price for each ticker."""
        query = """
            SELECT DISTINCT ON (ticker) 
                ticker, date, open, high, low, close, adj_close, volume
            FROM prices
            ORDER BY ticker, date DESC
        """
        return pd.read_sql(query, self.repo.engine)


def run_ingestion(days: int = 365, delay: float = 0.5):
    """
    Run full market data ingestion.

    Usage:
        python -c "from src.data.market import run_ingestion; run_ingestion()"
    """
    ingester = MarketDataIngester()

    def progress(current, total, ticker):
        pct = int(current / total * 100)
        print(f"[{pct:3d}%] ({current}/{total}) {ticker}")

    results = ingester.ingest_universe(days=days, delay=delay, progress_callback=progress)

    print("\n" + "=" * 50)
    print(f"SUCCESS: {len(results['success'])} tickers")
    print(f"FAILED:  {len(results['failed'])} tickers")

    if results['failed']:
        print(f"\nFailed tickers: {results['failed']}")

    return results