"""
Alpha Platform - Insider Trading Data

Fetches insider transactions from Finnhub and SEC.
"""

import os
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.db.connection import get_connection
from src.db.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InsiderTransaction:
    """Insider transaction data."""
    ticker: str
    filing_date: str
    transaction_date: str
    insider_name: str
    insider_title: str
    transaction_type: str  # BUY, SELL, GRANT, EXERCISE
    shares: int
    price: float
    value: float
    shares_owned_after: int = 0


class InsiderDataFetcher:
    """Fetches insider trading data."""

    def __init__(self, repository: Optional[Repository] = None):
        self.repo = repository or Repository()
        self.finnhub_key = os.getenv("FINNHUB_API_KEY", "")

    def get_finnhub_insider(self, ticker: str, days_back: int = 90) -> List[InsiderTransaction]:
        """
        Fetch insider transactions from Finnhub.

        Args:
            ticker: Stock ticker
            days_back: Days of history to fetch

        Returns:
            List of InsiderTransaction objects
        """
        if not self.finnhub_key:
            logger.debug(f"{ticker}: Finnhub API key not configured")
            return []

        try:
            url = "https://finnhub.io/api/v1/stock/insider-transactions"
            params = {
                "symbol": ticker,
                "token": self.finnhub_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            transactions = data.get('data', [])

            # Filter by date
            cutoff = datetime.now() - timedelta(days=days_back)
            cutoff_str = cutoff.strftime("%Y-%m-%d")

            results = []
            for tx in transactions:
                tx_date = tx.get('transactionDate', '')
                if tx_date < cutoff_str:
                    continue

                # Determine transaction type
                change = tx.get('change', 0)
                tx_code = tx.get('transactionCode', '')

                if tx_code in ['P', 'A']:  # Purchase or Award
                    tx_type = 'BUY'
                elif tx_code in ['S', 'F']:  # Sale or Tax
                    tx_type = 'SELL'
                elif tx_code == 'M':  # Exercise
                    tx_type = 'EXERCISE'
                elif tx_code == 'G':  # Gift
                    tx_type = 'GIFT'
                else:
                    tx_type = 'OTHER'

                price = float(tx.get('transactionPrice', 0) or 0)
                shares = abs(int(change)) if change else 0

                results.append(InsiderTransaction(
                    ticker=ticker,
                    filing_date=tx.get('filingDate', ''),
                    transaction_date=tx_date,
                    insider_name=tx.get('name', ''),
                    insider_title=tx.get('position', ''),
                    transaction_type=tx_type,
                    shares=shares,
                    price=price,
                    value=shares * price,
                    shares_owned_after=int(tx.get('share', 0) or 0)
                ))

            logger.info(f"{ticker}: Found {len(results)} insider transactions")
            return results

        except Exception as e:
            logger.error(f"{ticker}: Finnhub insider error - {e}")
            return []

    def save_transaction(self, tx: InsiderTransaction) -> bool:
        """Save insider transaction to database."""
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO insider_transactions (
                            ticker, filing_date, transaction_date, insider_name,
                            insider_title, transaction_type, shares, price, value,
                            shares_owned_after
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, insider_name, transaction_date, transaction_type) 
                        DO UPDATE SET
                            shares = EXCLUDED.shares,
                            price = EXCLUDED.price,
                            value = EXCLUDED.value
                    """, (
                        tx.ticker,
                        tx.filing_date,
                        tx.transaction_date,
                        tx.insider_name,
                        tx.insider_title,
                        tx.transaction_type,
                        tx.shares,
                        tx.price,
                        tx.value,
                        tx.shares_owned_after
                    ))
            return True
        except Exception as e:
            logger.error(f"Error saving transaction: {e}")
            return False

    def fetch_and_save(self, ticker: str, days_back: int = 90) -> int:
        """Fetch and save insider transactions for a ticker."""
        transactions = self.get_finnhub_insider(ticker, days_back)

        saved = 0
        for tx in transactions:
            if self.save_transaction(tx):
                saved += 1

        return saved

    def fetch_universe(self, days_back: int = 90, delay: float = 0.3,
                       progress_callback=None) -> Dict[str, int]:
        """Fetch insider transactions for all tickers."""
        tickers = self.repo.get_universe()
        total = len(tickers)

        results = {}

        for i, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(i + 1, total, ticker)

            count = self.fetch_and_save(ticker, days_back)
            results[ticker] = count

            if delay > 0:
                time.sleep(delay)

        total_saved = sum(results.values())
        logger.info(f"Fetched {total_saved} insider transactions for {total} tickers")

        return results

    def get_insider_signal(self, ticker: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze insider transactions and generate a signal.

        Returns:
            Dict with insider_signal score (0-100) and details
        """
        import pandas as pd

        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        query = """
            SELECT transaction_type, shares, value
            FROM insider_transactions
            WHERE ticker = %(ticker)s AND transaction_date >= %(cutoff)s
        """

        try:
            df = pd.read_sql(query, self.repo.engine, params={
                "ticker": ticker,
                "cutoff": cutoff
            })
        except:
            df = pd.DataFrame()

        if len(df) == 0:
            return {
                'insider_signal': 50,  # Neutral
                'buy_count': 0,
                'sell_count': 0,
                'net_value': 0,
                'summary': 'No recent insider activity'
            }

        # Count buys and sells
        buys = df[df['transaction_type'] == 'BUY']
        sells = df[df['transaction_type'] == 'SELL']

        buy_count = len(buys)
        sell_count = len(sells)
        buy_value = buys['value'].sum() if len(buys) > 0 else 0
        sell_value = sells['value'].sum() if len(sells) > 0 else 0
        net_value = buy_value - sell_value

        # Calculate signal (0-100)
        # More buys than sells = bullish
        # More sells than buys = bearish

        total_count = buy_count + sell_count
        if total_count == 0:
            signal = 50
        else:
            buy_ratio = buy_count / total_count
            signal = int(30 + buy_ratio * 40)  # Range 30-70 based on count

            # Adjust by value
            if net_value > 1_000_000:  # Over $1M net buying
                signal = min(100, signal + 20)
            elif net_value > 100_000:
                signal = min(100, signal + 10)
            elif net_value < -1_000_000:  # Over $1M net selling
                signal = max(0, signal - 20)
            elif net_value < -100_000:
                signal = max(0, signal - 10)

        # Generate summary
        if net_value > 100_000:
            summary = f"Net insider buying: ${net_value:,.0f}"
        elif net_value < -100_000:
            summary = f"Net insider selling: ${abs(net_value):,.0f}"
        else:
            summary = f"{buy_count} buys, {sell_count} sells in last {days_back} days"

        return {
            'insider_signal': signal,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'buy_value': buy_value,
            'sell_value': sell_value,
            'net_value': net_value,
            'summary': summary
        }


def get_institutional_holdings(ticker: str, repository: Repository = None) -> Dict[str, Any]:
    """
    Get institutional holdings summary.

    Note: This would typically come from 13F filings or a data provider.
    For now, returns placeholder data.
    """
    # TODO: Implement actual institutional holdings fetching
    # Could use SEC 13F filings or a data provider

    return {
        'institutional_signal': 50,
        'institutional_ownership_pct': None,
        'num_institutions': None,
        'summary': 'Institutional data not available'
    }