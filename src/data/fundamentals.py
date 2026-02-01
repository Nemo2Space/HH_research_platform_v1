"""
Alpha Platform - Fundamental Data Ingestion

Fetches fundamental data (PE, market cap, etc.) using yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import date
from typing import Optional, Dict, Any, List
import time

from src.db.repository import Repository
from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FundamentalDataIngester:
    """Fetches and stores fundamental data."""

    def __init__(self, repository: Optional[Repository] = None):
        self.repo = repository or Repository()

    def _normalize_percentage(self, value: Any, field_name: str) -> Optional[float]:
        """
        Normalize percentage values from yfinance.

        yfinance is inconsistent - sometimes returns:
        - 0.0348 (decimal, correct)
        - 3.48 (percentage, needs /100)
        - 348 (error, needs /100)

        We normalize everything to decimal (0.0348 = 3.48%)
        """
        if value is None:
            return None

        try:
            val = float(value)

            # If value > 1.0 (100%), it's likely already a percentage
            if field_name == 'dividend_payout_ratio':
                # Payout ratios can be > 100%, but not > 500%
                if val > 5.0:  # 500%
                    val = val / 100
            elif field_name in ('profit_margin', 'operating_margin',
                               'gross_margin', 'roe', 'roa', 'revenue_growth',
                               'earnings_growth', 'eps_growth'):
                # These should all be decimals (0.05 = 5%)
                # If > 1.0 (100%), likely needs division
                if val > 1.0:
                    val = val / 100
            elif field_name == 'dividend_yield':
                # Special handling for dividend yield
                # yfinance sometimes returns weird values
                # Valid dividend yields are typically 0-15% (0.0 to 0.15 as decimal)

                # If > 1.0, definitely needs division (e.g., 17 -> 0.17)
                if val > 1.0:
                    val = val / 100

                # If still > 0.15 (15%), likely still wrong - divide again or flag
                if val > 0.15:
                    logger.warning(f"Dividend yield {val:.4f} ({val*100:.2f}%) seems high, dividing by 100")
                    val = val / 100

                # Final sanity check: if > 0.20 (20%), mark as suspicious
                if val > 0.20:
                    logger.error(f"Dividend yield {val:.4f} ({val*100:.2f}%) is unrealistic, setting to None")
                    return None

            return val
        except (ValueError, TypeError):
            return None

    def _calculate_dividend_yield(self, info: Dict) -> Optional[float]:
        """
        Calculate dividend yield from dividend rate and price.
        This is more reliable than using yfinance's dividendYield directly.
        """
        try:
            # Get dividend rate (annual dividend per share)
            div_rate = info.get('dividendRate') or info.get('trailingAnnualDividendRate')

            # Get current price
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')

            if div_rate and price and price > 0:
                calculated_yield = div_rate / price

                # Sanity check: yield should be between 0 and 20%
                if 0 <= calculated_yield <= 0.20:
                    return calculated_yield
                else:
                    logger.warning(f"Calculated dividend yield {calculated_yield:.4f} is out of range")

            # Fallback to yfinance's dividendYield with normalization
            yf_yield = info.get('dividendYield')
            if yf_yield is not None:
                return self._normalize_percentage(yf_yield, 'dividend_yield')

            return None

        except Exception as e:
            logger.debug(f"Error calculating dividend yield: {e}")
            return None

    def fetch_fundamentals(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch fundamental data for a single ticker.

        Returns:
            Dict with fundamental metrics or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or 'symbol' not in info:
                logger.warning(f"{ticker}: No info returned")
                return None

            # Calculate dividend yield ourselves for accuracy
            dividend_yield = self._calculate_dividend_yield(info)

            # Log if there's a big discrepancy
            yf_yield = info.get('dividendYield')
            if yf_yield and dividend_yield:
                if abs(yf_yield - dividend_yield) > 0.01:  # More than 1% difference
                    logger.info(f"{ticker}: Using calculated yield {dividend_yield:.4f} vs yfinance {yf_yield:.4f}")

            fundamentals = {
                'ticker': ticker,
                'date': date.today(),

                # Valuation (these are ratios, not percentages)
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'peg_ratio': info.get('pegRatio'),

                # Profitability (normalize percentages)
                'profit_margin': self._normalize_percentage(info.get('profitMargins'), 'profit_margin'),
                'operating_margin': self._normalize_percentage(info.get('operatingMargins'), 'operating_margin'),
                'gross_margin': self._normalize_percentage(info.get('grossMargins'), 'gross_margin'),
                'roe': self._normalize_percentage(info.get('returnOnEquity'), 'roe'),
                'roa': self._normalize_percentage(info.get('returnOnAssets'), 'roa'),

                # Growth (normalize percentages)
                'revenue_growth': self._normalize_percentage(info.get('revenueGrowth'), 'revenue_growth'),
                'earnings_growth': self._normalize_percentage(info.get('earningsGrowth'), 'earnings_growth'),
                'eps_growth': self._normalize_percentage(info.get('earningsQuarterlyGrowth'), 'eps_growth'),

                # Dividend - use calculated yield for accuracy
                'dividend_yield': dividend_yield,
                'dividend_payout_ratio': self._normalize_percentage(info.get('payoutRatio'), 'dividend_payout_ratio'),

                # Financial Health (ratios, not percentages)
                'current_ratio': info.get('currentRatio'),
                'debt_to_equity': info.get('debtToEquity'),
                'free_cash_flow': info.get('freeCashflow'),

                # Per Share
                'eps': info.get('trailingEps'),
                'book_value_per_share': info.get('bookValue'),
                'revenue_per_share': info.get('revenuePerShare'),
            }

            return fundamentals

        except Exception as e:
            logger.error(f"{ticker}: Error fetching - {e}")
            return None

    def save_fundamentals(self, data: Dict[str, Any]) -> bool:
        """Save fundamental data to database."""
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO fundamentals (
                            ticker, date, market_cap, pe_ratio, forward_pe, pb_ratio, ps_ratio, peg_ratio,
                            profit_margin, operating_margin, gross_margin, roe, roa,
                            revenue_growth, earnings_growth, eps_growth,
                            dividend_yield, dividend_payout_ratio,
                            current_ratio, debt_to_equity, free_cash_flow,
                            eps, book_value_per_share, revenue_per_share
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            market_cap = EXCLUDED.market_cap,
                            pe_ratio = EXCLUDED.pe_ratio,
                            forward_pe = EXCLUDED.forward_pe,
                            pb_ratio = EXCLUDED.pb_ratio,
                            ps_ratio = EXCLUDED.ps_ratio,
                            peg_ratio = EXCLUDED.peg_ratio,
                            profit_margin = EXCLUDED.profit_margin,
                            operating_margin = EXCLUDED.operating_margin,
                            gross_margin = EXCLUDED.gross_margin,
                            roe = EXCLUDED.roe,
                            roa = EXCLUDED.roa,
                            revenue_growth = EXCLUDED.revenue_growth,
                            earnings_growth = EXCLUDED.earnings_growth,
                            eps_growth = EXCLUDED.eps_growth,
                            dividend_yield = EXCLUDED.dividend_yield,
                            dividend_payout_ratio = EXCLUDED.dividend_payout_ratio,
                            current_ratio = EXCLUDED.current_ratio,
                            debt_to_equity = EXCLUDED.debt_to_equity,
                            free_cash_flow = EXCLUDED.free_cash_flow,
                            eps = EXCLUDED.eps,
                            book_value_per_share = EXCLUDED.book_value_per_share,
                            revenue_per_share = EXCLUDED.revenue_per_share
                    """, (
                        data['ticker'], data['date'],
                        data.get('market_cap'), data.get('pe_ratio'), data.get('forward_pe'),
                        data.get('pb_ratio'), data.get('ps_ratio'), data.get('peg_ratio'),
                        data.get('profit_margin'), data.get('operating_margin'), data.get('gross_margin'),
                        data.get('roe'), data.get('roa'),
                        data.get('revenue_growth'), data.get('earnings_growth'), data.get('eps_growth'),
                        data.get('dividend_yield'), data.get('dividend_payout_ratio'),
                        data.get('current_ratio'), data.get('debt_to_equity'), data.get('free_cash_flow'),
                        data.get('eps'), data.get('book_value_per_share'), data.get('revenue_per_share'),
                    ))
            return True
        except Exception as e:
            logger.error(f"{data.get('ticker')}: Error saving - {e}")
            return False

    def ingest_ticker(self, ticker: str) -> bool:
        """Fetch and save fundamental data for a single ticker."""
        data = self.fetch_fundamentals(ticker)

        if data is None:
            return False

        success = self.save_fundamentals(data)
        if success:
            logger.info(f"{ticker}: Saved fundamentals")
        return success

    def ingest_universe(self, delay: float = 0.5, progress_callback=None) -> dict:
        """
        Fetch and save fundamental data for all tickers.

        Returns:
            Dict with success/failed counts
        """
        tickers = self.repo.get_universe()
        total = len(tickers)

        results = {
            'success': [],
            'failed': [],
            'total': total
        }

        logger.info(f"Starting fundamental ingestion for {total} tickers")

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

        logger.info(f"Complete: {len(results['success'])} success, {len(results['failed'])} failed")

        return results

    def get_latest_fundamentals(self) -> pd.DataFrame:
        """Get the most recent fundamentals for each ticker."""
        query = """
            SELECT DISTINCT ON (ticker) *
            FROM fundamentals
            ORDER BY ticker, date DESC
        """
        return pd.read_sql(query, self.repo.engine)